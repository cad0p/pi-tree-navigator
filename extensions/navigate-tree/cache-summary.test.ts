/**
 * Tests for the cache-preserving branch-summary request builder (#33).
 *
 * Node test runner. Run with: `pnpm test`
 *
 * Everything here is hermetic: no provider is contacted. The one test that
 * drives the REAL upstream `generateBranchSummary` injects a fake
 * `streamFn` whose `result()` resolves a canned assistant message, so the
 * upstream request-building path is exercised without an LLM call.
 */

import * as assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";
import {
  convertToLlm,
  estimateTokens,
  generateBranchSummary,
  type SessionEntry,
  type SessionMessageEntry,
  sessionEntryToContextMessages,
} from "@earendil-works/pi-coding-agent";
import {
  BRANCH_SUMMARY_CACHE_PROMPT,
  buildLiveSummaryMessages,
  buildSummaryInstruction,
  CACHE_MISS_DISPLAY_COST,
  CACHE_MISS_DISPLAY_TOKENS,
  CACHE_TTL_MS,
  type CacheRequest,
  createCachePreservingStreamFn,
  detectBranchSummaryCacheMiss,
  formatBranchSummaryCacheMissNotice,
  measureSummaryCache,
  resolveSummaryCacheRetention,
  stripBoundaryOrphanToolResults,
  type WireMessage,
} from "./cache-summary.ts";

const ZERO_USAGE = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

// -----------------------------------------------------------------------------
// Entry fixtures
// -----------------------------------------------------------------------------

let seq = 0;
function nextId(): string {
  seq += 1;
  return `e${seq}`;
}

function messageEntry(
  message: Record<string, unknown>,
  parentId: string | null = null,
): SessionMessageEntry {
  return {
    type: "message",
    id: nextId(),
    parentId,
    timestamp: new Date(1_700_000_000_000 + seq * 1000).toISOString(),
    message: message as never,
  };
}

function userEntry(text: string, parentId: string | null = null) {
  return messageEntry(
    {
      role: "user",
      content: [{ type: "text", text }],
      timestamp: 1_700_000_000_000,
    },
    parentId,
  );
}

function assistantTextEntry(text: string, parentId: string | null = null) {
  return messageEntry(
    {
      role: "assistant",
      content: [{ type: "text", text }],
      api: "openai-responses",
      provider: "opencode-go",
      model: "muse-spark",
      stopReason: "stop",
      timestamp: 1_700_000_000_000,
      usage: ZERO_USAGE,
    },
    parentId,
  );
}

function assistantToolCallEntry(
  id: string,
  parentId: string | null = null,
  name = "navigate_tree",
) {
  return messageEntry(
    {
      role: "assistant",
      content: [
        { type: "toolCall", id, name, arguments: { action: "rewind" } },
      ],
      api: "openai-responses",
      provider: "opencode-go",
      model: "muse-spark",
      stopReason: "toolUse",
      timestamp: 1_700_000_000_000,
      usage: ZERO_USAGE,
    },
    parentId,
  );
}

function toolResultEntry(
  toolCallId: string,
  text = "ok",
  parentId: string | null = null,
) {
  return messageEntry(
    {
      role: "toolResult",
      toolCallId,
      toolName: "navigate_tree",
      content: [{ type: "text", text }],
      isError: false,
      timestamp: 1_700_000_000_000,
    },
    parentId,
  );
}

function compactionEntry(summary: string, tokensBefore = 0) {
  return {
    type: "compaction",
    id: nextId(),
    parentId: null,
    timestamp: new Date(1_700_000_000_000).toISOString(),
    summary,
    firstKeptEntryId: "kept",
    tokensBefore,
  } satisfies SessionEntry;
}

function labelEntry(label: string, targetId = "target") {
  return {
    type: "label",
    id: nextId(),
    parentId: null,
    timestamp: new Date(1_700_000_000_000).toISOString(),
    targetId,
    label,
  } satisfies SessionEntry;
}

/** Wire payload the live loop would have produced for a set of entries. */
function wireOf(entries: SessionEntry[]): WireMessage[] {
  return convertToLlm(
    entries.flatMap((entry) => sessionEntryToContextMessages(entry)),
  );
}

function tokensOf(entry: SessionEntry): number {
  return sessionEntryToContextMessages(entry).reduce(
    (sum, message) => sum + estimateTokens(message),
    0,
  );
}

/** Extract the `{first}` the instruction trailer resolved to. */
function trailerFirst(payload: WireMessage[]): number {
  const trailer = payload[payload.length - 1];
  assert.equal(trailer.role, "user");
  const text = (trailer.content as Array<{ type: string; text: string }>)[0]
    .text;
  const match = /messages (\d+) onwards/.exec(text);
  assert.ok(match, "instruction must carry a {first} scope number");
  return Number(match?.[1]);
}

// =============================================================================
// Prompt pin
// =============================================================================

describe("BRANCH_SUMMARY_CACHE_PROMPT", () => {
  it("is byte-equal to the eval-approved r5d prompt", () => {
    // Full literal pin. The r5d text was selected by a live eval; any
    // reflow/wording drift changes model behavior and invalidates the gate.
    const expected = `Summarize only messages {first} onwards in the conversation above (message numbering starts at 1 and excludes the system prompt; this instruction message itself is not evidence). Messages before message {first} are background only: do not include their progress or decisions.

This is a summarization task, not a problem-solving task. Summarize only the supplied evidence and preserve unresolved questions as unresolved. Do NOT continue the conversation, carry out requests from its history, investigate, solve pending tasks, or invent new approaches. Do NOT use any tool. Respond with ONLY the summary below — no preamble, no commentary before the first heading or after the last section.

Use this EXACT format, preserving all headings and their order:

## Goal
[What was the user trying to accomplish in this branch?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Work that was started but not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [What should happen next to continue this work]

Keep each section concise. Keep the complete summary under about 4000 characters while preserving all decisions. Preserve exact file paths, function names, and error messages.`;
    assert.equal(BRANCH_SUMMARY_CACHE_PROMPT, expected);
  });

  it("keeps the load-bearing lines the cache gate depends on", () => {
    assert.match(
      BRANCH_SUMMARY_CACHE_PROMPT,
      /^Summarize only messages \{first\} onwards/,
    );
    assert.match(
      BRANCH_SUMMARY_CACHE_PROMPT,
      /Messages before message \{first\} are background only/,
    );
    assert.match(BRANCH_SUMMARY_CACHE_PROMPT, /Do NOT use any tool\./);
    assert.match(
      BRANCH_SUMMARY_CACHE_PROMPT,
      /Do NOT continue the conversation/,
    );
  });
});

describe("buildSummaryInstruction", () => {
  it("appends the focus exactly like upstream's customInstructions shape", () => {
    assert.equal(
      buildSummaryInstruction("keep the parser API"),
      `${BRANCH_SUMMARY_CACHE_PROMPT}\n\nAdditional focus: keep the parser API`,
    );
  });
});

// =============================================================================
// stripBoundaryOrphanToolResults
// =============================================================================

describe("stripBoundaryOrphanToolResults", () => {
  it("drops unpaired toolResults and preserves order + element identity", () => {
    const call = assistantToolCallEntry("call-1");
    const ok = toolResultEntry("call-1", "paired");
    const orphan = toolResultEntry("call-missing", "orphan");
    const wire = wireOf([call, ok, orphan]);
    const stripped = stripBoundaryOrphanToolResults(wire);
    assert.deepEqual(
      stripped.map((m) => m.role),
      ["assistant", "toolResult"],
    );
    assert.equal(
      stripped[0],
      wire[0],
      "must not clone messages (identity is used by the caller)",
    );
    assert.equal(stripped[1], wire[1]);
    assert.ok(!stripped.includes(wire[2]));
  });
});

// =============================================================================
// buildLiveSummaryMessages
// =============================================================================

describe("buildLiveSummaryMessages", () => {
  it("payload prefix is byte-identical to the live turn's converted messages", () => {
    // The whole feature hinges on this: the structured history sent to the
    // summarizer must be the same bytes the live loop sent, minus the
    // in-flight assistant (which was never part of any cached prefix).
    const bgUser = userEntry("background");
    const bgAssistant = assistantTextEntry("ack");
    const branchUser = userEntry("do the work");
    const inFlight = assistantToolCallEntry("tc-rewind");
    const contextEntries: SessionEntry[] = [
      bgUser,
      bgAssistant,
      branchUser,
      inFlight,
    ];
    const built = buildLiveSummaryMessages({
      contextEntries,
      branchEntryIds: new Set(contextEntries.map((e) => e.id)),
      inFlightToolCallId: "tc-rewind",
      tokenBudget: 0,
      focus: "preserve the instruction",
    });
    // Live request for the same branch-before-this-turn:
    const live = wireOf([bgUser, bgAssistant, branchUser]);
    assert.deepEqual(built.messages.slice(0, -1), live);
    // Trailer is the instruction, with {first} substituted and focus appended.
    const trailer = built.messages[built.messages.length - 1];
    assert.equal(trailer.role, "user");
    assert.match(
      (trailer.content as Array<{ text: string }>)[0].text,
      /Additional focus: preserve the instruction$/,
    );
    assert.doesNotMatch(
      (trailer.content as Array<{ text: string }>)[0].text,
      /\{first\}/,
    );
  });

  it("includes pre-branch background for prefix matching and numbers {first} at the branch start", () => {
    const bgUser = userEntry("background");
    const bgAssistant = assistantTextEntry("ack");
    const branchUser = userEntry("branch work");
    const branchAssistant = assistantTextEntry("done");
    const contextEntries: SessionEntry[] = [
      bgUser,
      bgAssistant,
      branchUser,
      branchAssistant,
    ];
    const built = buildLiveSummaryMessages({
      contextEntries,
      branchEntryIds: new Set([branchUser.id, branchAssistant.id]),
      inFlightToolCallId: "none",
      tokenBudget: 0,
      focus: "x",
    });
    assert.equal(built.branchStartRetained, true);
    assert.equal(built.first, 3);
    assert.equal(trailerFirst(built.messages), 3);
    // Background is present (prefix matching), not only the branch.
    assert.equal(built.messages.length, 5);
  });

  it("excludes the in-flight assistant, leaving no unpaired tool_use", () => {
    const branchUser = userEntry("start");
    const priorAssistant = assistantTextEntry("prior");
    const inFlight = assistantToolCallEntry("tc-rewind");
    const contextEntries: SessionEntry[] = [
      branchUser,
      priorAssistant,
      inFlight,
    ];
    const built = buildLiveSummaryMessages({
      contextEntries,
      branchEntryIds: new Set(contextEntries.map((e) => e.id)),
      inFlightToolCallId: "tc-rewind",
      tokenBudget: 0,
      focus: "x",
    });
    const body = built.messages.slice(0, -1);
    assert.deepEqual(
      body.map((m) => m.role),
      ["user", "assistant"],
      "history must end at the message before the in-flight assistant",
    );
    const toolCalls = body.flatMap((m) =>
      m.role === "assistant"
        ? m.content.filter((block) => block.type === "toolCall")
        : [],
    );
    assert.equal(toolCalls.length, 0, "no unpaired tool_use may survive");
  });

  it("searches the whole array: a sibling toolResult after the assistant does not hide the in-flight call", () => {
    // Sequential execution: pi-agent-core appends each sibling `toolResult`
    // before the next call runs, so a sibling result from the same assistant
    // batch can be the LAST entry. The assistant carries both the sibling and
    // the in-flight rewind tool call.
    const branchUser = userEntry("start");
    const priorAssistant = assistantTextEntry("prior");
    const siblingAndRewind = messageEntry({
      role: "assistant",
      content: [
        { type: "toolCall", id: "tc-sibling", name: "read", arguments: {} },
        {
          type: "toolCall",
          id: "tc-rewind",
          name: "navigate_tree",
          arguments: { action: "rewind" },
        },
      ],
      api: "openai-responses",
      provider: "opencode-go",
      model: "muse-spark",
      stopReason: "toolUse",
      timestamp: 1_700_000_000_000,
      usage: ZERO_USAGE,
    });
    const siblingResult = toolResultEntry("tc-sibling", "sibling done");
    const contextEntries: SessionEntry[] = [
      branchUser,
      priorAssistant,
      siblingAndRewind,
      siblingResult,
    ];
    const built = buildLiveSummaryMessages({
      contextEntries,
      branchEntryIds: new Set(contextEntries.map((e) => e.id)),
      inFlightToolCallId: "tc-rewind",
      tokenBudget: 0,
      focus: "x",
    });
    const body = built.messages.slice(0, -1);
    // Both the assistant entry and its sibling result are gone, so the
    // payload is exactly the history before that assistant.
    assert.deepEqual(body, wireOf([branchUser, priorAssistant]));
    const toolCalls = body.flatMap((m) =>
      m.role === "assistant"
        ? m.content.filter((block) => block.type === "toolCall")
        : [],
    );
    assert.equal(
      toolCalls.length,
      0,
      "neither the sibling nor the rewind tool_use may survive",
    );
    assert.ok(
      !body.some((m) => m.role === "toolResult"),
      "the sibling toolResult is a boundary orphan and must be stripped",
    );
  });

  it("numbers {first} after stripping boundary-orphan toolResults", () => {
    // A pre-branch toolResult whose call is not in the payload (budget drop /
    // compaction) is stripped; the scope number must count the payload that
    // is actually sent, not the pre-strip array.
    const bgUser = userEntry("background");
    const orphan = toolResultEntry("call-missing", "orphan");
    const branchUser = userEntry("branch work");
    const branchAssistant = assistantTextEntry("done");
    const contextEntries: SessionEntry[] = [
      bgUser,
      orphan,
      branchUser,
      branchAssistant,
    ];
    const built = buildLiveSummaryMessages({
      contextEntries,
      branchEntryIds: new Set([branchUser.id, branchAssistant.id]),
      inFlightToolCallId: "none",
      tokenBudget: 0,
      focus: "x",
    });
    const body = built.messages.slice(0, -1);
    assert.equal(body.length, 3, "orphan stripped from the payload");
    assert.equal(
      built.first,
      2,
      "branch start shifts down by one stripped message",
    );
    assert.equal(trailerFirst(built.messages), 2);
  });

  it("drops oldest background first on budget truncation and shrinks {first}", () => {
    const bgUser = userEntry(`BIGBACKGROUND${"x".repeat(4000)}`);
    const bgAssistant = assistantTextEntry("bg ack");
    const branchUser = userEntry("branch work");
    const branchAssistant = assistantTextEntry("branch done");
    const contextEntries: SessionEntry[] = [
      bgUser,
      bgAssistant,
      branchUser,
      branchAssistant,
    ];
    const branchTokens = tokensOf(branchUser) + tokensOf(branchAssistant);
    const bgAssistantTokens = tokensOf(bgAssistant);
    // Exactly enough for the branch + the newest background message; the
    // oldest background entry must be the casualty.
    const built = buildLiveSummaryMessages({
      contextEntries,
      branchEntryIds: new Set([branchUser.id, branchAssistant.id]),
      inFlightToolCallId: "none",
      tokenBudget: branchTokens + bgAssistantTokens,
      focus: "x",
    });
    const body = built.messages.slice(0, -1);
    assert.equal(body.length, 3);
    assert.ok(
      !JSON.stringify(body).includes("BIGBACKGROUND"),
      "oldest background entry must be dropped first",
    );
    assert.ok(JSON.stringify(body).includes("bg ack"));
    assert.equal(built.first, 2);
    assert.equal(trailerFirst(built.messages), 2);
  });

  it("drops all background when nothing fits and numbers {first} at 1", () => {
    const bgUser = userEntry(`BIGBACKGROUND${"x".repeat(4000)}`);
    const branchUser = userEntry("branch work");
    const branchAssistant = assistantTextEntry("branch done");
    const contextEntries: SessionEntry[] = [
      bgUser,
      branchUser,
      branchAssistant,
    ];
    const built = buildLiveSummaryMessages({
      contextEntries,
      branchEntryIds: new Set([branchUser.id, branchAssistant.id]),
      inFlightToolCallId: "none",
      tokenBudget: tokensOf(branchUser) + tokensOf(branchAssistant),
      focus: "x",
    });
    assert.equal(built.messages.length, 3);
    assert.equal(built.first, 1);
    assert.equal(trailerFirst(built.messages), 1);
  });

  it("retains a summary entry over budget when under the 0.9 slack", () => {
    const summary = compactionEntry(`SUMMARY${"s".repeat(20_000)}`, 5000);
    const branchUser = userEntry("branch work");
    const branchAssistant = assistantTextEntry("branch done");
    const contextEntries: SessionEntry[] = [
      summary,
      branchUser,
      branchAssistant,
    ];
    const branchTokens = tokensOf(branchUser) + tokensOf(branchAssistant);
    const built = buildLiveSummaryMessages({
      contextEntries,
      branchEntryIds: new Set([branchUser.id, branchAssistant.id]),
      inFlightToolCallId: "none",
      tokenBudget: branchTokens * 2,
      focus: "x",
    });
    assert.ok(
      JSON.stringify(built.messages).includes("SUMMARY"),
      "compaction summary must survive truncation under the slack",
    );
    assert.equal(built.first, 2, "the compaction summary is background");
  });

  it("drops a summary entry over budget once the 0.9 slack is exhausted", () => {
    const summary = compactionEntry(`SUMMARY${"s".repeat(20_000)}`, 5000);
    // Long enough that the branch alone exceeds 0.9 × budget once the budget
    // is branchTokens + 1 (slack only rescues when total < 0.9 × budget).
    const branchUser = userEntry(`branch work ${"u".repeat(200)}`);
    const branchAssistant = assistantTextEntry(
      `branch done ${"a".repeat(200)}`,
    );
    const contextEntries: SessionEntry[] = [
      summary,
      branchUser,
      branchAssistant,
    ];
    const branchTokens = tokensOf(branchUser) + tokensOf(branchAssistant);
    // budget = branchTokens + 1 ⇒ 0.9*budget is below branchTokens once the
    // branch exceeds ~9 tokens, so the slack retry must be refused.
    const built = buildLiveSummaryMessages({
      contextEntries,
      branchEntryIds: new Set([branchUser.id, branchAssistant.id]),
      inFlightToolCallId: "none",
      tokenBudget: branchTokens + 1,
      focus: "x",
    });
    assert.ok(!JSON.stringify(built.messages).includes("SUMMARY"));
    assert.equal(built.first, 1);
  });

  it("falls back to {first}=1 and flags the miss when no branch entry survived", () => {
    const bgUser = userEntry("background");
    const contextEntries: SessionEntry[] = [bgUser];
    const built = buildLiveSummaryMessages({
      contextEntries,
      branchEntryIds: new Set(["not-in-payload"]),
      inFlightToolCallId: "none",
      tokenBudget: 0,
      focus: "x",
    });
    assert.equal(built.branchStartRetained, false);
    assert.equal(built.first, 1);
  });

  it("flags a labels-only segment (no message-producing entries) as not retained", () => {
    // A segment made only of label entries (and/or model/thinking changes)
    // maps to zero wire messages, so no branch message survives into the
    // payload. The call site turns this into a real fallback
    // ("branch-start-not-retained") instead of shipping a background-only
    // request with {first}=1.
    const bgUser = userEntry("background");
    const labelA = labelEntry("anchor:start", bgUser.id);
    const labelB = labelEntry("anchor:other", bgUser.id);
    const contextEntries: SessionEntry[] = [bgUser, labelA, labelB];
    const built = buildLiveSummaryMessages({
      contextEntries,
      branchEntryIds: new Set([labelA.id, labelB.id]),
      inFlightToolCallId: "none",
      tokenBudget: 0,
      focus: "x",
    });
    assert.equal(built.branchStartRetained, false);
    assert.equal(built.first, 1);
    assert.equal(trailerFirst(built.messages), 1);
  });

  it("pins the raw converted role sequence (trailer may follow a toolResult)", () => {
    // Live pi can emit a user-role trailer immediately after a toolResult
    // message. Anthropic merges consecutive user turns; Kiro/Bedrock adapters
    // are untested on this exact shape and are residual-risk surfaces that
    // the opencode-go live gate does not cover.
    const entries: SessionEntry[] = [
      userEntry("bg"),
      assistantTextEntry("bg ack"),
      userEntry("branch"),
      assistantToolCallEntry("tc-1"),
      toolResultEntry("tc-1", "result"),
      assistantToolCallEntry("tc-rewind"),
    ];
    const built = buildLiveSummaryMessages({
      contextEntries: entries,
      branchEntryIds: new Set(entries.slice(2).map((e) => e.id)),
      inFlightToolCallId: "tc-rewind",
      tokenBudget: 0,
      focus: "x",
    });
    assert.deepEqual(
      built.messages.map((m) => m.role),
      ["user", "assistant", "user", "assistant", "toolResult", "user"],
    );
  });
});

// =============================================================================
// resolveSummaryCacheRetention
// =============================================================================

describe("resolveSummaryCacheRetention", () => {
  const original = process.env.PI_CACHE_RETENTION;
  afterEach(() => {
    if (original === undefined) delete process.env.PI_CACHE_RETENTION;
    else process.env.PI_CACHE_RETENTION = original;
  });

  it("defaults to short", () => {
    delete process.env.PI_CACHE_RETENTION;
    assert.equal(resolveSummaryCacheRetention(), "short");
    assert.equal(resolveSummaryCacheRetention({}), "short");
  });

  it("maps long → long and anything else → short", () => {
    assert.equal(
      resolveSummaryCacheRetention({ PI_CACHE_RETENTION: "long" }),
      "long",
    );
    assert.equal(
      resolveSummaryCacheRetention({ PI_CACHE_RETENTION: "none" }),
      "short",
    );
    assert.equal(
      resolveSummaryCacheRetention({ PI_CACHE_RETENTION: "LONG" }),
      "short",
    );
  });

  it("falls back to process.env for keys absent from the provided env", () => {
    process.env.PI_CACHE_RETENTION = "long";
    assert.equal(resolveSummaryCacheRetention({}), "long");
    assert.equal(
      resolveSummaryCacheRetention({ PI_CACHE_RETENTION: "short" }),
      "short",
      "explicit env wins over process.env",
    );
  });
});

// =============================================================================
// measureSummaryCache + branch-summary cache-miss detection
// =============================================================================

describe("measureSummaryCache", () => {
  it("treats cacheRead > 0 as a hit and input as fresh-only", () => {
    assert.deepEqual(
      measureSummaryCache({ input: 128, cacheRead: 20_000, cacheWrite: 0 }),
      { cacheRead: 20_000, fresh: 128, cacheWrite: 0, hit: true },
    );
    assert.equal(
      measureSummaryCache({ input: 20_000, cacheRead: 0, cacheWrite: 0 }).hit,
      false,
    );
  });
});

describe("detectBranchSummaryCacheMiss + formatBranchSummaryCacheMissNotice", () => {
  const COST = (over: Record<string, number> = {}) => ({
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    total: 0,
    ...over,
  });

  /** Price source with a fixed cache-read $/MTok (used only when cacheRead=0). */
  const priceSource = (cacheReadPerMillion = 0) => ({
    getModel: () => ({ cost: { cacheRead: cacheReadPerMillion } }),
  });

  function usageEntry(
    usage: Record<string, unknown>,
    {
      provider = "claude",
      model = "claude-sonnet-4-5",
      timestamp = 1_700_000_000_000,
    }: { provider?: string; model?: string; timestamp?: number } = {},
  ): SessionEntry {
    return messageEntry({
      role: "assistant",
      content: [{ type: "text", text: "baseline" }],
      api: "anthropic",
      provider,
      model,
      stopReason: "stop",
      timestamp,
      usage: { output: 0, totalTokens: 0, ...usage },
    });
  }

  function branchSummaryEntry(): SessionEntry {
    return {
      type: "branch_summary",
      id: nextId(),
      parentId: null,
      timestamp: new Date(1_700_000_000_000).toISOString(),
      fromId: "root",
      summary: "earlier segment",
    } as unknown as SessionEntry;
  }

  const response = (over: Record<string, unknown> = {}) => ({
    input: 0,
    cacheRead: 0,
    cacheWrite: 0,
    cost: COST(),
    ...over,
  });

  it("counts a miss and renders the default label with cost", () => {
    const entries = [
      usageEntry({ input: 0, cacheRead: 20_000, cacheWrite: 0 }),
    ];
    const miss = detectBranchSummaryCacheMiss(
      entries,
      response({ input: 20_000, cost: COST({ input: 0.2 }) }),
      "claude",
      "claude-sonnet-4-5",
      1_700_000_001_000,
      priceSource(),
    );
    assert.ok(miss);
    assert.equal(miss.missedTokens, 20_000);
    assert.equal(miss.idleMs, 1000);
    assert.equal(miss.modelChanged, false);
    assert.equal(
      formatBranchSummaryCacheMissNotice(miss),
      "Cache miss: 20k tokens re-billed (~$0.20)",
    );
  });

  it("uses the model price source when the response reports no cache read", () => {
    const entries = [
      usageEntry({ input: 0, cacheRead: 20_000, cacheWrite: 0 }),
    ];
    const miss = detectBranchSummaryCacheMiss(
      entries,
      response({ input: 20_000, cost: COST({ input: 0.2 }) }),
      "claude",
      "claude-sonnet-4-5",
      1_700_000_001_000,
      priceSource(3),
    );
    assert.ok(miss);
    // paidPerToken = 1e-5, readPerToken = 3e-6 -> 20k * 7e-6 = 0.14
    assert.equal(miss.missedCost.toFixed(2), "0.14");
    assert.equal(
      formatBranchSummaryCacheMissNotice(miss),
      "Cache miss: 20k tokens re-billed (~$0.14)",
    );
  });

  it("suppresses a miss after a model switch", () => {
    const entries = [
      usageEntry(
        { input: 0, cacheRead: 20_000, cacheWrite: 0 },
        { provider: "openai", model: "gpt-5" },
      ),
    ];
    const miss = detectBranchSummaryCacheMiss(
      entries,
      response({ input: 20_000, cost: COST({ input: 0.2 }) }),
      "claude",
      "claude-sonnet-4-5",
      1_700_000_001_000,
      priceSource(),
    );
    assert.equal(miss, undefined);
  });

  it("suppresses hits and noise-floor misses", () => {
    // Warm read: no missed tokens.
    const warm = detectBranchSummaryCacheMiss(
      [usageEntry({ input: 0, cacheRead: 20_000, cacheWrite: 0 })],
      response({ input: 100, cacheRead: 20_000, cost: COST({ input: 0.001 }) }),
      "claude",
      "claude-sonnet-4-5",
      1_700_000_001_000,
      priceSource(),
    );
    assert.equal(warm, undefined);
    // Miss at/below the 1024-token noise floor.
    const tiny = detectBranchSummaryCacheMiss(
      [usageEntry({ input: 0, cacheRead: 50_000, cacheWrite: 0 })],
      response({ input: 1030, cacheRead: 49_000, cost: COST({ input: 0.01 }) }),
      "claude",
      "claude-sonnet-4-5",
      1_700_000_001_000,
      priceSource(),
    );
    assert.equal(tiny, undefined);
  });

  it("returns undefined with no baseline request", () => {
    const miss = detectBranchSummaryCacheMiss(
      [userEntry("hi")],
      response({ input: 20_000, cost: COST({ input: 0.2 }) }),
      "claude",
      "claude-sonnet-4-5",
      1_700_000_001_000,
      priceSource(),
    );
    assert.equal(miss, undefined);
  });

  it("keeps the baseline across branch_summary entries", () => {
    const entries = [
      usageEntry({ input: 0, cacheRead: 20_000, cacheWrite: 0 }),
      branchSummaryEntry(),
    ];
    const miss = detectBranchSummaryCacheMiss(
      entries,
      response({ input: 20_000, cost: COST({ input: 0.2 }) }),
      "claude",
      "claude-sonnet-4-5",
      1_700_000_001_000,
      priceSource(),
    );
    assert.ok(miss, "branch_summary must not reset the baseline");
  });

  it("counts a zero-cache miss after a compaction boundary when earlier activity proved caching (session-scoped capability)", () => {
    // `everReportedCache` is session-scoped: a compaction resets the prompt
    // baseline (`prev`) but must NOT reset the provider-cache capability —
    // otherwise a total miss on a cache-read-only provider would be mistaken
    // for a provider that never reports caching at all. Port of the fork's
    // `cache-stats.ts` scan; without the capability flag this test's response
    // (zero cache read AND zero cache write) would be silently ignored.
    const entries = [
      usageEntry({ input: 0, cacheRead: 20_000, cacheWrite: 0 }),
      compactionEntry("cut"),
      usageEntry({ input: 20_000, cacheRead: 0, cacheWrite: 0 }),
    ];
    const miss = detectBranchSummaryCacheMiss(
      entries,
      response({ input: 20_000, cost: COST({ input: 0.2 }) }),
      "claude",
      "claude-sonnet-4-5",
      1_700_000_001_000,
      priceSource(),
    );
    assert.ok(miss, "compaction must reset the baseline, not the capability");
    assert.equal(miss.missedTokens, 20_000);
  });

  it("renders the idle label once the gap spans the cache TTL", () => {
    const base = 1_700_000_000_000;
    const entries = [
      usageEntry(
        { input: 0, cacheRead: 20_000, cacheWrite: 0 },
        { timestamp: base },
      ),
    ];
    const miss = detectBranchSummaryCacheMiss(
      entries,
      response({ input: 20_000, cost: COST({ input: 0.2 }) }),
      "claude",
      "claude-sonnet-4-5",
      base + CACHE_TTL_MS + 60_000,
      priceSource(),
    );
    assert.ok(miss);
    assert.equal(
      formatBranchSummaryCacheMissNotice(miss),
      "Cache miss after 6m idle: 20k tokens re-billed (~$0.20)",
    );
  });

  it("omits the cost below one cent and stays silent below the display floor", () => {
    const entries = [
      usageEntry({ input: 0, cacheRead: 20_000, cacheWrite: 0 }),
    ];
    const noCost = detectBranchSummaryCacheMiss(
      entries,
      response({ input: 20_000, cost: COST({ input: 0.001 }) }),
      "claude",
      "claude-sonnet-4-5",
      1_700_000_001_000,
      priceSource(),
    );
    assert.ok(noCost);
    assert.equal(
      formatBranchSummaryCacheMissNotice(noCost),
      "Cache miss: 20k tokens re-billed",
    );

    // Below the 20k-token floor and below $0.10 -> no notice.
    const below = detectBranchSummaryCacheMiss(
      [usageEntry({ input: 0, cacheRead: 50_000, cacheWrite: 0 })],
      response({ input: 15_000, cost: COST() }),
      "claude",
      "claude-sonnet-4-5",
      1_700_000_001_000,
      priceSource(),
    );
    assert.ok(below);
    assert.equal(formatBranchSummaryCacheMissNotice(below), null);
  });

  it("shows a cost-only miss above the dollar floor", () => {
    const miss = detectBranchSummaryCacheMiss(
      [usageEntry({ input: 0, cacheRead: 20_000, cacheWrite: 0 })],
      response({ input: 10_000, cost: COST({ input: 0.5 }) }),
      "claude",
      "claude-sonnet-4-5",
      1_700_000_001_000,
      priceSource(),
    );
    assert.ok(miss);
    assert.ok(miss.missedTokens < CACHE_MISS_DISPLAY_TOKENS);
    assert.ok(miss.missedCost >= CACHE_MISS_DISPLAY_COST);
    assert.equal(
      formatBranchSummaryCacheMissNotice(miss),
      "Cache miss: 10k tokens re-billed (~$0.50)",
    );
  });
});

// =============================================================================
// createCachePreservingStreamFn
// =============================================================================

interface Captured {
  model: unknown;
  context: {
    systemPrompt?: string;
    messages: WireMessage[];
    tools?: unknown[];
  };
  options: Record<string, unknown> | undefined;
}

function capturingRealStreamFn() {
  const calls: Captured[] = [];
  const realStreamFn = (async (
    model: unknown,
    context: Captured["context"],
    options: Record<string, unknown> | undefined,
  ) => {
    calls.push({ model, context, options });
    return { result: async () => fakeAssistantResponse() };
  }) as never;
  return { calls, realStreamFn };
}

function makeCacheRequest(): CacheRequest {
  const entries: SessionEntry[] = [
    userEntry("branch work"),
    assistantTextEntry("done"),
  ];
  const built = buildLiveSummaryMessages({
    contextEntries: entries,
    branchEntryIds: new Set(entries.map((e) => e.id)),
    inFlightToolCallId: "tc-rewind",
    tokenBudget: 0,
    focus: "preserve the parser API",
  });
  const tools = [
    { name: "read", description: "r", parameters: {} },
  ] as unknown as CacheRequest["context"]["tools"];
  return {
    context: {
      systemPrompt: "LIVE SYSTEM PROMPT",
      messages: built.messages,
      tools,
    },
    cacheRetention: "short" as const,
    sessionId: "live-session-id",
    reasoning: "high" as const,
    thinkingBudgets: { high: 8000 },
  };
}

describe("createCachePreservingStreamFn", () => {
  it("rewrites context + options to the live request shape", () => {
    const { calls, realStreamFn } = capturingRealStreamFn();
    const request = makeCacheRequest();
    const wrapped = createCachePreservingStreamFn({ realStreamFn, request });
    wrapped.streamFn({} as never, {} as never, {
      maxTokens: 2048,
      apiKey: "k",
      headers: { h: "1" },
      cacheRetention: "none",
      sessionId: "upstream-uuid",
    });
    assert.equal(wrapped.used.value, true);
    assert.equal(calls.length, 1);
    const captured = calls[0];
    assert.equal(captured.context.systemPrompt, "LIVE SYSTEM PROMPT");
    assert.equal(captured.context.messages, request.context.messages);
    assert.equal(captured.context.tools, request.context.tools);
    assert.equal(captured.options?.maxTokens, undefined);
    assert.equal(captured.options?.cacheRetention, "short");
    assert.equal(captured.options?.sessionId, "live-session-id");
    assert.equal(captured.options?.reasoning, "high");
    assert.deepEqual(captured.options?.thinkingBudgets, { high: 8000 });
    // Non-cache options survive.
    assert.equal(captured.options?.apiKey, "k");
    assert.deepEqual(captured.options?.headers, { h: "1" });
  });

  it("omits reasoning when off/undefined, sessionId when absent, budgets when unset", () => {
    const { calls, realStreamFn } = capturingRealStreamFn();
    const request = makeCacheRequest();
    request.reasoning = "off";
    delete (request as { sessionId?: unknown }).sessionId;
    delete (request as { thinkingBudgets?: unknown }).thinkingBudgets;
    const wrapped = createCachePreservingStreamFn({ realStreamFn, request });
    wrapped.streamFn({} as never, {} as never, { maxTokens: 4096 });
    const options = calls[0].options ?? {};
    assert.ok(!("reasoning" in options));
    assert.ok(!("sessionId" in options));
    assert.ok(!("thinkingBudgets" in options));
    assert.ok(!("maxTokens" in options));
  });

  it("delegates the cold context/options verbatim when request is null", () => {
    const { calls, realStreamFn } = capturingRealStreamFn();
    const wrapped = createCachePreservingStreamFn({
      realStreamFn,
      request: null,
    });
    const coldContext = { systemPrompt: "COLD", messages: [] };
    const coldOptions = { maxTokens: 2048, cacheRetention: "none" };
    wrapped.streamFn({} as never, coldContext as never, coldOptions as never);
    assert.equal(wrapped.used.value, false);
    assert.equal(calls[0].context, coldContext);
    assert.equal(calls[0].options, coldOptions);
  });
});

// =============================================================================
// Wrapper contract through the REAL upstream generateBranchSummary
// =============================================================================

const FAKE_SUMMARY_TEXT = "## Goal\nShip #33.\n## Progress\n### Done\nTests.";
const SUMMARIZATION_SYSTEM_PROMPT =
  "You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.\n\nDo NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.";
const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

function fakeAssistantResponse() {
  return {
    role: "assistant",
    content: [{ type: "text", text: FAKE_SUMMARY_TEXT }],
    api: "openai-responses",
    provider: "opencode-go",
    model: "muse-spark",
    stopReason: "stop",
    timestamp: 1_700_000_000_000,
    usage: {
      input: 420,
      output: 80,
      cacheRead: 20_000,
      cacheWrite: 0,
      totalTokens: 20_500,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
  };
}

function fakeModel() {
  return {
    id: "muse-spark",
    name: "muse-spark",
    api: "openai-responses",
    provider: "opencode-go",
    baseUrl: "https://opencode.ai",
    reasoning: true,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 200_000,
    maxTokens: 32_000,
  };
}

describe("createCachePreservingStreamFn through real generateBranchSummary", () => {
  it("forwards the live context + mirrored options into the provider stream", async () => {
    const entries: SessionEntry[] = [
      userEntry("branch work"),
      assistantTextEntry("done"),
    ];
    const { calls, realStreamFn } = capturingRealStreamFn();
    const request = makeCacheRequest();
    const wrapped = createCachePreservingStreamFn({ realStreamFn, request });

    const result = await generateBranchSummary(entries, {
      model: fakeModel() as never,
      apiKey: "test-key",
      headers: { "x-opencode-client": "pi" },
      env: { E: "1" },
      signal: new AbortController().signal,
      customInstructions: "preserve the parser API",
      streamFn: wrapped.streamFn,
    } as never);

    // The real upstream path built a cold context and passed it through the
    // wrapper, which swapped in the live request.
    assert.equal(wrapped.used.value, true);
    assert.equal(calls.length, 1);
    const captured = calls[0];
    assert.equal(captured.context.systemPrompt, request.context.systemPrompt);
    assert.equal(captured.context.messages, request.context.messages);
    assert.equal(captured.context.tools, request.context.tools);
    assert.equal(captured.options?.cacheRetention, "short");
    assert.equal(captured.options?.sessionId, "live-session-id");
    assert.equal(captured.options?.reasoning, "high");
    assert.deepEqual(captured.options?.thinkingBudgets, { high: 8000 });
    // maxTokens stripped — live turns let pi-ai clamp model.maxTokens itself.
    assert.ok(!("maxTokens" in (captured.options ?? {})));
    // Non-mirrored options survive.
    assert.equal(captured.options?.apiKey, "test-key");
    assert.deepEqual(captured.options?.headers, { "x-opencode-client": "pi" });
    assert.deepEqual(captured.options?.env, { E: "1" });

    // `.result()` contract: upstream unwraps the fake stream and returns the
    // canned summary + usage to the caller.
    assert.match(result.summary ?? "", /Ship #33/);
    assert.equal(result.usage?.cacheRead, 20_000);
  });

  it("delegates today's cold request when request is null (0.84.2 shape)", async () => {
    const entries: SessionEntry[] = [
      userEntry("branch work"),
      assistantTextEntry("done"),
    ];
    const { calls, realStreamFn } = capturingRealStreamFn();
    const wrapped = createCachePreservingStreamFn({
      realStreamFn,
      request: null,
    });
    await generateBranchSummary(entries, {
      model: fakeModel() as never,
      apiKey: "test-key",
      headers: {},
      signal: new AbortController().signal,
      customInstructions: "focus",
      streamFn: wrapped.streamFn,
    } as never);

    assert.equal(wrapped.used.value, false);
    const captured = calls[0];
    // Cold path: upstream's generic summarization prompt + serialized blob.
    assert.equal(captured.context.systemPrompt, SUMMARIZATION_SYSTEM_PROMPT);
    assert.equal(captured.context.messages.length, 1);
    assert.equal(captured.context.tools, undefined);
    // Upstream's output cap is version-dependent (0.84.2 = 2048, 0.85.1 =
    // 4096); assert only presence/type — a version-specific literal would
    // red on a pi bump (whereas the cache path asserts the cap is absent).
    assert.equal(typeof captured.options?.maxTokens, "number");
    assert.equal(captured.options?.cacheRetention, "none");
    assert.match(String(captured.options?.sessionId), UUID_RE);
    assert.ok(!("reasoning" in (captured.options ?? {})));
  });
});
