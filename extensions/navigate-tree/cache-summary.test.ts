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
  generateBranchSummary,
  type SessionEntry,
  type SessionMessageEntry,
} from "@earendil-works/pi-coding-agent";
import {
  BRANCH_SUMMARY_CACHE_PROMPT,
  buildSummaryInstruction,
  buildSummaryInstructionMessage,
  CACHE_MISS_DISPLAY_COST,
  CACHE_MISS_DISPLAY_TOKENS,
  CACHE_TTL_MS,
  type CacheRequest,
  createCachePreservingStreamFn,
  detectBranchSummaryCacheMiss,
  formatBranchSummaryCacheMissNotice,
  measureSummaryCache,
  resolveSummaryCacheRetention,
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

/**
 * A captured live request as `index.ts` builds it: the live head (prompt +
 * tool declarations in live order), the conversation, and the trailing
 * instruction. `toolsAdded` deliberately uses an order a session's
 * `state.tools` array would not have (grep before read) — the wrapper must
 * replay this array verbatim, never re-derive declarations.
 */
function makeCacheRequest(): CacheRequest {
  return {
    messages: [
      {
        role: "system",
        content: "LIVE SYSTEM PROMPT",
        toolsAdded: [
          { name: "grep", description: "g", parameters: {} },
          { name: "read", description: "r", parameters: {} },
        ],
      } as unknown as WireMessage,
      {
        role: "user",
        content: [{ type: "text", text: "branch work" }],
        timestamp: 1,
      } as WireMessage,
      buildSummaryInstructionMessage(2, "preserve the parser API"),
    ],
    cacheRetention: "short" as const,
    sessionId: "live-session-id",
    reasoning: "high" as const,
    thinkingBudgets: { high: 8000 },
  };
}

describe("buildSummaryInstructionMessage", () => {
  it("wraps the r5d instruction as a user message with {first} substituted", () => {
    const message = buildSummaryInstructionMessage(42, "finish the parser");
    assert.equal(message.role, "user");
    const blocks = Array.isArray(message.content) ? message.content : [];
    const text = blocks
      .filter((block) => block.type === "text")
      .map((block) => (block.type === "text" ? block.text : ""))
      .join("");
    assert.match(text, /Summarize only messages 42 onwards/);
    assert.match(text, /Additional focus: finish the parser/);
    assert.doesNotMatch(text, /\{first\}/);
    assert.equal(typeof message.timestamp, "number");
  });
});

describe("createCachePreservingStreamFn", () => {
  it("replays the captured messages verbatim and mirrors the live options", () => {
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
    // Identity: the captured array is delegated untouched. No
    // `systemPrompt`/`tools` fields — the captured leading system message is
    // the only prompt/tool source, so pi-ai's `normalizeContext` prepends
    // nothing.
    assert.equal(captured.context.messages, request.messages);
    assert.equal(captured.context.systemPrompt, undefined);
    assert.equal(captured.context.tools, undefined);
    assert.equal(captured.options?.maxTokens, undefined);
    assert.equal(captured.options?.cacheRetention, "short");
    assert.equal(captured.options?.sessionId, "live-session-id");
    assert.equal(captured.options?.reasoning, "high");
    assert.deepEqual(captured.options?.thinkingBudgets, { high: 8000 });
    // Non-cache options survive.
    assert.equal(captured.options?.apiKey, "k");
    assert.deepEqual(captured.options?.headers, { h: "1" });
  });

  it("keeps the captured head's tool order and message prefix (cache-parity regression)", () => {
    const request = makeCacheRequest();
    const headTools = (
      request.messages[0] as { toolsAdded?: Array<{ name: string }> }
    ).toolsAdded;
    assert.deepEqual(
      headTools?.map((tool) => tool.name),
      ["grep", "read"],
      "fixture head must declare a tool order a re-derivation would not produce",
    );
    const { calls, realStreamFn } = capturingRealStreamFn();
    const wrapped = createCachePreservingStreamFn({ realStreamFn, request });
    wrapped.streamFn({} as never, {} as never, {});
    const wire = calls[0].context.messages;
    assert.deepEqual(
      (wire[0] as { toolsAdded?: Array<{ name: string }> }).toolsAdded?.map(
        (tool) => tool.name,
      ),
      ["grep", "read"],
    );
    // Byte-identical prefix: everything apart from the trailing instruction
    // equals the capture (JSON is the serialization a serialized-prefix
    // provider cache keys on for same-value message arrays).
    assert.equal(
      JSON.stringify(wire.slice(0, -1)),
      JSON.stringify(request.messages.slice(0, -1)),
    );
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
    const coldContext = { messages: [] };
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

const FAKE_SUMMARY_TEXT = "## Goal\nShip #75.\n## Progress\n### Done\nTests.";
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
  it("forwards the captured messages + mirrored options into the provider stream", async () => {
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
    // wrapper, which swapped in the captured request.
    assert.equal(wrapped.used.value, true);
    assert.equal(calls.length, 1);
    const captured = calls[0];
    assert.equal(captured.context.messages, request.messages);
    assert.equal(captured.context.systemPrompt, undefined);
    assert.equal(captured.context.tools, undefined);
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
    assert.match(result.summary ?? "", /Ship #75/);
    assert.equal(result.usage?.cacheRead, 20_000);
  });

  it("delegates today's cold request when request is null (upstream shape)", async () => {
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
    // Cold path: upstream's own normalized context (leading generic
    // summarization system message + serialized conversation blob), no live
    // fields.
    assert.equal(captured.context.systemPrompt, undefined);
    assert.equal(captured.context.messages.length, 2);
    assert.equal(captured.context.messages[0].role, "system");
    assert.match(
      String(captured.context.messages[0].content),
      /summarization assistant/,
    );
    // Upstream's output cap is version-dependent; assert only presence/type
    // (the capture path asserts the cap is absent).
    assert.equal(typeof captured.options?.maxTokens, "number");
    assert.equal(captured.options?.cacheRetention, "none");
    assert.match(String(captured.options?.sessionId), UUID_RE);
    assert.ok(!("reasoning" in (captured.options ?? {})));
  });
});
