/**
 * Tests for navigate-tree dispatch + reflection bootstrap.
 *
 * Bun test runner. Run with: bun test
 *
 * The pi extension loader treats `./index.ts` as the entry point and ignores
 * sibling files \u2014 so this test file is not loaded as a separate extension.
 *
 * Test architecture:
 *   - Each test gets a fresh in-memory SessionManager (no fs writes).
 *   - A minimal fake `ExtensionAPI` captures the registered tool and routes
 *     `setLabel` calls into `sm.appendLabelChange` so `sm.getLabel` reflects
 *     state set by the tool.
 *   - Tests that exercise reflection install a fake AgentSession-shaped
 *     object on the module-internal `sessionInstances` array via
 *     `__testHooks.captureSession` (the prototype patch is restored between
 *     tests via `__testHooks.resetPrototype()` in `afterEach`).
 *   - The `summarize` DI seam injects a stub for `generateBranchSummary` so
 *     no real LLM call fires during rewind tests.
 */

import { afterEach, describe, it } from "bun:test";
import * as assert from "node:assert/strict";
import {
  type AgentSession,
  type ExtensionAPI,
  type SessionEntry,
  SessionManager,
} from "@earendil-works/pi-coding-agent";
import navigateTree, { __testHooks } from "./index.ts";

afterEach(() => {
  __testHooks.resetPrototype();
});

// -----------------------------------------------------------------------------
// Fakes
// -----------------------------------------------------------------------------

interface CapturedTool {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  executionMode?: string;
  execute: (
    toolCallId: string,
    params: unknown,
    signal: AbortSignal | undefined,
    onUpdate: unknown,
    ctx: unknown,
  ) => Promise<{
    content: Array<{ type: "text"; text: string }>;
    details: Record<string, unknown>;
    isError?: boolean;
  }>;
  [k: string]: unknown;
}

interface FakePi {
  pi: ExtensionAPI;
  setLabelCalls: Array<[string, string | undefined]>;
  registered: CapturedTool[];
}

/**
 * Build a minimal fake ExtensionAPI. `registerTool` captures the registered
 * tool definition; `setLabel` writes through to the SessionManager so
 * `sm.getLabel(...)` reflects the current state. Other methods throw on
 * access \u2014 the tool's `execute` only touches `setLabel`, so unknown calls
 * indicate a contract drift the tests want to surface.
 */
function makeFakePi(sm: SessionManager): FakePi {
  const setLabelCalls: Array<[string, string | undefined]> = [];
  const registered: CapturedTool[] = [];
  const pi = {
    registerTool(tool: CapturedTool) {
      registered.push(tool);
    },
    setLabel(entryId: string, label: string | undefined) {
      setLabelCalls.push([entryId, label]);
      // Route the label through the SessionManager so reads via
      // `sm.getLabel(...)` reflect the state the tool just set. Production
      // pi does this via the ExtensionRunner; in tests we shortcut.
      sm.appendLabelChange(entryId, label);
    },
  } as unknown as ExtensionAPI;
  return { pi, setLabelCalls, registered };
}

interface FakeCtx {
  sessionManager: SessionManager;
  model:
    | {
        api: string;
        provider: string;
        id: string;
        contextWindow?: number;
      }
    | undefined;
  modelRegistry: {
    getApiKeyAndHeaders(
      _model: unknown,
    ): Promise<
      | { ok: true; apiKey: string; headers: Record<string, string> }
      | { ok: false; error: string }
    >;
  };
}

function makeCtx(
  sm: SessionManager,
  opts: {
    contextWindow?: number;
    noModel?: boolean;
    authError?: string;
  } = {},
): FakeCtx {
  const model = opts.noModel
    ? undefined
    : {
        api: "anthropic",
        provider: "claude",
        id: "claude-sonnet-4-5",
        contextWindow: opts.contextWindow ?? 1_000_000,
      };
  return {
    sessionManager: sm,
    model,
    modelRegistry: {
      async getApiKeyAndHeaders(_m: unknown) {
        if (opts.authError) return { ok: false, error: opts.authError };
        return { ok: true, apiKey: "test-key", headers: {} };
      },
    },
  };
}

/**
 * Append a complete user/assistant turn so the chain has a labelable assistant
 * leaf with usage. Returns the assistant entry id so the caller can label it.
 */
function appendTurn(
  sm: SessionManager,
  userText: string,
  assistantText: string,
  totalTokens = 100,
): { userId: string; assistantId: string } {
  const userId = sm.appendMessage({
    role: "user",
    content: [{ type: "text", text: userText }],
  } as never);
  const assistantId = sm.appendMessage({
    role: "assistant",
    content: [{ type: "text", text: assistantText }],
    api: "anthropic",
    provider: "claude",
    model: "claude-sonnet-4-5",
    stopReason: "endTurn",
    timestamp: Date.now(),
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
  } as never);
  return { userId, assistantId };
}

/**
 * Stub for `generateBranchSummary`. Returns a fixed structured summary so the
 * rewind happy path can run without an LLM call. Matches the SDK's
 * BranchSummaryResult shape.
 */
async function fakeSummarize() {
  return {
    summary:
      "## Goal\nTest the rewind happy path.\n## Done\nAppended turns.\n## Next Steps\nVerify the synthetic.",
    readFiles: [] as string[],
    modifiedFiles: [] as string[],
    aborted: false,
  };
}

/**
 * Register the navigate-tree tool against a fresh fake pi + SessionManager
 * and return the captured tool plus the helpers tests need.
 */
function setup(
  opts: {
    contextWindow?: number;
    noModel?: boolean;
    authError?: string;
    summarize?: typeof fakeSummarize;
  } = {},
): {
  sm: SessionManager;
  pi: FakePi;
  ctx: FakeCtx;
  tool: CapturedTool;
} {
  const sm = SessionManager.inMemory("/tmp");
  const pi = makeFakePi(sm);
  navigateTree(pi.pi, {
    summarize: (opts.summarize ?? fakeSummarize) as never,
  });
  assert.equal(pi.registered.length, 1);
  const tool = pi.registered[0];
  const ctx = makeCtx(sm, opts);
  return { sm, pi, ctx, tool };
}

/**
 * Build a fake AgentSession-shaped object with the mutable `agent.state`
 * surface the reflection bootstrap touches. Cast through unknown so the test
 * doesn't depend on every field of the real class.
 */
interface FakeAgentSession {
  sessionManager: SessionManager;
  agent: {
    state: { systemPrompt: string; messages: unknown[]; tools: unknown[] };
    prepareNextTurn?: unknown;
  };
}

function makeFakeSession(sm: SessionManager): FakeAgentSession {
  return {
    sessionManager: sm,
    agent: {
      state: { systemPrompt: "S", messages: [], tools: [] },
      prepareNextTurn: undefined,
    },
  };
}

// =============================================================================
// Schema shape (Kiro compatibility)
// =============================================================================

describe("schema shape \u2014 Kiro compatibility (LD-4)", () => {
  it("registers a tool with a flat object root and `action` required", () => {
    // The Kiro/CodeWhisperer adapter forwards inputSchema.json verbatim and
    // 400s on non-`type: "object"` roots (anyOf/oneOf/discriminated unions).
    // Pin the working shape so a future TypeBox refactor surfaces here.
    const { tool } = setup();
    const params = tool.parameters as {
      type: string;
      required?: string[];
      anyOf?: unknown;
      oneOf?: unknown;
    };
    assert.equal(params.type, "object");
    assert.ok(params.required?.includes("action"), "action must be required");
    assert.equal(params.anyOf, undefined);
    assert.equal(params.oneOf, undefined);
  });
});

// =============================================================================
// buildSyntheticAssistant shape (LD-1)
// =============================================================================

describe("buildSyntheticAssistant shape (LD-1)", () => {
  it("builds an assistant message with toolUse stop reason and zero cost", () => {
    // The synthetic must (a) carry a tool_call id matching the in-flight call,
    // (b) survive Kiro's normalizeMessages filter (stopReason must not be
    // 'error' / 'aborted'), (c) report zero cost so the TUI footer doesn't
    // double-count, (d) propagate model fields used for downstream display.
    const m = __testHooks.buildSyntheticAssistant(
      "call-123",
      "navigate_tree",
      { action: "rewind" },
      { api: "anthropic", provider: "claude", id: "claude-sonnet-4-5" },
      12_345,
    );
    assert.equal(m.role, "assistant");
    assert.equal(m.stopReason, "toolUse");
    assert.equal(m.content[0].type, "toolCall");
    assert.equal(m.content[0].id, "call-123");
    assert.equal(m.content[0].name, "navigate_tree");
    assert.deepEqual(m.content[0].arguments, { action: "rewind" });
    assert.equal(m.usage.totalTokens, 12_345);
    assert.equal(m.usage.cost.total, 0);
    assert.equal(m.usage.input, 0);
    assert.equal(m.usage.output, 0);
    assert.equal(m.api, "anthropic");
    assert.equal(m.provider, "claude");
    assert.equal(m.model, "claude-sonnet-4-5");
  });

  it("falls back to 'unknown' for api/provider/model when the model is undefined", () => {
    const m = __testHooks.buildSyntheticAssistant(
      "call-123",
      "navigate_tree",
      {},
      undefined,
      0,
    );
    assert.equal(m.api, "unknown");
    assert.equal(m.provider, "unknown");
    assert.equal(m.model, "unknown");
  });
});

// =============================================================================
// dispatch: list
// =============================================================================

describe("dispatch: list action", () => {
  it("returns 'No labels' on an empty session", async () => {
    const { tool, ctx } = setup();
    const result = await tool.execute(
      "tc-1",
      { action: "list" },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assert.match(result.content[0].text, /No labels on the active branch\./);
    assert.equal(result.details.count, 0);
  });

  it("renders anchored labels in chronological order", async () => {
    const { sm, pi, tool, ctx } = setup();
    const t1 = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(t1.assistantId, "anchor:first");
    const t2 = appendTurn(sm, "u2", "a2", 200);
    pi.pi.setLabel(t2.assistantId, "anchor:second");

    const result = await tool.execute(
      "tc-1",
      { action: "list" },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assert.equal(result.details.count, 2);
    const text = result.content[0].text;
    // Chronological order: 'first' appears before 'second' in the text.
    const firstIdx = text.indexOf("first");
    const secondIdx = text.indexOf("second");
    assert.ok(firstIdx >= 0 && secondIdx > firstIdx);
  });

  it("omits ' of <window>' from the header when contextWindow is 0", async () => {
    const { sm, tool, ctx } = setup({ contextWindow: 0 });
    appendTurn(sm, "u", "a");
    const result = await tool.execute(
      "tc-1",
      { action: "list" },
      undefined,
      undefined,
      ctx,
    );
    // Header includes context as 'Nk' fallback but no ' of 1.0M' tail.
    const header = result.content[0].text.split("\n")[0];
    assert.ok(!header.includes(" of "), `header had ' of ': ${header}`);
  });

  it("includes the reflection-bootstrap warning when no captured session", async () => {
    // No __testHooks.captureSession() called \u2014 reflection finds nothing.
    const { tool, ctx } = setup();
    const result = await tool.execute(
      "tc-1",
      { action: "list" },
      undefined,
      undefined,
      ctx,
    );
    assert.match(result.content[0].text, /reflection bootstrap missing/);
    assert.equal(result.details.reflectionOk, false);
  });

  it("clears the reflection warning when the owning session is captured", async () => {
    const { sm, tool, ctx } = setup();
    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);
    const result = await tool.execute(
      "tc-1",
      { action: "list" },
      undefined,
      undefined,
      ctx,
    );
    assert.doesNotMatch(result.content[0].text, /reflection bootstrap missing/);
    assert.equal(result.details.reflectionOk, true);
  });
});

// =============================================================================
// dispatch: anchor
// =============================================================================

describe("dispatch: anchor action", () => {
  const invalidNames: Array<[string, unknown]> = [
    ["empty string", ""],
    ["undefined", undefined],
    ["uppercase", "Impl-Start"],
    ["snake_case", "snake_case"],
    ["space", "with space"],
    ["leading hyphen", "-leading"],
    ["trailing hyphen", "trailing-"],
    ["double hyphen", "a--b"],
    ["over max length", "a".repeat(41)],
  ];
  for (const [label, value] of invalidNames) {
    it(`rejects invalid name (${label}) with kebab-case message`, async () => {
      const { sm, tool, ctx } = setup();
      appendTurn(sm, "u", "a");
      const result = await tool.execute(
        "tc-1",
        { action: "anchor", name: value },
        undefined,
        undefined,
        ctx,
      );
      assert.equal(result.isError, true);
      assert.match(result.content[0].text, /kebab-case/);
    });
  }

  it("errors when the session has no entries yet", async () => {
    // Fresh inMemory SM \u2014 no appendMessage \u2014 leafId is null.
    const { tool, ctx } = setup();
    const result = await tool.execute(
      "tc-1",
      { action: "anchor", name: "iter-start" },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /No session entries yet/);
  });

  it("happy path: setLabel called with the prefixed label and details surface entryId", async () => {
    const { sm, pi, tool, ctx } = setup();
    const t = appendTurn(sm, "u", "a");
    const result = await tool.execute(
      "tc-1",
      { action: "anchor", name: "impl-start" },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    // pi.setLabel called with prefixed label on the leaf.
    const labelCall = pi.setLabelCalls.find(
      ([, lbl]) => lbl === "anchor:impl-start",
    );
    assert.ok(labelCall, "setLabel was not called with anchor:impl-start");
    assert.equal(labelCall?.[0], t.assistantId);
    assert.equal(result.details.label, "impl-start");
    assert.equal(result.details.entryId, t.assistantId);
  });

  it("move-on-collision: re-anchoring the same name moves the label off the prior entry", async () => {
    // First anchor at the leaf, append more turns, then re-anchor with the
    // same name. The prior label should be cleared and the new leaf labeled.
    const { sm, pi, tool, ctx } = setup();
    const t1 = appendTurn(sm, "u1", "a1");
    await tool.execute(
      "tc-1",
      { action: "anchor", name: "iter-start" },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(sm.getLabel(t1.assistantId), "anchor:iter-start");

    const t2 = appendTurn(sm, "u2", "a2");
    const result = await tool.execute(
      "tc-2",
      { action: "anchor", name: "iter-start" },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assert.equal(sm.getLabel(t1.assistantId), undefined);
    assert.equal(sm.getLabel(t2.assistantId), "anchor:iter-start");
    assert.equal(result.details.movedFromPriorEntry, t1.assistantId);
    // Verify the move actually went through pi.setLabel as a clear + set
    // pair (not a single in-place move).
    const clearCall = pi.setLabelCalls.find(
      ([id, lbl]) => id === t1.assistantId && lbl === undefined,
    );
    assert.ok(clearCall, "expected a clear of the prior label");
  });
});

// =============================================================================
// dispatch: rewind validation guards
// =============================================================================

describe("dispatch: rewind validation guards", () => {
  // Each row produces an isError=true result whose text matches the regex.
  // The order matches the guard order in execute(): labelStart, labelEnd,
  // summaryFocus, then label-existence.
  const cases: Array<{
    name: string;
    params: Record<string, unknown>;
    want: RegExp;
  }> = [
    {
      name: "missing labelStart",
      params: { action: "rewind" },
      want: /labelStart.*kebab-case/,
    },
    {
      name: "labelStart present, missing labelEnd",
      params: { action: "rewind", labelStart: "ok" },
      want: /labelEnd.*kebab-case/,
    },
    {
      name: "labelStart and labelEnd valid, missing summaryFocus",
      params: { action: "rewind", labelStart: "ok", labelEnd: "ok2" },
      want: /summaryFocus/,
    },
    {
      name: "summaryFocus shorter than min length",
      params: {
        action: "rewind",
        labelStart: "ok",
        labelEnd: "ok2",
        summaryFocus: "x".repeat(19), // MIN_SUMMARY_FOCUS_LENGTH = 20
      },
      want: /summaryFocus/,
    },
    {
      name: "summaryFocus shorter after trim",
      params: {
        action: "rewind",
        labelStart: "ok",
        labelEnd: "ok2",
        summaryFocus: `  ${"x".repeat(19)}  `,
      },
      want: /summaryFocus/,
    },
    {
      name: "summaryFocus mentions the 20-char threshold in error text",
      params: {
        action: "rewind",
        labelStart: "ok",
        labelEnd: "ok2",
        summaryFocus: "short",
      },
      want: /\u226520/, // \u22650
    },
    {
      name: "labelStart kebab-invalid (uppercase)",
      params: { action: "rewind", labelStart: "Bad-Name" },
      want: /labelStart.*kebab-case/,
    },
    {
      name: "labelEnd kebab-invalid (snake_case)",
      params: { action: "rewind", labelStart: "ok", labelEnd: "snake_case" },
      want: /labelEnd.*kebab-case/,
    },
    {
      name: "all valid but no such labelStart on the active branch",
      params: {
        action: "rewind",
        labelStart: "missing",
        labelEnd: "after",
        summaryFocus: "x".repeat(20),
      },
      want: /No label 'missing'/,
    },
  ];
  for (const c of cases) {
    it(c.name, async () => {
      const { sm, tool, ctx } = setup();
      // Append a turn so the no-such-label case can pass earlier guards
      // and reach the label-existence check.
      appendTurn(sm, "u", "a");
      const result = await tool.execute(
        "tc-1",
        c.params,
        undefined,
        undefined,
        ctx,
      );
      assert.equal(result.isError, true);
      assert.match(result.content[0].text, c.want);
    });
  }

  it("summaryFocus exactly 20 chars after trim passes the guard", async () => {
    // Boundary: summaryFocus.trim().length >= 20 (the >= boundary). The
    // call still errors at the next guard (no labelStart on chain), but it
    // moves past the focus-length check.
    const { sm, tool, ctx } = setup();
    appendTurn(sm, "u", "a");
    const result = await tool.execute(
      "tc-1",
      {
        action: "rewind",
        labelStart: "missing",
        labelEnd: "after",
        summaryFocus: "x".repeat(20),
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, true);
    // Past the focus guard, into the label-existence guard.
    assert.match(result.content[0].text, /No label 'missing'/);
  });
});

// =============================================================================
// dispatch: rewind happy path
// =============================================================================

describe("dispatch: rewind happy path", () => {
  it("collapses the chain, labels the summary, injects a synthetic, and refreshes", async () => {
    const { sm, pi, tool, ctx } = setup();
    // Build chain: t1 (anchor target) -> t2 -> t3 (leaf).
    const t1 = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendTurn(sm, "u2", "a2", 200);
    appendTurn(sm, "u3", "a3", 300);

    // Capture a fake session so reflection finds it.
    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        labelStart: "start",
        labelEnd: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );

    assert.equal(result.isError, undefined);
    // Response prose contract.
    assert.match(result.content[0].text, /\[rewind 'start' \u2192 'end'\]/);
    assert.match(result.content[0].text, /## Done/);
    assert.match(result.content[0].text, /## Next Steps/);
    // Details surface the contextBefore > contextAfter invariant.
    const before = result.details.contextBefore as number;
    const after = result.details.contextAfter as number;
    assert.ok(before > after, `expected before (${before}) > after (${after})`);
    // The new summary entry carries the labelEnd.
    const summaryId = result.details.summaryId as string;
    assert.equal(sm.getLabel(summaryId), "anchor:end");
    // The leaf is the synthetic assistant, NOT the branch_summary.
    const leafId = sm.getLeafId();
    assert.ok(leafId, "expected a leaf after rewind");
    assert.notEqual(leafId, summaryId);
    const leaf = sm.getEntry(leafId);
    assert.ok(leaf && leaf.type === "message");
    if (leaf.type === "message") {
      assert.equal(leaf.message.role, "assistant");
      const c0 = (
        leaf.message.content as Array<{ type: string; id?: string }>
      )[0];
      assert.equal(c0.type, "toolCall");
      assert.equal(c0.id, "tc-rewind");
    }
    // Reflection ran successfully, agent.state.messages was updated.
    assert.equal(result.details.agentMessagesRefreshed, true);
    assert.deepEqual(
      fake.agent.state.messages,
      sm.buildSessionContext().messages,
    );
  });

  it("reflection-failed path: warns in response and reports refreshed=false", async () => {
    const { sm, pi, tool, ctx } = setup();
    const t1 = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendTurn(sm, "u2", "a2", 200);
    // No captureSession call \u2014 reflection finds no owning session.

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        labelStart: "start",
        labelEnd: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assert.match(result.content[0].text, /Reflection failed/);
    assert.equal(result.details.agentMessagesRefreshed, false);
  });

  it("chained rewinds: a\u2192b then b\u2192c keeps prior labels and re-anchors the new summary", async () => {
    // Rewind A\u2192B leaves the leaf as a synthetic assistant whose parent is
    // the labeled branch_summary. The next rewind B\u2192C must find the
    // labeled summary (via getBranch walk-up), not the synthetic leaf.
    const { sm, pi, tool, ctx } = setup();
    const tA = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(tA.assistantId, "anchor:a");
    appendTurn(sm, "u2", "a2", 200);
    appendTurn(sm, "u3", "a3", 300);

    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);

    const r1 = await tool.execute(
      "tc-1",
      {
        action: "rewind",
        labelStart: "a",
        labelEnd: "b",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(r1.isError, undefined);
    const sumB = r1.details.summaryId as string;

    // Append more turns post-rewind.
    appendTurn(sm, "u4", "a4", 400);
    appendTurn(sm, "u5", "a5", 500);

    const r2 = await tool.execute(
      "tc-2",
      {
        action: "rewind",
        labelStart: "b",
        labelEnd: "c",
        summaryFocus: "Preserve user instructions across the second rewind.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(r2.isError, undefined);
    const sumC = r2.details.summaryId as string;
    // Final summary carries labelEnd.
    assert.equal(sm.getLabel(sumC), "anchor:c");
    // The b-summary still carries 'anchor:b'. (The b-summary isn't on the
    // active branch anymore \u2014 it's an ancestor of sumC in storage but the
    // active path goes through sumC. Verify via getEntry, which is
    // global-storage-keyed.)
    const bEntry = sm.getEntry(sumB);
    assert.ok(bEntry, "b-summary entry still present in storage");
    // Active leaf is the synthetic for the second rewind.
    const leafId = sm.getLeafId();
    assert.ok(leafId, "expected a leaf after second rewind");
    assert.notEqual(leafId, sumC);
    const leaf = sm.getEntry(leafId);
    if (leaf && leaf.type === "message") {
      const c0 = (
        leaf.message.content as Array<{ type: string; id?: string }>
      )[0];
      assert.equal(c0.id, "tc-2");
    }
  });
});

// =============================================================================
// installPrepareNextTurn (LD-2)
// =============================================================================

describe("installPrepareNextTurn (LD-2)", () => {
  it("basic: returns a context whose messages match buildSessionContext()", async () => {
    const sm = SessionManager.inMemory("/tmp");
    sm.appendMessage({
      role: "user",
      content: [{ type: "text", text: "hi" }],
    } as never);
    const fake = makeFakeSession(sm);
    __testHooks.installPrepareNextTurn(fake as unknown as AgentSession);
    const fn = fake.agent.prepareNextTurn as (
      signal?: AbortSignal,
    ) => Promise<{ context?: { messages: unknown[] } }>;
    const result = await fn();
    assert.deepEqual(
      result.context?.messages,
      sm.buildSessionContext().messages,
    );
    // Marker symbol is set so /reload re-installs unwrap correctly.
    const marked = fake.agent.prepareNextTurn as Record<symbol, unknown>;
    assert.equal(marked[__testHooks.PNT_MARKER], true);
  });

  it("chains over a prior non-marker prepareNextTurn from another extension", async () => {
    const sm = SessionManager.inMemory("/tmp");
    const fake = makeFakeSession(sm);
    let priorRan = false;
    fake.agent.prepareNextTurn = async () => {
      priorRan = true;
      return { model: "from-prior", thinkingLevel: "high" };
    };
    __testHooks.installPrepareNextTurn(fake as unknown as AgentSession);
    const fn = fake.agent.prepareNextTurn as (
      signal?: AbortSignal,
    ) => Promise<{ model?: unknown; thinkingLevel?: unknown }>;
    const result = await fn();
    assert.equal(priorRan, true);
    assert.equal(result.model, "from-prior");
    assert.equal(result.thinkingLevel, "high");
  });

  it("re-install (/reload) unwraps __prior; doesn't capture self as prior", async () => {
    const sm = SessionManager.inMemory("/tmp");
    const fake = makeFakeSession(sm);
    __testHooks.installPrepareNextTurn(fake as unknown as AgentSession);
    const after1 = fake.agent.prepareNextTurn;
    __testHooks.installPrepareNextTurn(fake as unknown as AgentSession);
    const after2 = fake.agent.prepareNextTurn as Record<
      string | symbol,
      unknown
    >;
    // New wrapper installed (different function reference).
    assert.notEqual(after2, after1);
    // The new wrapper's __prior is undefined: we recovered after1's
    // own __prior (which was undefined), not after1 itself.
    assert.equal(after2.__prior, undefined);
  });
});

// =============================================================================
// refreshAgentMessages (LD-3)
// =============================================================================

describe("refreshAgentMessages (LD-3)", () => {
  it("returns false when no session was captured", () => {
    const sm = SessionManager.inMemory("/tmp");
    assert.equal(__testHooks.refreshAgentMessages(sm), false);
  });

  it("returns false when agent.state is missing", () => {
    const sm = SessionManager.inMemory("/tmp");
    const fake = {
      sessionManager: sm,
      // agent present but state missing
      agent: { state: undefined, prepareNextTurn: undefined },
    };
    __testHooks.captureSession(fake as unknown as AgentSession);
    assert.equal(__testHooks.refreshAgentMessages(sm), false);
  });

  it("mutates agent.state.messages on success", () => {
    const sm = SessionManager.inMemory("/tmp");
    sm.appendMessage({
      role: "user",
      content: [{ type: "text", text: "hello" }],
    } as never);
    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);
    assert.equal(fake.agent.state.messages.length, 0);
    assert.equal(__testHooks.refreshAgentMessages(sm), true);
    assert.deepEqual(
      fake.agent.state.messages,
      sm.buildSessionContext().messages,
    );
  });
});

// =============================================================================
// captureSession reaping
// =============================================================================

describe("captureSession reaping", () => {
  it("deduplicates the same session and bounds growth across many distinct sessions", () => {
    // Register 100 short-lived sessions; the seenSessions WeakSet
    // dedupes within the same identity, and the sessionInstances array
    // gets reaped once it exceeds MAX_SESSION_REFS=16.
    //
    // Since WeakRef GC timing is non-deterministic, the bounded
    // assertion is the contract: the array doesn't grow proportional to
    // the number of pushes. (`Bun.gc(true)` is best-effort; if the
    // runtime doesn't expose it, we still get the dedupe + reap
    // bookkeeping.)
    for (let i = 0; i < 100; i++) {
      const sm = SessionManager.inMemory("/tmp");
      const fake = makeFakeSession(sm);
      __testHooks.captureSession(fake as unknown as AgentSession);
      // Drop our reference each iteration so the WeakRef can be reaped.
    }
    // Best-effort GC nudge.
    const g = (globalThis as { Bun?: { gc?: (sync: boolean) => void } }).Bun;
    if (typeof g?.gc === "function") g.gc(true);
    // Force the reaper to run by capturing one more session. The reaper
    // is gated inside `captureSession` on `length > MAX_SESSION_REFS` \u2014
    // it only runs at push time, never externally.
    const trailerSm = SessionManager.inMemory("/tmp");
    const trailerFake = makeFakeSession(trailerSm);
    __testHooks.captureSession(trailerFake as unknown as AgentSession);

    const refs = __testHooks.sessionRefCount();
    // Loose bound: the array shouldn't grow proportional to the number
    // of pushes once the reaper has fired.
    assert.ok(refs <= 101, `sessionRefCount=${refs} grew unbounded`);
    // Tighter bound when GC ran: post-reap, only live ref(s) remain.
    // (101 - 100 dead = 1, plus the trailer = at most ~2.)
    if (typeof g?.gc === "function") {
      assert.ok(
        refs <= 16,
        `sessionRefCount=${refs}; expected the reaper to bound it at MAX_SESSION_REFS=16`,
      );
    }
  });

  it("dedupes the same session on repeated capture", () => {
    const sm = SessionManager.inMemory("/tmp");
    const fake = makeFakeSession(sm);
    const before = __testHooks.sessionRefCount();
    __testHooks.captureSession(fake as unknown as AgentSession);
    const afterFirst = __testHooks.sessionRefCount();
    __testHooks.captureSession(fake as unknown as AgentSession);
    __testHooks.captureSession(fake as unknown as AgentSession);
    const afterRepeats = __testHooks.sessionRefCount();
    assert.equal(afterFirst - before, 1);
    assert.equal(afterRepeats, afterFirst);
  });
});

// =============================================================================
// findLabelHint depth limit
// =============================================================================

describe("findLabelHint", () => {
  it("returns null when no text is found within MAX_HINT_WALK_DEPTH (50)", () => {
    // Build a chain whose head is a custom_message with no text content,
    // followed by 100 user-text entries deep below \u2014 the walker should
    // give up at depth 50 and not reach the texts.
    const sm = SessionManager.inMemory("/tmp");
    // First push the deep texts.
    for (let i = 0; i < 100; i++) {
      sm.appendMessage({
        role: "user",
        content: [{ type: "text", text: `deep ${i}` }],
      } as never);
    }
    // Now push 60 thinking-only assistant messages on top so the walk
    // bottoms out before reaching the user texts.
    for (let i = 0; i < 60; i++) {
      sm.appendMessage({
        role: "assistant",
        content: [{ type: "thinking", thinking: "..." }],
        api: "anthropic",
        provider: "claude",
        model: "claude-sonnet-4-5",
        stopReason: "endTurn",
        timestamp: Date.now(),
        usage: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
      } as never);
    }
    const leafId = sm.getLeafId();
    assert.ok(leafId);
    const hint = __testHooks.findLabelHint(sm, leafId, 50);
    assert.equal(hint, null);
  });

  it("returns a preview when text is within the depth limit", () => {
    const sm = SessionManager.inMemory("/tmp");
    sm.appendMessage({
      role: "user",
      content: [{ type: "text", text: "the user's instruction" }],
    } as never);
    // 5 thinking entries on top \u2014 well within the 50-step cap.
    for (let i = 0; i < 5; i++) {
      sm.appendMessage({
        role: "assistant",
        content: [{ type: "thinking", thinking: "..." }],
        api: "anthropic",
        provider: "claude",
        model: "claude-sonnet-4-5",
        stopReason: "endTurn",
        timestamp: Date.now(),
        usage: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
      } as never);
    }
    const leafId = sm.getLeafId();
    assert.ok(leafId);
    const hint = __testHooks.findLabelHint(sm, leafId, 50);
    assert.ok(hint);
    if (hint) assert.match(hint, /the user's instruction/);
  });
});

// =============================================================================
// findLabeledEntry
// =============================================================================

describe("findLabeledEntry", () => {
  it("returns null when the label isn't on the active branch", () => {
    const sm = SessionManager.inMemory("/tmp");
    appendTurn(sm, "u", "a");
    assert.equal(__testHooks.findLabeledEntry(sm, "anchor:nope"), null);
  });

  it("returns the entry id when the label is set", () => {
    const sm = SessionManager.inMemory("/tmp");
    const t = appendTurn(sm, "u", "a");
    sm.appendLabelChange(t.assistantId, "anchor:found");
    assert.equal(
      __testHooks.findLabeledEntry(sm, "anchor:found"),
      t.assistantId,
    );
  });
});

// =============================================================================
// adversarial inputs
// =============================================================================

describe("dispatch: adversarial inputs", () => {
  it("100KB summaryFocus passes the length guard and reaches the no-such-label guard", async () => {
    // Pin behavior: no input-size cap on summaryFocus. The guard checks
    // a minimum, not a maximum. The test reaches the next guard
    // (no labelStart on active branch) without crashing.
    const { sm, tool, ctx } = setup();
    appendTurn(sm, "u", "a");
    const huge = "x".repeat(100_000);
    const result = await tool.execute(
      "tc-1",
      {
        action: "rewind",
        labelStart: "missing",
        labelEnd: "after",
        summaryFocus: huge,
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /No label 'missing'/);
  });

  it("labelEnd === labelStart passes name validation; surfaces as a 'no such label' error if not pre-anchored", async () => {
    // Both pass isValidName; pinning that the dispatch doesn't reject
    // labelStart === labelEnd up front. (Without a pre-set anchor, the
    // label-existence guard fires.)
    const { sm, tool, ctx } = setup();
    appendTurn(sm, "u", "a");
    const result = await tool.execute(
      "tc-1",
      {
        action: "rewind",
        labelStart: "same",
        labelEnd: "same",
        summaryFocus: "x".repeat(20),
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /No label 'same'/);
  });

  it("anchor over an existing label moves it (no duplicate)", async () => {
    // Already exercised in 'move-on-collision' above; pin the side-effect
    // shape adversarially here too: the same name set twice yields a
    // single live label on the active branch.
    const { sm, tool, ctx } = setup();
    const t1 = appendTurn(sm, "u1", "a1");
    await tool.execute(
      "tc-1",
      { action: "anchor", name: "dup" },
      undefined,
      undefined,
      ctx,
    );
    appendTurn(sm, "u2", "a2");
    await tool.execute(
      "tc-2",
      { action: "anchor", name: "dup" },
      undefined,
      undefined,
      ctx,
    );
    // Walk the branch and count entries labeled 'anchor:dup'.
    const branch: SessionEntry[] = sm.getBranch();
    const labeled = branch.filter((e) => sm.getLabel(e.id) === "anchor:dup");
    assert.equal(labeled.length, 1);
    assert.notEqual(labeled[0].id, t1.assistantId);
  });

  it("rewind without auth fails fast with a clear error", async () => {
    const { sm, pi, tool, ctx } = setup({ authError: "no api key" });
    const t1 = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendTurn(sm, "u2", "a2", 200);
    const result = await tool.execute(
      "tc-1",
      {
        action: "rewind",
        labelStart: "start",
        labelEnd: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /Auth resolution failed: no api key/);
  });

  it("rewind without a configured model fails with a clear error", async () => {
    const { sm, pi, tool, ctx } = setup({ noModel: true });
    const t1 = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendTurn(sm, "u2", "a2", 200);
    const result = await tool.execute(
      "tc-1",
      {
        action: "rewind",
        labelStart: "start",
        labelEnd: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /No model configured/);
  });
});

// =============================================================================
// salvage path (post-branchWithSummary failures)
// =============================================================================

/**
 * Wrap an existing `pi.setLabel` so the Nth call (1-indexed) throws. Earlier
 * calls go through to the original. Used to inject a throw on the rewind's
 * labelEnd write (the second setLabel call after the anchor write).
 */
function throwOnNthSetLabel(pi: FakePi, n: number, err: Error): void {
  const orig = pi.pi.setLabel.bind(pi.pi);
  let count = 0;
  (pi.pi as { setLabel: typeof pi.pi.setLabel }).setLabel = (
    entryId: string,
    label: string | undefined,
  ) => {
    count++;
    if (count === n) throw err;
    return orig(entryId, label);
  };
}

/**
 * Wrap `sm.appendMessage` so the next call throws. Used to inject a throw on
 * the synthetic append. Returns a restore function.
 */
function throwOnNextAppendMessage(sm: SessionManager, err: Error): () => void {
  const orig = sm.appendMessage.bind(sm);
  let armed = true;
  (sm as { appendMessage: typeof sm.appendMessage }).appendMessage = ((
    msg: never,
  ) => {
    if (armed) {
      armed = false;
      throw err;
    }
    return orig(msg);
  }) as typeof sm.appendMessage;
  return () => {
    (sm as { appendMessage: typeof sm.appendMessage }).appendMessage = orig;
  };
}

describe("dispatch: rewind salvage path", () => {
  it("setLabel(labelEnd) throws \u2192 synthetic still appended; original error wraps salvage detail", async () => {
    const { sm, pi, tool, ctx } = setup();
    const t1 = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendTurn(sm, "u2", "a2", 200);
    appendTurn(sm, "u3", "a3", 300);

    // Capture so refresh path runs.
    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);

    // We patch AFTER the anchor write above, so the patch counter starts
    // at 0. The first patched call is the rewind's labelEnd write; the
    // second is the salvage retry. Throw on every call so BOTH the
    // original write and the retry fail, surfacing the salvage detail in
    // the wrapped error.
    let calls = 0;
    const origSetLabel = pi.pi.setLabel.bind(pi.pi);
    void origSetLabel; // kept for parity with the helper used elsewhere
    (pi.pi as { setLabel: typeof pi.pi.setLabel }).setLabel = (
      _entryId: string,
      _label: string | undefined,
    ) => {
      calls++;
      throw new Error(`setLabel boom #${calls}`);
    };

    let thrown: unknown;
    try {
      await tool.execute(
        "tc-rewind",
        {
          action: "rewind",
          labelStart: "start",
          labelEnd: "end",
          summaryFocus: "Preserve user instructions and continue.",
        },
        undefined,
        undefined,
        ctx,
      );
    } catch (e) {
      thrown = e;
    }
    assert.ok(thrown, "rewind should re-throw the original error");
    const msg = thrown instanceof Error ? thrown.message : String(thrown);
    assert.match(msg, /setLabel boom #1/);
    // Salvage detail (CORR2-3 + SB2-9) wraps the original: the retry
    // (#2) ALSO threw, so the salvage-failure clause is appended.
    assert.match(msg, /salvage:.*labelEnd retry failed/);
    assert.equal(calls, 2, "setLabel was called twice (original + retry)");
    // Synthetic landed on the chain so pi's tool_result will pair correctly.
    const leafId = sm.getLeafId();
    assert.ok(leafId);
    const leaf = sm.getEntry(leafId as string);
    assert.ok(leaf && leaf.type === "message");
    if (leaf && leaf.type === "message") {
      assert.equal(leaf.message.role, "assistant");
      const c0 = (
        leaf.message.content as Array<{ type: string; id?: string }>
      )[0];
      assert.equal(c0.type, "toolCall");
      assert.equal(c0.id, "tc-rewind");
    }
    // Refresh ran in the salvage path: agent.state.messages was mutated.
    assert.deepEqual(
      fake.agent.state.messages,
      sm.buildSessionContext().messages,
    );
  });

  it("setLabel throws then retry succeeds \u2192 no salvage detail in re-thrown error", async () => {
    const { sm, pi, tool, ctx } = setup();
    const t1 = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendTurn(sm, "u2", "a2", 200);

    // Throw on call #1 (original labelEnd write). The salvage retry runs
    // as call #2 and succeeds, so no salvage detail in the wrapped error.
    throwOnNthSetLabel(pi, 1, new Error("transient setLabel boom"));

    let thrown: unknown;
    try {
      await tool.execute(
        "tc-rewind",
        {
          action: "rewind",
          labelStart: "start",
          labelEnd: "end",
          summaryFocus: "Preserve user instructions and continue.",
        },
        undefined,
        undefined,
        ctx,
      );
    } catch (e) {
      thrown = e;
    }
    assert.ok(thrown);
    const msg = thrown instanceof Error ? thrown.message : String(thrown);
    assert.match(msg, /transient setLabel boom/);
    // Retry succeeded \u2014 no salvage detail.
    assert.ok(
      !/salvage:/.test(msg),
      `expected no salvage detail when retry succeeds; got: ${msg}`,
    );
    // labelEnd was eventually written, so a chained rewind could find it.
    const summaryLabel = "anchor:end";
    let foundLabelEnd = false;
    for (const e of sm.getBranch()) {
      if (sm.getLabel(e.id) === summaryLabel) {
        foundLabelEnd = true;
        break;
      }
    }
    assert.equal(foundLabelEnd, true, "labelEnd retry should have written");
  });

  it("estimateActiveBranchTokens throws \u2192 synthetic still appended with degenerate token count", async () => {
    const { sm, pi, tool, ctx } = setup();
    const t1 = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendTurn(sm, "u2", "a2", 200);

    // Make buildSessionContext throw \u2014 estimateActiveBranchTokens calls it.
    // Throw only AFTER the branchWithSummary has succeeded by counting calls.
    // The earliest in-flow caller is `beforeTokens` math (which uses
    // estimateAtEntry, not buildSessionContext directly, in the assistant
    // branch). Here we monkey-patch buildSessionContext on the SM and have
    // it throw on the second call (the one inside estimateActiveBranchTokens
    // post-branchWithSummary). The first call comes from
    // refreshAgentMessages \u2014 but no session is captured here, so refresh
    // returns false fast without calling buildSessionContext. The first
    // sm.buildSessionContext() call is inside estimateActiveBranchTokens
    // post-branchWithSummary.
    let thrown: unknown;
    const origBuild = sm.buildSessionContext.bind(sm);
    let buildCalls = 0;
    (
      sm as { buildSessionContext: typeof sm.buildSessionContext }
    ).buildSessionContext = ((...args: unknown[]) => {
      buildCalls++;
      if (buildCalls === 1) throw new Error("estimate boom");
      // biome-ignore lint/suspicious/noExplicitAny: forward to original
      return (origBuild as (...a: unknown[]) => unknown).apply(sm, args as any);
    }) as typeof sm.buildSessionContext;

    try {
      await tool.execute(
        "tc-rewind",
        {
          action: "rewind",
          labelStart: "start",
          labelEnd: "end",
          summaryFocus: "Preserve user instructions and continue.",
        },
        undefined,
        undefined,
        ctx,
      );
    } catch (e) {
      thrown = e;
    }
    assert.ok(thrown);
    const msg = thrown instanceof Error ? thrown.message : String(thrown);
    assert.match(msg, /estimate boom/);
    // Restore so leaf inspection works.
    (
      sm as { buildSessionContext: typeof sm.buildSessionContext }
    ).buildSessionContext = origBuild;
    // Synthetic landed.
    const leafId = sm.getLeafId();
    assert.ok(leafId);
    const leaf = sm.getEntry(leafId as string);
    assert.ok(leaf && leaf.type === "message");
    if (leaf && leaf.type === "message") {
      const c0 = (
        leaf.message.content as Array<{ type: string; id?: string }>
      )[0];
      assert.equal(c0.id, "tc-rewind");
    }
  });

  it("synthetic appendMessage throws \u2192 original error propagates cleanly (no recovery)", async () => {
    // Post-restructure the synthetic append is the recovery itself \u2014 if
    // it throws there's nothing more we can do. Pin: the throw escapes
    // verbatim (no double-handling, no swallowing).
    const { sm, pi, tool, ctx } = setup();
    const t1 = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendTurn(sm, "u2", "a2", 200);

    // Arm appendMessage to throw on the next call \u2014 the synthetic append
    // is the only appendMessage from inside the tool's rewind path
    // (branchWithSummary doesn't go through appendMessage). The summarize
    // stub doesn't append either.
    const restore = throwOnNextAppendMessage(
      sm,
      new Error("appendMessage boom"),
    );
    let thrown: unknown;
    try {
      await tool.execute(
        "tc-rewind",
        {
          action: "rewind",
          labelStart: "start",
          labelEnd: "end",
          summaryFocus: "Preserve user instructions and continue.",
        },
        undefined,
        undefined,
        ctx,
      );
    } catch (e) {
      thrown = e;
    }
    restore();
    assert.ok(thrown);
    const msg = thrown instanceof Error ? thrown.message : String(thrown);
    assert.match(msg, /appendMessage boom/);
  });
});

// =============================================================================
// rewind error branches (COV2-2)
// =============================================================================

describe("dispatch: rewind error branches", () => {
  it("'Already at <labelStart>' fires when the labelStart anchor is on the leaf", async () => {
    // setLabel itself advances the leaf (it appends a label-type entry as
    // child of the prior leaf), so anchoring then driving rewind doesn't
    // naturally hit `oldLeaf === target`. Reset the leaf back to the
    // labeled assistant via `sm.branch(...)` so the guard fires. This is
    // the contract the message pins ("Already at <labelStart>"); the
    // path-construction is test-internal.
    const { sm, pi, tool, ctx } = setup();
    const t1 = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(t1.assistantId, "anchor:here");
    sm.branch(t1.assistantId);
    const result = await tool.execute(
      "tc-1",
      {
        action: "rewind",
        labelStart: "here",
        labelEnd: "after",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, true);
    assert.match(
      result.content[0].text,
      /Already at 'here' \u2014 nothing to summarize/,
    );
  });

  it("summarize aborted \u2192 'Summarization aborted.'", async () => {
    const { sm, pi, tool, ctx } = setup({
      summarize: (async () => ({
        summary: "",
        readFiles: [],
        modifiedFiles: [],
        aborted: true,
      })) as never,
    });
    const t1 = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendTurn(sm, "u2", "a2", 200);
    const result = await tool.execute(
      "tc-1",
      {
        action: "rewind",
        labelStart: "start",
        labelEnd: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /Summarization aborted/);
  });

  it("summarize returns error \u2192 'Summarization failed: <error>'", async () => {
    const { sm, pi, tool, ctx } = setup({
      summarize: (async () => ({
        summary: "",
        readFiles: [],
        modifiedFiles: [],
        aborted: false,
        error: "rate limited",
      })) as never,
    });
    const t1 = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendTurn(sm, "u2", "a2", 200);
    const result = await tool.execute(
      "tc-1",
      {
        action: "rewind",
        labelStart: "start",
        labelEnd: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /Summarization failed: rate limited/);
  });

  it("summarize returns empty summary \u2192 'no summary text'", async () => {
    const { sm, pi, tool, ctx } = setup({
      summarize: (async () => ({
        summary: "",
        readFiles: [],
        modifiedFiles: [],
        aborted: false,
      })) as never,
    });
    const t1 = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendTurn(sm, "u2", "a2", 200);
    const result = await tool.execute(
      "tc-1",
      {
        action: "rewind",
        labelStart: "start",
        labelEnd: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, true);
    assert.match(
      result.content[0].text,
      /Summarization failed: no summary text/,
    );
  });

  it("chained-rewind no-turns: synthetic-only intervening trips the boundary guard (CORR2-6)", async () => {
    const { sm, pi, tool, ctx } = setup();
    const tA = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(tA.assistantId, "anchor:a");
    appendTurn(sm, "u2", "a2", 200);

    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);

    // First rewind a\u2192b succeeds and leaves the leaf as a synthetic
    // assistant whose parent is the labeled b-summary.
    const r1 = await tool.execute(
      "tc-1",
      {
        action: "rewind",
        labelStart: "a",
        labelEnd: "b",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(r1.isError, undefined);

    // Immediately rewind b\u2192c with no intervening turns. The only entry
    // between leaf (synthetic) and target (b-summary) is the synthetic
    // itself \u2014 the new guard trips before summarize is invoked.
    const r2 = await tool.execute(
      "tc-2",
      {
        action: "rewind",
        labelStart: "b",
        labelEnd: "c",
        summaryFocus: "Preserve user instructions across the second rewind.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(r2.isError, true);
    assert.match(
      r2.content[0].text,
      /Already at synthetic boundary \u2014 no work to summarize/,
    );
  });
});

// =============================================================================
// installPrepareNextTurn cross-extension preservation (COV2-5)
// =============================================================================

describe("installPrepareNextTurn cross-extension preservation", () => {
  it("foreign extension's prepareNextTurn survives /reload re-install", async () => {
    // A foreign extension installs first (no marker). We install our
    // wrapper (wrap A); we install AGAIN simulating /reload (wrap B). The
    // foreign hook should still run and its `model` should propagate
    // through both wrappers \u2014 catching a regression where wrap B captures
    // wrap A as `__prior` instead of unwrapping to recover the foreign
    // function.
    const sm = SessionManager.inMemory("/tmp");
    const fake = makeFakeSession(sm);
    let priorRan = 0;
    fake.agent.prepareNextTurn = async () => {
      priorRan++;
      return { model: "from-foreign", thinkingLevel: "high" };
    };
    __testHooks.installPrepareNextTurn(fake as unknown as AgentSession);
    __testHooks.installPrepareNextTurn(fake as unknown as AgentSession);

    type Pnt = (
      ...args: unknown[]
    ) => Promise<{ model?: unknown; thinkingLevel?: unknown }>;
    const fn = fake.agent.prepareNextTurn as Pnt;
    const r = await fn();
    // Foreign ran exactly once \u2014 not zero (which would signal that
    // wrap B captured wrap A as prior and the foreign got stranded), and
    // not twice (which would signal both A and B chained over the foreign).
    assert.equal(priorRan, 1);
    assert.equal(r.model, "from-foreign");
    assert.equal(r.thinkingLevel, "high");
  });

  it("prior wrapper's context.systemPrompt + tools propagate through the chain (CORR2-4)", async () => {
    // A foreign prepareNextTurn that returns a context with a custom
    // systemPrompt and tools. Our wrapper should preserve those fields and
    // override only `messages`.
    const sm = SessionManager.inMemory("/tmp");
    sm.appendMessage({
      role: "user",
      content: [{ type: "text", text: "hi" }],
    } as never);
    const fake = makeFakeSession(sm);
    fake.agent.prepareNextTurn = async () => ({
      context: {
        systemPrompt: "FOREIGN_PROMPT",
        messages: [{ role: "user", content: "should be overwritten" }],
        tools: ["foreign-tool"],
      },
    });
    __testHooks.installPrepareNextTurn(fake as unknown as AgentSession);
    type Pnt = (...args: unknown[]) => Promise<{
      context?: {
        systemPrompt?: unknown;
        messages?: unknown;
        tools?: unknown;
      };
    }>;
    const fn = fake.agent.prepareNextTurn as Pnt;
    const r = await fn();
    // systemPrompt + tools survive from the foreign wrapper.
    assert.equal(r.context?.systemPrompt, "FOREIGN_PROMPT");
    assert.deepEqual(r.context?.tools, ["foreign-tool"]);
    // messages is overridden by sm.buildSessionContext().messages.
    assert.deepEqual(r.context?.messages, sm.buildSessionContext().messages);
  });
});

// =============================================================================
// non-assistant oldLeafEntry fallback (COV2-6)
// =============================================================================

describe("dispatch: rewind beforeTokens fallback", () => {
  it("non-assistant leaf: beforeTokens uses estimateActiveBranchTokens fallback (smoke)", async () => {
    // Build chain ending in a user message (not an assistant). The
    // happy-path beforeTokens math gates on role==='assistant' &&
    // parentId; otherwise falls back to estimateActiveBranchTokens.
    // Smoke-level pin: a sensible non-zero contextBefore lands in details.
    const { sm, pi, tool, ctx } = setup();
    const t1 = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendTurn(sm, "u2", "a2", 200);
    // Append a trailing user message so the leaf isn't an assistant.
    sm.appendMessage({
      role: "user",
      content: [{ type: "text", text: "u3" }],
    } as never);

    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);

    const result = await tool.execute(
      "tc-1",
      {
        action: "rewind",
        labelStart: "start",
        labelEnd: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    const before = result.details.contextBefore as number;
    assert.ok(
      typeof before === "number" && before > 0,
      `expected non-zero contextBefore; got ${before}`,
    );
  });
});
