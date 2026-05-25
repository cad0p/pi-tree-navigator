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
  generateBranchSummary,
  SessionManager,
} from "@earendil-works/pi-coding-agent";
import { MAX_NAME_LENGTH } from "./helpers.ts";
import navigateTree, {
  __testHooks,
  MAX_HINT_WALK_DEPTH,
  MAX_SESSION_REFS,
  MAX_SYNTHETIC_FOCUS_LENGTH,
  MIN_SUMMARY_FOCUS_LENGTH,
} from "./index.ts";

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
      "## Goal\nTest the rewind happy path.\n## Progress\n### Done\nAppended turns.\n## Next Steps\nVerify the synthetic.",
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

/**
 * Build the canonical rewindable fixture: an anchored first assistant turn
 * plus N follow-up turns (default 1), optionally with a captured
 * AgentSession so the reflection bootstrap finds it.
 *
 * Companion to `rewindFixture()` (which additionally drives the rewind);
 * use this helper when the test needs to control the rewind invocation
 * directly (custom args, throw-arming between setup and execute, etc.).
 */
function setupRewindable(
  sm: SessionManager,
  pi: FakePi,
  opts: {
    capture?: boolean;
    turnsAfter?: number;
    labelStartName?: string;
    tokenCounts?: number[];
  } = {},
): { fake?: FakeAgentSession } {
  const labelStartName = opts.labelStartName ?? "start";
  const turnsAfter = opts.turnsAfter ?? 1;
  const tokenCounts = opts.tokenCounts ?? [100, 200, 300, 400, 500];
  const t1 = appendTurn(sm, "u1", "a1", tokenCounts[0]);
  pi.pi.setLabel(t1.assistantId, `anchor:${labelStartName}`);
  for (let i = 0; i < turnsAfter; i++) {
    appendTurn(sm, `u${i + 2}`, `a${i + 2}`, tokenCounts[i + 1] ?? 100);
  }
  if (opts.capture) {
    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);
    return { fake };
  }
  return {};
}

// =============================================================================
// Schema shape (Kiro compatibility)
// =============================================================================

describe("schema shape \u2014 Kiro compatibility", () => {
  it("registers a tool with a flat object root and `action` required", () => {
    // The Kiro/CodeWhisperer adapter forwards inputSchema.json verbatim and
    // 400s on non-`type: "object"` roots (anyOf/oneOf/discriminated unions).
    // Pin the working shape so a future TypeBox refactor surfaces here.
    const { tool } = setup();
    const params = tool.parameters as {
      type: string;
      required?: string[];
      properties?: Record<string, unknown>;
      anyOf?: unknown;
      oneOf?: unknown;
    };
    assert.equal(params.type, "object");
    // Tighter pin: `action` is the ONLY required field at the schema level.
    // Runtime guards (in execute) handle the action-conditional required-ness
    // for `name`, `labelStart`, `labelEnd`, `summaryFocus`. A regression
    // that lifts a runtime guard into the schema (e.g. adding
    // `summaryFocus` to `required`) would re-introduce the original Kiro
    // 400 — this assertion catches it.
    assert.deepEqual(params.required, ["action"]);
    // Each action-conditional field must still exist in `properties` so the
    // schema describes the full surface to the model.
    const props = params.properties ?? {};
    for (const key of ["name", "labelStart", "labelEnd", "summaryFocus"]) {
      assert.ok(key in props, `${key} must be a schema property`);
    }
    assert.equal(params.anyOf, undefined);
    assert.equal(params.oneOf, undefined);
  });

  it("declares executionMode: 'sequential' (concurrency contract)", () => {
    // Concurrent dispatch would race on the SessionManager's leaf pointer
    // and corrupt the tree. The tool relies on the pi runtime to serialize
    // calls. A regression that drops or flips this property is the only
    // practical defense — source has no in-process double-call guard.
    const { tool } = setup();
    assert.equal(tool.executionMode, "sequential");
  });

  it("each conditional-required field carries an action-conditional description", () => {
    // The schema descriptions encode the conditional-required contract
    // for the model (the `Required when action='X'` phrase). A drift in
    // this prose silently degrades model tool-use accuracy. Pin presence
    // of the canonical conditional phrase so a future copy-edit that
    // drops it surfaces here.
    const { tool } = setup();
    const props = (
      tool.parameters as {
        properties: Record<string, { description?: string }>;
      }
    ).properties;
    assert.match(props.name.description ?? "", /Required when action='anchor'/);
    assert.match(
      props.labelStart.description ?? "",
      /Required when action='rewind'/,
    );
    assert.match(
      props.labelEnd.description ?? "",
      /Required when action='rewind'/,
    );
    assert.match(
      props.summaryFocus.description ?? "",
      /Required when action='rewind'/,
    );
    // Pin schema-description interpolation of the named constants. The
    // runtime guard error pins MIN_SUMMARY_FOCUS_LENGTH symmetrically
    // (see the `summaryFocus` validation-guards table below); the
    // schema-description side is the model-facing surface and deserves
    // the same anti-drift pin. A regression that hardcodes "max 40" or
    // "≥20" (instead of interpolating) silently desyncs when the
    // constants are re-tuned.
    assert.match(
      props.name.description ?? "",
      new RegExp(`max ${MAX_NAME_LENGTH}`),
    );
    assert.match(
      props.summaryFocus.description ?? "",
      new RegExp(`\u2265${MIN_SUMMARY_FOCUS_LENGTH}`),
    );
  });
});

// =============================================================================
// production-default `summarize` resolution
// =============================================================================

describe("production-default summarize resolution", () => {
  it("the SDK exports `generateBranchSummary` as a callable", () => {
    // If the import becomes stale (SDK rename, barrel-import shuffle),
    // production code path — which falls back to this default — would
    // explode at the first real rewind. Pin the export at module init.
    assert.equal(typeof generateBranchSummary, "function");
  });

  it("registers without an `opts` argument and `list` succeeds", async () => {
    // Production callers (pi's extension loader) pass a single arg. Confirm
    // the registration succeeds and a non-summarizing action (`list`) runs
    // without invoking the default `summarize` path.
    const sm = SessionManager.inMemory("/tmp");
    const fakePi = makeFakePi(sm);
    navigateTree(fakePi.pi);
    assert.equal(fakePi.registered.length, 1);
    const tool = fakePi.registered[0];
    const ctx = makeCtx(sm);
    const result = await tool.execute(
      "tc-1",
      { action: "list" },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
  });
});

// =============================================================================
// buildSyntheticAssistant shape
// =============================================================================

describe("buildSyntheticAssistant shape", () => {
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

  it("pins the per-row line shape: percentage column then label column", async () => {
    // Pin the actual rendered shape so a formatting refactor that swaps
    // column order or drops the percentage prefix surfaces here. The
    // regex matches one or more leading spaces, a `\d+\.\d%` percent,
    // more whitespace, then a non-whitespace label. We don't pin
    // character offsets — padStart/padEnd widths are implementation
    // details — just the shape.
    const { sm, pi, tool, ctx } = setup();
    const t1 = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(t1.assistantId, "anchor:short");
    const result = await tool.execute(
      "tc-1",
      { action: "list" },
      undefined,
      undefined,
      ctx,
    );
    const text = result.content[0].text;
    assert.match(text, /^\s+\d+\.\d+%\s+\S+/m);
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

  it("list and rewind warnings cite their site-specific REFLECTION_BOOTSTRAP_WARNING constants verbatim", async () => {
    // Two SSOT constants — one per emission site — because the prose
    // differs (`list` is read-only; `rewind` wrote to disk). Pin literal
    // containment of the matching constant at each site so a future
    // drift (e.g. dropping the /reload recovery hint at one site, or
    // accidentally swapping the constants between sites) surfaces here.
    const listSentinel = __testHooks.REFLECTION_BOOTSTRAP_WARNING_LIST;
    const rewindSentinel = __testHooks.REFLECTION_BOOTSTRAP_WARNING_REWIND;
    assert.ok(listSentinel.length > 0, "list-warning must be non-empty");
    assert.ok(rewindSentinel.length > 0, "rewind-warning must be non-empty");
    // The two are distinct — distinct prose for distinct call shapes.
    // (If they ever converge again, this asserts that fact deliberately.)
    assert.notEqual(
      listSentinel,
      rewindSentinel,
      "list and rewind warnings must use site-specific phrasing",
    );
    // Both share the recovery hint with `/reload` mentioned
    // (`/reload` is the lighter-weight recovery and is named first;
    // restarting pi is the heavier alternative). A regression that
    // drops `/reload` from either constant surfaces here.
    assert.match(
      listSentinel,
      /\/reload/,
      "list-warning must mention /reload as a recovery option",
    );
    assert.match(
      rewindSentinel,
      /\/reload/,
      "rewind-warning must mention /reload as a recovery option",
    );

    // List site: no captured session — reflection finds nothing. The
    // list-specific constant must appear verbatim; the rewind-specific
    // constant must NOT (they're cross-site distinct).
    {
      const { tool, ctx } = setup();
      const result = await tool.execute(
        "tc-list",
        { action: "list" },
        undefined,
        undefined,
        ctx,
      );
      assert.ok(
        result.content[0].text.includes(listSentinel),
        "list output must include REFLECTION_BOOTSTRAP_WARNING_LIST verbatim",
      );
      assert.ok(
        !result.content[0].text.includes(rewindSentinel),
        "list output must NOT include the rewind-specific phrasing",
      );
    }

    // Rewind site: same conditions — the bootstrap-missing footer must
    // contain the rewind-specific constant verbatim, NOT the list one.
    {
      const { sm, pi, tool, ctx } = setup();
      setupRewindable(sm, pi);
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
      assert.ok(
        result.content[0].text.includes(rewindSentinel),
        "rewind output must include REFLECTION_BOOTSTRAP_WARNING_REWIND verbatim",
      );
      assert.ok(
        !result.content[0].text.includes(listSentinel),
        "rewind output must NOT include the list-specific phrasing",
      );
    }
  });
});

// =============================================================================
// dispatch: anchor
// =============================================================================

describe("dispatch: anchor action", () => {
  // The `isValidName` predicate is exhaustively pinned in helpers.test.ts;
  // here we just verify the predicate's verdict propagates into the dispatch
  // error path. Two rows: a representative invalid shape + the boundary case.
  const invalidNames: Array<[string, unknown]> = [
    ["uppercase", "Impl-Start"],
    ["over max length", "a".repeat(MAX_NAME_LENGTH + 1)],
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
    // Pin the anchor follow-up hint prose. This is the load-bearing
    // nudge that gets the agent to chain `anchor → rewind` correctly
    // with a populated `summaryFocus`. A future copy-edit that drops
    // the `summaryFocus` mention or the `MIN_SUMMARY_FOCUS_LENGTH`
    // interpolation would silently degrade the model's tool-use
    // accuracy on the very first rewind.
    const text = (result.content[0] as { text: string }).text;
    assert.match(text, /navigate_tree\(action='rewind'/);
    assert.match(text, /labelStart='impl-start'/);
    assert.match(text, /summaryFocus=/);
    assert.match(text, new RegExp(`\u2265${MIN_SUMMARY_FOCUS_LENGTH}`));
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
    // Pin write-before-clear ordering: setLabel(newLeaf, fullLabel) MUST
    // land before setLabel(prior, undefined). The reverse order leaves a
    // "prior cleared, new failed to install" window if the second call
    // throws — source comment at the anchor handler documents this as
    // load-bearing. A regression to clear-then-set would still produce
    // the right end state but lose the rollback property.
    const setIdx = pi.setLabelCalls.findIndex(
      ([id, lbl]) => id === t2.assistantId && lbl === "anchor:iter-start",
    );
    const clearIdx = pi.setLabelCalls.findIndex(
      ([id, lbl]) => id === t1.assistantId && lbl === undefined,
    );
    assert.ok(setIdx >= 0, "expected the new-leaf set call");
    assert.ok(clearIdx >= 0, "expected the prior-clear call");
    assert.ok(
      setIdx < clearIdx,
      `expected set-before-clear; got setIdx=${setIdx} clearIdx=${clearIdx}`,
    );
  });

  it("re-anchor on the same leaf with the same name is idempotent: no spurious clear", async () => {
    // Defensive `prior !== leafId` guard: if we capture `prior` for the
    // requested label and find it points at the very leaf we're about
    // to label, skip the prior-clear (otherwise we'd issue setLabel(leaf,
    // undefined) immediately after setLabel(leaf, fullLabel), wiping the
    // label we just wrote).
    //
    // Production pi's `setLabel` always advances the leaf (it appends
    // a label-change entry via `appendLabelChange`), so `prior === leafId`
    // doesn't normally arise. The default `makeFakePi` mirrors that
    // behavior. To exercise the guard directly, swap in an in-place
    // setLabel that mutates `labelsById` WITHOUT advancing the leaf —
    // this models a hypothetical future pi (or extension-runner) where
    // setLabel is leaf-stable. The guard's correctness should not depend
    // on which behavior pi exposes.
    const { sm, ctx } = setup();
    const t1 = appendTurn(sm, "u", "a");

    // In-place pi: setLabel mutates labelsById directly, no leaf advance.
    const setLabelCalls: Array<[string, string | undefined]> = [];
    const inPlacePi = {
      registerTool() {},
      setLabel(entryId: string, label: string | undefined) {
        setLabelCalls.push([entryId, label]);
        // Reach into the SM's internal map to set the label without
        // appending a new entry. The map is exposed for tests via the
        // SessionManager surface.
        const labelsMap = (sm as unknown as { labelsById: Map<string, string> })
          .labelsById;
        if (label === undefined) labelsMap.delete(entryId);
        else labelsMap.set(entryId, label);
      },
    } as unknown as ExtensionAPI;
    // Re-register the tool with the in-place pi.
    const inPlaceRegistered: CapturedTool[] = [];
    (
      inPlacePi as unknown as {
        registerTool: (t: CapturedTool) => void;
      }
    ).registerTool = (t: CapturedTool) => {
      inPlaceRegistered.push(t);
    };
    navigateTree(inPlacePi, { summarize: fakeSummarize as never });
    const inPlaceTool = inPlaceRegistered[0];

    // First anchor: leaf is the assistant entry t1.assistantId.
    await inPlaceTool.execute(
      "tc-1",
      { action: "anchor", name: "foo" },
      undefined,
      undefined,
      ctx,
    );
    // Verify the label landed on t1.assistantId and the leaf did NOT
    // advance (the precondition for the guard branch).
    assert.equal(sm.getLabel(t1.assistantId), "anchor:foo");
    assert.equal(sm.getLeafId(), t1.assistantId);

    // Second anchor: leaf is STILL t1.assistantId, prior also points
    // at t1.assistantId — prior === leafId, the guard branch fires.
    setLabelCalls.length = 0;
    await inPlaceTool.execute(
      "tc-2",
      { action: "anchor", name: "foo" },
      undefined,
      undefined,
      ctx,
    );
    // Pin: exactly one setLabel call (the re-set of the same label),
    // and zero clears. A regression that drops the `prior !== leafId`
    // guard would issue a setLabel(t1.assistantId, undefined) clearing
    // the label we just (re-)wrote.
    assert.equal(setLabelCalls.length, 1, "expected exactly one setLabel");
    assert.deepEqual(setLabelCalls[0], [t1.assistantId, "anchor:foo"]);
    const clears = setLabelCalls.filter(([, lbl]) => lbl === undefined);
    assert.equal(
      clears.length,
      0,
      "no clear should fire when prior === leafId",
    );
    // Label still present.
    assert.equal(sm.getLabel(t1.assistantId), "anchor:foo");
  });

  it("falls through with a misleading rewind error on unknown `action`", async () => {
    // The schema declares `action` as a Union(anchor|rewind|list), but the
    // runtime dispatch is three `if` checks with no explicit default —
    // an unknown action falls through to the rewind validation block.
    // Pin the current behavior so a future explicit `else` guard surfaces
    // here as a deliberate change.
    const { sm, tool, ctx } = setup();
    appendTurn(sm, "u", "a");
    const result = await tool.execute(
      "tc-1",
      { action: "bogus" } as unknown as Record<string, unknown>,
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, true);
    // Pin that the unknown-action falls through to the rewind-validation
    // site specifically (not just "any error"). The rewind dispatch's
    // first guard rejects a missing/invalid `labelStart` with a
    // "kebab-case" message; pinning that text confirms the fall-through
    // landed at the labelStart guard, not at a future explicit
    // unknown-action default.
    assert.match(
      (result.content[0] as { text: string }).text,
      /labelStart.*kebab-case/,
    );
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
        summaryFocus: "x".repeat(MIN_SUMMARY_FOCUS_LENGTH - 1), // just under the floor
      },
      want: /summaryFocus/,
    },
    {
      // Trim-presence pin on the rejection path: raw length is above the
      // floor (so a regression dropping `.trim()` would accept it), but
      // trimmed length is below it. The guard with `.trim()` rejects;
      // without `.trim()` it would slip past this row.
      name: "summaryFocus shorter than min length after trim",
      params: {
        action: "rewind",
        labelStart: "ok",
        labelEnd: "ok2",
        summaryFocus: `  ${"x".repeat(MIN_SUMMARY_FOCUS_LENGTH - 1)}  `,
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
      // Pin that the error text surfaces MIN_SUMMARY_FOCUS_LENGTH literally;
      // the regex tracks the constant if the floor is bumped.
      want: new RegExp(`\u2265${MIN_SUMMARY_FOCUS_LENGTH}`),
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
        summaryFocus: "x".repeat(MIN_SUMMARY_FOCUS_LENGTH),
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
    // Boundary: summaryFocus.trim().length >= MIN_SUMMARY_FOCUS_LENGTH (the
    // >= boundary, not strict >). The call still errors at the next guard
    // (no labelStart on chain), but it moves past the focus-length check —
    // pinning that the comparison is inclusive at exactly MIN. Trim-
    // presence on the rejection path is pinned by the
    // "shorter than min length after trim" row above, not here.
    const { sm, tool, ctx } = setup();
    appendTurn(sm, "u", "a");
    const result = await tool.execute(
      "tc-1",
      {
        action: "rewind",
        labelStart: "missing",
        labelEnd: "after",
        summaryFocus: `  ${"x".repeat(MIN_SUMMARY_FOCUS_LENGTH)}  `,
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
  /**
   * Set up a 3-turn chain anchored at the first assistant, capture a fake
   * session so reflection succeeds, and drive a rewind. Returns everything
   * the split tests need to pin distinct contracts on the same outcome.
   */
  async function rewindFixture() {
    const { sm, pi, tool, ctx } = setup();
    const t1 = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendTurn(sm, "u2", "a2", 200);
    appendTurn(sm, "u3", "a3", 300);

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
    return { sm, pi, tool, ctx, fake, result };
  }

  it("emits the [rewind] response prose with Done / In Progress / Blocked / Next Steps", async () => {
    const { result } = await rewindFixture();
    assert.equal(result.isError, undefined);
    assert.match(result.content[0].text, /\[rewind 'start' \u2192 'end'\]/);
    assert.match(result.content[0].text, /### Done/);
    assert.match(result.content[0].text, /### In Progress/);
    assert.match(result.content[0].text, /### Blocked/);
    assert.match(result.content[0].text, /are pending/);
    assert.match(result.content[0].text, /## Next Steps/);
  });

  it("labels the summary entry with anchor:<labelEnd> and leaves a synthetic leaf", async () => {
    const { sm, result } = await rewindFixture();
    // The new summary entry carries the labelEnd.
    const summaryId = result.details.summaryId as string;
    assert.equal(sm.getLabel(summaryId), "anchor:end");
    // The leaf is the synthetic assistant, NOT the branch_summary.
    const leafId = sm.getLeafId();
    assert.ok(leafId, "expected a leaf after rewind");
    assert.notEqual(leafId, summaryId);
    const leaf = sm.getEntry(leafId);
    assert.ok(leaf && leaf.type === "message");
    if (leaf && leaf.type === "message") {
      assert.equal(leaf.message.role, "assistant");
      const c0 = (
        leaf.message.content as Array<{ type: string; id?: string }>
      )[0];
      assert.equal(c0.type, "toolCall");
      assert.equal(c0.id, "tc-rewind");
    }
  });

  it("shrinks contextBefore→contextAfter and refreshes agent.state.messages", async () => {
    const { sm, fake, result } = await rewindFixture();
    // Token-math contract: the chain shrinks across the rewind.
    const before = result.details.contextBefore as number;
    const after = result.details.contextAfter as number;
    assert.ok(before > after, `expected before (${before}) > after (${after})`);
    // Reflection contract: agent.state.messages mutated to match the new
    // session context. (The pre-rewind snapshot is empty per fixture; the
    // post-rewind value must equal sm.buildSessionContext().messages.)
    assert.equal(result.details.agentMessagesRefreshed, true);
    assert.deepEqual(
      fake.agent.state.messages,
      sm.buildSessionContext().messages,
    );
  });

  it("pins the synthetic-token bias contract: usage.totalTokens === pre-synthetic chain estimate", async () => {
    // The synthetic's totalTokens is set to the chain size *before* the
    // synthetic itself is appended — a deliberate ~50-token understatement
    // documented in buildSyntheticAssistant's JSDoc. Pin the contract so a
    // future refactor that tries to "fix" the bias by computing AFTER the
    // append surfaces here.
    const { sm, result } = await rewindFixture();
    const summaryId = result.details.summaryId as string;
    // The synthetic's recorded baseline equals the active-branch token count
    // measured at the new branch_summary leaf, immediately after
    // branchWithSummary and before the synthetic was appended. The simplest
    // verifiable surface is `details.contextAfter` — production sets
    // afterTokens = tokensAtNewLeaf, the same value that flows into the
    // synthetic's `totalTokens`. We assert the round-trip.
    const after = result.details.contextAfter as number;
    const leafId = sm.getLeafId();
    assert.ok(leafId);
    const leaf = sm.getEntry(leafId as string);
    assert.ok(leaf && leaf.type === "message");
    if (leaf && leaf.type === "message") {
      const usage = (leaf.message as { usage?: { totalTokens?: number } })
        .usage;
      assert.equal(
        usage?.totalTokens,
        after,
        "synthetic.totalTokens must equal contextAfter (pre-synthetic chain estimate)",
      );
    }
    // Sanity: summaryId is on the chain and labeled.
    assert.equal(sm.getLabel(summaryId), "anchor:end");
  });

  it("bootstrap-missing path: warns in response and reports refreshed=false", async () => {
    const { sm, pi, tool, ctx } = setup();
    setupRewindable(sm, pi);
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
    assert.match(result.content[0].text, /reflection bootstrap missing/);
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
    // Pin label retention explicitly: the b-summary keeps its 'anchor:b'
    // label across the second rewind. Nothing in the rewind path clears
    // labelStart's label (it's only labelEnd that gets the new write +
    // move-on-collision), so the prior label survives. A regression that
    // accidentally cleared labelStart on rewind would surface here.
    assert.equal(
      sm.getLabel(sumB),
      "anchor:b",
      "the b-summary's anchor:b label must survive the second rewind",
    );
    // Active leaf is the synthetic for the second rewind.
    const leafId = sm.getLeafId();
    assert.ok(leafId, "expected a leaf after second rewind");
    assert.notEqual(leafId, sumC);
    const leaf = sm.getEntry(leafId);
    if (leaf?.type === "message" && leaf.message.role === "assistant") {
      const c0 = (
        leaf.message.content as Array<{ type: string; id?: string }>
      )[0];
      assert.equal(c0.id, "tc-2");
    }
  });

  it("labelEnd collides with an existing anchor: rewind moves the anchor to the new summary (mirrors anchor's move-on-collision)", async () => {
    // Namespace symmetry: anchor.name and rewind.labelEnd both write into
    // the `anchor:` namespace, so a `rewind` whose labelEnd already labels
    // another entry on the *post-move active branch* must move the label
    // to the new summary — mirroring `anchor`'s write-before-clear
    // move-on-collision. Without the move, two entries on the same active
    // branch would both carry `anchor:b`, breaking `findLabeledEntry`'s
    // uniqueness invariant.
    //
    // Setup ordering matters: the prior `anchor:b` must be on the
    // ancestral side of `labelStart` (= tA), so that branchWithSummary
    // leaves it on the *active* branch (between root and the new
    // summary), NOT on the abandoned one. We anchor 'b' on the FIRST
    // turn and 'a' on the SECOND turn:
    //   root → tB(anchor:b) → tA(anchor:a) → t3 → leaf
    // Then rewind a→b collapses [tA-child … leaf] into summaryId, leaving
    //   root → tB(anchor:b) → tA(anchor:a) → summaryId(anchor:b)
    // — anchor:b lives on both tB and summaryId on the active branch.
    // The move-on-collision clears tB's label.
    const { sm, pi, tool, ctx } = setup();
    const tB = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(tB.assistantId, "anchor:b");
    const tA = appendTurn(sm, "u2", "a2", 200);
    pi.pi.setLabel(tA.assistantId, "anchor:a");
    appendTurn(sm, "u3", "a3", 300);

    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);

    // Sanity: pre-move, anchor:b resolves to tB.
    assert.equal(
      __testHooks.findLabeledEntry(sm, "anchor:b"),
      tB.assistantId,
      "pre-move sanity: anchor:b lives on tB",
    );

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        labelStart: "a",
        labelEnd: "b",
        summaryFocus: "Preserve the user's instruction and continue the work.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    const summaryId = result.details.summaryId as string;

    // The new summary carries 'anchor:b'.
    assert.equal(
      sm.getLabel(summaryId),
      "anchor:b",
      "new branch_summary must carry the labelEnd anchor",
    );
    // The prior 'b'-labeled entry lost its label — cleared by the
    // move-on-collision branch in rewind. (Only its label was cleared;
    // the entry itself is still in storage.)
    assert.notEqual(
      sm.getLabel(tB.assistantId),
      "anchor:b",
      "prior 'anchor:b' entry must be cleared after the move",
    );
    // findLabeledEntry resolves 'anchor:b' uniquely to the new summary.
    // (Walks leaf→root and the active branch now passes through summaryId
    // with the label cleared on tB — only the new write remains.)
    assert.equal(
      __testHooks.findLabeledEntry(sm, "anchor:b"),
      summaryId,
      "findLabeledEntry must resolve 'anchor:b' to the moved summary",
    );
  });

  it("summaryFocus.length === MAX_SYNTHETIC_FOCUS_LENGTH (1024) is stored verbatim with no truncation marker", async () => {
    // Below-boundary case: the cap check is `length > 1024`, so a focus
    // exactly 1024 chars long must pass through unchanged. Pin the
    // strict-greater-than comparison so a regression to `>=` (which
    // would chop the last char + append the marker) surfaces.
    const { sm, pi, tool, ctx } = setup();
    setupRewindable(sm, pi, { capture: true });

    const focus = "x".repeat(MAX_SYNTHETIC_FOCUS_LENGTH);
    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        labelStart: "start",
        labelEnd: "end",
        summaryFocus: focus,
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);

    // Read the synthetic's args back: under-cap focus survives verbatim.
    const leafId = sm.getLeafId();
    assert.ok(leafId);
    const leaf = sm.getEntry(leafId as string);
    assert.ok(leaf && leaf.type === "message");
    if (leaf?.type === "message" && leaf.message.role === "assistant") {
      const c0 = (
        leaf.message.content as Array<{
          type: string;
          arguments?: { summaryFocus?: string };
        }>
      )[0];
      assert.equal(c0.type, "toolCall");
      assert.equal(
        c0.arguments?.summaryFocus?.length,
        MAX_SYNTHETIC_FOCUS_LENGTH,
        "focus at the cap boundary must be stored at full length",
      );
      assert.equal(
        c0.arguments?.summaryFocus,
        focus,
        "under-cap focus must be stored verbatim, no truncation",
      );
      assert.ok(
        !/\[truncated\]/.test(c0.arguments?.summaryFocus ?? ""),
        "under-cap focus must NOT carry the truncation marker",
      );
    }
  });

  it("summaryFocus.length === 1025 truncates to 1024 chars + '\u2026 [truncated]' marker", async () => {
    // Above-boundary case: the first char beyond the cap triggers the
    // truncation branch. Pin the marker shape so a regression that
    // drops the suffix (or moves the cap) surfaces.
    const { sm, pi, tool, ctx } = setup();
    setupRewindable(sm, pi, { capture: true });

    const focus = "x".repeat(MAX_SYNTHETIC_FOCUS_LENGTH + 1);
    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        labelStart: "start",
        labelEnd: "end",
        summaryFocus: focus,
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);

    const leafId = sm.getLeafId();
    assert.ok(leafId);
    const leaf = sm.getEntry(leafId as string);
    assert.ok(leaf && leaf.type === "message");
    if (leaf?.type === "message" && leaf.message.role === "assistant") {
      const c0 = (
        leaf.message.content as Array<{
          type: string;
          arguments?: { summaryFocus?: string };
        }>
      )[0];
      const stored = c0.arguments?.summaryFocus ?? "";
      // Stored = MAX_SYNTHETIC_FOCUS_LENGTH chars of x + the literal
      // marker suffix.
      assert.equal(
        stored,
        `${"x".repeat(MAX_SYNTHETIC_FOCUS_LENGTH)}\u2026 [truncated]`,
        "over-cap focus must be sliced at MAX_SYNTHETIC_FOCUS_LENGTH and carry the marker",
      );
      assert.match(
        stored,
        /\[truncated\]$/,
        "truncation marker must terminate the stored focus",
      );
    }
  });

  it("dual-channel: summarize sees the full focus while the synthetic stores the truncated copy", async () => {
    // The full focus is passed live to `generateBranchSummary`, so the
    // summarizer always sees the original; only the synthetic's
    // re-emitted args are trimmed. Pin both channels with one spy: a
    // 1500-char focus must reach `summarize`'s `customInstructions`
    // unchanged, while the synthetic's args carry the 1024 + marker
    // form. A regression that pre-truncates the focus everywhere
    // ("simpler \u2014 one source of truth") would silently lobotomize the
    // summarizer's input.
    let capturedCustomInstructions: unknown;
    const spySummarize = (async (_entries: unknown, opts: unknown) => {
      capturedCustomInstructions = (opts as { customInstructions?: unknown })
        .customInstructions;
      return {
        summary: "## Goal\nspy.\n## Progress\n### Done\nx.\n## Next Steps\ny.",
        readFiles: [] as string[],
        modifiedFiles: [] as string[],
        aborted: false,
      };
    }) as typeof fakeSummarize;

    const { sm, pi, tool, ctx } = setup({ summarize: spySummarize });
    const { fake } = setupRewindable(sm, pi, { capture: true });
    if (!fake) throw new Error("capture: true must return fake");

    const fullFocus = "y".repeat(1500);
    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        labelStart: "start",
        labelEnd: "end",
        summaryFocus: fullFocus,
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);

    // (a) summarize spy received the FULL 1500-char focus.
    assert.equal(
      typeof capturedCustomInstructions,
      "string",
      "summarize must be invoked with a string customInstructions",
    );
    assert.equal(
      (capturedCustomInstructions as string).length,
      1500,
      "summarize must see the un-truncated focus",
    );
    assert.equal(
      capturedCustomInstructions,
      fullFocus,
      "summarize must see the focus verbatim (no pre-truncation upstream)",
    );

    // (b) synthetic's args store the truncated copy.
    const leafId = sm.getLeafId();
    assert.ok(leafId);
    const leaf = sm.getEntry(leafId as string);
    if (leaf?.type === "message" && leaf.message.role === "assistant") {
      const c0 = (
        leaf.message.content as Array<{
          type: string;
          arguments?: { summaryFocus?: string };
        }>
      )[0];
      const stored = c0.arguments?.summaryFocus ?? "";
      assert.equal(
        stored,
        `${"y".repeat(MAX_SYNTHETIC_FOCUS_LENGTH)}\u2026 [truncated]`,
        "synthetic must store the truncated form",
      );
    }
  });
});

// =============================================================================
// installPrepareNextTurn
// =============================================================================

describe("installPrepareNextTurn", () => {
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
// refreshAgentMessages
// =============================================================================

describe("refreshAgentMessages", () => {
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
    // Register a batch of short-lived sessions well above
    // MAX_SESSION_REFS so the reaper has work to do; the seenSessions
    // WeakSet dedupes within the same identity, and the sessionInstances
    // array gets reaped once length exceeds MAX_SESSION_REFS.
    //
    // Since WeakRef GC timing is non-deterministic, the bounded
    // assertion is the contract: the array doesn't grow proportional to
    // the number of pushes. (`Bun.gc(true)` is best-effort; if the
    // runtime doesn't expose it, we still get the dedupe + reap
    // bookkeeping.)
    const pushes = MAX_SESSION_REFS * 6 + 4; // comfortably above the cap
    for (let i = 0; i < pushes; i++) {
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
    assert.ok(refs <= pushes + 1, `sessionRefCount=${refs} grew unbounded`);
    // Tighter bound when GC ran: post-reap, only live ref(s) remain.
    // (most of the prior pushes were dead, plus the trailer = at most
    // a few \u2014 well under MAX_SESSION_REFS.)
    if (typeof g?.gc === "function") {
      assert.ok(
        refs <= MAX_SESSION_REFS,
        `sessionRefCount=${refs}; expected the reaper to bound it at MAX_SESSION_REFS=${MAX_SESSION_REFS}`,
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

  it("resetPrototype clears seenSessions so a re-captured identity isn't deduped", () => {
    // Pre-reset: capturing the same identity twice dedupes (afterRepeat
    // === afterFirst). Post-reset: the SAME identity captures freshly,
    // proving the WeakSet was rebound. WeakSet has no .clear() so this
    // pins the rebind contract — a regression that drops the rebind in
    // resetPrototype would silently observe stale dedupe across tests.
    const sm = SessionManager.inMemory("/tmp");
    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);
    const afterFirst = __testHooks.sessionRefCount();
    __testHooks.captureSession(fake as unknown as AgentSession);
    const afterRepeat = __testHooks.sessionRefCount();
    assert.equal(afterRepeat, afterFirst, "pre-reset: dedupe is active");

    __testHooks.resetPrototype();
    // After reset, sessionInstances is drained and seenSessions is fresh.
    assert.equal(__testHooks.sessionRefCount(), 0);
    // Re-capturing the SAME identity must succeed (count goes from 0 → 1).
    __testHooks.captureSession(fake as unknown as AgentSession);
    assert.equal(__testHooks.sessionRefCount(), 1);
  });
});

// =============================================================================
// findLabelHint depth limit
// =============================================================================

describe("findLabelHint", () => {
  it("returns null when no text is found within MAX_HINT_WALK_DEPTH", () => {
    // Build a chain whose head is a custom_message with no text content,
    // followed by user-text entries beyond the walker's reach \u2014 the
    // walker should give up at depth MAX_HINT_WALK_DEPTH and never
    // reach the texts.
    const sm = SessionManager.inMemory("/tmp");
    // First push the deep texts (positioned beyond the walker's reach).
    for (let i = 0; i < 100; i++) {
      sm.appendMessage({
        role: "user",
        content: [{ type: "text", text: `deep ${i}` }],
      } as never);
    }
    // Now push thinking-only assistant messages on top so the walk
    // bottoms out before reaching the user texts. We need at least
    // MAX_HINT_WALK_DEPTH thinking entries; use +10 for safety margin.
    for (let i = 0; i < MAX_HINT_WALK_DEPTH + 10; i++) {
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

  it("branch_summary entry: returns 'summary: <stripped lead-in>'", () => {
    // findLabelHint walks the parent chain; when it lands on a
    // branch_summary entry, the hint should be prefixed with 'summary: '
    // and stripBranchSummaryBoilerplate should remove pi's prelude. A
    // regression that drops the prefix or the strip call would surface
    // here.
    const sm = SessionManager.inMemory("/tmp");
    appendTurn(sm, "u1", "a1");
    const leafId = sm.getLeafId();
    assert.ok(leafId);
    const summaryId = sm.branchWithSummary(
      leafId,
      "This is the summary content of the rewound branch.",
    );
    const hint = __testHooks.findLabelHint(sm, summaryId, 80);
    assert.ok(hint, "branch_summary entry should produce a hint");
    if (hint) {
      assert.match(hint, /^summary: /);
      assert.match(hint, /summary content of the rewound branch/);
    }
  });

  it("custom_message entry: hint extracts the content text", () => {
    // findLabelHint also walks into `custom_message` entries (e.g. the
    // pi /label command's audit entries). Pin: the text content lands
    // in the hint, no prefix.
    const sm = SessionManager.inMemory("/tmp");
    sm.appendCustomMessageEntry(
      "some-custom-type",
      [{ type: "text", text: "custom message content" }],
      true,
    );
    const leafId = sm.getLeafId();
    assert.ok(leafId);
    const hint = __testHooks.findLabelHint(sm, leafId, 80);
    assert.ok(hint, "custom_message entry should produce a hint");
    if (hint) assert.match(hint, /custom message content/);
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
        summaryFocus: "x".repeat(MIN_SUMMARY_FOCUS_LENGTH),
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /No label 'same'/);
  });

  it("rewind without auth fails fast with a clear error", async () => {
    const { sm, pi, tool, ctx } = setup({ authError: "no api key" });
    setupRewindable(sm, pi);
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
    setupRewindable(sm, pi);
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
    const { fake } = setupRewindable(sm, pi, {
      capture: true,
      turnsAfter: 2,
      tokenCounts: [100, 200, 300],
    });
    if (!fake) throw new Error("capture: true must return fake");

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
    // Salvage detail wraps the original: the retry
    // (#2) ALSO threw, so the salvage-failure clause is appended.
    assert.match(msg, /salvage:.*labelEnd retry failed/);
    // Error.cause carries the original throw verbatim so post-mortem
    // readers walking the cause chain (or callers doing
    // `err.cause instanceof TypeError`-style checks) can recover the
    // original error class + stack. The wrapped string-formatted message
    // is for the agent; `cause` is for the debugger.
    assert.ok(thrown instanceof Error);
    if (thrown instanceof Error) {
      assert.ok(
        thrown.cause instanceof Error,
        "thrown.cause must preserve the original Error",
      );
      if (thrown.cause instanceof Error) {
        assert.match(thrown.cause.message, /setLabel boom #1/);
      }
    }
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
    setupRewindable(sm, pi);

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
    // Error.cause still preserves the original throw even when the
    // salvage retry succeeded \u2014 the wrapped Error always carries the
    // first-failure cause, regardless of whether salvage detail was
    // appended to the message.
    assert.ok(thrown instanceof Error);
    if (thrown instanceof Error) {
      assert.ok(
        thrown.cause instanceof Error,
        "thrown.cause must preserve the original Error",
      );
      if (thrown.cause instanceof Error) {
        assert.match(thrown.cause.message, /transient setLabel boom/);
      }
    }
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
    // Synthetic's `usage.totalTokens === 0` regardless of retry outcome:
    // the salvage path can't safely run estimateActiveBranchTokens
    // post-throw (the SM may be in an unknown state), so the synthetic
    // is built with `totalTokens=0` whenever we landed in the salvage
    // catch block. This pins that the salvage skips the post-rewind
    // chain measurement — a future refactor that "fixes" this would
    // mask the salvage's degenerate-fallback contract.
    const leafId = sm.getLeafId();
    assert.ok(leafId);
    const leaf = sm.getEntry(leafId as string);
    assert.ok(leaf && leaf.type === "message");
    if (leaf && leaf.type === "message") {
      const usage = (leaf.message as { usage?: { totalTokens?: number } })
        .usage;
      assert.equal(
        usage?.totalTokens,
        0,
        "salvage synthetic must use totalTokens=0",
      );
    }
  });

  it("estimateActiveBranchTokens throws \u2192 synthetic still appended with degenerate token count", async () => {
    const { sm, pi, tool, ctx } = setup();
    setupRewindable(sm, pi);

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
    // Error.cause preserves the original throw from the estimate step
    // (different from the setLabel-throws cases above) \u2014 pinning that
    // every salvage-rethrow path attaches `cause`, regardless of which
    // step (`setLabel` vs `estimate`) failed first.
    assert.ok(thrown instanceof Error);
    if (thrown instanceof Error) {
      assert.ok(
        thrown.cause instanceof Error,
        "thrown.cause must preserve the original Error",
      );
      if (thrown.cause instanceof Error) {
        assert.match(thrown.cause.message, /estimate boom/);
      }
    }
    // Restore so leaf inspection works.
    (
      sm as { buildSessionContext: typeof sm.buildSessionContext }
    ).buildSessionContext = origBuild;
    // Synthetic landed.
    const leafId = sm.getLeafId();
    assert.ok(leafId);
    const leaf = sm.getEntry(leafId as string);
    assert.ok(leaf && leaf.type === "message");
    if (leaf?.type === "message" && leaf.message.role === "assistant") {
      const c0 = (
        leaf.message.content as Array<{ type: string; id?: string }>
      )[0];
      assert.equal(c0.id, "tc-rewind");
      // Pin the degenerate token count promised by the test name. When
      // the estimate step throws, the salvage path can't compute
      // `tokensAtNewLeaf`, so the synthetic must be built with
      // `totalTokens=0` (mirroring the 'setLabel throws then retry
      // succeeds' sibling test). A regression that re-uses the
      // pre-throw zero-init value silently (or worse, leaves it
      // `undefined`) would surface here.
      const usage = (leaf.message as { usage?: { totalTokens?: number } })
        .usage;
      assert.equal(
        usage?.totalTokens,
        0,
        "salvage synthetic must use totalTokens=0 after estimate throw",
      );
    }
    // labelEnd write succeeded BEFORE the estimate threw — so a chained
    // rewind could still find it. Pin: walking the active branch finds
    // an entry with `anchor:end`. A regression that moves the label
    // write inside the throwing closure (or aborts the salvage label
    // retry policy) would silently drop the label, surfacing here.
    let foundLabelEnd = false;
    for (const e of sm.getBranch()) {
      if (sm.getLabel(e.id) === "anchor:end") {
        foundLabelEnd = true;
        break;
      }
    }
    assert.equal(
      foundLabelEnd,
      true,
      "labelEnd should have landed before estimate threw",
    );
  });

  it("synthetic appendMessage throws \u2192 original error propagates cleanly (no recovery)", async () => {
    // Post-restructure the synthetic append is the recovery itself \u2014 if
    // it throws there's nothing more we can do. Pin: the throw escapes
    // verbatim (no double-handling, no swallowing).
    const { sm, pi, tool, ctx } = setup();
    setupRewindable(sm, pi);

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
    // Tight equality (not regex match): a regression that wraps the
    // throw (e.g. `throw new Error("salvage failed: " + e.message)`)
    // would still match a /appendMessage boom/ regex, defeating the
    // "no recovery, no double-handling" claim. Exact equality pins it.
    assert.equal(msg, "appendMessage boom");
  });

  it("findLabeledEntry(labelEnd) throws → synthetic still appended; original error propagates", async () => {
    // The labelEnd-collision lookup runs inside the salvage try as the
    // first step. If it throws (e.g. malformed branch traversal),
    // setLabel cannot run — but the synthetic append still must, so
    // pi's appended tool_result has a matching tool_use on the new
    // branch. Without the in-try lookup, an upstream throw from
    // findLabeledEntry would orphan the tool_result.
    const { sm, pi, tool, ctx } = setup();
    setupRewindable(sm, pi, {
      capture: true,
      turnsAfter: 2,
      tokenCounts: [100, 200, 300],
    });

    // Trip `getBranch` ONLY after a `branch_summary` entry exists on
    // the active branch — i.e. after `sm.branchWithSummary` ran. The
    // labelEnd-collision lookup (`findLabeledEntry(sm, fullLabelEnd)`)
    // is the first call past that point in the rewind handler. The
    // labelStart lookup, beforeTokens, and collectEntriesForBranchSummary
    // all run BEFORE the move and thus before the trip is armed.
    const origGetBranch = sm.getBranch.bind(sm);
    let thrown: unknown;
    (sm as { getBranch: typeof sm.getBranch }).getBranch = (...args) => {
      const branch = origGetBranch(...args);
      if (branch.some((e) => e.type === "branch_summary")) {
        throw new Error("getBranch boom");
      }
      return branch;
    };

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

    // Restore for cleanliness.
    (sm as { getBranch: typeof sm.getBranch }).getBranch = origGetBranch;

    assert.ok(thrown, "rewind should re-throw the original error");
    const msg = thrown instanceof Error ? thrown.message : String(thrown);
    // Pin: original error from the lookup throw propagates verbatim,
    // no salvage-detail wrapping (lookup-throw doesn't trigger the
    // setLabel-retry path).
    assert.match(msg, /getBranch boom/);
    assert.ok(
      !/salvage:.*labelEnd retry failed/.test(msg),
      "lookup-throw must not trigger the setLabel-retry diagnostic",
    );

    // Synthetic still landed. Find the leaf assistant whose toolCall id
    // matches our in-flight `toolCallId='tc-rewind'`.
    const leafId = sm.getLeafId();
    assert.ok(leafId, "expected a leaf after lookup-throw salvage");
    if (leafId) {
      const leaf = sm.getEntry(leafId);
      assert.ok(leaf && leaf.type === "message");
      if (leaf && leaf.type === "message") {
        assert.equal(leaf.message.role, "assistant");
        const c0 = (
          leaf.message.content as Array<{ type: string; id?: string }>
        )[0];
        assert.equal(c0.type, "toolCall");
        assert.equal(c0.id, "tc-rewind");
      }
    }
  });

  it("prior-clear (clearPrior) throws once → retry succeeds; new labelEnd lives, prior label is cleared, no salvage detail", async () => {
    // The move-on-collision pair is two distinct setLabel calls.
    // (A) writes the new labelEnd onto the summary; (B) clears the
    // prior entry's labelEnd. If (A) succeeds and (B) throws, the
    // salvage retry must re-run (B) — not (A) — so the duplicate-label
    // state doesn't survive. Discriminator is `failedStep` (`setLabelEnd`
    // vs `clearPrior`); without the split, the retry would re-run the
    // already-succeeded (A) and silently leave both entries labeled.
    //
    // Setup mirrors the move-on-collision happy-path test:
    //   root → tB(anchor:b) → tA(anchor:a) → t3 → leaf
    // Rewind a→b: branchWithSummary collapses [tA-child … leaf] into
    // summaryId; (A) writes anchor:b onto summaryId; (B) clears anchor:b
    // off tB. We arm setLabel to throw on call #2 (B) only — #1 (A)
    // succeeds, #3 (the salvage retry of B) succeeds.
    const { sm, pi, tool, ctx } = setup();
    const tB = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(tB.assistantId, "anchor:b");
    const tA = appendTurn(sm, "u2", "a2", 200);
    pi.pi.setLabel(tA.assistantId, "anchor:a");
    appendTurn(sm, "u3", "a3", 300);

    // Patch AFTER the pre-anchor writes so the counter starts at 0.
    // Call #1 inside execute = (A) the new labelEnd write; call #2 = (B)
    // the prior-clear (throws once); call #3 = the salvage retry of (B)
    // (succeeds).
    throwOnNthSetLabel(pi, 2, new Error("transient prior-clear boom"));

    let thrown: unknown;
    try {
      await tool.execute(
        "tc-rewind",
        {
          action: "rewind",
          labelStart: "a",
          labelEnd: "b",
          summaryFocus:
            "Preserve the user's instruction and continue the work.",
        },
        undefined,
        undefined,
        ctx,
      );
    } catch (e) {
      thrown = e;
    }
    assert.ok(thrown, "rewind should re-throw the original (B) error");
    const msg = thrown instanceof Error ? thrown.message : String(thrown);
    assert.match(msg, /transient prior-clear boom/);
    // Retry succeeded — no salvage detail.
    assert.ok(
      !/salvage:/.test(msg),
      `expected no salvage detail when prior-clear retry succeeds; got: ${msg}`,
    );
    // Error.cause preserves the original (B) throw.
    assert.ok(thrown instanceof Error);
    if (thrown instanceof Error) {
      assert.ok(
        thrown.cause instanceof Error,
        "thrown.cause must preserve the original (B) Error",
      );
      if (thrown.cause instanceof Error) {
        assert.match(thrown.cause.message, /transient prior-clear boom/);
      }
    }
    // (A) wrote successfully on the first call: anchor:b lives on the
    // new summary entry (the leaf-side synthetic's parent).
    let summaryWithLabelEnd: string | null = null;
    for (const e of sm.getBranch()) {
      if (e.type === "branch_summary" && sm.getLabel(e.id) === "anchor:b") {
        summaryWithLabelEnd = e.id;
        break;
      }
    }
    assert.ok(
      summaryWithLabelEnd,
      "the new branch_summary must carry anchor:b (call #1 succeeded)",
    );
    // (B) RETRY succeeded: the prior tB lost its label. Without the
    // discriminant split, the retry would re-run (A) instead, leaving
    // tB still labeled — anchor:b would resolve to two entries.
    assert.notEqual(
      sm.getLabel(tB.assistantId),
      "anchor:b",
      "prior anchor:b must be cleared by the salvage retry of (B)",
    );
    // findLabeledEntry resolves anchor:b uniquely to the new summary.
    assert.equal(
      __testHooks.findLabeledEntry(sm, "anchor:b"),
      summaryWithLabelEnd,
      "anchor:b must resolve uniquely to the new summary post-salvage",
    );
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
  });

  it("prior-clear (clearPrior) throws on every call → salvage detail surfaces 'prior-clear retry failed'; new labelEnd still lives", async () => {
    // Sister test to the retry-succeeds case above: when BOTH the
    // original (B) prior-clear AND the salvage retry of (B) throw,
    // the wrapped error must surface a salvage detail that names the
    // prior-clear (NOT "labelEnd retry failed" — that diagnostic is
    // for the (A) failure mode and would be misleading here). This
    // pins the salvage-detail prose introduced by the failedStep
    // split: a regression that re-conflated the discriminants would
    // either retry (A) (silently succeeding, no detail) or surface
    // the wrong salvage-detail string.
    const { sm, pi, tool, ctx } = setup();
    const tB = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(tB.assistantId, "anchor:b");
    const tA = appendTurn(sm, "u2", "a2", 200);
    pi.pi.setLabel(tA.assistantId, "anchor:a");
    appendTurn(sm, "u3", "a3", 300);

    // Patch after pre-anchors. Throw on call #2 onward (B and the
    // retry of B). Call #1 (A) succeeds.
    let calls = 0;
    const origSetLabel = pi.pi.setLabel.bind(pi.pi);
    (pi.pi as { setLabel: typeof pi.pi.setLabel }).setLabel = (
      entryId: string,
      label: string | undefined,
    ) => {
      calls++;
      if (calls >= 2) throw new Error(`prior-clear boom #${calls}`);
      return origSetLabel(entryId, label);
    };

    let thrown: unknown;
    try {
      await tool.execute(
        "tc-rewind",
        {
          action: "rewind",
          labelStart: "a",
          labelEnd: "b",
          summaryFocus:
            "Preserve the user's instruction and continue the work.",
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
    // Original (B) error propagates verbatim as the base.
    assert.match(msg, /prior-clear boom #2/);
    // Salvage detail names the prior-clear, NOT the labelEnd retry.
    // Tight on both sides: the correct diagnostic must be present, and
    // the wrong (labelEnd-retry) diagnostic must NOT be present.
    assert.match(msg, /salvage:.*prior-clear retry failed/);
    assert.ok(
      !/labelEnd retry failed/.test(msg),
      `prior-clear failure must not surface a labelEnd-retry diagnostic; got: ${msg}`,
    );
    // Three setLabel calls fired: (A), original (B), retry of (B).
    assert.equal(
      calls,
      3,
      "setLabel was called three times: (A) write + (B) original + (B) retry",
    );
    // Error.cause preserves the original (B) throw.
    assert.ok(thrown instanceof Error);
    if (thrown instanceof Error) {
      assert.ok(
        thrown.cause instanceof Error,
        "thrown.cause must preserve the original (B) Error",
      );
      if (thrown.cause instanceof Error) {
        assert.match(thrown.cause.message, /prior-clear boom #2/);
      }
    }
    // (A) succeeded: the new summary still carries anchor:b. Even when
    // the prior-clear permanently fails, the new write survives so
    // single-call navigation still resolves correctly via
    // findLabeledEntry's leaf→root walk.
    let summaryWithLabelEnd: string | null = null;
    for (const e of sm.getBranch()) {
      if (e.type === "branch_summary" && sm.getLabel(e.id) === "anchor:b") {
        summaryWithLabelEnd = e.id;
        break;
      }
    }
    assert.ok(
      summaryWithLabelEnd,
      "the new branch_summary must carry anchor:b (call #1 succeeded before (B) threw)",
    );
    // Synthetic landed.
    const leafId = sm.getLeafId();
    assert.ok(leafId);
    const leaf = sm.getEntry(leafId as string);
    assert.ok(leaf && leaf.type === "message");
    if (leaf?.type === "message" && leaf.message.role === "assistant") {
      const c0 = (
        leaf.message.content as Array<{ type: string; id?: string }>
      )[0];
      assert.equal(c0.type, "toolCall");
      assert.equal(c0.id, "tc-rewind");
    }
  });
});

// =============================================================================
// rewind error branches
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
    setupRewindable(sm, pi);
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
    setupRewindable(sm, pi);
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
    setupRewindable(sm, pi);
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

  it("chained-rewind no-turns: synthetic-only intervening trips the boundary guard", async () => {
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

  it("chained-rewind discriminator: real navigate_tree call with nonzero usage does NOT trip the guard", async () => {
    // The chained-rewind no-turns guard discriminates THIS extension's
    // synthetic (zero usage + stopReason: 'toolUse') from a real
    // navigate_tree assistant turn (nonzero usage from the model call).
    // False-positive avoidance pin: a single intervening message shaped
    // like navigate_tree but with nonzero `usage.input`/`output` must
    // fall through the synthetic-shape check, so the rewind proceeds
    // normally (summarize is invoked). A regression that simplifies the
    // discriminator ("just check the toolCall name") would re-classify
    // this as a synthetic and trip a spurious 'Already at synthetic
    // boundary' error \u2014 catching that here.
    let summarizeCalled = false;
    const spySummarize = (async () => {
      summarizeCalled = true;
      return {
        summary: "## Goal\nspy.\n## Progress\n### Done\nx.\n## Next Steps\ny.",
        readFiles: [] as string[],
        modifiedFiles: [] as string[],
        aborted: false,
      };
    }) as typeof fakeSummarize;

    const { sm, pi, tool, ctx } = setup({ summarize: spySummarize });
    const t1 = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(t1.assistantId, "anchor:b");

    // Append a single intervening assistant turn whose lone content
    // block is a navigate_tree toolCall AND whose usage carries
    // nonzero input/output \u2014 the shape of a real model-issued
    // navigate_tree call. The synthetic discriminator must NOT match.
    sm.appendMessage({
      role: "assistant",
      content: [
        {
          type: "toolCall",
          id: "real-navtree-tc",
          name: "navigate_tree",
          arguments: { action: "anchor", name: "impl-start" },
        },
      ],
      api: "anthropic",
      provider: "claude",
      model: "claude-sonnet-4-5",
      stopReason: "toolUse",
      timestamp: Date.now(),
      usage: {
        // Nonzero usage \u2014 this is what distinguishes a real model call
        // from this extension's synthetic (which pins both to 0).
        input: 100,
        output: 50,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 150,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
    } as never);

    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);

    const r = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        labelStart: "b",
        labelEnd: "c",
        summaryFocus:
          "Preserve user instructions and continue past the real navigate_tree call.",
      },
      undefined,
      undefined,
      ctx,
    );
    // Pin: rewind proceeded (no boundary-guard error), summarize fired.
    assert.equal(
      r.isError,
      undefined,
      "rewind must NOT trip the synthetic-boundary guard on a real navigate_tree call with nonzero usage",
    );
    assert.equal(
      summarizeCalled,
      true,
      "summarize must run when the lone intervening message is a real (nonzero-usage) navigate_tree call",
    );
  });

  it("chained-rewind discriminator: lone intervening text-only assistant does NOT trip the guard", async () => {
    // Synthetic-discriminator fall-through: a text-only assistant message
    // (no toolCall block) must fall through the guard's first check
    // (`block.type === 'toolCall'`) so the rewind proceeds normally and
    // summarize is invoked. A regression that drops the block-shape gate
    // would mis-classify any assistant turn as 'synthetic' and trip a
    // spurious 'Already at synthetic boundary' error — catching that here.
    let summarizeCalled = false;
    const spySummarize = (async () => {
      summarizeCalled = true;
      return {
        summary: "## Goal\nspy.\n## Progress\n### Done\nx.\n## Next Steps\ny.",
        readFiles: [] as string[],
        modifiedFiles: [] as string[],
        aborted: false,
      };
    }) as typeof fakeSummarize;

    const { sm, pi, tool, ctx } = setup({ summarize: spySummarize });
    const t1 = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(t1.assistantId, "anchor:b");

    // Append a single intervening assistant turn whose lone content
    // block is a `text` block (no toolCall). The guard's `block.type ===
    // 'toolCall'` predicate must short-circuit before reaching the
    // synthetic-shape check, so the rewind proceeds.
    sm.appendMessage({
      role: "assistant",
      content: [{ type: "text", text: "thinking aloud, no tool call" }],
      api: "anthropic",
      provider: "claude",
      model: "claude-sonnet-4-5",
      stopReason: "endTurn",
      timestamp: Date.now(),
      usage: {
        input: 100,
        output: 50,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 150,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
    } as never);

    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);

    const r = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        labelStart: "b",
        labelEnd: "c",
        summaryFocus:
          "Preserve user instructions and continue past the text-only intervening assistant.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(
      r.isError,
      undefined,
      "rewind must NOT trip the synthetic-boundary guard on a text-only assistant",
    );
    assert.equal(
      summarizeCalled,
      true,
      "summarize must run when the lone intervening message is a text-only assistant",
    );
  });

  it("chained-rewind discriminator: lone intervening non-navigate_tree toolCall does NOT trip the guard", async () => {
    // Synthetic-discriminator fall-through: a synthetic-shaped
    // (zero-usage, stopReason 'toolUse') assistant whose toolCall is for
    // a DIFFERENT tool (e.g. `bash`) must fall through the
    // `name === 'navigate_tree'` predicate so the rewind proceeds. A
    // regression that drops the name gate would mis-classify any
    // zero-usage toolCall as our synthetic and trip a spurious
    // 'Already at synthetic boundary' error — catching that here.
    let summarizeCalled = false;
    const spySummarize = (async () => {
      summarizeCalled = true;
      return {
        summary: "## Goal\nspy.\n## Progress\n### Done\nx.\n## Next Steps\ny.",
        readFiles: [] as string[],
        modifiedFiles: [] as string[],
        aborted: false,
      };
    }) as typeof fakeSummarize;

    const { sm, pi, tool, ctx } = setup({ summarize: spySummarize });
    const t1 = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(t1.assistantId, "anchor:b");

    // Append a single intervening assistant turn whose lone content
    // block is a non-navigate_tree toolCall (`bash`) AND is otherwise
    // synthetic-shaped (zero usage, stopReason 'toolUse'). The
    // discriminator's `name === 'navigate_tree'` clause must short-
    // circuit before the shape check matches, so the rewind proceeds.
    sm.appendMessage({
      role: "assistant",
      content: [
        {
          type: "toolCall",
          id: "bash-tc",
          name: "bash",
          arguments: { command: "ls" },
        },
      ],
      api: "anthropic",
      provider: "claude",
      model: "claude-sonnet-4-5",
      stopReason: "toolUse",
      timestamp: Date.now(),
      usage: {
        // Synthetic-shaped (zero input/output) — only the toolCall name
        // distinguishes this from our synthetic. The name gate is the
        // load-bearing discriminator here.
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
    } as never);

    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);

    const r = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        labelStart: "b",
        labelEnd: "c",
        summaryFocus:
          "Preserve user instructions and continue past the non-navigate_tree intervening toolCall.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(
      r.isError,
      undefined,
      "rewind must NOT trip the synthetic-boundary guard on a non-navigate_tree toolCall",
    );
    assert.equal(
      summarizeCalled,
      true,
      "summarize must run when the lone intervening message is a non-navigate_tree toolCall",
    );
  });
});

// =============================================================================
// installPrepareNextTurn cross-extension preservation
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

  it("prior wrapper's context.systemPrompt + tools propagate through the chain", async () => {
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

  it("partial prior context: missing tools is filled from agent.state.tools", async () => {
    // A foreign wrapper that returns a partial `context` (only
    // systemPrompt; no tools). Pi's loop wholesale-replaces context
    // (`currentContext = ctx ?? currentContext`), so without our
    // defensive fill the next turn's context.tools would be undefined.
    // Pin: our wrapper falls back to agent.state.tools when the prior
    // didn't supply one.
    const sm = SessionManager.inMemory("/tmp");
    sm.appendMessage({
      role: "user",
      content: [{ type: "text", text: "hi" }],
    } as never);
    const fake = makeFakeSession(sm);
    fake.agent.state.tools = ["agent-state-tool"];
    fake.agent.state.systemPrompt = "AGENT_STATE_PROMPT";
    fake.agent.prepareNextTurn = async () => ({
      // Partial: only systemPrompt; no tools, no messages.
      context: {
        systemPrompt: "FOREIGN_PROMPT",
      },
    });
    __testHooks.installPrepareNextTurn(fake as unknown as AgentSession);
    type Pnt = (...args: unknown[]) => Promise<{
      context?: {
        systemPrompt?: unknown;
        tools?: unknown;
        messages?: unknown;
      };
    }>;
    const fn = fake.agent.prepareNextTurn as Pnt;
    const r = await fn();
    // Foreign systemPrompt wins (it was set explicitly).
    assert.equal(r.context?.systemPrompt, "FOREIGN_PROMPT");
    // tools fell back to agent.state.tools — the defensive fill.
    assert.deepEqual(r.context?.tools, ["agent-state-tool"]);
    assert.deepEqual(r.context?.messages, sm.buildSessionContext().messages);
  });

  it("partial prior context: missing systemPrompt is filled from agent.state.systemPrompt", async () => {
    // Symmetric to the tools-fill test above. A foreign wrapper that
    // returns only `tools` (no systemPrompt) must have systemPrompt
    // filled from agent.state.systemPrompt by our defensive merge.
    // Pin: a regression that drops the systemPrompt fallback line
    // (keeping only the tools fill) would let the next turn's context
    // ship with `systemPrompt: undefined` and lose the agent's prompt.
    const sm = SessionManager.inMemory("/tmp");
    sm.appendMessage({
      role: "user",
      content: [{ type: "text", text: "hi" }],
    } as never);
    const fake = makeFakeSession(sm);
    fake.agent.state.tools = ["agent-state-tool"];
    fake.agent.state.systemPrompt = "AGENT_STATE_PROMPT";
    fake.agent.prepareNextTurn = async () => ({
      // Partial: only tools; no systemPrompt, no messages.
      context: {
        tools: ["foreign-tool"],
      },
    });
    __testHooks.installPrepareNextTurn(fake as unknown as AgentSession);
    type Pnt = (...args: unknown[]) => Promise<{
      context?: {
        systemPrompt?: unknown;
        tools?: unknown;
        messages?: unknown;
      };
    }>;
    const fn = fake.agent.prepareNextTurn as Pnt;
    const r = await fn();
    // Foreign tools wins (it was set explicitly).
    assert.deepEqual(r.context?.tools, ["foreign-tool"]);
    // systemPrompt fell back to agent.state.systemPrompt \u2014 the defensive fill.
    assert.equal(r.context?.systemPrompt, "AGENT_STATE_PROMPT");
    assert.deepEqual(r.context?.messages, sm.buildSessionContext().messages);
  });

  it("explicit-undefined prior context: tools=undefined still falls back to agent.state.tools", async () => {
    // Regression pin for the merge-order bug: when the prior wrapper
    // returns `{ context: { systemPrompt: 'X', tools: undefined } }`
    // (key present, value explicitly undefined), the merged context
    // must still fall back to `agent.state.tools` rather than
    // ship `tools: undefined` to pi's loop. The pre-fix shape
    // (defaults written first, `...priorContext` spread after) silently
    // re-introduced `undefined` because the trailing spread overwrote
    // the fallback. Post-fix the spread runs first and the explicit
    // fallbacks overlay any `undefined` value the spread brought back.
    const sm = SessionManager.inMemory("/tmp");
    sm.appendMessage({
      role: "user",
      content: [{ type: "text", text: "hi" }],
    } as never);
    const fake = makeFakeSession(sm);
    fake.agent.state.tools = ["agent-state-tool"];
    fake.agent.state.systemPrompt = "AGENT_STATE_PROMPT";
    fake.agent.prepareNextTurn = async () => ({
      context: {
        systemPrompt: "FOREIGN_PROMPT",
        tools: undefined,
      },
    });
    __testHooks.installPrepareNextTurn(fake as unknown as AgentSession);
    type Pnt = (...args: unknown[]) => Promise<{
      context?: {
        systemPrompt?: unknown;
        tools?: unknown;
        messages?: unknown;
      };
    }>;
    const fn = fake.agent.prepareNextTurn as Pnt;
    const r = await fn();
    // Foreign systemPrompt wins (set to a real value).
    assert.equal(r.context?.systemPrompt, "FOREIGN_PROMPT");
    // The critical pin: tools must NOT be undefined \u2014 the explicit
    // fallback overlays the spread's undefined re-introduction.
    assert.deepEqual(
      r.context?.tools,
      ["agent-state-tool"],
      "tools=undefined from prior must fall back to agent.state.tools",
    );
    assert.deepEqual(r.context?.messages, sm.buildSessionContext().messages);
  });

  it("prior throwing in prepareNextTurn: error propagates verbatim (no swallow)", async () => {
    // If a foreign extension's prepareNextTurn throws, our wrapper
    // re-throws — we don't swallow + log + fall through, because that
    // would mask the foreign extension's bug behind our context-refresh.
    // Pin: the throw escapes verbatim so the failure surfaces at the
    // pi loop boundary where it's actionable. A future change to catch
    // + best-effort recover would surface here.
    const sm = SessionManager.inMemory("/tmp");
    sm.appendMessage({
      role: "user",
      content: [{ type: "text", text: "hi" }],
    } as never);
    const fake = makeFakeSession(sm);
    fake.agent.prepareNextTurn = async () => {
      throw new Error("foreign boom");
    };
    __testHooks.installPrepareNextTurn(fake as unknown as AgentSession);
    type Pnt = (...args: unknown[]) => Promise<unknown>;
    const fn = fake.agent.prepareNextTurn as Pnt;
    let thrown: unknown;
    try {
      await fn();
    } catch (e) {
      thrown = e;
    }
    assert.ok(thrown instanceof Error);
    if (thrown instanceof Error) assert.equal(thrown.message, "foreign boom");
  });

  it("returns without throwing when agent is missing", () => {
    // Pin the `if (!agent) return` early-exit in installPrepareNextTurn:
    // a session-shaped object that lacks an `agent` field must not throw
    // when the patch's wrapper runs `installPrepareNextTurn(this)`. This
    // protects against pi internals shape changes that drop the field
    // or rename it — the bootstrap degrades to no-op rather than
    // crashing the prompt patch (which would prevent any session from
    // calling prompt() at all).
    const sm = SessionManager.inMemory("/tmp");
    const fakeMissingAgent = { sessionManager: sm };
    assert.doesNotThrow(() => {
      __testHooks.installPrepareNextTurn(
        fakeMissingAgent as unknown as AgentSession,
      );
    });
  });
});

// =============================================================================
// non-assistant oldLeafEntry fallback
// =============================================================================

describe("dispatch: rewind beforeTokens fallback", () => {
  it("non-assistant leaf: beforeTokens uses estimateActiveBranchTokens fallback (smoke)", async () => {
    // Build chain ending in a user message (not an assistant). The
    // happy-path beforeTokens math gates on role==='assistant' &&
    // parentId; otherwise falls back to estimateActiveBranchTokens.
    // Smoke-level pin: a sensible non-zero contextBefore lands in details.
    const { sm, pi, tool, ctx } = setup();
    setupRewindable(sm, pi);
    // Append a trailing user message so the leaf isn't an assistant.
    sm.appendMessage({
      role: "user",
      content: [{ type: "text", text: "u3" }],
    } as never);

    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);

    // Pre-snapshot: leaf at execute time IS NOT a role==='assistant'
    // entry. This is the precondition for the fallback branch — without
    // it the test would silently exercise the happy path.
    const oldLeafId = sm.getLeafId();
    assert.ok(oldLeafId);
    const oldLeaf = sm.getEntry(oldLeafId as string);
    assert.ok(oldLeaf && oldLeaf.type === "message");
    if (oldLeaf && oldLeaf.type === "message") {
      assert.equal(
        oldLeaf.message.role,
        "user",
        "precondition: oldLeaf must NOT be assistant for fallback branch",
      );
    }

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
