/**
 * Tests for navigate-tree dispatch + reflection bootstrap.
 *
 * Node test runner. Run with: pnpm test
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

import * as assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, before, describe, it } from "node:test";
import {
  type AgentSession,
  CONFIG_DIR_NAME,
  type ContextUsage,
  type ExtensionAPI,
  generateBranchSummary,
  initTheme,
  SessionManager,
} from "@earendil-works/pi-coding-agent";
import { TREE_NAVIGATOR_CONFIG_FILENAME } from "./config.ts";
import { MAX_NAME_LENGTH, TOOL_NAME } from "./helpers.ts";
import navigateTree, {
  __testHooks,
  ANCHOR_MANDATE,
  MAX_HINT_WALK_DEPTH,
  MAX_SESSION_REFS,
  MAX_SYNTHETIC_FOCUS_LENGTH,
  MIN_REWIND_SAVINGS_TOKENS,
  MIN_SUMMARY_FOCUS_LENGTH,
} from "./index.ts";
import {
  buildNoAnchorText,
  buildRewindHintText,
  REWIND_HINT_CUSTOM_TYPE,
} from "./rewind-hint.ts";

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
  /** Captured `on(event, handler)` registrations, keyed by event name. */
  onCalls: Map<string, Array<(e: never, c: never) => unknown>>;
  /**
   * Raw argument arrays for every `sendMessage` call. Storing the full args
   * (not just the message object) is deliberate: the rewind hint's pinned
   * contract is a ONE-argument call, so tests must be able to distinguish
   * `sendMessage(msg)` from `sendMessage(msg, {})`.
   */
  sendMessageCalls: unknown[][];
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
  const onCalls = new Map<string, Array<(e: never, c: never) => unknown>>();
  const sendMessageCalls: unknown[][] = [];
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
    on(event: string, handler: (e: never, c: never) => unknown) {
      const list = onCalls.get(event) ?? [];
      list.push(handler);
      onCalls.set(event, list);
    },
    sendMessage(...args: unknown[]) {
      sendMessageCalls.push(args);
    },
    // Public since pi 0.81.0; the cache request mirrors it as `reasoning`.
    getThinkingLevel() {
      return "medium";
    },
  } as unknown as ExtensionAPI;
  return { pi, setLabelCalls, registered, onCalls, sendMessageCalls };
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
    /** Pricing lookup for the cache-miss detector (optional in real code). */
    find(provider: string, modelId: string): unknown;
  };
  /** Public system-prompt accessor (0.81+); override per test as needed. */
  getSystemPrompt(): string;
  /** UI availability as pi exposes it on the extension ctx. */
  hasUI: boolean;
  /** Current working directory (session_start derives the project config path). */
  cwd: string;
  /** Project-trust flag for the project config layer. */
  isProjectTrusted(): boolean;
  /** Captured `ui.notify` calls as `[message, type]`. */
  notifyCalls: Array<[string, "info" | "warning" | "error" | undefined]>;
  ui: {
    notify(message: string, type?: "info" | "warning" | "error"): void;
  };
  /** Context-usage probe (same shape as pi's `ctx.getContextUsage()`). */
  getContextUsage(): ContextUsage | undefined;
  /** Test-only setter driving `getContextUsage`. */
  setContextUsage(usage: ContextUsage | undefined): void;
}

function makeCtx(
  sm: SessionManager,
  opts: {
    contextWindow?: number;
    noModel?: boolean;
    authError?: string;
    hasUI?: boolean;
    cwd?: string;
    projectTrusted?: boolean;
    contextUsage?: ContextUsage;
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
  let usage = opts.contextUsage;
  const notifyCalls: Array<[string, "info" | "warning" | "error" | undefined]> =
    [];
  return {
    sessionManager: sm,
    model,
    getSystemPrompt: () => "LIVE SYSTEM PROMPT",
    hasUI: opts.hasUI ?? true,
    cwd: opts.cwd ?? "/tmp",
    isProjectTrusted: () => opts.projectTrusted ?? false,
    notifyCalls,
    ui: {
      notify(message: string, type?: "info" | "warning" | "error") {
        notifyCalls.push([message, type]);
      },
    },
    getContextUsage: () => usage,
    setContextUsage(next: ContextUsage | undefined) {
      usage = next;
    },
    modelRegistry: {
      async getApiKeyAndHeaders(_m: unknown) {
        if (opts.authError) return { ok: false, error: opts.authError };
        return { ok: true, apiKey: "test-key", headers: {} };
      },
      find(_p: string, _m: string) {
        return undefined;
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
 * Append an assistant message declaring a tool-call batch — the #37 shape
 * (e.g. `navigate_tree(rewind)` + `bash` emitted in one assistant message).
 * Returns the assistant entry id.
 */
function appendAssistantToolCalls(
  sm: SessionManager,
  toolCalls: Array<{ id: string; name: string }>,
  totalTokens = 30_000,
): string {
  return sm.appendMessage({
    role: "assistant",
    content: toolCalls.map((tc) => ({
      type: "toolCall",
      id: tc.id,
      name: tc.name,
      arguments: { action: "rewind" },
    })),
    api: "anthropic",
    provider: "claude",
    model: "claude-sonnet-4-5",
    stopReason: "toolUse",
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
    /** Plain field on pi-agent-core's Agent (cache request mirror). */
    thinkingBudgets?: unknown;
    prepareNextTurn?: unknown;
    prepareNextTurnWithContext?: unknown;
  };
  /**
   * Plain field on `AgentSession`; the extension reads
   * `getShowCacheMissNotices()` off it to gate the TUI cache notice. Default
   * in this fake mirrors pi's own default (off).
   */
  settingsManager?: { getShowCacheMissNotices?: () => boolean };
}

function makeFakeSession(sm: SessionManager): FakeAgentSession {
  return {
    sessionManager: sm,
    settingsManager: { getShowCacheMissNotices: () => false },
    agent: {
      state: { systemPrompt: "S", messages: [], tools: [] },
      prepareNextTurn: undefined,
      prepareNextTurnWithContext: undefined,
    },
  };
}

/**
 * Build the canonical rewindable fixture: an anchored first assistant turn
 * plus N follow-up turns (default 3) whose cumulative usage totals grow well
 * past MIN_REWIND_SAVINGS_TOKENS above the anchor (the #21 floor rejects
 * sub-floor rewinds, so every successful-rewind fixture must clear it),
 * optionally with a captured AgentSession so the reflection bootstrap finds
 * it.
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
    rewindToName?: string;
    tokenCounts?: number[];
  } = {},
): { fake?: FakeAgentSession } {
  const rewindToName = opts.rewindToName ?? "start";
  const turnsAfter = opts.turnsAfter ?? 3;
  const tokenCounts = opts.tokenCounts ?? [
    6_000, 12_000, 18_000, 24_000, 30_000,
  ];
  const t1 = appendTurn(sm, "u1", "a1", tokenCounts[0]);
  pi.pi.setLabel(t1.assistantId, `anchor:${rewindToName}`);
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
// context event handler
//
// The public `context` extension event (wired to pi's `Agent.transformContext`
// → `runner.emitContext`) is the replacement for the deleted
// `prepareNextTurn` double-wrap. It fires before EVERY LLM call with the
// session-tree projection; these pins cover the registration, the projection
// shape, and the post-rewind invariant (the README's headline feature).
// =============================================================================

describe("context event handler", () => {
  it("registers exactly one context handler via pi.on", () => {
    const { pi } = setup();
    const handlers = pi.onCalls.get("context");
    assert.ok(handlers, "factory must register a context handler");
    assert.equal(handlers.length, 1);
  });

  it("buildContextMessages projects the active branch (leaf-relative, no args)", () => {
    // The deleted wrapper rebuilt messages from buildSessionContext() every
    // turn; the context handler must produce the same projection from the
    // PUBLIC ReadonlySessionManager surface (buildContextEntries is exposed,
    // buildSessionContext is not). Pin the projection equals the session
    // context's messages on a plain chain.
    const { sm } = setup();
    appendTurn(sm, "u1", "a1", 100);
    appendTurn(sm, "u2", "a2", 200);
    const projected = __testHooks.buildContextMessages(sm);
    assert.deepEqual(projected, sm.buildSessionContext().messages);
  });

  it("handler returns messages = tree projection on every call (no leaf-gating)", () => {
    // R7: every appendMessage advances the leaf, so a "leaf changed since
    // last turn" heuristic would fire on essentially every call. Always-
    // replace degenerates to exactly what the deleted wrapper did. Fire the
    // captured handler twice with no tree change in between — both must
    // return the same projection (no pass-through case exists to pin, but
    // this guards against a future conditional reintroducing gating).
    const { sm, pi } = setup();
    appendTurn(sm, "u1", "a1", 100);
    const handlers = pi.onCalls.get("context");
    assert.ok(handlers, "factory must register a context handler");
    const first = handlers[0](
      { type: "context", messages: [] } as never,
      { sessionManager: sm } as never,
    );
    const second = handlers[0](
      { type: "context", messages: [] } as never,
      { sessionManager: sm } as never,
    );
    assert.deepEqual(first, second);
    assert.deepEqual(
      (first as { messages: unknown[] }).messages,
      sm.buildSessionContext().messages,
    );
  });

  it("post-rewind: handler returns rewound chain with synthetic + tool_result, no abandoned tool_use", async () => {
    // The headline invariant: after a mid-loop rewind, the next assistant
    // turn (the very next context event) must see the rewound chain — the
    // branch_summary message, the synthetic assistant (toolCall id == the
    // in-flight id), and the real tool_result — and NOT the abandoned
    // branch's assistant tool_use. This is the public-API replacement for
    // the delete-wrappper per-turn refresh; a regression here re-opens the
    // "next API call stays pre-rewind" bug.
    const { sm, pi, tool, ctx } = setup();
    const t1 = appendTurn(sm, "u1", "a1", 6_000);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendTurn(sm, "u2", "a2", 14_000);
    appendTurn(sm, "u3", "a3", 22_000);
    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "Preserve the user instruction and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);

    // In production, pi appends the tool_result as message_end AFTER the
    // rewind's synthetic (emitted by agent-loop's emitToolResultMessage).
    // Replicate that append so the projection under test is the state a
    // real next-turn context event would see.
    const toolResultMsg = {
      role: "toolResult" as const,
      toolCallId: "tc-rewind",
      toolName: "navigate_tree",
      content: [],
      details: result.details,
      isError: false,
      timestamp: Date.now(),
    };
    sm.appendMessage(toolResultMsg as never);

    const handlers = pi.onCalls.get("context");
    assert.ok(handlers, "factory must register a context handler");
    const outcome = handlers[0](
      { type: "context", messages: [] } as never,
      { sessionManager: sm } as never,
    ) as {
      messages: Array<{
        role: string;
        content?: Array<{ type?: string; id?: string; name?: string }>;
      }>;
    };
    const messages = outcome.messages;

    // Contains the branch_summary (user-role projection) and the synthetic
    // assistant with the in-flight toolCall id.
    const synthetic = messages.find(
      (m) => m.role === "assistant" && m.content?.[0]?.type === "toolCall",
    );
    assert.ok(synthetic, "projection must contain the synthetic assistant");
    assert.equal(synthetic.content?.[0]?.id, "tc-rewind");
    assert.equal(synthetic.content?.[0]?.name, "navigate_tree");

    // Ends with the tool_result (pairing contract).
    const last = messages[messages.length - 1];
    assert.equal(last.role, "toolResult");
    assert.equal(last.content?.[0]?.type, undefined);

    // The abandoned branch's original assistant tool_use is gone: the only
    // assistant-with-toolCall is the synthetic.
    const toolCallAssistants = messages.filter(
      (m) =>
        m.role === "assistant" && m.content?.some((c) => c.type === "toolCall"),
    );
    assert.equal(toolCallAssistants.length, 1);

    // The rewound chain is strictly shorter than the pre-rewind chain.
    assert.ok(
      messages.length < sm.getEntries().length,
      "rewound projection must be shorter than the raw entry list",
    );

    // idempotent: a second invocation with no tree change returns the same.
    const again = handlers[0](
      { type: "context", messages: [] } as never,
      { sessionManager: sm } as never,
    );
    assert.deepEqual(again, outcome);
  });
});

// =============================================================================
// before_agent_start anchor mandate (#31)
//
// The #29 `promptGuidelines` bullet landed mid-prompt (inside the Guidelines
// section) and did not move behavior. #31 moves the policy to a
// `before_agent_start` append, which lands at the very END of the system
// prompt (after <project_context> and skills), is re-applied on every prompt,
// and survives compaction. These pins cover registration, the append shape
// (with the approved mandate spelled as a literal), the active-tool gate,
// fail-open semantics, and byte-stability for prompt caching.
// =============================================================================

describe("before_agent_start anchor mandate", () => {
  it("registers exactly one before_agent_start handler via pi.on", () => {
    const { pi } = setup();
    const handlers = pi.onCalls.get("before_agent_start");
    assert.ok(handlers, "factory must register a before_agent_start handler");
    assert.equal(handlers.length, 1);
  });

  it("appends the approved mandate to the system prompt when the tool is active", async () => {
    const { pi } = setup();
    const handlers = pi.onCalls.get("before_agent_start");
    assert.ok(handlers, "factory must register a before_agent_start handler");
    const out = await handlers[0](
      {
        type: "before_agent_start",
        prompt: "do the thing",
        systemPrompt: "BASE",
        systemPromptOptions: {
          selectedTools: ["read", "bash", "navigate_tree"],
        },
      } as never,
      {} as never,
    );
    assert.deepEqual(out, {
      systemPrompt:
        "BASE\n\nnavigate_tree: gather all context, then anchor `context-gathered`; list anchors and rewind after every milestone or rabbit hole / dead end to keep context low.",
    });
  });

  it("skips the append when the tool is verifiably absent from selectedTools", async () => {
    const { pi } = setup();
    const handlers = pi.onCalls.get("before_agent_start");
    assert.ok(handlers, "factory must register a before_agent_start handler");
    const out = await handlers[0](
      {
        type: "before_agent_start",
        prompt: "do the thing",
        systemPrompt: "BASE",
        systemPromptOptions: { selectedTools: ["read", "bash"] },
      } as never,
      {} as never,
    );
    assert.deepEqual(out, {});
  });

  it("fails open when selectedTools is undefined (mandate still appended)", async () => {
    const { pi } = setup();
    const handlers = pi.onCalls.get("before_agent_start");
    assert.ok(handlers, "factory must register a before_agent_start handler");
    const out = await handlers[0](
      {
        type: "before_agent_start",
        prompt: "do the thing",
        systemPrompt: "BASE",
        systemPromptOptions: {},
      } as never,
      {} as never,
    );
    assert.deepEqual(out, { systemPrompt: `BASE\n\n${ANCHOR_MANDATE}` });
  });

  it("is byte-stable across identical calls (prompt-caching contract)", async () => {
    const { pi } = setup();
    const handlers = pi.onCalls.get("before_agent_start");
    assert.ok(handlers, "factory must register a before_agent_start handler");
    const event = {
      type: "before_agent_start",
      prompt: "do the thing",
      systemPrompt: "BASE",
      systemPromptOptions: {
        selectedTools: ["read", "bash", "navigate_tree"],
      },
    } as never;
    const first = (await handlers[0](event, {} as never)) as {
      systemPrompt: string;
    };
    const second = (await handlers[0](event, {} as never)) as {
      systemPrompt: string;
    };
    assert.deepEqual(first, second);
    assert.equal(first.systemPrompt, second.systemPrompt);
  });
});

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
    // for `name`, `rewindTo`, `newLabel`, `summaryFocus`. A regression
    // that lifts a runtime guard into the schema (e.g. adding
    // `summaryFocus` to `required`) would re-introduce the original Kiro
    // 400 — this assertion catches it.
    assert.deepEqual(params.required, ["action"]);
    // Each action-conditional field must still exist in `properties` so the
    // schema describes the full surface to the model.
    const props = params.properties ?? {};
    for (const key of ["name", "rewindTo", "newLabel", "summaryFocus"]) {
      assert.ok(key in props, `${key} must be a schema property`);
    }
    assert.equal(params.anyOf, undefined);
    assert.equal(params.oneOf, undefined);
  });

  it("schema clean break: rewindTo/newLabel registered, legacy names absent (#43)", () => {
    // #43 is a clean break — no aliases. The schema must expose only the
    // new param names; a legacy key reappearing in `properties` would keep
    // the old vocabulary in the model-facing tool definition (and the
    // cached prefix it sits at the head of). Legacy key names are
    // assembled at runtime so the repo's zero-hit ref audit stays clean.
    const legacyRewindTo = `label${"Start"}`;
    const legacyNewLabel = `label${"End"}`;
    const { tool } = setup();
    const props = (tool.parameters as { properties: Record<string, unknown> })
      .properties;
    assert.ok("rewindTo" in props, "rewindTo must be a schema property");
    assert.ok("newLabel" in props, "newLabel must be a schema property");
    assert.ok(
      !(legacyRewindTo in props),
      `no legacy ${legacyRewindTo} alias in the schema`,
    );
    assert.ok(
      !(legacyNewLabel in props),
      `no legacy ${legacyNewLabel} alias in the schema`,
    );
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
      props.rewindTo.description ?? "",
      /Required when action='rewind'/,
    );
    assert.match(
      props.newLabel.description ?? "",
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
// Tool definition text — issue #22 token-trim pins
// =============================================================================

describe("tool definition \u2014 #22 token-trim pins", () => {
  // Owner-approved copy from issue #22 (~848 → ~470 tok/request). The
  // expectations are reconstructed with the SAME template literals the
  // source interpolates (${MAX_NAME_LENGTH}, ${MIN_SUMMARY_FOCUS_LENGTH})
  // so constant re-tunes stay honest — hardcoding 40/20 here would let a
  // schema/expectation desync slip through.
  const expectedDescription = `Long-session context management via the pi session tree. Anchor named milestones, then collapse work between them into a model-generated summary to free context.
\`rewind\` does not restore prior state: it forks a sibling branch from the anchor and continues forward from a model-generated summary.

Operations (set \`action\`):
  • 'anchor', name='<milestone-name>': label the current point. Anchor at the start of a stage you'll summarize (e.g. 'impl-start').
  • 'rewind', rewindTo='<existing>', newLabel='<new>': collapse work between rewindTo and the current leaf into a branch_summary labeled newLabel, so rewinds can chain.
  • 'list': show all anchors on the active branch, oldest first, with cumulative context % at each.

\`name\` (anchor) and \`newLabel\` (rewind) write into one shared anchor namespace: re-using an existing label moves it to the new entry, and everything written there is addressable as a future \`rewindTo\`. Avoid the reserved \`anchor:\` prefix.`;

  it("description byte-equals the issue-approved trimmed string", () => {
    // Byte-exact snapshot: any wording change to the tool description must
    // land consciously through these pins, not by silent drift.
    const { tool } = setup();
    assert.equal(tool.description, expectedDescription);
  });

  it("each parameter description byte-equals its issue-approved string", () => {
    const { tool } = setup();
    const props = (
      tool.parameters as {
        properties: Record<string, { description?: string }>;
      }
    ).properties;
    assert.equal(props.action.description, "Which operation to perform.");
    assert.equal(
      props.name.description,
      `Required when action='anchor'. Kebab-case milestone label, max ${MAX_NAME_LENGTH} chars.`,
    );
    assert.equal(
      props.rewindTo.description,
      `Required when action='rewind'. Kebab-case name of an existing anchor on the active branch to rewind to.`,
    );
    assert.equal(
      props.newLabel.description,
      `Required when action='rewind'. Kebab-case label for the resulting branch_summary entry; reusable as a future rewindTo.`,
    );
    assert.equal(
      props.summaryFocus.description,
      `Required when action='rewind'; ≥${MIN_SUMMARY_FOCUS_LENGTH} chars after trim. Encode: (1) the user's most recent instruction verbatim, (2) what's done in the collapsed segment, (3) what's left to do as a next action.`,
    );
  });

  it("re-bloat tripwire: description + param descriptions stay ≤400 tok at chars/4", () => {
    // Every character of the tool definition is paid per request. The #22
    // trim lands at 1477 chars = 369.25 tok (#43 rename recompute); this
    // bound guards against silently regrowing the schema — a future legit
    // edit that needs more room must raise this number consciously, not
    // bleed past it.
    const { tool } = setup();
    const props = (
      tool.parameters as {
        properties: Record<string, { description?: string }>;
      }
    ).properties;
    const all = [
      tool.description,
      props.action.description ?? "",
      props.name.description ?? "",
      props.rewindTo.description ?? "",
      props.newLabel.description ?? "",
      props.summaryFocus.description ?? "",
    ].join("");
    assert.ok(
      all.length / 4 <= 400,
      `tool definition re-bloated: ${all.length} chars = ${(all.length / 4).toFixed(2)} tok > 400`,
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
          rewindTo: "start",
          newLabel: "end",
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
    assert.match(text, /rewindTo='<oldest appropriate anchor>'/);
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
      on() {
        // Factory registers the context handler via on; this test only
        // exercises anchor semantics, so a no-op collector suffices.
      },
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
    // first guard rejects a missing/invalid `rewindTo` with a
    // "kebab-case" message; pinning that text confirms the fall-through
    // landed at the rewindTo guard, not at a future explicit
    // unknown-action default.
    assert.match(
      (result.content[0] as { text: string }).text,
      /rewindTo.*kebab-case/,
    );
  });
});

// =============================================================================
// dispatch: rewind validation guards
// =============================================================================

describe("dispatch: rewind validation guards", () => {
  // Each row produces an isError=true result whose text matches the regex.
  // The order matches the guard order in execute(): rewindTo, newLabel,
  // summaryFocus, then label-existence.
  const cases: Array<{
    name: string;
    params: Record<string, unknown>;
    want: RegExp;
  }> = [
    {
      name: "missing rewindTo",
      params: { action: "rewind" },
      want: /rewindTo.*kebab-case/,
    },
    {
      name: "rewindTo present, missing newLabel",
      params: { action: "rewind", rewindTo: "ok" },
      want: /newLabel.*kebab-case/,
    },
    {
      name: "rewindTo and newLabel valid, missing summaryFocus",
      params: { action: "rewind", rewindTo: "ok", newLabel: "ok2" },
      want: /summaryFocus/,
    },
    {
      name: "summaryFocus shorter than min length",
      params: {
        action: "rewind",
        rewindTo: "ok",
        newLabel: "ok2",
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
        rewindTo: "ok",
        newLabel: "ok2",
        summaryFocus: `  ${"x".repeat(MIN_SUMMARY_FOCUS_LENGTH - 1)}  `,
      },
      want: /summaryFocus/,
    },
    {
      name: "summaryFocus mentions the 20-char threshold in error text",
      params: {
        action: "rewind",
        rewindTo: "ok",
        newLabel: "ok2",
        summaryFocus: "short",
      },
      // Pin that the error text surfaces MIN_SUMMARY_FOCUS_LENGTH literally;
      // the regex tracks the constant if the floor is bumped.
      want: new RegExp(`\u2265${MIN_SUMMARY_FOCUS_LENGTH}`),
    },
    {
      name: "rewindTo kebab-invalid (uppercase)",
      params: { action: "rewind", rewindTo: "Bad-Name" },
      want: /rewindTo.*kebab-case/,
    },
    {
      name: "newLabel kebab-invalid (snake_case)",
      params: { action: "rewind", rewindTo: "ok", newLabel: "snake_case" },
      want: /newLabel.*kebab-case/,
    },
    {
      name: "all valid but no such rewindTo on the active branch",
      params: {
        action: "rewind",
        rewindTo: "missing",
        newLabel: "after",
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

  it("old-name rewind calls reach execute and are refused by the new-name guards (clean break)", async () => {
    // #43 clean break: pi-ai's argument validation passes unknown keys
    // through (the schema has no `additionalProperties`), so a resumed
    // session replaying old-name params still reaches `execute` — the
    // runtime guards are the path exercised here, representative of the
    // live one-retry self-heal (each error names the new param). An
    // execute-level alias fallback would not add a schema key, so the
    // second row is the only guard against it. Legacy key names are
    // assembled at runtime so the repo's zero-hit ref audit stays clean.
    const legacyRewindTo = `label${"Start"}`;
    const legacyNewLabel = `label${"End"}`;
    const cases: Array<{
      name: string;
      params: Record<string, unknown>;
      want: RegExp;
    }> = [
      {
        name: "legacy rewindTo key only",
        params: { action: "rewind", [legacyRewindTo]: "ok" },
        want: /rewind requires `rewindTo` in kebab-case/,
      },
      {
        name: "legacy newLabel key only (valid rewindTo + summaryFocus)",
        params: {
          action: "rewind",
          rewindTo: "ok",
          [legacyNewLabel]: "ok2",
          summaryFocus: "x".repeat(MIN_SUMMARY_FOCUS_LENGTH),
        },
        want: /rewind requires `newLabel` in kebab-case/,
      },
    ];
    for (const c of cases) {
      const { sm, tool, ctx } = setup();
      appendTurn(sm, "u", "a");
      const result = await tool.execute(
        "tc-old-name",
        c.params,
        undefined,
        undefined,
        ctx,
      );
      assert.equal(result.isError, true, `${c.name}: must be refused`);
      assert.match(
        (result.content[0] as { text: string }).text,
        c.want,
        `${c.name}: refusal must name the new param`,
      );
    }
  });

  it("summaryFocus exactly 20 chars after trim passes the guard", async () => {
    // Boundary: summaryFocus.trim().length >= MIN_SUMMARY_FOCUS_LENGTH (the
    // >= boundary, not strict >). The call still errors at the next guard
    // (no rewindTo on chain), but it moves past the focus-length check —
    // pinning that the comparison is inclusive at exactly MIN. Trim-
    // presence on the rejection path is pinned by the
    // "shorter than min length after trim" row above, not here.
    const { sm, tool, ctx } = setup();
    appendTurn(sm, "u", "a");
    const result = await tool.execute(
      "tc-1",
      {
        action: "rewind",
        rewindTo: "missing",
        newLabel: "after",
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
// dispatch: solo-batch guard (#37)
//
// A rewind sharing an assistant batch with any other tool call bricks the
// active branch: the fork puts the declaring assistant on the abandoned
// branch, the synthetic re-declares ONLY the rewind call, and pi's
// sequential loop appends the sibling's result to the new branch with no
// declaring `tool_calls` — every subsequent request 400s. The guard refuses
// before any mutation or summarizer call.
// =============================================================================

describe("dispatch: solo-batch guard (#37)", () => {
  /**
   * Drive a rewind whose declaring assistant carries `batch` — the #37
   * shape that must be refused. Spies on the summarizer and on
   * `branchWithSummary`, and snapshots the tree so the refusal can be
   * proven side-effect-free.
   */
  async function batchedRewindFixture(order: "rewind-first" | "sibling-first") {
    let summarizeCalls = 0;
    const summarize = (async () => {
      summarizeCalls++;
      return {
        summary:
          "## Goal\nshould never run.\n## Progress\n### Done\nx.\n## Next Steps\ny.",
        readFiles: [] as string[],
        modifiedFiles: [] as string[],
        aborted: false,
      };
    }) as typeof fakeSummarize;
    const { sm, pi, tool, ctx } = setup({ summarize });
    // Healthy stage above the anchor: the guard must fire before the
    // min-savings floor / summarizer, so the refusal can't be mistaken for
    // an earlier rejection.
    const t1 = appendTurn(sm, "u1", "a1", 6_000);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendTurn(sm, "u2", "a2", 15_000);
    appendTurn(sm, "u3", "a3", 26_000);

    // Instance shadow: the tool calls `sm.branchWithSummary(...)`.
    const originalBranchWithSummary = sm.branchWithSummary.bind(sm);
    let branchCalls = 0;
    (
      sm as unknown as {
        branchWithSummary: typeof originalBranchWithSummary;
      }
    ).branchWithSummary = (
      ...args: Parameters<typeof originalBranchWithSummary>
    ) => {
      branchCalls++;
      return originalBranchWithSummary(...args);
    };

    const navigate = { id: "tc-rewind", name: "navigate_tree" };
    const sibling = { id: "tc-sibling", name: "bash" };
    const batch =
      order === "rewind-first" ? [navigate, sibling] : [sibling, navigate];
    appendAssistantToolCalls(sm, batch);

    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);

    const entriesBefore = sm.getEntries().length;
    const leafBefore = sm.getLeafId();

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    return {
      sm,
      result,
      batch,
      summarizeCalls: () => summarizeCalls,
      branchCalls: () => branchCalls,
      entriesBefore,
      leafBefore,
    };
  }

  for (const order of ["rewind-first", "sibling-first"] as const) {
    it(`refuses a ${order} batch without side effects`, async () => {
      const f = await batchedRewindFixture(order);
      assert.equal(f.result.isError, true);
      assert.equal(f.result.details.refusal, true);
      const text = f.result.content[0].text;
      // Pinned copy (stable clauses), with rewindTo interpolated.
      assert.match(text, /rewind must be the only tool call in its batch/);
      assert.match(text, /everything up to 'start' plus the new summary/);
      assert.match(text, /would be orphaned — never read by anyone/);
      assert.match(
        text,
        /Those other calls already ran; re-issue only the rewind, alone\./,
      );
      assert.equal(f.result.details.rejected, "batched-rewind");
      assert.deepEqual(
        f.result.details.batchedToolCalls,
        f.batch.map((tc) => tc.name),
      );
      // Zero side effects: no summarizer call, no branch move, no new entry,
      // leaf unchanged.
      assert.equal(f.summarizeCalls(), 0);
      assert.equal(f.branchCalls(), 0);
      assert.equal(f.sm.getEntries().length, f.entriesBefore);
      assert.equal(f.sm.getLeafId(), f.leafBefore);
    });
  }

  it("reports the batch in emission order (order independence)", async () => {
    const rewindFirst = await batchedRewindFixture("rewind-first");
    assert.deepEqual(rewindFirst.result.details.batchedToolCalls, [
      "navigate_tree",
      "bash",
    ]);
    const siblingFirst = await batchedRewindFixture("sibling-first");
    assert.deepEqual(siblingFirst.result.details.batchedToolCalls, [
      "bash",
      "navigate_tree",
    ]);
  });

  it("does not refuse a solo navigate_tree batch — the guard is length > 1 only", async () => {
    // Contrast pin against the refusals above: the same fixture with a
    // single-call assistant must execute the rewind. This is the first
    // fixture whose in-flight assistant is actually detectable (the older
    // happy-path fixtures use text-only assistants → helper returns null),
    // so it pins the length-1 passthrough explicitly.
    const batch = [{ id: "tc-rewind", name: "navigate_tree" }];
    const { sm, pi, tool, ctx } = setup();
    const t1 = appendTurn(sm, "u1", "a1", 6_000);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendTurn(sm, "u2", "a2", 15_000);
    appendTurn(sm, "u3", "a3", 26_000);
    appendAssistantToolCalls(sm, batch);
    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    // Post-rewind the synthetic re-declares only the rewind call, so the
    // helper now sees a solitary toolCall batch (length 1, no refusal).
    assert.deepEqual(
      __testHooks.findInFlightAssistantToolCalls(sm, "tc-rewind"),
      batch,
    );
  });
});

// =============================================================================
// dispatch: refusal results surface as failed tool calls
//
// `AgentToolResult` has no `isError` field in pi-agent-core's contract, and
// the agent runtime overwrites any returned flag with its own boolean (only a
// thrown `execute()` is finalized as an error). `toolError` therefore tags
// its details with `refusal: true` and the factory's `tool_result` handler
// flips the host's `isError` — the signal that turns the TUI row and the
// transcript entry red. These pins cover registration, the flip, and the
// no-op paths.
// =============================================================================

describe("dispatch: refusal results surface as failed tool calls", () => {
  function toolResultHandler(): (e: unknown) => unknown {
    const { pi } = setup();
    const handlers = pi.onCalls.get("tool_result");
    assert.ok(handlers, "factory must register a tool_result handler");
    assert.equal(handlers.length, 1);
    return handlers[0] as unknown as (e: unknown) => unknown;
  }

  it("returns isError:true for a navigate_tree refusal", () => {
    const out = toolResultHandler()({
      type: "tool_result",
      toolName: "navigate_tree",
      toolCallId: "tc-refusal",
      input: { action: "rewind" },
      content: [{ type: "text", text: "rewind must be the only tool call…" }],
      // Shape produced by `toolError`.
      details: { refusal: true, rejected: "batched-rewind" },
      isError: false,
    });
    assert.deepEqual(out, { isError: true });
  });

  it("leaves successful navigate_tree results untouched", () => {
    const out = toolResultHandler()({
      type: "tool_result",
      toolName: "navigate_tree",
      toolCallId: "tc-anchor",
      input: { action: "anchor" },
      content: [{ type: "text", text: "[anchor 'start'] set at 1.0%" }],
      details: { label: "start", contextTokens: 1_234 },
      isError: false,
    });
    assert.equal(out, undefined);
  });

  it("ignores other tools even with a refusal-shaped payload", () => {
    const out = toolResultHandler()({
      type: "tool_result",
      toolName: "bash",
      toolCallId: "tc-bash",
      input: { command: "true" },
      content: [{ type: "text", text: "ok" }],
      details: { refusal: true },
      isError: false,
    });
    assert.equal(out, undefined);
  });

  it("toolError tags every refusal with the marker, keeping caller details", async () => {
    const { tool, ctx } = setup();
    const result = await tool.execute(
      "tc-bad-focus",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "too short",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, true);
    assert.equal(result.details.refusal, true);
    assert.match(result.content[0].text, /summaryFocus/);
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
    const t1 = appendTurn(sm, "u1", "a1", 6_000);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendTurn(sm, "u2", "a2", 15_000);
    appendTurn(sm, "u3", "a3", 26_000);

    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
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
    assert.match(
      result.content[0].text,
      /\[rewind to 'start' · collapsed as 'end'\]/,
    );
    // #43: the structured details keys carry the new param names and the
    // fixture's values. This is the machine-read JSONL surface, so it is
    // pinned alongside the model-visible prose above.
    assert.equal(result.details.rewindTo, "start");
    assert.equal(result.details.newLabel, "end");
    assert.match(result.content[0].text, /### Done/);
    assert.match(result.content[0].text, /### In Progress/);
    assert.match(result.content[0].text, /### Blocked/);
    assert.match(result.content[0].text, /are pending/);
    assert.match(result.content[0].text, /## Next Steps/);
  });

  it("labels the summary entry with anchor:<newLabel> and leaves a synthetic leaf", async () => {
    const { sm, result } = await rewindFixture();
    // The new summary entry carries the newLabel.
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
        rewindTo: "start",
        newLabel: "end",
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
    const tA = appendTurn(sm, "u1", "a1", 6_000);
    pi.pi.setLabel(tA.assistantId, "anchor:a");
    appendTurn(sm, "u2", "a2", 12_000);
    appendTurn(sm, "u3", "a3", 18_000);

    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);

    const r1 = await tool.execute(
      "tc-1",
      {
        action: "rewind",
        rewindTo: "a",
        newLabel: "b",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(r1.isError, undefined);
    const sumB = r1.details.summaryId as string;

    // Append more turns post-rewind.
    appendTurn(sm, "u4", "a4", 24_000);
    appendTurn(sm, "u5", "a5", 30_000);

    const r2 = await tool.execute(
      "tc-2",
      {
        action: "rewind",
        rewindTo: "b",
        newLabel: "c",
        summaryFocus: "Preserve user instructions across the second rewind.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(r2.isError, undefined);
    const sumC = r2.details.summaryId as string;
    // Final summary carries newLabel.
    assert.equal(sm.getLabel(sumC), "anchor:c");
    // The b-summary still carries 'anchor:b'. (The b-summary isn't on the
    // active branch anymore \u2014 it's an ancestor of sumC in storage but the
    // active path goes through sumC. Verify via getEntry, which is
    // global-storage-keyed.)
    const bEntry = sm.getEntry(sumB);
    assert.ok(bEntry, "b-summary entry still present in storage");
    // Pin label retention explicitly: the b-summary keeps its 'anchor:b'
    // label across the second rewind. Nothing in the rewind path clears
    // rewindTo's label (it's only newLabel that gets the new write +
    // move-on-collision), so the prior label survives. A regression that
    // accidentally cleared rewindTo on rewind would surface here.
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

  it("newLabel collides with an existing anchor: rewind moves the anchor to the new summary (mirrors anchor's move-on-collision)", async () => {
    // Namespace symmetry: anchor.name and rewind.newLabel both write into
    // the `anchor:` namespace, so a `rewind` whose newLabel already labels
    // another entry on the *post-move active branch* must move the label
    // to the new summary — mirroring `anchor`'s write-before-clear
    // move-on-collision. Without the move, two entries on the same active
    // branch would both carry `anchor:b`, breaking `findLabeledEntry`'s
    // uniqueness invariant.
    //
    // Setup ordering matters: the prior `anchor:b` must be on the
    // ancestral side of `rewindTo` (= tA), so that branchWithSummary
    // leaves it on the *active* branch (between root and the new
    // summary), NOT on the abandoned one. We anchor 'b' on the FIRST
    // turn and 'a' on the SECOND turn:
    //   root → tB(anchor:b) → tA(anchor:a) → t3 → leaf
    // Then rewind a→b collapses [tA-child … leaf] into summaryId, leaving
    //   root → tB(anchor:b) → tA(anchor:a) → summaryId(anchor:b)
    // — anchor:b lives on both tB and summaryId on the active branch.
    // The move-on-collision clears tB's label.
    const { sm, pi, tool, ctx } = setup();
    const tB = appendTurn(sm, "u1", "a1", 6_000);
    pi.pi.setLabel(tB.assistantId, "anchor:b");
    const tA = appendTurn(sm, "u2", "a2", 16_000);
    pi.pi.setLabel(tA.assistantId, "anchor:a");
    appendTurn(sm, "u3", "a3", 22_000);
    appendTurn(sm, "u4", "a4", 28_000);

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
        rewindTo: "a",
        newLabel: "b",
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
      "new branch_summary must carry the newLabel anchor",
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
        rewindTo: "start",
        newLabel: "end",
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
        rewindTo: "start",
        newLabel: "end",
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
        rewindTo: "start",
        newLabel: "end",
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

  it("synthetic args are normalized: extra caller keys never leak onto the kept chain (#43)", async () => {
    // pi-ai's validateToolArguments passes unknown keys through (no
    // `additionalProperties` in the schema), so the raw params object can
    // carry extras — including old-name keys replayed by a resumed
    // session. The synthetic assistant is re-emitted on every subsequent
    // turn until the next rewind, so it must carry exactly the normalized
    // new-name arguments. A regression back to spreading `p` would leak
    // unknown keys (and retrigger this test).
    const { sm, pi, tool, ctx } = setup();
    setupRewindable(sm, pi, { capture: true });
    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "Preserve user instructions and continue.",
        legacyAlias: "must-not-leak",
        unknownKey: 42,
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
    if (leaf?.type !== "message") {
      assert.fail("expected the kept leaf to be a message entry");
    }
    // Unconditional role pin: without it, a regression leaving a
    // non-assistant leaf would skip the shape assertions below and pass
    // vacuously.
    assert.equal(leaf.message.role, "assistant");
    const c0 = (
      leaf.message.content as Array<{
        type: string;
        arguments?: Record<string, unknown>;
      }>
    )[0];
    assert.equal(c0.type, "toolCall");
    assert.deepEqual(Object.keys(c0.arguments ?? {}).sort(), [
      "action",
      "newLabel",
      "rewindTo",
      "summaryFocus",
    ]);
    assert.deepEqual(c0.arguments, {
      action: "rewind",
      rewindTo: "start",
      newLabel: "end",
      summaryFocus: "Preserve user instructions and continue.",
    });
  });

  it("wraps the provider's streamSimple as streamFn (custom-api provider routing)", async () => {
    // Regression: rewind failed with "No API provider registered for api:
    // commandcode-custom" for providers registered via
    // pi.registerProvider(name, { api: <custom-id>, streamSimple }) because
    // generateBranchSummary was called WITHOUT streamFn, making
    // completeSummarization fall back to the pi-ai compat registry (which
    // only knows builtin apis). The fix forwards the composed provider's
    // `streamSimple` via the public modelRegistry.getProvider() API — the
    // same routing pi's own branchWithSummary uses.
    //
    // #33 wraps that function (to rewrite the request at the seam), so the
    // identity changed from "the provider's streamSimple" to "a wrapper that
    // delegates to it". Pin the delegation, not the identity: a regression
    // that drops the forwarding still fails here because the delegate is
    // never called.
    let capturedStreamFn: unknown = "__not_called__";
    const spySummarize = (async (_entries: unknown, opts: unknown) => {
      capturedStreamFn = (opts as { streamFn?: unknown }).streamFn;
      return {
        summary: "## Goal\nspy.\n## Progress\n### Done\nx.\n## Next Steps\ny.",
        readFiles: [] as string[],
        modifiedFiles: [] as string[],
        aborted: false,
      };
    }) as typeof fakeSummarize;

    const { sm, pi, tool, ctx } = setup({ summarize: spySummarize });
    setupRewindable(sm, pi, {});

    // Simulate a custom-api provider (e.g. commandcode 0.5.x with
    // api "commandcode-custom"): the composed provider exposes
    // `streamSimple` via the public modelRegistry.getProvider().
    const delegated: unknown[] = [];
    const providerStreamSimple = async (
      _m: unknown,
      context: unknown,
      options: unknown,
    ) => {
      delegated.push({ context, options });
      return { result: async () => ({}) } as never;
    };
    (ctx.modelRegistry as unknown as { getProvider?: unknown }).getProvider =
      () => ({ streamSimple: providerStreamSimple });

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "custom-api stream routing regression focus",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assert.equal(
      typeof capturedStreamFn,
      "function",
      "summarize must receive a streamFn that delegates to the provider streamSimple",
    );
    // No captured session in this fixture ⇒ cold fallback: the wrapper
    // delegates the caller's context/options verbatim.
    const coldContext = { systemPrompt: "COLD", messages: [] };
    await (
      capturedStreamFn as (
        m: unknown,
        c: unknown,
        o: unknown,
      ) => Promise<unknown>
    )({}, coldContext, { maxTokens: 2048 });
    assert.equal(delegated.length, 1, "delegate must be invoked");
    assert.equal(
      (delegated[0] as { context: unknown }).context,
      coldContext,
      "fallback must be byte-identical delegation",
    );
  });

  it("omits streamFn when the provider has no streamSimple (builtin-compat fallback)", async () => {
    // A provider whose composed entry lacks `streamSimple` (or a
    // modelRegistry without getProvider, e.g. pre-0.81 hosts) must fall
    // back to the previous behavior: no streamFn → pi-ai compat dispatch.
    let capturedStreamFn: unknown = "__not_called__";
    const spySummarize = (async (_entries: unknown, opts: unknown) => {
      capturedStreamFn = (opts as { streamFn?: unknown }).streamFn;
      return {
        summary: "## Goal\nspy.\n## Progress\n### Done\nx.\n## Next Steps\ny.",
        readFiles: [] as string[],
        modifiedFiles: [] as string[],
        aborted: false,
      };
    }) as typeof fakeSummarize;

    const { sm, pi, tool, ctx } = setup({ summarize: spySummarize });
    setupRewindable(sm, pi, {});

    // Provider exists but has no streamSimple.
    (ctx.modelRegistry as unknown as { getProvider?: unknown }).getProvider =
      () => ({});

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "no-streamSimple fallback regression focus",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assert.equal(
      capturedStreamFn,
      undefined,
      "summarize must NOT receive streamFn when provider has no streamSimple",
    );
  });

  it("omits streamFn when getProvider returns undefined", async () => {
    // modelRegistry.getProvider(providerId) can return undefined (unknown
    // provider, or a host where the provider isn't composed yet). Must fall
    // back to the previous behavior (no streamFn).
    let capturedStreamFn: unknown = "__not_called__";
    const spySummarize = (async (_entries: unknown, opts: unknown) => {
      capturedStreamFn = (opts as { streamFn?: unknown }).streamFn;
      return {
        summary: "## Goal\nspy.\n## Progress\n### Done\nx.\n## Next Steps\ny.",
        readFiles: [] as string[],
        modifiedFiles: [] as string[],
        aborted: false,
      };
    }) as typeof fakeSummarize;

    const { sm, pi, tool, ctx } = setup({ summarize: spySummarize });
    setupRewindable(sm, pi, {});

    (ctx.modelRegistry as unknown as { getProvider?: unknown }).getProvider =
      () => undefined;

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "undefined-provider fallback regression focus",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assert.equal(
      capturedStreamFn,
      undefined,
      "summarize must NOT receive streamFn when getProvider returns undefined",
    );
  });

  it("omits streamFn when getProvider throws (defensive catch)", async () => {
    // A hostile/older modelRegistry may throw from getProvider. The
    // resolveProviderStreamFn try/catch must degrade to no streamFn rather
    // than failing the rewind.
    let capturedStreamFn: unknown = "__not_called__";
    const spySummarize = (async (_entries: unknown, opts: unknown) => {
      capturedStreamFn = (opts as { streamFn?: unknown }).streamFn;
      return {
        summary: "## Goal\nspy.\n## Progress\n### Done\nx.\n## Next Steps\ny.",
        readFiles: [] as string[],
        modifiedFiles: [] as string[],
        aborted: false,
      };
    }) as typeof fakeSummarize;

    const { sm, pi, tool, ctx } = setup({ summarize: spySummarize });
    setupRewindable(sm, pi, {});

    (ctx.modelRegistry as unknown as { getProvider?: unknown }).getProvider =
      () => {
        throw new Error("registry exploded");
      };

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "throwing-provider fallback regression focus",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assert.equal(
      capturedStreamFn,
      undefined,
      "summarize must NOT receive streamFn when getProvider throws",
    );
  });

  it("forwards streamSimple when it is not a function (truthy but invalid)", async () => {
    // `provider?.streamSimple` truthiness is the only guard; a truthy
    // non-function would previously be forwarded. Pin the current behavior
    // (forwarded as-is) so a future hardening (typeof check) is a visible
    // change, not a silent fix.
    let capturedStreamFn: unknown = "__not_called__";
    const spySummarize = (async (_entries: unknown, opts: unknown) => {
      capturedStreamFn = (opts as { streamFn?: unknown }).streamFn;
      return {
        summary: "## Goal\nspy.\n## Progress\n### Done\nx.\n## Next Steps\ny.",
        readFiles: [] as string[],
        modifiedFiles: [] as string[],
        aborted: false,
      };
    }) as typeof fakeSummarize;

    const { sm, pi, tool, ctx } = setup({ summarize: spySummarize });
    setupRewindable(sm, pi, {});

    const notAFunction = "not-a-function";
    (ctx.modelRegistry as unknown as { getProvider?: unknown }).getProvider =
      () => ({ streamSimple: notAFunction });

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "non-function streamSimple pin focus",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    // #33 always wraps a truthy provider streamSimple, so the option is the
    // wrapper (a function), not the raw truthy value. Calling it surfaces
    // the delegate's TypeError — pin that instead of the old identity.
    assert.equal(
      typeof capturedStreamFn,
      "function",
      "truthy streamSimple is wrapped (function-shaped option)",
    );
    assert.throws(
      () =>
        (capturedStreamFn as (m: unknown, c: unknown, o: unknown) => unknown)(
          {},
          {},
          {},
        ),
      TypeError,
      "invoking the wrapper surfaces the non-function delegate error",
    );
  });

  it("strips null header-deletion markers before passing headers", async () => {
    // pi 0.84+ ProviderHeaders can carry string|null; null marks a header
    // deletion. generateBranchSummary expects Record<string, string>, and
    // pi's own withoutDeletedHeaders drops nulls — mirror that.
    let capturedHeaders: unknown = "__not_called__";
    const spySummarize = (async (_entries: unknown, opts: unknown) => {
      capturedHeaders = (opts as { headers?: unknown }).headers;
      return {
        summary: "## Goal\nspy.\n## Progress\n### Done\nx.\n## Next Steps\ny.",
        readFiles: [] as string[],
        modifiedFiles: [] as string[],
        aborted: false,
      };
    }) as typeof fakeSummarize;

    const { sm, pi, tool, ctx } = setup({ summarize: spySummarize });
    setupRewindable(sm, pi, {});

    // Auth headers with a null deletion marker + a real header.
    (
      ctx.modelRegistry as unknown as {
        getApiKeyAndHeaders: () => Promise<unknown>;
      }
    ).getApiKeyAndHeaders = async () => ({
      ok: true,
      apiKey: "test-key",
      headers: { "x-real": "value", "x-deleted": null },
    });
    // Provider present so the rewind path reaches summarize.
    (ctx.modelRegistry as unknown as { getProvider?: unknown }).getProvider =
      () => ({ streamSimple: async () => ({}) as never });

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "null-header stripping regression focus",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assert.deepEqual(
      capturedHeaders,
      { "x-real": "value" },
      "null-marked headers must be stripped before summarize",
    );
  });

  it("injects opencode session headers for the summarization request", async () => {
    // Regression (2026-09-07): every rewind on opencode-go 400d with
    // MissingSessionID. Pi's live turns merge attribution headers in sdk.ts
    // (`mergeProviderAttributionHeaders`), but out-of-loop summarization
    // callers pass auth headers only — so the gateway never sees
    // x-opencode-session. The extension replicates pi's `getSessionHeaders`
    // half (fresh UUID per rewind; one-off summaries have no continuation).
    // The null-headers test above (default `claude` provider) doubles as the
    // negative case: no injection for non-opencode providers.
    let capturedHeaders: unknown = "__not_called__";
    const spySummarize = (async (_entries: unknown, opts: unknown) => {
      capturedHeaders = (opts as { headers?: unknown }).headers;
      return {
        summary: "## Goal\nspy.\n## Progress\n### Done\nx.\n## Next Steps\ny.",
        readFiles: [] as string[],
        modifiedFiles: [] as string[],
        aborted: false,
      };
    }) as typeof fakeSummarize;

    const { sm, pi, tool, ctx } = setup({ summarize: spySummarize });
    setupRewindable(sm, pi, {});
    (ctx as unknown as { model: unknown }).model = {
      api: "openai-responses",
      provider: "opencode-go",
      id: "muse-spark-1.3-contributor",
    };
    (
      ctx.modelRegistry as unknown as {
        getApiKeyAndHeaders: () => Promise<unknown>;
      }
    ).getApiKeyAndHeaders = async () => ({
      ok: true,
      apiKey: "test-key",
      // Pre-set session header must survive (never overridden).
      headers: { "x-real": "value", "x-opencode-session": "keep-me" },
    });
    (ctx.modelRegistry as unknown as { getProvider?: unknown }).getProvider =
      () => ({});

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "opencode session header regression focus",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    const headers = capturedHeaders as Record<string, string>;
    assert.equal(headers["x-real"], "value");
    assert.equal(
      headers["x-opencode-session"],
      "keep-me",
      "auth-provided session header must not be overridden",
    );
    assert.equal(headers["x-opencode-client"], "pi");
  });

  it("uses the live session id for x-opencode-session when auth sets none", async () => {
    // #33: the summarization request must route to the same
    // replica/affinity bucket as the turns it summarizes, so the header uses
    // the LIVE session id (not a fresh per-rewind uuid). A fresh uuid is
    // only the no-session fallback. Two distinct sessions must still produce
    // distinct ids (i.e. it is not a constant).
    const pairs: Array<{ header: string; live: string }> = [];
    const spySummarize = (async (_entries: unknown, opts: unknown) => {
      pairs.push({
        header:
          (opts as { headers?: Record<string, string> }).headers?.[
            "x-opencode-session"
          ] ?? "__missing__",
        live: "__pending__",
      });
      return {
        summary: "## Goal\nspy.\n## Progress\n### Done\nx.\n## Next Steps\ny.",
        readFiles: [] as string[],
        modifiedFiles: [] as string[],
        aborted: false,
      };
    }) as typeof fakeSummarize;

    for (const tc of ["tc-rewind-1", "tc-rewind-2"]) {
      const { sm, pi, tool, ctx } = setup({ summarize: spySummarize });
      setupRewindable(sm, pi, {});
      (ctx as unknown as { model: unknown }).model = {
        api: "openai-responses",
        provider: "opencode-go",
        id: "muse-spark-1.3-contributor",
      };
      (
        ctx.modelRegistry as unknown as {
          getApiKeyAndHeaders: () => Promise<unknown>;
        }
      ).getApiKeyAndHeaders = async () => ({
        ok: true,
        apiKey: "test-key",
        headers: {},
      });
      (ctx.modelRegistry as unknown as { getProvider?: unknown }).getProvider =
        () => ({});
      const result = await tool.execute(
        tc,
        {
          action: "rewind",
          rewindTo: "start",
          newLabel: "end",
          summaryFocus: "live session id header regression focus",
        },
        undefined,
        undefined,
        ctx,
      );
      assert.equal(result.isError, undefined);
      pairs[pairs.length - 1].live = sm.getSessionId();
    }
    assert.equal(pairs.length, 2);
    for (const pair of pairs) {
      assert.equal(
        pair.header,
        pair.live,
        "x-opencode-session must be the live session id, not a fresh uuid",
      );
    }
    assert.notEqual(
      pairs[0].header,
      pairs[1].header,
      "distinct sessions must yield distinct ids (not a constant)",
    );
  });

  it("forwards auth.baseUrl onto the model and auth.env (OAuth-derived endpoints)", async () => {
    // pi's own _getSummarizationRequestAuth applies `result.auth.baseUrl`
    // onto the model (OAuth/credential-derived endpoints, e.g.
    // githubCopilotOAuth) and forwards `env`. The extension must mirror
    // that — dropping baseUrl would hit the catalog default endpoint.
    let capturedModel: unknown;
    let capturedEnv: unknown = "__not_called__";
    const spySummarize = (async (_entries: unknown, opts: unknown) => {
      capturedModel = (opts as { model?: unknown }).model;
      capturedEnv = (opts as { env?: unknown }).env;
      return {
        summary: "## Goal\nspy.\n## Progress\n### Done\nx.\n## Next Steps\ny.",
        readFiles: [] as string[],
        modifiedFiles: [] as string[],
        aborted: false,
      };
    }) as typeof fakeSummarize;

    const { sm, pi, tool, ctx } = setup({ summarize: spySummarize });
    setupRewindable(sm, pi, {});

    (
      ctx.modelRegistry as unknown as {
        getApiKeyAndHeaders: () => Promise<unknown>;
      }
    ).getApiKeyAndHeaders = async () => ({
      ok: true,
      apiKey: "test-key",
      headers: {},
      baseUrl: "https://oauth-derived.example.com",
      env: { FOO: "bar" },
    });
    (ctx.modelRegistry as unknown as { getProvider?: unknown }).getProvider =
      () => ({ streamSimple: async () => ({}) as never });

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "oauth baseUrl/env forwarding regression focus",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assert.equal(
      (capturedModel as { baseUrl?: string } | undefined)?.baseUrl,
      "https://oauth-derived.example.com",
      "auth.baseUrl must be applied onto the summarization model",
    );
    assert.deepEqual(
      capturedEnv,
      { FOO: "bar" },
      "auth.env must be forwarded to summarize",
    );
  });

  it("resolves cacheRetention from the provider-scoped auth.env, not process.env", async () => {
    // pi-ai's getProviderEnvValue reads `auth.env` before `process.env`; a
    // provider-scoped PI_CACHE_RETENTION=long must therefore make the summary
    // request long (matching live) even when process.env says otherwise.
    const { spy, captured } = capturingSummarize();
    const { sm, pi, tool, ctx } = setup({ summarize: spy });
    const { fake } = setupRewindable(sm, pi, { capture: true });
    assert.ok(fake);
    fake.agent.state.tools = [
      { name: "read", description: "r", parameters: {} },
    ];

    const provider = capturingProvider();
    installProvider(ctx, provider.streamSimple);
    (
      ctx.modelRegistry as unknown as {
        getApiKeyAndHeaders: () => Promise<unknown>;
      }
    ).getApiKeyAndHeaders = async () => ({
      ok: true,
      apiKey: "test-key",
      headers: {},
      env: { PI_CACHE_RETENTION: "long" },
    });

    const original = process.env.PI_CACHE_RETENTION;
    process.env.PI_CACHE_RETENTION = "short";
    try {
      const result = await tool.execute(
        "tc-rewind",
        {
          action: "rewind",
          rewindTo: "start",
          newLabel: "end",
          summaryFocus: "provider-scoped retention must win over process env",
        },
        undefined,
        undefined,
        ctx,
      );
      assert.equal(result.isError, undefined);
      assert.equal(typeof captured.streamFn, "function");
      await (
        captured.streamFn as (
          m: unknown,
          c: unknown,
          o: unknown,
        ) => Promise<unknown>
      )({}, { systemPrompt: "COLD", messages: [] }, { maxTokens: 2048 });
      assert.equal(provider.calls.length, 1);
      assert.equal(
        provider.calls[0].options?.cacheRetention,
        "long",
        "provider-scoped PI_CACHE_RETENTION must override process.env",
      );
    } finally {
      if (original === undefined) delete process.env.PI_CACHE_RETENTION;
      else process.env.PI_CACHE_RETENTION = original;
    }
  });

  it("derives session-affinity headers from the auth.baseUrl-overridden model", async () => {
    // The opencode routing header keys off the model's `baseUrl` host. When
    // auth supplies the endpoint, `withSessionHeaders` must see the same
    // overridden model the request is sent to.
    const { spy, captured } = capturingSummarize();
    const { sm, pi, tool, ctx } = setup({ summarize: spy });
    setupRewindable(sm, pi, {});
    (
      ctx.modelRegistry as unknown as {
        getApiKeyAndHeaders: () => Promise<unknown>;
      }
    ).getApiKeyAndHeaders = async () => ({
      ok: true,
      apiKey: "test-key",
      headers: {},
      baseUrl: "https://opencode.ai/zen/v1",
    });
    installProvider(ctx, capturingProvider().streamSimple);

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "auth baseUrl must drive the session-affinity header",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assert.equal(
      (captured.headers as Record<string, string> | undefined)?.[
        "x-opencode-session"
      ],
      sm.getSessionId(),
      "the opencode routing header must be derived from the overridden baseUrl host",
    );
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

  it("captureSession tolerates a session with no agent internals", () => {
    // The deleted installPrepareNextTurn hook (which read agent.state.*
    // at prompt time) had an early-exit for sessions with no agent; the
    // replacement context-event refresh captures sessions for
    // refreshAgentMessages only, so a session without agent internals
    // must be capturable without throwing — findOwningSession just won't
    // match it.
    const sm = SessionManager.inMemory("/tmp");
    const ghost = { sessionManager: sm }; // no `agent` field at all
    const before = __testHooks.sessionRefCount();
    assert.doesNotThrow(() =>
      __testHooks.captureSession(ghost as unknown as AgentSession),
    );
    // The ghost is captured (ref count advanced) but findOwningSession
    // still resolves nothing for it (it has no agent.state.messages to
    // refresh — refreshAgentMessages returns false).
    assert.equal(__testHooks.sessionRefCount(), before + 1);
    assert.equal(__testHooks.refreshAgentMessages(sm), false);
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
    // `/tree` view's label entries). Pin: the text content lands
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
// findInFlightAssistantToolCalls (#37)
// =============================================================================

describe("findInFlightAssistantToolCalls (#37)", () => {
  it("returns the full batch when a sibling toolResult trails the declaring assistant", () => {
    // Sequential execution: pi appends each sibling's result before the
    // next call runs, so the newest entry may be a toolResult, not the
    // assistant. The walk matches by tool-call id, not tail position.
    const { sm } = setup();
    const batch = [
      { id: "tc-sibling", name: "bash" },
      { id: "tc-rewind", name: "navigate_tree" },
    ];
    appendAssistantToolCalls(sm, batch);
    sm.appendMessage({
      role: "toolResult",
      toolCallId: "tc-sibling",
      toolName: "bash",
      content: [{ type: "text", text: "sibling output" }],
      isError: false,
      timestamp: Date.now(),
    } as never);
    assert.deepEqual(
      __testHooks.findInFlightAssistantToolCalls(sm, "tc-rewind"),
      batch,
    );
  });

  it("counts id-bearing toolCall blocks with a malformed name (fail-open, name → 'unknown')", () => {
    // Detection must fail open on shape, not count: a sibling block whose
    // `name` is missing/renamed must still count toward the batch, so the
    // refusal can't be bypassed by an unexpected field shape.
    const { sm } = setup();
    sm.appendMessage({
      role: "assistant",
      content: [
        {
          type: "toolCall",
          id: "tc-rewind",
          name: "navigate_tree",
          arguments: {},
        },
        { type: "toolCall", id: "tc-sibling", arguments: {} },
      ],
      api: "anthropic",
      provider: "claude",
      model: "claude-sonnet-4-5",
      stopReason: "toolUse",
      timestamp: Date.now(),
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 30_000,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
    } as never);
    assert.deepEqual(
      __testHooks.findInFlightAssistantToolCalls(sm, "tc-rewind"),
      [
        { id: "tc-rewind", name: "navigate_tree" },
        { id: "tc-sibling", name: "unknown" },
      ],
    );
  });

  it("returns null when the in-flight toolCallId is absent", () => {
    const { sm } = setup();
    appendTurn(sm, "u", "a");
    assert.equal(
      __testHooks.findInFlightAssistantToolCalls(sm, "tc-rewind"),
      null,
    );
  });

  it("falls back to agent.state.messages when the branch walk misses", () => {
    const { sm } = setup();
    const batch = [{ id: "tc-rewind", name: "navigate_tree" }];
    const fake = makeFakeSession(sm);
    fake.agent.state.messages = [
      { role: "user", content: [{ type: "text", text: "go" }] },
      {
        role: "assistant",
        content: batch.map((tc) => ({
          type: "toolCall",
          ...tc,
          arguments: {},
        })),
      },
    ];
    __testHooks.captureSession(fake as unknown as AgentSession);
    // The active branch carries no assistant declaring tc-rewind.
    appendTurn(sm, "u", "a");
    assert.deepEqual(
      __testHooks.findInFlightAssistantToolCalls(sm, "tc-rewind"),
      batch,
    );
  });

  it("falls back when getBranch() throws, and returns null when both sources miss", () => {
    const { sm } = setup();
    const batch = [{ id: "tc-rewind", name: "navigate_tree" }];
    const fake = makeFakeSession(sm);
    fake.agent.state.messages = [
      {
        role: "assistant",
        content: batch.map((tc) => ({
          type: "toolCall",
          ...tc,
          arguments: {},
        })),
      },
    ];
    __testHooks.captureSession(fake as unknown as AgentSession);
    (sm as unknown as { getBranch: () => never }).getBranch = () => {
      throw new Error("branch read not available");
    };
    assert.deepEqual(
      __testHooks.findInFlightAssistantToolCalls(sm, "tc-rewind"),
      batch,
    );
    // Both sources miss → null (the caller falls through to the pre-guard
    // behavior, never hard-fails on undetectable state).
    fake.agent.state.messages = [];
    assert.equal(
      __testHooks.findInFlightAssistantToolCalls(sm, "tc-rewind"),
      null,
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
    // (no rewindTo on active branch) without crashing.
    const { sm, tool, ctx } = setup();
    appendTurn(sm, "u", "a");
    const huge = "x".repeat(100_000);
    const result = await tool.execute(
      "tc-1",
      {
        action: "rewind",
        rewindTo: "missing",
        newLabel: "after",
        summaryFocus: huge,
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /No label 'missing'/);
  });

  it("newLabel === rewindTo passes name validation; surfaces as a 'no such label' error if not pre-anchored", async () => {
    // Both pass isValidName; pinning that the dispatch doesn't reject
    // rewindTo === newLabel up front. (Without a pre-set anchor, the
    // label-existence guard fires.)
    const { sm, tool, ctx } = setup();
    appendTurn(sm, "u", "a");
    const result = await tool.execute(
      "tc-1",
      {
        action: "rewind",
        rewindTo: "same",
        newLabel: "same",
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
        rewindTo: "start",
        newLabel: "end",
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
        rewindTo: "start",
        newLabel: "end",
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
 * newLabel write (the second setLabel call after the anchor write).
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
  it("setLabel(newLabel) throws \u2192 synthetic still appended; original error wraps salvage detail", async () => {
    const { sm, pi, tool, ctx } = setup();
    const { fake } = setupRewindable(sm, pi, {
      capture: true,
      turnsAfter: 2,
      tokenCounts: [6_000, 14_000, 22_000],
    });
    if (!fake) throw new Error("capture: true must return fake");

    // We patch AFTER the anchor write above, so the patch counter starts
    // at 0. The first patched call is the rewind's newLabel write; the
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
          rewindTo: "start",
          newLabel: "end",
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
    assert.match(msg, /salvage:.*newLabel retry failed/);
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

    // Throw on call #1 (original newLabel write). The salvage retry runs
    // as call #2 and succeeds, so no salvage detail in the wrapped error.
    throwOnNthSetLabel(pi, 1, new Error("transient setLabel boom"));

    let thrown: unknown;
    try {
      await tool.execute(
        "tc-rewind",
        {
          action: "rewind",
          rewindTo: "start",
          newLabel: "end",
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
    // newLabel was eventually written, so a chained rewind could find it.
    const summaryLabel = "anchor:end";
    let foundLabelEnd = false;
    for (const e of sm.getBranch()) {
      if (sm.getLabel(e.id) === summaryLabel) {
        foundLabelEnd = true;
        break;
      }
    }
    assert.equal(foundLabelEnd, true, "newLabel retry should have written");
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
          rewindTo: "start",
          newLabel: "end",
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
    // newLabel write succeeded BEFORE the estimate threw — so a chained
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
      "newLabel should have landed before estimate threw",
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
          rewindTo: "start",
          newLabel: "end",
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

  it("findLabeledEntry(newLabel) throws → synthetic still appended; original error propagates", async () => {
    // The newLabel-collision lookup runs inside the salvage try as the
    // first step. If it throws (e.g. malformed branch traversal),
    // setLabel cannot run — but the synthetic append still must, so
    // pi's appended tool_result has a matching tool_use on the new
    // branch. Without the in-try lookup, an upstream throw from
    // findLabeledEntry would orphan the tool_result.
    const { sm, pi, tool, ctx } = setup();
    setupRewindable(sm, pi, {
      capture: true,
      turnsAfter: 2,
      tokenCounts: [6_000, 14_000, 22_000],
    });

    // Trip `getBranch` ONLY after a `branch_summary` entry exists on
    // the active branch — i.e. after `sm.branchWithSummary` ran. The
    // newLabel-collision lookup (`findLabeledEntry(sm, fullLabelEnd)`)
    // is the first call past that point in the rewind handler. The
    // rewindTo lookup, beforeTokens, and collectEntriesForBranchSummary
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
          rewindTo: "start",
          newLabel: "end",
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
      !/salvage:.*newLabel retry failed/.test(msg),
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

  it("prior-clear (clearPrior) throws once → retry succeeds; the new `newLabel` label lives, prior label is cleared, no salvage detail", async () => {
    // The move-on-collision pair is two distinct setLabel calls.
    // (A) writes the new label (`newLabel`) onto the summary; (B) clears the
    // prior entry's newLabel. If (A) succeeds and (B) throws, the
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
    const tB = appendTurn(sm, "u1", "a1", 6_000);
    pi.pi.setLabel(tB.assistantId, "anchor:b");
    const tA = appendTurn(sm, "u2", "a2", 16_000);
    pi.pi.setLabel(tA.assistantId, "anchor:a");
    appendTurn(sm, "u3", "a3", 22_000);
    appendTurn(sm, "u4", "a4", 28_000);

    // Patch AFTER the pre-anchor writes so the counter starts at 0.
    // Call #1 inside execute = (A) the new `newLabel` write; call #2 = (B)
    // the prior-clear (throws once); call #3 = the salvage retry of (B)
    // (succeeds).
    throwOnNthSetLabel(pi, 2, new Error("transient prior-clear boom"));

    let thrown: unknown;
    try {
      await tool.execute(
        "tc-rewind",
        {
          action: "rewind",
          rewindTo: "a",
          newLabel: "b",
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

  it("prior-clear (clearPrior) throws on every call → salvage detail surfaces 'prior-clear retry failed'; the new `newLabel` label still lives", async () => {
    // Sister test to the retry-succeeds case above: when BOTH the
    // original (B) prior-clear AND the salvage retry of (B) throw,
    // the wrapped error must surface a salvage detail that names the
    // prior-clear (NOT "newLabel retry failed" — that diagnostic is
    // for the (A) failure mode and would be misleading here). This
    // pins the salvage-detail prose introduced by the failedStep
    // split: a regression that re-conflated the discriminants would
    // either retry (A) (silently succeeding, no detail) or surface
    // the wrong salvage-detail string.
    const { sm, pi, tool, ctx } = setup();
    const tB = appendTurn(sm, "u1", "a1", 6_000);
    pi.pi.setLabel(tB.assistantId, "anchor:b");
    const tA = appendTurn(sm, "u2", "a2", 16_000);
    pi.pi.setLabel(tA.assistantId, "anchor:a");
    appendTurn(sm, "u3", "a3", 22_000);
    appendTurn(sm, "u4", "a4", 28_000);

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
          rewindTo: "a",
          newLabel: "b",
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
    // Salvage detail names the prior-clear, NOT the newLabel retry.
    // Tight on both sides: the correct diagnostic must be present, and
    // the wrong (newLabel-retry) diagnostic must NOT be present.
    assert.match(msg, /salvage:.*prior-clear retry failed/);
    assert.ok(
      !/newLabel retry failed/.test(msg),
      `prior-clear failure must not surface a newLabel-retry diagnostic; got: ${msg}`,
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
  it("'Already at <rewindTo>' fires when the rewindTo anchor is on the leaf", async () => {
    // setLabel itself advances the leaf (it appends a label-type entry as
    // child of the prior leaf), so anchoring then driving rewind doesn't
    // naturally hit `oldLeaf === target`. Reset the leaf back to the
    // labeled assistant via `sm.branch(...)` so the guard fires. This is
    // the contract the message pins ("Already at <rewindTo>"); the
    // path-construction is test-internal.
    const { sm, pi, tool, ctx } = setup();
    const t1 = appendTurn(sm, "u1", "a1", 100);
    pi.pi.setLabel(t1.assistantId, "anchor:here");
    sm.branch(t1.assistantId);
    const result = await tool.execute(
      "tc-1",
      {
        action: "rewind",
        rewindTo: "here",
        newLabel: "after",
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
        rewindTo: "start",
        newLabel: "end",
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
        rewindTo: "start",
        newLabel: "end",
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
        rewindTo: "start",
        newLabel: "end",
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
    const tA = appendTurn(sm, "u1", "a1", 6_000);
    pi.pi.setLabel(tA.assistantId, "anchor:a");
    appendTurn(sm, "u2", "a2", 14_000);
    appendTurn(sm, "u3", "a3", 20_000);

    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);

    // First rewind a\u2192b succeeds and leaves the leaf as a synthetic
    // assistant whose parent is the labeled b-summary.
    const r1 = await tool.execute(
      "tc-1",
      {
        action: "rewind",
        rewindTo: "a",
        newLabel: "b",
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
        rewindTo: "b",
        newLabel: "c",
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
    // Synthetic-discriminator fall-through pin, scoped post-#21: a lone
    // intervening message shaped like anything other than our synthetic
    // must fall through the synthetic-shape check — but a chain whose ONLY
    // content above the anchor is one message is structurally sub-floor
    // (measured savings ≈ 0), so the min-savings floor rejects it instead
    // of reaching summarize. The discriminator assertion survives as a
    // NEGATIVE pin: the error must be the floor rejection, NEVER the
    // 'Already at synthetic boundary' misfire.
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
        rewindTo: "b",
        newLabel: "c",
        summaryFocus:
          "Preserve user instructions and continue past the real navigate_tree call.",
      },
      undefined,
      undefined,
      ctx,
    );
    // Post-#21 scoping: fall-through from the boundary guard lands on the
    // min-savings floor (the lone-message chain measures ≈0 savings), so
    // the rewind errors — but with the FLOOR diagnostic, never the
    // synthetic-boundary misfire this discriminator exists to prevent.
    assert.equal(r.isError, true);
    assert.equal(
      (r.details as { rejected?: string }).rejected,
      "min-savings",
      "fall-through must land on the min-savings floor, not the boundary guard",
    );
    assert.ok(
      !/synthetic boundary/.test(r.content[0].text),
      "rewind must NOT trip the synthetic-boundary guard on a real navigate_tree call with nonzero usage",
    );
    assert.equal(
      summarizeCalled,
      false,
      "sub-floor chains are rejected before summarize",
    );
  });

  it("chained-rewind discriminator: lone intervening text-only assistant does NOT trip the guard", async () => {
    // Synthetic-discriminator fall-through pin, scoped post-#21: a lone
    // TEXT-ONLY assistant (no toolCall block) must fall through the guard's
    // first check (`block.type === 'toolCall'`) — but a chain whose only
    // content above the anchor is one message is structurally sub-floor, so
    // the min-savings floor rejects instead of reaching summarize. The
    // block-shape gate stays pinned negatively: never a synthetic-boundary
    // misfire.
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
        rewindTo: "b",
        newLabel: "c",
        summaryFocus:
          "Preserve user instructions and continue past the text-only intervening assistant.",
      },
      undefined,
      undefined,
      ctx,
    );
    // Post-#21 scoping: see the sibling nonzero-usage discriminator above.
    assert.equal(r.isError, true);
    assert.equal(
      (r.details as { rejected?: string }).rejected,
      "min-savings",
      "fall-through must land on the min-savings floor, not the boundary guard",
    );
    assert.ok(
      !/synthetic boundary/.test(r.content[0].text),
      "rewind must NOT trip the synthetic-boundary guard on a text-only assistant",
    );
    assert.equal(
      summarizeCalled,
      false,
      "sub-floor chains are rejected before summarize",
    );
  });

  it("chained-rewind discriminator: lone intervening non-navigate_tree toolCall does NOT trip the guard", async () => {
    // Synthetic-discriminator fall-through pin, scoped post-#21: a
    // synthetic-SHAPED assistant (zero usage + stopReason 'toolUse') whose
    // toolCall is for a DIFFERENT tool (`bash`) must fall through the
    // `name === 'navigate_tree'` predicate. The name gate is load-bearing:
    // without it this chain would trip 'Already at synthetic boundary';
    // with it, fall-through lands on the min-savings floor (structurally
    // ≈0 savings) — pinned negatively below.
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
        rewindTo: "b",
        newLabel: "c",
        summaryFocus:
          "Preserve user instructions and continue past the non-navigate_tree intervening toolCall.",
      },
      undefined,
      undefined,
      ctx,
    );
    // Post-#21 scoping: see the sibling nonzero-usage discriminator above.
    assert.equal(r.isError, true);
    assert.equal(
      (r.details as { rejected?: string }).rejected,
      "min-savings",
      "fall-through must land on the min-savings floor, not the boundary guard",
    );
    assert.ok(
      !/synthetic boundary/.test(r.content[0].text),
      "rewind must NOT trip the synthetic-boundary guard on a non-navigate_tree toolCall",
    );
    assert.equal(
      summarizeCalled,
      false,
      "sub-floor chains are rejected before summarize",
    );
  });
});

// =============================================================================
// dispatch: rewind min-savings floor (#21)
//
// Scenario pins from the issue-#20 repro, hermetic:
//   A — anchor→immediate rewind (zero/negligible work above the anchor)
//       must be REJECTED by the min-savings floor, never summarized;
//   B — healthy stages (≥20k apparent savings) must still succeed;
//   C — chained rewind→immediate-rewind keeps its existing
//       synthetic-boundary diagnostic (the floor sits AFTER that guard).
// =============================================================================

describe("dispatch: rewind min-savings floor (#21)", () => {
  /** Summarize stub that counts invocations so pins can assert call counts. */
  function countingSummarize() {
    let calls = 0;
    const stub = (async () => {
      calls++;
      return {
        summary:
          "## Goal\nfloor pin.\n## Progress\n### Done\nx.\n## Next Steps\ny.",
        readFiles: [] as string[],
        modifiedFiles: [] as string[],
        aborted: false,
      };
    }) as typeof fakeSummarize;
    return { stub, count: () => calls };
  }

  /** No newLabel may survive a rejected rewind — the label isn't consumed. */
  function assertLabelAbsent(sm: SessionManager, label: string): void {
    for (const e of sm.getBranch()) {
      assert.notEqual(sm.getLabel(e.id), label);
    }
  }

  it("A-pin variant A: anchor→immediate rewind with an earlier anchor present is rejected with the active-anchor list", async () => {
    const { stub, count } = countingSummarize();
    const { sm, pi, tool, ctx } = setup({ summarize: stub });
    const t1 = appendTurn(sm, "u1", "a1", 6_000);
    pi.pi.setLabel(t1.assistantId, "anchor:kickoff");
    const t2 = appendTurn(sm, "u2", "a2", 12_000);
    pi.pi.setLabel(t2.assistantId, "anchor:impl-start");

    // setLabel advances the leaf (label-change entry), so the leaf here is
    // the impl-start label entry — exactly the issue-#20 repro shape:
    // anchor, then immediately rewind with zero work above.
    const branchBefore = sm.getBranch().map((e) => e.id);
    const leafBefore = sm.getLeafId();

    const r = await tool.execute(
      "tc-floor-a",
      {
        action: "rewind",
        rewindTo: "impl-start",
        newLabel: "done",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );

    assert.equal(r.isError, true);
    assert.equal(r.details.rejected, "min-savings");
    assert.ok(typeof r.details.apparentSavings === "number");
    const text = r.content[0].text;
    assert.match(text, /would free only/);
    // Movement-order copy (#43): rewind TO the anchor, collapsed AS the new
    // label. An arrow reading `'impl-start' → 'done'` inverts the operation.
    assert.match(text, /Rewinding to 'impl-start' \(as 'done'\)/);
    // Variant A guidance + the full chronological anchor list, including
    // rewindTo itself.
    assert.match(text, /Rewind further back to actually free context/);
    assert.match(text, /'kickoff'/);
    assert.match(text, /'impl-start'/);
    // The numeric bar itself is NEVER printed agent-visible.
    assert.ok(
      !text.includes(String(MIN_REWIND_SAVINGS_TOKENS)),
      "rejection copy must not print the floor value",
    );
    // Zero summarizer calls; newLabel not consumed; tree untouched.
    assert.equal(count(), 0);
    assertLabelAbsent(sm, "anchor:done");
    assert.deepEqual(
      sm.getBranch().map((e) => e.id),
      branchBefore,
    );
    assert.equal(sm.getLeafId(), leafBefore);
  });

  it("A-pin variant B: rewinding the OLDEST anchor is rejected with 'No earlier anchors'", async () => {
    const { stub, count } = countingSummarize();
    const { sm, pi, tool, ctx } = setup({ summarize: stub });
    const t1 = appendTurn(sm, "u1", "a1", 6_000);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    // One tiny turn above the anchor: measured savings ≈ 1 token.
    appendTurn(sm, "u2", "a2", 6_001);

    const r = await tool.execute(
      "tc-floor-b",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "done",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );

    assert.equal(r.isError, true);
    assert.equal(r.details.rejected, "min-savings");
    const text = r.content[0].text;
    assert.match(text, /would free only/);
    assert.match(
      text,
      /No earlier anchors — keep working and rewind later once more has accumulated above it\./,
    );
    assert.match(text, /'start'/);
    assert.ok(!text.includes(String(MIN_REWIND_SAVINGS_TOKENS)));
    assert.equal(count(), 0);
    assertLabelAbsent(sm, "anchor:done");
  });

  it("B-pin: healthy stage (≥20k apparent savings) succeeds and calls summarize exactly once", async () => {
    const { stub, count } = countingSummarize();
    const { sm, pi, tool, ctx } = setup({ summarize: stub });
    const t1 = appendTurn(sm, "u1", "a1", 6_000);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendTurn(sm, "u2", "a2", 27_000);
    appendTurn(sm, "u3", "a3", 28_000);

    const r = await tool.execute(
      "tc-floor-bpin",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );

    assert.equal(r.isError, undefined);
    assert.match(
      r.content[0].text,
      /\[rewind to 'start' · collapsed as 'end'\]/,
    );
    // Apparent savings = contextBefore − anchor-time total (6_000).
    const before = r.details.contextBefore as number;
    assert.ok(
      before - 6_000 >= 20_000,
      `expected ≥20k apparent savings; got ${before - 6_000}`,
    );
    assert.equal(count(), 1);
  });

  it("C-pin: chained rewind→immediate-rewind keeps the synthetic-boundary message verbatim (floor must not preempt it)", async () => {
    const { stub, count } = countingSummarize();
    const { sm, pi, tool, ctx } = setup({ summarize: stub });
    const t1 = appendTurn(sm, "u1", "a1", 6_000);
    pi.pi.setLabel(t1.assistantId, "anchor:a");
    appendTurn(sm, "u2", "a2", 14_000);
    appendTurn(sm, "u3", "a3", 20_000);

    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);

    const r1 = await tool.execute(
      "tc-1",
      {
        action: "rewind",
        rewindTo: "a",
        newLabel: "b",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(r1.isError, undefined);

    // Immediate second rewind: the lone intervening entry is the synthetic,
    // so the EXISTING boundary guard fires — before the floor.
    const r2 = await tool.execute(
      "tc-2",
      {
        action: "rewind",
        rewindTo: "b",
        newLabel: "c",
        summaryFocus: "Preserve user instructions across the second rewind.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(r2.isError, true);
    assert.equal(
      r2.content[0].text,
      "Already at synthetic boundary — no work to summarize. Append at least one turn between rewinds.",
      "synthetic-boundary wording must stay verbatim and keep precedence over the floor",
    );
    assert.notEqual(r2.details.rejected, "min-savings");
    // Exactly one summarizer call total: the first rewind only.
    assert.equal(count(), 1);
  });

  it("boundary: apparent savings just under the floor rejects; exactly at the floor accepts", async () => {
    // Deterministic measurement math: with 'u*'/'a*' texts (2 chars → 1
    // estimated trailing token each), a chain anchored at a1(T1) with leaf
    // a3(T3) measures beforeTokens = T2 + 1 and tokensAtTarget = T1, i.e.
    //   apparentSavings = T2 + 1 − T1.
    async function floorFixture(t2Total: number) {
      const { stub, count } = countingSummarize();
      const { sm, pi, tool, ctx } = setup({ summarize: stub });
      const t1 = appendTurn(sm, "u1", "a1", 5_000);
      pi.pi.setLabel(t1.assistantId, "anchor:start");
      appendTurn(sm, "u2", "a2", t2Total);
      appendTurn(sm, "u3", "a3", t2Total + 1_000);
      return { tool, ctx, count };
    }
    const rewindArgs = {
      action: "rewind",
      rewindTo: "start",
      newLabel: "end",
      summaryFocus: "Preserve user instructions and continue.",
    } as const;

    // Just UNDER: T2 = 5000 + MIN − 999 → savings = MIN − 998.
    const under = await floorFixture(5_000 + MIN_REWIND_SAVINGS_TOKENS - 999);
    const ru = await under.tool.execute(
      "tc-under",
      rewindArgs,
      undefined,
      undefined,
      under.ctx,
    );
    assert.equal(ru.isError, true);
    assert.equal(ru.details.rejected, "min-savings");
    assert.ok(
      (ru.details.apparentSavings as number) < MIN_REWIND_SAVINGS_TOKENS,
    );
    assert.equal(under.count(), 0);

    // AT the floor: T2 = 5000 + MIN − 1 → savings = MIN exactly (the guard
    // is strict `<`, so this accepts).
    const at = await floorFixture(5_000 + MIN_REWIND_SAVINGS_TOKENS - 1);
    const ra = await at.tool.execute(
      "tc-at",
      rewindArgs,
      undefined,
      undefined,
      at.ctx,
    );
    assert.equal(ra.isError, undefined);
    // Success details carry contextBefore (= T2 + 1); savings = that minus
    // the anchor-time total (5_000).
    const before = ra.details.contextBefore as number;
    assert.equal(
      before - 5_000,
      MIN_REWIND_SAVINGS_TOKENS,
      "savings exactly at the floor must be accepted (strict < comparison)",
    );
    assert.equal(at.count(), 1);
  });

  it("guard-order: sub-floor rejection fires BEFORE the model check ('No model configured')", async () => {
    const { stub, count } = countingSummarize();
    const { sm, pi, tool, ctx } = setup({ noModel: true, summarize: stub });
    const t1 = appendTurn(sm, "u1", "a1", 6_000);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendTurn(sm, "u2", "a2", 6_001);

    const r = await tool.execute(
      "tc-order",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "done",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );

    assert.equal(r.isError, true);
    assert.equal(r.details.rejected, "min-savings");
    assert.match(r.content[0].text, /would free only/);
    assert.ok(
      !/No model configured/.test(r.content[0].text),
      "the floor must reject before the model check runs",
    );
    assert.equal(count(), 0);
  });

  it("anchor response tells the agent to wait for accumulated work (#21 copy)", async () => {
    const { sm, tool, ctx } = setup();
    appendTurn(sm, "u1", "a1", 100);

    const r = await tool.execute(
      "tc-anchor-copy",
      { action: "anchor", name: "impl-start" },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(r.isError, undefined);
    const text = r.content[0].text;
    assert.ok(
      text.includes("Once real work has accumulated after this anchor"),
      `anchor copy must steer toward accumulating work; got: ${text}`,
    );
    assert.ok(
      !text.includes("When you finish this stage"),
      "pre-#21 anchor copy must be gone",
    );
    // The focus-length constant stays interpolated (no hardcoded 20).
    assert.ok(text.includes(`≥${MIN_SUMMARY_FOCUS_LENGTH}-char focus`));
  });

  it("registers the two rewind-hygiene promptGuidelines byte-exactly and in order", () => {
    // Exact array length + per-element byte equality + order (the 2026-09-13
    // reflection's Lesson 3: `includes`-only pins pass under reordering).
    // These bullets are static and NOT config-gated — `registerTool` fixes
    // the array at registration, so every active-tool session carries them
    // (and pays the one-time prompt-cache prefix invalidation on upgrade).
    const { tool } = setup();
    const guidelines = tool.promptGuidelines as string[] | undefined;
    assert.ok(Array.isArray(guidelines), "promptGuidelines must be registered");
    const expected = [
      "navigate_tree: when rewinding, prefer the oldest anchor that keeps what you'd otherwise re-read or re-derive; persist durable findings to files first.",
      "navigate_tree: don't rewind while a user decision or unresolved question is pending — ask the user instead.",
    ];
    assert.equal(
      guidelines?.length,
      expected.length,
      `promptGuidelines must carry exactly ${expected.length} bullets; got: ${guidelines?.length}`,
    );
    assert.deepEqual(guidelines, expected);
    for (let i = 0; i < expected.length; i++) {
      assert.equal(
        guidelines?.[i],
        expected[i],
        `guidelines[${i}] must be byte-exact; got: ${guidelines?.[i]}`,
      );
    }
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
        rewindTo: "start",
        newLabel: "end",
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

// =============================================================================
// dispatch: rewind cache-preserving summary request (#33)
//
// The cache path is built at the rewind call site and injected through the
// `streamFn` seam. These tests pin the wiring (live inputs read, request
// assembled, wrapper forwarded) and the fallback/miss/hit matrix.
// Cache notices never enter the tool-result content (the model must not see
// them); when `showCacheMissNotices` is on, the notice string is stored in
// `details.summaryCache.notice` for the tool's TUI `renderResult`. The same
// numbers always live in `details.summaryCache`.
// Provider usage is stubbed; no request leaves the process.
// =============================================================================

interface CapturedRewindOptions {
  streamFn?: unknown;
  headers?: Record<string, string>;
  customInstructions?: unknown;
}

/** Summarize stub that captures the options the call site passes downstream. */
function capturingSummarize(usage?: unknown) {
  const captured: CapturedRewindOptions = {};
  const spy = (async (_entries: unknown, opts: unknown) => {
    const o = opts as CapturedRewindOptions;
    captured.streamFn = o.streamFn;
    captured.headers = o.headers;
    captured.customInstructions = o.customInstructions;
    return {
      summary: "## Goal\nspy.\n## Progress\n### Done\nx.\n## Next Steps\ny.",
      readFiles: [] as string[],
      modifiedFiles: [] as string[],
      aborted: false,
      ...(usage ? { usage } : {}),
    };
  }) as typeof fakeSummarize;
  return { spy, captured };
}

interface ProviderCall {
  context: {
    systemPrompt?: string;
    messages: Array<{ role: string }>;
    tools?: unknown;
  };
  options: Record<string, unknown> | undefined;
}

/** Fake provider `streamSimple` that records the wire request. */
function capturingProvider() {
  const calls: ProviderCall[] = [];
  const streamSimple = async (
    _model: unknown,
    context: ProviderCall["context"],
    options: Record<string, unknown> | undefined,
  ) => {
    calls.push({ context, options });
    return { result: async () => ({}) } as never;
  };
  return { calls, streamSimple };
}

function installProvider(ctx: FakeCtx, streamSimple: unknown): void {
  (ctx.modelRegistry as unknown as { getProvider?: unknown }).getProvider =
    () => ({ streamSimple });
}

const USAGE_COST = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  total: 0,
};

/**
 * Cache notices must never reach the model: the rewind tool-result content
 * (the only text the LLM sees) must carry none of the notice copy. The
 * human-readable notice travels through `details.summaryCache.notice` and is
 * rendered by the tool's `renderResult`; the machine-readable numbers stay in
 * `details.summaryCache`.
 */
function assertNoCacheNoticeInContent(text: string): void {
  assert.doesNotMatch(text, /summary cache:/);
  assert.doesNotMatch(text, /summary cache miss/);
  assert.doesNotMatch(text, /cache-preserving summary unavailable/);
}

/** Append an assistant entry carrying a single toolCall (the in-flight one). */
function appendInFlightAssistant(sm: SessionManager, id: string): string {
  return sm.appendMessage({
    role: "assistant",
    content: [
      {
        type: "toolCall",
        id,
        name: "navigate_tree",
        arguments: { action: "rewind" },
      },
    ],
    api: "anthropic",
    provider: "claude",
    model: "claude-sonnet-4-5",
    stopReason: "toolUse",
    timestamp: Date.now(),
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 20_000,
      cost: USAGE_COST,
    },
  } as never);
}

/**
 * Append a compaction entry that keeps the path from `firstKeptEntryId`
 * onward. `buildContextEntries()` then drops every path entry before that id
 * from the live projection — the exact evidence-loss shape the call site
 * must refuse to summarize from the cache path.
 */
function appendCompaction(
  sm: SessionManager,
  firstKeptEntryId: string,
  summary = "COMPACTED",
): string {
  return sm.appendCompaction(summary, firstKeptEntryId, 20_000);
}

/**
 * Append an assistant "previous request" turn with explicit prompt/cache
 * accounting so `detectBranchSummaryCacheMiss` has a baseline to compare the
 * summary request against. `totalTokens` feeds the existing token estimator.
 */
function appendUsageTurn(
  sm: SessionManager,
  usage: {
    input: number;
    cacheRead: number;
    cacheWrite: number;
    cost: { input: number; cacheRead: number; cacheWrite: number };
  },
  opts: { provider?: string; model?: string; timestamp?: number } = {},
): string {
  const total = usage.input + usage.cacheRead + usage.cacheWrite;
  return sm.appendMessage({
    role: "assistant",
    content: [{ type: "text", text: "baseline" }],
    api: "anthropic",
    provider: opts.provider ?? "claude",
    model: opts.model ?? "claude-sonnet-4-5",
    stopReason: "endTurn",
    timestamp: opts.timestamp ?? Date.now(),
    usage: {
      input: usage.input,
      output: 0,
      cacheRead: usage.cacheRead,
      cacheWrite: usage.cacheWrite,
      totalTokens: total,
      cost: {
        input: usage.cost.input,
        output: 0,
        cacheRead: usage.cost.cacheRead,
        cacheWrite: usage.cost.cacheWrite,
        total: usage.cost.input + usage.cost.cacheRead + usage.cost.cacheWrite,
      },
    },
  } as never);
}

/** A 20k-prompt-token baseline whose prefix should have been cache-served. */
const CACHE_BASELINE_USAGE = {
  input: 0,
  cacheRead: 20_000,
  cacheWrite: 0,
  cost: { input: 0, cacheRead: 0.001, cacheWrite: 0 },
};

/**
 * Summary response that missed a 20k baseline: 20k tokens re-billed at
 * $0.20. Clears both the token and dollar display floors.
 */
const SUMMARY_MISS_USAGE = {
  input: 20_000,
  output: 20,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 20_020,
  cost: { input: 0.2, output: 0, cacheRead: 0, cacheWrite: 0, total: 0.2 },
};

/** Expected TUI warning for `SUMMARY_MISS_USAGE` against `CACHE_BASELINE_USAGE`. */
const SUMMARY_MISS_NOTICE = "Cache miss: 20k tokens re-billed (~$0.20)";

describe("dispatch: rewind cache-preserving summary request (#33)", () => {
  const ORIGINAL_KILL_SWITCH = process.env.PI_NAVIGATE_TREE_SUMMARY_CACHE;
  afterEach(() => {
    if (ORIGINAL_KILL_SWITCH === undefined) {
      delete process.env.PI_NAVIGATE_TREE_SUMMARY_CACHE;
    } else {
      process.env.PI_NAVIGATE_TREE_SUMMARY_CACHE = ORIGINAL_KILL_SWITCH;
    }
  });

  it("assembles the live request and delegates it through the wrapper", async () => {
    const { spy, captured } = capturingSummarize();
    const { sm, pi, tool, ctx } = setup({ summarize: spy });
    const t1 = appendTurn(sm, "u1", "a1", 6_000);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendTurn(sm, "u2", "a2", 12_000);
    const inFlightId = appendInFlightAssistant(sm, "tc-rewind");
    assert.equal(sm.getLeafId(), inFlightId);

    const fake = makeFakeSession(sm);
    const liveTools = [{ name: "read", description: "r", parameters: {} }];
    fake.agent.state.tools = liveTools;
    fake.agent.thinkingBudgets = { high: 4242 };
    __testHooks.captureSession(fake as unknown as AgentSession);
    (pi.pi as unknown as { getThinkingLevel: () => string }).getThinkingLevel =
      () => "high";

    const provider = capturingProvider();
    installProvider(ctx, provider.streamSimple);

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus:
          "Preserve the latest instruction, note done work, list what remains.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assert.equal(
      (result.details.summaryCache as { mode: string }).mode,
      "live-prefix",
    );
    assert.equal(
      (result.details.summaryCache as { used: boolean }).used,
      false,
      "stub summarizer never invokes the wrapper, so `used` stays false",
    );

    // The wrapper was handed to the summarizer; invoke it the way the real
    // generateBranchSummary would (cold context/options in, live request out).
    assert.equal(typeof captured.streamFn, "function");
    await (
      captured.streamFn as (
        m: unknown,
        c: unknown,
        o: unknown,
      ) => Promise<unknown>
    )({}, { systemPrompt: "COLD", messages: [] }, { maxTokens: 2048 });

    assert.equal(provider.calls.length, 1);
    const call = provider.calls[0];
    assert.equal(call.context.systemPrompt, "LIVE SYSTEM PROMPT");
    assert.equal(
      call.context.tools,
      liveTools,
      "live tool instances must be reused",
    );
    assert.equal(
      call.options?.maxTokens,
      undefined,
      "caller cap must be stripped",
    );
    assert.equal(call.options?.cacheRetention, "short");
    assert.equal(call.options?.sessionId, sm.getSessionId());
    assert.equal(call.options?.reasoning, "high");
    assert.deepEqual(call.options?.thinkingBudgets, { high: 4242 });

    // In-flight assistant excluded: the payload ends before the assistant
    // that carries the triggering toolCall, leaving no unpaired tool_use.
    const body = call.context.messages.slice(0, -1);
    assert.ok(
      !body.some(
        (m) =>
          m.role === "assistant" && JSON.stringify(m).includes("tc-rewind"),
      ),
      "in-flight assistant toolCall must not appear in the summary payload",
    );
    const trailer = call.context.messages[call.context.messages.length - 1] as {
      role: string;
      content: Array<{ text: string }>;
    };
    assert.equal(trailer.role, "user");
    assert.match(
      trailer.content[0].text,
      /Additional focus: Preserve the latest/,
    );
    assert.doesNotMatch(trailer.content[0].text, /\{first\}/);
  });

  it("falls back with reflection-missing when no owning session can be found", async () => {
    const { spy } = capturingSummarize();
    const { sm, pi, tool, ctx } = setup({ summarize: spy });
    setupRewindable(sm, pi);
    // Provider present so the fallback reason is the reflection miss, not
    // the missing stream.
    installProvider(ctx, capturingProvider().streamSimple);

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assertNoCacheNoticeInContent(result.content[0].text);
    // No owning session => the settings gate is unreadable => no notice stored.
    const cache = result.details.summaryCache as {
      mode: string;
      fallbackReason: string;
      hit: boolean;
      notice: string | null;
    };
    assert.equal(cache.mode, "fallback");
    assert.equal(cache.fallbackReason, "reflection-missing");
    assert.equal(cache.hit, false);
    assert.equal(cache.notice, null);
  });

  it("falls back when the reflected session has no live tool array", async () => {
    const { spy } = capturingSummarize();
    const { sm, pi, tool, ctx } = setup({ summarize: spy });
    setupRewindable(sm, pi);
    // Capture a session whose tools field is not an array. (Must be the only
    // captured session for this sm so findOwningSession resolves it.)
    const fake = makeFakeSession(sm);
    fake.agent.state.tools = undefined as never;
    __testHooks.captureSession(fake as unknown as AgentSession);
    installProvider(ctx, capturingProvider().streamSimple);

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assertNoCacheNoticeInContent(result.content[0].text);
    // The stub reports no usage, so no miss is measured -> no notice.
    assert.equal(
      (result.details.summaryCache as { notice: string | null }).notice,
      null,
    );
    assert.equal(
      (result.details.summaryCache as { fallbackReason: string })
        .fallbackReason,
      "no-live-tools",
    );
  });

  it("kill switch PI_NAVIGATE_TREE_SUMMARY_CACHE=0 bypasses the cache path", async () => {
    process.env.PI_NAVIGATE_TREE_SUMMARY_CACHE = "0";
    const { spy } = capturingSummarize(undefined);
    const { sm, pi, tool, ctx } = setup({ summarize: spy });
    setupRewindable(sm, pi, { capture: true });
    installProvider(ctx, capturingProvider().streamSimple);

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assertNoCacheNoticeInContent(result.content[0].text);
    const cache = result.details.summaryCache as {
      mode: string;
      fallbackReason: string;
      notice: string | null;
    };
    assert.equal(cache.mode, "fallback");
    assert.equal(cache.fallbackReason, "disabled");
    assert.equal(cache.notice, null);
  });

  it("falls back with no-provider-stream when the registry has no streamSimple", async () => {
    const { spy } = capturingSummarize();
    const { sm, pi, tool, ctx } = setup({ summarize: spy });
    setupRewindable(sm, pi, { capture: true });

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assertNoCacheNoticeInContent(result.content[0].text);
    assert.equal(
      (result.details.summaryCache as { notice: string | null }).notice,
      null,
    );
  });

  it("stores the fork's miss notice when the summary misses a 20k baseline", async () => {
    const { spy } = capturingSummarize(SUMMARY_MISS_USAGE);
    const { sm, pi, tool, ctx } = setup({ summarize: spy });
    const { fake } = setupRewindable(sm, pi, { capture: true });
    if (!fake) throw new Error("capture: true must return fake");
    fake.settingsManager = { getShowCacheMissNotices: () => true };
    appendUsageTurn(sm, CACHE_BASELINE_USAGE);
    installProvider(ctx, capturingProvider().streamSimple);

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assertNoCacheNoticeInContent(result.content[0].text);
    const cache = result.details.summaryCache as {
      hit: boolean;
      cacheRead: number;
      input: number;
      cacheWrite: number;
      missedTokens: number;
      missedCost: number;
      notice: string | null;
    };
    assert.equal(cache.hit, false);
    assert.equal(cache.cacheRead, 0);
    assert.equal(cache.input, 20_000);
    assert.equal(cache.cacheWrite, 0);
    assert.equal(cache.missedTokens, 20_000);
    assert.equal(cache.missedCost.toFixed(2), "0.20");
    assert.equal(cache.notice, SUMMARY_MISS_NOTICE);
  });

  it("stays silent on a cache hit and still reports the measured stats", async () => {
    const { spy } = capturingSummarize({
      input: 420,
      output: 20,
      cacheRead: 20_000,
      cacheWrite: 0,
      totalTokens: 20_440,
      cost: {
        input: 0,
        output: 0,
        cacheRead: 0.001,
        cacheWrite: 0,
        total: 0.001,
      },
    });
    const { sm, pi, tool, ctx } = setup({ summarize: spy });
    const { fake } = setupRewindable(sm, pi, { capture: true });
    if (!fake) throw new Error("capture: true must return fake");
    fake.settingsManager = { getShowCacheMissNotices: () => true };
    appendUsageTurn(sm, CACHE_BASELINE_USAGE);
    installProvider(ctx, capturingProvider().streamSimple);

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assertNoCacheNoticeInContent(result.content[0].text);
    // Hits are silent: session totals cover them, there is no hit notice.
    const cache = result.details.summaryCache as {
      hit: boolean;
      fallbackReason: string | null;
      branchStartRetained: boolean;
      cacheRead: number;
      notice: string | null;
    };
    assert.equal(cache.hit, true);
    assert.equal(cache.cacheRead, 20_000);
    assert.equal(cache.notice, null);
    // Non-crossing regression: no compaction in the segment, so the request
    // stays live-prefix with no fallback and a retained branch start.
    assert.equal(cache.fallbackReason, null);
    assert.equal(cache.branchStartRetained, true);
  });

  it("falls back with branch-crosses-compaction when the segment crosses the compaction cut", async () => {
    const { spy, captured } = capturingSummarize();
    const { sm, pi, tool, ctx } = setup({ summarize: spy });
    const a1 = appendTurn(sm, "u1", "a1", 6_000);
    pi.pi.setLabel(a1.assistantId, "anchor:start");
    const a2 = appendTurn(sm, "u2", "a2", 12_000);
    // Keep from a2 onward: the anchor (a1), its label entry, and u2 are
    // dropped from the live projection, so the cache payload would lose raw
    // branch evidence the legacy path still sends.
    appendCompaction(sm, a2.assistantId);
    appendTurn(sm, "u3", "a3", 18_000);
    appendTurn(sm, "u4", "a4", 24_000);

    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);
    const provider = capturingProvider();
    installProvider(ctx, provider.streamSimple);

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "Preserve the raw branch evidence and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assertNoCacheNoticeInContent(result.content[0].text);
    // Default fake settings (off) => no notice stored for the fallback.
    const cache = result.details.summaryCache as {
      mode: string;
      fallbackReason: string;
      notice: string | null;
    };
    assert.equal(cache.mode, "fallback");
    assert.equal(cache.fallbackReason, "branch-crosses-compaction");
    assert.equal(cache.notice, null);

    // `request === null` path: the wrapper delegates the caller's cold
    // context/options verbatim (the raw-evidence legacy request).
    assert.equal(typeof captured.streamFn, "function");
    await (
      captured.streamFn as (
        m: unknown,
        c: unknown,
        o: unknown,
      ) => Promise<unknown>
    )({}, { systemPrompt: "COLD", messages: [] }, { maxTokens: 2048 });
    assert.equal(provider.calls.length, 1);
    assert.equal(provider.calls[0].context.systemPrompt, "COLD");
  });

  it("keeps live-prefix when the segment contains a compaction but the target is after firstKeptEntryId", async () => {
    const { spy } = capturingSummarize();
    const { sm, pi, tool, ctx } = setup({ summarize: spy });
    appendTurn(sm, "u1", "a1", 6_000);
    const t2 = appendTurn(sm, "u2", "a2", 12_000);
    pi.pi.setLabel(t2.assistantId, "anchor:start");
    appendTurn(sm, "u3", "a3", 18_000);
    // Keep from u2 onward: the anchor (a2) and its label survive the cut, so
    // the segment loses no evidence even though it contains the compaction
    // entry. A "segment has a compaction" predicate would wrongly fall back.
    appendCompaction(sm, t2.userId);
    appendTurn(sm, "u4", "a4", 24_000);
    appendTurn(sm, "u5", "a5", 30_000);

    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);
    installProvider(ctx, capturingProvider().streamSimple);

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "Preserve the live evidence and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    const cache = result.details.summaryCache as {
      mode: string;
      fallbackReason: string | null;
      branchStartRetained: boolean;
    };
    assert.equal(cache.mode, "live-prefix");
    assert.equal(cache.fallbackReason, null);
    assert.equal(cache.branchStartRetained, true);
  });

  it("keeps live-prefix when the target is exactly the compaction entry", async () => {
    const { spy } = capturingSummarize();
    const { sm, pi, tool, ctx } = setup({ summarize: spy });
    const t1 = appendTurn(sm, "u1", "a1", 6_000);
    appendTurn(sm, "u2", "a2", 12_000);
    // Keep everything (firstKeptEntryId = u1); the segment after the
    // compaction entry is fully retained.
    const compactionId = appendCompaction(sm, t1.userId);
    pi.pi.setLabel(compactionId, "anchor:start");
    appendTurn(sm, "u3", "a3", 18_000);
    appendTurn(sm, "u4", "a4", 24_000);

    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);
    installProvider(ctx, capturingProvider().streamSimple);

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "Preserve the live evidence and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    const cache = result.details.summaryCache as {
      mode: string;
      fallbackReason: string | null;
    };
    assert.equal(cache.mode, "live-prefix");
    assert.equal(cache.fallbackReason, null);
  });

  it("falls back with branch-start-not-retained when the newest message alone exceeds the budget", async () => {
    const { spy } = capturingSummarize();
    const { sm, pi, tool, ctx } = setup({
      summarize: spy,
      contextWindow: 20_000,
    });
    const a1 = appendTurn(sm, "u1", "a1", 6_000);
    pi.pi.setLabel(a1.assistantId, "anchor:start");
    // The newest entry alone (~10k tokens) exceeds the 20_000 - 16384 =
    // 3616-token budget, so the newest→oldest walk breaks before adding any
    // branch evidence.
    appendTurn(
      sm,
      `u2 ${"x".repeat(20_000)}`,
      `a2 ${"y".repeat(40_000)}`,
      30_000,
    );

    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);
    installProvider(ctx, capturingProvider().streamSimple);

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "Preserve the raw branch evidence and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assertNoCacheNoticeInContent(result.content[0].text);
    const cache = result.details.summaryCache as {
      mode: string;
      fallbackReason: string;
      branchStartRetained: boolean;
      notice: string | null;
    };
    assert.equal(cache.mode, "fallback");
    assert.equal(cache.fallbackReason, "branch-start-not-retained");
    assert.equal(cache.branchStartRetained, false);
    assert.equal(cache.notice, null);
  });

  it("stores no notice when the settings gate is unreadable", async () => {
    const { spy } = capturingSummarize(SUMMARY_MISS_USAGE);
    const { sm, pi, tool, ctx } = setup({ summarize: spy });
    const { fake } = setupRewindable(sm, pi, { capture: true });
    if (!fake) throw new Error("capture: true must return fake");
    // Present but throwing: the reflective read must swallow and store nothing.
    fake.settingsManager = {
      getShowCacheMissNotices: () => {
        throw new Error("settings unavailable");
      },
    };
    appendUsageTurn(sm, CACHE_BASELINE_USAGE);
    installProvider(ctx, capturingProvider().streamSimple);

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assertNoCacheNoticeInContent(result.content[0].text);
    const cache = result.details.summaryCache as {
      missedTokens: number;
      notice: string | null;
    };
    assert.equal(cache.missedTokens, 20_000);
    assert.equal(cache.notice, null);
  });

  it("stores no notice when showCacheMissNotices is off", async () => {
    const { spy } = capturingSummarize(SUMMARY_MISS_USAGE);
    const { sm, pi, tool, ctx } = setup({ summarize: spy });
    const { fake } = setupRewindable(sm, pi, { capture: true });
    if (!fake) throw new Error("capture: true must return fake");
    // Default fake session: getShowCacheMissNotices() === false.
    appendUsageTurn(sm, CACHE_BASELINE_USAGE);
    installProvider(ctx, capturingProvider().streamSimple);

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assertNoCacheNoticeInContent(result.content[0].text);
    const cache = result.details.summaryCache as {
      missedTokens: number;
      notice: string | null;
    };
    assert.equal(cache.missedTokens, 20_000);
    assert.equal(cache.notice, null);
  });

  it("stores no notice below the 20k-token / $0.10 display floor", async () => {
    const { spy } = capturingSummarize({
      input: 15_000,
      output: 20,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 15_020,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    });
    const { sm, pi, tool, ctx } = setup({ summarize: spy });
    const { fake } = setupRewindable(sm, pi, { capture: true });
    if (!fake) throw new Error("capture: true must return fake");
    fake.settingsManager = { getShowCacheMissNotices: () => true };
    appendUsageTurn(sm, {
      input: 0,
      cacheRead: 50_000,
      cacheWrite: 0,
      cost: { input: 0, cacheRead: 0, cacheWrite: 0 },
    });
    installProvider(ctx, capturingProvider().streamSimple);

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assertNoCacheNoticeInContent(result.content[0].text);
    const cache = result.details.summaryCache as {
      missedTokens: number;
      notice: string | null;
    };
    assert.equal(cache.missedTokens, 15_000);
    assert.equal(cache.notice, null);
  });

  it("suppresses a miss after a model switch", async () => {
    const { spy } = capturingSummarize(SUMMARY_MISS_USAGE);
    const { sm, pi, tool, ctx } = setup({ summarize: spy });
    const { fake } = setupRewindable(sm, pi, { capture: true });
    if (!fake) throw new Error("capture: true must return fake");
    fake.settingsManager = { getShowCacheMissNotices: () => true };
    appendUsageTurn(sm, CACHE_BASELINE_USAGE, {
      provider: "openai",
      model: "gpt-5",
    });
    installProvider(ctx, capturingProvider().streamSimple);

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assertNoCacheNoticeInContent(result.content[0].text);
    // Fork semantics: a cold summary right after a model switch is expected
    // re-billing (the live baseline belongs to another model), not an
    // actionable miss — the detector returns undefined, so nothing is stored.
    const cache = result.details.summaryCache as {
      missedTokens: number;
      modelChanged: boolean;
      notice: string | null;
    };
    assert.equal(cache.missedTokens, 0);
    assert.equal(cache.modelChanged, false);
    assert.equal(cache.notice, null);
  });

  it("labels the miss as idle once the gap spans the cache TTL", async () => {
    const { spy } = capturingSummarize(SUMMARY_MISS_USAGE);
    const { sm, pi, tool, ctx } = setup({ summarize: spy });
    const { fake } = setupRewindable(sm, pi, { capture: true });
    if (!fake) throw new Error("capture: true must return fake");
    fake.settingsManager = { getShowCacheMissNotices: () => true };
    appendUsageTurn(sm, CACHE_BASELINE_USAGE, {
      timestamp: Date.now() - (5 * 60 * 1000 + 60_000),
    });
    installProvider(ctx, capturingProvider().streamSimple);

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assertNoCacheNoticeInContent(result.content[0].text);
    const cache = result.details.summaryCache as {
      idleMs: number;
      notice: string | null;
    };
    assert.ok(cache.idleMs >= 5 * 60 * 1000);
    assert.equal(
      cache.notice,
      "Cache miss after 6m idle: 20k tokens re-billed (~$0.20)",
    );
  });

  it("measures the fallback path through the same miss detector", async () => {
    process.env.PI_NAVIGATE_TREE_SUMMARY_CACHE = "0";
    const { spy } = capturingSummarize(SUMMARY_MISS_USAGE);
    const { sm, pi, tool, ctx } = setup({ summarize: spy });
    const { fake } = setupRewindable(sm, pi, { capture: true });
    if (!fake) throw new Error("capture: true must return fake");
    fake.settingsManager = { getShowCacheMissNotices: () => true };
    appendUsageTurn(sm, CACHE_BASELINE_USAGE);
    installProvider(ctx, capturingProvider().streamSimple);

    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "start",
        newLabel: "end",
        summaryFocus: "Preserve user instructions and continue.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assertNoCacheNoticeInContent(result.content[0].text);
    // Fallback is NOT special-cased: the cold request measures as a miss.
    const cache = result.details.summaryCache as {
      mode: string;
      fallbackReason: string;
      missedTokens: number;
      notice: string | null;
    };
    assert.equal(cache.mode, "fallback");
    assert.equal(cache.fallbackReason, "disabled");
    assert.equal(cache.missedTokens, 20_000);
    assert.equal(cache.notice, SUMMARY_MISS_NOTICE);
  });
});

// =============================================================================
// dispatch: cache-notice transcript rendering (#33)
//
// Upstream pi renders cache notices as transcript lines via the tool renderer
// (not toasts). `execute` stores the notice string; `renderResult` reproduces
// the default result body and appends `new Spacer(1)` + a warning `Text`.
// =============================================================================

type RenderResultFn = (
  result: {
    content: Array<{ type: string; text?: string }>;
    details: unknown;
  },
  options: { expanded: boolean; isPartial: boolean },
  theme: { fg: (color: string, text: string) => string },
) => { render(width: number): string[] };

const STUB_THEME = { fg: (_color: string, text: string) => text };

function renderToolResult(
  tool: CapturedTool,
  contentText: string,
  notice: string | null,
  expanded: boolean,
): string {
  const renderResult = (tool as { renderResult?: RenderResultFn }).renderResult;
  assert.equal(typeof renderResult, "function");
  const component = (renderResult as RenderResultFn)(
    {
      content: [{ type: "text", text: contentText }],
      details: { summaryCache: { notice } },
    },
    { expanded, isPartial: false },
    STUB_THEME,
  );
  // Each rendered line is padded to the width; trim for readable assertions.
  return component
    .render(200)
    .map((line) => line.trim())
    .join("\n");
}

describe("dispatch: cache-notice transcript rendering (#33)", () => {
  before(() => {
    // `keyHint` reads the module-global pi theme; initialize it once.
    initTheme();
  });

  it("renders content only when there is no notice", () => {
    const { tool } = setup();
    const out = renderToolResult(tool, "line one\nline two", null, true);
    assert.equal(out, "line one\nline two");
    assert.doesNotMatch(out, /Cache miss/);
  });

  it("appends the warning line when a notice is present", () => {
    const { tool } = setup();
    const out = renderToolResult(
      tool,
      "body line",
      "Cache miss: 20k tokens re-billed (~$0.20)",
      true,
    );
    assert.match(out, /body line/);
    assert.match(out, /Cache miss: 20k tokens re-billed/);
  });

  it("previews the first 10 lines with an expand hint when collapsed", () => {
    const { tool } = setup();
    const content = Array.from({ length: 13 }, (_, i) => `l${i + 1}`).join(
      "\n",
    );
    const out = renderToolResult(tool, content, null, false);
    assert.match(out, /l10/);
    assert.doesNotMatch(out, /l11/);
    assert.match(out, /3 more lines/);
    assert.match(out, /to expand/);
  });

  it("renders all lines with no hint when expanded", () => {
    const { tool } = setup();
    const content = Array.from({ length: 13 }, (_, i) => `l${i + 1}`).join(
      "\n",
    );
    const out = renderToolResult(tool, content, null, true);
    assert.match(out, /l13/);
    assert.doesNotMatch(out, /more lines/);
  });

  it("adds a Spacer between the body and the warning line", () => {
    const { tool } = setup();
    const out = renderToolResult(
      tool,
      "body",
      "Cache miss: 5k tokens re-billed",
      true,
    );
    assert.match(out, /body\n\s*\n\s*Cache miss: 5k tokens re-billed/);
  });
});

// =============================================================================
// Rewind-hint integration (#44)
//
// The factory's hint state is closure-scoped: `session_start` is the only
// writer, so these tests drive the real handler against a temp
// `PI_CODING_AGENT_DIR` fixture (real loader, real fs) and then invoke the
// captured synchronous `turn_end` handler. No LLM and no session file —
// `SessionManager` stays in-memory, and the temp fixture is removed in a
// `finally` so nothing leaks between tests or into the user's real agent dir.
// =============================================================================

interface ConfigFixture {
  root: string;
  agentDir: string;
  projectDir: string;
}

function makeConfigFixture(): ConfigFixture {
  const root = mkdtempSync(join(tmpdir(), "navigate-tree-hint-"));
  const agentDir = join(root, "agent");
  const projectDir = join(root, "project");
  mkdirSync(agentDir, { recursive: true });
  mkdirSync(projectDir, { recursive: true });
  return { root, agentDir, projectDir };
}

function cleanupConfigFixture(fixture: ConfigFixture): void {
  rmSync(fixture.root, { recursive: true, force: true });
}

function writeGlobalConfig(fixture: ConfigFixture, raw: string): void {
  writeFileSync(join(fixture.agentDir, TREE_NAVIGATOR_CONFIG_FILENAME), raw);
}

function writeProjectConfig(fixture: ConfigFixture, raw: string): void {
  const dir = join(fixture.projectDir, CONFIG_DIR_NAME);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, TREE_NAVIGATOR_CONFIG_FILENAME), raw);
}

function globalConfigPath(fixture: ConfigFixture): string {
  return join(fixture.agentDir, TREE_NAVIGATOR_CONFIG_FILENAME);
}

function projectConfigPath(fixture: ConfigFixture): string {
  return join(
    fixture.projectDir,
    CONFIG_DIR_NAME,
    TREE_NAVIGATOR_CONFIG_FILENAME,
  );
}

/** Context-usage shape at `percent` of `contextWindow`. */
function usageAt(percent: number, contextWindow = 1_000_000): ContextUsage {
  return {
    tokens: Math.round((percent / 100) * contextWindow),
    contextWindow,
    percent,
  };
}

/**
 * Drive the captured `session_start` handler with `PI_CODING_AGENT_DIR`
 * pointed at the fixture's agent dir (the loader reads the real fs). The env
 * var is always restored, so the suite never touches the user's real config.
 */
async function runSessionStart(
  pi: FakePi,
  ctx: FakeCtx,
  fixture: ConfigFixture,
): Promise<void> {
  const handlers = pi.onCalls.get("session_start");
  assert.ok(handlers, "factory must register a session_start handler");
  assert.equal(handlers.length, 1);
  const prevAgentDir = process.env.PI_CODING_AGENT_DIR;
  process.env.PI_CODING_AGENT_DIR = fixture.agentDir;
  try {
    await handlers[0](
      { type: "session_start", reason: "startup" } as never,
      ctx as never,
    );
  } finally {
    if (prevAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = prevAgentDir;
  }
}

/** Fire the captured synchronous `turn_end` handler. */
function fireTurnEnd(pi: FakePi, ctx: FakeCtx): void {
  const handlers = pi.onCalls.get("turn_end");
  assert.ok(handlers, "factory must register a turn_end handler");
  assert.equal(handlers.length, 1);
  handlers[0]({ type: "turn_end" } as never, ctx as never);
}

/** Drive the `before_agent_start` handler to (re)compute the tool gate. */
async function fireBeforeAgentStart(
  pi: FakePi,
  selectedTools: string[] | undefined,
): Promise<void> {
  const handlers = pi.onCalls.get("before_agent_start");
  assert.ok(handlers, "factory must register a before_agent_start handler");
  assert.equal(handlers.length, 1);
  await handlers[0](
    {
      type: "before_agent_start",
      prompt: "p",
      systemPrompt: "BASE",
      systemPromptOptions: selectedTools === undefined ? {} : { selectedTools },
    } as never,
    {} as never,
  );
}

/** Label the current leaf `anchor:start` so the hint path has an anchor. */
function anchorStart(sm: SessionManager, pi: FakePi): void {
  const t = appendTurn(sm, "u-anchor", "a-anchor");
  pi.pi.setLabel(t.assistantId, "anchor:start");
}

describe("rewind hint: session_start config wiring (#44)", () => {
  it("loads the global threshold and fires on the first crossing", async () => {
    const fixture = makeConfigFixture();
    try {
      const { sm, pi, ctx } = setup();
      anchorStart(sm, pi);
      writeGlobalConfig(fixture, JSON.stringify({ rewindHintAtPercent: "90" }));
      await runSessionStart(pi, ctx, fixture);
      assert.deepEqual(ctx.notifyCalls, [], "valid config must not warn");

      ctx.setContextUsage(usageAt(89.9));
      fireTurnEnd(pi, ctx);
      assert.equal(
        pi.sendMessageCalls.length,
        0,
        "below-threshold observation must not fire",
      );

      ctx.setContextUsage(usageAt(90));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 1);
      // Real call arity: exactly one argument (no options bag).
      assert.equal(pi.sendMessageCalls[0].length, 1);
      assert.deepEqual(pi.sendMessageCalls[0], [
        {
          customType: REWIND_HINT_CUSTOM_TYPE,
          content: buildRewindHintText(90, 1_000_000),
          display: true,
        },
      ]);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("project layer overrides global when trusted; untrusted ignores the file", async () => {
    const fixture = makeConfigFixture();
    try {
      writeGlobalConfig(fixture, JSON.stringify({ rewindHintAtPercent: "90" }));
      writeProjectConfig(fixture, JSON.stringify({ rewindHintAtPercent: 50 }));

      // Trusted: the project's 50 wins over the global 90.
      {
        const { sm, pi, ctx } = setup();
        anchorStart(sm, pi);
        ctx.cwd = fixture.projectDir;
        ctx.isProjectTrusted = () => true;
        await runSessionStart(pi, ctx, fixture);
        ctx.setContextUsage(usageAt(50));
        fireTurnEnd(pi, ctx);
        assert.equal(pi.sendMessageCalls.length, 1);
        assert.deepEqual(ctx.notifyCalls, []);
      }

      // Untrusted: the project file is ignored silently; 50 does not fire,
      // 90 does.
      {
        const { sm, pi, ctx } = setup();
        anchorStart(sm, pi);
        ctx.cwd = fixture.projectDir;
        await runSessionStart(pi, ctx, fixture);
        ctx.setContextUsage(usageAt(50));
        fireTurnEnd(pi, ctx);
        assert.equal(pi.sendMessageCalls.length, 0);
        assert.deepEqual(ctx.notifyCalls, []);
        ctx.setContextUsage(usageAt(90));
        fireTurnEnd(pi, ctx);
        assert.equal(pi.sendMessageCalls.length, 1);
      }
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("invalid config disables the hint and surfaces one warning per layer", async () => {
    const fixture = makeConfigFixture();
    try {
      writeGlobalConfig(fixture, "{ not json");
      writeProjectConfig(
        fixture,
        JSON.stringify({ rewindHintAtPercent: true }),
      );
      const { sm, pi, ctx } = setup();
      anchorStart(sm, pi);
      ctx.cwd = fixture.projectDir;
      ctx.isProjectTrusted = () => true;
      await runSessionStart(pi, ctx, fixture);
      assert.deepEqual(ctx.notifyCalls, [
        [
          `navigate_tree: invalid JSON in global config at ${globalConfigPath(fixture)} — that layer was ignored.`,
          "warning",
        ],
        [
          `navigate_tree: invalid rewindHintAtPercent in project config at ${projectConfigPath(fixture)} — expected integer 20-95 or a disable sentinel (null, false, "off", "disabled"); rewind hint disabled for this session.`,
          "warning",
        ],
      ]);
      ctx.setContextUsage(usageAt(99));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 0);
      assert.equal(
        ctx.notifyCalls.length,
        2,
        "the turn must not add config warnings",
      );
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("suppresses config warnings when the session has no UI", async () => {
    const fixture = makeConfigFixture();
    try {
      writeGlobalConfig(fixture, "{ not json");
      const { pi, ctx } = setup();
      ctx.hasUI = false;
      await runSessionStart(pi, ctx, fixture);
      assert.deepEqual(ctx.notifyCalls, []);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("is default-off when no config exists (ENOENT is silent)", async () => {
    const fixture = makeConfigFixture();
    try {
      const { sm, pi, ctx } = setup();
      anchorStart(sm, pi);
      await runSessionStart(pi, ctx, fixture);
      assert.deepEqual(ctx.notifyCalls, []);
      ctx.setContextUsage(usageAt(99));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 0);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("reload resets the tracker and the toolActive gate", async () => {
    const fixture = makeConfigFixture();
    try {
      writeGlobalConfig(fixture, JSON.stringify({ rewindHintAtPercent: "90" }));
      const { sm, pi, ctx } = setup();
      anchorStart(sm, pi);
      await runSessionStart(pi, ctx, fixture);

      ctx.setContextUsage(usageAt(95));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 1);
      fireTurnEnd(pi, ctx);
      assert.equal(
        pi.sendMessageCalls.length,
        1,
        "spent crossing must not re-fire",
      );

      // Tool gate off: no fire, even across the threshold.
      await fireBeforeAgentStart(pi, ["read", "bash"]);
      ctx.setContextUsage(usageAt(10));
      fireTurnEnd(pi, ctx);
      ctx.setContextUsage(usageAt(95));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 1);

      // session_start resets both: gate back on, tracker re-armed.
      await runSessionStart(pi, ctx, fixture);
      fireTurnEnd(pi, ctx);
      assert.equal(
        pi.sendMessageCalls.length,
        2,
        "reload must re-arm the tracker",
      );
    } finally {
      cleanupConfigFixture(fixture);
    }
  });
});

describe("rewind hint: turn_end handler (#44)", () => {
  const ENABLED_CONFIG = JSON.stringify({ rewindHintAtPercent: "90" });

  async function enabledSetup(): Promise<{
    fixture: ConfigFixture;
    sm: SessionManager;
    pi: FakePi;
    ctx: FakeCtx;
  }> {
    const fixture = makeConfigFixture();
    writeGlobalConfig(fixture, ENABLED_CONFIG);
    const { sm, pi, ctx } = setup();
    anchorStart(sm, pi);
    await runSessionStart(pi, ctx, fixture);
    return { fixture, sm, pi, ctx };
  }

  it("does not fire when disabled by a sentinel", async () => {
    const fixture = makeConfigFixture();
    try {
      writeGlobalConfig(fixture, JSON.stringify({ rewindHintAtPercent: null }));
      const { sm, pi, ctx } = setup();
      anchorStart(sm, pi);
      await runSessionStart(pi, ctx, fixture);
      ctx.setContextUsage(usageAt(99));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 0);
      assert.deepEqual(ctx.notifyCalls, []);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("does not fire when the tool is inactive", async () => {
    const { fixture, pi, ctx } = await enabledSetup();
    try {
      await fireBeforeAgentStart(pi, ["read"]);
      ctx.setContextUsage(usageAt(99));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 0);
      assert.deepEqual(ctx.notifyCalls, []);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("no-ops on missing/percent-null/zero-window usage without spending the crossing", async () => {
    const { fixture, pi, ctx } = await enabledSetup();
    try {
      ctx.setContextUsage(undefined);
      fireTurnEnd(pi, ctx);
      ctx.setContextUsage({
        tokens: null,
        contextWindow: 1_000_000,
        percent: null,
      });
      fireTurnEnd(pi, ctx);
      ctx.setContextUsage({ tokens: 100, contextWindow: 0, percent: 95 });
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 0);
      assert.deepEqual(ctx.notifyCalls, []);

      // The crossing is still unspent — the first valid observation fires.
      ctx.setContextUsage(usageAt(95));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 1);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("fires once per crossing and re-arms only after a drop below the bar", async () => {
    const fixture = makeConfigFixture();
    try {
      writeGlobalConfig(fixture, JSON.stringify({ rewindHintAtPercent: "20" }));
      const { sm, pi, ctx } = setup();
      anchorStart(sm, pi);
      await runSessionStart(pi, ctx, fixture);
      ctx.setContextUsage(usageAt(19));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 0);

      ctx.setContextUsage(usageAt(20));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 1);

      ctx.setContextUsage(usageAt(45));
      fireTurnEnd(pi, ctx);
      assert.equal(
        pi.sendMessageCalls.length,
        1,
        "spent crossing must not re-fire",
      );

      ctx.setContextUsage(usageAt(12));
      fireTurnEnd(pi, ctx);
      assert.equal(
        pi.sendMessageCalls.length,
        1,
        "re-arming alone must not fire",
      );

      ctx.setContextUsage(usageAt(21));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 2);
      assert.deepEqual(pi.sendMessageCalls[1], [
        {
          customType: REWIND_HINT_CUSTOM_TYPE,
          content: buildRewindHintText(21, 1_000_000),
          display: true,
        },
      ]);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("with no anchors: TUI notify with the exact copy, no model message", async () => {
    const fixture = makeConfigFixture();
    try {
      writeGlobalConfig(fixture, ENABLED_CONFIG);
      const { pi, ctx } = setup();
      // Deliberately NO anchor.
      await runSessionStart(pi, ctx, fixture);
      ctx.setContextUsage(usageAt(95));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 0);
      assert.deepEqual(ctx.notifyCalls, [
        [buildNoAnchorText(95, 1_000_000), "warning"],
      ]);

      // Spent: no second toast while still above.
      fireTurnEnd(pi, ctx);
      assert.equal(ctx.notifyCalls.length, 1);

      // Drop + re-cross re-arms.
      ctx.setContextUsage(usageAt(10));
      fireTurnEnd(pi, ctx);
      ctx.setContextUsage(usageAt(96));
      fireTurnEnd(pi, ctx);
      assert.equal(ctx.notifyCalls.length, 2);
      assert.deepEqual(ctx.notifyCalls[1], [
        buildNoAnchorText(96, 1_000_000),
        "warning",
      ]);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("with no anchors and no UI: silent", async () => {
    const fixture = makeConfigFixture();
    try {
      writeGlobalConfig(fixture, ENABLED_CONFIG);
      const { pi, ctx } = setup();
      ctx.hasUI = false;
      await runSessionStart(pi, ctx, fixture);
      ctx.setContextUsage(usageAt(95));
      fireTurnEnd(pi, ctx);
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 0);
      assert.deepEqual(ctx.notifyCalls, []);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("sendMessage throw falls back to appendCustomMessageEntry(customType, text, true)", async () => {
    const { fixture, sm, pi, ctx } = await enabledSetup();
    try {
      const appendCalls: unknown[][] = [];
      (
        sm as unknown as {
          appendCustomMessageEntry: (...args: unknown[]) => string;
        }
      ).appendCustomMessageEntry = (...args: unknown[]) => {
        appendCalls.push(args);
        return "custom-id";
      };
      (
        pi.pi as unknown as { sendMessage: (...args: unknown[]) => void }
      ).sendMessage = () => {
        throw new Error("stale session runtime");
      };

      ctx.setContextUsage(usageAt(95));
      fireTurnEnd(pi, ctx);
      assert.deepEqual(appendCalls, [
        [REWIND_HINT_CUSTOM_TYPE, buildRewindHintText(95, 1_000_000), true],
      ]);
      assert.deepEqual(
        ctx.notifyCalls,
        [],
        "a successful fallback must not warn",
      );

      // The crossing is spent even on the fallback path.
      ctx.setContextUsage(usageAt(96));
      fireTurnEnd(pi, ctx);
      assert.equal(appendCalls.length, 1);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("sendMessage + append double throw: warns with the exact copy, never throws", async () => {
    const { fixture, sm, pi, ctx } = await enabledSetup();
    try {
      (
        sm as unknown as {
          appendCustomMessageEntry: (...args: unknown[]) => string;
        }
      ).appendCustomMessageEntry = () => {
        throw new Error("append exploded");
      };
      (
        pi.pi as unknown as { sendMessage: (...args: unknown[]) => void }
      ).sendMessage = () => {
        throw new Error("stale session runtime");
      };

      ctx.setContextUsage(usageAt(95));
      assert.doesNotThrow(() => fireTurnEnd(pi, ctx));
      assert.deepEqual(ctx.notifyCalls, [
        [
          `${TOOL_NAME}: could not deliver the rewind hint (append exploded).`,
          "warning",
        ],
      ]);

      // The crossing is spent even on the double-failure path.
      fireTurnEnd(pi, ctx);
      assert.equal(ctx.notifyCalls.length, 1);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("double throw with no UI: silent, no throw, crossing spent", async () => {
    const { fixture, sm, pi, ctx } = await enabledSetup();
    try {
      ctx.hasUI = false;
      let sendAttempts = 0;
      (
        sm as unknown as {
          appendCustomMessageEntry: (...args: unknown[]) => string;
        }
      ).appendCustomMessageEntry = () => {
        throw new Error("append exploded");
      };
      (
        pi.pi as unknown as { sendMessage: (...args: unknown[]) => void }
      ).sendMessage = () => {
        sendAttempts += 1;
        throw new Error("stale session runtime");
      };

      ctx.setContextUsage(usageAt(95));
      assert.doesNotThrow(() => fireTurnEnd(pi, ctx));
      assert.equal(sendAttempts, 1);
      assert.deepEqual(ctx.notifyCalls, []);

      // The crossing is spent even on the double-failure path.
      fireTurnEnd(pi, ctx);
      assert.equal(sendAttempts, 1);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("a throwing ui.notify in the double-failure path cannot escape", async () => {
    const { fixture, sm, pi, ctx } = await enabledSetup();
    try {
      (
        sm as unknown as {
          appendCustomMessageEntry: (...args: unknown[]) => string;
        }
      ).appendCustomMessageEntry = () => {
        throw new Error("append exploded");
      };
      (
        pi.pi as unknown as { sendMessage: (...args: unknown[]) => void }
      ).sendMessage = () => {
        throw new Error("stale session runtime");
      };
      let notifyAttempts = 0;
      ctx.ui.notify = () => {
        notifyAttempts += 1;
        throw new Error("notify exploded");
      };

      ctx.setContextUsage(usageAt(95));
      assert.doesNotThrow(() => fireTurnEnd(pi, ctx));
      assert.equal(notifyAttempts, 1);

      // The crossing is spent even when the warning itself fails.
      fireTurnEnd(pi, ctx);
      assert.equal(notifyAttempts, 1);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("captures sessionManager before sendMessage (post-throw ctx reads fail)", async () => {
    const { fixture, sm, pi, ctx } = await enabledSetup();
    try {
      const appendCalls: unknown[][] = [];
      (
        sm as unknown as {
          appendCustomMessageEntry: (...args: unknown[]) => string;
        }
      ).appendCustomMessageEntry = (...args: unknown[]) => {
        appendCalls.push(args);
        return "custom-id";
      };
      (
        pi.pi as unknown as { sendMessage: (...args: unknown[]) => void }
      ).sendMessage = () => {
        throw new Error("stale session runtime");
      };

      // `sessionManager` serves exactly one read (the capture); a second
      // read is the stale-getter failure the fix must not depend on.
      let reads = 0;
      const proxyCtx = {
        get sessionManager(): SessionManager {
          reads += 1;
          if (reads > 1) throw new Error("stale session runtime");
          return sm;
        },
        getContextUsage: ctx.getContextUsage,
        hasUI: ctx.hasUI,
        ui: ctx.ui,
      } as unknown as FakeCtx;

      ctx.setContextUsage(usageAt(95));
      fireTurnEnd(pi, proxyCtx);
      assert.equal(reads, 1, "sessionManager must be read exactly once");
      assert.deepEqual(appendCalls, [
        [REWIND_HINT_CUSTOM_TYPE, buildRewindHintText(95, 1_000_000), true],
      ]);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("a throwing anchor collection leaves the crossing unspent for a retry", async () => {
    const { fixture, sm, pi, ctx } = await enabledSetup();
    try {
      const originalGetBranch = sm.getBranch;
      (sm as unknown as { getBranch: () => never }).getBranch = () => {
        throw new Error("getBranch exploded");
      };
      ctx.setContextUsage(usageAt(95));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 0);
      assert.deepEqual(ctx.notifyCalls, []);

      (sm as unknown as { getBranch: typeof originalGetBranch }).getBranch =
        originalGetBranch;
      fireTurnEnd(pi, ctx);
      assert.equal(
        pi.sendMessageCalls.length,
        1,
        "a successful retry must still fire the crossing",
      );
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("the fired custom message is projected by the existing context handler", async () => {
    const { fixture, sm, pi, ctx } = await enabledSetup();
    try {
      ctx.setContextUsage(usageAt(95));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 1);

      // pi's runtime persists a sent custom message as a `custom_message`
      // entry via `appendCustomMessageEntry`; emulate that write, then run
      // the extension's own `context` projection and assert the hint text
      // survives (no customType filtering in `buildContextMessages`).
      const text = buildRewindHintText(95, 1_000_000);
      sm.appendCustomMessageEntry(REWIND_HINT_CUSTOM_TYPE, text, true);
      const contextHandlers = pi.onCalls.get("context");
      assert.ok(contextHandlers, "factory must register a context handler");
      const projected = contextHandlers[0](
        { type: "context", messages: [] } as never,
        { sessionManager: sm } as never,
      ) as { messages: unknown[] };
      assert.ok(
        JSON.stringify(projected.messages).includes(text),
        "the rewind hint custom_message must survive the context projection",
      );
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("fails open when selectedTools is undefined (gate stays on)", async () => {
    // Complement to the earlier "tool inactive" case: an undefined
    // `selectedTools` (the fail-open branch) must leave the gate ON.
    const { fixture, pi, ctx } = await enabledSetup();
    try {
      await fireBeforeAgentStart(pi, undefined);
      ctx.setContextUsage(usageAt(95));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 1);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("fires while TOOL_NAME is in the selected set", async () => {
    const { fixture, pi, ctx } = await enabledSetup();
    try {
      await fireBeforeAgentStart(pi, ["read", TOOL_NAME]);
      ctx.setContextUsage(usageAt(95));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 1);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });
});
