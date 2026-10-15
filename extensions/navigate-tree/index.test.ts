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
 *   - `rewind` is in-band (#80): the tool call only returns the summary
 *     instruction; tests then append the model's summary turn to the
 *     SessionManager and fire the registered `turn_end` handlers, exactly as
 *     the agent loop would. No LLM call exists anywhere in the flow.
 */

import * as assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, it } from "node:test";
import {
  type AgentSession,
  CONFIG_DIR_NAME,
  type ContextUsage,
  type ExtensionAPI,
  SessionManager,
} from "@earendil-works/pi-coding-agent";
import {
  START_ANCHOR_DEFAULT_TURNS,
  START_ANCHOR_MAX_TURNS,
  START_ANCHOR_MIN_TURNS,
  TREE_NAVIGATOR_CONFIG_FILENAME,
} from "./config.ts";
import { MAX_NAME_LENGTH, TOOL_NAME } from "./helpers.ts";
import navigateTree, {
  __testHooks,
  ANCHOR_MANDATE,
  buildLateEntriesNote,
  buildRewindCancelledText,
  buildRewindDoneText,
  buildSummaryRetryText,
  extractSummaryText,
  MAX_HINT_WALK_DEPTH,
  MAX_SESSION_REFS,
  MAX_SUMMARY_ATTEMPTS,
  MIN_REWIND_SAVINGS_TOKENS,
  MIN_SUMMARY_FOCUS_LENGTH,
  REWIND_CUSTOM_TYPE,
  START_ANCHOR_LABEL,
  SUMMARY_PENDING_BLOCK_REASON,
} from "./index.ts";
import {
  buildNoAnchorText,
  buildRewindHintText,
  REWIND_HINT_CUSTOM_TYPE,
} from "./rewind-hint.ts";
import {
  BRANCH_SUMMARY_PREAMBLE,
  buildStoredSummary,
  buildSummaryInstruction,
} from "./summary-prompt.ts";

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
   * contract is a ONE-argument call (no options bag). Rewind messages never
   * go through `sendMessage`: they are `turn_end` boundary entries.
   */
  sendMessageCalls: unknown[][];
  /**
   * Raw argument arrays for every `sendUserMessage` call. Late input is
   * re-appended directly, never re-sent: tests pin this stays empty.
   */
  sendUserMessageCalls: unknown[][];
  /** Interleaved `sendMessage` / `sendUserMessage` call order. */
  sendLog: Array<"sendMessage" | "sendUserMessage">;
  /** Folded `turn_end` boundary results, one per `fireTurnEndAll`. */
  boundaryResults: BoundaryResult[];
}

/** A `turn_end` boundary entry draft (pi's custom_message shape). */
interface BoundaryDraft {
  type: string;
  customType: string;
  content: string;
  display: boolean;
  details: Record<string, unknown>;
}

/** The folded `turn_end` boundary result, as `emitBoundary` returns it. */
interface BoundaryResult {
  entries: BoundaryDraft[];
  continue: boolean;
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
  const sendUserMessageCalls: unknown[][] = [];
  const sendLog: FakePi["sendLog"] = [];
  const boundaryResults: BoundaryResult[] = [];
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
      sendLog.push("sendMessage");
    },
    sendUserMessage(...args: unknown[]) {
      sendUserMessageCalls.push(args);
      sendLog.push("sendUserMessage");
    },
  } as unknown as ExtensionAPI;
  return {
    pi,
    setLabelCalls,
    registered,
    onCalls,
    sendMessageCalls,
    sendUserMessageCalls,
    sendLog,
    boundaryResults,
  };
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
  };
}

const ZERO_COST = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  total: 0,
};

function usageOf(totalTokens: number) {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens,
    cost: ZERO_COST,
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
    usage: usageOf(totalTokens),
  } as never);
  return { userId, assistantId };
}

/** An assistant message as the agent loop hands it to `turn_end`. */
type AssistantTurn = {
  role: "assistant";
  content: Array<Record<string, unknown>>;
  stopReason: string;
  [k: string]: unknown;
};

function assistantMessage(
  content: Array<Record<string, unknown>>,
  stopReason: string,
  totalTokens = 0,
): AssistantTurn {
  return {
    role: "assistant",
    content,
    api: "anthropic",
    provider: "claude",
    model: "claude-sonnet-4-5",
    stopReason,
    timestamp: Date.now(),
    usage: usageOf(totalTokens),
  };
}

/**
 * Append an assistant message declaring tool calls (each with `arguments`),
 * the shape the model emits for the rewind call itself or for a failed
 * summary turn. Returns the appended message (for `turn_end`) and its id.
 */
function appendToolCallAssistant(
  sm: SessionManager,
  toolCalls: Array<{
    id: string;
    name: string;
    arguments?: Record<string, unknown>;
  }>,
  totalTokens = 30_000,
): { id: string; message: AssistantTurn } {
  const message = assistantMessage(
    toolCalls.map((tc) => ({
      type: "toolCall",
      id: tc.id,
      name: tc.name,
      arguments: tc.arguments ?? { action: "rewind" },
    })),
    "toolUse",
    totalTokens,
  );
  return { id: sm.appendMessage(message as never), message };
}

/** Append the tool_result pi records after a tool's `execute` returns. */
function appendToolResult(
  sm: SessionManager,
  toolCallId: string,
  toolName: string,
  text: string,
  details: unknown = {},
): string {
  return sm.appendMessage({
    role: "toolResult",
    toolCallId,
    toolName,
    content: [{ type: "text", text }],
    details,
    isError: false,
    timestamp: Date.now(),
  } as never);
}

/** Canonical well-formed summary turn text (r5d headings). */
const SUMMARY_TEXT =
  "## Goal\nTest the rewind happy path.\n\n## Progress\n### Done\n- [x] Appended turns.\n\n## Next Steps\n1. Verify the branch_summary.";

/**
 * Register the navigate-tree tool against a fresh fake pi + SessionManager
 * and return the captured tool plus the helpers tests need.
 */
function setup(opts: { contextWindow?: number; noModel?: boolean } = {}): {
  sm: SessionManager;
  pi: FakePi;
  ctx: FakeCtx;
  tool: CapturedTool;
} {
  const sm = SessionManager.inMemory("/tmp");
  const pi = makeFakePi(sm);
  navigateTree(pi.pi);
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
    prepareNextTurnWithContext?: unknown;
  };
  /** Plain field on `AgentSession`; the rewind emits `session_tree` on it (#79). */
  _extensionRunner?: { emit?: (event: unknown) => Promise<unknown> | unknown };
}

function makeFakeSession(sm: SessionManager): FakeAgentSession {
  return {
    sessionManager: sm,
    agent: {
      state: { systemPrompt: "SYSTEM", messages: [], tools: [] },
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
): { fake?: FakeAgentSession; anchorId: string } {
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
    return { fake, anchorId: t1.assistantId };
  }
  return { anchorId: t1.assistantId };
}

// -----------------------------------------------------------------------------
// In-band rewind drivers (#80)
// -----------------------------------------------------------------------------

/**
 * Fire every registered `turn_end` handler in registration order, awaiting
 * each, and fold their results exactly as pi's `ExtensionRunner.emitBoundary`
 * does: every handler sees the running `{ entries, continue }`, and a result
 * carrying `entries` / `continue` replaces it. The folded result is returned
 * and recorded on `pi.boundaryResults`. `entries` seeds the list (another
 * extension's drafts earlier in the chain). `commit` replicates pi's commit
 * of the drafts onto the session right after the handlers, for tests that
 * assert branch order. `turnIndex` defaults to 0 so the auto start-anchor
 * (default X=2) stays out of the way.
 */
async function fireTurnEndAll(
  pi: FakePi,
  ctx: FakeCtx,
  message: unknown,
  opts: {
    turnIndex?: number;
    entries?: BoundaryDraft[];
    commit?: boolean;
  } = {},
): Promise<BoundaryResult> {
  const handlers = pi.onCalls.get("turn_end");
  assert.ok(handlers, "factory must register turn_end handlers");
  let entries = opts.entries ?? [];
  let shouldContinue = false;
  for (const handler of handlers) {
    const result = (await handler(
      {
        type: "turn_end",
        turnIndex: opts.turnIndex ?? 0,
        message,
        toolResults: [],
        entries,
        continue: shouldContinue,
      } as never,
      ctx as never,
    )) as { entries?: BoundaryDraft[]; continue?: boolean } | undefined;
    if (result?.entries !== undefined) entries = result.entries;
    if (result?.continue !== undefined) shouldContinue = result.continue;
  }
  const folded = { entries, continue: shouldContinue };
  pi.boundaryResults.push(folded);
  if (opts.commit) {
    for (const draft of entries) {
      ctx.sessionManager.appendCustomMessageEntry(
        draft.customType,
        draft.content,
        draft.display,
        draft.details,
      );
    }
  }
  return folded;
}

/** Fire the `tool_call` gate for a tool call in `ctx`'s session. */
function fireToolCall(
  pi: FakePi,
  ctx: FakeCtx,
  toolName = "bash",
  input: Record<string, unknown> = { command: "true" },
): unknown {
  const handlers = pi.onCalls.get("tool_call");
  assert.ok(handlers, "factory must register a tool_call handler");
  assert.equal(handlers.length, 1);
  return handlers[0](
    { type: "tool_call", toolCallId: "tc-gate", toolName, input } as never,
    ctx as never,
  );
}

const REWIND_ARGS = {
  action: "rewind",
  rewindTo: "start",
  newLabel: "end",
  summaryFocus: "Preserve user instructions and continue.",
} as const;

/**
 * Phase 1 exactly as the agent loop runs it: the model's assistant turn
 * declaring the rewind call is persisted, `execute` returns the summary
 * instruction, pi persists the tool_result, and the turn ends (the handler
 * must skip that turn — it carries the rewind's own toolCallId).
 */
async function requestRewind(
  f: { sm: SessionManager; pi: FakePi; ctx: FakeCtx; tool: CapturedTool },
  args: Record<string, unknown> = REWIND_ARGS,
  toolCallId = "tc-rewind",
) {
  const call = appendToolCallAssistant(f.sm, [
    { id: toolCallId, name: TOOL_NAME, arguments: args },
  ]);
  const result = await f.tool.execute(
    toolCallId,
    args,
    undefined,
    undefined,
    f.ctx,
  );
  appendToolResult(
    f.sm,
    toolCallId,
    TOOL_NAME,
    result.content[0].text,
    result.details,
  );
  await fireTurnEndAll(f.pi, f.ctx, call.message);
  return { result, callId: call.id };
}

/**
 * The model's summary turn: persist the assistant message (as pi does on
 * `message_end`) and fire `turn_end` with it.
 */
async function summaryTurn(
  f: { sm: SessionManager; pi: FakePi; ctx: FakeCtx },
  text: string = SUMMARY_TEXT,
  stopReason = "stop",
  opts: { commit?: boolean } = {},
): Promise<{
  entryId: string;
  message: AssistantTurn;
  boundary: BoundaryResult;
}> {
  const message = assistantMessage([{ type: "text", text }], stopReason, 100);
  const entryId = f.sm.appendMessage(message as never);
  const boundary = await fireTurnEndAll(f.pi, f.ctx, message, opts);
  return { entryId, message, boundary };
}

/**
 * Rewind messages delivered as `turn_end` boundary entries, oldest first,
 * each with its boundary's folded `continue`. Also pins that no rewind
 * message ever goes through `pi.sendMessage`.
 */
function rewindMessages(
  pi: FakePi,
): Array<{ message: BoundaryDraft; continue: boolean }> {
  assert.ok(
    !pi.sendMessageCalls.some(
      (args) =>
        (args[0] as { customType?: string } | undefined)?.customType ===
        REWIND_CUSTOM_TYPE,
    ),
    "rewind messages must be boundary entries, never pi.sendMessage",
  );
  return pi.boundaryResults.flatMap((result) =>
    result.entries
      .filter((draft) => draft.customType === REWIND_CUSTOM_TYPE)
      .map((message) => ({ message, continue: result.continue })),
  );
}

/**
 * Full in-band rewind: canonical fixture, phase 1, summary turn. Returns the
 * fixture plus the phase-1 result and the continuation message.
 */
async function rewindFixture(
  opts: { capture?: boolean; summary?: string; commit?: boolean } = {},
) {
  const f = setup();
  const { fake, anchorId } = setupRewindable(f.sm, f.pi, {
    capture: opts.capture ?? true,
  });
  const oldLeafBeforeCall = f.sm.getLeafId();
  const { result } = await requestRewind(f);
  const leafBeforeSummary = f.sm.getLeafId();
  const summary = await summaryTurn(f, opts.summary ?? SUMMARY_TEXT, "stop", {
    commit: opts.commit,
  });
  const messages = rewindMessages(f.pi);
  return {
    ...f,
    fake,
    anchorId,
    result,
    oldLeafBeforeCall,
    leafBeforeSummary,
    summaryEntryId: summary.entryId,
    done: messages[messages.length - 1],
  };
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
    assert.deepEqual(
      projected,
      sm
        .buildSessionContext()
        .messages.filter((message) => message.role !== "system"),
    );
  });

  it("buildContextMessages is conversation-only (Pi re-attaches the head)", () => {
    // `context` handlers transform conversation messages only; the runner
    // filters system state out before the handler and re-attaches the current
    // head afterwards. Returning a projection that still carries the
    // persisted system entry would duplicate the head whenever
    // `forceSystemPrompt` is unset.
    const { sm } = setup();
    sm.appendMessage({ role: "system", content: "PERSISTED HEAD" } as never);
    appendTurn(sm, "u1", "a1", 100);
    const projected = __testHooks.buildContextMessages(sm);
    assert.ok(
      projected.every((message) => message.role !== "system"),
      "the context payload must not carry system messages",
    );
    assert.ok(
      sm.buildSessionContext().messages.some((m) => m.role === "system"),
      "the raw session context still contains the persisted system entry",
    );
  });

  it("buildContextMessages applies context_edit omissions (projection beats raw entries)", () => {
    const { sm } = setup();
    const t1 = appendTurn(sm, "u1", "a1", 100);
    appendTurn(sm, "u2", "a2", 200);
    // pi's overflow recovery and the fallback provider omit failed attempts
    // with a null-replacement context_edit. The wire projection must drop
    // the message; the raw buildContextEntries walk would reintroduce it.
    sm.appendContextEdit(t1.assistantId, null);
    const projected = __testHooks.buildContextMessages(sm);
    assert.ok(
      !projected.some((message) => JSON.stringify(message).includes("a1")),
      "the omitted assistant must not be reintroduced",
    );
    assert.ok(
      sm.buildContextEntries().some((entry) => entry.id === t1.assistantId),
      "the raw entry walk still contains the omitted entry",
    );
    assert.equal(projected.length, sm.buildSessionContext().messages.length);
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

  it("post-rewind: handler returns anchor chain + branch_summary + continuation, none of the abandoned segment", async () => {
    // The headline invariant (#80 shape): after the summary turn lands, the
    // next LLM call (the very next context event) must see the chain up to
    // the anchor, the branch_summary, and the continuation message — and
    // none of the collapsed turns, the rewind tool call, its instruction
    // result, or the summary turn itself (all on the abandoned branch).
    // `commit`: pi persists the boundary entry as a custom_message entry
    // right after the turn_end handlers, before the next LLM call.
    const f = await rewindFixture({ commit: true });
    assert.ok(f.done, "the rewind must deliver its continuation message");

    const handlers = f.pi.onCalls.get("context");
    assert.ok(handlers, "factory must register a context handler");
    const outcome = handlers[0](
      { type: "context", messages: [] } as never,
      { sessionManager: f.sm } as never,
    ) as {
      messages: Array<{
        role: string;
        customType?: string;
        summary?: string;
        content?: unknown;
      }>;
    };
    const messages = outcome.messages;
    const json = JSON.stringify(messages);

    // Ends with branch_summary then the continuation.
    const last = messages[messages.length - 1];
    const prev = messages[messages.length - 2];
    assert.equal(prev.role, "branchSummary");
    assert.equal(
      prev.summary,
      buildStoredSummary(SUMMARY_TEXT, [], []),
      "the projected summary is the stored text",
    );
    assert.equal(last.role, "custom");
    assert.equal(last.customType, REWIND_CUSTOM_TYPE);
    assert.equal(last.content, f.done.message.content);

    // Anchor turn retained; collapsed turns, the rewind call, its result
    // and the summary turn are gone.
    assert.ok(json.includes('"a1"'), "the anchor turn is retained");
    for (const gone of ['"a2"', '"a3"', '"a4"', "tc-rewind"]) {
      assert.ok(
        !json.includes(gone),
        `${gone} must be on the abandoned branch`,
      );
    }
    assert.ok(
      !messages.some(
        (m) => m.role === "assistant" && JSON.stringify(m).includes("## Goal"),
      ),
      "the summary turn itself is not replayed as an assistant message",
    );
    assert.ok(
      !messages.some((m) => m.role === "toolResult"),
      "no tool_result survives on the rewound chain",
    );

    // idempotent: a second invocation with no tree change returns the same.
    const again = handlers[0](
      { type: "context", messages: [] } as never,
      { sessionManager: f.sm } as never,
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

\`name\` (anchor) and \`newLabel\` (rewind) write into one shared anchor namespace: labels are unique on the active branch — both actions refuse a name that already exists — and everything written there is addressable as a future \`rewindTo\`. Avoid the reserved \`anchor:\` prefix.`;

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
// factory registration
// =============================================================================

describe("factory registration", () => {
  it("registers with the single pi argument and `list` succeeds", async () => {
    // Production callers (pi's extension loader) pass a single arg; the
    // in-band rewind (#80) has no injectable summarizer.
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

  it("registers the rewind summary turn_end handler BEFORE the hint/start-anchor handler", () => {
    // Order matters: a completed rewind must already be on the branch when
    // the hint handler reads context usage. The rewind handler is async;
    // the hint handler is synchronous by design.
    const { pi } = setup();
    const handlers = pi.onCalls.get("turn_end");
    assert.ok(handlers);
    assert.equal(handlers.length, 2);
    assert.equal(handlers[0].constructor.name, "AsyncFunction");
    assert.notEqual(handlers[1].constructor.name, "AsyncFunction");
    assert.equal(pi.onCalls.get("tool_call")?.length, 1);
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

    // Rewind site: same conditions — the bootstrap-missing footer on the
    // continuation message (#80: written once the summary turn lands) must
    // contain the rewind-specific constant verbatim, NOT the list one.
    {
      const { done } = await rewindFixture({ capture: false });
      assert.ok(
        done.message.content.includes(rewindSentinel),
        "rewind continuation must include REFLECTION_BOOTSTRAP_WARNING_REWIND verbatim",
      );
      assert.ok(
        !done.message.content.includes(listSentinel),
        "rewind continuation must NOT include the list-specific phrasing",
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

  it("estimate throws before the label write: the error propagates and no label is written (atomic anchor)", async () => {
    const { sm, pi, tool, ctx } = setup();
    const t = appendTurn(sm, "u1", "a1");

    // Force the token estimate to throw. The old ordering wrote the label
    // first, so a throw here still left `anchor:impl-start` on disk (the
    // tool reported an error, and a retry with the same name was rejected
    // as a duplicate). Compute-before-write is the fix this pins.
    const origBuild = sm.buildSessionContext.bind(sm);
    (
      sm as { buildSessionContext: typeof sm.buildSessionContext }
    ).buildSessionContext = (() => {
      throw new Error("anchor estimate boom");
    }) as typeof sm.buildSessionContext;

    let thrown: unknown;
    try {
      await tool.execute(
        "tc-1",
        { action: "anchor", name: "impl-start" },
        undefined,
        undefined,
        ctx,
      );
    } catch (e) {
      thrown = e;
    } finally {
      (
        sm as { buildSessionContext: typeof sm.buildSessionContext }
      ).buildSessionContext = origBuild;
    }

    assert.ok(thrown, "the estimate throw must propagate");
    assert.match(
      thrown instanceof Error ? thrown.message : String(thrown),
      /anchor estimate boom/,
    );
    assert.equal(
      pi.setLabelCalls.length,
      0,
      "anchor must not write its label before the estimate succeeds",
    );
    assert.equal(sm.getLabel(t.assistantId), undefined);
  });

  it("duplicate refusal: re-anchoring an existing name is refused with no write and no prior-clear", async () => {
    // Issue #55 (C8): labels are unique on the active branch and immutable
    // once written. Re-anchoring the same name must be refused BEFORE any
    // write; the prior label stays exactly where it was (retired
    // move-on-collision).
    const { sm, pi, tool, ctx } = setup();
    const t1 = appendTurn(sm, "u1", "a1");
    const first = await tool.execute(
      "tc-1",
      { action: "anchor", name: "iter-start" },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(first.isError, undefined);
    assert.equal(sm.getLabel(t1.assistantId), "anchor:iter-start");
    const setLabelCountBefore = pi.setLabelCalls.length;

    const t2 = appendTurn(sm, "u2", "a2");
    const result = await tool.execute(
      "tc-2",
      { action: "anchor", name: "iter-start" },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /already exists on the active branch/);
    assert.match(result.content[0].text, /labels are unique/);
    assert.match(result.content[0].text, /action='list'/);
    assert.match(result.content[0].text, /fresh name/);
    assert.equal(result.details.refusal, true);
    // No write at all: no set, no clear, and the prior label and the new
    // leaf are untouched.
    assert.equal(
      pi.setLabelCalls.length,
      setLabelCountBefore,
      "a refused anchor must not call setLabel",
    );
    assert.equal(sm.getLabel(t1.assistantId), "anchor:iter-start");
    assert.equal(sm.getLabel(t2.assistantId), undefined);
    assert.equal(result.details.movedFromPriorEntry, undefined);
  });

  it("refusal copy names the colliding name and points at list + a fresh name", async () => {
    const { sm, pi, tool, ctx } = setup();
    appendTurn(sm, "u1", "a1");
    await tool.execute(
      "tc-1",
      { action: "anchor", name: "phase1" },
      undefined,
      undefined,
      ctx,
    );
    appendTurn(sm, "u2", "a2");
    const result = await tool.execute(
      "tc-2",
      { action: "anchor", name: "phase1" },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, true);
    const text = result.content[0].text;
    // Byte-exact copy pin (same string the source builds).
    assert.equal(
      text,
      "A label 'phase1' already exists on the active branch — labels are unique and immutable once written. Use action='list' to review the active labels, then pick a fresh name.",
    );
    void pi;
  });

  it("manual /tree duplicate labels remain possible and resolve leaf-most", async () => {
    // Issue #55 (C8): the tool never creates duplicates, but a manual
    // `/tree` label can put the same name on two entries. `findLabeledEntry`
    // walks leaf→root, so `rewindTo` resolves to the leaf-most match —
    // documented and accepted.
    const { sm, pi } = setup();
    const t1 = appendTurn(sm, "u1", "a1");
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    const t2 = appendTurn(sm, "u2", "a2");
    pi.pi.setLabel(t2.assistantId, "anchor:start");
    assert.equal(
      __testHooks.findLabeledEntry(sm, "anchor:start"),
      t2.assistantId,
      "duplicate labels must resolve leaf-most",
    );
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
      content: [
        { type: "text", text: "Rewinding to 'start' would free only…" },
      ],
      // Shape produced by `toolError`.
      details: { refusal: true, rejected: "min-savings" },
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
// dispatch: rewind phase 1 — the summary request (#80)
//
// The rewind tool call no longer summarizes or mutates anything: its result
// IS the r5d summary instruction, and the live model writes the summary as
// its next turn. These pins cover the instruction text, the `{first}`
// numbering, and the no-mutation contract.
// =============================================================================

describe("dispatch: rewind phase 1 (summary request)", () => {
  it("returns exactly buildSummaryInstruction(first, focus) and the request details", async () => {
    const { sm, pi, tool, ctx } = setup();
    const { anchorId } = setupRewindable(sm, pi);
    appendToolCallAssistant(sm, [{ id: "tc-rewind", name: TOOL_NAME }]);
    const result = await tool.execute(
      "tc-rewind",
      REWIND_ARGS,
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assert.equal(result.content.length, 1);
    // u1 + a1 are background → the collapsed segment starts at message 3.
    assert.equal(
      result.content[0].text,
      buildSummaryInstruction(3, REWIND_ARGS.summaryFocus),
    );
    assert.deepEqual(result.details, {
      status: "summary-requested",
      rewindTo: "start",
      newLabel: "end",
      targetId: anchorId,
      first: 3,
      contextBefore: result.details.contextBefore,
      contextWindow: 1_000_000,
    });
    assert.equal(typeof result.details.contextBefore, "number");
  });

  it("does not mutate the tree, send a message, or write a label", async () => {
    const { sm, pi, tool, ctx } = setup();
    setupRewindable(sm, pi, { capture: true });
    appendToolCallAssistant(sm, [{ id: "tc-rewind", name: TOOL_NAME }]);
    const entriesBefore = sm.getEntries().length;
    const leafBefore = sm.getLeafId();
    const labelCallsBefore = pi.setLabelCalls.length;
    const result = await tool.execute(
      "tc-rewind",
      REWIND_ARGS,
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assert.equal(sm.getLeafId(), leafBefore);
    assert.equal(sm.getEntries().length, entriesBefore);
    assert.ok(
      !sm.getEntries().some((e) => e.type === "branch_summary"),
      "phase 1 must not branch",
    );
    assert.equal(pi.setLabelCalls.length, labelCallsBefore);
    assert.equal(pi.sendMessageCalls.length, 0);
  });

  it("needs no model or auth: contextWindow falls back to 0", async () => {
    const { sm, pi, tool, ctx } = setup({ noModel: true });
    setupRewindable(sm, pi);
    const result = await tool.execute(
      "tc-rewind",
      REWIND_ARGS,
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assert.equal(result.details.status, "summary-requested");
    assert.equal(result.details.contextWindow, 0);
  });

  it("interpolates the focus verbatim (no truncation, no {first} left over)", async () => {
    const { sm, pi, tool, ctx } = setup();
    setupRewindable(sm, pi);
    const focus = `user said: ship it. ${"detail ".repeat(400)}`.trim();
    const result = await tool.execute(
      "tc-rewind",
      { ...REWIND_ARGS, summaryFocus: focus },
      undefined,
      undefined,
      ctx,
    );
    assert.ok(result.content[0].text.endsWith(`Additional focus: ${focus}`));
    assert.doesNotMatch(result.content[0].text, /\{first\}/);
  });

  it("numbers {first} from the context-edited projection, not the raw entry walk", async () => {
    const { sm, pi, tool, ctx } = setup();
    // A failed attempt BEFORE the anchor, omitted by pi / the fallback
    // provider via a null-replacement context_edit. The raw entry walk
    // counts it as background and would report `{first}=4`; the projection
    // the model sees drops it, so the branch start stays message 3.
    const failedId = sm.appendMessage(
      assistantMessage([{ type: "text", text: "partial" }], "error") as never,
    );
    sm.appendContextEdit(failedId, null);
    setupRewindable(sm, pi);
    const result = await tool.execute(
      "tc-rewind",
      REWIND_ARGS,
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.details.first, 3);
    assert.match(result.content[0].text, /^Summarize only messages 3 onwards/);
  });

  it("numbers {first} past provider-invisible empty assistants (aborted attempt)", async () => {
    const { sm, pi, tool, ctx } = setup();
    // A persisted assistant with no text/thinking/toolCall blocks is dropped
    // by the provider serializer, so it must not consume a `{first}` number.
    sm.appendMessage(assistantMessage([], "aborted") as never);
    setupRewindable(sm, pi);
    const result = await tool.execute(
      "tc-rewind",
      REWIND_ARGS,
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.details.first, 3);
  });

  it("counts visible background before the anchor (system messages excluded)", async () => {
    const { sm, pi, tool, ctx } = setup();
    sm.appendMessage({ role: "system", content: "PERSISTED HEAD" } as never);
    appendTurn(sm, "u0", "a0", 1_000);
    setupRewindable(sm, pi);
    const result = await tool.execute(
      "tc-rewind",
      REWIND_ARGS,
      undefined,
      undefined,
      ctx,
    );
    // u0, a0, u1, a1 → first = 5; the system head is not numbered.
    assert.equal(result.details.first, 5);
  });

  it("numbers from 1 when a compaction cut drops the target from the projection", async () => {
    const { sm, pi, tool, ctx } = setup();
    setupRewindable(sm, pi);
    // Keep only from u3 onwards: the anchor (a1) is older than the cut, so
    // every message the summary turn sees is collapsed evidence.
    const kept = sm
      .getBranch()
      .find(
        (e) =>
          e.type === "message" &&
          e.message.role === "user" &&
          JSON.stringify(e.message.content).includes('"u3"'),
      );
    assert.ok(kept);
    sm.appendCompaction("COMPACTED", kept.id, 20_000);
    appendTurn(sm, "u5", "a5", 30_000);
    const result = await tool.execute(
      "tc-rewind",
      REWIND_ARGS,
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assert.equal(result.details.first, 1);
    assert.match(result.content[0].text, /^Summarize only messages 1 onwards/);
  });

  it("countBackgroundMessages stops at the target entry", () => {
    const projected = [
      {
        sourceEntry: { id: "x" },
        messages: [
          { role: "user", content: "a", timestamp: 0 },
          assistantMessage([{ type: "text", text: "b" }], "stop"),
        ],
      },
      {
        sourceEntry: { id: "target" },
        messages: [{ role: "user", content: "c", timestamp: 0 }],
      },
      {
        sourceEntry: { id: "after" },
        messages: [{ role: "user", content: "d", timestamp: 0 }],
      },
    ];
    assert.equal(
      __testHooks.countBackgroundMessages(projected as never, "target"),
      3,
    );
  });

  it("a refused rewind leaves nothing pending (the gate stays open)", async () => {
    const { sm, pi, tool, ctx } = setup();
    appendTurn(sm, "u", "a");
    const result = await tool.execute(
      "tc-rewind",
      REWIND_ARGS,
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, true);
    assert.equal(fireToolCall(pi, ctx), undefined);
  });
});

// =============================================================================
// rewind: tool_call gate while the summary is pending (#80)
// =============================================================================

describe("rewind: tool_call gate while pending", () => {
  it("does not block before any rewind", () => {
    const { pi, ctx } = setup();
    assert.equal(fireToolCall(pi, ctx), undefined);
  });

  it("blocks every tool call (including navigate_tree) while the summary is pending", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi);
    await requestRewind(f);
    const expected = { block: true, reason: SUMMARY_PENDING_BLOCK_REASON };
    assert.deepEqual(fireToolCall(f.pi, f.ctx), expected);
    assert.deepEqual(
      fireToolCall(f.pi, f.ctx, TOOL_NAME, { action: "list" }),
      expected,
    );
    assert.match(SUMMARY_PENDING_BLOCK_REASON, /starting with '## Goal'/);
  });

  it("does not block another session's tool calls", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi);
    await requestRewind(f);
    const otherSm = SessionManager.inMemory("/tmp");
    assert.notEqual(otherSm.getSessionId(), f.sm.getSessionId());
    assert.equal(fireToolCall(f.pi, makeCtx(otherSm)), undefined);
  });

  it("re-opens once the rewind completes", async () => {
    const f = await rewindFixture();
    assert.equal(f.done.message.details.status, "done");
    assert.equal(fireToolCall(f.pi, f.ctx), undefined);
  });

  it("fails open when the session id cannot be read", () => {
    const { pi } = setup();
    const ctx = {
      get sessionManager(): SessionManager {
        throw new Error("stale ctx");
      },
    } as unknown as FakeCtx;
    assert.equal(fireToolCall(pi, ctx), undefined);
  });
});

// =============================================================================
// rewind phase 2 — completion on the summary turn's turn_end (#80)
// =============================================================================

describe("rewind phase 2: completion", () => {
  it("skips the phase-1 turn (it carries the rewind's own toolCallId)", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi);
    await requestRewind(f);
    assert.equal(rewindMessages(f.pi).length, 0);
    assert.ok(!f.sm.getEntries().some((e) => e.type === "branch_summary"));
    // Still pending, and no attempt was spent: the summary turn completes
    // on the first try.
    assert.ok(fireToolCall(f.pi, f.ctx));
    await summaryTurn(f);
    assert.equal(rewindMessages(f.pi)[0].message.details.summaryAttempts, 1);
  });

  it("branches from the anchor with preamble + summary, labels it, and moves the leaf", async () => {
    const f = await rewindFixture();
    const summaryId = f.done.message.details.summaryId as string;
    const entry = f.sm.getEntry(summaryId);
    assert.ok(entry && entry.type === "branch_summary");
    assert.equal(entry.parentId, f.anchorId, "child of the anchor entry");
    assert.equal(entry.summary, `${BRANCH_SUMMARY_PREAMBLE}${SUMMARY_TEXT}`);
    assert.deepEqual(entry.details, { readFiles: [], modifiedFiles: [] });
    // Extension-provided summary, as pi's navigateTree marks it.
    assert.equal(entry.fromHook, true);
    assert.equal(f.sm.getLabel(summaryId), "anchor:end");
    // The leaf is the label-change entry `setLabel` appends on the summary
    // (same shape as pi's own `/tree` label write).
    const leaf = f.sm.getEntry(f.sm.getLeafId() as string);
    assert.ok(leaf && leaf.type === "label");
    assert.equal(leaf.parentId, summaryId, "the leaf moved onto the summary");
    assert.notEqual(f.leafBeforeSummary, summaryId);
    // The abandoned branch keeps the rewind call, its result and the
    // summary turn.
    const summaryTurnEntry = f.sm.getEntry(f.summaryEntryId);
    assert.ok(summaryTurnEntry);
    assert.ok(!f.sm.getBranch().some((e) => e.id === f.summaryEntryId));
  });

  it("delivers the continuation as a boundary entry: buildRewindDoneText content, continue, done details", async () => {
    const f = await rewindFixture();
    const calls = rewindMessages(f.pi);
    assert.equal(calls.length, 1);
    const { message } = calls[0];
    assert.equal(calls[0].continue, true);
    assert.equal(message.type, "custom_message");
    assert.equal(message.customType, REWIND_CUSTOM_TYPE);
    assert.equal(message.display, true);
    const d = message.details;
    assert.equal(
      message.content,
      buildRewindDoneText(
        {
          rewindTo: "start",
          newLabel: "end",
          beforeTokens: f.result.details.contextBefore as number,
          contextWindow: 1_000_000,
        },
        d.contextAfter as number,
      ),
    );
    assert.deepEqual(d, {
      status: "done",
      rewindTo: "start",
      newLabel: "end",
      targetId: f.anchorId,
      summaryId: d.summaryId,
      collapsedEntries: d.collapsedEntries,
      contextBefore: f.result.details.contextBefore,
      contextAfter: d.contextAfter,
      contextWindow: 1_000_000,
      summaryAttempts: 1,
      agentMessagesRefreshed: true,
      historyRewriteNotified: false,
      readFiles: [],
      modifiedFiles: [],
      reappendedLateMessages: 0,
    });
    // Collapsed: the anchor's label-change entry, u2..a4 (6), the rewind
    // call, its result, and the summary turn — all left on the abandoned
    // branch.
    assert.equal(d.collapsedEntries, 10);
    assert.ok(
      (d.contextBefore as number) > (d.contextAfter as number),
      "the rewound chain must be smaller",
    );
  });

  it("continuation prose: [rewind] header + pending-work guidance", async () => {
    const f = await rewindFixture();
    const text = f.done.message.content;
    assert.match(text, /^\[rewind to 'start' · collapsed as 'end'\] · /);
    assert.match(text, /### Done/);
    assert.match(text, /### In Progress/);
    assert.match(text, /### Blocked/);
    assert.match(text, /## Next Steps/);
    assert.match(text, /are pending/);
    assert.match(text, /It is the newest branch_summary, the current state/);
    assert.match(text, /continuing the latest branch_summary's work/);
    assert.match(text, /User messages retained above predate this rewind/);
    assert.match(text, /do not re-answer them/);
  });

  it("refreshes agent.state.messages to the rewound chain", async () => {
    const f = await rewindFixture();
    assert.ok(f.fake);
    assert.equal(f.done.message.details.agentMessagesRefreshed, true);
    assert.deepEqual(
      f.fake.agent.state.messages,
      f.sm.buildSessionContext().messages,
    );
  });

  it("bootstrap-missing: continuation carries the rewind warning and refreshed=false", async () => {
    const f = await rewindFixture({ capture: false });
    assert.equal(f.done.message.details.status, "done");
    assert.equal(f.done.message.details.agentMessagesRefreshed, false);
    assert.ok(
      f.done.message.content.endsWith(
        `\n\n${__testHooks.REFLECTION_BOOTSTRAP_WARNING_REWIND}`,
      ),
    );
  });

  it("strips any preamble before '## Goal' from the stored summary", async () => {
    const f = await rewindFixture({
      summary: `Sure — here is the summary.\n\n${SUMMARY_TEXT}\n\n`,
    });
    const entry = f.sm.getEntry(f.done.message.details.summaryId as string);
    assert.ok(entry && entry.type === "branch_summary");
    assert.equal(entry.summary, buildStoredSummary(SUMMARY_TEXT, [], []));
  });

  it("appends read/modified file tags from read/edit/write calls in the collapsed segment", async () => {
    const f = setup();
    const t1 = appendTurn(f.sm, "u1", "a1", 6_000);
    f.pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendToolCallAssistant(
      f.sm,
      [
        { id: "r1", name: "read", arguments: { path: "src/a.ts" } },
        { id: "r2", name: "read", arguments: { path: "src/b.ts" } },
        { id: "e1", name: "edit", arguments: { path: "src/b.ts" } },
        { id: "w1", name: "write", arguments: { path: "src/c.ts" } },
      ],
      12_000,
    );
    for (const id of ["r1", "r2", "e1", "w1"]) {
      appendToolResult(f.sm, id, "read", "ok");
    }
    appendTurn(f.sm, "u3", "a3", 24_000);
    await requestRewind(f);
    await summaryTurn(f);
    const { details } = rewindMessages(f.pi)[0].message;
    assert.deepEqual(details.readFiles, ["src/a.ts"]);
    assert.deepEqual(details.modifiedFiles, ["src/b.ts", "src/c.ts"]);
    const entry = f.sm.getEntry(details.summaryId as string);
    assert.ok(entry && entry.type === "branch_summary");
    assert.equal(
      entry.summary,
      `${BRANCH_SUMMARY_PREAMBLE}${SUMMARY_TEXT}\n\n<read-files>\nsrc/a.ts\n</read-files>\n\n<modified-files>\nsrc/b.ts\nsrc/c.ts\n</modified-files>`,
    );
    assert.deepEqual(entry.details, {
      readFiles: ["src/a.ts"],
      modifiedFiles: ["src/b.ts", "src/c.ts"],
    });
  });

  it("emits session_tree after the rewind so history mirrors rebuild (#79)", async () => {
    // pi-claude-bridge mirrors the conversation in a Claude Code session and
    // rebuilds it only on `session_tree` / `session_compact`. A tool-driven
    // rewind must emit the same event pi's own navigateTree emits, AFTER the
    // branch and the agent-state refresh, so a handler reading the session
    // sees the rewound chain.
    const f = setup();
    setupRewindable(f.sm, f.pi);
    const fake = makeFakeSession(f.sm);
    const events: Array<Record<string, unknown>> = [];
    let messagesAtEmit: unknown[] | undefined;
    let deliveredBeforeEmit = -1;
    fake._extensionRunner = {
      emit: async (event) => {
        events.push(event as Record<string, unknown>);
        messagesAtEmit = fake.agent.state.messages;
        deliveredBeforeEmit = rewindMessages(f.pi).length;
      },
    };
    __testHooks.captureSession(fake as unknown as AgentSession);
    await requestRewind(f);
    assert.equal(events.length, 0, "phase 1 must not emit");
    const { entryId } = await summaryTurn(f);

    const done = rewindMessages(f.pi)[0].message;
    assert.equal(done.details.historyRewriteNotified, true);
    assert.equal(events.length, 1);
    const event = events[0];
    assert.equal(event.type, "session_tree");
    assert.equal(event.oldLeafId, entryId, "old leaf = the summary turn");
    assert.equal(event.newLeafId, f.sm.getLeafId());
    assert.equal(
      f.sm.getEntry(event.newLeafId as string)?.parentId,
      done.details.summaryId,
      "new leaf = the label entry on the summary",
    );
    assert.equal(
      (event.summaryEntry as { id?: string } | undefined)?.id,
      done.details.summaryId,
    );
    assert.equal(event.fromExtension, true);
    assert.deepEqual(messagesAtEmit, f.sm.buildSessionContext().messages);
    assert.equal(
      deliveredBeforeEmit,
      0,
      "emit precedes the continuation message",
    );
  });

  it("reports historyRewriteNotified=false without a reachable runner (#79)", async () => {
    const f = await rewindFixture();
    assert.equal(f.done.message.details.agentMessagesRefreshed, true);
    assert.equal(f.done.message.details.historyRewriteNotified, false);
  });

  it("a throwing session_tree handler does not fail the rewind (#79)", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi);
    const fake = makeFakeSession(f.sm);
    fake._extensionRunner = {
      emit: async () => {
        throw new Error("handler boom");
      },
    };
    __testHooks.captureSession(fake as unknown as AgentSession);
    await requestRewind(f);
    await summaryTurn(f);
    const done = rewindMessages(f.pi)[0].message;
    assert.equal(done.details.status, "done");
    assert.equal(done.details.historyRewriteNotified, false);
    assert.match(done.content, /^\[rewind to 'start'/);
  });

  it("a throwing pi.sendMessage cannot affect delivery (rewind messages never use it)", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    (
      f.pi.pi as unknown as { sendMessage: (...args: unknown[]) => void }
    ).sendMessage = () => {
      throw new Error("stale session runtime");
    };
    await requestRewind(f);
    const { boundary } = await summaryTurn(f, SUMMARY_TEXT, "stop", {
      commit: true,
    });
    assert.equal(boundary.continue, true);
    assert.equal(boundary.entries.length, 1);
    assert.equal(boundary.entries[0].details.status, "done");
    // Committed by pi (replicated by `commit`) after the branch_summary.
    const leaf = f.sm.getEntry(f.sm.getLeafId() as string);
    assert.ok(leaf && leaf.type === "custom_message");
    assert.equal(leaf.customType, REWIND_CUSTOM_TYPE);
    assert.equal(leaf.display, true);
    assert.match(String(leaf.content), /^\[rewind to 'start'/);
    assert.ok(
      f.sm.getBranch().some((e) => e.type === "branch_summary"),
      "the continuation follows the branch_summary on the active branch",
    );
  });

  it("setLabel throw falls back to appendLabelChange for the new label", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    await requestRewind(f);
    (f.pi.pi as { setLabel: typeof f.pi.pi.setLabel }).setLabel = () => {
      throw new Error("stale session runtime");
    };
    await summaryTurn(f);
    const done = rewindMessages(f.pi)[0].message;
    assert.equal(done.details.status, "done");
    assert.equal(f.sm.getLabel(done.details.summaryId as string), "anchor:end");
  });

  it("chained rewinds: a→b then b→c keeps prior labels and re-anchors the new summary", async () => {
    const f = setup();
    const tA = appendTurn(f.sm, "u1", "a1", 6_000);
    f.pi.pi.setLabel(tA.assistantId, "anchor:a");
    appendTurn(f.sm, "u2", "a2", 12_000);
    appendTurn(f.sm, "u3", "a3", 18_000);
    __testHooks.captureSession(
      makeFakeSession(f.sm) as unknown as AgentSession,
    );

    await requestRewind(f, { ...REWIND_ARGS, rewindTo: "a", newLabel: "b" });
    await summaryTurn(f, SUMMARY_TEXT, "stop", { commit: true });
    const first = rewindMessages(f.pi)[0].message;
    assert.equal(first.details.status, "done");
    const sumB = first.details.summaryId as string;

    appendTurn(f.sm, "u4", "a4", 24_000);
    appendTurn(f.sm, "u5", "a5", 30_000);
    const { result: r2 } = await requestRewind(
      f,
      {
        ...REWIND_ARGS,
        rewindTo: "b",
        newLabel: "c",
        summaryFocus: "Preserve user instructions across the second rewind.",
      },
      "tc-2",
    );
    assert.equal(r2.details.targetId, sumB);
    await summaryTurn(f);
    const second = rewindMessages(f.pi)[1].message;
    assert.equal(second.details.status, "done");
    const sumC = second.details.summaryId as string;
    assert.equal(f.sm.getEntry(sumC)?.parentId, sumB);
    assert.equal(f.sm.getLabel(sumC), "anchor:c");
    assert.equal(
      f.sm.getLabel(sumB),
      "anchor:b",
      "the b-summary's anchor:b label must survive the second rewind",
    );
    assert.equal(f.sm.getLabel(tA.assistantId), "anchor:a");
    assert.equal(f.sm.getEntry(f.sm.getLeafId() as string)?.parentId, sumC);
  });

  it("immediate re-rewind of a just-landed summary is refused and leaves nothing pending", async () => {
    const f = await rewindFixture();
    const { result } = await requestRewind(
      f,
      { ...REWIND_ARGS, rewindTo: "end", newLabel: "again" },
      "tc-2",
    );
    assert.equal(result.isError, true);
    assert.equal(result.details.rejected, "min-savings");
    assert.equal(fireToolCall(f.pi, f.ctx), undefined);
  });

  it("never calls pi.sendMessage / sendUserMessage for rewind messages", async () => {
    const f = await rewindFixture();
    assert.equal(f.done.message.details.status, "done");
    assert.deepEqual(f.pi.sendMessageCalls, []);
    assert.deepEqual(f.pi.sendUserMessageCalls, []);
  });

  it("keeps entries from a handler earlier in the chain: [foreign, ours]", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    await requestRewind(f);
    const foreign: BoundaryDraft = {
      type: "custom_message",
      customType: "other-extension",
      content: "queued by another extension",
      display: false,
      details: { from: "other" },
    };
    const message = assistantMessage(
      [{ type: "text", text: SUMMARY_TEXT }],
      "stop",
      100,
    );
    f.sm.appendMessage(message as never);
    const boundary = await fireTurnEndAll(f.pi, f.ctx, message, {
      entries: [foreign],
    });
    assert.equal(boundary.continue, true);
    assert.equal(boundary.entries.length, 2);
    assert.equal(boundary.entries[0], foreign, "the foreign draft, untouched");
    assert.equal(boundary.entries[1].customType, REWIND_CUSTOM_TYPE);
    assert.equal(boundary.entries[1].details.status, "done");
  });

  it("drops earlier context_edit drafts the rewind made stale, keeps the rest (compactions included)", async () => {
    // pi validates the whole boundary list in a preview and drops ALL of it
    // when one draft fails; a draft targeting an entry the rewind abandoned
    // would take the continuation down with it.
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    const anchorId = f.sm
      .getBranch()
      .find((e) => f.sm.getLabel(e.id) === "anchor:start")?.id;
    assert.ok(anchorId);
    await requestRewind(f);
    const message = assistantMessage(
      [{ type: "text", text: SUMMARY_TEXT }],
      "stop",
      100,
    );
    const summaryTurnId = f.sm.appendMessage(message as never);
    const staleEdit = {
      type: "context_edit",
      targetId: summaryTurnId,
      replacement: { content: "x" },
    };
    // pi never validates a compaction's firstKeptEntryId, so compaction
    // drafts pass through untouched — null (self-retaining) and stale ids.
    const staleCompaction = {
      type: "compaction",
      summary: "s",
      firstKeptEntryId: summaryTurnId,
    };
    const selfCompaction = {
      type: "compaction",
      summary: "s",
      firstKeptEntryId: null,
    };
    const liveEdit = {
      type: "context_edit",
      targetId: anchorId,
      replacement: { content: "y" },
    };
    const foreign = {
      type: "custom_message",
      customType: "other-extension",
      content: "kept",
      display: true,
      details: {},
    };
    const boundary = await fireTurnEndAll(f.pi, f.ctx, message, {
      entries: [
        staleEdit,
        staleCompaction,
        selfCompaction,
        liveEdit,
        foreign,
      ] as never,
    });
    assert.equal(boundary.continue, true);
    assert.deepEqual(
      boundary.entries.slice(0, -1),
      [staleCompaction, selfCompaction, liveEdit, foreign],
      "only the stale context_edit dropped, the rest kept in order",
    );
    assert.equal(boundary.entries.at(-1)?.customType, REWIND_CUSTOM_TYPE);
  });

  it("returns nothing on turns without a pending rewind", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi);
    const handlers = f.pi.onCalls.get("turn_end");
    assert.ok(handlers);
    const result = await handlers[0](
      {
        type: "turn_end",
        turnIndex: 0,
        message: assistantMessage(
          [{ type: "text", text: SUMMARY_TEXT }],
          "stop",
        ),
        toolResults: [],
        entries: [],
        continue: false,
      } as never,
      f.ctx as never,
    );
    assert.equal(result, undefined);
  });

  it("ignores turn_end from another session and non-assistant messages", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi);
    await requestRewind(f);
    const otherSm = SessionManager.inMemory("/tmp");
    await fireTurnEndAll(
      f.pi,
      makeCtx(otherSm),
      assistantMessage([{ type: "text", text: SUMMARY_TEXT }], "stop"),
    );
    await fireTurnEndAll(f.pi, f.ctx, {
      role: "user",
      content: [{ type: "text", text: SUMMARY_TEXT }],
    });
    await fireTurnEndAll(f.pi, f.ctx, undefined);
    assert.equal(rewindMessages(f.pi).length, 0);
    assert.ok(fireToolCall(f.pi, f.ctx), "still pending");
    await summaryTurn(f);
    assert.equal(rewindMessages(f.pi)[0].message.details.summaryAttempts, 1);
  });
});

// =============================================================================
// rewind phase 2 — failed summary turns, cancellation, abort (#80)
// =============================================================================

describe("rewind phase 2: failures", () => {
  async function pending() {
    const f = setup();
    const { anchorId } = setupRewindable(f.sm, f.pi, { capture: true });
    await requestRewind(f);
    return { ...f, anchorId };
  }

  function assertNotBranched(sm: SessionManager): void {
    assert.ok(
      !sm.getEntries().some((e) => e.type === "branch_summary"),
      "no branch_summary may land",
    );
  }

  /** A failed summary turn that calls a tool instead of summarizing. */
  async function toolCallTurn(
    f: Awaited<ReturnType<typeof pending>>,
    id: string,
  ): Promise<void> {
    const { message } = appendToolCallAssistant(f.sm, [
      { id, name: "bash", arguments: { command: "ls" } },
    ]);
    await fireTurnEndAll(f.pi, f.ctx, message);
  }

  it("a tool-call turn spends an attempt silently (the blocked result re-asks)", async () => {
    const f = await pending();
    await toolCallTurn(f, "tc-bash-1");
    assert.equal(rewindMessages(f.pi).length, 0);
    assertNotBranched(f.sm);
    assert.ok(fireToolCall(f.pi, f.ctx), "still pending");
    await toolCallTurn(f, "tc-bash-2");
    assert.equal(rewindMessages(f.pi).length, 0);
    // Third turn is a valid summary: lands with the attempt count.
    await summaryTurn(f);
    const done = rewindMessages(f.pi)[0].message;
    assert.equal(done.details.status, "done");
    assert.equal(done.details.summaryAttempts, MAX_SUMMARY_ATTEMPTS);
  });

  it("a tool-call failure below the cap returns no boundary result", async () => {
    const f = await pending();
    const { message } = appendToolCallAssistant(f.sm, [
      { id: "tc-bash-1", name: "bash", arguments: { command: "ls" } },
    ]);
    const handlers = f.pi.onCalls.get("turn_end");
    assert.ok(handlers);
    const result = await handlers[0](
      {
        type: "turn_end",
        turnIndex: 0,
        message,
        toolResults: [],
        entries: [],
        continue: false,
      } as never,
      f.ctx as never,
    );
    assert.equal(result, undefined);
    assert.ok(fireToolCall(f.pi, f.ctx), "still pending");
  });

  it("a mixed text + tool-call turn is a failure even with a ## Goal heading", async () => {
    const f = await pending();
    const message = assistantMessage(
      [
        { type: "text", text: SUMMARY_TEXT },
        { type: "toolCall", id: "tc-x", name: "bash", arguments: {} },
      ],
      "toolUse",
    );
    f.sm.appendMessage(message as never);
    await fireTurnEndAll(f.pi, f.ctx, message);
    assertNotBranched(f.sm);
    assert.equal(rewindMessages(f.pi).length, 0);
    assert.ok(fireToolCall(f.pi, f.ctx));
  });

  it("text without '## Goal' sends a retry message", async () => {
    const f = await pending();
    await summaryTurn(f, "Okay, I'll summarize now.");
    const calls = rewindMessages(f.pi);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].continue, true);
    assert.equal(calls[0].message.customType, REWIND_CUSTOM_TYPE);
    assert.equal(calls[0].message.content, buildSummaryRetryText("start"));
    assert.equal(calls[0].message.display, true);
    assert.deepEqual(calls[0].message.details, {
      status: "summary-retry",
      rewindTo: "start",
      newLabel: "end",
      summaryAttempts: 1,
    });
    assertNotBranched(f.sm);
    assert.ok(fireToolCall(f.pi, f.ctx), "still pending");
  });

  it("a truncated summary (stopReason 'length') is a failure", async () => {
    const f = await pending();
    await summaryTurn(f, SUMMARY_TEXT, "length");
    const calls = rewindMessages(f.pi);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].message.details.status, "summary-retry");
    assertNotBranched(f.sm);
  });

  it(`cancels after MAX_SUMMARY_ATTEMPTS (${MAX_SUMMARY_ATTEMPTS}) failures and re-opens the gate`, async () => {
    const f = await pending();
    const leafBefore = f.sm.getLeafId();
    for (let i = 0; i < MAX_SUMMARY_ATTEMPTS; i++) {
      await summaryTurn(f, `attempt ${i}`);
    }
    const calls = rewindMessages(f.pi);
    assert.equal(calls.length, MAX_SUMMARY_ATTEMPTS);
    for (let i = 0; i < MAX_SUMMARY_ATTEMPTS - 1; i++) {
      assert.equal(calls[i].message.details.status, "summary-retry");
      assert.equal(calls[i].message.details.summaryAttempts, i + 1);
    }
    const cancel = calls[MAX_SUMMARY_ATTEMPTS - 1];
    const reason = `no summary after ${MAX_SUMMARY_ATTEMPTS} attempts`;
    assert.equal(cancel.continue, true);
    assert.equal(
      cancel.message.content,
      buildRewindCancelledText("start", reason),
    );
    assert.deepEqual(cancel.message.details, {
      status: "cancelled",
      rewindTo: "start",
      newLabel: "end",
      reason,
      summaryAttempts: MAX_SUMMARY_ATTEMPTS,
    });
    assertNotBranched(f.sm);
    assert.notEqual(f.sm.getLeafId(), leafBefore, "failed turns stay on");
    assert.equal(fireToolCall(f.pi, f.ctx), undefined, "gate re-opened");
    // A later valid summary turn no longer completes anything.
    await summaryTurn(f);
    assert.equal(rewindMessages(f.pi).length, MAX_SUMMARY_ATTEMPTS);
    assertNotBranched(f.sm);
  });

  it("tool-call failures count toward the cap and cancel on the last one", async () => {
    const f = await pending();
    for (let i = 0; i < MAX_SUMMARY_ATTEMPTS; i++) {
      await toolCallTurn(f, `tc-bash-${i}`);
    }
    const calls = rewindMessages(f.pi);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].message.details.status, "cancelled");
    assert.equal(fireToolCall(f.pi, f.ctx), undefined);
  });

  it("an aborted turn drops the rewind silently", async () => {
    const f = await pending();
    await summaryTurn(f, SUMMARY_TEXT, "aborted");
    assert.equal(rewindMessages(f.pi).length, 0);
    assertNotBranched(f.sm);
    assert.equal(fireToolCall(f.pi, f.ctx), undefined);
    await summaryTurn(f);
    assertNotBranched(f.sm);
  });

  it("an error turn keeps the rewind pending without spending an attempt", async () => {
    const f = await pending();
    await summaryTurn(f, "", "error");
    assert.equal(rewindMessages(f.pi).length, 0);
    assertNotBranched(f.sm);
    assert.ok(fireToolCall(f.pi, f.ctx), "still pending after an error");
    await summaryTurn(f);
    const done = rewindMessages(f.pi)[0].message;
    assert.equal(done.details.status, "done");
    assert.equal(done.details.summaryAttempts, 1);
  });

  it("agent_end drops a pending rewind", async () => {
    const f = await pending();
    fireAgentEnd(f.pi, f.ctx);
    assert.equal(fireToolCall(f.pi, f.ctx), undefined);
    await summaryTurn(f);
    assert.equal(rewindMessages(f.pi).length, 0);
    assertNotBranched(f.sm);
  });

  it("session_start clears a pending rewind", async () => {
    const fixture = makeConfigFixture();
    try {
      const f = await pending();
      await runSessionStart(f.pi, f.ctx, fixture, "reload");
      assert.equal(fireToolCall(f.pi, f.ctx), undefined);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("an error turn survives agent_end (pi's auto-retry follows it) and the retry completes", async () => {
    const f = await pending();
    await summaryTurn(f, "", "error");
    fireAgentEnd(f.pi, f.ctx);
    assert.ok(fireToolCall(f.pi, f.ctx), "still pending across agent_end");
    await summaryTurn(f);
    const done = rewindMessages(f.pi)[0].message;
    assert.equal(done.details.status, "done");
    assert.equal(done.details.summaryAttempts, 1);
  });

  it("a non-error turn after an error resets the mark: agent_end drops again", async () => {
    const f = await pending();
    await summaryTurn(f, "", "error");
    await summaryTurn(f, "Okay, summarizing now.");
    assert.equal(
      rewindMessages(f.pi)[0].message.details.status,
      "summary-retry",
    );
    fireAgentEnd(f.pi, f.ctx);
    assert.equal(fireToolCall(f.pi, f.ctx), undefined);
  });

  it("before_agent_start (a new user prompt) drops a pending rewind, even an errored one", async () => {
    const f = await pending();
    await summaryTurn(f, "", "error");
    await fireBeforeAgentStart(f.pi, f.ctx, undefined);
    assert.equal(fireToolCall(f.pi, f.ctx), undefined);
    await summaryTurn(f);
    assert.equal(rewindMessages(f.pi).length, 0);
    assertNotBranched(f.sm);
  });

  it("cancels when a compaction landed after the rewind call", async () => {
    const f = await pending();
    f.sm.appendCompaction("COMPACTED", f.anchorId, 20_000);
    await summaryTurn(f);
    const calls = rewindMessages(f.pi);
    assert.equal(calls.length, 1);
    const reason =
      "the context was compacted while the summary was being written";
    assert.equal(
      calls[0].message.content,
      buildRewindCancelledText("start", reason),
    );
    assert.equal(calls[0].message.details.reason, reason);
    assertNotBranched(f.sm);
    assert.equal(fireToolCall(f.pi, f.ctx), undefined);
  });

  it("cancels when the rewind call is no longer on the active branch", async () => {
    const f = await pending();
    // Move the leaf back above the rewind call (anchor and labels intact).
    const a4 = f.sm
      .getBranch()
      .find(
        (e) =>
          e.type === "message" &&
          e.message.role === "assistant" &&
          JSON.stringify(e.message.content).includes('"a4"'),
      );
    assert.ok(a4);
    f.sm.branch(a4.id);
    await summaryTurn(f);
    const calls = rewindMessages(f.pi);
    assert.equal(calls.length, 1);
    assert.equal(
      calls[0].message.details.reason,
      "the rewind call is no longer on the active branch",
    );
    assertNotBranched(f.sm);
  });

  it("a throwing branchWithSummary cancels: message delivered, pending cleared, gate reopened", async () => {
    const f = await pending();
    (f.sm as unknown as { branchWithSummary: () => string }).branchWithSummary =
      () => {
        throw new Error("disk full");
      };
    const labelCallsBefore = f.pi.setLabelCalls.length;
    await summaryTurn(f);
    const calls = rewindMessages(f.pi);
    assert.equal(calls.length, 1);
    const reason = "the branch could not be written (disk full)";
    assert.equal(calls[0].continue, true);
    assert.equal(
      calls[0].message.content,
      buildRewindCancelledText("start", reason),
    );
    assert.deepEqual(calls[0].message.details, {
      status: "cancelled",
      rewindTo: "start",
      newLabel: "end",
      reason,
      summaryAttempts: 0,
    });
    assertNotBranched(f.sm);
    assert.equal(f.pi.setLabelCalls.length, labelCallsBefore, "no label");
    assert.equal(fireToolCall(f.pi, f.ctx), undefined, "gate reopened");
    await summaryTurn(f);
    assert.equal(rewindMessages(f.pi).length, 1, "nothing left to complete");
  });

  it("cancels when the anchor no longer resolves to the same entry", async () => {
    const f = await pending();
    f.pi.pi.setLabel(f.anchorId, undefined);
    await summaryTurn(f);
    const calls = rewindMessages(f.pi);
    assert.equal(calls.length, 1);
    const reason = "'start' is no longer on the active branch";
    assert.equal(
      calls[0].message.content,
      buildRewindCancelledText("start", reason),
    );
    assert.equal(calls[0].message.details.status, "cancelled");
    assert.equal(calls[0].message.details.reason, reason);
    assertNotBranched(f.sm);
    assert.equal(fireToolCall(f.pi, f.ctx), undefined);
  });

  it("cancels when the anchor label moved to a different entry", async () => {
    const f = await pending();
    const leaf = f.sm.getLeafId() as string;
    f.pi.pi.setLabel(leaf, "anchor:start");
    await summaryTurn(f);
    const calls = rewindMessages(f.pi);
    assert.equal(calls[0].message.details.status, "cancelled");
    assertNotBranched(f.sm);
  });

  it("cancels when newLabel was taken before completion", async () => {
    const f = await pending();
    f.pi.pi.setLabel(f.sm.getLeafId() as string, "anchor:end");
    await summaryTurn(f);
    const calls = rewindMessages(f.pi);
    assert.equal(calls.length, 1);
    const reason = "a label 'end' now exists on the active branch";
    assert.equal(
      calls[0].message.content,
      buildRewindCancelledText("start", reason),
    );
    assert.equal(calls[0].message.details.reason, reason);
    assertNotBranched(f.sm);
  });
});

// =============================================================================
// extractSummaryText
// =============================================================================

describe("extractSummaryText", () => {
  it("returns the text from the '## Goal' heading on, trimmed", () => {
    assert.equal(extractSummaryText(`${SUMMARY_TEXT}\n\n`), SUMMARY_TEXT);
  });

  it("drops a preamble before the heading", () => {
    assert.equal(
      extractSummaryText(`Here you go:\n\n${SUMMARY_TEXT}`),
      SUMMARY_TEXT,
    );
  });

  it("returns null without a line-start '## Goal' heading", () => {
    for (const text of [
      "",
      "Okay, summarizing.",
      "### Goal\nx",
      "## Goals\nx",
      "text ## Goal inline",
      "## goal\nlowercase",
    ]) {
      assert.equal(extractSummaryText(text), null, JSON.stringify(text));
    }
  });

  it("refuses a bare heading line: the body after it must be non-empty", () => {
    assert.equal(extractSummaryText("## Goal"), null);
    assert.equal(extractSummaryText("## Goal: ship"), null);
    assert.equal(extractSummaryText("preamble\n## Goal   \n\n  "), null);
    assert.equal(extractSummaryText("## Goal\nx"), "## Goal\nx");
    assert.equal(extractSummaryText("## Goal: ship\nx"), "## Goal: ship\nx");
  });
});
// =============================================================================
// rewind phase 2: multi-block summary turns and late-entry re-append (#80)
// =============================================================================

describe("rewind phase 2: multi-block text and late entries", () => {
  async function summaryBlocksTurn(
    f: { sm: SessionManager; pi: FakePi; ctx: FakeCtx },
    texts: string[],
  ): Promise<void> {
    const message = assistantMessage(
      texts.map((text) => ({ type: "text", text })),
      "stop",
      100,
    );
    f.sm.appendMessage(message as never);
    await fireTurnEndAll(f.pi, f.ctx, message);
  }

  it("joins text blocks with a newline: a preamble block before the summary completes", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    await requestRewind(f);
    await summaryBlocksTurn(f, ["Here is the summary:", SUMMARY_TEXT]);
    const done = rewindMessages(f.pi)[0].message;
    assert.equal(done.details.status, "done");
    const entry = f.sm.getEntry(done.details.summaryId as string);
    assert.ok(entry && entry.type === "branch_summary");
    assert.equal(entry.summary, `${BRANCH_SUMMARY_PREAMBLE}${SUMMARY_TEXT}`);
  });

  it("a heading split across blocks is joined with a newline, not glued", async () => {
    // `""` join would yield "## GoalShip it." — no `## Goal\b` heading.
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    await requestRewind(f);
    await summaryBlocksTurn(f, ["## Goal", "Ship it.\n\n## Next Steps\n1. x"]);
    const done = rewindMessages(f.pi)[0].message;
    assert.equal(done.details.status, "done");
    const entry = f.sm.getEntry(done.details.summaryId as string);
    assert.ok(entry && entry.type === "branch_summary");
    assert.equal(
      entry.summary,
      `${BRANCH_SUMMARY_PREAMBLE}## Goal\nShip it.\n\n## Next Steps\n1. x`,
    );
  });

  const NOTE_ONE =
    "The message between the branch_summary and this one arrived while the rewind was pending — it is new and still unanswered; handle it now.";
  const NOTE_TWO =
    "The 2 messages between the branch_summary and this one arrived while the rewind was pending — they are new and still unanswered; handle them now.";

  function appendLateUser(sm: SessionManager, text: string): string {
    return sm.appendMessage({
      role: "user",
      content: [{ type: "text", text }],
      timestamp: 1,
    } as never);
  }

  /** Active-branch entries after the new branch_summary, as compact tags. */
  function tailAfterSummary(sm: SessionManager): string[] {
    const branch = sm.getBranch();
    const at = branch.findIndex((e) => e.type === "branch_summary");
    assert.ok(at >= 0, "a branch_summary must be on the active branch");
    return branch.slice(at).map((e) => {
      if (e.type === "message") {
        return `${e.message.role}:${JSON.stringify(
          (e.message as { content?: unknown }).content,
        )}`;
      }
      if (e.type === "custom_message") {
        return `custom:${e.customType}:${JSON.stringify(e.content)}`;
      }
      return e.type;
    });
  }

  it("re-appends a late user steer and another extension's message, in order, onto the new branch", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    await requestRewind(f);
    const userId = appendLateUser(f.sm, "steer: also fix the docs");
    const customId = f.sm.appendCustomMessageEntry(
      "subagent-result",
      "subagent finished: 3 files changed",
      true,
      { job: 7 },
    );
    await summaryTurn(f);

    const done = rewindMessages(f.pi)[0].message;
    assert.equal(done.details.status, "done");
    assert.equal(done.details.reappendedLateMessages, 2);
    assert.deepEqual(tailAfterSummary(f.sm), [
      "branch_summary",
      "label",
      'user:[{"type":"text","text":"steer: also fix the docs"}]',
      'custom:subagent-result:"subagent finished: 3 files changed"',
    ]);
    const leaf = f.sm.getEntry(f.sm.getLeafId() as string);
    assert.ok(leaf && leaf.type === "custom_message");
    assert.equal(leaf.display, true);
    assert.deepEqual(leaf.details, { job: 7 });
    // Re-appended copies, not the originals (those stay on the abandoned
    // segment).
    const branchIds = new Set(f.sm.getBranch().map((e) => e.id));
    for (const id of [userId, customId]) {
      assert.ok(f.sm.getEntry(id), "the original stays on disk");
      assert.ok(!branchIds.has(id), "but off the active branch");
    }
    // No steer queue: re-appends never go back through input handlers.
    assert.deepEqual(f.pi.sendUserMessageCalls, []);
    assert.ok(done.content.endsWith(`\n\n${NOTE_TWO}`));
  });

  it("the continuation follows the re-appended entries once delivered", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    await requestRewind(f);
    appendLateUser(f.sm, "steer");
    await summaryTurn(f, SUMMARY_TEXT, "stop", { commit: true });
    const done = rewindMessages(f.pi)[0].message;
    assert.deepEqual(tailAfterSummary(f.sm), [
      "branch_summary",
      "label",
      'user:[{"type":"text","text":"steer"}]',
      `custom:${REWIND_CUSTOM_TYPE}:${JSON.stringify(done.content)}`,
    ]);
  });

  it("the late note is byte-exact for one message", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    await requestRewind(f);
    appendLateUser(f.sm, "steer");
    await summaryTurn(f);
    const done = rewindMessages(f.pi)[0].message;
    assert.equal(done.details.reappendedLateMessages, 1);
    assert.ok(done.content.endsWith(`\n\n${NOTE_ONE}`));
    assert.equal(buildLateEntriesNote(1), NOTE_ONE);
    assert.equal(buildLateEntriesNote(2), NOTE_TWO);
    assert.equal(
      buildLateEntriesNote(3),
      "The 3 messages between the branch_summary and this one arrived while the rewind was pending — they are new and still unanswered; handle them now.",
    );
  });

  it("does not re-append this extension's own rewind messages (e.g. a summary-retry)", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    await requestRewind(f);
    // `commit`: pi persists the delivered retry message.
    await summaryTurn(f, "Okay, summarizing now.", "stop", { commit: true });
    const retry = rewindMessages(f.pi)[0].message;
    assert.equal(retry.details.status, "summary-retry");
    await summaryTurn(f);
    const done = rewindMessages(f.pi)[1].message;
    assert.equal(done.details.status, "done");
    assert.equal(done.details.reappendedLateMessages, 0);
    assert.deepEqual(tailAfterSummary(f.sm), ["branch_summary", "label"]);
    assert.ok(!done.content.includes("arrived while the rewind was pending"));
  });

  it("does not re-append a rewind-hint message that landed after the rewind call", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    await requestRewind(f);
    f.sm.appendCustomMessageEntry(
      REWIND_HINT_CUSTOM_TYPE,
      expectedHint(91),
      true,
      undefined,
    );
    appendLateUser(f.sm, "steer");
    await summaryTurn(f);
    const done = rewindMessages(f.pi)[0].message;
    assert.equal(done.details.status, "done");
    assert.equal(done.details.reappendedLateMessages, 1);
    assert.deepEqual(tailAfterSummary(f.sm), [
      "branch_summary",
      "label",
      'user:[{"type":"text","text":"steer"}]',
    ]);
  });

  it("a re-appended hidden custom message stays hidden (display: false)", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    await requestRewind(f);
    f.sm.appendCustomMessageEntry("timer", "tick", false, { n: 1 });
    await summaryTurn(f);
    const done = rewindMessages(f.pi)[0].message;
    assert.equal(done.details.reappendedLateMessages, 1);
    assert.deepEqual(tailAfterSummary(f.sm), [
      "branch_summary",
      "label",
      'custom:timer:"tick"',
    ]);
    const leaf = f.sm.getEntry(f.sm.getLeafId() as string);
    assert.ok(leaf && leaf.type === "custom_message");
    assert.equal(leaf.display, false);
    assert.deepEqual(leaf.details, { n: 1 });
  });

  it("entries before the rewind call are not re-appended", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    f.sm.appendCustomMessageEntry("subagent-result", "early", true, {});
    appendLateUser(f.sm, "early user");
    appendTurn(f.sm, "u9", "a9", 34_000);
    await requestRewind(f);
    await summaryTurn(f);
    const done = rewindMessages(f.pi)[0].message;
    assert.equal(done.details.reappendedLateMessages, 0);
    assert.deepEqual(tailAfterSummary(f.sm), ["branch_summary", "label"]);
  });

  it("a throwing re-append does not fail the rewind and is not counted", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    await requestRewind(f);
    appendLateUser(f.sm, "lost steer");
    f.sm.appendCustomMessageEntry("subagent-result", "kept", true, {});
    // Arm the appendMessage throw only once the branch has been written.
    const sm = f.sm as unknown as {
      branchWithSummary: (...a: unknown[]) => string;
      appendMessage: (...a: unknown[]) => string;
    };
    const origBranch = sm.branchWithSummary.bind(f.sm);
    sm.branchWithSummary = (...a: unknown[]) => {
      const id = origBranch(...a);
      sm.appendMessage = () => {
        throw new Error("append boom");
      };
      return id;
    };
    await summaryTurn(f);
    const done = rewindMessages(f.pi)[0].message;
    assert.equal(done.details.status, "done");
    assert.equal(done.details.reappendedLateMessages, 1);
    assert.ok(done.content.endsWith(`\n\n${NOTE_ONE}`));
    assert.deepEqual(tailAfterSummary(f.sm), [
      "branch_summary",
      "label",
      'custom:subagent-result:"kept"',
    ]);
  });

  it("an estimator throw after the branch still delivers the continuation with contextAfter 0", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    await requestRewind(f);
    const sm = f.sm as unknown as {
      branchWithSummary: (...a: unknown[]) => string;
      buildSessionContext: () => unknown;
    };
    const origBranch = sm.branchWithSummary.bind(f.sm);
    sm.branchWithSummary = (...a: unknown[]) => {
      const id = origBranch(...a);
      sm.buildSessionContext = () => {
        throw new Error("estimate boom");
      };
      return id;
    };
    await summaryTurn(f);
    const done = rewindMessages(f.pi)[0].message;
    assert.equal(done.details.status, "done");
    assert.equal(done.details.contextAfter, 0);
    assert.equal(done.details.agentMessagesRefreshed, false);
    assert.match(done.content, / → 0\.0% of 1\.0M\n\n/);
    assert.ok(
      f.sm.getBranch().some((e) => e.type === "branch_summary"),
      "the branch landed",
    );
  });

  it("a throwing getBranch before the write cancels with the completion reason", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    await requestRewind(f);
    const sm = f.sm as unknown as { getBranch: () => unknown };
    const origGetBranch = sm.getBranch;
    sm.getBranch = () => {
      throw new Error("branch boom");
    };
    try {
      await assert.doesNotReject(summaryTurn(f));
    } finally {
      sm.getBranch = origGetBranch;
    }
    const calls = rewindMessages(f.pi);
    assert.equal(calls.length, 1);
    const reason = "the rewind could not be completed (branch boom)";
    assert.equal(calls[0].continue, true);
    assert.equal(calls[0].message.type, "custom_message");
    assert.equal(calls[0].message.display, true);
    assert.equal(
      calls[0].message.content,
      buildRewindCancelledText("start", reason),
    );
    assert.deepEqual(calls[0].message.details, {
      status: "cancelled",
      rewindTo: "start",
      newLabel: "end",
      reason,
      summaryAttempts: 0,
    });
    assert.ok(!f.sm.getEntries().some((e) => e.type === "branch_summary"));
    assert.equal(fireToolCall(f.pi, f.ctx), undefined, "gate reopened");
  });
});

// =============================================================================
// rewind message copy — byte-exact literals (no builder recomputation)
// =============================================================================

describe("rewind message copy (byte-exact)", () => {
  it("buildRewindDoneText", () => {
    assert.equal(
      buildRewindDoneText(
        {
          rewindTo: "start",
          newLabel: "end",
          beforeTokens: 30_000,
          contextWindow: 1_000_000,
        },
        4_000,
      ),
      "[rewind to 'start' · collapsed as 'end'] · context 3.0% → 0.4% of 1.0M\n\nA branch_summary of the collapsed work has been appended to your context. It is the newest branch_summary, the current state; earlier branch_summary messages, if present, are older collapsed segments. Items under '### Done' are complete. Items under '### In Progress', '### Blocked', or '## Next Steps' are pending — execute them next without re-confirming with the user, continuing the latest branch_summary's work. User messages retained above predate this rewind — do not re-answer them.",
    );
  });

  it("buildRewindDoneText without a context window uses the token form", () => {
    assert.equal(
      buildRewindDoneText(
        {
          rewindTo: "start",
          newLabel: "end",
          beforeTokens: 30_000,
          contextWindow: 0,
        },
        4_000,
      ).split("\n")[0],
      "[rewind to 'start' · collapsed as 'end'] · tokens 30000 → 4000",
    );
  });

  it("SUMMARY_PENDING_BLOCK_REASON", () => {
    assert.equal(
      SUMMARY_PENDING_BLOCK_REASON,
      "navigate_tree: a rewind is waiting for its summary — tool calls are blocked until it lands. Your entire next reply must be the summary in the format the rewind result requested, starting with '## Goal' — no acknowledgement, no tool calls.",
    );
  });

  it("buildSummaryRetryText", () => {
    assert.equal(
      buildSummaryRetryText("start"),
      "[navigate_tree] The rewind to 'start' is still waiting for its summary. Your entire reply must be the summary in the format the rewind result requested, starting with '## Goal' — no acknowledgement, no tool calls.",
    );
  });

  it("buildRewindCancelledText", () => {
    assert.equal(
      buildRewindCancelledText("start", "no summary after 3 attempts"),
      "[navigate_tree] Rewind to 'start' cancelled: no summary after 3 attempts. Nothing was collapsed — continue the work and rewind again later.",
    );
  });
});

// =============================================================================
// dispatch: rewind newLabel uniqueness (#55)
// =============================================================================

describe("dispatch: rewind newLabel uniqueness", () => {
  it("newLabel collision: rewind refuses before the summary request and mutates nothing", async () => {
    // Issue #55 (C8): a `newLabel` already on the active branch is refused
    // BEFORE the summary request or any tree mutation. Setup ordering mirrors
    // the retired move-on-collision test: the prior `anchor:b` sits on the
    // ancestral side of `rewindTo` (= tA), so the old behavior would have
    // moved it to the new summary.
    //   root → tB(anchor:b) → tA(anchor:a) → t3 → leaf
    const { sm, pi, tool, ctx } = setup();
    const tB = appendTurn(sm, "u1", "a1", 6_000);
    pi.pi.setLabel(tB.assistantId, "anchor:b");
    const tA = appendTurn(sm, "u2", "a2", 16_000);
    pi.pi.setLabel(tA.assistantId, "anchor:a");
    appendTurn(sm, "u3", "a3", 22_000);
    appendTurn(sm, "u4", "a4", 28_000);

    const fake = makeFakeSession(sm);
    __testHooks.captureSession(fake as unknown as AgentSession);

    const entriesBefore = sm.getEntries().length;
    const leafBefore = sm.getLeafId();

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
    assert.equal(result.isError, true);
    assert.equal(result.details.refusal, true);
    assert.match(
      result.content[0].text,
      /A label 'b' already exists on the active branch/,
    );
    assert.match(result.content[0].text, /fresh `newLabel`/);
    assert.match(result.content[0].text, /stays addressable via `rewindTo`/);
    // Refusal before the summary request: nothing is pending.
    assert.equal(
      fireToolCall(pi, ctx),
      undefined,
      "a duplicate-newLabel refusal must not leave a rewind pending",
    );
    assert.equal(pi.sendMessageCalls.length, 0);
    // No mutation: entry count, leaf, and both labels are untouched.
    assert.equal(sm.getEntries().length, entriesBefore);
    assert.equal(sm.getLeafId(), leafBefore);
    assert.equal(sm.getLabel(tB.assistantId), "anchor:b");
    assert.equal(sm.getLabel(tA.assistantId), "anchor:a");
    assert.equal(
      __testHooks.findLabeledEntry(sm, "anchor:b"),
      tB.assistantId,
      "the existing label must stay where it was",
    );
  });

  it("newLabel collision refusal copy is byte-exact", async () => {
    const { sm, pi, tool, ctx } = setup();
    const t1 = appendTurn(sm, "u1", "a1", 6_000);
    pi.pi.setLabel(t1.assistantId, "anchor:taken");
    appendTurn(sm, "u2", "a2", 16_000);
    const result = await tool.execute(
      "tc-rewind",
      {
        action: "rewind",
        rewindTo: "taken",
        newLabel: "taken",
        summaryFocus: "Preserve the user's instruction and continue the work.",
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, true);
    assert.equal(
      result.content[0].text,
      "A label 'taken' already exists on the active branch — labels are unique and immutable once written, so rewind requires a fresh `newLabel`. The existing 'taken' label stays addressable via `rewindTo`. Use action='list' to review the active labels, then pick a fresh newLabel.",
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
});

// =============================================================================
// dispatch: rewind min-savings floor (#21)
//
// Scenario pins from the issue-#20 repro, hermetic:
//   A — anchor→immediate rewind (zero/negligible work above the anchor)
//       must be REJECTED by the min-savings floor, never summarized;
//   B — healthy stages (≥20k apparent savings) must still request the
//       summary. (The immediate re-rewind of a just-landed summary is
//       pinned in the phase-2 block.)
// =============================================================================

describe("dispatch: rewind min-savings floor (#21)", () => {
  /** No newLabel may survive a rejected rewind — the label isn't consumed. */
  function assertLabelAbsent(sm: SessionManager, label: string): void {
    for (const e of sm.getBranch()) {
      assert.notEqual(sm.getLabel(e.id), label);
    }
  }

  it("A-pin variant A: anchor→immediate rewind with an earlier anchor present is rejected with the active-anchor list", async () => {
    const { sm, pi, tool, ctx } = setup();
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
    // Byte-exact pin (#84 moved the anchor walk into formatActiveAnchors).
    assert.equal(
      text,
      "Rewinding to 'impl-start' (as 'done') would free only ~0.0k tokens. Rewind further back to actually free context, or keep working. Active anchors: 'kickoff' 0.6% · 'impl-start' 1.2% of 1.0M",
    );
    // The numeric bar itself is NEVER printed agent-visible.
    assert.ok(
      !text.includes(String(MIN_REWIND_SAVINGS_TOKENS)),
      "rejection copy must not print the floor value",
    );
    // Nothing pending; newLabel not consumed; tree untouched.
    assert.equal(fireToolCall(pi, ctx), undefined);
    assertLabelAbsent(sm, "anchor:done");
    assert.deepEqual(
      sm.getBranch().map((e) => e.id),
      branchBefore,
    );
    assert.equal(sm.getLeafId(), leafBefore);
  });

  it("A-pin variant B: rewinding the OLDEST anchor is rejected with 'No earlier anchors'", async () => {
    const { sm, pi, tool, ctx } = setup();
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
    assert.equal(
      text,
      "Rewinding to 'start' (as 'done') would free only ~0.0k tokens. No earlier anchors — keep working and rewind later once more has accumulated above it. Active anchors: 'start' 0.6% of 1.0M",
    );
    assert.ok(!text.includes(String(MIN_REWIND_SAVINGS_TOKENS)));
    assert.equal(fireToolCall(pi, ctx), undefined);
    assertLabelAbsent(sm, "anchor:done");
  });

  it("A-pin variant B at contextWindow 0: k-format percents, no window suffix", async () => {
    const { sm, pi, tool, ctx } = setup({ contextWindow: 0 });
    const t1 = appendTurn(sm, "u1", "a1", 6_000);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendTurn(sm, "u2", "a2", 6_001);
    const r = await tool.execute(
      "tc-floor-b0",
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
    assert.equal(r.details.rejected, "min-savings");
    assert.equal(
      r.content[0].text,
      "Rewinding to 'start' (as 'done') would free only ~0.0k tokens. No earlier anchors — keep working and rewind later once more has accumulated above it. Active anchors: 'start' 6.0k",
    );
  });

  it("an empty-named manual label never counts as the oldest anchor", async () => {
    const { sm, pi, tool, ctx } = setup();
    const t0 = appendTurn(sm, "u0", "a0", 3_000);
    pi.pi.setLabel(t0.assistantId, "anchor:");
    const t1 = appendTurn(sm, "u1", "a1", 6_000);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    appendTurn(sm, "u2", "a2", 6_001);
    const r = await tool.execute(
      "tc-floor-empty",
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
    assert.equal(r.details.rejected, "min-savings");
    assert.match(
      r.content[0].text,
      /No earlier anchors — keep working and rewind later once more has accumulated above it\./,
    );
  });

  it("B-pin: healthy stage (≥20k apparent savings) requests the summary", async () => {
    const { sm, pi, tool, ctx } = setup();
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
    assert.equal(r.details.status, "summary-requested");
    // Apparent savings = contextBefore − anchor-time total (6_000).
    const before = r.details.contextBefore as number;
    assert.ok(
      before - 6_000 >= 20_000,
      `expected ≥20k apparent savings; got ${before - 6_000}`,
    );
    assert.ok(fireToolCall(pi, ctx), "the rewind is pending");
  });

  it("boundary: apparent savings just under the floor rejects; exactly at the floor accepts", async () => {
    // Deterministic measurement math: with 'u*'/'a*' texts (2 chars → 1
    // estimated trailing token each), a chain anchored at a1(T1) with leaf
    // a3(T3) measures beforeTokens = T2 + 1 and tokensAtTarget = T1, i.e.
    //   apparentSavings = T2 + 1 − T1.
    async function floorFixture(t2Total: number) {
      const { sm, pi, tool, ctx } = setup();
      const t1 = appendTurn(sm, "u1", "a1", 5_000);
      pi.pi.setLabel(t1.assistantId, "anchor:start");
      appendTurn(sm, "u2", "a2", t2Total);
      appendTurn(sm, "u3", "a3", t2Total + 1_000);
      return { pi, tool, ctx };
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
    assert.equal(fireToolCall(under.pi, under.ctx), undefined);

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
    assert.ok(fireToolCall(at.pi, at.ctx), "the accepted rewind is pending");
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
 * The hint text at `percent` of `contextWindow` (via `usageAt`), with the
 * reserve the test config sets and the active-anchor list the handler must
 * render (default: the lone `anchorStart` label at 0.0% of 1.0M).
 */
function expectedHint(
  percent: number,
  anchors = "'start' 0.0% of 1.0M",
  reserve = 100_000,
  contextWindow = 1_000_000,
): string {
  const usage = usageAt(percent, contextWindow);
  return buildRewindHintText(
    percent,
    contextWindow - (usage.tokens ?? 0),
    reserve,
    contextWindow,
    anchors,
  );
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
  reason: "startup" | "reload" | "new" | "resume" | "fork" = "startup",
): Promise<void> {
  const handlers = pi.onCalls.get("session_start");
  assert.ok(handlers, "factory must register a session_start handler");
  assert.equal(handlers.length, 1);
  const prevAgentDir = process.env.PI_CODING_AGENT_DIR;
  process.env.PI_CODING_AGENT_DIR = fixture.agentDir;
  try {
    await handlers[0]({ type: "session_start", reason } as never, ctx as never);
  } finally {
    if (prevAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = prevAgentDir;
  }
}

/**
 * Fire the captured synchronous hint / start-anchor `turn_end` handler. The
 * factory registers two `turn_end` handlers: the async rewind summary
 * handler (#80) first, this one last. Only this one is fired here so the
 * stale-ctx pins (exactly one `sessionManager` read) stay about this
 * handler; rewind tests use `fireTurnEndAll`.
 */
function fireTurnEnd(pi: FakePi, ctx: FakeCtx, turnIndex = 0): void {
  const handlers = pi.onCalls.get("turn_end");
  assert.ok(handlers, "factory must register a turn_end handler");
  assert.equal(handlers.length, 2);
  handlers[1]({ type: "turn_end", turnIndex } as never, ctx as never);
}

/** Fire the captured `agent_end` handler (run-end fallback). */
function fireAgentEnd(pi: FakePi, ctx: FakeCtx): void {
  const handlers = pi.onCalls.get("agent_end");
  assert.ok(handlers, "factory must register an agent_end handler");
  assert.equal(handlers.length, 1);
  handlers[0]({ type: "agent_end", messages: [] } as never, ctx as never);
}

/** Drive the `before_agent_start` handler to (re)compute the tool gate. */
async function fireBeforeAgentStart(
  pi: FakePi,
  ctx: FakeCtx,
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
    ctx as never,
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
      writeGlobalConfig(
        fixture,
        JSON.stringify({ rewindHintReserveTokens: 100_000 }),
      );
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
          content: expectedHint(90),
          display: true,
        },
      ]);
      // Byte-exact pin: the anchors are named, so no `list` call is needed.
      assert.equal(
        pi.sendMessageCalls[0][0].content,
        "[navigate_tree hint] Context is at 90.0% of 1.0M — 100.0k tokens left, within the 100.0k reserve. Persist what matters to files now, then rewind to the oldest appropriate anchor. Active anchors: 'start' 0.0% of 1.0M",
      );
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("project layer overrides global when trusted; untrusted ignores the file", async () => {
    const fixture = makeConfigFixture();
    try {
      writeGlobalConfig(
        fixture,
        JSON.stringify({ rewindHintReserveTokens: 100_000 }),
      );
      writeProjectConfig(
        fixture,
        JSON.stringify({ rewindHintReserveTokens: 500_000 }),
      );

      // Trusted: the project's 500k reserve wins over the global 100k.
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

      // Untrusted: the project file is ignored silently; 50% does not
      // fire (500k left > 100k), 90% does.
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
        JSON.stringify({ rewindHintReserveTokens: true }),
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
          `navigate_tree: invalid rewindHintReserveTokens in project config at ${projectConfigPath(fixture)} — expected integer 1000-1000000 or a disable sentinel (null, false, "off", "disabled"); rewind hint disabled for this session.`,
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

  it("the removed rewindHintAtPercent warns and leaves the hint off", async () => {
    const fixture = makeConfigFixture();
    try {
      writeGlobalConfig(fixture, JSON.stringify({ rewindHintAtPercent: "90" }));
      const { sm, pi, ctx } = setup();
      anchorStart(sm, pi);
      await runSessionStart(pi, ctx, fixture);
      assert.deepEqual(ctx.notifyCalls, [
        [
          `navigate_tree: rewindHintAtPercent in global config at ${globalConfigPath(fixture)} was removed in 0.3.0 and is ignored — use rewindHintReserveTokens (tokens left in the window when the hint fires, e.g. 50000).`,
          "warning",
        ],
      ]);
      ctx.setContextUsage(usageAt(99));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 0);
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
      writeGlobalConfig(
        fixture,
        JSON.stringify({ rewindHintReserveTokens: 100_000 }),
      );
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
      await fireBeforeAgentStart(pi, ctx, ["read", "bash"]);
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
  const ENABLED_CONFIG = JSON.stringify({ rewindHintReserveTokens: 100_000 });

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
      writeGlobalConfig(
        fixture,
        JSON.stringify({ rewindHintReserveTokens: null }),
      );
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
      await fireBeforeAgentStart(pi, ctx, ["read"]);
      ctx.setContextUsage(usageAt(99));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 0);
      assert.deepEqual(ctx.notifyCalls, []);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("keeps the tool gate per-session (#58): a sibling session must not clear this session's hint", async () => {
    const fixture = makeConfigFixture();
    try {
      writeGlobalConfig(fixture, ENABLED_CONFIG);
      const { sm, pi, ctx } = setup();
      anchorStart(sm, pi);
      await runSessionStart(pi, ctx, fixture);

      // Session A runs with the tool active, recording A's per-session gate.
      await fireBeforeAgentStart(pi, ctx, [TOOL_NAME, "read"]);

      // Sibling session B (e.g. an in-process subagent) starts without it.
      const smB = SessionManager.inMemory("/tmp");
      const ctxB = makeCtx(smB);
      await fireBeforeAgentStart(pi, ctxB, ["read"]);

      // A crosses: the hint must still fire — i.e. B's `before_agent_start`
      // did not clear A's per-session gate.
      ctx.setContextUsage(usageAt(95));
      fireTurnEnd(pi, ctx);
      assert.equal(
        pi.sendMessageCalls.length,
        1,
        "a sibling session's before_agent_start must not clear A's hint gate",
      );

      // Re-arm the shared tracker, then flip the shared fallback back on.
      ctx.setContextUsage(usageAt(10));
      fireTurnEnd(pi, ctx);
      await fireBeforeAgentStart(pi, ctx, [TOOL_NAME, "read"]);

      // B's own false gate must be honored (not silently fall back to the
      // shared true): B stays silent, must not spend the crossing, and A
      // must still fire after it.
      ctxB.setContextUsage(usageAt(95));
      fireTurnEnd(pi, ctxB);
      assert.equal(ctxB.notifyCalls.length, 0);
      assert.equal(pi.sendMessageCalls.length, 1);

      ctx.setContextUsage(usageAt(95));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 2);
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

  it("no-ops on tokens-null usage even when percent is set", async () => {
    const { fixture, pi, ctx } = await enabledSetup();
    try {
      // A 50k window under the 100k reserve: without the tokens guard,
      // `contextWindow - null` (50k) would read as inside the reserve.
      ctx.setContextUsage({
        tokens: null,
        contextWindow: 50_000,
        percent: 99,
      });
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 0);
      ctx.setContextUsage(usageAt(95));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 1);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("fires on an overrun window (tokens > contextWindow) with 0.0k left", async () => {
    const { fixture, pi, ctx } = await enabledSetup();
    try {
      ctx.setContextUsage({
        tokens: 1_050_000,
        contextWindow: 1_000_000,
        percent: 105,
      });
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 1);
      assert.equal(
        (pi.sendMessageCalls[0][0] as { content: string }).content,
        "[navigate_tree hint] Context is at 105.0% of 1.0M — 0.0k tokens left, within the 100.0k reserve. Persist what matters to files now, then rewind to the oldest appropriate anchor. Active anchors: 'start' 0.0% of 1.0M",
      );
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("a spent crossing skips the anchor walk entirely", async () => {
    const { fixture, sm, pi, ctx } = await enabledSetup();
    try {
      ctx.setContextUsage(usageAt(95));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 1);

      const calls = { getEntries: 0, getBranch: 0, getLabel: 0 };
      const target = sm as unknown as Record<
        keyof typeof calls,
        (...args: unknown[]) => unknown
      >;
      const originals = { ...calls } as unknown as typeof target;
      for (const name of Object.keys(calls) as Array<keyof typeof calls>) {
        const original = target[name].bind(sm);
        originals[name] = target[name];
        target[name] = (...args: unknown[]) => {
          calls[name]++;
          return original(...args);
        };
      }
      try {
        fireTurnEnd(pi, ctx);
        fireTurnEnd(pi, ctx);
      } finally {
        Object.assign(target, originals);
      }
      assert.equal(pi.sendMessageCalls.length, 1);
      // The start anchor's existence check may read the branch; the hint's
      // walk (getEntries + per-anchor estimates) must not run at all.
      assert.equal(calls.getEntries, 0);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("fires once per crossing and re-arms only after a drop below the bar", async () => {
    const fixture = makeConfigFixture();
    try {
      writeGlobalConfig(
        fixture,
        JSON.stringify({ rewindHintReserveTokens: 800_000 }),
      );
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
          content: expectedHint(21, undefined, 800_000),
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
        [REWIND_HINT_CUSTOM_TYPE, expectedHint(95), true],
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
        [REWIND_HINT_CUSTOM_TYPE, expectedHint(95), true],
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

  it("a throwing getEntries leaves the crossing unspent for a retry", async () => {
    const { fixture, sm, pi, ctx } = await enabledSetup();
    try {
      const originalGetEntries = sm.getEntries;
      (sm as unknown as { getEntries: () => never }).getEntries = () => {
        throw new Error("getEntries exploded");
      };
      ctx.setContextUsage(usageAt(95));
      assert.doesNotThrow(() => fireTurnEnd(pi, ctx));
      assert.equal(pi.sendMessageCalls.length, 0);

      (sm as unknown as { getEntries: typeof originalGetEntries }).getEntries =
        originalGetEntries;
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 1);
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
      const text = expectedHint(95);
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
      await fireBeforeAgentStart(pi, ctx, undefined);
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
      await fireBeforeAgentStart(pi, ctx, ["read", TOOL_NAME]);
      ctx.setContextUsage(usageAt(95));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 1);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });
});

// =============================================================================
// automatic start anchor (#55)
//
// The extension writes a silent `anchor:start` label at the X-th turn of the
// agent run (config `anchorStartAfterTurns`, default 2), or at the run's end
// when the run finishes earlier. Per-session state (settle marker, config
// value, tool-active gate) is keyed by session id; the `findLabeledEntry`
// existence check on the active branch is authoritative. These tests drive
// the real session_start / turn_end / agent_end handlers against temp config
// fixtures (real loader, real fs) and in-memory SessionManagers. No LLM is
// involved.
// =============================================================================

describe("rewind hint: suppressed on turns a rewind owns (#80)", () => {
  /** Hint `sendMessage` calls (the only kind this extension makes). */
  function hintCalls(pi: FakePi): unknown[][] {
    return pi.sendMessageCalls.filter(
      (args) =>
        (args[0] as { customType?: string } | undefined)?.customType ===
        REWIND_HINT_CUSTOM_TYPE,
    );
  }

  it("skips the rewind call, a failed attempt and the completing turn; the crossing fires on the next normal turn", async () => {
    const fixture = makeConfigFixture();
    try {
      writeGlobalConfig(
        fixture,
        JSON.stringify({ rewindHintReserveTokens: 100_000 }),
      );
      const f = setup();
      setupRewindable(f.sm, f.pi, { capture: true });
      await runSessionStart(f.pi, f.ctx, fixture);
      f.ctx.setContextUsage(usageAt(95));

      await requestRewind(f);
      assert.equal(hintCalls(f.pi).length, 0, "the rewind call's turn");
      await summaryTurn(f, "Okay, summarizing now.");
      assert.equal(
        rewindMessages(f.pi)[0].message.details.status,
        "summary-retry",
      );
      assert.equal(hintCalls(f.pi).length, 0, "a failed attempt");
      await summaryTurn(f);
      assert.equal(rewindMessages(f.pi)[1].message.details.status, "done");
      assert.equal(hintCalls(f.pi).length, 0, "the completing turn");

      // The crossing stayed unspent: the next normal turn fires it once.
      const next = assistantMessage([{ type: "text", text: "next" }], "stop");
      f.sm.appendMessage(next as never);
      const boundary = await fireTurnEndAll(f.pi, f.ctx, next);
      assert.deepEqual(boundary, { entries: [], continue: false });
      assert.deepEqual(hintCalls(f.pi), [
        [
          {
            customType: REWIND_HINT_CUSTOM_TYPE,
            content: expectedHint(95, "'start' 0.6% · 'end' 0.6% of 1.0M"),
            display: true,
          },
        ],
      ]);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("skips a tool-call failure turn too, and the start anchor still runs on it", async () => {
    const fixture = makeConfigFixture();
    try {
      writeGlobalConfig(
        fixture,
        JSON.stringify({ rewindHintReserveTokens: 100_000 }),
      );
      const f = setup();
      // Anchor `a`, not `start`: the auto start anchor must be the writer.
      setupRewindable(f.sm, f.pi, { capture: true, rewindToName: "a" });
      await runSessionStart(f.pi, f.ctx, fixture);
      assert.deepEqual(f.ctx.notifyCalls, []);
      f.ctx.setContextUsage(usageAt(95));
      await requestRewind(f, { ...REWIND_ARGS, rewindTo: "a" });
      assert.equal(hintCalls(f.pi).length, 0);
      assert.equal(
        __testHooks.findLabeledEntry(f.sm, START_ANCHOR_LABEL),
        null,
        "turn 1 is below the default X=2",
      );
      const { message } = appendToolCallAssistant(f.sm, [
        { id: "tc-bash-1", name: "bash", arguments: { command: "ls" } },
      ]);
      await fireTurnEndAll(f.pi, f.ctx, message, { turnIndex: 1 });
      assert.equal(hintCalls(f.pi).length, 0, "a tool-call failure turn");
      assert.ok(
        __testHooks.findLabeledEntry(f.sm, START_ANCHOR_LABEL),
        "the start anchor is not gated by the rewind",
      );
    } finally {
      cleanupConfigFixture(fixture);
    }
  });
});

describe("automatic start anchor (#55)", () => {
  const ENV_KEY = "PI_NAVIGATE_TREE_START_ANCHOR";
  let savedEnv: string | undefined;

  beforeEach(() => {
    savedEnv = process.env[ENV_KEY];
    delete process.env[ENV_KEY];
  });

  afterEach(() => {
    if (savedEnv === undefined) delete process.env[ENV_KEY];
    else process.env[ENV_KEY] = savedEnv;
    savedEnv = undefined;
  });

  it("registers exactly one agent_end handler", () => {
    const { pi } = setup();
    const handlers = pi.onCalls.get("agent_end");
    assert.ok(handlers, "factory must register an agent_end handler");
    assert.equal(handlers.length, 1);
  });

  it("defaults to X=2 with no config: silent write at the 2nd turn, no model message, no UI", async () => {
    const fixture = makeConfigFixture();
    try {
      const { sm, pi, ctx } = setup();
      await runSessionStart(pi, ctx, fixture);
      assert.deepEqual(ctx.notifyCalls, [], "absent config must not warn");

      appendTurn(sm, "u1", "a1");
      const leaf1 = sm.getLeafId();
      fireTurnEnd(pi, ctx, 0); // turn 1: below X
      assert.equal(pi.setLabelCalls.length, 0);
      assert.equal(sm.getLabel(leaf1 as string), undefined);

      appendTurn(sm, "u2", "a2");
      const leaf2 = sm.getLeafId();
      fireTurnEnd(pi, ctx, 1); // turn 2: triggers
      assert.equal(pi.setLabelCalls.length, 1);
      assert.deepEqual(pi.setLabelCalls[0], [leaf2, START_ANCHOR_LABEL]);
      assert.equal(sm.getLabel(leaf2 as string), START_ANCHOR_LABEL);
      assert.equal(__testHooks.findLabeledEntry(sm, START_ANCHOR_LABEL), leaf2);
      // Silent: label entry only — no model message, no UI.
      assert.equal(pi.sendMessageCalls.length, 0);
      assert.deepEqual(ctx.notifyCalls, []);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("honors a configured X and anchors the X-th turn's persisted tail", async () => {
    const fixture = makeConfigFixture();
    try {
      writeGlobalConfig(fixture, JSON.stringify({ anchorStartAfterTurns: 3 }));
      const { sm, pi, ctx } = setup();
      await runSessionStart(pi, ctx, fixture);

      appendTurn(sm, "u1", "a1");
      fireTurnEnd(pi, ctx, 0);
      assert.equal(pi.setLabelCalls.length, 0, "turn 1 must not fire at X=3");
      appendTurn(sm, "u2", "a2");
      fireTurnEnd(pi, ctx, 1);
      assert.equal(pi.setLabelCalls.length, 0, "turn 2 must not fire at X=3");
      appendTurn(sm, "u3", "a3");
      const leaf = sm.getLeafId();
      fireTurnEnd(pi, ctx, 2);
      assert.deepEqual(pi.setLabelCalls, [[leaf, START_ANCHOR_LABEL]]);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("project X overrides global when trusted; an untrusted project is ignored", async () => {
    const fixture = makeConfigFixture();
    try {
      writeGlobalConfig(fixture, JSON.stringify({ anchorStartAfterTurns: 5 }));
      writeProjectConfig(fixture, JSON.stringify({ anchorStartAfterTurns: 2 }));

      // Trusted: project 2 fires at the 2nd turn.
      {
        const { sm, pi, ctx } = setup();
        ctx.cwd = fixture.projectDir;
        ctx.isProjectTrusted = () => true;
        await runSessionStart(pi, ctx, fixture);
        appendTurn(sm, "u1", "a1");
        appendTurn(sm, "u2", "a2");
        fireTurnEnd(pi, ctx, 1);
        assert.equal(pi.setLabelCalls.length, 1);
      }

      // Untrusted: global 5 applies; the 2nd turn must not fire.
      {
        const { sm, pi, ctx } = setup();
        ctx.cwd = fixture.projectDir;
        await runSessionStart(pi, ctx, fixture);
        appendTurn(sm, "u1", "a1");
        appendTurn(sm, "u2", "a2");
        fireTurnEnd(pi, ctx, 1);
        assert.equal(pi.setLabelCalls.length, 0);
      }
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("invalid project X warns and falls through to the valid global layer", async () => {
    const fixture = makeConfigFixture();
    try {
      writeGlobalConfig(fixture, JSON.stringify({ anchorStartAfterTurns: 3 }));
      writeProjectConfig(
        fixture,
        JSON.stringify({ anchorStartAfterTurns: "nope" }),
      );
      const { sm, pi, ctx } = setup();
      ctx.cwd = fixture.projectDir;
      ctx.isProjectTrusted = () => true;
      await runSessionStart(pi, ctx, fixture);
      assert.deepEqual(ctx.notifyCalls, [
        [
          `navigate_tree: invalid anchorStartAfterTurns in project config at ${projectConfigPath(fixture)} — expected integer ${START_ANCHOR_MIN_TURNS}-${START_ANCHOR_MAX_TURNS}; falling back to the other layer's valid value, else the default ${START_ANCHOR_DEFAULT_TURNS}.`,
          "warning",
        ],
      ]);
      appendTurn(sm, "u1", "a1");
      appendTurn(sm, "u2", "a2");
      const leaf = sm.getLeafId();
      fireTurnEnd(pi, ctx, 2); // global X=3
      assert.deepEqual(pi.setLabelCalls, [[leaf, START_ANCHOR_LABEL]]);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("loads the field for all five session_start reasons", async () => {
    const reasons = ["startup", "reload", "new", "resume", "fork"] as const;
    for (const reason of reasons) {
      const fixture = makeConfigFixture();
      try {
        writeGlobalConfig(
          fixture,
          JSON.stringify({ anchorStartAfterTurns: 4 }),
        );
        const { sm, pi, ctx } = setup();
        await runSessionStart(pi, ctx, fixture, reason);
        assert.deepEqual(
          ctx.notifyCalls,
          [],
          `${reason}: valid config must not warn`,
        );
        for (let i = 0; i < 3; i++) {
          appendTurn(sm, `u${i}`, `a${i}`);
          fireTurnEnd(pi, ctx, i);
        }
        assert.equal(pi.setLabelCalls.length, 0, `${reason}: below X`);
        appendTurn(sm, "u4", "a4");
        fireTurnEnd(pi, ctx, 3);
        assert.equal(pi.setLabelCalls.length, 1, `${reason}: X-th turn writes`);
      } finally {
        cleanupConfigFixture(fixture);
      }
    }
  });

  it("agent_end fallback settles a run shorter than X at the run's end", async () => {
    const fixture = makeConfigFixture();
    try {
      writeGlobalConfig(fixture, JSON.stringify({ anchorStartAfterTurns: 5 }));
      const { sm, pi, ctx } = setup();
      await runSessionStart(pi, ctx, fixture);
      appendTurn(sm, "u1", "a1");
      fireTurnEnd(pi, ctx, 0);
      appendTurn(sm, "u2", "a2");
      fireTurnEnd(pi, ctx, 1);
      const leaf = sm.getLeafId();
      assert.equal(pi.setLabelCalls.length, 0);
      fireAgentEnd(pi, ctx);
      assert.deepEqual(pi.setLabelCalls, [[leaf, START_ANCHOR_LABEL]]);
      assert.equal(sm.getLabel(leaf as string), START_ANCHOR_LABEL);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("agent_end after a run with no turns anchors the last persisted entry (user message)", async () => {
    const fixture = makeConfigFixture();
    try {
      const { sm, pi, ctx } = setup();
      await runSessionStart(pi, ctx, fixture);
      const userId = sm.appendMessage({
        role: "user",
        content: [{ type: "text", text: "only a user message" }],
      } as never);
      fireAgentEnd(pi, ctx);
      assert.deepEqual(pi.setLabelCalls, [[userId, START_ANCHOR_LABEL]]);
      assert.equal(sm.getLabel(userId), START_ANCHOR_LABEL);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("settles after the first write: later turns and a second agent run never re-insert", async () => {
    const fixture = makeConfigFixture();
    try {
      const { sm, pi, ctx } = setup();
      await runSessionStart(pi, ctx, fixture);
      appendTurn(sm, "u1", "a1");
      appendTurn(sm, "u2", "a2");
      const firstLeaf = sm.getLeafId();
      fireTurnEnd(pi, ctx, 1);
      assert.equal(pi.setLabelCalls.length, 1);
      fireAgentEnd(pi, ctx); // same run's end: settled, no-op
      appendTurn(sm, "u3", "a3");
      fireTurnEnd(pi, ctx, 1); // second run's 2nd turn: no-op
      fireAgentEnd(pi, ctx);
      assert.equal(pi.setLabelCalls.length, 1, "must never re-insert");
      assert.equal(
        __testHooks.findLabeledEntry(sm, START_ANCHOR_LABEL),
        firstLeaf,
        "the original label target must not move",
      );
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("an existing anchor:start wins (manual /tree label): skip + settle, never write", async () => {
    const fixture = makeConfigFixture();
    try {
      const { sm, pi, ctx } = setup();
      await runSessionStart(pi, ctx, fixture);
      const t = appendTurn(sm, "u1", "a1");
      pi.pi.setLabel(t.assistantId, START_ANCHOR_LABEL);
      appendTurn(sm, "u2", "a2");
      const writesBefore = pi.setLabelCalls.length;
      fireTurnEnd(pi, ctx, 1);
      fireAgentEnd(pi, ctx);
      appendTurn(sm, "u3", "a3");
      fireTurnEnd(pi, ctx, 1);
      assert.equal(
        pi.setLabelCalls.length,
        writesBefore,
        "an existing label must settle the session without a write",
      );
      assert.equal(sm.getLabel(t.assistantId), START_ANCHOR_LABEL);
      assert.equal(
        __testHooks.findLabeledEntry(sm, START_ANCHOR_LABEL),
        t.assistantId,
      );
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("toolActive gate: inactive tool skips; reactivated tool writes", async () => {
    const fixture = makeConfigFixture();
    try {
      const { sm, pi, ctx } = setup();
      await runSessionStart(pi, ctx, fixture);
      appendTurn(sm, "u1", "a1");
      appendTurn(sm, "u2", "a2");
      await fireBeforeAgentStart(pi, ctx, ["read"]);
      fireTurnEnd(pi, ctx, 1);
      fireAgentEnd(pi, ctx);
      assert.equal(pi.setLabelCalls.length, 0);
      await fireBeforeAgentStart(pi, ctx, ["read", TOOL_NAME]);
      fireAgentEnd(pi, ctx);
      assert.equal(pi.setLabelCalls.length, 1);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("kill switch is read at write time: =0 skips both triggers; unset retries", async () => {
    const fixture = makeConfigFixture();
    try {
      const { sm, pi, ctx } = setup();
      await runSessionStart(pi, ctx, fixture);
      appendTurn(sm, "u1", "a1");
      appendTurn(sm, "u2", "a2");
      const leaf = sm.getLeafId();
      process.env[ENV_KEY] = "0";
      fireTurnEnd(pi, ctx, 1);
      fireAgentEnd(pi, ctx);
      assert.equal(
        pi.setLabelCalls.length,
        0,
        "kill switch must suppress both triggers",
      );
      delete process.env[ENV_KEY];
      // A skipped settle left the session unsettled: the next trigger retries.
      fireAgentEnd(pi, ctx);
      assert.deepEqual(pi.setLabelCalls, [[leaf, START_ANCHOR_LABEL]]);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("setLabel throw falls back to appendLabelChange(leafId, 'anchor:start') and settles", async () => {
    const fixture = makeConfigFixture();
    try {
      const { sm, pi, ctx } = setup();
      await runSessionStart(pi, ctx, fixture);
      appendTurn(sm, "u1", "a1");
      appendTurn(sm, "u2", "a2");
      const leaf = sm.getLeafId();
      const appendCalls: unknown[][] = [];
      const origAppend = sm.appendLabelChange.bind(sm);
      (
        sm as unknown as {
          appendLabelChange: (...args: unknown[]) => string;
        }
      ).appendLabelChange = (...args: unknown[]) => {
        appendCalls.push(args);
        return origAppend(args[0] as string, args[1] as string | undefined);
      };
      (
        pi.pi as unknown as { setLabel: (...args: unknown[]) => void }
      ).setLabel = () => {
        throw new Error("stale session runtime");
      };
      assert.doesNotThrow(() => fireTurnEnd(pi, ctx, 1));
      assert.deepEqual(appendCalls, [[leaf, START_ANCHOR_LABEL]]);
      assert.equal(sm.getLabel(leaf as string), START_ANCHOR_LABEL);
      // Settled via the fallback: later triggers are no-ops.
      fireTurnEnd(pi, ctx, 1);
      fireAgentEnd(pi, ctx);
      assert.equal(appendCalls.length, 1);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("double write failure stays fail-open and unsettled for the next retry", async () => {
    const fixture = makeConfigFixture();
    try {
      const { sm, pi, ctx } = setup();
      await runSessionStart(pi, ctx, fixture);
      appendTurn(sm, "u1", "a1");
      appendTurn(sm, "u2", "a2");
      const origSetLabel = pi.pi.setLabel.bind(pi.pi);
      const origAppend = sm.appendLabelChange.bind(sm);
      (
        pi.pi as unknown as { setLabel: (...args: unknown[]) => void }
      ).setLabel = () => {
        throw new Error("stale session runtime");
      };
      (
        sm as unknown as {
          appendLabelChange: (...args: unknown[]) => string;
        }
      ).appendLabelChange = () => {
        throw new Error("append exploded");
      };
      assert.doesNotThrow(() => fireTurnEnd(pi, ctx, 1));
      assert.equal(pi.setLabelCalls.length, 0);
      // Restore both: the unsettled session retries and succeeds at the
      // run end.
      (pi.pi as unknown as { setLabel: typeof pi.pi.setLabel }).setLabel =
        origSetLabel;
      (
        sm as unknown as { appendLabelChange: typeof sm.appendLabelChange }
      ).appendLabelChange = origAppend;
      fireAgentEnd(pi, ctx);
      assert.equal(pi.setLabelCalls.length, 1);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("existence-check read failure leaves the session unsettled (retries later)", async () => {
    const fixture = makeConfigFixture();
    try {
      const { sm, pi, ctx } = setup();
      await runSessionStart(pi, ctx, fixture);
      appendTurn(sm, "u1", "a1");
      appendTurn(sm, "u2", "a2");
      const origGetBranch = sm.getBranch.bind(sm);
      let armed = true;
      (sm as unknown as { getBranch: () => unknown }).getBranch = () => {
        if (armed) throw new Error("getBranch boom");
        return origGetBranch();
      };
      assert.doesNotThrow(() => fireTurnEnd(pi, ctx, 1));
      assert.equal(pi.setLabelCalls.length, 0);
      armed = false;
      fireAgentEnd(pi, ctx);
      assert.equal(pi.setLabelCalls.length, 1);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("setLabel throw with a missing appendLabelChange stays unsettled and retries", async () => {
    const fixture = makeConfigFixture();
    try {
      const { sm, pi, ctx } = setup();
      await runSessionStart(pi, ctx, fixture);
      appendTurn(sm, "u1", "a1");
      appendTurn(sm, "u2", "a2");
      const leaf = sm.getLeafId();
      const origSetLabel = pi.pi.setLabel.bind(pi.pi);
      const origAppend = sm.appendLabelChange.bind(sm);
      (
        pi.pi as unknown as { setLabel: (...args: unknown[]) => void }
      ).setLabel = () => {
        throw new Error("stale session runtime");
      };
      (sm as unknown as { appendLabelChange: unknown }).appendLabelChange =
        undefined;
      // A missing fallback method is NOT a successful write: no label, no
      // settle, and nothing throws into the loop.
      assert.doesNotThrow(() => fireTurnEnd(pi, ctx, 1));
      assert.equal(
        sm.getLabel(leaf as string),
        undefined,
        "no write may be recorded",
      );
      // Restore both: the still-unsettled session retries at the run end.
      (pi.pi as unknown as { setLabel: typeof pi.pi.setLabel }).setLabel =
        origSetLabel;
      (
        sm as unknown as { appendLabelChange: typeof sm.appendLabelChange }
      ).appendLabelChange = origAppend;
      fireAgentEnd(pi, ctx);
      assert.deepEqual(pi.setLabelCalls, [[leaf, START_ANCHOR_LABEL]]);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("appendLabelChange fallback preserves the method receiver (this === SessionManager)", async () => {
    const fixture = makeConfigFixture();
    try {
      const { sm, pi, ctx } = setup();
      await runSessionStart(pi, ctx, fixture);
      appendTurn(sm, "u1", "a1");
      appendTurn(sm, "u2", "a2");
      const leaf = sm.getLeafId();
      const origAppend = sm.appendLabelChange.bind(sm);
      // Prototype-style method that needs its receiver: an unbound
      // `append(...)` call loses `this` and throws instead of writing.
      (
        sm as unknown as {
          appendLabelChange: (
            entryId: string,
            label: string | undefined,
          ) => string;
        }
      ).appendLabelChange = function (
        this: unknown,
        entryId: string,
        label: string | undefined,
      ) {
        if (this !== sm) {
          throw new Error("appendLabelChange lost its receiver");
        }
        return origAppend(entryId, label);
      };
      (
        pi.pi as unknown as { setLabel: (...args: unknown[]) => void }
      ).setLabel = () => {
        throw new Error("stale session runtime");
      };
      assert.doesNotThrow(() => fireTurnEnd(pi, ctx, 1));
      assert.equal(sm.getLabel(leaf as string), START_ANCHOR_LABEL);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("per-session config: a later session_start must not retune an earlier session", async () => {
    const fixtureA = makeConfigFixture();
    const fixtureB = makeConfigFixture();
    try {
      writeGlobalConfig(fixtureA, JSON.stringify({ anchorStartAfterTurns: 5 }));
      writeGlobalConfig(fixtureB, JSON.stringify({ anchorStartAfterTurns: 1 }));
      const smA = SessionManager.inMemory("/tmp");
      const smB = SessionManager.inMemory("/tmp");
      const pi = makeFakePi(smA);
      navigateTree(pi.pi);
      const ctxA = makeCtx(smA);
      const ctxB = makeCtx(smB);
      const writes: Array<[string, string | undefined]> = [];
      let target = smA;
      (
        pi.pi as unknown as {
          setLabel: (entryId: string, label: string | undefined) => void;
        }
      ).setLabel = (entryId, label) => {
        writes.push([entryId, label]);
        target.appendLabelChange(entryId, label);
      };
      // B starts second and loads X=1; A must keep its own X=5.
      await runSessionStart(pi, ctxA, fixtureA);
      await runSessionStart(pi, ctxB, fixtureB);

      appendTurn(smA, "a-u1", "a-a1");
      appendTurn(smA, "a-u2", "a-a2");
      appendTurn(smB, "b-u1", "b-a1");

      fireTurnEnd(pi, ctxA, 1); // A: 2 < 5, no write
      assert.equal(writes.length, 0, "A must keep its X=5");

      target = smB;
      const leafB = smB.getLeafId();
      fireTurnEnd(pi, ctxB, 0); // B: 1 >= 1, writes at its first turn
      assert.deepEqual(writes, [[leafB, START_ANCHOR_LABEL]]);
      assert.equal(
        __testHooks.findLabeledEntry(smA, START_ANCHOR_LABEL),
        null,
        "A must stay unanchored",
      );
    } finally {
      cleanupConfigFixture(fixtureA);
      cleanupConfigFixture(fixtureB);
    }
  });

  it("per-session toolActive: a sibling session's inactive gate must not suppress this one", async () => {
    const fixture = makeConfigFixture();
    try {
      const smA = SessionManager.inMemory("/tmp");
      const smB = SessionManager.inMemory("/tmp");
      const pi = makeFakePi(smA);
      navigateTree(pi.pi);
      const ctxA = makeCtx(smA);
      const ctxB = makeCtx(smB);
      const writes: Array<[string, string | undefined]> = [];
      let target = smA;
      (
        pi.pi as unknown as {
          setLabel: (entryId: string, label: string | undefined) => void;
        }
      ).setLabel = (entryId, label) => {
        writes.push([entryId, label]);
        target.appendLabelChange(entryId, label);
      };
      await runSessionStart(pi, ctxA, fixture);
      await runSessionStart(pi, ctxB, fixture);
      appendTurn(smA, "a-u1", "a-a1");
      appendTurn(smA, "a-u2", "a-a2");
      appendTurn(smB, "b-u1", "b-a1");
      appendTurn(smB, "b-u2", "b-a2");

      // A has the tool active; B is a restricted sibling session.
      await fireBeforeAgentStart(pi, ctxA, ["read", TOOL_NAME]);
      await fireBeforeAgentStart(pi, ctxB, ["read"]);

      target = smA;
      fireTurnEnd(pi, ctxA, 1);
      assert.equal(writes.length, 1, "A must write despite B's inactive gate");
      target = smB;
      fireTurnEnd(pi, ctxB, 1);
      assert.equal(writes.length, 1, "B must stay gated");
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("per-session keying: interleaved sessions settle independently", async () => {
    const fixture = makeConfigFixture();
    try {
      const smA = SessionManager.inMemory("/tmp");
      const smB = SessionManager.inMemory("/tmp");
      const pi = makeFakePi(smA);
      navigateTree(pi.pi);
      const ctxA = makeCtx(smA);
      const ctxB = makeCtx(smB);
      const writes: Array<[string, string | undefined]> = [];
      let target = smA;
      (
        pi.pi as unknown as {
          setLabel: (entryId: string, label: string | undefined) => void;
        }
      ).setLabel = (entryId, label) => {
        writes.push([entryId, label]);
        target.appendLabelChange(entryId, label);
      };
      await runSessionStart(pi, ctxA, fixture);
      await runSessionStart(pi, ctxB, fixture);

      appendTurn(smA, "a-u1", "a-a1");
      appendTurn(smB, "b-u1", "b-a1");
      appendTurn(smA, "a-u2", "a-a2");
      appendTurn(smB, "b-u2", "b-a2");

      // B settles first; that must not suppress A's independent settle.
      target = smB;
      fireTurnEnd(pi, ctxB, 1);
      assert.equal(writes.length, 1);
      target = smA;
      fireTurnEnd(pi, ctxA, 1);
      assert.equal(writes.length, 2);

      // Both settled: interleaved re-fires are no-ops.
      fireTurnEnd(pi, ctxB, 1);
      fireTurnEnd(pi, ctxA, 1);
      fireAgentEnd(pi, ctxB);
      fireAgentEnd(pi, ctxA);
      assert.equal(writes.length, 2);
      assert.ok(__testHooks.findLabeledEntry(smA, START_ANCHOR_LABEL));
      assert.ok(__testHooks.findLabeledEntry(smB, START_ANCHOR_LABEL));
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("anchor(name='start') is refused when the automatic label exists (reservation moot)", async () => {
    const fixture = makeConfigFixture();
    try {
      const { sm, pi, tool, ctx } = setup();
      await runSessionStart(pi, ctx, fixture);
      appendTurn(sm, "u1", "a1");
      appendTurn(sm, "u2", "a2");
      fireTurnEnd(pi, ctx, 1);
      assert.equal(pi.setLabelCalls.length, 1);
      appendTurn(sm, "u3", "a3");
      const result = await tool.execute(
        "tc-1",
        { action: "anchor", name: "start" },
        undefined,
        undefined,
        ctx,
      );
      assert.equal(result.isError, true);
      assert.match(
        result.content[0].text,
        /A label 'start' already exists on the active branch/,
      );
    } finally {
      cleanupConfigFixture(fixture);
    }
  });
});
