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
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, it } from "node:test";
import {
  type AgentSession,
  CONFIG_DIR_NAME,
  type ContextUsage,
  type ExtensionAPI,
  SessionManager,
  SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { visibleWidth } from "@earendil-works/pi-tui";
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
  buildRewindAcceptedText,
  buildRewindDoneText,
  buildRewindFailedText,
  buildRewindOutcomeRow,
  buildSummaryFailedText,
  buildSummaryRequestRow,
  extractSummaryText,
  MAX_HINT_WALK_DEPTH,
  MAX_SESSION_REFS,
  MAX_SUMMARY_ATTEMPTS,
  MIN_REWIND_SAVINGS_TOKENS,
  MIN_SUMMARY_FOCUS_LENGTH,
  NESTED_REWIND_REFUSAL,
  REWIND_ABORTED_TEXT,
  REWIND_CUSTOM_TYPE,
  REWIND_OUTCOME_CUSTOM_TYPE,
  SOLO_REWIND_REFUSAL,
  START_ANCHOR_LABEL,
  SUMMARY_PENDING_BLOCK_REASON,
  SUMMARY_REQUEST_CUSTOM_TYPE,
  wrapPlainLine,
} from "./index.ts";
import {
  buildForcedRewindText,
  buildNoAnchorText,
  buildReserveOrderWarning,
  buildRewindCompactionFailedText,
  buildRewindHintText,
  buildWithheldNoticeText,
  COMPACTION_FAILED_ABORTED,
  COMPACTION_FAILED_NO_ANCHOR,
  COMPACTION_FAILED_NO_REWIND,
  COMPACTION_FAILED_OVERFLOWED_AGAIN,
  COMPACTION_FAILED_RUN_ENDED,
  COMPACTION_FAILED_STILL_OVER,
  FORCED_REWIND_BLOCK_REASON,
  REWIND_COMPACTION_OFF_TEXT,
  REWIND_HINT_CUSTOM_TYPE,
} from "./rewind-hint.ts";
import {
  BRANCH_SUMMARY_PREAMBLE,
  buildStoredSummary,
  buildSummaryInstruction,
  buildSummaryRequestText,
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
  /** `appendEntry(customType, data)` calls (routed to `sm.appendCustomEntry`). */
  appendEntryCalls: Array<[string, unknown]>;
  /** `registerMessageRenderer` registrations, keyed by customType. */
  messageRenderers: Map<string, (...args: never[]) => unknown>;
  /** `registerEntryRenderer` registrations, keyed by customType. */
  entryRenderers: Map<string, (...args: never[]) => unknown>;
}

/**
 * A boundary entry draft (pi's `SessionBoundaryDraft` union, loosely typed):
 * `custom_message` (customType/content/display/details), `custom`
 * (customType/data), `context_edit` (targetId/replacement) or `compaction`.
 */
interface BoundaryDraft {
  type: string;
  customType: string;
  content: string;
  display: boolean;
  details: Record<string, unknown>;
  data?: Record<string, unknown>;
  targetId?: string;
  replacement?: { content: unknown } | null;
  summary?: string;
  firstKeptEntryId?: string | null;
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
  const appendEntryCalls: Array<[string, unknown]> = [];
  const messageRenderers = new Map<string, (...args: never[]) => unknown>();
  const entryRenderers = new Map<string, (...args: never[]) => unknown>();
  const pi = {
    appendEntry(customType: string, data?: unknown) {
      appendEntryCalls.push([customType, data]);
      sm.appendCustomEntry(customType, data);
    },
    registerMessageRenderer(
      customType: string,
      renderer: (...args: never[]) => unknown,
    ) {
      messageRenderers.set(customType, renderer);
    },
    registerEntryRenderer(
      customType: string,
      renderer: (...args: never[]) => unknown,
    ) {
      entryRenderers.set(customType, renderer);
    },
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
    appendEntryCalls,
    messageRenderers,
    entryRenderers,
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
  /** The current abort signal (pi: undefined when not streaming). */
  signal: AbortSignal | undefined;
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
    signal: undefined,
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
 * Commit boundary drafts the way pi's `_commitBoundaryDrafts` does: all or
 * nothing. pi previews the whole list (`runner.emitBoundary` →
 * `buildContext`), and a `context_edit` whose target is off the active
 * branch invalidates it, dropping every draft. Returns false when dropped.
 */
function commitBoundaryDrafts(
  sm: SessionManager,
  drafts: readonly BoundaryDraft[],
): boolean {
  let onBranch: Set<string>;
  try {
    onBranch = new Set(sm.getBranch().map((e) => e.id));
  } catch {
    return false;
  }
  if (
    drafts.some(
      (d) => d.type === "context_edit" && !onBranch.has(d.targetId ?? ""),
    )
  ) {
    return false;
  }
  for (const draft of drafts) {
    switch (draft.type) {
      case "custom_message":
        sm.appendCustomMessageEntry(
          draft.customType,
          draft.content,
          draft.display,
          draft.details,
        );
        break;
      case "custom":
        sm.appendCustomEntry(draft.customType, draft.data);
        break;
      case "context_edit":
        sm.appendContextEdit(
          draft.targetId as string,
          (draft.replacement ?? null) as never,
        );
        break;
      case "compaction":
        sm.appendCompaction(
          draft.summary ?? "",
          (draft.firstKeptEntryId ?? null) as never,
          0,
          draft.details,
          true,
        );
        break;
    }
  }
  return true;
}

/**
 * The turn's persisted entry ids, as pi resolves them for `turn_end`: the
 * assistant's entry (`messageEntryId`) and the tool results that follow it
 * on the branch (`toolResults` / `toolResultEntryIds`).
 */
function resolveTurnEntries(
  sm: SessionManager,
  message: unknown,
): {
  messageEntryId: string | undefined;
  toolResults: unknown[];
  toolResultEntryIds: string[];
} {
  let branch: ReturnType<SessionManager["getBranch"]>;
  try {
    branch = sm.getBranch();
  } catch {
    // A test that stubs getBranch to throw drives the handlers anyway.
    branch = [];
  }
  const index = branch.findIndex(
    (e) => e.type === "message" && (e.message as unknown) === message,
  );
  if (index < 0) {
    return {
      messageEntryId: undefined,
      toolResults: [],
      toolResultEntryIds: [],
    };
  }
  const toolResults: unknown[] = [];
  const toolResultEntryIds: string[] = [];
  for (const entry of branch.slice(index + 1)) {
    if (entry.type !== "message" || entry.message.role !== "toolResult") break;
    toolResults.push(entry.message);
    toolResultEntryIds.push(entry.id);
  }
  return { messageEntryId: branch[index].id, toolResults, toolResultEntryIds };
}

/**
 * Fold boundary handlers exactly as pi's `ExtensionRunner.emitBoundary`
 * does: every handler sees the running `{ entries, continue }`, and a result
 * carrying `entries` / `continue` replaces it. Then commit the drafts
 * all-or-nothing (pi's `_commitBoundaryDrafts`); an invalid list folds to
 * `{ entries: [], continue: false }` like pi's preview failure.
 */
async function foldBoundary(
  handlers: Array<(e: never, c: never) => unknown>,
  ctx: FakeCtx,
  base: Record<string, unknown>,
  opts: { entries?: BoundaryDraft[]; continue?: boolean; commit?: boolean },
): Promise<BoundaryResult> {
  let entries = opts.entries ?? [];
  let shouldContinue = opts.continue ?? false;
  for (const handler of handlers) {
    const result = (await handler(
      { ...base, entries, continue: shouldContinue } as never,
      ctx as never,
    )) as { entries?: BoundaryDraft[]; continue?: boolean } | undefined;
    if (result?.entries !== undefined) entries = result.entries;
    if (result?.continue !== undefined) shouldContinue = result.continue;
  }
  if (opts.commit === false) return { entries, continue: shouldContinue };
  if (!commitBoundaryDrafts(ctx.sessionManager, entries)) {
    return { entries: [], continue: false };
  }
  return { entries, continue: shouldContinue };
}

/**
 * Fire every registered `turn_end` handler in registration order, awaiting
 * each, with the event fields pi passes (`messageEntryId`, `toolResults`,
 * `toolResultEntryIds`, `outcome`), and fold + commit their results (see
 * `foldBoundary`). The folded result is returned and recorded on
 * `pi.boundaryResults`. `entries` seeds the list (another extension's drafts
 * earlier in the chain). `commit: false` skips pi's commit. `turnIndex`
 * defaults to 0 so the auto start-anchor (default X=2) stays out of the way.
 */
async function fireTurnEndAll(
  pi: FakePi,
  ctx: FakeCtx,
  message: unknown,
  opts: {
    turnIndex?: number;
    entries?: BoundaryDraft[];
    commit?: boolean;
    messageEntryId?: string;
  } = {},
): Promise<BoundaryResult> {
  const handlers = pi.onCalls.get("turn_end");
  assert.ok(handlers, "factory must register turn_end handlers");
  const resolved = resolveTurnEntries(ctx.sessionManager, message);
  const stopReason = (message as { stopReason?: string } | undefined)
    ?.stopReason;
  const folded = await foldBoundary(
    handlers,
    ctx,
    {
      type: "turn_end",
      turnIndex: opts.turnIndex ?? 0,
      message,
      toolResults: resolved.toolResults,
      messageEntryId: opts.messageEntryId ?? resolved.messageEntryId,
      toolResultEntryIds: resolved.toolResultEntryIds,
      outcome:
        stopReason === "aborted"
          ? "aborted"
          : stopReason === "error"
            ? "error"
            : "completed",
    },
    opts,
  );
  pi.boundaryResults.push(folded);
  return folded;
}

/** Fire every `agent_before_settle` handler (a boundary, folded + committed). */
async function fireAgentBeforeSettle(
  pi: FakePi,
  ctx: FakeCtx,
  opts: {
    outcome?: "completed" | "aborted" | "error";
    continue?: boolean;
    entries?: BoundaryDraft[];
    commit?: boolean;
  } = {},
): Promise<BoundaryResult> {
  const handlers = pi.onCalls.get("agent_before_settle") ?? [];
  const folded = await foldBoundary(
    handlers,
    ctx,
    { type: "agent_before_settle", outcome: opts.outcome ?? "completed" },
    opts,
  );
  pi.boundaryResults.push(folded);
  return folded;
}

/** Fire every `agent_settled` handler. */
async function fireAgentSettled(
  pi: FakePi,
  ctx: FakeCtx,
  aborted = false,
): Promise<void> {
  for (const handler of pi.onCalls.get("agent_settled") ?? []) {
    await handler({ type: "agent_settled", aborted } as never, ctx as never);
  }
}

/** Append a compaction (as pi does) and fire every `session_compact` handler. */
async function fireSessionCompact(
  pi: FakePi,
  ctx: FakeCtx,
  firstKeptEntryId: string,
  reason: "manual" | "threshold" | "overflow" = "threshold",
): Promise<string> {
  const sm = ctx.sessionManager;
  const id = sm.appendCompaction("compacted", firstKeptEntryId, 1000);
  for (const handler of pi.onCalls.get("session_compact") ?? []) {
    await handler(
      {
        type: "session_compact",
        compactionEntry: sm.getEntry(id),
        fromExtension: false,
        reason,
        willRetry: reason === "overflow",
      } as never,
      ctx as never,
    );
  }
  return id;
}

/** Fire the `context` handler and return the messages it would send. */
function fireContext(pi: FakePi, ctx: FakeCtx): Array<Record<string, unknown>> {
  const handlers = pi.onCalls.get("context");
  assert.ok(handlers, "factory must register a context handler");
  assert.equal(handlers.length, 1);
  const result = handlers[0](
    { type: "context", messages: [] } as never,
    ctx as never,
  ) as { messages: Array<Record<string, unknown>> };
  return result.messages;
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
  totalTokens?: number,
) {
  const call = appendToolCallAssistant(
    f.sm,
    [{ id: toolCallId, name: TOOL_NAME, arguments: args }],
    totalTokens,
  );
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
  const boundary = await fireTurnEndAll(f.pi, f.ctx, message, {
    commit: opts.commit ?? false,
  });
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

/** The latest summary-instruction draft returned by a boundary (#90). */
function summaryRequestDraftOf(pi: FakePi): BoundaryDraft {
  const drafts = pi.boundaryResults.flatMap((r) =>
    r.entries.filter((d) => d.customType === SUMMARY_REQUEST_CUSTOM_TYPE),
  );
  const draft = drafts.at(-1);
  assert.ok(draft, "a summary instruction draft was returned");
  return draft;
}

/** Outcome entries (`navigate-tree-rewind-outcome`) on the active branch. */
function outcomeEntries(sm: SessionManager): Array<Record<string, unknown>> {
  return sm
    .getBranch()
    .filter(
      (e) => e.type === "custom" && e.customType === REWIND_OUTCOME_CUSTOM_TYPE,
    )
    .map((e) => (e as { data: Record<string, unknown> }).data);
}

/** Ids hidden (last `context_edit` null) on the active branch. */
function hiddenIds(sm: SessionManager): Set<string> {
  const last = new Map<string, unknown>();
  for (const e of sm.getBranch()) {
    if (e.type === "context_edit") last.set(e.targetId, e.replacement);
  }
  return new Set([...last].filter(([, r]) => r === null).map(([id]) => id));
}

/**
 * A failed summary attempt that calls tools: persist the assistant, the
 * blocked tool results pi records for it, and fire `turn_end`.
 */
async function toolCallAttempt(
  f: { sm: SessionManager; pi: FakePi; ctx: FakeCtx },
  ids: string[] = ["tc-read"],
  stopReason = "toolUse",
): Promise<{ entryId: string; resultIds: string[]; boundary: BoundaryResult }> {
  const message = assistantMessage(
    ids.map((id) => ({
      type: "toolCall",
      id,
      name: "read",
      arguments: { path: `src/${id}.ts` },
    })),
    stopReason,
    100,
  );
  const entryId = f.sm.appendMessage(message as never);
  const resultIds = ids.map((id) =>
    appendToolResult(f.sm, id, "read", SUMMARY_PENDING_BLOCK_REASON),
  );
  const boundary = await fireTurnEndAll(f.pi, f.ctx, message);
  return { entryId, resultIds, boundary };
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

  it("renders the shared anchor table with an aligned (after:) column for a 40-char name (#101 D19)", async () => {
    const { sm, pi, tool, ctx } = setup();
    const long = "a".repeat(MAX_NAME_LENGTH);
    const t1 = appendTurn(sm, "u1", "first reply", 100);
    pi.pi.setLabel(t1.assistantId, "anchor:start");
    const t2 = appendTurn(sm, "u2", "second reply", 200);
    pi.pi.setLabel(t2.assistantId, `anchor:${long}`);
    const result = await tool.execute(
      "tc-1",
      { action: "list" },
      undefined,
      undefined,
      ctx,
    );
    const body = result.content[0].text.split("\n\n")[1];
    assert.equal(
      body,
      [
        "Active anchors (root → leaf, % of 1.0M):",
        `   0.0%  ${"start".padEnd(MAX_NAME_LENGTH)}  (after: “first reply”)`,
        `   0.0%  ${long}  (after: “second reply”)`,
      ].join("\n"),
    );
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
  it("returns only the byte-exact 'accepted' text and the request details (#90 S2)", async () => {
    const { sm, pi, tool, ctx } = setup();
    const { anchorId } = setupRewindable(sm, pi);
    appendToolCallAssistant(sm, [
      { id: "tc-rewind", name: TOOL_NAME, arguments: REWIND_ARGS },
    ]);
    const result = await tool.execute(
      "tc-rewind",
      REWIND_ARGS,
      undefined,
      undefined,
      ctx,
    );
    assert.equal(result.isError, undefined);
    assert.equal(result.content.length, 1);
    assert.equal(
      result.content[0].text,
      "[navigate_tree] Rewind to 'start' accepted (collapse as 'end'). Follow the navigate_tree message below to complete it.",
    );
    assert.equal(
      result.content[0].text,
      buildRewindAcceptedText("start", "end"),
    );
    // u1 + a1 are background → the collapsed segment starts at message 3.
    assert.deepEqual(result.details, {
      status: "summary-requested",
      rewindTo: "start",
      newLabel: "end",
      targetId: anchorId,
      first: 3,
      contextBefore: result.details.contextBefore,
      tokensAtTarget: result.details.tokensAtTarget,
      contextWindow: 1_000_000,
    });
    assert.equal(typeof result.details.contextBefore, "number");
    assert.equal(typeof result.details.tokensAtTarget, "number");
  });

  it("the rewind turn's turn_end returns the instruction custom message + continue (#90 S1/S3)", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi);
    const { result } = await requestRewind(f);
    const boundary = f.pi.boundaryResults.at(-1);
    assert.ok(boundary);
    assert.equal(boundary.continue, true);
    assert.equal(boundary.entries.length, 1);
    const draft = boundary.entries[0];
    assert.equal(draft.type, "custom_message");
    assert.equal(draft.customType, SUMMARY_REQUEST_CUSTOM_TYPE);
    assert.equal(draft.display, true);
    assert.equal(
      draft.content,
      `[navigate_tree]\n${buildSummaryInstruction(3, REWIND_ARGS.summaryFocus)}`,
    );
    assert.equal(
      draft.content,
      buildSummaryRequestText(3, REWIND_ARGS.summaryFocus),
    );
    assert.deepEqual(draft.details, {
      toolCallId: "tc-rewind",
      rewindTo: "start",
      newLabel: "end",
      contextBefore: result.details.contextBefore,
      tokensAtTarget: result.details.tokensAtTarget,
      contextWindow: 1_000_000,
      first: 3,
    });
    // Committed right after the rewind's tool result.
    const leaf = f.sm.getEntry(f.sm.getLeafId() as string);
    assert.ok(leaf && leaf.type === "custom_message");
    assert.equal(leaf.customType, SUMMARY_REQUEST_CUSTOM_TYPE);
    const parent = f.sm.getEntry(leaf.parentId as string);
    assert.ok(parent && parent.type === "message");
    assert.equal(parent.message.role, "toolResult");
  });

  it("Esc during the rewind turn ends it aborted: no instruction, no continue (#90 O6)", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi);
    const controller = new AbortController();
    controller.abort();
    f.ctx.signal = controller.signal;
    await requestRewind(f);
    const boundary = f.pi.boundaryResults.at(-1);
    assert.ok(boundary);
    assert.equal(boundary.continue, false);
    assert.ok(
      !boundary.entries.some(
        (d) => d.customType === SUMMARY_REQUEST_CUSTOM_TYPE,
      ),
    );
    const outcome = outcomeEntries(f.sm);
    assert.equal(outcome.length, 1);
    assert.equal(outcome[0].status, "aborted");
    assert.equal(
      outcome[0].text,
      "Summarization aborted. Nothing was collapsed.",
    );
    assert.equal(fireToolCall(f.pi, f.ctx), undefined, "nothing pending");
    // The tool result now reads as the abort, as an error.
    f.ctx.signal = undefined;
    const sent = fireContext(f.pi, f.ctx);
    const toolResult = sent.find((m) => m.role === "toolResult");
    assert.ok(toolResult);
    assert.equal(toolResult.isError, true);
    assert.deepEqual(toolResult.content, [
      { type: "text", text: "Summarization aborted. Nothing was collapsed." },
    ]);
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

  it("interpolates the focus verbatim into the instruction (no truncation, no {first} left over)", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi);
    const focus = `user said: ship it. ${"detail ".repeat(400)}`.trim();
    const { result } = await requestRewind(f, {
      ...REWIND_ARGS,
      summaryFocus: focus,
    });
    assert.equal(result.isError, undefined);
    const draft = summaryRequestDraftOf(f.pi);
    assert.ok(draft.content.endsWith(`Additional focus: ${focus}`));
    assert.doesNotMatch(draft.content, /\{first\}/);
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
    assert.equal(
      SUMMARY_PENDING_BLOCK_REASON,
      "navigate_tree: blocked: the model tried to call a tool instead of summarizing.",
    );
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
    // call, its result, the instruction message, and the summary turn — all
    // left on the abandoned branch.
    assert.equal(d.collapsedEntries, 11);
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

describe("rewind phase 2: failed attempts are hidden and re-sent (#90)", () => {
  async function pending() {
    const f = setup();
    const { anchorId } = setupRewindable(f.sm, f.pi, { capture: true });
    const { callId } = await requestRewind(f);
    return { ...f, anchorId, callId };
  }

  function assertNotBranched(sm: SessionManager): void {
    assert.ok(
      !sm.getEntries().some((e) => e.type === "branch_summary"),
      "no branch_summary may land",
    );
  }

  /** A committed text attempt (pi persists it, then commits the drafts). */
  function attempt(
    f: { sm: SessionManager; pi: FakePi; ctx: FakeCtx },
    text: string,
    stopReason = "stop",
  ) {
    return summaryTurn(f, text, stopReason, { commit: true });
  }

  function rewindResultEntryId(sm: SessionManager): string {
    const entry = sm
      .getBranch()
      .find(
        (e) =>
          e.type === "message" &&
          e.message.role === "toolResult" &&
          e.message.toolCallId === "tc-rewind",
      );
    assert.ok(entry);
    return entry.id;
  }

  function instructionEntryId(sm: SessionManager): string {
    const entry = sm
      .getBranch()
      .find(
        (e) =>
          e.type === "custom_message" &&
          e.customType === SUMMARY_REQUEST_CUSTOM_TYPE,
      );
    assert.ok(entry);
    return entry.id;
  }

  /**
   * §15.2/§15.4: once a rewind has ended, what pi stores (the projection
   * that `{first}`, estimates and compaction read) and what the `context`
   * handler sends agree, except the `isError` flag on the rewind's result.
   */
  function assertProjectionAgrees(f: {
    sm: SessionManager;
    pi: FakePi;
    ctx: FakeCtx;
  }): void {
    const stored = f.sm
      .buildSessionProjection()
      .messages.filter((m) => m.role !== "system")
      .map((m) =>
        m.role === "toolResult" && m.toolCallId === "tc-rewind"
          ? { ...m, isError: true }
          : m,
      );
    assert.deepEqual(fireContext(f.pi, f.ctx), stored);
  }

  it("a text attempt without '## Goal' hides only that assistant and re-sends, nothing appended", async () => {
    const f = await pending();
    const before = fireContext(f.pi, f.ctx);
    const { entryId, boundary } = await attempt(f, "Okay, I'll summarize now.");
    assert.equal(boundary.continue, true);
    assert.deepEqual(boundary.entries, [
      { type: "context_edit", targetId: entryId, replacement: null },
    ]);
    assert.deepEqual([...hiddenIds(f.sm)], [entryId]);
    assert.equal(outcomeEntries(f.sm).length, 0);
    assertNotBranched(f.sm);
    assert.ok(fireToolCall(f.pi, f.ctx), "still pending");
    assert.deepEqual(fireContext(f.pi, f.ctx), before, "identical retry");
  });

  it("a tool-call attempt hides the assistant and ALL its tool results together", async () => {
    const f = await pending();
    const { entryId, resultIds, boundary } = await toolCallAttempt(f, [
      "tc-a",
      "tc-b",
    ]);
    assert.equal(boundary.continue, true);
    assert.deepEqual(
      boundary.entries.map((d) => [d.type, d.targetId, d.replacement]),
      [entryId, ...resultIds].map((id) => ["context_edit", id, null]),
    );
    assertNotBranched(f.sm);
  });

  it("identical retries: a user steer and an extension message mid-pending never reach the request", async () => {
    const f = await pending();
    const before = fireContext(f.pi, f.ctx);
    // Steer drained after the instruction + another extension's message.
    f.sm.appendMessage({
      role: "user",
      content: [{ type: "text", text: "also check X" }],
      timestamp: Date.now(),
    } as never);
    f.sm.appendCustomMessageEntry("subagent-result", "done", true);
    assert.deepEqual(fireContext(f.pi, f.ctx), before, "attempt 1 request");
    await toolCallAttempt(f);
    assert.deepEqual(fireContext(f.pi, f.ctx), before, "attempt 2 request");
    await attempt(f, "no heading");
    assert.deepEqual(fireContext(f.pi, f.ctx), before, "attempt 3 request");
    // The request ends at the instruction.
    const last = before.at(-1) as { role?: string; customType?: string };
    assert.equal(last.role, "custom");
    assert.equal(last.customType, SUMMARY_REQUEST_CUSTOM_TYPE);
  });

  const causes: Array<{
    name: string;
    run: (f: Awaited<ReturnType<typeof pending>>) => Promise<unknown>;
    text: string;
    reason: string;
  }> = [
    {
      name: "cut off",
      run: (f) => attempt(f, SUMMARY_TEXT, "length"),
      text: "Summarization failed: no summary after 3 attempts (last: cut off). Nothing was collapsed. Try again with a tighter summaryFocus.",
      reason: "no summary after 3 attempts (last: cut off)",
    },
    {
      name: "called a tool",
      run: (f) => toolCallAttempt(f, [`tc-${Math.random()}`]),
      text: "Summarization failed: no summary after 3 attempts (last: called a tool). Nothing was collapsed. Try again; the reply after the rewind must be only the summary.",
      reason: "no summary after 3 attempts (last: called a tool)",
    },
    {
      name: "no '## Goal' summary",
      run: (f) => attempt(f, "Sure, summarizing."),
      text: "Summarization failed: no summary after 3 attempts (last: no '## Goal' summary). Nothing was collapsed. Try again; the reply after the rewind must be only the summary.",
      reason: "no summary after 3 attempts (last: no '## Goal' summary)",
    },
  ];
  for (const c of causes) {
    it(`3rd failure (last: ${c.name}) ends the rewind as a failed tool call (O5)`, async () => {
      const f = await pending();
      const resultId = rewindResultEntryId(f.sm);
      const instructionId = instructionEntryId(f.sm);
      // Attempts 1-2 use another cause: the LAST attempt names the ending.
      await attempt(f, "first try");
      await toolCallAttempt(f, ["tc-2"]);
      await c.run(f);
      const boundary = f.pi.boundaryResults.at(-1);
      assert.ok(boundary);
      assert.equal(boundary.continue, true);
      assert.equal(c.text, buildSummaryFailedText(c.name as never));
      const outcome = outcomeEntries(f.sm);
      assert.equal(outcome.length, 1);
      assert.equal(outcome[0].status, "failed");
      assert.equal(outcome[0].text, c.text);
      assert.equal(outcome[0].reason, c.reason);
      assert.equal(outcome[0].toolCallId, "tc-rewind");
      assert.equal(outcome[0].toolResultEntryId, resultId);
      assert.ok(hiddenIds(f.sm).has(instructionId), "instruction hidden");
      // The tool result now reads as the failure, as an error (clone).
      const sent = fireContext(f.pi, f.ctx);
      const result = sent.find(
        (m) => m.role === "toolResult" && m.toolCallId === "tc-rewind",
      );
      assert.ok(result);
      assert.equal(result.isError, true);
      assert.deepEqual(result.content, [{ type: "text", text: c.text }]);
      assert.ok(
        !sent.some(
          (m) =>
            (m as { customType?: string }).customType ===
            SUMMARY_REQUEST_CUSTOM_TYPE,
        ),
      );
      assert.ok(
        !sent.some(
          (m, i) => i > sent.indexOf(result) && m.role === "assistant",
        ),
        "every attempt hidden",
      );
      // The session's own message objects are never mutated.
      const stored = f.sm.getEntry(resultId);
      assert.ok(stored && stored.type === "message");
      assert.equal((stored.message as { isError?: boolean }).isError, false);
      assertProjectionAgrees(f);
      assertNotBranched(f.sm);
      assert.equal(fireToolCall(f.pi, f.ctx), undefined, "gate re-opened");
      // A later valid summary turn no longer completes anything.
      await attempt(f, SUMMARY_TEXT);
      assertNotBranched(f.sm);
    });
  }

  it("a cut-off reply that also calls a tool counts as 'cut off'", async () => {
    const f = await pending();
    await attempt(f, "x");
    await attempt(f, "y");
    await toolCallAttempt(f, ["tc-cut"], "length");
    assert.equal(
      outcomeEntries(f.sm)[0].text,
      buildSummaryFailedText("cut off"),
    );
  });

  it("a summary plus a tool call is a failure ('called a tool')", async () => {
    const f = await pending();
    const message = assistantMessage(
      [
        { type: "text", text: SUMMARY_TEXT },
        { type: "toolCall", id: "tc-x", name: "bash", arguments: {} },
      ],
      "toolUse",
    );
    const entryId = f.sm.appendMessage(message as never);
    const resultId = appendToolResult(f.sm, "tc-x", "bash", "blocked");
    await fireTurnEndAll(f.pi, f.ctx, message);
    assertNotBranched(f.sm);
    assert.deepEqual(hiddenIds(f.sm), new Set([entryId, resultId]));
    assert.ok(fireToolCall(f.pi, f.ctx));
  });

  it("late input becomes visible again after a failure, right after the failed result (O1b)", async () => {
    const f = await pending();
    const steerId = f.sm.appendMessage({
      role: "user",
      content: [{ type: "text", text: "also check X" }],
      timestamp: Date.now(),
    } as never);
    for (let i = 0; i < MAX_SUMMARY_ATTEMPTS; i++) await attempt(f, `a${i}`);
    const sent = fireContext(f.pi, f.ctx);
    const resultAt = sent.findIndex(
      (m) => m.role === "toolResult" && m.toolCallId === "tc-rewind",
    );
    const steerAt = sent.findIndex(
      (m) =>
        m.role === "user" && JSON.stringify(m.content).includes("also check X"),
    );
    assert.ok(resultAt >= 0 && steerAt === resultAt + 1);
    assert.ok(!hiddenIds(f.sm).has(steerId));
  });

  it("an aborted attempt ends the rewind aborted: attempt hidden, no continue", async () => {
    const f = await pending();
    const { entryId, boundary } = await attempt(
      f,
      "## Goal\npartial",
      "aborted",
    );
    assert.equal(boundary.continue, false);
    const outcome = outcomeEntries(f.sm);
    assert.equal(outcome.length, 1);
    assert.equal(outcome[0].status, "aborted");
    assert.equal(outcome[0].text, REWIND_ABORTED_TEXT);
    assert.equal(
      REWIND_ABORTED_TEXT,
      "Summarization aborted. Nothing was collapsed.",
    );
    assert.ok(hiddenIds(f.sm).has(entryId));
    assertProjectionAgrees(f);
    assertNotBranched(f.sm);
    assert.equal(fireToolCall(f.pi, f.ctx), undefined);
  });

  it("an error turn keeps the rewind pending without spending an attempt", async () => {
    const f = await pending();
    const { boundary } = await attempt(f, "", "error");
    assert.deepEqual(boundary.entries, []);
    assert.ok(fireToolCall(f.pi, f.ctx), "still pending after an error");
    await summaryTurn(f);
    const done = rewindMessages(f.pi)[0].message;
    assert.equal(done.details.status, "done");
    assert.equal(done.details.summaryAttempts, 1);
  });

  it("pi-fallback-provider coexistence: its null edit on the errored attempt + ours stay valid; not counted", async () => {
    const f = await pending();
    const errored = await attempt(f, "", "error");
    // The fallback provider hides the errored attempt at agent_before_settle
    // and continues on another model.
    const settle = await fireAgentBeforeSettle(f.pi, f.ctx, {
      outcome: "error",
      continue: true,
      entries: [
        {
          type: "context_edit",
          targetId: errored.entryId,
          replacement: null,
        } as BoundaryDraft,
      ],
    });
    assert.equal(settle.continue, true);
    assert.equal(settle.entries.length, 1, "the fallback's edit survives");
    assert.ok(hiddenIds(f.sm).has(errored.entryId));
    assert.ok(fireToolCall(f.pi, f.ctx), "still pending");
    const failed = await attempt(f, "no heading");
    // Same target hidden again by both: still a valid list.
    const both = await fireTurnEndAll(f.pi, f.ctx, failed.message, {
      entries: [
        {
          type: "context_edit",
          targetId: failed.entryId,
          replacement: null,
        } as BoundaryDraft,
      ],
    });
    assert.equal(both.entries.length, 2);
    await summaryTurn(f);
    const done = rewindMessages(f.pi).at(-1)?.message;
    assert.equal(done?.details.status, "done");
  });

  it("agent_end keeps a pending rewind (agent_before_settle decides)", async () => {
    const f = await pending();
    fireAgentEnd(f.pi, f.ctx);
    assert.ok(fireToolCall(f.pi, f.ctx), "still pending");
  });

  it("agent_before_settle ends a pending rewind: 'the run ended before the summary', continue", async () => {
    const f = await pending();
    const result = await fireAgentBeforeSettle(f.pi, f.ctx);
    assert.equal(result.continue, true);
    const outcome = outcomeEntries(f.sm);
    assert.equal(outcome.length, 1);
    assert.equal(outcome[0].status, "failed");
    assert.equal(
      outcome[0].text,
      "Summarization failed: the run ended before the summary. Nothing was collapsed. Try again.",
    );
    assert.equal(
      outcome[0].text,
      buildRewindFailedText("the run ended before the summary"),
    );
    assert.ok(hiddenIds(f.sm).has(instructionEntryId(f.sm)));
    assertProjectionAgrees(f);
    assert.equal(fireToolCall(f.pi, f.ctx), undefined);
  });

  it("agent_before_settle leaves it pending on an error outcome or an explicit continue", async () => {
    const f = await pending();
    await fireAgentBeforeSettle(f.pi, f.ctx, { outcome: "error" });
    assert.ok(fireToolCall(f.pi, f.ctx));
    await fireAgentBeforeSettle(f.pi, f.ctx, { continue: true });
    assert.ok(fireToolCall(f.pi, f.ctx));
    assert.equal(outcomeEntries(f.sm).length, 0);
  });

  it("Esc while a blocked attempt's tools resolve ends it aborted, not failed (O6)", async () => {
    const f = await pending();
    const controller = new AbortController();
    controller.abort();
    f.ctx.signal = controller.signal;
    const { entryId, resultIds, boundary } = await toolCallAttempt(f);
    f.ctx.signal = undefined;
    assert.equal(boundary.continue, false);
    const outcome = outcomeEntries(f.sm);
    assert.equal(outcome.length, 1);
    assert.equal(outcome[0].status, "aborted");
    assert.equal(outcome[0].text, REWIND_ABORTED_TEXT);
    for (const id of [entryId, ...resultIds]) {
      assert.ok(hiddenIds(f.sm).has(id), "the aborted attempt is hidden");
    }
    assert.equal(fireToolCall(f.pi, f.ctx), undefined);
  });

  it("agent_before_settle on an aborted outcome ends it aborted without continue", async () => {
    const f = await pending();
    const result = await fireAgentBeforeSettle(f.pi, f.ctx, {
      outcome: "aborted",
    });
    assert.equal(result.continue, false);
    assert.equal(outcomeEntries(f.sm)[0].status, "aborted");
  });

  it("agent_settled with the rewind still pending (errored, no retry) ends it aborted outside a boundary", async () => {
    const f = await pending();
    const errored = await attempt(f, "", "error");
    await fireAgentBeforeSettle(f.pi, f.ctx, { outcome: "error" });
    await fireAgentSettled(f.pi, f.ctx);
    assert.equal(f.pi.appendEntryCalls.length, 1);
    assert.equal(f.pi.appendEntryCalls[0][0], REWIND_OUTCOME_CUSTOM_TYPE);
    const outcome = outcomeEntries(f.sm);
    assert.equal(outcome[0].status, "aborted");
    assert.equal(outcome[0].text, REWIND_ABORTED_TEXT);
    assert.ok(hiddenIds(f.sm).has(instructionEntryId(f.sm)));
    // The errored attempt is pi's to hide; ours are exact ids only.
    assert.ok(!hiddenIds(f.sm).has(errored.entryId));
    const replaced = f.sm
      .getBranch()
      .find(
        (e) =>
          e.type === "context_edit" && e.targetId === rewindResultEntryId(f.sm),
      );
    assert.ok(replaced, "the tool result rewrite is persisted");
    assert.equal(fireToolCall(f.pi, f.ctx), undefined);
  });

  it("agent_settled names the reason: aborted run vs an error with no retry", async () => {
    const f = await pending();
    await attempt(f, "", "error");
    await fireAgentSettled(f.pi, f.ctx, true);
    assert.equal(outcomeEntries(f.sm)[0].reason, "aborted");
    assert.equal(outcomeEntries(f.sm)[0].text, REWIND_ABORTED_TEXT);
    const g = await pending();
    await attempt(g, "", "error");
    await fireAgentSettled(g.pi, g.ctx, false);
    assert.equal(
      outcomeEntries(g.sm)[0].reason,
      "the run ended after an error",
    );
  });

  it("before_agent_start (a new prompt) ends a pending rewind aborted", async () => {
    const f = await pending();
    await attempt(f, "", "error");
    await fireBeforeAgentStart(f.pi, f.ctx, undefined);
    assert.equal(outcomeEntries(f.sm)[0].status, "aborted");
    assert.equal(fireToolCall(f.pi, f.ctx), undefined);
    await summaryTurn(f);
    assertNotBranched(f.sm);
  });

  it("session_compact while pending ends it before the next request (O7)", async () => {
    const f = await pending();
    await fireSessionCompact(f.pi, f.ctx, f.anchorId);
    const outcome = outcomeEntries(f.sm);
    assert.equal(outcome.length, 1);
    assert.equal(
      outcome[0].text,
      "Summarization failed: the context was compacted. Nothing was collapsed. Try again.",
    );
    assert.equal(fireToolCall(f.pi, f.ctx), undefined);
  });

  it("a compaction entry after the rewind call ends it at the next request / turn_end", async () => {
    const f = await pending();
    f.sm.appendCompaction("COMPACTED", f.anchorId, 20_000);
    fireContext(f.pi, f.ctx);
    assert.equal(outcomeEntries(f.sm)[0].reason, "the context was compacted");
    assert.equal(fireToolCall(f.pi, f.ctx), undefined);
  });

  it("a compaction seen first at the attempt's turn_end ends it there", async () => {
    const f = await pending();
    f.sm.appendCompaction("COMPACTED", f.anchorId, 20_000);
    const { boundary } = await attempt(f, SUMMARY_TEXT);
    assert.equal(boundary.continue, true);
    assert.equal(outcomeEntries(f.sm)[0].reason, "the context was compacted");
    assertNotBranched(f.sm);
  });

  it("instruction never committed: the request ends it 'could not be delivered' (§15.5)", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    // Another handler's invalid draft drops the whole list (instruction too).
    const call = appendToolCallAssistant(f.sm, [
      { id: "tc-rewind", name: TOOL_NAME, arguments: REWIND_ARGS },
    ]);
    const result = await f.tool.execute(
      "tc-rewind",
      REWIND_ARGS,
      undefined,
      undefined,
      f.ctx,
    );
    appendToolResult(f.sm, "tc-rewind", TOOL_NAME, result.content[0].text);
    await fireTurnEndAll(f.pi, f.ctx, call.message, { commit: false });
    assert.ok(fireToolCall(f.pi, f.ctx), "pending");
    const sent = fireContext(f.pi, f.ctx);
    const outcome = outcomeEntries(f.sm);
    assert.equal(outcome.length, 1);
    assert.equal(
      outcome[0].text,
      "Summarization failed: the summary request could not be delivered. Nothing was collapsed. Try again.",
    );
    const toolResult = sent.find(
      (m) => m.role === "toolResult" && m.toolCallId === "tc-rewind",
    );
    assert.equal(toolResult?.isError, true);
  });

  it("instruction never committed and no request seen: the attempt's turn_end ends it", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    const call = appendToolCallAssistant(f.sm, [
      { id: "tc-rewind", name: TOOL_NAME, arguments: REWIND_ARGS },
    ]);
    const result = await f.tool.execute(
      "tc-rewind",
      REWIND_ARGS,
      undefined,
      undefined,
      f.ctx,
    );
    appendToolResult(f.sm, "tc-rewind", TOOL_NAME, result.content[0].text);
    await fireTurnEndAll(f.pi, f.ctx, call.message, { commit: false });
    const { entryId } = await attempt(f, SUMMARY_TEXT);
    assert.equal(
      outcomeEntries(f.sm)[0].reason,
      "the summary request could not be delivered",
    );
    assert.ok(hiddenIds(f.sm).has(entryId));
    assertNotBranched(f.sm);
  });

  it("ends when the rewind call is no longer on the active branch", async () => {
    const f = await pending();
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
    const drafts = f.pi.boundaryResults.at(-1)?.entries ?? [];
    const outcome = drafts.find(
      (d) => d.customType === REWIND_OUTCOME_CUSTOM_TYPE,
    );
    assert.equal(
      outcome?.data?.reason,
      "the rewind call is no longer on the active branch",
    );
    assertNotBranched(f.sm);
  });

  it("a throwing branchWithSummary ends it failed and clears the branch: label", async () => {
    const f = await pending();
    (f.sm as unknown as { branchWithSummary: () => string }).branchWithSummary =
      () => {
        throw new Error("disk full");
      };
    const { entryId, boundary } = await attempt(f, SUMMARY_TEXT);
    assert.equal(boundary.continue, true);
    const outcome = outcomeEntries(f.sm);
    assert.equal(
      outcome[0].text,
      "Summarization failed: the branch could not be written (disk full). Nothing was collapsed. Try again.",
    );
    assert.equal(f.sm.getLabel(entryId), undefined, "branch: label cleared");
    assert.ok(hiddenIds(f.sm).has(entryId), "the summary turn is hidden too");
    assertNotBranched(f.sm);
    assert.equal(fireToolCall(f.pi, f.ctx), undefined, "gate reopened");
  });

  it("a branch_summary disk write that throws after pi moved the leaf restores the leaf before failing", async () => {
    const f = await pending();
    // pi's branchWithSummary moves the leaf and appends in memory, THEN the
    // file append throws (session-manager `_appendEntry` \u2192 `_persist`).
    const internals = f.sm as unknown as { _persist: (e: unknown) => void };
    const origPersist = internals._persist.bind(f.sm);
    internals._persist = (entry: unknown) => {
      if ((entry as { type?: string }).type === "branch_summary") {
        throw new Error("ENOSPC");
      }
      origPersist(entry);
    };
    const { entryId, boundary } = await attempt(f, SUMMARY_TEXT);
    assert.equal(boundary.continue, true);
    const branch = f.sm.getBranch();
    assert.ok(
      !branch.some((e) => e.type === "branch_summary"),
      "the active branch is not collapsed",
    );
    assert.ok(
      branch.some((e) => e.id === entryId),
      "the leaf is back on the summary turn's branch",
    );
    assert.ok(
      branch.some(
        (e) =>
          e.type === "label" &&
          e.targetId === entryId &&
          e.label === "branch:end",
      ),
      "restored to the leaf right before the branch call (label entry kept)",
    );
    const outcome = outcomeEntries(f.sm);
    assert.equal(outcome.length, 1);
    assert.equal(
      outcome[0].text,
      "Summarization failed: the branch could not be written (ENOSPC). Nothing was collapsed. Try again.",
    );
    assert.ok(hiddenIds(f.sm).has(entryId), "the summary turn is hidden");
    assert.ok(hiddenIds(f.sm).has(instructionEntryId(f.sm)));
    assert.equal(f.sm.getLabel(entryId), undefined, "branch: label cleared");
  });

  it("ends when the anchor no longer resolves to the same entry", async () => {
    const f = await pending();
    f.pi.pi.setLabel(f.anchorId, undefined);
    await attempt(f, SUMMARY_TEXT);
    assert.equal(
      outcomeEntries(f.sm)[0].text,
      "Summarization failed: 'start' is no longer on the active branch. Nothing was collapsed. Try again.",
    );
    assertNotBranched(f.sm);
  });

  it("ends when the anchor label moved to a different entry", async () => {
    const f = await pending();
    f.pi.pi.setLabel(f.sm.getLeafId() as string, "anchor:start");
    await attempt(f, SUMMARY_TEXT);
    assert.equal(outcomeEntries(f.sm)[0].status, "failed");
    assertNotBranched(f.sm);
  });

  it("ends when newLabel was taken before completion", async () => {
    const f = await pending();
    f.pi.pi.setLabel(f.sm.getLeafId() as string, "anchor:end");
    await attempt(f, SUMMARY_TEXT);
    assert.equal(
      outcomeEntries(f.sm)[0].reason,
      "a label 'end' now exists on the active branch",
    );
    assertNotBranched(f.sm);
  });

  it("success after hidden attempts: their blocked calls stay out of the file lists (O3-B)", async () => {
    const f = await pending();
    await toolCallAttempt(f, ["tc-ghost"]);
    await summaryTurn(f);
    const done = rewindMessages(f.pi)[0].message;
    assert.equal(done.details.status, "done");
    assert.equal(done.details.summaryAttempts, 2);
    assert.deepEqual(done.details.readFiles, []);
    const summary = f.sm.getEntry(done.details.summaryId as string);
    assert.ok(summary && summary.type === "branch_summary");
    assert.doesNotMatch(summary.summary, /tc-ghost/);
  });
});

describe("rewind: an ending dropped by pi (#90)", () => {
  // Another extension's invalid draft (e.g. a later handler's) makes pi drop
  // the whole boundary list, our ending included (`commit: false`): it must
  // be re-applied outside the boundary.

  it("a dropped agent_before_settle ending is re-applied at the next request", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    await requestRewind(f);
    await fireAgentBeforeSettle(f.pi, f.ctx, { commit: false });
    assert.equal(outcomeEntries(f.sm).length, 0, "pi dropped the list");
    const sent = fireContext(f.pi, f.ctx);
    const outcome = outcomeEntries(f.sm);
    assert.equal(outcome.length, 1);
    assert.equal(
      outcome[0].text,
      "Summarization failed: the run ended before the summary. Nothing was collapsed. Try again.",
    );
    const instruction = f.sm
      .getBranch()
      .find(
        (e) =>
          e.type === "custom_message" &&
          e.customType === SUMMARY_REQUEST_CUSTOM_TYPE,
      );
    assert.ok(instruction && hiddenIds(f.sm).has(instruction.id));
    const result = sent.find(
      (m) => m.role === "toolResult" && m.toolCallId === "tc-rewind",
    );
    assert.equal(result?.isError, true);
    assert.equal(fireToolCall(f.pi, f.ctx), undefined, "gate reopened");
    // Not written twice.
    fireContext(f.pi, f.ctx);
    assert.equal(outcomeEntries(f.sm).length, 1);
  });

  it("a dropped turn_end ending is re-applied at agent_settled", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    await requestRewind(f);
    const message = assistantMessage([], "aborted", 100);
    const entryId = f.sm.appendMessage(message as never);
    await fireTurnEndAll(f.pi, f.ctx, message, { commit: false });
    assert.equal(outcomeEntries(f.sm).length, 0, "pi dropped the list");
    await fireAgentSettled(f.pi, f.ctx, true);
    const outcome = outcomeEntries(f.sm);
    assert.equal(outcome.length, 1);
    assert.equal(outcome[0].status, "aborted");
    assert.equal(outcome[0].text, REWIND_ABORTED_TEXT);
    assert.ok(hiddenIds(f.sm).has(entryId), "the aborted attempt is hidden");
  });

  it("a dropped exhausted-attempts ending is re-applied at agent_before_settle with continue", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    await requestRewind(f);
    await summaryTurn(f, "no goal 1", "stop", { commit: true });
    await summaryTurn(f, "no goal 2", "stop", { commit: true });
    // Attempt 3 exhausts the attempts; pi drops its turn_end list
    // (another extension's invalid draft): entries [], continue false.
    const third = await summaryTurn(f, "no goal 3", "stop", {
      commit: false,
    });
    assert.equal(third.boundary.continue, true, "our ending asked to continue");
    assert.equal(outcomeEntries(f.sm).length, 0, "pi dropped the list");
    const result = await fireAgentBeforeSettle(f.pi, f.ctx);
    const outcome = outcomeEntries(f.sm);
    assert.equal(outcome.length, 1);
    assert.equal(outcome[0].status, "failed");
    assert.ok(hiddenIds(f.sm).has(third.entryId), "attempt 3 is hidden");
    assert.equal(
      result.continue,
      true,
      "the model reads the failure in the same run",
    );
  });

  it("a dropped failed ending re-applied at an aborted / errored settle does not continue", async () => {
    for (const outcome of ["aborted", "error"] as const) {
      const f = setup();
      setupRewindable(f.sm, f.pi, { capture: true });
      await requestRewind(f);
      await summaryTurn(f, "no goal 1", "stop", { commit: true });
      await summaryTurn(f, "no goal 2", "stop", { commit: true });
      await summaryTurn(f, "no goal 3", "stop", { commit: false });
      const result = await fireAgentBeforeSettle(f.pi, f.ctx, { outcome });
      assert.equal(outcomeEntries(f.sm)[0].status, "failed");
      assert.equal(result.continue, false, outcome);
    }
  });

  it("a committed ending is not re-applied", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    await requestRewind(f);
    await fireAgentBeforeSettle(f.pi, f.ctx);
    fireContext(f.pi, f.ctx);
    await fireAgentSettled(f.pi, f.ctx);
    assert.equal(outcomeEntries(f.sm).length, 1);
    assert.equal(f.pi.appendEntryCalls.length, 0);
  });
});

describe("rewind: reused tool-call ids (#90)", () => {
  // OpenAI-compatible servers may send "" or reuse ids: everything a
  // rewind keys on is its own entry ids, never the toolCallId alone.
  for (const id of ["X", ""]) {
    it(`two rewinds sharing toolCallId ${JSON.stringify(id)}: frozen view, rewrite and ending follow each rewind's own entries`, async () => {
      const f = setup();
      setupRewindable(f.sm, f.pi, { capture: true });
      await requestRewind(f, REWIND_ARGS, id);
      await fireAgentBeforeSettle(f.pi, f.ctx);
      assert.equal(outcomeEntries(f.sm).length, 1);
      appendTurn(f.sm, "u9", "a9", 30_000);
      await requestRewind(f, { ...REWIND_ARGS, newLabel: "end2" }, id);
      assert.ok(fireToolCall(f.pi, f.ctx), "the second rewind is pending");
      const instructions = f.sm
        .getBranch()
        .filter(
          (e) =>
            e.type === "custom_message" &&
            e.customType === SUMMARY_REQUEST_CUSTOM_TYPE,
        );
      assert.equal(instructions.length, 2);
      const sent = fireContext(f.pi, f.ctx);
      // Frozen at the SECOND instruction (the first one is hidden).
      assert.equal(
        (sent.at(-1) as { details?: { newLabel?: string } }).details?.newLabel,
        "end2",
      );
      assert.ok(
        sent.some((m) => JSON.stringify(m.content).includes('"a9"')),
        "work after the first rewind is still sent",
      );
      const results = () =>
        fireContext(f.pi, f.ctx).filter(
          (m) => m.role === "toolResult" && m.toolCallId === id,
        );
      // Only the first (ended) rewind's result reads as failed.
      let r = results();
      assert.equal(r.length, 2);
      assert.equal(r[0].isError, true);
      assert.notEqual(r[1].isError, true);
      assert.match(JSON.stringify(r[1].content), /accepted/);
      // The second ends aborted: its own result and instruction only.
      await fireAgentBeforeSettle(f.pi, f.ctx, { outcome: "aborted" });
      const outcomes = outcomeEntries(f.sm);
      assert.equal(outcomes.length, 2);
      assert.notEqual(
        outcomes[0].toolResultEntryId,
        outcomes[1].toolResultEntryId,
      );
      assert.ok(hiddenIds(f.sm).has(instructions[1].id));
      r = results();
      assert.equal(r.length, 2);
      assert.equal(
        (r[0].content as Array<{ text: string }>)[0].text,
        "Summarization failed: the run ended before the summary. Nothing was collapsed. Try again.",
      );
      assert.equal(
        (r[1].content as Array<{ text: string }>)[0].text,
        REWIND_ABORTED_TEXT,
      );
      assert.equal(r[1].isError, true);
    });

    it(`an attempt calling a tool with the rewind's id ${JSON.stringify(id)} is an attempt, not the rewind turn (matched by messageEntryId)`, async () => {
      const f = setup();
      setupRewindable(f.sm, f.pi, { capture: true });
      await requestRewind(f, REWIND_ARGS, id);
      const { entryId, resultIds } = await toolCallAttempt(f, [id]);
      const instructions = f.sm
        .getBranch()
        .filter(
          (e) =>
            e.type === "custom_message" &&
            e.customType === SUMMARY_REQUEST_CUSTOM_TYPE,
        );
      assert.equal(instructions.length, 1, "no second instruction");
      for (const hidden of [entryId, ...resultIds]) {
        assert.ok(hiddenIds(f.sm).has(hidden), "the attempt is hidden");
      }
      assert.equal(outcomeEntries(f.sm).length, 0);
      assert.ok(fireToolCall(f.pi, f.ctx), "still pending, re-sent");
    });

    it(`the rewind turn's boundary skipped: a later turn calling ${JSON.stringify(id)} is not taken for it (matched by messageEntryId)`, async () => {
      const f = setup();
      setupRewindable(f.sm, f.pi, { capture: true });
      appendToolCallAssistant(f.sm, [
        { id, name: TOOL_NAME, arguments: REWIND_ARGS },
      ]);
      const result = await f.tool.execute(
        id,
        REWIND_ARGS,
        undefined,
        undefined,
        f.ctx,
      );
      appendToolResult(f.sm, id, TOOL_NAME, result.content[0].text);
      // pi never fired the rewind turn's turn_end.
      const { entryId } = await toolCallAttempt(f, [id]);
      assert.ok(
        !f.pi.boundaryResults.some((r) =>
          r.entries.some((d) => d.customType === SUMMARY_REQUEST_CUSTOM_TYPE),
        ),
        "no instruction for a turn that did not issue the rewind",
      );
      const outcome = outcomeEntries(f.sm);
      assert.equal(outcome.length, 1);
      assert.equal(
        outcome[0].reason,
        "the summary request could not be delivered",
      );
      assert.ok(hiddenIds(f.sm).has(entryId));
    });
  }
});

describe("rewind: restart while pending (#90 O8)", () => {
  async function withSessionStart(
    run: (fixture: ConfigFixture) => Promise<void>,
  ): Promise<void> {
    const fixture = makeConfigFixture();
    try {
      await run(fixture);
    } finally {
      cleanupConfigFixture(fixture);
    }
  }

  it("an orphan instruction (no branch: label, no outcome) is ended aborted at session_start", async () => {
    await withSessionStart(async (fixture) => {
      const f = setup();
      setupRewindable(f.sm, f.pi, { capture: true });
      await requestRewind(f);
      const attempt = await summaryTurn(f, "partial", "stop", {
        commit: false,
      });
      // A fresh extension instance (restart / reload): nothing pending.
      const g = { ...f, pi: makeFakePi(f.sm) };
      navigateTree(g.pi.pi);
      await runSessionStart(g.pi, f.ctx, fixture, "resume");
      const outcome = outcomeEntries(f.sm);
      assert.equal(outcome.length, 1);
      assert.equal(outcome[0].status, "aborted");
      assert.equal(outcome[0].text, REWIND_ABORTED_TEXT);
      assert.ok(hiddenIds(f.sm).has(attempt.entryId), "the attempt is hidden");
      // Idempotent: the outcome resolves it.
      await runSessionStart(g.pi, f.ctx, fixture, "resume");
      assert.equal(outcomeEntries(f.sm).length, 1);
    });
  });

  it("orphan hiding skips a steer typed while pending (it sits before attempt 1); the steer stays visible", async () => {
    await withSessionStart(async (fixture) => {
      const f = setup();
      setupRewindable(f.sm, f.pi, { capture: true });
      await requestRewind(f);
      const steerId = f.sm.appendMessage({
        role: "user",
        content: [{ type: "text", text: "steer" }],
      } as never);
      const attempt = await summaryTurn(f, "partial", "stop", {
        commit: false,
      });
      const g = { ...f, pi: makeFakePi(f.sm) };
      navigateTree(g.pi.pi);
      await runSessionStart(g.pi, f.ctx, fixture, "resume");
      const hidden = hiddenIds(f.sm);
      assert.ok(hidden.has(attempt.entryId), "attempt 1 after the steer");
      assert.ok(!hidden.has(steerId), "the steer stays visible");
    });
  });

  it("orphan hiding stops after MAX_SUMMARY_ATTEMPTS assistant turns", async () => {
    await withSessionStart(async (fixture) => {
      const f = setup();
      setupRewindable(f.sm, f.pi, { capture: true });
      await requestRewind(f);
      const ids: string[] = [];
      for (let i = 0; i < 4; i++) {
        // User messages (steers) between turns do not end the window.
        f.sm.appendMessage({
          role: "user",
          content: [{ type: "text", text: `steer ${i}` }],
        } as never);
        ids.push(
          f.sm.appendMessage(
            assistantMessage(
              [{ type: "text", text: `t${i}` }],
              "stop",
              100,
            ) as never,
          ),
        );
      }
      const g = { ...f, pi: makeFakePi(f.sm) };
      navigateTree(g.pi.pi);
      await runSessionStart(g.pi, f.ctx, fixture, "resume");
      const hidden = hiddenIds(f.sm);
      for (const id of ids.slice(0, 3)) assert.ok(hidden.has(id));
      assert.ok(!hidden.has(ids[3]), "a 4th assistant turn was not an attempt");
    });
  });

  it("the same session_start in-process also drops the in-memory pending rewind", async () => {
    await withSessionStart(async (fixture) => {
      const f = setup();
      setupRewindable(f.sm, f.pi);
      await requestRewind(f);
      await runSessionStart(f.pi, f.ctx, fixture, "reload");
      assert.equal(fireToolCall(f.pi, f.ctx), undefined);
      assert.equal(outcomeEntries(f.sm)[0].status, "aborted");
    });
  });

  it("success leaves branch:<newLabel> on the summary turn; going back to the old branch is not an orphan", async () => {
    await withSessionStart(async (fixture) => {
      const f = await rewindFixture({ commit: true });
      assert.equal(f.sm.getLabel(f.summaryEntryId), "branch:end");
      // `list` never shows it.
      const list = await f.tool.execute(
        "tc-list",
        { action: "list" },
        undefined,
        undefined,
        f.ctx,
      );
      assert.doesNotMatch(list.content[0].text, /branch:|\bend\b.*branch/);
      // Back onto the abandoned branch (e.g. /tree), then a restart.
      f.sm.branch(f.summaryEntryId);
      await runSessionStart(f.pi, f.ctx, fixture, "resume");
      assert.equal(outcomeEntries(f.sm).length, 0);
    });
  });

  it("a later rewind's outcome with the same toolCallId does not resolve an orphan (matched by instructionEntryId)", async () => {
    await withSessionStart(async (fixture) => {
      const f = setup();
      setupRewindable(f.sm, f.pi, { capture: true });
      // Rewind 1 (id "X") dies mid-flight: no outcome is ever written.
      await requestRewind(f, REWIND_ARGS, "X");
      // A fresh instance (nothing pending) runs rewind 2, reusing "X", and
      // ends it: its outcome lands after rewind 1's instruction.
      const g = { ...f, pi: makeFakePi(f.sm) };
      navigateTree(g.pi.pi);
      g.tool = g.pi.registered[0];
      await requestRewind(g, { ...REWIND_ARGS, newLabel: "end2" }, "X");
      await fireAgentBeforeSettle(g.pi, g.ctx);
      const instructions = f.sm
        .getBranch()
        .filter(
          (e) =>
            e.type === "custom_message" &&
            e.customType === SUMMARY_REQUEST_CUSTOM_TYPE,
        );
      assert.equal(instructions.length, 2);
      assert.equal(outcomeEntries(f.sm).length, 1);
      assert.equal(
        outcomeEntries(f.sm)[0].instructionEntryId,
        instructions[1].id,
      );
      // Restart: rewind 1 is still an orphan.
      const h = { ...f, pi: makeFakePi(f.sm) };
      navigateTree(h.pi.pi);
      await runSessionStart(h.pi, f.ctx, fixture, "resume");
      const outcomes = outcomeEntries(f.sm);
      assert.equal(outcomes.length, 2, "rewind 1 is ended at restart");
      assert.equal(outcomes[1].instructionEntryId, instructions[0].id);
      assert.equal(outcomes[1].status, "aborted");
      assert.ok(hiddenIds(f.sm).has(instructions[0].id));
    });
  });

  it("a failed rewind's outcome resolves its instruction", async () => {
    await withSessionStart(async (fixture) => {
      const f = setup();
      setupRewindable(f.sm, f.pi);
      await requestRewind(f);
      await fireAgentBeforeSettle(f.pi, f.ctx);
      await runSessionStart(f.pi, f.ctx, fixture, "resume");
      assert.equal(outcomeEntries(f.sm).length, 1);
    });
  });
});

describe("rewind: solo-call guard (#37, restored by #90 S6)", () => {
  async function rewindIn(batch: Array<{ id: string; name: string }>) {
    const f = setup();
    setupRewindable(f.sm, f.pi);
    appendToolCallAssistant(
      f.sm,
      batch.map((c) => ({
        ...c,
        arguments: c.id === "tc-rewind" ? REWIND_ARGS : { path: "x" },
      })),
    );
    const result = await f.tool.execute(
      "tc-rewind",
      REWIND_ARGS,
      undefined,
      undefined,
      f.ctx,
    );
    return { ...f, result };
  }

  it("refuses a rewind batched with other calls; nothing is recorded", async () => {
    const f = await rewindIn([
      { id: "tc-read", name: "read" },
      { id: "tc-rewind", name: TOOL_NAME },
    ]);
    assert.equal(f.result.isError, true);
    assert.equal(
      f.result.content[0].text,
      "rewind must be the only tool call in its message — re-issue only the rewind, alone. The other calls in this message ran normally.",
    );
    assert.equal(f.result.content[0].text, SOLO_REWIND_REFUSAL);
    assert.deepEqual(f.result.details.batchedToolCalls, ["read", TOOL_NAME]);
    assert.equal(fireToolCall(f.pi, f.ctx), undefined, "nothing pending");
  });

  it("allows a solo rewind", async () => {
    const f = await rewindIn([{ id: "tc-rewind", name: TOOL_NAME }]);
    assert.equal(f.result.isError, undefined);
  });

  it("fails open when getBranch throws", () => {
    const sm = {
      getBranch() {
        throw new Error("boom");
      },
    } as unknown as SessionManager;
    assert.equal(
      __testHooks.findInFlightAssistantToolCalls(sm, "tc-rewind"),
      null,
    );
  });

  it("finds the declaring assistant by id even with sibling results after it", () => {
    const sm = SessionManager.inMemory("/tmp");
    appendToolCallAssistant(sm, [
      { id: "tc-a", name: "read" },
      { id: "tc-rewind", name: TOOL_NAME },
    ]);
    appendToolResult(sm, "tc-a", "read", "ok");
    assert.deepEqual(
      __testHooks.findInFlightAssistantToolCalls(sm, "tc-rewind"),
      [
        { id: "tc-a", name: "read" },
        { id: "tc-rewind", name: TOOL_NAME },
      ],
    );
    assert.equal(__testHooks.findInFlightAssistantToolCalls(sm, "nope"), null);
  });

  it("blocks a nested rewind (issued by another tool)", () => {
    const { pi, ctx } = setup();
    const handlers = pi.onCalls.get("tool_call");
    assert.ok(handlers);
    const result = handlers[0](
      {
        type: "tool_call",
        toolCallId: "tc-code/1",
        parentToolCallId: "tc-code",
        toolName: TOOL_NAME,
        input: { action: "rewind" },
      } as never,
      ctx as never,
    );
    assert.deepEqual(result, { block: true, reason: NESTED_REWIND_REFUSAL });
    assert.equal(
      NESTED_REWIND_REFUSAL,
      "rewind cannot be called from inside another tool — call navigate_tree directly.",
    );
    // A nested list is fine.
    assert.equal(
      handlers[0](
        {
          type: "tool_call",
          toolCallId: "tc-code/2",
          parentToolCallId: "tc-code",
          toolName: TOOL_NAME,
          input: { action: "list" },
        } as never,
        ctx as never,
      ),
      undefined,
    );
  });
});

describe("rewind: TUI renderers (#90 O1 / O9)", () => {
  const theme = { fg: (color: string, text: string) => `<${color}>${text}` };

  it("summary request row: byte-exact, with and without a window", () => {
    assert.equal(
      buildSummaryRequestRow({
        rewindTo: "context-gathered",
        newLabel: "parser-done",
        contextBefore: 304_000,
        tokensAtTarget: 120_000,
        contextWindow: 1_000_000,
      }),
      "[rewind to 'context-gathered' · collapse as 'parser-done'] · summarizing · context 30.4% → ~12.0% of 1.0M",
    );
    assert.equal(
      buildSummaryRequestRow({
        rewindTo: "a",
        newLabel: "b",
        contextBefore: 30400,
        tokensAtTarget: 12000,
        contextWindow: 0,
      }),
      "[rewind to 'a' · collapse as 'b'] · summarizing · tokens 30400 → ~12000",
    );
  });

  it("outcome rows: failed with its reason, aborted without", () => {
    assert.equal(
      buildRewindOutcomeRow({
        status: "failed",
        reason: "no summary after 3 attempts (last: called a tool)",
        rewindTo: "context-gathered",
        newLabel: "parser-done",
      }),
      "[rewind to 'context-gathered' · collapse as 'parser-done'] · failed: no summary after 3 attempts (last: called a tool) · nothing collapsed",
    );
    assert.equal(
      buildRewindOutcomeRow({
        status: "aborted",
        reason: "aborted",
        rewindTo: "a",
        newLabel: "b",
      }),
      "[rewind to 'a' · collapse as 'b'] · aborted · nothing collapsed",
    );
  });

  it("registers a message renderer: collapsed = the row, expanded = row + full text", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi);
    await requestRewind(f);
    const renderer = f.pi.messageRenderers.get(SUMMARY_REQUEST_CUSTOM_TYPE);
    assert.ok(renderer);
    const draft = summaryRequestDraftOf(f.pi);
    const message = {
      role: "custom",
      customType: draft.customType,
      content: draft.content,
      display: true,
      details: draft.details,
      timestamp: 0,
    };
    const collapsed = (
      renderer as (...a: unknown[]) => { render(w: number): string[] }
    )(message, { expanded: false, outputPad: 1 }, theme);
    const row = buildSummaryRequestRow(draft.details as never);
    assert.deepEqual(collapsed.render(400), [` <customMessageText>${row}`]);
    const expanded = (
      renderer as (...a: unknown[]) => { render(w: number): string[] }
    )(message, { expanded: true, outputPad: 1 }, theme);
    const lines = expanded.render(10_000);
    assert.equal(lines[0], ` <customMessageText>${row}`);
    assert.ok(lines.some((l) => l.includes("[navigate_tree]")));
    for (const line of collapsed.render(30)) {
      assert.ok(line.replace(/<[^>]+>/g, "").length <= 30, "wrapped to width");
    }
  });

  it("registers an entry renderer for the outcome, error-styled", () => {
    const f = setup();
    const renderer = f.pi.entryRenderers.get(REWIND_OUTCOME_CUSTOM_TYPE) as (
      ...a: unknown[]
    ) => { render(w: number): string[] } | undefined;
    assert.ok(renderer);
    const out = renderer(
      {
        type: "custom",
        customType: REWIND_OUTCOME_CUSTOM_TYPE,
        data: {
          status: "aborted",
          reason: "aborted",
          rewindTo: "a",
          newLabel: "b",
        },
      },
      { expanded: false },
      theme,
    );
    assert.ok(out);
    assert.deepEqual(out.render(200), [
      " <error>[rewind to 'a' · collapse as 'b'] · aborted · nothing collapsed",
    ]);
    assert.equal(
      renderer({ type: "custom", data: {} }, { expanded: false }, theme),
      undefined,
    );
  });

  it("wrapPlainLine word-wraps and hard-breaks long words", () => {
    assert.deepEqual(wrapPlainLine("aaa bbb ccc", 7), ["aaa bbb", "ccc"]);
    assert.deepEqual(wrapPlainLine("abcdefghij", 4), ["abcd", "efgh", "ij"]);
    assert.deepEqual(wrapPlainLine("short", 10), ["short"]);
  });

  it("wrapPlainLine wraps by terminal cells: CJK, emoji and tabs never exceed the width", () => {
    for (const text of [
      "保持用户指令并继续工作，直到完成所有任务为止。".repeat(3),
      "🚀🔥✅ emoji focus 🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪",
      "\tTabbed\tfocus\twith\ttabs\teverywhere\t\t\t",
    ]) {
      for (const width of [1, 2, 3, 5, 8, 13, 400]) {
        for (const line of wrapPlainLine(text, width)) {
          assert.ok(
            visibleWidth(line) <= width,
            `${JSON.stringify(line)} is ${visibleWidth(line)} cells > ${width}`,
          );
          assert.ok(!line.includes("\t"), `${JSON.stringify(line)} has a tab`);
        }
      }
    }
  });

  it("renderers never emit a line wider than the terminal (pi-tui crashes on one)", async () => {
    const ansi = {
      fg: (_color: string, text: string) => `\x1b[31m${text}\x1b[39m`,
    };
    const f = setup();
    setupRewindable(f.sm, f.pi);
    await requestRewind(f, {
      ...REWIND_ARGS,
      newLabel: "cjk-end",
      summaryFocus:
        "保持用户指令并继续工作。🚀🔥✅🧪\tTabbed\tfocus\t保持用户指令并继续工作。",
    });
    const renderer = f.pi.messageRenderers.get(SUMMARY_REQUEST_CUSTOM_TYPE) as (
      ...a: unknown[]
    ) => { render(w: number): string[] };
    const draft = summaryRequestDraftOf(f.pi);
    const message = {
      role: "custom",
      customType: draft.customType,
      content: draft.content,
      display: true,
      details: draft.details,
      timestamp: 0,
    };
    const outcome = f.pi.entryRenderers.get(REWIND_OUTCOME_CUSTOM_TYPE) as (
      ...a: unknown[]
    ) => { render(w: number): string[] };
    for (const width of [1, 2, 3, 7, 10, 17, 40, 400]) {
      const rendered = [
        ...renderer(message, { expanded: true, outputPad: 1 }, ansi).render(
          width,
        ),
        ...outcome(
          {
            type: "custom",
            customType: REWIND_OUTCOME_CUSTOM_TYPE,
            data: {
              status: "failed",
              reason: "the branch could not be written (磁盘已满 🚫\tdisk)",
              rewindTo: "开始",
              newLabel: "结束🚀",
            },
          },
          { expanded: true },
          ansi,
        ).render(width),
      ];
      for (const line of rendered) {
        assert.ok(
          visibleWidth(line) <= width,
          `${JSON.stringify(line)} is ${visibleWidth(line)} cells > ${width}`,
        );
        // Tabs are expanded to spaces: no raw tab reaches a row.
        assert.ok(!line.includes("\t"), `${JSON.stringify(line)} has a tab`);
      }
    }
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

  it("does not re-append this extension's own instruction message (#90)", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    await requestRewind(f);
    // A failed attempt first: the retry re-sends, nothing appended.
    await summaryTurn(f, "Okay, summarizing now.", "stop", { commit: true });
    assert.equal(rewindMessages(f.pi).length, 0);
    await summaryTurn(f);
    const done = rewindMessages(f.pi)[0].message;
    assert.equal(done.details.status, "done");
    assert.equal(done.details.summaryAttempts, 2);
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

  it("a throwing getBranch before the write ends it with the completion reason", async () => {
    const f = setup();
    setupRewindable(f.sm, f.pi, { capture: true });
    await requestRewind(f);
    const sm = f.sm as unknown as { getBranch: () => unknown };
    const origGetBranch = sm.getBranch;
    sm.getBranch = () => {
      throw new Error("branch boom");
    };
    let boundary: BoundaryResult | undefined;
    try {
      await assert.doesNotReject(async () => {
        boundary = (await summaryTurn(f)).boundary;
      });
    } finally {
      sm.getBranch = origGetBranch;
    }
    assert.ok(boundary);
    assert.equal(boundary.continue, true);
    const outcome = boundary.entries.find(
      (d) => d.customType === REWIND_OUTCOME_CUSTOM_TYPE,
    );
    assert.equal(
      outcome?.data?.text,
      "Summarization failed: the rewind could not be completed (branch boom). Nothing was collapsed. Try again.",
    );
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

  it("SUMMARY_PENDING_BLOCK_REASON (TUI-only, #90 O9)", () => {
    assert.equal(
      SUMMARY_PENDING_BLOCK_REASON,
      "navigate_tree: blocked: the model tried to call a tool instead of summarizing.",
    );
  });

  it("buildRewindAcceptedText (#90 S2)", () => {
    assert.equal(
      buildRewindAcceptedText("context-gathered", "parser-done"),
      "[navigate_tree] Rewind to 'context-gathered' accepted (collapse as 'parser-done'). Follow the navigate_tree message below to complete it.",
    );
  });

  it("ending texts (#90 O5 / O6)", () => {
    assert.equal(
      buildRewindFailedText("the context was compacted"),
      "Summarization failed: the context was compacted. Nothing was collapsed. Try again.",
    );
    assert.equal(
      REWIND_ABORTED_TEXT,
      "Summarization aborted. Nothing was collapsed.",
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
    assert.match(text, / {2}kickoff\n/);
    assert.match(text, / {2}impl-start$/);
    // Byte-exact pin (#84 moved the anchor walk into formatActiveAnchors).
    assert.equal(
      text,
      "Rewinding to 'impl-start' (as 'done') would free only ~0.0k tokens. Rewind further back to actually free context, or keep working.\nActive anchors (root → leaf, % of 1.0M):\n   0.6%  kickoff\n   1.2%  impl-start",
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
      "Rewinding to 'start' (as 'done') would free only ~0.0k tokens. No earlier anchors — keep working and rewind later once more has accumulated above it.\nActive anchors (root → leaf, % of 1.0M):\n   0.6%  start",
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
      "Rewinding to 'start' (as 'done') would free only ~0.0k tokens. No earlier anchors — keep working and rewind later once more has accumulated above it.\nActive anchors (root → leaf):\n   6.0k  start",
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
    // The rejection lists only names rewindTo accepts (no `''`).
    assert.match(
      r.content[0].text,
      /\nActive anchors \(root → leaf, % of 1\.0M\):\n +[\d.]+% {2}start$/,
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
  // pi's auto-compaction defaults to on, which turns the hint off (#101,
  // D9): the fixture's pi settings turn it off unless a test says otherwise.
  writePiSettings(
    { root, agentDir, projectDir },
    { compaction: { enabled: false } },
  );
  return { root, agentDir, projectDir };
}

/** Write the fixture's global pi `settings.json` (#101). */
function writePiSettings(fixture: ConfigFixture, settings: unknown): void {
  writeFileSync(
    join(fixture.agentDir, "settings.json"),
    JSON.stringify(settings),
  );
}

/** Write the fixture's project pi `.pi/settings.json` (#101). */
function writeProjectPiSettings(
  fixture: ConfigFixture,
  settings: unknown,
): void {
  const dir = join(fixture.projectDir, CONFIG_DIR_NAME);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, "settings.json"), JSON.stringify(settings));
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
  anchors = "Active anchors (root → leaf, % of 1.0M):\n   0.0%  start",
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
        "[navigate_tree hint] Context is at 90.0% of 1.0M — 100.0k tokens left, within the 100.0k reserve. Persist what matters to files now, then rewind to the oldest appropriate anchor.\nActive anchors (root → leaf, % of 1.0M):\n   0.0%  start",
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
        "[navigate_tree hint] Context is at 105.0% of 1.0M — 0.0k tokens left, within the 100.0k reserve. Persist what matters to files now, then rewind to the oldest appropriate anchor.\nActive anchors (root → leaf, % of 1.0M):\n   0.0%  start",
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

  it("lists only names rewindTo accepts", async () => {
    const fixture = makeConfigFixture();
    try {
      writeGlobalConfig(fixture, ENABLED_CONFIG);
      const { sm, pi, ctx } = setup();
      const t0 = appendTurn(sm, "u-empty", "a-empty");
      pi.pi.setLabel(t0.assistantId, "anchor:");
      const t1 = appendTurn(sm, "u-upper", "a-upper");
      pi.pi.setLabel(t1.assistantId, "anchor:Foo");
      anchorStart(sm, pi);
      await runSessionStart(pi, ctx, fixture);
      ctx.setContextUsage(usageAt(95));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 1);
      const text = String(
        (pi.sendMessageCalls[0][0] as { content: unknown }).content,
      );
      assert.match(
        text,
        /\nActive anchors \(root → leaf, % of 1\.0M\):\n +[\d.]+% {2}start$/,
      );
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("an empty-named manual label alone counts as no anchors", async () => {
    const fixture = makeConfigFixture();
    try {
      writeGlobalConfig(fixture, ENABLED_CONFIG);
      const { sm, pi, ctx } = setup();
      // `anchor:` is not a valid `rewindTo`, so the model must not be
      // pointed at it.
      const t = appendTurn(sm, "u-empty", "a-empty");
      pi.pi.setLabel(t.assistantId, "anchor:");
      await runSessionStart(pi, ctx, fixture);
      ctx.setContextUsage(usageAt(95));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 0);
      assert.deepEqual(ctx.notifyCalls, [
        [buildNoAnchorText(95, 1_000_000), "warning"],
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
        JSON.stringify(projected.messages).includes(
          JSON.stringify(text).slice(1, -1),
        ),
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
      await summaryTurn(f, "Okay, summarizing now.", "stop", { commit: true });
      assert.equal(rewindMessages(f.pi).length, 0);
      assert.equal(hintCalls(f.pi).length, 0, "a failed attempt");
      await summaryTurn(f);
      assert.equal(rewindMessages(f.pi)[0].message.details.status, "done");
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
            content: expectedHint(
              95,
              "Active anchors (root → leaf, % of 1.0M):\n   0.6%  start\n   0.6%  end",
            ),
            display: true,
          },
        ],
      ]);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("skips a tool-call failure turn too, and the start anchor skips it as well (#90)", async () => {
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
      assert.equal(
        __testHooks.findLabeledEntry(f.sm, START_ANCHOR_LABEL),
        null,
        "never on an attempt that is about to be hidden",
      );
      // agent_end while pending writes nothing either.
      fireAgentEnd(f.pi, f.ctx);
      assert.equal(
        __testHooks.findLabeledEntry(f.sm, START_ANCHOR_LABEL),
        null,
      );
    } finally {
      cleanupConfigFixture(fixture);
    }
  });
});

describe("rewind hint: per-model modelOverrides (#86)", () => {
  const SONNET = "claude/claude-sonnet-4-5";

  async function overrideSetup(config: unknown): Promise<{
    fixture: ConfigFixture;
    pi: FakePi;
    ctx: FakeCtx;
  }> {
    const fixture = makeConfigFixture();
    writeGlobalConfig(fixture, JSON.stringify(config));
    const { sm, pi, ctx } = setup();
    anchorStart(sm, pi);
    await runSessionStart(pi, ctx, fixture);
    return { fixture, pi, ctx };
  }

  it("a matching entry overrides the base reserve", async () => {
    const { fixture, pi, ctx } = await overrideSetup({
      rewindHintReserveTokens: 20_000,
      modelOverrides: { [SONNET]: { rewindHintReserveTokens: 100_000 } },
    });
    try {
      // 92% of 1M = 80k left: inside the 100k entry, outside the 20k base.
      ctx.setContextUsage(usageAt(92));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 1);
      assert.match(
        String((pi.sendMessageCalls[0][0] as { content: unknown }).content),
        /within the 100\.0k reserve/,
      );
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("an off entry disables the hint for that model only", async () => {
    const { fixture, pi, ctx } = await overrideSetup({
      rewindHintReserveTokens: 100_000,
      modelOverrides: { [SONNET]: { rewindHintReserveTokens: "off" } },
    });
    try {
      ctx.setContextUsage(usageAt(99));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 0);
      // Switching to an unmatched model falls back to the base reserve.
      ctx.model = { api: "openai", provider: "x", id: "other" };
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 1);
      assert.match(
        String((pi.sendMessageCalls[0][0] as { content: unknown }).content),
        /within the 100\.0k reserve/,
      );
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("an entry enables a model while the base is off", async () => {
    const { fixture, pi, ctx } = await overrideSetup({
      modelOverrides: { [SONNET]: { rewindHintReserveTokens: 100_000 } },
    });
    try {
      ctx.setContextUsage(usageAt(95));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 1);
      // An unmatched model gets no hint (base off), but its turns still
      // observe the climb back out, so the next crossing fires again.
      ctx.model = { api: "openai", provider: "x", id: "other" };
      ctx.setContextUsage(usageAt(96));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 1);
      ctx.setContextUsage(usageAt(10));
      fireTurnEnd(pi, ctx);
      ctx.model = {
        api: "anthropic",
        provider: "claude",
        id: "claude-sonnet-4-5",
      };
      ctx.setContextUsage(usageAt(96));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 2);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("a model switch re-reads the reserve on the next turn", async () => {
    const { fixture, pi, ctx } = await overrideSetup({
      rewindHintReserveTokens: 20_000,
      modelOverrides: { "x/big": { rewindHintReserveTokens: 100_000 } },
    });
    try {
      // 80k left: outside the 20k base for the default model.
      ctx.setContextUsage(usageAt(92));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 0);
      ctx.model = { api: "openai", provider: "x", id: "big" };
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 1);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("no model falls back to the base reserve", async () => {
    const { fixture, pi, ctx } = await overrideSetup({
      rewindHintReserveTokens: 100_000,
      modelOverrides: { [SONNET]: { rewindHintReserveTokens: "off" } },
    });
    try {
      ctx.model = undefined;
      ctx.setContextUsage(usageAt(95));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 1);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("a throwing ctx.model getter skips the turn without spending the crossing", async () => {
    const { fixture, pi, ctx } = await overrideSetup({
      rewindHintReserveTokens: 100_000,
      modelOverrides: { "x/y": { rewindHintReserveTokens: 50_000 } },
    });
    try {
      const live = ctx.model;
      Object.defineProperty(ctx, "model", {
        configurable: true,
        get() {
          throw new Error("stale ctx");
        },
      });
      ctx.setContextUsage(usageAt(95));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 0);
      Object.defineProperty(ctx, "model", {
        configurable: true,
        writable: true,
        value: live,
      });
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 1);
    } finally {
      cleanupConfigFixture(fixture);
    }
  });

  it("without modelOverrides, ctx.model is never read", async () => {
    const { fixture, pi, ctx } = await overrideSetup({
      rewindHintReserveTokens: 100_000,
    });
    try {
      let reads = 0;
      const live = ctx.model;
      Object.defineProperty(ctx, "model", {
        configurable: true,
        get() {
          reads++;
          return live;
        },
      });
      ctx.setContextUsage(usageAt(95));
      fireTurnEnd(pi, ctx);
      assert.equal(pi.sendMessageCalls.length, 1);
      assert.equal(reads, 0);
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

describe("README: history-mirroring providers (#90)", () => {
  const readme = readFileSync(
    new URL("../../README.md", import.meta.url),
    "utf8",
  );

  it("does not claim the summary is the paused Claude Code query", () => {
    assert.doesNotMatch(readme, /paused Claude Code query/);
  });

  it("says the #90 flow is untested on history-mirroring providers", () => {
    assert.match(
      readme,
      /instruction message and its hidden re-sends \(#90\) are untested on history-mirroring providers/,
    );
  });
});

describe("README: rewind-driven compaction naming (#101)", () => {
  const readme = readFileSync(
    new URL("../../README.md", import.meta.url),
    "utf8",
  );

  it("says 'overflow recovery' only inside the quoted final error", () => {
    const lines = readme
      .split("\n")
      .filter((line) => /overflow recovery/i.test(line));
    assert.ok(lines.length > 0);
    for (const line of lines) {
      assert.ok(
        line.startsWith(
          "[navigate_tree] Context overflow recovery failed: <reason>.",
        ),
        line,
      );
    }
  });

  it("documents the either-or error and the D8 text byte-exact", () => {
    assert.ok(readme.includes(REWIND_COMPACTION_OFF_TEXT));
    assert.ok(readme.includes(buildRewindCompactionFailedText("<reason>")));
    assert.ok(
      readme.includes(
        buildReserveOrderWarning(10000, 16384, "anthropic/claude-haiku-5-5"),
      ),
    );
    assert.ok(readme.includes(buildWithheldNoticeText(19_000)));
    assert.ok(
      readme.includes(
        buildForcedRewindText(
          100_000,
          2,
          "Active anchors (root → leaf, % of 100k):\n   4.1%  start\n  12.0%  context-gathered",
        ),
      ),
    );
  });
});

// =============================================================================
// Rewind-driven compaction (#101)
//
// The overflow turns are simulated: the overflowed assistant is appended as
// pi persists it, then the boundaries fire in pi's order (`turn_end`,
// `agent_end`, `agent_before_settle`, `agent_settled`). Window 100k, hint
// reserve 20k, pi's compaction off with its default reserve (16384), so the
// target is 83 616 tokens.
// =============================================================================

describe("rewind-driven compaction (#101)", () => {
  const WINDOW = 100_000;
  const BIG = "x".repeat(48_000); // ~12k tokens
  const theme = { fg: (color: string, text: string) => `<${color}>${text}` };

  interface Forced {
    fixture: ConfigFixture;
    sm: SessionManager;
    pi: FakePi;
    ctx: FakeCtx;
    tool: CapturedTool;
    /** The batch: the issuing assistant and its two big results (r1, r2). */
    issuerId: string;
    r1: string;
    r2: string;
  }

  async function forcedSetup(
    opts: {
      config?: Record<string, unknown>;
      piSettings?: Record<string, unknown>;
      anchor?: boolean;
      issuerTokens?: number;
    } = {},
  ): Promise<Forced> {
    const fixture = makeConfigFixture();
    writeGlobalConfig(
      fixture,
      JSON.stringify(opts.config ?? { rewindHintReserveTokens: 20_000 }),
    );
    if (opts.piSettings) writePiSettings(fixture, opts.piSettings);
    const f = setup({ contextWindow: WINDOW });
    (f.ctx.model as { maxTokens?: number }).maxTokens = 64_000;
    f.ctx.cwd = fixture.projectDir;
    await runSessionStart(f.pi, f.ctx, fixture);
    const t = appendTurn(f.sm, "u1", "a1", 5_000);
    if (opts.anchor ?? true) f.pi.pi.setLabel(t.assistantId, "anchor:start");
    appendTurn(f.sm, "u2", "a2", 20_000);
    const issuer = appendToolCallAssistant(
      f.sm,
      [
        { id: "r1", name: "read", arguments: { path: "a" } },
        { id: "r2", name: "read", arguments: { path: "b" } },
      ],
      opts.issuerTokens ?? 60_000,
    );
    const r1 = appendToolResult(f.sm, "r1", "read", BIG);
    const r2 = appendToolResult(f.sm, "r2", "read", BIG);
    return { fixture, ...f, issuerId: issuer.id, r1, r2 };
  }

  function errorOverflow(sm: SessionManager, provider = "claude") {
    const message = {
      ...assistantMessage([], "error", 0),
      provider,
      errorMessage: "prompt is too long: 106000 tokens > 100000 maximum",
    };
    return { id: sm.appendMessage(message as never), message };
  }

  /** pi's order after an error turn: turn_end, agent_end, before_settle. */
  async function runErrorEnd(f: Forced, message: unknown, cont = false) {
    const turn = await fireTurnEndAll(f.pi, f.ctx, message);
    for (const h of f.pi.onCalls.get("agent_end") ?? []) {
      h({ type: "agent_end", messages: [message] } as never, f.ctx as never);
    }
    const settle = await fireAgentBeforeSettle(f.pi, f.ctx, {
      outcome: "error",
      continue: cont,
    });
    return { turn, settle };
  }

  function forcedDrafts(result: BoundaryResult) {
    return result.entries.filter((d) => d.customType === REWIND_CUSTOM_TYPE);
  }

  function textAttempt(f: Forced, text = "I will rewind soon.") {
    const message = assistantMessage([{ type: "text", text }], "stop", 100);
    f.sm.appendMessage(message as never);
    return fireTurnEndAll(f.pi, f.ctx, message);
  }

  const D5_ONE = `[navigate_tree] Context overflowed the window (100k): 1 tool result was withheld. Rewind now — only navigate_tree rewind is allowed until it lands.
Active anchors (root → leaf, % of 100k):
   5.0%  start`;

  it("texts are byte-exact (D4, D5, D8, D9, D10, gate)", () => {
    assert.equal(
      buildWithheldNoticeText(19_000),
      "[navigate_tree] Output withheld: adding it overflowed the context window (~19k tokens). The call ran; only its output was dropped. Rewind first, then re-run it if you still need the output.",
    );
    assert.equal(buildWithheldNoticeText(200), buildWithheldNoticeText(1_000));
    assert.equal(
      buildForcedRewindText(100_000, 2, "TABLE"),
      "[navigate_tree] Context overflowed the window (100k): 2 tool results were withheld. Rewind now — only navigate_tree rewind is allowed until it lands.\nTABLE",
    );
    assert.match(
      buildForcedRewindText(100_000, 0, "T"),
      /: no tool results were withheld\. /,
    );
    assert.equal(
      buildRewindCompactionFailedText(COMPACTION_FAILED_NO_REWIND),
      "[navigate_tree] Context overflow recovery failed: no rewind after 3 attempts. The session is at the context limit; rewind manually with /tree, /compact, or start a new session.",
    );
    assert.equal(
      REWIND_COMPACTION_OFF_TEXT,
      "navigate_tree: the rewind hint and rewind-driven compaction are off because pi's auto-compaction is on. Disable compaction.enabled or remove rewindHintReserveTokens.",
    );
    assert.equal(
      buildReserveOrderWarning(10_000, 16_384, "anthropic/claude-haiku-5-5"),
      "rewindHintReserveTokens (10000) is below pi's compaction.reserveTokens (16384) for anthropic/claude-haiku-5-5: rewind-driven compaction may fire before the hint, giving the model no time to prepare and persist what matters to files.",
    );
    assert.equal(
      FORCED_REWIND_BLOCK_REASON,
      "navigate_tree: blocked: only rewind is allowed until the forced rewind lands.",
    );
  });

  it("explicit error: nothing at turn_end, recovery at agent_before_settle in the same run", async () => {
    const f = await forcedSetup();
    try {
      const err = errorOverflow(f.sm);
      const { turn, settle } = await runErrorEnd(f, err.message);
      assert.deepEqual(turn.entries, []);
      assert.equal(settle.continue, true);
      const edits = settle.entries.filter((d) => d.type === "context_edit");
      assert.deepEqual(
        edits.map((d) => [d.targetId, d.replacement === null]),
        [
          [err.id, true],
          [f.r2, false],
        ],
        "the error is hidden, then only the newest result is withheld",
      );
      assert.deepEqual(edits[1].replacement, {
        content: [
          {
            type: "text",
            text: "[navigate_tree] Output withheld: adding it overflowed the context window (~12k tokens). The call ran; only its output was dropped. Rewind first, then re-run it if you still need the output.",
          },
        ],
      });
      const [d5] = forcedDrafts(settle);
      assert.equal(d5.content, D5_ONE);
      assert.equal(d5.display, true);
      assert.deepEqual(d5.details, {
        status: "forced",
        withheld: 1,
        contextWindow: WINDOW,
      });
      // The tool call stays; the result keeps its toolCallId.
      const wire = fireContext(f.pi, f.ctx);
      const r2 = wire.find((m) => m.toolCallId === "r2");
      assert.ok(r2);
      assert.match(JSON.stringify(r2.content), /Output withheld/);
      assert.ok(!wire.some((m) => m.stopReason === "error"));
      // No start anchor at the limit.
      assert.equal(f.sm.getLabel(err.id), undefined);
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("squeezed length: recovered at turn_end, the reply and its failed calls hidden", async () => {
    const f = await forcedSetup();
    try {
      const message = {
        ...assistantMessage(
          [{ type: "toolCall", id: "w1", name: "write", arguments: {} }],
          "length",
          0,
        ),
        usage: { ...usageOf(99_000), output: 100 },
      };
      const id = f.sm.appendMessage(message as never);
      const failed = appendToolResult(f.sm, "w1", "write", "truncated");
      const turn = await fireTurnEndAll(f.pi, f.ctx, message, {
        turnIndex: 1,
      });
      assert.equal(turn.continue, true);
      const edits = turn.entries.filter((d) => d.type === "context_edit");
      assert.deepEqual(
        edits.map((d) => [d.targetId, d.replacement === null]),
        [
          [id, true],
          [failed, true],
          [f.r2, false],
        ],
      );
      assert.equal(forcedDrafts(turn)[0].content, D5_ONE);
      // Rewind-owned: no start anchor on the overflowed turn (X = 2).
      assert.equal(f.sm.getLabel(id), undefined);
      assert.equal(f.pi.sendMessageCalls.length, 0, "no hint");
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("pi compaction on: the overflowed reply gets no start anchor (#101, D14)", async () => {
    // Pi compaction is on: rewind-driven compaction stands down, so the
    // overflowed turn is not rewind-owned. It must still not be anchored.
    const f = await forcedSetup({
      anchor: false,
      piSettings: { compaction: { enabled: true } },
    });
    try {
      const message = {
        ...assistantMessage([{ type: "text", text: "done" }], "stop", 0),
        usage: { ...usageOf(105_000), input: 105_000 },
      };
      const id = f.sm.appendMessage(message as never);
      await fireTurnEndAll(f.pi, f.ctx, message, { turnIndex: 1 });
      assert.equal(f.sm.getLabel(id), undefined);
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("silent stop overflow: the reply stays visible, results withheld until the estimate fits", async () => {
    const f = await forcedSetup();
    try {
      const message = {
        ...assistantMessage([{ type: "text", text: "done" }], "stop", 0),
        usage: { ...usageOf(105_000), input: 105_000 },
      };
      const id = f.sm.appendMessage(message as never);
      const turn = await fireTurnEndAll(f.pi, f.ctx, message);
      assert.equal(turn.continue, true);
      const edits = turn.entries.filter((d) => d.type === "context_edit");
      assert.deepEqual(
        edits.map((d) => d.targetId),
        [f.r2, f.r1],
      );
      assert.ok(!hiddenIds(f.sm).has(id));
      assert.match(forcedDrafts(turn)[0].content, /: 2 tool results were/);
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("does not trigger: another model, a non-overflow error, compaction on, another extension's continue", async () => {
    // Another model.
    {
      const f = await forcedSetup();
      try {
        const err = errorOverflow(f.sm, "other");
        const { settle } = await runErrorEnd(f, err.message);
        assert.deepEqual(settle.entries, []);
        assert.equal(fireToolCall(f.pi, f.ctx), undefined);
      } finally {
        cleanupConfigFixture(f.fixture);
      }
    }
    // A rate limit is not an overflow.
    {
      const f = await forcedSetup();
      try {
        const message = {
          ...assistantMessage([], "error", 0),
          errorMessage: "rate limit exceeded",
        };
        f.sm.appendMessage(message as never);
        const { settle } = await runErrorEnd(f, message);
        assert.deepEqual(settle.entries, []);
      } finally {
        cleanupConfigFixture(f.fixture);
      }
    }
    // pi's compaction on (pi's default): pi owns the overflow.
    {
      const f = await forcedSetup({ piSettings: {} });
      try {
        const err = errorOverflow(f.sm);
        const { settle } = await runErrorEnd(f, err.message);
        assert.deepEqual(settle.entries, []);
      } finally {
        cleanupConfigFixture(f.fixture);
      }
    }
    // No hint reserve: off.
    {
      const f = await forcedSetup({ config: {} });
      try {
        const err = errorOverflow(f.sm);
        const { settle } = await runErrorEnd(f, err.message);
        assert.deepEqual(settle.entries, []);
      } finally {
        cleanupConfigFixture(f.fixture);
      }
    }
    // Another extension already continues (e.g. a fallback provider).
    {
      const f = await forcedSetup();
      try {
        const err = errorOverflow(f.sm);
        const { settle } = await runErrorEnd(f, err.message, true);
        assert.deepEqual(settle.entries, []);
        assert.equal(settle.continue, true);
      } finally {
        cleanupConfigFixture(f.fixture);
      }
    }
  });

  it("S4: a per-model compaction reserve sets the target (nothing withheld under it)", async () => {
    const f = await forcedSetup({
      piSettings: {
        compaction: {
          enabled: false,
          modelOverrides: {
            "claude/claude-sonnet-4-5": { reserveTokens: 2_000 },
          },
        },
      },
    });
    try {
      const err = errorOverflow(f.sm);
      const { settle } = await runErrorEnd(f, err.message);
      assert.equal(settle.continue, true);
      assert.match(
        forcedDrafts(settle)[0].content,
        /: no tool results were withheld\. /,
      );
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("gate: only a direct rewind passes; blocked calls terminate the batch", async () => {
    const f = await forcedSetup();
    try {
      await runErrorEnd(f, errorOverflow(f.sm).message);
      const blocked = {
        block: true,
        reason: FORCED_REWIND_BLOCK_REASON,
        terminate: true,
      };
      assert.deepEqual(
        fireToolCall(f.pi, f.ctx, "read", { path: "x" }),
        blocked,
      );
      assert.deepEqual(
        fireToolCall(f.pi, f.ctx, TOOL_NAME, { action: "list" }),
        blocked,
      );
      assert.deepEqual(
        fireToolCall(f.pi, f.ctx, TOOL_NAME, { action: "anchor", name: "x" }),
        blocked,
      );
      assert.equal(
        fireToolCall(f.pi, f.ctx, TOOL_NAME, { action: "rewind" }),
        undefined,
      );
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("attempts are hidden and re-sent; the third ends with D8 and lifts the gate", async () => {
    const f = await forcedSetup();
    try {
      await runErrorEnd(f, errorOverflow(f.sm).message);
      const before = JSON.stringify(fireContext(f.pi, f.ctx));
      const a1 = await textAttempt(f);
      assert.equal(a1.continue, true);
      assert.deepEqual(forcedDrafts(a1), []);
      assert.equal(JSON.stringify(fireContext(f.pi, f.ctx)), before);
      // A refused rewind is not an attempt and stays visible (D7).
      const refusal = appendToolCallAssistant(f.sm, [
        {
          id: "bad",
          name: TOOL_NAME,
          arguments: { ...REWIND_ARGS, rewindTo: "nope" },
        },
      ]);
      const refused = await f.tool.execute(
        "bad",
        { ...REWIND_ARGS, rewindTo: "nope" },
        undefined,
        undefined,
        f.ctx,
      );
      appendToolResult(
        f.sm,
        "bad",
        TOOL_NAME,
        refused.content[0].text,
        refused.details,
      );
      const r = await fireTurnEndAll(f.pi, f.ctx, refusal.message);
      assert.deepEqual(r.entries, []);
      assert.ok(!hiddenIds(f.sm).has(refusal.id));
      const a2 = await textAttempt(f);
      assert.equal(a2.continue, true);
      const a3 = await textAttempt(f);
      assert.equal(a3.continue, false);
      assert.equal(
        forcedDrafts(a3)[0].content,
        buildRewindCompactionFailedText(COMPACTION_FAILED_NO_REWIND),
      );
      assert.deepEqual(forcedDrafts(a3)[0].details, {
        status: "failed",
        reason: COMPACTION_FAILED_NO_REWIND,
      });
      assert.equal(fireToolCall(f.pi, f.ctx), undefined, "gate lifted");
      await fireAgentSettled(f.pi, f.ctx);
      assert.equal(f.pi.sendMessageCalls.length, 0, "no second D8 row");
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("the run continuing after a D8 on a turn with tool calls never starts a second forced rewind", async () => {
    const f = await forcedSetup();
    try {
      await runErrorEnd(f, errorOverflow(f.sm).message);
      await textAttempt(f);
      await textAttempt(f);
      // Attempt 3 calls a tool: the gate blocks it with `terminate`, so pi
      // ends the batch; a queued steer could still continue the loop.
      assert.equal(
        (fireToolCall(f.pi, f.ctx) as { terminate?: boolean }).terminate,
        true,
      );
      const a3 = await toolCallAttempt(f, ["tc-x"]);
      assert.equal(a3.boundary.continue, false);
      assert.equal(forcedDrafts(a3.boundary).length, 1);
      // pi continues anyway (a steer): the next turn overflows again.
      const message = {
        ...assistantMessage([], "length", 0),
        usage: { ...usageOf(99_000), output: 10 },
      };
      const id = f.sm.appendMessage(message as never);
      const again = await fireTurnEndAll(f.pi, f.ctx, message, {
        turnIndex: 5,
      });
      assert.deepEqual(again.entries, [], "no second D5, no second D8");
      assert.equal(
        f.sm.getLabel(id),
        undefined,
        "no start anchor at the limit",
      );
      // Settled: the next run gets a fresh forced rewind.
      await fireAgentSettled(f.pi, f.ctx);
      const { settle } = await runErrorEnd(f, errorOverflow(f.sm).message);
      assert.equal(settle.continue, true);
      assert.equal(forcedDrafts(settle)[0].details.status, "forced");
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("a landed rewind lifts the gate (D7); the next overflow is a fresh one (D15)", async () => {
    const f = await forcedSetup();
    try {
      await runErrorEnd(f, errorOverflow(f.sm).message);
      await requestRewind(f);
      const summary = await summaryTurn(f, SUMMARY_TEXT, "stop", {
        commit: true,
      });
      assert.equal(summary.boundary.continue, true);
      assert.equal(forcedDrafts(summary.boundary)[0].details.status, "done");
      assert.equal(fireToolCall(f.pi, f.ctx), undefined);
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("three failed summaries during a forced rewind: #93 ending plus D8, the run ends", async () => {
    const f = await forcedSetup();
    try {
      await runErrorEnd(f, errorOverflow(f.sm).message);
      await requestRewind(f);
      assert.deepEqual(fireToolCall(f.pi, f.ctx), {
        block: true,
        reason: SUMMARY_PENDING_BLOCK_REASON,
        terminate: true,
      });
      await summaryTurn(f, "no summary", "stop", { commit: true });
      await summaryTurn(f, "no summary", "stop", { commit: true });
      const last = await summaryTurn(f, "no summary", "stop", { commit: true });
      assert.equal(last.boundary.continue, false);
      assert.equal(outcomeEntries(f.sm).length, 1);
      assert.equal(
        forcedDrafts(last.boundary).at(-1)?.content,
        buildRewindCompactionFailedText(COMPACTION_FAILED_NO_REWIND),
      );
      assert.equal(fireToolCall(f.pi, f.ctx), undefined);
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("D8: a forced rewind's failed ending re-applied at agent_before_settle keeps the gate (#101)", async () => {
    const f = await forcedSetup();
    try {
      await runErrorEnd(f, errorOverflow(f.sm).message);
      await requestRewind(f);
      await summaryTurn(f, "no summary", "stop", { commit: true });
      await summaryTurn(f, "no summary", "stop", { commit: true });
      // Attempt 3 exhausts the attempts; pi drops its turn_end list.
      await summaryTurn(f, "no summary", "stop", { commit: false });
      assert.equal(outcomeEntries(f.sm).length, 0, "pi dropped the list");
      const settle = await fireAgentBeforeSettle(f.pi, f.ctx);
      assert.equal(outcomeEntries(f.sm).length, 1, "the ending is re-applied");
      assert.equal(settle.continue, false, "D8 gate, no continue");
      assert.equal(
        forcedDrafts(settle).at(-1)?.content,
        buildRewindCompactionFailedText(COMPACTION_FAILED_NO_REWIND),
      );
      assert.equal(fireToolCall(f.pi, f.ctx), undefined, "the gate is lifted");
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("D7 (fix 7): a failed summary that is not the third keeps the gate and continues", async () => {
    const f = await forcedSetup();
    try {
      await runErrorEnd(f, errorOverflow(f.sm).message);
      await requestRewind(f);
      // The anchor goes missing before the summary turn: not an exhausted attempt.
      const start = f.sm
        .getBranch()
        .find((e) => f.sm.getLabel(e.id) === START_ANCHOR_LABEL);
      assert.ok(start);
      f.pi.pi.setLabel(start.id, undefined);
      const { boundary } = await summaryTurn(f, SUMMARY_TEXT, "stop", {
        commit: true,
      });
      assert.equal(
        outcomeEntries(f.sm)[0].text,
        "Summarization failed: 'start' is no longer on the active branch. Nothing was collapsed. Try again.",
      );
      assert.deepEqual(forcedDrafts(boundary), [], "no D8 row");
      assert.equal(boundary.continue, true, "the run continues");
      assert.notEqual(fireToolCall(f.pi, f.ctx), undefined, "the gate stays");
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("S6: a second overflow during the forced rewind ends it", async () => {
    const f = await forcedSetup();
    try {
      await runErrorEnd(f, errorOverflow(f.sm).message);
      const second = errorOverflow(f.sm);
      const { settle } = await runErrorEnd(f, second.message);
      assert.equal(settle.continue, false);
      assert.equal(
        forcedDrafts(settle)[0].content,
        buildRewindCompactionFailedText(COMPACTION_FAILED_OVERFLOWED_AGAIN),
      );
      assert.ok(hiddenIds(f.sm).has(second.id));
      assert.equal(fireToolCall(f.pi, f.ctx), undefined);
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("S6 during the summary turn (D13): #93 failed ending plus D8", async () => {
    const f = await forcedSetup();
    try {
      await runErrorEnd(f, errorOverflow(f.sm).message);
      await requestRewind(f);
      const { settle } = await runErrorEnd(f, errorOverflow(f.sm).message);
      assert.equal(settle.continue, false);
      assert.equal(
        outcomeEntries(f.sm)[0].reason,
        COMPACTION_FAILED_OVERFLOWED_AGAIN,
      );
      assert.equal(
        forcedDrafts(settle).at(-1)?.content,
        buildRewindCompactionFailedText(COMPACTION_FAILED_OVERFLOWED_AGAIN),
      );
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("D14: no anchor ends right away, and no start anchor lands on the overflow", async () => {
    const f = await forcedSetup({ anchor: false });
    try {
      const err = errorOverflow(f.sm);
      const { settle } = await runErrorEnd(f, err.message);
      assert.equal(settle.continue, false);
      assert.equal(
        forcedDrafts(settle)[0].content,
        buildRewindCompactionFailedText(COMPACTION_FAILED_NO_ANCHOR),
      );
      assert.ok(
        !f.sm
          .getBranch()
          .some((e) => f.sm.getLabel(e.id) === START_ANCHOR_LABEL),
      );
      await fireAgentSettled(f.pi, f.ctx);
      // The next run that does not overflow writes `start`.
      appendTurn(f.sm, "u", "a", 100);
      fireAgentEnd(f.pi, f.ctx);
      assert.ok(
        f.sm
          .getBranch()
          .some((e) => f.sm.getLabel(e.id) === START_ANCHOR_LABEL),
      );
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("D16 (fix 6): agent_settled reads Esc from the last reply, not only the event flag", async () => {
    // pi < 1.1.0 sends no `aborted` on agent_settled: the reply carries it.
    const f = await forcedSetup();
    try {
      await runErrorEnd(f, errorOverflow(f.sm).message);
      f.sm.appendMessage(assistantMessage([], "aborted", 0) as never);
      await fireAgentSettled(f.pi, f.ctx, false);
      assert.deepEqual(f.pi.sendMessageCalls, [
        [
          {
            customType: REWIND_CUSTOM_TYPE,
            content: buildRewindCompactionFailedText(COMPACTION_FAILED_ABORTED),
            display: true,
            details: { status: "failed", reason: COMPACTION_FAILED_ABORTED },
          },
        ],
      ]);
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("D16: a run that settles mid-gate lifts it with one D8 row (aborted, or ended)", async () => {
    for (const [aborted, reason] of [
      [true, COMPACTION_FAILED_ABORTED],
      [false, COMPACTION_FAILED_RUN_ENDED],
    ] as const) {
      const f = await forcedSetup();
      try {
        await runErrorEnd(f, errorOverflow(f.sm).message);
        await fireAgentSettled(f.pi, f.ctx, aborted);
        assert.deepEqual(f.pi.sendMessageCalls, [
          [
            {
              customType: REWIND_CUSTOM_TYPE,
              content: buildRewindCompactionFailedText(reason),
              display: true,
              details: { status: "failed", reason },
            },
          ],
        ]);
        assert.equal(fireToolCall(f.pi, f.ctx), undefined);
      } finally {
        cleanupConfigFixture(f.fixture);
      }
    }
  });

  it("D16: Esc on a forced attempt hides it and ends the gate with one D8 row (#101)", async () => {
    const f = await forcedSetup();
    try {
      await runErrorEnd(f, errorOverflow(f.sm).message);
      assert.notEqual(fireToolCall(f.pi, f.ctx), undefined, "the gate is up");
      const message = assistantMessage(
        [{ type: "text", text: "partial" }],
        "aborted",
        100,
      );
      const id = f.sm.appendMessage(message as never);
      const turn = await fireTurnEndAll(f.pi, f.ctx, message);
      assert.equal(turn.continue, false);
      assert.ok(hiddenIds(f.sm).has(id), "the aborted attempt is hidden");
      assert.deepEqual(
        forcedDrafts(turn).map((d) => d.details),
        [{ status: "failed", reason: COMPACTION_FAILED_ABORTED }],
      );
      assert.equal(fireToolCall(f.pi, f.ctx), undefined, "the gate is lifted");
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("D16: Esc between turns (agent_before_settle, aborted) ends a forced rewind with one D8 row (#101)", async () => {
    const f = await forcedSetup();
    try {
      await runErrorEnd(f, errorOverflow(f.sm).message);
      const settle = await fireAgentBeforeSettle(f.pi, f.ctx, {
        outcome: "aborted",
      });
      assert.equal(settle.continue, false);
      assert.deepEqual(
        forcedDrafts(settle).map((d) => d.details),
        [{ status: "failed", reason: COMPACTION_FAILED_ABORTED }],
      );
      assert.equal(fireToolCall(f.pi, f.ctx), undefined, "the gate is lifted");
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("D16: Esc with a pending rewind keeps the aborted ending and adds one D8 row (#101)", async () => {
    const f = await forcedSetup();
    try {
      await runErrorEnd(f, errorOverflow(f.sm).message);
      await requestRewind(f, REWIND_ARGS, "tc-rewind", 90_000);
      const message = assistantMessage([], "aborted", 100);
      f.sm.appendMessage(message as never);
      const turn = await fireTurnEndAll(f.pi, f.ctx, message);
      assert.equal(turn.continue, false);
      assert.equal(outcomeEntries(f.sm)[0].status, "aborted");
      assert.deepEqual(
        forcedDrafts(turn).map((d) => d.details),
        [{ status: "failed", reason: COMPACTION_FAILED_ABORTED }],
      );
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("D11: a pending rewind's summary request overflows — withhold before the call, re-send", async () => {
    const f = await forcedSetup();
    try {
      await requestRewind(f, REWIND_ARGS, "tc-rewind", 90_000);
      const before = fireContext(f.pi, f.ctx).length;
      const err = errorOverflow(f.sm);
      const { settle } = await runErrorEnd(f, err.message);
      assert.equal(settle.continue, true);
      assert.deepEqual(
        forcedDrafts(settle),
        [],
        "no D5: the request is re-sent",
      );
      const edits = settle.entries.filter((d) => d.type === "context_edit");
      assert.deepEqual(
        edits.map((d) => [d.targetId, d.replacement === null]),
        [
          [err.id, true],
          [f.r2, false],
        ],
      );
      const wire = fireContext(f.pi, f.ctx);
      assert.equal(wire.length, before, "the frozen request, withheld");
      assert.equal(wire.at(-1)?.customType, SUMMARY_REQUEST_CUSTOM_TYPE);
      // The summary lands.
      const done = await summaryTurn(f, SUMMARY_TEXT, "stop", { commit: true });
      assert.equal(forcedDrafts(done.boundary)[0].details.status, "done");
      assert.equal(fireToolCall(f.pi, f.ctx), undefined);
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("D11: nothing to withhold on a squeezed summary — still over, D8 (the D11 check runs first)", async () => {
    const f = await forcedSetup();
    try {
      await requestRewind(f, REWIND_ARGS, "tc-rewind", 200_000);
      const message = {
        ...assistantMessage([{ type: "text", text: "## Goal" }], "length", 0),
        usage: { ...usageOf(200_000), output: 100 },
      };
      const id = f.sm.appendMessage(message as never);
      const turn = await fireTurnEndAll(f.pi, f.ctx, message);
      assert.equal(
        turn.continue,
        false,
        "no #93 re-send of a squeezed summary",
      );
      assert.ok(hiddenIds(f.sm).has(id));
      assert.equal(
        outcomeEntries(f.sm)[0].reason,
        COMPACTION_FAILED_STILL_OVER,
      );
      assert.equal(
        forcedDrafts(turn).at(-1)?.content,
        buildRewindCompactionFailedText(COMPACTION_FAILED_STILL_OVER),
      );
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("D11: still over the limit after withholding — #93 failed ending plus D8", async () => {
    const f = await forcedSetup();
    try {
      await requestRewind(f, REWIND_ARGS, "tc-rewind", 200_000);
      const { settle } = await runErrorEnd(f, errorOverflow(f.sm).message);
      assert.equal(settle.continue, false);
      assert.equal(
        outcomeEntries(f.sm)[0].reason,
        COMPACTION_FAILED_STILL_OVER,
      );
      assert.equal(
        forcedDrafts(settle).at(-1)?.content,
        buildRewindCompactionFailedText(COMPACTION_FAILED_STILL_OVER),
      );
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("D17: the hint stays quiet during the forced rewind", async () => {
    const f = await forcedSetup();
    try {
      await runErrorEnd(f, errorOverflow(f.sm).message);
      f.ctx.setContextUsage(usageAt(95, WINDOW));
      fireTurnEnd(f.pi, f.ctx);
      assert.equal(f.pi.sendMessageCalls.length, 0);
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });

  it("D9: compaction on → TUI error once per crossing, no hint; headless silent; re-read per crossing", async () => {
    const f = await forcedSetup({
      piSettings: { compaction: { enabled: true } },
    });
    try {
      f.ctx.setContextUsage(usageAt(90, WINDOW));
      fireTurnEnd(f.pi, f.ctx);
      fireTurnEnd(f.pi, f.ctx);
      assert.deepEqual(f.ctx.notifyCalls, [
        [REWIND_COMPACTION_OFF_TEXT, "error"],
      ]);
      assert.equal(f.pi.sendMessageCalls.length, 0);
      // Settings change between crossings: the next crossing hints.
      writePiSettings(f.fixture, { compaction: { enabled: false } });
      f.ctx.setContextUsage(usageAt(10, WINDOW));
      fireTurnEnd(f.pi, f.ctx);
      f.ctx.setContextUsage(usageAt(90, WINDOW));
      fireTurnEnd(f.pi, f.ctx);
      assert.equal(f.pi.sendMessageCalls.length, 1);
    } finally {
      cleanupConfigFixture(f.fixture);
    }
    const g = await forcedSetup({ piSettings: {} });
    try {
      g.ctx.hasUI = false;
      g.ctx.setContextUsage(usageAt(90, WINDOW));
      fireTurnEnd(g.pi, g.ctx);
      assert.deepEqual(g.ctx.notifyCalls, []);
      assert.equal(g.pi.sendMessageCalls.length, 0);
    } finally {
      cleanupConfigFixture(g.fixture);
    }
  });

  it("D9: project settings count only when the project is trusted", async () => {
    for (const trusted of [false, true]) {
      const fixture = makeConfigFixture();
      try {
        writeGlobalConfig(
          fixture,
          JSON.stringify({ rewindHintReserveTokens: 20_000 }),
        );
        writeProjectPiSettings(fixture, { compaction: { enabled: true } });
        const f = setup({ contextWindow: WINDOW });
        f.ctx.cwd = fixture.projectDir;
        f.ctx.isProjectTrusted = () => trusted;
        await runSessionStart(f.pi, f.ctx, fixture);
        anchorStart(f.sm, f.pi);
        f.ctx.setContextUsage(usageAt(90, WINDOW));
        fireTurnEnd(f.pi, f.ctx);
        assert.equal(f.pi.sendMessageCalls.length, trusted ? 0 : 1);
        assert.equal(f.ctx.notifyCalls.length, trusted ? 1 : 0);
      } finally {
        cleanupConfigFixture(fixture);
      }
    }
  });

  it("D9 (S3): the live session's compaction setting decides, not the disk (#101)", async () => {
    for (const live of [true, false]) {
      // The disk says the opposite of the session's in-memory settings.
      const f = await forcedSetup({
        piSettings: { compaction: { enabled: !live } },
      });
      try {
        const session = {
          sessionManager: f.sm,
          settingsManager: SettingsManager.inMemory({
            compaction: { enabled: live },
          }),
        };
        __testHooks.captureSession(session as unknown as AgentSession);
        f.ctx.setContextUsage(usageAt(90, WINDOW));
        fireTurnEnd(f.pi, f.ctx);
        assert.equal(
          f.pi.sendMessageCalls.length,
          live ? 0 : 1,
          `live=${live}`,
        );
        assert.equal(f.ctx.notifyCalls.length, live ? 1 : 0, `live=${live}`);
      } finally {
        cleanupConfigFixture(f.fixture);
      }
    }
  });

  it("D10: warns per model when the hint reserve is below pi's, only with compaction off", async () => {
    const f = await forcedSetup({
      config: {
        rewindHintReserveTokens: 10_000,
        modelOverrides: { "openai/gpt-x": { rewindHintReserveTokens: 50_000 } },
      },
      piSettings: {
        compaction: {
          enabled: false,
          modelOverrides: { "openai/gpt-x": { reserveTokens: 60_000 } },
        },
      },
    });
    try {
      assert.deepEqual(f.ctx.notifyCalls, [
        [
          buildReserveOrderWarning(10_000, 16_384, "claude/claude-sonnet-4-5"),
          "warning",
        ],
        [buildReserveOrderWarning(50_000, 60_000, "openai/gpt-x"), "warning"],
      ]);
    } finally {
      cleanupConfigFixture(f.fixture);
    }
    const g = await forcedSetup({
      config: { rewindHintReserveTokens: 10_000 },
      piSettings: {},
    });
    try {
      assert.deepEqual(g.ctx.notifyCalls, [], "compaction on: D9's case");
    } finally {
      cleanupConfigFixture(g.fixture);
    }
  });

  it("renders D8 as one error-styled row; the forced message keeps pi's default box", async () => {
    const f = await forcedSetup();
    try {
      const renderer = f.pi.messageRenderers.get(REWIND_CUSTOM_TYPE) as (
        ...a: unknown[]
      ) => { render(w: number): string[] } | undefined;
      assert.ok(renderer);
      const text = buildRewindCompactionFailedText(COMPACTION_FAILED_NO_ANCHOR);
      const row = renderer(
        {
          role: "custom",
          customType: REWIND_CUSTOM_TYPE,
          content: text,
          display: true,
          details: { status: "failed" },
        },
        { expanded: false, outputPad: 1 },
        theme,
      );
      assert.deepEqual(row?.render(1000), [` <error>${text}`]);
      assert.equal(
        renderer(
          {
            role: "custom",
            customType: REWIND_CUSTOM_TYPE,
            content: "x",
            display: true,
            details: { status: "forced" },
          },
          { expanded: false, outputPad: 1 },
          theme,
        ),
        undefined,
      );
    } finally {
      cleanupConfigFixture(f.fixture);
    }
  });
});
