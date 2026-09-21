/**
 * navigate-tree — agent-callable session tree navigation.
 *
 * See README “Implementation notes” for the user-facing narrative
 * (Anthropic tool_use↔tool_result pairing, same-loop context refresh,
 * reflection bootstrap). This file-level JSDoc carries only
 * source-internal facts the README doesn't.
 *
 * `/tree`-visible artifacts (empirical):
 *   • Synthetic assistant message right after each `branch_summary`,
 *     with one tool_call sharing the in-flight `toolCallId`.
 *   • Dangling tool_use on the anchor entry whose original
 *     tool_results were cut off by the rewind. Anthropic accepts this
 *     — the dangling tool_use is buffered behind the branch_summary's
 *     user-text rendering and the API doesn't reject it. No walk-up
 *     logic at anchor time.
 *
 * Reflection bootstrap replicates pi's own slash-command line
 * verbatim (kept symmetric so a pi rename here surfaces as the
 * runtime warning rather than silently drifting):
 *
 *   this.agent.state.messages = this.sessionManager.buildSessionContext().messages;
 *
 * Risks of the reflection approach:
 *   • If pi switches either of the two fields this extension reads —
 *     `AgentSession.prototype.prompt` or `agent.state.messages` —
 *     to ES `#` private fields, this breaks fundamentally.
 *   • If pi renames or restructures any of these fields, this breaks.
 *   • Patches `AgentSession.prototype.prompt` globally on import; not
 *     reversible without a process restart; affects every session in
 *     the pi process, including sessions that never call
 *     `navigate_tree`.
 *
 * Verified against pi 0.81.0 / 0.83.0 / 0.84.2. In-loop context
 * refresh runs through the public `context` extension event (see
 * `buildContextMessages`), not `agent.prepareNextTurn*` reflection.
 */

import { randomUUID } from "node:crypto";
import { join } from "node:path";
import {
  type AgentTool,
  estimateContextTokens,
  type StreamFn,
  type ThinkingLevel,
} from "@earendil-works/pi-agent-core";
import {
  AgentSession,
  buildSessionContext,
  CONFIG_DIR_NAME,
  collectEntriesForBranchSummary,
  type ExtensionAPI,
  generateBranchSummary,
  getAgentDir,
  keyHint,
  type ModelRegistry,
  type SessionEntry,
  type SessionManager,
  sessionEntryToContextMessages,
} from "@earendil-works/pi-coding-agent";
import { Container, Spacer, Text } from "@earendil-works/pi-tui";
import { Type } from "typebox";
import {
  buildLiveSummaryMessages,
  type CacheRequest,
  createCachePreservingStreamFn,
  detectBranchSummaryCacheMiss,
  formatBranchSummaryCacheMissNotice,
  measureSummaryCache,
  resolveSummaryCacheRetention,
} from "./cache-summary.ts";
import {
  loadTreeNavigatorConfig,
  TREE_NAVIGATOR_CONFIG_FILENAME,
} from "./config.ts";
import {
  extractTextContent,
  formatContextDelta,
  formatPct1,
  formatWindow,
  isValidName,
  LABEL_PREFIX,
  MAX_NAME_LENGTH,
  stripBranchSummaryBoilerplate,
  TOOL_NAME,
  toOneLine,
} from "./helpers.ts";
import {
  buildNoAnchorText,
  buildRewindHintText,
  collectAnchorNames,
  REWIND_HINT_CUSTOM_TYPE,
  RewindHintTracker,
} from "./rewind-hint.ts";

/**
 * Always-on anchor mandate appended to the system prompt on every agent
 * start (gated on the tool being active). Same lever as pi-napkin's vault
 * mandate: `promptGuidelines` bullets land mid-prompt; this append lands
 * at the very end of the system prompt, after <project_context> and
 * skills, and is re-applied on every prompt (never compacted). Byte-stable
 * (module constant, no per-session interpolation) for provider caching.
 */
export const ANCHOR_MANDATE = `${TOOL_NAME}: gather all context, then anchor \`context-gathered\`; list anchors and rewind after every milestone or rabbit hole / dead end to keep context low.`;

// ---------------------------------------------------------------------------
// Exported boundary constants below (MAX_SESSION_REFS, MAX_HINT_WALK_DEPTH,
// MIN_SUMMARY_FOCUS_LENGTH, MIN_REWIND_SAVINGS_TOKENS,
// MAX_SYNTHETIC_FOCUS_LENGTH).
//
// Stability: these are internal tunables. Exported only so the test suite
// can pin boundary cases by constant rather than literal. Re-tuning is
// NOT a semver-breaking change for this package — production callers
// should rely on the registered `navigate_tree` tool surface, not import
// these constants directly. The `__testHooks` JSDoc carries the same
// caveat for module-internal helpers.
// ---------------------------------------------------------------------------

// Cap on captured AgentSession refs across /new + /resume + /reload cycles.
// Worst case is ~one ref per long-lived session before reaping dead WeakRefs;
// 16 leaves headroom for the deepest session-fanout pattern observed (a few
// /resume cycles on top of a couple of /new cycles) without prematurely
// reaping a still-live session. Bump if the reaper fires while a session
// is still live.
export const MAX_SESSION_REFS = 16;
// Cap on parentId chain walks in `findLabelHint` (UX preview only;
// no need to walk to the root for a 50-char snippet).
export const MAX_HINT_WALK_DEPTH = 50;
// Floor on `summaryFocus` length (after trim) for `rewind`. The user's most
// recent instruction lives on the chain about to be collapsed; if the focus
// is shorter than this, it almost always elides that instruction (a terse
// "finish parser fix" is 17 chars and conveys nothing the next turn can
// act on). 20 is the empirical threshold below which the post-rewind turn
// reliably loses continuity — raising it forces more useful focus text
// without inviting verbosity.
export const MIN_SUMMARY_FOCUS_LENGTH = 20;
// Floor on the *apparent* token savings (`beforeTokens − tokensAtTarget`) a
// `rewind` must clear to execute. Absolute threshold, deliberately NOT
// window-relative: % semantics drift with window size, so a fixed token bar
// is the only stable shape. Every rewind carries ≈350–900 tokens of fixed
// overhead (the branch_summary adds ~300–800 permanent tokens on the kept
// chain; the synthetic assistant re-emits ~50 on every subsequent turn), so
// apparent savings OVERSTATES true freed context. Empirically the degenerate
// anchor→immediate-rewind case measures ~100 apparent tokens while a healthy
// stage measures ~20k — 4k separates them cleanly (~200×) and roughly equals
// one 300-line source-file read (~5.9k tokens measured). Below the floor a
// rewind burns a summarizer LLM call, forks a junk branch_summary, consumes
// a milestone label, and GROWS live context.
export const MIN_REWIND_SAVINGS_TOKENS = 4000;
// Cap on the `summaryFocus` length stored in the synthetic assistant's
// arguments. The full focus is passed live to `generateBranchSummary`, so
// the summarizer always sees the original; we only need a trimmed copy in
// the synthetic's args because pi's `convertToLlm` re-emits the synthetic's
// toolCall block (including its arguments) on every subsequent turn until
// another rewind. Without a cap, a 100K-char focus inflates every later
// turn's input by ~100K chars indefinitely. 1024 chars is generous — well
// above empirically useful focus length, and the agent already saw the
// full focus string when it issued the rewind.
export const MAX_SYNTHETIC_FOCUS_LENGTH = 1024;
// Hint length cap for the per-row hint shown in `list` output. 50 chars
// fits one terminal column without wrapping in typical 80-column TUIs.
const LIST_HINT_MAX_LENGTH = 50;
// Hint length cap for the hint shown in the `anchor` response. The anchor
// response is a single block of prose (not a column-aligned table) so it
// can afford a longer hint than `list`'s per-row preview.
const ANCHOR_HINT_MAX_LENGTH = 60;
// padStart width for the percentage column in `list` output. The longest
// percent label is "100.0%" = 6 chars; "99.9%" = 5 chars covers the
// realistic worst case and keeps the column tight.
const LIST_PCT_COL_WIDTH = 5;
// padEnd width for the anchor-name column in `list` output. MAX_NAME_LENGTH
// is 40, but the typical kebab-case name is 8–20 chars; 28 keeps the
// hint column visible without truncating common names.
const LIST_LABEL_COL_WIDTH = 28;
const ORIG_PROMPT_KEY = Symbol.for("navigate-tree.orig-prompt");

// Two warnings: list-site (read-only path; warns about the next turn's
// context view) and rewind-site (wrote to disk; leads with that). Both
// suggest /reload first, then restart pi, in that order.
const REFLECTION_BOOTSTRAP_WARNING_LIST =
  "⚠ reflection bootstrap missing — anchors and rewinds still work, but the next assistant turn may snapshot pre-bootstrap context. Run `/reload` (or restart pi) to recover.";
const REFLECTION_BOOTSTRAP_WARNING_REWIND =
  "⚠ reflection bootstrap missing — the rewind landed on disk but the next assistant turn may still see the pre-rewind context. Run `/reload` (or restart pi) to recover.";

// ---------------------------------------------------------------------------
// Typed views over pi internals.
//
// pi-coding-agent doesn't expose `agent`, `state`, `prepareNextTurn`, or
// `sessionManager` on `AgentSession` in its public types, but they are plain
// (non-`#`-private) fields on the class. Each cast point is a fragility
// surface for pi version bumps; grouping them here makes the dependency
// surface explicit.
// ---------------------------------------------------------------------------

interface PiInternals {
  agent: {
    state: {
      messages: unknown[];
      /** Live system prompt (cache-preserving request mirror; #33). */
      systemPrompt: string;
      /** Live tool instances (cache-preserving request mirror; #33). */
      tools: unknown[];
    };
    /** Live thinking token budgets (plain field on pi-agent-core's Agent). */
    thinkingBudgets?: unknown;
  };
  /**
   * Plain field on `AgentSession`; the extension ctx does not expose
   * settings. Used only to gate the TUI cache notice on pi's own
   * `showCacheMissNotices` setting (default off).
   */
  settingsManager?: {
    getShowCacheMissNotices?: () => boolean;
  };
  sessionManager: SessionManager;
}

function asInternals(session: AgentSession): PiInternals {
  return session as unknown as PiInternals;
}

// =============================================================================
// Reflection bootstrap & in-loop refresh
// =============================================================================

const sessionInstances: WeakRef<AgentSession>[] = [];
let seenSessions = new WeakSet<AgentSession>();

function captureSession(session: AgentSession): void {
  if (seenSessions.has(session)) return;
  seenSessions.add(session);
  sessionInstances.push(new WeakRef(session));
  // Reap dead WeakRefs occasionally so the array doesn't grow unbounded
  // across /new and /resume cycles.
  if (sessionInstances.length > MAX_SESSION_REFS) {
    for (let i = sessionInstances.length - 1; i >= 0; i--) {
      if (!sessionInstances[i].deref()) sessionInstances.splice(i, 1);
    }
  }
}

function patchAgentSessionPrototype(): void {
  const proto = AgentSession.prototype as unknown as Record<
    PropertyKey,
    unknown
  >;
  // Stash the truly-original prompt the FIRST time we patch. On subsequent
  // /reloads the value is already there — we don't overwrite, we just read it
  // back so the new wrapper still calls the original (not a previous wrapper).
  if (!proto[ORIG_PROMPT_KEY]) {
    proto[ORIG_PROMPT_KEY] = proto.prompt;
  }
  const orig = proto[ORIG_PROMPT_KEY] as (...args: unknown[]) => unknown;

  // Always replace the wrapper, even if a previous load already patched. On
  // /reload the previous wrapper closes over the previous module's
  // `sessionInstances` — if we don't replace, captures land in the dead
  // module and reflection finds nothing.
  const patched = function (this: AgentSession, ...args: unknown[]) {
    captureSession(this);
    return orig.apply(this, args);
  };
  proto.prompt = patched;
}

function findOwningSession(sm: SessionManager): AgentSession | null {
  for (const ref of sessionInstances) {
    const s = ref.deref();
    if (s && asInternals(s).sessionManager === sm) {
      return s;
    }
  }
  return null;
}

/**
 * Resolve the provider's `streamSimple` for summarization routing via the
 * PUBLIC modelRegistry API (no reflection).
 *
 * Custom providers registered via `pi.registerProvider(name, { api:
 * <custom-id>, streamSimple })` are composed into the ModelRuntime provider
 * returned by `ctx.modelRegistry.getProvider(...)` — its `streamSimple`
 * dispatches to the extension handler (provider-composer `streamWith`).
 * Without passing it as `streamFn` to `generateBranchSummary`,
 * completeSummarization falls back to the pi-ai compat registry (builtin
 * apis only) and throws "No API provider registered for api: <custom-id>".
 */
function resolveProviderStreamFn(
  modelRegistry: ModelRegistry,
  providerId: string,
): { streamFn: StreamFn } | undefined {
  try {
    const provider = modelRegistry.getProvider(providerId);
    return provider?.streamSimple
      ? { streamFn: provider.streamSimple as StreamFn }
      : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Drop `null` header-deletion markers (pi 0.84+ `ProviderHeaders` can carry
 * `string | null` values). Mirrors pi's own `withoutDeletedHeaders` in
 * agent-session.js — `generateBranchSummary` expects `Record<string, string>`.
 */
function stripNullHeaders(
  headers: Record<string, string | null> | undefined,
): Record<string, string> | undefined {
  if (!headers) return undefined;
  const entries = Object.entries(headers).filter((entry) => entry[1] !== null);
  return entries.length > 0
    ? (Object.fromEntries(entries) as Record<string, string>)
    : undefined;
}

/**
 * Session-routing headers for the summarization request.
 *
 * Pi's live turns merge these in `sdk.ts` via `mergeProviderAttributionHeaders`
 * (session headers + install-telemetry attribution). Out-of-loop summarization
 * callers — pi's own `agent-session.navigateTree` and this extension — pass only
 * auth headers, so providers that REQUIRE the session header 400 the request
 * (`MissingSessionID`; observed on opencode-go 2026-09-07: every rewind and every
 * native `/tree` summary fails deterministically).
 *
 * The tool-execute ctx exposes neither `settingsManager` (telemetry attribution)
 * nor the live routing `sessionId` on 0.81.x, so this replicates only the session
 * half of pi's merge (`getSessionHeaders` in `provider-attribution.ts`): for
 * opencode-family providers, inject `x-opencode-session` + `x-opencode-client`.
 *
 * `sessionId` is the LIVE session id when the caller can supply one
 * (`sm.getSessionId()`), so the summarization request routes to the same
 * replica/affinity bucket as the turns it summarizes. A fresh UUID is only the
 * fallback for callers that cannot supply one (upstream's `completeSummarization`
 * does the same via `uuidv7()`). The header is routing-only — on
 * openai-responses the *cache key* is `prompt_cache_key`, derived from the
 * forwarded `sessionId` option, not from this header (pi-ai sets the header from
 * `sessionId` only when the provider opts into session affinity).
 *
 * Never overrides a header the auth layer already set.
 */
function withSessionHeaders(
  model: { provider?: string; baseUrl?: string } | undefined,
  headers: Record<string, string> | undefined,
  sessionId?: string,
): Record<string, string> | undefined {
  const provider = model?.provider ?? "";
  let host = "";
  try {
    host = model?.baseUrl ? new URL(model.baseUrl).hostname : "";
  } catch {
    host = "";
  }
  const needsSession =
    provider === "opencode" ||
    provider === "opencode-go" ||
    host === "opencode.ai";
  if (!needsSession) return headers;
  const merged = { ...(headers ?? {}) };
  if (!merged["x-opencode-session"]) {
    merged["x-opencode-session"] = sessionId || randomUUID();
  }
  if (!merged["x-opencode-client"]) {
    merged["x-opencode-client"] = "pi";
  }
  return merged;
}

/**
 * Project the session tree's active branch into the messages the next LLM
 * call should see — the public-API replacement for the deleted
 * `prepareNextTurn` double-wrap (which rebuilt `messages` from
 * `sessionManager.buildSessionContext()` between every turn).
 *
 * Registered as a `context` extension handler: pi fires the `context` event
 * inside `streamAssistantResponse` before EVERY LLM call (wired through
 * `Agent.transformContext` → `runner.emitContext`), so this projection is
 * applied at the wire boundary itself rather than at a turn-boundary
 * snapshot. The loop's own `currentContext.messages` is never mutated — the
 * handler only replaces the clone that `convertToLlm` consumes.
 *
 * We always replace (no leaf-gating): every `appendMessage` advances the
 * session leaf, so a "leaf changed since last turn" heuristic would fire on
 * essentially every call anyway — and always-replace is byte-for-byte the
 * behavior of the old wrapper (per-turn rebuild from the tree).
 *
 * The parameter is typed structurally because `ReadonlySessionManager` is
 * NOT re-exported from the package barrel (only `SessionManager` and
 * `sessionEntryToContextMessages` are public); `ctx.sessionManager` at the
 * event site is `ReadonlySessionManager`, which has `buildContextEntries()`.
 */
export function buildContextMessages(sm: {
  buildContextEntries(): SessionEntry[];
}): ReturnType<typeof sessionEntryToContextMessages> {
  return sm
    .buildContextEntries()
    .flatMap((entry) => sessionEntryToContextMessages(entry));
}

function refreshAgentMessages(sm: SessionManager): boolean {
  // Manually replicate the agent-state refresh that pi's
  // commandCtx.navigateTree does after branchWithSummary. Returns true on
  // success, false if reflection couldn't find the owning AgentSession (in
  // which case the rewind is structurally complete on disk, but the next LLM
  // call will still see stale messages).
  const session = findOwningSession(sm);
  if (!session) return false;
  try {
    const sessionContext = sm.buildSessionContext();
    const agent = asInternals(session).agent;
    if (!agent?.state) return false;
    agent.state.messages = sessionContext.messages;
    return true;
  } catch {
    return false;
  }
}

// =============================================================================
// Helpers (extension-internal; pure helpers in ./helpers.ts)
// =============================================================================

function findLabeledEntry(
  sm: SessionManager,
  fullLabel: string,
): string | null {
  const path = sm.getBranch();
  for (let i = path.length - 1; i >= 0; i--) {
    if (sm.getLabel(path[i].id) === fullLabel) return path[i].id;
  }
  return null;
}

/** Shape of a `toolCall` content block on an assistant message. */
interface ToolCallBlock {
  id: string;
  name?: unknown;
}

/**
 * Find the assistant message that declared the in-flight tool call and
 * return its full tool-call batch as emitted (emission order), or `null`
 * when neither the active branch nor the captured session's live messages
 * carry it.
 *
 * `rewind` is only structurally safe as a solo call (#37): it forks the
 * tree and re-declares ONLY its own tool call in a synthetic assistant, so
 * a sibling call's result — appended by pi's sequential loop after the
 * fork — would land on the new branch with no declaring assistant and
 * brick every subsequent request. This helper is the detection seam for
 * the pre-mutation refusal guard.
 *
 * Primary source is `sm.getBranch()`: pi persists the assistant
 * SessionEntry on `message_end` before `tool_execution_start`, so the
 * in-flight assistant is already on the active branch when `execute` runs
 * (asserted by `scripts/pi-upstream-probe.mjs`). The walk matches by
 * tool-call id, not by tail position, because preceding sibling results
 * may already trail the assistant. `getBranch()` reads internal session
 * state and can throw, and tree-write lag can leave it stale; both cases
 * fall back to the same scan over the captured session's
 * `agent.state.messages`. A miss returns `null` so callers fall through to
 * the pre-guard behavior — never hard-fail on undetectable state.
 *
 * Detection fails open on shape, not count: every `toolCall` block with a
 * string `id` counts toward the returned batch, even if its `name` is
 * missing or malformed (`name` falls back to `"unknown"`), so a sibling
 * call can't be silently dropped from the count by an unexpected shape.
 */
function findInFlightAssistantToolCalls(
  sm: SessionManager,
  toolCallId: string,
): Array<{ id: string; name: string }> | null {
  const scan = (
    items: ReadonlyArray<unknown>,
  ): Array<{ id: string; name: string }> | null => {
    for (let i = items.length - 1; i >= 0; i--) {
      const item = items[i] as
        | { type?: unknown; message?: unknown }
        | null
        | undefined;
      if (!item || typeof item !== "object") continue;
      // SessionEntry: `{ type: "message", message: {...} }`. Raw
      // `agent.state.messages` entries are the message itself.
      const candidate = (item.type === "message" ? item.message : item) as
        | { role?: unknown; content?: unknown }
        | null
        | undefined;
      if (
        !candidate ||
        typeof candidate !== "object" ||
        candidate.role !== "assistant" ||
        !Array.isArray(candidate.content)
      ) {
        continue;
      }
      const toolCalls = candidate.content.filter(
        (block): block is ToolCallBlock =>
          !!block &&
          typeof block === "object" &&
          (block as { type?: unknown }).type === "toolCall" &&
          typeof (block as { id?: unknown }).id === "string",
      );
      if (toolCalls.some((block) => block.id === toolCallId)) {
        return toolCalls.map((block) => ({
          id: block.id,
          name: typeof block.name === "string" ? block.name : "unknown",
        }));
      }
    }
    return null;
  };

  let branch: SessionEntry[] | undefined;
  try {
    branch = sm.getBranch();
  } catch {
    // getBranch() reads internal session state; fall through to the
    // captured-session scan instead of aborting the rewind.
  }
  if (branch) {
    const onBranch = scan(branch);
    if (onBranch) return onBranch;
  }
  const session = findOwningSession(sm);
  const liveMessages = session
    ? asInternals(session).agent?.state?.messages
    : undefined;
  if (!Array.isArray(liveMessages)) return null;
  try {
    return scan(liveMessages);
  } catch {
    return null;
  }
}

function estimateActiveBranchTokens(sm: SessionManager): number {
  return estimateContextTokens(sm.buildSessionContext().messages).tokens;
}

function estimateAtEntry(
  entries: SessionEntry[],
  entryId: string,
  byId: Map<string, SessionEntry>,
): number {
  return estimateContextTokens(
    buildSessionContext(entries, entryId, byId).messages,
  ).tokens;
}

/**
 * Rejection copy for the rewind min-savings floor (#21). Reports the
 * apparent savings plus every active anchor — chronological root→leaf,
 * same walk order as `list`, including rewindTo itself — with its
 * cumulative context percentage, so the agent can pick a genuinely earlier
 * anchor or keep working. Move-on-collision means the tool itself never
 * creates duplicate anchor names; a manual `/tree` label `anchor:x` CAN
 * put the same name on two entries — duplicates print twice here, exactly as
 * `list` prints them today.
 *
 * Deliberately does NOT print the floor value anywhere: an agent-visible
 * numeric bar invites grinding up to it instead of doing real work.
 */
function buildMinSavingsRejection(
  rewindTo: string,
  newLabel: string,
  apparentSavings: number,
  sm: SessionManager,
  cw: number,
  allEntries: SessionEntry[],
  byId: Map<string, SessionEntry>,
): string {
  const anchors: string[] = [];
  let oldestAnchorName: string | null = null;
  for (const e of sm.getBranch()) {
    const lbl = sm.getLabel(e.id);
    if (!lbl?.startsWith(LABEL_PREFIX)) continue;
    const name = lbl.slice(LABEL_PREFIX.length);
    if (!oldestAnchorName) oldestAnchorName = name;
    anchors.push(
      `'${name}' ${formatPct1(estimateAtEntry(allEntries, e.id, byId), cw)}`,
    );
  }
  // cw=0 → percents fall back to formatPct1's k-format and the window
  // suffix is omitted, mirroring `list`'s header behavior.
  const suffix = cw > 0 ? ` of ${formatWindow(cw)}` : "";
  const guidance =
    oldestAnchorName === rewindTo
      ? "No earlier anchors — keep working and rewind later once more has accumulated above it."
      : "Rewind further back to actually free context, or keep working.";
  return `Rewinding to '${rewindTo}' (as '${newLabel}') would free only ~${formatPct1(apparentSavings, 0)} tokens. ${guidance} Active anchors: ${anchors.join(" · ")}${suffix}`;
}

/**
 * Walk parentId chain back from `fromId` and return a one-line preview of
 * the first entry that has meaningful text content. Branch summaries are
 * prefixed with `summary:` so the source is clear; user/assistant text is
 * shown as-is.
 */
function findLabelHint(
  sm: SessionManager,
  fromId: string,
  maxLen: number,
): string | null {
  let cur: string | null | undefined = fromId;
  let depth = 0;
  while (cur && depth < MAX_HINT_WALK_DEPTH) {
    const e = sm.getEntry(cur);
    if (!e) break;
    let text = "";
    let prefix = "";
    if (e.type === "branch_summary" && e.summary) {
      text = stripBranchSummaryBoilerplate(e.summary);
      prefix = "summary: ";
    } else if (e.type === "message") {
      const role = e.message.role;
      if (role === "user" || role === "assistant") {
        text = extractTextContent(e.message.content);
      }
    } else if (e.type === "custom_message") {
      text = extractTextContent(e.content);
    }
    const oneLine = toOneLine(text, maxLen - prefix.length);
    if (oneLine) return prefix + oneLine;
    cur = e.parentId;
    depth++;
  }
  return null;
}

/**
 * Build a synthetic assistant message containing a single tool_call whose id
 * matches the in-flight tool_call id. Appended after `branchWithSummary` so
 * the real tool_result lands paired with a matching tool_use.
 *
 * `usage` fields are zeroed except `totalTokens`, which is set to the
 * chain size measured BEFORE this synthetic is appended (the post-rewind
 * baseline that pi-agent-core's `estimateContextTokens` reads off the last
 * assistant). `stopReason: "toolUse"` survives Kiro's `normalizeMessages`
 * filter (which strips `error` / `aborted`); without it the synthetic would
 * be filtered out and the tool_result would re-orphan.
 */
function buildSyntheticAssistant(
  toolCallId: string,
  toolName: string,
  args: Record<string, unknown>,
  model: { api?: string; provider?: string; id?: string } | undefined,
  totalTokens: number,
) {
  return {
    role: "assistant" as const,
    content: [
      {
        type: "toolCall" as const,
        id: toolCallId,
        name: toolName,
        arguments: args,
      },
    ],
    api: model?.api ?? "unknown",
    provider: model?.provider ?? "unknown",
    model: model?.id ?? "unknown",
    stopReason: "toolUse" as const,
    timestamp: Date.now(),
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
  };
}

// =============================================================================
// Extension
// =============================================================================

interface ToolResult {
  content: Array<{ type: "text"; text: string }>;
  details: Record<string, unknown>;
  isError?: boolean;
}

function toolError(
  text: string,
  details: Record<string, unknown> = {},
): ToolResult {
  return {
    content: [{ type: "text", text }],
    // `isError` is a statement of intent only: pi-agent-core ignores the flag
    // on a *returned* result (only a thrown `execute()` is finalized as
    // failed), so without the `refusal` marker below the host records this
    // as a successful call and the TUI paints the row with `toolSuccessBg`.
    // The `tool_result` handler in the factory promotes the marker into the
    // host's real error channel.
    details: { ...details, refusal: true },
    isError: true,
  };
}

export default function (
  pi: ExtensionAPI,
  opts?: { summarize?: typeof generateBranchSummary },
) {
  // DI seam: tests inject a stub `summarize` to avoid hitting the real
  // model. Production callers (pi's extension loader) pass no second
  // argument, so this falls back to the real `generateBranchSummary`
  // import.
  const summarize = opts?.summarize ?? generateBranchSummary;
  patchAgentSessionPrototype();

  // Rewind-hint closure state (#44): fresh per factory invocation and reset
  // on every `session_start`, so `/new` / `/resume` / `/reload` sessions
  // never inherit a spent crossing or another session's config. `toolActive`
  // fails open until `before_agent_start` says otherwise.
  let toolActive = true;
  let rewindHintAtPercent: number | null = null;
  let hintTracker = new RewindHintTracker();

  // Rewind-hint config (#44): load once per session start. Every ctx getter
  // is read BEFORE the first await — `ctx.ui` / `isProjectTrusted` / `cwd` /
  // `hasUI` assert against the live session (pi-steering precedent), and
  // `loadTreeNavigatorConfig` is async.
  pi.on("session_start", async (_event, ctx) => {
    const cwd = ctx.cwd;
    const projectTrusted = ctx.isProjectTrusted();
    const hasUI = ctx.hasUI;
    const ui = ctx.ui;
    rewindHintAtPercent = null;
    hintTracker = new RewindHintTracker();
    toolActive = true;
    let threshold: number | null = null;
    let warnings: string[] = [];
    try {
      const loaded = await loadTreeNavigatorConfig({
        agentDir: getAgentDir(),
        projectPath: join(cwd, CONFIG_DIR_NAME, TREE_NAVIGATOR_CONFIG_FILENAME),
        projectTrusted,
      });
      threshold = loaded.config.rewindHintAtPercent;
      warnings = loaded.warnings;
    } catch {
      // The loader is fail-closed and should never throw; this guard exists
      // so a loader bug can never break session start. Silent: every loader
      // failure already carries a user-facing warning, and a new string here
      // would only fire on an unreachable path.
      threshold = null;
      warnings = [];
    }
    rewindHintAtPercent = threshold;
    if (hasUI) {
      for (const warning of warnings) ui.notify(warning, "warning");
    }
  });

  // Public context event: replace the wire messages with the session-tree
  // projection before every LLM call. This is the public-API replacement for
  // the deleted `agent.prepareNextTurnWithContext` per-turn refresh — it
  // fires inside `streamAssistantResponse` (via `Agent.transformContext` →
  // `runner.emitContext`), before EVERY LLM call including the turn right
  // after a mid-loop rewind, so the rewound chain (branch_summary +
  // synthetic assistant + tool_result) reaches the next API call.
  pi.on("context", (_event, ctx) => ({
    messages: buildContextMessages(ctx.sessionManager),
  }));

  // Anchor mandate (#31): a before_agent_start append lands at the END of
  // the system prompt — stronger than the mid-prompt Guidelines block, and
  // re-applied on every prompt instead of living in the conversation. Skip
  // only when the tool is verifiably absent from the active set; fail-open
  // when `selectedTools` is undefined (this extension always registers it).
  // The same check gates the rewind hint (#44): an inactive tool must not
  // fire `turn_end` nudges the model can't act on.
  pi.on("before_agent_start", async (event) => {
    const selected = event.systemPromptOptions?.selectedTools;
    toolActive = !Array.isArray(selected) || selected.includes(TOOL_NAME);
    if (!toolActive) return {};
    return { systemPrompt: `${event.systemPrompt}\n\n${ANCHOR_MANDATE}` };
  });

  // Refusals must render (and be recorded) as failed tool calls. A returned
  // `isError: true` never reaches the host's error channel — pi-agent-core
  // only finalizes a call as failed when `execute()` throws, overwriting the
  // flag with its own boolean — so `toolError` tags its details with
  // `refusal: true` and this handler flips the host's `isError`, which drives
  // the TUI's `toolErrorBg` shell, the transcript entry, and any other
  // extension's `tool_result` view. Content is left untouched, so the model
  // still reads the refusal copy.
  pi.on("tool_result", (event) => {
    if (event.toolName !== TOOL_NAME) return undefined;
    const details = event.details as { refusal?: boolean } | undefined;
    return details?.refusal === true ? { isError: true } : undefined;
  });

  // Rewind hint (#44): one check per turn, after the turn's tool results and
  // before the agent loop's steering drain — so a `pi.sendMessage` from this
  // handler is delivered in the same run. Synchronous by design: no await
  // before any ctx use. Fires once per threshold crossing; see
  // `RewindHintTracker` for the re-arm semantics.
  pi.on("turn_end", (_event, ctx) => {
    if (!toolActive || rewindHintAtPercent === null) return;
    const usage = ctx.getContextUsage();
    // `percent == null` is pi's post-compaction state (unknown tokens), and
    // `contextWindow <= 0` has no meaningful bar to compare against. Both
    // are no-ops that leave the crossing unspent.
    if (!usage || usage.percent == null || usage.contextWindow <= 0) return;
    if (usage.percent < rewindHintAtPercent) {
      // Below the bar: re-arm only.
      hintTracker.observe(usage.percent, rewindHintAtPercent);
      return;
    }
    // Collect anchors BEFORE marking the crossing spent: a throwing
    // `getBranch` / `getLabel` must leave the crossing unspent so the next
    // turn can retry — never burn the one shot on a failed state read.
    // `sm` is captured BEFORE the throw-capable `pi.sendMessage` below, so
    // the fallback append never re-reads a lazy ctx getter after the throw.
    // Typed via `typeof ctx.sessionManager` (ReadonlySessionManager, not
    // re-exported from the package root).
    let sm: typeof ctx.sessionManager;
    let anchorNames: string[];
    try {
      sm = ctx.sessionManager;
      anchorNames = collectAnchorNames(sm);
    } catch {
      return;
    }
    if (!hintTracker.observe(usage.percent, rewindHintAtPercent)) return;
    if (anchorNames.length > 0) {
      const text = buildRewindHintText(usage.percent, usage.contextWindow);
      try {
        // Pinned delivery semantics — do not "improve": NO options bag
        // (never `triggerTurn`, never `sendUserMessage`). Streaming → the
        // loop steers and reacts once in the same run; idle → a plain
        // append, no turn started (the user-confirmation property is
        // intentional).
        pi.sendMessage({
          customType: REWIND_HINT_CUSTOM_TYPE,
          content: text,
          display: true,
        });
      } catch {
        // Defensively unreachable by construction on pi 0.84.2 / 0.85.1:
        // `ctx.getContextUsage()` — this handler's first API touch — asserts
        // the same staleness flag as `pi.sendMessage`, and after `bindCore`
        // `sendMessage`'s only synchronous throw is that flag (delivery
        // delegates to an async wrapper that catches). Kept as
        // defense-in-depth for host drift or a future async window in this
        // handler: the captured SessionManager outlives `assertActive()`,
        // the lazy ctx getters do not. The direct append is safe — the
        // `context` handler rebuilds wire messages from the session tree, so
        // the text still reaches the next LLM call; only the live TUI render
        // is deferred. Unlike pi-napkin's `onFallbackFailure`, the notify
        // below is itself try-guarded: a failure path must never throw.
        try {
          (sm as Partial<SessionManager>).appendCustomMessageEntry?.(
            REWIND_HINT_CUSTOM_TYPE,
            text,
            true,
          );
        } catch (err) {
          try {
            if (ctx.hasUI) {
              ctx.ui.notify(
                `${TOOL_NAME}: could not deliver the rewind hint (${err instanceof Error ? err.message : String(err)}).`,
                "warning",
              );
            }
          } catch {
            // Stale ctx getters — nothing left to surface with.
          }
        }
      }
    } else if (ctx.hasUI) {
      ctx.ui.notify(
        buildNoAnchorText(usage.percent, usage.contextWindow),
        "warning",
      );
    }
  });

  pi.registerTool({
    name: TOOL_NAME,
    label: "Navigate Tree",
    // Stateful: every action mutates SessionManager; concurrent calls would
    // race on `leafId` / `labelsById` and produce an undefined tree.
    executionMode: "sequential",
    promptGuidelines: [
      `${TOOL_NAME}: when rewinding, prefer the oldest anchor that keeps what you'd otherwise re-read or re-derive; persist durable findings to files first.`,
      `${TOOL_NAME}: don't rewind while a user decision or unresolved question is pending — ask the user instead.`,
    ],
    description: `Long-session context management via the pi session tree. Anchor named milestones, then collapse work between them into a model-generated summary to free context.
\`rewind\` does not restore prior state: it forks a sibling branch from the anchor and continues forward from a model-generated summary.

Operations (set \`action\`):
  • 'anchor', name='<milestone-name>': label the current point. Anchor at the start of a stage you'll summarize (e.g. 'impl-start').
  • 'rewind', rewindTo='<existing>', newLabel='<new>': collapse work between rewindTo and the current leaf into a branch_summary labeled newLabel, so rewinds can chain.
  • 'list': show all anchors on the active branch, oldest first, with cumulative context % at each.

\`name\` (anchor) and \`newLabel\` (rewind) write into one shared anchor namespace: re-using an existing label moves it to the new entry, and everything written there is addressable as a future \`rewindTo\`. Avoid the reserved \`${LABEL_PREFIX}\` prefix.`,
    promptSnippet:
      "Use to anchor named milestones and rewind the conversation tree to a prior point with a model-generated summary, for token-efficient long autonomous sessions.",
    // The schema is intentionally a flat `Type.Object` with everything-but-
    // `action` optional, with action-conditional required-ness enforced at
    // runtime in `execute`. Discriminated unions / property-level required-
    // when-action shapes break the Kiro/CodeWhisperer adapter, which forwards
    // `inputSchema.json` verbatim and 400s on non-`type: "object"` roots.
    // The runtime guards in `execute` provide the conditional-required
    // behavior the schema can't express.
    parameters: Type.Object({
      action: Type.Union(
        [Type.Literal("anchor"), Type.Literal("rewind"), Type.Literal("list")],
        {
          description: "Which operation to perform.",
        },
      ),
      name: Type.Optional(
        Type.String({
          description: `Required when action='anchor'. Kebab-case milestone label, max ${MAX_NAME_LENGTH} chars.`,
        }),
      ),
      rewindTo: Type.Optional(
        Type.String({
          description: `Required when action='rewind'. Kebab-case name of an existing anchor on the active branch to rewind to.`,
        }),
      ),
      newLabel: Type.Optional(
        Type.String({
          description: `Required when action='rewind'. Kebab-case label for the resulting branch_summary entry; reusable as a future rewindTo.`,
        }),
      ),
      summaryFocus: Type.Optional(
        Type.String({
          description: `Required when action='rewind'; ≥${MIN_SUMMARY_FOCUS_LENGTH} chars after trim. Encode: (1) the user's most recent instruction verbatim, (2) what's done in the collapsed segment, (3) what's left to do as a next action.`,
        }),
      ),
    }),
    execute: async (toolCallId, params, signal, _onUpdate, ctx) => {
      const sm = ctx.sessionManager as SessionManager;
      const p = params as {
        action: "anchor" | "rewind" | "list";
        name?: string;
        rewindTo?: string;
        newLabel?: string;
        summaryFocus?: string;
      };

      // --- list ---
      if (p.action === "list") {
        const path = sm.getBranch();
        const allEntries = sm.getEntries();
        const byId = new Map<string, SessionEntry>();
        for (const e of allEntries) byId.set(e.id, e);
        const cw = ctx.model?.contextWindow ?? 0;
        const totalTokens = estimateActiveBranchTokens(sm);
        const reflectionOk = !!findOwningSession(sm);

        const lines: string[] = [];
        for (const e of path) {
          const lbl = sm.getLabel(e.id);
          if (lbl?.startsWith(LABEL_PREFIX)) {
            const name = lbl.slice(LABEL_PREFIX.length);
            const tokensAt = estimateAtEntry(allEntries, e.id, byId);
            const pct = formatPct1(tokensAt, cw).padStart(LIST_PCT_COL_WIDTH);
            const hint = findLabelHint(sm, e.id, LIST_HINT_MAX_LENGTH);
            const hintPart = hint ? `  (after: “${hint}”)` : "";
            lines.push(
              `  ${pct}  ${name.padEnd(LIST_LABEL_COL_WIDTH)}${hintPart}`,
            );
          }
        }

        const reflectionWarning = reflectionOk
          ? ""
          : ` · ${REFLECTION_BOOTSTRAP_WARNING_LIST}`;
        const header = `[list] · ${lines.length} label${lines.length === 1 ? "" : "s"} · ctx ${formatPct1(totalTokens, cw)}${cw > 0 ? ` of ${formatWindow(cw)}` : ""}${reflectionWarning}`;
        const body = lines.length
          ? `Active labels (root → leaf):\n${lines.join("\n")}`
          : "No labels on the active branch.";
        return {
          content: [{ type: "text", text: `${header}\n\n${body}` }],
          details: {
            count: lines.length,
            contextTokens: totalTokens,
            contextWindow: cw,
            reflectionOk,
          },
        };
      }

      // --- anchor ---
      if (p.action === "anchor") {
        if (!isValidName(p.name)) {
          return toolError(
            `anchor requires \`name\` in kebab-case, max ${MAX_NAME_LENGTH} chars (e.g. 'impl-start').`,
          );
        }
        const leafId = sm.getLeafId();
        if (!leafId) {
          return toolError("No session entries yet — nothing to anchor.");
        }
        // Write the new label first, then clear the prior. If the second
        // setLabel throws, two labels of the same name briefly coexist on
        // the active branch — `findLabeledEntry` walks leaf→root and
        // returns the leaf-side match, so navigation behavior is correct
        // during the overlap. The pre-PR "no enforcement" semantics already
        // tolerated this. The reverse order (clear-then-set) was move-then-
        // lose under failure: a partial collapse left the active branch
        // with no anchor of the requested name at all.
        const fullLabel = LABEL_PREFIX + p.name;
        const prior = findLabeledEntry(sm, fullLabel);
        pi.setLabel(leafId, fullLabel);
        if (prior && prior !== leafId) {
          pi.setLabel(prior, undefined);
        }
        const cw = ctx.model?.contextWindow ?? 0;
        const tokensHere = estimateActiveBranchTokens(sm);
        const labelHint = findLabelHint(sm, leafId, ANCHOR_HINT_MAX_LENGTH);
        const positionLine = `${formatPct1(tokensHere, cw)}${cw > 0 ? ` of ${formatWindow(cw)}` : ""}`;
        const hintLine = labelHint ? ` (after: “${labelHint}”)` : "";
        return {
          content: [
            {
              type: "text",
              text:
                `[anchor '${p.name}'] set at ${positionLine}${hintLine}\n\n` +
                `Once real work has accumulated after this anchor, collapse it into a summary with: ${TOOL_NAME}(action='rewind', rewindTo='<oldest appropriate anchor>', newLabel='<milestone-name>', summaryFocus='<≥${MIN_SUMMARY_FOCUS_LENGTH}-char focus: latest user instruction + done + remaining>').`,
            },
          ],
          details: {
            label: p.name,
            entryId: leafId,
            contextTokens: tokensHere,
            labelHint,
            movedFromPriorEntry: prior && prior !== leafId ? prior : null,
          },
        };
      }

      // --- rewind ---
      if (!isValidName(p.rewindTo)) {
        return toolError(
          `rewind requires \`rewindTo\` in kebab-case, max ${MAX_NAME_LENGTH} chars.`,
        );
      }
      if (!isValidName(p.newLabel)) {
        return toolError(
          `rewind requires \`newLabel\` in kebab-case, max ${MAX_NAME_LENGTH} chars.`,
        );
      }
      if (
        !p.summaryFocus ||
        p.summaryFocus.trim().length < MIN_SUMMARY_FOCUS_LENGTH
      ) {
        const focusLen = p.summaryFocus?.trim().length ?? 0;
        return toolError(
          `\`summaryFocus\` must be ≥${MIN_SUMMARY_FOCUS_LENGTH} chars after trim (got ${focusLen}). The user's most recent instruction (which triggered this rewind) lives on the chain that's about to be collapsed — if summaryFocus doesn't preserve it, the post-rewind turn won't know what's left to do.\n\n` +
            `Include in summaryFocus:\n` +
            `  1. the user's most recent instruction verbatim,\n` +
            `  2. which parts have already been done in the work being collapsed,\n` +
            `  3. which parts remain unactioned.`,
        );
      }

      // Solo-batch guard (#37): a rewind must be the only tool call in its
      // assistant batch. Pi runs a batch containing this sequential-mode tool
      // sequentially; `branchWithSummary` then puts the declaring assistant
      // on the abandoned branch, and the synthetic assistant re-declares
      // ONLY this call — so any sibling result pi appends afterwards would
      // land on the new branch with no declaring `tool_calls`, 400-ing every
      // subsequent request. Refuse before any mutation or LLM call. A missed
      // detection returns null and falls through to today's behavior. The
      // validations above ran first so the copy can interpolate a valid
      // rewindTo.
      const inFlightBatch = findInFlightAssistantToolCalls(sm, toolCallId);
      if (inFlightBatch && inFlightBatch.length > 1) {
        return toolError(
          `rewind must be the only tool call in its batch — after it, the next context is everything up to '${p.rewindTo}' plus the new summary, so any other call's result would be orphaned — never read by anyone. Those other calls already ran; re-issue only the rewind, alone.`,
          {
            rejected: "batched-rewind",
            batchedToolCalls: inFlightBatch.map((call) => call.name),
          },
        );
      }

      const target = findLabeledEntry(sm, LABEL_PREFIX + p.rewindTo);
      if (!target) {
        return toolError(
          `No label '${p.rewindTo}' on the active branch. Use action='list' to see available labels.`,
        );
      }

      const oldLeaf = sm.getLeafId();
      if (!oldLeaf || oldLeaf === target) {
        return toolError(`Already at '${p.rewindTo}' — nothing to summarize.`);
      }

      // Token math hoisted ABOVE the model check / auth await (#21): these
      // are pure SessionManager reads, and the min-savings floor below must
      // be evaluable — and able to reject — before any auth resolution.
      // `allEntries`/`byId` are hoisted out of the parented-assistant branch
      // because the floor rejection copy reports per-anchor percentages from
      // the same maps.
      //
      // The leaf at execute time is the assistant that just streamed the
      // rewind tool call. Its `usage.input` is the *minimum* of recent API
      // calls in this turn, so estimating from it understates what the
      // user just saw. Use the chain up to its parent (which has the prior
      // assistant's usage as baseline) so beforeTokens matches the value
      // `list` would have reported on the previous turn.
      const oldLeafEntry = sm.getEntry(oldLeaf);
      const allEntries = sm.getEntries();
      const byId = new Map<string, SessionEntry>();
      for (const e of allEntries) byId.set(e.id, e);
      let beforeTokens: number;
      if (
        oldLeafEntry &&
        oldLeafEntry.type === "message" &&
        oldLeafEntry.message.role === "assistant" &&
        oldLeafEntry.parentId
      ) {
        beforeTokens = estimateAtEntry(allEntries, oldLeafEntry.parentId, byId);
      } else {
        beforeTokens = estimateActiveBranchTokens(sm);
      }
      const tokensAtTarget = estimateAtEntry(allEntries, target, byId);
      const contextWindow = ctx.model?.contextWindow ?? 0;

      const { entries } = collectEntriesForBranchSummary(sm, oldLeaf, target);
      if (entries.length === 0) {
        return toolError(
          `No entries between leaf and '${p.rewindTo}' — nothing to summarize.`,
        );
      }
      // Chained-rewind-no-turns guard: bail if the only message between
      // leaf and target matches our synthetic shape (single navigate_tree
      // toolCall block + stopReason "toolUse" + zero usage), meaning the
      // agent didn't append a real turn between rewinds. Keep only message
      // entries — label / compaction / branch_summary / etc. carry no
      // rewindable semantic content for this guard. The synthetic-shape
      // predicate avoids false-positives on real navigate_tree calls
      // (which have nonzero usage from the model).
      const messageEntries = entries.filter((e) => e.type === "message");
      if (messageEntries.length === 1) {
        const lone = messageEntries[0];
        if (lone.type === "message" && lone.message.role === "assistant") {
          const msg = lone.message as {
            content: Array<{ type?: string; name?: string }>;
            stopReason?: string;
            usage?: { input?: number; output?: number };
          };
          const block = msg.content[0];
          const isSyntheticShape =
            msg.stopReason === "toolUse" &&
            (msg.usage?.input ?? 0) === 0 &&
            (msg.usage?.output ?? 0) === 0;
          if (
            block &&
            block.type === "toolCall" &&
            block.name === TOOL_NAME &&
            isSyntheticShape
          ) {
            return toolError(
              `Already at synthetic boundary — no work to summarize. Append at least one turn between rewinds.`,
            );
          }
        }
      }

      // Minimum-savings floor (#21): refuse rewinds whose apparent savings
      // is below MIN_REWIND_SAVINGS_TOKENS — collapsing a near-empty segment
      // burns a summarizer LLM call, forks a junk branch_summary, consumes
      // a milestone label, and grows live context. Placed AFTER the
      // synthetic-boundary guard so the chained rewind-with-no-turns case
      // keeps its specific diagnostic, and BEFORE the model/auth checks so
      // the rejection fires without touching provider config.
      const apparentSavings = beforeTokens - tokensAtTarget;
      if (apparentSavings < MIN_REWIND_SAVINGS_TOKENS) {
        return toolError(
          buildMinSavingsRejection(
            p.rewindTo,
            p.newLabel,
            apparentSavings,
            sm,
            contextWindow,
            allEntries,
            byId,
          ),
          { rejected: "min-savings", apparentSavings },
        );
      }

      if (!ctx.model) {
        return toolError("No model configured for summarization.");
      }

      const auth = await ctx.modelRegistry.getApiKeyAndHeaders(ctx.model);
      if (!auth.ok) {
        return toolError(`Auth resolution failed: ${auth.error}`);
      }

      // -----------------------------------------------------------------
      // Cache-preserving summary request (#33)
      //
      // `generateBranchSummary` builds a cold, standalone request (generic
      // summarization prompt, serialized blob, no tools, cacheRetention
      // "none", fresh session id). The live turns being collapsed were just
      // cache-served, so the summary re-bills the whole input. We cannot
      // patch upstream (the extension loader aliases @earendil-works/* to the
      // host process), but the `streamFn` seam we already inject is the exact
      // point where upstream's cold context/options exist — wrap it and
      // rewrite them to the live shape. Every live input is read
      // defensively: on any miss we leave `summaryCacheRequest` null and the
      // wrapper delegates today's cold request (with a user-visible warning).
      //
      // Two evidence guards also refuse the cache path and fall back to the
      // cold request, because raw branch evidence must beat a cache hit when
      // the two disagree:
      //
      //   - "branch-crosses-compaction": `buildContextEntries()` applies the
      //     compaction cut — entries before the latest compaction's
      //     `firstKeptEntryId` are dropped from the live projection. When a
      //     rewind segment reaches older than that cut, the cache payload
      //     would summarize the lossy compacted projection while the legacy
      //     path summarizes the raw segment. The compaction entry itself is
      //     retained, so `branchStartRetained` cannot catch this; the
      //     id-difference check below does (every other entry type — labels,
      //     model changes, thinking-level changes — stays in the projection,
      //     so only compaction-dropped entries are missing).
      //   - "branch-start-not-retained": no message from the collapsed
      //     segment survived into the payload (labels-only segment) or the
      //     newest message alone exceeded the token budget. The legacy path
      //     then either summarizes the raw evidence or returns "No content to
      //     summarize" before any wire call.
      // -----------------------------------------------------------------
      const providerStream = resolveProviderStreamFn(
        ctx.modelRegistry,
        ctx.model.provider,
      );
      let summaryCacheRequest: CacheRequest | null = null;
      let summaryCacheFallbackReason: string | undefined;
      let branchStartRetained = true;

      if (process.env.PI_NAVIGATE_TREE_SUMMARY_CACHE === "0") {
        // Kill switch: an env-var escape hatch to force the pre-#33 cold
        // request without unloading the extension.
        summaryCacheFallbackReason = "disabled";
      } else if (!providerStream) {
        summaryCacheFallbackReason = "no-provider-stream";
      } else {
        const owning = findOwningSession(sm);
        const internals = owning ? asInternals(owning) : undefined;
        const liveTools = internals?.agent?.state?.tools;
        if (!internals) {
          summaryCacheFallbackReason = "reflection-missing";
        } else if (!Array.isArray(liveTools)) {
          summaryCacheFallbackReason = "no-live-tools";
        } else {
          // Public accessor first (0.81+), reflected field as backstop. Both
          // are read here AND in refreshAgentMessages; the reflection probe
          // covers the plain-field shape.
          let systemPrompt: string | undefined;
          if (typeof ctx.getSystemPrompt === "function") {
            try {
              systemPrompt = ctx.getSystemPrompt();
            } catch {
              systemPrompt = undefined;
            }
          }
          if (typeof systemPrompt !== "string" || systemPrompt.length === 0) {
            const reflected = internals.agent?.state?.systemPrompt;
            systemPrompt =
              typeof reflected === "string" ? reflected : undefined;
          }
          if (typeof systemPrompt !== "string" || systemPrompt.length === 0) {
            summaryCacheFallbackReason = "no-system-prompt";
          } else {
            const contextEntries = sm.buildContextEntries();
            const contextIds = new Set(contextEntries.map((e) => e.id));
            // Evidence-loss guard: any segment entry missing from the live
            // projection means the compaction cut dropped raw evidence, so
            // the cache payload would diverge from the legacy raw segment.
            // Use the id-difference predicate — NOT a "segment contains a
            // compaction" check: a segment that contains the compaction
            // entry but whose target sits at/after `firstKeptEntryId` loses
            // no evidence and must still take the cache path.
            if (entries.some((e) => !contextIds.has(e.id))) {
              summaryCacheFallbackReason = "branch-crosses-compaction";
            } else {
              const branchEntryIds = new Set(entries.map((e) => e.id));
              // Upstream's default `reserveTokens`; the fallback path will
              // recompute the same budget inside generateBranchSummary.
              const tokenBudget = (ctx.model.contextWindow || 128000) - 16384;
              const built = buildLiveSummaryMessages({
                contextEntries,
                branchEntryIds,
                inFlightToolCallId: toolCallId,
                tokenBudget,
                focus: p.summaryFocus,
              });
              branchStartRetained = built.branchStartRetained;
              if (!built.branchStartRetained) {
                summaryCacheFallbackReason = "branch-start-not-retained";
              } else {
                const sessionId = sm.getSessionId();
                // Public since 0.81.0; `ctx.thinkingLevel` only exists >=
                // 0.84.2, so never read it directly. "off" is omitted,
                // mirroring the live loop's request shape.
                const reasoning =
                  typeof pi.getThinkingLevel === "function"
                    ? pi.getThinkingLevel()
                    : undefined;
                summaryCacheRequest = {
                  context: {
                    systemPrompt,
                    messages: built.messages,
                    tools: liveTools as AgentTool[],
                  },
                  // Provider-scoped `PI_CACHE_RETENTION` (auth.env) must win
                  // over `process.env`, mirroring pi-ai's
                  // `getProviderEnvValue`: the live turns resolve retention
                  // from the provider env, so a provider-scoped "long" with
                  // a process-level "short" would make live=long /
                  // summary=short and silently miss.
                  cacheRetention: resolveSummaryCacheRetention(auth.env),
                  ...(sessionId ? { sessionId } : {}),
                  ...(typeof reasoning === "string" && reasoning !== "off"
                    ? { reasoning: reasoning as ThinkingLevel }
                    : {}),
                  ...(internals.agent?.thinkingBudgets
                    ? { thinkingBudgets: internals.agent.thinkingBudgets }
                    : {}),
                };
              }
            }
          }
        }
      }

      // Always wrap when a provider stream exists: with a request it rewrites
      // the live shape, without one it delegates today's cold request
      // byte-for-byte (the `used` flag records which happened).
      const cacheWrapped = providerStream
        ? createCachePreservingStreamFn({
            realStreamFn: providerStream.streamFn,
            request: summaryCacheRequest,
          })
        : undefined;

      // `auth.baseUrl` (OAuth/credential-derived endpoint, e.g.
      // githubCopilotOAuth) must be applied to the model, mirroring pi's
      // own `_getSummarizationRequestAuth` (`result.auth.baseUrl ?
      // { ...model, baseUrl: result.auth.baseUrl } : model`). Use this one
      // resolved model for BOTH the summarization call and the session
      // headers: the routing header is derived from the endpoint actually
      // contacted, so reading the un-overridden `ctx.model` there would
      // diverge.
      const requestModel = auth.baseUrl
        ? { ...ctx.model, baseUrl: auth.baseUrl }
        : ctx.model;

      const result = await summarize(entries, {
        model: requestModel,
        apiKey: auth.apiKey ?? "",
        headers: withSessionHeaders(
          requestModel,
          stripNullHeaders(auth.headers),
          sm.getSessionId(),
        ),
        ...(auth.env ? { env: auth.env } : {}),
        signal: signal ?? new AbortController().signal,
        customInstructions: p.summaryFocus,
        // Route through the composed provider's `streamSimple` (public
        // modelRegistry API) instead of the pi-ai compat registry.
        // Custom providers registered via `pi.registerProvider(name,
        // { api: <custom-id>, streamSimple })` are NOT visible to the
        // compat registry (which only knows builtin apis) — without this,
        // rewind fails with "No API provider registered for api:
        // <custom-id>" for any custom-api provider (e.g. commandcode 0.5.x
        // with api "commandcode-custom"). The wrapper delegates to it in
        // both the live-prefix and fallback cases.
        ...(cacheWrapped ? { streamFn: cacheWrapped.streamFn } : {}),
      });
      if (result.aborted) {
        return toolError("Summarization aborted.");
      }
      if (result.error || !result.summary) {
        return toolError(
          `Summarization failed: ${result.error ?? "no summary text"}`,
        );
      }

      // Cache outcome measured from provider usage. Mode reflects whether a
      // cache-preserving request was BUILT; `used` reflects whether the
      // wrapper actually delegated it (false when the summarizer is stubbed,
      // aborted before the wire call, or the fallback path ran).
      const summaryCacheMode = summaryCacheRequest ? "live-prefix" : "fallback";
      const summaryCacheStats = measureSummaryCache(
        result.usage ?? { input: 0, cacheRead: 0, cacheWrite: 0 },
      );

      // Fork-faithful miss accounting (cad0p/pi@eval/branch-summary-prompt):
      // scan the entries as they stand (the branch_summary is not appended
      // yet) and compare the summary's measured usage against the previous
      // request's baseline — the baseline survives branch_summary entries
      // (the summary request reuses the live prefix), while a summary miss
      // after a model switch is suppressed as expected re-billing. HITS ARE
      // SILENT — the footer/session totals already cover them; only an
      // actionable miss is worth a transcript line. The fallback path (cache
      // request not built) is NOT special-cased: the legacy cold request
      // measures as a miss exactly when the numbers say so, and
      // `fallbackReason` lives in `details.summaryCache` only. Upstream pi
      // renders cache notices as transcript lines gated by
      // `showCacheMissNotices`; the extension does the same by storing the
      // notice string and rendering it from the tool's `renderResult`. The
      // model-visible tool-result content stays cache-free (the model was
      // proven to echo the line verbatim). The setting is read reflectively
      // off the captured session — the extension ctx does not expose
      // `settingsManager`. `Date.now()` is the summary timestamp, as the
      // fork's `navigateTree` call site uses. Headless runs and setting-off
      // runs surface the same numbers via `details.summaryCache`.
      const summaryCacheMiss = result.usage
        ? detectBranchSummaryCacheMiss(
            allEntries,
            result.usage,
            requestModel.provider,
            requestModel.id,
            Date.now(),
            {
              getModel: (provider, model) =>
                ctx.modelRegistry.find(provider, model),
            },
          )
        : undefined;
      let summaryCacheNotice: string | null = null;
      if (summaryCacheMiss) {
        const notice = formatBranchSummaryCacheMissNotice(summaryCacheMiss);
        if (notice !== null) {
          let showCacheNotices = false;
          try {
            const owning = findOwningSession(sm);
            showCacheNotices =
              (owning
                ? asInternals(owning).settingsManager
                : undefined
              )?.getShowCacheMissNotices?.() ?? false;
          } catch {
            // Unreadable settings -> stay silent (mirror pi's default off).
            showCacheNotices = false;
          }
          if (showCacheNotices) summaryCacheNotice = notice;
        }
      }

      // Move the tree.
      const summaryId = sm.branchWithSummary(target, result.summary, {
        readFiles: result.readFiles ?? [],
        modifiedFiles: result.modifiedFiles ?? [],
      });

      // Chain-validity invariants once `branchWithSummary` succeeds: a
      // synthetic must land on every path with toolCallId === this
      // in-flight call (so pi's appended tool_result pairs), and
      // stopReason: "toolUse" (survives Kiro's normalizeMessages filter
      // — see `buildSyntheticAssistant` JSDoc). The synthetic append
      // sits OUTSIDE the try so it runs exactly once regardless of
      // which earlier step threw. newLabel write moves before clear,
      // mirroring `anchor`'s move-on-collision so duplicate anchors
      // can't survive a chained rewind.
      const fullLabelEnd = LABEL_PREFIX + p.newLabel;
      let priorLabelEnd: ReturnType<typeof findLabeledEntry> = null;
      let tokensAtNewLeaf = 0;
      let originalErr: unknown;
      let salvageDetail = "";
      let failedStep:
        | "lookup"
        | "setLabelEnd"
        | "clearPrior"
        | "estimate"
        | null = null;
      try {
        failedStep = "lookup";
        priorLabelEnd = findLabeledEntry(sm, fullLabelEnd);
        failedStep = "setLabelEnd";
        pi.setLabel(summaryId, fullLabelEnd);
        // `summaryId` was freshly allocated by branchWithSummary above;
        // no pre-existing label can already point at it.
        if (priorLabelEnd) {
          failedStep = "clearPrior";
          pi.setLabel(priorLabelEnd, undefined);
        }

        // Compute afterTokens NOW — before we append the synthetic. This
        // captures the chain size at the new leaf (branch_summary) using
        // the prior real assistant's usage as the baseline.
        failedStep = "estimate";
        tokensAtNewLeaf = estimateActiveBranchTokens(sm);
        failedStep = null;
      } catch (err) {
        originalErr = err;
        // Best-effort retry of the specific failed step (pi.setLabel is
        // idempotent under re-application). Per-step recovery shape:
        //   - setLabelEnd: retry pi.setLabel(summaryId, fullLabelEnd).
        //   - clearPrior:  retry pi.setLabel(priorLabelEnd, undefined).
        //   - lookup / estimate: no retry — either prior state unknown
        //     or both labels already wrote; redundant retry would mask
        //     the real cause.
        if (failedStep === "setLabelEnd") {
          try {
            pi.setLabel(summaryId, fullLabelEnd);
          } catch (retryErr) {
            salvageDetail = `newLabel retry failed: ${
              retryErr instanceof Error ? retryErr.message : String(retryErr)
            }`;
          }
        } else if (failedStep === "clearPrior" && priorLabelEnd) {
          try {
            pi.setLabel(priorLabelEnd, undefined);
          } catch (retryErr) {
            salvageDetail = `prior-clear retry failed: ${
              retryErr instanceof Error ? retryErr.message : String(retryErr)
            }`;
          }
        }
      }

      // Synthetic append: runs in BOTH the happy path and the salvage path.
      // If `originalErr` is set we use a degenerate synthetic
      // (totalTokens=0, since `tokensAtNewLeaf` may not have been computed).
      // The synthetic's matching toolCallId is the only structural
      // requirement for chain validity — pi's appended tool_result pairs
      // with this synthetic regardless of which earlier step threw.
      //
      // The full `summaryFocus` is already live in the LLM call to
      // `generateBranchSummary`; we only need a trimmed copy in the
      // synthetic's args (which pi will re-emit on every subsequent turn).
      // Truncate to MAX_SYNTHETIC_FOCUS_LENGTH so a long focus string
      // doesn't inflate every later turn indefinitely.
      const syntheticArgs: Record<string, unknown> = {
        action: "rewind",
        rewindTo: p.rewindTo,
        newLabel: p.newLabel,
        summaryFocus: p.summaryFocus,
      };
      if (
        typeof p.summaryFocus === "string" &&
        p.summaryFocus.length > MAX_SYNTHETIC_FOCUS_LENGTH
      ) {
        syntheticArgs.summaryFocus = `${p.summaryFocus.slice(0, MAX_SYNTHETIC_FOCUS_LENGTH)}… [truncated]`;
      }
      const syntheticMsg = buildSyntheticAssistant(
        toolCallId,
        TOOL_NAME,
        syntheticArgs,
        ctx.model as
          | { api?: string; provider?: string; id?: string }
          | undefined,
        originalErr ? 0 : tokensAtNewLeaf,
      );
      const syntheticId = sm.appendMessage(syntheticMsg);

      // Refresh agent.state.messages so the next prompt() snapshot reflects
      // the rewound chain. Runs in both paths; `refreshAgentMessages`
      // already swallows internal throws, so it can't re-trigger salvage.
      const refreshed = refreshAgentMessages(sm);

      if (originalErr) {
        // Salvage path: synthetic landed (chain is valid), newLabel retry
        // and refresh were best-effort. Re-throw the original error with
        // any salvage detail attached so the failure surfaces to the
        // agent and post-mortem reviewers can tell what was recovered.
        // Preserve the original via `Error.cause` (ES2022) so callers
        // doing `instanceof` checks against typed subclasses, or
        // post-mortem readers walking the cause chain, can recover the
        // original throw. Older runtimes silently ignore the options
        // bag, so this is forward-compatible without a feature gate.
        const baseMsg =
          originalErr instanceof Error
            ? originalErr.message
            : String(originalErr);
        throw new Error(
          salvageDetail ? `${baseMsg} (salvage: ${salvageDetail})` : baseMsg,
          { cause: originalErr },
        );
      }

      const afterTokens = tokensAtNewLeaf;

      return {
        content: [
          {
            type: "text",
            text:
              `[rewind to '${p.rewindTo}' · collapsed as '${p.newLabel}'] · ${formatContextDelta(beforeTokens, afterTokens, contextWindow)}\n\n` +
              `A branch_summary recording the work just collapsed has been appended to your context. Items under '### Done' are complete. Items under '### In Progress', '### Blocked', or '## Next Steps' are pending — execute them next without re-confirming with the user. Other branch_summary messages, if present, record earlier collapsed segments.` +
              (refreshed ? "" : `\n\n${REFLECTION_BOOTSTRAP_WARNING_REWIND}`),
          },
        ],
        details: {
          rewindTo: p.rewindTo,
          newLabel: p.newLabel,
          targetId: target,
          summaryId,
          syntheticAssistantId: syntheticId,
          collapsedEntries: entries.length,
          contextBefore: beforeTokens,
          contextAfter: afterTokens,
          contextWindow,
          agentMessagesRefreshed: refreshed,
          summaryCache: {
            mode: summaryCacheMode,
            fallbackReason: summaryCacheFallbackReason ?? null,
            branchStartRetained,
            used: cacheWrapped?.used.value ?? false,
            cacheRead: summaryCacheStats.cacheRead,
            input: summaryCacheStats.fresh,
            cacheWrite: summaryCacheStats.cacheWrite,
            hit: summaryCacheStats.hit,
            missedTokens: summaryCacheMiss?.missedTokens ?? 0,
            missedCost: summaryCacheMiss?.missedCost ?? 0,
            idleMs: summaryCacheMiss?.idleMs ?? 0,
            modelChanged: summaryCacheMiss?.modelChanged ?? false,
            notice: summaryCacheNotice,
          },
          readFiles: result.readFiles ?? [],
          modifiedFiles: result.modifiedFiles ?? [],
        },
      };
    },
    // Upstream pi renders cache notices as transcript lines (its
    // `maybeShowCacheMissNotice` + `createResultFallback`), not toasts.
    // Reproduce the default tool-result rendering and append the notice that
    // `execute` stored. `showCacheMissNotices` gating already happened in
    // `execute`; this renderer only runs in the TUI.
    renderResult(result, { expanded }, theme) {
      const output = result.content
        .filter((block) => block.type === "text")
        .map((block) => (block.type === "text" ? block.text : ""))
        .join("\n");
      const lines = output.split("\n");
      const displayLines = expanded ? lines : lines.slice(0, 10);
      const remaining = lines.length - displayLines.length;
      let text = displayLines
        .map((line) => theme.fg("toolOutput", line))
        .join("\n");
      if (remaining > 0) {
        text += `${theme.fg("muted", `\n... (${remaining} more lines,`)} ${keyHint(
          "app.tools.expand",
          "to expand",
        )}${theme.fg("muted", ")")}`;
      }
      const container = new Container();
      container.addChild(new Text(text, 0, 0));
      const notice = (
        result.details as
          | { summaryCache?: { notice?: string | null } }
          | undefined
      )?.summaryCache?.notice;
      if (typeof notice === "string" && notice.length > 0) {
        container.addChild(new Spacer(1));
        container.addChild(new Text(theme.fg("warning", notice), 1, 0));
      }
      return container;
    },
  });
}

/**
 * Non-stable testing-only hooks. **Do NOT import in production code.**
 *
 * The `__` prefix and individual member names are subject to change in any
 * release without a semver-major bump. Intended exclusively for hermetic
 * tests within this package; the hooks reach into module-internal state
 * (the `AgentSession.prototype` patch, the `seenSessions` WeakSet, the
 * `sessionInstances` array, the `prepareNextTurn` marker symbol) and are
 * not designed for external consumption.
 *
 * If you found this via `node_modules` archaeology, you're holding it
 * wrong — use the registered tool surface (`navigate_tree`) instead.
 */
export const __testHooks = {
  /**
   * Restore the original `AgentSession.prototype.prompt` (stashed by
   * `patchAgentSessionPrototype` under the `ORIG_PROMPT_KEY` symbol) and
   * drain the captured-session refs. Idempotent: a no-op if the patch was
   * never installed or has already been reset.
   */
  resetPrototype(): void {
    const proto = AgentSession.prototype as unknown as Record<
      PropertyKey,
      unknown
    >;
    const orig = proto[ORIG_PROMPT_KEY];
    if (typeof orig === "function") {
      proto.prompt = orig;
      delete proto[ORIG_PROMPT_KEY];
    }
    sessionInstances.length = 0;
    // WeakSet has no .clear(); rebind to a fresh instance so a test that
    // re-captures the SAME session identity post-reset isn't deduped by
    // stale state from the previous test's capture.
    seenSessions = new WeakSet();
  },
  /** Module-internal helpers exposed for hermetic unit tests. */
  buildSyntheticAssistant,
  findLabelHint,
  findLabeledEntry,
  findInFlightAssistantToolCalls,
  buildContextMessages,
  refreshAgentMessages,
  captureSession,
  /** Read-only view of captured-session ref count for reaping assertions. */
  sessionRefCount(): number {
    return sessionInstances.length;
  },
  /**
   * The reflection-bootstrap-missing warning strings, split per site.
   * Exported so tests can pin the per-site verbatim wording (the `list`
   * site is read-only and uses the read-only phrasing; the `rewind` site
   * writes to disk and uses the write phrasing). Tests assert literal
   * containment at each site to catch drift.
   */
  REFLECTION_BOOTSTRAP_WARNING_LIST,
  REFLECTION_BOOTSTRAP_WARNING_REWIND,
};
