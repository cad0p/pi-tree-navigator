/**
 * navigate-tree — agent-callable session tree navigation.
 *
 * See README “Implementation notes” for the user-facing narrative
 * (in-band summary turn, same-loop context refresh, reflection
 * bootstrap). This file-level JSDoc carries only source-internal facts
 * the README doesn't.
 *
 * `/tree`-visible artifacts (empirical):
 *   • The abandoned branch of every rewind ends with the rewind tool
 *     call, its "accepted" result, the `navigate-tree-summary-request`
 *     instruction message, any hidden failed attempts, and the model's
 *     summary turn (labeled `branch:<newLabel>`); the active branch
 *     continues from the `branch_summary` with the `navigate-tree-rewind`
 *     continuation message.
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
 *   • `AgentSession._extensionRunner` (#79) is read only to emit
 *     `session_tree` after a rewind; losing it degrades silently
 *     (`historyRewriteNotified: false`, history mirrors stay stale).
 *   • Patches `AgentSession.prototype.prompt` globally on import; not
 *     reversible without a process restart; affects every session in
 *     the pi process, including sessions that never call
 *     `navigate_tree`.
 *
 * Verified against pi 1.3.0 via `scripts/pi-upstream-probe.mjs`.
 * In-loop context refresh runs through the public `context` extension
 * event (see `buildContextMessages`), not `agent.prepareNextTurn*`
 * reflection. The rewind summary is written in-band by the live model
 * (#80): no out-of-loop summarization request exists.
 */

import { join } from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import {
  type AssistantMessage,
  isContextOverflow,
  isRecoverableLength,
} from "@earendil-works/pi-ai";
import {
  AgentSession,
  buildSessionContext,
  buildSessionProjection,
  CONFIG_DIR_NAME,
  collectEntriesForBranchSummary,
  convertToLlm,
  type ExtensionAPI,
  estimateTokens,
  getAgentDir,
  prepareBranchEntries,
  type SessionBoundaryDraft,
  type SessionEntry,
  type SessionManager,
  SettingsManager,
  sessionEntryToContextMessages,
} from "@earendil-works/pi-coding-agent";
import {
  truncateToWidth,
  visibleWidth,
  wrapTextWithAnsi,
} from "@earendil-works/pi-tui";
import { Type } from "typebox";
import {
  loadTreeNavigatorConfig,
  resolveRewindHintReserve,
  START_ANCHOR_DEFAULT_TURNS,
  TREE_NAVIGATOR_CONFIG_FILENAME,
} from "./config.ts";
import {
  estimateContextTokens,
  newestUsageIndex,
  splitContextEstimate,
} from "./context-tokens.ts";
import {
  type AnchorTableRow,
  BRANCH_LABEL_PREFIX,
  extractTextContent,
  formatAnchorTable,
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
  RewindHintTracker,
} from "./rewind-hint.ts";
import {
  buildStoredSummary,
  buildSummaryRequestText,
  computeFileLists,
} from "./summary-prompt.ts";

/**
 * Always-on anchor mandate appended to the system prompt on every agent
 * start (gated on the tool being active). Same lever as pi-napkin's vault
 * mandate: `promptGuidelines` bullets land mid-prompt; this append lands
 * at the very end of the system prompt, after <project_context> and
 * skills, and is re-applied on every prompt (never compacted). Byte-stable
 * (module constant, no per-session interpolation) for provider caching.
 */
export const ANCHOR_MANDATE = `${TOOL_NAME}: gather all context, then anchor \`context-gathered\`; list anchors and rewind after every milestone or rabbit hole / dead end to keep context low.`;

/**
 * Label written by the automatic start anchor (#55). Exported for the test
 * suite; production callers should rely on the registered tool surface.
 */
export const START_ANCHOR_LABEL = `${LABEL_PREFIX}start`;

// ---------------------------------------------------------------------------
// Exported boundary constants below (MAX_SESSION_REFS, MAX_HINT_WALK_DEPTH,
// MIN_SUMMARY_FOCUS_LENGTH, MIN_REWIND_SAVINGS_TOKENS,
// MAX_SUMMARY_ATTEMPTS).
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
// chain, plus the continuation message), so
// apparent savings OVERSTATES true freed context. Empirically the degenerate
// anchor→immediate-rewind case measures ~100 apparent tokens while a healthy
// stage measures ~20k — 4k separates them cleanly (~200×) and roughly equals
// one 300-line source-file read (~5.9k tokens measured). Below the floor a
// rewind burns a summary turn, forks a junk branch_summary, consumes
// a milestone label, and GROWS live context.
export const MIN_REWIND_SAVINGS_TOKENS = 4000;
// Summary attempts a pending rewind gets in total (#80, #90 S5). An attempt
// fails when it calls a tool (blocked while the rewind is pending), carries
// no `## Goal` summary, or is cut off; a failed attempt is hidden from the
// model and the same messages are re-sent (nothing appended). After the last
// one the rewind's tool result reads as a failed tool call.
export const MAX_SUMMARY_ATTEMPTS = 3;
// Hint length cap for the per-row hint shown in `list` output. 50 chars
// fits one terminal column without wrapping in typical 80-column TUIs.
const LIST_HINT_MAX_LENGTH = 50;
// Hint length cap for the hint shown in the `anchor` response. The anchor
// response is a single block of prose (not a column-aligned table) so it
// can afford a longer hint than `list`'s per-row preview.
const ANCHOR_HINT_MAX_LENGTH = 60;
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
    };
  };
  /**
   * Plain field on `AgentSession` (#79). Used only to emit `session_tree`
   * after a rewind, mirroring pi's own `navigateTree`, so extensions that
   * mirror the conversation (pi-claude-bridge's Claude Code session) rebuild
   * from the rewound chain instead of continuing on their stale copy.
   */
  _extensionRunner?: {
    emit?: (event: unknown) => Promise<unknown> | unknown;
  };
  sessionManager: SessionManager;
  /**
   * Plain field on `AgentSession` in pi 1.3.0 (#101, S3): the live settings
   * pi decides compaction from. Optional: older pi versions may lack it.
   */
  settingsManager?: SettingsManager;
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
 * Prefers the canonical `buildSessionProjection().messages` (>= 0.87.0): that
 * is exactly what pi itself treats as the model context, applying
 * `context_edit` omissions/replacements written by pi's own overflow recovery
 * and by extensions such as the fallback provider. Returning the raw
 * `buildContextEntries()` view would reintroduce edited-away messages into the
 * request (and desync the summary payload's `{first}` numbering from what the
 * model actually sees). The entry walk remains only as a defensive fallback
 * for structural test doubles.
 *
 * The result is CONVERSATION-ONLY: the `context` event contract is "transform
 * conversation messages without prompt and tool system messages; Pi restores
 * that state afterward" (pi docs, `context` section), and `runner.emitContext`
 * hands handlers `currentMessages.filter(role !== "system")` and re-attaches
 * the current head via `restoreSystemMessages`. Returning the projection's
 * system entries would make that re-attach prepend the head to a list still
 * carrying the persisted system entry — a duplicated head on runs where
 * `forceSystemPrompt` is unset.
 *
 * We always replace (no leaf-gating): every `appendMessage` advances the
 * session leaf, so a "leaf changed since last turn" heuristic would fire on
 * essentially every call anyway — and always-replace is byte-for-byte the
 * behavior of the old wrapper (per-turn rebuild from the tree).
 *
 * The parameter is typed structurally because `ReadonlySessionManager` is
 * NOT re-exported from the package barrel (only `SessionManager` and
 * `sessionEntryToContextMessages` are public); `ctx.sessionManager` at the
 * event site is `ReadonlySessionManager`.
 */
export function buildContextMessages(
  sm: {
    buildContextEntries(): SessionEntry[];
    buildSessionProjection?(): {
      entries?: Array<{ sourceEntry: SessionEntry; messages: AgentMessage[] }>;
      messages: AgentMessage[];
    };
    getBranch?(): SessionEntry[];
  },
  pendingInstructionEntryId?: string | null,
  usageOverride?: { tokens: number; entryId?: string },
): ReturnType<typeof sessionEntryToContextMessages> {
  let entries:
    | Array<{ sourceEntry: SessionEntry | null; messages: AgentMessage[] }>
    | undefined;
  try {
    if (typeof sm.buildSessionProjection === "function") {
      const projection = sm.buildSessionProjection();
      entries = Array.isArray(projection.entries)
        ? projection.entries
        : [{ sourceEntry: null, messages: projection.messages }];
    }
  } catch {
    // Fall through to the raw entry projection — pi rewrites the context
    // handlers' output at the wire boundary anyway.
  }
  entries ??= sm.buildContextEntries().map((entry) => ({
    sourceEntry: entry,
    messages: sessionEntryToContextMessages(entry),
  }));
  // Frozen view (#90, O2): while a rewind waits for its summary, the
  // request ends at its instruction, so every attempt re-sends the same
  // messages — late input and hidden attempts never reach it. Keyed by the
  // instruction's entry id: tool-call ids can be "" or reused.
  if (pendingInstructionEntryId) {
    const at = entries.findIndex(
      (e) => e.sourceEntry?.id === pendingInstructionEntryId,
    );
    if (at >= 0) entries = entries.slice(0, at + 1);
  }
  // A failed / aborted rewind reads as a failed tool call (#90, O5): its
  // tool result's content is already replaced by a persisted `context_edit`,
  // but `isError` is not editable — set it on a clone, per request (never
  // mutate the session's own message objects). Keyed by the tool-result
  // entry id the outcome recorded.
  const outcomes = collectRewindOutcomes(sm);
  // Stale usage (#101): an assistant's `usage` measures the whole context it
  // was sent with. One produced after a `context_edit`'s target existed but
  // before that edit applied was sent with the edited-away content counted, so
  // pi's output clamp (window - estimate - 4096, floored at 1) would size
  // max_tokens from an overstated count. Zero those (see staleUsageFlags); pi-ai
  // then falls back to an older exact usage or chars/3.5. Assistants before the
  // target, and after the edit, keep their exact usage.
  const stale = staleUsageFlags(entries);
  // #107: pi's clamp reads the newest non-zero assistant usage, and stale ones
  // are zeroed below — so the calibrated post-withhold size must land on the
  // assistant whose usage is the plan's baseline: the planner's entry by id,
  // NOT the newest stale one, which another extension's later `context_edit`
  // can push the target onto (shrinking a usage that was exact).
  const tokens = usageOverride?.tokens;
  const overrideAt =
    tokens === undefined
      ? -1
      : overrideTargetIndex(entries, usageOverride?.entryId);
  let at = 0;
  const messages = entries.flatMap((e, i) => {
    const text = e.sourceEntry ? outcomes.get(e.sourceEntry.id) : undefined;
    const withOutcome =
      text === undefined
        ? e.messages
        : e.messages.map((message) =>
            message.role === "toolResult"
              ? {
                  ...message,
                  content: [{ type: "text" as const, text }],
                  isError: true,
                }
              : message,
          );
    const start = at;
    const projected = stale[i] ? withOutcome.map(dropStaleUsage) : withOutcome;
    at += projected.length;
    if (tokens === undefined || overrideAt < 0) return projected;
    return projected.map((message, j) =>
      start + j === overrideAt ? withContextTokens(message, tokens) : message,
    );
  });
  return messages.filter((message) => message.role !== "system");
}

/**
 * Per entry, whether its assistant usage is stale (#101). An entry at `i` is
 * stale iff some `context_edit` at `e` targets an entry at `t` with
 * `t < i < e`: it was produced after the target existed but before the edit
 * applied. Indices are branch order over the projection, which keeps the
 * `context_edit` entries themselves (with no messages). Targets outside it are
 * ignored.
 */
function staleUsageFlags(
  entries: Array<{ sourceEntry: SessionEntry | null }>,
): boolean[] {
  const position = new Map<string, number>();
  for (const [i, e] of entries.entries()) {
    if (e.sourceEntry && !position.has(e.sourceEntry.id)) {
      position.set(e.sourceEntry.id, i);
    }
  }
  const stale = entries.map(() => false);
  for (const [edit, e] of entries.entries()) {
    const source = e.sourceEntry;
    if (source?.type !== "context_edit" || !source.targetId) continue;
    const target = position.get(source.targetId);
    if (target === undefined) continue;
    for (let i = target + 1; i < edit; i++) stale[i] = true;
  }
  return stale;
}

/**
 * `message` with a zeroed usage when it is an assistant with one (#101). Zeroed,
 * not deleted: pi-ai reads `usage.totalTokens` unguarded and would throw. Zero
 * totals are skipped, and every field is zeroed because `calculateContextTokens`
 * falls back to the four parts when `totalTokens` is 0.
 */
function dropStaleUsage(message: AgentMessage): AgentMessage {
  if (message.role !== "assistant" || !message.usage) return message;
  return {
    ...message,
    usage: {
      ...message.usage,
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
    },
  };
}

/**
 * The message index the calibrated size goes on (#107). The planner's entry
 * id is the target: its newest assistant usage is the plan's baseline, so the
 * override replaces exactly the number it was computed from. "Newest stale"
 * would drift onto a later assistant that an unrelated `context_edit` made
 * stale — shrinking a usage that was measured with the withheld content
 * already gone.
 *
 * The entry is absent from the D11 frozen view (the request ends at the
 * pending instruction, so the failed attempt that supplied the baseline was
 * sliced off) — then the newest assistant usage in the view is the one pi's
 * clamp will actually read, and it carries the calibrated size instead.
 */
function overrideTargetIndex(
  entries: Array<{
    sourceEntry: SessionEntry | null;
    messages: AgentMessage[];
  }>,
  entryId?: string,
): number {
  let index = 0;
  for (const e of entries) {
    if (entryId !== undefined && e.sourceEntry?.id === entryId) {
      const found = newestAssistantUsageIndex(e.messages, index);
      if (found >= 0) return found;
    }
    index += e.messages.length;
  }
  return newestAssistantUsageIndex(entries.flatMap((e) => e.messages));
}

/**
 * Index of the newest assistant message carrying a usage in `messages`,
 * offset by `start`, or -1 with none (#107). Zeroed stale usages still count:
 * the override is applied after them, and pi reads the newest one.
 */
function newestAssistantUsageIndex(
  messages: readonly AgentMessage[],
  start = 0,
): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (message.role === "assistant" && message.usage) return start + i;
  }
  return -1;
}

/**
 * `message` with the calibrated post-withhold size in its usage (#107), the
 * number pi's clamp sizes the reply from. A clone: the session's own message
 * objects are never mutated.
 */
function withContextTokens(
  message: AgentMessage,
  tokens: number,
): AgentMessage {
  if (message.role !== "assistant" || !message.usage) return message;
  return {
    ...message,
    usage: {
      ...message.usage,
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: Math.max(0, Math.round(tokens)),
    },
  };
}

/** True for the summary instruction entry of the rewind `toolCallId` (#90). */
function isSummaryRequestEntry(
  entry: SessionEntry,
  toolCallId: string,
): boolean {
  return (
    entry.type === "custom_message" &&
    entry.customType === SUMMARY_REQUEST_CUSTOM_TYPE &&
    (entry.details as { toolCallId?: unknown } | undefined)?.toolCallId ===
      toolCallId
  );
}

/** Outcome entry `data` on `entry`, or null (#90). */
function readOutcomeData(entry: SessionEntry): RewindOutcomeData | null {
  if (
    entry.type !== "custom" ||
    entry.customType !== REWIND_OUTCOME_CUSTOM_TYPE
  ) {
    return null;
  }
  const data = entry.data as Partial<RewindOutcomeData> | undefined;
  if (
    !data ||
    typeof data.toolCallId !== "string" ||
    typeof data.text !== "string"
  ) {
    return null;
  }
  return data as RewindOutcomeData;
}

/**
 * Failed / aborted rewinds on the active branch: tool-result entry id →
 * result text.
 */
function collectRewindOutcomes(sm: {
  getBranch?(): SessionEntry[];
}): Map<string, string> {
  const out = new Map<string, string>();
  if (typeof sm.getBranch !== "function") return out;
  let branch: SessionEntry[];
  try {
    branch = sm.getBranch();
  } catch {
    return out;
  }
  for (const entry of branch) {
    const data = readOutcomeData(entry);
    if (data && typeof data.toolResultEntryId === "string") {
      out.set(data.toolResultEntryId, data.text);
    }
  }
  return out;
}

/**
 * True when the provider's serializer will actually send this message.
 * pi-ai's adapters skip assistant messages that carry no text, thinking, or
 * tool-call blocks (openai-completions: "Skip assistant messages that have no
 * content and no tool calls"; anthropic-messages drops zero-block messages),
 * so such an entry must not consume a `{first}` number. Non-assistant
 * messages and unknown block types pass (conservative: never under-count a
 * message the model might see).
 */
function isProviderVisibleMessage(message: AgentMessage): boolean {
  if (message.role !== "assistant") return true;
  const content: unknown = message.content;
  if (typeof content === "string") return content.trim().length > 0;
  if (!Array.isArray(content)) return true;
  for (const block of content as unknown[]) {
    if (typeof block !== "object" || block === null) continue;
    const record = block as { [key: string]: unknown };
    const type = record.type;
    if (type === "text") {
      const text = record.text;
      if (typeof text === "string" && text.trim().length > 0) return true;
    } else if (type === "thinking") {
      const thinking = record.thinking;
      const signature = record.thinkingSignature;
      const hasThinking =
        typeof thinking === "string" && thinking.trim().length > 0;
      const hasSignature =
        typeof signature === "string" && signature.trim().length > 0;
      if (hasThinking || hasSignature || record.redacted === true) return true;
    } else if (type !== undefined) {
      // Includes `toolCall` and any future block type: assume it serializes.
      return true;
    }
  }
  return false;
}

/**
 * Count the non-system wire messages contributed by the projection entries
 * at or before `target` — the background prefix the summary turn sees ahead
 * of the collapsed segment. Substituted into the summary
 * instruction's `{first}` placeholder.
 *
 * System messages are excluded because the instruction numbering explicitly
 * excludes the system prompt (and any tool-delta system messages replayed
 * into the head). Provider-invisible empty assistants are excluded too (see
 * `isProviderVisibleMessage`) so an aborted, non-`context_edit`ed attempt
 * cannot inflate `{first}`. Quantity is taken from the PROJECted entries
 * (`buildSessionProjection()`), so a compaction cut before `target` is
 * counted the way the live projection presents it AND messages omitted by
 * `context_edit` (pi's overflow recovery, the fallback provider's
 * failed-attempt edits) no longer inflate `{first}` past what the model sees.
 */
function countBackgroundMessages(
  projectedEntries: Array<{
    sourceEntry: { id: string };
    messages: AgentMessage[];
  }>,
  target: string,
): number {
  let count = 0;
  for (const entry of projectedEntries) {
    for (const message of convertToLlm(entry.messages)) {
      if (message.role !== "system" && isProviderVisibleMessage(message)) {
        count++;
      }
    }
    if (entry.sourceEntry.id === target) break;
  }
  return count;
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

async function emitSessionTree(
  sm: SessionManager,
  oldLeafId: string | null,
  summaryId: string,
): Promise<boolean> {
  // Replicate the `session_tree` emit that pi's own `navigateTree` does after
  // `branchWithSummary` (#79). A tool-driven rewind skips it, so extensions
  // that mirror the conversation never learn the history was rewritten:
  // pi-claude-bridge keeps resuming its Claude Code session with the full
  // pre-rewind context. Returns false when the runner is unreachable or a
  // handler throws; the rewind itself is already complete either way.
  const session = findOwningSession(sm);
  const runner = session ? asInternals(session)._extensionRunner : undefined;
  if (typeof runner?.emit !== "function") return false;
  try {
    await runner.emit({
      type: "session_tree",
      newLeafId: sm.getLeafId(),
      oldLeafId,
      summaryEntry: sm.getEntry(summaryId),
      fromExtension: true,
    });
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

function estimateActiveBranchTokens(sm: SessionManager): number {
  return estimateContextTokens(sm.buildSessionContext().messages);
}

function estimateAtEntry(
  entries: SessionEntry[],
  entryId: string,
  byId: Map<string, SessionEntry>,
): number {
  return estimateContextTokens(
    buildSessionContext(entries, entryId, byId).messages,
  );
}

/**
 * Every active anchor — chronological root→leaf, same walk order as `list`
 * — with its cumulative context percentage, rendered as the shared anchor
 * table (#101, D19). Shared by the min-savings rejection, the rewind hint
 * and the forced-rewind message so all name the same targets the same way. A manual
 * `/tree` label `anchor:x` CAN put the same name on two entries —
 * duplicates print twice, exactly as `list` prints them. Names `rewindTo`
 * would reject (e.g. an empty `anchor:` label) are skipped: both callers
 * point the model at rewind targets.
 */
function formatActiveAnchors(
  sm: Pick<SessionManager, "getBranch" | "getLabel">,
  cw: number,
  allEntries: SessionEntry[],
  byId: Map<string, SessionEntry>,
): { names: string[]; text: string } {
  const names: string[] = [];
  const rows: AnchorTableRow[] = [];
  for (const e of sm.getBranch()) {
    const lbl = sm.getLabel(e.id);
    if (!lbl?.startsWith(LABEL_PREFIX)) continue;
    const name = lbl.slice(LABEL_PREFIX.length);
    if (!isValidName(name)) continue;
    names.push(name);
    rows.push({
      pct: formatPct1(estimateAtEntry(allEntries, e.id, byId), cw),
      name,
    });
  }
  // cw=0 → percents fall back to formatPct1's k-format and the header
  // drops the window, mirroring `list`.
  return { names, text: formatAnchorTable(rows, cw) };
}

/**
 * Rejection copy for the rewind min-savings floor (#21). Reports the
 * apparent savings plus every active anchor (including rewindTo itself)
 * with its cumulative context percentage, so the agent can pick a
 * genuinely earlier anchor or keep working.
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
  const anchors = formatActiveAnchors(sm, cw, allEntries, byId);
  const guidance =
    anchors.names[0] === rewindTo
      ? "No earlier anchors — keep working and rewind later once more has accumulated above it."
      : "Rewind further back to actually free context, or keep working.";
  return `Rewinding to '${rewindTo}' (as '${newLabel}') would free only ~${formatPct1(apparentSavings, 0)} tokens. ${guidance}\n${anchors.text}`;
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

// =============================================================================
// Extension
// =============================================================================

/**
 * Earlier handlers' boundary drafts that still apply after a rewind moved the
 * branch. A `context_edit` draft can target an entry the rewind just
 * abandoned; pi validates the whole list in a preview (`appendContextEdit`
 * throws for a target off the active branch) and drops ALL of it, ours
 * included, when one draft fails — so such drafts are dropped here instead.
 * Every other draft passes through: pi never validates a compaction's
 * `firstKeptEntryId` (`null` is a valid self-retaining compaction).
 */
function keepValidDrafts<T>(sm: SessionManager, drafts: readonly T[]): T[] {
  let onBranch: Set<string>;
  try {
    onBranch = new Set(sm.getBranch().map((e) => e.id));
  } catch {
    return [...drafts];
  }
  return drafts.filter((draft) => {
    const d = draft as { type?: string; targetId?: string };
    if (d?.type === "context_edit") return onBranch.has(d.targetId ?? "");
    return true;
  });
}

/** A `navigate-tree-rewind` message, delivered as a `turn_end` boundary entry. */
interface RewindMessageDraft {
  type: "custom_message";
  customType: string;
  content: string;
  display: boolean;
  details: Record<string, unknown>;
}

function rewindMessageDraft(
  content: string,
  details: Record<string, unknown>,
): RewindMessageDraft {
  return {
    type: "custom_message",
    customType: REWIND_CUSTOM_TYPE,
    content,
    display: true,
    details,
  };
}

/** The model fields rewind-driven compaction reads (#101). */
interface OverflowModel {
  provider: string;
  id: string;
  contextWindow?: number;
  maxTokens?: number;
}

/**
 * True when `message` is a context overflow by pi's own checks (#101, D1):
 * an explicit overflow error, a silent overflow, or a reply squeezed by the
 * context (`isRecoverableLength`). Same model only (pi's `sameModel`): a
 * reply from another model says nothing about this window.
 */
function overflowed(
  message: unknown,
  model: OverflowModel | undefined,
): boolean {
  const m = message as Partial<AssistantMessage> | undefined;
  if (m?.role !== "assistant" || !model) return false;
  if (m.provider !== model.provider || m.model !== model.id) return false;
  try {
    return (
      isContextOverflow(m as AssistantMessage, model.contextWindow) ||
      isRecoverableLength(m as AssistantMessage, model.maxTokens ?? 0)
    );
  } catch {
    return false;
  }
}

/** What withholding tool results plans (#101, D3/D4). */
interface WithholdPlan {
  drafts: SessionBoundaryDraft[];
  withheld: number;
  fits: boolean;
  /** The calibrated post-withhold estimate (#107): pi's clamp basis. */
  projectedTokens: number;
  /**
   * The entry whose assistant usage is that estimate's baseline (#107): the
   * override must land on it by id, not on whatever is newest later.
   */
  overrideEntryId?: string;
}

/**
 * The provider's real size of the overflowed request (#101, A): input +
 * cache read + cache write of a `length` or silent `stop` reply that carries
 * usage. An explicit error, or no usage, gives `undefined`: the estimate is
 * then used as is (ratio 1).
 */
function overflowRequestTokens(
  sm: SessionManager,
  entryId: string,
): number | undefined {
  const entry = sm.getEntry(entryId);
  if (entry?.type !== "message" || entry.message.role !== "assistant") {
    return undefined;
  }
  const reply = entry.message as {
    stopReason?: string;
    usage?: { input?: number; cacheRead?: number; cacheWrite?: number };
  };
  if (reply.stopReason !== "length" && reply.stopReason !== "stop") {
    return undefined;
  }
  const usage = reply.usage;
  const total =
    (usage?.input ?? 0) + (usage?.cacheRead ?? 0) + (usage?.cacheWrite ?? 0);
  return total > 0 ? total : undefined;
}

/**
 * Withhold the tool results that overflowed (#101, D3): walk back from
 * `fromEntryId` (excluded) through every older turn, and replace their
 * content newest-first with the D4 notice until the estimate fits `target`.
 * The walk stops at a compaction or branch summary (a context boundary: what
 * is above it is not projected). User and custom messages are never
 * withheld; a navigate_tree result, a result already hidden or replaced, and
 * a result not in the projection are skipped. The estimate is the projection without
 * `hiddenIds` (sliced at `instructionEntryId`, the frozen request of a
 * pending rewind), minus what each notice saves.
 *
 * #101, A: `realTokens` (the overflowed request's provider count) scales the
 * estimate's tail after the newest usage baseline by `ratio` =
 * (real − baseline) / tail, clamped to at least 1; the planner then fits
 * `baseline + ratio × tail` under `target`. Without `realTokens` the
 * ratio is 1.
 */
function planWithhold(
  sm: SessionManager,
  fromEntryId: string,
  hiddenIds: readonly string[],
  target: number,
  instructionEntryId?: string,
  realTokens?: number,
): WithholdPlan {
  const branch = sm.getBranch();
  const lastEdit = new Map<string, unknown>();
  for (const entry of branch) {
    if (entry.type === "context_edit") {
      lastEdit.set(entry.targetId, entry.replacement);
    }
  }
  const hidden = new Set(hiddenIds);
  let entries = sm.buildSessionProjection().entries;
  if (instructionEntryId) {
    const at = entries.findIndex(
      (e) => e.sourceEntry.id === instructionEntryId,
    );
    if (at >= 0) entries = entries.slice(0, at + 1);
  }
  // The messages the estimate counts, and each entry's first position in them
  // (#101, walk fix): an entry outside this list is not in the context.
  const messages: AgentMessage[] = [];
  // The entry that owns each message, so the baseline's owner can be recovered
  // once the anchor is known (#107).
  const owners: string[] = [];
  const positions = new Map<string, number>();
  for (const e of entries) {
    if (hidden.has(e.sourceEntry.id)) continue;
    for (const m of e.messages) {
      if (m.role === "system") continue;
      if (!positions.has(e.sourceEntry.id)) {
        positions.set(e.sourceEntry.id, messages.length);
      }
      messages.push(m);
      owners.push(e.sourceEntry.id);
    }
  }
  const anchor = newestUsageIndex(messages);
  const split = splitContextEstimate(messages);
  let baseline = split.baseline;
  let tail = split.tail;
  const ratio =
    realTokens === undefined || tail <= 0
      ? 1
      : Math.max(1, (realTokens - baseline) / tail);
  const estimate = () => baseline + ratio * tail;
  const drafts: SessionBoundaryDraft[] = [];
  const start = branch.findIndex((e) => e.id === fromEntryId);
  // Newest first, across turns: the failed attempt's own batch is not always
  // enough (#101, walk fix). A compaction or branch summary is a context
  // boundary: nothing above it is in the projected context.
  for (let i = start - 1; i >= 0 && estimate() > target; i--) {
    const entry = branch[i];
    if (entry.type === "compaction" || entry.type === "branch_summary") {
      break;
    }
    if (entry.type !== "message" || entry.message.role !== "toolResult") {
      continue;
    }
    const at = positions.get(entry.id);
    if (
      at === undefined ||
      lastEdit.has(entry.id) ||
      entry.message.toolName === TOOL_NAME
    ) {
      continue;
    }
    const original = estimateTokens(entry.message);
    const text = buildWithheldNoticeText(original);
    const content = [{ type: "text" as const, text }];
    const saving = original - estimateTokens({ ...entry.message, content });
    // A result before the newest usage is already in its real count, so its
    // saving comes off the baseline, as the unscaled estimate (it under-credits
    // a dense result, which only withholds more). It must not go on the tail:
    // the ratio would scale a negative tail into a false fit. A result after
    // the usage is in the tail, which the ratio scales as before.
    if (at < anchor) {
      baseline -= saving;
    } else {
      tail -= saving;
    }
    drafts.push({
      type: "context_edit",
      targetId: entry.id,
      replacement: { content },
    });
  }
  return {
    drafts,
    withheld: drafts.length,
    fits: estimate() <= target,
    projectedTokens: Math.round(estimate()),
    // The assistant whose real usage is the baseline the estimate was built
    // from: the one the override must replace (#107).
    overrideEntryId: anchor >= 0 ? owners[anchor] : undefined,
  };
}

/** `context_edit` drafts hiding `ids` (a failed attempt, #90 O3). */
function hideDrafts(ids: readonly string[]): SessionBoundaryDraft[] {
  return ids.map((targetId) => ({
    type: "context_edit",
    targetId,
    replacement: null,
  }));
}

/** Why a summary attempt failed (#90, O3/O5); the last one names the ending. */
export type SummaryFailureCause =
  | "cut off"
  | "called a tool"
  | "no '## Goal' summary";

/** A rewind waiting for its in-band summary turn (#80, #90). */
interface PendingRewind {
  /** The rewind tool call; its own turn is not the summary turn. */
  toolCallId: string;
  /** The assistant entry that issued the rewind (leaf at phase 1). */
  rewindEntryId: string;
  target: string;
  rewindTo: string;
  newLabel: string;
  beforeTokens: number;
  contextWindow: number;
  /** `{first}` of the summary instruction. */
  first: number;
  summaryFocus: string;
  /** Estimated context at the anchor (before the summary is added). */
  tokensAtTarget: number;
  /** The rewind's tool-result entry, set when its turn ends. */
  toolResultEntryId?: string;
  /** The rewind turn ended and the instruction draft was returned. */
  instructionSent?: boolean;
  /** The instruction's entry, once it is seen committed on the branch. */
  instructionEntryId?: string;
  /** Entries of failed attempts hidden so far (assistant + its results). */
  hiddenEntryIds: string[];
  /** Failed summary turns so far (tool call / no `## Goal` / cut off). */
  attempts: number;
  /** The cause of the latest failed attempt. */
  lastCause?: SummaryFailureCause;
  /**
   * An ending returned as boundary drafts, not yet seen committed (#90): pi
   * drops a boundary's whole list when any handler's draft is invalid, so
   * the next event checks for the outcome entry and re-applies the ending
   * outside a boundary when it is missing.
   */
  committing?: {
    status: RewindOutcomeData["status"];
    reason: string;
    text: string;
    extraHidden: readonly string[];
    /** JSON of the outcome `data` the drafts carry. */
    outcomeKey: string;
  };
}

/** customType of the rewind continuation message (#80). */
export const REWIND_CUSTOM_TYPE = "navigate-tree-rewind";

/** customType of the summary instruction message (#90). */
export const SUMMARY_REQUEST_CUSTOM_TYPE = "navigate-tree-summary-request";

/**
 * customType of the `custom` entry recording a failed / aborted rewind
 * (#90). The persistent marker: the `context` handler turns the rewind's
 * tool result into an error from it, the restart check reads it, and the
 * TUI renders it.
 */
export const REWIND_OUTCOME_CUSTOM_TYPE = "navigate-tree-rewind-outcome";

/**
 * Reason of a tool call blocked while a rewind waits for its summary. Only
 * a summary attempt can call a tool then (a batched rewind is refused, #37),
 * and every failed attempt is hidden from the model, so this text is
 * TUI-only.
 */
export const SUMMARY_PENDING_BLOCK_REASON = `${TOOL_NAME}: blocked: the model tried to call a tool instead of summarizing.`;

/** Refusal of a rewind that shares its assistant message with other calls (#37). */
export const SOLO_REWIND_REFUSAL =
  "rewind must be the only tool call in its message — re-issue only the rewind, alone. The other calls in this message ran normally.";

/** pi-codemode's tool name: its scripts issue nested calls (#103). */
const CODEMODE_TOOL_NAME = "codemode";
/** Refusal of a codemode script's rewind that is not its first nested call (#103). */
export const NESTED_REWIND_REFUSAL = `rewind from a codemode script must be the script's first nested call — call ${TOOL_NAME} directly, or start the script with it.`;

/** The rewind's tool result (#90): it only reports that the rewind was accepted. */
export function buildRewindAcceptedText(
  rewindTo: string,
  newLabel: string,
): string {
  return `[${TOOL_NAME}] Rewind to '${rewindTo}' accepted (collapse as '${newLabel}'). Follow the ${TOOL_NAME} message below to complete it.`;
}

/** `[rewind to '<rewindTo>' · collapse as '<newLabel>']`, the TUI row prefix. */
function rewindRowPrefix(rewindTo: unknown, newLabel: unknown): string {
  return `[rewind to '${String(rewindTo)}' · collapse as '${String(newLabel)}']`;
}

/**
 * The summary request's one-line TUI row (#90, O1). `~` marks the estimate at
 * the anchor (before the summary is added); the continuation reports the
 * real figure.
 */
export function buildSummaryRequestRow(details: {
  rewindTo: string;
  newLabel: string;
  contextBefore: number;
  tokensAtTarget: number;
  contextWindow: number;
}): string {
  const cw = details.contextWindow;
  const delta =
    cw > 0
      ? `context ${formatPct1(details.contextBefore, cw)} → ~${formatPct1(details.tokensAtTarget, cw)} of ${formatWindow(cw)}`
      : `tokens ${details.contextBefore} → ~${details.tokensAtTarget}`;
  return `${rewindRowPrefix(details.rewindTo, details.newLabel)} · summarizing · ${delta}`;
}

/** Reason of a failed rewind whose attempts ran out (#90, O5). */
export function buildAttemptsExhaustedReason(
  cause: SummaryFailureCause,
): string {
  return `no summary after ${MAX_SUMMARY_ATTEMPTS} attempts (last: ${cause})`;
}

/**
 * The rewind's tool result after its attempts ran out (#90, O5), keyed on
 * the last attempt's cause: a tighter focus only helps a cut-off summary.
 */
export function buildSummaryFailedText(cause: SummaryFailureCause): string {
  const advice =
    cause === "cut off"
      ? "Try again with a tighter summaryFocus."
      : "Try again; the reply after the rewind must be only the summary.";
  return `Summarization failed: ${buildAttemptsExhaustedReason(cause)}. Nothing was collapsed. ${advice}`;
}

/** The rewind's tool result after any other failure (#90, O6). */
export function buildRewindFailedText(reason: string): string {
  return `Summarization failed: ${reason}. Nothing was collapsed. Try again.`;
}

/** The rewind's tool result after an abort or a new prompt (#90, O6). */
export const REWIND_ABORTED_TEXT =
  "Summarization aborted. Nothing was collapsed.";

/** The rewind's tool-result text for an ending (#90, O5 / O6). */
function endingText(
  status: "failed" | "aborted",
  reason: string,
  rewind: { lastCause?: SummaryFailureCause },
): string {
  if (status === "aborted") return REWIND_ABORTED_TEXT;
  if (
    rewind.lastCause &&
    reason === buildAttemptsExhaustedReason(rewind.lastCause)
  ) {
    return buildSummaryFailedText(rewind.lastCause);
  }
  return buildRewindFailedText(reason);
}

/**
 * The summary instruction (#90, S1/S3): a `navigate_tree` custom message,
 * returned as a boundary entry by the rewind turn's `turn_end`.
 */
function summaryRequestDraft(pending: PendingRewind): SessionBoundaryDraft {
  return {
    type: "custom_message",
    customType: SUMMARY_REQUEST_CUSTOM_TYPE,
    content: buildSummaryRequestText(pending.first, pending.summaryFocus),
    display: true,
    details: {
      toolCallId: pending.toolCallId,
      rewindTo: pending.rewindTo,
      newLabel: pending.newLabel,
      contextBefore: pending.beforeTokens,
      tokensAtTarget: pending.tokensAtTarget,
      contextWindow: pending.contextWindow,
      first: pending.first,
    },
  };
}

/** Persisted `data` of a `navigate-tree-rewind-outcome` entry (#90, O8). */
export interface RewindOutcomeData {
  toolCallId: string;
  status: "failed" | "aborted";
  /** Machine-readable cause; the TUI row is built from it, never from `text`. */
  reason: string;
  /** What the rewind's tool result now says (model-visible). */
  text: string;
  rewindTo: string;
  newLabel: string;
  /** Entries hidden from model context by this rewind (audit). */
  hiddenIds: string[];
  toolResultEntryId: string | null;
  /**
   * The instruction this outcome ends (the O8 check matches on it; absent
   * on entries written before it existed).
   */
  instructionEntryId?: string | null;
}

/** The outcome entry's TUI row (#90, O9). */
export function buildRewindOutcomeRow(
  data: Pick<RewindOutcomeData, "status" | "reason" | "rewindTo" | "newLabel">,
): string {
  const middle =
    data.status === "aborted" ? "aborted" : `failed: ${data.reason}`;
  return `${rewindRowPrefix(data.rewindTo, data.newLabel)} · ${middle} · nothing collapsed`;
}

/**
 * Minimal TUI component (pi-tui's `Component` shape) rendering plain lines,
 * word-wrapped to the width in terminal cells and styled per line. pi-tui
 * throws on any rendered line wider than the terminal, so every line is
 * measured and wrapped with pi-tui's own helpers (pi's extension loader
 * aliases `@earendil-works/pi-tui` to its bundled copy at runtime).
 */
class StyledLines {
  private readonly lines: string[];
  private readonly style: (line: string) => string;
  private readonly padX: number;

  constructor(lines: string[], style: (line: string) => string, padX = 1) {
    this.lines = lines;
    this.style = style;
    this.padX = padX;
  }

  render(width: number): string[] {
    const inner = Math.max(1, width - 2 * this.padX);
    const pad = " ".repeat(Math.max(0, Math.min(this.padX, width - 1)));
    const out: string[] = [];
    for (const line of this.lines) {
      for (const piece of wrapPlainLine(line, inner)) {
        const row = pad + this.style(piece);
        // Final guard: pi-tui throws on a line wider than the terminal.
        out.push(
          visibleWidth(row) <= width
            ? row
            : truncateToWidth(row, Math.max(0, width), ""),
        );
      }
    }
    return out;
  }

  invalidate(): void {}
}

/**
 * Word-wrap one plain-text line to `width` terminal cells (hard-break words
 * longer than the width). Cells, not code points: `summaryFocus` and error
 * reasons can carry wide characters (CJK, emoji) and tabs. Tabs are
 * expanded first (pi-tui measures one as 3 cells); every piece is then
 * truncated to the width as a final guard (a 2-cell character cannot fit a
 * 1-cell width).
 */
export function wrapPlainLine(line: string, width: number): string[] {
  const w = Math.max(1, Math.floor(width));
  const text = line.replace(/\t/g, "   ");
  if (visibleWidth(text) <= w) return [text];
  return wrapTextWithAnsi(text, w).map((piece) =>
    visibleWidth(piece) <= w ? piece : truncateToWidth(piece, w, ""),
  );
}

/**
 * The summary turn's text, from its `## Goal` heading on (r5d's first
 * required heading), or `null` when the heading is missing — an
 * acknowledgement or commentary turn is a failed attempt, never a stored
 * summary. Any preamble before the heading is dropped.
 */
export function extractSummaryText(text: string): string | null {
  const start = text.search(/^## Goal\b/m);
  if (start < 0) return null;
  const summary = text.slice(start).trim();
  // A bare heading is not a summary: storing it would lose the collapsed work.
  const body = summary.replace(/^## Goal\b[^\n]*/, "").trim();
  return body.length > 0 ? summary : null;
}

/**
 * Why a pending rewind no longer matches the active branch, or `null`. The
 * assistant that issued the rewind (`afterEntryId`) must still be on the
 * branch, and no compaction may have landed after it: pi's threshold
 * compaction can run before the summary turn, so the instruction's
 * `{first}` and the evidence the model summarized would no longer match the
 * raw segment.
 */
function rewindStaleReason(
  sm: SessionManager,
  afterEntryId: string,
): string | null {
  let branch: SessionEntry[];
  try {
    branch = sm.getBranch();
  } catch {
    return null;
  }
  const start = branch.findIndex((e) => e.id === afterEntryId);
  if (start < 0) return "the rewind call is no longer on the active branch";
  return branch.slice(start + 1).some((e) => e.type === "compaction")
    ? "the context was compacted"
    : null;
}

/**
 * Find the assistant message that declared the in-flight tool call on the
 * active branch and return its tool-call batch (emission order), or `null`
 * when it is not found or the branch cannot be read (fail-open). pi persists
 * the assistant at `message_end`, before tools run, so it is on the branch
 * when `execute` runs; the walk matches by tool-call id, not tail position,
 * because earlier siblings' results may already trail it. Every `toolCall`
 * block with a string `id` counts, whatever its `name` (#37, restored #90).
 */
function findInFlightAssistantToolCalls(
  sm: Pick<SessionManager, "getBranch">,
  toolCallId: string,
): Array<{ id: string; name: string }> | null {
  let branch: SessionEntry[];
  try {
    branch = sm.getBranch();
  } catch {
    return null;
  }
  for (let i = branch.length - 1; i >= 0; i--) {
    const entry = branch[i];
    if (entry?.type !== "message") continue;
    const message = entry.message as { role?: unknown; content?: unknown };
    if (message.role !== "assistant" || !Array.isArray(message.content)) {
      continue;
    }
    const calls = (message.content as unknown[]).filter(
      (block): block is { id: string; name?: unknown } =>
        !!block &&
        typeof block === "object" &&
        (block as { type?: unknown }).type === "toolCall" &&
        typeof (block as { id?: unknown }).id === "string",
    );
    if (calls.some((call) => call.id === toolCallId)) {
      return calls.map((call) => ({
        id: call.id,
        name: typeof call.name === "string" ? call.name : "unknown",
      }));
    }
  }
  return null;
}

/**
 * The rewind's tool-result entry on `branch` (#90), by position: the first
 * result for `toolCallId` after the assistant that issued the rewind
 * (`afterEntryId`), or the nearest one before its instruction
 * (`beforeEntryId`). Never by `toolCallId` alone: OpenAI-compatible servers
 * can send "" or reuse ids, so an older rewind's result may share it.
 */
function findToolResultEntryId(
  branch: readonly SessionEntry[],
  toolCallId: string,
  at: { afterEntryId?: string; beforeEntryId?: string },
): string | undefined {
  const isResult = (entry: SessionEntry | undefined) =>
    entry?.type === "message" &&
    entry.message.role === "toolResult" &&
    entry.message.toolCallId === toolCallId;
  if (at.afterEntryId !== undefined) {
    const start = branch.findIndex((e) => e.id === at.afterEntryId);
    if (start < 0) return undefined;
    for (let i = start + 1; i < branch.length; i++) {
      if (isResult(branch[i])) return branch[i].id;
    }
    return undefined;
  }
  if (at.beforeEntryId !== undefined) {
    const end = branch.findIndex((e) => e.id === at.beforeEntryId);
    for (let i = end - 1; i >= 0; i--) {
      if (isResult(branch[i])) return branch[i].id;
    }
  }
  return undefined;
}

/**
 * The pending rewind's committed instruction entry, or undefined (#90).
 * Recorded on first sight: the boundary commits it after our handler
 * returns, so its id is only known from the branch afterwards.
 */
function pendingInstructionEntryId(
  branch: readonly SessionEntry[],
  pending: PendingRewind,
): string | undefined {
  if (
    pending.instructionEntryId &&
    branch.some((e) => e.id === pending.instructionEntryId)
  ) {
    return pending.instructionEntryId;
  }
  const id = findInstructionEntryId(
    branch,
    pending.toolCallId,
    pending.rewindEntryId,
  );
  if (id) pending.instructionEntryId = id;
  return id;
}

/**
 * The instruction entry of the rewind issued by `afterEntryId` (#90): the
 * first summary request for `toolCallId` after it on `branch`.
 */
function findInstructionEntryId(
  branch: readonly SessionEntry[],
  toolCallId: string,
  afterEntryId: string,
): string | undefined {
  const start = branch.findIndex((e) => e.id === afterEntryId);
  if (start < 0) return undefined;
  for (let i = start + 1; i < branch.length; i++) {
    if (isSummaryRequestEntry(branch[i], toolCallId)) return branch[i].id;
  }
  return undefined;
}

/** True when a compaction landed on `branch` after `afterEntryId`. */
function compactedAfter(
  branch: readonly SessionEntry[],
  afterEntryId: string,
): boolean {
  const start = branch.findIndex((e) => e.id === afterEntryId);
  return (
    start >= 0 && branch.slice(start + 1).some((e) => e.type === "compaction")
  );
}

/** What the ending of a rewind writes (#90, §5). */
interface EndingPlan {
  edits: Array<{
    targetId: string;
    replacement: { content: Array<{ type: "text"; text: string }> } | null;
  }>;
  data: RewindOutcomeData;
}

/** The fields of a rewind an ending needs (a pending one, or an orphan). */
interface EndableRewind {
  toolCallId: string;
  rewindTo: string;
  newLabel: string;
  hiddenEntryIds: readonly string[];
  toolResultEntryId?: string;
  /** The assistant that issued the rewind (a pending rewind). */
  rewindEntryId?: string;
  /** The instruction entry (an orphan, or a resolved pending rewind). */
  instructionEntryId?: string;
}

/**
 * Plan the ending of a rewind (#90, §5): hide the failed attempts so far,
 * `extraHidden` (the turn that ended it) and the instruction; replace the
 * rewind's tool result with `text`; record the outcome. Exact ids only,
 * each filtered to the active branch (pi rejects a `context_edit` target off
 * it, dropping a boundary's whole list); a target already hidden is skipped.
 */
function planEnding(
  sm: Pick<SessionManager, "getBranch">,
  rewind: EndableRewind,
  status: RewindOutcomeData["status"],
  reason: string,
  text: string,
  extraHidden: readonly string[] = [],
): EndingPlan {
  let branch: SessionEntry[] = [];
  try {
    branch = sm.getBranch();
  } catch {
    branch = [];
  }
  const onBranch = new Set(branch.map((e) => e.id));
  const lastEdit = new Map<string, unknown>();
  for (const entry of branch) {
    if (entry.type === "context_edit") {
      lastEdit.set(entry.targetId, entry.replacement);
    }
  }
  // Exact entries only (#90): tool-call ids can be "" or reused.
  const instructionId =
    rewind.instructionEntryId ??
    (rewind.rewindEntryId !== undefined
      ? findInstructionEntryId(branch, rewind.toolCallId, rewind.rewindEntryId)
      : undefined);
  const hiddenIds: string[] = [];
  for (const id of [
    ...rewind.hiddenEntryIds,
    ...extraHidden,
    ...(instructionId ? [instructionId] : []),
  ]) {
    if (onBranch.has(id) && !hiddenIds.includes(id)) hiddenIds.push(id);
  }
  const edits: EndingPlan["edits"] = hiddenIds
    .filter((id) => !(lastEdit.has(id) && lastEdit.get(id) === null))
    .map((targetId) => ({ targetId, replacement: null }));
  let toolResultEntryId: string | undefined = rewind.toolResultEntryId;
  if (!toolResultEntryId || !onBranch.has(toolResultEntryId)) {
    toolResultEntryId =
      rewind.rewindEntryId !== undefined
        ? findToolResultEntryId(branch, rewind.toolCallId, {
            afterEntryId: rewind.rewindEntryId,
          })
        : findToolResultEntryId(branch, rewind.toolCallId, {
            beforeEntryId: instructionId,
          });
  }
  if (toolResultEntryId) {
    edits.push({
      targetId: toolResultEntryId,
      replacement: { content: [{ type: "text", text }] },
    });
  }
  return {
    edits,
    data: {
      toolCallId: rewind.toolCallId,
      status,
      reason,
      text,
      rewindTo: rewind.rewindTo,
      newLabel: rewind.newLabel,
      hiddenIds,
      toolResultEntryId: toolResultEntryId ?? null,
      instructionEntryId: instructionId ?? null,
    },
  };
}

/**
 * Entries that arrived on the active branch after `afterEntryId` (the
 * assistant that issued the rewind) from outside the conversation's own
 * turns: user messages (steers drained at the rewind turn's boundary) and
 * other extensions' custom messages (subagent results, timers). Oldest
 * first; this extension's own rewind and rewind-hint messages are excluded
 * (a pre-rewind rewind hint is stale once it lands).
 */
type LateEntry =
  | {
      kind: "user";
      message: Extract<SessionEntry, { type: "message" }>["message"];
    }
  | {
      kind: "custom";
      customType: string;
      content: unknown;
      display: boolean;
      details: unknown;
    };

function collectLateEntries(
  sm: SessionManager,
  afterEntryId: string,
): LateEntry[] {
  let branch: SessionEntry[];
  try {
    branch = sm.getBranch();
  } catch {
    return [];
  }
  const start = branch.findIndex((e) => e.id === afterEntryId);
  if (start < 0) return [];
  const late: LateEntry[] = [];
  for (const entry of branch.slice(start + 1)) {
    if (entry.type === "message" && entry.message.role === "user") {
      late.push({ kind: "user", message: entry.message });
    } else if (
      entry.type === "custom_message" &&
      entry.customType !== REWIND_CUSTOM_TYPE &&
      entry.customType !== SUMMARY_REQUEST_CUSTOM_TYPE &&
      entry.customType !== REWIND_HINT_CUSTOM_TYPE
    ) {
      late.push({
        kind: "custom",
        customType: entry.customType,
        content: entry.content,
        display: entry.display,
        details: entry.details,
      });
    }
  }
  return late;
}

export function buildRewindDoneText(
  pending: Pick<
    PendingRewind,
    "rewindTo" | "newLabel" | "beforeTokens" | "contextWindow"
  >,
  afterTokens: number,
): string {
  return (
    `[rewind to '${pending.rewindTo}' · collapsed as '${pending.newLabel}'] · ${formatContextDelta(pending.beforeTokens, afterTokens, pending.contextWindow)}\n\n` +
    `A branch_summary of the collapsed work has been appended to your context. It is the newest branch_summary, the current state; earlier branch_summary messages, if present, are older collapsed segments. Items under '### Done' are complete. Items under '### In Progress', '### Blocked', or '## Next Steps' are pending — execute them next without re-confirming with the user, continuing the latest branch_summary's work. User messages retained above predate this rewind — do not re-answer them.`
  );
}

export function buildLateEntriesNote(count: number): string {
  return count === 1
    ? "The message between the branch_summary and this one arrived while the rewind was pending — it is new and still unanswered; handle it now."
    : `The ${count} messages between the branch_summary and this one arrived while the rewind was pending — they are new and still unanswered; handle them now.`;
}

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

export default function (pi: ExtensionAPI) {
  patchAgentSessionPrototype();

  // Pending in-band rewinds (#80), keyed by session id: set by the rewind
  // tool call, completed or ended by a later boundary / event (#90).
  const pendingRewinds = new Map<string, PendingRewind>();
  // pi-codemode nested calls (#103): nested calls seen so far per codemode
  // call, and the codemode call each accepted nested rewind lands on.
  const nestedCallCounts = new Map<string, number>();
  const nestedRewindCallers = new Map<string, string>();
  // Codemode calls with a refused nested navigate_tree (#103, C): a refusal
  // counts as a direct one (D7), so turn_end does not count it as an attempt.
  const refusedNestedCallers = new Set<string>();
  // Forgets every nested call (#103, F): an abort after `beforeToolCall`
  // leaves a call with no tool_result, so its entries would outlive the run.
  const clearNestedCalls = (): void => {
    nestedCallCounts.clear();
    nestedRewindCallers.clear();
    refusedNestedCallers.clear();
  };

  // Turns the rewind handler owned (the rewind call's turn, summary
  // attempts, the completing turn): the hint and the auto start anchor skip
  // them, so neither lands on a rewind in progress or on a hidden attempt.
  const rewindTurns = new WeakSet<object>();
  // Turns that overflowed the window (#101, D14): marked by the rewind handler,
  // whether or not rewind-driven compaction is on, so the start anchor skips
  // them.
  const overflowedTurns = new WeakSet<object>();

  // A forced rewind (#101, rewind-driven compaction), keyed by session id:
  // set when the forced-rewind message goes out, deleted when the rewind
  // lands. A final error (D8) inside a run marks it `ended` instead (the gate
  // lifts, no second forced rewind in the same run, no start anchor) and
  // `agent_settled` deletes it.
  const forcedRewinds = new Map<string, ForcedRewind>();
  // Read by tests through `__testHooks.forcedRecord` (#107).
  forcedRewindsByExtension.set(pi, forcedRewinds);

  // Where pi's settings are read from (#101, S4): the session's cwd, agent
  // dir and project trust, set at `session_start`. `null` = cannot tell.
  let piSettingsSource: {
    cwd: string;
    agentDir: string;
    projectTrusted: boolean;
  } | null = null;

  /**
   * pi's compaction settings for `model` (#101, S3/S4/D9), or `null`. Read
   * from the captured session's live settings manager, the one pi decides
   * from; the disk (with the session's project trust) only when no session
   * is captured.
   */
  function piCompaction(
    sm: SessionManager,
    model: OverflowModel | undefined,
  ): { enabled: boolean; reserveTokens: number } | null {
    try {
      const session = findOwningSession(sm);
      let settingsManager = session
        ? asInternals(session).settingsManager
        : undefined;
      if (!settingsManager) {
        if (!piSettingsSource) return null;
        settingsManager = SettingsManager.create(
          piSettingsSource.cwd,
          piSettingsSource.agentDir,
          { projectTrusted: piSettingsSource.projectTrusted },
        );
      }
      const settings = settingsManager.getCompactionSettings(model);
      return {
        enabled: settings.enabled,
        reserveTokens: settings.reserveTokens,
      };
    } catch {
      // Invalid settings: pi's own compaction throws on them too.
      return null;
    }
  }

  /**
   * The window and target of rewind-driven compaction for `model`, or
   * `null` when it is off (#101, S2/S3): the model has a hint reserve,
   * `navigate_tree` is active for the session, and pi's auto-compaction is
   * off.
   */
  function rewindCompactionTarget(
    sm: SessionManager,
    sessionId: string,
    model: OverflowModel | undefined,
  ): { contextWindow: number; target: number } | null {
    if (!model || !(model.contextWindow && model.contextWindow > 0)) {
      return null;
    }
    if ((toolActiveBySession.get(sessionId) ?? toolActive) === false) {
      return null;
    }
    const reserve = resolveRewindHintReserve(
      {
        rewindHintReserveTokens,
        rewindHintModelReserveTokens: hintByModel,
      },
      `${model.provider}/${model.id}`,
    );
    if (reserve === null) return null;
    const pc = piCompaction(sm, model);
    if (!pc || pc.enabled) return null;
    return {
      contextWindow: model.contextWindow,
      target: model.contextWindow - pc.reserveTokens,
    };
  }

  /**
   * End rewind-driven compaction with the final error (#101, D8): the record
   * is marked ended (created if absent) and the D8 message is returned as a
   * boundary draft. The caller returns no `continue`.
   */
  function compactionFailed(
    sessionId: string,
    reason: string,
  ): SessionBoundaryDraft[] {
    const record = forcedRewinds.get(sessionId) ?? { attempts: 0 };
    record.ended = true;
    // #107: the override is only in force while the forced rewind is pending.
    delete record.projectedTokens;
    delete record.overrideEntryId;
    forcedRewinds.set(sessionId, record);
    return [
      rewindMessageDraft(buildRewindCompactionFailedText(reason), {
        status: "failed",
        reason,
      }),
    ];
  }

  /** Esc during a forced rewind ends it with the D8 row (#101, D16). */
  function abortedForced(sessionId: string): SessionBoundaryDraft[] {
    return forcedActive(sessionId)
      ? compactionFailed(sessionId, COMPACTION_FAILED_ABORTED)
      : [];
  }

  /** True while a forced rewind gates the session (#101, D6). */
  function forcedActive(sessionId: string): boolean {
    const record = forcedRewinds.get(sessionId);
    return !!record && !record.ended;
  }

  /**
   * Rewind-driven compaction (#101, D2–D5, D11, D14, S6): the one entry
   * point of both boundaries, after an overflow. `failedEntryId` is the
   * overflowed assistant (the withholding walk starts above it), `failedIds`
   * the entries of the failed attempt to hide. Returns the boundary drafts
   * and `continue`, or `null` to stand down.
   */
  function startRewindCompaction(
    sm: SessionManager,
    sessionId: string,
    at: { contextWindow: number; target: number },
    failedEntryId: string,
    failedIds: readonly string[],
  ): { drafts: SessionBoundaryDraft[]; continue: boolean } | null {
    const record = forcedRewinds.get(sessionId);
    // A final error already ended it in this run: pi ends the run.
    if (record?.ended) return null;
    const pending = livePending(sm, sessionId);
    // S6: one recovery per overflow.
    if (record) {
      const ending = pending
        ? endAtBoundary(
            sm,
            sessionId,
            pending,
            "failed",
            COMPACTION_FAILED_OVERFLOWED_AGAIN,
            failedIds,
          )
        : hideDrafts(failedIds);
      return {
        drafts: [
          ...ending,
          ...compactionFailed(sessionId, COMPACTION_FAILED_OVERFLOWED_AGAIN),
        ],
        continue: false,
      };
    }
    // D11: the pending rewind's summary request overflowed. Keep the
    // rewind, withhold before its call and re-send the frozen request.
    if (pending) {
      let instructionEntryId: string | undefined;
      try {
        instructionEntryId = pendingInstructionEntryId(sm.getBranch(), pending);
      } catch {
        instructionEntryId = undefined;
      }
      const plan = planWithhold(
        sm,
        pending.rewindEntryId,
        failedIds,
        at.target,
        instructionEntryId,
        overflowRequestTokens(sm, failedEntryId),
      );
      // D11 runs before #93's classification: a squeezed summary is
      // handled here, even with nothing to withhold.
      if (!plan.fits) {
        return {
          drafts: [
            ...endAtBoundary(
              sm,
              sessionId,
              pending,
              "failed",
              COMPACTION_FAILED_STILL_OVER,
              failedIds,
            ),
            ...compactionFailed(sessionId, COMPACTION_FAILED_STILL_OVER),
          ],
          continue: false,
        };
      }
      pending.hiddenEntryIds.push(...failedIds);
      forcedRewinds.set(sessionId, {
        attempts: 0,
        projectedTokens: plan.projectedTokens,
        overrideEntryId: plan.overrideEntryId,
      });
      hintTracker.markSpent();
      return {
        drafts: [...hideDrafts(failedIds), ...plan.drafts],
        continue: true,
      };
    }
    // A fresh forced rewind.
    const allEntries = sm.getEntries();
    const byId = new Map<string, SessionEntry>();
    for (const e of allEntries) byId.set(e.id, e);
    const anchors = formatActiveAnchors(sm, at.contextWindow, allEntries, byId);
    // D14: nothing to rewind to.
    if (anchors.names.length === 0) {
      return {
        drafts: [
          ...hideDrafts(failedIds),
          ...compactionFailed(sessionId, COMPACTION_FAILED_NO_ANCHOR),
        ],
        continue: false,
      };
    }
    const plan = planWithhold(
      sm,
      failedEntryId,
      failedIds,
      at.target,
      undefined,
      overflowRequestTokens(sm, failedEntryId),
    );
    forcedRewinds.set(sessionId, {
      attempts: 0,
      projectedTokens: plan.projectedTokens,
      overrideEntryId: plan.overrideEntryId,
    });
    hintTracker.markSpent();
    return {
      drafts: [
        ...hideDrafts(failedIds),
        ...plan.drafts,
        rewindMessageDraft(
          buildForcedRewindText(at.contextWindow, plan.withheld, anchors.text),
          {
            status: "forced",
            withheld: plan.withheld,
            contextWindow: at.contextWindow,
          },
        ),
      ],
      continue: true,
    };
  }

  /** Best-effort label write: `pi.setLabel`, else the captured manager. */
  function writeLabel(
    sm: SessionManager,
    entryId: string,
    label: string | undefined,
  ): boolean {
    try {
      pi.setLabel(entryId, label);
      return true;
    } catch {
      // Stale runtime: the captured SessionManager still writes.
      try {
        sm.appendLabelChange(entryId, label);
        return true;
      } catch {
        return false;
      }
    }
  }

  /** An ending as boundary drafts: hides, tool-result rewrite, outcome (#90). */
  function endingDrafts(plan: EndingPlan): SessionBoundaryDraft[] {
    return [
      ...plan.edits.map(
        (edit): SessionBoundaryDraft => ({
          type: "context_edit",
          targetId: edit.targetId,
          replacement: edit.replacement,
        }),
      ),
      {
        type: "custom",
        customType: REWIND_OUTCOME_CUSTOM_TYPE,
        data: plan.data,
      },
    ];
  }

  /** End the pending rewind at a boundary (`turn_end`, `agent_before_settle`). */
  function endAtBoundary(
    sm: SessionManager,
    sessionId: string,
    pending: PendingRewind,
    status: RewindOutcomeData["status"],
    reason: string,
    extraHidden: readonly string[] = [],
  ): SessionBoundaryDraft[] {
    const text = endingText(status, reason, pending);
    const plan = planEnding(sm, pending, status, reason, text, extraHidden);
    // Kept (as committing) until the outcome entry is seen on the branch.
    pendingRewinds.set(sessionId, pending);
    pending.committing = {
      status,
      reason,
      text,
      extraHidden,
      outcomeKey: JSON.stringify(plan.data),
    };
    return endingDrafts(plan);
  }

  /**
   * Settle a committing ending (#90): drop the pending rewind, and when pi
   * did not commit its drafts (no matching outcome entry on the branch),
   * apply the same ending outside a boundary. Returns the status and the
   * D8 reason (#101, D7) of an ending re-applied here, else `null`.
   */
  function settleCommittingEnding(
    sm: SessionManager,
    sessionId: string,
  ): { status: RewindOutcomeData["status"]; reason: string } | null {
    const pending = pendingRewinds.get(sessionId);
    const committing = pending?.committing;
    if (!pending || !committing) return null;
    pendingRewinds.delete(sessionId);
    let committed: boolean;
    try {
      committed = sm.getBranch().some((e) => {
        const data = readOutcomeData(e);
        return data !== null && JSON.stringify(data) === committing.outcomeKey;
      });
    } catch {
      // Cannot tell: never write a second outcome.
      committed = true;
    }
    if (committed) return null;
    endOutsideBoundary(
      sm,
      pending,
      committing.status,
      committing.reason,
      committing.text,
      committing.extraHidden,
    );
    const reason =
      pending.lastCause &&
      committing.reason === buildAttemptsExhaustedReason(pending.lastCause)
        ? COMPACTION_FAILED_NO_REWIND
        : committing.reason;
    return { status: committing.status, reason };
  }

  /** `sessionId`'s pending rewind, after settling a committing ending. */
  function livePending(
    sm: SessionManager,
    sessionId: string,
  ): PendingRewind | undefined {
    settleCommittingEnding(sm, sessionId);
    return pendingRewinds.get(sessionId);
  }

  /**
   * End a rewind outside a boundary (`session_compact`, `before_agent_start`,
   * `agent_settled`, `session_start`, the `context` handler): the same
   * `context_edit`s, appended directly, then the outcome entry. Every write
   * is guarded: an ending must never throw into pi.
   */
  function endOutsideBoundary(
    sm: SessionManager,
    rewind: EndableRewind,
    status: RewindOutcomeData["status"],
    reason: string,
    text: string,
    extraHidden: readonly string[] = [],
  ): void {
    const plan = planEnding(sm, rewind, status, reason, text, extraHidden);
    for (const edit of plan.edits) {
      try {
        sm.appendContextEdit(edit.targetId, edit.replacement);
      } catch {
        // Target gone: nothing to hide or rewrite there.
      }
    }
    try {
      pi.appendEntry(REWIND_OUTCOME_CUSTOM_TYPE, plan.data);
    } catch {
      try {
        sm.appendCustomEntry(REWIND_OUTCOME_CUSTOM_TYPE, plan.data);
      } catch {
        // Fail-open: the context edits above already landed.
      }
    }
  }

  /** End `sessionId`'s pending rewind (if any) outside a boundary. */
  function endPendingOutsideBoundary(
    sm: SessionManager,
    sessionId: string,
    status: RewindOutcomeData["status"],
    reason: string,
  ): void {
    const pending = livePending(sm, sessionId);
    if (!pending) return;
    pendingRewinds.delete(sessionId);
    endOutsideBoundary(
      sm,
      pending,
      status,
      reason,
      endingText(status, reason, pending),
    );
  }

  /**
   * O8 (#90): at session start, end every rewind that died mid-flight — an
   * instruction on the active branch (after the newest compaction) with
   * neither its `branch:<newLabel>` completion label nor an outcome entry
   * after it. The assistant turns after such an instruction, up to
   * MAX_SUMMARY_ATTEMPTS (user messages, e.g. steers, skipped and kept),
   * were attempts (tool calls were blocked), so those are hidden with it.
   */
  function resolveOrphanRewinds(sm: SessionManager): void {
    let branch: SessionEntry[];
    try {
      branch = sm.getBranch();
    } catch {
      return;
    }
    let start = 0;
    for (let i = branch.length - 1; i >= 0; i--) {
      if (branch[i].type === "compaction") {
        start = i + 1;
        break;
      }
    }
    for (let i = start; i < branch.length; i++) {
      const entry = branch[i];
      if (
        entry.type !== "custom_message" ||
        entry.customType !== SUMMARY_REQUEST_CUSTOM_TYPE
      ) {
        continue;
      }
      const details = (entry.details ?? {}) as {
        toolCallId?: unknown;
        rewindTo?: unknown;
        newLabel?: unknown;
      };
      if (typeof details.toolCallId !== "string") continue;
      const toolCallId = details.toolCallId;
      const newLabel = String(details.newLabel ?? "");
      const later = branch.slice(i + 1);
      const resolved = later.some((e) => {
        const data = readOutcomeData(e);
        if (data) {
          // By instruction entry id (tool-call ids can be "" or reused);
          // by toolCallId only for outcomes written without it.
          if (data.instructionEntryId === entry.id) return true;
          if (
            data.instructionEntryId === undefined &&
            data.toolCallId === toolCallId
          ) {
            return true;
          }
        }
        try {
          return sm.getLabel(e.id) === BRANCH_LABEL_PREFIX + newLabel;
        } catch {
          return false;
        }
      });
      if (resolved) continue;
      // The attempt window: at most MAX_SUMMARY_ATTEMPTS assistant turns
      // (and their tool results). User messages are skipped, not a stop:
      // steers typed while pending sit between the instruction and
      // attempt 1, and stay visible.
      const attempts: string[] = [];
      let assistants = 0;
      for (const e of later) {
        if (e.type !== "message") continue;
        if (e.message.role === "assistant") {
          if (assistants >= MAX_SUMMARY_ATTEMPTS) break;
          assistants++;
        } else if (e.message.role !== "toolResult") {
          continue;
        }
        attempts.push(e.id);
      }
      endOutsideBoundary(
        sm,
        {
          toolCallId,
          rewindTo: String(details.rewindTo ?? ""),
          newLabel,
          hiddenEntryIds: [],
          instructionEntryId: entry.id,
        },
        "aborted",
        "the session restarted",
        REWIND_ABORTED_TEXT,
        attempts,
      );
    }
  }

  /**
   * Why the pending rewind cannot go on to a summary attempt, or `null`
   * (#90, §15.5): a compaction landed after the rewind call (O7), or the
   * rewind turn has passed and its instruction is not on the branch (another
   * extension's invalid draft dropped the whole boundary list, or pi skipped
   * the boundary). Compaction is read from a compaction entry, never from a
   * missing instruction.
   */
  function undeliveredReason(
    sm: SessionManager,
    pending: PendingRewind,
  ): string | null {
    let branch: SessionEntry[];
    try {
      branch = sm.getBranch();
    } catch {
      return null;
    }
    if (!branch.some((e) => e.id === pending.rewindEntryId)) {
      return "the rewind call is no longer on the active branch";
    }
    if (compactedAfter(branch, pending.rewindEntryId)) {
      return "the context was compacted";
    }
    const rewindTurnPassed =
      pending.instructionSent === true ||
      findToolResultEntryId(branch, pending.toolCallId, {
        afterEntryId: pending.rewindEntryId,
      }) !== undefined;
    if (!rewindTurnPassed) return null;
    return pendingInstructionEntryId(branch, pending) !== undefined
      ? null
      : "the summary request could not be delivered";
  }

  // Phase 2 of an in-band rewind (#80): branch from the anchor with the
  // summary the live model just wrote. Nothing is in flight — the summary
  // turn has ended — so the rewind tool call, its tool result, the
  // instruction and the summary all stay on the abandoned branch and no
  // synthetic tool_use is needed to keep the new branch structurally valid.
  // Returns the boundary drafts: the continuation, or an ending.
  async function completeRewind(
    sm: SessionManager,
    sessionId: string,
    pending: PendingRewind,
    summaryText: string,
    summaryEntryId: string | undefined,
    turnIds: readonly string[],
  ): Promise<SessionBoundaryDraft[]> {
    const fail = (reason: string) =>
      endAtBoundary(sm, sessionId, pending, "failed", reason, turnIds);
    // A compaction or tree move during the summary turn invalidates the
    // rewind: the anchor must still be the same labeled entry on the active
    // branch, and the new label must still be free.
    if (
      findLabeledEntry(sm, LABEL_PREFIX + pending.rewindTo) !== pending.target
    ) {
      return fail(`'${pending.rewindTo}' is no longer on the active branch`);
    }
    if (findLabeledEntry(sm, LABEL_PREFIX + pending.newLabel)) {
      return fail(
        `a label '${pending.newLabel}' now exists on the active branch`,
      );
    }
    const staleReason = rewindStaleReason(sm, pending.rewindEntryId);
    if (staleReason) return fail(staleReason);
    const oldLeafId = sm.getLeafId();
    // Input that arrived after the rewind call (user steers drained at the
    // rewind turn's boundary, other extensions' messages) sits on the segment
    // about to be abandoned. It is new input, not collapsed work: it is
    // re-appended onto the new branch below, in order, ahead of the
    // continuation (which flags it as unanswered).
    const lateEntries = collectLateEntries(sm, pending.rewindEntryId);
    // Completion marker (#90, O8): `branch:<newLabel>` on the summary turn,
    // the abandoned branch's last real entry, written before the branch so
    // the restart check never reads this rewind as died mid-flight.
    const branchLabel = BRANCH_LABEL_PREFIX + pending.newLabel;
    const branchLabelWritten = summaryEntryId
      ? writeLabel(sm, summaryEntryId, branchLabel)
      : false;
    let entries: SessionEntry[];
    let readFiles: string[];
    let modifiedFiles: string[];
    let summaryId: string;
    // The leaf right before the branch call (the label write above appended
    // an entry, so it is not `oldLeafId`).
    let leafBeforeBranch: string | null | undefined;
    try {
      ({ entries } = collectEntriesForBranchSummary(
        sm,
        oldLeafId,
        pending.target,
      ));
      // Hidden attempts' tool calls never ran (blocked): keep them out of
      // the file lists (#90, O3-B).
      const hidden = new Set(pending.hiddenEntryIds);
      const { fileOps } = prepareBranchEntries(
        entries.filter((e) => !hidden.has(e.id)),
        Number.MAX_SAFE_INTEGER,
      );
      ({ readFiles, modifiedFiles } = computeFileLists(fileOps));
      // `fromHook: true` pairs with the `session_tree` payload's
      // `fromExtension: true`, as pi's own navigateTree does for an
      // extension-provided summary.
      leafBeforeBranch = sm.getLeafId();
      summaryId = sm.branchWithSummary(
        pending.target,
        buildStoredSummary(summaryText, readFiles, modifiedFiles),
        { readFiles, modifiedFiles },
        true,
      );
    } catch (err) {
      // pi's branchWithSummary moves the leaf and appends in memory before
      // the file append that may throw: put the leaf back, or the session
      // stays collapsed while the model is told nothing was.
      try {
        if (leafBeforeBranch && sm.getLeafId() !== leafBeforeBranch) {
          sm.branch(leafBeforeBranch);
        }
      } catch {
        // Fail-open: the ending below still reaches the model.
      }
      // Nothing branched: undo the completion marker, then tell the model,
      // or a text-only summary turn would end the run with no word of it.
      if (branchLabelWritten && summaryEntryId) {
        writeLabel(sm, summaryEntryId, undefined);
      }
      return fail(
        `the branch could not be written (${err instanceof Error ? err.message : String(err)})`,
      );
    }
    pendingRewinds.delete(sessionId);
    // A forced rewind landed: the gate lifts (#101, D7).
    forcedRewinds.delete(sessionId);
    // The summary landed unlabeled if both writes fail; `list` simply won't
    // show it.
    writeLabel(sm, summaryId, LABEL_PREFIX + pending.newLabel);
    // Direct appends, not the steer queue: no second pass through other
    // extensions' input handlers, and anything still queued is delivered
    // after these, keeping the original order.
    let reappended = 0;
    for (const late of lateEntries) {
      try {
        if (late.kind === "user") {
          sm.appendMessage(
            late.message as Parameters<SessionManager["appendMessage"]>[0],
          );
        } else {
          sm.appendCustomMessageEntry(
            late.customType,
            late.content as Parameters<
              SessionManager["appendCustomMessageEntry"]
            >[1],
            late.display,
            late.details,
          );
        }
        reappended++;
      } catch {
        // Fail-open: the original stays on disk on the abandoned branch.
      }
    }
    // The branch has landed: every step below is guarded so the
    // continuation always reaches the model.
    let afterTokens = 0;
    try {
      afterTokens = estimateActiveBranchTokens(sm);
    } catch {
      afterTokens = 0;
    }
    const refreshed = refreshAgentMessages(sm);
    const historyRewriteNotified = await emitSessionTree(
      sm,
      oldLeafId,
      summaryId,
    );
    return [
      rewindMessageDraft(
        buildRewindDoneText(pending, afterTokens) +
          (reappended > 0 ? `\n\n${buildLateEntriesNote(reappended)}` : "") +
          (refreshed ? "" : `\n\n${REFLECTION_BOOTSTRAP_WARNING_REWIND}`),
        {
          status: "done",
          rewindTo: pending.rewindTo,
          newLabel: pending.newLabel,
          targetId: pending.target,
          summaryId,
          collapsedEntries: entries.length,
          contextBefore: pending.beforeTokens,
          contextAfter: afterTokens,
          contextWindow: pending.contextWindow,
          summaryAttempts: pending.attempts + 1,
          agentMessagesRefreshed: refreshed,
          historyRewriteNotified,
          readFiles,
          modifiedFiles,
          reappendedLateMessages: reappended,
        },
      ),
    ];
  }

  // While a rewind waits for its summary, every tool call is blocked: the
  // next turn must be the summary itself. The blocked attempt is hidden from
  // the model (#90).
  //
  // pi-codemode (#103): a script's nested calls reach this gate as tool calls
  // whose `parentToolCallId` is the codemode call's id. A nested rewind is
  // accepted only as the script's first nested call; `execute` lands it on
  // the codemode call, so it behaves like a direct rewind. Under a forced
  // rewind (#101) the top-level `codemode` call passes so its script can
  // rewind, and every other nested call is blocked: the gate still means
  // "rewind only".
  pi.on("tool_call", (event, ctx) => {
    const isRewind =
      event.toolName === TOOL_NAME &&
      (event.input as { action?: unknown } | undefined)?.action === "rewind";
    const parentId =
      typeof event.parentToolCallId === "string" &&
      event.parentToolCallId.length > 0
        ? event.parentToolCallId
        : undefined;
    if (parentId !== undefined) {
      const priorNested = nestedCallCounts.get(parentId) ?? 0;
      nestedCallCounts.set(parentId, priorNested + 1);
      if (isRewind) {
        if (priorNested > 0) {
          refusedNestedCallers.add(parentId);
          return { block: true, reason: NESTED_REWIND_REFUSAL };
        }
        nestedRewindCallers.set(event.toolCallId, parentId);
      }
    }
    let sessionId: string;
    try {
      sessionId = ctx.sessionManager.getSessionId();
    } catch {
      return undefined;
    }
    // A forced rewind (#101, D6): `terminate` so a turn of blocked calls
    // ends the run unless `turn_end` re-sends (an ending's `continue: false`
    // must hold).
    const forced = forcedActive(sessionId);
    if (livePending(ctx.sessionManager as SessionManager, sessionId)) {
      // A blocked call gets no tool_result (pi-agent-core agent-loop.js: an
      // immediate result skips afterToolCall), so its nested-rewind entry
      // (#103) is retired here.
      nestedRewindCallers.delete(event.toolCallId);
      return {
        block: true,
        reason: SUMMARY_PENDING_BLOCK_REASON,
        ...(forced ? { terminate: true } : {}),
      };
    }
    const topLevelCodemode =
      event.toolName === CODEMODE_TOOL_NAME && parentId === undefined;
    if (forced && !isRewind && !topLevelCodemode) {
      return {
        block: true,
        reason: FORCED_REWIND_BLOCK_REASON,
        terminate: true,
      };
    }
    return undefined;
  });

  // Rewind turn and summary attempts (#80, #90). Registered before the hint
  // / start-anchor handler so a completed rewind is already on the branch
  // when that handler reads the context usage.
  pi.on("turn_end", async (event, ctx) => {
    let sm: SessionManager;
    let sessionId: string;
    try {
      sm = ctx.sessionManager as SessionManager;
      sessionId = sm.getSessionId();
    } catch {
      return;
    }
    const pending = livePending(sm, sessionId);
    // Every return keeps other handlers' entries (pi replaces the list per
    // handler) and drops `context_edit`s whose target left the branch, ours
    // included (one invalid draft makes pi drop the whole list).
    const deliverRaw = (drafts: SessionBoundaryDraft[], cont: boolean) => ({
      entries: keepValidDrafts(sm, [...(event.entries ?? []), ...drafts]),
      continue: cont,
    });
    // The three-attempts ending of the pending rewind during a forced rewind
    // ends rewind-driven compaction too (#101, D7): D8, no `continue`. Other
    // failed endings leave the gate as it is.
    const deliver = (drafts: SessionBoundaryDraft[], cont: boolean) => {
      const pendingNow = pendingRewinds.get(sessionId);
      const ending = pendingNow?.committing;
      const last = pendingNow?.lastCause;
      if (
        ending?.status !== "failed" ||
        !forcedActive(sessionId) ||
        !last ||
        ending.reason !== buildAttemptsExhaustedReason(last)
      ) {
        return deliverRaw(drafts, cont);
      }
      return deliverRaw(
        [
          ...drafts,
          ...compactionFailed(sessionId, COMPACTION_FAILED_NO_REWIND),
        ],
        false,
      );
    };
    const message = event.message as {
      role?: string;
      content?: unknown;
      stopReason?: string;
    };
    if (message?.role !== "assistant") return;
    const messageEntryId =
      typeof event.messageEntryId === "string" ? event.messageEntryId : "";
    const turnIds = messageEntryId
      ? [
          messageEntryId,
          ...(event.toolResultEntryIds ?? []).filter(
            (id): id is string => typeof id === "string",
          ),
        ]
      : [];
    // Rewind-driven compaction (#101, D2): a `length` / silent overflow is
    // handled here, so the model never retries the oversized call; an
    // explicit error ends the loop and is handled at `agent_before_settle`.
    // Another extension's `continue` owns the turn.
    if (!event.continue && messageEntryId && overflowed(message, ctx.model)) {
      overflowedTurns.add(message);
      const at = rewindCompactionTarget(sm, sessionId, ctx.model);
      if (at) {
        rewindTurns.add(message);
        if (message.stopReason === "error") return;
        // A silent `stop` reply stays visible (§9.1); a squeezed reply and
        // its failed calls, or a summary attempt, are hidden.
        const failedIds =
          message.stopReason === "length" || pending ? turnIds : [];
        const result = startRewindCompaction(
          sm,
          sessionId,
          at,
          messageEntryId,
          failedIds,
        );
        if (result) return deliverRaw(result.drafts, result.continue);
      }
    }
    if (!pending) {
      // A forced-rewind attempt (#101, D6): anything but an accepted
      // `rewind` (which makes the rewind pending) is hidden and re-sent.
      if (!forcedActive(sessionId)) return;
      rewindTurns.add(message);
      let aborted = message.stopReason === "aborted";
      try {
        aborted ||= ctx.signal?.aborted === true;
      } catch {
        // Stale ctx: treat as not aborted.
      }
      // Esc ends the gate here (D16): the attempt is hidden, one D8 row.
      // An error: pi's retry, or `agent_settled`.
      if (aborted) {
        return deliverRaw(
          [
            ...hideDrafts(turnIds),
            ...compactionFailed(sessionId, COMPACTION_FAILED_ABORTED),
          ],
          false,
        );
      }
      if (message.stopReason === "error") return;
      // A refused `rewind` is not an attempt (D7): it stays visible.
      // A nested refusal (#103) is recorded under its codemode call.
      const results = (event.toolResults ?? []) as Array<{
        toolName?: string;
        toolCallId?: string;
        details?: unknown;
      }>;
      const refused = results.some(
        (result) =>
          refusedNestedCallers.has(result.toolCallId ?? "") ||
          (result.toolName === TOOL_NAME &&
            (result.details as { refusal?: unknown } | undefined)?.refusal ===
              true),
      );
      for (const result of results) {
        refusedNestedCallers.delete(result.toolCallId ?? "");
      }
      if (refused) return;
      const record = forcedRewinds.get(sessionId);
      if (!record) return;
      record.attempts++;
      if (record.attempts < MAX_SUMMARY_ATTEMPTS) {
        return deliverRaw(hideDrafts(turnIds), true);
      }
      return deliverRaw(
        [
          ...hideDrafts(turnIds),
          ...compactionFailed(sessionId, COMPACTION_FAILED_NO_REWIND),
        ],
        false,
      );
    }
    rewindTurns.add(message);
    const content = Array.isArray(message.content)
      ? (message.content as Array<{
          type?: string;
          id?: string;
          text?: string;
        }>)
      : [];
    // This turn's tool results by toolCallId (`toolResultEntryIds` holds only
    // the resolved ids, so never pair it with `toolResults` by index).
    const resultIds = new Map<string, string>();
    for (const id of event.toolResultEntryIds ?? []) {
      try {
        const entry = sm.getEntry(id);
        if (entry?.type === "message" && entry.message.role === "toolResult") {
          resultIds.set(entry.message.toolCallId, id);
        }
      } catch {
        // Unresolvable: the pending truncation still covers it.
      }
    }

    // The turn that issued the rewind: answer it with the instruction.
    // Matched by entry id when pi resolves it: a summary attempt may call a
    // tool with the same ("" or reused) id as the rewind.
    const isRewindTurn =
      typeof event.messageEntryId === "string"
        ? event.messageEntryId === pending.rewindEntryId
        : !pending.instructionSent &&
          content.some(
            (b) => b?.type === "toolCall" && b.id === pending.toolCallId,
          );
    if (isRewindTurn) {
      pending.toolResultEntryId = resultIds.get(pending.toolCallId);
      let aborted = message.stopReason === "aborted";
      try {
        aborted ||= ctx.signal?.aborted === true;
      } catch {
        // Stale ctx: treat as not aborted.
      }
      if (aborted) {
        // Esc before the instruction was committed (#90, O6).
        return deliver(
          [
            ...endAtBoundary(sm, sessionId, pending, "aborted", "aborted"),
            ...abortedForced(sessionId),
          ],
          false,
        );
      }
      pending.instructionSent = true;
      // A boundary entry: pi commits it right after the rewind's tool
      // result, BEFORE it drains queued steers, and `continue` guarantees
      // the summary request (#90, S1).
      return deliver([summaryRequestDraft(pending)], true);
    }

    // A summary attempt. Its own entries — the assistant and every
    // resolvable tool result, never one without the other (O3) — are what
    // a failure hides (`turnIds`).
    // Esc while a blocked attempt's tools resolve: the assistant completed
    // (`toolUse`) but the run was aborted. Esc while pending is an abort
    // (O6), never a failed attempt.
    let attemptAborted = message.stopReason === "aborted";
    try {
      attemptAborted ||= ctx.signal?.aborted === true;
    } catch {
      // Stale ctx: treat as not aborted.
    }
    if (attemptAborted) {
      return deliver(
        [
          ...endAtBoundary(
            sm,
            sessionId,
            pending,
            "aborted",
            "aborted",
            turnIds,
          ),
          ...abortedForced(sessionId),
        ],
        false,
      );
    }
    // Provider errors go through pi's auto-retry (or pi-fallback-provider's
    // retry on another model): not an attempt. The rewind waits; if no
    // retry comes, `agent_settled` / `before_agent_start` end it.
    if (message.stopReason === "error") return;
    const undelivered = undeliveredReason(sm, pending);
    if (undelivered) {
      return deliver(
        endAtBoundary(sm, sessionId, pending, "failed", undelivered, turnIds),
        true,
      );
    }
    const hasToolCall = content.some((b) => b?.type === "toolCall");
    const summary = extractSummaryText(
      content
        .filter((b) => b?.type === "text" && typeof b.text === "string")
        .map((b) => b.text)
        .join("\n"),
    );
    // Precedence: a cut-off reply that also calls a tool is "cut off" (the
    // advice that helps is a tighter focus).
    const cause: SummaryFailureCause | null =
      message.stopReason === "length"
        ? "cut off"
        : hasToolCall
          ? "called a tool"
          : summary === null
            ? "no '## Goal' summary"
            : null;
    if (cause !== null || summary === null) {
      // (`summary === null` implies a cause; the check narrows the type.)
      const failure = cause ?? "no '## Goal' summary";
      pending.attempts++;
      pending.lastCause = failure;
      if (pending.attempts >= MAX_SUMMARY_ATTEMPTS) {
        return deliver(
          endAtBoundary(
            sm,
            sessionId,
            pending,
            "failed",
            buildAttemptsExhaustedReason(failure),
            turnIds,
          ),
          true,
        );
      }
      // Re-send (#90, S4): hide the attempt and ask again with the same
      // messages — nothing appended.
      pending.hiddenEntryIds.push(...turnIds);
      return deliver(
        turnIds.map(
          (targetId): SessionBoundaryDraft => ({
            type: "context_edit",
            targetId,
            replacement: null,
          }),
        ),
        true,
      );
    }
    try {
      return deliver(
        await completeRewind(
          sm,
          sessionId,
          pending,
          summary,
          messageEntryId || undefined,
          turnIds,
        ),
        true,
      );
    } catch (err) {
      // A throw before the branch write (the stale checks read the session)
      // leaves the rewind pending: end it so the model hears about it.
      // Every step after the write is guarded, so the branch never lands
      // without its continuation.
      const still = pendingRewinds.get(sessionId);
      if (!still || still.committing) return;
      return deliver(
        endAtBoundary(
          sm,
          sessionId,
          pending,
          "failed",
          `the rewind could not be completed (${err instanceof Error ? err.message : String(err)})`,
          turnIds,
        ),
        true,
      );
    }
  });

  // The run would settle with a rewind still pending (#90, O6): another
  // handler cleared `continue`, or the last attempt never reached
  // `turn_end`. Not on an error outcome or an explicit continue (pi's
  // auto-retry / pi-fallback-provider retry the turn): those wait for
  // `agent_settled`.
  pi.on("agent_before_settle", (event, ctx) => {
    let sm: SessionManager;
    let sessionId: string;
    try {
      sm = ctx.sessionManager as SessionManager;
      sessionId = sm.getSessionId();
    } catch {
      return;
    }
    // A failed ending pi dropped (e.g. the exhausted attempts' turn_end)
    // comes with `continue`, as it would have at its boundary (O6): the
    // model reads the failure in the same run.
    const reapplied = settleCommittingEnding(sm, sessionId);
    if (
      reapplied?.status === "failed" &&
      event.outcome !== "error" &&
      event.outcome !== "aborted"
    ) {
      // A forced rewind's failed ending (#101, D8): the D8 gate, no
      // `continue`, as `deliver()` gives it.
      if (forcedRewinds.get(sessionId)?.ended) {
        return {
          entries: keepValidDrafts(sm, [
            ...(event.entries ?? []),
            ...compactionFailed(sessionId, reapplied.reason),
          ]),
          continue: false,
        };
      }
      return { continue: true };
    }
    // Rewind-driven compaction (#101, D2): an explicit overflow error ended
    // the loop; pi did not retry (overflow) or compact (off). Recover in the
    // same run. Another extension's `continue` owns the recovery.
    if (event.outcome === "error" && !event.continue) {
      let last: SessionEntry | undefined;
      try {
        last = sm
          .getBranch()
          .findLast(
            (e) => e.type === "message" && e.message.role === "assistant",
          );
      } catch {
        last = undefined;
      }
      const failed = last?.type === "message" ? last.message : undefined;
      if (
        last &&
        (failed as { stopReason?: string } | undefined)?.stopReason ===
          "error" &&
        overflowed(failed, ctx.model)
      ) {
        const at = rewindCompactionTarget(sm, sessionId, ctx.model);
        const result = at
          ? startRewindCompaction(sm, sessionId, at, last.id, [last.id])
          : null;
        if (result) {
          return {
            entries: keepValidDrafts(sm, [
              ...(event.entries ?? []),
              ...result.drafts,
            ]),
            continue: result.continue,
          };
        }
      }
    }
    const pending = pendingRewinds.get(sessionId);
    // Esc during a forced rewind (#101, D16): the D8 row, after a pending
    // rewind's aborted ending.
    if (event.outcome === "aborted" && forcedActive(sessionId)) {
      const ending = pending
        ? endAtBoundary(sm, sessionId, pending, "aborted", "aborted")
        : [];
      return {
        entries: keepValidDrafts(sm, [
          ...(event.entries ?? []),
          ...ending,
          ...abortedForced(sessionId),
        ]),
        continue: false,
      };
    }
    if (!pending || event.outcome === "error" || event.continue) return;
    if (event.outcome === "aborted") {
      return {
        entries: keepValidDrafts(sm, [
          ...(event.entries ?? []),
          ...endAtBoundary(sm, sessionId, pending, "aborted", "aborted"),
        ]),
      };
    }
    const ending = endAtBoundary(
      sm,
      sessionId,
      pending,
      "failed",
      "the run ended before the summary",
    );
    // During a forced rewind: D8, the run ends (#101, D7).
    const failed = forcedActive(sessionId)
      ? compactionFailed(sessionId, "the run ended before the summary")
      : [];
    return {
      entries: keepValidDrafts(sm, [
        ...(event.entries ?? []),
        ...ending,
        ...failed,
      ]),
      continue: failed.length === 0,
    };
  });

  // The last assistant reply ended with Esc (stopReason "aborted", #101, D16).
  function lastReplyAborted(sm: SessionManager): boolean {
    try {
      const last = sm
        .getBranch()
        .findLast(
          (e) => e.type === "message" && e.message.role === "assistant",
        );
      return (
        last?.type === "message" &&
        (last.message as { stopReason?: string }).stopReason === "aborted"
      );
    } catch {
      return false;
    }
  }

  // Fully settled with the rewind still pending: the run was aborted, or
  // the errored attempt's retry never came (#90, O6).
  pi.on("agent_settled", (event, ctx) => {
    // Nothing is in flight once settled: drop any nested-call leftovers (#103, F).
    clearNestedCalls();
    // Esc ended the run (#101, D16). The event flag exists on pi >= 1.1.0;
    // on older pi the reply Esc cut short says it (peer floor 0.87.0).
    let aborted = event.aborted === true;
    try {
      const sm = ctx.sessionManager as SessionManager;
      aborted ||= lastReplyAborted(sm);
      endPendingOutsideBoundary(
        sm,
        sm.getSessionId(),
        "aborted",
        aborted ? "aborted" : "the run ended after an error",
      );
    } catch {
      // Stale ctx: nothing to end.
    }
    // A forced rewind still gating the session (#101, D16): lift the gate
    // and say so. An ended one already said it (D8).
    let sm: SessionManager;
    let sessionId: string;
    try {
      sm = ctx.sessionManager as SessionManager;
      sessionId = sm.getSessionId();
    } catch {
      return;
    }
    const record = forcedRewinds.get(sessionId);
    forcedRewinds.delete(sessionId);
    if (!record || record.ended) return;
    const reason = aborted
      ? COMPACTION_FAILED_ABORTED
      : COMPACTION_FAILED_RUN_ENDED;
    const content = buildRewindCompactionFailedText(reason);
    const details = { status: "failed", reason };
    try {
      // Idle: a plain append, no turn started.
      pi.sendMessage({
        customType: REWIND_CUSTOM_TYPE,
        content,
        display: true,
        details,
      });
    } catch {
      try {
        sm.appendCustomMessageEntry(REWIND_CUSTOM_TYPE, content, true, details);
      } catch {
        // Fail-open: the gate is lifted either way.
      }
    }
  });

  // The session is going away (quit, reload, new, resume, fork): a forced
  // rewind cannot outlive it (#101).
  pi.on("session_shutdown", (_event, ctx) => {
    clearNestedCalls();
    try {
      forcedRewinds.delete(
        (ctx.sessionManager as SessionManager).getSessionId(),
      );
    } catch {
      // Fail-open: session_start clears it too.
    }
  });

  // A compaction while pending (threshold or overflow recovery) runs before
  // the next request: end the rewind before that request goes out (#90, O7).
  pi.on("session_compact", (_event, ctx) => {
    try {
      const sm = ctx.sessionManager as SessionManager;
      endPendingOutsideBoundary(
        sm,
        sm.getSessionId(),
        "failed",
        "the context was compacted",
      );
    } catch {
      // Stale ctx: the next turn_end / context call ends it.
    }
  });

  // TUI rows (#90, O1 / O9). Local components styled through `theme`.
  pi.registerMessageRenderer(
    SUMMARY_REQUEST_CUSTOM_TYPE,
    (message, options, theme) => {
      const d = (message.details ?? {}) as Partial<{
        rewindTo: string;
        newLabel: string;
        contextBefore: number;
        tokensAtTarget: number;
        contextWindow: number;
      }>;
      const row = buildSummaryRequestRow({
        rewindTo: d.rewindTo ?? "?",
        newLabel: d.newLabel ?? "?",
        contextBefore: d.contextBefore ?? 0,
        tokensAtTarget: d.tokensAtTarget ?? 0,
        contextWindow: d.contextWindow ?? 0,
      });
      const lines = [row];
      if (options.expanded) {
        lines.push("", ...extractTextContent(message.content).split("\n"));
      }
      return new StyledLines(
        lines,
        (line) => theme.fg("customMessageText", line),
        options.outputPad ?? 1,
      );
    },
  );
  pi.registerEntryRenderer(
    REWIND_OUTCOME_CUSTOM_TYPE,
    (entry, _options, theme) => {
      const data = entry.data as Partial<RewindOutcomeData> | undefined;
      if (!data || (data.status !== "failed" && data.status !== "aborted")) {
        return undefined;
      }
      return new StyledLines(
        [
          buildRewindOutcomeRow({
            status: data.status,
            reason: String(data.reason ?? ""),
            rewindTo: String(data.rewindTo ?? "?"),
            newLabel: String(data.newLabel ?? "?"),
          }),
        ],
        (line) => theme.fg("error", line),
      );
    },
  );
  // The final error of rewind-driven compaction (#101, D8): one
  // error-styled row, full text, nothing to expand. Every other rewind
  // message (the continuation, the forced-rewind message) keeps pi's
  // default rendering.
  pi.registerMessageRenderer(REWIND_CUSTOM_TYPE, (message, options, theme) => {
    const status = (message.details as { status?: unknown } | undefined)
      ?.status;
    if (status !== "failed") return undefined;
    return new StyledLines(
      extractTextContent(message.content).split("\n"),
      (line) => theme.fg("error", line),
      options.outputPad ?? 1,
    );
  });

  // Rewind-hint closure state (#44): fresh per factory invocation and reset
  // on every `session_start`, so `/new` / `/resume` / `/reload` sessions
  // never inherit a spent crossing or another session's config. `toolActive`
  // fails open until `before_agent_start` says otherwise. `hintByModel`
  // (#86) holds the `modelOverrides` reserves; the reserve is resolved
  // against the live `ctx.model` on every `turn_end`, so a model switch
  // mid-session takes effect on the next turn.
  let toolActive = true;
  let rewindHintReserveTokens: number | null = null;
  let hintByModel: ReadonlyMap<string, number | null> = new Map();
  let hintTracker = new RewindHintTracker();

  // Auto start-anchor closure state (#55). All of it is keyed by session id
  // and NEVER cleared in-process: once a session has settled (write or
  // found-existing), no later prompt / agent loop re-inserts the label.
  // `/reload` builds a fresh closure, so a reloaded session re-settles once
  // via the existence check (no duplicate write). The per-session maps keep
  // two in-process sessions (e.g. the main session and a subagent) from
  // sharing each other's config value, tool-active gate, or settle marker.
  const anchorStartSettled = new Set<string>();
  const anchorStartTurnsBySession = new Map<string, number>();
  const toolActiveBySession = new Map<string, boolean>();

  // Auto start anchor (#55): silent `anchor:start` write, shared by the
  // `turn_end` trigger and the `agent_end` fallback. `completedTurns` is the
  // number of finished turns in the current agent run
  // (`Number.POSITIVE_INFINITY` for the run-end fallback). Fail-open
  // everywhere: any read/write failure leaves the session unsettled so the
  // next turn or run end retries, and nothing throws into the agent loop.
  function writeStartAnchorIfNeeded(
    sm: SessionManager,
    completedTurns: number,
  ): void {
    // Kill switch read at write time:
    // a mid-process change applies without an extension reload.
    if (process.env.PI_NAVIGATE_TREE_START_ANCHOR === "0") return;
    let sessionId: string;
    try {
      sessionId = sm.getSessionId();
    } catch {
      return;
    }
    // Per-session tool gate (#55): a sibling session's `before_agent_start`
    // (e.g. a subagent whose tool subset omits `navigate_tree`) must not
    // clear this session's gate. Absent = fail-open until a gate is set.
    if (toolActiveBySession.get(sessionId) === false) return;
    if (anchorStartSettled.has(sessionId)) return;
    // Never at the context limit (#101, D14): not during a forced rewind,
    // nor after its final error in the same run.
    if (forcedRewinds.has(sessionId)) return;
    // Per-session threshold (#55): two sessions started with different
    // project configs must not share one closure value.
    const startAfterTurns =
      anchorStartTurnsBySession.get(sessionId) ?? START_ANCHOR_DEFAULT_TURNS;
    if (completedTurns < startAfterTurns) return;
    try {
      // The existence check is authoritative: a `start` already on the
      // active branch (manual `/tree` label, pre-reload write, prior
      // settle) wins and settles the session without another write.
      if (findLabeledEntry(sm, START_ANCHOR_LABEL)) {
        anchorStartSettled.add(sessionId);
        return;
      }
    } catch {
      return;
    }
    let leafId: string | null;
    try {
      leafId = sm.getLeafId();
    } catch {
      return;
    }
    if (!leafId) return;
    try {
      pi.setLabel(leafId, START_ANCHOR_LABEL);
      anchorStartSettled.add(sessionId);
    } catch {
      // Loader-level setLabel can throw on a stale runtime while the
      // captured SessionManager still writes; same fallback shape as the
      // hint's appendCustomMessageEntry path. A successful direct append
      // settles; a missing or throwing append stays unsettled and retries
      // later (an optional call on a missing method must NOT read as a
      // successful write).
      try {
        const append = (sm as Partial<SessionManager>).appendLabelChange;
        if (typeof append !== "function") return;
        append.call(sm, leafId, START_ANCHOR_LABEL);
        anchorStartSettled.add(sessionId);
      } catch {
        // Fail-open: never throw into the agent loop.
      }
    }
  }

  // Config (#44 hint + #55 start anchor): load once per session start. Every
  // ctx getter is read BEFORE the first await — `ctx.ui` / `isProjectTrusted`
  // / `cwd` / `hasUI` assert against the live session (pi-steering
  // precedent), and `loadTreeNavigatorConfig` is async. The settled set is
  // deliberately NOT reset: it is per-session state keyed by session id.
  pi.on("session_start", async (_event, ctx) => {
    const cwd = ctx.cwd;
    const projectTrusted = ctx.isProjectTrusted();
    const hasUI = ctx.hasUI;
    const ui = ctx.ui;
    const sm = ctx.sessionManager as SessionManager;
    const sessionId = sm.getSessionId();
    let model: OverflowModel | undefined;
    try {
      model = ctx.model;
    } catch {
      model = undefined;
    }
    const agentDir = getAgentDir();
    // pi's settings as this session reads them (#101, S4).
    piSettingsSource = { cwd, agentDir, projectTrusted };
    // A committing ending pi dropped is re-applied with its own reason.
    settleCommittingEnding(sm, sessionId);
    pendingRewinds.delete(sessionId);
    forcedRewinds.delete(sessionId);
    // O8 (#90): a rewind that died mid-flight (restart, crash) left its
    // instruction in context with nothing waiting for it — end it.
    resolveOrphanRewinds(sm);
    rewindHintReserveTokens = null;
    hintByModel = new Map();
    hintTracker = new RewindHintTracker();
    toolActive = true;
    let reserve: number | null = null;
    let byModel: ReadonlyMap<string, number | null> = new Map();
    let startTurns = START_ANCHOR_DEFAULT_TURNS;
    let warnings: string[] = [];
    try {
      const loaded = await loadTreeNavigatorConfig({
        agentDir,
        projectPath: join(cwd, CONFIG_DIR_NAME, TREE_NAVIGATOR_CONFIG_FILENAME),
        projectTrusted,
      });
      reserve = loaded.config.rewindHintReserveTokens;
      byModel = loaded.config.rewindHintModelReserveTokens;
      startTurns = loaded.config.anchorStartAfterTurns;
      warnings = loaded.warnings;
    } catch {
      // The loader is fail-closed and should never throw; this guard exists
      // so a loader bug can never break session start. Silent: every loader
      // failure already carries a user-facing warning, and a new string here
      // would only fire on an unreachable path.
      reserve = null;
      byModel = new Map();
      startTurns = START_ANCHOR_DEFAULT_TURNS;
      warnings = [];
    }
    rewindHintReserveTokens = reserve;
    hintByModel = byModel;
    anchorStartTurnsBySession.set(sessionId, startTurns);
    // Fresh leg: the per-session gate re-arms on the next
    // `before_agent_start`; absent = fail-open in the meantime.
    toolActiveBySession.delete(sessionId);
    if (reserve !== null || byModel.size > 0) {
      warnings.push(...reserveOrderWarnings(model));
    }
    if (hasUI) {
      for (const warning of warnings) ui.notify(warning, "warning");
    }
  });

  /**
   * D10 (#101): one warning per model whose hint reserve is below pi's
   * `compaction.reserveTokens` (both have per-model overrides), while pi's
   * compaction is off (on, both features are off and D9 says so).
   */
  function reserveOrderWarnings(model: OverflowModel | undefined): string[] {
    if (!piSettingsSource) return [];
    const warnings: string[] = [];
    try {
      const settings = SettingsManager.create(
        piSettingsSource.cwd,
        piSettingsSource.agentDir,
        { projectTrusted: piSettingsSource.projectTrusted },
      );
      if (settings.getCompactionSettings(model).enabled) return [];
      const keys = new Set<string>();
      if (model) keys.add(`${model.provider}/${model.id}`);
      for (const key of hintByModel.keys()) keys.add(key);
      const overrides = settings.getSettings().compaction?.modelOverrides;
      for (const key of Object.keys(overrides ?? {})) keys.add(key);
      for (const key of keys) {
        const slash = key.indexOf("/");
        if (slash <= 0) continue;
        // Per key: one model's invalid setting does not skip the others.
        try {
          const hint = resolveRewindHintReserve(
            {
              rewindHintReserveTokens,
              rewindHintModelReserveTokens: hintByModel,
            },
            key,
          );
          const piReserve = settings.getCompactionReserveTokens({
            provider: key.slice(0, slash),
            id: key.slice(slash + 1),
          });
          if (hint !== null && hint < piReserve) {
            warnings.push(buildReserveOrderWarning(hint, piReserve, key));
          }
        } catch {
          // Invalid pi setting for this model: pi reports it itself.
        }
      }
    } catch {
      // Invalid pi settings: pi reports them itself.
    }
    return warnings;
  }

  // Public context event: replace the wire messages with the session-tree
  // projection before every LLM call. This is the public-API replacement for
  // the deleted `agent.prepareNextTurnWithContext` per-turn refresh — it
  // fires inside `streamAssistantResponse` (via `Agent.transformContext` →
  // `runner.emitContext`), before EVERY LLM call including the turn right
  // after a mid-loop rewind, so the rewound chain (branch_summary +
  // continuation message) reaches the next API call.
  //
  // While a rewind is pending the request is frozen at its instruction
  // (#90, O2). A pending rewind whose instruction never landed, or whose
  // context was compacted, is ended here, before the request goes out.
  pi.on("context", (_event, ctx) => {
    const sm = ctx.sessionManager as SessionManager;
    let pending: PendingRewind | undefined;
    let sessionId: string | undefined;
    try {
      sessionId = sm.getSessionId();
      pending = livePending(sm, sessionId);
    } catch {
      pending = undefined;
    }
    if (pending && sessionId !== undefined) {
      const reason = undeliveredReason(sm, pending);
      if (reason) {
        endPendingOutsideBoundary(sm, sessionId, "failed", reason);
        pending = undefined;
      }
    }
    let instructionEntryId: string | undefined;
    if (pending) {
      try {
        instructionEntryId = pendingInstructionEntryId(sm.getBranch(), pending);
      } catch {
        instructionEntryId = undefined;
      }
    }
    // #107: hand pi the planner's calibrated post-withhold size, so the forced
    // summary turn is not sized from a usage measured before the withholding.
    // An ended record (D8) is out of force whatever it still holds: the same
    // run must not re-send the size it just gave up on.
    const forced =
      sessionId === undefined ? undefined : forcedRewinds.get(sessionId);
    return {
      messages: buildContextMessages(
        sm,
        instructionEntryId,
        forced && !forced.ended && forced.projectedTokens !== undefined
          ? {
              tokens: forced.projectedTokens,
              entryId: forced.overrideEntryId,
            }
          : undefined,
      ),
    };
  });

  // Anchor mandate (#31): a before_agent_start append lands at the END of
  // the system prompt — stronger than the mid-prompt Guidelines block, and
  // re-applied on every prompt instead of living in the conversation. Skip
  // only when the tool is verifiably absent from the active set; fail-open
  // when `selectedTools` is undefined (this extension always registers it).
  // The same check gates the rewind hint (#44): an inactive tool must not
  // fire `turn_end` nudges the model can't act on.
  pi.on("before_agent_start", async (event, ctx) => {
    const selected = event.systemPromptOptions?.selectedTools;
    toolActive = !Array.isArray(selected) || selected.includes(TOOL_NAME);
    // Per-session copy of the gate (#55): the shared `toolActive` above
    // drives the hint, but an in-process sibling session must not clear this
    // session's auto-anchor gate.
    try {
      const sessionId = ctx.sessionManager.getSessionId();
      toolActiveBySession.set(sessionId, toolActive);
      // A new prompt is not a retry (pi retries via `agent.continue()`,
      // which skips this event): a rewind still waiting is aborted (#90, O6).
      // pi 1.3.0 also fires this for an extension `sendMessage` run.
      endPendingOutsideBoundary(
        ctx.sessionManager as SessionManager,
        sessionId,
        "aborted",
        "a new prompt started",
      );
    } catch {
      // Fail-open: the per-session gate stays absent.
    }
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
    // Retire the pi-codemode bookkeeping of this call (#103).
    nestedRewindCallers.delete(event.toolCallId);
    nestedCallCounts.delete(event.toolCallId);
    if (event.toolName !== TOOL_NAME) return undefined;
    const details = event.details as { refusal?: boolean } | undefined;
    if (details?.refusal === true) {
      if (event.parentToolCallId)
        refusedNestedCallers.add(event.parentToolCallId);
      return { isError: true };
    }
    return undefined;
  });

  // Rewind hint (#44): one check per turn, after the turn's tool results and
  // before the agent loop's steering drain — so a `pi.sendMessage` from this
  // handler is delivered in the same run. Synchronous by design: no await
  // before any ctx use. Fires once per reserve crossing; see
  // `RewindHintTracker` for the re-arm semantics.
  pi.on("turn_end", (event, ctx) => {
    // Single `sessionManager` capture shared by the auto start-anchor and
    // the rewind hint: the stale-ctx contract is exactly one read, before
    // any throw-capable call. A throwing getter (replaced session) skips
    // both paths without throwing into the loop.
    let sm: typeof ctx.sessionManager | null;
    try {
      sm = ctx.sessionManager;
    } catch {
      sm = null;
    }
    // Auto start anchor (#55): fires at the X-th turn of the run. Runs
    // BEFORE and independently of the hint block — the two fields have
    // separate config semantics (the hint can be off/null while the start
    // anchor still applies). Native per-run `turnIndex` is 0-based, hence
    // the pre-increment. (The turn's entry ids, `messageEntryId` /
    // `toolResultEntryIds`, exist since pi 0.87; the rewind handler reads
    // them, this one needs only `turnIndex` and `message`.) A turn a rewind
    // owns is skipped (#90): an attempt may be hidden, and the summary turn
    // carries the `branch:` completion label.
    const rewindOwned =
      !!event.message &&
      typeof event.message === "object" &&
      rewindTurns.has(event.message);
    // An overflowed turn is never anchored (#101, D14): with pi compaction on,
    // rewind-driven compaction stands down and does not mark it owned.
    const overflowedTurn =
      !!event.message &&
      typeof event.message === "object" &&
      overflowedTurns.has(event.message);
    if (sm && !rewindOwned && !overflowedTurn) {
      writeStartAnchorIfNeeded(sm as SessionManager, event.turnIndex + 1);
    }
    // Per-session hint gate (#58): an in-process sibling session's
    // `before_agent_start` (e.g. a subagent whose tool subset omits
    // `navigate_tree`) must not clear THIS session's hint. Mirrors the auto
    // start-anchor's per-session gate; the shared `toolActive` remains the
    // fallback for sessions with no recorded gate (fail-open).
    let hintToolActive = toolActive;
    if (sm) {
      try {
        const gated = toolActiveBySession.get(sm.getSessionId());
        if (gated !== undefined) hintToolActive = gated;
      } catch {
        // Stale ctx: keep the shared fallback.
      }
    }
    // Hint off everywhere: nothing can fire or re-arm.
    if (
      !hintToolActive ||
      (rewindHintReserveTokens === null && hintByModel.size === 0)
    ) {
      return;
    }
    let modelKey: string | undefined;
    if (hintByModel.size > 0) {
      try {
        const model = ctx.model;
        if (model) modelKey = `${model.provider}/${model.id}`;
      } catch {
        // Stale ctx: no hint this turn.
        return;
      }
    }
    const reserve = resolveRewindHintReserve(
      {
        rewindHintReserveTokens,
        rewindHintModelReserveTokens: hintByModel,
      },
      modelKey,
    );
    // A rewind owns this turn (its call, a summary attempt, or the
    // completion): a rewind hint would be stale or push a second
    // rewind. The crossing stays unspent for the next turn.
    if (
      event.message &&
      typeof event.message === "object" &&
      rewindTurns.has(event.message)
    ) {
      return;
    }
    const usage = ctx.getContextUsage();
    // `tokens == null` / `percent == null` is pi's post-compaction state
    // (unknown tokens), and `contextWindow <= 0` has no meaningful bar to
    // compare against. All are no-ops that leave the crossing unspent.
    if (
      !usage ||
      usage.tokens == null ||
      usage.percent == null ||
      usage.contextWindow <= 0
    ) {
      return;
    }
    const tokensLeft = usage.contextWindow - usage.tokens;
    if (reserve === null || tokensLeft > reserve) {
      // Hint off for this model, or outside the reserve: re-arm only.
      hintTracker.observe(tokensLeft, reserve);
      return;
    }
    // Spent: skip the anchor walk below (it estimates every anchor).
    if (hintTracker.isSpent()) return;
    // Collect anchors BEFORE marking the crossing spent: a throwing
    // `getBranch` / `getLabel` / `getEntries` must leave the crossing
    // unspent so the next turn can retry — never burn the one shot on a
    // failed state read. The shared `sm` was captured before any
    // throw-capable call, so the fallback append never re-reads a lazy ctx
    // getter after a throw.
    if (!sm) return;
    let anchors: { names: string[]; text: string };
    try {
      const allEntries = sm.getEntries();
      const byId = new Map<string, SessionEntry>();
      for (const e of allEntries) byId.set(e.id, e);
      anchors = formatActiveAnchors(sm, usage.contextWindow, allEntries, byId);
    } catch {
      return;
    }
    if (!hintTracker.observe(tokensLeft, reserve)) return;
    // Either-or with pi's compaction (#101, S3/D9), read at the crossing
    // (between crossings the hint has no effect, so no per-turn read): on,
    // the hint and rewind-driven compaction are off, and the crossing shows
    // why (TUI only; headless stays silent). The crossing is spent.
    // (`compaction.enabled` has no per-model override: no `ctx.model` read.)
    if (piCompaction(sm as SessionManager, undefined)?.enabled) {
      if (ctx.hasUI) ctx.ui.notify(REWIND_COMPACTION_OFF_TEXT, "error");
      return;
    }
    if (anchors.names.length > 0) {
      const text = buildRewindHintText(
        usage.percent,
        tokensLeft,
        reserve,
        usage.contextWindow,
        anchors.text,
      );
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

  // Auto start anchor (#55) run-end fallback: a run that finishes before its
  // X-th turn still settles (a one-turn first run anchors at turn 1).
  // `agent_end` fires on complete / aborted / errored runs; the settled set
  // makes this a no-op once the turn trigger already wrote.
  pi.on("agent_end", (event, ctx) => {
    let sm: typeof ctx.sessionManager;
    try {
      sm = ctx.sessionManager;
      // A run that ended on an overflow (#101, D14): its leaf is at the
      // context limit, a useless rewind target.
      const last = (event.messages ?? []).findLast(
        (m) => m.role === "assistant",
      );
      if (last && overflowed(last, ctx.model)) return;
    } catch {
      return;
    }
    // A pending rewind is NOT dropped here (#90): `agent_end` fires before
    // pi's auto-retry and `agent_before_settle`, which end it. While one is
    // pending the leaf may be an attempt about to be hidden: no start anchor.
    try {
      if (livePending(sm as SessionManager, sm.getSessionId())) return;
    } catch {
      return;
    }
    writeStartAnchorIfNeeded(sm as SessionManager, Number.POSITIVE_INFINITY);
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

\`name\` (anchor) and \`newLabel\` (rewind) write into one shared anchor namespace: labels are unique on the active branch — both actions refuse a name that already exists — and everything written there is addressable as a future \`rewindTo\`. Avoid the reserved \`${LABEL_PREFIX}\` prefix.`,
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
    execute: async (toolCallId, params, _signal, _onUpdate, ctx) => {
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

        const rows: AnchorTableRow[] = [];
        for (const e of path) {
          const lbl = sm.getLabel(e.id);
          if (lbl?.startsWith(LABEL_PREFIX)) {
            const name = lbl.slice(LABEL_PREFIX.length);
            const tokensAt = estimateAtEntry(allEntries, e.id, byId);
            const hint = findLabelHint(sm, e.id, LIST_HINT_MAX_LENGTH);
            rows.push({
              pct: formatPct1(tokensAt, cw),
              name,
              ...(hint ? { after: hint } : {}),
            });
          }
        }

        const reflectionWarning = reflectionOk
          ? ""
          : ` · ${REFLECTION_BOOTSTRAP_WARNING_LIST}`;
        const header = `[list] · ${rows.length} label${rows.length === 1 ? "" : "s"} · ctx ${formatPct1(totalTokens, cw)}${cw > 0 ? ` of ${formatWindow(cw)}` : ""}${reflectionWarning}`;
        const body = rows.length
          ? formatAnchorTable(rows, cw)
          : "No labels on the active branch.";
        return {
          content: [{ type: "text", text: `${header}\n\n${body}` }],
          details: {
            count: rows.length,
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
        // Labels are unique on the active branch (issue #55, C8): an
        // existing label is never moved or overwritten, and the caller
        // must pick a fresh name. Manual `/tree` duplicates are the only
        // duplicate path; move-on-collision is retired for both writers.
        const fullLabel = LABEL_PREFIX + p.name;
        if (findLabeledEntry(sm, fullLabel)) {
          return toolError(
            `A label '${p.name}' already exists on the active branch — labels are unique and immutable once written. Use action='list' to review the active labels, then pick a fresh name.`,
          );
        }
        // Atomic anchor: compute the estimate BEFORE writing the label. A
        // throw from the estimator must not leave a label on disk —
        // otherwise the agent sees an error yet a retry with the same
        // name is rejected as a duplicate.
        const cw = ctx.model?.contextWindow ?? 0;
        const tokensHere = estimateActiveBranchTokens(sm);
        const labelHint = findLabelHint(sm, leafId, ANCHOR_HINT_MAX_LENGTH);
        const positionLine = `${formatPct1(tokensHere, cw)}${cw > 0 ? ` of ${formatWindow(cw)}` : ""}`;
        const hintLine = labelHint ? ` (after: “${labelHint}”)` : "";
        pi.setLabel(leafId, fullLabel);
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
          },
        };
      }

      // --- rewind ---
      // Solo-call guard (#37, restored by #90 S6): a rewind sharing its
      // assistant message with other calls is refused before anything is
      // recorded. pi runs the batch in order (`executionMode: sequential`):
      // calls before it already ran, calls after it run normally since
      // nothing is pending. Detection fails open (`null`).
      // A rewind from a codemode script is keyed on the codemode call (#103):
      // that call is the one in the assistant message and the transcript.
      // A nested rewind is solo too (#103, D): a sibling nested call of the
      // same script, issued before or in flight, refuses it like a batch. The
      // siblings run normally, as with a direct batch.
      const nestedCaller = nestedRewindCallers.get(toolCallId);
      const callerId = nestedCaller ?? toolCallId;
      const batch = findInFlightAssistantToolCalls(sm, callerId);
      const nestedSiblings =
        nestedCaller !== undefined &&
        (nestedCallCounts.get(nestedCaller) ?? 0) > 1;
      if ((batch && batch.length > 1) || nestedSiblings) {
        return toolError(SOLO_REWIND_REFUSAL, {
          rejected: "batched-rewind",
          batchedToolCalls: batch?.map((call) => call.name) ?? [],
        });
      }
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

      // Label uniqueness (issue #55, C8): a `newLabel` already on the
      // active branch is refused BEFORE the summary turn or any mutation —
      // labels are immutable once written, so the anchor→rewind
      // rename-forward idiom is retired. The check sits with the
      // name/summaryFocus validations so no summary turn is wasted; the
      // existing label stays addressable via `rewindTo`.
      if (findLabeledEntry(sm, LABEL_PREFIX + p.newLabel)) {
        return toolError(
          `A label '${p.newLabel}' already exists on the active branch — labels are unique and immutable once written, so rewind requires a fresh \`newLabel\`. The existing '${p.newLabel}' label stays addressable via \`rewindTo\`. Use action='list' to review the active labels, then pick a fresh newLabel.`,
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

      // Token math (#21): pure SessionManager reads, so the min-savings
      // floor below can reject before a pending rewind is recorded.
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
      // Minimum-savings floor (#21): refuse rewinds whose apparent savings
      // is below MIN_REWIND_SAVINGS_TOKENS — collapsing a near-empty segment
      // burns a summary turn, forks a junk branch_summary, consumes a
      // milestone label, and grows live context.
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

      // `{first}` numbers the collapsed segment's first message the way the
      // summary turn sees the conversation: the context-edit-aware
      // projection the `context` handler sends on every LLM call (system
      // messages and provider-invisible assistants excluded). A target the
      // projection no longer carries (older than the latest compaction cut)
      // means every visible message is collapsed evidence.
      let first = 1;
      try {
        const projection = buildSessionProjection(allEntries, oldLeaf, byId);
        if (projection.entries.some((e) => e.sourceEntry.id === target)) {
          first = countBackgroundMessages(projection.entries, target) + 1;
        }
      } catch {
        first = 1;
      }

      let sessionId: string;
      try {
        sessionId = sm.getSessionId();
      } catch (err) {
        return toolError(
          `Could not read the session id: ${err instanceof Error ? err.message : String(err)}`,
        );
      }

      // In-band summary (#80, #90): no separate summarization request. This
      // tool result only reports "accepted"; the rewind turn's `turn_end`
      // returns the instruction as a custom message, so the live model writes
      // the summary as its next turn — an append to the live conversation,
      // served from the prompt cache. Until it lands every tool call is
      // blocked.
      pendingRewinds.set(sessionId, {
        toolCallId: callerId,
        rewindEntryId: oldLeaf,
        target,
        rewindTo: p.rewindTo,
        newLabel: p.newLabel,
        beforeTokens,
        contextWindow,
        first,
        summaryFocus: p.summaryFocus,
        tokensAtTarget,
        hiddenEntryIds: [],
        attempts: 0,
      });

      return {
        content: [
          {
            type: "text",
            text: buildRewindAcceptedText(p.rewindTo, p.newLabel),
          },
        ],
        details: {
          status: "summary-requested",
          rewindTo: p.rewindTo,
          newLabel: p.newLabel,
          targetId: target,
          first,
          contextBefore: beforeTokens,
          tokensAtTarget,
          contextWindow,
        },
      };
    },
  });
}

/**
 * A forced rewind's record (#101, rewind-driven compaction): the attempts
 * spent, whether a final error (D8) ended it, and the plan's calibrated
 * post-withhold size with the entry it replaces (#107).
 */
interface ForcedRewind {
  attempts: number;
  ended?: boolean;
  projectedTokens?: number;
  /** The entry the projected size replaces (#107). */
  overrideEntryId?: string;
}

/**
 * The live `forcedRewinds` map of each extension instance, for
 * `__testHooks.forcedRecord`: a D8 cleanup is reported by no event, so a
 * test reads the record itself. Weak, so it changes no lifetime.
 */
const forcedRewindsByExtension = new WeakMap<
  object,
  Map<string, ForcedRewind>
>();

/**
 * Non-stable testing-only hooks. **Do NOT import in production code.**
 *
 * The `__` prefix and individual member names are subject to change in any
 * release without a semver-major bump. Intended exclusively for hermetic
 * tests within this package; the hooks reach into module-internal state
 * (the `AgentSession.prototype` patch, the `seenSessions` WeakSet, the
 * `sessionInstances` array) and are
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
  findLabelHint,
  findLabeledEntry,
  buildContextMessages,
  refreshAgentMessages,
  captureSession,
  countBackgroundMessages,
  findInFlightAssistantToolCalls,
  planWithhold,
  /**
   * A copy of one extension instance's forced-rewind record (#107). The D8
   * cleanup reports nothing an event can see, so the test reads it here.
   * `extension` is the `ExtensionAPI` the factory was called with.
   */
  forcedRecord(
    extension: unknown,
    sessionId: string,
  ): ForcedRewind | undefined {
    const record = forcedRewindsByExtension
      .get(extension as object)
      ?.get(sessionId);
    return record ? { ...record } : undefined;
  },
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
