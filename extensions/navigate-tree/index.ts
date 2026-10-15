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
 *     call, its instruction result, and the model's summary turn; the
 *     active branch continues from the `branch_summary` with the
 *     `navigate-tree-rewind` continuation message.
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
 * Verified against pi 1.2.0 via `scripts/pi-upstream-probe.mjs`.
 * In-loop context refresh runs through the public `context` extension
 * event (see `buildContextMessages`), not `agent.prepareNextTurn*`
 * reflection. The rewind summary is written in-band by the live model
 * (#80): no out-of-loop summarization request exists.
 */

import { join } from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import {
  AgentSession,
  buildSessionContext,
  buildSessionProjection,
  CONFIG_DIR_NAME,
  collectEntriesForBranchSummary,
  convertToLlm,
  type ExtensionAPI,
  getAgentDir,
  prepareBranchEntries,
  type SessionEntry,
  type SessionManager,
  sessionEntryToContextMessages,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import {
  loadTreeNavigatorConfig,
  resolveRewindHintReserve,
  START_ANCHOR_DEFAULT_TURNS,
  TREE_NAVIGATOR_CONFIG_FILENAME,
} from "./config.ts";
import { estimateContextTokens } from "./context-tokens.ts";
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
  REWIND_HINT_CUSTOM_TYPE,
  RewindHintTracker,
} from "./rewind-hint.ts";
import {
  buildStoredSummary,
  buildSummaryInstruction,
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
// Summary turns a pending rewind tolerates before it is cancelled (#80). A
// summary turn fails when it calls a tool (blocked while the rewind is
// pending) or carries no `## Goal` summary; each failure re-asks. Failed
// attempts sit on the segment the rewind collapses, so a retry costs one
// cached turn and nothing after the rewind lands.
export const MAX_SUMMARY_ATTEMPTS = 3;
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
export function buildContextMessages(sm: {
  buildContextEntries(): SessionEntry[];
  buildSessionProjection?(): { messages: AgentMessage[] };
}): ReturnType<typeof sessionEntryToContextMessages> {
  let messages: AgentMessage[] | undefined;
  try {
    if (typeof sm.buildSessionProjection === "function") {
      messages = sm.buildSessionProjection().messages;
    }
  } catch {
    // Fall through to the raw entry projection — pi rewrites the context
    // handlers' output at the wire boundary anyway.
  }
  messages ??= sm
    .buildContextEntries()
    .flatMap((entry) => sessionEntryToContextMessages(entry));
  return messages.filter((message) => message.role !== "system");
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
 * — with its cumulative context percentage, rendered as
 * `'a' 4.1% · 'b' 22.0% of 1.0M`. Shared by the min-savings rejection and
 * the rewind hint so both name the same targets the same way. A manual
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
  const parts: string[] = [];
  for (const e of sm.getBranch()) {
    const lbl = sm.getLabel(e.id);
    if (!lbl?.startsWith(LABEL_PREFIX)) continue;
    const name = lbl.slice(LABEL_PREFIX.length);
    if (!isValidName(name)) continue;
    names.push(name);
    parts.push(
      `'${name}' ${formatPct1(estimateAtEntry(allEntries, e.id, byId), cw)}`,
    );
  }
  // cw=0 → percents fall back to formatPct1's k-format and the window
  // suffix is omitted, mirroring `list`'s header behavior.
  const suffix = cw > 0 ? ` of ${formatWindow(cw)}` : "";
  return { names, text: `${parts.join(" · ")}${suffix}` };
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
  return `Rewinding to '${rewindTo}' (as '${newLabel}') would free only ~${formatPct1(apparentSavings, 0)} tokens. ${guidance} Active anchors: ${anchors.text}`;
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

/** A rewind waiting for its in-band summary turn (#80). */
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
  /** Failed summary turns so far (tool call / no `## Goal` / truncated). */
  attempts: number;
  /** The latest summary turn ended in a provider error (auto-retry pending). */
  lastTurnErrored?: boolean;
}

/** customType of the rewind continuation / retry / cancel messages (#80). */
export const REWIND_CUSTOM_TYPE = "navigate-tree-rewind";

export const SUMMARY_PENDING_BLOCK_REASON = `${TOOL_NAME}: a rewind is waiting for its summary — tool calls are blocked until it lands. Your entire next reply must be the summary in the format the rewind result requested, starting with '## Goal' — no acknowledgement, no tool calls.`;

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
    ? "the context was compacted while the summary was being written"
    : null;
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

export function buildSummaryRetryText(rewindTo: string): string {
  return `[${TOOL_NAME}] The rewind to '${rewindTo}' is still waiting for its summary. Your entire reply must be the summary in the format the rewind result requested, starting with '## Goal' — no acknowledgement, no tool calls.`;
}

export function buildRewindCancelledText(
  rewindTo: string,
  reason: string,
): string {
  return `[${TOOL_NAME}] Rewind to '${rewindTo}' cancelled: ${reason}. Nothing was collapsed — continue the work and rewind again later.`;
}

/** One session's rewind-hint config and crossing (#44, #85, #86). */
interface HintState {
  rewindHintReserveTokens: number | null;
  rewindHintModelReserveTokens: ReadonlyMap<string, number | null>;
  tracker: RewindHintTracker;
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
  // tool call, completed (or cancelled) by the summary turn's `turn_end`.
  const pendingRewinds = new Map<string, PendingRewind>();

  // Turns the rewind handler owned (the rewind call's turn, summary
  // attempts, the completing turn): the hint handler skips them, so a stale
  // rewind hint never lands on a rewind in progress.
  const rewindTurns = new WeakSet<object>();

  function cancelRewind(
    sessionId: string,
    pending: PendingRewind,
    reason: string,
  ): RewindMessageDraft {
    pendingRewinds.delete(sessionId);
    return rewindMessageDraft(
      buildRewindCancelledText(pending.rewindTo, reason),
      {
        status: "cancelled",
        rewindTo: pending.rewindTo,
        newLabel: pending.newLabel,
        reason,
        summaryAttempts: pending.attempts,
      },
    );
  }

  // Phase 2 of an in-band rewind (#80): branch from the anchor with the
  // summary the live model just wrote. Nothing is in flight — the summary
  // turn has ended — so the rewind tool call, its instruction result and the
  // summary all stay on the abandoned branch and no synthetic tool_use is
  // needed to keep the new branch structurally valid.
  async function completeRewind(
    sm: SessionManager,
    sessionId: string,
    pending: PendingRewind,
    summaryText: string,
  ): Promise<RewindMessageDraft> {
    // A compaction or tree move during the summary turn invalidates the
    // rewind: the anchor must still be the same labeled entry on the active
    // branch, and the new label must still be free.
    if (
      findLabeledEntry(sm, LABEL_PREFIX + pending.rewindTo) !== pending.target
    ) {
      return cancelRewind(
        sessionId,
        pending,
        `'${pending.rewindTo}' is no longer on the active branch`,
      );
    }
    if (findLabeledEntry(sm, LABEL_PREFIX + pending.newLabel)) {
      return cancelRewind(
        sessionId,
        pending,
        `a label '${pending.newLabel}' now exists on the active branch`,
      );
    }
    const staleReason = rewindStaleReason(sm, pending.rewindEntryId);
    if (staleReason) {
      return cancelRewind(sessionId, pending, staleReason);
    }
    const oldLeafId = sm.getLeafId();
    // Input that arrived after the rewind call (user steers drained at the
    // rewind turn's boundary, other extensions' messages) sits on the segment
    // about to be abandoned. It is new input, not collapsed work: it is
    // re-appended onto the new branch below, in order, ahead of the
    // continuation (which flags it as unanswered).
    const lateEntries = collectLateEntries(sm, pending.rewindEntryId);
    let entries: SessionEntry[];
    let readFiles: string[];
    let modifiedFiles: string[];
    let summaryId: string;
    try {
      ({ entries } = collectEntriesForBranchSummary(
        sm,
        oldLeafId,
        pending.target,
      ));
      const { fileOps } = prepareBranchEntries(
        entries,
        Number.MAX_SAFE_INTEGER,
      );
      ({ readFiles, modifiedFiles } = computeFileLists(fileOps));
      // `fromHook: true` pairs with the `session_tree` payload's
      // `fromExtension: true`, as pi's own navigateTree does for an
      // extension-provided summary.
      summaryId = sm.branchWithSummary(
        pending.target,
        buildStoredSummary(summaryText, readFiles, modifiedFiles),
        { readFiles, modifiedFiles },
        true,
      );
    } catch (err) {
      // Nothing branched: tell the model, or a text-only summary turn would
      // end the run with no word of the failed rewind.
      return cancelRewind(
        sessionId,
        pending,
        `the branch could not be written (${err instanceof Error ? err.message : String(err)})`,
      );
    }
    pendingRewinds.delete(sessionId);
    const fullLabel = LABEL_PREFIX + pending.newLabel;
    try {
      pi.setLabel(summaryId, fullLabel);
    } catch {
      // Stale runtime: the captured SessionManager still writes.
      try {
        sm.appendLabelChange(summaryId, fullLabel);
      } catch {
        // The summary landed unlabeled; `list` simply won't show it.
      }
    }
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
    return rewindMessageDraft(
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
    );
  }

  // While a rewind waits for its summary, every tool call is blocked: the
  // next turn must be the summary itself. The blocked result re-asks.
  pi.on("tool_call", (_event, ctx) => {
    let sessionId: string;
    try {
      sessionId = ctx.sessionManager.getSessionId();
    } catch {
      return undefined;
    }
    if (!pendingRewinds.has(sessionId)) return undefined;
    return { block: true, reason: SUMMARY_PENDING_BLOCK_REASON };
  });

  // Summary turn (#80). Registered before the hint / start-anchor handler so
  // a completed rewind is already on the branch when that handler reads the
  // context usage.
  pi.on("turn_end", async (event, ctx) => {
    let sm: SessionManager;
    let sessionId: string;
    try {
      sm = ctx.sessionManager as SessionManager;
      sessionId = sm.getSessionId();
    } catch {
      return;
    }
    const pending = pendingRewinds.get(sessionId);
    if (!pending) return;
    if (event.message && typeof event.message === "object") {
      rewindTurns.add(event.message);
    }
    // Rewind messages are returned as boundary entries: pi commits them right
    // after this handler, BEFORE it drains queued steers, and `continue`
    // guarantees one more provider request — so the continuation reaches a
    // real turn in the same run, ahead of anything the user typed meanwhile.
    // Other handlers' entries are kept (pi replaces the list per handler).
    const deliver = (draft: RewindMessageDraft) => ({
      entries: [...keepValidDrafts(sm, event.entries ?? []), draft],
      continue: true,
    });
    const message = event.message as {
      role?: string;
      content?: unknown;
      stopReason?: string;
    };
    if (message?.role !== "assistant") return;
    const content = Array.isArray(message.content)
      ? (message.content as Array<{
          type?: string;
          id?: string;
          text?: string;
        }>)
      : [];
    // The turn that issued the rewind ends after its own tool result; the
    // summary is the NEXT turn.
    if (
      content.some((b) => b?.type === "toolCall" && b.id === pending.toolCallId)
    ) {
      return;
    }
    if (message.stopReason === "aborted") {
      // The user's abort wins: drop the rewind without a model message.
      pendingRewinds.delete(sessionId);
      return;
    }
    // Provider errors go through pi's auto-retry, which emits `agent_end`
    // BEFORE it re-runs the turn (`agent.continue()`): mark the rewind so
    // that `agent_end` keeps it. A new user prompt (`before_agent_start`)
    // drops it if the retry never comes.
    if (message.stopReason === "error") {
      pending.lastTurnErrored = true;
      return;
    }
    pending.lastTurnErrored = false;
    const hasToolCall = content.some((b) => b?.type === "toolCall");
    const summary = extractSummaryText(
      content
        .filter((b) => b?.type === "text" && typeof b.text === "string")
        .map((b) => b.text)
        .join("\n"),
    );
    if (hasToolCall || summary === null || message.stopReason === "length") {
      pending.attempts++;
      if (pending.attempts >= MAX_SUMMARY_ATTEMPTS) {
        return deliver(
          cancelRewind(
            sessionId,
            pending,
            `no summary after ${pending.attempts} attempts`,
          ),
        );
      }
      if (!hasToolCall) {
        // A text-only turn ends the loop; blocked tool calls already re-ask.
        return deliver(
          rewindMessageDraft(buildSummaryRetryText(pending.rewindTo), {
            status: "summary-retry",
            rewindTo: pending.rewindTo,
            newLabel: pending.newLabel,
            summaryAttempts: pending.attempts,
          }),
        );
      }
      return;
    }
    try {
      return deliver(await completeRewind(sm, sessionId, pending, summary));
    } catch (err) {
      // A throw before the branch write (the stale checks read the session)
      // leaves the rewind pending: cancel it so the model hears about it.
      // Every step after the write is guarded, so the branch never lands
      // without its continuation.
      return deliver(
        cancelRewind(
          sessionId,
          pending,
          `the rewind could not be completed (${err instanceof Error ? err.message : String(err)})`,
        ),
      );
    }
  });

  // Rewind-hint state (#44, #85), keyed by session id and replaced on that
  // session's `session_start`, so `/new` / `/resume` / `/reload` sessions
  // never inherit a spent crossing, and a sibling session served by the
  // same extension instance never resets, reconfigures, or spends another
  // session's hint. Hardening, like the other per-session maps: pi-subagents
  // currently builds a fresh extension instance per child. A session with
  // no entry (no `session_start` seen) gets no hint.
  // `byModel` (#86) holds the `modelOverrides` reserves; the reserve is
  // resolved against the live `ctx.model` on every `turn_end`, so a model
  // switch mid-session takes effect on the next turn.
  const hintBySession = new Map<string, HintState>();

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
    const sessionId = ctx.sessionManager.getSessionId();
    pendingRewinds.delete(sessionId);
    let reserve: number | null = null;
    let byModel: ReadonlyMap<string, number | null> = new Map();
    let startTurns = START_ANCHOR_DEFAULT_TURNS;
    let warnings: string[] = [];
    try {
      const loaded = await loadTreeNavigatorConfig({
        agentDir: getAgentDir(),
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
    hintBySession.set(sessionId, {
      rewindHintReserveTokens: reserve,
      rewindHintModelReserveTokens: byModel,
      tracker: new RewindHintTracker(),
    });
    anchorStartTurnsBySession.set(sessionId, startTurns);
    // Fresh leg: the per-session gate re-arms on the next
    // `before_agent_start`; absent = fail-open in the meantime.
    toolActiveBySession.delete(sessionId);
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
  // continuation message) reaches the next API call.
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
  pi.on("before_agent_start", async (event, ctx) => {
    const selected = event.systemPromptOptions?.selectedTools;
    const toolActive = !Array.isArray(selected) || selected.includes(TOOL_NAME);
    // Per-session gate (#55, #58): an in-process sibling session must not
    // clear this session's hint or auto-anchor gate.
    try {
      const sessionId = ctx.sessionManager.getSessionId();
      toolActiveBySession.set(sessionId, toolActive);
      // A new prompt is not a retry (pi retries via `agent.continue()`,
      // which skips this event): a rewind left waiting across an errored
      // run's end is dropped.
      pendingRewinds.delete(sessionId);
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
    if (event.toolName !== TOOL_NAME) return undefined;
    const details = event.details as { refusal?: boolean } | undefined;
    return details?.refusal === true ? { isError: true } : undefined;
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
    // the pre-increment. This is the only TurnEndEvent field read here: it
    // exists on both pi 0.84.2 (dev-dep) and 0.99.2 (host); host-only
    // fields (`messageEntryId` / `outcome`) must never be read.
    if (sm) writeStartAnchorIfNeeded(sm as SessionManager, event.turnIndex + 1);
    // Per-session hint gate (#58): an in-process sibling session's
    // `before_agent_start` (e.g. a subagent whose tool subset omits
    // `navigate_tree`) must not clear THIS session's hint. Mirrors the auto
    // start-anchor's per-session gate; a session with no recorded gate
    // fails open.
    let hintToolActive = true;
    let hint: HintState | undefined;
    if (sm) {
      try {
        const sessionId = sm.getSessionId();
        hintToolActive = toolActiveBySession.get(sessionId) ?? true;
        hint = hintBySession.get(sessionId);
      } catch {
        // Stale ctx: no hint state.
      }
    }
    if (!hintToolActive || !hint) return;
    // Hint off everywhere: nothing can fire or re-arm.
    if (
      hint.rewindHintReserveTokens === null &&
      hint.rewindHintModelReserveTokens.size === 0
    ) {
      return;
    }
    let modelKey: string | undefined;
    if (hint.rewindHintModelReserveTokens.size > 0) {
      try {
        const model = ctx.model;
        if (
          typeof model?.provider === "string" &&
          typeof model.id === "string"
        ) {
          modelKey = `${model.provider}/${model.id}`;
        }
      } catch {
        // Stale ctx: no hint this turn.
        return;
      }
    }
    const rewindHintReserveTokens = resolveRewindHintReserve(hint, modelKey);
    const hintTracker = hint.tracker;
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
    if (
      rewindHintReserveTokens === null ||
      tokensLeft > rewindHintReserveTokens
    ) {
      // Hint off for this model, or outside the reserve: re-arm only.
      hintTracker.observe(tokensLeft, rewindHintReserveTokens);
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
    if (!hintTracker.observe(tokensLeft, rewindHintReserveTokens)) return;
    if (anchors.names.length > 0) {
      const text = buildRewindHintText(
        usage.percent,
        tokensLeft,
        rewindHintReserveTokens,
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
  pi.on("agent_end", (_event, ctx) => {
    let sm: typeof ctx.sessionManager;
    try {
      sm = ctx.sessionManager;
    } catch {
      return;
    }
    writeStartAnchorIfNeeded(sm as SessionManager, Number.POSITIVE_INFINITY);
    // A run that ends with a rewind still waiting for its summary drops it
    // — unless the last turn errored: pi emits `agent_end` before its
    // auto-retry re-runs the turn, so that rewind waits for the retry.
    try {
      const sessionId = sm.getSessionId();
      if (!pendingRewinds.get(sessionId)?.lastTurnErrored) {
        pendingRewinds.delete(sessionId);
      }
    } catch {
      // Stale ctx: nothing to drop.
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

      // In-band summary (#80): no separate summarization request. This tool
      // result IS the summary instruction, so the live model writes the
      // summary as its next turn — an append to the live conversation,
      // served from the prompt cache on every provider. The `turn_end`
      // handler branches from the anchor with that text; until then every
      // tool call is blocked.
      pendingRewinds.set(sessionId, {
        toolCallId,
        rewindEntryId: oldLeaf,
        target,
        rewindTo: p.rewindTo,
        newLabel: p.newLabel,
        beforeTokens,
        contextWindow,
        attempts: 0,
      });

      return {
        content: [
          {
            type: "text",
            text: buildSummaryInstruction(first, p.summaryFocus),
          },
        ],
        details: {
          status: "summary-requested",
          rewindTo: p.rewindTo,
          newLabel: p.newLabel,
          targetId: target,
          first,
          contextBefore: beforeTokens,
          contextWindow,
        },
      };
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
