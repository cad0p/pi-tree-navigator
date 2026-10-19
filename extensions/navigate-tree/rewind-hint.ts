/**
 * rewind-hint — context-pressure nudge for the navigate-tree extension.
 *
 * Pure state + copy builders. `RewindHintTracker` fires exactly once per
 * reserve crossing and re-arms only when the tokens left climb back above
 * the reserve (a rewind or compaction). The builders render the two
 * user-visible strings (agent hint + TUI-only no-anchor warning).
 *
 * No pi runtime imports — the tracker and builders take plain values; the
 * caller renders the active-anchor list (it needs the session's token
 * estimator).
 */

import { formatWindow, LABEL_PREFIX, TOOL_NAME } from "./helpers.ts";

export const REWIND_HINT_CUSTOM_TYPE = "navigate-tree-rewind-hint";

/**
 * One-shot crossing tracker. `observe` returns true exactly on a fresh
 * crossing (`tokensLeft <= reserve` while unspent); an observation with
 * more than `reserve` tokens left re-arms, and a spent tracker stays silent
 * while still inside the reserve. A `null` reserve (hint off for the
 * active model, #86) never fires, and always re-arms.
 */
export class RewindHintTracker {
  private spent = false;

  observe(tokensLeft: number, reserve: number | null): boolean {
    if (reserve === null || tokensLeft > reserve) {
      this.spent = false;
      return false;
    }
    if (this.spent) return false;
    this.spent = true;
    return true;
  }

  /** True while the current crossing's one hint has been sent. */
  isSpent(): boolean {
    return this.spent;
  }

  /**
   * Spend the current crossing without a hint (#101, D17): a forced rewind
   * is already asking for one. `observe` re-arms as usual.
   */
  markSpent(): void {
    this.spent = true;
  }

  /** Re-arm for a fresh session (`session_start`). */
  reset(): void {
    this.spent = false;
  }
}

/** `12.3k` / `1.0M`; negative (an overrun window) clamps to `0.0k`. */
function formatTokensK(tokens: number): string {
  const t = Math.max(0, tokens);
  // 999_950+ would round to `1000.0k`.
  if (t >= 999_950) return `${(t / 1_000_000).toFixed(1)}M`;
  return `${(t / 1000).toFixed(1)}k`;
}

/**
 * The agent-facing hint (#84). Dynamic trigger + ask, plus the active anchors
 * (`anchorTable`, pre-rendered by the caller with `formatAnchorTable`, #101
 * D19): a model that never anchored itself can't
 * assume there is nothing to rewind to, and it needs no `list` call. The
 * static how-to (persist first; don't rewind while a decision is pending)
 * lives in the tool's `promptGuidelines`, per the 2026-09-13
 * guidance-placement rule.
 */
export function buildRewindHintText(
  percent: number,
  tokensLeft: number,
  reserve: number,
  contextWindow: number,
  anchorTable: string,
): string {
  return `[${TOOL_NAME} hint] Context is at ${percent.toFixed(1)}% of ${formatWindow(contextWindow)} — ${formatTokensK(tokensLeft)} tokens left, within the ${formatTokensK(reserve)} reserve. Persist what matters to files now, then rewind to the oldest appropriate anchor.\n${anchorTable}`;
}

/** The TUI-only warning shown when a crossing has no anchor to rewind to. */
export function buildNoAnchorText(
  percent: number,
  contextWindow: number,
): string {
  return `${TOOL_NAME}: context at ${percent.toFixed(1)}% of ${formatWindow(contextWindow)} — no anchors on the active branch, so no rewind hint was sent. To enable one: /tree, select the entry to rewind to, press shift+l, label it \`${LABEL_PREFIX}<name>\` (e.g. ${LABEL_PREFIX}context-gathered), then ask me to rewind to it.`;
}

// -----------------------------------------------------------------------------
// Rewind-driven compaction (#101): the texts, byte-exact as decided.
// -----------------------------------------------------------------------------

/** `~Nk` for the withheld notice: thousands, at least `~1k`. */
function formatWithheldTokens(tokens: number): string {
  return `~${Math.max(1, Math.round(tokens / 1000))}k`;
}

/** The content that replaces a withheld tool result (D4). */
export function buildWithheldNoticeText(tokens: number): string {
  return `[${TOOL_NAME}] Output withheld: adding it overflowed the context window (${formatWithheldTokens(tokens)} tokens). The call ran; only its output was dropped. Rewind first, then re-run it if you still need the output.`;
}

/** The forced-rewind message (D5, owner wording of §12), then the anchor table. */
export function buildForcedRewindText(
  contextWindow: number,
  withheld: number,
  anchorTable: string,
): string {
  const count =
    withheld === 0
      ? "no tool results were withheld"
      : withheld === 1
        ? "1 tool result was withheld"
        : `${withheld} tool results were withheld`;
  return `[${TOOL_NAME}] Context overflowed the window (${formatWindow(contextWindow)}): ${count}. Rewind now — only ${TOOL_NAME} rewind is allowed until it lands.\n${anchorTable}`;
}

/** The final error of rewind-driven compaction (D8). */
export function buildRewindCompactionFailedText(reason: string): string {
  return `[${TOOL_NAME}] Context overflow recovery failed: ${reason}. The session is at the context limit; rewind manually with /tree, /compact, or start a new session.`;
}

/** D8 reasons (D8, D14, D16, §9.6). */
export const COMPACTION_FAILED_OVERFLOWED_AGAIN =
  "the context overflowed again during the forced rewind";
export const COMPACTION_FAILED_NO_REWIND = `no rewind after 3 attempts`;
export const COMPACTION_FAILED_NO_ANCHOR = "no anchor to rewind to";
export const COMPACTION_FAILED_ABORTED = "the forced rewind was aborted";
export const COMPACTION_FAILED_STILL_OVER =
  "the context is still over the limit after withholding tool results";
export const COMPACTION_FAILED_RUN_ENDED = "the run ended before the rewind";

/** Reason of a tool call blocked during a forced rewind (TUI-only, §9.6). */
export const FORCED_REWIND_BLOCK_REASON = `${TOOL_NAME}: blocked: only rewind is allowed until the forced rewind lands.`;

/** The either-or error at a hint crossing (D9), TUI only. */
export const REWIND_COMPACTION_OFF_TEXT = `${TOOL_NAME}: the rewind hint and rewind-driven compaction are off because pi's auto-compaction is on. Disable compaction.enabled or remove rewindHintReserveTokens.`;

/** The reserve-ordering warning (D10), numbers raw, model key as written. */
export function buildReserveOrderWarning(
  hintReserve: number,
  piReserve: number,
  modelKey: string,
): string {
  return `rewindHintReserveTokens (${hintReserve}) is below pi's compaction.reserveTokens (${piReserve}) for ${modelKey}: rewind-driven compaction may fire before the hint, giving the model no time to prepare and persist what matters to files.`;
}
