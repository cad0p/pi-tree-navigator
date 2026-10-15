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
 * while still inside the reserve.
 */
export class RewindHintTracker {
  private spent = false;

  observe(tokensLeft: number, reserve: number): boolean {
    if (tokensLeft > reserve) {
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

  /** Re-arm for a fresh session (`session_start`). */
  reset(): void {
    this.spent = false;
  }
}

/** `12.3k`; negative (an overrun window) clamps to `0.0k`. */
function formatTokensK(tokens: number): string {
  return `${(Math.max(0, tokens) / 1000).toFixed(1)}k`;
}

/**
 * The agent-facing hint (#84). Dynamic trigger + ask, plus the active anchors
 * (`anchorList`, pre-rendered by the caller in the min-savings rejection's
 * `'name' 20.9% · ...` form): a model that never anchored itself can't
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
  anchorList: string,
): string {
  return `[${TOOL_NAME} hint] Context is at ${percent.toFixed(1)}% of ${formatWindow(contextWindow)} — ${formatTokensK(tokensLeft)} tokens left, within the ${formatTokensK(reserve)} reserve. Persist what matters to files now, then rewind to the oldest appropriate anchor. Active anchors: ${anchorList}`;
}

/** The TUI-only warning shown when a crossing has no anchor to rewind to. */
export function buildNoAnchorText(
  percent: number,
  contextWindow: number,
): string {
  return `${TOOL_NAME}: context at ${percent.toFixed(1)}% of ${formatWindow(contextWindow)} — no anchors on the active branch, so no rewind hint was sent. To enable one: /tree, select the entry to rewind to, press shift+l, label it \`${LABEL_PREFIX}<name>\` (e.g. ${LABEL_PREFIX}context-gathered), then ask me to rewind to it.`;
}
