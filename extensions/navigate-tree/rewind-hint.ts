/**
 * rewind-hint — context-pressure nudge for the navigate-tree extension.
 *
 * Pure state + copy builders. `RewindHintTracker` fires exactly once per
 * threshold crossing and re-arms only when usage drops below the threshold
 * (a rewind or compaction). The builders render the two user-visible strings
 * (agent hint + TUI-only no-anchor warning). `collectAnchorNames` is the
 * anchors-exist predicate: root→leaf, `anchor:` stripped, capped.
 *
 * No pi runtime imports — the tracker and builders take plain values, and
 * `collectAnchorNames` takes the ReadonlySessionManager surface
 * structurally.
 */

import { formatWindow, LABEL_PREFIX, TOOL_NAME } from "./helpers.ts";

export const REWIND_HINT_CUSTOM_TYPE = "navigate-tree-rewind-hint";
export const REWIND_HINT_MAX_ANCHORS = 3;

/**
 * One-shot crossing tracker. `observe` returns true exactly on a fresh
 * crossing (`percent >= threshold` while unspent); a below-threshold
 * observation re-arms and a spent tracker stays silent while still above.
 */
export class RewindHintTracker {
  private spent = false;

  observe(percent: number, threshold: number): boolean {
    if (percent < threshold) {
      this.spent = false;
      return false;
    }
    if (this.spent) return false;
    this.spent = true;
    return true;
  }

  /** Re-arm for a fresh session (`session_start`). */
  reset(): void {
    this.spent = false;
  }
}

/**
 * The agent-facing hint. Dynamic trigger + ask only — the static how-to
 * (persist first; don't rewind while a decision is pending) lives in the
 * tool's `promptGuidelines`, per the 2026-09-13 guidance-placement rule.
 */
export function buildRewindHintText(
  percent: number,
  contextWindow: number,
): string {
  return `[${TOOL_NAME} hint] Context is at ${percent.toFixed(1)}% of ${formatWindow(contextWindow)} — running low. Persist what matters to files now, then list anchors and rewind to the appropriate one.`;
}

/** The TUI-only warning shown when a crossing has no anchor to rewind to. */
export function buildNoAnchorText(
  percent: number,
  contextWindow: number,
): string {
  return `${TOOL_NAME}: context at ${percent.toFixed(1)}% of ${formatWindow(contextWindow)} — no anchors on the active branch, so no rewind hint was sent. To enable one: /tree, select the entry to rewind to, press shift+l, label it \`${LABEL_PREFIX}<name>\` (e.g. ${LABEL_PREFIX}context-gathered), then ask me to rewind to it.`;
}

/**
 * Collect `anchor:`-labeled names on the active branch, root→leaf, capped at
 * `REWIND_HINT_MAX_ANCHORS`. Used as an existence predicate (`.length > 0`);
 * the cap keeps the walk bounded — the names are reserved for future copy
 * (the hint itself deliberately lists no anchors; the model calls `list` for
 * per-anchor percentages).
 */
export function collectAnchorNames(sm: {
  getBranch(): Array<{ id: string }>;
  getLabel(id: string): string | undefined;
}): string[] {
  const names: string[] = [];
  for (const entry of sm.getBranch()) {
    const label = sm.getLabel(entry.id);
    if (!label?.startsWith(LABEL_PREFIX)) continue;
    names.push(label.slice(LABEL_PREFIX.length));
    if (names.length >= REWIND_HINT_MAX_ANCHORS) break;
  }
  return names;
}
