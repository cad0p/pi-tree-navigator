/**
 * Pure helpers for the navigate-tree extension.
 *
 * Imported by `./index.ts` and `./helpers.test.ts`. Pi's extension loader
 * loads `./index.ts` and ignores everything else in this directory unless
 * referenced from there — so this file isn't loaded as a separate extension.
 *
 * No pi runtime imports — these are pure functions over plain JS values.
 */

// ---------------------------------------------------------------------------
// Exported boundary constants below (MAX_NAME_LENGTH, MAX_BOILERPLATE_LEAD_IN).
//
// Stability: these are internal tunables. Exported only so the test suite
// can pin boundary cases by constant rather than literal. Re-tuning is
// NOT a semver-breaking change for this package — production callers
// should rely on the registered `navigate_tree` tool surface, not import
// these constants directly.
// ---------------------------------------------------------------------------

// Hard cap on label-name length. 40 chars accommodates descriptive names
// (e.g. 'parser-edge-case-investigation', 31 chars) while keeping list
// output column-friendly under common terminal widths and preventing a
// runaway label string from poisoning the JSONL on disk.
export const MAX_NAME_LENGTH = 40;
/**
 * Kebab-case anchor name: lowercase alphanumeric segments separated by
 * single hyphens. No leading / trailing / double hyphens (the `isValidName`
 * test suite enforces these rejections). Mirrors the pattern documented in
 * the tool description and README — if this regex relaxes, both surfaces
 * need the same update.
 */
const NAME_RE = /^[a-z0-9]+(-[a-z0-9]+)*$/;

// Maximum lead-in distance for pi's branch-summary boilerplate marker.
// Pi's standard prelude ("The user explored a different conversation
// branch...") fits in the first ~150 chars; 200 is a generous upper bound.
// A "## Goal" found later than this is treated as in-content prose, not
// the boilerplate marker, and the strip is a no-op.
export const MAX_BOILERPLATE_LEAD_IN = 200;

// Sentinel that pi's branch-summary prelude always starts with. Gating the
// strip on this prefix makes the helper a no-op for any non-boilerplate
// text that happens to contain `## Goal` early (e.g. a user-authored
// markdown doc whose first H2 is "Goal"). If pi's wording later changes,
// the strip falls back to no-op and `findLabelHint` shows the unmodified
// prelude — still useful, just less concise.
//
// Pi 0.75.5's branch_summary prelude begins with this exact text, derived
// from `branch-summarization.js` in @earendil-works/pi-coding-agent. If pi
// reworks the prelude wording (e.g. "navigated" instead of "explored"),
// the strip degrades to a no-op (graceful) — but `findLabelHint` will then
// show the unmodified prelude in `list` output. Verify against pi's
// `dist/core/compaction/branch-summarization.js` when bumping the
// peer-dep floor and update if the leading copy has changed.
const BRANCH_SUMMARY_SENTINEL =
  "The user explored a different conversation branch";

// Pi 0.75.5's branch-summary boilerplate places `## Goal` after the prelude
// (verified against `dist/core/compaction/branch-summarization.js`). Used to
// anchor the strip cut-point in `stripBranchSummaryBoilerplate` — a single
// constant so the indexOf probe and the slice-length advance can't drift.
const GOAL_HEADER = "## Goal";

/** Validate a kebab-case name suitable for use as a navigate-tree label. */
export function isValidName(s: unknown): s is string {
  return (
    typeof s === "string" &&
    s.length > 0 &&
    s.length <= MAX_NAME_LENGTH &&
    NAME_RE.test(s)
  );
}

/** Truncate text to a one-line preview, collapsing whitespace. */
export function toOneLine(text: string, maxLen: number): string | null {
  const t = text.replace(/\s+/g, " ").trim();
  if (t.length === 0) return null;
  if (maxLen <= 1) return t.length > 0 ? "…" : null;
  return t.length > maxLen ? `${t.slice(0, maxLen - 1)}…` : t;
}

/**
 * Format token count as a percentage with one decimal, or as `Nk` if no window.
 *
 * Negative-input contract: a negative `tokens` value produces a negative
 * formatted string ("-0.0%", "-5.0%") rather than being clamped to 0 —
 * negative tokens shouldn't reach this helper in production (token counters
 * are unsigned), but if a caller bug feeds one in, the helper renders the
 * sign and stays non-throwing. Tests pin the JS `toFixed` behavior so a
 * future precision change surfaces immediately.
 */
export function formatPct1(tokens: number, contextWindow: number): string {
  if (contextWindow <= 0) return `${(tokens / 1000).toFixed(1)}k`;
  return `${((tokens / contextWindow) * 100).toFixed(1)}%`;
}

/** Format a context window size as `1.0M` or `200k`. Empty string when unknown. */
export function formatWindow(contextWindow: number): string {
  if (contextWindow <= 0) return "";
  if (contextWindow >= 1_000_000)
    return `${(contextWindow / 1_000_000).toFixed(1)}M`;
  return `${(contextWindow / 1000).toFixed(0)}k`;
}

export function formatContextDelta(
  beforeTokens: number,
  afterTokens: number,
  contextWindow: number,
): string {
  if (contextWindow > 0) {
    return `context ${formatPct1(beforeTokens, contextWindow)} → ${formatPct1(afterTokens, contextWindow)} of ${formatWindow(contextWindow)}`;
  }
  return `tokens ${beforeTokens} → ${afterTokens}`;
}

/**
 * Strip pi's standard branch-summary boilerplate ("The user explored a
 * different conversation branch...") so the hint shows the actual content.
 */
export function stripBranchSummaryBoilerplate(text: string): string {
  if (!text.startsWith(BRANCH_SUMMARY_SENTINEL)) return text;
  const goalIdx = text.indexOf(GOAL_HEADER);
  if (goalIdx > 0 && goalIdx < MAX_BOILERPLATE_LEAD_IN) {
    return text.slice(goalIdx + GOAL_HEADER.length);
  }
  return text;
}

/**
 * Extract a string from a message-like content field. Handles both the
 * legacy string shape and the modern array-of-blocks shape, joining all
 * text blocks with spaces.
 *
 * Note: empty-text blocks survive the `type === "text"` filter and are
 * joined with the separator, producing leading / trailing / double spaces
 * in the output (e.g. `[{text: ""}, {text: "kept"}]` → `" kept"`). Callers
 * that render the result should run it through `toOneLine()` or `.trim()`.
 */
export function extractTextContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter(
      (c): c is { type: "text"; text: string } =>
        typeof c === "object" &&
        c !== null &&
        (c as { type?: string }).type === "text" &&
        typeof (c as { text?: unknown }).text === "string",
    )
    .map((c) => c.text)
    .join(" ");
}
