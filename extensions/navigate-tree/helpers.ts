/**
 * Pure helpers for the navigate-tree extension.
 *
 * Imported by `./index.ts` and `./helpers.test.ts`. Pi's extension loader
 * loads `./index.ts` and ignores everything else in this directory unless
 * referenced from there — so this file isn't loaded as a separate extension.
 *
 * No pi runtime imports — these are pure functions over plain JS values.
 */

// Hard cap on label-name length. 40 chars accommodates descriptive names
// (e.g. 'parser-edge-case-investigation', 31 chars) while keeping list
// output column-friendly under common terminal widths and preventing a
// runaway label string from poisoning the JSONL on disk.
export const MAX_NAME_LENGTH = 40;
export const NAME_RE = /^[a-z0-9][a-z0-9-]*$/;

// Maximum lead-in distance for pi's branch-summary boilerplate marker.
// Pi's standard prelude ("The user explored a different conversation
// branch...") fits in the first ~150 chars; 200 is a generous upper bound.
// A "## Goal" found later than this is treated as in-content prose, not
// the boilerplate marker, and the strip is a no-op.
const MAX_BOILERPLATE_LEAD_IN = 200;

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

/** Format token count as a percentage with one decimal, or as `Nk` if no window. */
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
  const goalIdx = text.indexOf("## Goal");
  if (goalIdx > 0 && goalIdx < MAX_BOILERPLATE_LEAD_IN) {
    return text.slice(goalIdx + "## Goal".length);
  }
  return text;
}

/**
 * Extract a string from a message-like content field. Handles both the
 * legacy string shape and the modern array-of-blocks shape, joining all
 * text blocks with spaces.
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
