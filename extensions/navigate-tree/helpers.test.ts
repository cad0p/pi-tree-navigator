/**
 * Tests for navigate-tree pure helpers.
 *
 * Bun test runner (project-pinned to bun). Run with: bun test
 *
 * The pi extension loader treats `./index.ts` as the entry point and ignores
 * sibling files — so this test file is not loaded as a separate extension.
 */

import { describe, it } from "bun:test";
import * as assert from "node:assert/strict";
import {
  extractTextContent,
  formatContextDelta,
  formatPct1,
  formatWindow,
  isValidName,
  MAX_NAME_LENGTH,
  stripBranchSummaryBoilerplate,
  toOneLine,
} from "./helpers.ts";

describe("isValidName", () => {
  it("accepts kebab-case", () => {
    assert.equal(isValidName("a"), true);
    assert.equal(isValidName("impl-start"), true);
    assert.equal(isValidName("step-1-of-2"), true);
    assert.equal(isValidName("0design"), true);
  });
  it("rejects empty, undefined, non-strings", () => {
    assert.equal(isValidName(""), false);
    assert.equal(isValidName(undefined), false);
    assert.equal(isValidName(null), false);
    assert.equal(isValidName(123), false);
    assert.equal(isValidName({ a: 1 }), false);
  });
  it("rejects leading hyphen, uppercase, underscores, spaces", () => {
    assert.equal(isValidName("-leading"), false);
    assert.equal(isValidName("Impl-Start"), false);
    assert.equal(isValidName("snake_case"), false);
    assert.equal(isValidName("with space"), false);
  });
  it("rejects trailing hyphen", () => {
    assert.equal(isValidName("a-"), false);
  });
  it("rejects double hyphen", () => {
    assert.equal(isValidName("a--b"), false);
  });
  it("rejects all-hyphen / hyphen-only", () => {
    assert.equal(isValidName("---"), false);
    assert.equal(isValidName("-"), false);
  });
  it("rejects digit-then-trailing-hyphen", () => {
    assert.equal(isValidName("0-"), false);
  });
  it("enforces max length", () => {
    assert.equal(isValidName("a".repeat(MAX_NAME_LENGTH)), true);
    assert.equal(isValidName("a".repeat(MAX_NAME_LENGTH + 1)), false);
  });
});

describe("toOneLine", () => {
  it("collapses whitespace", () => {
    assert.equal(toOneLine("a   b\n\n c", 100), "a b c");
  });
  it("returns null for empty / whitespace-only", () => {
    assert.equal(toOneLine("", 10), null);
    assert.equal(toOneLine("   \n  \t  ", 10), null);
  });
  it("truncates with ellipsis", () => {
    assert.equal(toOneLine("hello world", 8), "hello w…");
  });
  it("does not truncate when under maxLen", () => {
    assert.equal(toOneLine("short", 100), "short");
  });
  it("handles maxLen <= 1", () => {
    assert.equal(toOneLine("text", 1), "…");
    assert.equal(toOneLine("", 1), null);
  });
});

describe("formatPct1", () => {
  it("formats with one decimal when window is known", () => {
    assert.equal(formatPct1(19_000, 1_000_000), "1.9%");
    assert.equal(formatPct1(560_000, 1_000_000), "56.0%");
    assert.equal(formatPct1(0, 1_000_000), "0.0%");
  });
  it("falls back to k-tokens when window is unknown", () => {
    assert.equal(formatPct1(19_000, 0), "19.0k");
    assert.equal(formatPct1(123, 0), "0.1k");
    assert.equal(formatPct1(560_000, -1), "560.0k");
  });
  it("renders 100.0% when tokens === window exactly", () => {
    // Boundary: at the window, percent is exactly 100.0% (not clamped).
    assert.equal(formatPct1(1_000_000, 1_000_000), "100.0%");
  });
  it("renders > 100.0% when tokens overflow the window", () => {
    // Pinning behavior, not a bug guard: a token estimate over the
    // window can happen if the synthetic-assistant baseline lags the
    // actual chain. The column should keep formatting; downstream
    // formatters (formatContextDelta, list output) inherit this.
    assert.equal(formatPct1(1_500_000, 1_000_000), "150.0%");
  });
  it("rounds tiny fractions down to 0.0%", () => {
    // 50 / 1_000_000 = 0.005% → toFixed(1) yields "0.0%". Pinning the
    // round-down behavior so a future toFixed(2) refactor surfaces here.
    assert.equal(formatPct1(50, 1_000_000), "0.0%");
  });
  it("renders negative tokens with a leading minus", () => {
    // Negative tokens shouldn't reach this helper in production (token
    // counters are unsigned), but if a caller bug feeds a negative the
    // helper must not crash; pin the actual JS toFixed output.
    assert.equal(formatPct1(-1, 1_000_000), "-0.0%");
    assert.equal(formatPct1(-50_000, 1_000_000), "-5.0%");
  });
});

describe("formatWindow", () => {
  it("uses M for >=1M", () => {
    assert.equal(formatWindow(1_000_000), "1.0M");
    assert.equal(formatWindow(1_500_000), "1.5M");
  });
  it("uses k below 1M", () => {
    assert.equal(formatWindow(200_000), "200k");
    assert.equal(formatWindow(8_000), "8k");
  });
  it("returns empty string when unknown", () => {
    assert.equal(formatWindow(0), "");
    assert.equal(formatWindow(-1), "");
  });
});

describe("formatContextDelta", () => {
  it("renders before → after with window", () => {
    assert.equal(
      formatContextDelta(560_000, 19_000, 1_000_000),
      "context 56.0% → 1.9% of 1.0M",
    );
  });
  it("falls back to tokens when no window", () => {
    assert.equal(formatContextDelta(123, 45, 0), "tokens 123 → 45");
  });
});

describe("stripBranchSummaryBoilerplate", () => {
  it("strips lead-in up to ## Goal", () => {
    const text =
      "The user explored a different conversation branch before returning here.\nSummary of that exploration:\n\n## Goal\nFix it.";
    assert.equal(stripBranchSummaryBoilerplate(text), "\nFix it.");
  });
  it("returns text unchanged when no early ## Goal", () => {
    assert.equal(stripBranchSummaryBoilerplate("just text"), "just text");
  });
  it("does not strip without sentinel even with late ## Goal", () => {
    // Negative pin: missing sentinel short-circuits the strip even when a
    // "## Goal" appears later in the text. Distinct from the lead-in
    // boundary check (which is exercised in the 199/200 pair below).
    const long = `${"x".repeat(300)}## Goal\nlate`;
    assert.equal(stripBranchSummaryBoilerplate(long), long);
  });
  it("does not strip when ## Goal appears past MAX_BOILERPLATE_LEAD_IN even with sentinel", () => {
    // Lead-in boundary check WITH the sentinel: prelude prefix matches but
    // the ## Goal is too late, so the strip is short-circuited by the
    // distance check, not the sentinel guard. Pairs with the 199/200 tests
    // below to fully cover the boundary.
    const sentinel = "The user explored a different conversation branch";
    const padding = "x".repeat(250);
    const text = `${sentinel}${padding}## Goal\nlate`;
    assert.equal(stripBranchSummaryBoilerplate(text), text);
  });
  it("does not strip when ## Goal is at index 0", () => {
    // Boilerplate always has prose before ## Goal, so a leading ## Goal isn't
    // the boilerplate pattern — preserve it.
    assert.equal(stripBranchSummaryBoilerplate("## Goal\nfoo"), "## Goal\nfoo");
  });
  it("does not strip when text lacks the pi boilerplate sentinel", () => {
    // A user-authored doc whose first H2 is "Goal" must survive untouched
    // — the strip is gated on pi's actual prelude prefix.
    const userDoc =
      "Some quick notes on the parser refactor.\n\n## Goal\nMake it faster.";
    assert.equal(stripBranchSummaryBoilerplate(userDoc), userDoc);
  });
  it("strips when ## Goal is at index 199 (just inside the boundary)", () => {
    // Sentinel is 49 chars; pad to put '## Goal' at index 199 exactly.
    // The boundary check is `goalIdx < 200`, so 199 should still strip.
    const sentinel = "The user explored a different conversation branch";
    const padding = "x".repeat(199 - sentinel.length);
    const text = `${sentinel}${padding}## Goal\nbody`;
    assert.equal(text.indexOf("## Goal"), 199);
    assert.equal(stripBranchSummaryBoilerplate(text), "\nbody");
  });
  it("does not strip when ## Goal is at index 200 (boundary)", () => {
    // Boundary case: at exactly 200, the strict-less-than check fails and
    // the text is preserved as-is.
    const sentinel = "The user explored a different conversation branch";
    const padding = "x".repeat(200 - sentinel.length);
    const text = `${sentinel}${padding}## Goal\nbody`;
    assert.equal(text.indexOf("## Goal"), 200);
    assert.equal(stripBranchSummaryBoilerplate(text), text);
  });
});

describe("extractTextContent", () => {
  it("returns string content as-is", () => {
    assert.equal(extractTextContent("hello"), "hello");
  });
  it("joins text blocks with spaces", () => {
    const blocks = [
      { type: "text", text: "first" },
      { type: "thinking", thinking: "ignored" },
      { type: "text", text: "second" },
    ];
    assert.equal(extractTextContent(blocks), "first second");
  });
  it("filters out non-text blocks", () => {
    const blocks = [
      { type: "toolCall", id: "x", name: "bash", arguments: {} },
      { type: "text", text: "kept" },
    ];
    assert.equal(extractTextContent(blocks), "kept");
  });
  it("returns empty string for non-string non-array", () => {
    assert.equal(extractTextContent(undefined), "");
    assert.equal(extractTextContent(null), "");
    assert.equal(extractTextContent(42), "");
    assert.equal(extractTextContent({ type: "text", text: "x" }), "");
  });
  it("ignores text blocks where text isn't a string", () => {
    const blocks = [
      { type: "text", text: null },
      { type: "text", text: "kept" },
    ];
    assert.equal(extractTextContent(blocks), "kept");
  });
  it("ignores tool_result, image, and redactedThinking blocks", () => {
    // Pi emits these block types alongside text on real chains. The helper
    // is field-shape-based (`type === 'text'` and `text` is a string), so
    // these all fall through; pin the realistic mix to lock that behavior.
    const blocks = [
      {
        type: "toolResult",
        id: "x",
        output: [{ type: "text", text: "ignored" }],
      },
      { type: "image", source: { type: "base64", data: "…" } },
      { type: "redactedThinking", data: "…" },
      { type: "text", text: "kept" },
    ];
    assert.equal(extractTextContent(blocks), "kept");
  });
  it("returns empty string for empty array", () => {
    assert.equal(extractTextContent([]), "");
  });
  it("preserves empty-string text blocks (joined with surrounding space)", () => {
    // Pinning a slightly-surprising actual behavior: ['', 'kept'].join(' ')
    // === ' kept'. If a future refactor filters empty-string blocks the
    // join shape changes; surface it here.
    assert.equal(
      extractTextContent([
        { type: "text", text: "" },
        { type: "text", text: "kept" },
      ]),
      " kept",
    );
  });
});
