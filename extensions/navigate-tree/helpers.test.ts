/**
 * Tests for navigate-tree pure helpers.
 *
 * Run with: bun test
 *
 * Uses node:test + node:assert/strict so the file is also runnable directly
 * via `node --test extensions/navigate-tree/helpers.test.ts` on Node 22.6+
 * (built-in TypeScript stripping). Tested against bun 1.3.13 and Node 24.
 *
 * The pi extension loader treats `./index.ts` as the entry point and ignores
 * sibling files — so this test file is not loaded as a separate extension.
 */

import * as assert from "node:assert/strict";
import { describe, it } from "node:test";
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
  it("does not strip when ## Goal appears far into the text", () => {
    const long = `${"x".repeat(300)}## Goal\nlate`;
    assert.equal(stripBranchSummaryBoilerplate(long), long);
  });
  it("does not strip when ## Goal is at index 0", () => {
    // Boilerplate always has prose before ## Goal, so a leading ## Goal isn't
    // the boilerplate pattern — preserve it.
    assert.equal(stripBranchSummaryBoilerplate("## Goal\nfoo"), "## Goal\nfoo");
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
});
