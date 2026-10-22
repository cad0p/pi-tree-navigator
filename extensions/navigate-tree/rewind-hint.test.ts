/**
 * Tests for the rewind-hint tracker and copy builders.
 *
 * Node test runner. Run with: pnpm test
 *
 * The pi extension loader treats `./index.ts` as the entry point and ignores
 * sibling files — so this test file is not loaded as a separate extension.
 * These are pure-value tests: no pi session, no fs, no LLM.
 */

import * as assert from "node:assert/strict";
import { describe, it } from "node:test";
import { LABEL_PREFIX, TOOL_NAME } from "./helpers.ts";
import {
  buildNoAnchorText,
  buildRewindHintText,
  REWIND_HINT_CUSTOM_TYPE,
  RewindHintTracker,
} from "./rewind-hint.ts";

describe("RewindHintTracker", () => {
  it("fires on the first observation at or inside the reserve", () => {
    const tracker = new RewindHintTracker();
    assert.equal(tracker.observe(20_000, 20_000), true);
  });

  it("does not fire outside the reserve", () => {
    const tracker = new RewindHintTracker();
    assert.equal(tracker.observe(20_001, 20_000), false);
    assert.equal(tracker.observe(20_000, 20_000), true);
  });

  it("does not re-fire while spent and still inside", () => {
    const tracker = new RewindHintTracker();
    assert.equal(tracker.observe(10_000, 20_000), true);
    assert.equal(tracker.observe(5_000, 20_000), false);
    assert.equal(tracker.observe(-70_000, 20_000), false);
  });

  it("re-arms after the tokens left climb back outside the reserve", () => {
    const tracker = new RewindHintTracker();
    assert.equal(tracker.observe(10_000, 20_000), true);
    assert.equal(tracker.observe(80_000, 20_000), false);
    assert.equal(tracker.observe(19_000, 20_000), true);
    assert.equal(tracker.observe(18_000, 20_000), false);
  });

  it("a null reserve never fires and re-arms (#86)", () => {
    const t = new RewindHintTracker();
    assert.equal(t.observe(1_000, null), false);
    assert.equal(t.observe(50_000, 50_000), true);
    assert.equal(t.observe(1_000, null), false);
    assert.equal(t.isSpent(), false);
  });

  it("isSpent() tracks the crossing", () => {
    const tracker = new RewindHintTracker();
    assert.equal(tracker.isSpent(), false);
    tracker.observe(10_000, 20_000);
    assert.equal(tracker.isSpent(), true);
    tracker.observe(80_000, 20_000);
    assert.equal(tracker.isSpent(), false);
  });

  it("reset() re-arms a spent tracker", () => {
    const tracker = new RewindHintTracker();
    assert.equal(tracker.observe(10_000, 20_000), true);
    assert.equal(tracker.observe(10_000, 20_000), false);
    tracker.reset();
    assert.equal(tracker.observe(10_000, 20_000), true);
  });

  it("reset() on a fresh tracker is a no-op", () => {
    const tracker = new RewindHintTracker();
    tracker.reset();
    assert.equal(tracker.observe(10_000, 20_000), true);
  });
});

describe("buildRewindHintText", () => {
  it("is byte-exact at 98.6% of a 100k window, naming the anchors", () => {
    assert.equal(
      buildRewindHintText(
        98.6,
        100_000,
        "Active anchors (root → leaf, % of 100k):\n  20.9%  start",
      ),
      "[navigate_tree hint] Context is at 98.6% of 100k. Persist what matters to files now, then rewind to the oldest appropriate anchor.\nActive anchors (root → leaf, % of 100k):\n  20.9%  start",
    );
  });

  it("is byte-exact at a 1M window with several anchors", () => {
    assert.equal(
      buildRewindHintText(
        95.2,
        1_000_000,
        "Active anchors (root → leaf, % of 1.0M):\n   4.1%  start\n  12.0%  context-gathered",
      ),
      "[navigate_tree hint] Context is at 95.2% of 1.0M. Persist what matters to files now, then rewind to the oldest appropriate anchor.\nActive anchors (root → leaf, % of 1.0M):\n   4.1%  start\n  12.0%  context-gathered",
    );
  });

  it("never prints a tokens-left or reserve clause (#114)", () => {
    const text = buildRewindHintText(99.9, 1_000_000, "'a' 1.0%");
    assert.doesNotMatch(text, /tokens left/);
    assert.doesNotMatch(text, /reserve/);
    assert.doesNotMatch(text, /—/);
  });

  it("starts with the TOOL_NAME-derived header (rename-desync pin)", () => {
    assert.ok(
      buildRewindHintText(91.2, 1_000_000, "'a' 1.0%").startsWith(
        `[${TOOL_NAME} hint] `,
      ),
    );
  });

  it("renders the percent with one decimal (toFixed(1))", () => {
    assert.match(
      buildRewindHintText(90, 1_000_000, "'a' 1.0%"),
      /at 90\.0% of 1\.0M/,
    );
    assert.match(
      buildRewindHintText(89.94, 1_000_000, "'a' 1.0%"),
      /at 89\.9%/,
    );
  });
});

describe("buildNoAnchorText", () => {
  it("is byte-exact at 91.2% of a 1M window", () => {
    assert.equal(
      buildNoAnchorText(91.2, 1_000_000),
      "navigate_tree: context at 91.2% of 1.0M — no anchors on the active branch, so no rewind hint was sent. To enable one: /tree, select the entry to rewind to, press shift+l, label it `anchor:<name>` (e.g. anchor:context-gathered), then ask me to rewind to it.",
    );
  });

  it("is byte-exact at a 200k window (k-window formatting)", () => {
    assert.equal(
      buildNoAnchorText(87.5, 200_000),
      "navigate_tree: context at 87.5% of 200k — no anchors on the active branch, so no rewind hint was sent. To enable one: /tree, select the entry to rewind to, press shift+l, label it `anchor:<name>` (e.g. anchor:context-gathered), then ask me to rewind to it.",
    );
  });

  it("starts with the TOOL_NAME-derived prefix (rename-desync pin)", () => {
    assert.ok(
      buildNoAnchorText(91.2, 1_000_000).startsWith(
        `${TOOL_NAME}: context at `,
      ),
    );
  });

  it("renders the LABEL_PREFIX-derived label instructions (rename-desync pin)", () => {
    const text = buildNoAnchorText(91.2, 1_000_000);
    assert.ok(
      text.includes(
        `label it \`${LABEL_PREFIX}<name>\` (e.g. ${LABEL_PREFIX}context-gathered)`,
      ),
    );
  });
});

describe("REWIND_HINT_CUSTOM_TYPE", () => {
  it("is the pinned customType", () => {
    assert.equal(REWIND_HINT_CUSTOM_TYPE, "navigate-tree-rewind-hint");
  });
});
