/**
 * Tests for the rewind-hint tracker, copy builders, and anchor collection.
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
  collectAnchorNames,
  REWIND_HINT_CUSTOM_TYPE,
  REWIND_HINT_MAX_ANCHORS,
  RewindHintTracker,
} from "./rewind-hint.ts";

describe("RewindHintTracker", () => {
  it("fires on the first observation at or above the threshold", () => {
    const tracker = new RewindHintTracker();
    assert.equal(tracker.observe(90, 90), true);
  });

  it("does not fire below the threshold", () => {
    const tracker = new RewindHintTracker();
    assert.equal(tracker.observe(89.9, 90), false);
    assert.equal(tracker.observe(90, 90), true);
  });

  it("does not re-fire while spent and still above", () => {
    const tracker = new RewindHintTracker();
    assert.equal(tracker.observe(95, 90), true);
    assert.equal(tracker.observe(96, 90), false);
    assert.equal(tracker.observe(99, 90), false);
  });

  it("re-arms after a drop below the threshold", () => {
    const tracker = new RewindHintTracker();
    assert.equal(tracker.observe(95, 90), true);
    assert.equal(tracker.observe(12, 90), false);
    assert.equal(tracker.observe(91, 90), true);
    assert.equal(tracker.observe(92, 90), false);
  });

  it("reset() re-arms a spent tracker", () => {
    const tracker = new RewindHintTracker();
    assert.equal(tracker.observe(95, 90), true);
    assert.equal(tracker.observe(95, 90), false);
    tracker.reset();
    assert.equal(tracker.observe(95, 90), true);
  });

  it("reset() on a fresh tracker is a no-op", () => {
    const tracker = new RewindHintTracker();
    tracker.reset();
    assert.equal(tracker.observe(95, 90), true);
  });
});

describe("buildRewindHintText", () => {
  it("is byte-exact at 91.2% of a 1M window", () => {
    assert.equal(
      buildRewindHintText(91.2, 1_000_000),
      "[navigate_tree hint] Context is at 91.2% of 1.0M — running low. Persist what matters to files now, then list anchors and rewind to the oldest appropriate one.",
    );
  });

  it("is byte-exact at a 200k window (k-window formatting)", () => {
    assert.equal(
      buildRewindHintText(87.5, 200_000),
      "[navigate_tree hint] Context is at 87.5% of 200k — running low. Persist what matters to files now, then list anchors and rewind to the oldest appropriate one.",
    );
  });

  it("starts with the TOOL_NAME-derived header (rename-desync pin)", () => {
    assert.ok(
      buildRewindHintText(91.2, 1_000_000).startsWith(`[${TOOL_NAME} hint] `),
    );
  });

  it("renders the percent with one decimal (toFixed(1))", () => {
    assert.match(buildRewindHintText(90, 1_000_000), /at 90\.0% of 1\.0M/);
    assert.match(buildRewindHintText(89.94, 1_000_000), /at 89\.9%/);
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

describe("collectAnchorNames", () => {
  function fakeSm(
    order: string[],
    labels: Record<string, string | undefined>,
  ): {
    getBranch(): Array<{ id: string }>;
    getLabel(id: string): string | undefined;
  } {
    return {
      getBranch: () => order.map((id) => ({ id })),
      getLabel: (id: string) => labels[id],
    };
  }

  it("returns anchor names root→leaf with the prefix stripped", () => {
    const sm = fakeSm(["a", "b", "c"], {
      a: "anchor:context-gathered",
      b: "anchor:plan-approved",
      c: undefined,
    });
    assert.deepEqual(collectAnchorNames(sm), [
      "context-gathered",
      "plan-approved",
    ]);
  });

  it("matches labels built from LABEL_PREFIX (rename-desync pin)", () => {
    const sm = fakeSm(["a", "b"], {
      a: `${LABEL_PREFIX}context-gathered`,
      b: `${LABEL_PREFIX}plan-approved`,
    });
    assert.deepEqual(collectAnchorNames(sm), [
      "context-gathered",
      "plan-approved",
    ]);
  });

  it("skips non-anchor labels", () => {
    const sm = fakeSm(["a", "b", "c"], {
      a: "other:label",
      b: "anchor:kept",
      c: "anchorish",
    });
    assert.deepEqual(collectAnchorNames(sm), ["kept"]);
  });

  it("caps at REWIND_HINT_MAX_ANCHORS (root side wins)", () => {
    const order = ["a", "b", "c", "d", "e"];
    const labels: Record<string, string> = {};
    for (const id of order) labels[id] = `anchor:${id}`;
    const names = collectAnchorNames(fakeSm(order, labels));
    assert.equal(names.length, REWIND_HINT_MAX_ANCHORS);
    assert.deepEqual(names, ["a", "b", "c"]);
  });

  it("returns an empty array when nothing is anchored", () => {
    assert.deepEqual(
      collectAnchorNames(fakeSm(["a", "b"], { a: undefined, b: "x" })),
      [],
    );
  });

  it("returns an empty array on an empty branch", () => {
    assert.deepEqual(collectAnchorNames(fakeSm([], {})), []);
  });
});
