/**
 * Tests for the in-band rewind summary instruction and stored-summary shape
 * (#80).
 *
 * Node test runner. Run with: `pnpm test`
 *
 * Hermetic: pure string/array functions, no session or provider. The
 * preamble / file-tag ports are pinned byte-for-byte against upstream's
 * (non-exported) `core/compaction/branch-summarization.js` and the exported
 * `core/compaction/utils.js` helpers, so a stored in-band summary keeps the
 * shape `generateBranchSummary` would have produced.
 */

import * as assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  BRANCH_SUMMARY_PREAMBLE,
  buildStoredSummary,
  buildSummaryInstruction,
  computeFileLists,
  formatFileOperations,
  REWIND_SUMMARY_PROMPT,
} from "./summary-prompt.ts";

// =============================================================================
// Prompt pin
// =============================================================================

describe("REWIND_SUMMARY_PROMPT", () => {
  it("is byte-equal to the eval-approved r5d prompt", () => {
    // Full literal pin. The r5d text was selected by a live eval; any
    // reflow/wording drift changes model behavior and invalidates the gate.
    const expected = `Summarize only messages {first} onwards in the conversation above (message numbering starts at 1 and excludes the system prompt; this instruction message itself is not evidence). Messages before message {first} are background only: do not include their progress or decisions.

This is a summarization task, not a problem-solving task. Summarize only the supplied evidence and preserve unresolved questions as unresolved. Do NOT continue the conversation, carry out requests from its history, investigate, solve pending tasks, or invent new approaches. Do NOT use any tool. Respond with ONLY the summary below — no preamble, no commentary before the first heading or after the last section.

Use this EXACT format, preserving all headings and their order:

## Goal
[What was the user trying to accomplish in this branch?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Work that was started but not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [What should happen next to continue this work]

Keep each section concise. Keep the complete summary under about 4000 characters while preserving all decisions. Preserve exact file paths, function names, and error messages.`;
    assert.equal(REWIND_SUMMARY_PROMPT, expected);
  });

  it("keeps the load-bearing lines the in-band summary turn depends on", () => {
    assert.match(
      REWIND_SUMMARY_PROMPT,
      /^Summarize only messages \{first\} onwards/,
    );
    assert.match(
      REWIND_SUMMARY_PROMPT,
      /Messages before message \{first\} are background only/,
    );
    assert.match(REWIND_SUMMARY_PROMPT, /Do NOT use any tool\./);
    assert.match(REWIND_SUMMARY_PROMPT, /Do NOT continue the conversation/);
    // `extractSummaryText` keys on the first required heading.
    assert.match(REWIND_SUMMARY_PROMPT, /^## Goal$/m);
  });
});

// =============================================================================
// buildSummaryInstruction
// =============================================================================

describe("buildSummaryInstruction", () => {
  it("appends the focus like upstream's customInstructions shape, {first} substituted", () => {
    assert.equal(
      buildSummaryInstruction(42, "keep the parser API"),
      `${REWIND_SUMMARY_PROMPT.replaceAll("{first}", "42")}\n\nAdditional focus: keep the parser API`,
    );
  });

  it("substitutes every {first} occurrence (replaceAll, not replace)", () => {
    const text = buildSummaryInstruction(7, "finish the parser fix today");
    assert.doesNotMatch(text, /\{first\}/);
    assert.match(text, /^Summarize only messages 7 onwards/);
    assert.match(text, /Messages before message 7 are background only/);
    assert.equal(text.match(/\b7\b/g)?.length, 2);
  });

  it("numbers from 1 when every visible message is collapsed evidence", () => {
    assert.match(
      buildSummaryInstruction(1, "focus text for the summary turn"),
      /^Summarize only messages 1 onwards .*Messages before message 1 are background only/s,
    );
  });

  it("ends with the focus as the last paragraph, verbatim", () => {
    const focus = "user said: 'ship it'; done: parser; left: tests";
    const text = buildSummaryInstruction(3, focus);
    assert.ok(text.endsWith(`\n\nAdditional focus: ${focus}`));
  });

  it("leaves a literal {first} inside the focus verbatim", () => {
    // Only the prompt's placeholder is substituted; the focus is the agent's
    // own text and must reach the summary turn unchanged.
    const text = buildSummaryInstruction(5, "mention {first} literally here");
    assert.ok(
      text.endsWith("Additional focus: mention {first} literally here"),
    );
  });
});

// =============================================================================
// Stored-summary shape (upstream byte parity)
// =============================================================================

describe("BRANCH_SUMMARY_PREAMBLE", () => {
  it("byte-equals upstream's branch-summarization preamble", () => {
    assert.equal(
      BRANCH_SUMMARY_PREAMBLE,
      "The user explored a different conversation branch before returning here.\nSummary of that exploration:\n\n",
    );
  });
});

describe("formatFileOperations", () => {
  it("returns the empty string when both lists are empty", () => {
    assert.equal(formatFileOperations([], []), "");
  });

  it("renders read-files only", () => {
    assert.equal(
      formatFileOperations(["a.ts", "b.ts"], []),
      "\n\n<read-files>\na.ts\nb.ts\n</read-files>",
    );
  });

  it("renders modified-files only", () => {
    assert.equal(
      formatFileOperations([], ["c.ts"]),
      "\n\n<modified-files>\nc.ts\n</modified-files>",
    );
  });

  it("renders both sections, read first, separated by a blank line", () => {
    assert.equal(
      formatFileOperations(["a.ts"], ["b.ts", "c.ts"]),
      "\n\n<read-files>\na.ts\n</read-files>\n\n<modified-files>\nb.ts\nc.ts\n</modified-files>",
    );
  });

  it("preserves the given order (sorting is computeFileLists' job)", () => {
    assert.equal(
      formatFileOperations(["z.ts", "a.ts"], []),
      "\n\n<read-files>\nz.ts\na.ts\n</read-files>",
    );
  });
});

describe("computeFileLists", () => {
  it("splits read-only files from modified (edited ∪ written), both sorted", () => {
    const out = computeFileLists({
      read: new Set(["z.ts", "shared.ts", "a.ts"]),
      written: new Set(["w.ts", "shared.ts"]),
      edited: new Set(["e.ts"]),
    });
    assert.deepEqual(out, {
      readFiles: ["a.ts", "z.ts"],
      modifiedFiles: ["e.ts", "shared.ts", "w.ts"],
    });
  });

  it("dedupes a file both edited and written", () => {
    const out = computeFileLists({
      read: new Set(),
      written: new Set(["x.ts"]),
      edited: new Set(["x.ts"]),
    });
    assert.deepEqual(out, { readFiles: [], modifiedFiles: ["x.ts"] });
  });

  it("returns empty lists for empty ops", () => {
    assert.deepEqual(
      computeFileLists({
        read: new Set(),
        written: new Set(),
        edited: new Set(),
      }),
      { readFiles: [], modifiedFiles: [] },
    );
  });
});

describe("buildStoredSummary", () => {
  it("is preamble + summary + file tags (upstream generateBranchSummary order)", () => {
    const summary = "## Goal\nx\n\n## Next Steps\n1. y";
    assert.equal(
      buildStoredSummary(summary, ["r.ts"], ["m.ts"]),
      `${BRANCH_SUMMARY_PREAMBLE}${summary}\n\n<read-files>\nr.ts\n</read-files>\n\n<modified-files>\nm.ts\n</modified-files>`,
    );
  });

  it("omits the file tags when no files were touched", () => {
    assert.equal(
      buildStoredSummary("## Goal\nx", [], []),
      `${BRANCH_SUMMARY_PREAMBLE}## Goal\nx`,
    );
  });
});
