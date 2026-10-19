/**
 * summary-prompt — the in-band rewind summary instruction (#80, #90).
 *
 * `rewind` does not issue a separate summarization request. Its tool result
 * only reports that the rewind was accepted; the instruction below arrives
 * right after it as a `navigate_tree` custom message (#90), so the live model
 * writes the summary as the next turn of the live conversation — a plain
 * append, served from the provider's prompt cache. `index.ts` then branches
 * from the anchor with that text.
 */

/**
 * The eval-approved "r5d" branch-summary instruction, byte-exact from
 * `cad0p/pi@eval/branch-summary-prompt`
 * `packages/coding-agent/src/core/compaction/branch-summarization.ts`
 * (`BRANCH_SUMMARY_PROMPT`). Do NOT reflow, re-wrap, or "fix" the wording:
 * the r5d text was selected by a live eval and the `{first}` scope sentence
 * is what keeps pre-branch background out of the summary. `{first}` is
 * substituted (via `replaceAll`) with the 1-based number of the first
 * collapsed message in the projection the model sees.
 */
export const REWIND_SUMMARY_PROMPT = `Summarize only messages {first} onwards in the conversation above (message numbering starts at 1 and excludes the system prompt; this instruction message itself is not evidence). Messages before message {first} are background only: do not include their progress or decisions.

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

/**
 * Compose the summary instruction: r5d with `{first}` substituted, plus the
 * focus. Mirrors upstream's `customInstructions` append shape
 * (`${PROMPT}\n\nAdditional focus: ...`).
 */
export function buildSummaryInstruction(first: number, focus: string): string {
  // Substitute in the prompt only: a focus that quotes `{first}` stays
  // verbatim.
  const prompt = REWIND_SUMMARY_PROMPT.replaceAll("{first}", String(first));
  return `${prompt}\n\nAdditional focus: ${focus}`;
}

/**
 * The content of the `navigate-tree-summary-request` custom message (#90):
 * a `[navigate_tree]` source line, then the instruction. The message reaches
 * the provider as a user message right after the rewind's tool result.
 */
export function buildSummaryRequestText(first: number, focus: string): string {
  return `[navigate_tree]\n${buildSummaryInstruction(first, focus)}`;
}

/**
 * Byte-exact copy of upstream's (non-exported) `BRANCH_SUMMARY_PREAMBLE`
 * (`core/compaction/branch-summarization.js`), so a stored in-band summary
 * has the same shape as one produced by `generateBranchSummary`.
 */
export const BRANCH_SUMMARY_PREAMBLE = `The user explored a different conversation branch before returning here.
Summary of that exploration:

`;

/**
 * Byte-exact port of upstream's (non-exported) `formatFileOperations`
 * (`core/compaction/utils.js`): file lists appended to the stored summary.
 */
export function formatFileOperations(
  readFiles: string[],
  modifiedFiles: string[],
): string {
  const sections: string[] = [];
  if (readFiles.length > 0) {
    sections.push(`<read-files>\n${readFiles.join("\n")}\n</read-files>`);
  }
  if (modifiedFiles.length > 0) {
    sections.push(
      `<modified-files>\n${modifiedFiles.join("\n")}\n</modified-files>`,
    );
  }
  if (sections.length === 0) return "";
  return `\n\n${sections.join("\n\n")}`;
}

/**
 * Port of upstream's (non-exported) `computeFileLists`: read-only files
 * (read but never edited/written) and modified files, both sorted.
 */
export function computeFileLists(fileOps: {
  read: Set<string>;
  written: Set<string>;
  edited: Set<string>;
}): { readFiles: string[]; modifiedFiles: string[] } {
  const modified = new Set([...fileOps.edited, ...fileOps.written]);
  const readFiles = [...fileOps.read].filter((f) => !modified.has(f)).sort();
  const modifiedFiles = [...modified].sort();
  return { readFiles, modifiedFiles };
}

/** Compose the stored branch_summary text from the model's summary turn. */
export function buildStoredSummary(
  summaryText: string,
  readFiles: string[],
  modifiedFiles: string[],
): string {
  return (
    BRANCH_SUMMARY_PREAMBLE +
    summaryText +
    formatFileOperations(readFiles, modifiedFiles)
  );
}
