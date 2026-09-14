#!/usr/bin/env node
/**
 * summary-quality-check — offline structural smoke for `navigate_tree`
 * rewind summaries.
 *
 * Usage:
 *   node scripts/summary-quality-check.mjs <session.jsonl> [--all]
 *
 * The unit suite proves the summary *request* is shaped live-identically,
 * but it cannot prove the model's output is scope-clean. This script is the
 * second half of the live gate (see README "Live summary verification"):
 * run it on the session file produced by a real `rewind` and eyeball the
 * printed summary.
 *
 * Structural checks per `branch_summary` entry:
 *   - pi's standard preamble is allowed before the first heading, nothing else
 *     (the r5d prompt forbids model preamble/commentary);
 *   - the eight r5d headings are present in order;
 *   - total length <= 4500 chars (r5d asks for "under about 4000");
 *   - no leftover `{first}` placeholder.
 * It also prints the newest rewind tool-result `details.summaryCache` block
 * when present (provider-measured cache evidence).
 *
 * Scope is deliberately NOT asserted here: "describes only the collapsed
 * branch, background excluded, work not continued" needs a reader. The
 * script prints the full summary for that eyeball check.
 *
 * Exit code: 0 = all checked summaries pass; 1 = a check failed or no
 * `branch_summary` entry was found.
 */

import { readFileSync } from "node:fs";

/** pi's `BRANCH_SUMMARY_PREAMBLE`, prepended to every stored summary. */
const PREAMBLE =
  "The user explored a different conversation branch before returning here.\nSummary of that exploration:\n\n";

const HEADINGS = [
  "## Goal",
  "## Constraints & Preferences",
  "## Progress",
  "### Done",
  "### In Progress",
  "### Blocked",
  "## Key Decisions",
  "## Next Steps",
];

const MAX_SUMMARY_CHARS = 4500;

function fail(message) {
  console.error(`FAIL  ${message}`);
  return false;
}

/**
 * @param {string} summary stored branch_summary text (preamble + model text + file-ops appendix)
 * @param {string} label entry id for messages
 * @returns {boolean}
 */
function checkSummary(summary, label) {
  let ok = true;
  if (typeof summary !== "string" || summary.trim().length === 0) {
    return fail(`${label}: empty summary`);
  }
  const body = summary.startsWith(PREAMBLE)
    ? summary.slice(PREAMBLE.length)
    : summary;
  if (summary.startsWith(PREAMBLE)) {
    console.log(`ok    ${label}: standard preamble present`);
  } else {
    ok = fail(`${label}: missing pi summary preamble`) && ok;
  }
  const firstHeading = body.indexOf("## Goal");
  if (firstHeading === -1) {
    ok = fail(`${label}: no "## Goal" heading`) && ok;
  } else if (body.slice(0, firstHeading).trim().length > 0) {
    ok = fail(
      `${label}: model preamble/commentary before "## Goal": ${JSON.stringify(
        body.slice(0, firstHeading).trim().slice(0, 120),
      )}`,
    );
  }
  let cursor = -1;
  for (const heading of HEADINGS) {
    const at = body.indexOf(heading);
    if (at === -1) {
      ok = fail(`${label}: missing heading "${heading}"`) && ok;
    } else if (at < cursor) {
      ok = fail(`${label}: heading "${heading}" out of order`) && ok;
    } else {
      cursor = at;
    }
  }
  if (body.includes("{first}")) {
    ok = fail(`${label}: unresolved "{first}" placeholder`) && ok;
  }
  if (summary.length > MAX_SUMMARY_CHARS) {
    ok = fail(
      `${label}: ${summary.length} chars > ${MAX_SUMMARY_CHARS} (r5d target ~4000)`,
    );
  } else {
    console.log(`ok    ${label}: ${summary.length} chars`);
  }
  return ok;
}

function main() {
  const args = process.argv.slice(2);
  const all = args.includes("--all");
  const file = args.find((a) => !a.startsWith("--"));
  if (!file) {
    console.error(
      "usage: node scripts/summary-quality-check.mjs <session.jsonl> [--all]",
    );
    process.exit(2);
  }

  const entries = readFileSync(file, "utf8")
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    })
    .filter(Boolean);

  const summaries = entries.filter((e) => e.type === "branch_summary");
  if (summaries.length === 0) {
    console.error(`FAIL  no branch_summary entry in ${file}`);
    process.exit(1);
  }

  const cacheBlocks = [];
  for (const e of entries) {
    if (e.type !== "message") continue;
    const details = e.message?.details;
    if (details?.summaryCache) cacheBlocks.push(details.summaryCache);
  }
  if (cacheBlocks.length > 0) {
    const latest = cacheBlocks[cacheBlocks.length - 1];
    console.log(
      `cache ${JSON.stringify(latest)}`,
    );
  } else {
    console.log("cache (no details.summaryCache found in this session)");
  }

  const checked = all ? summaries : [summaries[summaries.length - 1]];
  let ok = true;
  for (const entry of checked) {
    const label = `branch_summary ${String(entry.id ?? "?").slice(0, 8)}`;
    ok = checkSummary(entry.summary, label) && ok;
  }

  console.log("--- summary text (eyeball scope: branch only, background excluded, work not continued) ---");
  for (const entry of checked) {
    console.log(entry.summary);
    console.log("--- end ---");
  }
  process.exit(ok ? 0 : 1);
}

main();
