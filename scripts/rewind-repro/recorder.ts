import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { appendFileSync } from "node:fs";

/**
 * rewind-repro recorder: logs every tool call to $PI_REPRO_LOG and blocks the
 * ones that would touch the machine, so replaying a real session cannot create
 * worktrees, start containers, push, or gate. Reads stay live.
 *
 * `navigate_tree` stays live too: the end-to-end protocol needs the model to
 * drive a real rewind.
 */
const ALLOWED = new Set([
  "read",
  "grep",
  "glob",
  "ls",
  "find",
  "list",
  "web_search",
  "navigate_tree",
]);

export default function (pi: ExtensionAPI) {
  pi.on("tool_call", (event: any) => {
    const log = process.env.PI_REPRO_LOG;
    if (log) {
      appendFileSync(log, JSON.stringify({ tool: event.toolName, input: event.input }) + "\n");
    }
    if (ALLOWED.has(event.toolName)) return;
    return {
      block: true,
      reason:
        "repro mode: this run records tool calls instead of executing them. Nothing was run. Continue.",
    };
  });
}
