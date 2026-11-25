import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { appendFileSync } from "node:fs";

/**
 * rewind-repro recorder: logs every tool call to $PI_REPRO_LOG and blocks the
 * ones that would touch the machine, so replaying a real session cannot create
 * worktrees, start containers, push, or gate. Reads stay live.
 *
 * `navigate_tree` stays live too: the end-to-end protocol needs the model to
 * drive a real rewind.
 *
 * PI_REPRO_CODEMODE=1 (run_e2e.sh --codemode on|only) also lets the top-level
 * `codemode` call run. Its script's tool calls reach this same gate as nested
 * calls, with `parentToolCallId` set to the script's call, so the allowlist
 * still decides what runs; the log line carries that id as `parent`. A nested
 * `codemode` call is blocked (pi refuses it anyway). The gate does not see
 * `models.*` (classifiers, image generation): they are not tool calls, so a
 * script can run them. They cannot touch the machine, but they can cost money.
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
  // PI_REPRO_CUSTOM_PROMPT=1 (run_e2e.sh --request custom): deliver the rewind
  // request as a custom message, not a user message, so no user entry is added
  // to the collapsed segment — as when the model rewinds on the hint.
  pi.on("input", async (event: any) => {
    if (process.env.PI_REPRO_CUSTOM_PROMPT !== "1")
      return { action: "continue" };
    if (event.source === "extension") return { action: "continue" };
    pi.sendMessage(
      {
        customType: "repro-rewind-request",
        content: event.text,
        display: true,
      },
      { triggerTurn: true },
    );
    // Print mode exits when the prompt returns: hold it until the run settles.
    await new Promise<void>((resolve) => {
      pi.on("agent_settled", () => resolve());
    });
    return { action: "handled" };
  });
  pi.on("tool_call", (event: any) => {
    const parent =
      typeof event.parentToolCallId === "string" &&
      event.parentToolCallId.length > 0
        ? event.parentToolCallId
        : undefined;
    const log = process.env.PI_REPRO_LOG;
    if (log) {
      const line = JSON.stringify({
        tool: event.toolName,
        id: event.toolCallId,
        ...(parent ? { parent } : {}),
        input: event.input,
      });
      appendFileSync(log, `${line}\n`);
    }
    if (ALLOWED.has(event.toolName)) return;
    if (
      process.env.PI_REPRO_CODEMODE === "1" &&
      event.toolName === "codemode" &&
      parent === undefined
    ) {
      return;
    }
    return {
      block: true,
      reason:
        "repro mode: this run records tool calls instead of executing them. Nothing was run. Continue.",
    };
  });
}
