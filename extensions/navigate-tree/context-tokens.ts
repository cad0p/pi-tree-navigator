/**
 * context-tokens — local `estimateContextTokens` parity helper.
 *
 * ## Why this exists
 *
 * `@earendil-works/pi-agent-core` exported `estimateContextTokens` through
 * 0.99.2 but dropped it from the root export in **1.0.0** (its runtime keys
 * are now `Agent`, `agentLoop`, `agentLoopContinue`, `runAgentLoop`,
 * `runAgentLoopContinue`, `runToolCall`, `setDefaultStreamFn`, and
 * `streamProxy`). `@earendil-works/pi-coding-agent` never re-exported it at
 * the root in any version (0.87.1 / 0.99.2 / 1.0.0); its implementation
 * lives at `dist/core/compaction/compaction.js`, but the package `exports`
 * map allows only `.` and `./rpc-entry`, so it is not importable. Importing
 * the old name as a runtime value therefore left it `undefined` on pi
 * 1.0.0, and every `navigate_tree` action threw
 * `(0, _piAgentCore.estimateContextTokens) is not a function`.
 *
 * This module reproduces the removed helper's semantics on top of two
 * `@earendil-works/pi-coding-agent` root exports that are public across the
 * whole peer range (≥0.81.0, the peer floor):
 *   • `calculateContextTokens(usage)` — provider usage → context tokens.
 *   • `estimateTokens(message)` — per-message chars/4 heuristic.
 *
 * Semantics (identical to the removed helper): walk newest→oldest for the
 * first assistant message with a *valid* usage — `stopReason` neither
 * `"aborted"` nor `"error"` and `calculateContextTokens(usage) > 0` — and
 * return that baseline plus `estimateTokens` for every message after it.
 * With no valid usage anywhere, sum `estimateTokens` over every message.
 *
 * ## Known divergences (no cross-version numeric parity claim)
 *
 * - **System messages.** pi 1.0.0's `estimateTokens` adds a `system` case
 *   (counts `content`, `sections`, and `toolsAdded`) where 0.84.2 returned
 *   `0`. `buildSessionContext(...).messages` carries `system` entries on
 *   1.0.0, so an estimate is self-consistent per host and matches pi's own
 *   `estimateProjectedContextTokens` there — but a chain containing system
 *   entries does not produce the same number on both versions. No
 *   cross-version parity is claimed.
 * - **`JSON.stringify` vs `safeJsonStringify`.** The removed pi-agent-core
 *   helper wrapped toolCall `arguments` serialization in a
 *   `safeJsonStringify` guard (falling back to `"[unserializable]"`); the
 *   public `estimateTokens` uses plain `JSON.stringify` (0.84.2 and 1.0.0
 *   alike — v1 restricts toolCall arguments to JSON-compatible values).
 *   The guard is deliberately not reproduced.
 */

import type { AgentMessage } from "@earendil-works/pi-agent-core";
import {
  calculateContextTokens,
  estimateTokens,
} from "@earendil-works/pi-coding-agent";

type Usage = Parameters<typeof calculateContextTokens>[0];

/**
 * Valid assistant usage for baseline purposes: `stopReason` not
 * `"aborted"`/`"error"` and a nonzero context-token count. Mirrors the
 * removed pi-agent-core helper's `getAssistantUsage` predicate exactly.
 */
function validAssistantUsage(message: AgentMessage): Usage | undefined {
  if (message.role !== "assistant") return undefined;
  const assistant = message as {
    stopReason?: string;
    usage?: Usage;
  };
  if (
    assistant.stopReason !== "aborted" &&
    assistant.stopReason !== "error" &&
    assistant.usage &&
    calculateContextTokens(assistant.usage) > 0
  ) {
    return assistant.usage;
  }
  return undefined;
}

/**
 * Total estimated context tokens for `messages`.
 *
 * Uses the newest valid assistant usage as the baseline plus
 * `estimateTokens` for every later message (the newest usage covers
 * everything before it); with no valid usage, sums `estimateTokens` over
 * all messages. See the file JSDoc for the version boundary and the
 * known system-message / `safeJsonStringify` divergences.
 */
export function estimateContextTokens(messages: AgentMessage[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    const usage = validAssistantUsage(messages[i]);
    if (!usage) continue;
    let trailingTokens = 0;
    for (let j = i + 1; j < messages.length; j++) {
      trailingTokens += estimateTokens(messages[j]);
    }
    return calculateContextTokens(usage) + trailingTokens;
  }

  let estimated = 0;
  for (const message of messages) {
    estimated += estimateTokens(message);
  }
  return estimated;
}
