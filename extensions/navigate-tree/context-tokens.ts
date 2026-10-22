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
 * lives at `dist/core/compaction/compaction.js`, but it is not reachable
 * through the package `exports` map (on 0.87.1+: root, `./rpc-entry`, and
 * source-only subpaths only), so it is not importable. Importing
 * the old name as a runtime value therefore left it `undefined` on pi
 * 1.0.0, and every `navigate_tree` action threw
 * `(0, _piAgentCore.estimateContextTokens) is not a function`.
 *
 * This module reproduces the removed helper's semantics on top of two
 * `@earendil-works/pi-coding-agent` root exports that are public across the
 * whole peer range (≥0.87.0, the peer floor):
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
 *   1.0.0, so an estimate is self-consistent per host — but a chain
 *   containing system entries does not produce the same number on both
 *   versions. No cross-version parity is claimed. Nor is parity with pi's
 *   `estimateProjectedContextTokens` on the no-usage path: pi folds the
 *   system entries into the current system state (`getCurrentSystemMessage`)
 *   while this helper sums every entry, so replaced or removed sections and
 *   tool changes count here but not in pi.
 * - **`JSON.stringify` vs `safeJsonStringify`.** The removed pi-agent-core
 *   helper wrapped toolCall `arguments` serialization in a
 *   `safeJsonStringify` guard (falling back to `"[unserializable]"`); the
 *   public `estimateTokens` uses plain `JSON.stringify` (0.84.2 and 1.0.0
 *   alike — v1 restricts toolCall arguments to JSON-compatible values).
 *   The guard is deliberately not reproduced.
 * - **Stale usage baseline after `context_edit` / compaction.** Both this
 *   helper and pi's `estimateProjectedContextTokens` (behind
 *   `ctx.getContextUsage()`) estimate the projected messages, but pi drops
 *   the last assistant usage as a baseline when it sits at or before the
 *   latest `context_edit` or `compaction` entry; this helper keeps it. On
 *   such a branch an anchor percentage (this helper) can exceed what pi
 *   reports for the same point (after compaction pi reports `tokens: null`
 *   until the next assistant usage). Moving live-branch numbers onto pi's
 *   estimate is tracked in #64.
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
 * `estimateContextTokens` split at its baseline (#101, A): `baseline` is the
 * newest valid assistant usage (0 with none) and `tail` the `estimateTokens`
 * sum of the messages after it (all messages with none). The total is
 * `baseline + tail`; the withholding planner scales `tail` alone, because
 * the provider's real count can only disagree with the estimate there.
 */
export function splitContextEstimate(messages: AgentMessage[]): {
  baseline: number;
  tail: number;
} {
  const anchor = newestUsageIndex(messages);
  const usage = anchor >= 0 ? validAssistantUsage(messages[anchor]) : undefined;
  let tail = 0;
  for (let j = anchor + 1; j < messages.length; j++) {
    tail += estimateTokens(messages[j]);
  }
  return {
    baseline: usage ? calculateContextTokens(usage) : 0,
    tail,
  };
}

/**
 * Index of the newest message carrying a valid assistant usage (the baseline
 * of `splitContextEstimate`), or -1 with none. Messages at or before it are
 * already counted by its real usage.
 */
export function newestUsageIndex(messages: AgentMessage[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (validAssistantUsage(messages[i])) return i;
  }
  return -1;
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
  const { baseline, tail } = splitContextEstimate(messages);
  return baseline + tail;
}

// ---------------------------------------------------------------------------
// pi-ai's own per-message estimator (#107, round 3)
// ---------------------------------------------------------------------------

/**
 * pi-ai's constants (`dist/utils/estimate.js` on 1.3.0 and 1.4.0, the two
 * versions this port was byte-verified against).
 */
const PI_CHARS_PER_TOKEN = 3.5;
const PI_ESTIMATED_IMAGE_CHARS = 4_800;

/** pi-ai's `safeJsonStringify`: a throwing `JSON.stringify` degrades instead. */
function piSafeJsonStringify(value: unknown): string {
  try {
    return JSON.stringify(value) ?? "undefined";
  } catch {
    return "[unserializable]";
  }
}

/** pi-ai's `estimateTextTokens`: chars/3.5, rounded up. */
function piEstimateTextTokens(chars: number): number {
  return Math.ceil(chars / PI_CHARS_PER_TOKEN);
}

/**
 * pi-ai's `estimateTextAndImageContentChars`: string content is its length;
 * an array sums text lengths and counts every other block as one image.
 */
function piContentChars(content: unknown): number {
  if (typeof content === "string") return content.length;
  if (!Array.isArray(content)) return 0;
  let chars = 0;
  for (const block of content as Array<{ type?: string; text?: string }>) {
    chars +=
      block?.type === "text"
        ? (block.text ?? "").length
        : PI_ESTIMATED_IMAGE_CHARS;
  }
  return chars;
}

/**
 * pi-ai's `getSystemMessageText`: the content text followed by every non-null
 * section, the non-empty parts joined by a blank line.
 */
function piSystemMessageText(message: {
  content?: unknown;
  sections?: Record<string, string | null>;
}): string {
  const content = message.content;
  const text =
    typeof content === "string"
      ? content
      : Array.isArray(content)
        ? (content as Array<{ type?: string; text?: string }>)
            .filter((block) => block?.type === "text")
            .map((block) => block.text ?? "")
            .join("\n")
        : "";
  const parts = [text];
  for (const section of Object.values(message.sections ?? {})) {
    if (section !== null) parts.push(section);
  }
  return parts.filter((part) => part.length > 0).join("\n\n");
}

/** pi-ai's `estimateToolsTokens`: the JSON of the tool list, or 0. */
function piToolsTokens(tools: unknown): number {
  if (!Array.isArray(tools) || tools.length === 0) return 0;
  return piEstimateTextTokens(piSafeJsonStringify(tools).length);
}

/**
 * Faithful port of pi-ai's `estimateMessageTokens` (#107, round 3).
 *
 * pi sizes the output clamp from THIS estimator — `clampMaxTokensToContext`
 * computes `contextWindow - estimateContextTokens(context) - 4096` — while
 * pi-compaction's `estimateTokens` (what `splitContextEstimate` uses) is
 * chars/4. The 4 -> 3.5 gap is what starved the forced summary turn: pi saw
 * 37,518 tokens for a 31,530-token request and clamped `max_tokens` to 1.
 * The withholding planner therefore has to fit BOTH views.
 *
 * Byte-verified against `@earendil-works/pi-ai/dist/utils/estimate.js` on
 * 1.3.0 and 1.4.0 (the two files are identical): chars/3.5 `Math.ceil`, an
 * image block counts 4800 chars, `system` uses the section text plus the
 * `toolsAdded`/`toolsRemoved` JSON, `user`/`toolResult` use text+image
 * content, and an assistant sums text + thinking + tool-call name + JSON
 * arguments. Only LLM roles are handled: a non-LLM agent message (a custom
 * message, a branch/compaction summary) is converted by pi's `convertToLlm`
 * before the clamp ever sees it, so callers must convert first.
 */
export function piEstimateMessageTokens(message: AgentMessage): number {
  const role = (message as { role?: string }).role;
  if (role === "system") {
    const system = message as unknown as {
      content?: unknown;
      sections?: Record<string, string | null>;
      toolsAdded?: unknown;
      toolsRemoved?: unknown;
    };
    return (
      piEstimateTextTokens(piSystemMessageText(system).length) +
      piToolsTokens(system.toolsAdded) +
      piToolsTokens(system.toolsRemoved)
    );
  }
  if (role === "user" || role === "toolResult") {
    return piEstimateTextTokens(
      piContentChars((message as { content?: unknown }).content),
    );
  }
  const content = (message as { content?: unknown }).content;
  let chars = 0;
  for (const block of Array.isArray(content)
    ? (content as Array<{
        type?: string;
        text?: string;
        thinking?: string;
        name?: string;
        arguments?: unknown;
      }>)
    : []) {
    if (block?.type === "text") {
      chars += (block.text ?? "").length;
    } else if (block?.type === "thinking") {
      chars += (block.thinking ?? "").length;
    } else {
      chars +=
        (block?.name ?? "").length +
        piSafeJsonStringify(block?.arguments).length;
    }
  }
  return piEstimateTextTokens(chars);
}
