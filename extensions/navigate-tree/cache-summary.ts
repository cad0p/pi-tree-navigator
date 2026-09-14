/**
 * cache-summary — cache-preserving branch-summary request construction.
 *
 * ## Why this exists
 *
 * `rewind` collapses a conversation segment by calling pi's upstream
 * `generateBranchSummary`, which builds a *cold, standalone* request: a
 * generic summarization system prompt, the conversation serialized to a
 * text blob, no tools, `cacheRetention: "none"`, a fresh session id, and
 * no reasoning forwarding. The live turns the summary covers were just
 * prompt-cache-served, so the summary re-bills the entire branch input
 * (measured ~77k tokens cold vs a few hundred warm). The fix exists in
 * the upstream fork (`cad0p/pi` PR #3) but cannot be imported: pi's
 * extension loader aliases `@earendil-works/*` to the host process's own
 * modules, so an extension can never ship a patched coding-agent.
 *
 * ## How
 *
 * `index.ts` already injects a `streamFn` into `generateBranchSummary`
 * (for custom-provider routing). That seam receives the fully built
 * `(model, context, options)` triple *after* upstream applied its cold
 * choices — including `completeSummarization`'s forced
 * `cacheRetention: "none"` + fresh `sessionId` — and before the wire
 * call. `createCachePreservingStreamFn` replaces that triple with the
 * live request shape: the session's own system prompt, tool array, and
 * session id, the conversation as structured `Message`s (so the bytes
 * prefix-match the live turns), and the same cache/reasoning params live
 * turns send.
 *
 * ## Residual risks (see README "Limitations")
 *
 *  - Every param this module does not mirror is a silent cache miss:
 *    the summary still runs, just cold (and a cold *structured* request
 *    can bill more than branch-only evidence). The extension measures the
 *    summary response with pi's own miss detector and, when it clears the
 *    display floor, records the notice string in `details.summaryCache`
 *    (gated by `showCacheMissNotices`); `index.ts`'s `renderResult` renders
 *    it as a TUI transcript line. Neither surface reaches the model.
 *  - The request depends on plain (non-`#`-private) pi internals for
 *    `systemPrompt` / `tools`; `index.ts` falls back to the cold request
 *    when any live input is unavailable.
 */

import type {
  AgentTool,
  StreamFn,
  ThinkingLevel,
} from "@earendil-works/pi-agent-core";
import {
  convertToLlm,
  estimateTokens,
  type SessionEntry,
  sessionEntryToContextMessages,
} from "@earendil-works/pi-coding-agent";

// ---------------------------------------------------------------------------
// Wire-message type
//
// `Message` / `Usage` live only in `@earendil-works/pi-ai`, which is NOT a
// peer dependency of this package (it's a transitive dep of the pi packages
// themselves). Importing it would either add an undeclared dependency or
// resolve to a duplicated instance under a different node_modules root. Both
// types are fully structural, so derive them:
//   - `convertToLlm`'s return element type IS the wire `Message` union;
//   - usage need only these three counters for cache accounting.
// ---------------------------------------------------------------------------

/** LLM-compatible wire message, derived from pi's own `convertToLlm`. */
export type WireMessage = ReturnType<typeof convertToLlm>[number];

/**
 * Structural subset of pi-ai's `Usage` this module needs. `input` counts
 * FRESH (uncached) tokens only on cache-serving providers, which is why the
 * cache-hit metric is `cacheRead > 0` rather than `cacheRead/input`.
 */
export interface SummaryCacheUsage {
  input: number;
  cacheRead: number;
  cacheWrite: number;
}

// ---------------------------------------------------------------------------
// Prompt
// ---------------------------------------------------------------------------

/**
 * The eval-approved "r5d" branch-summary instruction, byte-exact from
 * `cad0p/pi@eval/branch-summary-prompt`
 * `packages/coding-agent/src/core/compaction/branch-summarization.ts`
 * (`BRANCH_SUMMARY_PROMPT`). Do NOT reflow, re-wrap, or "fix" the wording:
 * the r5d text was selected by a live eval and the `{first}` scope sentence
 * is what keeps pre-branch background out of the summary. `{first}` is
 * substituted (via `replaceAll`) with the strip-adjusted 1-based number of
 * the first branch message after the payload is final.
 *
 * The fallback (cold) path keeps using upstream's older branch prompt;
 * divergence is intentional (the fallback is a degraded path) and documented
 * in the README.
 */
export const BRANCH_SUMMARY_CACHE_PROMPT = `Summarize only messages {first} onwards in the conversation above (message numbering starts at 1 and excludes the system prompt; this instruction message itself is not evidence). Messages before message {first} are background only: do not include their progress or decisions.

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
 * Compose the trailing instruction message text. Mirrors upstream's
 * `customInstructions` append shape (`${PROMPT}\n\nAdditional focus: ...`),
 * so the fallback and cache paths differ only in prompt body + payload
 * shaping, not in how `summaryFocus` is conveyed.
 */
export function buildSummaryInstruction(focus: string): string {
  return `${BRANCH_SUMMARY_CACHE_PROMPT}\n\nAdditional focus: ${focus}`;
}

// ---------------------------------------------------------------------------
// Payload construction
// ---------------------------------------------------------------------------

/**
 * Drop `toolResult` messages whose matching assistant `toolCall` is not in
 * the payload (a branch cut between a call and its result; compaction
 * boundaries can also split them). Providers reject result blocks that
 * reference calls outside the request, so the structured summary request
 * must strip them. Port of the fork's `stripBoundaryOrphanToolResults`:
 * preserves order, never mutates, and preserves element identity (callers
 * use identity to count how many stripped messages preceded the branch).
 */
export function stripBoundaryOrphanToolResults(
  messages: WireMessage[],
): WireMessage[] {
  const callIds = new Set<string>();
  for (const message of messages) {
    if (message.role !== "assistant") continue;
    for (const block of message.content) {
      if (block.type === "toolCall") callIds.add(block.id);
    }
  }
  return messages.filter((message) => {
    if (message.role !== "toolResult") return true;
    return callIds.has(message.toolCallId);
  });
}

/**
 * Newest index of an assistant entry whose content carries a `toolCall` with
 * `inFlightToolCallId`, or -1 when none exists. Searches from the end because
 * sequential execution can leave sibling `toolResult` entries after the
 * assistant that owns the in-flight call.
 */
function findInFlightAssistantIndex(
  entries: SessionEntry[],
  inFlightToolCallId: string,
): number {
  for (let i = entries.length - 1; i >= 0; i--) {
    const entry = entries[i];
    if (
      entry.type === "message" &&
      entry.message.role === "assistant" &&
      Array.isArray(entry.message.content) &&
      entry.message.content.some(
        (block) => block.type === "toolCall" && block.id === inFlightToolCallId,
      )
    ) {
      return i;
    }
  }
  return -1;
}

export interface BuildLiveSummaryArgs {
  /**
   * The live projection of the active branch (`sessionManager
   * .buildContextEntries()`), i.e. the exact entries the live turns send.
   * Using the projection (rather than reconstructing prefix+branch) is what
   * makes the resulting payload byte-identical to the previous live
   * request's message list.
   */
  contextEntries: SessionEntry[];
  /**
   * Ids of the entries being collapsed (`collectEntriesForBranchSummary`).
   * Messages from other entries are pre-branch background: sent for cache
   * prefix matching only, excluded from the summary via the `{first}` scope
   * sentence.
   */
  branchEntryIds: Set<string>;
  /**
   * Id of the tool call whose assistant message triggered this rewind. That
   * assistant entry was never part of any cached live prefix (it is the
   * response being streamed), and an unpaired `tool_use` immediately
   * followed by a user message is rejected by Anthropic. The newest retained
   * assistant entry carrying a `toolCall` with this id is removed by index —
   * NOT merely from the tail: `navigate_tree` runs `executionMode:
   * "sequential"`, so pi-agent-core appends each sibling `toolResult` before
   * the next call executes and a sibling result can follow this assistant.
   * Dropping the assistant makes the retained history byte-identical to the
   * previous live request.
   */
  inFlightToolCallId: string;
  /** Context window minus the response reserve (upstream default 16384). */
  tokenBudget: number;
  /** `summaryFocus` from the tool call. */
  focus: string;
}

export interface LiveSummaryMessages {
  /** Structured history (stripped) + the trailing instruction message. */
  messages: WireMessage[];
  /**
   * 1-based number of the first branch message in `messages` (numbering
   * excludes the system prompt; the instruction itself is not evidence).
   * Substituted into `{first}`.
   */
  first: number;
  /**
   * False when no retained entry belongs to the collapsed branch: a
   * labels-only segment, the newest message alone exceeding the budget, or
   * the branch start being dropped by compaction. `first` is then 1 — every
   * retained message is background and gets summarized. The index.ts call
   * site now treats `false` as a real fallback (reason
   * `"branch-start-not-retained"`), so a live-prefix request always carries
   * `true`; the flag is kept in `details.summaryCache` for diagnostics. A
   * retained survivor by definition implies a hit, so there is no clamp
   * step.
   */
  branchStartRetained: boolean;
}

/**
 * Build the cache-preserving summary payload.
 *
 * Walk the live projection newest→oldest, dropping oldest entries first when
 * over budget (a truncated request no longer prefix-matches live turns; the
 * system prompt + tools still do). Summary entries (`compaction` /
 * `branch_summary`) get upstream's 0.9-slack retry so they survive
 * truncation when they are the thing that must not be lost. Then strip
 * boundary-orphan tool results and adjust `{first}` by however many stripped
 * messages preceded the branch start, so the instruction's numbering always
 * matches the payload actually sent.
 */
export function buildLiveSummaryMessages(
  args: BuildLiveSummaryArgs,
): LiveSummaryMessages {
  const {
    contextEntries,
    branchEntryIds,
    inFlightToolCallId,
    tokenBudget,
    focus,
  } = args;

  // --- in-flight assistant exclusion (must happen before anything else) ---
  // Search the WHOLE retained array, not just the tail. `navigate_tree`
  // declares `executionMode: "sequential"`, so pi-agent-core runs the batch
  // through `executeToolCallsSequential`: calls execute in order and each
  // `toolResult` is appended before the next call executes. When a sibling
  // tool call precedes the rewind call in the same assistant turn, the last
  // session entry is that sibling's `toolResult` — not the assistant — so a
  // tail-only check would leave the assistant (and its unpaired `tool_use`)
  // in the payload and Anthropic would reject the summary request. Remove the
  // assistant at its index; the sibling `toolResult`s that follow then have
  // no matching call and are dropped by `stripBoundaryOrphanToolResults`
  // below (single removal path — do not add a second one here).
  const retained = contextEntries.slice();
  const excludedAt = findInFlightAssistantIndex(retained, inFlightToolCallId);
  if (excludedAt >= 0) retained.splice(excludedAt, 1);

  // --- newest→oldest walk with the upstream token budget ---
  const evidence: WireMessage[] = [];
  const inBranch: boolean[] = [];
  let totalTokens = 0;
  for (let i = retained.length - 1; i >= 0; i--) {
    const entry = retained[i];
    const entryMessages = sessionEntryToContextMessages(entry);
    let overBudget = false;
    for (let j = entryMessages.length - 1; j >= 0; j--) {
      const agentMessage = entryMessages[j];
      // convertToLlm is a pure per-message map+filter (verified against
      // 0.84.2 `messages.js`), so converting one message at a time keeps
      // the branch/background flag exact without diverging from what the
      // live loop produces for the same AgentMessage.
      const wire = convertToLlm([agentMessage]);
      if (wire.length === 0) continue;
      const tokens = estimateTokens(agentMessage);
      const fits = tokenBudget <= 0 || totalTokens + tokens <= tokenBudget;
      if (!fits) {
        // Summary entries are load-bearing context: upstream retries them
        // when under 90% of budget. Mirror that before giving up.
        if (
          (entry.type === "compaction" || entry.type === "branch_summary") &&
          totalTokens < tokenBudget * 0.9
        ) {
          evidence.unshift(...wire);
          inBranch.unshift(...wire.map(() => branchEntryIds.has(entry.id)));
          totalTokens += tokens;
        }
        overBudget = true;
        break;
      }
      evidence.unshift(...wire);
      inBranch.unshift(...wire.map(() => branchEntryIds.has(entry.id)));
      totalTokens += tokens;
    }
    if (overBudget) break;
  }

  const firstBranchIdx = inBranch.indexOf(true);
  const branchStartRetained = firstBranchIdx >= 0;
  // Pre-truncation counting would misnumber; count only retained messages
  // before the branch start.
  const firstRaw = branchStartRetained ? 1 + firstBranchIdx : 1;

  const stripped = stripBoundaryOrphanToolResults(evidence);
  const removedBeforeFirst = evidence
    .slice(0, firstRaw - 1)
    .filter((message) => !stripped.includes(message)).length;
  const first = Math.max(1, firstRaw - removedBeforeFirst);

  const instruction: WireMessage = {
    role: "user",
    content: [
      {
        type: "text",
        text: buildSummaryInstruction(focus).replaceAll(
          "{first}",
          String(first),
        ),
      },
    ],
    timestamp: Date.now(),
  };

  return { messages: [...stripped, instruction], first, branchStartRetained };
}

// ---------------------------------------------------------------------------
// Cache retention
// ---------------------------------------------------------------------------

/**
 * Mirror pi-ai's `resolveCacheRetention`: explicit env wins key-by-key, with
 * `process.env` as the fallback. Live turns default to `"short"`; the summary
 * must match or its single-use trailer breakpoints land differently and the
 * provider keys on a different retention class. Hence: resolved, never
 * hardcoded.
 */
export function resolveSummaryCacheRetention(
  env?: Record<string, string | undefined>,
): "short" | "long" {
  const value = env?.PI_CACHE_RETENTION ?? process.env.PI_CACHE_RETENTION;
  return value === "long" ? "long" : "short";
}

// ---------------------------------------------------------------------------
// Measurement + notice
// ---------------------------------------------------------------------------
// Cache-miss detection (behavioral port of pi's `cache-stats.js`)
//
// `detectCacheMiss` is a behavioral port of upstream pi's detector of the
// same name (`dist/core/cache-stats.js`): same scan (reset the baseline on
// `compaction` OR `branch_summary`, baseline-local sticky `reportedCache`),
// same `detectMiss` math (1024-token noise floor, cost rates, `modelChanged`).
// pi does not export it from the package barrel, so the extension carries its
// own copy. The display half (`formatCacheMissNotice` copy + thresholds)
// mirrors upstream `interactive-mode.js`'s `addCacheMissNotice`.
// ---------------------------------------------------------------------------

/**
 * Prompt-cache TTL: idle gaps longer than this are worth mentioning as the
 * likely cause of a miss. Anthropic's default cache TTL is 5 minutes.
 */
export const CACHE_TTL_MS = 5 * 60 * 1000;

/** Per-turn misses at or below this are cache breakpoint granularity noise. */
const NOISE_FLOOR_TOKENS = 1024;

/** Display floor: only misses at/above this many tokens warn. */
export const CACHE_MISS_DISPLAY_TOKENS = 20_000;
/** Display floor: only misses at/above this many dollars warn. */
export const CACHE_MISS_DISPLAY_COST = 0.1;

/** A counted cache miss on the just-completed request (summary or live turn). */
export interface CacheMiss {
  /** Prompt tokens in the previous request's prompt but not read from cache. */
  missedTokens: number;
  /** Extra dollars paid vs. a full cache hit; 0 when pricing is unknown. */
  missedCost: number;
  /** Milliseconds since the previous request (which last refreshed the cache). */
  idleMs: number;
  /** True when the model changed relative to the previous request. */
  modelChanged: boolean;
}

/** Minimal pricing lookup; cost is $/million tokens. Satisfied by ModelRegistry. */
export interface ModelPriceSource {
  getModel(
    provider: string,
    modelId: string,
  ): { cost?: { cacheRead?: number } } | undefined;
}

/** The last request seen by the scan; everything in its prompt should be cached. */
interface PreviousRequest {
  promptTokens: number;
  modelKey: string;
  timestamp: number;
  /**
   * Sticky within the current baseline: an earlier request since the last
   * context boundary reported cache activity. Distinguishes a total miss on a
   * cache-read-only provider from a provider that never reports caching at
   * all. Reset with the baseline on `compaction` / `branch_summary`.
   */
  reportedCache: boolean;
}

interface MissUsage {
  input: number;
  cacheRead: number;
  cacheWrite: number;
  cost?: { input?: number; cacheRead?: number; cacheWrite?: number };
}

export interface MissAssistantMessage {
  provider?: string;
  model?: string;
  usage: MissUsage;
  timestamp: number;
}

function modelKey(provider: string, model: string): string {
  return `${provider}/${model}`;
}

function detectMiss(
  prev: PreviousRequest | undefined,
  message: MissAssistantMessage,
  models: ModelPriceSource,
): CacheMiss | undefined {
  const usage = message.usage;
  const promptTokens = usage.input + usage.cacheRead + usage.cacheWrite;
  // A zero-cache turn only counts when cache activity was reported before:
  // on cache-read-only providers that is a total miss, while on providers
  // that never report caching it means nothing.
  if (
    !prev ||
    promptTokens <= 0 ||
    (usage.cacheRead + usage.cacheWrite === 0 && !prev.reportedCache)
  ) {
    return undefined;
  }

  const missedTokens =
    Math.min(prev.promptTokens, promptTokens) - usage.cacheRead;
  if (missedTokens <= NOISE_FLOOR_TOKENS) return undefined;

  // Extra cost = missed tokens billed at the actual paid rate (input/cacheWrite,
  // incl. write premium) instead of the cache-read rate. Missed tokens can only
  // land in the input or cacheWrite buckets, so the paid rate comes straight
  // from this message's own cost breakdown.
  const cost = usage.cost ?? {};
  const paidTokens = usage.input + usage.cacheWrite;
  const paidPerToken =
    paidTokens > 0
      ? ((cost.input ?? 0) + (cost.cacheWrite ?? 0)) / paidTokens
      : 0;
  const readPerToken =
    usage.cacheRead > 0
      ? (cost.cacheRead ?? 0) / usage.cacheRead
      : (models.getModel(message.provider ?? "", message.model ?? "")?.cost
          ?.cacheRead ?? 0) / 1_000_000;

  return {
    missedTokens,
    missedCost: missedTokens * Math.max(0, paidPerToken - readPerToken),
    idleMs: Math.max(0, message.timestamp - prev.timestamp),
    modelChanged:
      modelKey(message.provider ?? "", message.model ?? "") !== prev.modelKey,
  };
}

function asPreviousRequest(
  message: MissAssistantMessage,
  reportedCache: boolean,
): PreviousRequest | undefined {
  const usage = message.usage;
  const promptTokens = usage.input + usage.cacheRead + usage.cacheWrite;
  if (promptTokens <= 0) return undefined;
  return {
    promptTokens,
    modelKey: modelKey(message.provider ?? "", message.model ?? ""),
    timestamp: message.timestamp,
    reportedCache: reportedCache || usage.cacheRead + usage.cacheWrite > 0,
  };
}

function scan(entries: SessionEntry[]): PreviousRequest | undefined {
  let prev: PreviousRequest | undefined;
  for (const entry of entries) {
    if (entry.type === "compaction" || entry.type === "branch_summary") {
      // The context legitimately changed; the next turn's prompt is new content,
      // not re-billed content. Model switches are NOT exempt: they re-bill the
      // full prompt and should be counted.
      prev = undefined;
      continue;
    }
    if (entry.type === "message" && entry.message.role === "assistant") {
      const message = entry.message as unknown as MissAssistantMessage;
      // `reportedCache` is baseline-local sticky: it never survives the
      // context boundary above (mirrors upstream `asPreviousRequest(_,
      // prev?.reportedCache ?? false)`).
      prev = asPreviousRequest(message, prev?.reportedCache ?? false) ?? prev;
    }
  }
  return prev;
}

/**
 * Detect a cache miss on a just-completed assistant message. `entries` must
 * not yet contain `message` (for a branch summary: the session BEFORE the
 * summary entry is appended).
 *
 * Same name and semantics as upstream pi's `detectCacheMiss`
 * (`dist/core/cache-stats.js`), which is not exported from the package
 * barrel. Model switches are NOT suppressed: they re-bill the full prompt and
 * are surfaced with the `Cache miss after model switch` label.
 */
export function detectCacheMiss(
  entries: SessionEntry[],
  message: MissAssistantMessage,
  models: ModelPriceSource,
): CacheMiss | undefined {
  return detectMiss(scan(entries), message, models);
}

/**
 * Compact token formatting, byte-for-byte port of upstream pi's
 * `interactive-mode/components/footer.js` `formatTokens` (the same helper the
 * miss notice uses): 999 → "999", 9999 → "10.0k", 20000 → "20k".
 */
function formatTokens(count: number): string {
  if (count < 1000) return count.toString();
  if (count < 10_000) return `${(count / 1000).toFixed(1)}k`;
  if (count < 1_000_000) return `${Math.round(count / 1000)}k`;
  if (count < 10_000_000) return `${(count / 1_000_000).toFixed(1)}M`;
  return `${Math.round(count / 1_000_000)}M`;
}

/**
 * TUI warning copy for a counted cache miss, or `null` when the miss is below
 * the display floor. Mirrors upstream `addCacheMissNotice` thresholds and
 * label selection verbatim. `index.ts` stores the non-null result in
 * `details.summaryCache.notice` and renders it as a transcript line in the
 * tool's `renderResult`; it is NEVER appended to the rewind tool-result
 * content the model sees.
 */
export function formatCacheMissNotice(miss: CacheMiss): string | null {
  if (
    miss.missedTokens < CACHE_MISS_DISPLAY_TOKENS &&
    miss.missedCost < CACHE_MISS_DISPLAY_COST
  ) {
    return null;
  }
  const cost =
    miss.missedCost >= 0.01 ? ` (~$${miss.missedCost.toFixed(2)})` : "";
  const reBilled = `${formatTokens(miss.missedTokens)} tokens re-billed${cost}`;
  let label = "Cache miss";
  if (miss.modelChanged) {
    label = "Cache miss after model switch";
  } else if (miss.idleMs >= CACHE_TTL_MS) {
    label = `Cache miss after ${Math.round(miss.idleMs / 60_000)}m idle`;
  }
  return `${label}: ${reBilled}`;
}

// ---------------------------------------------------------------------------

/**
 * Cache accounting for the summary request. `usage.input` counts FRESH
 * (uncached) tokens only on cache-serving providers, which is why the
 * cache-hit metric is `cacheRead > 0` rather than `cacheRead/input`. This is
 * retained purely as the machine-readable `details.summaryCache` surface;
 * the TUI warning is derived from the miss detector above.
 */
export interface SummaryCacheStats {
  /** Fresh (uncached) input tokens billed for this request. */
  cacheRead: number;
  fresh: number;
  cacheWrite: number;
  hit: boolean;
}

/**
 * Derive cache accounting from provider usage. On cache-serving providers
 * `usage.input` counts FRESH tokens only (a cached read can exceed it), so
 * the hit predicate is `cacheRead > 0`, never a ratio.
 */
export function measureSummaryCache(
  usage: SummaryCacheUsage,
): SummaryCacheStats {
  return {
    cacheRead: usage.cacheRead,
    fresh: usage.input,
    cacheWrite: usage.cacheWrite,
    hit: usage.cacheRead > 0,
  };
}

// ---------------------------------------------------------------------------
// StreamFn wrapper
// ---------------------------------------------------------------------------

/**
 * The exact request live turns send, rebuilt for the summarization call.
 * `index.ts` assembles this from public pi APIs + two plain reflected fields
 * and passes it to `createCachePreservingStreamFn`.
 */
export interface CacheRequest {
  /** Live system prompt + live tool array + structured history + trailer. */
  context: {
    systemPrompt: string;
    messages: WireMessage[];
    tools: AgentTool[];
  };
  /**
   * Resolved (never hardcoded) cache retention. Explicit `"short"` matters:
   * reads key on sessionId + prefix bytes, so `"none"` (upstream's forced
   * value) would send no session id and could never hit.
   */
  cacheRetention: "short" | "long";
  /** Live session id: joins the session's cache namespace. */
  sessionId?: string;
  /**
   * Live thinking level, forwarded as `reasoning`. Providers that key the
   * cache on reasoning effort will miss a request that omits it even when
   * the prefix bytes match — this is a product requirement, not probe
   * hygiene. `"off"` is omitted, mirroring the live loop.
   */
  reasoning?: ThinkingLevel;
  /**
   * Live thinking token budgets (plain field on pi-agent-core's `Agent`).
   * Inert on some APIs, part of Anthropic's thinking body; forwarded only
   * when the host exposes it.
   */
  thinkingBudgets?: unknown;
}

/**
 * Wrap the `streamFn` handed to `generateBranchSummary` so the request is
 * rewritten to the live shape at the last possible moment (after upstream's
 * `completeSummarization` forced `cacheRetention: "none"` + a fresh
 * `sessionId`).
 *
 * `request === null` → delegate the caller's context and options untouched:
 * today's cold standalone request. This is the fallback for every "live input
 * unavailable" case.
 *
 * The `maxTokens` strip: upstream's summary caller caps output
 * (`maxTokens: 2048` in 0.84.2), but live turns let pi-ai fill
 * `clampMaxTokensToContext(model, liveContext, options?.maxTokens ??
 * model.maxTokens)`. A caller cap therefore diverges from the live
 * `max_output_tokens` and can break a gateway that keys on it. Stripping the
 * cap lets the provider compute the same value live gets. Residual: when the
 * context window is near-full the clamp differs by the trailer size (~1k);
 * r5d bounds output length in prose instead, and a miss is flagged by the
 * notice. (The recorded value differs across 0.80.2 = 2048 / 0.84.2 = 2048 /
 * fork = 4096, which is exactly why we strip whatever is there rather than
 * assume a constant.)
 *
 * Returned `used.value` flips true iff the live request was actually
 * delegated, so `index.ts` can report whether the wrapper engaged (it stays
 * false when the summarizer is stubbed or errors before the wire call).
 */
export function createCachePreservingStreamFn(args: {
  realStreamFn: StreamFn;
  request: CacheRequest | null;
}): { streamFn: StreamFn; used: { value: boolean } } {
  const { realStreamFn, request } = args;
  const used = { value: false };
  const streamFn: StreamFn = (model, coldContext, options) => {
    if (request === null) {
      used.value = false;
      return realStreamFn(model, coldContext, options);
    }
    used.value = true;
    const rest = { ...(options as Record<string, unknown> | undefined) };
    // Strip whatever caller cap upstream set; live turns let pi-ai clamp
    // model.maxTokens to the real context window.
    delete rest.maxTokens;
    const next = {
      ...rest,
      cacheRetention: request.cacheRetention,
      ...(request.sessionId ? { sessionId: request.sessionId } : {}),
      ...(request.reasoning && request.reasoning !== "off"
        ? { reasoning: request.reasoning }
        : {}),
      ...(request.thinkingBudgets
        ? { thinkingBudgets: request.thinkingBudgets }
        : {}),
    } as NonNullable<Parameters<StreamFn>[2]>;
    const context = request.context as Parameters<StreamFn>[1];
    return realStreamFn(model, context, next);
  };
  return { streamFn, used };
}
