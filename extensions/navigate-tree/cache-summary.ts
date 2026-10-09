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
 * This extension observes every live request through pi's public
 * `context_with_system` event and stores the exact `[head, ...messages]`
 * array the provider is about to see, keyed by session. At rewind time
 * `index.ts` converts that capture with the same public `convertToLlm` the
 * live loop uses, appends the summary instruction, and hands the result to
 * `createCachePreservingStreamFn`. The wrapper passes it to
 * `modelRegistry.streamSimple` unchanged, so the summary request's messages
 * (and the tool declarations replayed from its leading system message) are
 * byte-identical to the live request by construction — no field mirroring.
 * The wrapper only strips upstream's `maxTokens` cap and sets the live
 * cache/reasoning options.
 *
 * ## Residual risks (see README "Limitations")
 *
 *  - Providers that key on request attributes outside the message
 *    transcript (headers, retry/timeout defaults, `onPayload` /
 *    `transformHeaders` hooks) still see this extension-built request, not
 *    the SDK-built one. `index.ts` keeps the session-routing headers the
 *    providers require (`withSessionHeaders`).
 *  - When the capture is unavailable, stale, or unsafe, `index.ts` hands
 *    the rewind to upstream `generateBranchSummary` unchanged: a correct
 *    summary on one cold bill, recorded in `details.summaryCache` as a
 *    fallback reason. The extension also measures the served response with
 *    pi's own miss detector and, when it clears the display floor, records
 *    the notice string in `details.summaryCache` (gated by
 *    `showCacheMissNotices`); `index.ts`'s `renderResult` renders it as a
 *    TUI transcript line. Neither surface reaches the model.
 */

import type { StreamFn, ThinkingLevel } from "@earendil-works/pi-agent-core";
import type {
  convertToLlm,
  SessionEntry,
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
// Instruction message
// ---------------------------------------------------------------------------

/**
 * Build the trailing instruction wire message for a captured payload.
 *
 * `first` is the 1-based number of the first branch message in the payload
 * (non-system messages only — the captured leading system message carries the
 * prompt and tool declarations and is not part of the conversation
 * numbering). Everything before it is background for prefix matching only.
 */
export function buildSummaryInstructionMessage(
  first: number,
  focus: string,
): WireMessage {
  return {
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
// Branch-summary cache-miss detection (port of pi's `cache-stats.ts`)
//
// Byte-for-byte behavioral port of `detectBranchSummaryCacheMiss` from
// `cad0p/pi@eval/branch-summary-prompt`
// `packages/coding-agent/src/core/cache-stats.ts`. pi 0.84.2 does not export
// this symbol, and its public `cache-stats` surface is an older revision, so
// the extension carries its own copy. The display half (`CacheMiss` copy +
// thresholds) mirrors `interactive-mode.ts`'s `addCacheMissNotice`.
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

/** A counted cache miss on the just-completed branch-summary request. */
export interface BranchSummaryCacheMiss {
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
   * Sticky: some earlier request in this session reported cache activity.
   * Session-scoped (never reset by context boundaries): provider cache
   * capability does not change across compactions, while the prompt baseline
   * legitimately does. Distinguishes a total miss on a cache-read-only
   * provider from a provider that never reports caching at all.
   */
  reportedCache: boolean;
}

interface MissUsage {
  input: number;
  cacheRead: number;
  cacheWrite: number;
  cost?: { input?: number; cacheRead?: number; cacheWrite?: number };
}

interface MissAssistantMessage {
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
): BranchSummaryCacheMiss | undefined {
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

function scan(
  entries: SessionEntry[],
  keepBaselineAcrossBranchSummary: boolean,
): PreviousRequest | undefined {
  let prev: PreviousRequest | undefined;
  // Session-level cache capability: any measured cache activity (assistant
  // turns AND summary requests) proves the provider reports caching, so a
  // later zero-read is a real miss even across a context boundary.
  let everReportedCache = false;
  for (const entry of entries) {
    if (
      entry.type === "compaction" ||
      (entry.type === "branch_summary" && !keepBaselineAcrossBranchSummary)
    ) {
      // The context legitimately changed; the next turn's prompt is new content,
      // not re-billed content. Model switches are NOT exempt: they re-bill the
      // full prompt and should be counted.
      if (entry.usage && entry.usage.cacheRead + entry.usage.cacheWrite > 0) {
        everReportedCache = true;
      }
      prev = undefined;
      continue;
    }
    if (entry.type === "branch_summary") {
      // Probe-only path (keepBaselineAcrossBranchSummary): the summary request
      // reuses the live prompt-cache prefix, so the parent baseline survives.
      // Fold cache activity into the session capability flag but never reset
      // prev and never become prev (only assistant messages do).
      if (entry.usage && entry.usage.cacheRead + entry.usage.cacheWrite > 0) {
        everReportedCache = true;
      }
      continue;
    }
    if (entry.type === "message" && entry.message.role === "assistant") {
      const message = entry.message as unknown as MissAssistantMessage;
      if (
        message.usage &&
        message.usage.cacheRead + message.usage.cacheWrite > 0
      ) {
        everReportedCache = true;
      }
      prev =
        asPreviousRequest(
          message,
          (prev?.reportedCache ?? false) || everReportedCache,
        ) ?? prev;
    }
  }
  return prev;
}

/**
 * Detect a cache miss on a just-completed branch-summary response from its
 * measured usage. `entries` is the session BEFORE the summary entry is
 * appended. Live-turn accounting counts model switches as misses; summary
 * probes suppress them instead — a cold summary right after a switch is
 * expected re-billing, not an actionable miss.
 */
export function detectBranchSummaryCacheMiss(
  entries: SessionEntry[],
  responseUsage: MissUsage,
  provider: string,
  model: string,
  timestamp: number,
  models: ModelPriceSource,
): BranchSummaryCacheMiss | undefined {
  const prev = scan(entries, true);
  if (prev && prev.modelKey !== modelKey(provider, model)) return undefined;
  return detectMiss(
    prev,
    { provider, model, usage: responseUsage, timestamp },
    models,
  );
}

/**
 * Compact token formatting, byte-for-byte port of the fork's
 * `interactive-mode/components/footer.ts` `formatTokens` (the same helper the
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
 * TUI warning copy for a counted branch-summary cache miss, or `null` when
 * the miss is below the display floor. Mirrors the fork's `addCacheMissNotice`
 * thresholds and label selection verbatim. `index.ts` stores the non-null
 * result in `details.summaryCache.notice` and renders it as a transcript line
 * in the tool's `renderResult`; it is NEVER appended to the rewind tool-result
 * content the model sees.
 */
export function formatBranchSummaryCacheMissNotice(
  miss: BranchSummaryCacheMiss,
): string | null {
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
 * The captured live request, replayed for the summarization call.
 * `index.ts` assembles this from the `context_with_system` capture (already
 * converted to wire messages) plus the trailing instruction, and passes it
 * to `createCachePreservingStreamFn`.
 */
export interface CacheRequest {
  /**
   * The captured live request's wire messages (`[head, ...messages]`) plus
   * the trailing summary instruction. The leading system message carries the
   * live prompt and tool declarations, so the provider replays both verbatim
   * — no separate `systemPrompt`/`tools` fields (passing those would make
   * pi-ai's `normalizeContext` prepend a declaration live never sends).
   */
  messages: WireMessage[];
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
 * rewritten to the captured live shape at the last possible moment (after
 * upstream's `completeSummarization` forced `cacheRetention: "none"` + a
 * fresh `sessionId`).
 *
 * `request === null` → delegate the caller's context and options untouched:
 * today's cold standalone request. This is the fallback for every
 * "capture unavailable/unsafe" case.
 *
 * With a request, the wrapper swaps the context for
 * `{ messages: request.messages }` only. Omitting `systemPrompt`/`tools`
 * means pi-ai's `normalizeContext` prepends nothing, and the provider
 * replays the captured leading system message — which carries the live
 * prompt and tool declarations in the live order — verbatim.
 *
 * The `maxTokens` strip: upstream's summary caller caps output
 * (`maxTokens: min(4096, model.maxTokens)`), but live turns let pi-ai fill
 * `clampMaxTokensToContext(model, liveContext, options?.maxTokens ??
 * model.maxTokens)`. A caller cap therefore diverges from the live
 * `max_output_tokens` and can break a gateway that keys on it. Stripping the
 * cap lets the provider compute the same value live gets. Residual: when the
 * context window is near-full the clamp differs by the trailer size (~1k);
 * r5d bounds output length in prose instead, and a miss is flagged by the
 * notice.
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
    return realStreamFn(
      model,
      { messages: request.messages } as Parameters<StreamFn>[1],
      next,
    );
  };
  return { streamFn, used };
}
