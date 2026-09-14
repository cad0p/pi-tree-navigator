# pi-tree-navigator

🌳 Agent-callable session tree navigation for [pi](https://github.com/badlogic/pi-mono).

Lets a pi agent anchor named milestones in its own conversation, then collapse work between them into a model-generated `branch_summary` to free up context — without tripping Anthropic's `tool_use` ↔ `tool_result` validation, and with the freed context immediately available to the next assistant turn (even within the same `prompt()` call).

## Install

Stable npm release:

```bash
pi install npm:@cad0p/pi-tree-navigator
```

Pre-release npm snapshots from `main` are published with the `next` dist-tag:

```bash
pi install npm:@cad0p/pi-tree-navigator@next
```

You can also install directly from the git source when testing unreleased branches:

```bash
pi install git:github.com/cad0p/pi-tree-navigator
```

## Publishing

This repo uses [`cad0p/semver-calver-release`](https://github.com/cad0p/semver-calver-release)'s npm-package workflow:

- Pushes to `main` compute the next hybrid SemVer + CalVer version, tag a GitHub prerelease, and publish to npm with the `next` dist-tag.
- Curated release PRs from `release/from-v*` branches bump the base `package.json` version and publish stable npm releases.
- npm publishing uses GitHub OIDC / npm trusted publishing via `.github/workflows/release.yml` (`id-token: write`) and `publishConfig.access: public`.

### Requirements

- **pi 0.81+** (node ≥22.19, which pi 0.81+ itself requires) with at least one model provider configured.
- Peer dependencies (the source of truth is `package.json` `peerDependencies`):
  - `@earendil-works/pi-coding-agent >=0.81.0`
  - `@earendil-works/pi-agent-core >=0.81.0`
  - `@earendil-works/pi-tui >=0.81.0` (used by the `renderResult` cache-notice transcript line: `Container`, `Spacer`, `Text`).
  - `typebox ^1.0.0` (used to declare the tool's parameter schema; bundled with pi but listed explicitly so a standalone install resolves correctly).
- The reflection bootstrap depends on plain (not `#`-private) internal pi/agent fields: `AgentSession.prototype.prompt` (patched for session capture) and `agent.state.messages` (refreshed after a rewind so the next prompt snapshots the rewound chain). Since #33 the cache-preserving summary request also **reads** `agent.state.tools`, `agent.state.systemPrompt`, and `agent.thinkingBudgets` off the same captured session (read-only; `ctx.getSystemPrompt()` is the public primary for the prompt), plus `settingsManager.getShowCacheMissNotices()` to gate the cache-notice transcript line. Per-turn in-loop context refresh runs through the **public** `context` extension event (see “In-loop context refresh” below) — no `agent.prepareNextTurn*` reflection. Verified against pi 0.81.0 / 0.83.0 / 0.84.2.

## What you get

A single agent-callable tool, `navigate_tree`, with three actions:

| action | params | effect |
|---|---|---|
| `anchor` | `name` | Label the current point in the conversation as a milestone. |
| `rewind` | `labelStart`, `labelEnd`, `summaryFocus` | Collapse work between `labelStart` and the current leaf into a `branch_summary` entry. The summary is itself labeled with `labelEnd`, so you can chain rewinds. `summaryFocus` is required (non-trivial focus required; floor enforced at runtime by `MIN_SUMMARY_FOCUS_LENGTH`). Despite the verb, `rewind` does not restore prior state — it forks a sibling branch from `labelStart` and continues forward from a model-generated summary; the original subtree is preserved on disk but no longer on the active path. |
| `list` | — | Show all anchors on the active branch with cumulative context %. |

`name` (written by `anchor`) and `labelEnd` (written by `rewind`) both share the reserved `anchor:` label prefix; `labelStart` resolves against that same namespace. Every label written by `anchor` and every `labelEnd` written by `rewind` is referenceable by any subsequent `rewind`'s `labelStart`, and `list` shows all of them.

**Anchoring is mandated, not suggested.** On every agent start the extension appends a one-line mandate to the end of the system prompt (`before_agent_start`), gated on the tool being active: `navigate_tree: gather all context, then anchor \`context-gathered\`; list anchors and rewind after every milestone or rabbit hole / dead end.` The append lands after project context and skills, is re-applied on every prompt, and survives compaction — unlike the `promptGuidelines` bullet it replaced. ~35 tokens, constant for prompt caching.

## How it works

A typical autonomous-loop pattern:

```
agent: navigate_tree(action="anchor", name="impl-start")
  → [anchor 'impl-start'] set at 1.9% of 1.0M (after: “implement the parser”)

agent: ...does work, runs tools, accumulates context to 30%...

agent: navigate_tree(action="rewind", labelStart="impl-start", labelEnd="impl-end",
                     summaryFocus="record only the public API of the parser
                                   and the open issue with edge case X")
  → [rewind 'impl-start' → 'impl-end'] · context 30.4% → 4.1% of 1.0M
  → A branch_summary recording the work just collapsed has been appended
    to your context. Items under '### Done' are complete. ...

agent: ...continues with the freed context, the next API call is back at ~4%...
```

The freed context is available to the **next assistant turn within the same `prompt()` call**, not just on the next user prompt. This is the key feature — autonomous agents don't have to wait for a user round-trip to benefit from a rewind.

## Implementation notes

Why this is more involved than just calling pi's `branchWithSummary`:

1. **Anthropic's tool_use ↔ tool_result pairing.** When a tool call rewinds the session tree, the tool's own `tool_use` lives in the assistant message that issued it — which `branchWithSummary` puts on the abandoned branch. Pi unconditionally writes the tool's `tool_result` to the new branch, leaving the result orphaned. Anthropic 400s the next API call with `Improperly formed request`. The fix is to inject a synthetic assistant message whose single `tool_call` has the same id as the in-flight call, *after* `branchWithSummary` but *before* the tool returns. Pi then writes the real `tool_result` as a child of that synthetic assistant — and the chain stays structurally valid.

2. **In-loop context refresh.** Pi's `Agent` class snapshots `state.messages` once at the start of `prompt()` and pushes new messages onto its own array. A rewind issued mid-loop wouldn't reduce the next API call's size until the user sent a fresh prompt. We register a handler on pi's **public `context` extension event** — fired via `Agent.transformContext` → `runner.emitContext` before *every* LLM call, including the turn right after a mid-loop rewind — that replaces the wire messages with the session-tree projection (`sessionManager.buildContextEntries()` → `sessionEntryToContextMessages`). After a rewind, the very next assistant turn within the same `prompt()` sees the rewound chain. (This replaced the earlier `agent.prepareNextTurnWithContext` reflection wrapper.)

3. **Reflection bootstrap.** Pi's slash-command `navigateTree` has access to `commandCtx.navigateTree`, which mutates `agent.state.messages`. Tool executes don't get that ctx, so we capture every `AgentSession` instance via the prompt patch and replicate the mutation manually. Without it, the on-disk leaf moves but `agent.state.messages` stays stale.

4. **`summaryFocus` is mandatory.** The summary is the only thing the agent will see of the collapsed work. The first time the agent uses `rewind`, blanket prompts produce vague summaries; subsequent rewinds are weaker. Forcing the agent to articulate `summaryFocus` (passed to pi's `generateBranchSummary` as `customInstructions`) measurably improves what survives.

### Cache-preserving summary request (#33)

Upstream `generateBranchSummary` builds a *cold* standalone request: a generic summarization system prompt, the conversation serialized into a text blob, no tools, `cacheRetention: "none"`, a fresh session id, and no reasoning forwarding. The live turns the summary collapses were just prompt-cache-served, so the summary re-billed the whole branch input (measured ~77k tokens cold vs a few hundred warm on opencode-go). The upstream fork fix (`cad0p/pi` PR #3) cannot be imported — pi's extension loader aliases `@earendil-works/*` to the host process's own modules — so the extension rewrites the request at the `streamFn` seam it already injects. That seam receives the fully-built `(model, context, options)` triple *after* `completeSummarization` applied its cold choices, and before the wire call.

The wrapper (`cache-summary.ts`) replaces that triple with the live request shape:

- **system prompt** — `ctx.getSystemPrompt()` when available, else the reflected `agent.state.systemPrompt`.
- **tools** — the same live tool array (`agent.state.tools`), by reference.
- **messages** — the live projection (`buildContextEntries`) as structured `Message`s, minus the in-flight assistant (it was never in a cached prefix, and an unpaired `tool_use` followed by a user message is rejected by Anthropic). Pre-branch background is included so the bytes prefix-match the previous live request; boundary-orphan `tool_result`s are stripped and the `{first}` scope number is adjusted to the payload actually sent.
- **params** — `cacheRetention` (resolved from `PI_CACHE_RETENTION`, never hardcoded), the live `sessionId` (also used for the opencode routing header), `reasoning` from `pi.getThinkingLevel()` (`"off"` omitted), and `thinkingBudgets` when the host exposes them. The caller's `maxTokens` cap is stripped: live turns let pi-ai clamp `model.maxTokens` to the context, and gateways that key the cache on params must see the same value.

The history walks newest→oldest against `contextWindow − 16384` tokens, dropping the oldest background first; `compaction` / `branch_summary` entries get upstream's 0.9-slack retry so they survive truncation. A truncated request no longer prefix-matches live turns (system + tools still do) — same as the fork.

The summary instruction uses the eval-approved r5d prompt and **must not drift**: it is pinned byte-for-byte in `cache-summary.test.ts`. The fallback (cold) path keeps upstream's older branch prompt — intentional divergence for a degraded path.

Every live input is read defensively. When any is unavailable (no provider `streamSimple`, no captured session, no live tools, no system prompt) or when the kill switch is set, the wrapper delegates today's cold request — and that path is **not** special-cased for notices: the summary response is measured by the fork's miss detector either way (the legacy cold request reports `cacheRead≈0`, so it misses exactly when the numbers say so). **Hits are silent** — the session totals/footer already cover them and there is no read/fresh notice. A **miss** is shown only when it clears the display floor (≥20k tokens or ≥$0.10), as a **TUI transcript line** appended by the tool's `renderResult` — the same mechanism upstream pi uses for its other cache notices (a `Spacer(1)` plus a warning-text line after the tool-result body) — gated by pi's own `showCacheMissNotices` setting (default **off**). The copy is `Cache miss: <n> tokens re-billed[ (~$<n>)]`, prefixed `Cache miss after <n>m idle` once the gap spans the 5-minute cache TTL (the fork's `Cache miss after model switch` branch is retained for parity, but summary misses after a model switch are suppressed as expected re-billing). Cache text is **never** in the rewind tool-result content the model sees. When the setting is off, or in a headless run (e.g. `-p` / RPC without UI, where the renderer never runs), nothing is rendered; the same data is available via `details.summaryCache`, which carries `mode`, `fallbackReason`, `branchStartRetained`, `used`, `cacheRead`, `input`, `cacheWrite`, `hit`, `missedTokens`, `missedCost`, `idleMs`, `modelChanged`, and `notice` (the rendered notice string, or `null`). `mode` records whether a cache-preserving request was **built**; `used` records whether the wrapper actually **delegated** it — a built-but-undelegated request (stub summarizer, upstream "No content to summarize" before `streamFn`, or an early abort) has `mode: "live-prefix"` / `used: false` and no measurable usage. Kill switch: `PI_NAVIGATE_TREE_SUMMARY_CACHE=0`.

Two of those fallback reasons are **evidence guards**, not param mirroring: they deliberately refuse the cache path so the summary keeps full-fidelity evidence (correctness over a cache hit), and they are the only known cases where the extension's request would silently diverge from the legacy path:

- **`branch-crosses-compaction`** — `buildContextEntries()` applies the compaction cut: entries before the latest compaction's `firstKeptEntryId` are dropped from the live projection. When a rewind segment reaches older than that cut (anchor/target older than `firstKeptEntryId`), the cache payload would summarize the lossy compacted projection while the legacy path summarizes the raw segment, so the request falls back. The predicate is the id difference between the collapsed entries and the live projection — **not** "the segment contains a compaction": a segment that contains the compaction entry but whose target sits at/after `firstKeptEntryId` loses no evidence and still takes the cache path.
- **`branch-start-not-retained`** — `buildLiveSummaryMessages` found no collapsed-segment message in the payload (a labels-only segment) or the newest message alone exceeded the token budget. The cache payload would be background-only; the legacy path either summarizes the raw evidence or returns "No content to summarize" before any wire call.

`details.summaryCache.branchStartRetained` is kept for diagnostics; it is now always `true` on a live-prefix request, because `false` is treated as a real fallback.

### Synthetic assistant token bias

The synthetic assistant we inject after each rewind carries the **post-rewind chain estimate** in `usage.totalTokens` (so `estimateContextTokens` reads a sensible baseline immediately after the move). The synthetic itself adds a ~50-token toolCall block re-emitted on every subsequent turn until the next rewind — that overhead is **not** reflected in any `usage.*` field, so future `estimateContextTokens` calls understate the chain by ~50 tokens until the next assistant turn writes a fresh usage block. Negligible at typical anchor cadence; mention if you're benchmarking exact token deltas, ignore otherwise.

## Limitations

- **Brittle to pi version bumps.** The fix uses two independent reflection points on internals that aren't part of pi's public API: `AgentSession.prototype.prompt` (session capture) and `agent.state.messages` (refreshed after a rewind). A third reflection point (`agent.prepareNextTurn*`) was eliminated in v0.2.0 via the public `context` extension event; the per-turn systemPrompt/tools/model/thinkingLevel refreshes come from pi's own `_installAgentNextTurnRefresh` (construction-installed since 0.80.3). If a future pi release renames the two remaining fields, switches them to private (`#`) fields, or restructures the class hierarchy, this breaks. The extension fails loudly: `anchor` still works, `rewind` reports `⚠ reflection bootstrap missing — the rewind landed on disk but the next assistant turn may still see the pre-rewind context. Run \`/reload\` (or restart pi) to recover.`, and you'd see context corruption return on the next prompt.

  **Audited against pi 0.81.0 / 0.83.0 / 0.84.2 (2026-08; #33 re-audit 2026-09):** two of the original five reflection points stay eliminated — `agent.prepareNextTurnWithContext` (replaced by the public `context` extension event, which fires via `transformContext` before **every** LLM call) and the `prepareNextTurn` double-wrap. The remaining surface is `AgentSession.prototype.prompt` (session capture) and `agent.state.messages` (post-rewind refresh); both are NOT eliminable for the tool-based design: there is no public per-prompt hook for tool executes, and pi's own `navigateTree` refresh lives on `ExtensionCommandContext`, not the `tool.execute` ctx. #33 adds four **read-only** reads on the same captured session — `agent.state.tools`, `agent.state.systemPrompt` (public `ctx.getSystemPrompt()` is the primary; the reflected field is the backstop), `agent.thinkingBudgets`, and `settingsManager.getShowCacheMissNotices()` (the AgentSession-level gate for the cache-notice transcript line) — all plain fields/accessors on pi-agent-core's `Agent` / `AgentSession`, all covered by `scripts/pi-upstream-probe.mjs`. Their failure mode is a cache miss or a suppressed notice, never a hard failure: the request falls back to the cold path with the transcript notice below.

- **Anchor early in the turn.** Whatever's in `agent.state.messages` *before* the `anchor` tool call stays in the kept chain. Everything after gets summarized. Anchor at the *start* of a stage for maximum context savings.

- **Tiny rewinds are rejected by a minimum-savings floor.** A `rewind` whose measured savings falls below an internal floor (~4k tokens of apparent context freed) is refused with guidance listing the active anchors instead of executing — collapsing a near-empty segment burns a summarizer LLM call and can even grow live context once the summary and its synthetic assistant land on the kept chain. This pairs with anchoring early: anchor at the start of a stage, then rewind only once real work has accumulated above the anchor.

- **Abandoned branches grow the JSONL forever.** Each rewind preserves the abandoned subtree on disk. Session files get bigger over time even as live context shrinks. For very long autonomous runs (days), session files can hit hundreds of MB.

- **Tested against Anthropic and Kiro providers.** The synthetic-tool_use trick is specifically for Anthropic's strict tool_use/tool_result pairing; the synthetic's `stopReason: "toolUse"` survives Kiro's `normalizeMessages` filter. Other providers may have different validation rules — untested.

- **Cache-preserving summary request (#33).** `rewind` mirrors the live request (system prompt, tool array, session id, cache retention, reasoning effort, thinking budgets) so the summary can be served from the same prompt-cache prefix as the turns it collapses. Any param the live loop sends that this extension does not mirror is a silent miss — the summary still runs, just cold, and a cold *structured* request can bill more than branch-only evidence. The response surfaces misses as TUI transcript lines (gated by `showCacheMissNotices`, ≥20k tokens / ≥$0.10 display floor, hits silent) and in `details.summaryCache` — never in the model-visible tool-result content (see “Cache-preserving summary request” above) — and `PI_NAVIGATE_TREE_SUMMARY_CACHE=0` forces the pre-#33 cold path. A segment whose raw evidence is not in the live projection (it crosses the latest compaction's `firstKeptEntryId`) or whose branch evidence is entirely dropped (labels-only / oversized newest message) deliberately refuses the cache path and re-bills cold — raw evidence beats a cache hit when the two disagree.

- **Cache path bypasses the SDK live request hooks.** The wrapper builds the summary request itself rather than routing through the SDK live path, so `onPayload` / `before_provider_request` and `transformHeaders` hooks registered by other extensions do not run for the summary request. Any extension that mutates the live request through those hooks is an additional silent-miss surface: the summary still runs, just cold.

- **`images.blockImages` divergence (cache path).** The summary shapes its history with the public `convertToLlm`, message by message. Pi's live loop wraps that in `convertToLlmWithBlockImages` when the `images.blockImages` setting is on, which filters image blocks out of the wire payload — so with that setting enabled and images in the collapsed context, the summary payload diverges and misses the cache (correct summary, cold bill). The live gate runs without images.

- **Summary trailer role alternation is untested on Kiro/Bedrock.** When the last retained message is user-role (a `toolResult` or user text), the summarization instruction lands as a consecutive user turn. Anthropic merges consecutive user turns; pi's Kiro/Bedrock adapters are untested on this exact shape. Unit tests pin the sequence; validate manually there before relying on the cache path.

- **Loading the extension monkey-patches `AgentSession.prototype.prompt` globally.** Every session in the host pi process picks up the patch on import, including sessions that never call `navigate_tree`. The patch is install-on-import and not reversible within a running pi process; restart pi to fully unload it.

- **`anchor:` is a reserved label prefix.** Any label written via pi's `/label` command or by another extension that begins with `anchor:` will be picked up by `list` and addressable by `rewind`'s `labelStart` / `labelEnd`. Avoid the prefix in manually-set labels.

- **Disk-fault during `rewind` (rare).** Pi's `branchWithSummary` advances the in-memory leaf before persisting the new entry to disk. If pi's session-write fails mid-call (full disk, FS error on a persisted session), the in-memory leaf has already moved past the original assistant turn but the synthetic-assistant injection in this extension never runs — pi's tool-result then lands without a matching tool_use, surfacing as the same `context_length_exceeded` 400 the synthetic exists to prevent. Production risk: low (in-memory tests don't reach this case; pi's session-write is robust on POSIX disk). Tracked for an additional salvage layer wrapping `branchWithSummary` itself in v0.2.0.

## Development

```bash
pnpm install
pnpm test          # helpers + dispatch / reflection bootstrap / salvage path
pnpm run lint      # biome check extensions/
pnpm run typecheck # tsc --noEmit
```

Tests cover `extensions/navigate-tree/helpers.ts` (pure helpers in `helpers.test.ts`), `extensions/navigate-tree/index.ts` (action dispatch, schema shape, synthetic-assistant injection, context-event projection, reflection bootstrap, salvage path, and the #33 cache-request call site — in `index.test.ts`), and `extensions/navigate-tree/cache-summary.ts` (payload shaping, `{first}` numbering + budget truncation, the r5d prompt pin, the notice matrix, and the wrapper contract driven through the real upstream `generateBranchSummary` — in `cache-summary.test.ts`). The `summarize` factory option injects a stub for `generateBranchSummary` so no real LLM call fires during rewind tests, and the wrapper tests use a fake capturing `streamFn`; the suite is fully offline. Additional manual e2e validation against the current pi release (0.84.x at time of writing) is recommended for any pi version bump (the reflection bootstrap depends on `AgentSession.prototype.prompt` / `agent.state.messages` field shapes; the cache path additionally reads `agent.state.tools` / `agent.state.systemPrompt` / `agent.thinkingBudgets`).

### Live summary verification

The unit suite is fully offline; it cannot prove that a real provider serves the summary from cache or that the model's output is scope-clean. For any change to `cache-summary.ts` or the rewind call site, run both halves against a configured provider:

1. **Cache gate** — in a fresh session with only this extension loaded (`pi --no-extensions -e <repo>/extensions/navigate-tree/index.ts`), anchor at the start of a stage, accumulate real work (e.g. two file reads), then `rewind`. Confirm `cacheRead > 0` / `hit: true` in `details.summaryCache` in the session JSONL (the authoritative, always-present surface); with pi's `showCacheMissNotices` setting enabled you will also see a cache-notice transcript line on a miss (≥20k tokens / ≥$0.10) and nothing at all on a hit. Cache text never appears in the tool-result content the model sees. A miss notice means a mirrored request param diverged — bisect in this order: caller `maxTokens` (must be stripped), `reasoning`, `cacheRetention`, session headers.
2. **Quality smoke** — `node scripts/summary-quality-check.mjs ~/.pi/agent/sessions/<dir>/<file>.jsonl` checks the r5d headings/length/preamble (and prints the newest `details.summaryCache` block); eyeball the printed summary for scope: branch only, pre-branch background excluded, unresolved work preserved, no continuation of the collapsed work.

## License

MIT — see [LICENSE](LICENSE).
