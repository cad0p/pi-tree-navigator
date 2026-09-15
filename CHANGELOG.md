# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-09-15

<!-- USER-EDITABLE SECTION START -->
Headline release: `rewind` summaries now ride the prompt cache.

`rewind` collapses a conversation segment by asking a model to summarize it, and that summary request used to be built cold — generic summarization system prompt, conversation serialized to a blob, no tools, fresh session id, no reasoning params, `cacheRetention: "none"` — so it re-billed the entire branch even though the turns it collapsed had just been prompt-cache-served (measured ~77k tokens / $0.0075 cold vs ~$0.0004 warm on opencode-go). The extension now rewrites the request at the `streamFn` seam to mirror the live turns: same system prompt, same tool array, the live structured message prefix (minus the in-flight assistant), same session routing, cache retention, reasoning effort, and thinking budgets. Measured on opencode-go: ~98–99% of the summary request input served from the live prefix cache (~20–24k tokens read, a few hundred fresh tokens billed). No upstream fork or patched pi host needed.

Safety stays ahead of cache hits: a segment whose raw evidence is not in the live projection (it crosses the latest compaction's cut) or whose branch evidence is entirely dropped deliberately refuses the cache path and re-bills cold, and `PI_NAVIGATE_TREE_SUMMARY_CACHE=0` forces the pre-cache path. A cache miss that clears the ≥20k-token / ≥$0.10 display floor is surfaced as a TUI transcript line when pi's `showCacheMissNotices` setting is on (default off); hits are silent, and every rewind always records `mode` / `used` / `cacheRead` / `hit` / `missedTokens` / `missedCost` / `fallbackReason` in `details.summaryCache` in the session JSONL. The `@earendil-works/pi-tui` peer powers the notice line. See the README's “Cache-preserving summary request (#33)” and “Live summary verification” for the full contract and the cache-gate checklist.

Also in this release: `rewind` is now refused before any mutation when it shares an assistant batch with sibling tool calls — sibling results land after the collapse with no declaring call, which is exactly how sessions got bricked (#37). The model re-issues the rewind solo. And refusals now actually look refused: every action rejected by the tool is recorded and rendered as a **failed** call — the TUI row flips from the green success background to the red error background, the session transcript carries `isError: true`, and Anthropic receives `is_error: true`. pi-agent-core ignores `isError` on returned tool results (only a thrown `execute()` is finalized as failed), so refusals are now promoted through the public `tool_result` event; the refusal copy and `details` are unchanged.
<!-- USER-EDITABLE SECTION END -->

### 🚀 Features

- *(navigate-tree)* Preserve prompt-cache prefix on rewind summaries (closes #33)

### 🐛 Bug Fixes

- *(rewind)* Refuse sibling-batched rewinds — rewind must be the only tool call in its batch (closes #37)
- *(rewind)* Surface refusals as failed tool calls (closes #40)


## [0.1.3] - 2026-09-14

<!-- USER-EDITABLE SECTION START -->
Last version before adopting the branch summarization cache hit mechanism to save everyone money -> https://github.com/earendil-works/pi/issues/9411
<!-- USER-EDITABLE SECTION END -->

### 🚀 Features

- *(ci)* Daily pi-upstream reflection probe (closes #17)
- *(guidelines)* Steer agents to anchor early at context-gathered (closes #29)
- Mandate anchoring via before_agent_start system prompt (closes #31)

### 🐛 Bug Fixes

- *(rewind)* Reject degenerate rewinds below a min-savings floor (closes #21, closes #20)
- *(rewind)* Inject opencode session headers for summarization (closes #26)

### ⚙️ Miscellaneous Tasks

- *(tooldef)* Trim navigate_tree tool definition (~848 → ~470 tok/request) (closes #22)


## [0.1.2] - 2026-08-20

<!-- USER-EDITABLE SECTION START -->
Patch release: fixes rewind summarization for custom-api providers and eliminates 3 of the 5 reflection points via public APIs.

**Rewind summarization for custom providers (#13):** `rewind` failed with `No API provider registered for api: <custom-id>` for any provider registered via `pi.registerProvider(name, { api: <custom-id>, streamSimple })` (e.g. pi-commandcode-provider 0.5.x with `api: "commandcode-custom"`). The summarizer now routes through the composed provider's `streamSimple`, obtained via the public `ctx.modelRegistry.getProvider(providerId)` API (pi ≥0.81.0) — the same routing pi's own `branchWithSummary` uses. Peer floor bumped to `>=0.81.0`; `null` header-deletion markers stripped (pi 0.84+ `ProviderHeaders`).

**Reflection surface five → two (#14):** the per-turn in-loop context refresh no longer reads `agent.state.systemPrompt` / `agent.state.tools` or wraps `agent.prepareNextTurnWithContext` / `agent.prepareNextTurn`. Instead a `pi.on("context")` handler (the public `context` extension event, fired via `Agent.transformContext` before every LLM call) returns the session-tree projection `sessionManager.buildContextEntries().flatMap(sessionEntryToContextMessages)`. Pi's own `_installAgentNextTurnRefresh` keeps systemPrompt/tools/model/thinkingLevel fresh per turn. Remaining reflection points: `AgentSession.prototype.prompt` (session capture) and `agent.state.messages` (post-rewind refresh so the next prompt snapshots the rewound chain) — neither is eliminable via public APIs.

Verified live on pi 0.84.2 (tmux TUI, commandcode provider): a 580KB file read (22.6% of 400k) rewound mid-loop to 1.0% of 400k with the very next LLM call at the rewound cache prefix (cacheRead 90,368 → 2,944). 125 tests pass; CI green.
<!-- USER-EDITABLE SECTION END -->

### 🐛 Bug Fixes

- *(rewind)* Route summarization through provider streamSimple (public API) (closes #13)

### 🚜 Refactor

- Eliminate 3 of 5 reflection points via public context event (closes #14)

### 📚 Documentation

- Update reflection field count/versions and dev commands for v0.1.1 reality ([#9](https://github.com/cad0p/pi-tree-navigator/pull/9))
- Port AGENTS.md bootstrap instructions from pi-napkin ([#11](https://github.com/cad0p/pi-tree-navigator/pull/11))


## [0.1.1] - 2026-07-31

<!-- USER-EDITABLE SECTION START -->
Patch release: restores the mid-loop context refresh on pi ≥0.80.3. No behavior change on pi ≤0.80.2.

**The bug (pi ≥0.80.3):** pi 0.80.3 added `AgentSession._installAgentNextTurnRefresh()`, which installs pi's own `agent.prepareNextTurnWithContext` in the constructor, and pi-agent-core's `Agent.createLoopConfig` now prefers that field over `agent.prepareNextTurn`. Since this extension only wrapped `prepareNextTurn`, its mid-loop context replacement was dead code: after a `rewind`, the branch summary landed correctly, but every remaining turn of the same loop still sent the full pre-rewind context to the API, and the footer's context-% re-anchored on that stale usage (jumping back up right after the rewind). Rewinds only actually saved context on the *next* user prompt.

**The fix:** `installPrepareNextTurn` now wraps both hook fields with the same marker/`__prior` chaining discipline. On pi ≥0.80.3 the `prepareNextTurnWithContext` wrapper chains pi's own (keeping its per-turn `systemPrompt`/`tools`/`model`/`thinkingLevel` refreshes) and overrides only `messages`; on pi ≤0.80.2 the new field is never read and `prepareNextTurn` does the work as before.

Verified live on pi 0.83.0 (persisted session): after a rewind at 31.5% context, the footer stays at ~1.6% for the rest of the loop (previously bounced back to ~33.5%), and the post-rewind API call goes out with ~3.7k tokens instead of ~80.5k.
<!-- USER-EDITABLE SECTION END -->

### 🚀 Features

- Discriminated-union schema makes summaryFocus required at the wire level ([#1](https://github.com/cad0p/pi-tree-navigator/pull/1))

### 🐛 Bug Fixes

- Revert discriminated-union parameters — Kiro rejects non-object root schemas ([#2](https://github.com/cad0p/pi-tree-navigator/pull/2))
- Wrap prepareNextTurnWithContext — in-loop context refresh dead since pi 0.80.3 ([#8](https://github.com/cad0p/pi-tree-navigator/pull/8))

### 🚜 Refactor

- Nest extension under extensions/navigate-tree/ per pi-napkin convention

### 📚 Documentation

- Promote npm install and publishing ([#4](https://github.com/cad0p/pi-tree-navigator/pull/4))

### ⚙️ Miscellaneous Tasks

- Release-grade cleanup for v0.1.0 ([#3](https://github.com/cad0p/pi-tree-navigator/pull/3))
- Switch from bun to node + pnpm for local dev and CI ([#6](https://github.com/cad0p/pi-tree-navigator/pull/6))


## [0.1.0] - 2026-05-25

<!-- USER-EDITABLE SECTION START -->

Initial release.

`navigate_tree` is an agent-callable pi tool with three actions:

- `anchor` — label the current point in the conversation as a milestone.
- `rewind` — collapse work between an anchor and the current leaf into a model-generated `branch_summary`, freeing context.
- `list` — show all anchors on the active branch with cumulative context %.

Designed for long autonomous sessions where the agent itself decides when to summarize. Survives mid-loop rewinds (the next assistant turn within the same `prompt()` call sees the reduced context) and produces structurally valid Anthropic chains by injecting a synthetic `tool_use` to pair with the rewind's `tool_result`.

User-visible specifics worth knowing on day one:

- Anchor names are kebab-case (lowercase alphanumeric segments separated by single hyphens; max 40 chars). Re-anchoring with a name already on the active branch moves the prior label to the new leaf rather than duplicating it; the same move-on-collision applies to `rewind`'s `labelEnd`.
- `rewind` requires a `summaryFocus` of ≥20 chars after trim; the rejection message lists what the focus should preserve so the agent can self-correct without user intervention.
- The `branch_summary` boilerplate strip in `list` hints is sentinel-anchored — a user-authored doc whose first H2 happens to be `## Goal` is preserved untouched.
- If the `AgentSession.prototype` patch isn't installed (typically only after a pi internals shape change), `list` and `rewind` surface a `⚠ reflection bootstrap missing` warning. The hint suggests `/reload` first (lighter — re-runs the prototype patch on the current process) and `Restart pi` as the heavier-handed alternative.

<!-- USER-EDITABLE SECTION END -->
