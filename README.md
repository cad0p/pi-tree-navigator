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
  - `typebox ^1.0.0` (used to declare the tool's parameter schema; bundled with pi but listed explicitly so a standalone install resolves correctly).
- The reflection bootstrap depends on two plain (not `#`-private) internal pi/agent fields: `AgentSession.prototype.prompt` (patched for session capture) and `agent.state.messages` (refreshed after a rewind so the next prompt snapshots the rewound chain). Per-turn in-loop context refresh runs through the **public** `context` extension event (see “In-loop context refresh” below) — no `agent.prepareNextTurn*` reflection. Verified against pi 0.81.0 / 0.83.0 / 0.84.2.

## What you get

A single agent-callable tool, `navigate_tree`, with three actions:

| action | params | effect |
|---|---|---|
| `anchor` | `name` | Label the current point in the conversation as a milestone. |
| `rewind` | `labelStart`, `labelEnd`, `summaryFocus` | Collapse work between `labelStart` and the current leaf into a `branch_summary` entry. The summary is itself labeled with `labelEnd`, so you can chain rewinds. `summaryFocus` is required (non-trivial focus required; floor enforced at runtime by `MIN_SUMMARY_FOCUS_LENGTH`). Despite the verb, `rewind` does not restore prior state — it forks a sibling branch from `labelStart` and continues forward from a model-generated summary; the original subtree is preserved on disk but no longer on the active path. |
| `list` | — | Show all anchors on the active branch with cumulative context %. |

`name` (written by `anchor`) and `labelEnd` (written by `rewind`) both share the reserved `anchor:` label prefix; `labelStart` resolves against that same namespace. Every label written by `anchor` and every `labelEnd` written by `rewind` is referenceable by any subsequent `rewind`'s `labelStart`, and `list` shows all of them.

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

### Synthetic assistant token bias

The synthetic assistant we inject after each rewind carries the **post-rewind chain estimate** in `usage.totalTokens` (so `estimateContextTokens` reads a sensible baseline immediately after the move). The synthetic itself adds a ~50-token toolCall block re-emitted on every subsequent turn until the next rewind — that overhead is **not** reflected in any `usage.*` field, so future `estimateContextTokens` calls understate the chain by ~50 tokens until the next assistant turn writes a fresh usage block. Negligible at typical anchor cadence; mention if you're benchmarking exact token deltas, ignore otherwise.

## Limitations

- **Brittle to pi version bumps.** The fix uses two independent reflection points on internals that aren't part of pi's public API: `AgentSession.prototype.prompt` (session capture) and `agent.state.messages` (refreshed after a rewind). A third reflection point (`agent.prepareNextTurn*`) was eliminated in v0.2.0 via the public `context` extension event; the per-turn systemPrompt/tools/model/thinkingLevel refreshes come from pi's own `_installAgentNextTurnRefresh` (construction-installed since 0.80.3). If a future pi release renames the two remaining fields, switches them to private (`#`) fields, or restructures the class hierarchy, this breaks. The extension fails loudly: `anchor` still works, `rewind` reports `⚠ reflection bootstrap missing — the rewind landed on disk but the next assistant turn may still see the pre-rewind context. Run \`/reload\` (or restart pi) to recover.`, and you'd see context corruption return on the next prompt.

  **Audited against pi 0.81.0 / 0.83.0 / 0.84.2 (2026-08):** three of the five reflection points have been eliminated — `agent.state.systemPrompt` and `agent.state.tools` reads (deleted with the `prepareNextTurn` double-wrap; no longer needed since pi's own per-turn wrapper and the public `ctx.getSystemPrompt()` / `pi.getAllTools()` cover them) and `agent.prepareNextTurnWithContext` (replaced by the public `context` extension event, which fires via `transformContext` before **every** LLM call). The remaining two are NOT eliminable for the tool-based design: `AgentSession.prototype.prompt` (no public per-prompt hook for tool executes — the `context` event fires too late to install hooks before `createLoopConfig`) and `agent.state.messages` (pi's own `navigateTree` refresh is only reachable via `ctx.navigateTree()`, which exists solely on `ExtensionCommandContext`, not the `ExtensionContext` a `tool.execute` receives).

- **Anchor early in the turn.** Whatever's in `agent.state.messages` *before* the `anchor` tool call stays in the kept chain. Everything after gets summarized. Anchor at the *start* of a stage for maximum context savings.

- **Tiny rewinds are rejected by a minimum-savings floor.** A `rewind` whose measured savings falls below an internal floor (~4k tokens of apparent context freed) is refused with guidance listing the active anchors instead of executing — collapsing a near-empty segment burns a summarizer LLM call and can even grow live context once the summary and its synthetic assistant land on the kept chain. This pairs with anchoring early: anchor at the start of a stage, then rewind only once real work has accumulated above the anchor.

- **Abandoned branches grow the JSONL forever.** Each rewind preserves the abandoned subtree on disk. Session files get bigger over time even as live context shrinks. For very long autonomous runs (days), session files can hit hundreds of MB.

- **Tested against Anthropic and Kiro providers.** The synthetic-tool_use trick is specifically for Anthropic's strict tool_use/tool_result pairing; the synthetic's `stopReason: "toolUse"` survives Kiro's `normalizeMessages` filter. Other providers may have different validation rules — untested.

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

Tests cover `extensions/navigate-tree/helpers.ts` (pure helpers in `helpers.test.ts`) and `extensions/navigate-tree/index.ts` (action dispatch, schema shape, synthetic-assistant injection, context-event projection, reflection bootstrap, salvage path — in `index.test.ts`). The `summarize` factory option injects a stub for `generateBranchSummary` so no real LLM call fires during rewind tests. Additional manual e2e validation against the current pi release (0.84.x at time of writing) is recommended for any pi version bump (the reflection bootstrap depends on `AgentSession.prototype.prompt` / `agent.state.messages` field shapes).

## License

MIT — see [LICENSE](LICENSE).
