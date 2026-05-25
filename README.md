# pi-tree-navigator

🌳 Agent-callable session tree navigation for [pi](https://github.com/badlogic/pi-mono).

Lets a pi agent anchor named milestones in its own conversation, then collapse work between them into a model-generated `branch_summary` to free up context — without tripping Anthropic's `tool_use` ↔ `tool_result` validation, and with the freed context immediately available to the next assistant turn (even within the same `prompt()` call).

## Install

```bash
pi install git:github.com/cad0p/pi-tree-navigator
```

> **Status:** v0.1.0 is not yet on the npm registry (pending OIDC trusted-publisher setup). Install from the git source for now.

<details>
<summary>Once published / pre-release installs</summary>

- Stable (once published to npm):

  ```bash
  pi install npm:@cad0p/pi-tree-navigator
  ```

- Pre-release (calver snapshots from `main`, published to npm `@next` on every push):

  ```bash
  pi install npm:@cad0p/pi-tree-navigator@next
  ```

</details>

### Requirements

- **pi 0.74+** with at least one model provider configured.
- The reflection bootstrap depends on three internal pi/agent fields being plain (not `#`-private): `AgentSession.prototype.prompt`, `agent.state.messages`, and `agent.prepareNextTurn`. Verified against pi 0.75.x.

## What you get

A single agent-callable tool, `navigate_tree`, with three actions:

| action | params | effect |
|---|---|---|
| `anchor` | `name` | Label the current point in the conversation as a milestone. |
| `rewind` | `labelStart`, `labelEnd`, `summaryFocus` | Collapse work between `labelStart` and the current leaf into a `branch_summary` entry. The summary is itself labeled with `labelEnd`, so you can chain rewinds. `summaryFocus` is required (≥20 chars after trim). Despite the verb, `rewind` does not restore prior state — it forks a sibling branch from `labelStart` and continues forward from a model-generated summary; the original subtree is preserved on disk but no longer on the active path. |
| `list` | — | Show all anchors on the active branch with cumulative context %. |

`name`, `labelStart`, and `labelEnd` all live in a single namespace under the reserved `anchor:` label prefix — every label written by `anchor` and every `labelEnd` written by `rewind` is referenceable by any subsequent `rewind`'s `labelStart`, and `list` shows all of them.

## How it works

A typical autonomous-loop pattern:

```
agent: navigate_tree(action="anchor", name="impl-start")
  → [anchor 'impl-start'] set at 1.9% of 1.0M (after user: "implement the parser")

agent: ...does work, runs tools, accumulates context to 30%...

agent: navigate_tree(action="rewind", labelStart="impl-start", labelEnd="impl-end",
                     summaryFocus="record only the public API of the parser
                                   and the open issue with edge case X")
  → [rewind 'impl-start' → 'impl-end'] · context 30.4% → 4.1% of 1.0M
  → A branch_summary recording the work just collapsed has been appended
    to your context. Items under '## Done' are complete. ...

agent: ...continues with the freed context, the next API call is back at ~4%...
```

The freed context is available to the **next assistant turn within the same `prompt()` call**, not just on the next user prompt. This is the key feature — autonomous agents don't have to wait for a user round-trip to benefit from a rewind.

## Implementation notes

Why this is more involved than just calling pi's `branchWithSummary`:

1. **Anthropic's tool_use ↔ tool_result pairing.** When a tool call rewinds the session tree, the tool's own `tool_use` lives in the assistant message that issued it — which `branchWithSummary` puts on the abandoned branch. Pi unconditionally writes the tool's `tool_result` to the new branch, leaving the result orphaned. Anthropic 400s the next API call with `Improperly formed request`. The fix is to inject a synthetic assistant message whose single `tool_call` has the same id as the in-flight call, *after* `branchWithSummary` but *before* the tool returns. Pi then writes the real `tool_result` as a child of that synthetic assistant — chain valid.

2. **In-loop context refresh.** Pi's `Agent` class snapshots `state.messages` once at the start of `prompt()` and pushes new messages onto its own array. A rewind issued mid-loop wouldn't reduce the next API call's size until the user sent a fresh prompt. We wire `agent.prepareNextTurn` from a prototype patch on `AgentSession.prototype.prompt`, returning a fresh context built from `sessionManager.buildSessionContext()` between every turn boundary. After a rewind, the very next assistant turn within the same `prompt()` sees the rewound chain.

3. **Reflection bootstrap.** Pi's slash-command `navigateTree` has access to `commandCtx.navigateTree`, which mutates `agent.state.messages`. Tool executes don't get that ctx, so we capture every `AgentSession` instance via the prompt patch and replicate the mutation manually. Without it, the on-disk leaf moves but `agent.state.messages` stays stale.

4. **`summaryFocus` is mandatory.** The summary is the only thing the agent will see of the collapsed work. The first time the agent uses `rewind`, blanket prompts produce vague summaries; subsequent rewinds are weaker. Forcing the agent to articulate `summaryFocus` (passed to pi's `BRANCH_SUMMARY_PROMPT` as `Additional focus: …`) measurably improves what survives.

### Synthetic assistant token bias

The synthetic assistant we inject after each rewind shows up in `usage.totalTokens` as ~30–50 input tokens (the wrapper around its empty content). It's a one-time cost per rewind, not per subsequent turn — the `branch_summary` entry already absorbs the abandoned content. Mention if you're benchmarking exact token deltas; ignore otherwise.

## Limitations

- **Brittle to pi version bumps.** The fix uses three independent reflection points on internals that aren't part of pi's public API: `AgentSession.prototype.prompt`, `agent.state.messages`, and `agent.prepareNextTurn`. If a future pi release renames any of these, switches them to private (`#`) fields, or restructures the class hierarchy, this breaks. The extension fails loudly: `anchor` still works, `rewind` reports `⚠ Reflection failed`, and you'd see context corruption return on the next prompt.

- **Anchor early in the turn.** Whatever's in `agent.state.messages` *before* the `anchor` tool call stays in the kept chain. Everything after gets summarized. Anchor at the *start* of a stage for maximum context savings.

- **Abandoned branches grow the JSONL forever.** Each rewind preserves the abandoned subtree on disk. Session files get bigger over time even as live context shrinks. For very long autonomous runs (days), session files can hit hundreds of MB.

- **Anthropic only (today).** The synthetic-tool_use trick is specifically for Anthropic's strict tool_use/tool_result pairing. Other providers may have different validation rules — untested.

- **Loading the extension monkey-patches `AgentSession.prototype.prompt` globally.** Every session in the host pi process picks up the patch on import, including sessions that never call `navigate_tree`. The patch is install-on-import and not reversible within a running pi process; restart pi to fully unload it.

- **`anchor:` is a reserved label prefix.** Any label written via pi's `/label` command or by another extension that begins with `anchor:` will be picked up by `list` and addressable by `rewind`'s `labelStart` / `labelEnd`. Avoid the prefix in manually-set labels.

## Development

```bash
bun install
bun test          # ~110 tests covering helpers + dispatch + reflection bootstrap
bunx biome check extensions/
bunx tsc --noEmit
```

Tests cover `extensions/navigate-tree/helpers.ts` (pure helpers in `helpers.test.ts`) and `extensions/navigate-tree/index.ts` (action dispatch, schema shape, synthetic-assistant injection, reflection bootstrap, salvage path — in `index.test.ts`). The `summarize` factory option injects a stub for `generateBranchSummary` so no real LLM call fires during rewind tests. Additional manual e2e validation against pi 0.75.x is recommended for any pi version bump (the reflection bootstrap depends on internal field shapes).

## License

MIT — see [LICENSE](LICENSE).
