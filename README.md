# pi-tree-navigator

🌳 Agent-callable session tree navigation for [pi](https://github.com/badlogic/pi-mono).

Lets a pi agent anchor named milestones in its own conversation, then collapse work between them into a model-generated `branch_summary` to free up context. The summary is written by the live model as one more turn of the live conversation, so it is served from the prompt cache on every provider, and the freed context is immediately available to the next assistant turn (even within the same `prompt()` call).

This way you can have a session that looks like this :)

<img width="2686" height="708" alt="image" src="https://github.com/user-attachments/assets/dc3beecf-c18c-4ab2-b5a8-19b9f94d1498" />


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

- **pi 0.87+** (node ≥22.19, which pi 0.87+ itself requires) with at least one model provider configured. The floor is set by `SessionManager.buildSessionProjection` (the context-edit-aware projection used both for the wire context and to number the summary's `{first}` branch start; added in 0.87.0).
- Peer dependencies (the source of truth is `package.json` `peerDependencies`):
  - `@earendil-works/pi-coding-agent >=0.87.0`
  - `@earendil-works/pi-agent-core >=0.87.0`
  - `typebox ^1.0.0` (used to declare the tool's parameter schema; bundled with pi but listed explicitly so a standalone install resolves correctly).
- The reflection bootstrap depends on plain (not `#`-private) internal pi/agent fields: `AgentSession.prototype.prompt` (patched for session capture) and `agent.state.messages` (refreshed after a rewind so the next prompt snapshots the rewound chain). After a rewind the extension also emits `session_tree` through the captured session's plain `_extensionRunner` field (#79; see [History-mirroring providers](#history-mirroring-providers-pi-claude-bridge)). Per-turn in-loop context refresh runs through the **public** `context` extension event (see “In-loop context refresh” below) — no `agent.prepareNextTurn*` reflection. The rewind's same-run continuation relies on pi's `turn_end` boundary contract (handlers awaited before the steering drain, returned entries committed, `continue` honoured; public since 0.87). Verified against pi 1.2.0 via `scripts/pi-upstream-probe.mjs`.

## What you get

A single agent-callable tool, `navigate_tree`, with three actions:

| action | params | effect |
|---|---|---|
| `anchor` | `name` | Label the current point in the conversation as a milestone. |
| `rewind` | `rewindTo`, `newLabel`, `summaryFocus` | Collapse work between `rewindTo` and the current leaf into a `branch_summary` entry. The tool result is the summary instruction; the model's next reply is the summary, and the extension then branches with it (see [In-band summary](#in-band-summary-80)). The summary is itself labeled with `newLabel`, so you can chain rewinds. `summaryFocus` is required (non-trivial focus required; floor enforced at runtime by `MIN_SUMMARY_FOCUS_LENGTH`). Despite the verb, `rewind` does not restore prior state — it forks a sibling branch from `rewindTo` and continues forward from a model-generated summary; the original subtree is preserved on disk but no longer on the active path. |
| `list` | — | Show all anchors on the active branch with cumulative context %. |

`name` (written by `anchor`) and `newLabel` (written by `rewind`) both share the reserved `anchor:` label prefix; `rewindTo` resolves against that same namespace. Every label written by `anchor` and every `newLabel` written by `rewind` is referenceable by any subsequent `rewind`'s `rewindTo`, and `list` shows all of them.

Labels are unique on the active branch: `anchor` refuses a `name` and `rewind` refuses a `newLabel` that already exists (the `rewind` check runs before summarization), and an existing label is never moved or cleared. The old move-on-collision rename-forward idiom (`anchor x` → `rewind(newLabel=x)`) is retired — a rewind needs a fresh `newLabel`; the existing label stays addressable via `rewindTo`. Manual `/tree` labels can still duplicate; `list` shows every match and `rewindTo` resolves leaf-most. The extension itself also writes one silent label per session leg: `anchor:start` (see [Automatic start anchor](#automatic-start-anchor)).

**Anchoring is mandated, not suggested.** On every agent start the extension appends a one-line mandate to the end of the system prompt (`before_agent_start`), gated on the tool being active: `navigate_tree: gather all context, then anchor \`context-gathered\`; list anchors and rewind after every milestone or rabbit hole / dead end to keep context low.` The append lands after project context and skills, is re-applied on every prompt, and survives compaction — unlike the `promptGuidelines` bullet it replaced. ~40 tokens (159 chars / 4), constant for prompt caching.

## Configuration (`tree-navigator.json`)

Both the rewind hint and the automatic start anchor read one two-layer config file:

| Layer | Path |
|---|---|
| Global | `join(getAgentDir(), "tree-navigator.json")` (`PI_CODING_AGENT_DIR`-aware) |
| Project | `join(cwd, CONFIG_DIR_NAME, "tree-navigator.json")` (`.pi/` by default; read only when the project is trusted — untrusted projects are ignored silently) |

```json
{ "rewindHintReserveTokens": 50000, "anchorStartAfterTurns": 2 }
```

A project key, when present, wins (including the hint's disable sentinels); absent → global; absent everywhere → the field's default. File-level problems (non-ENOENT read error, invalid JSON, non-object root) warn and make that layer contribute nothing for *every* field, so the other layer still applies; a missing file is silent. Each field-level invalid value emits its own field-scoped warning when the session has a UI (headless sessions stay silent, like every `ui.notify`).

## Rewind hint (optional)

Long autonomous sessions can blow through context silently — the model can't read pi's footer, and `rewind` is a targeted alternative to pi's lossy auto-compaction, but nothing tells the agent *when* to use it. The rewind hint is an opt-in `turn_end` observer that fires **once per crossing**: when the tokens left in the context window drop to the configured reserve, the agent gets a persisted nudge to persist what matters to files and rewind to an anchor. Default: **disabled** (pi's own compaction behavior is unchanged).

Enable it with `rewindHintReserveTokens` (see [Configuration](#configuration-tree-navigatorjson) for the layer paths and the shared file-level failure semantics):

```json
{ "rewindHintReserveTokens": 50000 }
```

- Value: integer **1000–1000000** inclusive — the number of tokens left in the window (`contextWindow − context tokens`) at which the hint fires. A bare number is canonical (`50000`); a digits-only string (`"50000"`, surrounding whitespace allowed) is also accepted. Size it like pi's `compaction.reserveTokens`: it is an absolute budget for the persist + summary turns, so it means the same thing on a 100k and a 1M window. A reserve at or above the window fires on the first observed turn and never re-arms while that window stays (the tokens left can never climb above it); only a switch to a model with a larger window can re-arm it.
- Explicit-disable sentinels: `null`, `false`, `"off"`, `"disabled"` (case-insensitive) — the project layer can switch off a global value.
- Precedence: a project key, when present, wins (including sentinels). Absent → global. Absent everywhere → disabled.
- Invalid value (out of range, non-integer, a string other than digits — signs, decimals, exponents — `true`, object): the hint stays off for the session — no fallback to the other layer's reserve — and its field-scoped warning is shown when the session has a UI.
- One hint per crossing: fires when `tokens left <= reserve`, re-arms only after the tokens left climb back above it (a rewind or compaction); no escalation, no re-fire while spent.
- **Removed in 0.3.0: `rewindHintAtPercent`.** A percentage meant very different budgets on different windows (90% leaves 100k on 1M but 10k on 100k). The key is now ignored with a field-scoped warning pointing at `rewindHintReserveTokens`; migrate by picking the tokens you want left (e.g. `"90"` on a 1M window → `100000`).

When the crossing has at least one `anchor:` label on the active branch, the agent receives a persisted custom message naming every active anchor with its cumulative context percentage — so the model knows a rewind target exists (including the automatic `start`) without a `list` call. Mid-run this steers the running loop (one reaction turn in the same run); when the agent is idle it only appends and waits for the user's next prompt, so an idle-agent rewind stays user-confirmed:

```text
[navigate_tree hint] Context is at 95.2% of 1.0M — 48.0k tokens left, within the 50.0k reserve. Persist what matters to files now, then rewind to the oldest appropriate anchor. Active anchors: 'start' 4.1% · 'context-gathered' 12.0% of 1.0M
```

With no anchors the extension sends **no model message** (a rewind is impossible) and shows a TUI-only warning explaining how to add one manually: `/tree`, select the entry to rewind to, press `shift+l`, label it `anchor:<name>` (e.g. `anchor:context-gathered`), then ask the agent to rewind to it.

### Rewind hygiene (always on)

Independent of the hint, the tool ships two static `promptGuidelines` bullets, present whenever `navigate_tree` is active: **when rewinding, prefer the oldest anchor that keeps what you'd otherwise re-read or re-derive**, persisting durable findings to files first; and **don't rewind while a user decision or unresolved question is pending** — ask the user instead. The bullets are not config-gated (`registerTool` fixes the array at registration), cost ~64 tokens per request (257 chars / 4), and change the cached system-prompt prefix once per active session on upgrade (the provider-facing tool schema is unchanged).

### Compaction guidance

The hint is an alternative to auto-compaction. Auto-compaction fires at `contextWindow − reserveTokens` (`compaction.reserveTokens`, default 16384); the hint fires once `contextWindow − tokens <= rewindHintReserveTokens`, and pi's threshold compaction once `tokens > contextWindow − reserveTokens`, so — in sessions without context edits or compaction, where both measure the same last-usage tokens — the hint lands first on every window whenever `rewindHintReserveTokens` is at least the active model's compaction reserve (`compaction.modelOverrides["<provider>/<modelId>"].reserveTokens`, else `compaction.reserveTokens`); size it well above that to leave room for the persist and summary turns. Overflow-triggered compaction (a provider context-length error) is independent of either threshold. Recommended: disable auto-compaction (`"compaction": { "enabled": false }`) so running out mid-persist produces pi's loud context-window error instead of a silent lossy compaction. Keep compaction enabled only as a safety net if you'd rather the model continue and compact mid-write; the hint never overrides it.

## Automatic start anchor

Independent of the hint, the extension writes a silent `anchor:start` label at the X-th turn of the agent run (or at the run's end if it finishes earlier), so every session leg has a guaranteed low-floor fallback rewind target (typically ~4% of the window) with no user action. `start` shows up in `list` output like any other anchor; nothing else model-visible changes in v1.

`anchorStartAfterTurns` (see [Configuration](#configuration-tree-navigatorjson)):

- Value: integer **1–50** inclusive. A bare number is canonical (`2`); numeric strings (`"2"`) are accepted. Default **2**.
- **Non-optional**: unlike the hint, an absent `anchorStartAfterTurns` — or no config file at all — still fires at the default; absence is never a warning.
- Invalid value (out of range, decimal, non-numeric string, `null`, `false`, object): a field-scoped warning, then the other layer's valid value applies, else the default 2 — a touched-but-invalid value must not disable a non-optional feature.
- Precedence: trusted project > global > default 2.
- Kill switch: `PI_NAVIGATE_TREE_START_ANCHOR=0` suppresses the write, read at write time (a mid-process change applies without `/reload`).

Behavior and edge cases:

- **Run-bounded**: X counts native turns of the current agent run. A run shorter than X anchors at its end (`agent_end`); `agent_end` fires on completed, aborted, and errored runs, so settlement always happens by the end of the first run (with X=2, even a one-turn run anchors at turn 1).
- **Once per session**: `start` marks the start of the current session leg. Once the label is written — or found already on the active branch (e.g. a manual `/tree` label) — the extension never writes it again in that session, including later prompts and agent loops. `/reload` re-evaluates only via the existence check, so no duplicate is ever written.
- **Every session**, including subagents, gated only on `navigate_tree` being in the active tool set (headless included).
- **Labels are unique** (see [What you get](#what-you-get)): the tool refuses a duplicate `name`/`newLabel`, so `start` is never moved or overwritten by the agent. A manual `/tree` duplicate remains possible; `list` shows both and `rewindTo` resolves leaf-most.
- Silent by design: a label entry only — no model message, no UI.

## How it works

A typical autonomous-loop pattern:

```
agent: navigate_tree(action="anchor", name="impl-start")
  → [anchor 'impl-start'] set at 1.9% of 1.0M (after: “implement the parser”)

agent: ...does work, runs tools, accumulates context to 30%...

agent: navigate_tree(action="rewind", rewindTo="impl-start", newLabel="impl-end",
                     summaryFocus="record only the public API of the parser
                                   and the open issue with edge case X")
  → Summarize only messages 4 onwards in the conversation above ... (the summary instruction)

agent: ## Goal ... ## Next Steps ...        ← the summary, a plain live turn (cache hit)
  → [rewind to 'impl-start' · collapsed as 'impl-end'] · context 30.4% → 4.1% of 1.0M
    A branch_summary of the collapsed work has been appended to your context.
    It is the newest branch_summary, the current state. User messages retained
    above predate this rewind — do not re-answer them. ...

agent: ...continues with the freed context, the next API call is back at ~4%...
```

The freed context is available to the **next assistant turn within the same `prompt()` call**, not just on the next user prompt. This is the key feature — autonomous agents don't have to wait for a user round-trip to benefit from a rewind.

## Implementation notes

Why this is more involved than just calling pi's `branchWithSummary`:

1. **The summary is written in-band (#80).** The rewind tool does not call a summarizer. Its tool result is the summary instruction, so the live model writes the summary as its next turn and the tree moves only after that turn ends (see [In-band summary](#in-band-summary-80)). Nothing is in flight when `branchWithSummary` runs, so the rewind's own `tool_use`/`tool_result`, the instruction and the summary turn all stay on the abandoned branch and the new branch is structurally valid without any synthetic message.

2. **In-loop context refresh.** Pi's `Agent` class snapshots `state.messages` once at the start of `prompt()` and pushes new messages onto its own array. A rewind issued mid-loop wouldn't reduce the next API call's size until the user sent a fresh prompt. We register a handler on pi's **public `context` extension event** — fired via `Agent.transformContext` → `runner.emitContext` before *every* LLM call, including the turn right after a mid-loop rewind — that returns the conversation-only session projection (`sessionManager.buildSessionProjection().messages`, context-edit-aware; Pi re-attaches the current system head itself). After a rewind, the very next assistant turn within the same `prompt()` sees the rewound chain. (This replaced the earlier `agent.prepareNextTurnWithContext` reflection wrapper.)

3. **Reflection bootstrap.** Pi's slash-command `navigateTree` has access to `commandCtx.navigateTree`, which mutates `agent.state.messages`. Tool executes don't get that ctx, so we capture every `AgentSession` instance via the prompt patch and replicate the mutation manually. Without it, the on-disk leaf moves but `agent.state.messages` stays stale.

4. **`summaryFocus` is mandatory.** The summary is the only thing the agent will see of the collapsed work. The first time the agent uses `rewind`, blanket prompts produce vague summaries; subsequent rewinds are weaker. Forcing the agent to articulate `summaryFocus` (appended to the summary instruction as `Additional focus:`) measurably improves what survives.

### In-band summary (#80)

Earlier releases summarized with a separate request: first pi's cold `generateBranchSummary` (re-billing the whole branch), then a replay of the captured live request (#33/#75) that hit the cache only while the replay stayed byte-identical to the live turn, with fallbacks (`stale-capture`, `capture-diverged`, `overflow`, ...) that re-billed cold. A separate request also cannot reach a provider that serves turns from its own session, such as pi-claude-bridge. The summary is now just the next turn of the live conversation:

1. **`rewind` (tool call).** Validates as before (names, `summaryFocus`, label uniqueness, the minimum-savings floor), records a pending rewind for the session, and returns the eval-approved r5d instruction with `{first}`, the 1-based number of the first collapsed message in the projection the model sees. The tree does not move yet.
2. **The summary turn.** The model's next reply is the summary: an append to the live conversation, so it is served from the prompt cache like any other turn. While the rewind is pending every tool call is blocked (`tool_call` → `block`) with a reason that asks for the summary.
3. **`turn_end`.** A text-only turn with the r5d `## Goal` heading completes the rewind: any preamble before `## Goal` is dropped, the stored `branch_summary` gets upstream's preamble and `<read-files>`/`<modified-files>` lists, `branchWithSummary` forks from the anchor, the label is written, `agent.state.messages` is refreshed and `session_tree` is emitted. The continuation message (`customType: "navigate-tree-rewind"`) is delivered in the same run, so the rewound context reaches a real turn — and the footer's token count updates — without a user prompt.

A summary turn fails when it calls a tool, has no `## Goal` heading, or is truncated. A failed turn re-asks (the blocked tool results do it, or a `summary-retry` message for a text-only turn); after `MAX_SUMMARY_ATTEMPTS` (3) the rewind is cancelled with a `cancelled` message and nothing is collapsed. Failed attempts sit on the segment the rewind collapses, so they cost nothing once it lands. An aborted summary turn drops the rewind silently. A provider error keeps it pending across pi's auto-retry (pi emits `agent_end` before it retries, so the rewind survives that run end); the next user prompt drops a rewind the retry never completed, and any other run end without a summary drops it. Input that arrives after the rewind call — user steers drained at the rewind turn's boundary, other extensions' messages (subagent results, timers) — lands before the summary turn, on the segment being collapsed; it is re-appended onto the new branch in its original order, and the continuation message flags it as new and still unanswered. If the anchor moved, `newLabel` was taken, the context was compacted, or the branch could not be written while the summary was pending, the rewind is cancelled with a `cancelled` message.

The continuation message is returned as a `turn_end` boundary entry with `continue: true`: pi commits it right after the handler, before draining queued steers, and guarantees one more provider request — so it lands ahead of anything typed meanwhile and the run continues even after a text-only summary turn. Its `details` carry `status`, `rewindTo`, `newLabel`, `targetId`, `summaryId`, `collapsedEntries`, `contextBefore`/`contextAfter`, `contextWindow`, `summaryAttempts`, `agentMessagesRefreshed`, `historyRewriteNotified`, `reappendedLateMessages`, and the file lists. The rewind hint stays silent on any turn a rewind owns.

### History-mirroring providers (pi-claude-bridge)

[pi-claude-bridge](https://www.npmjs.com/package/pi-claude-bridge) serves each turn from its own Claude Code session and rebuilds that session from pi's history only when pi signals a rewrite (`session_tree` / `session_compact`). A tool-driven rewind skipped that signal, so the Claude Code query parked at the `navigate_tree` call kept answering with the whole pre-rewind context: the footer stayed at the old percentage because the real request did too (#79).

After every rewind the extension emits `session_tree` with the payload pi's own `navigateTree` emits (`newLeafId`, `oldLeafId`, `summaryEntry`, `fromExtension: true`), after the `agent.state.messages` refresh. The bridge then rebuilds its session from the rewound chain on the next request. `details.historyRewriteNotified` on the continuation message records whether the event went out (`false` when the runner is unreachable or the emit threw; the rewind itself is complete either way).

Cache behavior on the bridge:

- **Summary turn.** The in-band summary is the paused Claude Code query receiving its tool result: a cache hit on the live session, no bridge-specific path.
- **First turn after the rewind.** That turn is a bridge session rebuild, which per the bridge's README usually misses the prompt cache past the system prompt. It is a one-time write of the rewound chain (tens of k) instead of re-reading the full pre-rewind context on every turn.

## Limitations

- **Brittle to pi version bumps.** The fix uses two independent reflection points on internals that aren't part of pi's public API: `AgentSession.prototype.prompt` (session capture) and `agent.state.messages` (refreshed after a rewind). A soft one (#79) is `AgentSession._extensionRunner`, used only to emit `session_tree` after a rewind; if it disappears the rewind still works, but history-mirroring providers (pi-claude-bridge) stop rebuilding and `details.historyRewriteNotified` reports `false`. A former third reflection point (`agent.prepareNextTurn*`) was eliminated in v0.2.0 via the public `context` extension event; the per-turn systemPrompt/tools/model/thinkingLevel refreshes come from pi's own `_installAgentNextTurnRefresh` (construction-installed since 0.80.3). If a future pi release renames the two remaining fields, switches them to private (`#`) fields, or restructures the class hierarchy, this breaks. The extension fails loudly: `anchor` still works, the rewind continuation message reports `⚠ reflection bootstrap missing — the rewind landed on disk but the next assistant turn may still see the pre-rewind context. Run \`/reload\` (or restart pi) to recover.`, and you'd see context corruption return on the next prompt.

  **Audited against pi 1.2.0 via `scripts/pi-upstream-probe.mjs` (2026-10; #80 re-audit):** the eliminated reflection points stay eliminated (`agent.prepareNextTurnWithContext`, replaced by the public `context` extension event, and the `prepareNextTurn` double-wrap). The remaining surface is `AgentSession.prototype.prompt` (session capture) and `agent.state.messages` (post-rewind refresh); both are NOT eliminable for the tool-based design: there is no public per-prompt hook for tool executes, and pi's own `navigateTree` refresh lives on `ExtensionCommandContext`, not the `tool.execute` ctx. #80 deleted the cache-preserving summary request and with it every #33/#75 read (`agent.transformContext` capture, `modelRegistry.streamSimple`, `agent.thinkingBudgets`, `settingsManager.getShowCacheMissNotices()`).

- **Anchor early in the turn.** Whatever's in `agent.state.messages` *before* the `anchor` tool call stays in the kept chain. Everything after gets summarized. Anchor at the *start* of a stage for maximum context savings.

- **Tiny rewinds are rejected by a minimum-savings floor.** A `rewind` whose measured savings falls below an internal floor (~4k tokens of apparent context freed) is refused with guidance listing the active anchors instead of executing — collapsing a near-empty segment burns a summary turn and can even grow live context once the summary and its continuation message land on the kept chain. This pairs with anchoring early: anchor at the start of a stage, then rewind only once real work has accumulated above the anchor.

- **Abandoned branches grow the JSONL forever.** Each rewind preserves the abandoned subtree on disk. Session files get bigger over time even as live context shrinks. For very long autonomous runs (days), session files can hit hundreds of MB.

- **The summary costs one live turn.** The summary turn reads the whole live context (from cache) and its output tokens are billed like any turn. A model that keeps calling tools instead of summarizing exhausts `MAX_SUMMARY_ATTEMPTS` and the rewind is cancelled.

- **Other extensions' `turn_end` drafts on the summary turn.** The rewind moves the branch inside its `turn_end` handler. Earlier handlers' `context_edit` drafts that target an entry the rewind abandoned are dropped (pi would otherwise reject the whole boundary list, the continuation included); a handler registered AFTER this extension that returns such a draft still makes pi discard the list, and the run ends on the summary turn with the rewind landed but unannounced.

- **Loading the extension monkey-patches `AgentSession.prototype.prompt` globally.** Every session in the host pi process picks up the patch on import, including sessions that never call `navigate_tree`. The patch is install-on-import and not reversible within a running pi process; restart pi to fully unload it.

- **`anchor:` is a reserved label prefix.** Any label written via the `/tree` view (`Shift+L` on the selected entry) or by another extension that begins with `anchor:` will be picked up by `list` and addressable by `rewind`'s `rewindTo` / `newLabel`. Avoid the prefix in manually-set labels.

- **Disk-fault during `rewind` (rare).** Pi's `branchWithSummary` advances the in-memory leaf before persisting the new entry to disk. If pi's session-write fails mid-call (full disk, FS error on a persisted session), the throw surfaces from the `turn_end` handler; the summary turn is already on disk, and the rewind can be re-issued. Production risk: low (pi's session-write is robust on POSIX disk).

## Development

```bash
pnpm install
pnpm test          # helpers + dispatch / in-band rewind / reflection bootstrap
pnpm run lint      # biome check extensions/
pnpm run typecheck # tsc --noEmit
```

Tests cover `extensions/navigate-tree/helpers.ts` (pure helpers in `helpers.test.ts`), `extensions/navigate-tree/index.ts` (action dispatch, schema shape, the in-band rewind: instruction result, tool-call gate, summary-turn completion / retry / cancel, context-event projection, reflection bootstrap, `session_tree` emit — in `index.test.ts`), and `extensions/navigate-tree/summary-prompt.ts` (the byte-exact r5d prompt pin and the stored-summary shape — in `summary-prompt.test.ts`). The suite is fully offline: the summary turn is simulated by appending an assistant message and firing `turn_end`. Manual e2e validation against the current pi release is recommended for any pi version bump (the reflection bootstrap depends on `AgentSession.prototype.prompt` / `agent.state.messages` field shapes, and the same-run continuation on pi's `turn_end` ordering).

### Live summary verification

The unit suite cannot prove that a real provider serves the summary turn from cache, that the continuation lands in the same run, or that the model's summary is scope-clean. For any change to the rewind flow, run an interactive session (tmux is fine) with the extension under test loaded (`pi -e <worktree>/extensions/navigate-tree/index.ts`, plus the provider's extension when it is one), gather some real context, and rewind to an anchor (e.g. the automatic `start`). Then check the session JSONL (`~/.pi/agent/sessions/--<cwd-slug>--/<file>.jsonl`):

1. **Cache gate** — the summary turn (the assistant entry after the rewind's instruction result) reports `cacheRead` ≈ the previous turn's context and a small `cacheWrite`.
2. **Same-run continuation** — the `navigate-tree-rewind` message (`status: "done"`) is followed by an assistant turn in the same run, with no user message in between, whose usage reflects the rewound context; the footer drops without a new prompt.
3. **Quality smoke** — `node scripts/summary-quality-check.mjs <jsonl>` checks the r5d headings/length/preamble; eyeball the printed summary for scope: branch only, pre-branch background excluded, unresolved work preserved, no continuation of the collapsed work.

## License

MIT — see [LICENSE](LICENSE).
