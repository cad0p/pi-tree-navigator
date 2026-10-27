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
  - `@earendil-works/pi-ai >=0.87.0` (`isContextOverflow` / `isRecoverableLength`, the overflow checks rewind-driven compaction shares with pi)
  - `typebox ^1.0.0` (used to declare the tool's parameter schema; bundled with pi but listed explicitly so a standalone install resolves correctly).
- The reflection bootstrap depends on plain (not `#`-private) internal pi/agent fields: `AgentSession.prototype.prompt` (patched for session capture) and `agent.state.messages` (refreshed after a rewind so the next prompt snapshots the rewound chain). After a rewind the extension also emits `session_tree` through the captured session's plain `_extensionRunner` field (#79; see [History-mirroring providers](#history-mirroring-providers-pi-claude-bridge)). Per-turn in-loop context refresh runs through the **public** `context` extension event (see “In-loop context refresh” below) — no `agent.prepareNextTurn*` reflection. The rewind's same-run continuation relies on pi's `turn_end` boundary contract (handlers awaited before the steering drain, returned entries committed, `continue` honoured; public since 0.87). Verified against pi 1.3.0 via `scripts/pi-upstream-probe.mjs`.

## What you get

A single agent-callable tool, `navigate_tree`, with three actions:

| action | params | effect |
|---|---|---|
| `anchor` | `name` | Label the current point in the conversation as a milestone. |
| `rewind` | `rewindTo`, `newLabel`, `summaryFocus` | Collapse work between `rewindTo` and the current leaf into a `branch_summary` entry. The tool result only reports the rewind as accepted; the summary instruction follows as a `navigate_tree` message, the model's next reply is the summary, and the extension then branches with it (see [In-band summary](#in-band-summary-80)). A rewind must be the only tool call in its message. The summary is itself labeled with `newLabel`, so you can chain rewinds. `summaryFocus` is required (non-trivial focus required; floor enforced at runtime by `MIN_SUMMARY_FOCUS_LENGTH`). Despite the verb, `rewind` does not restore prior state — it forks a sibling branch from `rewindTo` and continues forward from a model-generated summary; the original subtree is preserved on disk but no longer on the active path. |
| `list` | — | Show all anchors on the active branch with cumulative context %. |

`name` (written by `anchor`) and `newLabel` (written by `rewind`) both share the reserved `anchor:` label prefix; `rewindTo` resolves against that same namespace. Every label written by `anchor` and every `newLabel` written by `rewind` is referenceable by any subsequent `rewind`'s `rewindTo`, and `list` shows all of them. A completed rewind also writes `branch:<newLabel>` on the summary turn of the abandoned branch, so `/tree` can jump back to the full collapsed segment; `branch:` labels are not anchors (`rewindTo` does not resolve them, `list` does not show them).

Labels are unique on the active branch: `anchor` refuses a `name` and `rewind` refuses a `newLabel` that already exists (the `rewind` check runs before summarization), and an existing label is never moved or cleared. The old move-on-collision rename-forward idiom (`anchor x` → `rewind(newLabel=x)`) is retired — a rewind needs a fresh `newLabel`; the existing label stays addressable via `rewindTo`. Manual `/tree` labels can still duplicate; `list` shows every match and `rewindTo` resolves leaf-most. The extension itself also writes one silent label per session leg: `anchor:start` (see [Automatic start anchor](#automatic-start-anchor)).

**Anchoring is mandated, not suggested.** On every agent start the extension appends a one-line mandate to the end of the system prompt (`before_agent_start`), gated on the tool being active: `navigate_tree: gather all context, then anchor \`context-gathered\`; list anchors and rewind after every milestone or rabbit hole / dead end to keep context low.` The append lands after project context and skills, is re-applied on every prompt, and survives compaction — unlike the `promptGuidelines` bullet it replaced. ~40 tokens (159 chars / 4), constant for prompt caching.

## Configuration (`tree-navigator.json`)

Both the rewind hint and the automatic start anchor read one two-layer config file:

| Layer | Path |
|---|---|
| Global | `join(getAgentDir(), "tree-navigator.json")` (`PI_CODING_AGENT_DIR`-aware) |
| Project | `join(cwd, CONFIG_DIR_NAME, "tree-navigator.json")` (`.pi/` by default; read only when the project is trusted — untrusted projects are ignored silently) |

```json
{ "rewindHintReserveTokens": 50000, "anchorStartAfterTurns": 0 }
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

When the crossing has at least one `anchor:` label on the active branch, the agent receives a persisted custom message naming every active anchor with its cumulative context percentage — so the model knows a rewind target exists (including the automatic `start`) without a `list` call. The message prints the percentage only, not a tokens-left count: a model reads "N tokens left" as "I am running out of tokens" and stalls instead of rewinding (#114). Mid-run this steers the running loop (one reaction turn in the same run); when the agent is idle it only appends and waits for the user's next prompt, so an idle-agent rewind stays user-confirmed:

```text
[navigate_tree hint] Context is at 95.2% of 1.0M. Persist what matters to files now, then rewind to the oldest appropriate anchor.
Active anchors (root → leaf, % of 1.0M):
   4.1%  start
  12.0%  context-gathered
```

The same anchor table (cumulative context % at each anchor, root → leaf) ends the min-savings refusal and the forced-rewind message, and is the body of `list` (which adds an `(after: “…”)` preview column).

With no anchors the extension sends **no model message** (a rewind is impossible) and shows a TUI-only warning explaining how to add one manually: `/tree`, select the entry to rewind to, press `shift+l`, label it `anchor:<name>` (e.g. `anchor:context-gathered`), then ask the agent to rewind to it.

### Per-model reserve (`modelOverrides`)

Different models need different budgets: a model that writes long summaries or grows context fast per turn needs a larger reserve. `modelOverrides` sets `rewindHintReserveTokens` per model. It is keyed like pi's `compaction.modelOverrides`, but each entry takes `rewindHintReserveTokens` (not pi's `reserveTokens`) with this extension's value rules, including the disable sentinels pi does not accept:

```json
{
  "rewindHintReserveTokens": 40000,
  "modelOverrides": {
    "opencode-go/deepseek-v4.1-flash": { "rewindHintReserveTokens": 50000 }
  }
}
```

- Keys are the exact `provider/modelId` of the active model (as in pi's `/model` list). The reserve is resolved against the live model on every turn, so a mid-session `/model` switch takes effect on the next turn. The crossing is shared across the switch and re-arms whenever the tokens left exceed the active model's reserve, so switching to a model with a smaller reserve (or one with the hint off) can re-arm it and a later switch back can fire again.
- A matching entry wins over the base `rewindHintReserveTokens`, including an entry sentinel (`"off"` disables the hint for that model only) and an entry that enables the hint while the base is off. No matching entry, or no active model → the base value.
- Layers merge per key: a trusted project entry that sets `rewindHintReserveTokens` replaces the global entry for the same model (an entry without it is ignored, so the global entry stays), other keys pass through. As in pi, a matching entry from either layer beats the base value from either layer, so a global model entry beats a project-wide base.
- Entry values follow the base field's rules. An invalid entry value disables the hint for that model (no fallback to the base) with a field-scoped warning. A non-object `modelOverrides` or entry warns and is ignored; an entry without `rewindHintReserveTokens` (e.g. a pasted pi `reserveTokens` entry) is ignored, and so are other keys inside an entry.

### Rewind hygiene (always on)

Independent of the hint, the tool ships two static `promptGuidelines` bullets, present whenever `navigate_tree` is active: **when rewinding, prefer the oldest anchor that keeps what you'd otherwise re-read or re-derive**, persisting durable findings to files first; and **don't rewind while a user decision or unresolved question is pending** — ask the user instead. The bullets are not config-gated (`registerTool` fixes the array at registration), cost ~64 tokens per request (257 chars / 4), and change the cached system-prompt prefix once per active session on upgrade (the provider-facing tool schema is unchanged).

### Rewind-driven compaction

`rewindHintReserveTokens` is one switch for two things: the hint (an early warning, above) and **rewind-driven compaction**, which takes over when the context still overflows. Wherever the hint is on for the active model (and `navigate_tree` is active), a real overflow forces a rewind instead of ending the run. It needs pi's auto-compaction off; see [Compaction guidance](#compaction-guidance).

It acts only on a real overflow, detected with pi's own checks (`isContextOverflow` / `isRecoverableLength` from `@earendil-works/pi-ai`): the provider's context-length error, a silent overflow (input over the window), or a reply cut short because the context left no room for output. Only a reply from the active model counts, as in pi. There is no prediction and no capping of tool output.

On an overflow, in the same run:

1. The failed reply is hidden from the model (a provider error, or a cut-off reply and its failed tool calls; a silent overflow's complete reply stays).
2. The newest tool results of the batch that caused it are replaced, one by one, until the context fits `contextWindow − compaction.reserveTokens` (pi's reserve for the active model, `compaction.modelOverrides` included) under BOTH counts of it: the calibrated estimate below, and the one pi's output clamp computes for the request that follows — pi-ai's chars/3.5 estimate over the prefix the extension hands back plus every message after it, which leaves `reserveTokens − 4096` of output room. Fitting only the chars/4 estimate is what starved the forced turn in #107 (a 31,530-token request inside the target was still clamped to one output token). When the failed reply carries usage (a cut-off reply or a silent overflow), the calibrated estimate's tail is scaled by the provider's real count: ratio = (real input + cache read + cache write − newest usage) ÷ (estimated tail), never below 1. Each one reads: `[navigate_tree] Output withheld: adding it overflowed the context window (~19k tokens). The call ran; only its output was dropped. Rewind first, then re-run it if you still need the output.` User messages and other extensions' messages are never withheld.
3. The model gets the forced-rewind message:

```text
[navigate_tree] Context overflowed the window (100k): 2 tool results were withheld. Rewind now — only navigate_tree rewind is allowed until it lands.
Active anchors (root → leaf, % of 100k):
   4.1%  start
  12.0%  context-gathered
```

Until the rewind lands, every other tool call is blocked (`list` and `anchor` included). A `codemode` call passes so its script can rewind, but a nested call other than `rewind` is blocked (the gate still means rewind only). A reply without a `rewind` call is hidden and the same request is sent again, 3 attempts in total. A `rewind` refused by validation (min-savings floor, unknown anchor) stays visible and does not count. An accepted `rewind` runs the normal in-band flow (summary, branch, continuation), and the gate lifts when it lands. The hint does not also fire during it.

When it cannot recover, the run ends with one error message, and the gate lifts:

```text
[navigate_tree] Context overflow recovery failed: <reason>. The session is at the context limit; rewind manually with /tree, /compact, or start a new session.
```

Reasons: `the context overflowed again during the forced rewind` (one forced rewind per overflow, as pi gives up after one compact-and-retry), `no rewind after 3 attempts` (also when the rewind's summary fails 3 times), `no anchor to rewind to` (an overflow before any anchor exists; the automatic `start` is not written on an overflowed run), `the context is still over the limit after withholding tool results` (an overflow while a voluntary rewind waited for its summary, when withholding cannot make room; also when a fresh forced plan cannot fit the target after withholding, where the forced message would only be starved again), `the forced rewind was aborted` (Esc), and `the run ended before the rewind`. Another rewind failure during a forced rewind ends it with that rewind's reason.

An overflow while a voluntary rewind waits for its summary keeps the model's rewind: the summary attempt is hidden, tool results before the `rewind` call are withheld, and the summary request is sent again.

There is no limit after a landing: if the rewind freed too little and the context overflows again later, that is a fresh forced rewind.

### Compaction guidance

**Either-or.** With `rewindHintReserveTokens` set and pi's `compaction.enabled` true, both the hint and rewind-driven compaction are off, and each crossing shows this TUI error instead of the hint: `navigate_tree: the rewind hint and rewind-driven compaction are off because pi's auto-compaction is on. Disable compaction.enabled or remove rewindHintReserveTokens.` pi's default is `true`, so set it off in pi's `settings.json` (global or trusted project):

```json
{ "compaction": { "enabled": false } }
```

`anchor`, `list` and `rewind` keep working either way, and pi's own compaction (when on) works as before. pi's compaction settings are read from the session's live settings (the ones pi decides from) at each crossing and each overflow, so a `/settings` change applies at the next one. Before the session is captured (its first prompt) they are read from disk with the session's project-trust decision.

**Headless sessions (subagents).** A headless session with the hint set and pi's compaction on (pi's default, or enabled by a project or agent) runs silently without the hint and without rewind-driven compaction: there is no UI for the error.

**Reserve ordering.** The hint must fire before rewind-driven compaction's target: `rewindHintReserveTokens` ≥ the active model's `compaction.reserveTokens` (`compaction.modelOverrides["<provider>/<modelId>"].reserveTokens`, else `compaction.reserveTokens`, default 16384). Size it well above that to leave room for the persist and summary turns. With compaction off, session start warns per model when it is below (both still run), e.g.: `rewindHintReserveTokens (10000) is below pi's compaction.reserveTokens (16384) for anthropic/claude-haiku-5-5: rewind-driven compaction may fire before the hint, giving the model no time to prepare and persist what matters to files.`

**Room for the forced turns.** The target `contextWindow − compaction.reserveTokens` must hold the forced message, the rewind call, the summary instruction and the summary, thinking included. At high thinking levels size pi's reserve per model, e.g.:

```json
{ "compaction": { "modelOverrides": { "anthropic/claude-haiku-5-5": { "reserveTokens": 30000 } } } }
```

The thinking level is not changed for the forced turns, so they stay cached.

## Automatic start anchor

Independent of the hint, the extension writes a silent `anchor:start` label so every session leg has a guaranteed low-floor fallback rewind target with no user action. By default (X=0) it labels the first user prompt — the leaf right after it, before the first assistant reply — so the floor is only the base load (system prompt + first message). With X ≥ 1 it labels the X-th turn of the agent run instead (or the run's end if it finishes earlier). `start` shows up in `list` output like any other anchor; nothing else model-visible changes.

Why 0 (#116): on small windows the first turns are expensive. Haiku 5.5 subagents (100k) measured 12–15k at the first prompt but 19–40k after two turns, and a session landing at 40% on every rewind refilled the window with re-reads before making any edit. In isolated replays the first-prompt anchor gave more working room and did not make the retained first prompt take over after a rewind (the stale-request re-answers happened at X=2, where the retained turns show the agent working on the old request). It also gives rewind-driven compaction an anchor when the very first reply overflows (#104).

`anchorStartAfterTurns` (see [Configuration](#configuration-tree-navigatorjson)):

- Value: integer **0–50** inclusive. `0` = the first user prompt; `N ≥ 1` = the N-th turn of the run. A bare number is canonical (`0`); numeric strings (`"0"`) are accepted. Default **0**.
- **Non-optional**: unlike the hint, an absent `anchorStartAfterTurns` — or no config file at all — still fires at the default; absence is never a warning.
- Invalid value (out of range, decimal, non-numeric string, `null`, `false`, object): a field-scoped warning, then the other layer's valid value applies, else the default 0 — a touched-but-invalid value must not disable a non-optional feature.
- Precedence: trusted project > global > default 0.
- Kill switch: `PI_NAVIGATE_TREE_START_ANCHOR=0` suppresses the write, read at write time (a mid-process change applies without `/reload`).

Behavior and edge cases:

- **First prompt (X=0)**: written when the run's first assistant message starts (pi persists the prompt, and any context messages sent with it, just before), so it lands even if that reply errors or overflows. The first `turn_end` and `agent_end` back it up.
- **Run-bounded (X ≥ 1)**: X counts native turns of the current agent run. A run shorter than X anchors at its end (`agent_end`); `agent_end` fires on completed, aborted, and errored runs, so settlement always happens by the end of the first run (with X=2, even a one-turn run anchors at turn 1). An overflowed turn is never anchored.
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
  → [navigate_tree] Rewind to 'impl-start' accepted (collapse as 'impl-end'). Follow the
    navigate_tree message below to complete it.
  → [navigate_tree]
    Summarize only messages 4 onwards in the conversation above ... (the summary instruction)

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

1. **The summary is written in-band (#80).** The rewind tool does not call a summarizer. The summary instruction arrives as a `navigate_tree` message right after the rewind's tool result, so the live model writes the summary as its next turn and the tree moves only after that turn ends (see [In-band summary](#in-band-summary-80)). Nothing is in flight when `branchWithSummary` runs, so the rewind's own `tool_use`/`tool_result`, the instruction message and the summary turn all stay on the abandoned branch and the new branch is structurally valid without any synthetic message.

2. **In-loop context refresh.** Pi's `Agent` class snapshots `state.messages` once at the start of `prompt()` and pushes new messages onto its own array. A rewind issued mid-loop wouldn't reduce the next API call's size until the user sent a fresh prompt. We register a handler on pi's **public `context` extension event** — fired via `Agent.transformContext` → `runner.emitContext` before *every* LLM call, including the turn right after a mid-loop rewind — that returns the conversation-only session projection (`sessionManager.buildSessionProjection().messages`, context-edit-aware; Pi re-attaches the current system head itself). After a rewind, the very next assistant turn within the same `prompt()` sees the rewound chain. (This replaced the earlier `agent.prepareNextTurnWithContext` reflection wrapper.)

3. **Reflection bootstrap.** Pi's slash-command `navigateTree` has access to `commandCtx.navigateTree`, which mutates `agent.state.messages`. Tool executes don't get that ctx, so we capture every `AgentSession` instance via the prompt patch and replicate the mutation manually. Without it, the on-disk leaf moves but `agent.state.messages` stays stale.

4. **`summaryFocus` is mandatory.** The summary is the only thing the agent will see of the collapsed work. The first time the agent uses `rewind`, blanket prompts produce vague summaries; subsequent rewinds are weaker. Forcing the agent to articulate `summaryFocus` (appended to the summary instruction as `Additional focus:`) measurably improves what survives.

### In-band summary (#80)

Earlier releases summarized with a separate request: first pi's cold `generateBranchSummary` (re-billing the whole branch), then a replay of the captured live request (#33/#75) that hit the cache only while the replay stayed byte-identical to the live turn, with fallbacks (`stale-capture`, `capture-diverged`, `overflow`, ...) that re-billed cold. A separate request also cannot reach a provider that serves turns from its own session, such as pi-claude-bridge. The summary is now just the next turn of the live conversation:

1. **`rewind` (tool call).** Refuses a rewind that shares its assistant message with other tool calls (`rewind must be the only tool call in its message — re-issue only the rewind, alone. The other calls in this message ran normally.`) and a rewind issued from a codemode script that is not the script's first nested call (the same rule, without the last sentence: `rewind must be the only tool call in its message — re-issue only the rewind, alone.`). A script whose first nested call is a `rewind` is accepted like a direct one only if the script issues no other nested call: a sibling still in flight refuses it with the batch message above, and the siblings run normally. Otherwise validates as before (names, `summaryFocus`, label uniqueness, the minimum-savings floor), records a pending rewind for the session, and returns a short result: `[navigate_tree] Rewind to '<rewindTo>' accepted (collapse as '<newLabel>'). Follow the navigate_tree message below to complete it.` The tree does not move yet.
2. **The instruction message.** The rewind turn's `turn_end` returns a `navigate-tree-summary-request` custom message as a boundary entry with `continue: true`: `[navigate_tree]`, a newline, then the eval-approved r5d instruction with `{first}` (the 1-based number of the first collapsed message in the projection the model sees) and `Additional focus: <summaryFocus>`. pi commits it right after the rewind's tool result and before queued steers. The model receives it as a user message rather than as a tool result, so it no longer reads like an instruction injected through a tool. In the TUI it renders as one line, `[rewind to '<rewindTo>' · collapse as '<newLabel>'] · summarizing · context 30.4% → ~12.0% of 1.0M` (the `~` figure is the estimate at the anchor before the summary is added), and expands to the full text.
3. **The summary turn.** The model's next reply is the summary: an append to the live conversation, so it is served from the prompt cache like any other turn. While the rewind is pending, the request is frozen at the instruction message: the `context` handler drops everything after it, so steers and other extensions' messages that arrive meanwhile are not sent, and every retry sends the same messages. Every tool call is blocked (TUI-only reason: `navigate_tree: blocked: the model tried to call a tool instead of summarizing.`).
4. **`turn_end`.** A text-only turn with the r5d `## Goal` heading completes the rewind: any preamble before `## Goal` is dropped, the summary turn is labeled `branch:<newLabel>`, the stored `branch_summary` gets upstream's preamble and `<read-files>`/`<modified-files>` lists (tool calls of hidden attempts excluded), `branchWithSummary` forks from the anchor, the label is written, `agent.state.messages` is refreshed and `session_tree` is emitted. The continuation message (`customType: "navigate-tree-rewind"`) is delivered in the same run, so the rewound context reaches a real turn — and the footer's token count updates — without a user prompt.

A summary attempt fails when it calls a tool, has no `## Goal` summary (including an empty or thinking-only reply), or is cut off (`stopReason: "length"`); a cut-off reply that also calls a tool counts as cut off. A failed attempt is hidden from model context (`context_edit` with `null` on the assistant and all its tool results, together) and the same request is sent again with nothing appended. After `MAX_SUMMARY_ATTEMPTS` (3 in total) the rewind fails: the instruction and the attempts are hidden, the rewind's tool result is replaced with the failure text for the last attempt's cause, and it is sent as a tool error (`isError: true`):

```
Summarization failed: no summary after 3 attempts (last: cut off). Nothing was collapsed. Try again with a tighter summaryFocus.
Summarization failed: no summary after 3 attempts (last: called a tool). Nothing was collapsed. Try again; the reply after the rewind must be only the summary.
Summarization failed: no summary after 3 attempts (last: no '## Goal' summary). Nothing was collapsed. Try again; the reply after the rewind must be only the summary.
```

Every other way a pending rewind can end goes through the same path, so the model always sees its rewind call fail and no instruction is left behind in context:

- Esc during the rewind turn or a summary attempt, a new prompt while pending, a run that settles after a provider error with no retry, or a restart with a rewind in flight: `Summarization aborted. Nothing was collapsed.` (an aborted attempt is hidden too; no `continue`).
- `Summarization failed: <reason>. Nothing was collapsed. Try again.` with `continue: true` where a boundary is available. The reasons are: `'<rewindTo>' is no longer on the active branch`, `a label '<newLabel>' now exists on the active branch`, `the branch could not be written (<error>)`, `the context was compacted` (detected at `session_compact`, before the next request), `the run ended before the summary` (`agent_before_settle`, when another handler left the run settling), and `the summary request could not be delivered` (the instruction message never reached the session, e.g. because another extension's invalid boundary draft made pi drop the list).

A provider error keeps the rewind pending without spending an attempt (pi's auto-retry or pi-fallback-provider continues the run); `agent_end` no longer drops it. Each ending is persisted: at a boundary as drafts, otherwise with `appendContextEdit` and `appendEntry`. A `navigate-tree-rewind-outcome` custom entry (`toolCallId`, `status`, `reason`, `text`, `rewindTo`, `newLabel`, `hiddenIds`, `toolResultEntryId`) records it. It renders in the TUI as `[rewind to '<rewindTo>' · collapse as '<newLabel>'] · failed: <reason> · nothing collapsed` (or `· aborted · nothing collapsed`). The `context` handler uses it to send the rewind's tool result with `isError: true`. On `session_start`, an instruction on the active branch with neither a `branch:<newLabel>` label nor an outcome entry after it gets an aborted ending.

Input that arrives after the rewind call (user steers, other extensions' messages such as subagent results or timers) is hidden while the rewind is pending. On success it is re-appended onto the new branch in its original order, and the continuation message flags it as new and still unanswered. On failure it simply becomes visible again, right after the failed rewind's tool result. To pick a different anchor, press Esc and ask again; a steer without Esc does not cancel the rewind.

The continuation message is returned as a `turn_end` boundary entry with `continue: true`: pi commits it right after the handler, before draining queued steers, and guarantees one more provider request — so it lands ahead of anything typed meanwhile and the run continues even after a text-only summary turn. Its `details` carry `status`, `rewindTo`, `newLabel`, `targetId`, `summaryId`, `collapsedEntries`, `contextBefore`/`contextAfter`, `contextWindow`, `summaryAttempts`, `agentMessagesRefreshed`, `historyRewriteNotified`, `reappendedLateMessages`, and the file lists. The rewind hint stays silent on any turn a rewind owns.

### History-mirroring providers (pi-claude-bridge)

[pi-claude-bridge](https://www.npmjs.com/package/pi-claude-bridge) serves each turn from its own Claude Code session and rebuilds that session from pi's history only when pi signals a rewrite (`session_tree` / `session_compact`). A tool-driven rewind skipped that signal, so the Claude Code query parked at the `navigate_tree` call kept answering with the whole pre-rewind context: the footer stayed at the old percentage because the real request did too (#79).

After every rewind the extension emits `session_tree` with the payload pi's own `navigateTree` emits (`newLeafId`, `oldLeafId`, `summaryEntry`, `fromExtension: true`), after the `agent.state.messages` refresh. The bridge then rebuilds its session from the rewound chain on the next request. `details.historyRewriteNotified` on the continuation message records whether the event went out (`false` when the runner is unreachable or the emit threw; the rewind itself is complete either way).

Cache behavior on the bridge:

- **Summary turn.** The summary instruction message and its hidden re-sends (#90) are untested on history-mirroring providers.
- **First turn after the rewind.** That turn is a bridge session rebuild, which per the bridge's README usually misses the prompt cache past the system prompt. It is a one-time write of the rewound chain (tens of k) instead of re-reading the full pre-rewind context on every turn.

## Limitations

- **Brittle to pi version bumps.** The fix uses two independent reflection points on internals that aren't part of pi's public API: `AgentSession.prototype.prompt` (session capture) and `agent.state.messages` (refreshed after a rewind). A soft one (#79) is `AgentSession._extensionRunner`, used only to emit `session_tree` after a rewind; if it disappears the rewind still works, but history-mirroring providers (pi-claude-bridge) stop rebuilding and `details.historyRewriteNotified` reports `false`. A former third reflection point (`agent.prepareNextTurn*`) was eliminated in v0.2.0 via the public `context` extension event; the per-turn systemPrompt/tools/model/thinkingLevel refreshes come from pi's own `_installAgentNextTurnRefresh` (construction-installed since 0.80.3). If a future pi release renames the two remaining fields, switches them to private (`#`) fields, or restructures the class hierarchy, this breaks. The extension fails loudly: `anchor` still works, the rewind continuation message reports `⚠ reflection bootstrap missing — the rewind landed on disk but the next assistant turn may still see the pre-rewind context. Run \`/reload\` (or restart pi) to recover.`, and you'd see context corruption return on the next prompt.

  **Audited against pi 1.2.0 via `scripts/pi-upstream-probe.mjs` (2026-10; #80 re-audit):** the eliminated reflection points stay eliminated (`agent.prepareNextTurnWithContext`, replaced by the public `context` extension event, and the `prepareNextTurn` double-wrap). The remaining surface is `AgentSession.prototype.prompt` (session capture) and `agent.state.messages` (post-rewind refresh); both are NOT eliminable for the tool-based design: there is no public per-prompt hook for tool executes, and pi's own `navigateTree` refresh lives on `ExtensionCommandContext`, not the `tool.execute` ctx. #80 deleted the cache-preserving summary request and with it every #33/#75 read (`agent.transformContext` capture, `modelRegistry.streamSimple`, `agent.thinkingBudgets`, `settingsManager.getShowCacheMissNotices()`).

- **Anchor early in the turn.** Whatever's in `agent.state.messages` *before* the `anchor` tool call stays in the kept chain. Everything after gets summarized. Anchor at the *start* of a stage for maximum context savings.

- **Tiny rewinds are rejected by a minimum-savings floor.** A `rewind` whose measured savings falls below an internal floor (~4k tokens of apparent context freed) is refused with guidance listing the active anchors instead of executing — collapsing a near-empty segment burns a summary turn and can even grow live context once the summary and its continuation message land on the kept chain. This pairs with anchoring early: anchor at the start of a stage, then rewind only once real work has accumulated above the anchor.

- **The estimate can undercount.** The chars/4 estimate ran about 1.45× low on a live run of 15 parallel reads. The real-usage ratio corrects this only when the overflowed reply carries usage (a cut-off reply or a silent overflow). An explicit provider context-length error has none, so its ratio is 1 and a miss falls through to the second-overflow path (D8).

- **Abandoned branches grow the JSONL forever.** Each rewind preserves the abandoned subtree on disk. Session files get bigger over time even as live context shrinks. For very long autonomous runs (days), session files can hit hundreds of MB.

- **The summary costs one live turn.** The summary turn reads the whole live context (from cache) and its output tokens are billed like any turn. Each retry re-sends the same messages, so it is a cached turn too (a retry on a different model, e.g. after a pi-fallback-provider switch or a manual `/model` change, is a cold cache). A model that keeps calling tools instead of summarizing exhausts `MAX_SUMMARY_ATTEMPTS` and the rewind fails with a tool error.

- **Hidden attempts stay in the TUI and on disk.** pi does not re-render on `context_edit`, so failed attempts and their blocked calls stay on screen and in the JSONL; only model context drops them. After a failure the rewind's tool-result row in the TUI and its JSONL entry still read "accepted" (the result was rendered before it was replaced); the `navigate-tree-rewind-outcome` entry and the `context_edit` entries are authoritative.

- **Settle-time endings and later extensions.** `agent_before_settle` handlers run in load order. If an extension loaded after this one sets `continue` for the run, the rewind may already have ended with `the run ended before the summary`. On pi 1.3.0, `before_agent_start` also fires for runs started by an extension's `sendMessage`, and that ends a pending rewind as aborted like a new prompt does.

- **Other extensions' `turn_end` drafts on the summary turn.** The rewind moves the branch inside its `turn_end` handler. Earlier handlers' `context_edit` drafts that target an entry the rewind abandoned are dropped (pi would otherwise reject the whole boundary list, the continuation included); a handler registered AFTER this extension that returns such a draft still makes pi discard the list, and the run ends on the summary turn with the rewind landed but unannounced.

- **Rewind-driven compaction loses the withheld output.** The model sees only the notice; it must re-run the call after the rewind if it still needs the output. Withheld results and hidden attempts still show in full in the TUI and the JSONL (pi does not re-render on `context_edit`).

- **Rewind-driven compaction: same model only.** An overflow is handled only when the failed reply's provider and model match the active model; with a virtual model selection (pi resolves the physical model internally) it is skipped and the run ends on the error, as without this extension.

- **Rewind-driven compaction on pi < 1.3.0: z.ai's overflow message.** `@earendil-works/pi-ai` before 1.3.0 does not recognize z.ai's overflow error (`prompt exceeds max length`, the CN endpoint), so that overflow is not detected and ends the run on the error, as without this extension.

- **The reserve-order warning reads pi's settings from disk.** The `session_start` warning (hint reserve below pi's `compaction.reserveTokens`), and rewind-driven compaction before the session is captured, read pi's settings from disk with the session's project trust. An SDK host that runs a session with in-memory settings, or different trust, can disagree with what they read; the disk source is also shared by sessions in one process (like the hint state).

- **Rewind-driven compaction and later extensions.** At `turn_end` and `agent_before_settle` it stands down when an earlier handler already set `continue` (e.g. a fallback provider retrying on a larger model). An extension loaded after this one that also recovers runs after the drafts were returned.

- **`codemode.mode: "only"`.** With `navigate_tree` reachable only from `codemode` scripts, a rewind is issued from a script whose first nested call is `navigate_tree` `rewind` (for example `await navigate_tree({ action: "rewind" })`). The same applies to voluntary rewinds: a script's first nested rewind is accepted whether or not a forced rewind is active. A script that makes no rewind counts as a failed forced attempt (`no rewind after 3 attempts`). Keep `navigate_tree` a direct tool (`codemode.mode: "on"`, pi's default) if you do not want rewinds to go through scripts.

- **A restart during a forced rewind** leaves the forced-rewind message in context with no gate.

- **Loading the extension monkey-patches `AgentSession.prototype.prompt` globally.** Every session in the host pi process picks up the patch on import, including sessions that never call `navigate_tree`. The patch is install-on-import and not reversible within a running pi process; restart pi to fully unload it.

- **`anchor:` and `branch:` are reserved label prefixes.** Any label written via the `/tree` view (`Shift+L` on the selected entry) or by another extension that begins with `anchor:` will be picked up by `list` and addressable by `rewind`'s `rewindTo` / `newLabel`. `branch:<newLabel>` marks the summary turn of a completed rewind; clearing or renaming it makes a restart on that branch read the rewind as unfinished. Avoid both prefixes in manually-set labels.

- **Disk-fault during `rewind` (rare).** Pi's `branchWithSummary` advances the in-memory leaf before persisting the new entry to disk. If pi's session-write fails mid-call (full disk, FS error on a persisted session), the throw surfaces from the `turn_end` handler; the summary turn is already on disk, and the rewind can be re-issued. Production risk: low (pi's session-write is robust on POSIX disk).

## Development

```bash
pnpm install
pnpm test          # helpers + dispatch / in-band rewind / reflection bootstrap
pnpm run lint      # biome check extensions/
pnpm run typecheck # tsc --noEmit
```

Tests cover `extensions/navigate-tree/helpers.ts` (pure helpers in `helpers.test.ts`), `extensions/navigate-tree/index.ts` (action dispatch, schema shape, the in-band rewind: accepted result, solo/nested guards, instruction message, frozen request, hidden re-sends, every ending and its outcome entry, rewind-driven compaction (overflow signals, withholding, forced-rewind gate and attempts, final error, either-or with pi's compaction), TUI renderers, tool-call gate, summary-turn completion, context-event projection, reflection bootstrap, `session_tree` emit — in `index.test.ts`), and `extensions/navigate-tree/summary-prompt.ts` (the byte-exact r5d prompt pin and the stored-summary shape — in `summary-prompt.test.ts`). The suite is fully offline: the summary turn is simulated by appending an assistant message and firing `turn_end`. Manual e2e validation against the current pi release is recommended for any pi version bump (the reflection bootstrap depends on `AgentSession.prototype.prompt` / `agent.state.messages` field shapes, and the same-run continuation on pi's `turn_end` ordering).

### Live summary verification

The unit suite cannot prove that a real provider serves the summary turn from cache, that the continuation lands in the same run, or that the model's summary is scope-clean. For any change to the rewind flow, run an interactive session (tmux is fine) with the extension under test loaded (`pi -e <worktree>/extensions/navigate-tree/index.ts`, plus the provider's extension when it is one), gather some real context, and rewind to an anchor (e.g. the automatic `start`). Then check the session JSONL (`~/.pi/agent/sessions/--<cwd-slug>--/<file>.jsonl`):

1. **Cache gate** — the summary turn (the assistant entry after the `navigate-tree-summary-request` message) reports `cacheRead` ≈ the previous turn's context and a small `cacheWrite`.
2. **Same-run continuation** — the `navigate-tree-rewind` message (`status: "done"`) is followed by an assistant turn in the same run, with no user message in between, whose usage reflects the rewound context; the footer drops without a new prompt.
3. **Quality smoke** — `node scripts/summary-quality-check.mjs <jsonl>` checks the r5d headings/length/preamble; eyeball the printed summary for scope: branch only, pre-branch background excluded, unresolved work preserved, no continuation of the collapsed work.

## License

MIT — see [LICENSE](LICENSE).
