# Changelog

All notable changes to this project will be documented in this file.

## [calver-released]

<!-- USER-EDITABLE SECTION START -->
<!-- Add your curated release notes here. -->
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
