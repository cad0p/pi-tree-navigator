# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-05-25

<!-- USER-EDITABLE SECTION START -->

Initial release.

`navigate_tree` is an agent-callable pi tool with three actions:

- `anchor` — label the current point in the conversation as a milestone.
- `rewind` — collapse work between an anchor and the current leaf into a model-generated `branch_summary`, freeing context.
- `list` — show all anchors on the active branch with cumulative context %.

Designed for long autonomous sessions where the agent itself decides when to summarize. Survives mid-loop rewinds (the next assistant turn within the same `prompt()` call sees the reduced context) and produces structurally valid Anthropic chains by injecting a synthetic `tool_use` to pair with the rewind's `tool_result`.

<!-- USER-EDITABLE SECTION END -->


### 🚀 Features

- *(navigate-tree)* Initial extension: anchor / rewind / list actions
- *(navigate-tree)* In-loop context refresh via `agent.prepareNextTurn`, so rewinds free context for subsequent turns within the same `prompt()` call
- *(navigate-tree)* Synthetic-tool_call injection so Anthropic accepts the chain after a rewind

### Behavior

- *(navigate-tree)* Re-anchoring with a `name` already on the active branch moves the prior label to the new leaf instead of creating a duplicate.
- *(navigate-tree)* Anchor names follow strict kebab-case (lowercase alphanumeric segments separated by single hyphens; no leading/trailing/double hyphens).
- *(navigate-tree)* `rewind` requires a `summaryFocus` of ≥20 chars after trim; the rejection message lists what the focus should preserve so the agent can self-correct without user intervention.
- *(navigate-tree)* `list` and `rewind` surface a `⚠ reflection bootstrap missing` warning with a "Restart pi to recover" recovery hint when the `AgentSession.prototype` patch isn't installed (typically only after a pi internals shape change).
- *(navigate-tree)* `branch_summary` boilerplate strip is sentinel-anchored — it only fires when the entry begins with pi's actual prelude, so a user-authored doc whose first H2 happens to be `## Goal` is preserved untouched.
- *(navigate-tree)* Rewinds preserve chain validity even on partial label-write failure: every code path where `branchWithSummary` succeeds appends a synthetic tool_call exactly once, with best-effort retry of the labelEnd write and the original error wrapped with salvage detail before re-throw.
- *(navigate-tree)* `prepareNextTurn` chain composition propagates the prior wrapper's full `context` (systemPrompt, tools, future fields), overriding only `messages`. Co-installed extensions that mutate context survive across `/reload`.
- *(navigate-tree)* `anchor` writes the new label before clearing the prior, so a mid-collision crash leaves at most a brief two-label overlap (resolvable on the next leaf→root walk) rather than "prior cleared, new failed to install".
