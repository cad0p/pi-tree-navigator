# Changelog

All notable changes to this project will be documented in this file.

## [calver-released]

<!-- USER-EDITABLE SECTION START -->

Initial release.

`navigate_tree` is an agent-callable pi tool with three actions:

- `anchor` — label the current point in the conversation as a milestone.
- `rewind` — collapse work between an anchor and the current leaf into a model-generated `branch_summary`, freeing context.
- `list` — show all anchors on the active branch with cumulative context %.

Designed for long autonomous sessions where the agent itself decides when to summarize. Survives mid-loop rewinds (the next assistant turn within the same `prompt()` call sees the reduced context) and produces structurally valid Anthropic chains by injecting a synthetic `tool_use` to pair with the rewind's `tool_result`.

<!-- USER-EDITABLE SECTION END -->


### 🚜 Refactor

- Nest extension under extensions/navigate-tree/ per pi-napkin convention


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
