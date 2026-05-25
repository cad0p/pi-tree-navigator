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

User-visible specifics worth knowing on day one:

- Anchor names are kebab-case (lowercase alphanumeric segments separated by single hyphens; max 40 chars). Re-anchoring with a name already on the active branch moves the prior label to the new leaf rather than duplicating it; the same move-on-collision applies to `rewind`'s `labelEnd`.
- `rewind` requires a `summaryFocus` of ≥20 chars after trim; the rejection message lists what the focus should preserve so the agent can self-correct without user intervention.
- The `branch_summary` boilerplate strip in `list` hints is sentinel-anchored — a user-authored doc whose first H2 happens to be `## Goal` is preserved untouched.
- If the `AgentSession.prototype` patch isn't installed (typically only after a pi internals shape change), `list` and `rewind` surface a `⚠ reflection bootstrap missing` warning. The hint suggests `/reload` first (lighter — re-runs the prototype patch on the current process) and `Restart pi` as the heavier-handed alternative.

<!-- USER-EDITABLE SECTION END -->
