**The daily pi-upstream reflection probe failed.** Upstream pi changed one of the two remaining reflection points that `navigate-tree` relies on — the extension will silently degrade (or fail loudly at load) until adapted.

## What broke

Step: `STEP_PLACEHOLDER` — see the probe run for the exact FAIL line(s): RUN_URL_PLACEHOLDER

Probed against `@earendil-works/pi-coding-agent` **VERSION_PLACEHOLDER** (latest).

## The two remaining reflection points (after #14)

1. `AgentSession.prototype.prompt` — must remain a **writable plain data property** (the extension stashes the original and replaces it with a capture wrapper). If pi switches it to `#`-private or a getter-only accessor, the patch silently no-ops → `anchor`/`list` keep working but session capture dies → post-rewind `agent.state.messages` refresh returns false (warning surfaces in rewind output).
2. `agent.state.messages` — must remain a **plain writable field** on pi-agent-core's `Agent` (the extension assigns `agent.state.messages = sessionManager.buildSessionContext().messages` after a rewind). If pi makes `state`/`messages` `#`-private or read-only, the refresh throws (caught → returns false → stale messages until next prompt).

Plus transitive deps: `AgentSession` constructor must assign `this.sessionManager` (plain field), `SessionManager.prototype.buildSessionContext` must exist, `AgentSession`/`SessionManager` must stay exported.

## What to do

1. Read the probe output (run URL above) — it names the exact check that failed.
2. Inspect the corresponding pi dist source (`node_modules/@earendil-works/pi-coding-agent/dist/core/agent-session.js` / `pi-agent-core/dist/agent.js`) at the failing version.
3. Adapt `extensions/navigate-tree/index.ts` (and/or bump the peer floor in `package.json`) to the new shape.
4. Re-run the probe (`workflow_dispatch`) until green; add regression coverage.

## Risk window

The **shipped extension keeps running** on the last compatible pi (peer floor `>=0.81.0`); npm consumers on the NEW pi version will hit the break. No user action required beyond this issue.
