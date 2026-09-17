# Verification — CI gates, independent review, offline gates, live proof

This is the verification checklist `/impl` looks for: what CI proves on every PR, how independent review works in this repo, the offline gates every change runs, and what must be proven live for which change type. It composes the repo's authoritative sources — `.github/workflows/*`, README “Development” and “Live summary verification”, `scripts/` — into one file. If a needed signal is missing here, that is a docs gap: add it in the same PR that needed it.

## 1. How to use this file

1. Identify your change type in §5 and note its live-proof row.
2. Run the offline gates (§4) locally in your worktree.
3. Push and confirm CI is green (§2) on the exact head SHA.
4. Get independent review (§3) — verdicts recorded on that SHA, fresh instances for fixes.
5. Prove live (§5) whenever the change type has a live surface — real output, never a substitute.
6. Close out (§6): evidence in the PR body, kanban + vault updated, PR left **draft** (merge is the owner's call).

## 2. CI gates

PR checks (`.github/workflows/ci.yml`, `validate-package-version.yml`, `validate-release-pr.yml`):

| Check | What it proves |
|---|---|
| `test (ubuntu-latest)` / `test (macos-latest)` | Matrix job: `biome check extensions/` (lint), `tsc --noEmit` (typecheck), `node --test extensions/**/*.test.ts` (full suite). Node 24 + pnpm 11, `pnpm install --frozen-lockfile`. |
| `validate` (Validate Package Version) | `cad0p/semver-calver-release/validate-package-version` — feature PRs must **not** bump `package.json` version; release PRs (`release/from-v*`) own it. |
| `validate` (Validate Release PR) | `cad0p/semver-calver-release/validate-release-pr` — release-PR shape only; green/skip on ordinary PRs. |

How to read checks:

```bash
gh pr checks <pr>            # one line per check: pass / fail / pending / skipping
gh pr checks <pr> --watch    # wait until all checks settle
gh run view <run-id> --log   # full log when something is red
```

Anything other than a pass (or a legitimate path-gated skip) is a blocker: fix on the branch and re-verify — a review verdict applies to the commit it reviewed, so a new push needs a fresh pass (§3).

**Not a PR gate:** `Pi Upstream Probe` (`.github/workflows/pi-upstream-probe.yml`) runs on a daily schedule / `workflow_dispatch` only. It daily-probes upstream `@earendil-works/pi-coding-agent` for the two remaining reflection points the extension depends on (`AgentSession.prototype.prompt`, `agent.state.messages`); on failure it files an alarm per `scripts/alarm-body.md`. Run `node scripts/pi-upstream-probe.mjs` locally when touching the reflection bootstrap or bumping the peer floor — a red probe is a real signal about upstream drift, not a PR blocker by itself.

## 3. Independent review protocol

Every change gets an independent verdict from an agent that did **not** author it (orchestrator spawns the reviewer; implementation subagents never review their own work):

1. **Functional/correctness review** — does it do what the issue/plan says; are the tests real and meaningful; boundaries and failure paths.
2. **Adversarial lens** — break it: mutation-test byte-exact string pins (reorder, single-char drift, added/removed period, changed length must fail), probe off-by-one ranges, invalid/absent config layers, `null`/`undefined` API returns, throw paths, tool-inactive gating, and copy drift against the issue's pinned strings.
3. **Pi-api lens** — for code touching pi APIs: signatures typecheck against the locked `@earendil-works/*` devDependency **and** the semantics hold on the host runtime version (the repo is a pi extension; local types can lag the host `pi`). Check `dist` sources, not just `.d.ts`.

Rules:

- Findings must be **reproduced** before reporting: a failing command, a mutated test, a quoted line, or a reverted-fix test.
- Each reviewer records its own findings + verdict (e.g. `functional: FINDINGS @ <sha>`, `adversarial: CLEAN @ <sha>`) and the PR records all verdicts against the same SHA.
- Any commit pushed after a verdict needs a fresh verify pass by every lens whose scope it touched, seeded with that lens's findings.
- Triage every finding: fix-now / defer (with a tracked issue + vault note) / decline (with rationale). Blockers and majors are never deferred.
- Trust but verify: an agent's summary is intent, not outcome — the orchestrator reads the actual diff before reporting work as done.

## 4. Offline gates

Run in the worktree (all must be green before push):

```bash
pnpm install --frozen-lockfile
pnpm test            # node --test extensions/**/*.test.ts (offline; DI stubs, no LLM calls)
pnpm run lint        # biome check extensions/
pnpm run typecheck   # tsc --noEmit
node scripts/pi-upstream-probe.mjs   # extra: reflection-surface probe (not a CI gate)
```

Notes:

- Node ≥22.19 is required (pi 0.81+; native TS type-stripping for `node --test` on `.ts`). CI uses Node 24.
- The suite is fully offline: `summarize` is DI-stubbed and the cache-path tests use a capturing `streamFn`.
- Scope discipline: a feature PR must not edit `CHANGELOG.md` or the `package.json` version (release policy); release PRs own them.

## 5. Live proof per change type

Live proof means running the real flow against the real surface and capturing the authoritative signal — a green CI check is not live proof. Record the evidence (commands, output, session JSONL path, run URL) in the PR.

**Model/provider rule (all rows).** Prefer running the host `pi` without `--model`/`--provider` (it resolves `settings.json` defaults — the model the user actually runs). Two traps: (1) pi does **not** read `PI_MODEL`/`PI_PROVIDER`/`PI_REASONING_LEVEL` for model selection — it only sets them for spawned bash children (`dist/core/tools/bash.js`); (2) `--no-extensions` disables provider extension aliases (e.g. `opencode-go-2` in this fleet), so headless resolution can silently fall back to an arbitrary model. Where the session's provider is an extension alias, **load the alias extension alongside the subject** (`-e <alias-ext> -e <subject-ext>`) so the real account resolves; otherwise pin only the **base** provider id + the session's model slug (`--provider <base> --model <$PI_MODEL>`), never a different provider/model, and **assert from the run's session JSONL (assistant entry `provider`/`model`) that the intended model actually served the run**. Check readiness with `pi auth check --provider <p> --model <m>` (refreshes OAuth credentials — not a quota check; quota surfaces as a runtime 429). Pins rot (quota, rotation, renames).

| Change type | Live proof |
|---|---|
| **Rewind call site / `cache-summary.ts` / `generateBranchSummary` wiring** | README “Live summary verification”, **both halves**: (1) *cache gate* — fresh session, only this extension (`pi --no-extensions -e <worktree>/extensions/navigate-tree/index.ts`), anchor → real work → rewind; assert `cacheRead > 0` / `hit: true` in `details.summaryCache` in the session JSONL (`~/.pi/agent/sessions/--<cwd-slug>--/<file>.jsonl`), or a recorded legitimate `fallbackReason`; (2) *quality smoke* — `node scripts/summary-quality-check.mjs <jsonl>` passes (r5d headings/length/preamble) and the printed summary is scope-clean (branch only, pre-branch background excluded, unresolved work preserved, collapsed work not continued). A miss notice means a mirrored request param diverged — bisect `maxTokens` strip → `reasoning` → `cacheRetention` → session headers. |
| **`promptGuidelines` / `before_agent_start` mandate / tool description / config loader / custom messages (hint, refusal copy)** | Live session (`pi -e <worktree>/extensions/navigate-tree/index.ts`, or interactive TUI for rendering): confirm the exact rendered system prompt (pi-context-view or equivalent prompt dump; guidelines mid-prompt, mandate at the end) and the persisted session-JSONL entries (e.g. `custom_message` with the pinned `customType` + byte-exact copy; anchor/rewind/refusal result lines). `hasUI`-gated surfaces (`ui.notify` warnings, TUI-only copy) are unit-pinned when headless; **config-error warnings are proven live by the TUI warning** in a short tmux session, and any TUI-visible output the change ships gets an interactive check (evidence standard below). |
| **Reflection bootstrap / pi version bump** | `node scripts/pi-upstream-probe.mjs` green against the version in play + a manual anchor → rewind → list session on the host pi (the two reflection points: session capture, post-rewind `agent.state.messages` refresh). |
| **Docs only (README, this file)** | CI green on the head SHA; links/commands in the change actually exist (spot-check anything command-like). No live run required — state so explicitly in the PR. |

**TUI-only output (evidence standard).** Output that renders only in an interactive TUI (`ui.notify` warnings, custom-message rows, transcript lines) cannot be proven by a headless run — a headless run never reaches `hasUI`-gated surfaces, and a unit pin is not live proof. When the changeset ships such output and it is reachable on the current host version, trigger it in a short interactive session (tmux is fine) and capture the pane: `tmux capture-pane -p -J` (normalize trailing whitespace; `-J` joins soft-wrapped lines) showing presence + byte-exact copy against the builder/constant is sufficient. TUI content is text and styling is theme-driven, so a screenshot or attribute capture (`tmux capture-pane -e`) is needed only when layout, color, or truncation is itself the change subject. For deliberately unreachable defensive output — a path that cannot fire on the current host by construction — the unit pin plus an explicit unreachability statement in the PR is the evidence.

Live-run hygiene: back up any global extension config you must modify (`~/.pi/agent/tree-navigator.json`, etc.) and restore/remove it afterwards; never leave test thresholds or credentials behind.

## 6. Closeout

- [ ] CI green on the exact PR head SHA (all checks listed in §2).
- [ ] Offline gates green locally on the head commit (§4).
- [ ] Independent verdict(s) recorded on the same SHA (§3); fix commits re-verified.
- [ ] Live proof for the change type captured with real output (§5) — or an explicit statement why it does not apply.
- [ ] Scope check: `git diff --name-only <base>...HEAD` matches the plan; no `CHANGELOG.md` / `package.json` version edits; no stray artifacts.
- [ ] PR body carries summary, verification evidence, release-note callouts (e.g. prompt-cache invalidation, behavior changes), and `Closes #N` lines.
- [ ] PR left **draft**; kanban + vault notes updated; worktree/branch kept until the owner merges, then cleaned up.
