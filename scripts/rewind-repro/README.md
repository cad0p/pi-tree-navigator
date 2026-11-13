# rewind-repro — measuring what a rewind's continuation actually does

A scratch harness for one question: **after a rewind, does the model resume the branch summary's
`## Next Steps`, or does it act on a retained user message that reads like a live request?**
It exists because the answer decides whether guidance in the rewind copy is worth anything.

Everything here runs against a **real session file** cut around a real rewind, so the context the model
sees is the context it saw then.

## The rule that makes or breaks the measurement

**A user turn injected after the collapse invalidates a pass.**

`pi -p "Continue." --session fixture.jsonl` appends a user message after the summary. That settles
"which user message is live?" in the model's favour: it now has a benign request that is newer than the
retained one, so it reads the summary and resumes. Measured that way, a summary change looked like a
5/6 fix. Measured the real way — a run where the model drives the rewind and the extension continues it —
the same change scored 0/7 on the fixture it had "fixed" 3/3.

So there are two protocols, and only the first one can show that something works:

| protocol | how | cost |
|---|---|---|
| **end-to-end (default)** | cut the session *before* the rewind, then let the model drive it: `-p "Rewind to the anchor '<name>' as '<label>', summaryFocus='…'"`. The model calls the tool, writes the summary as its next turn, the extension branches and continues in the same loop. | one full pre-rewind context per data point (~365k tokens for each fixture used so far) |
| continuation-only (diagnostics only) | cut *at* the rewind landing and resume with `-p "Continue."` | cheap (post-rewind context only), but every pass is suspect |

If you use continuation-only, treat every pass as unproven; a *failure* there is still informative (the
shipped copy fails it even with the benign turn present). One attempt to start the run without a user
turn — `pi.sendMessage(…, { triggerTurn: true })` from `session_start` in a headless run with no prompt —
exited without running a turn and was not pursued.

The end-to-end protocol has its own caveat: the rewind is requested by a `-p` user message. That message
sits after the anchor, so the rewind collapses it and it is not in the continuation's context — but in the
real sessions the model rewound on its own (context-pressure hint), not because it was asked.

**Fixtures are real session files.** They hold private conversation content: keep them in a scratch
directory, never in the repo.

## Files

| file | what |
|---|---|
| `find_rewinds.py` | scan `~/.pi/agent/sessions` for landed rewinds; prints session, file line, `rewindTo`, `contextBefore/After`, and the first assistant turn after it |
| `build_fixture.py` | `cut`: cut a session at a chosen line. `rewrite` (continuation-only diagnostics): replace the continuation prose, the branch-summary preamble, or append a summary section |
| `dump_tail.py` | print the last landed rewind of a session: its numbers, the continuation entry, and the next assistant turn(s) with provider/model — the thing you score. Check the label it prints: if the run died before rewinding, the "last rewind" is an old one from the fixture |
| `recorder.ts` | extension: logs every `tool_call` to `$PI_REPRO_LOG` and blocks everything but reads and `navigate_tree`, so a replay of a real session cannot touch the machine |
| `run_e2e.sh` | the end-to-end runner: pinned provider, recorder, extension under test, rewind prompt |

## Procedure

```bash
# 1. find a candidate: a rewind whose collapsed segment answered a retained ask,
#    and whose next assistant turn drifted back to it
python3 scripts/rewind-repro/find_rewinds.py | less

# 2. cut before the rewind attempt (the line before the model's rewind call)
python3 scripts/rewind-repro/build_fixture.py cut \
  --session ~/.pi/agent/sessions/<dir>/<file>.jsonl \
  --line 11125 --out /tmp/repro/pre-b.jsonl

# 3. run the flow with the extension under test
scripts/rewind-repro/run_e2e.sh \
  --fixture /tmp/repro/pre-b.jsonl \
  --extension extensions/navigate-tree \
  --anchor workstream-pkg-heartbeat-done --label e2e-b --root /tmp/repro \
  --provider opencode-go-11 --model deepseek-v4.1-flash --thinking max \
  --provider-extension ~/.pi/agent/npm/node_modules/@cad0p/pi-fallback-provider/index.ts

# 4. score it
python3 scripts/rewind-repro/dump_tail.py /tmp/repro/session-e2e-b.jsonl
```

Score the **first assistant turn after the continuation**: does it resume the summary's `## Next Steps`,
or does it answer/re-run something from before the collapse? Record the provider/model of that turn — a
fallback extension can cycle to another model mid-run and invalidate the data point (pin `--models`).

## Cost

Each end-to-end run sends the whole pre-rewind context (the two fixtures used so far were ~365k tokens
each) and takes 1–2 minutes. That is real spend: the first investigation exhausted one provider's weekly
quota and another alias's 5-hour quota. Decide the matrix before running it.

## What it has shown so far

End-to-end, DeepSeek V4.1 Flash (max), two fixtures (2026-11-13):

| extension under test | fixture A | fixture B |
|---|---|---|
| shipped (0.3.0-20261113.0) | 1 / 2 resumed | 0 / 3 resumed |
| + a `## Requests Above This Point` summary section (six wordings) | 1 / 1 | 0 / 7 |

- The section gets written correctly every time — the model records "answered earlier, here is where" for
  the retained ask — and on fixture B the next turn answers it again anyway. One run's reasoning, after
  quoting that disposition: "But the user is asking now — this is the current question."
- On fixture A the baseline run that resumed had a summary that happened to record "Answered the original
  demo question (demo already shipped)" under Done; the run that drifted reasoned "the branch summary is
  about a different workstream".
- Small numbers. They rule out "the copy fixed it"; they do not establish a fix.
