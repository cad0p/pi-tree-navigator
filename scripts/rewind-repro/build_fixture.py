#!/usr/bin/env python3
"""Build a rewind-repro fixture from a real pi session file.

Two subcommands:

  cut      cut a session at a line, writing a fixture whose leaf is that entry
  rewrite  rewrite the last rewind's continuation prose and/or the last branch
           summary's preamble / trailing section, in place

Cutting *before* the rewind attempt gives the end-to-end protocol (the model
drives the rewind itself). Cutting *at* the rewind landing gives the
continuation-only protocol, which is diagnostic only: a headless start needs a
user turn, and a user turn settles the question the harness is measuring.
"""

import argparse
import json
import sys


def read_entries(path):
    with open(path, errors="ignore") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_entries(path, entries, cwd=None):
    if cwd:
        for entry in entries:
            if entry.get("type") == "session":
                entry["cwd"] = cwd
    with open(path, "w") as fh:
        fh.write("\n".join(json.dumps(e) for e in entries) + "\n")


def cmd_cut(args):
    entries = read_entries(args.session)[: args.line]
    write_entries(args.out, entries, args.cwd)
    last = entries[-1] if entries else {}
    print(
        json.dumps(
            {
                "out": args.out,
                "entries": len(entries),
                "leafType": last.get("type"),
                "leafRole": (last.get("message") or {}).get("role"),
            }
        )
    )


def last_index(entries, predicate):
    for i in range(len(entries) - 1, -1, -1):
        if predicate(entries[i]):
            return i
    return None


def cmd_rewrite(args):
    entries = read_entries(args.session)

    ridx = last_index(
        entries,
        lambda e: e.get("type") == "custom_message"
        and e.get("customType") == "navigate-tree-rewind",
    )
    if ridx is None:
        sys.exit("no navigate-tree-rewind custom message in this fixture")

    header = (entries[ridx].get("content") or "").split("\n\n")[0]
    prose = ""
    if args.copy:
        prose = open(args.copy).read().strip()
    entries[ridx]["content"] = header + ("\n\n" + prose if prose else "")

    sidx = last_index(entries, lambda e: e.get("type") == "branch_summary")
    if sidx is not None:
        summary = entries[sidx].get("summary") or ""
        if args.preamble:
            marker = "Summary of that exploration:"
            preamble = open(args.preamble).read().rstrip("\n")
            summary = preamble + (
                summary.split(marker, 1)[1] if marker in summary else summary
            )
        if args.section:
            summary = summary.rstrip("\n") + "\n\n" + open(args.section).read().strip() + "\n"
        entries[sidx]["summary"] = summary

    write_entries(args.session, entries, args.cwd)
    print(json.dumps({"fixture": args.session, "rewindIndex": ridx, "summaryIndex": sidx}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    cut = sub.add_parser("cut", help="cut a session at a line")
    cut.add_argument("--session", required=True)
    cut.add_argument("--line", type=int, required=True, help="1-based, inclusive")
    cut.add_argument("--out", required=True)
    cut.add_argument("--cwd", default=None, help="rewrite session.cwd (scratch workdir)")
    cut.set_defaults(func=cmd_cut)

    rewrite = sub.add_parser("rewrite", help="rewrite the rewind copy / summary framing")
    rewrite.add_argument("--session", required=True, help="fixture, rewritten in place")
    rewrite.add_argument("--copy", default=None, help="continuation prose file (empty = header only)")
    rewrite.add_argument("--preamble", default=None, help="branch-summary preamble file")
    rewrite.add_argument("--section", default=None, help="text appended to the summary")
    rewrite.add_argument("--cwd", default=None)
    rewrite.set_defaults(func=cmd_rewrite)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
