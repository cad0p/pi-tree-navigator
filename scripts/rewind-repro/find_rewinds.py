#!/usr/bin/env python3
"""Find landed rewinds in the pi session store and show what happened next.

Prints one record per landed rewind: session file, line, anchor, context
before/after, whether the collapsed summary was produced (status), and the
first assistant turn after the landing — which is what you score.

  python3 find_rewinds.py [--min-size 3000000] [--max-after 200000]
"""

import argparse
import json
import os


def first_assistant_turn(lines, start, scan=8):
    for line in lines[start : start + scan]:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("type") != "message":
            continue
        message = entry.get("message") or {}
        if message.get("role") != "assistant":
            continue
        for part in message.get("content") or []:
            if isinstance(part, dict) and part.get("type") in ("thinking", "text"):
                text = part.get("thinking") or part.get("text") or ""
                return " ".join(text.split())[:240]
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.path.expanduser("~/.pi/agent/sessions"))
    parser.add_argument("--min-size", type=int, default=3_000_000)
    parser.add_argument("--max-after", type=int, default=200_000)
    args = parser.parse_args()

    for directory in sorted(os.listdir(args.root)):
        path = os.path.join(args.root, directory)
        if not os.path.isdir(path):
            continue
        for name in sorted(os.listdir(path)):
            if not name.endswith(".jsonl"):
                continue
            session = os.path.join(path, name)
            try:
                if os.path.getsize(session) < args.min_size:
                    continue
                with open(session, errors="ignore") as fh:
                    lines = fh.read().splitlines()
            except OSError:
                continue
            for index, line in enumerate(lines):
                if "navigate-tree-rewind" not in line:
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if entry.get("type") != "custom_message":
                    continue
                if entry.get("customType") != "navigate-tree-rewind":
                    continue
                details = entry.get("details") or {}
                if details.get("status") != "done":
                    continue
                after = details.get("contextAfter") or 0
                if after > args.max_after:
                    continue
                print(
                    json.dumps(
                        {
                            "session": session,
                            "line": index + 1,
                            "anchor": details.get("rewindTo"),
                            "label": details.get("newLabel"),
                            "before": details.get("contextBefore"),
                            "after": after,
                            "collapsed": details.get("collapsedEntries"),
                            "next": first_assistant_turn(lines, index + 1),
                        }
                    )
                )


if __name__ == "__main__":
    main()
