#!/usr/bin/env python3
"""Print the post-rewind tail of a session file: the stored summary, the
continuation entry and the next assistant turn — the evidence you score.

  python3 dump_tail.py <session.jsonl> [--turns 1]
"""

import argparse
import json


def flat(text, limit):
    return " ".join(str(text).split())[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument("--width", type=int, default=2000)
    args = parser.parse_args()

    with open(args.session, errors="ignore") as fh:
        entries = [json.loads(line) for line in fh if line.strip()]

    last = None
    for i, entry in enumerate(entries):
        if entry.get("type") == "custom_message" and entry.get("customType") == "navigate-tree-rewind":
            last = i
    if last is None:
        raise SystemExit("no landed rewind in this session")

    details = entries[last].get("details") or {}
    print(f"rewind: {details.get('rewindTo')} -> {details.get('newLabel')} "
          f"({details.get('contextBefore')} -> {details.get('contextAfter')} tokens)")
    print(f"continuation entry: {flat(entries[last].get('content'), 400)}")

    seen = 0
    for entry in entries[last:]:
        if entry.get("type") == "message" and (entry.get("message") or {}).get("role") == "assistant":
            message = entry["message"]
            print(
                f"assistant turn {seen + 1}: {message.get('provider')}/{message.get('model')} "
                f"{message.get('stopReason')}"
            )
            for part in message.get("content") or []:
                kind = part.get("type")
                if kind == "thinking":
                    print("  thinking:", flat(part.get("thinking"), args.width))
                elif kind == "text":
                    print("  text:", flat(part.get("text"), args.width))
                elif kind == "tool_use":
                    print("  tool:", part.get("name"), flat(json.dumps(part.get("input")), 300))
            seen += 1
            if seen >= args.turns:
                break


if __name__ == "__main__":
    main()
