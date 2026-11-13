#!/usr/bin/env bash
# End-to-end rewind repro run: the model drives the rewind, writes the summary
# and continues in the same loop. No user turn is injected after the collapse —
# see README.md, "The rule that makes or breaks the measurement".
#
# usage: run_e2e.sh --fixture F --extension DIR --anchor NAME --label L
#                   [--provider P] [--model M] [--thinking LEVEL] [--root DIR]
#                   [--provider-extension FILE]   (the extension that registers
#                   the provider alias, when the provider is not built in)
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"

fixture="" extension="" anchor="" label=""
provider="opencode-go-11" model="deepseek-v4.1-flash" thinking="max"
root="${TMPDIR:-/tmp}/rewind-repro"
provider_extension=""

while [ $# -gt 0 ]; do
  case "$1" in
    --fixture) fixture="$2"; shift 2 ;;
    --extension) extension="$2"; shift 2 ;;
    --anchor) anchor="$2"; shift 2 ;;
    --label) label="$2"; shift 2 ;;
    --provider) provider="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --thinking) thinking="$2"; shift 2 ;;
    --root) root="$2"; shift 2 ;;
    --provider-extension) provider_extension="$2"; shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

[ -n "$fixture" ] && [ -n "$extension" ] && [ -n "$anchor" ] && [ -n "$label" ] || {
  echo "usage: run_e2e.sh --fixture F --extension DIR --anchor NAME --label L" >&2
  exit 2
}

mkdir -p "$root/work" "$root/logs"
fixture="$(cd "$(dirname "$fixture")" && pwd)/$(basename "$fixture")"
extension="$(cd "$extension" && pwd)"
session="$root/session-$label.jsonl"
cp "$fixture" "$session"
log="$root/logs/$label.log"
: > "$log"

echo "session:  $session"
echo "log:      $log"
echo "model:    $provider/$model ($thinking)"
echo "anchor:   $anchor -> $label"

cd "$root/work"
PI_REPRO_LOG="$log" pi --no-extensions \
  -e "$here/recorder.ts" \
  -e "$extension/index.ts" \
  ${provider_extension:+-e "$provider_extension"} \
  --models "$provider/$model" \
  --session "$session" --provider "$provider" --model "$model" --thinking "$thinking" \
  -p "Rewind to the anchor '$anchor' as '$label', summaryFocus='the open work and what unblocks it'. The rewind must be the only tool call in its message."

echo
echo "score it: python3 $here/dump_tail.py $session"
