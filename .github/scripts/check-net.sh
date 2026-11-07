#!/usr/bin/env bash
# Run an engine binary by itself, copied into an empty directory, and check
# that it evaluates with its built-in network. v6.3.0 to v6.5.0 shipped the
# network only as a separate file, so a tester who ran the binary alone got
# the hand-crafted evaluation.
#
#   check-net.sh BINARY
set -euo pipefail
bin=$1
dir=$(mktemp -d)
cp "$bin" "$dir/"
out=$(cd "$dir" && printf 'uci\nisready\nposition startpos\neval\nquit\n' | "./$(basename "$bin")" 2>&1)
rm -rf "$dir"
printf '%s\n' "$out" | grep -E '^(id name|info string|eval)' || true
if ! printf '%s\n' "$out" | grep -q 'NNUE evaluation using built-in'; then
  echo "$bin: the built-in network did not load" >&2
  exit 1
fi
if ! printf '%s\n' "$out" | grep -q '^eval cp'; then
  echo "$bin: no eval" >&2
  exit 1
fi
