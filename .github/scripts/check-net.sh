#!/usr/bin/env bash
# Run an engine binary by itself, copied into an empty directory, and check
# that it evaluates with its built-in network, and names that network by its
# real name. v6.3.0 to v6.5.0 shipped the network only as a separate file, so
# a tester who ran the binary alone got the hand-crafted evaluation; v6.6.0's
# binaries called gen768r_06q by the default file name of v6.3.0's network.
#
#   check-net.sh BINARY
#
# The real name is the file name of the asset fetch-net.sh pins (its url=
# line). The binary must report its built-in network under that name and
# offer it as NNUEFile's default, or the check fails.
set -euo pipefail
bin=$1
want=$(sed -n 's#^url=.*/##p' "$(dirname "$0")/fetch-net.sh")
[ -n "$want" ] || { echo "no url= line in fetch-net.sh" >&2; exit 1; }
dir=$(mktemp -d)
cp "$bin" "$dir/"
out=$(cd "$dir" && printf 'uci\nisready\nposition startpos\neval\nquit\n' | "./$(basename "$bin")" 2>&1 | tr -d '\r')
# Windows can keep the copied binary locked for a moment after it exits, so
# the cleanup retries and never fails the check.
for _ in 1 2 3 4 5; do rm -rf "$dir" 2> /dev/null && break; sleep 1; done
printf '%s\n' "$out" | grep -E '^(id name|info string|eval|option name NNUEFile)' || true
if ! printf '%s\n' "$out" | grep -q 'NNUE evaluation using built-in'; then
  echo "$bin: the built-in network did not load" >&2
  exit 1
fi
if ! printf '%s\n' "$out" | grep -qF "NNUE evaluation using built-in $want ("; then
  echo "$bin: the built-in network does not report itself as $want, the pinned network" >&2
  exit 1
fi
if ! printf '%s\n' "$out" | grep -qxF "option name NNUEFile type string default $want"; then
  echo "$bin: NNUEFile's default is not $want, the pinned network" >&2
  exit 1
fi
if ! printf '%s\n' "$out" | grep -q '^eval cp'; then
  echo "$bin: no eval" >&2
  exit 1
fi
