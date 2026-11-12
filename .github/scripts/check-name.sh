#!/usr/bin/env bash
# Print an engine binary's UCI name and, given a development build id, check
# that the binary reports it ("id name Luminex dev-20261112-d4774f1a"), as the
# luminex-dev pre-release binaries do.
#
#   check-name.sh BINARY [BUILD_ID]
set -euo pipefail
bin=$1
want=${2:-}
out=$(printf 'uci\nquit\n' | "$bin" 2>&1)
out=${out//$'\r'/}
name=$(grep -m1 '^id name' <<< "$out" || true)
echo "$name"
if [ -n "$want" ] && [ "$name" != "id name Luminex $want" ]; then
  echo "$bin: expected 'id name Luminex $want'" >&2
  exit 1
fi
