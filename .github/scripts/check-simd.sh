#!/usr/bin/env bash
# Check that every SIMD kernel level of an engine binary evaluates alike. The
# binary's nnuecheck (luminex nnuecheck [NETFILE]) plays a fixed set of games
# under every level the CPU supports, fails on any difference, and prints a
# digest of the evaluations, which every x86-64 build of the same network
# must share (build.yml compares the release builds' with OpenBench's).
#
#   check-simd.sh BINARY [--net FILE] [CHIP "LEVELS"]
#
# Without CHIP it runs on this CPU. With CHIP it runs under Intel SDE (the
# launcher's path in $SDE, from fetch-sde.sh) as the CPU of that SDE chip
# option (spr, icx, skx, arl, hsw, snb, nhm, ...): LEVELS must be exactly the
# levels the binary finds there, best first, and SDE's chip check stops the
# binary if it executes an instruction that CPU lacks. $DIGEST, if set, is the
# digest the run must print (another run's). With $GITHUB_OUTPUT set, the
# digest is also written there as digest=...
set -euo pipefail
bin=$1
shift
net=
if [ "${1:-}" = --net ]; then net=$2; shift 2; fi
chip=${1:-}
want=${2:-}
if [ -n "$chip" ]; then
  : "${SDE:?SDE must name the sde launcher (fetch-sde.sh)}"
  run=("$SDE" "-$chip" -chip-check-exe-only -- "$bin" nnuecheck)
  where="on $chip (SDE)"
else
  run=("$bin" nnuecheck)
  where="on this CPU"
fi
[ -z "$net" ] || run+=("$net")
status=0
out=$("${run[@]}" 2>&1 | tr -d '\r') || status=$?
printf '%s\n' "$out" | grep -vE '^(id |option )' || true
if [ "$status" != 0 ] || ! printf '%s\n' "$out" | grep -qx 'nnuecheck ok'; then
  echo "$bin: nnuecheck failed $where (exit status $status)" >&2
  exit 1
fi
levels=$(printf '%s\n' "$out" | sed -n 's/^nnuecheck: levels \(.*\) (in use: .*)$/\1/p')
if [ -n "$want" ] && [ "$levels" != "$want" ]; then
  echo "$bin $where: levels '$levels', expected '$want'" >&2
  exit 1
fi
digest=$(printf '%s\n' "$out" | sed -n 's/^nnuecheck: .* digest \([0-9a-f]*\)$/\1/p')
if [ -z "$digest" ]; then
  echo "$bin $where: nnuecheck printed no digest" >&2
  exit 1
fi
if [ -n "${DIGEST:-}" ] && [ "$digest" != "$DIGEST" ]; then
  echo "$bin $where: digest $digest, expected $DIGEST" >&2
  exit 1
fi
echo "$bin $where: levels $levels agree, digest $digest"
[ -z "${GITHUB_OUTPUT:-}" ] || echo "digest=$digest" >> "$GITHUB_OUTPUT"
