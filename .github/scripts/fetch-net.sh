#!/usr/bin/env bash
# Download the network the release binaries carry and check its sha256, so
# every build embeds exactly the shipped file. Prints the file's path and
# which network it holds: the file is always saved under the engine's default
# name, so the path alone does not say.
#
#   fetch-net.sh [DIR]   (default: net)
#
# The pin is gen768r_06q (luminex_gen768r_06q_i8.nnue on the engine-archives
# pre-release). Shipping a new network means attaching it to a release and
# moving url, sha256 and about here. The url's file name is the network's real
# name and its only source: the file is saved under it, CMake makes it the
# engine's default network (DEFAULT_NET) and the name the built-in network
# reports, and releases attach the file under it (check-net.sh fails a build
# that names another network). about, the network's name and a line on how it
# was made, goes into the development build's notes (dev-release.sh).
set -euo pipefail
url=https://github.com/changcheng967/Luminex/releases/download/engine-archives/luminex_gen768r_06q_i8.nnue
sha256=e3397b0ff38743be8a632178ea1bd6b62994ebb0e581953ccc2d6a8a402d6243
about="gen768r_06q: gen768r_05 trained on to 1.95M steps (from 1.50M on t91 December 2025, January 2026 and June-August and October 2026, 6.0B positions, at loss power 3.0), then 30K steps through the engine's int8 rounding (QAT); the last gen768r network"
name=${url##*/}
dir=${1:-net}
out=$dir/$name
mkdir -p "$dir"
[ -f "$out" ] || curl -fsSL --retry 3 -o "$out" "$url"
if command -v sha256sum > /dev/null; then
  got=$(sha256sum "$out" | cut -d' ' -f1)
else
  got=$(shasum -a 256 "$out" | cut -d' ' -f1)
fi
if [ "$got" != "$sha256" ]; then
  echo "$out: sha256 $got, expected $sha256" >&2
  rm -f "$out"
  exit 1
fi
echo "$out (OpenBench id $(printf '%s' "${sha256:0:8}" | tr a-f A-F)), sha256 checked"
