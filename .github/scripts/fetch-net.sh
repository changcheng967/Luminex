#!/usr/bin/env bash
# Download the network the release binaries carry and check its sha256, so
# every build embeds exactly the shipped file. Prints the file's path and
# which network it holds: the file is always saved under the engine's default
# name, so the path alone does not say.
#
#   fetch-net.sh [DIR]   (default: net)
#
# The pin is gen768s_01q (luminex_gen768s_01q_i8.nnue on the engine-archives
# pre-release). Shipping a new network means attaching it to a release and
# moving url, sha256 and about here. The url's file name is the network's real
# name and its only source: the file is saved under it, CMake makes it the
# engine's default network (DEFAULT_NET) and the name the built-in network
# reports, and releases attach the file under it (check-net.sh fails a build
# that names another network). about, the network's name and a line on how it
# was made, goes into the development build's notes (dev-release.sh).
set -euo pipefail
url=https://github.com/changcheng967/Luminex/releases/download/engine-archives/luminex_gen768s_01q_i8.nnue
sha256=6665f301cb3d299a94ac773bc148d1b4962d76a9529dbf67310015f83fef813c
about="gen768s_01q: gen768r_06 trained on for 450K steps with a fresh optimizer (lr 5e-5, loss power 3.0) on Leela BT4's relabel of t91 (November 2025, January-August and October 2026, 6.1B positions), then 30K steps through the engine's int8 rounding (QAT)"
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
