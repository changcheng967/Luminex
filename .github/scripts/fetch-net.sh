#!/usr/bin/env bash
# Download the network the release binaries carry and check its sha256, so
# every build embeds exactly the shipped file. Prints the file's path and
# which network it holds: the file is always saved under the engine's default
# name, so the path alone does not say.
#
#   fetch-net.sh [DIR]   (default: net)
#
# The pin is gen768s_03q (luminex_gen768s_03q_i8.nnue on the engine-archives
# pre-release). Shipping a new network means attaching it to a release and
# moving url, sha256 and about here. The url's file name is the network's real
# name and its only source: the file is saved under it, CMake makes it the
# engine's default network (DEFAULT_NET) and the name the built-in network
# reports, and releases attach the file under it (ci.sh net fails a build
# that names another network). about, the network's name and a line on how it
# was made, goes into the development build's notes (ci.sh dev-release).
set -euo pipefail
url=https://github.com/changcheng967/Luminex/releases/download/engine-archives/luminex_gen768s_03q_i8.nnue
sha256=001d5bcd3ff0cd8f566450a809c065d7ecdf0f751227a1f844bfd8c61e2d6a4b
about="gen768s_03q: gen768s_01 trained on for two more 450K-step runs with fresh optimizers (lr 5e-5, loss power 3.0) on Leela BT4's relabel of t91 (gen768s_02: December 2025 and seven 2026 months; gen768s_03: November 2025 and the 2026 months), then 30K steps of QAT with one int8 scale per tail row (LNR8)"
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
