#!/usr/bin/env bash
# Download the network the release binaries carry and check its sha256, so
# every build embeds exactly the shipped file. Prints the file's path and
# which network it holds: the file is always saved under the engine's default
# name, so the path alone does not say.
#
#   fetch-net.sh [DIR]   (default: net)
#
# The pin is gen768t_01q (luminex_gen768t_01q_i8.nnue on the engine-archives
# pre-release). Shipping a new network means attaching it to a release and
# moving url, sha256 and about here. The url's file name is the network's real
# name and its only source: the file is saved under it, CMake makes it the
# engine's default network (DEFAULT_NET) and the name the built-in network
# reports, and releases attach the file under it (ci.sh net fails a build
# that names another network). about, the network's name and a line on how it
# was made, goes into the development build's notes (ci.sh dev-release).
set -euo pipefail
url=https://github.com/changcheng967/Luminex/releases/download/engine-archives/luminex_gen768t_01q_i8.nnue
sha256=87e4a40fdc2e47dca427053a59014fdfa345501ec8f75c4a370eedc48df1721a
about="gen768t_01q: V5 (gen768t, docs/nnue/V5-GEN768T.md), gen768s's feature layer with 55,960 int8 relation rows (attacks, and pawn pairs in contact) summed into 256 of its 768 lanes (the LNT8 format); from gen768s_03's float network, 450K steps on Leela BT4's relabel of every t91 month (January's tail held out) and the BT4 corpus, then 30K steps of QAT with int8 relation rows and one int8 scale per tail row"
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
