#!/usr/bin/env bash
# Download the network the release binaries carry and check its sha256, so
# every build embeds exactly the shipped file. Prints the file's path.
#
#   fetch-net.sh [DIR]   (default: net)
#
# The pin is gen768r_03 (luminex_gen768r_03_i8.nnue on the engine-archives
# pre-release). Shipping a new network means attaching it to a release and
# moving url and sha256 here; whatever the asset is called, the file is saved
# as DEFAULT_NET (src/nnue.h), the name the engine loads.
set -euo pipefail
name=luminex_sfq_i8.nnue
url=https://github.com/changcheng967/Luminex/releases/download/engine-archives/luminex_gen768r_03_i8.nnue
sha256=5bbd5cc136463ef7352e4a3f380639c4ef33ad1175782a72f364537f0458dcde
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
echo "$out"
