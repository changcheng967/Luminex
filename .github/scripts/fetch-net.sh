#!/usr/bin/env bash
# Download the network the release binaries carry and check its sha256, so
# every build embeds exactly the shipped file. Prints the file's path.
#
#   fetch-net.sh [DIR]   (default: net)
#
# The pin is the v6.5.0 release asset. Shipping a new network means attaching
# it to a release and moving url and sha256 here; the name must stay
# DEFAULT_NET (src/nnue.h).
set -euo pipefail
name=luminex_sfq_i8.nnue
url=https://github.com/changcheng967/Luminex/releases/download/v6.5.0/$name
sha256=1f42c03d417239782b6c1f7a593083417a4d588de3cc4c7375649ae4f22c3626
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
