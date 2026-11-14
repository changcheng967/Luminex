#!/usr/bin/env bash
# Download Intel's Software Development Emulator, which runs a binary as on
# another Intel CPU (its cpuid, and with "chip check" only that CPU's
# instructions), so CI can check the universal build's AVX-512 and AVX-VNNI
# kernels on GitHub's runners, which have neither. The version and the
# sha256 of each package are Intel's, from its download page
# (https://www.intel.com/content/www/us/en/download/684897/intel-software-development-emulator.html).
# Prints the path of the sde launcher.
#
#   fetch-sde.sh [DIR]   (default: $RUNNER_TEMP/sde, or ./sde)
set -euo pipefail
version=10.13.1-2026-07-28
mirror=https://downloadmirror.intel.com/924984
case "$(uname -s)" in
  Linux*)
    os=lin; exe=sde64
    sha256=94e97d623fec54385686e1e7ba65ebc9941748c05ee451423948334892bf2b50 ;;
  MINGW*|MSYS*|CYGWIN*)
    os=win; exe=sde.exe
    sha256=74e626ede09b0baa5011fc9e51b58627ea92c3fc0bae5fd7db34b490f335f651 ;;
  *) echo "no SDE package for $(uname -s)" >&2; exit 1 ;;
esac
dir=${1:-${RUNNER_TEMP:-.}/sde}
# Windows runners give RUNNER_TEMP as a Windows path; Git Bash takes POSIX ones.
if command -v cygpath > /dev/null; then dir=$(cygpath -u "$dir"); fi
pkg=sde-external-$version-$os
mkdir -p "$dir"
if [ ! -f "$dir/$pkg/$exe" ]; then
  curl -fsSL --retry 3 -o "$dir/$pkg.tar.xz" "$mirror/$pkg.tar.xz"
  got=$(sha256sum "$dir/$pkg.tar.xz" | cut -d' ' -f1)
  if [ "$got" != "$sha256" ]; then
    echo "$mirror/$pkg.tar.xz: sha256 $got, expected $sha256" >&2
    rm -f "$dir/$pkg.tar.xz"
    exit 1
  fi
  tar -xf "$dir/$pkg.tar.xz" -C "$dir"
  rm -f "$dir/$pkg.tar.xz"
fi
echo "$dir/$pkg/$exe"
