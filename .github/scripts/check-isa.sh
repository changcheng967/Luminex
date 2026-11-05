#!/usr/bin/env bash
# Check that an x86 engine binary has the network code for the instruction set
# its name promises, so a modern build without AVX2 or a compatibility build
# that needs AVX2 fails CI instead of shipping (both happened in v6.3.0).
#
#   check-isa.sh BINARY avx2|baseline
#
# vpmaddubsw on ymm registers only comes from the AVX2 network kernels, so it
# is a precise marker: present in AVX2 builds, absent from baseline builds.
# Counting all ymm use would not work on Windows, where the C++ runtime links
# in its own CPU-dispatched AVX2 string routines.
set -eu
bin=$1
want=$2
dis=$(command -v llvm-objdump || command -v objdump)
n=$("$dis" -d --no-show-raw-insn "$bin" | grep -c 'vpmaddubsw.*ymm' || true)
echo "$bin: $n AVX2 network instructions (vpmaddubsw on ymm); expected: $want"
case $want in
  avx2) test "$n" -gt 0 ;;
  baseline) test "$n" -eq 0 ;;
  *) echo "unknown level: $want" >&2; exit 2 ;;
esac
