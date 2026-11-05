#!/usr/bin/env bash
# Check that an engine binary has the network kernels for the instruction set
# its name promises, so a mislabelled build fails CI instead of shipping. In
# v6.3.0 the Windows modern build had no AVX2 kernels, both compatibility
# builds needed AVX2, and the macOS build had no NEON dot-product kernels.
#
#   check-isa.sh BINARY avx2|baseline|dotprod
#
# The markers come only from the network kernels: vpmaddubsw on ymm registers
# from the AVX2 ones (present in avx2 builds, absent from baseline builds) and
# sdot from the NEON ones. Counting all ymm use would not work on Windows,
# where the C++ runtime links in its own CPU-dispatched AVX2 string routines.
set -eu
bin=$1
want=$2
case $want in
  avx2|baseline) marker='vpmaddubsw.*ymm'; what='AVX2 network instructions (vpmaddubsw on ymm)' ;;
  dotprod) marker='sdot'; what='NEON dot-product instructions (sdot)' ;;
  *) echo "unknown level: $want" >&2; exit 2 ;;
esac
dis=$(command -v llvm-objdump || command -v objdump)
n=$("$dis" -d --no-show-raw-insn "$bin" | grep -c "$marker" || true)
echo "$bin: $n $what; expected: $want"
case $want in
  avx2|dotprod) test "$n" -gt 0 ;;
  baseline) test "$n" -eq 0 ;;
esac
