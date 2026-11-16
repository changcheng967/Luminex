#!/usr/bin/env bash
# Check that an engine binary has the network kernels for the instruction set
# its name promises, so a mislabelled build fails CI instead of shipping. In
# v6.3.0 the Windows modern build had no AVX2 kernels, both compatibility
# builds needed AVX2, and the macOS build had no NEON dot-product kernels.
#
#   check-isa.sh BINARY universal|avx2|baseline|dotprod
#
# universal is the x86-64 release build (CMake LUMINEX_DISPATCH), which must
# carry every level's kernels; avx2 and baseline are single-level x86 builds,
# dotprod the Apple silicon one. The markers come only from the network
# kernels, each level's own instructions:
#   avx512vnni  vpdpbusd on zmm registers (the dense layers)
#   avx512      vpaddsw on zmm (the accumulator)
#   avxvnni     vpdpbusd in its VEX form, which objdump prints as {vex}
#   avx2        vpmaddubsw on ymm (the dense layers)
#   base        paddsw without the VEX v prefix (the SSE2 accumulator)
#   dotprod     sdot (NEON)
# Counting all ymm use would not work on Windows, where the C++ runtime links
# in its own CPU-dispatched AVX2 string routines.
set -eu
bin=$1
want=$2
dis=$(command -v llvm-objdump || command -v objdump)
asm=$("$dis" -d --no-show-raw-insn "$bin")
count() { printf '%s\n' "$asm" | grep -cE "$1" || true; }
check() {   # what, marker, expected: present or absent
  n=$(count "$2")
  echo "$bin: $n $1 ($2); expected $3"
  if [ "$3" = present ]; then test "$n" -gt 0; else test "$n" -eq 0; fi
}
case $want in
  universal)
    check 'AVX512-VNNI kernel instructions' 'vpdpbusd.*zmm' present
    check 'AVX-512 kernel instructions' 'vpaddsw.*zmm' present
    check 'AVX-VNNI kernel instructions' '\{vex\}[[:space:]]*vpdpbusd' present
    check 'AVX2 kernel instructions' 'vpmaddubsw.*ymm' present
    check 'SSE2 kernel instructions' '[[:space:]]paddsw[[:space:]]' present ;;
  avx2) check 'AVX2 network instructions' 'vpmaddubsw.*ymm' present ;;
  baseline) check 'AVX2 network instructions' 'vpmaddubsw.*ymm' absent ;;
  dotprod) check 'NEON dot-product instructions' 'sdot' present ;;
  *) echo "unknown level: $want" >&2; exit 2 ;;
esac
