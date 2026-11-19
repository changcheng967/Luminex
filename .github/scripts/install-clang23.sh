#!/usr/bin/env bash
# clang 23 and lld 23 for the Linux builds, from apt.llvm.org (LLVM's own
# packages), with LLVM's libc++ 23: Ubuntu 22.04's libstdc++ (12) has no
# <format>, and libc++ comes from the same source as the compiler, so no gcc
# is installed for its library. clang 23 with lld and no PIE is the fastest
# build measured: +25% game speed and +16-22% bench over gcc 14 on the molab
# SPRT workers (2026-11-17).
#
# The compiler must run and say it is clang 23 after the install: apt takes a
# name it has no package for as a pattern, and a half-loaded index once made
# "g++-13" install clang-13 and its kin with exit code 0 (2026-11-17).
# Writes CC=clang-23 and CXX=clang++-23 to $GITHUB_ENV.
set -uo pipefail
V=23

sudo apt-get update -q && sudo apt-get install -y -q ninja-build lsb-release wget gnupg software-properties-common || exit 1

for i in 1 2 3; do
    if curl -fsSL --retry 3 --max-time 60 https://apt.llvm.org/llvm.sh -o /tmp/llvm.sh \
        && sudo bash /tmp/llvm.sh "$V" \
        && apt-cache pkgnames "libc++-$V-dev" | grep -xF "libc++-$V-dev" > /dev/null \
        && sudo apt-get install -y -q "clang-$V" "lld-$V" "libc++-$V-dev" "libc++abi-$V-dev" \
        && "clang++-$V" --version | sed -n 1p | grep -E "clang version $V\." \
        && "ld.lld-$V" --version | sed -n 1p; then
        echo "CC=clang-$V" >> "$GITHUB_ENV"
        echo "CXX=clang++-$V" >> "$GITHUB_ENV"
        exit 0
    fi
    echo "::warning::apt.llvm.org: attempt $i of 3 failed"
    [ "$i" -lt 3 ] && sleep $((i * 30))
done
echo "::error::no clang $V from apt.llvm.org"
exit 1
