#!/usr/bin/env bash
# gcc-13 for the Linux builds on Ubuntu 22.04, whose own gcc (11) has no <format>.
#
# First choice: the ubuntu-toolchain-r PPA, three tries, each counted only when
# g++-13 then runs. When Launchpad cannot serve it (its 503s failed every Linux
# job on 2026-11-17), the same gcc 13 from conda-forge instead. Its compiler targets an older glibc through its own sysroot,
# so the binary still runs on glibc 2.35 and newer; check-deps.py checks that
# whichever compiler built it.
#
# GCC13_SOURCE=conda skips the PPA, to exercise the fallback on purpose.
# Writes CXX=<the compiler> to $GITHUB_ENV.
set -uo pipefail

sudo apt-get update -q && sudo apt-get install -y -q ninja-build || exit 1

# A down PPA costs apt minutes per try (it fails package by package), so ask its
# index first and skip straight to the fallback when even that is refused.
PPA=https://ppa.launchpadcontent.net/ubuntu-toolchain-r/test/ubuntu/dists/jammy/Release
if [ "${GCC13_SOURCE:-}" != conda ] && ! curl -fsS -o /dev/null --max-time 20 "$PPA"; then
    echo "::warning::toolchain PPA index unreachable"
    GCC13_SOURCE=conda
fi

if [ "${GCC13_SOURCE:-}" != conda ]; then
    for i in 1 2 3; do
        # apt takes a name it has no package for as a pattern. When the PPA's
        # lists half-loaded (main, 2026-11-17 21:18 UTC), "g++-13" matched
        # clang-13 and its kin, apt installed 17 of those and exited 0, and
        # CMake found no g++-13. So the exact package must be known before
        # the install, and the compiler must run after it.
        if sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test \
            && apt-cache pkgnames g++-13 | grep -xF g++-13 > /dev/null \
            && sudo apt-get install -y -q g++-13 \
            && g++-13 --version | sed -n 1p; then
            echo "CXX=g++-13" >> "$GITHUB_ENV"
            echo "gcc-13 from the toolchain PPA"
            exit 0
        fi
        echo "::warning::toolchain PPA: attempt $i of 3 failed"
        [ "$i" -lt 3 ] && sleep $((i * 30))
    done
    # Launchpad is unreachable: drop its source so no later apt call trips on it
    sudo add-apt-repository -y --remove ppa:ubuntu-toolchain-r/test || true
fi

curl -fsSL --retry 3 https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$HOME" bin/micromamba || exit 1
"$HOME/bin/micromamba" create -y -q -p "$HOME/gcc13" -c conda-forge "gxx_linux-64=13" || exit 1
CXX="$HOME/gcc13/bin/x86_64-conda-linux-gnu-g++"
"$CXX" --version | sed -n 1p || exit 1   # sed reads it all: no SIGPIPE for pipefail to see
echo "CXX=$CXX" >> "$GITHUB_ENV"
echo "::warning::gcc-13 from conda-forge, because the toolchain PPA could not serve it"
