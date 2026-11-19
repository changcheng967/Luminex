#!/usr/bin/env bash
# LLVM 23 for the macOS build: Homebrew's llvm (clang) and lld (ld64.lld, which
# links clang 23's LTO: Apple's linker reads only Apple clang's bitcode), or
# llvm@23 once Homebrew's llvm has moved past 23. Writes CXX (Homebrew's
# clang++) and LUMINEX_SDK (the SDK, whose libc++ headers the build uses) to
# $GITHUB_ENV and lld's folder to $GITHUB_PATH.
set -euo pipefail
V=23
brew install ninja llvm lld
LLVM=$(brew --prefix llvm)
if ! "$LLVM/bin/clang++" --version | sed -n 1p | grep -qE "clang version $V\."; then
    brew install "llvm@$V"
    LLVM=$(brew --prefix "llvm@$V")
fi
"$LLVM/bin/clang++" --version | sed -n 1p | grep -E "clang version $V\."
LLD=$(brew --prefix lld)/bin
"$LLD/ld64.lld" --version | sed -n 1p
echo "$LLD" >> "$GITHUB_PATH"
echo "CXX=$LLVM/bin/clang++" >> "$GITHUB_ENV"
echo "LUMINEX_SDK=$(xcrun --show-sdk-path)" >> "$GITHUB_ENV"
