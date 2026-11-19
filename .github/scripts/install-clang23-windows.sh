#!/usr/bin/env bash
# LLVM 23 for the Windows builds (Git Bash on a windows-2022 runner): LLVM's
# own x86-64 Windows (MSVC) release tarball, from llvm/llvm-project's newest
# 23.x release (LLVM 23 ships Windows as clang+llvm-*.tar.xz, no installer),
# unpacked into C:\LLVM23, then LLVMInstallDir and LLVMToolsVersion in
# $GITHUB_ENV, which MSBuild reads as properties: Visual Studio's ClangCL
# toolset (cmake -T ClangCL) then compiles with LLVM 23's clang-cl and links
# with its lld-link instead of the LLVM that Visual Studio bundles.
# check-compiler.sh confirms what CMake found. Needs GH_TOKEN.
set -euo pipefail
V=23

tag=$(gh release list -R llvm/llvm-project --limit 100 --json tagName -q '.[].tagName' \
      | grep -E "^llvmorg-$V\.[0-9]+\.[0-9]+$" | sort -t. -k2,2n -k3,3n | tail -n 1)
[ -n "$tag" ] || { echo "::error::no llvmorg-$V.x.y release found"; exit 1; }
ver=${tag#llvmorg-}
asset="clang+llvm-$ver-x86_64-pc-windows-msvc.tar.xz"
echo "LLVM $ver ($asset)"
for i in 1 2 3; do
    gh release download "$tag" -R llvm/llvm-project -p "$asset" -D "$RUNNER_TEMP" --clobber && break
    [ "$i" -lt 3 ] && sleep $((i * 20))
done
mkdir -p /c/LLVM23
tar -xJf "$RUNNER_TEMP/$asset" -C /c/LLVM23 --strip-components=1
/c/LLVM23/bin/clang-cl.exe --version | sed -n 1p | grep -E "clang version $V\."
ls /c/LLVM23/bin/lld-link.exe
ls -d "/c/LLVM23/lib/clang/$V"
echo 'LLVMInstallDir=C:\LLVM23' >> "$GITHUB_ENV"
echo "LLVMToolsVersion=$V" >> "$GITHUB_ENV"
