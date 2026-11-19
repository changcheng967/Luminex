#!/usr/bin/env bash
# LLVM 23 for the Windows builds (Git Bash on a windows-2022 runner): LLVM's
# own release installer, from llvm/llvm-project's newest 23.x release,
# unpacked into C:\LLVM23 by 7-Zip (no install, no registry), then
# LLVMInstallDir and LLVMToolsVersion in $GITHUB_ENV, which MSBuild reads as
# properties: Visual Studio's ClangCL toolset (cmake -T ClangCL) then
# compiles with LLVM 23's clang-cl instead of the LLVM that Visual Studio
# bundles. check-compiler.sh confirms what CMake found. Needs GH_TOKEN.
set -euo pipefail
V=23

tag=$(gh release list -R llvm/llvm-project --limit 100 --json tagName -q '.[].tagName' \
      | grep -E "^llvmorg-$V\.[0-9]+\.[0-9]+$" | sort -t. -k2,2n -k3,3n | tail -n 1)
[ -n "$tag" ] || { echo "::error::no llvmorg-$V.x.y release found"; exit 1; }
ver=${tag#llvmorg-}
exe="LLVM-$ver-win64.exe"
echo "LLVM $ver ($exe)"
for i in 1 2 3; do
    gh release download "$tag" -R llvm/llvm-project -p "$exe" -D "$RUNNER_TEMP" --clobber && break
    [ "$i" -lt 3 ] && sleep $((i * 20))
done
7z x -y -o'C:\LLVM23' "$RUNNER_TEMP/$exe" > /dev/null
/c/LLVM23/bin/clang-cl.exe --version | sed -n 1p | grep -E "clang version $V\."
ls -d "/c/LLVM23/lib/clang/$V"
echo 'LLVMInstallDir=C:\LLVM23' >> "$GITHUB_ENV"
echo "LLVMToolsVersion=$V" >> "$GITHUB_ENV"
