#!/usr/bin/env bash
# clang 23, the compiler of every build: clang 23 + lld + no PIE is the fastest
# build measured, +25% game speed and +16-22% bench over gcc 14 on the molab
# SPRT workers (2026-11-17).
#
#   clang23.sh install     install clang 23 on this GitHub runner (by its OS)
#   clang23.sh check DIR   fail unless the compiler CMake found for DIR is clang 23
#
# install, per runner OS, and what it writes to $GITHUB_ENV:
#   Linux    clang 23, lld 23 and libc++ 23 from apt.llvm.org (Ubuntu 22.04's
#            libstdc++ has no <format>; no gcc is installed): CC, CXX
#   Windows  LLVM's own Windows tarball (LLVM 23 ships no installer) unpacked
#            into C:\LLVM23: LLVMInstallDir and LLVMToolsVersion, which MSBuild
#            reads, so Visual Studio's ClangCL toolset (cmake -T ClangCL)
#            compiles with clang-cl 23. Needs GH_TOKEN.
#   macOS    Homebrew's llvm and lld (ld64.lld links clang 23's LTO; Apple's
#            linker reads only Apple clang's bitcode): CXX and LUMINEX_SDK, the
#            SDK whose libc++ the build compiles against so macOS 13.3 holds;
#            lld's folder goes to $GITHUB_PATH
# Each install ends by running the compiler: apt takes a name it has no
# package for as a pattern, and a half-loaded index once made "g++-13" install
# clang-13 and its kin with exit code 0 (2026-11-17).
set -uo pipefail
V=23

install_linux() {
    sudo apt-get update -q && sudo apt-get install -y -q ninja-build lsb-release wget gnupg software-properties-common || return 1
    for i in 1 2 3; do
        if curl -fsSL --retry 3 --max-time 60 https://apt.llvm.org/llvm.sh -o /tmp/llvm.sh \
            && sudo bash /tmp/llvm.sh "$V" \
            && apt-cache pkgnames "libc++-$V-dev" | grep -xF "libc++-$V-dev" > /dev/null \
            && sudo apt-get install -y -q "clang-$V" "lld-$V" "libc++-$V-dev" "libc++abi-$V-dev" \
            && "clang++-$V" --version | sed -n 1p | grep -E "clang version $V\." \
            && "ld.lld-$V" --version | sed -n 1p; then
            echo "CC=clang-$V" >> "$GITHUB_ENV"
            echo "CXX=clang++-$V" >> "$GITHUB_ENV"
            return 0
        fi
        echo "::warning::apt.llvm.org: attempt $i of 3 failed"
        [ "$i" -lt 3 ] && sleep $((i * 30))
    done
    return 1
}

install_windows() {
    local tag ver asset
    tag=$(gh release list -R llvm/llvm-project --limit 100 --json tagName -q '.[].tagName' \
          | grep -E "^llvmorg-$V\.[0-9]+\.[0-9]+$" | sort -t. -k2,2n -k3,3n | tail -n 1)
    [ -n "$tag" ] || { echo "::error::no llvmorg-$V.x.y release"; return 1; }
    ver=${tag#llvmorg-}
    asset="clang+llvm-$ver-x86_64-pc-windows-msvc.tar.xz"
    echo "LLVM $ver ($asset)"
    for i in 1 2 3; do
        gh release download "$tag" -R llvm/llvm-project -p "$asset" -D "$RUNNER_TEMP" --clobber && break
        [ "$i" -lt 3 ] && sleep $((i * 20))
    done
    mkdir -p /c/LLVM23
    # a POSIX path: GNU tar reads "D:\a\..." as a remote host D
    tar -xJf "$(cygpath -u "$RUNNER_TEMP")/$asset" -C /c/LLVM23 --strip-components=1 || return 1
    /c/LLVM23/bin/clang-cl.exe --version | sed -n 1p | grep -E "clang version $V\." || return 1
    ls /c/LLVM23/bin/lld-link.exe "/c/LLVM23/lib/clang/$V" > /dev/null || return 1
    echo 'LLVMInstallDir=C:\LLVM23' >> "$GITHUB_ENV"
    echo "LLVMToolsVersion=$V" >> "$GITHUB_ENV"
}

install_macos() {
    local llvm lld
    brew install ninja llvm lld || return 1
    llvm=$(brew --prefix llvm)
    if ! "$llvm/bin/clang++" --version | sed -n 1p | grep -qE "clang version $V\."; then
        brew install "llvm@$V" || return 1   # Homebrew's llvm has moved past 23
        llvm=$(brew --prefix "llvm@$V")
    fi
    "$llvm/bin/clang++" --version | sed -n 1p | grep -E "clang version $V\." || return 1
    lld=$(brew --prefix lld)/bin
    "$lld/ld64.lld" --version | sed -n 1p || return 1
    echo "$lld" >> "$GITHUB_PATH"
    echo "CXX=$llvm/bin/clang++" >> "$GITHUB_ENV"
    echo "LUMINEX_SDK=$(xcrun --show-sdk-path)" >> "$GITHUB_ENV"
}

check() {
    local f id ver
    f=$(ls "$1"/CMakeFiles/*/CMakeCXXCompiler.cmake 2>/dev/null | head -n 1)
    [ -n "$f" ] || { echo "::error::no CMakeCXXCompiler.cmake under $1/CMakeFiles"; return 1; }
    id=$(sed -n 's/^set(CMAKE_CXX_COMPILER_ID "\(.*\)")$/\1/p' "$f")
    ver=$(sed -n 's/^set(CMAKE_CXX_COMPILER_VERSION "\(.*\)")$/\1/p' "$f")
    echo "C++ compiler: $id $ver"
    [ "$id" = Clang ] && [ "${ver%%.*}" = "$V" ] || { echo "::error::the build must be clang $V, CMake found $id $ver"; return 1; }
}

case "${1:-}" in
    install)
        case "${RUNNER_OS:-}" in
            Linux) install_linux ;;
            Windows) install_windows ;;
            macOS) install_macos ;;
            *) echo "::error::unknown runner OS '${RUNNER_OS:-}'"; false ;;
        esac || { echo "::error::no clang $V on this runner"; exit 1; } ;;
    check)
        check "${2:?usage: clang23.sh check BUILD_DIR}" || exit 1 ;;
    *)
        echo "usage: clang23.sh install | clang23.sh check BUILD_DIR" >&2; exit 2 ;;
esac
