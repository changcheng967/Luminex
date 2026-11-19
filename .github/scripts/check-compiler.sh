#!/usr/bin/env bash
# Every build is clang 23: the compiler CMake identified for BUILD_DIR must be
# Clang, major version MAJOR. A runner image or package manager that quietly
# hands over another compiler (Apple clang, gcc, an older LLVM) fails here.
#   check-compiler.sh BUILD_DIR [MAJOR]     (MAJOR defaults to 23)
set -euo pipefail
dir=$1
major=${2:-23}
f=$(ls "$dir"/CMakeFiles/*/CMakeCXXCompiler.cmake 2>/dev/null | head -n 1)
[ -n "$f" ] || { echo "no CMakeCXXCompiler.cmake under $dir/CMakeFiles" >&2; exit 1; }
id=$(sed -n 's/^set(CMAKE_CXX_COMPILER_ID "\(.*\)")$/\1/p' "$f")
ver=$(sed -n 's/^set(CMAKE_CXX_COMPILER_VERSION "\(.*\)")$/\1/p' "$f")
echo "C++ compiler: $id $ver"
if [ "$id" != Clang ] || [ "${ver%%.*}" != "$major" ]; then
    echo "::error::the build must be clang $major, CMake found $id $ver" >&2
    exit 1
fi
