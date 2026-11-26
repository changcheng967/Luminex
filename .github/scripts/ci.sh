#!/usr/bin/env bash
# Luminex's CI tools, one subcommand each (the workflows call them as
# bash .github/scripts/ci.sh SUBCOMMAND ARGS):
#   clang install | clang check DIR   clang 23 for this runner; fail unless CMake found it
#   fetch-net [DIR]                   the pinned network, downloaded into DIR (net) and its sha256 checked
#   isa BINARY LEVEL                  the binary's own code needs no more than LEVEL
#   name BINARY NAME                  the binary reports itself as Luminex NAME (a build's id, or a version)
#   net [--no-embed] BINARY           the built-in network loads and is the pinned one
#   simd BINARY [--net FILE] [CHIP "LEVELS"]   every kernel level evaluates alike
#   sde                               fetch Intel SDE, print the launcher's path
#   dev-release DIST                  publish the rolling luminex-dev pre-release
#   deps BINARY ASSET                 the binary needs no more from the system than its notes say
#   bench-line SHA BENCH              the commit's last Bench: line is the node count the builds measured
# Below them: the network pin, which CMake reads too; and, before deps, the
# format of a release's notes.

# ----- the network pin
# The network the release binaries carry. fetch-net downloads it and checks
# its sha256, so every build embeds exactly the shipped file. The pin is
# gen768t_05q (luminex_gen768t_05q_i8.nnue on the engine-archives
# pre-release). Shipping a new network means attaching it to a release and
# moving pin_url, pin_sha256 and pin_about here. The url's file name is the
# network's real name and its only source: fetch-net saves the file under it,
# CMake (from the pin_url line) makes it the engine's default network
# (DEFAULT_NET) and the name the built-in network reports, and releases attach
# the file under it (net fails a build that names another network).
# pin_about, the network's name and a line on how it was made, goes into the
# development build's notes (dev-release).
pin_url=https://github.com/changcheng967/Luminex/releases/download/engine-archives/luminex_gen768t_05q_i8.nnue
pin_sha256=1d9f44604e289cda055e246166b7f9a5fe419bf351d4b25142e9a2c35eb75544
pin_about="gen768t_05q: V5 (gen768t, docs/NNUE.md), gen768s's feature layer with 55,960 int8 relation rows (attacks, and pawn pairs in contact) summed into 256 of its 768 lanes (the LNT8 format); gen768t_04's float network trained on for 2,693,718 more steps (353.1B positions, a fresh optimizer, lr 5e-5) on Leela data relabelled by BT4: eleven months of test91 games and six corpus files (leela96 splits 2-4, test80 games of April, May and July 2023), then 30K steps of QAT with int8 relation rows and one int8 scale per tail row"

# ----- fetch-net
# Download the pinned network and check its sha256. Prints the file's path and
# which network it holds: the file is always saved under the network's real
# name, so the path alone says.
#
#   ci.sh fetch-net [DIR]   (default: net)
cmd_fetch_net() {
set -euo pipefail
dir=${1:-net}
out=$dir/${pin_url##*/}
mkdir -p "$dir"
[ -f "$out" ] || curl -fsSL --retry 3 -o "$out" "$pin_url"
if command -v sha256sum > /dev/null; then
  got=$(sha256sum "$out" | cut -d' ' -f1)
else
  got=$(shasum -a 256 "$out" | cut -d' ' -f1)
fi
if [ "$got" != "$pin_sha256" ]; then
  echo "$out: sha256 $got, expected $pin_sha256" >&2
  rm -f "$out"
  exit 1
fi
echo "$out (OpenBench id $(printf '%s' "${pin_sha256:0:8}" | tr a-f A-F)), sha256 checked"
}

# ----- clang
# clang 23, the compiler of every build: clang 23 + lld + no PIE is the fastest
# build measured, +25% game speed and +16-22% bench over gcc 14 on the molab
# SPRT workers (2026-11-17).
#
#   ci.sh clang install     install clang 23 on this GitHub runner (by its OS)
#   ci.sh clang check DIR   fail unless the compiler CMake found for DIR is clang 23
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
    local llvm lld f
    # only what the runner lacks: brew warns of a formula already installed
    for f in ninja llvm lld; do
        brew list --formula "$f" > /dev/null 2>&1 || brew install "$f" || return 1
    done
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

clang_check() {
    local f id ver
    f=$(ls "$1"/CMakeFiles/*/CMakeCXXCompiler.cmake 2>/dev/null | head -n 1)
    [ -n "$f" ] || { echo "::error::no CMakeCXXCompiler.cmake under $1/CMakeFiles"; return 1; }
    id=$(sed -n 's/^set(CMAKE_CXX_COMPILER_ID "\(.*\)")$/\1/p' "$f")
    ver=$(sed -n 's/^set(CMAKE_CXX_COMPILER_VERSION "\(.*\)")$/\1/p' "$f")
    echo "C++ compiler: $id $ver"
    [ "$id" = Clang ] && [ "${ver%%.*}" = "$V" ] || { echo "::error::the build must be clang $V, CMake found $id $ver"; return 1; }
}

cmd_clang() {
    set -uo pipefail
    case "${1:-}" in
        install)
            case "${RUNNER_OS:-}" in
                Linux) install_linux ;;
                Windows) install_windows ;;
                macOS) install_macos ;;
                *) echo "::error::unknown runner OS '${RUNNER_OS:-}'"; false ;;
            esac || { echo "::error::no clang $V on this runner"; exit 1; } ;;
        check)
            clang_check "${2:?usage: ci.sh clang check BUILD_DIR}" || exit 1 ;;
        *)
            echo "usage: ci.sh clang install | ci.sh clang check BUILD_DIR" >&2; exit 2 ;;
    esac
}

# ----- isa
# Check that an engine binary has the network kernels for the instruction set
# its name promises, so a mislabelled build fails CI instead of shipping. In
# v6.3.0 the Windows modern build had no AVX2 kernels, both compatibility
# builds needed AVX2, and the macOS build had no NEON dot-product kernels.
#
#   ci.sh isa BINARY universal|avx2|baseline|dotprod
#
# universal is the x86-64 release build (CMake LUMINEX_DISPATCH), which must
# carry every level's kernels; avx2 and baseline are single-level x86 builds,
# dotprod the Apple silicon one. The markers come only from the network
# kernels, each level's own instructions:
#   avx512vnni  vpdpbusd on zmm registers (the dense layers)
#   avx512      vpaddw or vpsubw on zmm (the accumulator's wrapping adds)
#   avxvnni     vpdpbusd in its VEX form, which objdump prints as {vex}
#   avx2        vpmaddubsw on ymm (the dense layers)
#   base        paddw or psubw without the VEX v prefix (the SSE2 accumulator)
# The accumulator wraps instead of saturating since PR #106; in the universal
# build (gcc 16) every zmm vpaddw/vpsubw comes from the two AVX-512 kernel
# files and every legacy paddw/psubw from nnue_k_base.
#   dotprod     sdot (NEON)
# Counting all ymm use would not work on Windows, where the C++ runtime links
# in its own CPU-dispatched AVX2 string routines.
cmd_isa() {
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
    check 'AVX-512 kernel instructions' 'vp(add|sub)w.*zmm' present
    check 'AVX-VNNI kernel instructions' '\{vex\}[[:space:]]*vpdpbusd' present
    check 'AVX2 kernel instructions' 'vpmaddubsw.*ymm' present
    check 'SSE2 kernel instructions' '[[:space:]]p(add|sub)w[[:space:]]' present ;;
  avx2) check 'AVX2 network instructions' 'vpmaddubsw.*ymm' present ;;
  baseline) check 'AVX2 network instructions' 'vpmaddubsw.*ymm' absent ;;
  dotprod) check 'NEON dot-product instructions' 'sdot' present ;;
  *) echo "unknown level: $want" >&2; exit 2 ;;
esac
}

# ----- name
# Print an engine binary's UCI name and, given a development build id, check
# that the binary reports it ("id name Luminex dev-20261112-d4774f1a"), as the
# luminex-dev pre-release binaries do.
#
#   ci.sh name BINARY [BUILD_ID]
cmd_name() {
set -euo pipefail
bin=$1
want=${2:-}
out=$(printf 'uci\nquit\n' | "$bin" 2>&1)
out=${out//$'\r'/}
name=$(grep -m1 '^id name' <<< "$out" || true)
echo "$name"
if [ -n "$want" ] && [ "$name" != "id name Luminex $want" ]; then
  echo "$bin: expected 'id name Luminex $want'" >&2
  exit 1
fi
}

# ----- net
# Run an engine binary by itself, copied into an empty directory, and check
# that it evaluates with its built-in network, and names that network by its
# real name. v6.3.0 to v6.5.0 shipped the network only as a separate file, so
# a tester who ran the binary alone got the hand-crafted evaluation; v6.6.0's
# first binaries called gen768r_06q by the default file name of v6.3.0's
# network.
#
#   ci.sh net BINARY              a build with the network inside
#   ci.sh net --no-embed BINARY   a build without one (the Makefile's, as
#                                    OpenBench builds): only NNUEFile's default
#
# The real name is the file name of the pinned network (pin_url, above). The
# binary must offer it as NNUEFile's default and, with the network inside,
# report its built-in network under it, or the check fails.
cmd_net() {
set -euo pipefail
embed=1
if [ "${1:-}" = --no-embed ]; then embed=0; shift; fi
bin=$1
want=${pin_url##*/}
dir=$(mktemp -d)
cp "$bin" "$dir/"
out=$(cd "$dir" && printf 'uci\nisready\nposition startpos\neval\nquit\n' | "./$(basename "$bin")" 2>&1 | tr -d '\r')
# Windows can keep the copied binary locked for a moment after it exits, so
# the cleanup retries and never fails the check.
for _ in 1 2 3 4 5; do rm -rf "$dir" 2> /dev/null && break; sleep 1; done
printf '%s\n' "$out" | grep -E '^(id name|info string|eval|option name NNUEFile)' || true
if ! printf '%s\n' "$out" | grep -qxF "option name NNUEFile type string default $want"; then
  echo "$bin: NNUEFile's default is not $want, the pinned network" >&2
  exit 1
fi
if [ "$embed" = 1 ]; then
  if ! printf '%s\n' "$out" | grep -q 'NNUE evaluation using built-in'; then
    echo "$bin: the built-in network did not load" >&2
    exit 1
  fi
  if ! printf '%s\n' "$out" | grep -qF "NNUE evaluation using built-in $want ("; then
    echo "$bin: the built-in network does not report itself as $want, the pinned network" >&2
    exit 1
  fi
fi
if ! printf '%s\n' "$out" | grep -q '^eval cp'; then
  echo "$bin: no eval" >&2
  exit 1
fi
}

# ----- simd
# Check that every SIMD kernel level of an engine binary evaluates alike. The
# binary's nnuecheck (luminex nnuecheck [NETFILE]) plays a fixed set of games
# under every level the CPU supports, fails on any difference, and prints a
# digest of the evaluations, which every x86-64 build of the same network
# must share (build.yml compares the release builds' with OpenBench's).
#
#   ci.sh simd BINARY [--net FILE] [CHIP "LEVELS"]
#
# Without CHIP it runs on this CPU. With CHIP it runs under Intel SDE (the
# launcher's path in $SDE, from ci.sh sde) as the CPU of that SDE chip
# option (spr, icx, skx, arl, hsw, snb, nhm, ...): LEVELS must be exactly the
# levels the binary finds there, best first, and SDE's chip check stops the
# binary if it executes an instruction that CPU lacks. $DIGEST, if set, is the
# digest the run must print (another run's). With $GITHUB_OUTPUT set, the
# digest is also written there as digest=...
cmd_simd() {
set -euo pipefail
bin=$1
shift
net=
if [ "${1:-}" = --net ]; then net=$2; shift 2; fi
chip=${1:-}
want=${2:-}
if [ -n "$chip" ]; then
  : "${SDE:?SDE must name the sde launcher (ci.sh sde)}"
  run=("$SDE" "-$chip" -chip-check-exe-only -- "$bin" nnuecheck)
  where="on $chip (SDE)"
else
  run=("$bin" nnuecheck)
  where="on this CPU"
fi
[ -z "$net" ] || run+=("$net")
status=0
out=$("${run[@]}" 2>&1 | tr -d '\r') || status=$?
printf '%s\n' "$out" | grep -vE '^(id |option )' || true
if [ "$status" != 0 ] || ! printf '%s\n' "$out" | grep -qx 'nnuecheck ok'; then
  echo "$bin: nnuecheck failed $where (exit status $status)" >&2
  exit 1
fi
levels=$(printf '%s\n' "$out" | sed -n 's/^nnuecheck: levels \(.*\) (in use: .*)$/\1/p')
if [ -n "$want" ] && [ "$levels" != "$want" ]; then
  echo "$bin $where: levels '$levels', expected '$want'" >&2
  exit 1
fi
digest=$(printf '%s\n' "$out" | sed -n 's/^nnuecheck: .* digest \([0-9a-f]*\)$/\1/p')
if [ -z "$digest" ]; then
  echo "$bin $where: nnuecheck printed no digest" >&2
  exit 1
fi
if [ -n "${DIGEST:-}" ] && [ "$digest" != "$DIGEST" ]; then
  echo "$bin $where: digest $digest, expected $DIGEST" >&2
  exit 1
fi
echo "$bin $where: levels $levels agree, digest $digest"
[ -z "${GITHUB_OUTPUT:-}" ] || echo "digest=$digest" >> "$GITHUB_OUTPUT"
}

# ----- sde
# Download Intel's Software Development Emulator, which runs a binary as on
# another Intel CPU (its cpuid, and with "chip check" only that CPU's
# instructions), so CI can check the universal build's AVX-512 and AVX-VNNI
# kernels on GitHub's runners, which have neither. The version and the
# sha256 of each package are Intel's, from its download page
# (https://www.intel.com/content/www/us/en/download/684897/intel-software-development-emulator.html).
# Prints the path of the sde launcher.
#
#   ci.sh sde [DIR]   (default: $RUNNER_TEMP/sde, or ./sde)
cmd_sde() {
set -euo pipefail
version=10.13.1-2026-07-28
mirror=https://downloadmirror.intel.com/924984
case "$(uname -s)" in
  Linux*)
    os=lin; exe=sde64
    sha256=94e97d623fec54385686e1e7ba65ebc9941748c05ee451423948334892bf2b50 ;;
  MINGW*|MSYS*|CYGWIN*)
    os=win; exe=sde.exe
    sha256=74e626ede09b0baa5011fc9e51b58627ea92c3fc0bae5fd7db34b490f335f651 ;;
  *) echo "no SDE package for $(uname -s)" >&2; exit 1 ;;
esac
dir=${1:-${RUNNER_TEMP:-.}/sde}
# Windows runners give RUNNER_TEMP as a Windows path; Git Bash takes POSIX ones.
if command -v cygpath > /dev/null; then dir=$(cygpath -u "$dir"); fi
pkg=sde-external-$version-$os
mkdir -p "$dir"
if [ ! -f "$dir/$pkg/$exe" ]; then
  curl -fsSL --retry 3 -o "$dir/$pkg.tar.xz" "$mirror/$pkg.tar.xz"
  got=$(sha256sum "$dir/$pkg.tar.xz" | cut -d' ' -f1)
  if [ "$got" != "$sha256" ]; then
    echo "$mirror/$pkg.tar.xz: sha256 $got, expected $sha256" >&2
    rm -f "$dir/$pkg.tar.xz"
    exit 1
  fi
  tar -xf "$dir/$pkg.tar.xz" -C "$dir"
  rm -f "$dir/$pkg.tar.xz"
fi
echo "$dir/$pkg/$exe"
}

# ----- dev-release
# Publish the rolling development build. The binaries in DIR, built and tested
# by build.yml's jobs, become the pre-release luminex-$BUILD_ID; then every
# older luminex-dev release and tag is deleted, so one development build stands
# next to the stable releases. Only tags of the exact form
# luminex-dev-YYYYMMDD-<8 hex digits> are deleted: the v* releases and
# engine-archives (OpenBench's store of old builds and networks) never match.
# DRY_RUN=1, the default, prints the notes, the assets and the deletions
# without making them.
#
#   GH_TOKEN=... BUILD_ID=dev-20261112-d4774f1a [DRY_RUN=0] ci.sh dev-release DIR
cmd_dev_release() {
set -euo pipefail
dir=$1
: "${BUILD_ID:?}"
dry=${DRY_RUN:-1}
repo=${GITHUB_REPOSITORY:-changcheng967/Luminex}
url=https://github.com/$repo
sha=$(git rev-parse HEAD)
tag=luminex-$BUILD_ID
pattern='^luminex-dev-[0-9]{8}-[0-9a-f]{8}$'
[[ $tag =~ $pattern ]] || { echo "$tag is not a development build tag" >&2; exit 1; }
[[ $BUILD_ID == *-"${sha:0:8}" ]] || { echo "$BUILD_ID does not name commit $sha" >&2; exit 1; }

# The network the binaries carry, as the pin above names it: its file name on
# engine-archives, its OpenBench id (the first 8 hex digits of its sha256) and
# its about line. fetch-net saved it under its own name, the name it is
# attached under, so the notes and the file say which network it is.
here=$(dirname "$0")
net_name=${pin_url##*/}
net_about=$pin_about
net_id=$(sha256sum "$dir/$net_name" | cut -c1-8 | tr a-f A-F)
if [ -n "$net_about" ]; then
  net_line="the network ${net_about%%: *} (OpenBench id \`$net_id\`): ${net_about#*: }."
else
  net_line="the network \`$net_name\` (OpenBench id \`$net_id\`)."
fi

# Every build job's binary, and the network on its own
assets=(luminex-linux-x86-64 luminex-windows-x86-64.exe luminex-macos-arm64 "$net_name")
files=()
for a in "${assets[@]}"; do
  [ -f "$dir/$a" ] || { echo "missing $dir/$a" >&2; exit 1; }
  files+=("$dir/$a")
done
chmod +x "$dir/luminex-linux-x86-64"
bash "$here/ci.sh" name "$dir/luminex-linux-x86-64" "$BUILD_ID"

# The notes cover everything since the latest stable release, since each push
# to main replaces the previous development build. Commits are split into
# those that change the engine binary (its sources, or the network it carries)
# and those that do not (training, tools, CI, docs), each linked to its pull
# request.
stable=$(gh release view --repo "$repo" --json tagName --jq .tagName)
mapfile -t dev_tags < <(gh release list --repo "$repo" --limit 100 --json tagName --jq '.[].tagName' \
                        | grep -E "$pattern" | grep -vx "$tag" || true)
base=$(git rev-parse "refs/tags/$stable^{commit}")
# The data tools' file; the other names are its parts before they shared it, for older commits.
tool_only='^src/(tools\.cpp|eval_trace\.cpp|featurize\.cpp|frame_merge\.cpp|lc0pack\.cpp|lc0_move_table\.h|verify_frames\.c)$'
engine_lines=()
other_lines=()
while read -r h; do
  subject=$(git show -s --format=%s "$h")
  pr=$(gh api "repos/$repo/commits/$h/pulls" --jq '[.[] | select(.merged_at != null)][0].number // empty' 2> /dev/null || true)
  # A squash merge's subject already ends in " (#N)", which GitHub links by
  # itself: the pull request's link replaces it, so the number shows once
  [ -z "$pr" ] || subject="${subject%" (#$pr)"} ([#$pr]($url/pull/$pr))"
  line="* [\`${h:0:8}\`]($url/commit/$h) $subject"
  engine=$(git show --name-only --format= "$h" | grep -E '^src/' | grep -vE "$tool_only" || true)
  # A move of the network pin counts (pin_sha256 here, sha256 in fetch-net.sh
  # before the pin moved into this file), not the files' other edits
  [ -n "$engine" ] || engine=$(git show --format= "$h" -- .github/scripts/ci.sh .github/scripts/fetch-net.sh \
                               | grep -E '^[-+](pin_)?sha256=' || true)
  if [ -n "$engine" ]; then engine_lines+=("$line"); else other_lines+=("$line"); fi
done < <(git rev-list --no-merges "$base..$sha")
count=$(( ${#engine_lines[@]} + ${#other_lines[@]} ))
[ "$count" = 1 ] && commits="1 commit" || commits="$count commits"
when=$(TZ=UTC git show -s --format=%cd --date=format-local:'%d %B %Y' "$sha")
when=${when#0}
# The stable release's network: its pin in ci.sh, or in fetch-net.sh before the pin moved here
stable_net=$( { git show "$stable:.github/scripts/ci.sh" 2> /dev/null | sed -n 's/^pin_sha256=\([0-9a-f]\{8\}\).*/\1/p'
                git show "$stable:.github/scripts/fetch-net.sh" 2> /dev/null | sed -n 's/^sha256=\([0-9a-f]\{8\}\).*/\1/p'; } \
             | head -n 1 | tr a-f A-F || true)
if [ -z "$stable_net" ]; then
  net_since=""
elif [ "$stable_net" = "$net_id" ]; then
  net_since=" It is also the network in $stable."
else
  net_since=" It replaces \`$stable_net\`, the network in $stable."
fi
{
  echo "**Luminex \`$BUILD_ID\`**: the engine as it stands on \`main\` on $when, $commits after [$stable]($url/releases/tag/$stable). CI built every binary below from commit [\`${sha:0:8}\`]($url/commit/$sha) and checked that each one runs, evaluates with its built-in network and names itself after this build."
  echo
  echo "A development build has no version number yet: it is what the next release is being made from. Every push to \`main\` publishes a new one and removes the old, so there is only ever one. For a numbered release, take [$stable]($url/releases/tag/$stable)."
  echo
  echo "## Network"
  echo
  echo "Every binary carries $net_line$net_since The same file is attached on its own as \`$net_name\`, for builds without a network inside: load it with the \`NNUEFile\` option."
  echo
  echo "## Changes since $stable"
  echo
  echo "### In the engine (${#engine_lines[@]})"
  echo
  echo "These change how Luminex searches, evaluates or reports, or the network it plays with."
  echo
  if [ ${#engine_lines[@]} -gt 0 ]; then printf '%s\n' "${engine_lines[@]}"; else echo "None yet."; fi
  echo
  echo "### Training, tools and CI (${#other_lines[@]})"
  echo
  echo "These leave the engine's play unchanged."
  echo
  if [ ${#other_lines[@]} -gt 0 ]; then printf '%s\n' "${other_lines[@]}"; else echo "None yet."; fi
  echo
  echo "[Every change since $stable, file by file]($url/compare/$stable...$sha)"
  echo
  echo "## Which binary"
  echo
  echo "| Platform | Binary | CPU | Needs |"
  echo "|---|---|---|---|"
  echo "| Linux | \`luminex-linux-x86-64\` | any x86-64 with SSE4.2 and POPCNT (x86-64-v2) | Ubuntu 22.04 or newer (glibc 2.35) |"
  echo "| Windows | \`luminex-windows-x86-64.exe\` | any x86-64 with POPCNT | nothing to install |"
  echo "| macOS | \`luminex-macos-arm64\` | Apple silicon | macOS 13.3 or newer |"
  echo
  echo "One binary per platform: the x86-64 ones carry the network kernels for every instruction-set level (AVX-512 with VNNI, AVX-512, AVX-VNNI, AVX2 and a baseline) and run the best one the CPU supports; the \`info string\` that reports the network names it. Each binary is about 76 MB, the network included."
  echo
  echo "## Reporting a result or a bug"
  echo
  echo "The engine names itself \`Luminex $BUILD_ID\` (UCI \`id name\`). Quote that name with any game, rating or bug report, so it can be traced to this exact commit. Search and evaluation changes are measured on [OpenBench](https://bench.luminexchess.com/) before they merge; [luminexchess.com](https://luminexchess.com/) has the project's games and ratings."
} > notes.md

mapfile -t stale_releases < <(printf '%s\n' "${dev_tags[@]}" | grep -E "$pattern" || true)
mapfile -t stale_tags < <(gh api "repos/$repo/git/matching-refs/tags/luminex-dev-" --jq '.[].ref' \
                          | sed 's#^refs/tags/##' | grep -E "$pattern" | grep -vx "$tag" || true)

if [ "$dry" != 0 ]; then
  echo "DRY RUN: would publish $tag at $sha with ${#files[@]} assets:"
  printf '  %s\n' "${files[@]}"
  echo "DRY RUN: would delete the releases: ${stale_releases[*]:-none}; the tags: ${stale_tags[*]:-none}"
  echo "----- notes -----"
  cat notes.md
  exit 0
fi

# Only main's tip publishes: a run that ends after a newer push must not
# replace the newer build (runs of this job queue one at a time)
tip=$(gh api "repos/$repo/commits/main" --jq .sha)
if [ "$tip" != "$sha" ]; then
  echo "main has moved on to ${tip:0:8}: ${sha:0:8} is not published"
  exit 0
fi

if gh release view "$tag" --repo "$repo" > /dev/null 2>&1; then
  # A re-run of the same commit refreshes its release
  gh release upload "$tag" "${files[@]}" --repo "$repo" --clobber
  gh release edit "$tag" --repo "$repo" --prerelease --notes-file notes.md
else
  gh release create "$tag" "${files[@]}" --repo "$repo" --target "$sha" --prerelease --latest=false \
    --title "Luminex $BUILD_ID" --notes-file notes.md
fi
echo "published $url/releases/tag/$tag"

# The new build is up; the older ones go, then any tag a cut-short deletion left
for old in "${stale_releases[@]}"; do
  gh release delete "$old" --repo "$repo" --cleanup-tag --yes
  echo "deleted the release $old and its tag"
done
for old in "${stale_tags[@]}"; do
  if gh api "repos/$repo/git/refs/tags/$old" > /dev/null 2>&1; then
    gh api -X DELETE "repos/$repo/git/refs/tags/$old"
    echo "deleted the tag $old"
  fi
done
}

# ----- release notes
# The format of a v* release's notes (the release job in build.yml creates the
# release; its notes are written by hand, in this shape):
#
# > **Bold one line:** the Elo over the previous release and the rating-list
# > estimate. Under it, one plain sentence on what the release brings.
#
# ## What's new
#
# One bold line per area (network, time management, search, speed, ...) with
# its PR links, then short plain bullets: what changed, a sentence each. No
# figures: the release's own Elo is in Strength, and each change's is on its
# PR and test page.
#
# ## Strength
#
# | Match | Games | W-D-L | Result |
# |---|---:|---:|---|
# | vX vs the previous release (test N) | N | W-D-L | **Elo ± 95%** |
# | The gauntlet (gauntlet N) | N | W-D-L | **rating ±** per list |
#
# One line of conditions (threads, hash, time control, openings), then any
# caveat. The results against each opponent go in a collapsed `<details>` table.
# Quote each Elo as its OpenBench test page shows it, with the 95% interval,
# and link the page: commit messages keep the figure of their day, which drifts
# when OpenBench's error model changes (v6.6.0's first notes quoted several
# that had).
#
# No table of each change's SPRT: it was the longest part of the notes and
# few read it (v6.6.0's was removed); the PRs and their test pages keep them.
#
# ## Downloads
#
# | Platform | Binary | CPU | Needs |
# |---|---|---|---|
# | Linux | `luminex-linux-x86-64` | any x86-64 with SSE4.2 and POPCNT (x86-64-v2) | Ubuntu 22.04 or newer (glibc 2.35) |
# | Windows | `luminex-windows-x86-64.exe` | any x86-64 with POPCNT | nothing to install |
# | macOS | `luminex-macos-arm64` | Apple silicon | macOS 13.3 or newer |
#
# One binary per platform: the x86-64 ones carry the network kernels of every
# instruction-set level (AVX-512 with VNNI, AVX-512, AVX-VNNI, AVX2, a
# baseline), use the best the CPU supports, and name it at the end of the
# `info string` that reports the network. Every level evaluates alike
# (nnuecheck in CI).
#
# The first release with these binaries says, under Downloads, that the old
# x86-64 assets are gone, since the README told users with fast CPUs to take
# `-modern`:
#
# > **One x86-64 download per OS now.** The `-modern`, `-glibc235` and
# > `-glibc235-modern` binaries are retired. If you used `-modern`, take
# > `luminex-linux-x86-64` or `luminex-windows-x86-64.exe`: it picks the
# > AVX2 or AVX-512 kernels `-modern` had by itself, and AVX-VNNI on CPUs
# > such as Arrow Lake.
#
# Needs is what the binary needs from the system: glibc and libstdc++ on Linux,
# any runtime to install on Windows, the oldest macOS. ci.sh deps reads it
# from every binary's headers in CI and fails a build that needs more than its
# row says, so a changed limit changes the row.
#
# Every asset carries its real name: the network is attached as
# `luminex_<architecture>_<NN>_i8.nnue`, the name the binaries report, never
# under a past network's name or a neutral one. ci.sh net fails a build that
# reports any other.
#
# ## Options
#
# | To play with | Set |
# |---|---|
#
# The UCI options a user may need.
#
# ## Credits
#
# Training data, testing, attribution; the license line.

# ----- bench-line
# The commit's last "Bench: N" line must be BENCH, the node count every build
# measured: OpenBench takes a test's bench from that line, and a worker whose
# build counts another number refuses the test. A commit without the line (a
# merge of main into a branch) gets a warning. The message is read through the
# API, so the job needs only this script.
cmd_bench_line() {
    local sha=${1:-} bench=${2:-} msg line n
    [ -n "$sha" ] && [ -n "$bench" ] || { echo "usage: ci.sh bench-line SHA BENCH" >&2; exit 2; }
    msg=$(gh api "repos/$GITHUB_REPOSITORY/commits/$sha" --jq .commit.message) || { echo "cannot read commit $sha" >&2; exit 1; }
    line=$(printf '%s\n' "$msg" | tr -d '\r' | grep -E '^Bench: [0-9]+ *$' | tail -n 1 || true)
    if [ -z "$line" ]; then
        echo "::warning::commit ${sha:0:8} has no 'Bench: N' line; the builds count $bench"
        return 0
    fi
    n=${line#Bench: }
    n=${n%% *}
    echo "commit ${sha:0:8}: Bench: $n; the builds count $bench"
    if [ "$n" != "$bench" ]; then
        echo "::error::commit ${sha:0:8} says Bench: $n, but the builds count $bench (the last Bench line is the one read)"
        exit 1
    fi
}

# ----- deps (Python: reads the ELF, PE or Mach-O headers)
cmd_deps() {
    local py=python3
    [ "${RUNNER_OS:-}" = Windows ] && py=python
    "$py" - "$@" <<'PY'
# Check what a release binary needs from the system, read from its headers,
# against what the release notes promise for it: the System column of the
# Binaries table (the release notes' format, in ci.sh).
#
#   ci.sh deps BINARY ASSET
#
# ASSET is the binary's release name: luminex-linux-x86-64,
# luminex-windows-x86-64.exe or luminex-macos-arm64. A binary that needs more
# than its row says fails the build. v6.6.0's first binaries needed macOS 26,
# as clang builds for the runner's macOS without a deployment target, and on
# Windows the Visual C++ Redistributable, for the DLL runtime; their notes
# said neither. A limit here and its row in the notes change together.
import struct
import sys

# Linux: the newest symbol version each library may be asked for. The Linux
# binary is built on Ubuntu 22.04 by clang 23 with LLVM's libc++ and libc++abi
# and libgcc linked in, so it needs glibc 2.35, Ubuntu 22.04's, and none of
# the C++ runtime libraries (libstdc++ too, for a build that took it).
LINUX = {"GLIBC": (2, 35)}
LINUX_NO_LIBS = ("libstdc++.so.6", "libgcc_s.so.1", "libc++.so.1", "libc++abi.so.1", "libunwind.so.1")
# macOS 13.3, CMakeLists.txt's deployment target: the first with std::to_chars
# for floating point, which std::format needs.
MACOS_MIN = (13, 3, 0)
# Windows: no DLL runtime, so nothing to install.
WINDOWS_NO_DLLS = ("msvcp", "vcruntime", "concrt", "ucrtbase", "api-ms-win-crt-")


class Binary:
    def __init__(self, path):
        self.f = open(path, "rb")

    def read(self, off, n):
        self.f.seek(off)
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError(f"short read at {off}")
        return b

    def cstr(self, off):
        out = b""
        while True:
            self.f.seek(off)
            chunk = self.f.read(256)
            if not chunk:
                raise ValueError(f"unterminated string at {off}")
            j = chunk.find(b"\0")
            if j >= 0:
                return (out + chunk[:j]).decode()
            out += chunk
            off += len(chunk)


def version(s):
    """'GLIBC_2.38' -> ('GLIBC', (2, 38)); None for names like GLIBC_PRIVATE."""
    prefix, _, num = s.rpartition("_")
    parts = num.split(".")
    if not prefix or not all(p.isdigit() for p in parts):
        return None
    return prefix, tuple(int(p) for p in parts)


def elf_needs(b):
    h = b.read(0, 64)
    if h[4] != 2 or h[5] != 1:
        raise ValueError("not a 64-bit little-endian ELF")
    phoff, = struct.unpack_from("<Q", h, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", h, 0x36)
    loads, dyn = [], None
    for i in range(phnum):
        ptype, _, off, vaddr, _, filesz, _, _ = struct.unpack(
            "<IIQQQQQQ", b.read(phoff + i * phentsize, 56))
        if ptype == 1:
            loads.append((vaddr, off, filesz))
        elif ptype == 2:
            dyn = (off, filesz)
    if dyn is None:
        return [], {}

    def to_off(v):
        for vaddr, off, filesz in loads:
            if vaddr <= v < vaddr + filesz:
                return v - vaddr + off
        raise ValueError(f"address {v:#x} outside the file")

    tags, needed = {}, []
    raw = b.read(*dyn)
    for k in range(0, len(raw), 16):
        tag, val = struct.unpack_from("<qQ", raw, k)
        if tag == 0:
            break
        if tag == 1:
            needed.append(val)
        else:
            tags[tag] = val
    strtab = to_off(tags[5])
    libs = [b.cstr(strtab + n) for n in needed]
    versions = {}
    if 0x6FFFFFFE in tags:
        vn = to_off(tags[0x6FFFFFFE])
        for _ in range(tags.get(0x6FFFFFFF, 0)):
            _, cnt, file, aux, nxt = struct.unpack("<HHIII", b.read(vn, 16))
            lib = b.cstr(strtab + file)
            a = vn + aux
            for _ in range(cnt):
                _, _, _, name, anext = struct.unpack("<IHHII", b.read(a, 16))
                versions.setdefault(lib, set()).add(b.cstr(strtab + name))
                a += anext
            if nxt == 0:
                break
            vn += nxt
    return libs, versions


def check_linux(b):
    libs, versions = elf_needs(b)
    limits = LINUX
    errors, newest = [], {}
    for lib, names in versions.items():
        for name in names:
            v = version(name)
            if v is None:
                continue
            prefix, num = v
            if num > newest.get(prefix, ()):
                newest[prefix] = num
            if prefix in limits and num > limits[prefix]:
                errors.append(f"{lib} {name}, newer than {prefix}_{'.'.join(map(str, limits[prefix]))}")
    errors += [f"links {lib}, which this build carries inside" for lib in libs if lib in LINUX_NO_LIBS]
    summary = ", ".join(f"{p}_{'.'.join(map(str, n))}" for p, n in sorted(newest.items()))
    return f"libraries {', '.join(libs)}; newest versions {summary}", errors


def check_macos(b):
    magic, = struct.unpack("<I", b.read(0, 4))
    if magic != 0xFEEDFACF:
        raise ValueError(f"not a thin 64-bit Mach-O (magic {magic:#x})")
    ncmds, sizeofcmds = struct.unpack("<II", b.read(16, 8))
    lc = b.read(32, sizeofcmds)
    off, minos, dylibs = 0, None, []
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", lc, off)
        if cmd == 0x32:  # LC_BUILD_VERSION
            v, = struct.unpack_from("<I", lc, off + 12)
            minos = (v >> 16, (v >> 8) & 0xFF, v & 0xFF)
        elif cmd == 0x24:  # LC_VERSION_MIN_MACOSX
            v, = struct.unpack_from("<I", lc, off + 8)
            minos = (v >> 16, (v >> 8) & 0xFF, v & 0xFF)
        elif cmd in (0xC, 0x80000018):  # LC_LOAD_DYLIB, LC_LOAD_WEAK_DYLIB
            noff, = struct.unpack_from("<I", lc, off + 8)
            dylibs.append(lc[off + noff:off + size].split(b"\0")[0].decode())
        off += size
    if minos is None:
        return "no minimum macOS version in the load commands", ["no LC_BUILD_VERSION"]
    shown = ".".join(map(str, minos))
    errors = []
    if minos > MACOS_MIN:
        errors.append(f"needs macOS {shown}, newer than {'.'.join(map(str, MACOS_MIN))}")
    return f"macOS {shown} or newer; dylibs {', '.join(dylibs)}", errors


def check_windows(b):
    lfanew, = struct.unpack("<I", b.read(0x3C, 4))
    coff = b.read(lfanew, 24)
    if coff[:4] != b"PE\0\0":
        raise ValueError("not a PE file")
    nsec, = struct.unpack_from("<H", coff, 6)
    optsize, = struct.unpack_from("<H", coff, 20)
    opt = b.read(lfanew + 24, optsize)
    if struct.unpack_from("<H", opt, 0)[0] != 0x20B:
        raise ValueError("not PE32+")
    dirs = {i: struct.unpack_from("<II", opt, 112 + 8 * i) for i in (1, 13)}
    table = b.read(lfanew + 24 + optsize, 40 * nsec)
    sections = [struct.unpack_from("<8sIIII", table, 40 * i)[1:] for i in range(nsec)]

    def to_off(rva):
        for vsize, va, rawsize, rawptr in sections:
            if va <= rva < va + max(vsize, rawsize):
                return rva - va + rawptr
        raise ValueError(f"RVA {rva:#x} outside the sections")

    dlls = []
    for index, size, name_at in ((1, 20, 12), (13, 32, 4)):  # imports, delay imports
        rva = dirs[index][0]
        if not rva:
            continue
        d = to_off(rva)
        while True:
            entry = b.read(d, size)
            if entry == b"\0" * size:
                break
            name_rva, = struct.unpack_from("<I", entry, name_at)
            dlls.append(b.cstr(to_off(name_rva)))
            d += size
    errors = [f"imports {dll}, the DLL runtime (install the Visual C++ Redistributable)"
              for dll in dlls if dll.lower().startswith(WINDOWS_NO_DLLS)]
    return f"imports {', '.join(dlls)}", errors


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: ci.sh deps BINARY ASSET")
    path, asset = sys.argv[1], sys.argv[2]
    b = Binary(path)
    if asset.startswith("luminex-linux-"):
        summary, errors = check_linux(b)
    elif asset.startswith("luminex-macos-"):
        summary, errors = check_macos(b)
    elif asset.startswith("luminex-windows-"):
        summary, errors = check_windows(b)
    else:
        sys.exit(f"{asset}: not a release binary's name")
    print(f"{asset}: {summary}")
    for e in errors:
        print(f"{asset}: {e}", file=sys.stderr)
    if errors:
        sys.exit(f"{asset} needs more from the system than the release notes promise")


if __name__ == "__main__":
    main()
PY
}

cmd=${1:-}
shift || true
case "$cmd" in
    clang|fetch-net|isa|name|net|simd|sde|dev-release|deps|bench-line) "cmd_${cmd//-/_}" "$@" ;;
    *) sed -n '2,15p' "$0" >&2; exit 2 ;;
esac
