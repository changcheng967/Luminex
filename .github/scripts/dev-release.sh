#!/usr/bin/env bash
# Publish the rolling development build. The binaries in DIR, built and tested
# by build.yml's jobs, become the pre-release luminex-$BUILD_ID; then every
# older luminex-dev release and tag is deleted, so one development build stands
# next to the stable releases. Only tags of the exact form
# luminex-dev-YYYYMMDD-<8 hex digits> are deleted: the v* releases and
# engine-archives (OpenBench's store of old builds and networks) never match.
# DRY_RUN=1, the default, prints the notes, the assets and the deletions
# without making them.
#
#   GH_TOKEN=... BUILD_ID=dev-20261112-d4774f1a [DRY_RUN=0] dev-release.sh DIR
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

# The network the binaries carry, as fetch-net.sh pins it: its file name on
# engine-archives, its OpenBench id (the first 8 hex digits of its sha256) and
# its about line. fetch-net.sh saved it under the engine's default file name;
# it is attached under its own name, so the notes and the file say which
# network it is.
here=$(dirname "$0")
net_name=$(sed -n 's#^url=.*/##p' "$here/fetch-net.sh")
net_about=$(sed -n 's#^about="\(.*\)"$#\1#p' "$here/fetch-net.sh")
[ -n "$net_name" ] || { echo "no url= line in fetch-net.sh" >&2; exit 1; }
[ "$net_name" = luminex_sfq_i8.nnue ] || cp "$dir/luminex_sfq_i8.nnue" "$dir/$net_name"
net_id=$(sha256sum "$dir/$net_name" | cut -c1-8 | tr a-f A-F)
if [ -n "$net_about" ]; then
  net_line="the network ${net_about%%: *} (OpenBench id \`$net_id\`): ${net_about#*: }."
else
  net_line="the network \`$net_name\` (OpenBench id \`$net_id\`)."
fi

# Every build job's binary, and the network on its own
assets=(luminex-linux-x86-64-modern luminex-linux-x86-64
        luminex-linux-x86-64-glibc235-modern luminex-linux-x86-64-glibc235
        luminex-windows-x86-64-modern.exe luminex-windows-x86-64.exe
        luminex-macos-arm64 "$net_name")
files=()
for a in "${assets[@]}"; do
  [ -f "$dir/$a" ] || { echo "missing $dir/$a" >&2; exit 1; }
  files+=("$dir/$a")
done
chmod +x "$dir/luminex-linux-x86-64"
bash "$here/check-name.sh" "$dir/luminex-linux-x86-64" "$BUILD_ID"

# The notes cover everything since the latest stable release, since each push
# to main replaces the previous development build. Commits are split into
# those that change the engine binary (its sources, or the network it carries)
# and those that do not (training, tools, CI, docs), each linked to its pull
# request.
stable=$(gh release view --repo "$repo" --json tagName --jq .tagName)
mapfile -t dev_tags < <(gh release list --repo "$repo" --limit 100 --json tagName --jq '.[].tagName' \
                        | grep -E "$pattern" | grep -vx "$tag" || true)
base=$(git rev-parse "refs/tags/$stable^{commit}")
tool_only='^src/(eval_trace\.cpp|featurize\.cpp|frame_merge\.cpp|lc0pack\.cpp|lc0_move_table\.h|verify_frames\.c)$'
engine_lines=()
other_lines=()
while read -r h; do
  line="* [\`${h:0:8}\`]($url/commit/$h) $(git show -s --format=%s "$h")"
  pr=$(gh api "repos/$repo/commits/$h/pulls" --jq '[.[] | select(.merged_at != null)][0].number // empty' 2> /dev/null || true)
  [ -z "$pr" ] || line+=" ([#$pr]($url/pull/$pr))"
  engine=$(git show --name-only --format= "$h" | grep -E '^src/|^\.github/scripts/fetch-net\.sh$' | grep -vE "$tool_only" || true)
  if [ -n "$engine" ]; then engine_lines+=("$line"); else other_lines+=("$line"); fi
done < <(git rev-list --no-merges "$base..$sha")
count=$(( ${#engine_lines[@]} + ${#other_lines[@]} ))
[ "$count" = 1 ] && commits="1 commit" || commits="$count commits"
when=$(TZ=UTC git show -s --format=%cd --date=format-local:'%d %B %Y' "$sha")
when=${when#0}
stable_net=$(git show "$stable:.github/scripts/fetch-net.sh" 2> /dev/null \
             | sed -n 's/^sha256=\([0-9a-f]\{8\}\).*/\1/p' | tr a-f A-F || true)
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
  echo "| Platform | Binary | CPU |"
  echo "|----------|--------|-----|"
  echo "| Linux | \`luminex-linux-x86-64-modern\` | AVX2 (x86-64-v3) |"
  echo "| Linux | \`luminex-linux-x86-64\` | SSE4.2 and POPCNT (x86-64-v2) |"
  echo "| Linux, glibc 2.35 (Ubuntu 22.04) | \`luminex-linux-x86-64-glibc235-modern\` | AVX2 (x86-64-v3) |"
  echo "| Linux, glibc 2.35 (Ubuntu 22.04) | \`luminex-linux-x86-64-glibc235\` | SSE4.2 and POPCNT (x86-64-v2) |"
  echo "| Windows | \`luminex-windows-x86-64-modern.exe\` | AVX2 |"
  echo "| Windows | \`luminex-windows-x86-64.exe\` | POPCNT, no AVX2 needed |"
  echo "| macOS | \`luminex-macos-arm64\` | Apple silicon |"
  echo
  echo "The \`-modern\` builds need AVX2, which most x86-64 processors since 2015 have; on an older one take the plain build. Each binary is about 76 MB, the network included."
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
