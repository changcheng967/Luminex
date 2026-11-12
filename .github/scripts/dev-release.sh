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

# Every build job's binary, and the network on its own
assets=(luminex-linux-x86-64-modern luminex-linux-x86-64
        luminex-linux-x86-64-glibc235-modern luminex-linux-x86-64-glibc235
        luminex-windows-x86-64-modern.exe luminex-windows-x86-64.exe
        luminex-macos-arm64 luminex_sfq_i8.nnue)
files=()
for a in "${assets[@]}"; do
  [ -f "$dir/$a" ] || { echo "missing $dir/$a" >&2; exit 1; }
  files+=("$dir/$a")
done
chmod +x "$dir/luminex-linux-x86-64"
bash "$(dirname "$0")/check-name.sh" "$dir/luminex-linux-x86-64" "$BUILD_ID"

# The notes list the commits since the previous development build, or since
# the latest stable release when there is none
stable=$(gh release view --repo "$repo" --json tagName --jq .tagName)
mapfile -t dev_tags < <(gh release list --repo "$repo" --limit 100 --json tagName --jq '.[].tagName' \
                        | grep -E "$pattern" | grep -vx "$tag" || true)
since=$stable
label="since $stable"
if [ ${#dev_tags[@]} -gt 0 ] && git rev-parse -q --verify "refs/tags/${dev_tags[0]}^{commit}" > /dev/null; then
  since=${dev_tags[0]}
  label="since the previous development build (${dev_tags[0]#luminex-})"
fi
base=$(git rev-parse "refs/tags/$since^{commit}")
count=$(git rev-list --count --no-merges "$base..$sha")
{
  echo "Precompiled binaries of the latest Luminex development version, built automatically from every push to \`main\` that passes CI."
  echo
  echo "> [!NOTE]"
  echo "> For stable, thoroughly tested builds use the [official releases]($url/releases/latest) (newest: [$stable]($url/releases/tag/$stable)). Development builds may contain bugs."
  echo
  echo "Built from [\`${sha:0:8}\`]($url/commit/$sha); the engine reports itself as \`Luminex $BUILD_ID\`. Each binary carries the network \`luminex_sfq_i8.nnue\`, which is also attached on its own."
  echo
  echo "## Commits $label"
  echo
  git log --no-merges --format="* [\`%h\`]($url/commit/%H) %s" --abbrev=8 -n 30 "$base..$sha"
  [ "$count" -le 30 ] || echo "* and $((count - 30)) more"
  echo
  echo "[All commits since $stable]($url/compare/$stable...$sha)"
  echo
  echo "## Binaries"
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
