# Release Notes Template

> **Bold one line:** the Elo over the previous release and the rating-list
> estimate. Under it, one plain sentence on what the release brings.

## What's new

One bold line per area (network, time management, search, speed, ...) with
its PR links, then short plain bullets: what changed, a sentence each. No
figures here; they go in Measured gains.

## Strength

| Match | Games | W-D-L | Result |
|---|---:|---:|---|
| vX vs the previous release (test N) | N | W-D-L | **Elo ± 95%** |
| The gauntlet (gauntlet N) | N | W-D-L | **rating ±** per list |

One line of conditions (threads, hash, time control, openings), then any
caveat. The results against each opponent go in a collapsed `<details>` table.

## Measured gains

| Change | Elo | PR | Test |
|---|---:|:---:|:---:|

One row per SPRT, grouped under bold rows by area. Quote each Elo as its
OpenBench test page shows it, with the 95% interval, and link the page:
commit messages keep the figure of their day, which drifts when OpenBench's
error model changes (v6.6.0's first notes quoted several that had). Mark a
non-regression test, and a test stopped undecided.

## Downloads

| Platform | Binary | CPU | Needs |
|---|---|---|---|

Needs is what the binary needs from the system: glibc and libstdc++ on Linux,
any runtime to install on Windows, the oldest macOS. check-deps.py reads it
from every binary's headers in CI and fails a build that needs more than its
row says, so a changed limit changes the row.

Every asset carries its real name: the network is attached as
`luminex_<architecture>_<NN>_i8.nnue`, the name the binaries report, never
under a past network's name or a neutral one. check-net.sh fails a build that
reports any other.

## Options

| To play with | Set |
|---|---|

The UCI options a user may need.

## Credits

Training data, testing, attribution; the license line.
