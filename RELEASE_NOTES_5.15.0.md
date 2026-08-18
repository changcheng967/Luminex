# Luminex v5.15.0

**Handcrafted-evaluation refinements — sharper passed-pawn scoring and a trade-down bonus.**

## What's New
- **Passed-pawn path decomposition** (+15 Elo): passed pawns are now scored by decomposing the path to promotion into *blockade*, *safe-advance*, and *clear-path* components instead of a single flat bonus.
- **Trade-down bonus** (+12 Elo): the winning side gains a bonus as non-pawn material comes off the board — converting a material advantage into won endgames.
- **Transposition table tuned to 128 MB** default (less L2/L3 cache pressure at bullet time controls).
- **C++23 modernization**: `std::format`, `std::to_underlying`, defaulted `operator==`, `[[assume]]`, `std::byteswap`.
- **NNUE evaluation (experimental)**: an NNUE inference path (HalfKAv2 features, AVX-512/VNNI) is available via `UseNNUE`. It is competitive with HCE at equal depth; **HCE remains the default and recommended path.**

## Test Results
| Opponent | Time Control | Games | W–D–L | Score | Est. Elo |
|---|---|---|---|---|---|
| Stash V20 (~2509) | 1+0.01 bullet | 200 | 80–18–102 | 44.5% | **~2470** |

## Estimated Strength
~2470 Elo (blitz) — ≈ Stash V20 (2509) − 40.

## Binaries
| Platform | File |
|---|---|
| Windows x64 | `luminex-5.15.0-windows-x64.exe` |
| Linux x64 | `luminex-5.15.0-linux-x64` |

## Usage
UCI engine. Default is handcrafted evaluation (HCE):
```
position startpos
go depth 20
```
Optional NNUE (experimental): set `option UseNNUE true` and `option NNUEFile <path-to-net>`.

*GPL-3.0, with Stockfish attribution.*
