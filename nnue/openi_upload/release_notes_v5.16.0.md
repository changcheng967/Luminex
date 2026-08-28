# Luminex v5.16.0

**Distilled evaluation (fit6) — the handcrafted basis refit against 22M Stockfish search scores — plus a 256 MB transposition table. ≈ +42 Elo over v5.15.0.**

## What's New
- **Distilled evaluation, fit5 → fit6** (+43 Elo vs the v5.15.0 eval): the handcrafted feature basis was refit by anchored ridge regression against 22.1M Stockfish search scores from real engine games (targets tail-clipped at ±1000 cp, rank-1 gauge anchors pinning effective piece values, λ=3e7). Adopted in [`8656041`](https://github.com/changcheng967/Luminex/commit/8656041) (fit5: +98.4 ± 28.5, LOS 100%, 500-game A/B vs the pre-fit baseline) and [`95ec6b4`](https://github.com/changcheng967/Luminex/commit/95ec6b4) (fit6: +12 vs fit5 over 1000 games; first >50% score vs Stash V20 at 0.504).
- **Transposition table default 128 MB → 256 MB** ([`c5637e7`](https://github.com/changcheng967/Luminex/commit/c5637e7)): self-play A/B +16.7 ± 26.7 (LOS 89%); ladder confirm vs Stash V20 improved 0.504 → 0.509. The 128 MB default was an unmeasured carry-over assumption from v5.15.0.
- **Sharp-root time extension** ([`59c1afe`](https://github.com/changcheng967/Luminex/commit/59c1afe)): when the root aspiration window must be re-widened (score dropped outside the window), the search extends its time budget instead of stopping on the stability heuristic. SPRT +3.7 at a 12,000-game cap (LLR +0.62 — small positive, kept).
- **Evaluation basis prepared for future fits** (zero-weight, behavior identical): per-file pawn-shield resolution, imbalance-polynomial features, overloaded-defender flags ([`54c7b10`](https://github.com/changcheng967/Luminex/commit/54c7b10)).

## Test Results
| Opponent | Time Control | Games | W–D–L | Score | Est. Elo |
|---|---|---|---|---|---|
| Stash V20 (~2509), fit6 + 256 MB TT | 1+0.01 bullet | 500 | 233–224–43 | 50.9% | **~2512** |
| Stash V20 (~2509), fit6 only | 1+0.01 bullet | 500 | 232–228–40 | 50.4% | ~2509 |

## Estimated Strength
~2512 Elo (blitz) — ≈ Stash V20 (2509) + 3; **+42 over v5.15.0** (~2470).

## Binaries
| Platform | File |
|---|---|
| Windows x64 (AVX2/BMI2) | `luminex-windows-x86-64-modern.exe` |
| Windows x64 | `luminex-windows-x86-64.exe` |
| Linux x64 (AVX2) | `luminex-linux-x86-64-modern` |
| Linux x64 | `luminex-linux-x86-64` |
| macOS Apple Silicon | `luminex-macos-arm64` |

## Usage
UCI engine. Default is the handcrafted distilled evaluation (HCE):
```
position startpos
go depth 20
```
Optional NNUE (experimental): set `option UseNNUE true` and `option NNUEFile <path-to-net>`.

*GPL-3.0, with Stockfish attribution.*
