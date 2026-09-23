# Stockfish NNUE Evolution: SFNNv1 → SFNNv16
> Confidence: v1-v2 CPW-verified; v3+ cross-validated from nnue-pytorch ecosystem +
> this session's three-agent proposal cross-check. Items marked ⚠ are community-timeline,
> treat exact version boundaries as approximate.

| Gen | Era | Feature set | Width | Tail | Act | Key change |
|---|---|---|---|---|---|---|
| v1 | SF12 2020 | HalfKP 41024x2 + factorizer | 256x2 | 512→32→32→1 | CReLU | birth (Nodchip port, +80 Elo fishtest-confirmed) |
| ⚠v2-v3 | SF13 | HalfKA vertical flip | 256→1024? | widen | CReLU | flip-not-rotate; hybrid→full NNUE |
| v2' | SF14 Jul 2021 | HalfKAv2 (no king redundancy) | 1024→2x520-era | 16→32→1 ×8 buckets | CReLU | PSQT taps + material-count OUTPUT SUBNETS |
| ⚠v4-v5 | SF15 | HalfKAv2_hm (horizontal mirror, halves king space) | 1536x2 | 16→16→1 | CReLU→SCReLU (SF15-16 era) | mirroring; small/big dual-net era begins (SF16.1) |
| ⚠v6-v9 | SF16-17 | hm + increasing L2 | 1536x2 | 15→15→1 / 16→16→1 | SCReLU | king input buckets; factorizer maturing |
| v10 | SF17.x | + FullThreats (PieceSq-PieceSq attack pairs) | 1536x2 | 16→16→1 | SCReLU | THREAT INPUTS — net sees attack geometry |
| ⚠v11-v12 | | threat pruning iterations | | | | remove king-piece threat subsets |
| v13 | | threats mature -> accumulator SHRINKS | reduced | L2 16→32 | SCReLU | the key trade: features buy width back |
| ⚠v14-v15 | SF18 | more layer-stack complexity | | | | tail depth experiments |
| v16 | SF19 2026 | threats pruned to non-redundant + PAWN-PAIR features (adjacent-file) | ~1536x2 | 16-32 | SCReLU + QAT | pawn-pairs supersede pawn threat pairs; small net RETIRED; QAT standard |

## The arc in one sentence
Every generation after v2 was an **update-path-cost reduction campaign** (prune features
-> afford width/L2 -> prune again) — EXCEPT the capacity jumps, which were always
purchased with more Lc0 data first.

## Milestone side-facts
- Hybrid patch Aug 2020 (+~20 Elo); HCE fully removed SF16 (2023)
- Dual small/big nets SF16.1+ (big net for deep, small for speed), retired in v16 era
- Bullet trainer (Jamie Whiting, Rust) became the community standard ~2025
- Our session verified SF19 on our silicon: 622K nps, 1536-class width
