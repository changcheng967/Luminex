# SFNNv1 — HalfKP (Stockfish 12, Sept 2020)
> Confidence: CPW (verified snapshot)

## Feature Set
- HalfKP: per side, (own king square 64) × (non-king piece-square 640+1) = 41,024 inputs/side
- Factorizer: virtual features (sub-network decomposition) to aid training

## Architecture Graph
```
41024x2 sparse -> [FT 256x2 accumulator, int16] -> concat 512
              -> 512x32 -> 32x32 -> 32x1 (CReLU-ish clipped activations)
```

## Signature Innovations
- Factorizer shadow features; dual perspective concat (stm-first); hybrid era
  (NNUE only for balanced material, +20 Elo) started Aug 2020, full default later
- Rotation (xor 63) instead of vertical flip — a Shogi relic, later removed
