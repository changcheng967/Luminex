# HalfKA / HalfKAv2 (Stockfish 13-14, 2020-21)
> Lead: Tomasz Sobczyk (Sopel) | Confidence: CPW

## Feature Set
- HalfKA: 12x64x64 = 45,056/side, vertical flip (not rotation)
- HalfKAv2 (SF14, Jul 2021): removes king-square redundancy -> 11x64x64
- halfka features: piece ANY square incl. king position identity

## Architecture Graph
```
HalfKAv2 45056x2 -> [FT 1024x2? -> v2 shipped 2x520-era] 
  -> concat + 8x2 PSQT taps direct to output (unbalanced-material learning)
  -> eight 512x2 -> 16 -> 32 -> 1 output SUBNETS by piece-count bucket (0-7)
```

## Signature Innovations
- PSQT taps from FT to output (linear bypass, born here)
- OUTPUT BUCKETS -> material-count subnets (LayerStacks) — the capacity-free trick
- nnue-pytorch trainer by Gary Linscott; Lc0 data collaboration (Feb 2021)
