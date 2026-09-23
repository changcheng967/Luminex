# SFNNv16-class (Stockfish 19, 2026)
> Confidence: cross-validated + SOURCE-VERIFIED 2026-09-23 (nnue-pytorch master read)

## Source-verified details (2026-09-23)
- ComposedFeatureTransformer: per-block SEPARATE weight tables (rows l1_size+psqt wide),
  SUMMED into one accumulator via shared bias. Threat flips pay full-L1-width rows.
- HalfKAv2_hm KingBuckets: a 32-entry remap table in the feature index — currently 1:1
  in master, but the MECHANISM permits king-square collapsing (within-bucket king moves
  would be index-stable = no refresh). An untapped knob for engines without Finny hit-rate.
- PSQT bypass = extra FT output columns (num_psqt_buckets), not a separate layer.

## Architecture Graph
```
HalfKAv2_hm 24576/side + threats(non-redundant) + pawn-pairs(adjacent file)
   -> [FT 1536x2 int16 saturating accumulator, Finny-cached king buckets]
   -> SCReLU int16->uint8
   -> concat 3072 (+PSQT bypass taps)
   -> 16->16->1 hidden (int8 VNNI) x output buckets by material
```

## Signature Innovations
- Threat inputs then SUBTRACTION of them; pawn-pairs as cheaper superset
- QAT with power-of-2 scales + STE as standard practice
- Finny Tables (Finn Eggers): king-square-keyed accumulator cache
- Measured on our Xeon 2.6GHz: 622K nps single-thread
