# SFNNv16-class (Stockfish 19, 2026)
> Confidence: cross-validated (session benchmarks + 3-agent proposal agreement)

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
