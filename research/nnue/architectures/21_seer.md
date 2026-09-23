# Seer (Connor McMonigle)
> Confidence: verified-from-source (seer-training read this session)

## Architecture (from model.py)
```
HalfKA -> [FactoredBlock: sparse features PROJECTED THROUGH AN INTERMEDIATE
           FACTORIZATION (feature -> inter_dim -> output) with a learned
           sparse conversion matrix] -> accumulator
  -> SCReLU tail
```

## Signature Innovations
- FACTORIZED feature transformer: instead of dense per-feature rows, features
  share parameters through an intermediate factor space. The most structurally
  original FT in the ecosystem.
- Loss (from source): fixed λ=0.6 sigmoid-MSE eval/outcome mix
- Loss-space survey reference (this session): sigmoid-MSE family
