# Berserk (Jay Honnold)
> Confidence: verified-from-source (berserk-trainer read this session) + CPW

## Architecture (from trainer source)
```
HalfKA features -> per-perspective accumulator (N_HIDDEN) 
  -> output = outBias + DotProduct(accumulator, outWeights)   [very shallow tail]
```
- Loss (from source): `λ=0.6 · (σ(score)−σ(pred))² + 0.4 · (result−σ(pred))²`
  — sigmoid-space MSE with FIXED outcome mixing. Family: SF/Berserk sigmoid-space.

## Signature Innovations
- Minimalist tail (dot-product output) — proves FT does the heavy lifting
- Cross-engine loss survey reference point (this session)
- Berserk 14: CEDR ~3713, top-10 engine
