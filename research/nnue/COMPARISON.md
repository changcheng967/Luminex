# Comparison Matrix (session-verified where marked)

| Engine | Features | Width | Tail | Act | Loss/space | Verified |
|---|---|---|---|---|---|---|
| Nasu 2018 | king-indexed Shogi | small | tiny | — | — | CPW |
| SF12 v1 | HalfKP+factorizer | 256x2 | 32→32→1 | CReLU | — | CPW |
| SF14 | HalfKAv2+PSQT taps | ~1024x2 | 16→32→1 x8 buckets | CReLU | — | CPW |
| SF19 v16-class | hm+threats+pawn-pairs | 1536x2 | 16-16-1 | SCReLU+QAT | pow~2.6 σ-space λ | session+cross |
| Berserk | HalfKA | N_HIDDEN | DOT ONLY | σ(WINV) | λ=0.6 σ-MSE | source✓ |
| Seer | HalfKA FACTORIZED | — | SCReLU | σ-MSE λ=0.6 | source✓ |
| v12p2 (ours) | HalfKAv2_hg | 512x2 | 16→32→1 | SCReLU | pow2.6 σ-space | source✓ |
| HCE (ours) | hand-crafted | — | — | — | — | 1.88M nps✓ |
| Leela | 112-plane | 40+ResNet SE | policy+value | — | WDL CE | local clone✓ |

## Where Luminex sits
- Feature set: SF14-era (pre-threat) — one generation behind SF19
- Width: SF12-era 512 — two generations behind
- Loss: SF-class (pow-2.6 σ-space) — current
- Kernels: SF-class or better (int16 saturating + fused + batched L3 — our own)
- Data: 4.29B Leela labels, fat-tail preserved — quality-competitive
- The Luminex-Gen v1.2 plan = close the feature/width gap with SF-current
  while keeping our kernel edge
