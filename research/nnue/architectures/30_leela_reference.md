# Leela Chess Zero (paradigm reference, NOT NNUE)
> Confidence: verified (local clone + session data pipeline built on its data)

```
112-plane input (classical/canonical) -> ResNet w/ SE blocks (40+ blocks, 256 filters)
  -> policy head (1858 moves) + value head (WDL) + MLH
```
- GPU-bound: ~1-5K nps CPU-class vs NNUE 600-900K (session-measured SF19=622K)
- Value: TRAINING DATA (our entire 4.29B dataset is Leela deep-search labels)
  + win-rate-model research (SF's double-sigmoid descends from Leela's WDL work)
- Mindset import: verification culture (vdlp/vizvezdenec), data quality absolutism
