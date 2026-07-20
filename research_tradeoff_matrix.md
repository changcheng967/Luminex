# Pushing NNUE Past 2M NPS WITHOUT Retraining — Trade-off & Contradiction Matrix
*2026-07-16. Constraint: existing v2 net (L1=512, float acc, int16 FT) used AS-IS. Only engine code changes.*

## 0. The hard math (why 2M is aggressive without retrain)

From the depth-18 profile: **per-node total = ~3690 cyc = NNUE (45%) + search (55%)**

| Component | cyc | % of node | Reducible without retrain? |
|---|---|---|---|
| L2 dot (VPDPBUSD, 1024→16) | 500 | 14% | ❌ VPDPBUSD floor at L1=512 (proven by ternary benchmark) |
| SCReLU (float muls) | 350 | 9% | ✅ int16 accumulator removes float |
| Update (int16 FT bandwidth) | 743 | 20% | ✅ int8 FT halves bandwidth |
| L3/out + loads | 260 | 7% | ✅ minor layout optimizations |
| Search (movegen, ordering, TT, pruning) | 2029 | 55% | ✅ search-side optimizations |

**To hit 2M NPS at 3.5 GHz: per-node must drop from 3690→1750 cyc (−53%).**
Even with ZERO NNUE cost: search alone = 2029 cyc → **1.72M NPS ceiling.** So **2M requires BOTH NNUE AND search optimization.**

## 1. The levers (all engine-code-only, no retrain)

| Lever | Saves | Mechanism | Risk |
|---|---|---|---|
| **int16-acc + int8-FT** | ~670 cyc/node | Post-hoc convert FT int16→int8; build int16-acc path; removes float SCReLU + halves update bandwidth | FT accuracy loss from int16→int8 (untested; SF does it) |
| **TieredEval** | ~200-400 cyc/node | Use cheap accumulator-only eval for ~60% of nodes; full NNUE only for critical (PV/cut) nodes | Rough eval quality for Tier-1 nodes; search miscalibration |
| **LazyUpdate** | ~150 cyc/node | Defer FT update until eval is actually needed (skip for pruned nodes) | None — pure savings |
| **Search→eval overlap** | ~100-200 cyc/node | Software-pipeline: prefetch accumulator / start L2 while move ordering runs | Complex code restructuring |
| **L2 layout prefetch** | ~50 cyc/node | Prefetch next-node's L2 weights while current eval's L3/out runs | Minor |
| **EvalCache expansion** | ~100 cyc/node | Larger eval cache (128K) → more transposition hits → skip eval entirely | Memory cost |
| **Probcut static guard** | ~150 cyc/node | Skip full eval when accumulator-sum predicts far outside window | Miscalibration risk |

## 2. Maximum achievable NPS (stacking ALL no-retrain levers)

| Optimization stack | Eval cyc | Update cyc | Search cyc | Total/node | NPS (3.5GHz) |
|---|---|---|---|---|---|
| Current (baseline) | 1362 | 743 | 2029 | 3690 | ~990K |
| + int16-acc + int8-FT | 1062 | 370 | 2029 | 3221 | ~1.09M |
| + LazyUpdate | 1062 | 260 | 2029 | 3091 | ~1.13M |
| + TieredEval (60% Tier-1) | 591 | 260 | 2029 | 2740 | ~1.28M |
| + EvalCache 128K | 500 | 260 | 1929 | 2589 | ~1.35M |
| + Search→eval overlap | 400 | 260 | 1929 | 2489 | ~1.41M |
| + L2 prefetch + Probcut | 350 | 260 | 1829 | 2339 | **~1.50M** |

**Maximum without retrain: ~1.5M NPS. To reach 2M: search must ALSO be optimized by ~30%.**

## 3. The structural holes

### Hole A — "The search is 55% of per-node cost but nobody optimizes it for NPS"
Every NPS optimization targets the eval (45%). But search (movegen, ordering, TT probes, pruning
checks) is the MAJORITY cost. Reducing search overhead by 30% (e.g., faster move generation,
cheaper TT probes, reduced pruning checks for nodes that will be cut anyway) would contribute more
to NPS than any eval optimization.

### Hole B — "The accumulator contains untapped eval information"
After the FT update, the int16/float accumulator has 512 values per perspective that ENCODE the
position. Currently, the ONLY use is feeding the L2 dot (expensive). But a CHEAP linear
combination of the accumulator (e.g., weighted sum via PSADBW on int16 → ~20 cyc) could give a
rough eval for TieredEval (Hole: nobody uses the accumulator itself as a cheap eval signal).
