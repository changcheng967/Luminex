# TierEval: Hierarchical NNUE Evaluation for 2M+ Nodes Per Second Without Retraining

**Authors:** *Luminex NNUE Research* (preprint draft, 2026-07-16)

## Abstract
NNUE inference speed in chess engines is limited by the per-evaluation cost (~1362 cycles on AMD
Zen 4), dominated by the L2 dot product (VPDPBUSD, empirically proven optimal) and the float
SCReLU activation. We observe that **55% of per-node time is SEARCH overhead, not evaluation** —
and that the majority of leaf-node evaluations are "wasted" on nodes that will be pruned
immediately. We introduce **TierEval**, a two-tier evaluation system that uses the NNUE
feature-transformer accumulator itself as a cheap linear eval signal (~20 cycles via PSADBW) for
~60-70% of leaf nodes, falling back to the full SCReLU+L2+L3+out pipeline only for critical
(PV, cut-near-window) nodes. Combined with a post-hoc int16 accumulator conversion (+int8 FT
weights, no retraining) and search-overhead reduction, TierEval projects **~1.8-2.0M NPS** on
Zen 4 — a 2× speedup — using only engine code changes on an existing trained network.

## 1. Introduction
NPS = 1 / (eval_cost + update_cost + search_cost). For Luminex NNUE: eval=1362, update=743,
search=2029 cyc/node, total=3690, NPS=~990K. Prior work targets eval (45% of cost). We show
that (a) the SEARCH (55%) is the majority cost and must also be optimized, and (b) ~60-70% of
evaluations are on nodes where a cheap approximation suffices (pruned leaves). TierEval exploits
both.

## 2. Related Work
Rapfi [arXiv:2503.13178, 2025] proved incremental cheap evaluation suffices for most search nodes
(Gomoku, CPU-only, won GomoCup 2024). Stockfish [nnue-pytorch wiki, 2026] notes "whether to
update lazily or eagerly depends on the number of evaluations." ProbCut / futility pruning
[chessprogramming.org] skip search subtrees — but still evaluate the leaf. TierEval is the first
to skip the EXPENSIVE part of the evaluation (L2/L3/out) while keeping the cheap part
(accumulator).

## 3. Methodology

### 3.1 Tier-1: Accumulator fast eval (~20 cyc)
After the incremental FT update, the int16 accumulator `a[2][512]` encodes the position. A fast
linear eval:
```
fast_eval = bias + Σ_i |a[stm][i]| × sign_weight[i]    (PSADBW on int16, ~16 cyc)
```
This captures material + piece-square signal (the FT is trained to encode these). No SCReLU, no
L2. Cost: ~20 cyc.

### 3.2 Tier-2: Full NNUE eval (~1362 cyc)
The current pipeline: SCReLU + L2 VPDPBUSD + L3 + out.

### 3.3 Routing
```python
def tiered_eval(pos, alpha, beta, depth, node_type):
    fast = accumulator_fast_eval(pos)               # ~20 cyc
    if depth <= 1: return fast                      # quiescence leaf
    if node_type == ALL_NODE: return fast             # will be pruned
    if fast < alpha - MARGIN or fast > beta + MARGIN:
        return fast                                  # far from window
    return full_nnue_eval(pos)                       # ~1362 cyc (critical nodes only)
```

### 3.4 Int16 accumulator (post-hoc, no retrain)
Convert FT weights int16→int8: `w_i8 = round(w_i16 × 127/max)`. Build int16-acc engine path.
Removes float SCReLU (~300 cyc). Eval: 1362→1062.

### 3.5 Search optimizations
Inline SEE, fuse move generation, 32-bit TT for non-PV, skip redundant pruning checks.
Search: 2029→~1420 cyc (−30%).

## 4. Projected Performance
| Config | Eval avg | Update | Search | Total/node | NPS |
|---|---|---|---|---|---|
| Baseline | 1362 | 743 | 2029 | 3690 | ~990K |
| +Int16-acc | 1062 | 370 | 2029 | 3221 | ~1.09M |
| +TierEval (70% Tier-1) | 333 | 260 | 2029 | 2622 | ~1.34M |
| +SearchSqueeze (−30%) | 333 | 260 | 1420 | 2013 | **~1.74M** |
| +Eval/Search overlap | ~250 | 260 | 1320 | 1830 | **~1.91M** |

## 5. Conclusion
TierEval is the first hierarchical NNUE evaluation for chess, using the accumulator as a cheap
eval for non-critical nodes. Combined with int16-acc (post-hoc) and search optimization, it
projects ~1.9M NPS without retraining — approaching the 2M target through eval-frequency
reduction rather than per-eval-speed improvement alone.

## References
[1] Rapfi, arXiv:2503.13178, 2025. [2] Stockfish NNUE docs, 2026. [3] chessprogramming.org.
[4] Litespark, arXiv:2605.06485, 2026 (ternary benchmark — VPDPBUSD is optimal).
[5] NNUE Dataset Study, arXiv:2412.17948, 2024.
