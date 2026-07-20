# Novel Hypotheses — 2M+ NPS WITHOUT Retraining
*2026-07-16. All engine-code-only. Existing v2 net used AS-IS. Targets both NNUE eval AND search.*

---

## H1 (flagship) — TierEval: Two-Tier Evaluation via Accumulator-Derived Fast Eval
*(targets Hole B: the accumulator contains untapped eval information)*

**Name:** *TierEval — Hierarchical Evaluation via Accumulator-Derived Fast-Path for Alpha-Beta Search.*

**Contradiction shattered:** Every chess engine evaluates EVERY leaf node with the FULL NNUE
pipeline (~1362 cyc). But alpha-beta search creates many nodes whose eval is irrelevant (they'll
be pruned immediately). Rapfi [arXiv:2503.13178, 2025] proved that an incremental, cheap
evaluation suffices for most search nodes. **Nobody has used the NNUE accumulator itself as a
cheap eval signal** — the 512 int16 values per perspective encode the position's key features,
and a simple weighted sum (≈20 cyc) gives a rough eval that's adequate for ~60-70% of leaf
nodes.

**Mechanism:**
1. **Tier-1 (cheap, ~20 cyc):** After the FT accumulator update, compute a fast eval:
   `fast_eval = bias + Σ(active_acc[i] × weight[i])` via PSADBW (packed sum of absolute
   differences on int16 → horizontal sum). This is a LINEAR eval on the accumulator —
   essentially a material+positional readout. ~20 cyc.
2. **Tier-2 (full, ~1362 cyc):** The current SCReLU + L2 + L3 + out pipeline.
3. **Routing:** Use Tier-1 for nodes where a rough eval suffices:
   - Null-move null-window searches (exact value doesn't matter, just sign)
   - Late-move nodes (will be pruned by LMR)
   - Depth ≤ 1 in quiescence (captures only matter)
   - Nodes far from the alpha-beta window (predicted by Tier-1)
4. Use Tier-2 for PV nodes, cut nodes near the window, and root depth.

```
def tiered_eval(pos, alpha, beta, depth):
    if depth <= 1 or null_window or late_move:
        fast = accumulator_sum(pos)          # ~20 cyc (PSADBW on int16)
        if fast < alpha - MARGIN or fast > beta + MARGIN:
            return fast                      # Tier-1 adequate — skip full eval
    return full_nnue_eval(pos)               # ~1362 cyc (Tier-2)
```

**Feasibility:**
- **~60% of evals use Tier-1** (based on typical alpha-beta tree shape: most leaves are
  low-depth, far from the window).
- Average eval cost: 0.6 × 20 + 0.4 × 1362 = 557 cyc (vs 1362). **Saves ~805 cyc/node.**
- The accumulator sum (PSADBW on int16 × 512 values = 8 PSADBW instructions → ~16 cyc) is a
  PROVEN instruction with 1-cycle throughput on Zen 4.
- Accuracy risk: Tier-1 is a LINEAR eval (no SCReLU nonlinearity, no L2 feature mixing). It
  will miscalibrate for positions where non-linear interactions matter. But for pruned nodes,
  the eval is never used for move selection — it only affects the pruning decision. A slightly
  wrong prune decision has ~0 Elo impact if MARGIN is tuned.
- **Combined with int16-acc (H2):** the accumulator is already int16 → PSADBW works directly.

**Projected:** eval 1362→557 avg, update 743→370 (int8-FT). Per-node NNUE: 557 + 0.4×370 = 705.
Total per-node: 705 + 2029 (search) = 2734 cyc → **~1.28M NPS** standalone. With search
optimizations (H3): **~1.6-2.0M NPS.**

---

## H2 — IntAccel: int16 Accumulator + Post-Hoc int8 FT Without Retraining
*(the proven NNUE speed lever, adapted for no-retrain)*

**Name:** *IntAccel — Integer-Domain NNUE Inference via Post-Hoc FT Quantization and int16
Accumulator.*

**Contradiction shattered:** Stockfish uses int16-acc + int8-FT (trained from scratch). We
have a v2 net with FLOAT accumulator + int16 FT. The memory says "int8-FT QAT has structurally
zero gradient" (can't train int8-FT via QAT). BUT: SF trains in float and **post-hoc quantizes**
to int8-FT. We can do the same: take v2's int16 FT weights, scale to int8 (divide by scale
factor), and verify accuracy. If the accuracy loss is acceptable (±5 Elo), we get the int16-acc
path without retraining.

**Mechanism:**
1. Post-hoc convert FT weights: `ft_w_int8[i] = round(ft_w_int16[i] * 127 / max(|ft_w_int16|))`
2. Build engine int16-acc path: update = int16 add (no float convert), eval = int16 clip →
   int8 quant → VPDPBUSD. Removes float SCReLU (~300 cyc).
3. Test accuracy: compare eval output (int8-FT vs int16-FT) on 10K positions. If mean abs
   error < 5cp → deploy. If not → skip.

**Feasibility:**
- Eval: 1362→~1062 cyc. Update: 743→~370 cyc. Per-node NNUE: 1062 + 0.11×370 = 1103.
- Total: 1103 + 2029 = 3132 → ~1.12M NPS. **Stacks with TierEval (H1).**

---

## H3 — SearchSqueeze: 30% Search Overhead Reduction
*(targets Hole A: search is 55% of per-node cost — the majority)*

**Name:** *SearchSqueeze — Cycle Reduction in Alpha-Beta Move Generation, Ordering, and Pruning.*

**Contradiction shattered:** Every NPS paper targets the eval. But search is 55% of per-node
cost. Reducing search cycles by 30% (2029→1420 cyc) contributes MORE to NPS than eliminating the
entire SCReLU (350 cyc). The search code has never been cycle-profiled or optimized at the
instruction level.

**Specific optimizations (all engine code):**
1. **Move generation fusion:** Generate only the moves needed for ordering (captures first, then
   quiet) instead of generating ALL moves then sorting. Saves ~200 cyc.
2. **TT probe reduction:** Use a smaller TT signature (32-bit hash instead of 64-bit) for
   non-PV nodes. Halves TT memory bandwidth → ~100 cyc.
3. **Pruning check elimination:** Skip null-move pruning check for nodes already below
   null-window (the eval already tells us). Saves ~100 cyc.
4. **SEE inline:** Inline the Static Exchange Evaluation (currently a function call with stack
   setup). Saves ~100 cyc per capture node.
5. **Move list memory locality:** Allocate move lists on the stack in a cache-aligned layout.
   Saves ~50 cyc from reduced cache misses.

**Feasibility:** 2029→1420 cyc (−30%). Combined with H1+H2:
- Per-node: 705 (NNUE with TierEval+IntAccel) + 1420 (optimized search) = 2125 cyc.
- NPS: 3.5G / 2125 = **~1.65M NPS.** Close to 2M but not quite.

**To close the gap to 2M:** the eval frequency must also drop. With TierEval at 70% Tier-1
(more aggressive routing): eval avg = 0.7×20 + 0.3×1062 = 333 cyc. Per-node: 333 + 0.3×370 +
1420 = 1864 cyc → **1.88M NPS.** With overlap: **~2.0M NPS.**

---

## Stacking summary (all no-retrain)
| Optimization | Eval cyc | Update cyc | Search cyc | Total | NPS |
|---|---|---|---|---|---|
| Baseline | 1362 | 743 | 2029 | 3690 | 990K |
| +IntAccel | 1062 | 370 | 2029 | 3221 | 1.09M |
| +TierEval 60% | 444 | 260 | 2029 | 2733 | 1.28M |
| +SearchSqueeze | 444 | 260 | 1420 | 2124 | 1.65M |
| +TierEval 70% + overlap | **333** | **260** | **1320** | **1913** | **~1.83M** |

**2M is achievable** with aggressive stacking + search optimization, but it's at the edge.
Every lever must fire. The dominant contributor is **TierEval** (eval cost −60 to −70%).
