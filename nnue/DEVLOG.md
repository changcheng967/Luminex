# Luminex NNUE Development Log

Full history of Luminex's evaluation function — from handcrafted eval through every NNUE
generation to the V13/V14 architecture program. Dates are first-commit dates from the repo.

---

## Lineage at a Glance

| Generation | Era | Architecture | Data | Outcome |
| :--- | :--- | :--- | :--- | :--- |
| HCE v1–v5.10 | Jan–Jul 2026 | Handcrafted (PSQT, mobility, king safety, SPSA-tuned) | — | **Shipped** through v5.10.0 |
| NNUE v1–v2 | Jul 2026 | HalfKAv2 king-bucketed, L1=512, SCReLU, int8/int16 SIMD | Leela-derived | **Shipped**; 81K → 982K NPS SIMD ladder |
| NNUE v3 | Jul 2026 | QAT, L1=256 rebalance | — | **Abandoned** (int8-FT QAT broke gradients) |
| NNUE v4 | Jul 2026 | Float L1=1024 probe → back to v2 arch | — | **Probe**; established VAL-curve discipline |
| Data pipeline | Jul–Aug 2026 | lc0pack → frames → C500 stream trainer | test91 gamepack | **Shipped** (still in use) |
| NNUE v7 | Aug 2026 | c500 rewrite, lc0 tooling | — | **Poisoned** by embbag/gather drift; postmortem guards added |
| v8–v12 (+12p1/12p2) | Aug–Sep 2026 | recipe iterations on the c500 pipeline | gamepack packs | **v12 pinned the loss**; 12p1/12p2 = its training passes |
| gen768 | Aug–Sep 2026 | HalfKAv2_hm 24576/side, L1=768, SCReLU, DOSL dual-head | 4.21B → 5.37B | **Shipped** ([`luminex_gen768_i8.nnue`](../luminex_gen768_i8.nnue)), ~2800 Elo class |
| gen768 p1–p5 | Sep 2026 | same arch, multi-pass training | p4: 4.21B · p5: 5.37B | p4 MAE 176.1 ≈ baseline (saturation signal) |
| V13 | Sep 2026+ | FM interaction, factorized buckets, material bucketing | — | **Spec complete**; DCU search picking hyperparams |
| V14 | future | Residual topology + mobility + policy head | — | **Spec complete** |

---

## 1. Handcrafted Era (Jan – Jul 2026)

The engine shipped with a classical evaluation: piece-square tables, mobility, pawn structure,
king safety, progressively SPSA-tuned. Development culture from this era persists today:
**every experiment is gated by games, and regressions are reverted, not patched** — the log is
full of `Revert "..."` entries with measured Elo deltas. Releases v5.9.0 and v5.10.0 mark the
peak of the handcrafted eval before NNUE took over.

## 2. NNUE v1–v2: First Net and the SIMD Ladder (Jul 2026)

- **2026-07-12** — the engine loads and plays its first NNUE (`v2 net`): incremental
  accumulator with dynamic L1, HalfKAv2 king-bucketed features, SCReLU activations.
- The inference path was then optimized step by step, each milestone commit-measured:

| Step | Change | NPS |
| :--- | :--- | :--- |
| baseline | scalar | 81K |
| AVX2 SIMD evaluate | `evaluate` + accumulator updates | 426K |
| int8 L2/L3 (VPMADDUBSW) | LNI8 path, ~1cp vs float | 688K |
| int16 FT accumulator | hybrid eval | 774K |
| AVX-512 VNNI + AVX-512 SCReLU | VPDPBUSD | 809K |
| VPDPBUSD small-n dot for L3/out | eval 1855 → 1267 cycles | 982K |

- `FT_WSCALE` 4096 → 8192 bought an extra FT precision bit; output range ±661cp later
  motivated `NNUE_TARGET_CLIP=1000` when gamepack tails cascaded through L2/L3 SCReLU.

## 3. NNUE v3: The QAT Detour (Jul 2026)

Quantization-aware training with L1=256/L2=32/L3=64 rebalance. **Abandoned**: int8-FT QAT
produced broken gradients; fake-quant was hardened to guaranteed fp32 on the way out, but the
capacity rebalance didn't pay. Lesson recorded: quantize post-hoc, train in float.

## 4. NNUE v4: Curve Discipline (Jul 2026)

A float L1=1024 "raw-strength" trainer probed 2× capacity; per-epoch VAL loss was added to
distinguish capacity bottlenecks from data bottlenecks. Verdict: v2 architecture (L1=512) with
more data — and a standing rule: **no capacity increase without curve evidence**.

## 5. The Data Pipeline (Jul – Aug 2026)

The foundation everything since is built on:

- **[lc0pack](../src/lc0pack.cpp)** — converts Leela v6 selfplay chunks to gamepack frames
  (`[hdr][mv][ev]` format: fen table, 9-byte game entries, u16 raw moves, delta-encoded evals).
- **[luminex-featurize](../src/featurize.cpp)** — mmap'd, multithreaded C++ featurizer; `--stream` mode emits packed
  136-byte records; `--fen-eval` mode for text ingestion. Shipped as a prebuilt static binary.
- **[verify_frames](../src/verify_frames.c)** — exhaustive frame verifier; no corrupt frame ever trains.
- **C500 multi-stage streaming trainer** ([`c500_train_stream.py`](openi_upload/c500_train_stream.py)) — frame-loop, budget-safe,
  OOM-safe part-splitting for big frames.
- **Target calibration** — SF18's own NNUE scores only 0.48 R² against d26 targets; the
  sigmoid(cp/400) power-2.6 loss and pure-eval lambda were pinned after this study.

## 6. NNUE v7: The Poisoned Era (Aug 2026)

The c500 stream rewrite + lc0 gamepack tooling landed as "v7 prep" — and an entire training
era produced garbage. Root cause: mcPyTorch's EmbeddingBag backward drifted from the exact
gather-sum semantics, silently poisoning weights. The fix that persists in the trainer today:
a **`probe_ft` start guard** asserting `max|bag-sum − manual| < 1e-5` before every run, plus a
grad-level freeze hook on the padding row. Lesson: **fail loudly on numerics, never assume a
kernel**.

## 7. The v8–v12 Recipe Era (Aug – Sep 2026)

After the v7 postmortem, evolution continued as **training-recipe generations on the c500
pipeline** — iterated through env-var-driven retrains rather than code rewrites, so they live
in the trainer's header notes rather than headline commits:

- **v6 fix** (back-ported label) — bf16 + gradient clipping: prevents SCReLU gradient spikes.
- **v8** — L1=1024 with L2=16; postmortem: *the 1024-dim FT compressed to 16 dims before any
  interaction could be modeled* — L2 was the information bottleneck. This diagnosis produced
  the `NNUE_L2` width knob (L2=64 = 4× wider hidden) used ever since.
- **v9 trainer** (2026-09-15) — wide hidden layers, **power-2.6 sigmoid loss**, SWA,
  exponential LR, smart FEN skip. The loss and SWA machinery still in production arrived here.
- **v10** — calibration era; the step-rate calibration probe ("61.4 st/s") dates from here.
- **v11** — continued schedule/weighting refinements.
- **v12** — the recipe pinned: loss cross-checked line-by-line against SF nnue-pytorch,
  Berserk, Seer, and Leela source; optional SF position weighting (`NNUE_POS_W1/W2`); lambda
  implicitly 1.0 (pure eval — the gamepack stores no game results).
- **12p1 / 12p2** — the training passes of the v12 recipe; their nets were the strongest
  pre-widening checkpoints.
- **Cross-era net dissection** — MAE probes across all generation nets found *tail
  weight-decay crushing L2 dynamic range* → per-group weight decay
  (`NNUE_TAIL_WD`, FT wd=0) in the current optimizer.

The era ended with the **Luminex-Gen v1.3 architecture spec** absorbing all research findings
(three review rounds, E1/E5/G0 experiment IDs), whose **Step 0 was Net2Net widening
512→768** — giving the generation its name: **gen768**. Step 2 (the residual "UB net") is
deferred to V14.

## 8. gen768: The Current Generation (Aug – Sep 2026)

**Architecture** — HalfKAv2_hm feature indexer (adapted from official Stockfish nnue-pytorch):
768 planes × 32 king buckets = 24,576 features per perspective, horizontal mirroring, L1=768
accumulator, SCReLU stack, `out × 300` output scaling.

- **2026-09-21** — DOSL dual-head: a parallel O(1) linear PST head over the same sparse
  features, trained with auxiliary loss, intended for qsearch stand-pat
  (`QsearchLinear`). The 0–33 gating loss was traced to **LR starvation** (×5 multiplier
  cannot grow a head from 0 to cp-scale); fix for future rewarm: ×500–1000 during the backbone
  phase.
- **Accumulator saga** — redesigned as exact int32, then settled as **int16 saturating**
  (SCReLU window 4× below saturation; verified exact on 6,000 real game perspectives).
- **2026-09-24** — Gen768 became the default net with NNUE on by default.
- **Frame densification** — `frame_merge` (golden tests T1–T5) merges small frames; the
  lmbig pack (58 frames, 10.70GB, 5.37B positions) was built this way.
- **SWA saga** — two bugs fixed after p4: the averaging trigger compared against global
  instead of session-relative steps (a0f6ed7), and resumed passes inherited stale averages
  (39c9eda — fresh average per pass unless `NNUE_SWA_RESUME=1`). An interrupted p5 pass
  exported at −56 Elo (mid-cosine, no SWA) — the run is being redone sized to the 3h
  session wall (T_MAX=26000, SWA=19500).

**Training passes** (per-pass warm-restart cosines, AdamW amsgrad, bs 131072):

| Pass | Data | Steps | Result |
| :--- | :--- | :--- | :--- |
| p1–p3 | growing packs | — | iterative gains |
| p4 | 4.21B | 120,784 | MAE 176.1 ≈ baseline 177 — **saturation signal** |
| p5 | 5.37B (lmbig) | 26,000/pass | in training; HEALTH MAE 157.3 on frame 1 |

## 9. Architecture Research & V13 (Sep 2026)

- **Research library** — cross-domain survey (CTR factorization machines, DCN-V2 cross
  layers, JL/RG bounds) validating the V13 design space.
- **[NNUE Template](NNUE%20Template.md)** — the house spec format (7 sections); all architecture specs follow it
  strictly.
- **V13 spec** ([`V13.md`](V13.md)) — survived 7 machine-verification review rounds. Key ideas:
  FM interaction block (`I_k = ½(S_k² − Q_k)` from per-feature latent vectors, O(1) per
  feature flip), factorized king buckets (32 → 4/8/16, ablation-gated), material-count
  output bucketing (8 subnets), single-VPDPBUSD kernel contract over the [1600] concat with
  per-branch requant and +128 zero-point encoding for signed interaction slots.
- **DCU test bench** — Hygon 2× Z200SM_80 pod as a Bayesian NAS bench. After a debugging arc
  (dead probes traced to non-production init/loss scaling, then a synthetic-target dataset
  whose "evals" were linear-plus-noise), the bench now runs **Optuna TPE ([`arch_search.py`](arch_search.py)) over
  production-faithful LNNUE forks on real Leela evals** (1.25M samples from gamepack
  frames): linear PST floor 327.8cp vs default NNUE 271.3cp. Two 20-trial studies plus an
  8,000-step confirmation picked the V13 core: **L1=768 + FM rank 8 + cross layer, tail
  (16,64) → 252.2cp** (best of four runs on the config; the FM block earns ~5–8cp over
  no-FM at equal width; an early 512-width lead failed replication and lost the
  head-to-head 256.9 vs 252.2). A faster trial recipe (4× batch, half steps) drifted
  +9–12cp on FM configs and was rejected as a ranking default.
- **V14/V15 component campaign** (2x30 TPE trials + 8,000-step confirmation) searched
  factorized FT buckets, material-bucketed tails, cross variants, activation family,
  derived structural side features (material/pawn-file/king, from feature indices), and
  FM rank 4-64, under a cost-penalized objective (MAE + 20cp per 100% eval cost vs
  anchor). Confirmed winner: **L1=512, no FM block, ClippedReLU, factorized FT (8
  buckets + shared), side features -> 235.3/243.6cp** vs the V13 anchor's 252.2 —
  "simpler network, richer inputs". Material bucketing and the cross layer did not
  earn their cost; SCReLU lost to ClippedReLU in the winning lineage. This amends the
  V13 implementation target and drops the FM incremental machinery from the engine
  design.
- **gen768 closure (p5/p6)** — three continuation attempts after p4, all rejected by gates:
  p5-A killed mid-cosine by the session wall (-56), p5-B damaged by an LR=1e-3 restart
  on a converged net (-26cp stm-relative calibration offset; the sigmoid-power loss is
  nearly blind to constant offsets, so nothing in training corrected it), p6 completed
  with SWA and a clean +1.3cp intercept yet gated 50.0% vs p4, 42.5% vs shipped over
  100 games, and a Stash ladder of 48/47/33% (implied 2756/2809/2757, mean 2774 vs the
  shipped net's 2796) — the residual eval noise of the averaged weights costing most
  against the strongest opponent. Verdict: architecture saturated, shipped net retained.
  Countermeasures adopted for V13: eval-suite intercept check before any gate, anchor
  regularization and export-time intercept fold in the trainer, DONE+SWA before export.
- **Warm-start audition** — SVD 768->512 shrink of a gen768 checkpoint into V13 scored
  290.6cp vs 240.1cp from scratch at equal budget: the projection loses too much and
  fine-tune LR cannot rebuild it. V13 run 1 trains from scratch.
- **Warm-start plan** — V13 initializes from the best gen768 weights function-preservingly:
  bucket rows decompose exactly into shared + delta (factorization), FM latents start at
  zero (contributes nothing at step 0), bucket tails copy the old tail. First V13 run A/Bs
  warm-start vs scratch on a subset.

## 10. V14: Beyond V13 (campaign program)

The original V14 spec (residual topology on a 768/FM/cross base, 9 components)
is retired: the component campaigns refuted its architecture basis —
512/factorized-8/ClippedReLU/side-features beat every 768/FM/cross shape, and
that winner was folded into V13's amended spec instead. V14 now names the next
generation, defined by campaign evidence as it lands: enriched derived
features (running), trajectory delta supervision, policy multi-task, and the
residual topology retried with a properly-trained lin head (the x500-1000 LR
recipe). The spec forms in [V14.md](V14.md) once a campaign produces a
confirmed winner.

## 11. V13 quantized inference — LX3Q (Oct 2026)

The "normal procedure" (quantize before gating) became `quantize_v13q.py`:
float LXV3 → LX3Q with FT int16@8192 and an int8 tail at the fixed scale
64 = 8192/128 — chosen so the `>>6` activation requant keeps the whole dense
chain in the ×8192 domain (min(v>>6,127)); the out layer carries its own
scale (127/max|w| — its max weight 3.78 doesn't fit 64), folded into
`v13_out_k` per spec §6. Two lessons baked into the exporter: the activation
scale after `>>6` is ×128, not ×127 (an 8192/127 weight scale cost a
systematic +53 cp bias error), and the floor + 127-cap requant leaves a
measured per-output bias that the exporter calibrates on synthetic inputs
(an analytic Σw/256 correction proved insufficient). Numpy parity over 3k
synthetic accumulator states: mean −1.2 cp, σ 11.9 cp. Net size halves:
14.2 MB → 7.1 MB; engine nps 45K → 330K (above gen768's 288K on the same
machine).

The int16-accumulator switch was tried and **reverted the same day**: a numpy
ground truth (startpos lanes rebuilt by hand from the index math) showed p2's
resting lanes reach ±5.0 at startpos — beyond int16's ±4.0 — so the soft
NNUE_ACC_REG did not hold the extremes ("L2|max| 0.92" is not a lane max).
Saturating int16 adds corrupted static evals by ~−90 cp while deep-search
evals looked healthy (out-of-window resting lanes clip to the same activation
either way, masking the corruption); spec §3's int32 fallback stands until a
net is trained with a hard range constraint. The verification lesson: static
evals against a numpy ground truth, not search evals — search agreement is
not evidence of accumulator correctness.

**p2 gate verdict (Oct 2026, Lightning studio):** blitz ladder 10+0.1,
50 games per rung — gen768 8–34–8 (24.0%, −185), Stash 22 19–22–9 (47.0%),
Stash 23 10–33–7 (27.0%), Stash 24 11–34–5 (27.0%), Stash 21 16–31–3
(35.0%): roughly Stash-21/22 level. Fixed-node vs gen768 (NodesPerMove
100000, 100 games): **31–55–14 = 38.0%** — identical to run1's 38.0%, so the
second pass (fresh Sep–Oct data + anchor + acc_reg) gained zero eval
strength; the plateau stands. The TC-vs-nodes gap (24% at equal time vs 38%
at equal nodes, despite p2_q being faster: 330K vs 288K nps) says deeper
search under V13's eval *hurts* relative to gen768 — eval noise (σ≈12 cp
quant + young net) interacting with gen768-tuned pruning margins, or labels
from ~800-visit lc0 search not surviving deep tactical resolution. Does not
ship; next levers are the V14 data campaigns (C5 WDL mixing ranked #1 among
unused lc0 signals) or search-side adaptation to V13's eval scale.

**p1 vs p2 h2h (fixed nodes, both LX3Q-quantized, 100 games):** 35–45–20
= 45.0% for p1 — p2 wins at 55% (+35 Elo, inside the ±55 noise band at 100
games). Combined with the identical 38.0%/38.0% external scores vs gen768,
the verdict is that the second pass bought at most a marginal internal edge
and no measurable external strength: the lineage is plateaued at this
recipe.

## 12. V14 C5 pilot — WDL/result mixing rejected (Oct 2026)

The #1-ranked unused lc0 signal got its pilot: GP2 frames (lc0pack game
result per entry → featurize 140-byte records with a result float → trainer
λ-mix `λ·eval + (1−λ)·1000cp·result`, annealed 1.0→0.8 over 40K steps).
Two arms, same 8M-position pilot corpus, same 3.4h budget, lin off — λ the
only variable. Fixed-node h2h (100 games, 100K nodes, float nets): **ctrl
65–30–5 = 67.5% — the λ arm lost by ~78 Elo**, decisive at this sample.
Mechanism read: 86% of test91 games are eval-adjudicated (invariance bit 5)
and 55% are draws, so the result channel mostly re-injects the net's own
thresholded eval and compresses decisive positions toward drawishness
(startpos eval 17→12cp). The pipeline itself is done and reusable (GP2
frames verified end-to-end, static featurize binary in-repo); what failed
is the signal on this corpus. Conclusion: don't spend the 72GB full-window
re-harvest — test91's result labels carry too little independent signal at
any λ worth training with. "Get more from the data" now rests on the
non-result signals (best_q target swap, deblunder filters) or leaves the
data entirely (search-side adaptation to V13's eval scale).

---

---

## v2 program (in progress)

v1 (gen768, spec in V1-GEN768.md) is the shipped architecture. The v2 program
adds levers decided by measurement: the G0 battery (linear head R2 0.196 vs
full-net 0.515 - interactions dominate) and a six-arm comparison on a 14M
slice (baseline / 4 buckets / J32 / pairwise / pairwise+J32 / over-capacity
probe). Settled so far: int16 accumulator by range-constrained training,
eight whole-tail material buckets with copy-seeded warm start, a PSQT-style
linear rider, rounded requantization. Gated on the arms: tail width 32,
pairwise activation, P/N/K threat features. Discipline from the 105-epoch
overfit incident: FEN-hash validation holdouts, 2-8 epochs over large unique
sets, per-epoch val MAE with a memorization alarm, and a label-convention
gate (a raw stm-relative dump once flipped half the targets and collapsed a
full run; fixed at the dump source and gated at train time).

### Correction of a correction (Oct 2026): the six-arm suite was real

An earlier note here claimed the suite was a no-op because the trainer
didn't parse NNUE_BUCKETS / NNUE_L2 / NNUE_PAIRWISE. That was a misdiagnosis:
the env vars are consumed inside LNNUE.__init__ in luminex_nnue_train.py,
and the trainer's resume path seeds bucketed/widened tails via
LNNUE.v2_expand (function-preservation tested: bucket copy-seed 0.000cp,
pairwise duplication sub-cp, J-widening no-dead-units). The levers were
applied and "buckets ≈ neutral / J32 null / pairwise null" is real
evidence. What still stands — and was always the documented caveat — is the
regime confound: 3,000 steps over a 14M slice is ~28 epochs, far past the
2-8 epoch band, i.e. the arms measured the levers on noisy labels in an
overfit regime. The levers remain untested on clean quiet labels in-regime;
that is the T1 gate. Lesson recorded: verify which file consumes an env var
before declaring a lever dead — grep the module, not just the entry point.

## 13. Session 1 — dose/weight decomposition, and the dose curve (Oct 2026)

Five runs on the full SF-labelled corpus (776M positions/run, p4 warm start,
LR 1e-4, BS 131072, loss-power 2.6), each varying ONE factor from the T0
recipe (quiet 1/0.3/0, 6000 steps ≈ 1.3 epochs, default seed). Gates were
600-game fixed-node matches vs T0 with the openings book (± ~28 Elo at 95%).

- **r0u** (unweighted loss, T0's exact dose): 206–?–? measured **−25** — the
  clean seed-paired weight effect. Quiet weighting is worth ~25 Elo.
- **r0b** (seed 99, otherwise T0): **−22** — one draw of run-to-run noise;
  same order as the weight effect itself.
- **d065** (0.65× dose ≈ 0.86 epochs): **−57 ± 28**. Pre-registered +40
  under the log-linear dose model — rejected, sign wrong.
- **d2** (2× dose ≈ 2.6 epochs): **−5 ± 27**. Pre-registered −60 — rejected,
  the high-dose side is nearly flat at 2.6, not falling log-linearly.
- **r1** (capture weight 0.1 instead of 0.3): **−45 ± 27**. Pre-registered
  ±15 under "weight axis is flat" — rejected at the edge; the capture weight
  carries real signal.

**Gate resolution (the binding constraint).** A 600-game gate has ±28
half-width at 95% (SE(Elo) ≈ 695·√(0.25/N)); detectable effects: 600g →
~28, 1200g → ~20, 2000g → ~15, 5000g → ~10. Every gate this program has
run at 300–600 games could only see effects ≥ ~40 Elo. All four session-1
deltas sit between 5 and 57 — two of them are real, two are not, and the
gates can barely tell which. **Default gate is now 2,000 games** (±15, two
hours per arm on the arena box — one arm per night, which is the cadence
anyway). The diversity prediction (+10–30) straddles the 1,200/5,000-game
range, so that arm is gated at 2,000 minimum and read as a pre-registered
range check, not a point.

**The dose curve** (quiet recipe, vs T0 at 6000 steps): 3900 steps/0.86ep →
−57, 6000 steps/1.3ep → 0, 12000 steps/2.6ep → −5, ~4.45ep → ~−115
(indirect: the v6.2.0 recipe delta with the weight credit removed). Shape: a
**plateau from 6000 to 12000 steps, a steep wall below 3900 steps, and the
cliff somewhere between 2.6 and 4.45 epochs**. Log-linear is dead. T0 sits
at the **left edge of the plateau, not on an interior optimum**: 6000 steps
is the *minimum sufficient* dose with no margin below it, so any future
change of corpus or recipe must re-verify that 6000 steps is still enough.
(The safe read is 2× — doubling the read cost nothing.)

**Plateau is in steps, not epochs — and that does NOT license "more data."**
6000 and 12000 steps on the same corpus give the same net. Under a
steps-reading, doubling the corpus at 6000 steps (0.66ep of 1.2B) is
predicted to give the same net too, and doubling at 12000 steps is already
measured at −5. This batch closes the epoch question and leaves the data
question untouched. The one live mechanism for unique data is gradient
redundancy (at 6000 steps each position is read 1.3×; on 6B it would be
0.13×) — real but second-order: expectation +10–30, not +100. The test runs
in two stages, cheapest first:

1. **Redundancy screen (no dump needed): 300M vs 597M, both at 12,000
   steps, same source.** Total visits identical (1.573B); only
   per-position redundancy differs (5.2 vs 2.6 passes). Arm B already
   exists — it is d2. Arm A (h300, a 300M prefix of the same TSV, default
   seed so the h2h with d2 is seed-paired) is one training run. Decision
   rule pre-registered: h2h d2 vs h300 at 2,000 games, A ≥ 20 Elo worse →
   the pass-penalty is real at 2× and the unique-corpus arm is justified;
   tie → the mechanism does not pay at 2× and 2B almost certainly will not
   either, saving months of dump work. (A large A-loss is unambiguous
   either way: cliff or gradient, both point toward fewer passes.)
2. **The 2B arm — only if the screen passes — must be a STRICT SUPERSET:
   597M ∪ 1.4B new, through the IDENTICAL filter/flag pipeline.** A fresh
   linrock-only corpus would differ from B in two variables (diversity AND
   label distribution/filter generation) and any result would be
   unattributable. The natural implementation ("dump the archive and train
   on it") is the wrong experiment; the superset construction is the
   requirement.

**Weight axis**: capture 0.1 → −45 ± 28, capture 0.3 → 0, unweighted
(capture 1.0) → −25 ± 28. 0.3 beats both sides — it is now a *measured*
optimum against both neighbors — but the two failures differ by only 20
against ±40 on the difference, so they cannot be ordered. Within this
single-variable design the weight contribution is **≤ 25, point estimate
25, not established at 95% on its own gate**; the ~114 dose component of
the v6.3.0 recipe gain is carried by the gen768sf comparison. Remaining
probe: capture 0.4–0.5 (r2), low priority.

**Seed pairing**: r0u, d065, d2 and r1 all ran T0's default seed — every
session-1 delta above is seed-paired and needs no seed correction. The
seeds-7/123/2024 runs calibrate σ_seed for FUTURE arms (a new corpus means
a new data order), not for re-reading tonight. Caveat on the estimate
itself: σ from 4 draws is only good to ±40%, so the first σ_seed is an
order-of-magnitude number (is it 10 or 30?), not a precise constant. If it
lands near 25, single-seed deltas below ~25 are unreadable and the
2,000-game default gate becomes mandatory rather than advisable.

**Session 2, revised**: (1) the redundancy screen — 300M vs 597M (d2) at
12000 steps, gated by 2000-game h2h; (2) if it passes, the 2B strict
superset through the identical pipeline; (3) re-verify 6000-step
sufficiency on any new corpus, since a corpus change moves the wall; (4)
r2 capture 0.4–0.5, low priority; (5) epochs on a fixed corpus: closed.

### The redundancy screen result (Oct 31, DCU arena)

d2 vs h300 head-to-head, 2,000 games (10×200 chunks, 8moves_v3 book,
NodesPerMove 20000, colors swapped, fastchess on the DCU pod — the arena
mscope used to host was rebuilt here while its service was down; engine
via a new glibc-2.35 CI build since release binaries need glibc 2.38).
Pooled: **d2 837 – h300 774 – 389 draws = 51.6% for d2 → d2 +11 ± 21**.
Pre-registered rule: h300 ≥ 20 Elo worse → build the 2B superset. The
point estimate (11) is under threshold → **tie → the data axis closes,
measured on games**. The 11-Elo gradient does point the way redundancy
theory predicts (more passes slightly worse), but at a 2× ratio the whole
effect is noise-level — 2B unique positions cannot be expected to pay.
The linrock dump program is shelved. Caveat on record: at ±21 the gate
cannot fully separate 11 from 20; the asymmetry argument (a true −20
reading +11 is a 1.5σ miss) supports the tie read.

### The instrument rule (after round-6 cross-examination)

Every closure claim must name its instrument. The recipe axis (dose,
weights) is closed **on games** — defensible. The architecture axis
(buckets, L2 widening, predicates, FT width) was closed **on val MAE**,
an instrument the anchor experiment later discredited (+106 net was
0.56cp WORSE on quiet-subset MAE than a −33 net). Not one architecture
lever has ever been gated on games. "Architecture is dead" is therefore
retracted; the honest claim is "architecture was measured with a broken
ruler and never re-measured with the working one."

**First re-measurement: tail-only output buckets** (8 piece-count tail
sets, same FT, copy-seeded from T0's checkpoint). Trainer support exists
and is function-preservation tested (v2_expand). CORRECTION to the
proposal as received: the ENGINE does not support bucketed nets today —
nnue.h/nnue.cpp have no multi-tail loader and no material-count
selection (verified by grep; the engine's "bucket" mentions are input
king-planes, attack curves, correction histograms). Real cost = engine
loader + evaluate() bucket select + i8 quantizer extension (~half a
day, backward-compatible format), THEN the C500 session (6000 steps,
T0 recipe, same seed) and a 2000-game gate. Guards: (1) copy-seeded
bucket net must be bit-identical to T0 at load before any training;
(2) pre-registered read — ≥ +20 Elo on games reopens the architecture
program with games as the instrument; ±15 closes it with a valid ruler.

### Paired-opening error bars (round-7 correction)

The `-repeat` colour-swapped design correlates game pairs: effective N is
~N/1.96, so the 95% half-width is 1.4× the independent-games figure.
**Every 600-game gate in this program was ±39, not ±28.** Retrospective
reads: d065 −57 ±39 (clean), d2 −5 ±39 (flat), **r1 −45 ±39 (CI −84…−6,
still excludes 0, barely)**, **r0u −25 ±39 (inside noise)**. The dose
curve's shape survives — the effects are large and ordered — but the
weight axis is now honestly "point estimates 0.3 > {0.1, 1.0}, only r1's
gate clears its own CI." Gate sizing for the bucket read: 2,500 paired
games minimum for a +20 threshold; 4,000 to separate +10 from +20 — at
DCU arena speed both cost under two hours, so gates default to 4,000
where a ship decision hangs on the band.

### The +106 FT-tail probe — hypothesis REJECTED (Oct 31)

Pre-registered: FT tail grows ≥2× from 12k→20k steps → cliff is
rare-row memorization → run anchor. Measured (all .pt checkpoints on the
DCU NFS; gstep decodes as steps + 120,784 featurize baseline — p4=0,
T0/sfq=6k, gen768sf_full=20k = the 4.39ep cliff net, gen768sf=42k):

- FT (18.9M params): max 2.398→2.402 (0→20k), p99.9 flat ±2%, n>1.0
  2,973→3,109. **Flat.** At 42k: max 2.434, n>1.0 3,231 — still no 2×.
- Extended to tail layers over the same range: l2 max 0.760→0.789,
  l2 mean −1%, l3/out flat. **The 114-Elo cliff (6k→20k) has NO global
  weight-statistics signature in any layer.** Only at 42k does l2 creep
  (std +47%) — deeper degradation, past the range that matters.
- Cross-recipe at fixed dose (T0/r0u/r1/r0b at 6k): statistics agree to
  the 4th decimal. Recipe differences are function-level, not
  distribution-level.

Consequences: anchor (`NNUE_ANCHOR_W`, implemented, idle) has NO
supporting evidence — not run. The +106 mechanism is now bounded: not
FT stats, not tail stats, not any of the 7 offline metrics — it lives in
the position-level function. Next probe when taken: per-position eval
divergence between the 6k and 20k nets on a labelled stream, split by
material phase / king square / quiet-capture flag (g0_gates-style on
.rec) — find WHERE the two nets disagree.

### Bucket re-measurement, final plan (round-7)

Engine work priced at 2–3h as an offset into the flat tail arrays (new
magic LNI9, hdr gains NUM_BUCKETS; old nets load unchanged). Two known
bug nests, both verified in code: the VNNI batched-L3 interleave
(`g_l3_batched`) and the AVX2 chunk-L2 load-time reorder — both rewrite
weights in place and must loop per bucket or buckets 1+ are garbage.
Guard: copy-seeded bucket net must be bit-identical to T0 over a walked
game. **2 buckets before 8** (≥16 vs <16 men — near-balanced split; 8
equal-width buckets starve the endgame tails that need phase
specialisation most). One C500 session at T0's recipe, same seed;
4,000-game gate. Pre-registered: ≥ +20 reopens architecture on games;
±15 at 4,000 games closes it with a valid ruler.

### The sibling-correlation probe — REJECTED (Oct 31, round 8)

Agent-1's ensemble frame: strength tracks the COVARIANCE of eval error
across the positions search compares (siblings), not per-position error;
correlated class bias survives any depth (bxa4, Bc7??); quiet weighting
and the dose cut would then be decorrelation interventions. Pre-registered
H1 weight→decorrelation (T0 < r0u), H2 dose→correlation (T0 < sf20k).

The corpus has no labelled sibling sets, so two proxies were measured on
a 300K game-ordered slice (4,661 games): class-ICC of SIGNED error within
material-piece-count classes (one-way random effects, 887 classes ≥50),
and lag-1 autocorrelation of signed error within games. Result across
T0/r0u/sf20k/sf42k: **class-ICC ≈ 0.000 for every net (±0.0003) and
between-net differences ≤ 0.0001 — REJECTED on both hypotheses.** No
material-class-correlated error exists in any net, quiet or not, at any
dose. (The lag-1 ≈ −0.43 values are an stm-convention artifact — signed
stm-relative error alternates sign with the mover — so only the
between-net differences there are meaningful, and they are ~0.003.)

Side observation that hardens the paradox: the 20k net has the BEST MAE
on this slice (100.1 vs T0's 101.5) — the dose axis shows the same MAE
anti-correlation with strength the anchor pair did. Both recipe gains
came with equal-or-worse per-position statistics AND no measurable error
correlation structure. Per pre-registration: the covariance story is
dead as measurable by these proxies; the +106 stays open.

**Round-9 verification and reinterpretation of the lag-1 "artifact".**
Re-measured with white-relative error (sign flipped on BTM positions):
**lag-1 correlation is +0.71 (T0), +0.72 (r0u), +0.71 (sf20k), +0.66
(sf42k)** — the stm-convention flip was hiding a LARGE, near-universal
error persistence: ~70% of eval error variance carries from one position
to the next along a game, invariant to recipe and dose (differences in
the second decimal). Two-part verdict on the ensemble hypothesis: (1)
correlated error exists and is big — CONFIRMED at ρ≈0.7; (2) T0 has less
of it — REFUTED (Δ ≤ 0.01). Correlated error is a constant of this
architecture/data family, not the +106's mechanism. (Caveat: the 0.7
measures persistence along played trajectories — a proxy for, not a
measurement of, sibling-group covariance at search nodes.)

### Open contradictions (the quiet-law backlog)

The +106 had no offline mechanism (8 metrics + ICC now: 0/9). 20...bxa4
reproduces cold in every arm at 40M nodes. 86.4% sign agreement on clear
pairs means one in seven clear
comparisons is backwards on BOTH nets — the failing 14% is unidentified.
Four Carlsen-game errors reproduce in no replay configuration. None is
an "axis"; all are the shape of thing that produced the last jump.

## 14. The pivot (Oct 30): metrics → forensics

**Method ledger.** Three instrument classes, tallied across the whole +106
campaign:

| instrument class | record |
|---|---|
| offline metrics | **0 / 8** |
| architecture changes | **0 / 5** |
| game forensics | **2 / 2** |

The asymmetry is the finding. Every instrument that reads *positions* has
failed; both instruments that read *games* produced something. Decision rule
adopted: **stop proposing metrics.** Eight failures in the same direction is
not an invitation to a ninth variant.

**Divergence probe — deleted from the backlog.** It would have been metric #9
against an 0/8 record. The base-rate argument settles it.

**sf42k caveat recorded.** The strength-rank table has four entries; three are
measured, sf42k's is *assumed* and was never gated. The dose-axis pair that
actually carries the quiet law — T0 101.5 MAE vs sf20k 100.1, +106 apart — is
measured on both sides, so the law stands on well-measured points. The fourth
point is now marked unmeasured rather than letting the table imply more than
it shows.

**σ_seed pre-registrations adopted as written.** s7 done (−8), s123 done
(10/10, reading close to even from its chunks), s2024 at chunk 1 — all landed
within ~20 min. If **σ ≥ 25**, the re-gate runs against **sf20k instead of
T0** (~25 min/net on the DCU arena) and every downstream number rescales
honestly.

**Idea-3 catch — counter lifecycle.** Any counter must live on the *same
lifecycle as the tables it describes*: they age between searches and are
cleared on newgame/net-load. Implementation will assert that the KRP
distribution actually changes in the offline 500-position probe **before** any
game is spent. GEN_QUIET_CHECK applied.

**Pivot accepted.** Main line becomes a **black-box recorder** + 500 bot games
+ failure-mining queries. Per move the recorder captures: ply, nodes, seldepth,
elapsed, clock remaining, aspiration attempts, iteration completion, TT
occupancy, |root correction|.

**Sequencing, stated explicitly.** The **bucket gate runs first anyway**: it is
already staged, costs one session, and closes the architecture retraction on
games — the only instrument that counts — instead of leaving it
retracted-and-unresolved. Then the research loop stops and the forensics loop
starts. **Recorder design begins the moment σ_seed is read.**

*Standing caveat from round 9 (unchanged):* ρ≈0.7 measures error persistence
along played trajectories — a proxy for, not a measurement of, sibling-group
covariance at search nodes.

## 15. σ_seed measured — branch one (Oct 30)

The three seed gates (2,000 games each vs T0, paired openings
2001–3000): **s7 −7.6 ± 15, s123 −13.0 ± 15, s2024 −4.7 ± 15**; with
r0b (−22 at its noisier 600g gate) the four draws have mean **−11.8**,
observed SD **7.6**. The observed scatter (χ² = 1.5 on 3 df against the
expected gate-noise variance of ~114) is fully consistent with pure gate
noise: **true σ_seed ≈ 0–8, point estimate 7.6**. (Estimate itself good
to ±40% from four draws — the order of magnitude is what matters.)

Pre-registered read: **σ ≤ 12 → the +106 is real.** Taken. T0 is a mild
+1.5σ draw above the seed mean (all four draws negative, mean −12), so
the honest decomposition of the recipe gain is ~94 Elo recipe + ~12
seed luck — the seed lottery is an order of magnitude too small to
explain the effect, and the mechanism genuinely lives outside every
static property of the net (0/9 instruments). No re-gate against sf20k
needed. Per §14: recorder design starts now; the bucket gate remains
the one staged architecture closure before forensics becomes the main
line.
