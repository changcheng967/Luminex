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
| gen768 | Aug–Sep 2026 | HalfKAv2_hm 24576/side, L1=768, SCReLU, DOSL dual-head | 4.21B → 5.37B | **Shipped** (`luminex_gen768_i8.nnue`), ~2800 Elo class |
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

- **lc0pack** — converts Leela v6 selfplay chunks to gamepack frames
  (`[hdr][mv][ev]` format: fen table, 9-byte game entries, u16 raw moves, delta-encoded evals).
- **luminex-featurize** — mmap'd, multithreaded C++ featurizer; `--stream` mode emits packed
  136-byte records; `--fen-eval` mode for text ingestion. Shipped as a prebuilt static binary.
- **verify_frames** — exhaustive frame verifier; no corrupt frame ever trains.
- **C500 multi-stage streaming trainer** (`c500_train_stream.py`) — frame-loop, budget-safe,
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
- **NNUE Template** — the house spec format (7 sections); all architecture specs follow it
  strictly.
- **V13 spec** (`nnue/V13.md`) — survived 7 machine-verification review rounds. Key ideas:
  FM interaction block (`I_k = ½(S_k² − Q_k)` from per-feature latent vectors, O(1) per
  feature flip), factorized king buckets (32 → 4/8/16, ablation-gated), material-count
  output bucketing (8 subnets), single-VPDPBUSD kernel contract over the [1600] concat with
  per-branch requant and +128 zero-point encoding for signed interaction slots.
- **DCU test bench** — Hygon 2× Z200SM_80 pod as a Bayesian NAS bench. After a debugging arc
  (dead probes traced to non-production init/loss scaling, then a synthetic-target dataset
  whose "evals" were linear-plus-noise), the bench now runs **Optuna TPE over
  production-faithful LNNUE forks on real Leela evals** (1.25M samples from gamepack
  frames): linear PST floor 327.8cp vs default NNUE 271.3cp. Study S1 winner:
  **L1=512 + FM-16, no cross, tail (8,64) → 257.1cp at 13.0M params** — FM-16 beats plain
  L1=768 (262.4cp) with 32% fewer parameters; the cross layer earned nothing at this budget.
- **Warm-start plan** — V13 initializes from the best gen768 weights function-preservingly:
  bucket rows decompose exactly into shared + delta (factorization), FM latents start at
  zero (contributes nothing at step 0), bucket tails copy the old tail. First V13 run A/Bs
  warm-start vs scratch on a subset.

## 10. V14: Residual Topology (Spec Complete)

`nnue/V14.md` — the 9-component upgrade: DOSL backbone trained first with the NNUE learning
a residual δ (±600cp correction range = 60% better quantization resolution than full-eval
training), mobility summary features (+40/perspective from engine attack maps), cross layer
on the interaction vector (triplewise terms), trajectory delta supervision, best-move policy
head, lazy accumulator updates, confidence-gated eval. Targets engine v7.0+.

---

## Standing Lessons

1. **Gate everything by games; revert what fails.** The 1,470-commit log is the evidence.
2. **Verify numerics against a reference before trusting a kernel** (probe_ft guard, frame
   verifier, export asserts).
3. **No capacity increases without VAL-curve evidence** (v4 rule).
4. **Post-hoc quantization, float training** (v3's QAT grave).
5. **Data first, architecture second — until saturation.** p4's flat MAE at 4.21B was the
   signal that gen768's architecture, not its data, is now the bottleneck.
6. **Search harnesses must mirror production** — the DCU probes only produced signal once
   they forked the real trainer's init, scaling, and loss on real targets.
