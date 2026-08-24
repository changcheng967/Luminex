# HCE Tuning & Distillation Record

Chronological record of every hand-crafted-eval tuning campaign on Luminex:
what was tried, the measurement, and the verdict. All Elo numbers from
cutechess (500 games tc=1+0.01 unless noted; ±~29 Elo at 500g). Stash ladder
Elo: V20=2509, V21=2714.

---

## 1. Texel MSE tuning — DEAD END (8 attempts)

Tuning the EXISTING eval terms by gradient descent on quiet-position MSE.
All 8 attempts failed, including a proper-scale 4.18M-row self-play dataset
with verified-correct gradients. **MSE-tuning existing terms does not
improve play: the eval at play-strength is already at a local optimum.**

- Infra preserved: `tuner_v4` + `/hyperai/home/overnight_quiet.txt`.
- Do not re-attempt tuning existing terms in place.

## 2. SPSA search-constant tuning — NO GAIN

12 search constants (LMR scales, futility, NMP, razoring, reverse futility,
aspiration) via the SPSA harness, fixed-node N=100K evaluation, tc=1+0.01
gates. Result: **tuned ≈ default (+6 Elo = noise over 1000g).** HCE search
constants are near-optimal; path forward is eval, not search tuning.

- 4 harness bugs fixed (run_match last-line, _singular wiring, v3 gains,
  theta_bar normalization).
- Real SPSA signal needs ±25 Elo/eval (~200g/eval, ~24h) — not worth it here.

## 3. SF distillation ridge fits — THE WIN

Fit the linear feature basis (`eval_feat.h`) to 22.14M Stockfish-eval rows
(1.5GB `fens_clean.txt`, from the 14.5B gamepack; 2.47M held out) with a
closed-form eigen-path ridge, centered on the current engine coefs.

Pipeline: `luminex-evaltrace --solve/--score/--verify/--dump-coefs` →
`solve_ridge.py` (whitened ridge + rank-1 gauge anchors from
`gen_anchors.py`) → `gen_fitted.py` → `src/eval_fitted.h` (regenerate +
rebuild applies a fit).

### Era 1 — unclipped targets, from pre-fit baseline (c0)

| fit | λ | A/B vs baseline | vs StashV20 |
|---|---|---|---|
| c0 baseline | — | — | −45.4 ± 29.9 |
| fit2 (anchored) | 1e6 | −47.5 ± 28.7 | −20.2 ± 29.6 |
| fit3 | 3e6 | +37 ± 28 | −48.9 |
| fit4 | 1e7 | +62.5 ± 28.3 | −2.8 ± 29.4 |
| **fit5 ADOPTED (8656041)** | **3e7** | **+98.4 ± 28.5 LOS 100%** | −2.8 ± 29.1 |
| fit6 | 1e8 | +68.3 ± 27.4 | — |
| fit7 | 3e8 | +24.4 ± 26.4 | — |

- **fit1 (unanchored λ=3000) REJECTED**: holdout R² 0.22→0.56 but **−210 Elo**
  — in-distribution agreement ≠ play strength. Gauge freedom (material↔PST
  collinearity) let the fit put chess-nonsense values (queen MG effective
  −52) that explode out-of-distribution. Fix: 12 rank-1 anchors pinning
  effective piece values (MAT + mean PST).
- **λ-curve is unimodal, peak λ=3e7.** mean|Δ| at peak ≈ 11cp: small
  data-directed corrections on the hand values win; big reallocations lose.

### Era 2 — tail-clipped targets (fit6, ADOPTED, 95ec6b4)

Root cause of leverage garbage: the gamepack eval tail runs to ±9096cp; a
handful of near-mate targets dominate squared error. Fix `NNUE_TARGET_MAX=1000`
(drops 79.7K/24.9M rows). **All future fits must clip.**

Cleaned λ-curve from c0 (A/B vs fit5, 500g each):

| λ | A/B vs fit5 |
|---|---|
| 1e8 | −27.9 |
| **3e7 (fit6)** | **+18.1, +6.3 (avg +12)** |
| 1e7 | +5.6 |
| (from fit5 center) 1e7 | −35.6 |
| (from fit5 center) 3e6 | −47.5 |

fit6 vs StashV20: **232-228-40 (0.504, +2.8) — first >50% vs V20** ⇒
≈2512, ladder floor advanced. Post-fit movement in any direction loses:
fit5/fit6 are at the play-optimum for the current basis.

### Margin-rescale ×0.862 — TESTED & DEAD

Scaling all cp-coupled search margins by the fit5-vs-c0 eval-std ratio:
−8.3 ± 26.5 (neutral, reverted). Eval STD ratio does NOT transfer to margin
scale — margins guard eval ERROR, not spread.

### Tooling bug (fixed): evaltrace --score OOB

TEMPO features (idx ≥ NSOLVE) fell into the EG branch and read past the
coef arrays — stack garbage poisoned every holdout base-R² of the fit era.
All PLAY conclusions stand (game matches were always ground truth);
old R² numbers were noise. Diagnosed via identical-arrays-different-scores
+ `ETRACE_DEBUG` fill print.

## 4. Rung 3a — imbalance-polynomial family (CLOSED: neutral)

Deprioritized 200M-row data rung (more data only refines post-fit
directions already shown play-harmful); expanding the BASIS instead.
Three imbalance-polynomial features added (NPHASE 607→610):

| feature | value | λ=1e7 | λ=3e7 | λ=1e8 |
|---|---|---|---|---|
| ROOK_PAWN | rooks × pawns per side | −11.0 | −5.7 | −2.4 |
| NN_PAIR | both knights present | −27.5 | −11.9 | −4.3 |
| NB_PAIR | knight + bishop present | −28.2 | −12.0 | −4.1 |

Chess-coherent: rooks devalue with own pawns (want open files; 2R×8P side
≈ −91cp vs pawn-less rooks at λ=3e7), second-minor penalties beyond the
existing bishop-pair / knight-pawn terms.

Holdout (2.46M rows, clipped): base c0 0.4123 / λ=1e7 **0.4918** /
λ=3e7 0.4641 / λ=1e8 0.4385.

**A/B λ=3e7 vs fit6: 186-206-108 (0.480), −13.9 ± 27.0, LOS 15.6% —
neutral-to-negative, not adopted.** But the refit drifted the 1214 old
coefs too (mean |Δ| 5.3cp, max 133), conflating the feature family with
refit noise.

**Isolation A/B (fit6 coefs exactly + ONLY the 6 new values): 189-196-115
(0.493), −4.9 ± 26.7, LOS 36.1% — DEAD NEUTRAL. VERDICT: the imbalance
polynomial family (ROOK_PAWN / NN_PAIR / NB_PAIR) adds no play value at
fitted magnitudes; two independent 500g tests agree (−13.9 confounded,
−4.9 clean).** The distillation already absorbed these interactions into
the fitted material/PST terms. Features stay in the code zero-initialized
(engine identical to fit6); fitted values NOT adopted. Next rung-3
candidate: per-file shield family.

**Verify anomaly resolved (same day):** the first fitted build failed
`--verify` (3005 mirror mismatches) — the tracer's new block emitted
features but never added `FE_MG/FE_EG` to its mirror accumulator; engine
and reconstruction were correct all along. Fixed; verify 0/0 with fitted
coefs. **The solve/coefs/holdout scores were unaffected** (they use the
feature vector + SF target, never the mirror). Committed at the
feature-expansion commit with coefs zero-initialized (engine identical to
fit6 until a fit is adopted).

## 5. Rung 3b — per-file pawn-shield family (CLOSED: neutral)

The coarse shield terms (SHIELD_R1-3, one weight per rank-distance summed
across all 3 files; OPEN_KFILE/OPEN_ADJ which count ENEMY pawns as cover)
collapse the file dimension. New family (NPHASE 610→621, 2f6d78d, zero-init):

| feature | meaning | λ=1e7 | λ=3e7 | λ=1e8 |
|---|---|---|---|---|
| SHIELD_KF_R1-3 | own pawn on (king file, rel-rank r) | 1.2/2.5/1.4 | 0.4/1.4/0.8 | 0.1/0.6/0.3 |
| SHIELD_ADJ_R1-3 | own pawns on (kf±1, r) | 1.4/−2.3/0.1 | 1.2/−1.8/0.0 | 0.7/−1.1/0.0 |
| SHIELD_EXT_R1-3 | own pawns on (kf±2, r) | 6.2/2.9/1.9 | 3.2/0.9/0.7 | 1.1/0.2/0.2 |
| HOLE_KF / HOLE_ADJ | own-pawn-missing flags | −4.4/−2.8 | −2.7/−1.6 | −1.2/−0.8 |

Residual-refinement magnitudes (|0.4..3.2|cp) — the family sits ON TOP of
the existing shield weights, so only the file-resolution delta is fitted.
Chess-coherent signs where interpretable (ADJ_R2 negative = pushed f3/h3
pawns weaken the king; HOLE penalties; EG values near-zero or noise).
Gates pass all λ. Verify 0/0 both zero-init and fitted.

**Isolation A/B (fit6 + ONLY the 22 new values, λ=3e7): 183-183-134
(0.500), 0.0 ± 26.1, LOS 50.0% — DEAD NEUTRAL, not adopted.**

**PATTERN (3a + 3b): feature-basis expansion on top of the distilled basis
yields ZERO play value even when holdout R² improves.** The fitted
material/PST terms already encode these interactions; explicit features
only re-express absorbed signal. Rung-3 basis expansion is EXHAUSTED at
this data scale — the remaining HCE levers are more DATA (refit with 9×
rows at scaled λ) or structural search, not more features.

## 6. Rung 3c — 9× data refit — NEGATIVE, not adopted

1-in-66 gamepack dump → **223.56M rows** (9.05× the 22.1M), tail-clipped
(804K dropped), anchored refit from c0=fit6. Effective-shrinkage mapping
(λ ∝ N) puts the fit6-analog at λ=3e8 — shield/imbalance slots reproduced
their 22M values to 2 decimals (−5.80 vs −5.72, 3.16 vs 3.16), confirming
the scaling; the payload was the 1220 old-coef refinements (mean |Δ| 5.3,
max 134).

**A/B λ=3e8 vs fit6: 177-210-113 (0.467), −23.0 ± 26.8, LOS 4.7%.**
λ=1e8 (bigger drift, mean 9.2) untested — era-2's curve says bigger
post-fit movement loses MORE; no upside story.

**Conclusion: more data on the same basis is play-NEGATIVE at matched
shrinkage.** The distillation play-optimum is not the MSE-optimum — 9×
data moves estimates toward the true SF-eval MSE minimizer and AWAY from
whatever imperfect-agreement distance plays best (fit1's −210 at the
extreme). fit6 is a delicate sweet spot, not a step toward a limit.

**Fit-lever scoreboard (all measured):** Texel-tune existing terms DEAD ×8;
SPSA search constants NO GAIN; distillation@22M +43 (the one win); 3a
features NEUTRAL; 3b features NEUTRAL; 3c 9×-data refit NEGATIVE. The
linear-basis fit is exhausted in every direction around fit6. Remaining
HCE levers: nonlinear expressiveness (king-danger curvature — untested
axis) and search structure.

## 7. Gap analysis vs Stash/SF (2026-08-19) — the measured weakness

Fresh 500g vs StashV20: 0.501 (0.7 ± 29.5 — even), PGNs saved
(`-pgnout`; note: this cutechess rejects `-pgn`). Attribution
(`analyze_gap.py`): **75% of losses = ≥5 material swing within ≤4 plies**
(bullet punches), opening disasters 16%, endgame-conversion losses **0**
(July's #1 weakness is GONE — fit6 fixed it), flags 46-0 in our favor,
draw rate 6.6% (was ~40%).

Fixed-depth-12 duel on the 177 pre-swing loss positions (`swingduel.py`):
we repeat the game move 86%, Stash 82%, engines agree 83% — no dedicated
tactical search gap. But eval magnitudes: **stash_cp ≈ 2.02 × our_cp − 91**
(r=0.93) — our eval ~2× compressed on sharp sac positions.

**SF18 referee (depth 24, ground truth) on the same 177** (`referee.py` +
`threeway.py`): SF18 itself plays the game move 79% (razor positions, not
blunders); position at swing start **79 won / 41 balanced / 57 lost** —
45% of our losses begin from WON positions. cp MAE vs truth: **us 301,
Stash 256** (we misprice sharp positions 18% worse). Bestmove agreement:
77% vs 81%. True-blunder rate: 6 vs 6 (symmetric).

**Diagnosis: won-position conversion under fire + ~300cp eval error on
attack-heavy positions (king-danger tail mispricing).** The linear basis
cannot fix this (3a/3b proved feature-refits absorbed); the hand-shaped
au²/8 danger curve with one fitted scale is the suspect. Fix: KS_AUB8-40
cumulative attack-unit buckets (NPHASE 621→626) — the solver grafts a
learned shape correction onto the fixed quadratic. Isolation A/B gates it.

**3d RESULT: BOTH DIRECTIONS DEAD NEUTRAL, not adopted.** Fitted buckets
(λ=3e7: MG −4.7/−4.5/−2.9/−2.4/−0.7 = mild FLATTENING — the MSE-fit votes
danger overshoots in the high-AU band, opposite the play-gap prediction):
A/B vs fit6 **−4.2 ± 26.6, LOS 38%**. Sign-flipped mirror (same magnitudes,
referee-directed STEEPENING): **−11.1 ± 26.1, LOS 20%**. Conclusion:
±15cp tail corrections are ~20× too small to move the measured 300cp
decisive-regime mispricing; the KS_AUB axis is exhausted. If the
danger mispricing is real, it lives in the attack-unit WEIGHTS (pawn=2
N=4 B=3 R=5 Q=7 + check bonuses + 2-attacker gate + /4 no-queen rules),
which no fitted feature can correct — that would need game-outcome-driven
tuning of the KS accumulation itself, not eval features. Verify 0/0 both
candidates. Features stay zero-init (engine = fit6).

## 8. Rung 3d — danger-shape buckets — NEUTRAL, not adopted

KS_AUB8-40 cumulative attack-unit buckets on the king-danger curve
(NPHASE 626). Fresh 22M dump + solve + anchored fit. Fitted MG coefs came
out NEGATIVE (−4.7/−4.5/−2.9/−2.4/−0.7; cumulative ≈ −15cp at au≥40) —
the data says the hand-shaped au²/8 curve OVERSTATES tail danger, the
OPPOSITE of the SF18-referee compression prediction (under-press −273 /
under-defend +462). Data-optimum and play-relevant direction disagree.

**Isolation A/B (fit6 + only 10 bucket values): 183-182-135 (0.501),
+0.7 ± 26.0, LOS 52.1% — DEAD NEUTRAL.** Fourth consecutive fit-lever
null (3a/3b neutral, 3c negative, 3d neutral). **The eval-fit lever is
fully exhausted: fit6 is the play-optimum of the linear + hand-shaped
architecture. The measured sharp-position compression is real but not
addressable by eval refits.** Remaining levers: search structure (validated
compounding track record), time management (46-0 flag headroom), NNUE.

## 9. Rung 4a — bounded qsearch checks — NEGATIVE, reverted

The old "quiet checks removed for speed" comment (qsearch tail) predates
measurement. Re-tested with a different design point: quiet checks via the
dedicated GEN_QUIET_CHECK generator, TOP qsearch ply only (depth>=0),
capped at 3 moves, SEE-safe, not-in-check, gated on alpha<beta.
**A/B vs fit6: 181-205-114 (0.476), −16.7 ± 26.8, LOS 11.1%.** Even the
bounded variant loses: the NPS cost at bullet outweighs the tactical gain.
Reverted both ends; captures-only qsearch confirmed optimal.
(NPS IS Elo at bullet — matches the v7-bullet-baselines finding.)

### 4a audit conclusion — search playbook-complete

Full inventory vs SF15/16-era: history gravity + counter + continuation ✓;
corrhist-CORRECTED eval (6 tables) wired into futility/RFP/prune margins ✓;
improving + improving_deep + opponent-worsening ✓; rich LMR (ttPv,
cut-node, killers, checks, TT-noisy, capture-response, eval-adaptive,
hindsight) ✓; progressive aspiration + root PVS ✓; generation-aged
depth-preferred TT ✓; soft-time with best-move-stability + node-effort
dominance + score-drop extension ✓ (ideal_time initially looked unused —
wrong, it drives the stability-scaled soft-stop); singular/probcut/NMP/
IIR per works list ✓; modern qsearch ✓. SPSA already says constants
optimal. **Nothing structural remains from the shared playbook.**

**STRATEGIC CONCLUSION: Luminex shares the SF playbook with Stash on BOTH
axes — eval distilled from SF evals, search built from SF patterns. Every
adopt-the-known-pattern rung converges toward Stash: parity, not
superiority (empirically: 4 fit rungs null, search complete, 0.501 vs
V20). The ladder above ≈2512 requires asymmetric ideas OUTSIDE the shared
playbook** — next rungs come from the Innovation Protocol (fresh re-read
before use) with the SF18-referee gap analysis as the targeting instrument.

## 10. Rung 7 — logistic trainer (Grant/Texel method) — first results

Built `--texel` into evaltrace (Adam log-loss on the --solve sparse rows,
L2-anchored to fit6, holdout monitoring; committed 6f83bf9, 5e12269).
Fixed readout: static-eval MAE vs frozen refs — sharp = 177 swing
positions (SF18 d24), quiet = fixed 5K sample (gamepack SF evals).
Baseline fit6+TT256: **sharp 754.4 / quiet 199.9**.

**Control (σ(eval) labels, 24.9M rows, 40 epochs): REFUTED the clean
loss-mismatch hypothesis.** Readout sharp 764.8 / quiet 169.1 — logistic
loss IMPROVED the mid-range 31cp and shrank the tail harder. Mechanism:
log-loss on σ(eval) is near-invariant to monotone rescaling → cp scale
drifts free (material values walked −31..−58cp; fit1's gauge disease via
the loss function). No A/B (gate failed). Lesson: any σ-link fit needs
gauge/scale anchors as hard as the ridge fits had.

**Outcome fit v1 (890K rows from V35-vs-V36 corpus, L2 1e-7): FAIL**
(sharp 751.3 / quiet 211.7 — quiet regressed). **v2 (L2 1e-8, 200 ep):
outcome logloss 0.5469→0.5363 (holdout clean, no overfit) but readout
WORSE both axes (757.5 / 236.9).** Natural-distribution outcome labels
inherit the quiet-majority drag — the SAME imbalance as MSE distillation.

**v3 (FULL corpus: 3.31M rows from 23,008 V35-vs-V36 games, tension
row-weights WSHARP=2, anchored L2 1e-7, holdout 66K clean): outcome
logloss 0.5387 — and the A/B: 179-230-91 (0.449), −35.6 ± 27.7, LOS
0.6%. DECISIVELY NEGATIVE.**

**RUNG-7 VERDICT: the Grant/Texel outcome-tuning recipe does NOT
transfer to a baseline already at its play-optimum.** Stash v21's +205
harvested low-hanging fruit from badly-tuned hand values; fit6's
neighborhood punishes even small eval-space movement (mean drift 1.7cp
→ −35.6 Elo; cf. 3c's −23 at matched shrinkage, era-2's "any movement
loses"). The eval-parameter optimum is SHARP. Infrastructure preserved:
--texel trainer (tension-weighted, holdout), 3.3M-row strong-teacher
outcome corpus, readout harness — reusable for search-side outcome
learning or regularization studies.

## 11. TT 256MB default — ADOPTED (c5637e7)

Config-only probe after five dead rungs: Hash=256 vs 128. Self-play A/B:
**203-179-118 (0.524), +16.7 ± 26.7, LOS 89%**. Ladder confirm vs
StashV20: **233-224-43 (0.509)** vs the 128-era 0.504 — consistent. The old
"256 too much cache pressure" comment in main.cpp was an unmeasured
assumption. Default bumped 128→256 (main.cpp + uci option default).

## 12. Adjacent negative results (isolated eval-term changes)

- King-safety isolated tweaks regress (quadratic-KS −25.7, coordinated
  redesign −40, king-activity −30, king-defender −13 vs Stash) — but the
  4× scale fix (/8→/2) was NEUTRAL head-to-head (144-149-107, 49.4%, 400g);
  the apparent regressions were the stale-tuning confound. Retune WITH the
  fixed scale remains untested (superseded by distillation).
- Eval-diff move ordering −12, 2× mobility scale + 2D phase −100,
  TT-eval upgrade −153, forcing-line fast-eval −249 (full list in git log).

## 13. Standing lessons

1. **Never judge a fitted eval by R² or spot-evals — game-match it.**
2. Anchored ridge only; sanity gates (piece values in range, mobility
   centered, PST std ≤ ~3×) BEFORE any 500g match.
3. Clip targets (|eval| > 1000cp) in every future fit.
4. λ ≈ 3e7 is the play-optimum regime for this data at 22M rows.
5. Small data-directed corrections win; big reallocations lose.
**Qsearch TT-before-stand-pat (Stash's +35.7 patch transplanted): 171-212-117
(0.459), -28.6 +- 26.7, LOS 1.8% — NEGATIVE, reverted (b11dee6).** Our qsearch
already fronts a 64K eval-cache serving the same skip-the-eval role; the TT
probe adds cost without new benefit. LESSON: even proven Stash patches do not
transplant — codebase context dominates patch value.

## Session summary (2026-08-19/20 night)
12 ideas measured, 1 adopted (TT 256MB, +16.7 self-play / 0.509 ladder).
The eval-parameter optimum is SHARP: every refit (data-scaled, logistic,
outcome-weighted) and every transplanted patch reads negative. At 500g
screening precision (<+20 invisible), the measured base rate of candidate
ideas is 1/12. The binding constraint is VERIFICATION SCALE, not idea supply:
Stash compounded +890 via ~150 patches at 1.5K-90K-game SPRT each.
NEXT LEVER: SPRT batch harness (box does ~7K games/h; LLR early-stop),
then patch pipeline at that precision.

## Rung 10 — Stash eval PORT (the whatever-it-takes path) — IN FLIGHT

After ~25 zero-verdict ideas and the falsification of every local lever,
the strategic pivot: **port Morgan Houppin's Stash eval verbatim**
(GPL-3.0, proven 3399-ladder HCE, our-license-compatible) into our
search behind UCI "UseStashEval". ~930 lines translated (59abb7c):
PSQT + all piece terms with their tuned mobility tables, quadratic-MG
king safety, storm/shelter, threats, passed king-distance tables, EG
scaling, KXK, pawn-structure hash.

Status:
- Faithfulness: **depth-1 differential vs real Stash V36 on 1500 shared
  positions: MAE 129, p50 89, p90 256** — the core is ported correctly
  (disagreements concentrate in KXK mate endings, VICTORY scale).
- Speed: initial port 1.39M NPS (loop-built bitboards per call!) —
  fixed to 1.71M (f404794); fit6 = 2.17M. Remaining gap ~21% (~-37 Elo).
- Bullet smoke (tc=1+0.01): 0.370, then 0.345 after speed fix —
  -92 to -111. With -37 explained by NPS, the residual ~-60 is the
  SEARCH-MARGIN SCALE MISMATCH: our futility/razoring/aspiration are
  calibrated to fit6's cp scale; Stash's EG scores live at 2x scale.
- Isolation test IN FLIGHT: fixed-node 100K/move smoke (NodesPerMove)
  splits eval-quality from speed. Then: margin retune (their
  search_params.c) or NPS recovery per its verdict.

**SPRT era (harness 61442e4, validated live against the known Hash-128-vs-256
**SPRT-era scoreboard (phase 1 — parameter probes at ±5 precision):**
- mtg 25→20 (more time/move): **H0 in 1,500 games** — the stability-reduction
  TM already spends time optimally.
- eval-cache 64K→256K: **H0 at ~3,000 games** (−7 trend — worse locality).
- Non-PV extra LMR reduction (Stash v26's +5.1): **H0 by 16K-game cap**,
  −1.5 ± ~3 — exactly neutral; our LMR tables already calibrated.

Three rejections, zero acceptances. **Parameter space is saturated,
now confirmed at ±5 precision (SPSA's earlier ±27 claim reproduced with
sharper tools). Transplant base rate: 0/3 for external patches.**

**drop2x (score-drop extension x1.5 -> x2): first SPRT run +6.6 +- 4.5
at 12K games (LLR +2.09, one short of accept at elo1=5); confirmation
run resolved to ~+1.6 at 10K (LLR -0.13). Pooled ~+2-3 over 22K games —
NOT adopted.** The replication gate caught what the first run would have
claimed. Sharp-root extension (aspiration-widening-aware time spend,
59c1afe): SPRT resolved +3.7 at 12K cap, LLR +0.62 — same small-positive-
unresolved profile. Combined (drop2x + sharproot) SPRT: +4.3 at 14K when
the box dropped; definitive 40K-game run (combfinal) relaunched.

## Rung 8 — overloaded-defender features (protocol v26 D-derivation) — NEGATIVE

Full-protocol run on "where does the next Elo live for a saturated HCE":
Phase 0 assumption graph post-falsification (A1-A5 dead; A8 "eval = purely
static features" the surviving upstream flip) → Pathway D derivation:
"compressed micro-proofs" — features encoding 2-3-ply tactical facts.
Triage survivor: OVERLOAD2/OVERLOAD3 (sole defender of 2+/3+ attacked
own pieces; the punch corpus's dominant motif by duel-FEN inspection).
Implemented (NPHASE 629), mirror-verified 0/0, fitted anchored λ=3e7 on
24.6M rows. Fitted signs INVERTED vs chess theory (MG -6.6/-5.9 through
the feature's negative-sign convention = overload correlates BETTER for
the owner — an activity confound: pieces that defend things are developed
pieces). **Isolation A/B: 186-205-109 (0.481), -13.2 +- 27.0, LOS 17% —
neutral-to-negative, NOT adopted.** Features stay zero-init. D-derivation
#1 complete per protocol 7.6: the anomaly-driven micro-proof family, in
its first instantiation, does not convert; whether the FAMILY has other
instantiations (discovered-attack potential) remains open but priors are
now low.

**Rung 9 (material-simplification root tiebreak, the A6 opponent-stateless
flip): SPRT resolved -1.4 at 12K cap, LLR -1.98 — NOT adopted, reverted
(d3ddb62).** The GM trade-when-ahead rule as a root decision overrides the
search's genuine preference among ties and loses ~2-4 Elo. Tenth SPRT-era
verdict: zero adoptions.

**PORT REVERTED (2026-08-24, user directive: self-derived only).** The port
survives only as DIAGNOSTIC FACTS: their eval + our search (scale-aligned)
beats our eval at 100K fixed nodes +56 +- 67 — a real ~50-100 Elo eval-knowledge
gap, now the target for SELF-DERIVED fixes. METHOD POSTMORTEM (why 25+ attempts
read zero): (1) granularity — tested families/refits, never single-term
micro-patches; (2) anchoring — every fit L2-pinned to fit6's basin, never
allowed to travel; (3) precision — 500g/12-16K gates cannot see +3-8 Elo
(fishtest uses 40-100K+). Corrected methodology: weakness map (fit6 vs real
Stash static-eval differential by position class), then micro-patches at
fishtest-scale SPRT precision, unanchored basin exploration allowed.

## WEAKNESS MAP (2026-08-24, fit6 vs Stash V36 static-eval differential, 4000 gamepack positions)

| regime | our SF-error | V36 SF-error | gap |
|---|---|---|---|
| MG | 155 | 142 | -13 (parity) |
| MG-EG | 286 | 302 | +17 (we win) |
| EG (npm<10) | 758 | 518 | **-240 THE GAP** |

Root cause candidate: only 7% of the 24.9M training rows are EG — the EG
terms were fit on 1/14th of the MG data. Not saturation: STARVATION.
First self-derived fix: EG-upsampled refit (EG rows x5, streamed solve,
~28M rows) in flight.
