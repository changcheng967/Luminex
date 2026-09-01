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

### Depth-1 control (artifact isolation): EG gap corrected

Our depth-1 (qsearch-inclusive) eval vs V36 depth-1 on the same positions:
EG bias -276 -> -1.1, disagreement 646 -> 413, sfMAE 758 -> 687. So ~230cp
of the apparent gap was static-vs-depth-1 artifact; REAL depth-1 EG gap =
687 vs 518 = ~170cp, still THE weakness (MG dead parity 141 vs 142).
EG-upsampled refit (50.26M rows) FALSIFIED data-starvation: EG sfMAE
unchanged (757.7). Remaining attack: EG RESOLUTION — qsearch depth-4 cutoff
(our qsearch already recovers 70cp in EG at depth 1) and EG expressiveness.
First micro-patch under test: qsearch depth -4 -> -6 at fixed nodes.

**qsearch depth -4 -> -6 (global): 36-41-23 (0.475), -17.4 +- 60.3 at
100K fixed nodes — NEUTRAL, reverted.** Global deepening pays NPS cost
everywhere for EG gains that wash out. Next: EG-CONDITIONAL qsearch depth
(extend only when npm < 10) + EG-passed-pawn/king-activity expressiveness
micro-patches at 40K-game SPRT precision.

### EG-type decomposition (1732 EG positions, 25K sample, depth-1 both engines)

Top rows (/R +10151, /NB +7560) are SCORING-SCALE ARTIFACTS: SF scores
won endgames at +-3000..9000; both engines score VICTORY-scale (~+10000
V36, ~+15600 us); the error gap is convention, not ignorance (verified:
true KRK we score +15486 correctly; corpus /R example SF=-3096).
REAL weaknesses: **RP-vs-R endings +651** (the conversion ending) and a
consistent **+50..130 deficit across ALL minor-piece ending types**.
Next self-derived targets: (1) rook-endgame scaling for RPvR (one-sided
pawns drawishness), (2) minor-ending EG terms (N/B EG PST refinement).

**Rook one-wing scaling (classical rule, targeted at the +651 RPvR gap):
ACCEPT H0 in 1,500 games — -51.5 Elo, LLR -3.30. REVERTED (a62133d).**
The rule as implemented HURTS: our engine was already converting these
endings well enough, and blunting its EG scores there loses games. The
+651 static-eval gap vs V36 does not translate into a play gap — V36
may simply be scoring its own conversion intent differently.
Lesson repeated: eval-agreement gaps are NOT play gaps (3rd time:
danger compression, outcome-logloss, now RPvR static error).

## A1: unanchored outcome retune — NEGATIVE, the objective space is CLOSED

The audit's never-pulled lever (engine_audit.md A1): outcome-logistic,
L2=1e-9 (no anchor), 120 epochs on 3.3M strong-teacher outcome rows,
holdout clean throughout (0.5379, zero overfit), genuine travel from
fit6 (mean drift 2.1cp, max 6.7cp). A/B vs fit6+TT256: 176-217-107
(0.459), **-28.6 +- 27.1, LOS 2% — decisively negative.**

**This completes the falsification of the ENTIRE tuning-objective space:**
MSE-to-SF-evals (8x), sigma(eval)-logistic, outcome-logistic anchored
(3 variants), outcome-logistic UNANCHORED with correct methodology —
every objective, data source, and constraint configuration lands at or
below the same play value. The linear function class on this feature
basis is at its play ceiling (~2515). This is a property of the
architecture, not of any tuning method. Remaining self-derived levers
per the audit: A3 (uncertainty-as-depth-allocation) and structural
search innovation.

**A3 v1 (uncertainty-as-depth: -1 LMR when eval_uncertainty > 120):
ACCEPT H0 in 13,500 games — -3.1, LLR -3.17. REVERTED (6c0d295).**
The extra depth at high-uncertainty nodes costs what it gains; the
correction-history signal is already fully exploited via margin
modulation (futility/reverse-futility/LMP adjustments). A3 v1 joins the
ledger as the 6th gap-targeted rejection.

**A3 v2 (uncertainty-gated pruning disable: skip futility+rev-futility
when eval_uncertainty > 150): H0 (combined with v1's record at 13.75K
games, LLR -3.21). REVERTED (26ee0ca).** Both doses of the
uncertainty-as-search-allocation idea are dead: the correction-history
signal's play value is fully expressed through margin modulation alone.
The uncertainty axis is CLOSED.

## Rung 11 — self-play outcome fit (the exact Houppin recipe, our own games) — DEFLATED TO H0, not adopted

The last untested tuning configuration: 10K games our-engine-vs-our-engine
(the position distribution our search actually visits), 1.44M positions,
outcome-logistic via --texel, 150 epochs, holdout clean throughout
(0.5818), genuine travel (drift mean 1.17cp / max 4.4cp), mirror-verified
0/0. 40K-game SPRT vs fit6+TT256: started −19 → peaked **+4.7 Elo
(LLR +2.03) at n≈23K** → deflated steadily to **+1.6 (LLR −1.20) at
n=33.5K** when the box rebooted (killed before formal termination). The
same mid-run-inflation-then-deflation profile as drop2x; at the cap it
reads H0. NOT adopted; luminex_spfit binary shelved.

**This closes the "iterative self-play re-centering" hypothesis for our
baseline.** Houppin's V21 +205 harvested the gap between an UNTUNED hand
baseline and its outcome-optimum — a one-time harvest. Our baseline is a
least-squares endpoint already (fit to SF depth-26 fishtest search scores),
so the outcome gradient at fit6 is ~zero: five independent
outcome-tuning configurations (anchored ×3, unanchored, self-play) now
all land at or below zero. The recipe was never magic; the headroom was.

## E0 — model-class oracle (2026-08-28): interactions help a little, and only in sharp positions

Question: is the linear basis the binding constraint, and how much would
unrestricted nonlinearity buy on OUR features? Measurement: dumped
2.46M rows (features + SF-d26 targets) from the exact engine basis
(evaltrace --dump-rows, sample of fens22m), then compared on a shared
245,965-row holdout: (a) ridge in the ENGINE's model class
(MG*wmg | EG*weg | ph), (b) LightGBM (63 leaves × 800 trees —
interactions + thresholds unrestricted). Pre-registered rule: ΔR² <0.05
basis information-poor; >0.10 wholesale interaction refit; between →
selective families.

| model | holdout R² | MAE |
|---|---|---|
| linear (engine class) | 0.5754 | 97.4 |
| GBM (unrestricted) | 0.6472 | 87.5 |

**ΔR² = +0.072 → the middle band: selective interaction families, not a
wholesale quadratic refit.** Bucket MAEs show WHERE: quiet (|y|<100)
75.6 → 68.6 (−9%), 100–300: 83.5 → 76.5 (−8%), 300–600: 191.7 → 165.9
(−13.5%), 600–1000: 309.4 → 271.1 (−12.4%) — the interaction advantage
CONCENTRATES in the sharp/decisive regimes, exactly the measured play
weakness (razor mispricings). Equally important: **even unrestricted
nonlinearity on current features caps at R²≈0.65 — ~35% of SF's eval is
not representable by ANY model on this feature set.** That residual is
missing relational information (SF11-audit families we lack: per-type
safe checks, passed-pawn path-safety states × asymmetric king
proximity). Next rungs, in order: (E1) selective sharp-regime
interaction families + wholesale refit through the full gate ladder;
(E2) missing relational families (self-implemented) + refit. R² is not
play — both go through 500g → 40K SPRT before any adoption.

Infra notes: chunked parse (oracle_chunk_*.npz kept, raw dump deleted);
the single-process version OOM'd the box twice — all heavy jobs now
ulimit-capped, nice'd, n_jobs≤4 (see MEMORY.md box-crash triggers).

## E0b — target-metric calibration (2026-08-28): R² vs d26 targets SATURATES at ~0.5 for everyone, including SF18

Prompted by "how do we get to R²=1.0": scored SF18's own NNUE eval (the
strongest eval in existence) against the SAME fishtest-d26 targets our
distillation trains on — 45,721 shared non-check positions, pairing
validated by shifted-controls ≈ 0 (the in-check rows had been silently
shifting all pairings; "Final evaluation: none (in check)").

| eval | R² vs d26 targets | MAE |
|---|---|---|
| SF18 raw NNUE (plays ~3600) | **0.4789** | 150.3 |
| our fit6 (same rows) | 0.3802 | **117.8** |
| our linear α300 ridge (E0 holdout) | 0.5754 | 97.4 |

**Reading: the best eval in the world agrees with raw depth-26 search
scores only ~0.48 — barely different from ours. Our distillation is
ALREADY at (or beyond, by MAE) the world's static-agreement level. R²
against search targets is a SATURATED metric: it cannot distinguish us
from SF, and pushing it toward 1.0 asks the eval to statically duplicate
what the search computes dynamically — which SF's own design avoids and
which our own 3c falsification measured as play-NEGATIVE (9× data moved
the fit toward the true MSE optimum of this exact target and lost 23
Elo).** Note the signature: SF18 wins squared error (150.7 RMSE vs our
164.4) while we win absolute error (117.8 vs 150.3) — our compression
buys small bulk errors at the cost of the heavy decisive-regime tails
(the measured 300cp sharp-position mispricing). The play-relevant axis
is TAIL STRUCTURE, not R². Consequences: E1 (interactions for +0.07
R²) is DEMOTED — its gain is on a dead metric. E2 (new relational
features) stays plausible but must be justified and gated by play
exclusively. The 1000-Elo gap to SF lives in tail structure, search-eval
coevolution, and feature information — none of which this metric sees.

## E0c — cross-engine static calibration (2026-08-28) — CORRECTED same day: convention error in band joins

The original E0c band analysis (864cp decisive-band MAE, slope ~0,
"our static does not express decisive magnitude") was WRONG: the
tracer's mirror/static field is STM-perspective, and the band joins
never flipped it by side-to-move — every black-to-move row (half the
corpus) was sign-inverted, destroying within-band slopes. Corrected
band stats (statics flipped to white-POV; overall rows unaffected —
--score/eval-command paths flip correctly internally):

| eval | overall R² | <100 slope/MAE | 300-600 slope/MAE | 600+ slope/MAE |
|---|---|---|---|---|
| SF11 HCE (~3500) | 0.113 | 0.06 / 104 | 0.73 / 221 | 0.96 / 313 |
| our fit6 static | 0.319* | 0.17 / 107 | **1.19 / 174** | **1.32 / 252** |

(*overall r on this sample; the --score R²=0.38 uses the exact
prediction path.) **Corrected findings: (1) our static DOES commit in
decisive bands — slope 1.19-1.32, mildly OVER-committed vs SF11's
0.73-0.96 — and its in-band MAE (252 @600+) BEATS SF11's static (313).
(2) Our shape mirrors the strong-HCE philosophy: quiet-flat (0.17 vs
SF11's 0.06), decisive-committed. (3) Therefore the 1000-Elo play gap
is NOT explained by decisive-band static calibration vs d26 — the
eval-side hunt must target the RAZOR class (75% of losses = quiet-sac
punches; E0 oracle's interaction gain concentrates exactly there) and
search, not EG-decisive scaling.** Depth-1 harness also abandoned (TT
pollution + unresolved reporting; no conclusions drawn). The one real
artifact found while chasing this: tracer mirror is stm-POV —
documented; every future join must flip by stm.

## Rung E2a — unanchored-EG-block refit — REJECTED (decisively negative)

Built during the (mistaken) 864cp hunt: solve8.bin re-solved with MG
anchored (λ=3e7) and the EG block freed (λ_EG sweep 3e7→1e4, rank-1
gauge anchors kept; EG drift mean 5→77cp, gates PASS, EG-PST-std ≤
2.79×). λ_EG=1e6 candidate read out on corrected bands: quiet MAE
107→81 but 300-600 MAE 174→202 and 600+ 252→284 with slope rising
further over-commitment (1.32→1.46) — bulk-MSE-better/tail-worse, the
classic non-converting pattern. Play A/B 1500g tc=1+0.01 vs
fit6+TT256: **495-695-310 (0.433), ≈ −47 Elo, LOS≈0 — REJECTED.**
Seventh refit configuration, same verdict: the coefficient space
around fit6 is play-exhausted in every anchoring geometry. Candidate
binary deleted; coefs kept in sftest/dump for the record.

## Rung E2b — razor-class interaction features (HANGAU/THRAU/MATAU) — SPRT in flight

First INTERACTION family (products, not new predicates): the E0 oracle
measured interactions buying −13% MAE exactly in the |y|>300 bands, and
the linear basis cannot express products. Three features (NPHASE
629→632, zero-init committed 00758f2, --verify 0/0), emitted in the KS
block (attacker≥2, au≤48), mirrored term-for-term in the tracer:

- RAZOR_HANGAU = enemy-hanging-count × min(au,48)/16
- RAZOR_THRAU = enemy-pieces-attacked × min(au,48)/16
- RAZOR_MATAU = clamp(matmargin,±400)/100 × min(au,48)/16

Fresh 22.14M-row solve (run_v9.bin, c0 remapped 1258→1264, anchors
remapped, gen_fitted patched for 632). Fitted at λ=3e7: MATAU MG
−10.07 (CHESS-COHERENT: under attack by a side that SACRIFICED →
~60cp extra danger at AU40/300cp-sac — the initiative is real;
attacked by the material-ahead side → less incremental danger, base
already prices material). HANGAU −3.68 (rung-8-style activity confound
sign). THRAU ≈ 0. λ=1e7 dose ≈ 1.5× magnitudes.

Isolation A/Bs (c0 + ONLY the 6 new values), 500g tc=1+0.01 vs
fit6+TT256:
- λ=3e7: 195-197-108 (0.498), −1.4 — dead neutral.
- λ=1e7: 196-184-120 (0.512), +8.4 ± ~28 — maybe.

**40K SPRT (elo0=0, elo1=5) on the λ=1e7 dose launched — verdict
pending.** Coverage limitation noted: the ±500 early-exit skips the KS
section, so the features do not fire on already-decisive positions;
the family prices the pre-decisive razor band only.

**VERDICT: ACCEPT H1 at n=27,500 (10,501-10,149-6,850, 0.506, +5.1,
LLR +3.00) — ADOPTED (94896af).** Trajectory: +28.8 (n=250) → 0.000
(n=6.25K) → +3.4 (n=18.25K) → accept — the middle-run dip was noise,
the effect held. **First eval-side adoption since fit6; first
acceptance of the SPRT era after ten rejections; new baseline ≈2520.**
The interaction-feature axis is VALIDATED as adoptable — the first new
adoptable axis since the distillation. Immediately stacked: LMP P2
probe (changelog-mined structural aggressiveness) on the new baseline.

**LMP P2 (threshold base + ¾·d²) vs razor baseline: ACCEPT H0 at
n=5,000 (−7.5 trend at cutoff, LLR −2.94) — REJECTED, direction
closed.** Dose-response logic: a 25% threshold cut at depth 5 lost
decisively; the milder P1 (4% cut) cannot reach +5 from there, and
the conservative direction costs NPS with no evidence. Our LMP is at
its play-optimum — consistent with the earlier SPSA null on scalars;
Stash's +24-class LMP gains were theirs, not ours (transplant base
rate now 0/6). Changelog-mined follow-up: mobility-zone refinement
(+19.95 theirs) — eval-side, own-value change, goes through the game
gates directly.

## Rung E3 — mobility-zone refinement (rammed/low-rank exclusion) — SPRT in flight

Zone now also excludes squares covered by own RAMMED pawns (enemy
pawn directly ahead — can never advance) and own undeveloped pawns on
the two lowest ranks: those squares are not real maneuvering space.
Mirrored term-identically in the tracer (verify 0/0 over 45,721).
This is a feature-VALUE change on existing coefficients (no refit):
500g A/B vs razor baseline: 190-184-126 (0.506), +4.2 — the same
neutral-positive screen profile the razor family showed before its
SPRT accept. 40K SPRT launched. If it accepts, a re-centered refit on
the new zone definition is the natural amplifier (their +19.95 came
tuned, not raw). Box incident note: the first A/B launch after the
12h restart silently ran razor-vs-razor (venv cmake missing, pipeline
exit masked the failed build) — killed, cmake reinstalled, md5
verified before relaunch; add pip-cmake to the restart checklist.

**VERDICT: ACCEPT H0 at n=31,250 (11,710-11,692-7,848, 0.500, +0.2,
LLR −2.96) — REJECTED, reverted both ends.** The +36.8 opening and
+4.2 screen dissolved to exact zero over the full run. Transplant
base rate 0/7. The raw zone change carries no play value here; the
Stash +19.95 was in THEIR feature geometry, tuned.

## Rung E5 — one-queen KS trigger — neutral; box rebuild; bullet-ladder recalibration

**Volume-swap incident (2026-08-30):** the 12h reprovision attached a
WRONG volume (foreign data in /hyperai/home; all box artifacts gone:
Luminex tree, binaries, fens22m.txt, dumps, cutechess, sprt.py).
Nothing irreplaceable lost — every adopted state lives in git. Box
rebuilt from scratch: clean clone (main = razor baseline), cutechess
1.5.1 built from source (needed qt6-svg-dev + fresh apt index), sprt.py
restored from nnue/openi_upload, openings book rebuilt by concatenating
the 23K corpus games (ECO-diverse).

**QKS (lone-queen king-safety trigger): 183-185-132 (0.498) — dead
neutral, rejected.** Implementation lesson (the second of its kind):
the first version was DEAD CODE — `enemy_qu` is consumed by the
pop_lsb loop before the gate; compiler DCE'd it; binaries were
byte-identical and only a 45K-position eval-diff (645 changed) proved
the fixed version alive. **Mirror-verify alone cannot detect
behavior-null changes — always eval-diff against base before any
match.**

**Bullet-ladder recalibration (major measurement correction):** on the
rebuilt fast setup (500 bullet games ≈ 70s at 7-way), luminex-razor:
vs Stash V20: 191-185-124 (0.506, 500g); vs Stash V21: 765-779-456
(0.496, 2000g, ±14) — **bullet-parity with V21** whose LTC ladder is
2714. V20 and V21 both read "even" → the +205 V20→V21 LTC gap largely
does not exist at 1+0.01 (eval-knowledge gains need depth). The
standing convention (bullet-parity ⇒ ladder-Elo ≈ opponent's) gives
"≈2714" but that is a CONVENTION, not a measurement: our true
blitz/LTC Elo remains unmeasured; an honest ladder claim needs an LTC
match. Note also: matches on this hardware/book draw at 23-24%
(vs 8.6% on the old book) — book sensitivity is real; all future
ladder numbers quote the book.

**Fast-harness era + two more probes (same day):** the rebuilt setup
runs 500 bullet games in ~70s (vs ~8 min before) — screens are now
1000g, SPRTs ~90 min at conc 7. Aspiration-disable at |score|>2000
(Stash V30 idea): 379-378-243 (0.500) at 1000g — neutral, rejected.
Early-exit threshold 500→650 (extends KS/razor coverage into more
decisive positions; 516-position eval-diff = provably alive): 373-374-
253 (0.499) — neutral, rejected; the razor family's ±500 coverage
limitation costs nothing. Transplant base rate 0/8.

**Pins-in-mobility (ray-accurate: pinned pieces count only king-sniper-
line squares; 608-position eval-diff, mirror 0/0): 368-398-234 (0.485)
at 1000g — REJECTED.** Chess-correct, but the per-side pin scan's NPS
cost outweighs the precision at bullet. Note: the first A/B of this
probe was INVALID (stale binary — the build failed with venv-cmake
missing and the pipeline masked it AGAIN); hard gate added: matches
only launch when candidate/base md5s DIFFER. Base rate 0/9.

**Box incident closure (2026-08-30):** the "volume wipe" was a
mis-routed port — port 30616 was someone else's instance (foreign
project data); the real box (new port) has the original volume intact
(adopted binary, fens22m, run_v9.bin, all tooling). All my artifacts
removed from the foreign instance; their data untouched.

## Rung E6 — THE LTC INVERSION (2026-08-30): bullet parity is a bullet number

The autonomous LTC ladder (tc=10+0.1, 300g, old book): **V20 beats us
97-172-31 (0.375) ≈ −88 Elo.** The bullet parity (0.504-0.509 at
1+0.01, three eras) INVERTS at LTC. Every ladder estimate in this
ledger ("≈2512", "bullet-parity with V21") is a bullet-convention
number; on the LTC clock we sit ≈ V20 −88 (ladder-equivalent ≈2420).
Mechanism question (fixed-node 100K/move match vs V20 in flight):
if we also lose at equal nodes → knowledge/search-quality gap (eval
rungs, and our bullet-tuned search constants may be mis-tuned for
LTC); if even at equal nodes → the deficit is depth-per-second (NPS)
and time-allocation, i.e. speed rungs. STRATEGIC COROLLARY: the 3300
target must name its clock — CCRL-style blitz/LTC is the standard;
gates at bullet select NPS-favoring changes. All SPRT-era adoptions
(razor +5.1, TT256 +16.7) are bullet-verified; they should be
LTC-confirmed before being treated as ladder progress. V21 LTC leg
re-running (binary was on the wrong instance).

**V21 LTC verdict: 58-216-26 (0.237) ≈ −181 Elo (n=300).** LTC ladder
under the (sharp, low-draw ~10%) old book: V20 −88, V21 −181 —
LTC-equivalent ≈2420 by the local V20 anchor. Cross-check tension:
through us, V21−V20 ≈ 112 vs the ladder's 205 (inside combined noise
at n=300 each, sharp-book caveat). The knowledge-vs-speed split still
needs the per-FEN fixed-node duel (both engines driven by `go nodes`
— cutechess's NodesPerMove option was the failure, NOT go-nodes which
every UCI engine honors; referee methodology from the gap analysis
applies). NOTE: fixed-node via cutechess option is INVALID for
Stash (no such option advertised — it played 60s/move in that
configuration; that match was killed, its 0-4 partial discarded).

## Rung E4 — post-adoption re-centered refit + MATAU dose — both negative, family calibrated closed

Re-centered solve (run_v9.bin, c0 = adopted razor coefs, λ=3e7,
gates pass, drift mean 5.02): the fit wants MORE MATAU (−15.47 →
−20.81) with the usual global wobble. **Whole-refit A/B: 176-208-116
(0.468), −22.2 — rejected at screen, era-2 law holds on the new
baseline too.** Targeted follow-up: MATAU-only at the refit value
(−20.8, all else adopted): **186-201-113 (0.485), −9.8 — the adopted
−15.5 dose is the play-optimum; escalation loses.** The razor family
is now fully calibrated: values adopted, dose confirmed, direction
closed. Ledger state: 1 adoption (+5.1 razor), 3 rejections this
batch (LMP P2 H0@5K, mobzone H0@31K, re-center −22.2@500g).

## Rung E7 — the NPS gap, measured and mapped (2026-08-31): the LTC deficit is 32-35% engine speed

Depth bench (10s/move, 10 positions, same box): **luminex 2.18M NPS
vs stash-v20 3.18M / v21 3.37M** — we are 32-35% slower; per-position
depths lag 0-5 plies (worst: rook endings, quiet openings). At bullet
the stronger-per-node eval compensates; at LTC the depth deficit
compounds — this IS the −88/−181 inversion. Speed math: +1.5x NPS ≈
+40-60 Elo at LTC; closing to 3.2M+ is the single biggest measured
lever on the board.

Profile map (callgrind breaks on this engine; gprof at -O2 has
LTO-mushy attribution — and profiling builds MUST pass -DNDEBUG or
asserts dominate and lie: a 30% fake assert cost burned an hour).
True release spread: generate<LEGAL> + legal() + attackers_to + see
≈ 14-18% (pseudo-legal generation + pin-mask legality would cut most
of it); **set_check_info runs PER MOVE, not per node** (12.9M calls
≈ 8% — pinned/checkers need computing once per node, ~10x fewer);
122 g_stats counter sites ≈ 2-4%; evaluate 7.4% self at 27% of nodes
(fine); do/undo ≈ 5% (fine). No single smoking gun — the gap is many
structural overheads. The NNUE-accumulator-destructor 22% reading was
an LTO attribution ghost; all NNUE guards verified correct in code.

**Attack plan (NPS rung): N1 = set_check_info per-node (+~8%
target), N2 = pseudo-legal staged movegen + pin-mask legality
(+10-15%), N3 = stats-counter gating (+2-4%). Each gated by: bench
NPS delta + depth-bench + 1000g bullet A/B + 300g LTC confirm.
Validated en route: our `go nodes` works (fixed-node duel methodology
available for knowledge-vs-speed splits); valgrind/callgrind fail on
this engine — use gprof -DNDEBUG or differential builds.**

## Rung E7 outcome — N1/N2 speed complex REVERTED; measurement-precision lesson

The NPS attack's first two candidates failed verification and are
reverted (9933905): (1) lazy pins (+3.1% in one session) screened
+11.2 at 1000g but read −6.7 at SPRT n=6,000 — the null-pins "fix"
costs more than the speed buys; (2) the preserved-semantics variant
and pin-mask-legality-alone BOTH still produce depth-16 trees that
differ from baseline (unexplained; perft blind to it) — disqualified
under the verification discipline regardless of measured speed.
**Critical methodology discovery: the box is a noisy-neighbor VM —
the SAME baseline binary measured 1.66M/1.69M/2.18M/2.23M NPS across
sessions (±30%). Small NPS deltas are UNMEASURABLE on this hardware
without interleaved A/B/A discipline; only the original interleaved
depth-bench (us 2.18M vs Stash 3.18-3.37M, one session) stands.**
The 32-35% speed gap vs Stash is real but must be attacked with
interleaved measurement and tree-identity proofs (fixed-depth node
counts — now a standard gate; it caught every divergence perft
missed). Speed rung suspended pending a methodology that fits the
hardware's variance floor.

## Rung E8 — the verified gap decomposition (2026-08-31): where the ~800 lives

E1 time-handicap ladder (Stash V20 fixed at 10+0.1, us at 10/15/20s,
300g LTC each): **0.642 / 0.560 / 0.505 — Stash's 10 seconds equals
our 20.** Linear in log-time (+53, +50 per ×1.5). E2 equal-nodes duel
(800 positions, both engines exactly 300K nodes, SF18-d24 referee):
truth-agreement **lux 39.5% vs stash 41.7%**, head-to-head 45.3%.

**Decomposition of the −98 LTC vs V20 (both methods agree):**
- SPEED FAMILY ≈ 53 Elo: raw NPS 1.47x deficit (~30) PLUS
  nodes-to-depth efficiency 1.36x (~23) — their tree extracts more
  depth per node (selectivity/pruning/ordering), on top of raw speed.
  (Time-equivalence 2.0x ÷ NPS ratio 1.47x = 1.36x per-node
  efficiency gap.)
- KNOWLEDGE/eval-at-depth ≈ 45 Elo: the E1 residual at speed parity
  matches the E2 referee gap (2.2pp bestmove agreement ≈ 33-45 Elo).
- Scaling to V21 (−181): ≈ 100 speed-family + ≈ 80 knowledge.

**Closure priorities by measured size: (1) NPS grind (~30) — many
small verified wins, needs interleaved measurement; (2) per-node
selectivity (~23) — LMR/ordering quality at deep plies, the LTC
counterpart of our bullet-tuned constants; (3) knowledge (~45 vs V20)
— razor-style interaction rungs at ~+5 each, or a bigger eval
structure. Rough full-gap projection to 3300: ~350-400 speed-family,
~350-400 knowledge, remainder beyond-V21 extrapolation.**

## Rung E8b — FINAL decomposition: the entire LTC gap is middlegame NPS

Follow-ups that closed the system: (1) nodes-to-depth curves are
EQUAL (lux/stash 0.74-1.05x; we are 2x LEANER to depth 14) — the
per-node selectivity hypothesis is DEAD; (2) referee duel at 5M nodes:
truth-agreement 43.8% vs 43.8% — the knowledge gap measured at 300K
(2.2pp) vanishes at real depth: ZERO knowledge deficit at equal
nodes; (3) interleaved NPS by regime: MIDDLEGAME 1.62M vs 3.28M
(ratio 0.49) but endgame 2.64M vs 3.44M (0.77).

**The middlegame NPS ratio (0.49) exactly equals E1's time
equivalence (their 10s = our 20s). All measurements now cohere:
THE ENTIRE −98 LTC GAP vs V20 IS MIDDLEGAME PER-NODE COST.**
Knowledge: zero (measured). Selectivity: zero (measured). Endgame
speed: near-parity. Our per-node overhead explodes with board density
— the profile's generate/legality/attackers/check-info/stats bundle
(~30%+) plus eval-path cost on dense positions vs Stash's leaner
scaling.

**The road to closing the hundreds: 2x middlegame NPS = ~+98 Elo at
LTC, pure engineering with provable tree identity (no knowledge risk,
no SPRT needed — node-identity + interleaved midgame-NPS gates).
Attack order: (1) eval cost on dense positions (attack tables per
call), (2) per-move legality/check-info, (3) stats counters, (4)
movegen staging. Each verified by: identical depth-16 node counts +
interleaved midgame NPS gain.**

## Rung E9 — eval cost measured: 60 percent of node time (2026-09-01)

Differential build (evaluate() replaced by popcount-material blend,
everything else untouched), interleaved A/B/A/B middlegame NPS:
**baseline 1.68M vs eval-free 4.15M = the eval costs 60% of node
time.** Eval-free we would be FASTER than Stash's 3.28M — the search
machinery is fine; evaluate() at ~480ns/call (0.74 calls/node) is
2x Stash's ENTIRE node budget. Eval-cache hit rates: main 18%,
qsearch 3% (6.1M full evals per 10.4M nodes — the full-position
cache cannot see qsearch capture chains; pawn hash already exists and
works). gprof/LTO attributions proven FICTIONAL on this codebase
(Book 5.5M calls impossible, vector 34% = merged symbols) —
differential builds are the only trusted instrument; the D1 harness
(/tmp/lxd1 pattern) is the template.

**Eval-cost campaign (each step output-identical + tree-identity
gated, cumulative target 2-3x cheaper eval = 2.4-3.3M NPS = the full
speed half of the gap): (a) king-safety second pass reuses
attacks_by instead of recomputing per-piece attacks; (b) threat
loops: piece_type_on loads per target replaced by extract-from-
bitboard iteration; (c) eval cache 64K -> 1M (main-path locality);
(d) misc: shared subexpression hoisting, branchless term selects.
No knowledge risk; the 5M-node referee duel already proved output
quality is at parity — speed here converts 1:1 into LTC Elo.**
