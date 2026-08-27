# Luminex Systems Audit — 2026-08-25

The car-engineer document: every subsystem of our engine, what it does,
what it structurally CANNOT do, and where four months of play data says
it fails. Written from our own code and our own loss corpus. No other
engine's terms were consulted.

## Machine overview

Alpha-beta (PVS) + staged move ordering + 1250-feature linear eval
distilled from SF static evals. 2.4M NPS. Bullet ladder ≈2515.

## Subsystem-by-subsystem

### 1. Quiescence search
What it is: captures + check evasions only, SEE-pruned (−60×depth
margin), MVV-LVA ordered, horizon at depth −4, delta pruning, eval-cache
in front of stand-pat.
Structurally cannot: see QUIET-move tactics (checks, threats, pawn
races). Depth −4 cannot resolve any exchange longer than 4 captures.
Play data: 75% of losses are ≥5-point swings within ≤4 plies that START
from quiet moves (sac initiations) — exactly what qsearch never
searches. Tried: quiet-check generation (−16.7), deeper horizon global
(neutral) and EG-conditional (−41.9). The NPS cost exceeds the vision
gain at every configuration tested.

### 2. Move ordering (staged, novel)
What it is: strict priority phases — TT move, good captures (MVV-LVA +
SEE + check bonus), killers/counters/continuation/history quiets, bad
captures last. Generation is lazy per phase (genuine design win).
Structurally cannot: order what it cannot see — quiet SAC moves score as
plain quiets (history only). The 86%-repeat finding: at the razor
positions, the winning/rebutting move is typically a quiet sac that
ordering surfaces too late (or never at the searched depth).
Measured: 83% agreement with Stash on chosen moves — ordering is NOT a
differentiator vs peers; it is a differentiator vs depth-24 truth.

### 3. Pruning stack
Razoring, reverse futility, futility, NMP (eval-adaptive R + zugzwang
verification), ProbCut (dynamic SEE), continuation pruning, LMP.
Structurally cannot: distinguish "eval is high because position is won"
from "eval is high because the eval is wrong here." eval_uncertainty
(correction-history magnitude) partially compensates — our invention —
but uncertainty only modulates margins, never redirects depth.
SPRT: whole stack saturated at ±5 Elo (4 probes).

### 4. LMR
Dual tables (noisy/quiet), improving/worsening/cut-node/TT-PV modifiers,
history-based reduction, check/promotion/pawn-7th/castling/capture-
response discounts, eval-adaptive distance. Hindsight-depth adjustment.
Structurally cannot: reduce selectively in the razor class — every
signal it uses (history, eval distance) is exactly the signal that is
WRONG in razor positions (measured: our eval says +1123 where truth at
depth 24 says losing).
Play data: SPRT non-PV extra reduction = exactly 0 at 16K games. The
reduction surface is at ITS play-optimum for this eval.

### 5. Extensions
Singular (+double + multi-cut), check, recapture, forcing-line (ours,
+72 validated). 
Structurally cannot: extend on QUIET sac initiation — no signal exists
to detect "this quiet move starts a 10-ply forced sequence" cheaply.

### 6. History tables
Plain gravity, counter (1-ply), continuation (2-ply), capture history,
low-ply bonus. Correction history: 6 tables (ours).
Structurally cannot: learn from UNSEEN positions — bullet games are
unique; razor positions recur never. History is fast-adapting
ordering, not knowledge.

### 7. Eval (the known ceiling)
Linear in 1250 features. Distilled from SF STATIC evals at 22M rows.
PROVEN play-optimum of its function class: 5 feature families, 3 data
scales (up to 223M rows), 4 objective functions (MSE, σ(eval)-logistic,
outcome-logistic, tension-weighted outcome) all land at the same play
value. 2× compressed in decisive regimes vs every strong engine
(measured 3 independent ways) — and every decompression is play-neutral
(5 attempts).
Structurally cannot: express feature interactions at all (linear), and
the distillation target (static eval) carries no information about
which errors matter in play.

### 8. Time management
Stability tiers, node-effort dominance, score-drop extension,
sharp-root extension (aspiration-retry aware). Flag dominance 46-0 (we
are faster than peers). mtg and drop-multiplier probes: dead.

## The two structural failure modes (from OUR loss corpus)

F1. RAZOR BLINDNESS: we enter and lose won positions through quiet-sac
punches whose refutation is deeper than bullet search reaches. All
engines at our depth share this — the winner is decided by who enters
fewer unresolvable razors. Our eval cannot price "unresolvable."

F2. KNOWLEDGE SOURCE: every coefficient comes from SF static-eval
distillation. Houppin's came from HIS OWN GAMES via outcome-AdaGrad —
a base continuously re-centered on PLAY outcomes under HIS search. Our
one unanchored attempt was ridge-to-SF-evals (fit1, −210, chess-nonsense
values); our outcome attempts were all L2-ANCHORED to fit6 (drift
≤1.7cp, methodologically prevented from traveling — the postmortem's
anchoring bug). The unanchored OUTCOME tune with chess-sanity gates
(instead of anchors) has NEVER been run.

## What year-end 3300 requires (honest math)

Stash: +890 over 15 versions ≈ 83 accepted patches at +2..+35 each,
each verified at 10-90K games, on a base that MOVED under play-based
retunes. Our box: ~7K games/h → one 40K SPRT per ~6h → 3-4 tests/day.
80 accepted patches at 1-in-5 acceptance ≈ 400 candidate tests ≈ 100+
days of 24/7 autonomous testing. Feasible ONLY as an always-on pipeline,
and ONLY if candidates come from a source that actually moves the base
(the unanchored outcome retune), not from the exhausted static-eval fit.

## Attack plan (self-derived, from this audit)

A1. UNANCHORED OUTCOME RETUNE: --texel trainer, outcome labels, NO L2
to c0 (or 1e-9), init from fit6, effective-value GATES (not anchors) as
chess-sanity guard, 100+ epochs, drift allowed to travel. Gate output
through 500g A/B then 40K SPRT. This is our v21-equivalent swing — the
one big lever never actually pulled (postmortem bug 2).

A2. ALWAYS-ON SPRT PIPELINE: cron-driven candidate queue, resume
harness, 24/7. Candidates from: A1 iterates, per-term micro-patches,
search-structure probes. Every verdict auto-recorded.

A3. RAZOR-CLASS UNCERTAINTY (research): extend eval_uncertainty from
margin-modulation to depth-allocation — at nodes where correction
history has been very wrong, distrust the eval enough to search deeper
rather than prune. Not in any engine (uncertainty-as-margin is ours;
uncertainty-as-depth-allocation would be the novel step). Design and
falsify cheaply before building.
