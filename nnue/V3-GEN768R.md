# NNUE Architecture Specification: Luminex gen768r

gen768 with its frozen tail units revived and routed through eight output
buckets. Every choice below comes from a measurement listed in "Evidence";
the history's instrument rule stands (nnue/DEVLOG.md §13-14): offline numbers
locate weaknesses, only games decide whether a net ships.

## 1. General Metadata
- **Network Name**: gen768r
- **Target Engine / Application**: Luminex 6.6+ (engine: LNB8 loader, #49; trainer: #49 + #50)
- **Primary Use Case**: Chess Position Evaluation
- **Weight Quantization Scheme**: Post-hoc, unchanged (FT float in file, int16 x8192 in RAM; tail int8 with a per-layer, per-bucket scale)
- **Internal Output Scale**: float centipawns x300 (trainer), int32 centipawns (engine)

---

## 2. Input Features & Transformer Layer
- **Feature Set Name**: HalfKAv2_hm (unchanged)
- **Input Feature Dimension**: `[24576]` per perspective
- **Feature Breakdown**:
  1. **Feature Group 1**: king bucket (32, horizontally mirrored) x piece-square (768)
- **Perspective Symmetry**: Dual Perspective (`side-to-move` / `other-side`)

Why unchanged: the feature transformer is the healthy part of the net (no
dead units; its activation has 360-475 effective dimensions), and our static
error against Stockfish 19 is about twice Stockfish's own in every group we
split by (1.84-1.99x with and without passed pawns), so no single missing
input stands out. The measured bottleneck is downstream.

---

## 3. Perspective Accumulator Subnet
- **Input Dimension**: `[24576]` (sparse boolean)
- **Output Dimension**: `[768]` per perspective, concatenated `[1536]`
- **Layer Breakdown**:
  - `Sparse Linear`: `[24576] -> [768]` | Weights `f32 file / i16 x8192 RAM`, Bias `f32`
  - `Activation`: SCReLU `clamp(x, 0, 1)^2` after the stm / nstm concatenation

---

## 4. Subnet Selection & Bucketing Logic
- **Routing Factor**: pieces on the board, kings included
- **Bucket Selection Logic**: `Index = (pieces - 1) * 8 / 32` (2-4, 5-8, ..., 29-32 men)
- **Number of Buckets**: `8`
- **Combined Perspective State**: the shared `[1536]` activation feeds the selected bucket's tail

Why: one tail cannot hold the per-phase scale the data needs. Our static eval
against Stockfish's runs from 0.66x (21-24 men) to 1.88x (29-32 men), and
compresses big advantages about twice as much as Stockfish's static eval does.

---

## 5. Main Subnet Architecture (one per bucket)

| Layer | Type | Input | Output | Activation | Fixed point |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L2** | Dense Linear | `[1536]` | `[16]` | SCReLU | `i8`, per-bucket scale |
| **L3** | Dense Linear | `[16]` | `[32]` | SCReLU | `i8`, per-bucket scale |
| **Out** | Dense Linear | `[32]` | `[1]` | none | `i8`, own scale |

All units live at the start: the shipped tail's 7 dead + 1 stuck L2 units and
10 dead L3 units are redrawn before training (`revive_tail.py`, function
preserved to 0.0002 cp), then copied into all eight buckets. Tail weights
train at 4x the base learning rate so the revived units' outgoing weights,
which start at zero, can reach the live units' scale within one T0 dose.

---

## 6. Output Conversion & Post-Processing
- **Perspective Average Shortcut**: none
- **Output Addition**: none
- **Final Rescaling Formula**: `Value = out x 300` centipawns, side-to-move relative
- **Final Return Type**: int32 centipawns

---

## 7. Dataflow Execution Diagram

```text
[ Board ] -> stm features -> Sparse Linear [24576 -> 768] --+
          -> nstm features -> Sparse Linear [24576 -> 768] -+-> concat [1536] -> SCReLU
                                                                     |
                                              bucket = (pieces - 1) * 8 / 32
                                                                     |
                      +---------------------+------------------------+-----+
                      v                     v                              v
                  [ tail 0 ]  ...       [ tail b ]        ...         [ tail 7 ]
               16 -> 32 -> 1          16 -> 32 -> 1                 16 -> 32 -> 1
                                            |
                                            v
                                      x 300 = cp (stm)
```

---

## Evidence

All on the mscope pod; scripts in the session notes (`referee.py`, `analyze.py`,
`replay.py`, `attrib.py`, `probe_net.py`, `revive_tail.py`).

**1. Where the Elo goes (game forensics).** Stockfish 19 (150K nodes, WDL on)
refereed every position of 300 equal-node games, Luminex 100K vs Stash 37
100K (2moves_v1, colours swapped). Expected-score loss per game:

| | Luminex | Stash 37 |
|---|---|---|
| total | 1.27 | 1.20 |
| in winning positions (referee > 85%) | **0.59** | 0.12 |
| in balanced positions | 0.66 | 1.00 |
| 29-32 men | 0.21 | 0.32 |
| 17-24 men | 0.50 | 0.37 |
| <= 16 men | 0.26 | 0.19 |

We reach winning positions three times as often as Stash and give five
times as much back from them.

**2. Search or eval.** Each of our 1,604 mistakes (>= 0.05 expected score)
was replayed cold at 1x, 8x and 64x the game's nodes and re-scored by
Stockfish 19 from the same root at 300K nodes (1,267 confirmed, total loss
300.8). The persistent ones were split by Stockfish's static eval and by a
depth-9 search, the kind of label we train on.

| fixed by | share of our loss | in winning positions (52% of it) |
|---|---|---|
| a cold re-search at the game's 100K nodes | 32.8% | 34.3% |
| 8x the nodes | 20.7% | 19.2% |
| 64x the nodes | 17.9% | 14.6% |
| nothing, though our static eval prefers the right move (search) | 11.7% | 14.3% |
| nothing: Stockfish's static eval gets it right (eval, learnable) | 6.3% | 6.9% |
| nothing: only Stockfish at depth 9 gets it right (eval, deep) | 7.1% | 7.9% |
| nothing: depth 9 gets it wrong too (beyond our labels) | 3.5% | 2.8% |

The split is the same in every phase. The first row is not a hash problem:
on 1,500 random moves a cold re-search picks a different move 36% of the
time, and where it does, its move is not better (cold minus warm -0.012 +-
0.010 expected score). More than a third of our decisions at 100K nodes are
near-ties that flip with the search state, and the mistakes in that row are
the unlucky draws. The search-defect row is mostly active moves that leave a
piece hanging (Nd5, Nf5, Bxg2, Bxe3): our search sees the material go and
never finds the compensation Stockfish's search finds.

**2b. Near-ties are exactly where our eval has no signal.** On pairs where
our warm and cold searches chose different moves and Stockfish's search
prefers one by >= 0.05, our static eval of the two children picks the better
one 48.9% of the time (88 pairs), Stockfish's static eval 62.5%; at >= 0.10,
49.3% vs 63.8% (69 pairs). Small samples (+-9%), one direction.

**3. The net.** On positions from our games:
- feature transformer: no dead units, about half its lanes live, 360-475
  effective dimensions at 90% of variance;
- L2: 7 of 16 units dead on every position, 1 saturated on every one;
  effective dimension 4 (90%) / 6 (99%);
- L3: 10 of 32 units dead on every position;
- the September gen768 net has 7 + 2 and 11: the units died in the 512-wide
  era and every net since inherited them.

**4. The eval against Stockfish 19** (5,796 positions, depth-10 truth):
- static error about twice Stockfish's static error in every group: 2.07x
  with material level, 1.84-1.99x with and without passed pawns, 1.96-2.10x
  when a side is up or down two pawns;
- big advantages compressed: SF depth 10 +300..+600 reads 139 cp low
  (Stockfish's static eval: 93 cp low);
- per-phase scale against Stockfish's static eval 0.66 (21-24 men) to 1.88
  (29-32 men);
- compensation (down two pawns, Stockfish says fine) is hard for both
  statics (bias -180 / -141); our ratio there is 1.24, our closest.

## Decisions

**D1. Architecture now: gen768r (this document).** Revive the frozen tail,
route it through 8 output buckets, train the tail weights at 4x. The FT stays
as it is. Cost per eval: one popcount (identical trees with today's net).
Why this and not a wider net: most of our loss at STC node counts is
search-limited (evidence 2: 71% fixed by more nodes or a re-search, 12% a
search defect, 17% persistent eval), so accuracy must not cost nodes; this design
adds capacity at zero NPS cost exactly where the net is narrowest (evidence 3)
and gives each phase its own scale (evidence 4). Gate: net-only SPRT against
the shipped net, dev_network vs base_network on one commit.

**D2. Data: about 0.8B unique positions, one pass (T0's dose).** t91 April +
May (Leela self-play, Stockfish-relabelled), 6,000 steps x 131,072. Basis:
- T0's dose sits on the measured plateau (6,000-12,000 steps); fewer steps
  lose (-57 at 0.65x), more passes over fixed data fall off a cliff between
  2.6 and 4.45 epochs;
- twice the unique data at the same steps measured +11 +- 21 (tie), on a
  tail that could use six dimensions of what the FT learned;
- not a few million: the FT has 18.9M weights, and T0 gives each about a
  million updates;
- not 100B: a 3-hour C500 session trains about 0.8B samples, so 100B is
  ~125 sessions, and nothing measured says this capacity could use it.
Re-open after D1: the same 1x-vs-2x unique-data screen on gen768r. If 2x
pays, scale to 2-4B positions (more t91 months); if it ties again, data stays
closed until the FT grows.

**D3. Next components, with the rule that decides each.** These are where
the measurements say novelty pays: not in a larger or more elaborate shape
(that costs the nodes our losses are short of), but in what the net is
trained to do.
- *Liveness-preserving training* (built, #50): function-preserving revival
  of frozen tail units, bit-identical in int8, and a frozen-unit count in
  every training run.
- *A sibling-ranking objective.* The pointwise loss scores each position
  alone; the search's hard decisions compare the children of one parent
  (evidence 2, 2b). Add a pairwise term that penalizes ordering two
  siblings differently from a deep search: for children a, b of one parent
  with Stockfish preferring a, a margin loss on sigmoid(f(b)) -
  sigmoid(f(a)), mixed with the pointwise loss. First build the data that
  also confirms 2b at size: our top two moves at 100K nodes for many
  positions from our own games, both scored by Stockfish 19 (the pod scores
  about a million positions a day). Gate: held-out near-tie ordering, then
  games.
- *Tail width.* If gen768r's middlegame buckets use >= 12 of their 16 L2
  dimensions (99% of variance), the tail is the bottleneck again: L2 = 32 per
  bucket, after a VNNI kernel for it; ship only if the fixed-node gain beats
  the NPS loss.
- *Our own positions, deeper labels.* A tail fitted on 31K positions from our
  games, labelled by Stockfish 19 at 150K nodes, cut held-out error 167 ->
  133 cp in seven epochs (frozen FT). The pair data above is that
  distribution with those labels; it feeds both the ranking term and a
  pointwise fine-tune. Only games decide: MAE has predicted strength 0/8
  times.
- *New inputs* only for a knowledge hole the forensics name. None so far:
  the static error ratio is flat across every split tried.

**D4. What this does not do.** Stockfish's static eval errs half as much as
ours and ranks near-tie siblings where ours guesses; its search converts the
same evaluations deeper. gen768r and the ranking objective can close part
of the eval gap at no speed cost. They cannot by themselves take us past
Stockfish: that needs both a much better eval (capacity, data, labels) and
the search work the attribution points at (39% of our loss is fixed by 8x
or 64x the nodes, 12% is a search defect in lines that give up material).
