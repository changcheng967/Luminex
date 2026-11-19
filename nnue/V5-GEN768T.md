# NNUE Architecture Specification: Luminex gen768t (V5, draft)

## 1. General Metadata
- **Network Name**: gen768t (working name; files `luminex_gen768t_<NN>_i8.nnue`)
- **Target Engine / Application**: Luminex. No net of this kind is trained yet. It is gen768s ([V4-GEN768S.md](../V4-GEN768S.md)) with a second sparse input set, attack relations and pawn pairs, whose int8 rows add into the first `T = 256` lanes of the feature layer, and with that layer's integers on the engine's x2048 grid, where every lane is an exact int16 sum. Net file format LNT8, little-endian, every tensor preceded by its int32 element count as in LNR8:
  - `"LNT8"`, then int32 `L1 = 768`, `L2 = 16`, `L3 = 32`, `NI = 24576`, `NB = 8`
  - int32 `T = 256` (relation lanes; a multiple of 64), `NR = 55960` (rows of feature groups 2 and 3), `FS = 11` (the feature layer's grid: lanes, piece-square weights and bias are integers x `2^FS` = x2048), `RS = 4` (relation shift), `PPM` (pawn-pair mode, 0 or 1), `LAYOUT = 1` (section 2's index layout)
  - `int16 psq[NI][L1]` (feature-major, x2048), `int32 psq_bias[L1]` (x2048)
  - `int8 rel[NR][T]` (feature-major, one row per relation or pawn-pair feature)
  - `NB` dense tails, each exactly LNR8's
  - optional tagged sections after the tails (LHE1, the heads), as today

  An engine that reads LNT8 still reads LNR8 (a net with no relation table). An engine without the LNT8 code rejects the magic
- **Primary Use Case**: Chess position evaluation. The value is the only output the search reads. The error and draw heads are gen768s's; the engine computes neither
- **Weight Quantization Scheme**: QAT.
  - Piece-square table: int16 on the x2048 grid (`FS = 11`, the engine's grid for gen768s since #93), weights clipped to `|w| <= 3.99`. int16 ends at 16.0 on this grid, and every final lane stays under it (section 3, range rule).
  - Relation table: int8, `|q| <= 127`, in steps of `2^RS` on the x2048 grid (`RS = 4`: steps of 1/128, `|w| <= 0.992`).
  - Dense layers: int8 with one scale per row of L2 and L3 and f32 biases (LNR8).
  - Float phase: the relation weights are clipped to the int8 range after each step, and a draw logit trains jointly with the value as in gen768s and is dropped before QAT. Both heads are fitted on the QAT net's activations
- **Internal Output Scale**: value: float centipawns (trainer), int32 centipawns (engine). Heads: expected-score units in `[0, 1]`, in steps of 1/10000 in the engine

---

## 2. Input Features & Transformer Layer
- **Feature Set Name**: HalfKAv2_hm + AR (attack relations) + PP (pawn pairs)
- **Input Feature Dimension**: `[24576]` + `[54472]` + `[1488]` = `[80536]` per perspective (Sparse boolean vector). Groups 2 and 3 share one table of `55960` rows, used by both perspectives
- **Feature Breakdown**:
  1. **Feature Group 1 (piece-squares, unchanged)**: King bucket (32, horizontally mirrored) x piece-square (12 piece types x 64 squares = 768), kings included; at most 32 active rows
  2. **Feature Group 2 (attack relations, 54,472)**: a piece on one square attacking a piece on another, on the real board, for both colours and for attacks on own pieces (defences). Not king-bucketed; oriented like group 1
  3. **Feature Group 3 (pawn pairs, 1,488)**: two pawns of either colour on the same or adjacent files. With `PPM = 0` only the 210 rows in which one pawn attacks the other's square are ever active
- **Perspective Symmetry**: Dual Perspective (`side-to-move` / `other-side`)

**Frame.** Squares are a1 = 0 ... h8 = 63. For perspective `P` with its own king on `k`:
- `o(s) = s XOR (56 if P is Black) XOR (7 if k is on files a-d)`, group 1's orientation;
- a piece's side is `own` (colour `P`) or `their`;
- own pawns capture toward rank 8 of the oriented board, their pawns toward rank 1.

**Group 2: which attacks are features.** A relation `(a on x) -> (v on y)` holds when `a` attacks square `y` and `v` stands on it: a pawn attacks its two capture squares, a knight or king its jumps, a bishop, rook or queen each of its lines up to and including the first occupied square. `a` and `v` may be of one colour. A relation is not a feature when another input always carries it:
1. `v` is a king.
2. `a` is the perspective's own king (group 1's king bucket and `v`'s piece-square carry it exactly).
3. `a` and `v` are both pawns (group 3).
4. Mutual attacks are told once:
   - `a` and `v` of one type among N, B, R, Q: on one side, one feature per unordered pair; across sides, the feature whose attacker is the own piece.
   - Q -> B along a diagonal (the feature is B -> Q); Q -> R along a rank or file (the feature is R -> Q).
   - B -> P or Q -> P when that pawn attacks the bishop's or queen's square (the feature is P -> B or P -> Q).

**Group 2: index.** Attacker class `a = (side, type)`, victim class `v = (side, type)`, oriented squares `x' = o(x)`, `y' = o(y)`. Before indexing, a pair of one type on one side is ordered `x' < y'`, and a pair of one type with a `their` attacker and an `own` victim swaps roles.

`E(a)[x']` is the set of squares class `a` attacks from `x'` on an empty board (pawns stand on ranks 2-7 only). `G(a->v)[x']` is `E(a)[x']` restricted as follows:
- `v` a pawn: ranks 2-7 only;
- Q -> B: rank and file lines only; Q -> R: diagonals only;
- B -> P and Q -> P: without the squares from which that pawn attacks `x'` (an own pawn: the two squares diagonally below `x'`; their pawn: the two diagonally above);
- one type, one side: only `y' > x'`.

```text
index = base(a->v) + off(a->v)[x'] + popcount( G(a->v)[x'] & (bit(y') - 1) )
off(a->v)[x'] = sum of |G(a->v)[u]| over u < x'
base(a->v)    = sum of the sizes of all earlier blocks
```

Block order: attacker side (own, their), attacker type (P, N, B, R, Q, K), victim side (own, their), victim type (P, N, B, R, Q); a block with no feature is skipped. Block sizes (rows):

| attacker \ victim | own P | own N | own B | own R | own Q | their P | their N | their B | their R | their Q | rows |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| own P | - | 84 | 84 | 84 | 84 | - | 84 | 84 | 84 | 84 | 672 |
| own N | 284 | 168 | 336 | 336 | 336 | 284 | 336 | 336 | 336 | 336 | 3,088 |
| own B | 364 | 560 | 280 | 560 | 560 | 364 | 560 | 560 | 560 | 560 | 4,928 |
| own R | 672 | 896 | 896 | 448 | 896 | 672 | 896 | 896 | 896 | 896 | 8,064 |
| own Q | 1,036 | 1,456 | 896 | 560 | 728 | 1,036 | 1,456 | 896 | 560 | 1,456 | 10,080 |
| their P | - | 84 | 84 | 84 | 84 | - | 84 | 84 | 84 | 84 | 672 |
| their N | 284 | - | 336 | 336 | 336 | 284 | 168 | 336 | 336 | 336 | 2,752 |
| their B | 364 | 560 | - | 560 | 560 | 364 | 560 | 280 | 560 | 560 | 4,368 |
| their R | 672 | 896 | 896 | - | 896 | 672 | 896 | 896 | 448 | 896 | 7,168 |
| their Q | 1,036 | 1,456 | 896 | 560 | - | 1,036 | 1,456 | 896 | 560 | 728 | 8,624 |
| their K | 348 | 420 | 420 | 420 | 420 | 348 | 420 | 420 | 420 | 420 | 4,056 |

Total 54,472; every row is reached by exactly one (class pair, `x'`, `y'`).

**Group 3: pawn pairs.** A pawn's id is `48 * side + o(s) - 8` (0-95). A pair is two pawns on different squares whose files differ by at most one: 480 same-file and 1,008 adjacent-file rows.

```text
index = 54472 + rank of (lower id, higher id) among all valid pairs in lexicographic order
```

- `PPM = 0`: a pair is active only when one pawn attacks the other's square: an own pawn defending an own pawn (70 rows), their pawn defending their pawn (70), an own and their pawn attacking each other (70).
- `PPM = 1`: every pair on the board is active.

**What a move changes.** Three steps on single squares; each emits removals (`-`) and additions (`+`) of physical features, and each perspective maps them to its own indices (a relation that is not a feature in that perspective is skipped).

```text
remove(z):     - every relation the piece on z attacks in, - every relation it is the victim of,
               - its pawn pairs; empty z;
               + (s -> w) for every slider s whose line reaches z, w the first piece beyond z
add(z, p):     - (s -> w) for every slider s whose line reaches z, w the first piece beyond z;
               put p on z;
               + every relation p attacks in, + every relation it is the victim of, + its pawn pairs
replace(z, p): - the old piece's relations and pawn pairs; put p on z; + the new piece's
               (the occupancy is unchanged, so no line changes)
```

A slider's line reaches `z` when `z` is on one of its lines with only empty squares between. Every `+` and `-` applies the rules above (kings as victims, mutual attacks told once, pawn on pawn as a pair).

| Move | Steps, in order |
| :--- | :--- |
| Quiet | `remove(from)`, `add(to, piece)` |
| Capture | `remove(from)`, `replace(to, piece)` |
| En passant | `remove(from)`, `remove(captured pawn's square)`, `add(to, pawn)` |
| Promotion | `remove(from)`, then `add(to, promoted piece)` or `replace(to, promoted piece)` |
| Castling | `remove(king from)`, `remove(rook from)`, `add(king to, king)`, `add(rook to, rook)` |
| Null move | nothing |
| Own king crosses the d/e files | the mover's perspective: every index changes, groups 2 and 3 are rebuilt from the board; the other perspective: the steps above |

A feature a move both adds and removes may be cancelled or applied twice: the sums are exact either way.

**Conformance.** White's view, then Black's: active features of groups 2 and 3, and the sum of their indices.

| Position | `PPM = 0` White | `PPM = 0` Black | `PPM = 1` White | `PPM = 1` Black |
| :--- | :--- | :--- | :--- | :--- |
| start position | 33 / 923,020 | 33 / 923,020 | 69 / 2,897,986 | 69 / 2,897,986 |
| `r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -` | 47 / 1,268,017 | 47 / 1,275,960 | 77 / 2,913,373 | 77 / 2,925,714 |
| `2kr3r/ppp2ppp/2n1bn2/2b1p3/4P3/2NP1N2/PPPB1PPP/2KR1B1R b - -` (both kings mirrored) | 34 / 762,273 | 35 / 866,598 | 63 / 2,353,560 | 64 / 2,457,869 |
| `8/5pk1/6p1/8/3R4/6P1/r4PK1/8 w - -` | 5 / 250,781 | 5 / 229,138 | 9 / 469,911 | 9 / 448,268 |

Single features, in the third position unless named:

| Feature | View | Oriented | Block | base + off + rank = index |
| :--- | :--- | :--- | :--- | :--- |
| Ne5 attacks pawn d7 (second position) | White | e5 -> d7 | own N -> their P | 2132 + 168 + 6 = 2306 |
| the same | Black | e4 -> d2 | their N -> own P | 27504 + 116 + 0 = 27620 |
| Bc5 attacks pawn f2 | White | f5 -> c2 | their B -> own P | 30256 + 228 + 0 = 30484 |
| the same | Black | f4 -> c7 | own B -> their P | 6084 + 152 + 6 = 6242 |
| Rd1 defends Bd2 | White | e1 -> e2 | own R -> own B | 10256 + 56 + 7 = 10319 |
| rooks d8 and h8 see each other | Black | a1, e1 | own R -> own R | 11152 + 0 + 3 = 11155 |
| the same | White | a8, e8 | their R -> their R | 40448 + 420 + 3 = 40871 |
| pawns e4 and d5 attack each other (after 1. e4 d5) | White | ids 20, 75 | pawn pair | 54472 + 577 = 55049 |
| the same | Black | ids 19, 76 | pawn pair | 54472 + 552 = 55024 |

---

## 3. Perspective Accumulator Subnet
- **Input Dimension**: `[24576]` piece-squares + `[55960]` relations and pawn pairs (Sparse boolean)
- **Output Dimension**: `[1536]` (Combined `[768]` side-to-move + `[768]` other side), shared by the value tail and both heads
- **Layer Breakdown**:
  - `Sparse Linear (piece-squares)`: Input `[24576]` -> Output `[768]` per perspective | Weights: `FP<i16, x2048>`, Bias: `FP<i32, x2048>`; int16 lanes holding the exact sum of the bias and the active rows
  - `Sparse Linear (relations)`: Input `[55960]` -> Output `[256]` per perspective | Weights: `i8` (one step = `2^RS` on the x2048 grid), no bias; int16 lanes holding the exact sum of the active rows, in steps. At most 248 features are active in any position (248 x 127 < 32768), so the lanes never overflow and the order of additions never matters
  - `Combination`: lane `i < 256`: `a_i = sat16(psq_i + clamp(rel_i, -L, L) * 2^RS)` with `L = 511` steps (3.99 at `RS = 4`), a saturating int16 addition; lane `i >= 256`: `a_i = psq_i`. The activation reads `a_i` only through `clamp(a, 0, 1)`, so a saturated sum gives the activation the exact sum gives
  - `Activation`: SCReLU `clamp(a / 2048, 0, 1)^2` after the stm / nstm concatenation, requantized to `u8` (x127)
  - `Range rule`: every piece-square and relation lane is an exact integer sum. The piece-square lanes stay under 16.0, int16's end on this grid, on every position the trainer's export checks; a net whose largest lane reaches 16.0 is not written. A move's rows are added and subtracted in wrapping int16 arithmetic, so a lane equals its exact sum whenever that sum fits, whatever its partial sums do; a rebuild from the board sums in int32 and saturates once. That is what the trainer computes, bit for bit. The relation lanes hold at most 248 x 127 steps and never overflow, and the combination saturates harmlessly (above). An engine that saturates an intermediate sum and then updates from it does not compute this network
- **Maintenance**:
  - The piece-square accumulator is maintained as in gen768s: updated per move, rebuilt through the refresh cache when the own king moves. Under the range rule a lane never clips.
  - The relation accumulator's indices depend on the own king only through the mirror. It is updated per move, own-king moves included (the king is a blocker on lines like any piece), and rebuilt from the board only when the own king crosses the d/e files.
  - Like the piece-square lanes it is brought current lazily, before an evaluation. A move's rows are found then, from the board before the move, which the engine keeps per ply; a ply that no evaluation needs is never processed.

---

## 4. Subnet Selection & Bucketing Logic
- **Routing Factor**: Piece count (men on the board, kings included)
- **Bucket Selection Logic**: `Index = (piece_count - 1) * 8 / 32` (2-4, 5-8, ..., 29-32 men)
- **Number of Buckets**: `8`; each bucket holds a value tail and one readout of each head
- **Combined Perspective State**: Concatenated `[1536]` `u8` tensor, fed to the selected bucket's value tail on every evaluation

---

## 5. Main Subnet Architecture

Value tail (per bucket, every evaluation; unchanged from gen768s, LNR8):

| Layer Index | Layer Type | Input Shape | Output Shape | Activation / Operation | Fixed-Point Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L2** | Dense Linear | `[1536]` | `[16]` | SCReLU `clamp(in, 0, 1)^2`, requantized to `u8` (x127) | `i8` weights (one scale per row), `f32` bias |
| **L3** | Dense Linear | `[16]` | `[32]` | SCReLU `clamp(in, 0, 1)^2`, requantized to `u8` (x127) | `i8` weights (one scale per row), `f32` bias |
| **Out** | Dense Linear | `[32]` | `[1]` | None | `i8` weights (own scale), `f32` bias |

Heads (per bucket; unchanged from gen768s, not computed by the engine):

| Layer Index | Layer Type | Input Shape | Output Shape | Activation / Operation | Fixed-Point Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Error** | Linear readout | `[1536]` activations + `\|v\|/400` | `[1]` | clamp to `[0, 1]` | `i8` weights (one scale per head per bucket), `f32` coefficient of `\|v\|/400`, `f32` bias |
| **Draw** | Linear readout | `[1536]` activations + `\|v\|/400` | `[1]` | clamp to `[0, 1]` | `i8` weights (one scale per head per bucket), `f32` coefficient of `\|v\|/400`, `f32` bias |

---

## 6. Output Conversion & Post-Processing
- **Perspective Average Shortcut**: None
- **Output Addition**: None
- **Final Rescaling Formula**: as gen768s
  $$
  \text{Value} = \text{round}(out \times 300)
  $$
- **Final Return Type**: Signed integer (`int32`) centipawns, side-to-move relative

---

## 7. Dataflow Execution Diagram (Text Pipeline)

```text
[ Board State ]
│
├──► Side-To-Move Perspective ─┬─► Piece-squares: Sparse Linear [24576 -> 768] (i16) ───────────────┐
│                              └─► Relations + pawn pairs: Sparse Linear [55960 -> 256] (i8)        │
│                                        └─► clamp, << RS, added to lanes 0..255 ───────────────────┤
│                                                                                                   ├─► Accumulator [1536]
└──► Opponent Perspective ─────┬─► Piece-squares: Sparse Linear [24576 -> 768] (i16) ───────────────┤         │
                               └─► Relations + pawn pairs: Sparse Linear [55960 -> 256] (i8)        │         │
                                         └─► clamp, << RS, added to lanes 0..255 ───────────────────┘         │
                                                                                                              ▼
                                                                                             SCReLU clamp(0,1)^2 -> u8
                                                                                                              │
                                                                                                  Bucket Routing Logic
                                                                                               (piece_count - 1) * 8 / 32
                                                                                                              │
                                                                                                              ▼
                                                                                                [ Value tail, bucket b ]
                                                                                                              │
                                                                                                      Dense Linear (16)
                                                                                                              │
                                                                                                      SCReLU clamp(0,1)^2
                                                                                                              │
                                                                                                      Dense Linear (32)
                                                                                                              │
                                                                                                      SCReLU clamp(0,1)^2
                                                                                                              │
                                                                                                      Dense Linear (1)
                                                                                                              │
                                                                                                        x 300 -> int32
                                                                                                              │
                                                                                                              ▼
                                                                                                  [ Evaluation Output ]
```

What each side computes:

```text
Engine                                                   Trainer
------                                                   -------
read LNT8 (and LNR8: no relation table, T = 0)           write LNT8; float checkpoints carry rel[NR][T]
on a move: keep the new ply's board (mailbox and         per batch: both perspectives' active indices of
  bitboards); nothing else                                 groups 2 and 3 from the position record
before an evaluation, for each ply to replay: the        float phase: rel rows summed into lanes 0..T-1
  section 2 steps on the board before its move ->          before the bias and SCReLU; rows start at zero;
  per perspective, rows to add and to subtract;            weights clipped to +-127 steps after each step
  prefetch them, then the piece-square lanes, then
  relation lanes = parent's lanes + adds - subs
own king across d/e: rebuild that perspective's          QAT: rel rows rounded to int8 steps (straight
  relation lanes from that ply's board                     through), the lane sum clamped at +-L steps,
evaluation: lanes 0..T-1 = sat(psq + clamp(rel) << RS)     piece-squares on the x2048 grid, tail per row
debug builds: every lane == a rebuild from the board     export: largest |psq lane| over the checked
  at every evaluation, inside searches                     positions below 16.0, or the net is not written
                                                         parity: indices equal the engine's on the same
                                                           positions; QAT activations equal bit for bit,
                                                           for positions reached inside a search as well
```
