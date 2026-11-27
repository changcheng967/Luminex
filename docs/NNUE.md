# Luminex's network architectures

The specification of every network architecture the engine reads, oldest first, each in the template's
seven parts (the template is last). A network is named `luminex_<architecture>_<NN>.nnue`, its architecture
the network name in its section (README, "Networks are named").

- [V1-GEN768](#v1-gen768): gen768
- [V2-GEN768Q](#v2-gen768q): gen768q
- [V3-GEN768R](#v3-gen768r): gen768r
- [V4-GEN768S](#v4-gen768s): gen768s
- [V5-GEN768T](#v5-gen768t): gen768t
- [The template](#the-template)

---

## V1-GEN768

### 1. General Metadata
- **Network Name**: gen768
- **Target Engine / Application**: Luminex v6.1.0 – v6.5.1 (shipped as gen768, then retrained as sf1 in v6.2.0 and sfq in v6.3.0 – v6.5.1; ended, replaced by gen768r)
- **Primary Use Case**: Chess Position Evaluation
- **Weight Quantization Scheme**: Post-hoc (float training; FT stored float32 in file, int16 x8192 in engine RAM; tail int8 at weight scale 64)
- **Internal Output Scale**: float centipawns (trainer), int32 centipawns (engine tail)

---

### 2. Input Features & Transformer Layer
- **Feature Set Name**: HalfKAv2_hm
- **Input Feature Dimension**: `[24576]` (Sparse boolean vector)
- **Feature Breakdown**:
  1. **Feature Group 1**: King square x piece-square, both colors, horizontal mirror, 32 rows max active
- **Perspective Symmetry**: Dual Perspective (`side-to-move` / `other-side`)

---

### 3. Perspective Accumulator Subnet
- **Input Dimension**: `[24576]` (Sparse boolean)
- **Output Dimension**: `[768]` per perspective
- **Layer Breakdown**:
  - `Sparse Linear`: Input `[24576]` -> Output `[768]` | Weights: `FP<f32 file / i16 x8192 RAM>`, Bias: `f32`
  - `Activation`: SCReLU `clamp(in, 0, 1)^2` applied after stm/nstm concatenation

---

### 4. Subnet Selection & Bucketing Logic
- **Routing Factor**: None (single shared tail; bucketing is a v2 item)
- **Bucket Selection Logic**: n/a
- **Number of Buckets**: `1`
- **Combined Perspective State**: Concatenated `[1536]` tensor (stm `[768]` ++ nstm `[768]`)

---

### 5. Main Subnet Architecture

| Layer Index | Layer Type | Input Shape | Output Shape | Activation / Operation | Fixed-Point Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L2** | Dense Linear | `[1536]` | `[16]` | None | `i8 weights, scale 64` |
| **A2** | Activation | `[16]` | `[16]` | SCReLU `clamp(in,0,1)^2` | requant `min(v>>6, 127)` |
| **L3** | Dense Linear | `[16]` | `[32]` | None | `i8 weights, scale 64` |
| **A3** | Activation | `[32]` | `[32]` | SCReLU `clamp(in,0,1)^2` | requant `min(v>>6, 127)` |
| **Out** | Dense Linear | `[32]` | `[1]` | None | `i8, own scale` |

---

### 6. Output Conversion & Post-Processing
- **Perspective Average Shortcut**: None
- **Output Addition**: None
- **Final Rescaling Formula**:
  $$
  \text{Value} = \text{float}(out) \times 300 = \text{centipawns (stm-relative)}
  $$
- **Final Return Type**: Float / Centipawns (stm-relative; negamax consumes directly)

---

### 7. Dataflow Execution Diagram (Text Pipeline)

```text
[ Board State ]
│
├──► Side-To-Move Perspective ──► Sparse Linear [24576->768] ──────┐
│                                                                  ├─► concat [1536]
└──► Opponent Perspective    ──► Sparse Linear [24576->768] ──────┘        │
                                                                           ▼
                                                            SCReLU clamp(0,1)^2
                                                                           │
                                                                           ▼
                                                            Dense Linear [1536->16]
                                                                           │
                                                            SCReLU clamp(0,1)^2
                                                                           │
                                                                           ▼
                                                            Dense Linear [16->32]
                                                                           │
                                                            SCReLU clamp(0,1)^2
                                                                           │
                                                                           ▼
                                                            Dense Linear [32->1]
                                                                           │
                                                                     x 300 (cp)
                                                                           │
                                                                           ▼
                                                            [ Evaluation Output ]
```

---

## V2-GEN768Q

### 1. General Metadata
- **Network Name**: gen768q
- **Target Engine / Application**: None: a design that was never trained or released
- **Primary Use Case**: Chess Position Evaluation
- **Weight Quantization Scheme**: Post-hoc (float training; FT stored float32 in file, int16 x8192 in engine RAM; tail int8 at weight scale 64)
- **Internal Output Scale**: float centipawns (trainer), int32 centipawns (engine tail)

---

### 2. Input Features & Transformer Layer
- **Feature Set Name**: HalfKAv2_hm + Pawn Conjunctions
- **Input Feature Dimension**: `[24704]` minimum (T2a) — `[25185]` full max if all probe-gated blocks pass
- **Feature Breakdown**:
  1. **Feature Group 1**: King square x piece-square, both colors, horizontal mirror, 24576 indices (gen768 set unchanged)
  2. **Feature Group 2**: Passed pawn, `24576 + side*64 + (pov==BLACK ? sq^56 : sq)`, side in {own, other} relative to the side to move — **128 indices, pooled across perspectives (no perspective multiplier: which half of the concat fires carries it), king-independent, vertical flip only, NO horizontal mirror** (rook-file vs centre-file is absolute signal). T2a.
  3. **Feature Group 3**: Isolated pawn, same indexing as Group 2 — 128 indices. Conditional (probe-gated), sequenced after T2a.
  4. **Feature Group 4**: Doubled pawn, same indexing — 128 indices. Conditional (probe-gated).
  5. **Feature Group 5**: King shelter — conjunction-only; two candidate forms (225-row KS-lite vs 512-row persp x KingBuckets x class) decided by the ridge probe and slice arm. Weakest block, first to be cut.
- **Perspective Symmetry**: Dual Perspective (`side-to-move` / `other-side`)

---

### 3. Perspective Accumulator Subnet
- **Input Dimension**: per Feature Group table (Sparse boolean)
- **Output Dimension**: `[768]` per perspective
- **Layer Breakdown**:
  - `Sparse Linear`: Input `[N]` -> Output `[768]` | Weights: `FP<f32 file / i16 x8192 RAM>`, Bias: `f32`
  - `Activation`: SCReLU `clamp(in, 0, 1)^2` applied after stm/nstm concatenation
- **Warm Start**: new FT columns zero-initialized — bit-identical to T0 at step 0 (verified static AND along a walked 500-move game)

---

### 4. Subnet Selection & Bucketing Logic
- **Routing Factor**: Total piece count (men on board, kings included)
- **Bucket Selection Logic**: `Index = min(7, (32 - piece_count) / 4)`
- **Number of Buckets**: `8` — **conditional on the buckets-from-T0 dose-matched fork** (pre-registered: quiet val MAE <= -0.6cp confirms, > -0.3cp kills)
- **Combined Perspective State**: Concatenated `[1536]` tensor (stm `[768]` ++ nstm `[768]`) fed into the selected Main Subnet

---

### 5. Main Subnet Architecture

| Layer Index | Layer Type | Input Shape | Output Shape | Activation / Operation | Fixed-Point Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L2** | Dense Linear | `[1536]` | `[16]` | None | `i8 weights, scale 64` |
| **A2** | Activation | `[16]` | `[16]` | SCReLU `clamp(in,0,1)^2` | requant `min(v>>6, 127)` |
| **L3** | Dense Linear | `[16]` | `[32]` | None | `i8 weights, scale 64` |
| **A3** | Activation | `[32]` | `[32]` | SCReLU `clamp(in,0,1)^2` | requant `min(v>>6, 127)` |
| **Out** | Dense Linear | `[32]` | `[1]` | None | `i8, own scale` |

L2 stays 16 (widening dead on two bases); FT stays 768 (Elo ledger prices any width
change; slim-FT-640 is the only live width experiment, conditional, from scratch).

---

### 6. Output Conversion & Post-Processing
- **Perspective Average Shortcut**: None
- **Output Addition**: None
- **Final Rescaling Formula**:
  $$
  \text{Value} = \text{float}(out) \times 300 = \text{centipawns (stm-relative)}
  $$
- **Final Return Type**: Float / Centipawns (stm-relative; negamax consumes directly)

---

### 7. Dataflow Execution Diagram (Text Pipeline)

```text
[ Board State ]
│
├──► Side-To-Move Perspective ──► Sparse Linear [N->768] ──┐
│                                                           ├─► concat [1536]
└──► Opponent Perspective    ──► Sparse Linear [N->768] ──┘        │
                                                                    ▼
                                                     SCReLU clamp(0,1)^2
                                                                    │
                                                    (bucket routing if fork passes)
                                                                    │
                                                                    ▼
                                                    Dense Linear [1536->16]
                                                    SCReLU -> [16->32] -> SCReLU
                                                    Dense Linear [32->1] -> x 300 (cp)
                                                                    │
                                                                    ▼
                                                        [ Evaluation Output ]
```

---

### Program rules (round-5b reconciliation, three-agent consensus)

- **Ladder policy**: every lever forks from T0 (`luminex_sfq_full.pt`) in parallel with a
  dose-matched control; no chaining (chaining accumulates dose the Data Law rations and
  entangles attribution). Dose ledger: cumulative <= ~2 epochs on any shipped lineage.
- **Gate order**: ridge probe (kill-only, <0.3% residual R^2) -> 10k-position featurizer-
  engine index diff -> zero-init bit-identity (static + walked game) -> DCU slice arm vs
  dose-matched control -> nps bench with the Elo ledger (ship iff fixed-node gain >= nps
  loss + 10 at 1,000 games) -> 300-game fixed-node book gate -> C500 full run.
- **First C500 session is recipe, not architecture**: single-variable forks from T0 at
  matched dose (capture weight {0.1, 0, 0.6}, dose {0.65x, 2x}, loss power 2.0), anchor
  re-run as control, judged on quiet-subset val MAE against the published run-repeat bar.
- **Rejected permanently**: ply (layer-stack) bucketing — eval must remain a function of
  the position (TT and repetition consistency); FT-1024; L2 widening; PSQT rider.

---

## V3-GEN768R

### 1. General Metadata
- **Network Name**: gen768r
- **Target Engine / Application**: Luminex v6.6.0, its only release (LNB8 net format; the line ended with gen768r_06q)
- **Primary Use Case**: Chess Position Evaluation
- **Weight Quantization Scheme**: Post-hoc (float training; FT stored float32 in the file, int16 x8192 in engine RAM; dense layers int8, one scale per layer per bucket)
- **Internal Output Scale**: float centipawns (trainer), int32 centipawns (engine)

---

### 2. Input Features & Transformer Layer
- **Feature Set Name**: HalfKAv2_hm
- **Input Feature Dimension**: `[24576]` per perspective (Sparse boolean vector)
- **Feature Breakdown**:
  1. **Feature Group 1**: King bucket (32, horizontally mirrored) x piece-square (12 piece types x 64 squares = 768), kings included; at most 32 active rows
- **Perspective Symmetry**: Dual Perspective (`side-to-move` / `other-side`)

---

### 3. Perspective Accumulator Subnet
- **Input Dimension**: `[24576]` (Sparse boolean)
- **Output Dimension**: `[1536]` (Combined `[768]` side-to-move + `[768]` other side)
- **Layer Breakdown**:
  - `Sparse Linear`: Input `[24576]` -> Output `[768]` per perspective | Weights: `FP<f32 file / i16, x8192 RAM>`, Bias: `FP<f32 file / i32, x8192 RAM>`
  - `Activation`: SCReLU `clamp(in, 0, 1)^2` after the stm / nstm concatenation, requantized to `u8` (x127)

---

### 4. Subnet Selection & Bucketing Logic
- **Routing Factor**: Piece count (men on the board, kings included)
- **Bucket Selection Logic**: `Index = (piece_count - 1) * 8 / 32` (2-4, 5-8, ..., 29-32 men)
- **Number of Buckets**: `8` (Main Subnets `0` through `7`)
- **Combined Perspective State**: Concatenated `[1536]` tensor fed into the selected Main Subnet

---

### 5. Main Subnet Architecture

| Layer Index | Layer Type | Input Shape | Output Shape | Activation / Operation | Fixed-Point Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L2** | Dense Linear | `[1536]` | `[16]` | SCReLU `clamp(in, 0, 1)^2`, requantized to `u8` (x127) | `i8` weights (per-bucket scale), `f32` bias |
| **L3** | Dense Linear | `[16]` | `[32]` | SCReLU `clamp(in, 0, 1)^2`, requantized to `u8` (x127) | `i8` weights (per-bucket scale), `f32` bias |
| **Out** | Dense Linear | `[32]` | `[1]` | None | `i8` weights (own scale), `f32` bias |

---

### 6. Output Conversion & Post-Processing
- **Perspective Average Shortcut**: None
- **Output Addition**: None
- **Final Rescaling Formula**:
  $$
  \text{Value} = \text{round}(out \times 300)
  $$
- **Final Return Type**: Signed Integer (`int32`) / Centipawns, side-to-move relative

---

### 7. Dataflow Execution Diagram (Text Pipeline)

```text
[ Board State ]
│
├──► Side-To-Move Perspective ──► Sparse Linear [24576 -> 768] ──┐
│                                                                 ├─► Accumulator [1536]
└──► Opponent Perspective    ──► Sparse Linear [24576 -> 768] ──┘         │
                                                                          ▼
                                                               SCReLU clamp(0,1)^2 -> u8
                                                                          │
                                                              Bucket Routing Logic
                                                           (piece_count - 1) * 8 / 32
                                                                          │
                          ┌───────────────────────────────────────────────┴───────────────────────────────────────────────┐
                          ▼                                                                                               ▼
                    [ Main Subnet 0 ]                                                                                 [ Main Subnet 7 ]
                          │                                                                                               │
                  Dense Linear (16)                                                                               Dense Linear (16)
                          │                                                                                               │
                  SCReLU clamp(0,1)^2                                                                             SCReLU clamp(0,1)^2
                          │                                                                                               │
                  Dense Linear (32)                                                                               Dense Linear (32)
                          │                                                                                               │
                  SCReLU clamp(0,1)^2                                                                             SCReLU clamp(0,1)^2
                          │                                                                                               │
                  Dense Linear (1)                                                                                Dense Linear (1)
                          └───────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                                          │
                                                                          ▼
                                                                  x 300 -> int32
                                                                          │
                                                                          ▼
                                                             [ Evaluation Output ]
```

---

## V4-GEN768S

### 1. General Metadata
- **Network Name**: gen768s
- **Target Engine / Application**: None: main's network between v6.6.0 and v6.7.0 (gen768s_01q, then gen768s_03q), in no release; replaced by gen768t (LNS1 net format: gen768r's LNB8 plus one error head per output bucket)
- **Primary Use Case**: Chess position evaluation, and the evaluation's own expected error, read by the search's pruning margins
- **Weight Quantization Scheme**: QAT, as gen768r_03q and 04q (feature layer on the engine's int16 grid x8192, dense layers int8 with one scale per layer per bucket). The error head trains after the value net, on its frozen accumulator, through the same int8 rounding
- **Internal Output Scale**: float centipawns (trainer), int32 centipawns (engine), for both outputs

---

### 2. Input Features & Transformer Layer
- **Feature Set Name**: HalfKAv2_hm (unchanged from gen768r)
- **Input Feature Dimension**: `[24576]` per perspective (Sparse boolean vector)
- **Feature Breakdown**:
  1. **Feature Group 1**: King bucket (32, horizontally mirrored) x piece-square (12 piece types x 64 squares = 768), kings included; at most 32 active rows
- **Perspective Symmetry**: Dual Perspective (`side-to-move` / `other-side`)

---

### 3. Perspective Accumulator Subnet
- **Input Dimension**: `[24576]` (Sparse boolean)
- **Output Dimension**: `[1536]` (Combined `[768]` side-to-move + `[768]` other side), shared by the value tail and the error head
- **Layer Breakdown**:
  - `Sparse Linear`: Input `[24576]` -> Output `[768]` per perspective | Weights: `FP<i16, x8192>`, Bias: `FP<i32, x8192>`
  - `Activation`: SCReLU `clamp(in, 0, 1)^2` after the stm / nstm concatenation, requantized to `u8` (x127)

---

### 4. Subnet Selection & Bucketing Logic
- **Routing Factor**: Piece count (men on the board, kings included)
- **Bucket Selection Logic**: `Index = (piece_count - 1) * 8 / 32` (2-4, 5-8, ..., 29-32 men)
- **Number of Buckets**: `8`; each bucket holds a value tail and an error head
- **Combined Perspective State**: Concatenated `[1536]` tensor fed into the selected bucket's value tail, and into its error head when the search asks for it

---

### 5. Main Subnet Architecture

Value tail (per bucket, every evaluation; unchanged from gen768r: the capacity test keeps L2 at 16, Luminex-NNUE's DEVLOG §19 S2):

| Layer Index | Layer Type | Input Shape | Output Shape | Activation / Operation | Fixed-Point Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L2** | Dense Linear | `[1536]` | `[16]` | SCReLU `clamp(in, 0, 1)^2`, requantized to `u8` (x127) | `i8` weights (per-bucket scale), `f32` bias |
| **L3** | Dense Linear | `[16]` | `[32]` | SCReLU `clamp(in, 0, 1)^2`, requantized to `u8` (x127) | `i8` weights (per-bucket scale), `f32` bias |
| **Out** | Dense Linear | `[32]` | `[1]` | None | `i8` weights (own scale), `f32` bias |

Error head (per bucket, computed only where the search reads it):

| Layer Index | Layer Type | Input Shape | Output Shape | Activation / Operation | Fixed-Point Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **E1** | Dense Linear | `[1536]` | `[8]` | SCReLU `clamp(in, 0, 1)^2`, requantized to `u8` (x127) | `i8` weights (per-bucket scale), `f32` bias |
| **E2** | Dense Linear | `[8]` | `[1]` | ReLU `max(0, in)` | `i8` weights (own scale), `f32` bias |

---

### 6. Output Conversion & Post-Processing
- **Perspective Average Shortcut**: None
- **Output Addition**: None
- **Final Rescaling Formula**:
  $$
  \text{Value} = \text{round}(out \times 300) \qquad \text{Error} = \text{round}(e \times 300)
  $$
  Error is non-negative: the predicted size of the gap between this static evaluation and a deep search's value of the position (the label and the quantile are set by Luminex-NNUE's DEVLOG §19, gate 1)
- **Final Return Type**: Signed Integer (`int32`) / Centipawns, side-to-move relative (value); unsigned centipawns (error)

---

### 7. Dataflow Execution Diagram (Text Pipeline)

```text
[ Board State ]
│
├──► Side-To-Move Perspective ──► Sparse Linear [24576 -> 768] ──┐
│                                                                 ├─► Accumulator [1536]
└──► Opponent Perspective    ──► Sparse Linear [24576 -> 768] ──┘         │
                                                                          ▼
                                                               SCReLU clamp(0,1)^2 -> u8
                                                                          │
                                                              Bucket Routing Logic
                                                           (piece_count - 1) * 8 / 32
                                                                          │
                          ┌───────────────────────────────────────────────┴───────────────────────────┐
                          ▼                                                                           ▼
             [ Value tail, bucket b ]                                                  [ Error head, bucket b ]
                          │                                                          (only when the search asks)
                  Dense Linear (16)                                                                   │
                          │                                                                   Dense Linear (8)
                  SCReLU clamp(0,1)^2                                                                 │
                          │                                                                   SCReLU clamp(0,1)^2
                  Dense Linear (32)                                                                   │
                          │                                                                   Dense Linear (1)
                  SCReLU clamp(0,1)^2                                                                 │
                          │                                                                     ReLU max(0, x)
                  Dense Linear (1)                                                                    │
                          │                                                                    x 300 -> cp
                    x 300 -> int32                                                                    │
                          │                                                                           ▼
                          ▼                                                          [ Expected error (cp) ]
               [ Evaluation Output ]                                               read by the pruning margins
```

---

## V5-GEN768T

### 1. General Metadata
- **Network Name**: gen768t
- **Target Engine / Application**: None: main's network between v6.6.0 and v6.7.0 (gen768t_01q to gen768t_05q), in no release; replaced by gen768u (LNT8 net format: LNR8 plus a relation table)
- **Primary Use Case**: Chess Position Evaluation
- **Weight Quantization Scheme**: QAT (piece-square table int16 on the engine's x2048 grid; relation table int8 in steps of 1/128; dense layers int8 with one scale per row, `f32` biases)
- **Internal Output Scale**: float centipawns (trainer), int32 centipawns (engine)

---

### 2. Input Features & Transformer Layer
- **Feature Set Name**: HalfKAv2_hm + attack relations + pawn pairs (threat inputs: prior art in Stockfish's Full_Threats and PP_3Wide and in other engines, listed in the README)
- **Input Feature Dimension**: `[24576]` + `[55960]` per perspective (Sparse boolean vector)
- **Feature Breakdown**:
  1. **Feature Group 1**: King bucket (32, horizontally mirrored) x piece-square (12 piece types x 64 squares = 768), kings included; at most 32 active rows
  2. **Feature Group 2**: Attack relations (54,472): a piece attacking a piece of either colour (a pawn on its two capture squares, a knight or king on its jumps, a bishop, rook or queen along each line up to the first occupied square), on squares oriented as group 1, without the king bucket. Not features: a king as the victim, the perspective's own king as the attacker, a pawn attacking a pawn (group 3); a mutual attack between two pieces is one feature
  3. **Feature Group 3**: Pawn pairs (1,488): two pawns of either colour on the same or adjacent files, oriented as group 1; a pair is active only when one pawn attacks the other's square (210 rows); groups 2 and 3 together have at most 248 active rows
- **Perspective Symmetry**: Dual Perspective (`side-to-move` / `other-side`)

---

### 3. Perspective Accumulator Subnet
- **Input Dimension**: `[24576]` + `[55960]` (Sparse boolean)
- **Output Dimension**: `[1536]` (Combined `[768]` side-to-move + `[768]` other side)
- **Layer Breakdown**:
  - `Sparse Linear (piece-squares)`: Input `[24576]` -> Output `[768]` per perspective | Weights: `FP<i16, x2048>`, Bias: `FP<i32, x2048>`
  - `Sparse Linear (relations)`: Input `[55960]` -> Output `[256]` per perspective | Weights: `FP<i8, x128>`, no bias
  - `Combination`: lanes `0..255`: `sat16(psq + clamp(rel, -511, 511) * 16)` on the x2048 grid; lanes `256..767`: `psq`
  - `Activation`: SCReLU `clamp(in, 0, 1)^2` after the stm / nstm concatenation, requantized to `u8` (x127)

---

### 4. Subnet Selection & Bucketing Logic
- **Routing Factor**: Piece count (men on the board, kings included)
- **Bucket Selection Logic**: `Index = (piece_count - 1) * 8 / 32` (2-4, 5-8, ..., 29-32 men)
- **Number of Buckets**: `8`; each bucket holds a value tail and the two heads
- **Combined Perspective State**: Concatenated `[1536]` tensor fed into the selected bucket's value tail

---

### 5. Main Subnet Architecture

Value tail (per bucket, every evaluation):

| Layer Index | Layer Type | Input Shape | Output Shape | Activation / Operation | Fixed-Point Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L2** | Dense Linear | `[1536]` | `[16]` | SCReLU `clamp(in, 0, 1)^2`, requantized to `u8` (x127) | `i8` weights (one scale per row), `f32` bias |
| **L3** | Dense Linear | `[16]` | `[32]` | SCReLU `clamp(in, 0, 1)^2`, requantized to `u8` (x127) | `i8` weights (one scale per row), `f32` bias |
| **Out** | Dense Linear | `[32]` | `[1]` | None | `i8` weights (own scale), `f32` bias |

Heads (per bucket, stored after the tails; not computed by the engine):

| Layer Index | Layer Type | Input Shape | Output Shape | Activation / Operation | Fixed-Point Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Error** | Linear Readout | `[1536]` + `\|v\| / 400` | `[1]` | clamp to `[0, 1]` | `i8` weights (one scale per head), `f32` coefficient and bias |
| **Draw** | Linear Readout | `[1536]` + `\|v\| / 400` | `[1]` | clamp to `[0, 1]` | `i8` weights (one scale per head), `f32` coefficient and bias |

---

### 6. Output Conversion & Post-Processing
- **Perspective Average Shortcut**: None
- **Output Addition**: None
- **Final Rescaling Formula**:
  $$
  \text{Value} = \text{round}(out \times 300)
  $$
- **Final Return Type**: Signed Integer (`int32`) / Centipawns, side-to-move relative

---

### 7. Dataflow Execution Diagram (Text Pipeline)

```text
[ Board State ]
│
├──► Side-To-Move Perspective ──┬──► Piece-squares: Sparse Linear [24576 -> 768] ──────────────────┐
│                               └──► Relations: Sparse Linear [55960 -> 256]                       │
│                                    clamp(-511, 511) x 16 -> + lanes 0..255 (saturating) ─────────┤
│                                                                                                  ├─► Accumulator [1536]
└──► Opponent Perspective ──────┬──► Piece-squares: Sparse Linear [24576 -> 768] ──────────────────┤         │
                                └──► Relations: Sparse Linear [55960 -> 256]                       │         │
                                     clamp(-511, 511) x 16 -> + lanes 0..255 (saturating) ─────────┘         │
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

---

## V6-GEN768U

### 1. General Metadata
- **Network Name**: gen768u
- **Target Engine / Application**: Luminex v6.7.0 (LNT8 net format with 73,728 piece-square rows: the table once per phase block)
- **Primary Use Case**: Chess Position Evaluation
- **Weight Quantization Scheme**: QAT (piece-square tables int16 on the engine's x2048 grid; relation table int8 in steps of 1/128; dense layers int8 with one scale per row, `f32` biases)
- **Internal Output Scale**: float centipawns (trainer), int32 centipawns (engine)

---

### 2. Input Features & Transformer Layer
- **Feature Set Name**: HalfKAv2_hm by phase block + attack relations + pawn pairs (phase blocks: Triumviratus's input buckets by piece count, related to Stockfish's K32H3; threat inputs: prior art in Stockfish's Full_Threats and PP_3Wide and in other engines, listed in the README)
- **Input Feature Dimension**: `[73728]` + `[55960]` per perspective (Sparse boolean vector)
- **Feature Breakdown**:
  1. **Feature Group 1**: Phase block (3, by the men on the board, kings and pawns included: block 0 for 26 or more, 1 for 13 to 25, 2 for 12 or fewer) x king bucket (32, horizontally mirrored) x piece-square (12 piece types x 64 squares = 768), kings included: feature = block x 24,576 + the HalfKAv2_hm index, the same block for both perspectives; at most 32 active rows
  2. **Feature Group 2**: Attack relations (54,472): a piece attacking a piece of either colour (a pawn on its two capture squares, a knight or king on its jumps, a bishop, rook or queen along each line up to the first occupied square), on squares oriented as group 1, without the king bucket or the block. Not features: a king as the victim, the perspective's own king as the attacker, a pawn attacking a pawn (group 3); a mutual attack between two pieces is one feature
  3. **Feature Group 3**: Pawn pairs (1,488): two pawns of either colour on the same or adjacent files, oriented as group 1; a pair is active only when one pawn attacks the other's square (210 rows); groups 2 and 3 together have at most 248 active rows
- **Perspective Symmetry**: Dual Perspective (`side-to-move` / `other-side`)

---

### 3. Perspective Accumulator Subnet
- **Input Dimension**: `[73728]` + `[55960]` (Sparse boolean)
- **Output Dimension**: `[1536]` (Combined `[768]` side-to-move + `[768]` other side)
- **Layer Breakdown**:
  - `Sparse Linear (piece-squares)`: Input `[73728]` (three blocks of `[24576]`) -> Output `[768]` per perspective | Weights: `FP<i16, x2048>`, Bias: `FP<i32, x2048>` (one bias for every block)
  - `Sparse Linear (relations)`: Input `[55960]` -> Output `[256]` per perspective | Weights: `FP<i8, x128>`, no bias
  - `Combination`: lanes `0..255`: `sat16(psq + clamp(rel, -511, 511) * 16)` on the x2048 grid; lanes `256..767`: `psq`
  - `Activation`: SCReLU `clamp(in, 0, 1)^2` after the stm / nstm concatenation, requantized to `u8` (x127)

---

### 4. Subnet Selection & Bucketing Logic
- **Routing Factor**: Piece count (men on the board, kings included), twice: the feature layer's phase block and the output bucket
- **Bucket Selection Logic**: phase block `0` for 26 or more men, `1` for 13 to 25, `2` for 12 or fewer; output bucket `Index = (piece_count - 1) * 8 / 32` (2-4, 5-8, ..., 29-32 men)
- **Number of Buckets**: `3` phase blocks of the piece-square table; `8` output buckets, each holding a value tail
- **Combined Perspective State**: Concatenated `[1536]` tensor fed into the selected bucket's value tail

---

### 5. Main Subnet Architecture

Value tail (per bucket, every evaluation):

| Layer Index | Layer Type | Input Shape | Output Shape | Activation / Operation | Fixed-Point Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L2** | Dense Linear | `[1536]` | `[16]` | SCReLU `clamp(in, 0, 1)^2`, requantized to `u8` (x127) | `i8` weights (one scale per row), `f32` bias |
| **L3** | Dense Linear | `[16]` | `[32]` | SCReLU `clamp(in, 0, 1)^2`, requantized to `u8` (x127) | `i8` weights (one scale per row), `f32` bias |
| **Out** | Dense Linear | `[32]` | `[1]` | None | `i8` weights (own scale), `f32` bias |

---

### 6. Output Conversion & Post-Processing
- **Perspective Average Shortcut**: None
- **Output Addition**: None
- **Final Rescaling Formula**:
  $$
  \text{Value} = \text{round}(out \times 300)
  $$
- **Final Return Type**: Signed Integer (`int32`) / Centipawns, side-to-move relative

---

### 7. Dataflow Execution Diagram (Text Pipeline)

```text
[ Board State ] ──► phase block p by the men on the board: 0 for 26 or more, 1 for 13 to 25, 2 for 12 or fewer
│
├──► Side-To-Move Perspective ──┬──► Piece-squares: Sparse Linear [73728 -> 768], block p ─────────┐
│                               └──► Relations: Sparse Linear [55960 -> 256]                       │
│                                    clamp(-511, 511) x 16 -> + lanes 0..255 (saturating) ─────────┤
│                                                                                                  ├─► Accumulator [1536]
└──► Opponent Perspective ──────┬──► Piece-squares: Sparse Linear [73728 -> 768], block p ─────────┤         │
                                └──► Relations: Sparse Linear [55960 -> 256]                       │         │
                                     clamp(-511, 511) x 16 -> + lanes 0..255 (saturating) ─────────┘         │
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

---

## The template

A new architecture's section, after the last one (its title is the version and the network name, as
`V5-GEN768T`):

````markdown
## V<N>-<NETWORK NAME>
### 1. General Metadata
- **Network Name**: [e.g., SFNNv16]
- **Target Engine / Application**: [e.g., Stockfish 16+]
- **Primary Use Case**: [e.g., Chess Position Evaluation]
- **Weight Quantization Scheme**: [e.g., QAT (Quantization Aware Training)]
- **Internal Output Scale**: [e.g., int64 evaluation units]

---

### 2. Input Features & Transformer Layer
- **Feature Set Name**: [e.g., HalfKAv2_hm + FullThreats + 2P Wide]
- **Input Feature Dimension**: `[N]` (Sparse boolean vector)
- **Feature Breakdown**:
  1. **Feature Group 1**: [Description & index size, e.g., King square + piece square]
  2. **Feature Group 2**: [Description & index size, e.g., Attack threat pairs]
- **Perspective Symmetry**: Dual Perspective (`side-to-move` / `other-side`)

---

### 3. Perspective Accumulator Subnet
- **Input Dimension**: `[N]` (Sparse boolean)
- **Output Dimension**: `[1024]` (Combined `[512]` "our" + `[512]` "their")
- **Layer Breakdown**:
  - `Sparse Linear`: Input `[N]` -> Output `[1024]` | Weights: `FP<i16, 256>`, Bias: `FP<i16, 256>`
  - `Activation`: ClippedReLU `clamp(in, 0, 255/256)`
  - `Element-wise Multiply`: Output `[512]` = `a[512] * b[512]`

---

### 4. Subnet Selection & Bucketing Logic
- **Routing Factor**: [e.g., Material count / Piece count]
- **Bucket Selection Logic**: `Index = (piece_count - 1) / 4`
- **Number of Buckets**: `8` (Main Subnets `0` through `7`)
- **Combined Perspective State**: Concatenated `[1024]` tensor fed into selected Main Subnet

---

### 5. Main Subnet Architecture

| Layer Index | Layer Type | Input Shape | Output Shape | Activation / Operation | Fixed-Point Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L1** | Dense Linear | `[1024]` | `[32]` | None | `FP<i32, 128*64>` |
| **L2** | Parallel Activations | `[32]` | `[64]` | SqrClippedReLU `[32]` + ClippedReLU `[32]` | `FP<i16, 125>` |
| **L3** | Dense Linear | `[64]` | `[32]` | None | `FP<i32, 128*64>` |
| **L4** | Parallel Activations | `[32]` | `[128]` | SqrClippedReLU `[32]` + ClippedReLU `[32]` | `FP<i16, 125>` |
| **L5** | Dot Product | `[128]` | `[1]` | Output Dot Product with Weights | `FP<i32, 128*125>` |

---

### 6. Output Conversion & Post-Processing
- **Perspective Average Shortcut**: `out = (our[8] - their[8]) / 2`
- **Output Addition**: Main Subnet Output + Perspective Average Offset
- **Final Rescaling Formula**:
  $$
  \text{Value} = \frac{\text{int64}(in) \times \text{Multiplier}}{\text{Denominator}}
  $$
- **Final Return Type**: Signed Integer (`int64`) / Centipawns

---

### 7. Dataflow Execution Diagram (Text Pipeline)

```text
[ Board State ]
│
├──► Side-To-Move Perspective ──► Sparse Linear ──► ClippedReLU ──┐
│                                                                 ├─► Accumulator [1024]
└──► Opponent Perspective    ──► Sparse Linear ──► ClippedReLU ──┘         │
                                                                          ▼
                                                              Bucket Routing Logic
                                                              (piece_count - 1) / 4
                                                                          │
                          ┌───────────────────────────────────────────────┴───────────────────────────────────────────────┐
                          ▼                                                                                               ▼
                    [ Main Subnet 0 ]                                                                                 [ Main Subnet 7 ]
                          │                                                                                               │
                    Dense Linear (32)                                                                                 Dense Linear (32)
                          │                                                                                               │
              SqrClippedReLU / ClippedReLU                                                                    SqrClippedReLU / ClippedReLU
                          │                                                                                               │
                    Dense Linear (32)                                                                                 Dense Linear (32)
                          │                                                                                               │
              SqrClippedReLU / ClippedReLU                                                                    SqrClippedReLU / ClippedReLU
                          │                                                                                               │
                    Dot Product (1)                                                                                   Dot Product (1)
                          └───────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                                          │
                                                                          ▼
                                                              + Perspective Offset
                                                                          │
                                                                          ▼
                                                              Rescaling / int64
                                                                          │
                                                                          ▼
                                                              [ Evaluation Output ]
```
````
