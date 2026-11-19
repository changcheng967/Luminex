# NNUE Architecture Specification: Luminex gen768t

## 1. General Metadata
- **Network Name**: gen768t
- **Target Engine / Application**: Luminex, from PR #107 (LNT8 net format: LNR8 plus a relation table); no network released yet
- **Primary Use Case**: Chess Position Evaluation
- **Weight Quantization Scheme**: QAT (piece-square table int16 on the engine's x2048 grid; relation table int8 in steps of 1/128; dense layers int8 with one scale per row, `f32` biases)
- **Internal Output Scale**: float centipawns (trainer), int32 centipawns (engine)

---

## 2. Input Features & Transformer Layer
- **Feature Set Name**: HalfKAv2_hm + attack relations + pawn pairs
- **Input Feature Dimension**: `[24576]` + `[55960]` per perspective (Sparse boolean vector)
- **Feature Breakdown**:
  1. **Feature Group 1**: King bucket (32, horizontally mirrored) x piece-square (12 piece types x 64 squares = 768), kings included; at most 32 active rows
  2. **Feature Group 2**: Attack relations (54,472): a piece attacking a piece of either colour (a pawn on its two capture squares, a knight or king on its jumps, a bishop, rook or queen along each line up to the first occupied square), on squares oriented as group 1, without the king bucket. Not features: a king as the victim, the perspective's own king as the attacker, a pawn attacking a pawn (group 3); a mutual attack between two pieces is one feature
  3. **Feature Group 3**: Pawn pairs (1,488): two pawns of either colour on the same or adjacent files, oriented as group 1; a pair is active only when one pawn attacks the other's square (210 rows); groups 2 and 3 together have at most 248 active rows
- **Perspective Symmetry**: Dual Perspective (`side-to-move` / `other-side`)

---

## 3. Perspective Accumulator Subnet
- **Input Dimension**: `[24576]` + `[55960]` (Sparse boolean)
- **Output Dimension**: `[1536]` (Combined `[768]` side-to-move + `[768]` other side)
- **Layer Breakdown**:
  - `Sparse Linear (piece-squares)`: Input `[24576]` -> Output `[768]` per perspective | Weights: `FP<i16, x2048>`, Bias: `FP<i32, x2048>`
  - `Sparse Linear (relations)`: Input `[55960]` -> Output `[256]` per perspective | Weights: `FP<i8, x128>`, no bias
  - `Combination`: lanes `0..255`: `sat16(psq + clamp(rel, -511, 511) * 16)` on the x2048 grid; lanes `256..767`: `psq`
  - `Activation`: SCReLU `clamp(in, 0, 1)^2` after the stm / nstm concatenation, requantized to `u8` (x127)

---

## 4. Subnet Selection & Bucketing Logic
- **Routing Factor**: Piece count (men on the board, kings included)
- **Bucket Selection Logic**: `Index = (piece_count - 1) * 8 / 32` (2-4, 5-8, ..., 29-32 men)
- **Number of Buckets**: `8`; each bucket holds a value tail and the two heads
- **Combined Perspective State**: Concatenated `[1536]` tensor fed into the selected bucket's value tail

---

## 5. Main Subnet Architecture

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

## 6. Output Conversion & Post-Processing
- **Perspective Average Shortcut**: None
- **Output Addition**: None
- **Final Rescaling Formula**:
  $$
  \text{Value} = \text{round}(out \times 300)
  $$
- **Final Return Type**: Signed Integer (`int32`) / Centipawns, side-to-move relative

---

## 7. Dataflow Execution Diagram (Text Pipeline)

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
