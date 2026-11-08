# NNUE Architecture Specification: Luminex gen768r

## 1. General Metadata
- **Network Name**: gen768r
- **Target Engine / Application**: Luminex 6.6+ (LNB8 net format)
- **Primary Use Case**: Chess Position Evaluation
- **Weight Quantization Scheme**: Post-hoc (float training; FT stored float32 in the file, int16 x8192 in engine RAM; dense layers int8, one scale per layer per bucket)
- **Internal Output Scale**: float centipawns (trainer), int32 centipawns (engine)

---

## 2. Input Features & Transformer Layer
- **Feature Set Name**: HalfKAv2_hm
- **Input Feature Dimension**: `[24576]` per perspective (Sparse boolean vector)
- **Feature Breakdown**:
  1. **Feature Group 1**: King bucket (32, horizontally mirrored) x piece-square (12 piece types x 64 squares = 768), kings included; at most 32 active rows
- **Perspective Symmetry**: Dual Perspective (`side-to-move` / `other-side`)

---

## 3. Perspective Accumulator Subnet
- **Input Dimension**: `[24576]` (Sparse boolean)
- **Output Dimension**: `[1536]` (Combined `[768]` side-to-move + `[768]` other side)
- **Layer Breakdown**:
  - `Sparse Linear`: Input `[24576]` -> Output `[768]` per perspective | Weights: `FP<f32 file / i16, x8192 RAM>`, Bias: `FP<f32 file / i32, x8192 RAM>`
  - `Activation`: SCReLU `clamp(in, 0, 1)^2` after the stm / nstm concatenation, requantized to `u8` (x127)

---

## 4. Subnet Selection & Bucketing Logic
- **Routing Factor**: Piece count (men on the board, kings included)
- **Bucket Selection Logic**: `Index = (piece_count - 1) * 8 / 32` (2-4, 5-8, ..., 29-32 men)
- **Number of Buckets**: `8` (Main Subnets `0` through `7`)
- **Combined Perspective State**: Concatenated `[1536]` tensor fed into the selected Main Subnet

---

## 5. Main Subnet Architecture

| Layer Index | Layer Type | Input Shape | Output Shape | Activation / Operation | Fixed-Point Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L2** | Dense Linear | `[1536]` | `[16]` | SCReLU `clamp(in, 0, 1)^2`, requantized to `u8` (x127) | `i8` weights (per-bucket scale), `f32` bias |
| **L3** | Dense Linear | `[16]` | `[32]` | SCReLU `clamp(in, 0, 1)^2`, requantized to `u8` (x127) | `i8` weights (per-bucket scale), `f32` bias |
| **Out** | Dense Linear | `[32]` | `[1]` | None | `i8` weights (own scale), `f32` bias |

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
