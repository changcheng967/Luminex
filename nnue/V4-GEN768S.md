# NNUE Architecture Specification: Luminex gen768s

## 1. General Metadata
- **Network Name**: gen768s
- **Target Engine / Application**: Luminex 6.7+ (LNS1 net format: gen768r's LNB8 plus one error head per output bucket)
- **Primary Use Case**: Chess position evaluation, and the evaluation's own expected error, read by the search's pruning margins
- **Weight Quantization Scheme**: QAT, as gen768r_03q and 04q (feature layer on the engine's int16 grid x8192, dense layers int8 with one scale per layer per bucket). The error head trains after the value net, on its frozen accumulator, through the same int8 rounding
- **Internal Output Scale**: float centipawns (trainer), int32 centipawns (engine), for both outputs

---

## 2. Input Features & Transformer Layer
- **Feature Set Name**: HalfKAv2_hm (unchanged from gen768r)
- **Input Feature Dimension**: `[24576]` per perspective (Sparse boolean vector)
- **Feature Breakdown**:
  1. **Feature Group 1**: King bucket (32, horizontally mirrored) x piece-square (12 piece types x 64 squares = 768), kings included; at most 32 active rows
- **Perspective Symmetry**: Dual Perspective (`side-to-move` / `other-side`)

---

## 3. Perspective Accumulator Subnet
- **Input Dimension**: `[24576]` (Sparse boolean)
- **Output Dimension**: `[1536]` (Combined `[768]` side-to-move + `[768]` other side), shared by the value tail and the error head
- **Layer Breakdown**:
  - `Sparse Linear`: Input `[24576]` -> Output `[768]` per perspective | Weights: `FP<i16, x8192>`, Bias: `FP<i32, x8192>`
  - `Activation`: SCReLU `clamp(in, 0, 1)^2` after the stm / nstm concatenation, requantized to `u8` (x127)

---

## 4. Subnet Selection & Bucketing Logic
- **Routing Factor**: Piece count (men on the board, kings included)
- **Bucket Selection Logic**: `Index = (piece_count - 1) * 8 / 32` (2-4, 5-8, ..., 29-32 men)
- **Number of Buckets**: `8`; each bucket holds a value tail and an error head
- **Combined Perspective State**: Concatenated `[1536]` tensor fed into the selected bucket's value tail, and into its error head when the search asks for it

---

## 5. Main Subnet Architecture

Value tail (per bucket, every evaluation; unchanged from gen768r: the capacity test keeps L2 at 16, DEVLOG §19 S2):

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

## 6. Output Conversion & Post-Processing
- **Perspective Average Shortcut**: None
- **Output Addition**: None
- **Final Rescaling Formula**:
  $$
  \text{Value} = \text{round}(out \times 300) \qquad \text{Error} = \text{round}(e \times 300)
  $$
  Error is non-negative: the predicted size of the gap between this static evaluation and a deep search's value of the position (the label and the quantile are set by DEVLOG §19, gate 1)
- **Final Return Type**: Signed Integer (`int32`) / Centipawns, side-to-move relative (value); unsigned centipawns (error)

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
