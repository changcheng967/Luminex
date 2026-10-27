# NNUE Architecture Specification: Luminex gen768q

## 1. General Metadata
- **Network Name**: gen768q
- **Target Engine / Application**: Luminex v6.4.0+ (planned)
- **Primary Use Case**: Chess Position Evaluation
- **Weight Quantization Scheme**: Post-hoc (float training; FT stored float32 in file, int16 x8192 in engine RAM; tail int8 at weight scale 64)
- **Internal Output Scale**: float centipawns (trainer), int32 centipawns (engine tail)

---

## 2. Input Features & Transformer Layer
- **Feature Set Name**: HalfKAv2_hm + Pawn Conjunctions + Shelter
- **Input Feature Dimension**: `[25856]` (Sparse boolean vector)
- **Feature Breakdown**:
  1. **Feature Group 1**: King square x piece-square, both colors, horizontal mirror, 24576 indices (gen768 set unchanged)
  2. **Feature Group 2**: Passed pawn, perspective x relative colour x oriented square, 256 indices
  3. **Feature Group 3**: Isolated pawn, same indexing, 256 indices
  4. **Feature Group 4**: Doubled pawn, same indexing, 256 indices
  5. **Feature Group 5**: King shelter, perspective x king bucket (32, oriented) x shield class (8: popcount(own shield pawns) x 2 + storm bit), 512 indices, exactly one active per perspective
- **Perspective Symmetry**: Dual Perspective (`side-to-move` / `other-side`)

---

## 3. Perspective Accumulator Subnet
- **Input Dimension**: `[25856]` (Sparse boolean)
- **Output Dimension**: `[768]` per perspective
- **Layer Breakdown**:
  - `Sparse Linear`: Input `[25856]` -> Output `[768]` | Weights: `FP<f32 file / i16 x8192 RAM>`, Bias: `f32`
  - `Activation`: SCReLU `clamp(in, 0, 1)^2` applied after stm/nstm concatenation

---

## 4. Subnet Selection & Bucketing Logic
- **Routing Factor**: Total piece count (men on board, kings included)
- **Bucket Selection Logic**: `Index = min(7, (32 - piece_count) / 4)` (bucket 0 = full board, 7 = bare kings)
- **Number of Buckets**: `8` (Main Subnets `0` through `7`)
- **Combined Perspective State**: Concatenated `[1536]` tensor (stm `[768]` ++ nstm `[768]`) fed into the selected Main Subnet

---

## 5. Main Subnet Architecture

| Layer Index | Layer Type | Input Shape | Output Shape | Activation / Operation | Fixed-Point Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L2** | Dense Linear | `[1536]` | `[16]` | None | `i8 weights, scale 64` |
| **A2** | Activation | `[16]` | `[16]` | SCReLU `clamp(in,0,1)^2` | requant `min(v>>6, 127)` |
| **L3** | Dense Linear | `[16]` | `[32]` | None | `i8 weights, scale 64` |
| **A3** | Activation | `[32]` | `[32]` | SCReLU `clamp(in,0,1)^2` | requant `min(v>>6, 127)` |
| **Out** | Dense Linear | `[32]` | `[1]` | None | `i8, own scale` |

All 8 Main Subnets share this shape; each owns independent weights. L2 stays 16
(widening to 32 measured dead on two bases); FT stays 768 (the nps curve is
convex toward smaller widths; inputs beat width).

---

## 6. Output Conversion & Post-Processing
- **Perspective Average Shortcut**: None
- **Output Addition**: None
- **Final Rescaling Formula**:
  $$
  \text{Value} = \text{float}(out) \times 300 = \text{centipawns (stm-relative)}
  $$
- **Final Return Type**: Float / Centipawns (stm-relative; negamax consumes directly)

---

## 7. Dataflow Execution Diagram (Text Pipeline)

```text
[ Board State ]
│
├──► Side-To-Move Perspective ──► Sparse Linear [25856->768] ──┐
│                                                               ├─► concat [1536]
└──► Opponent Perspective    ──► Sparse Linear [25856->768] ──┘        │
                                                                        ▼
                                                         SCReLU clamp(0,1)^2
                                                                        │
                                                                        ▼
                                                        Bucket Routing Logic
                                                        min(7, (32 - men) / 4)
                                                                        │
                      ┌─────────────────────────────────────────────────┴─────────────────────────────┐
                      ▼                                                                                   ▼
                [ Main Subnet 0 ]                                                                   [ Main Subnet 7 ]
                      │                                                                                   │
                Dense Linear [1536->16]                                                              Dense Linear [1536->16]
                      │                                                                                   │
                SCReLU clamp(0,1)^2                                                                  SCReLU clamp(0,1)^2
                      │                                                                                   │
                Dense Linear [16->32]                                                                Dense Linear [16->32]
                      │                                                                                   │
                SCReLU clamp(0,1)^2                                                                  SCReLU clamp(0,1)^2
                      │                                                                                   │
                Dense Linear [32->1]                                                                 Dense Linear [32->1]
                      └─────────────────────────────────────────────────┬─────────────────────────────┘
                                                                        │
                                                                        ▼
                                                                  x 300 (cp)
                                                                        │
                                                                        ▼
                                                            [ Evaluation Output ]
```
