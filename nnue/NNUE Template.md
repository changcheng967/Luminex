# NNUE Architecture Specification: [Network Name / Version]

## 1. General Metadata
- **Network Name**: [e.g., SFNNv16]
- **Target Engine / Application**: [e.g., Stockfish 16+]
- **Primary Use Case**: [e.g., Chess Position Evaluation]
- **Weight Quantization Scheme**: [e.g., QAT (Quantization Aware Training)]
- **Internal Output Scale**: [e.g., int64 evaluation units]

---

## 2. Input Features & Transformer Layer
- **Feature Set Name**: [e.g., HalfKAv2_hm + FullThreats + 2P Wide]
- **Input Feature Dimension**: `[N]` (Sparse boolean vector)
- **Feature Breakdown**:
  1. **Feature Group 1**: [Description & index size, e.g., King square + piece square]
  2. **Feature Group 2**: [Description & index size, e.g., Attack threat pairs]
- **Perspective Symmetry**: Dual Perspective (`side-to-move` / `other-side`)

---

## 3. Perspective Accumulator Subnet
- **Input Dimension**: `[N]` (Sparse boolean)
- **Output Dimension**: `[1024]` (Combined `[512]` "our" + `[512]` "their")
- **Layer Breakdown**:
  - `Sparse Linear`: Input `[N]` -> Output `[1024]` | Weights: `FP<i16, 256>`, Bias: `FP<i16, 256>`
  - `Activation`: ClippedReLU `clamp(in, 0, 255/256)`
  - `Element-wise Multiply`: Output `[512]` = `a[512] * b[512]`

---

## 4. Subnet Selection & Bucketing Logic
- **Routing Factor**: [e.g., Material count / Piece count]
- **Bucket Selection Logic**: `Index = (piece_count - 1) / 4`
- **Number of Buckets**: `8` (Main Subnets `0` through `7`)
- **Combined Perspective State**: Concatenated `[1024]` tensor fed into selected Main Subnet

---

## 5. Main Subnet Architecture

| Layer Index | Layer Type | Input Shape | Output Shape | Activation / Operation | Fixed-Point Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L1** | Dense Linear | `[1024]` | `[32]` | None | `FP<i32, 128*64>` |
| **L2** | Parallel Activations | `[32]` | `[64]` | SqrClippedReLU `[32]` + ClippedReLU `[32]` | `FP<i16, 125>` |
| **L3** | Dense Linear | `[64]` | `[32]` | None | `FP<i32, 128*64>` |
| **L4** | Parallel Activations | `[32]` | `[128]` | SqrClippedReLU `[32]` + ClippedReLU `[32]` | `FP<i16, 125>` |
| **L5** | Dot Product | `[128]` | `[1]` | Output Dot Product with Weights | `FP<i32, 128*125>` |

---

## 6. Output Conversion & Post-Processing
- **Perspective Average Shortcut**: `out = (our[8] - their[8]) / 2`
- **Output Addition**: Main Subnet Output + Perspective Average Offset
- **Final Rescaling Formula**:
  $$
  \text{Value} = \frac{\text{int64}(in) \times \text{Multiplier}}{\text{Denominator}}
  $$
- **Final Return Type**: Signed Integer (`int64`) / Centipawns

---

## 7. Dataflow Execution Diagram (Text Pipeline)

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