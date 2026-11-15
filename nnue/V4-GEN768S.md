# NNUE Architecture Specification: Luminex gen768s

## 1. General Metadata
- **Network Name**: gen768s
- **Target Engine / Application**: Luminex. gen768s_01q has been main's net since #84 (value net only, LNB8). LNR8 (PR #89) is LNB8 with one int8 scale per tail row. The heads are an optional LHE1 section after the tails (PR #87), and engines without that code skip it
- **Primary Use Case**: Chess position evaluation. Per position the net also gives the evaluation's expected error (the error head, read by the search's pruning margins) and the chance of a draw (the draw head, stored but not read by the search)
- **Weight Quantization Scheme**: QAT. The feature layer is on the engine's int16 grid x8192, as in gen768r. The dense layers are int8:
  - gen768s_01q and 02q: one scale per layer per bucket (LNB8).
  - gen768s_02qr: one scale per row (output neuron) of L2 and L3 (LNR8), from a QAT that rounds each row on its own scale.

  In the float phase, a draw logit of the heads' form trains jointly with the value (binary cross-entropy, weight 1e-3). It shapes the accumulator, and it is dropped before QAT. Both shipped heads are fitted on the QAT net's activations (ridge regression per bucket), with int8 weights, one scale per head per bucket
- **Internal Output Scale**: value: float centipawns (trainer), int32 centipawns (engine). Heads: expected-score units in `[0, 1]`, in steps of 1/10000 in the engine

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
- **Output Dimension**: `[1536]` (Combined `[768]` side-to-move + `[768]` other side), shared by the value tail and both heads
- **Layer Breakdown**:
  - `Sparse Linear`: Input `[24576]` -> Output `[768]` per perspective | Weights: `FP<i16, x8192>`, Bias: `FP<i32, x8192>`
  - `Activation`: SCReLU `clamp(in, 0, 1)^2` after the stm / nstm concatenation, requantized to `u8` (x127)

---

## 4. Subnet Selection & Bucketing Logic
- **Routing Factor**: Piece count (men on the board, kings included)
- **Bucket Selection Logic**: `Index = (piece_count - 1) * 8 / 32` (2-4, 5-8, ..., 29-32 men)
- **Number of Buckets**: `8`; each bucket holds a value tail and one readout of each head
- **Combined Perspective State**: Concatenated `[1536]` `u8` tensor. It feeds the selected bucket's value tail on every evaluation, and its error head when the search reads it (`HeadK > 0`). The engine never computes the draw head

---

## 5. Main Subnet Architecture

Value tail (per bucket, every evaluation; unchanged from gen768r):

| Layer Index | Layer Type | Input Shape | Output Shape | Activation / Operation | Fixed-Point Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L2** | Dense Linear | `[1536]` | `[16]` | SCReLU `clamp(in, 0, 1)^2`, requantized to `u8` (x127) | `i8` weights (one scale per bucket in LNB8, per row in LNR8), `f32` bias |
| **L3** | Dense Linear | `[16]` | `[32]` | SCReLU `clamp(in, 0, 1)^2`, requantized to `u8` (x127) | `i8` weights (one scale per bucket in LNB8, per row in LNR8), `f32` bias |
| **Out** | Dense Linear | `[32]` | `[1]` | None | `i8` weights (own scale), `f32` bias |

Heads (per bucket; each a single linear readout with no hidden layer, 1,536 multiply-adds):

| Layer Index | Layer Type | Input Shape | Output Shape | Activation / Operation | Fixed-Point Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Error** | Linear readout | `[1536]` activations + `\|v\|/400` | `[1]` | clamp to `[0, 1]` | `i8` weights (one scale per head per bucket), `f32` coefficient of `\|v\|/400`, `f32` bias |
| **Draw** | Linear readout | `[1536]` activations + `\|v\|/400` | `[1]` | clamp to `[0, 1]` | `i8` weights (one scale per head per bucket), `f32` coefficient of `\|v\|/400`, `f32` bias |

`v` is the value tail's output for the same position.

---

## 6. Output Conversion & Post-Processing
- **Perspective Average Shortcut**: None
- **Output Addition**: None
- **Final Rescaling Formula**:
  $$
  \text{Value} = \text{round}(out \times 300) \qquad e = \text{clamp}\Big(\frac{q \cdot a}{127\,s} + w_p \frac{|v|}{400} + b,\ 0,\ 1\Big)
  $$
  Here `q` are a head's int8 weights, `s` their scale, `a` the `[1536]` activation bytes, and `w_p` and `b` the head's float coefficient and bias. Each head stores `round(e x 10000)`.
  - Error head: trained toward `|sigmoid(v/400) - sigmoid(t/400)|`, the expected-score gap between the value and its training target `t`. The search converts it to centipawns at the position's eval: `cp = min(400, e x 400 / sigmoid'(v/400))`, with `sigmoid'` floored at 0.01.
  - Draw head: trained toward "the game was drawn".
- **Final Return Type**: value: signed integer (`int32`) centipawns, side-to-move relative. Heads: integer, 1/10000 of expected score

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
                     ┌────────────────────────────────────────────────────┼──────────────────────────────┐
                     ▼                                                    ▼                              ▼
        [ Value tail, bucket b ]                          [ Error head, bucket b ]          [ Draw head, bucket b ]
                     │                                    (when the search reads it)          (not computed by the
             Dense Linear (16)                                            │                         engine)
                     │                                       Linear readout + |v|/400                    │
             SCReLU clamp(0,1)^2                                          │                  Linear readout + |v|/400
                     │                                             clamp [0, 1]                          │
             Dense Linear (32)                                            │                         clamp [0, 1]
                     │                                     x 400 / sigmoid'(v/400), <= 400               │
             SCReLU clamp(0,1)^2                                          │                              ▼
                     │                                                    ▼                     [ P(draw) ]
             Dense Linear (1)                                 [ Expected error (cp) ]
                     │                                       read by the pruning margins
               x 300 -> int32
                     │
                     ▼
          [ Evaluation Output ] ──── v ────► both heads' |v|/400 input
```
