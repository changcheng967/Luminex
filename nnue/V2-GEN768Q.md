# NNUE Architecture Specification: Luminex gen768q

## 1. General Metadata
- **Network Name**: gen768q
- **Target Engine / Application**: Luminex v6.4.0+ (planned)
- **Primary Use Case**: Chess Position Evaluation
- **Weight Quantization Scheme**: Post-hoc (float training; FT stored float32 in file, int16 x8192 in engine RAM; tail int8 at weight scale 64)
- **Internal Output Scale**: float centipawns (trainer), int32 centipawns (engine tail)

---

## 2. Input Features & Transformer Layer
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

## 3. Perspective Accumulator Subnet
- **Input Dimension**: per Feature Group table (Sparse boolean)
- **Output Dimension**: `[768]` per perspective
- **Layer Breakdown**:
  - `Sparse Linear`: Input `[N]` -> Output `[768]` | Weights: `FP<f32 file / i16 x8192 RAM>`, Bias: `f32`
  - `Activation`: SCReLU `clamp(in, 0, 1)^2` applied after stm/nstm concatenation
- **Warm Start**: new FT columns zero-initialized — bit-identical to T0 at step 0 (verified static AND along a walked 500-move game)

---

## 4. Subnet Selection & Bucketing Logic
- **Routing Factor**: Total piece count (men on board, kings included)
- **Bucket Selection Logic**: `Index = min(7, (32 - piece_count) / 4)`
- **Number of Buckets**: `8` — **conditional on the buckets-from-T0 dose-matched fork** (pre-registered: quiet val MAE <= -0.6cp confirms, > -0.3cp kills)
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

L2 stays 16 (widening dead on two bases); FT stays 768 (Elo ledger prices any width
change; slim-FT-640 is the only live width experiment, conditional, from scratch).

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

## Program rules (round-5b reconciliation, three-agent consensus)

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
