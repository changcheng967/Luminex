# Situation: Luminex NNUE v10 net loses 0-35 at equal nodes; v2 (older net) wins 26-0. Find the root cause.

You are analyzing a chess-engine NNUE training pipeline. Analyze the evidence below
independently. Do NOT accept my hypothesis just because it's stated — your job is to
find the true root cause, consider the alternatives, and say what additional test
would discriminate between them.

## The engine (fixed across all tests)

- Luminex: alpha-beta engine with switchable eval: HCE (hand-crafted linear, 1250
  features) or NNUE (HalfKAv2_hm feature transformer, SCReLU activations, float32
  .nnue file format "LNN1"). UCI options: `UseNNUE`, `NNUEFile`.
- Engine reads L1/L2/L3 dims dynamically from the net header. Verified: engine's
  computed eval for a specific position matches a pure-Python reference implementation
  of the net EXACTLY (both say -37cp), including for the new L1=256 net. So the
  engine's inference (including L1=256) is arithmetically faithful.
- Matches run with cutechess-cli 1.5.1, both sides = SAME engine binary, Threads=1,
  nodes=100000 per move + tc=60+0.6 safety, 60 varied opening positions. Only
  `UseNNUE`/`NNUEFile` differ between the two sides.

## The three measurements (all with the same engine build, same match config)

1. **v10 net (new, L1=256) vs HCE: 0 - 35 - 0.** v10 loses every game.
   In-game searched evals for v10 stayed within roughly -75..+100cp the whole game,
   including after v10 was a full queen down (HCE side eval correctly jumped to
   +600/+1000/mate).
2. **v2 net (old, L1=512) vs HCE: 26 - 0 - 0 and counting.** v2 crushes.
   (Historical note: v2 reportedly scored ~+214 Elo at equal depth in an earlier
   session, against a different opponent.)
3. **Static-eval probes (net forward pass in pure Python, no engine):**

| position | v10 | v2 | "true" (material) |
|---|---|---|---|
| startpos | +34 | +48 | 0 |
| startpos, black queen removed (White to move) | +217 | +97 | ~+900 |
| startpos, white queen removed (Black to move) | +217 | +97 | ~+900 stm-rel |
| startpos, black rook removed | +174 | +97 | ~+500 |
| startpos, white knight removed (White to move) | -36 | -99 | ~-300 |
| real game position, Black down a queen for nothing (FEN r1bB1rk1/ppp2ppp/2n5/2bpp3/2B1P1n1/2PP1N2/PP3PPP/RN1Q1RK1 b - - 0 8) | **-37** | **-96** | ~-900 |

   Note: the startpos-minus-piece positions are out-of-distribution (never occur in
   real games); the "real game position" came from match game #1 after 8.Bxd8.
   Also note v2 prices OOD material poorly too, yet v2 plays crushingly — so OOD
   material pricing alone cannot be the whole story. The real-game queen-down
   position is the sharper discriminator (v10 -37 vs v2 -96 vs true ~-900).

4. Probe MAE on 30,497 labeled real-game positions (from the training frames):
   v10 = 82.1 (predSTD 173, target std 231), v2 = 87.3 (predSTD 198). v10 looks
   BETTER on this sample — yet plays catastrophically worse. (The sample labels come
   from the same dataset v10 trained on.)

## Training facts

- v10: fresh training, all previous bugs fixed (LR schedule, pad row, loss in sigmoid
  space, etc.). L1=256, L2=16, L3=32. 8.02B positions seen in one 2.83h block,
  AdamW, cosine LR fully annealed, bf16 autocast, power-2.6 loss in SIGMOID space
  (|sigmoid(pred/400) - sigmoid(target/400)|^2.6), output scale ×300, targets
  clamped ±1500cp. Per-frame out-of-sample MAE during training: 101.7 (frame 1) ->
  ~68-71 (frames 15-37), predSTD/tgtSTD ratio rose 0.66 -> 0.82. Training looked
  textbook-healthy.
- v2: old training (~2B positions, older trainer with known LR bug meaning effective
  live data was much smaller), L1=512, same L2=16/L3=32 head, same loss family.
- The DATASET is the big difference: v10 trained on a dataset rebuilt 2026-09-12;
  v2 trained on older data from a previous pipeline.

## The dataset (rebuilt 2026-09-12, "the frames")

Build pipeline (from pipeline_lightning.sh + pipeline.log on the build machine):

```
for f in filelist.txt:  curl https://huggingface.co/datasets/official-stockfish/fishtest_pgns/.../f.pgn.gz
~/Luminex/build/luminex-encode --threads 2 -o frames/raw_N  pgn/*.pgn.gz
xz -9e frames/raw_N -> frame_NNNN.xz     (74 frames, 15.98B positions total)
```

- Source = **official Stockfish Fishtest PGNs** (SF-vs-SF SPRT self-play games with
  `{+1.36/20 0.813s}`-style eval comments — no literal "%eval" strings; the encoder's
  fallback parser handles this format). Encoder: parses eval from PGN comments
  (pawns -> cp), quantizes to 8cp, stores WHITE-relative evals per game as:
  start_eval (int16) + per-ply deltas (int8, escape 0x80 + int16 absolute when
  |delta|>127). Decoder in featurize.cpp matches this format. 15.98B positions.
- **Label distributions measured (source = one raw fishtest .pgn.gz, 1.24M evals;
  frames = 40K labels dumped from frame_0000 via the featurizer itself):**

  | stat | SOURCE (raw PGN) | FRAMES (encoded) |
  |---|---|---|
  | mean abs eval | 179cp | 164cp |
  | >300cp | 15.9% | 16.6% |
  | >500cp | 6.5% | 6.2% |
  | >900/1000cp | 0.33% | 0.44% |
  | max | 9554cp | 8904cp |

  => **The encoding is faithful; there is NO encoder/decoder data-loss.** The fishtest
  self-play evals genuinely sit ~36% within... correction: ~63.7% of positions within
  ±100cp, with a thin decisive tail (0.3% beyond ±900cp).
- Historical project memory (from an earlier failure): "v6 died because its training
  data was balanced SF Fishtest engine-draws (median eval 48cp, only 4% >300cp, ~0%
  decisive)". A previous fix used "Gigafish" (Lichess decisive games + SF depth-10
  evals: median |eval| 316cp, 12% >1000cp) — but the Sep-12 rebuild returned to
  Fishtest PGNs (whose tail is thinner than Gigafish's by ~30x at the >1000cp mark).

## My working hypothesis (challenge it — it is PARTIALLY WRONG already)

Initially I suspected the encoder crushed the evals; measurement disproved that
(frames == source faithfully). Remaining hypothesis: the fishtest self-play
distribution itself (63.7% of positions within ±100cp, only 0.33% beyond ±900cp)
taught v10 a "everything is close to equal" prior: it never saw enough materially
wild MIDDLEGAME positions to price them, so in real play (where a queen changes
hands) its eval barely moves and the search gives material away. v2's older training
data (provenance unknown — the analyst may weigh this) apparently covered imbalanced
positions better: v2 saturates around ±661cp but still DIRECTIONALLY prices big
material, which suffices for search.

## Alternative hypotheses you must evaluate (rule in or out)

1. **Distribution/regime mismatch**: fishtest labels are 63.7% within ±100cp and
   0.33% beyond ±900cp; maybe v10 simply never learned imbalanced-middlegame pricing
   (rare in data), while its in-distribution MAE stays fine. Consider whether 8B
   positions (=> ~26M positions >900cp) should nevertheless have sufficed.
2. **Trainer-side scale weighting**: sigmoid-space power-2.6 loss with SCALE=400,
   output ×300, targets clamped ±1500 — analyze whether this loss under-weights the
   decisive tail (e.g. sigmoid compresses |cp|>600 into a tiny output interval, so
   gradient pressure to separate +300 from +900 is weak). v2 used plain sigmoid-MSE
   with the same SCALE and a ×300-ish output scale — maybe the loss change matters?
3. **L2=16 bottleneck**: both v2 and v10 have L2=16. v2 saturates ±661, v10 ±~220.
   Same width, different saturation — is 16 hidden units enough to represent the
   full range, and why would the two nets differ?
4. **L1=256 capacity** — v3 (L1=256, old data) probed 87.9 MAE ≈ v2's 87.3, but v3
   was never play-tested.
5. **Encoder/decoder delta bug** — RULED OUT by measurement (frames match source).
6. **Engine-side bug** — largely excluded (engine == Python reference exactly on v10
   evals including L1=256; v2 wins through the same binary).
7. **v2 data provenance**: what v2 trained on is unknown to us (old machine, gone).
   If you believe the data theory, v2's data likely had fat decisive tails.

## Questions to answer

1. What is the most probable root cause of v10 losing 0-35 while v2 wins 26-0 under
   identical conditions?
2. Is any single cause sufficient to explain ALL observations (v10's good probe MAE,
   healthy training curves with predSTD/tgtSTD reaching 0.82, the -37cp eval on a
   real queen-down position, v2's ±661 saturation yet dominant play)?
3. What is the cheapest discriminating test you would run NEXT, in order?
4. Fix options: (a) re-source data with fat decisive tails (Gigafish: Lichess
   decisive + SF evals, 12% >1000cp), (b) keep fishtest data but reweight/oversample
   the tail or change the loss (e.g. plain MSE, or cp-space loss), (c) fine-tune v10
   on tail-heavy data, (d) something else? Rank them.

## Files included in the upload (folder analysis_upload/)

- situation_for_analysis.md   (this file)
- src/encode.cpp              — gamepack encoder (eval parse, delta/escape writing)
- src/featurize.cpp           — decoder + the dump used to measure frame label stats
- src/nnue.cpp                — engine inference (int16 FT quant scale 8192, AVX2)
- nnue/mae_probe.py           — Python net evaluator used for the probe tables
- nnue/openi_upload/c500_train_stream.py  — trainer loop (loss, clamps, batch flow)
- nnue/openi_upload/luminex_nnue_train.py — model (arch, SCReLU, ×300, save format)
- analysis_evidence/full_game_dump_head.txt  — first 60 lines of frame label dump
- analysis_evidence/label_stats.txt          — frame label distribution (40K)
- analysis_evidence/source_vs_frames_stats.txt — corrected source-vs-frames table
- analysis_evidence/source_pgn_sample.txt    — raw fishtest PGN sample (eval format)
- analysis_evidence/pipeline_lightning.sh    — dataset build script (provenance)
- analysis_evidence/pipeline_log_head.txt    — build log (frame sizes, counts)
- analysis_evidence/v10_hce_nodes_pgn_head.txt — first games of the 0-35 match
- analysis_evidence/v2_match_score.txt       — v2-vs-HCE live score
- (optional, large) luminex_v10.nnue (25MB), luminex_v2_latest.nnue (50MB)


---

## UPDATE (post-hoc measurements — include in your analysis)

New facts established after the original writeup:

1. Engine FULLY exonerated: (a) accumulator self-check (full recompute vs incremental
   at every eval, incl. search with null moves/unmakes) silent on both a scalar Debug
   build and an AVX2 Release build with -DNNUE_FORCE_SELFCHECK; (b) engine == pure
   Python on net outputs; (c) one real latent bug found & fixed (missing *FT_WINV in
   the non-AVX2 refresh fallback) that never affected release builds.
2. v10 loses 0-60 vs HCE on the REBUILT engine too (the original 0-35 was on a stale
   Aug-31 binary; scores: v2 60-0, v10 0-60, identical engine+config).
3. Response curves (mean net eval by true-|eval| bin, signed): v10 tracks v2 closely
   (67.7/104.7/239.9/380.0/434.9 vs v2 64.2/104.0/260.0/409.4/463.8 for bins
   0-100/100-300/300-500/500-800/800-1200). Both saturate; v10 slightly flatter at
   the top. Error tails (p99/p999/max) are similar. Per-ply white-relative eval
   sensitivity similar (~20cp). => static metrics do NOT separate them; play does.
4. Node-budget utilization at "go nodes 100000": HCE 82k, v2 81k, v10 53k. v10's
   flatter evals presumably interact with eval-margin pruning (RFP/futility) and the
   iteration-abort projection, costing it depth.
5. **KEY NEW SIGNATURE — L2 weight scale by trainer era**: old trainer (uniform
   AdamW wd=1e-4): v2 l2|max|=2.16, v3=5.56, v4=5.16 (big). New trainer (tail-only
   wd=1e-2 on l2/l3/out weights, the "v6 fix"): v8=0.40, v9=0.41, v10=0.49 (crushed
   ~4-10x). v8/v9 are exactly the earlier "flat eval" nets.
6. Functional consequence measured on 8k real positions: v2's 16 L2 pre-activations
   span [-9.6,+9.3] (51% saturated at 0, 38% at 1 -> rich near-binary code;
   post-SCReLU std 0.478). v10 spans [-3.4,+5.1] with 60% dead at 0 (std 0.368).
   The bottleneck layer carries far less signal.
7. Hypothesis to evaluate: the tail weight-decay=1e-2 (introduced to stop "L2/SCReLU
   weight feedback runaway" diagnosed in the v7 era) suppresses the L2 layer's
   dynamic range; large L2 weights were not a bug but the functional operating mode.
   AdamW decay 1e-2 x lr 1e-3 x 62k steps ≈ ~50% shrinkage pressure on weights whose
   gradients don't fight back. The proposed discriminating experiment: fine-tune
   luminex_v10.pt with NNUE_TAIL_WD=1e-4, lr~3e-4, ~2h; if L2 weights grow, queen
   pricing recovers, and match results flip, decay is the primary root cause (data
   possibly innocent). Alternative/companion suspects remain: fishtest thin-tail
   data, power-2.6 sigmoid-space loss.
