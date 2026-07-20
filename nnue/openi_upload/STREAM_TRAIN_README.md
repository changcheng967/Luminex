# Train v6 on the 2.9B Game-Pack (streaming — minimal C500 disk)

The pack `/hyperai/home/gamepack.xz` (3.14 GB, ~2.9B positions) is too big to featurize to
.npy (would be ~395 GB — exceeds C500 VRAM and the 20 GB upload cap). So we featurize
**on-the-fly**: `luminex-featurize --stream` reads the .xz and pipes 136-byte feature records
straight to the trainer, which shuffle-buffers a chunk in CPU RAM and feeds the GPU. **C500
disk stays at just the .xz + binaries (~3–4 GB).** The featurizer (~2–3M pos/s) outruns
training (~900K), so pipe backpressure throttles it to the train rate.

## What to upload to OpenI
**Dataset:** `gamepack.xz`  (from the SSH `/hyperai/home/gamepack.xz`)
**Code (`/tmp/code/`):**
- `luminex-featurize`  (binary, from `/hyperai/home/Luminex/build_enc/luminex-featurize`)
- `luminex_nnue_train.py`  (from `/hyperai/home/luminex_nnue_train.py` — provides the LNNUE model + save_nnue)
- `c500_train_stream.py`  (the streaming trainer)

## Run on C500 (no args; env-tunable)
```
python c500_train_stream.py
```
Env defaults: `NNUE_XZ=/tmp/dataset/gamepack.xz FEAT=/tmp/code/luminex-featurize NNUE_L1=512
NNUE_EPOCHS=1 NNUE_BS=32768 NNUE_LR=1e-3 NNUE_BUF=2000000 NNUE_FEAT_THREADS=8`

It streams the full 2.9B for 1 epoch (~54 min at 900K pos/s), checkpoints per epoch, then
saves `luminex_v2.nnue` (float) + `luminex_v2_i8.nnue` (int8, what the engine loads) and
uploads. Bench `luminex_v2_i8.nnue` vs the current v2.

## Notes
- The featurizer reads the whole decompressed frame into RAM (~8 GB) then streams records out.
  Disk stays minimal; ensure C500 has ~10 GB free RAM (it does).
- `luminex-featurize` is a Linux x86_64 binary built on the SSH; it should run on the C500 host
  as-is. If it errors on missing libs, recompile on the C500:
  `g++ -std=c++23 -O2 -I Luminex/src encode.cpp featurize.cpp ... ` (sources in Luminex/src).
- Verified: featurizer `--stream` output is byte-identical to its `.npy` output (w/b/s/t all
  match on a test frame), so the streamed features are exactly what the existing trainer expects.
