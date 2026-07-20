# NNUE Game-Pack Data Pipeline

## Overview
Train the Luminex NNUE (HalfKAv2_hm, L1=512/L2=16/L3=32) on billions of Stockfish-evaluated positions from fishtest PGNs.

## Pipeline

### 1. Encode (`src/encode.cpp`)
Multi-threaded (one shared FEN dict, no merge needed). PGN → Game-Pack frame:
- Frame: `[u64 len_hdr][hdr][u64 len_mv][mv][u64 len_ev][ev]`
- `mv`: 1 byte/pos (move index into sorted legal list).
- `ev`: delta-encoded evals (white-perspective), quantized to 8 cp.
- `n_pos` is **uint16** (max 65535 plies — no 255 truncation).
- **Cascade-safe**: on bad_san, skip to next game (no wrong-position cascade).

### 2. Featurize (`src/featurize.cpp`)
HalfKAv2_hm feature extraction:
- mmap mode (`--input file`) for low-RAM machines (~6 GB RSS).
- stdin mode for pipe (`xz -dc | featurize --stream`).
- SIGSEGV backtrace handler + skip-on-divergence (robust).

### 3. Multi-stage encode (`nnue/openi_upload/multistage_encode.py`)
For the 20 GB disk constraint:
- Download fishtest PGNs in parallel (6 concurrent).
- Encode each batch → `frame_K.xz` (atomic + xz-verified, shutdown-safe).
- Paginated HF listing (follows API cursor past 1000-entry truncation).
- Trainer loops over the frames (no merge).

### 4. Train (`nnue/openi_upload/c500_train_stream.py`)
C500 streaming trainer: `xz -dc frame_K.xz | luminex-featurize --stream` per frame, shuffle-buffer, GPU train. Cosine LR annealed over the budget. int8 quantize done locally post-training.
