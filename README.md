# Luminex

A UCI chess engine written in C++23. Luminex ships with a self-engineered hand-crafted
evaluation (HCE) built from chess first principles, and an optional NNUE evaluation
(an efficiently-updatable neural network) for stronger play on AVX-512/VNNI hardware.

## Download

Latest release: [github.com/changcheng967/Luminex/releases](https://github.com/changcheng967/Luminex/releases)

| Platform | Binary |
|----------|--------|
| Linux (AVX2) | `luminex-linux-x86-64-modern` |
| Linux (SSE4.2) | `luminex-linux-x86-64` |
| Windows (ClangCL) | `luminex-windows-x86-64-modern.exe` |
| Windows (MSVC) | `luminex-windows-x86-64.exe` |
| macOS (Apple Silicon) | `luminex-macos-arm64` |

## Usage

```
uci
isready
position startpos
go movetime 1000
```

### UCI Options

| Option | Default | Description |
|--------|---------|-------------|
| `Hash` | 128 | Transposition-table size in MB |
| `Threads` | 1 | Search threads (lazy SMP; each thread owns its NNUE accumulator) |
| `UseNNUE` | false | Use the NNUE evaluation instead of HCE |
| `NNUEFile` | `luminex_v1.nnue` | Path to the `.nnue` network file |
| `SearchDepth` | 0 | If >0, search every node to a fixed depth (overrides time control) |
| `Contempt` | 0 | Draw avoidance (centipawns) |
| `BookFile` | `<empty>` | Polyglot opening-book path |

## Evaluation

Luminex has two interchangeable evaluation functions, selected at runtime via `UseNNUE`:

- **HCE (default)** — a hand-crafted evaluation: material, piece-square tables, mobility,
  passed-pawn path decomposition, king safety, and a multi-table correction history. No
  network file required.
- **NNUE (optional)** — a HalfKAv2_hg feature transformer (L1=512) feeding SCReLU
  activations through int8-quantized L2/L3/output layers (L2=16, L3=32), inferred with
  AVX-512 VNNI (`VPDPBUSD`). The accumulator is maintained incrementally on make/unmake,
  so only moved-piece feature deltas are applied per node.

On AVX-512/VNNI hardware the NNUE path reaches ~990K single-threaded nodes/sec and is
strength-even with HCE at bullet time controls while being substantially stronger at
equal search depth. If no network is loaded (or the CPU lacks AVX2), the engine
transparently falls back to HCE.

## NNUE Training Pipeline

The nets are trained on a self-built dataset derived from Leela Chess Zero's test91
self-play data: 4.29B positions / ~36.8M games / 132 gamepack frames (8.0 GB),
converted by `src/lc0pack.cpp` (Leela v6 training chunks → compact gamepack frames
with raw moves and delta-encoded evals). Leela's deep-search Q values keep the
natural fat-tail eval distribution (13.5% of positions beyond ±1000cp) — the property
that made fishtest-style thin-tail data fail for this engine.

Training uses a power-2.6 loss in sigmoid (win-probability) space — the same family
as Stockfish's `nnue-pytorch` and Berserk's trainer — with optional SF-style position
weighting. One trainer pass featurizes frames on the fly through `luminex-featurize`
(no intermediate tensor storage). Scripts live in `nnue/openi_upload/`;
`src/verify_frames.c` validates every frame (all 132 pass byte-exact closure).
Data acquisition used the overlapped download+pack pipeline in `nnue/pipeline5.sh`.

## Build

Requires a C++23 compiler (GCC 13+, Clang 15+, MSVC 2022+) and CMake 3.20+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The NNUE VNNI path auto-selects at compile time via `__AVX512VNNI__`; without it the
engine builds and runs using the AVX2 int8 path (or falls back to HCE if AVX2 is absent).

## Architecture

```
src/
  luminex.h          # Core types and constants
  types.h            # Value types, enums, MoveFlag encoding
  bitboard.h         # Bitboard operations
  magic.cpp          # Magic bitboard generation
  board.h / cpp      # Position representation, make/unmake
  movegen.h / cpp    # Legal move generation
  evaluation.h / cpp # Hand-crafted evaluation (HCE)
  nnue.h / cpp       # Optional NNUE evaluation + incremental accumulator
  search.h / cpp     # PVS search with LMR, phased move generation
  transposition.h / cpp # Transposition table
  book.h / cpp       # Polyglot opening book
  uci.h / cpp        # UCI protocol
  main.cpp           # Entry point

  # NNUE data pipeline (optional CMake targets)
  lc0pack.cpp        # Leela v6 training chunks -> gamepack frames (BUILD_LC0PACK)
  featurize.cpp      # gamepack frames -> training tensors, on the fly (BUILD_FEATURIZER)
  verify_frames.c    # exhaustive frame integrity verifier (BUILD_VERIFY)
  eval_trace.cpp     # HCE eval tracing for tuning (BUILD_EVALTRACE)

nnue/
  pipeline5.sh       # overlapped download+pack pipeline for the test91 dataset
  mae_probe.py       # net evaluation probe (cross-era MAE comparison)
  openi_upload/      # training scripts (per-frame streaming trainer, model, quantizer)
```

## License

Luminex is licensed under the [GNU General Public License v3.0](LICENSE).

Some foundational code derives from [Stockfish](https://github.com/official-stockfish/Stockfish),
which is also GPL-3.0 licensed. All original contributions to Luminex are released under
the same license.
