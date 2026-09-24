# Luminex

A UCI chess engine written in C++23. Luminex plays with an NNUE evaluation by
default — a 768-wide HalfKAv2_hg network (Gen768, function-preserving Net2Net
widening of the 512-wide predecessor) — and ships with a self-engineered
hand-crafted evaluation (HCE) as an always-available fallback and tuning
testbed.

## Download

Latest release: [github.com/changcheng967/Luminex/releases](https://github.com/changcheng967/Luminex/releases)

| Platform | Binary |
|----------|--------|
| Linux (AVX2) | `luminex-linux-x86-64-modern` |
| Linux (SSE4.2) | `luminex-linux-x86-64` |
| Windows (ClangCL) | `luminex-windows-x86-64-modern.exe` |
| Windows (MSVC) | `luminex-windows-x86-64.exe` |
| macOS (Apple Silicon) | `luminex-macos-arm64` |

Each release also attaches the current network (`luminex_gen768_i8.nnue`).
Place it next to the binary — NNUE then loads automatically. Without the file
(or on hardware without AVX2) the engine transparently falls back to HCE.

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
| `Hash` | 256 | Transposition-table size in MB |
| `Threads` | 1 | Search threads (lazy SMP; each thread owns its NNUE accumulator) |
| `UseNNUE` | true | NNUE evaluation (automatic HCE fallback if unavailable) |
| `NNUEFile` | `luminex_gen768_i8.nnue` | Path to the `.nnue` network file |
| `Ponder` | false | pondering |
| `Contempt` | 0 | Draw avoidance (centipawns) |
| `Move Overhead` | 10 | Time-management safety margin (ms) |
| `BookFile` | `<empty>` | Polyglot opening-book path |
| `SyzygyPath` | `<empty>` | Syzygy tablebase path |
| `Skill Level` | 20 | 0–20 strength handicap |
| `UCI_LimitStrength` / `UCI_Elo` | false / 1320 | Elo-capped play (1320–3190) |
| `UCI_Chess960` | false | Chess960 / FRC mode |
| `UCI_ShowWDL` | false | Report win/draw/loss in search output |
| `SearchDepth` | 0 | If >0, fixed-depth search (overrides time control) |
| `NodesPerMove` | 0 | If >0, fixed node budget per move |
| `QsearchLinear` | true | Qsearch stand-pat via the net's linear head (inert unless the loaded net carries a trained one) |

The engine also exposes ~40 hand-tunable HCE parameters (`BishopPairMG`,
`RookOpenMG`, `PawnShieldCenter`, …) for evaluation experiments; see `uci.cpp`.

## Evaluation

Luminex has two interchangeable evaluation functions, selected at runtime via
`UseNNUE`:

- **NNUE (default)** — HalfKAv2_hg feature transformer (L1=768, two int16
  saturating accumulator perspectives) feeding SCReLU activations through
  int8-quantized L2/L3/output layers (16→32→1). The accumulator is maintained
  incrementally on make/unmake with fused single-pass move deltas; the tail
  runs AVX-512 VNNI (`VPDPBUSD`) with a batched L3 GEMM. Measured up to ~1.2M
  single-threaded nodes/sec on AVX-512/VNNI hardware with the 768-wide net.
- **HCE (fallback)** — material, piece-square tables, mobility, passed-pawn
  path decomposition, king safety, and a multi-table correction history. No
  network file required.

## NNUE Training

The current net is a 768-wide HalfKAv2_hg network obtained by
function-preserving Net2Net widening of the 512-wide predecessor. Training and
data tooling live under `nnue/` and `src/lc0pack.cpp`.

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
  nnue.h / cpp       # NNUE evaluation + incremental accumulator
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
  mae_probe.py       # net evaluation probe (cross-era MAE comparison)
  openi_upload/      # training scripts (per-frame streaming trainer, model, quantizer)
```

## License

Luminex is licensed under the [GNU General Public License v3.0](LICENSE).

Some foundational code derives from [Stockfish](https://github.com/official-stockfish/Stockfish),
which is also GPL-3.0 licensed. All original contributions to Luminex are released under
the same license. Training data from [Leela Chess Zero](https://lczero.org/) self-play.
