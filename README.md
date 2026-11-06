# Luminex

<p align="center"><img src="luminex.jpg" alt="Luminex" width="280"></p>

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/changcheng967/Luminex)

A UCI chess engine written in C++23. Luminex plays with an NNUE evaluation by
default — a 768-wide HalfKAv2_hm network (Gen768), since v6.3.0 trained with a
quiet-weighted loss on Stockfish-relabel data (Leela self-play positions
relabeled by deep Stockfish search — the same public training data Stockfish
itself uses; the loss weights positions by what the search actually consumes:
quiet nodes) — and ships with a self-engineered hand-crafted evaluation (HCE)
as an always-available fallback and tuning testbed. Estimated strength:
~3255 Elo blitz. On OpenBench the v6.5.0 candidate climbed the Stash ladder
one version at a time, 1,000 games a step: +104 ± 20 against Stash 29 (CCRL
Blitz 3137), +79 against Stash 30, +40 against Stash 31 and +8 ± 18 against
Stash 32 (3252), then 48% against Stash 33 (3286), where the ladder stops.
Against v6.4.0 it scores +145 ± 12 over 2,020 games.

## Download

Latest release: [github.com/changcheng967/Luminex/releases](https://github.com/changcheng967/Luminex/releases)

| Platform | Binary | CPU |
|----------|--------|-----|
| Linux | `luminex-linux-x86-64-modern` | AVX2 (x86-64-v3) |
| Linux | `luminex-linux-x86-64` | SSE4.2 and POPCNT (x86-64-v2) |
| Windows | `luminex-windows-x86-64-modern.exe` | AVX2 |
| Windows | `luminex-windows-x86-64.exe` | POPCNT, no AVX2 needed |
| macOS | `luminex-macos-arm64` | Apple silicon |

Take the modern build unless the CPU lacks AVX2; most x86 CPUs from 2013 on
have it. Every build runs the network: the modern ones with AVX2 kernels, the
others with a portable int8 path (NEON dot-product on Apple silicon).

Every release binary carries the current network (`luminex_sfq_i8.nnue`)
inside it, so the binary alone plays at full strength; it loads at the first
`isready` or `go`. To play with another network, point `NNUEFile` at its
file, which is looked for as given and then next to the binary. Left at its
default name, `NNUEFile` selects the built-in network. Each release also
attaches the network as a separate file, for builds without one inside, which
look for it in the working directory and next to the binary. Without a
loadable network the engine says so with an `info string` and uses the
hand-crafted evaluation.

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
| `Hash` | 256 | Transposition-table size in MB (allocated at the first `isready` or search) |
| `Threads` | 1 | Search threads (lazy SMP; each thread owns its NNUE accumulator) |
| `MultiPV` | 1 | Number of principal variations reported |
| `UCI_ShowWDL` | false | Append `wdl W D L` (per mille, approximate model) to info lines |
| `Ponder` | false | Allow pondering (`go ponder` / `ponderhit`) |
| `UseNNUE` | true | NNUE evaluation (automatic HCE fallback if unavailable) |
| `NNUEFile` | `luminex_sfq_i8.nnue` | Network file to load; the default name selects the built-in network |
| `Contempt` | 0 | Draw avoidance (centipawns) |
| `Move Overhead` | 30 | Time-management safety margin (ms) |
| `BookFile` | `<empty>` | Polyglot opening-book path |
| `SearchDepth` | 0 | If >0, fixed-depth search (overrides time control) |
| `NodesPerMove` | 0 | If >0, fixed node budget per move |
| `Clear Hash` | button | Clear the transposition table |

The engine also exposes 24 hand-tunable HCE parameters (`BishopPairMG`,
`RookOpenMG`, `PawnShieldCenter`, …) for evaluation experiments; see `uci.cpp`.

Options accepted for GUI/bot compatibility but **not implemented** (silently
ignored): `Skill Level`, `UCI_LimitStrength`/`UCI_Elo` (no strength limiting)
and `SyzygyPath` (no tablebase probing).

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

The current net is a 768-wide HalfKAv2_hg network, warm-started from the
Gen768 lineage and fine-tuned on Stockfish-relabel data (Leela positions with
deep-Stockfish search labels, filtered to |eval| < 800 cp and ply 8-200).
Training and data tooling live under `nnue/` and `src/lc0pack.cpp`: the
streaming trainer, the SF-relabel binpack dumper, the featurizer, quantizers,
validation-split tooling, and the SPSA search tuner.

## Build

Requires a C++23 compiler (GCC 13+, Clang 15+, MSVC 2022+) and CMake 3.20+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

That binary has no network inside and loads `luminex_sfq_i8.nnue` from the
working directory or its own directory. To build the network in, as the
releases do, fetch the pinned file and name it at configure time:

```bash
bash .github/scripts/fetch-net.sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLUMINEX_EMBED_NET=net/luminex_sfq_i8.nnue
cmake --build build --config Release
```

The network kernels are chosen at compile time: AVX-512 VNNI
(`__AVX512VNNI__`), AVX2, NEON dot-product on ARM, or a portable int8 path
elsewhere. CMake targets the build machine with GCC and Clang, and AVX2 with
MSVC; for binaries that run elsewhere, set `LUMINEX_MARCH` (GCC/Clang, e.g.
`x86-64-v2`) or
`LUMINEX_MSVC_ARCH` (MSVC and clang-cl: `AVX512`, `AVX2` or `SSE2`). The
Makefile picks the AVX-512 VNNI build by itself on CPUs that support it.

## Architecture

```
src/
  luminex.h          # Core types and constants
  types.h            # Value types, enums, MoveFlag encoding
  bitboard.h         # Bitboard operations
  magic.cpp          # Magic bitboard tables (precomputed multipliers)
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
