<p align="center"><img src="luminex.jpg" alt="Luminex" width="240"></p>

<h1 align="center">Luminex</h1>

<p align="center">
A UCI chess engine in C++23: an NNUE evaluation, a hand-crafted fallback,
and time management solved from measured data.
</p>

<p align="center">
  <a href="https://luminexchess.com"><b>Homepage</b></a> ·
  <a href="https://bench.luminexchess.com"><b>OpenBench</b></a> ·
  <a href="https://github.com/changcheng967/Luminex/releases"><b>Releases</b></a> ·
  <a href="https://deepwiki.com/changcheng967/Luminex"><b>Ask DeepWiki</b></a> ·
  <a href="https://github.com/computer-chess-index/cci"><b>CCI rating list</b></a>
</p>

<p align="center">
  <a href="https://github.com/changcheng967/Luminex/releases/latest"><img src="https://img.shields.io/github/v/release/changcheng967/Luminex?label=release" alt="Latest release"></a>
  <a href="https://github.com/changcheng967/Luminex/actions/workflows/build.yml"><img src="https://github.com/changcheng967/Luminex/actions/workflows/build.yml/badge.svg" alt="Build and Test"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-blue" alt="License: GPL-3.0"></a>
  <a href="https://deepwiki.com/changcheng967/Luminex"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
</p>

---

## Contents

- [Links](#links)
- [Strength](#strength)
- [Download](#download)
- [Usage](#usage)
- [How it works](#how-it-works): [evaluation](#evaluation), [search](#search), [time management](#time-management)
- [Testing](#testing)
- [NNUE training](#nnue-training)
- [Build](#build)
- [Source layout](#source-layout)
- [License](#license)

## Links

| | |
|---|---|
| **Homepage** | [luminexchess.com](https://luminexchess.com) |
| **OpenBench**, where every change is tested | [bench.luminexchess.com](https://bench.luminexchess.com) |
| **Releases** (binaries with the network built in) | [github.com/changcheng967/Luminex/releases](https://github.com/changcheng967/Luminex/releases) |
| **Ask DeepWiki** about the code | [deepwiki.com/changcheng967/Luminex](https://deepwiki.com/changcheng967/Luminex) |
| **CCI**, the Computer Chess Index rating list | [github.com/computer-chess-index/cci](https://github.com/computer-chess-index/cci) |
| **OpenBench fork** behind the test server (branch `luminex`) | [github.com/changcheng967/OpenBench](https://github.com/changcheng967/OpenBench) |

## Strength

**CCI, v6.5.1** (E1 list: Intel Core Ultra 7 265T E-cores, 1 thread, November 2026):

| STC 8+0.08 | LTC 60+0.6 | VLTC 2m24+1.12 |
|---|---|---|
| 2619 | 3054 | 3173 |

**Stash ladder, v6.5.0.** On OpenBench, 1,000 games against each Stash version in turn:

| Opponent (CCRL Blitz) | Result |
|---|---|
| Stash 29 (3137) | +104 ± 20 |
| Stash 30 | +79 |
| Stash 31 | +40 |
| Stash 32 (3252) | +8 ± 18 |
| Stash 33 (3286) | −14 (48%; the ladder stops here) |

**Gains on `main` since v6.5.1.** Each is an SPRT on OpenBench, at 10+0.1 unless noted:

| Change | Elo |
|---|---|
| Search constants retuned by SPSA (#40) | +26.5 ± 10.4 |
| Clock spent by a solved time schedule (#44) | +72.2 ± 16.7 |
| Search stopped by measured best-move stability (#45) | +8.3 ± 5.3 |
| Margin over beta for shallow null moves (#41) | +4.0 ± 3.1 |
| New time management against the old formulas, at 60+0.6 | +37.5 ± 11.0 |
| Zero-increment clocks spent by value of time, at 10+0 (#47) | +5.6 ± 4.2 |
| Moves-to-go clocks spent by value of time, at 40 moves in 10 s (#47) | +11.0 ± 6.2 |

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
inside it, so the binary alone plays at full strength. The network loads at
the first `isready` or `go`. To play with another network, point `NNUEFile`
at its file; it is looked for as given, then next to the binary. Left at its
default name, `NNUEFile` selects the built-in network. Each release also
attaches the network as a separate file for builds without one inside, which
look for it in the working directory and next to the binary. Without a
loadable network the engine says so with an `info string` and uses the
hand-crafted evaluation.

## Usage

Luminex speaks UCI; any UCI GUI, cutechess-cli, fastchess or a lichess bot
bridge can drive it.

```
uci
isready
position startpos
go movetime 1000
```

### UCI options

| Option | Default | Description |
|--------|---------|-------------|
| `Hash` | 256 | Transposition-table size in MB (allocated at the first `isready` or search) |
| `Threads` | 1 | Search threads (lazy SMP; each thread owns its NNUE accumulator) |
| `MultiPV` | 1 | Number of principal variations reported |
| `UCI_ShowWDL` | false | Append `wdl W D L` (per mille, approximate model) to info lines |
| `Ponder` | false | Allow pondering (`go ponder` / `ponderhit`) |
| `UseNNUE` | true | NNUE evaluation (falls back to the HCE when no network loads) |
| `NNUEFile` | `luminex_sfq_i8.nnue` | Network file to load; the default name selects the built-in network |
| `Contempt` | 0 | Draw avoidance (centipawns) |
| `Move Overhead` | 30 | Time-management safety margin per move (ms) |
| `TimeSchedule` | 2 | Clock schedule with an increment: 0 = the old formula, 1 and 2 = schedules solved by dynamic programming (2: early moves are worth more) |
| `TimeStability` | 1 | When to stop searching: 0 = the old best-move-stability factors, 1 = factors measured on recorded searches |
| `TimeZeroInc` | 1 | Clock schedule without an increment: 0 = the old formula, 1 = the value-weighted horizon |
| `TimeMovesToGo` | 1 | Clock schedule with moves to go: 0 = the old formula, 1 = the value-weighted horizon to the next control |
| `BookFile` | `<empty>` | Polyglot opening-book path |
| `SearchDepth` | 0 | If > 0, fixed-depth search (overrides the time control) |
| `NodesPerMove` | 0 | If > 0, fixed node budget per move |
| `Clear Hash` | button | Clear the transposition table |

Tuning options:
- **Search constants** (`LmrScale`, `NmpBase`, `FutCoeff`, `SingularMargin`, …).
  These are the parameters SPSA tunes on OpenBench; the full list and their
  ranges are in [`src/spsa_params.h`](src/spsa_params.h).
- **24 HCE parameters** (`BishopPairMG`, `RookOpenMG`, `PawnShieldCenter`, …),
  for evaluation experiments; see [`src/uci.cpp`](src/uci.cpp).

Options accepted for GUI and bot compatibility but **not implemented** (they
are ignored): `Skill Level`, `UCI_LimitStrength`/`UCI_Elo` (no strength
limiting) and `SyzygyPath` (no tablebase probing).

## How it works

### Evaluation

Two interchangeable evaluation functions, selected at runtime with `UseNNUE`:

- **NNUE (default).**
  - Inputs: a HalfKAv2_hm feature transformer with 32 king buckets and
    horizontal mirroring.
  - First layer: 768 int16 lanes per perspective, so 1,536 inputs, with
    SCReLU activations.
  - Dense tail: int8 16 → 32 → 1 layers. On AVX-512 VNNI hardware it runs on
    `VPDPBUSD` with a batched L3.
  - The accumulator is updated lazily. A move only records its change, and the
    change is applied when a position is actually evaluated (many nodes are
    answered by the hash table first). King moves rebuild from a per-square
    refresh cache.
- **HCE (fallback).** Material, piece-square tables, mobility, passed-pawn
  path decomposition, king safety and a multi-table correction history. It
  needs no network file.

### Search

- **Core:** principal-variation search with iterative deepening, aspiration
  windows and a transposition table.
- **Pruning:** reverse futility, razoring, null move (needing a margin over
  beta at shallow depth), ProbCut, futility, late-move and history pruning, and
  SEE pruning.
- **Reductions and extensions:** late-move reductions driven by history and
  node type, singular and double extensions, check extensions, and internal
  iterative reductions.
- **Move ordering:** hash move, captures by capture history, killers, counter
  moves and continuation history, with correction history on the static
  evaluation.
- **Quiescence search** with SEE filtering. **Lazy SMP** for more than one
  thread.
- The search constants are tuned by SPSA on OpenBench.

### Time management

Luminex's time management comes from measurement, not from hand-picked
formulas:

1. **Value of thinking time.** We recorded the anytime search on 6,000
   positions from SPRT games (one log line per finished iteration, up to 2M
   nodes), and Stockfish 19 scored every move the search proposed along the
   way. That gives the gain of doubling the thinking time, by game phase and
   by what the search sees, such as whether the best move just changed.
2. **The clock schedule** (`TimeSchedule 2`). A dynamic program over move
   number × clock, using the lengths of 16,732 SPRT games, spends the clock
   where time is worth most, read back as a formula. It gained **+72 Elo** at
   10+0.1.
3. **The stop rule** (`TimeStability 1`). The soft time limit follows the
   measured value of more time for each best-move stability: 2.4× the ideal
   time right after the best move changes, 0.5× once it has held for six
   iterations, and halved when the game is decided. It gained **+8.3 Elo**.
4. **Long time control.** At 60+0.6 the two together score **+37.5 Elo**
   against the old formulas.
5. **Controls without an increment, and with moves to go** (`TimeZeroInc 1`,
   `TimeMovesToGo 1`). The same dynamic program has a closed form for them:
   divide the clock above a small reserve by the expected number of moves
   left, each weighted by its value of thinking time. They gained **+5.6 Elo**
   at 10+0 and **+11.0 Elo** at 40 moves in 10 s
   ([#47](https://github.com/changcheng967/Luminex/pull/47)).

## Testing

Every change is measured on **[OpenBench](https://bench.luminexchess.com)**
before it lands:

- **SPRTs.** Bounds [0, 5] Elo at 10+0.1, Threads=1, Hash=16, with the
  2moves_v1 book, run on Xeon 8562Y+ workers. The clock is scaled to a
  reference speed, so results compare across machines.
- **Tuning.** Parameter tuning runs there as SPSA, typically 80,000 games.
- **Records.** Pull requests and commit messages carry the full SPRT result
  (Elo, LLR, games, pentanomial counts). Failed tests stay on record, and the
  pull request is closed with its numbers.

## NNUE training

The current network is a 768-wide HalfKAv2_hm net. It was warm-started from
the Gen768 lineage and fine-tuned on Stockfish-relabel data: Leela positions
with deep-Stockfish search labels, filtered to |eval| < 800 cp and plies
8–200. The loss weights quiet positions, which are what the search consumes.

The training and data tooling lives in `nnue/` and `src/`:
- the streaming trainer;
- the Stockfish-relabel binpack dumper;
- the featurizer;
- quantizers;
- validation-split tools;
- the SPSA search tuner.

## Build

Requires a C++23 compiler (GCC 13+, Clang 15+, MSVC 2022+) and CMake 3.20+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

That binary has no network inside; it loads `luminex_sfq_i8.nnue` from the
working directory or its own directory. To build the network in, as the
releases do, fetch the pinned file and name it at configure time:

```bash
bash .github/scripts/fetch-net.sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLUMINEX_EMBED_NET=net/luminex_sfq_i8.nnue
cmake --build build --config Release
```

**Choosing the network kernels.** They are selected at compile time: AVX-512
VNNI (`__AVX512VNNI__`), AVX2, NEON dot-product on ARM, or a portable int8
path elsewhere.
- **Default target.** CMake targets the build machine with GCC and Clang, and
  AVX2 with MSVC.
- **Portable binaries.** For binaries that must run on other machines, set
  `LUMINEX_MARCH` (GCC/Clang, e.g. `x86-64-v2`) or `LUMINEX_MSVC_ARCH` (MSVC
  and clang-cl: `AVX512`, `AVX2` or `SSE2`).
- **Makefile.** The `Makefile` picks the AVX-512 VNNI build by itself on CPUs
  that support it.

## Source layout

```
src/
  luminex.h             core types, constants, engine version
  types.h               value types, enums, move-flag encoding
  bitboard.h            bitboard operations
  magic.cpp             magic bitboard tables (precomputed multipliers)
  board.h / .cpp        position representation, make/unmake
  movegen.h / .cpp      legal move generation
  evaluation.h / .cpp   hand-crafted evaluation (HCE)
  eval_feat.h, eval_fitted.h, kpk_bitbase.h   HCE features, fitted weights, KPK bitbase
  tuner_params.h / .cpp HCE material and mobility tables (148 tunable values)
  nnue.h / .cpp         NNUE evaluation and the lazy incremental accumulator
  embedded_net.h / .cpp the network built into release binaries
  search.h / .cpp       search and time management
  spsa_params.h         the SPSA-tunable search constants and their ranges
  transposition.h / .cpp  transposition table
  book.h / .cpp         Polyglot opening book
  uci.h / .cpp          UCI protocol and options
  main.cpp              entry point, bench

  # NNUE data pipeline (optional CMake targets)
  lc0pack.cpp, lc0_move_table.h   Leela v6 training chunks -> gamepack frames (BUILD_LC0PACK)
  frame_merge.cpp       merges gamepack frames into denser ones
  featurize.cpp         gamepack frames -> training tensors, on the fly (BUILD_FEATURIZER)
  verify_frames.c       exhaustive frame-integrity verifier (BUILD_VERIFY)
  eval_trace.cpp        HCE evaluation tracing for tuning (BUILD_EVALTRACE)

nnue/
  DEVLOG.md, V1-GEN768.md, V2-GEN768Q.md   training log and network notes
  openi_upload/         training scripts: the C500 streaming trainer, model,
                        int8 quantizer, Stockfish-binpack dumper, validation
                        split, SPSA tuner, lichess bot setup
```

## License

Luminex is licensed under the [GNU General Public License v3.0](LICENSE).

Some foundational code derives from [Stockfish](https://github.com/official-stockfish/Stockfish),
which is also GPL-3.0. All original contributions to Luminex are released
under the same license. Training data comes from
[Leela Chess Zero](https://lczero.org/) self-play.
