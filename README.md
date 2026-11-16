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

**v6.6.0 against v6.5.1** on OpenBench (one thread, 10+0.1, 2moves_v1):
**+246.55 ± 13.37** over 2,010 games (W 1292, D 653, L 65;
[test 131](https://bench.luminexchess.com/test/131/)).

**Gauntlet 3**, the development build with gen768r_05q against twelve
CCRL-rated engines (800 games each, 10+0.1;
[gauntlet 3](https://bench.luminexchess.com/gauntlet/3/)): **3471 ± 16 on
CCRL Blitz**, 3402 ± 21 on CCRL 40/15. v6.6.0's gen768r_06q adds
+3.09 ± 2.46 in self-play ([test 128](https://bench.luminexchess.com/test/128/)),
below what a repeat gauntlet resolves (about ± 10), so the rating stands
for v6.6.0. Main and the development builds now carry gen768s_03q.
gen768s_01q was +77.57 ± 15.21 over gen768r_06q in self-play
([test 140](https://bench.luminexchess.com/test/140/)), and gen768s_03q is
+17.80 ± 7.98 over gen768s_01q
([test 148](https://bench.luminexchess.com/test/148/)); the rating above
predates both.

| Opponent (CCRL Blitz) | Elo |
|---|---|
| Stash 37 (3417) | +94.3 |
| Smallbrain 7.0.0 (3454) | −13.0 |
| Rice 8.0.0 (3530) | −101.8 |
| Patricia 5 (3539) | −59.2 |
| Elixir v3.0 (3567) | −123.5 |
| Altair v7.0.0 (3579) | −160.3 |
| Renegade, Quanticade, Clover, Berserk, Obsidian, PlentyChess (3694–3773) | −350.3 to −444.4 |

**CCI, v6.5.1** (E1 list: Intel Core Ultra 7 265T E-cores, 1 thread, November 2026),
the last release CCI has rated:

| STC 8+0.08 | LTC 60+0.6 | VLTC 2m24+1.12 |
|---|---|---|
| 2619 | 3054 | 3173 |

**What v6.6.0 adds over v6.5.1.** Each is an SPRT on OpenBench against the
main of its day, at 10+0.1 unless noted:

| Change | Elo |
|---|---|
| Correction history keyed by side to move | [+61.41 ± 15.85](https://bench.luminexchess.com/test/42/) |
| Clock spent by a solved time schedule (#44) | [+72.22 ± 16.84](https://bench.luminexchess.com/test/51/) |
| Search constants retuned by SPSA (tune 45, #40; tune 54) | [+26.49 ± 10.38](https://bench.luminexchess.com/test/46/); [+5.88 ± 4.15](https://bench.luminexchess.com/test/60/) |
| Search stopped by measured best-move stability (#45) | [+8.32 ± 5.29](https://bench.luminexchess.com/test/52/) |
| Late quiet moves pruned without a history escape | [+4.86 ± 3.74](https://bench.luminexchess.com/test/43/) |
| Margin over beta for shallow null moves (#41) | [+4.00 ± 3.12](https://bench.luminexchess.com/test/47/) |
| Zero-increment clocks spent by value of time, at 10+0 (#47) | [+5.60 ± 4.17](https://bench.luminexchess.com/test/56/) |
| Moves-to-go clocks spent by value of time, at 40 moves in 10 s (#47) | [+11.01 ± 6.21](https://bench.luminexchess.com/test/57/) |
| The gen768r network: gen768r_03 with tune 65's constants, over the old network | [+60.08 ± 15.55](https://bench.luminexchess.com/test/68/) |
| then 03q (QAT), 04q, 05q, 06q, each over the one before | [+13.39](https://bench.luminexchess.com/test/73/), [+13.31](https://bench.luminexchess.com/test/82/), [+22.02](https://bench.luminexchess.com/test/112/), [+3.09](https://bench.luminexchess.com/test/128/) |
| gen768s_01q: gen768r_06 trained on Leela BT4's labels, over 06q | [+77.57 ± 15.21](https://bench.luminexchess.com/test/140/) |
| gen768s_03q: two more BT4 runs, then QAT for one int8 scale per tail row (LNR8), over 01q | [+17.80 ± 7.98](https://bench.luminexchess.com/test/148/) |

Each figure is its OpenBench page's. The hash prefetch at make-move, the
one-pass child accumulator and the lazy accumulator updates search the same
trees about 9.5% faster (+6.9% for the first two, +2.5% for the third).

## Download

Latest release: [github.com/changcheng967/Luminex/releases](https://github.com/changcheng967/Luminex/releases)

| Platform | Binary | CPU | Needs |
|---|---|---|---|
| Linux | `luminex-linux-x86-64` | any x86-64 with SSE4.2 and POPCNT (x86-64-v2) | Ubuntu 22.04 or newer (glibc 2.35) |
| Windows | `luminex-windows-x86-64.exe` | any x86-64 with POPCNT | nothing to install |
| macOS | `luminex-macos-arm64` | Apple silicon | macOS 13.3 or newer |

One binary per platform. The x86-64 binaries carry the network kernels of
every instruction-set level and use the best one the CPU supports, found at
startup; the `info string` that reports the network ends with it (`SIMD level
avxvnni`):

| Level | CPUs | Network kernels |
|---|---|---|
| `avx512vnni` | AVX-512 with VNNI: Intel Cascade Lake, Ice Lake, Sapphire Rapids and later Xeons, AMD Zen 4 and later | AVX-512 lanes, `VPDPBUSD` dense layers |
| `avx512` | AVX-512 without VNNI: Skylake-SP and Skylake-X | AVX-512 lanes, AVX2 dense layers |
| `avxvnni` | AVX2 with AVX-VNNI: Intel Alder Lake and later (Core Ultra included) | AVX2, dense layers on VEX `VPDPBUSD` |
| `avx2` | AVX2 (x86-64-v3): Intel since Haswell (2013), AMD since Excavator (2015) | AVX2 |
| `base` | every other x86-64 | SSE2; the dense layers on SSSE3's `pmaddubsw` on Linux (whose base is x86-64-v2) |

Every level computes the same evaluation, bit for bit; CI checks it on each
binary (`nnuecheck`, under Intel SDE for CPUs the runners lack). CI also
reads what every binary needs from its headers and fails a build that needs
more than its row (`.github/scripts/check-deps.py`). On Apple silicon the
network runs on NEON dot-product kernels.

Every release binary carries its network inside it (v6.6.0:
`luminex_gen768r_06q_i8.nnue`; main and the development builds:
`luminex_gen768s_03q_i8.nnue`), so the binary alone plays at full
strength, and reports it by that name. The network loads at
the first `isready` or `go`. To play with another network, point `NNUEFile`
at its file; it is looked for as given, then next to the binary. Left at its
default name, the network's own, `NNUEFile` selects the built-in network.
Each release also attaches the network under that name, for builds without
one inside, which look for it in the working directory and next to the
binary. Without a
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
| `NNUEFile` | `luminex_gen768s_03q_i8.nnue` | Network file to load; the default, the shipped network's own name, selects the built-in network |
| `SimdLevel` | auto | x86-64 release binaries only: the network kernels' instruction-set level, `auto` for the best the CPU supports, or a lower one for testing (`avx512vnni`, `avx512`, `avxvnni`, `avx2`, `base`; one the CPU lacks gives the best below it). Every level evaluates alike |
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
  - Dense tail: int8 16 → 32 → 1 layers, one per output bucket (eight, chosen
    by the number of pieces on the board). On AVX-512 VNNI hardware it runs on
    `VPDPBUSD` with a batched L3, on AVX-VNNI hardware on its VEX form.
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

The current network, gen768s_03q, keeps gen768r's architecture: gen768's
768-wide HalfKAv2_hm feature layer with a revived dense tail split into eight
output buckets, stored with one int8 scale per row of its dense layers (the
LNR8 format). What changed is the labels. gen768r_06, the gen768r line's last
network (1.95M steps of 131,072 positions on Leela Chess Zero's test91
self-play games), trained on for three more runs of 450K steps, each with a
fresh optimizer, on Leela BT4's relabel of the same games (the binpacks of
the t91 LGS store, with every position kept except those the binpacks mark to
skip, and January's last chunks held out): gen768s_01 on November 2025 and
January to August and October 2026 (6.1B positions), gen768s_02 on December
2025 and seven 2026 months, gen768s_03 on November 2025 and the 2026 months
again. The loss (power 3.0 on the win-probability error) weights quiet
positions, which are what the search consumes; the last 30K steps train
through the engine's int8 rounding, one scale per row (QAT).
[`nnue/DEVLOG.md`](nnue/DEVLOG.md) has the runs and the measurements behind
them.

The training and data tooling lives in `nnue/` and `src/`:
- the streaming trainer;
- the Stockfish-relabel binpack dumper;
- the featurizer;
- quantizers;
- validation-split tools;
- the SPSA search tuner.

Networks are named `luminex_<architecture>_<NN>.nnue`. The architecture is
the network name of its spec in `nnue/` (`gen768`, `gen768r`, ...), and `NN`
numbers that architecture's training runs in order, starting at `01`. The
trainer writes the float network under that name and its checkpoint as the
matching `.pt`; the quantized network the engine loads adds `_i8`
(`luminex_gen768r_01_i8.nnue`). The built-in network is pinned by name in
`src/nnue.h` (`DEFAULT_NET`) and by URL and sha256 in
`.github/scripts/fetch-net.sh`.

## Build

Requires a C++23 compiler (GCC 13+, Clang 15+, MSVC 2022+) and CMake 3.20+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

That binary has no network inside; it loads the shipped network,
`luminex_gen768s_03q_i8.nnue`, from the working directory or its own
directory. The name comes from `.github/scripts/fetch-net.sh`, which pins the
network: CMake reads it there, so a build always names the network it ships
with. To build the network in, as the releases do, fetch the pinned file and
ask for it at configure time:

```bash
bash .github/scripts/fetch-net.sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLUMINEX_EMBED_PINNED=ON
cmake --build build --config Release
```

**Choosing the network kernels.** The kernels (`src/nnue_kernels.inc`) are
written once for every level: AVX-512 VNNI, AVX-512, AVX-VNNI, AVX2, a
baseline, and NEON dot-product on ARM.
- **Default target.** A build has the one level its compiler targets, called
  directly: the build machine's with GCC and Clang, AVX2 with MSVC.
- **One binary for every x86-64 CPU** (the releases): `-DLUMINEX_DISPATCH=ON`
  compiles the engine for x86-64-v2 (x86-64 with POPCNT under clang-cl) and
  the kernels once per level, each file with its level's flags, and picks the
  best level the CPU and its OS support at startup (`cpuid`, `xgetbv`). It
  needs GCC, Clang or clang-cl (`-T ClangCL`).
- **One level for other machines.** Set `LUMINEX_MARCH` (GCC/Clang, e.g.
  `x86-64-v3`) or `LUMINEX_MSVC_ARCH` (MSVC and clang-cl: `AVX512`, `AVX2` or
  `SSE2`).
- **Makefile.** The `Makefile`, OpenBench's build, picks the AVX-512 VNNI build
  by itself on CPUs that support it.
- **Checking them.** `luminex nnuecheck [NETFILE]` (or `nnuecheck` at the UCI
  prompt) evaluates a fixed set of games under every level the CPU supports,
  fails on any difference, and prints a digest of the evaluations, the same
  for every x86-64 build of a network.

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
  nnue.h / .cpp         NNUE evaluation, the lazy incremental accumulator, kernel levels
  nnue_kernels.h / .inc the network's SIMD kernels, written once for every level
  nnue_k_<level>.cpp    the kernels compiled per level (LUMINEX_DISPATCH builds)
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
