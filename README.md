<p align="center"><img src="luminex.jpg" alt="Luminex" width="240"></p>

<h1 align="center">Luminex</h1>

<p align="center">
A UCI chess engine in C++23. Its NNUE evaluation also reads which pieces
attack which, every change to its search is tested on OpenBench, and its
time management is solved from measured data.
</p>

<p align="center">
  <a href="https://luminexchess.com"><b>Homepage</b></a> ·
  <a href="https://github.com/changcheng967/Luminex/releases"><b>Download</b></a> ·
  <a href="https://bench.luminexchess.com"><b>OpenBench</b></a> ·
  <a href="https://github.com/computer-chess-index/cci/blob/main/engines/Luminex.md"><b>CCI ratings</b></a> ·
  <a href="https://github.com/changcheng967/Luminex-NNUE"><b>Trainer</b></a>
</p>

<p align="center">
  <a href="https://github.com/changcheng967/Luminex/releases/latest"><img src="https://img.shields.io/github/v/release/changcheng967/Luminex?label=release" alt="Latest release"></a>
  <a href="https://github.com/changcheng967/Luminex/actions/workflows/build.yml"><img src="https://github.com/changcheng967/Luminex/actions/workflows/build.yml/badge.svg" alt="Build"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-blue" alt="License: GPL-3.0"></a>
  <a href="https://deepwiki.com/changcheng967/Luminex"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
</p>

## Strength

v6.7.0, the latest release:

| Measured | Result |
|---|---|
| Against v6.6.0, 2,010 games ([test 321](https://bench.luminexchess.com/test/321/)) | **+314.92 ± 14.52** Elo |
| Against eleven CCRL-rated engines, 8,876 games ([gauntlet 8](https://bench.luminexchess.com/gauntlet/8/)) | **3662 ± 8** CCRL Blitz, 3544 ± 11 CCRL 40/15 |

The games are played on OpenBench on one thread with 16 MB of hash, at
10+0.1 scaled to each worker's speed, from the 2moves_v1 openings. Earlier
releases' results are in their
[release notes](https://github.com/changcheng967/Luminex/releases).

## Download

Both kinds of build are on [Releases](https://github.com/changcheng967/Luminex/releases):

- **v6.7.0** is the latest release.
- **The development build** is `main` as it stands: the pre-release named
  `dev-<date>-<commit>`, rebuilt on every push to `main`. Only the newest is kept.

Every binary has its network built in, so the binary alone plays at full
strength. The network is also attached as a file, for builds without one
inside. There is one binary per platform:

| Platform | Binary | CPU | Needs |
|---|---|---|---|
| Linux | `luminex-linux-x86-64` | x86-64-v2 (SSE4.2, POPCNT) | glibc 2.35 or newer (Ubuntu 22.04) |
| Windows | `luminex-windows-x86-64.exe` | x86-64 with POPCNT | nothing to install |
| macOS | `luminex-macos-arm64` | Apple silicon | macOS 13.3 or newer |

The x86-64 binaries carry the network kernels for every instruction-set
level and use the best the CPU supports: AVX-512 VNNI, AVX-512, AVX-VNNI,
AVX2 or a baseline. Every level evaluates alike, bit for bit. The `info
string` that names the network ends with the level in use.

## Usage

Luminex speaks UCI. Any UCI GUI, cutechess-cli, fastchess or a lichess bot
bridge can run it.

```
uci
isready
position startpos moves e2e4
go wtime 60000 btime 60000 winc 600 binc 600
```

`luminex bench` searches a fixed set of positions and prints the nodes and
the speed. It is the bench OpenBench runs.

| Option | Default | Description |
|---|---|---|
| `Hash` | 256 | Hash table in MB, allocated at the first `isready` or search |
| `Threads` | 1 | Search threads, 1 to 64 |
| `MultiPV` | 1 | Principal variations reported |
| `Ponder` | false | Declared so GUIs offer pondering; `go ponder` and `ponderhit` work whatever its value |
| `UCI_ShowWDL` | false | Add `wdl` (win, draw and loss per mille) to the info lines |
| `UseNNUE` | true | Evaluate with the network; false uses the hand-crafted evaluation |
| `NNUEFile` | `luminex_gen768u_01q_i8.nnue` | The network. The default name selects the built-in network; any other file is looked for as given, then next to the binary |
| `SimdLevel` | auto | x86-64 release binaries: the kernels' level (`avx512vnni`, `avx512`, `avxvnni`, `avx2`, `base`), for testing |
| `Contempt` | 0 | Draw avoidance, in centipawns |
| `Move Overhead` | 30 | Milliseconds kept back on each move for the GUI and the connection |
| `TimeSchedule` | 2 | Schedule with an increment: 0 is the old formula, 1 and 2 are solved by dynamic programming (2 values early moves more) |
| `TimeStability` | 1 | When to stop: 0 uses the old stability factors, 1 the measured ones |
| `TimeZeroInc` | 1 | Schedule without an increment: 0 is the old formula, 1 the value-weighted horizon |
| `TimeMovesToGo` | 1 | Schedule with moves to go: 0 is the old formula, 1 the value-weighted horizon to the next control |
| `TimeLag` | 1 | Without an increment: 1 also pays for the lag the clock shows on each move, 0 does not |
| `BookFile` | empty | Polyglot opening book |
| `SearchDepth` | 0 | Above 0: a fixed depth (up to 30) instead of the clock |
| `NodesPerMove` | 0 | Above 0: a fixed node budget per move |
| `Clear Hash` | button | Empty the hash table |

For tuning, the search constants SPSA tunes (`SPSAParams` in
[`src/luminex.h`](src/luminex.h)) are options too. A spin option's value is
clamped into its declared range. There is no strength limit and no tablebase
probing, so the engine declares no `UCI_LimitStrength`, `UCI_Elo`,
`Skill Level` or `SyzygyPath`. An option it does not declare gets an
`info string` reply and is ignored.

## How it works

### Evaluation

The network is V6, gen768u ([spec](docs/NNUE.md#v6-gen768u)):

- **Feature layer.** HalfKAv2_hm (32 king buckets, mirrored horizontally,
  times 768 piece-squares) feeds 768 int16 lanes for each side, from one of
  three tables chosen by the men on the board: 26 or more, 13 to 25, or 12 or
  fewer. Splitting the table by piece count is an idea the Triumviratus
  engine published; Stockfish's K32H3 and nettest #531's K32Q2P4 are related
  work. Beside it, 55,960 relation features add int8 rows to 256 of those
  lanes: which piece attacks which, and pawn pairs in contact. These are
  threat inputs, which other engines had first: Stockfish's Full_Threats
  (crediting Monty, Yukari and PlentyChess), PlentyChess, Viridithas,
  Stormphrax, Berserk, Caissa, Integral and Triumviratus, and Stockfish's
  PP_3Wide (from Pawnocchio) for the pawn pairs. The relation set, its
  encoding into 256 lanes and the measurements are ours.
- **Tail.** SCReLU, then one of eight int8 tails, chosen by the number of
  men on the board: 1,536 → 16 → 32 → 1, with one scale per row.
- **Updates.** A move only records what it changes, and the accumulator
  catches up when a position is evaluated, since many nodes end at the hash
  table first. King moves, and captures that move the men into another
  table, rebuild from a refresh cache.
- **Kernels.** The kernels are written once for AVX-512 VNNI, AVX-512,
  AVX-VNNI, AVX2, SSE and NEON. Every level gives the same evaluation.
- **Bare kings.** Endings against a bare king are scored before the
  network, which has hardly ever seen one.

Without a network, or with `UseNNUE` off, the hand-crafted evaluation plays.
It scores material, piece-square tables, mobility, pawn structure with
passed-pawn paths, king safety and known endgames.

### Search

- **Core.** Principal-variation search with iterative deepening,
  aspiration windows and a hash table. A move prefetches its child's hash
  entry.
- **Pruning.** Reverse futility, razoring, null move, ProbCut, futility,
  late-move and history pruning. A null move needs a margin over beta at
  shallow depth and a piece besides pawns.
- **Reductions and extensions.** Late-move reductions driven by history and
  node type. Singular extensions, with double and negative extensions and
  multi-cut. Check extensions and internal iterative reductions.
- **Move ordering.** Hash move, captures by capture history, killers,
  counter moves and continuation history.
- **Correction history.** The static evaluation is corrected by pawn,
  non-pawn, minor-piece, major-piece and king-pawn tables, keyed by side to
  move.
- **Quiescence search** with SEE filtering.
- **Threads.** Lazy SMP. Every helper runs the main thread's own root
  iteration, so the helpers search its tree and fill its hash table.
- **Constants.** SPSA tunes them on OpenBench.

### Time management

The time management is measured and then solved, not hand-picked:

1. **Value of time.** We recorded the searches of 6,000 positions from SPRT
   games, and Stockfish 19 scored every move proposed along the way. That
   gives the gain from doubling the thinking time, by game phase and by how
   stable the best move is.
2. **Schedule** (`TimeSchedule 2`). A dynamic program over move number and
   clock, on the lengths of 16,732 games, spends the clock where time is
   worth most:
   [+72.22 ± 16.84](https://bench.luminexchess.com/test/51/) at 10+0.1.
3. **Stop rule** (`TimeStability 1`). The soft limit follows the measured
   value of more time at each stability. It is 2.4 times the ideal time just
   after the best move changes and 0.5 times once it has held for six
   iterations: [+8.32 ± 5.29](https://bench.luminexchess.com/test/52/).
4. **No increment, and moves to go** (`TimeZeroInc 1`, `TimeMovesToGo 1`).
   This is the same program in closed form: the clock above a small reserve,
   divided by the value-weighted number of moves left.
   It scored [+5.60 ± 4.17](https://bench.luminexchess.com/test/56/) at 10+0
   and [+11.01 ± 6.21](https://bench.luminexchess.com/test/57/) at 40 moves
   in 10 s.
5. **Pondering.** Time spent pondering counts as thinking time on the move,
   so a ponder hit continues the budget instead of starting a new one.

## Networks

The networks are trained with
[Luminex-NNUE](https://github.com/changcheng967/Luminex-NNUE). It holds the
GPU trainer, quantization-aware training, the data converters and network
forensics, and its
[development log](https://github.com/changcheng967/Luminex-NNUE/blob/main/docs/DEVLOG.md)
records every run. The current network learned only from Leela Chess Zero
data, every position relabelled by its BT4 network: test91's games
(standard and Chess960) and test80 games from the BT4 training corpus.
[`docs/NNUE.md`](docs/NNUE.md) specifies every architecture, V1 to V6.

A network's name gives its architecture and run. For example,
`luminex_gen768u_01q_i8.nnue` is gen768u's first run, finished with
quantization-aware training (`q`) and quantized to int8 (`_i8`). The built-in
network is pinned in [`.github/scripts/ci.sh`](.github/scripts/ci.sh)
(`pin_url`, `pin_sha256`), and CMake reads its name there.

## Testing

Every change is measured on [OpenBench](https://bench.luminexchess.com)
before it lands. The server runs a fork,
[changcheng967/OpenBench](https://github.com/changcheng967/OpenBench)
(branch `luminex`).

- **SPRT.** Bounds [0, 5] Elo at 10+0.1, one thread, 16 MB hash, 2moves_v1
  openings. The clock is scaled to each worker's measured speed, so machines
  compare.
- **SPSA** tunes the search constants.
- **Gauntlets** against CCRL-rated engines give the rating.

Every pull request and merge commit carries its test's numbers. A change
that fails is closed with them.

## Build

You need C++23, CMake 3.20 or newer, and clang 23. Clang 23 builds main,
every release from v6.7.0 and every OpenBench worker, and CI checks it. With
lld and without PIE it is the fastest build measured; CMakeLists.txt sets
both for clang on Linux.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++-23
cmake --build build --config Release -j
```

That binary has no network inside. It loads `luminex_gen768u_01q_i8.nnue`
from the working directory or its own. To build the network in, as the
releases do:

```bash
bash .github/scripts/ci.sh fetch-net
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++-23 -DLUMINEX_EMBED_PINNED=ON
cmake --build build --config Release -j
```

- **Instruction-set level.** By default a build targets its own machine.
  `-DLUMINEX_DISPATCH=ON` makes the releases' one binary for every x86-64
  CPU: the engine for x86-64-v2, the kernels once per level, and the best
  chosen at startup. `-DLUMINEX_MARCH=x86-64-v3` builds one fixed level.
- **Windows.** The releases build with Visual Studio's ClangCL toolset on
  LLVM 23 (`-G "Visual Studio 17 2022" -A x64 -T ClangCL`).
- **OpenBench.** The `Makefile` is OpenBench's build: `make -j EXE=luminex`.
- **Checks.** `luminex nnuecheck [NETFILE]` evaluates a fixed set of games
  at every level the CPU supports and fails on any difference.

## Source layout

```
src/
  luminex.h        the one header: types, bitboards, the position, move generation,
                   the hash table, the evaluation, the network's interface, search,
                   the SPSA-tunable constants, UCI, the version
  board.cpp        the position (make and unmake), move generation, magic bitboards,
                   the hash table
  evaluation.cpp   the hand-crafted evaluation, its tuned values, the KPK bitbase
  nnue.cpp         the network: loading, the lazy accumulator, the relation inputs,
                   the kernel levels, the built-in network
  nnue_kernels.h   the network's SIMD kernels, written once for every level
  search.cpp       search, time management, the threads
  uci.cpp          the UCI protocol and options, the opening book, bench, main
  tools.cpp        data tools for training, one optional CMake target each
docs/NNUE.md       the network specs, V1 to V6
.github/           CI and releases: workflows/build.yml, scripts/ci.sh
Makefile           OpenBench's build
```

## License

Luminex is licensed under the [GNU General Public License v3.0](LICENSE).
Some of its foundations derive from
[Stockfish](https://github.com/official-stockfish/Stockfish), and some
search and evaluation code from [Stash](https://github.com/mhouppin/stash-bot)
by Morgan Houppin, both also GPL-3.0.
The training data comes from [Leela Chess Zero](https://lczero.org/).
