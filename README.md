# Luminex

A UCI chess engine written in C++23 with pure handcrafted evaluation.

## Features

- **Search**: PVS with LMR, null move pruning (static R=3/4/5), singular extension with double extension, probcut, razoring, futility/reverse futility pruning, aspiration windows, phased move generation
- **Evaluation**: PeSTO-derived PSTs, Ethereal mobility tables, Stash-style piece-specific threats, SafetyTable king safety (capped 200), pawn structure (doubled/isolated/backward/candidate/connected/phalanx/lever), passed pawns with rook behind/king proximity/free passer, bishop pair, outpost pieces, far piece penalty, space evaluation, OCB scaling
- **Move ordering**: Phased generation (TT -> captures -> quiets) with killer/counter-move/2-ply continuation history, capture history, escape-aware ordering, previous root PV bonus
- **Heuristics**: TT cutoff stat updates with SF11 stat_bonus formula, ttPv LMR reduction, history gravity (32768 divisor), continuation pruning
- **Time management**: Sudden-death and increment time controls with stability-based allocation
- **Infrastructure**: Magic bitboards, transposition table with depth-preferred aging, eval cache (512K), pawn hash (16K), lazy SMP

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build --config Release
```

Requires C++23 compiler (MSVC, Clang, GCC) and CMake 3.15+.

Linux (cloud):
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -G 'Unix Makefiles' -DCMAKE_CXX_FLAGS='-O3 -march=native -mtune=native'
cmake --build build
```

## Usage

```
uci
setoption name Hash value 128
setoption name Threads value 1
setoption name Contempt value 0
position startpos
go wtime 60000 btime 60000 winc 1000 binc 1000
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| Hash | spin | 128 | TT size in MB |
| Threads | spin | 1 | Search threads (lazy SMP) |
| Contempt | spin | 0 | Draw avoidance (centipawns) |

## Testing

```bash
build/luminex.exe bench
# Expected: 20, 400, 8902, 197281
```

## Strength

~1835 ELO (pure HCE, no NNUE). Estimated from 40-game tests against Stash v13 (1972 ELO) at tc=1+0.01.
Best confirmed result: 12-27-1 (31.25% WR, -137 ELO) vs Stash v13.

## License

Open source.
