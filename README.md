# Luminex

A UCI chess engine written in C++23 with pure handcrafted evaluation. Targeting 3500+ ELO.

## Features

- **Search**: PVS with 2D LMR table (SF16-style multiplicative), null move pruning (static R), singular extension with double/negative extensions, probcut, razoring, futility/reverse futility pruning, aspiration windows, phased move generation, qsearch with check generation, TT probe in qsearch, mate score ply adjustment (value_to_tt/value_from_tt)
- **Evaluation**: PeSTO-derived PSTs, Ethereal mobility tables, SF11-style piece-specific threats, SafetyTable king safety, pawn structure (doubled/isolated/backward/candidate/connected/phalanx/lever), passed pawns with rook behind/king proximity/free passer, bishop pair, outpost pieces, far piece penalty, space evaluation, OCB scaling, initiative/complexity bonus (SF11), minor behind pawn, weak queen penalty, endgame king safety (back-rank mate detection)
- **Move ordering**: Phased generation (TT -> captures -> quiets) with killer/counter-move/2-ply continuation history, low-ply history, capture history with fail-low malus, escape-aware ordering, previous root PV bonus
- **Heuristics**: TT cutoff stat updates with SF11 stat_bonus formula, ttPv LMR reduction, history gravity (32768 divisor), continuation pruning, TT prefetch, correction history (pawn-key based)
- **Tuning**: UCI-tunable eval parameters, mobility/threat scaling factors
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

~1840 ELO (pure HCE, no NNUE). Estimated from 80-game CuteChess tests vs Stash v13 (1972 ELO) at tc=1+0.01.

| Opponent | Score | ELO Diff | Test |
|----------|-------|----------|------|
| Stash v13 (1972) | 24-53-3 (0.319) | -132 +/- 82 | 80 games |

## Stash ELO Reference (Blitz)

| Version | ELO | Version | ELO | Version | ELO |
|---------|-----|---------|-----|---------|-----|
| v36 | 3399 | v25 | 2937 | v14 | 2060 |
| v35 | 3358 | v24 | 2880 | v13 | 1972 |
| v34 | 3328 | v23 | 2830 | v12 | 1886 |
| v33 | 3286 | v22 | 2770 | v11 | 1690 |
| v32 | 3252 | v21 | 2714 | v10 | 1620 |
| v31 | 3220 | v20 | 2509 | v9 | 1275 |
| v30 | 3166 | v19 | 2473 | v8 | 1090 |
| v29 | 3137 | v18 | 2390 | | |
| v28 | 3092 | v17 | 2298 | | |
| v27 | 3057 | v16 | 2220 | | |
| v26 | 3000 | v15 | 2140 | | |

## License

Open source.
