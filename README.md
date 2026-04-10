# Luminex

A UCI chess engine written in C++23 with self-engineered handcrafted evaluation. All values derived from first principles, chess theory, and mathematical reasoning. No values borrowed from other engines. Targeting 3000+ ELO before any tuning or NNUE.

## Features

- **Search**: PVS with 2D LMR table, null move pruning (static R with verification), singular extensions with double/negative extensions, probcut, razoring, futility/reverse futility pruning, aspiration windows, phased move generation, qsearch with quiet check generation, TT probe in qsearch, mate score ply adjustment
- **Evaluation**: Self-engineered piece values (MG/EG), PST tables from center-distance theory, linear mobility formulas (base + slope calibrated per piece), sigmoid king safety (Hill equation), value-ratio threat model, pawn structure (doubled/isolated/backward/candidate/connected/phalanx/lever), passed pawns with rook behind/king proximity/free passer, bishop pair, bishop same-color pawn penalty, outpost pieces, attack density, pinned piece penalty, pawn shield, pawn storm, rook on 7th/open file, space evaluation, OCB scaling, KXK endgame detection, castling evaluation
- **Move ordering**: Phased generation (TT -> captures -> quiets) with killer/counter-move/2-ply continuation history, low-ply history, capture history, escape-aware ordering, previous root PV bonus
- **Heuristics**: TT cutoff stat updates, ttPv LMR reduction, history gravity, continuation pruning, TT prefetch, correction history (pawn-key based with depth^2 weighting)
- **Tuning**: 24 UCI-tunable eval parameters via SPSA infrastructure
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

Plus 24 tunable eval parameters (BishopPairMG/EG, RookOpenMG/EG, etc.) for SPSA tuning.

## Testing

```bash
build/luminex.exe bench
# Expected: 20, 400, 8902, 197281
```

## Strength

~2030 ELO (pure HCE, no tuning, no NNUE). Estimated from 320-game CuteChess tests vs Stash v13 (1972 ELO) at tc=1+0.01.

| Version | Opponent | Score | Win% | ELO Diff | Test |
|---------|----------|-------|------|----------|------|
| v5.1.0 | Stash v13 (1972) | 179-127-14 | 58.1% | +57 | 320 games |

### Progress

| Version | Change | vs Stash v13 |
|---------|--------|-------------|
| v4.5.0 | Baseline (correction history) | 0.380 |
| v5.0.0 | Eval rewrite from first principles | 0.533 (+23 ELO) |
| v5.1.0 | Sigmoid king safety | 0.581 (+57 ELO) |

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
