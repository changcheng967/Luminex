# Luminex

A UCI chess engine written in C++23.

## Features

- **Search**: PVS with LMR, null move pruning, singular extension, probcut, razoring, aspiration windows, phased move generation
- **Evaluation**: Handcrafted (PeSTO-derived) with pawn correction history (dynamic eval learning from search)
- **Move ordering**: Phased generation (TT -> captures -> quiets) with killer/counter-move/continuation history
- **Heuristics**: TT cutoff stat updates, ttPv-based LMR reduction, escape-aware ordering, previous PV bonus
- **Time management**: Sudden-death and increment time controls with stability-based allocation
- **Infrastructure**: Magic bitboards, transposition table with depth-preferred aging, eval cache, lazy SMP

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build --config Release
```

Requires C++23 compiler (MSVC, Clang, GCC) and CMake 3.15+.

## Usage

```
uci
setoption name Hash value 128
setoption name Contempt value 0
position startpos
go wtime 60000 btime 60000 winc 1000 binc 1000
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| Hash | spin | 128 | TT size in MB |
| Contempt | spin | 0 | Draw avoidance (centipawns) |

## Testing

```bash
build/luminex.exe bench
# Expected: 20, 400, 8902, 197281
```

## Strength

~3350 ELO (pure HCE, no NNUE). Tested against Stash v13 at tc=1+0.01.

## License

Open source.
