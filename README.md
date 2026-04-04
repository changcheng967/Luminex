# Luminex

A UCI chess engine written in C++23.

## Features

- **Search**: PVS with LMR, null move pruning, singular extension, probcut, razoring, aspiration windows
- **Evaluation**: Handcrafted with correction history (dynamic eval learning from search results)
- **Move ordering**: Phased generation (TT → captures → quiets) with killer/counter-move/continuation history
- **Time management**: Sudden-death and increment time controls
- **Infrastructure**: Magic bitboards, transposition table with depth-preferred aging, eval cache

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
position startpos
go wtime 60000 btime 60000 winc 1000 binc 1000
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| Hash | spin | 128 | TT size in MB |
| Contempt | spin | 0 | Draw avoidance |

## Testing

```bash
build/luminex.exe bench
# Expected: 20, 400, 8902, 197281
```

## Strength

~2600 ELO (pure HCE, no NNUE). Tested against Stash reference engines.

## License

Open source.
