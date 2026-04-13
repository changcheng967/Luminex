# Luminex

A UCI chess engine written in C++23. All evaluation values are self-engineered from chess first principles.

## Download

Latest release: [github.com/changcheng967/Luminex/releases](https://github.com/changcheng967/Luminex/releases)

| Platform | Binary |
|----------|--------|
| Linux (AVX2) | `luminex-linux-x86-64-modern` |
| Linux (SSE4.2) | `luminex-linux-x86-64` |
| Windows (ClangCL) | `luminex-windows-x86-64-modern.exe` |
| Windows (MSVC) | `luminex-windows-x86-64.exe` |
| macOS (Apple Silicon) | `luminex-macos-arm64` |
| Web (WASM) | `luminex-wasm.tar.gz` |

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
| `Hash` | 128 | TT size in MB |
| `Threads` | 1 | Search threads (lazy SMP) |
| `Contempt` | 0 | Draw avoidance (centipawns) |

### Web

Extract `luminex-wasm.tar.gz` and serve `index.html` from any static host. Runs entirely in-browser.

## Build

Requires C++23 compiler (GCC 12+, Clang 15+, MSVC 2022+) and CMake 3.15+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## Architecture

```
src/
  luminex.h          # Core types and constants
  types.h            # Value types, enums
  bitboard.h         # Bitboard operations
  magic.cpp          # Magic bitboard generation
  board.h / cpp      # Position representation, make/unmake
  movegen.h / cpp    # Legal move generation
  evaluation.h / cpp # Hand-crafted evaluation
  search.h / cpp     # PVS search with LMR
  transposition.h / cpp # Transposition table
  uci.h / cpp        # UCI protocol
  texel_tune.cpp     # Texel tuning
  main.cpp           # Entry point
```

## License

[Luminex License](LICENSE) — non-commercial use
