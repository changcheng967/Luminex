# Luminex

A UCI chess engine written in C++23. All evaluation values are self-engineered from chess first principles.

**Current version: v5.10.0** — [Download](https://github.com/changcheng967/Luminex/releases/latest)

## Version History

| Version | Key Feature | Est. Elo |
|---------|-------------|----------|
| v5.4.0 | Baseline HCE | ~2067 |
| v5.5.0 | Tactical qsearch eval | ~2173 |
| v5.6.0 | Correction history | ~2200 |
| v5.7.0 | Aspiration + centralization ordering | ~2258 |
| v5.8.0 | LMR positional awareness | ~2300 |
| v5.9.0 | Qsearch check generation + History-Adaptive LMP | ~2350 |
| v5.10.0 | Capture history TT + bad capture ordering | ~2390 |

## Download

Latest release: [github.com/changcheng967/Luminex/releases](https://github.com/changcheng967/Luminex/releases)

| Platform | Binary |
|----------|--------|
| Linux (AVX2) | `luminex-linux-x86-64-modern` |
| Linux (SSE4.2) | `luminex-linux-x86-64` |
| Windows (ClangCL) | `luminex-windows-x86-64-modern.exe` |
| Windows (MSVC) | `luminex-windows-x86-64.exe` |
| macOS (Apple Silicon) | `luminex-macos-arm64` |

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

## Build

Requires C++23 compiler (GCC 12+, Clang 15+, MSVC 2022+) and CMake 3.20+.

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
  search.h / cpp     # PVS search with LMR, phased move generation
  transposition.h / cpp # Transposition table
  uci.h / cpp        # UCI protocol
  main.cpp           # Entry point
```

## License

[Luminex License](LICENSE) — non-commercial use
