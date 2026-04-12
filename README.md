# Luminex

A portable UCI chess engine written in C++23.

All evaluation values are self-engineered from chess first principles. No values borrowed from other engines.

## Downloads

Pre-built binaries available for Linux, Windows, macOS, and Web (WASM):

| Platform | File | Notes |
|----------|------|-------|
| Linux x86-64 (modern) | `luminex-linux-x86-64-modern` | AVX2 + BMI2 (CPUs since ~2015) |
| Linux x86-64 | `luminex-linux-x86-64` | SSE4.2 + POPCNT (older CPUs) |
| Windows x86-64 (modern) | `luminex-windows-x86-64-modern.exe` | ClangCL optimized |
| Windows x86-64 | `luminex-windows-x86-64.exe` | MSVC compatible |
| macOS ARM64 | `luminex-macos-arm64` | Apple Silicon |
| Web (WASM) | `luminex-wasm.tar.gz` | Browser playable (JS + WASM) |

See [Releases](https://github.com/changcheng967/Luminex/releases) for downloads.

## Quick Start

```bash
./luminex
uci
isready
position startpos
go movetime 1000
```

### Web

Extract `luminex-wasm.tar.gz` and serve `index.html` from any static host. The engine runs entirely in-browser via WebAssembly.

## Build from Source

Requirements: C++23 compiler (GCC 12+, Clang 15+, MSVC 2022+) and CMake 3.15+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### Platform-specific

**Linux (optimized for current CPU):**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -mtune=native"
cmake --build build
```

**Windows (MSVC):**
```cmd
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

**WASM (requires [Emscripten](https://emscripten.org/)):**
```bash
emcmake cmake -B build-wasm -DCMAKE_BUILD_TYPE=Release -DWASM_BUILD=ON
cmake --build build-wasm
```

## UCI Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| Hash | spin | 128 | Transposition table size in MB |
| Threads | spin | 1 | Search threads (lazy SMP) |
| Contempt | spin | 0 | Draw avoidance in centipawns |

Plus 24 SPSA-tunable evaluation parameters (e.g. `BishopPairMG`, `RookOpenEG`).

## Features

**Search**
- PVS with aspiration windows
- LMR (log-depth x log-moves product formula)
- Null move pruning with verification search
- Singular extensions (single, double, negative)
- ProbCut, razoring, futility/reverse futility pruning
- Phased move generation (TT -> captures -> quiets)
- Late move pruning, continuation pruning
- Qsearch with SEE-based capture pruning

**Evaluation**
- MG/EG piece values with interpolation
- Piece-square tables from center-distance theory
- Linear mobility formulas calibrated per piece type
- Sigmoid king safety (Hill equation attack model)
- Value-ratio threat evaluation
- Pawn structure: doubled, isolated, backward, connected, phalanx, lever, candidate
- Passed pawns: rook behind, king proximity, free passer, blocked penalty
- Bishop pair, bad bishop (same-color pawns), outpost pieces
- Rook: open/semi-open file, 7th rank, trapped detection, x-ray
- Pawn shield, pawn storm, castling evaluation
- Space evaluation, material imbalance
- KXK endgame detection (KBN corner drive)
- OCB scaling, king proximity endgame scaling

**Move Ordering**
- TT move -> captures (MVV-LVA + capture history) -> quiets
- Killer moves, counter-move history, 2-ply continuation history
- Low-ply history (plies 1-3), escape-aware ordering
- TT cutoff stat updates, ttPv LMR reduction

**Infrastructure**
- Magic bitboards
- Transposition table with depth-preferred aging and generation counter
- Evaluation cache (512K entries)
- Pawn hash table (16K entries)
- Correction history (pawn-key based, depth^2 weighted)
- Lazy SMP multi-threading
- TT prefetch

**Time Management**
- Sudden-death and increment time controls
- Best-move stability allocation
- Score-based soft/hard time limits

## Strength

Tested via cutechess-cli at `tc=1+0.01` (bullet) against Stash:

| Version | Opponent | W-D-L | Score | Est. Elo |
|---------|----------|-------|-------|----------|
| v5.4.0 | Stash v15 (2140) | 149-19-232 | 39.6% | ~2067 |
| v5.4.0 | Stash v14 (2060) | 195-13-192 | 50.4% | ~2063 |
| v5.3.0 | Stash v14 (2060) | 106-5-89 | 54.3% | ~2092 |
| v5.2.0 | Stash v14 (2060) | 90-9-101 | 47.2% | ~2041 |
| v5.1.0 | Stash v13 (1972) | 179-14-127 | 55.2% | ~2086 |
| v5.0.0 | Stash v13 (1972) | 170-13-137 | 52.9% | ~2058 |

### Version History

| Version | Key Change |
|---------|-----------|
| v5.4.0 | Trapped knight detection, tested vs Stash v15 |
| v5.3.0 | Improving flag, recapture extension refinement |
| v5.2.0 | Search/board bug fixes, LMR product formula, best-move stability TM |
| v5.1.0 | Sigmoid king safety, value-ratio threats |
| v5.0.0 | Evaluation rewritten from first principles |
| v4.5.0 | Correction history baseline |

## Testing

```bash
./luminex bench
```

Perft values: `20 400 8902 197281`

### Match Testing

```bash
cutechess-cli \
  -each proto=uci tc=1+0.01 \
  -engine name=Luminex cmd=./luminex \
  -engine name=Opponent cmd=./opponent \
  -rounds 400 -repeat \
  -resign movecount=3 score=400 \
  -draw movenumber=40 movecount=8 score=10
```

## Architecture

```
src/
  luminex.h        # Core types, constants, board representation
  bitboard.h/cpp   # Magic bitboards, attack generation
  position.h/cpp   # Position class, make/unmake move
  movegen.h/cpp    # Legal and pseudo-legal move generation
  evaluation.h/cpp # Hand-crafted evaluation (all self-engineered)
  search.h/cpp     # PVS search with LMR, TT, heuristics
  uci.h/cpp        # UCI protocol handler
  tt.h/cpp         # Transposition table
  tune.h/cpp       # SPSA tuning parameters
```

## License

Open source.
