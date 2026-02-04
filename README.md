# Luminex Chess Engine

A world-class classical chess engine written in modern C++23.

## Current Status (v3.18.0)

**Active Development**: Fixing illegal move generation issues. Engine currently at ~85% legal move rate in testing.

**Known Issues**:
- Position state drift after multiple moves
- Working on comprehensive position validation and restoration

## Features

### Search
- Late Move Reduction (LMR) with history-based adjustments
- Null Move Pruning with zugzwang detection
- Razoring for low-depth pruning
- ProbCut for high-beta cutoffs
- Singular Extension for critical positions
- Aspiration Windows for efficient PV search
- Internal Iterative Deepening
- Tournament time management (wtime/btime with increments)

### Evaluation
- Piece-Square Tables (separate middle-game/end-game)
- Comprehensive mobility evaluation
- Pawn structure analysis (passed, doubled, isolated, backward pawns)
- King safety with pawn shields
- Connected rooks bonus
- Knight outpost bonus
- Bishop pair bonus
- Bad bishop penalty
- Piece coordination evaluation
- Threat detection (hanging pieces)
- Endgame king evaluation (centralization, opposition)

### Infrastructure
- Transposition table with depth-based replacement
- Killer moves (2 per ply)
- History heuristic
- Counter-move history
- Quiescence search with Static Exchange Evaluation (SEE)
- Contempt option for draw avoidance

## Building

### Requirements
- C++23 compatible compiler (Clang, GCC, MSVC)
- CMake 3.15+
- Ninja (recommended) or other build tools

### Compilation

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build --config Release
```

The executable will be at `build/luminex.exe` (Windows) or `build/luminex` (Linux/macOS).

## Usage

### UCI Protocol

Luminex supports the Universal Chess Interface (UCI) protocol:

```
uci
setoption name Contempt value 100
setoption name Hash value 128
position startpos
go depth 10
position startpos moves e2e4 e7e5
go wtime 60000 btime 60000
quit
```

### Options

- `Contempt` (0-1000): Draw avoidance. Positive values play for wins.
- `Hash` (1-1048576): Transposition table size in MB
- `Clear Hash`: Clear the transposition table

## Engine Info

- **Version**: 3.18.0
- **Author**: changcheng967
- **Language**: C++23
- **Code Size**: ~7000 LOC
- **Binary Size**: ~280KB

## Performance

Optimized for modern x86-64 processors with AVX-512 support:
- Bitboard-based representation
- Template metaprogramming for compile-time optimization
- Efficient move generation and legal move verification

## License

This project is open source and available for educational and competitive use.

## Contributing

Contributions are welcome! The codebase is designed to be clean and well-documented for easy modification.

## Acknowledgments

Inspired by world-class chess engines like Stockfish, Ethereal, and others.
Uses techniques and concepts from the Chess Programming Wiki (chessprogramming.org).
