# Luminex Chess Engine

A competitive chess engine written in modern C++23, built from scratch with magic bitboards, PVS search, and handcrafted evaluation.

## Current Version: v3.19.0

**Status**: Active development. Engine plays legal chess with 0 illegal moves. Estimated ~2200-2300 ELO.

## Recent Changes (v3.19.0)

- Fixed time management for sudden-death time controls (no more time losses)
- Improved PVS (Principal Variation Search) with proper zero-window scout and re-search
- Added LMR for PV nodes with reduced depth penalty
- Added mate distance pruning
- Improved aspiration window depth-start time check (60% threshold)
- Removed expensive counter-move history aging (uses gravity formula instead)

## Features

### Search
- Principal Variation Search (PVS) with zero-window scout
- Late Move Reduction (LMR) with history-based adjustments
- Null Move Pruning with endgame verification
- Razoring for low-depth pruning
- ProbCut for high-beta cutoffs
- Singular Extension for critical positions
- Mate distance pruning
- Aspiration Windows with iterative widening
- Internal Iterative Reduction (IIR)
- Phased move generation (TT move → captures → quiets)
- Quiescence search with SEE pruning and MVV-LVA ordering
- Check extensions

### Move Ordering
- TT move (highest priority)
- Captures ordered by MVV-LVA with SEE classification
- Killer moves (2 per ply)
- Counter-move history (1-ply)
- Continuation history (2-ply)
- Plain history heuristic with gravity formula
- Escape-aware ordering for pieces under pawn attack

### Evaluation (Handcrafted)
- PeSTO piece-square tables (MG/EG tapered)
- Pawn structure: doubled, isolated, passed pawns
- Passed pawn: blocker penalty, rook-behind bonus, king proximity
- Piece mobility (knight, bishop, rook, queen)
- Open/semi-open file bonuses for rooks
- Rook on 7th rank bonus
- Bishop pair bonus
- King safety with attack maps (non-linear danger scaling)
- King pawn shield evaluation
- Castling evaluation
- Tempo bonus

### Infrastructure
- Magic bitboard sliding piece attacks
- Transposition table with depth-preferred aging (32-bit key, 64-byte clusters)
- Eval cache (512K entries)
- Threaded search (separate search thread from UCI loop)
- Tournament time management with increment support

## Building

### Requirements
- C++23 compatible compiler (MSVC, Clang, GCC)
- CMake 3.15+
- Ninja (recommended)

### Compilation

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build --config Release
```

The executable will be at `build/luminex.exe` (Windows) or `build/luminex` (Linux/macOS).

### Testing

```bash
# Perft test (must pass before any commit)
build/luminex.exe bench
# Expected: 20, 400, 8902, 197281

# Match vs reference engine
cutechess-cli -rounds 20 -engine cmd=build/luminex.exe name=Luminex -engine cmd=stash.exe name=Stash -each proto=uci tc=1+0.01
```

## Usage

### UCI Protocol

```
uci
setoption name Hash value 128
setoption name Contempt value 0
position startpos
go wtime 60000 btime 60000 winc 1000 binc 1000
quit
```

### Options

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| Hash | spin | 128 | 1-1048576 | Transposition table size in MB |
| Contempt | spin | 0 | -1000 to 1000 | Draw avoidance tendency |
| Clear Hash | button | - | - | Clear the transposition table |

## Architecture

```
src/
├── main.cpp          - Entry point
├── uci.cpp           - UCI protocol, stdin/stdout, threading
├── search.cpp        - PVS, LMR, null move, iterative deepening
├── evaluation.cpp    - Handcrafted eval (PST, pawns, mobility, king safety)
├── board.cpp         - Position, do_move/undo_move, legal(), SEE
├── movegen.cpp       - Move generation (legal, capture, quiet)
├── transposition.cpp - TT with depth-preferred replacement
├── bitboard.cpp      - Magic bitboard initialization
└── *.h               - Headers
```

## Key Files

- `src/uci.cpp` — UCI loop, stdin polling, handle_position, handle_go
- `src/board.cpp` — Position, do_move/undo_move, legal(), attackers_to()
- `src/search.cpp` — Iterative deepening, alpha-beta, qsearch
- `src/evaluation.cpp` — Static eval (PST, pawn structure, king safety)
- `src/bitboard.h` — Bitboard utils, attack functions, magic bitboards

## Strength Estimate

Based on testing against Stash (known CCRL-rated engines):

| Opponent | Opponent ELO | Result | Est. Luminex ELO |
|----------|-------------|--------|------------------|
| Stash v17 | ~2298 | 10-9-1 (52.5%) | ~2300 |
| Stash v10 | ~1620 | 17-3 (85%) | ~1900-2000 |

Estimated strength: **~2200-2300 blitz ELO** (as of v3.19.0)

## Roadmap

- [ ] Tune LMR formula constants (+20-50 ELO potential)
- [ ] Add capture LMR (+30-50 ELO potential)
- [ ] Improve null move reduction formula (+20-40 ELO potential)
- [ ] Add pawn storm evaluation (+30-50 ELO potential)
- [ ] Add threat evaluation (+20-40 ELO potential)
- [ ] Add connected rooks bonus
- [ ] Add followup heuristic for move ordering
- [ ] Multi-threading (Lazy SMP)
- [ ] NNUE evaluation (long-term goal)

## Engine Info

- **Author**: changcheng967
- **Language**: C++23
- **Platform**: Windows (Linux/macOS compatible)
- **Protocol**: UCI

## License

Open source. Available for educational and competitive use.

## Acknowledgments

Inspired by Stockfish, Ethereal, and the chess programming community.
Uses techniques from the Chess Programming Wiki (chessprogramming.org) and PeSTO evaluation tables.
