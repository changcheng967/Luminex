# Luminex Chess Engine

A competitive chess engine written in modern C++23, built from scratch with magic bitboards, PVS search, and a "third-generation" evaluation approach featuring dynamic correction history.

## Current Version: v3.22.0

**Status**: Active development. Engine plays legal chess with 0 illegal moves. Estimated **~2600 ELO** (pure HCE, no NNUE).

## Recent Changes

### v3.22.0 — Major Evaluation & Search Overhaul (+300 ELO)
- **Correction History**: Dynamic eval that learns from search results (position-specific corrections)
- **Connected pawn bonus**: Rewards mutually-protecting pawn chains
- **Outpost knight bonus**: Protected knights on rank 4-6 that enemy pawns can't challenge
- **Bad bishop penalty**: Bishops hemmed in by own pawns on same color complex
- **Pawn storm evaluation**: Enemy pawns advancing toward castled king
- **Threat evaluation**: Hanging pieces, minor pieces threatened by enemy pawns
- **Connected rooks bonus**: Rooks connected on same file/rank
- **Endgame scale factors**: OCB draw detection, no-pawn endgames, material imbalance scaling
- **Recapture extension**: Extend when recapturing on the same square
- **SEE quiet move pruning**: Prune bad quiet moves at shallow depth
- **Counter-move LMR bonus**: Reduce less for counter-move suggestions
- **TT move LMR bonus**: Reduce less for transposition table move
- **Deeper/shallower LMR re-search**: Stockfish-inspired depth adjustment after LMR
- **Improved futility margins**: Better depth-dependent pruning thresholds
- **Capture LMR**: Losing captures get reduced search depth

### v3.19.0
- Fixed time management for sudden-death time controls (no more time losses)
- Improved PVS with proper zero-window scout and re-search
- Added LMR for PV nodes, mate distance pruning
- Fixed CI: all 4 platforms (Ubuntu GCC/Clang, Windows MSVC, macOS) pass

## Features

### Search
- Principal Variation Search (PVS) with zero-window scout
- Late Move Reduction (LMR) with history, killer, counter-move, and TT adjustments
- Deeper/shallower LMR re-search (Stockfish-inspired)
- Null Move Pruning with endgame verification
- Razoring for low-depth pruning
- ProbCut for high-beta cutoffs
- Singular Extension for critical positions
- Recapture extension
- Check extensions
- Mate distance pruning
- Aspiration Windows with iterative widening
- Internal Iterative Reduction (IIR)
- Phased move generation (TT move -> captures -> quiets)
- Quiescence search with SEE pruning and MVV-LVA ordering

### Move Ordering
- TT move (highest priority)
- Captures ordered by MVV-LVA with SEE classification
- Killer moves (2 per ply)
- Counter-move table (direct move suggestion)
- Counter-move history (1-ply)
- Continuation history (2-ply)
- Plain history heuristic with gravity formula
- Escape-aware ordering for pieces under pawn attack

### Evaluation ("Third Generation" HCE)
- **Correction History**: Dynamic eval adjustments learned during search
- PeSTO piece-square tables (MG/EG tapered)
- Pawn structure: doubled, isolated, backward, connected, passed pawns
- Passed pawn: blocker penalty, rook-behind bonus, king proximity
- Outpost knight bonus (protected, no enemy pawn can challenge)
- Bad bishop penalty (hemmed in by own pawns)
- Piece mobility (knight, bishop, rook, queen)
- Open/semi-open file bonuses for rooks
- Rook on 7th rank bonus
- Connected rooks bonus
- Bishop pair bonus
- King safety with attack maps (non-linear danger scaling)
- King pawn shield evaluation
- Pawn storm detection (enemy pawns advancing toward castled king)
- Castling evaluation
- Threat evaluation (hanging pieces, minor pieces attacked by pawns)
- Endgame scale factors (OCB draws, no-pawn endgames, material imbalance)
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

### Testing

```bash
# Perft test (must pass before any commit)
build/luminex.exe bench
# Expected: 20, 400, 8902, 197281

# Match vs reference engine
cutechess-cli -rounds 40 -engine cmd=build/luminex.exe name=Luminex -engine cmd=stash.exe name=Stash -each proto=uci tc=1+0.01
```

## Strength Estimate

Based on testing against Stash engines (known CCRL-rated):

| Opponent | Result | Score |
|----------|--------|-------|
| Stash v20 (~2550) | 29-11 | +168 ELO |
| Stash v19 (~2500) | 28-12 | +147 ELO |
| Stash v17 (~2300) | 19-19 | Even |

Estimated strength: **~2600 blitz ELO** (pure HCE, no NNUE)

## Roadmap

- [x] Tune LMR formula with history/killer/counter-move adjustments
- [x] Add capture LMR for losing captures
- [x] Add pawn storm evaluation
- [x] Add threat evaluation (hanging pieces, pawn threats)
- [x] Add connected rooks bonus
- [x] Add correction history (dynamic eval learning)
- [ ] Multi-threading (Lazy SMP)
- [ ] Position-type classification for adaptive search policy
- [ ] Search-eval fusion (eval tells search what to extend/reduce)
- [ ] "Third generation" evaluation beyond HCE and NNUE

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
