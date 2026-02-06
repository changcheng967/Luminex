# Luminex Progress Tracker

| Commit | Score vs Fruit | Illegal Moves | Key Changes | Status |
|--------|---------------|---------------|-------------|--------|
| 85bf001 | 0/20 | 2/20 | running_alpha fix | Illegal moves partially fixed |
| b947127 | 0/20 | 1/20 | perft validation, debug logging | Perft PASSED, illegal moves still occur |

## Measurements (20 games, tc=1+0.1)
- Score: 0-20
- Illegal Moves: 1 (intermittent)
- All losses: By checkmate

## Perft Validation
- Perft 1: 20 PASS
- Perft 2: 400 PASS
- Perft 3: 8902 PASS
- Perft 4: 197281 PASS

## Known Issues
1. Intermittent illegal moves (knight/rook geometry)
2. FEN fullmove number computed incorrectly
3. Very weak play - loses all games by checkmate
