# Luminex Progress Tracker

| Commit | Score vs Fruit | Illegal Moves | Key Changes | Status |
|--------|---------------|---------------|-------------|--------|
| 85bf001 | 0/20 | 2/20 | running_alpha fix | Illegal moves partially fixed |
| b947127 | 0/20 | 1/20 | perft validation, debug logging | Perft PASSED, illegal moves still occur |
| LATEST | 0/20 | 0/20 | game_ply_ fix in Position::set() | **ILLEGAL MOVES ELIMINATED** |

## Latest Measurements (20 games, tc=1+0.1)
- Score: 0-20 (19 by checkmate, 1 timeout)
- Illegal Moves: 0 ✅
- All losses: By checkmate (no illegal moves!)

## Perft Validation
- Perft 1: 20 PASS
- Perft 2: 400 PASS
- Perft 3: 8902 PASS
- Perft 4: 197281 PASS

## Root Cause Fixed
**game_ply_ computation bug**: Position::set() was resetting game_ply_ to 0 instead of computing from FEN fullmove number. This caused state corruption between engine and GUI, leading to illegal moves.
- Fix: `game_ply_ = (fullmove - 1) * 2 + (side_to_move_ == BLACK ? 1 : 0)`

## Current Issues
1. ~~Intermittent illegal moves~~ FIXED
2. ~~FEN fullmove number computed incorrectly~~ FIXED
3. Very weak play - loses all games by checkmate (NEXT PRIORITY)
