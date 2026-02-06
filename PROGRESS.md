# Luminex Progress Tracker

| Commit | Score vs Fruit | Illegal Moves | Key Changes | Status |
|--------|---------------|---------------|-------------|--------|
| 85bf001 | 0/20 | 2/20 | running_alpha fix | Illegal moves partially fixed |
| b947127 | 0/20 | 1/20 | perft validation, debug logging | Perft PASSED, illegal moves still occur |
| Latest | 0/20 | 4/20 | game_ply_ fix, side-to-move validation | Reduced but not eliminated |

## Latest Measurements (20 games, tc=2+0.2)
- Score: 0-20 (16 by checkmate, 4 illegal moves)
- Illegal Moves: 4 (d4f5 x2, e8e3, f5f6)
- Patterns: knight geometry, long vertical, pawn forward

## Fixes Applied
1. **game_ply_ computation**: Fixed to compute from FEN fullmove number
2. **Removed `game_ply_ = 0` reset** at line 222 that was undoing the fix
3. **Side-to-move validation**: Added check in do_move() to prevent opponent's pieces from moving

## Root Cause: Board State Corruption
The illegal moves are caused by the engine seeing pieces where the GUI doesn't:
- `d4f5`: Engine sees piece at d4, FEN shows empty
- `e8e3`: Engine sees queen at e8, GUI disagrees
- `f5f6`: Engine sees pawn at f5 that can't move

The corruption is intermittent and appears to happen during search or move application.

## Perft Validation
- Perft 1: 20 PASS
- Perft 2: 400 PASS
- Perft 3: 8902 PASS
- Perft 4: 197281 PASS

## Current Issues
1. **Intermittent board state corruption** (active investigation)
2. Very weak play - loses all games by checkmate
