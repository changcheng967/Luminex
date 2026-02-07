# Luminex Progress Tracker

| Commit | Score vs Fruit | Illegal Moves | Key Changes | Status |
|--------|---------------|---------------|-------------|--------|
| 26bc63b | 0/20 | 4/20 | game_ply fix, side-to-move validation | Reduced but not eliminated |
| bc45f6b | 0/20 | 1/20 | Fixed depth reporting bug | Illegal moves greatly reduced |
| Latest | 0/20 | 1/20 | - | Investigating ghost piece at f8 |

## Latest Measurements (20 games, tc=1+0.1)
- Score: 0-20 (19 by checkmate, 1 illegal move)
- Illegal Move: f8g8 (ghost piece at empty f8)
- Pattern: Engine sees piece at empty square

## Fixes Applied
1. **game_ply_ computation**: Fixed to compute from FEN fullmove number
2. **Side-to-move validation**: Added check in do_move() to prevent opponent's pieces from moving
3. **Depth reporting bug**: Fixed uci_info reporting wrong depth after loop increment

## Current Illegal Move Analysis: f8g8
Position: `q4k1r/p4p1p/1p4pB/2pnPN2/4P3/2P5/P1P2PPP/R2Q1RK1 b - - 1 15`

Last rank `q4k1r` means:
- a8: queen (q)
- b8-f8: 5 empty squares
- g8: king (k)
- h8: rook (r)

**Bug**: Engine tries to move from f8 (empty) to g8 (king's square)
**Expected**: f8 is empty, no piece should move from there
**Hypothesis**: Board corruption during search - engine piece array shows piece at f8 when there isn't one

## Perft Validation
- Perft 1: 20 PASS
- Perft 2: 400 PASS
- Perft 3: 8902 PASS
- Perft 4: 197281 PASS

## Current Issues
1. **Ghost piece bug**: Engine sees piece at empty f8 square (active investigation)
2. Very weak play - loses all games by checkmate
3. High node count - 36x more nodes than Fruit at same depth
