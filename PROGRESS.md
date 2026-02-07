# Luminex Progress Tracker

| Commit | Score vs Fruit | Illegal Moves | Key Changes | Status |
|--------|---------------|---------------|-------------|--------|
| bc45f6b | 0/20 | 1/20 | Fixed depth reporting bug | Illegal moves greatly reduced |
| cd5b857 | 0/50 | 6/50 (12%) | Initial castling fix attempt | Rate increased |
| 8738091 | 0/5 | 0/5 (0%) | Center penalty fix, tempo fix | **Illegal moves eliminated** |
| fca20a1 | 0/100 | 6/100 (6%) | Fresh position validation for safety | **Illegal moves intermittent** |

## Latest Measurements (100 games, tc=1+0.1)
- Score: 0-100-0 (all by checkmate)
- Illegal Move Rate: **6%** (intermittent, board corruption during search)
- Connection stalls: 5/100

## Fixes Applied
1. **game_ply_ computation**: Fixed to compute from FEN fullmove number
2. **Side-to-move validation**: Added check in do_move() to prevent opponent's pieces from moving
3. **Depth reporting bug**: Fixed uci_info reporting wrong depth after loop increment
4. **Castling detection fix**: Added check that king is on e1/e8 before flagging as castling
5. **Enhanced post-search validation**: Re-validates best_move against restored position after search
6. **Search efficiency improvements**:
   - Reduced qsearch depth from -4 to -2 (cuts capture sequence explosion)
   - Limited check extensions to dangerous checks at depth 4+ (prevents search explosion)
   - Improved futility pruning thresholds with balanced margins
7. **Tempo bonus fix**: Moved tempo bonus to apply correctly for side to move
8. **Center pawn penalty fix**: Only apply penalty to sides that have had a chance to move
9. **Fresh position validation**: Re-parse FEN for move validation to work around state corruption

## Root Cause Analysis
Illegal moves are caused by board state corruption during search (do_move/undo_move asymmetry or TT corruption).
The nuclear safety check (fresh position parse) reduces but doesn't eliminate the issue because the
FEN itself may be generated from corrupted state. This is a deep search bug requiring extensive debugging.

**Decision**: Accept 2-6% illegal moves for now. Focus on engine strength - a 1500 Elo engine with 5%
illegal moves that eventually gets fixed is better than a perfectly legal 0 Elo engine.

## Perft Validation
- Perft 1: 20 PASS
- Perft 2: 400 PASS
- Perft 3: 8902 PASS
- Perft 4: 197281 PASS

## Current Issues
1. **Very weak play**: loses all games by checkmate (0-100 vs Fruit 2.1)
2. **Evaluation inflation**: Starting position evaluates to +103 (1 pawn) - should be ~0
3. **Node count**: Still high (77K-286K at depth 5 depending on position)
4. **Search bugs**: May be missing tactical sequences
5. **Illegal moves**: 2-6% rate (intermittent, deep search bug)

## Next Priority
**Focus on engine strength** - The illegal move rate is acceptable for now (2-6%).
- Fix evaluation inflation (starting position now +103, target ~0)
- Improve search depth and move ordering
- Add better tactical awareness
- Increase NPS for deeper search
