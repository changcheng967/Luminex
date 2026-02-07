# Luminex Progress Tracker

| Commit | Score vs Fruit | Illegal Moves | Key Changes | Status |
|--------|---------------|---------------|-------------|--------|
| bc45f6b | 0/20 | 1/20 | Fixed depth reporting bug | Illegal moves greatly reduced |
| cd5b857 | 0/50 | 6/50 (12%) | Initial castling fix attempt | Rate increased |
| 8738091 | 0/5 | 0/5 (0%) | Center penalty fix, tempo fix | **Illegal moves eliminated** |

## Latest Measurements (5 games, tc=1+0.1)
- Score: 0-5 (all by checkmate)
- Illegal Move Rate: **0%**
- Starting eval: +103 (improved from +203)

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
8. **Center pawn penalty fix**: Only apply penalty to sides that have had a chance to move (don't penalize black at ply 1)

## Root Cause: Board State Corruption During Search
The illegal moves were caused by the search modifying the position in ways that aren't fully undone. When the position is restored via FEN after search, the best_move may reference a board state that no longer exists. Post-search validation catches most of these.

## Perft Validation
- Perft 1: 20 PASS
- Perft 2: 400 PASS
- Perft 3: 8902 PASS
- Perft 4: 197281 PASS

## Current Issues
1. **Very weak play**: loses all games by checkmate (0-5 vs Fruit 2.1)
2. **Evaluation inflation**: Starting position evaluates to +103 (1 pawn) - should be ~0
3. **Node count**: Still high (77K-286K at depth 5 depending on position)
4. **Search bugs**: May be missing tactical sequences

## Next Priority
**Focus on engine strength** - Illegal moves eliminated (0%). Engine loses all games by checkmate.
- Continue fixing evaluation inflation (starting position now +103, target ~0)
- Improve search depth and move ordering
- Add better tactical awareness
