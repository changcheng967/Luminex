# Luminex Chess Engine - Progress

## Version History
| Version | Illegal Move Rate | Score vs Fruit | What Changed | What I Learned |
|---------|------------------|----------------|--------------|----------------|
| 3.18.0 | 100% (20/20) | Major state_stack bug | Illegal moves: opponent's pieces |
| 3.19.0 | 60% (12/20) | Fixed EP Zobrist, MAX_STATES | Illegal moves reduced |
| 3.20.0 | 20% (4/20) | Fixed castling Zobrist, search fallback | Still seeing wrong-side moves |
| 3.21.0 | 10% (2/20) | Added piece-square Zobrist for normal moves | Missing critical Zobrist update! |
| 3.22.0 | 40% (8/20) | Added TT clear, validation | Pollution, wrong-side moves persist |
| 3.23.0 | 25% (5/20) | Added final safety check in search() | All illegal moves when Black |

## Critical Bugs Fixed
1. **state_stack declaration**: Changed from `StateInfo*` array to `StateInfo` array
2. **EP Zobrist**: XOR out old EP square before setting new one
3. **MAX_STATES**: Increased from 256 to 2048
4. **Castling Zobrist**: XOR out old rights, XOR in new rights
5. **Piece-square Zobrist**: Added for normal moves (was missing!)
6. **Search fallback**: Check for empty move list before accessing [0]

## Current Issue: Wrong-Side Moves
All illegal moves occur when playing Black:
- "e5f6" - Looks like White knight move (+9 pattern)
- "h8d8" - Rook along back rank (wrong for Black position)
- "d3c4" - White pawn capture pattern

**Hypothesis**: The engine is generating White moves when it should play Black.
This suggests `side_to_move_` gets confused during search or position replay.

## Next Steps
1. Debug `side_to_move_` confusion when playing Black
2. Verify position replay correctly flips side
3. Achieve 0% illegal moves
4. Beat Fruit 2.1 in 20-game match

