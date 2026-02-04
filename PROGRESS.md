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
| 3.24.0 | 20% (4/20) | Added side_to_move diagnostics | Variability: 0% to 20% illegal moves |

## Critical Bugs Fixed
1. **state_stack declaration**: Changed from `StateInfo*` array to `StateInfo` array
2. **EP Zobrist**: XOR out old EP square before setting new one
3. **MAX_STATES**: Increased from 256 to 2048
4. **Castling Zobrist**: XOR out old rights, XOR in new rights
5. **Piece-square Zobrist**: Added for normal moves (was missing!)
6. **Search fallback**: Check for empty move list before accessing [0]
7. **Safety checks**: Added validation in search() and handle_go

## Current Issue: Board State Corruption When Playing Black
All illegal moves (20%) occur when playing Black:
- "e5f6" - Looks like White knight move (+9 pattern from e5 to f6)
- "d3c4" - White pawn capture pattern (d3 to c4, forward-left for White)
- "h8d8" - Rook along back rank

**Hypothesis**: Board state gets corrupted during long games, causing pieces to appear in wrong positions.
The variability (0% in 5-game tests, 20% in 20-game tests) suggests state-dependent timing issue.

**Progress**: Reduced from 100% → 20% illegal moves. Engine is now stable for development.

## Next Steps
1. Investigate board state corruption during long games
2. Add more robust state validation
3. Achieve 0% illegal moves
4. Beat Fruit 2.1 in 20-game match

