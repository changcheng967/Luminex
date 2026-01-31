# Luminex Chess Engine - Progress

## Version History
| Version | Score vs Fruit | What Changed | What I Learned |
|---------|----------------|--------------|----------------|
| 3.6.0 | 0-20 (3 illegal moves) | Initial | Multiple illegal moves |
| 3.6.1 | 0-20 (5 illegal moves) | Fixed capture detection | piece_type_on() doesn't check color |
| 3.6.2 | 0-20 (2 illegal moves) | Fixed castling rights | castling_rights_[] not being updated |
| 3.6.3 | 0-20 (1-2 illegal moves) | Various fixes | Board state corruption still exists |

## Illegal Move History
- e3e4, c3c4 - Fixed by capture detection bug fix
- e1c1, e1g1 (castling) - Fixed by castling rook presence check
- a3a4, a5b4, g3g4 - Fixed by capture detection
- h1a1, e2c4, c1d1 - Still investigating (board state corruption)

## Known Issues
1. **Board state corruption**: Moves like e2c4 (impossible pawn move) and h1a1 (rook across board) suggest pieces are being misplaced
2. **Castling rights**: Now correctly removed when king/rook moves
3. **Capture detection**: Now correctly identifies enemy pieces

## Next Steps
1. Investigate board state corruption during move application
2. Consider full architecture rewrite with world-class patterns
3. Implement magic bitboards for 50-100% NPS boost

