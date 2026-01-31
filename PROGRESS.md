# Luminex Chess Engine - Progress

## Version History
| Version | Score vs Fruit | What Changed | What I Learned |
|---------|----------------|--------------|----------------|
| 3.6.0 | 0-20 (3 illegal moves) | Initial | Multiple illegal moves |
| 3.6.1 | 0-20 (5 illegal moves) | Fixed capture detection | piece_type_on() doesn't check color |
| 3.6.2 | 0-20 (2 illegal moves) | Fixed castling rights | castling_rights_[] not being updated |
| 3.6.3 | 0-20 (1-2 illegal moves) | Various fixes | Board state corruption still exists |
| 3.6.4 | 0-20 (7 illegal moves) | Added state stack bounds check | Reduced illegal moves with logging |
| 3.6.5 | 0-20 (4 illegal moves) | Fixed king_square initialization | king_square now reset in set() |
| 3.7.0 | 0-20 (4 illegal moves) | Added corruption logging | Identified board desync issue |

## Illegal Move History
- e3e4, c3c4 - Fixed by capture detection bug fix
- e1c1, e1g1 (castling) - Fixed by castling rook presence check
- a3a4, a5b4, g3g4 - Fixed by capture detection
- g8e7, f8g7, c7b6 - Fixed by king_square reset
- a5b5, a6b5, e4e6 - Still investigating (board desync between GUI and engine)

## Known Issues
1. **Board state desynchronization**: Luminex's internal board differs from GUI's board after move application
2. **king_square corruption**: Fixed by initializing to SQUARE_NONE in set()
3. **State stack overflow**: Added bounds check to prevent buffer overflow
4. **Remaining illegal moves**: Down to 4 per 20 games (20% from original 100%)

## Next Steps
1. Investigate move flag detection in handle_position()
2. Add validation to compare board state after UCI move application
3. Consider full architecture rewrite with world-class patterns
4. Implement magic bitboards for 50-100% NPS boost

