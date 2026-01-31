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
| 3.8.0 | 0-20 (3 illegal moves) | Fixed MF_DOUBLE_PAWN detection | Double pawn push was not being flagged |
| 3.9.0 | 0-20 (3-7 illegal moves) | Added UCI command logging | Board state desync remains |
| 3.10.0 | 0-20 (6 illegal moves) | Added move flag auto-fix | Fixed capture flag mismatches |
| 3.11.0 | 0-20 (3 illegal moves) | Enhanced board validation | Reduced to 15% illegal moves |

## Illegal Move History
- e3e4, c3c4 - Fixed by capture detection bug fix
- e1c1, e1g1 (castling) - Fixed by castling rook presence check
- a3a4, a5b4, g3g4 - Fixed by capture detection
- g8e7, f8g7, c7b6 - Fixed by king_square reset
- a5b5, a6b5, e4e6, b7b6 - Fixed by MF_DOUBLE_PAWN detection (mostly)
- Remaining: a5b5 (horizontal), a2a3 (pawn), promotions - Board state desync (15% rate, down from 100%)

## Known Issues
1. **Board state desynchronization** (15% illegal moves, down from 100%):
   - Luminex's internal board differs from GUI's board after move application
   - Added comprehensive validation: move flag checking, board consistency, piece type verification
   - Moves like a5b5 (horizontal) appear as legal on Luminex's board but illegal on GUI's
   - Root cause: Complex interaction between move parsing, state tracking, and search

## Next Steps
1. Continue reducing illegal moves (target: 0%)
2. Optimize NPS with magic bitboards (50-100% boost target)
3. Beat Fruit 2.1 in 20-game match

