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
| 3.12.0 | 0-5 (0 illegal moves) | Fixed Zobrist implementation | No illegal moves! But plays very weakly |
| 3.13.0 | 0-20 (5 illegal moves) | Fixed aggressive pruning, LMR | Regressed! New illegal moves appeared |
| 3.13.1 | 0-20 (4 illegal moves) | Fixed castling hash update, TT validation | Castling bugs eliminated! New: sliding piece corruption |

## Illegal Move History
- **3.13.0**: Castling bugs (a1e1, h8a8, f1h1)
- **3.13.1**: Castling bugs FIXED! New issues: sliding pieces (a8a1, d6h2)
  - Castling fix: Zobrist hash update for rook position
  - Castling fix: Rook verification before generating castling
  - Castling fix: Pseudo-legal validation ensures correct from-square
  - New bug: Sliding pieces moving through blockers (bitboard corruption)

## Known Issues
1. **Sliding piece corruption** - a8a1, d6h2 suggest occupied bitboard is wrong
2. **Board state desync** - Internal board doesn't match GUI state
3. **Very weak play** - Gets checkmated quickly by Fruit

## Next Steps
1. Fix bitboard corruption in sliding piece attacks
2. Investigate do_move/undo_move for state corruption
3. Beat Fruit 2.1 in 20-game match

