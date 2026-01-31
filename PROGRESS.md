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

## Illegal Move History
- Previous fixes resolved most illegal moves
- **3.12.0**: Fixed Zobrist hash implementation - 0 illegal moves in 5 games!
- **3.13.0**: New illegal moves: a1e1 (castling bug), f1h1, h3h4, e7f6, h8a8
- Root cause: Move generation or board corruption during search

## Known Issues
1. **Move generation bugs** - Illegal moves like a1e1 suggest castling/encoding issues
2. **Very weak play** - Even without illegal moves, engine gets checkmated quickly
3. **Board corruption during search** - Validation catches many corrupt positions

## Next Steps
1. Fix move generation bugs (castling, move encoding)
2. Investigate why engine plays so weakly
3. Beat Fruit 2.1 in 20-game match

