# Luminex Chess Engine - Progress

## Version History
| Version | Illegal Move Rate | Score vs Fruit | What Changed | What I Learned |
|---------|------------------|----------------|--------------|----------------|
| 3.6.0 | 15% (3/20) | Initial | Multiple illegal moves |
| 3.6.1 | 25% (5/20) | Fixed capture detection | piece_type_on() doesn't check color |
| 3.6.2 | 10% (2/20) | Fixed castling rights | castling_rights_[] not being updated |
| 3.6.3 | 5-10% (1-2/20) | Various fixes | Board state corruption still exists |
| 3.6.4 | 35% (7/20) | Added state stack bounds check | Reduced illegal moves with logging |
| 3.6.5 | 20% (4/20) | Fixed king_square initialization | king_square now reset in set() |
| 3.7.0 | 20% (4/20) | Added corruption logging | Identified board desync issue |
| 3.8.0 | 15% (3/20) | Fixed MF_DOUBLE_PAWN detection | Double pawn push was not being flagged |
| 3.9.0 | 15-35% (3-7/20) | Added UCI command logging | Board state desync remains |
| 3.10.0 | 30% (6/20) | Added move flag auto-fix | Fixed capture flag mismatches |
| 3.11.0 | 15% (3/20) | Enhanced board validation | Reduced to 15% illegal moves |
| 3.12.0 | 0% (0/5) | Fixed Zobrist implementation | No illegal moves! But plays very weakly |
| 3.13.0 | 25% (5/20) | Fixed aggressive pruning, LMR | Regressed! New illegal moves appeared |
| 3.13.1 | 20% (4/20) | Fixed castling hash update, TT validation | Castling bugs eliminated! New: sliding piece corruption |
| 3.18.0 | 100% (20/20) | Used legal move generation for position replay | REGRESSION! Now returning opponent's moves |

## Root Cause Identified (v3.18.0)
- **Search returns opponent's moves**: Patterns like c8d7 (White trying to move Black's bishop)
- **Position corruption during search**: Castling rights KQkq → - after search
- **do_move/undo_move bug**: Position not properly restored after search completes

## Illegal Move Patterns
- **Opponent's pieces**: c8d7 (White→Black bishop), c1d2 (Black→White bishop), f3d2 (Black→White knight)
- **100% of moves are illegal** when using legal move generation for replay
- This indicates the search itself is broken, not just move validation

## Known Issues
1. **Search corruption** - do_move/undo_move doesn't restore position correctly
2. **Castling rights lost** - After search, castling rights go from KQkq to -
3. **Opponent's moves returned** - Search generates moves for wrong color
4. **Very weak play** - Gets checkmated quickly by Fruit

## Next Steps
1. Fix do_move/undo_move to properly restore castling rights
2. Verify search returns moves for correct side_to_move
3. Achieve 0% illegal moves
4. Beat Fruit 2.1 in 20-game match

