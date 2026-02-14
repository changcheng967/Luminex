# Luminex Progress Tracker

## Recent Commits (2026-02-14)
| Commit | Description | Status |
|--------|-------------|--------|
| 8beda5b | Enable aspiration windows with wide initial delta | Performance improvement |
| 001164e | Remove duplicate pawn shield evaluation | Bug fix |
| 1b4f816 | Remove duplicate knight outpost evaluation | Bug fix |
| 56671f3 | Add SEE-based capture ordering and logarithmic LMR | Performance improvement |
| b7c93be | Fix depth 8+ hang and CuteChess stalls | Critical fix |

## Current State (2026-02-14)
- **Depth 8+ Hang**: **FIXED** - Aspiration window loop bug corrected
- **CuteChess Stalls**: **FIXED** - Stdin polling with prefix match and newline guard
- **Illegal Moves**: 0% - All perft tests pass
- **Perft Validation**: All passing (20, 400, 8902, 197281)

## Performance Metrics (startpos depth 8)
- Nodes: ~2.1M
- Time: ~2 seconds
- NPS: ~1M

## Key Improvements Made
1. **Aspiration window fix** - Changed `<=` to `<` in fail-low check
2. **SEE-based capture ordering** - Distinguishes winning vs losing captures
3. **Logarithmic LMR** - Better reduction formula for late moves
4. **Duplicate evaluation removal** - Knight outpost and pawn shield were evaluated twice
5. **Proper aspiration windows** - Enabled with 200cp delta at depth 5+

## Remaining Improvements
1. **Continuation history** - Attempted but caused performance issues
2. **Reverse futility pruning** - Not yet implemented
3. **Singular extensions** - Currently disabled
4. **Evaluation simplification** - Still complex with many overlapping terms

## Perft Validation
- Perft 1: 20 PASS
- Perft 2: 400 PASS
- Perft 3: 8902 PASS
- Perft 4: 197281 PASS
