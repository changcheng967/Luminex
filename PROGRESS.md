# Luminex Progress Tracker

## Recent Commits (2026-02-14)
| Commit | Description | Status |
|--------|-------------|--------|
| e40d880 | Add eval mode and update progress | Utility |
| 1ca7d26 | Improve check and evasion extensions | Performance |
| c6c1595 | Improve evaluation and search parameters | Performance |
| e3a95dc | Fix time management and reduce stall rate | Critical fix |

## Current State (2026-02-14)
- **Connection Stalls**: **FIXED** - 0% stall rate (down from 40-60%)
- **Illegal Moves**: 0% - All perft tests pass
- **Perft Validation**: All passing (20, 400, 8902, 197281)
- **Fruit Match Score**: ~1-19 (engine won 1 game in testing)

## Performance Metrics (startpos depth 8)
- Nodes: ~1.2M
- Time: ~2 seconds
- NPS: ~660K

## Key Improvements Made
1. **Time check frequency** - Changed from every 256 to every 64 nodes
2. **Qsearch depth** - Extended from -1 to -4 to reduce horizon effect
3. **Mobility weights** - Increased significantly for all pieces
4. **Piece values** - Knight 320→330, Queen 900→1000
5. **Pruning** - Reduced LMP and razoring aggression
6. **Extensions** - Improved check/evasion extensions (depth 5→3 for evasions)
7. **Node safety limit** - Added 50M node cap to prevent runaway searches
8. **History/Killer tables** - Cleared between searches

## Remaining Issues
1. **Strength gap** - Still losing most games to Fruit (~2300 ELO)
2. Need deeper search optimization
3. Need better move ordering (continuation history)

## Next Steps
1. Implement continuation history for better move ordering
2. Add reverse futility pruning
3. Improve king safety attack weighting
4. Consider evaluation simplification

## Perft Validation
- Perft 1: 20 PASS
- Perft 2: 400 PASS
- Perft 3: 8902 PASS
- Perft 4: 197281 PASS
