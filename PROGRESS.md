# Luminex Progress Tracker

## Recent Commits (2026-02-14)
| Commit | Description | Status |
|--------|-------------|--------|
| c6c1595 | Improve evaluation and search parameters | Performance improvement |
| e3a95dc | Fix time management and reduce stall rate | Critical fix |
| 8beda5b | Enable aspiration windows with wide initial delta | Performance improvement |

## Current State (2026-02-14)
- **Connection Stalls**: **FIXED** - Reduced from 40% to 0%
- **Illegal Moves**: 0% - All perft tests pass
- **Perft Validation**: All passing (20, 400, 8902, 197281)
- **Fruit Match Score**: 0-20 (all losses by checkmate)

## Performance Metrics (startpos depth 8)
- Nodes: ~1.2M
- Time: ~2 seconds
- NPS: ~660K

## Key Improvements Made
1. **Time check frequency** - Changed from every 256 to every 64 nodes
2. **Qsearch depth** - Extended from -1 to -4 to reduce horizon effect
3. **Mobility weights** - Increased significantly for all pieces
4. **Piece values** - Knight 320->330, Queen 900->1000
5. **Pruning** - Reduced LMP and razoring aggression
6. **Node safety limit** - Added 50M node cap to prevent runaway searches

## Remaining Issues
1. **Strength gap** - Losing all games to Fruit (~2300 ELO)
2. **King safety** - May need improved attack weighting
3. **Search depth** - Need deeper search for tactical vision

## Next Steps
1. Improve king safety evaluation with weighted attack counts
2. Add continuation history for better move ordering
3. Implement reverse futility pruning
4. Consider evaluation simplification to reduce noise

## Perft Validation
- Perft 1: 20 PASS
- Perft 2: 400 PASS
- Perft 3: 8902 PASS
- Perft 4: 197281 PASS
