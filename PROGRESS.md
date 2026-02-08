# Luminex Progress Tracker

| Commit | Score vs Stash v9 | Illegal Moves | Key Changes | Status |
|--------|------------------|---------------|-------------|--------|
| 466969a | 0/10 (all mates) | 0/10 (0%) | Reduced futility margins 20% | No regression |
| 002cfa3 | 0/15 (all mates) | 0/15 (0%) | Check extension depth 4->3 | No regression |
| 27b5b55 | 0/20 (19 mates, 1 stall) | 0/20 (0%) | Reverted qsearch fix | **Stable baseline** |
| 5d4eb37 | 0/20 (17 mates, 3 stalls) | 0/20 (0%) | Qsearch no stand-pat in check | **Reverted - caused stalls** |

## Current State (2025-02-08)
- **Baseline (27b5b55)**: 0/20 vs Stash v9 (5% stall rate, 0% illegal)
- **Small improvements added**: Check extension, reduced futility margins
- **Evaluation**: Seems correct (Ruy Lopez eval -225 cp reasonable)
- **Estimated Elo**: **<1000** (Stash v9 is ~1275)
- **NPS**: ~650K-900K depending on position

## Recent Changes
1. **Reverted qsearch fix** - Was causing illegal moves and increased stall rate
2. **Check extension improvement** - Lowered threshold from depth 4 to 3
3. **Futility margin reduction** - Reduced by ~20% for better tactical vision

## Known Issues
- **Extremely weak**: Loses all games by checkmate
- **Root cause**: Likely insufficient search depth or poor defensive awareness
- **NOT evaluation bug**: Manual testing shows eval returns reasonable values

## Next Priority
1. **Increase search depth** - Need to find positions where engine stops too early
2. **Better defensive play** - Engine doesn't see threats coming
3. **Move ordering** - Could improve efficiency

## Perft Validation
- Perft 1: 20 PASS
- Perft 2: 400 PASS
- Perft 3: 8902 PASS
- Perft 4: 197281 PASS
