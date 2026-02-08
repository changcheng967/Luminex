# Luminex Progress Tracker

| Commit | Score vs Stash v9 | Illegal Moves | Stalls | Key Changes | Status |
|--------|------------------|---------------|--------|-------------|--------|
| 466969a | 0/10 (all mates) | 0/10 (0%) | 0% | Reduced futility margins 20% | No regression |
| 002cfa3 | 0/15 (all mates) | 0/15 (0%) | 0% | Check extension depth 4->3 | No regression |
| 27b5b55 | 0/20 (19 mates, 1 stall) | 0/20 (0%) | 5% | Reverted qsearch fix | **Stable baseline** |
| 5d4eb37 | 0/20 (17 mates, 3 stalls) | 0/20 (0%) | 15% | Qsearch no stand-pat in check | **Reverted - caused stalls** |
| 3c684ab | 0/20 (17 mates, 2 illegal, 1 stall) | 2/20 (10%) | 5% | CPW time management + qsearch fix | **Time management fixed, illegal moves appeared** |

## Current State (2026-02-08)
- **Latest (3c684ab)**: 0/20 vs Stash v9 (85% checkmate losses, 10% illegal, 5% stalls)
- **Time Management**: **FIXED** - Stalls reduced from 100% to 5% with CPW formula
- **Illegal Moves**: **NEW ISSUE** - 2/20 games (a1d1, h2h3) - board state corruption?
- **Estimated Elo**: **<1000** (Stash v9 is ~1275)
- **NPS**: ~650K-900K depending on position

## Recent Changes
1. **Qsearch no stand-pat in check** - Must find evasions or detect checkmate
2. **CPW time management** - base/20 + inc/2, check every 512 nodes, soft bound at 2x ideal
3. **movestogo support** - Correctly parsed and used in time allocation

## Known Issues
- **Illegal moves (10%)**: a1d1, h2h3 - suggests board state drift during position replay
- **Extremely weak**: Loses 85% of games by checkmate
- **Root cause of weakness**: Likely insufficient search depth or poor defensive awareness
- **NOT evaluation bug**: Manual testing shows eval returns reasonable values

## Next Priority
1. **Fix illegal moves** - Investigate board state corruption in handle_position()
2. **Defensive play** - Engine doesn't see threats coming (losing by mate)
3. **Move ordering** - Could improve efficiency

## Perft Validation
- Perft 1: 20 PASS
- Perft 2: 400 PASS
- Perft 3: 8902 PASS
- Perft 4: 197281 PASS
