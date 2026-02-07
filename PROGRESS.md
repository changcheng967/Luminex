# Luminex Progress Tracker

| Commit | Score vs Stash v9 | Illegal Moves | Key Changes | Status |
|--------|------------------|---------------|-------------|--------|
| 5d4eb37 | 0/20 (17 mates, 3 stalls) | 0/20 (0%) | Qsearch no stand-pat in check | **Partial fix, has stalls** |
| 9adac2a | 0/20 (20 mates, 0 stalls) | 0/20 (0%) | Documentation update | Baseline |
| 91e7c52 | 0/20 | 0/20 (0%) | Reverted evasion ordering | Stable but weak |
| ed4ed24 | N/A | N/A | Evasion move ordering | Reverted (caused bugs) |

## Latest Measurements (20 games, tc=1+0.1 vs Stash v9)
- Score: 0-20 (17 checkmates, 3 connection stalls)
- Illegal Move Rate: **0%**
- Connection stalls: **15%** (3/20 games)
- Estimated Elo: **<1000** (Stash v9 is ~1275)

## Fixes Applied
1. **Qsearch stand-pat fix** (5d4eb37): Engine can no longer "stand pat" (accept static eval) when in check
   - Impact: Critical correctness fix - engine must find evasions when in check
   - Status: Introduces 15% stall rate (needs investigation)

## Known Issues
- **Stalls in 15% of games**: Engine stops responding in some positions
  - May be time management issue
  - May be infinite loop in search
  - Needs debugging with logging

## Current State
- **Engine is mostly stable**: 0% illegal moves
- **Extremely weak**: Still loses all games vs ~1275 Elo engine
- **Search depth**: ~7-8 in 1 second, ~900K NPS
- **Evaluation**: PST mirroring fixed from earlier commit
- **Root cause**: Engine lacks tactical awareness and king safety

## Next Priority
1. **Debug stalls** - Add logging to identify why engine stops responding
2. **King attack counting** - Utilize the `king_attackers` array that's computed but not used
3. **Better time management** - Ensure engine doesn't run out of time
4. **Move ordering improvements** - Better handling of evasions

## Perft Validation
- Perft 1: 20 PASS
- Perft 2: 400 PASS
- Perft 3: 8902 PASS
- Perft 4: 197281 PASS
