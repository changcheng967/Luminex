# Luminex Progress Tracker

| Commit | Score vs Stash v9 | Illegal Moves | Key Changes | Status |
|--------|------------------|---------------|-------------|--------|
| 91e7c52 | 0/20 | 0/20 (0%) | Reverted evasion ordering (bug), PST init, evasion extension | **Stable but weak** |
| a570a1d | 0/20 | 0/20 (0%) | PST mirroring fix | **PST now correct** |
| 217458f | 0/20 | 0/20 (0%) | Check evasion extension | **Helps but not enough** |
| 8467dd0 | 0/20 | 0/20 (0%) | Search stability fixes | **Stable engine!** |

## Latest Measurements (20 games, tc=1+0.1 vs Stash v9)
- Score: 0-20 (all by checkmate)
- Illegal Move Rate: **0%**
- Connection stalls: **0%**
- Game completion: **100%**
- Estimated Elo: **<1000** (Stash v9 is ~1275)

## Recent Attempts (Not successful yet)
1. **Check evasion extension** (217458f): Extend when in check to find defenses
   - Result: No improvement, still 0-20
2. **PST mirroring fix** (a570a1d): Call init_evaluation() to mirror BLACK PST from WHITE
   - Result: PST values now correct, but no rating improvement
3. **Evasion move ordering** (ed4ed24): Prioritize captures of checking piece
   - Result: **REGRESSION** - introduced illegal move bug, reverted

## Current State
- **Engine is stable**: No crashes, no stalls, no illegal moves (after revert)
- **Extremely weak**: Loses all games by checkmate vs ~1275 Elo engine
- **Search depth**: ~7-8 in 1 second, ~900K NPS
- **Evaluation**: PST mirroring fixed, but position evaluation still problematic
- **Root cause**: Engine doesn't understand tactical threats or defensive resources

## Diagnosed Issues
1. **PST was not being mirrored** - FIXED by calling init_evaluation()
2. **No extension when in check** - FIXED with evasion extension
3. **Eval bias** - Starting position shows small bias, may need further investigation
4. **Move ordering** - Evasion moves not ordered (attempted fix caused bugs)

## Next Priority (Safe, incremental fixes)
1. **Investigate evaluation bias** - Why does engine prefer losing positions?
2. **Improve qsearch** - Ensure tactical sequences are properly evaluated
3. **Better time management** - Use more time in critical positions
4. **King safety evaluation** - Ensure danger is properly assessed

## Perft Validation
- Perft 1: 20 PASS
- Perft 2: 400 PASS
- Perft 3: 8902 PASS
- Perft 4: 197281 PASS
