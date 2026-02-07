# Luminex Progress Tracker

| Commit | Score vs Stash v9 | Illegal Moves | Key Changes | Status |
|--------|------------------|---------------|-------------|--------|
| 8467dd0 | 0/20 | 0/20 (0%) | Search stability fixes | **Stable engine!** |
| 8c7b6aa | N/A | N/A | PVS, TT margin, IIR, eval fixes | Reverted (caused stalls) |

## Latest Measurements (20 games, tc=1+0.1 vs Stash v9)
- Score: 0-20 (all by checkmate)
- Illegal Move Rate: **0%**
- Connection stalls: **0%**
- Game completion: **100%**

## Fixes Applied (8467dd0)
1. **Reverted PVS for all nodes**: Was causing exponential blowup at depth 7+ (millions of nodes)
2. **Fixed aspiration window fail-low**: Was narrowing beta instead of keeping it (could cause infinite loop)
3. **Fixed eval cache type**: int32_t instead of int16_t to match Value type
4. **Fixed bare `go` command**: Default depth 6 instead of MAX_PLY-1 (was infinite search)
5. **Reverted TT depth margin**: Require exact depth match (depth-2 caused infinite loops with IIR)

## Previous Fixes (8c7b6aa - mostly reverted)
- UCI parser MF_DOUBLE_PAWN (kept - fixes en passant)
- Reduced center pawn bonus 50/30 → 15/10 (kept)
- Value int16_t → int32_t (kept)
- IIR instead of IID (reverted - wasn't effective)
- TT cutoff depth margin (reverted - caused infinite loops)
- PVS for all nodes (reverted - caused exponential blowup)

## Current State
- **Engine is stable**: No crashes, no stalls, no illegal moves
- **Still very weak**: Loses all games by checkmate vs 1275 Elo engine
- **Node count reasonable**: ~170K at depth 8 (vs startpos)
- **Time control works**: tc=1+0.1 completes games properly

## Next Priority
1. **Improve playing strength** - Engine needs to see tactics and avoid getting mated
2. **Fix remaining evaluation issues** - Starting position still has eval inflation
3. **Improve move ordering** - Better TT move ordering, killer moves
4. **Add check extensions** - Currently too conservative
5. **Consider PVS improvements** - Need to understand why PVS-for-all caused blowup before re-enabling

## Perft Validation
- Perft 1: 20 PASS
- Perft 2: 400 PASS
- Perft 3: 8902 PASS
- Perft 4: 197281 PASS
