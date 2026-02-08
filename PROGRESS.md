# Luminex Progress Tracker

| Commit | Score vs Stash v9 | Illegal Moves | Stalls | Key Changes | Status |
|--------|------------------|---------------|--------|-------------|--------|
| 466969a | 0/10 (all mates) | 0/10 (0%) | 0% | Reduced futility margins 20% | No regression |
| 002cfa3 | 0/15 (all mates) | 0/15 (0%) | 0% | Check extension depth 4->3 | No regression |
| 27b5b55 | 0/20 (19 mates, 1 stall) | 0/20 (0%) | 5% | Reverted qsearch fix | **Stable baseline** |
| 3c684ab | 0/20 (17 mates, 2 illegal, 1 stall) | 2/20 (10%) | 5% | CPW time management | Time mgmt fixed, illegal moves |
| e2a07c2 | N/A (direct tests only) | 0/10 (0%) | N/A | Castling rights fix + NDEBUG guards | **Correctness fixes** |
| 3ef4a81 | TBD | TBD | ~50% | Max depth limited to 8 (workaround) | **Depth 9+ hang workaround** |

## Current State (2026-02-08)
- **Latest (3ef4a81)**: Direct UCI tests work, CuteChess has 50% stalls
- **Castling Rights**: **FIXED** - Correctly revokes opponent's rights when capturing on rook squares
- **Stderr flooding**: **FIXED** - Wrapped debug output in NDEBUG guards
- **Depth 9+ Hang**: **WORKAROUND** - Limited max depth to 8, root cause still unknown
- **CuteChess Stalls**: **ONGOING** - 50% stall rate, affects either White or Black

## Critical Findings
1. **Depth 9+ Bug**: Search hangs at depth 9+ (longstanding bug, exists in earlier versions)
   - Direct tests with `go depth 9` hang indefinitely
   - Debug output shows millions of qsearch calls at ply=8 depth=0
   - Workaround: Limit max search depth to 8
2. **CuteChess Stalls**: Intermittent connection stalls
   - Direct UCI pipe tests work correctly
   - Affects either White or Black depending on timing
   - May be related to CuteChess's communication protocol
3. **Castling Rights Bug**: Fixed - was causing illegal moves via Zobrist corruption

## Known Issues
- **Depth 9+ hang**: Search tree explodes at depth 9, needs investigation
- **CuteChess stalls**: 50% games stall, cause unclear
- **Extremely weak**: <1000 Elo due to depth limitation

## Next Priority
1. **Fix depth 9+ hang** - Investigate qsearch recursion or TT corruption
2. **Fix CuteChess stalls** - Capture UCI conversation to identify issue
3. **Re-enable deeper search** - After depth 9 bug is fixed
4. **Apply search improvements** - PVS at PV nodes, extension fixes (from ANALYSIS.md)

## Perft Validation
- Perft 1: 20 PASS
- Perft 2: 400 PASS
- Perft 3: 8902 PASS
- Perft 4: 197281 PASS
