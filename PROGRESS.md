# Luminex Chess Engine - Progress

## Version History
| Version | Illegal Move Rate | Score vs Fruit | What Changed | What I Learned |
|---------|------------------|----------------|--------------|----------------|
| ... | ... | ... | Previous sessions | ... |
| **Current** | **0% (0/60)** | **0/60** | **slider_blockers + Zobrist + eval fixes** | **Engine stable, fundamentally weak** |

## Latest Session Fixes
| Commit | Bug | Fix | Impact |
|--------|-----|-----|--------|
| 9a7704f | Broken slider_blockers | Use attack functions with empty occupancy | Fixed pin detection, 0% illegal moves |
| 9a7704f | Promotion Zobrist key | XOR out pawn from origin | Fixed position key corruption |
| 6c676d8 | Wrong phase calculation | Standard piece-count formula | Proper MG/EG tapering |
| 6c676d8 | Mobility weights too high | Reduced to standard (K:4/6, B:5/7, R:3/5, Q:2/3) | More balanced evaluation |
| 18b4b4f | No PV prioritization at root | Prioritize previous best move | Improved search stability |

## Current Status
- **Commit:** 91aeb82
- **Illegal Move Rate:** **0%** ✅ (60 games tested)
- **Score vs Fruit 2.1:** **0-60** at tc=1+0.1 (all losses by checkmate)

## What Was Fixed
1. **slider_blockers()** - Completely broken, now correctly identifies pinned pieces
2. **Promotion Zobrist** - Pawn wasn't XORed out, causing key corruption
3. **Phase calculation** - Changed from `material/200` to standard formula
4. **Mobility weights** - Reduced from aggressive to standard values
5. **PV move ordering** - Previous iteration's best move now prioritized at root

## Remaining Issues
The engine is ~1000+ ELO weaker than Fruit 2.1. To be competitive, it needs:
1. Better tactical vision (search depth improvements)
2. More sophisticated evaluation (king safety, pawn structure)
3. Better time management
4. Opening book (for avoiding early mistakes)

## Test Results
| Test | Games | Illegal Moves | Score |
|------|-------|---------------|-------|
| Latest build | 60 | 0 ✅ | 0-60 |

## Conclusion
Engine is **legally sound** with 0% illegal moves across 60 games. However, significant evaluation and search improvements are needed to compete with strong engines like Fruit 2.1 (~2700 ELO).
