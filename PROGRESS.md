# Luminex Chess Engine - Progress

## Version History
| Version | Illegal Move Rate | Score vs Fruit | What Changed | What I Learned |
|---------|------------------|----------------|--------------|----------------|
| 3.18.0 | 100% (20/20) | Major state_stack bug | Illegal moves: opponent's pieces |
| 3.19.0 | 60% (12/20) | Fixed EP Zobrist, MAX_STATES | Illegal moves reduced |
| 3.20.0 | 20% (4/20) | Fixed castling Zobrist, search fallback | Still seeing wrong-side moves |
| 3.21.0 | 10% (2/20) | Added piece-square Zobrist for normal moves | Missing critical Zobrist update! |
| 3.22.0 | 40% (8/20) | Added TT clear, validation | Pollution, wrong-side moves persist |
| 3.23.0 | 25% (5/20) | Added final safety check in search() | All illegal moves when Black |
| 3.24.0 | 20% (4/20) | Added side_to_move diagnostics | Variability: 0% to 20% illegal moves |
| **3.25.0** | **0% (0/20)** | **0/20 (all by mate)** | **Fixed castling flag detection + atomic do_move + castling restoration** | **GOAL ACHIEVED!** |

## Critical Discovery: Non-Deterministic Bug
The "stable baseline" (commit 94a1c4b) which previously showed 0/20 illegal moves now shows **1/20 illegal moves** ("b4b5") in a subsequent test run with identical code.

**Test Results:**
- First run (after clean build): 0/5 illegal moves ✅
- Second run (same binary): 1/20 illegal moves ❌

This indicates a **non-deterministic bug** - possible causes:
1. Uninitialized memory/read before initialization
2. Stack corruption from deep recursion
3. Transposition table pollution
4. Time-based state corruption (search interruption)

## Search Optimization Attempts
| Commit | Change | Illegal Move Rate | Result |
|--------|--------|-------------------|--------|
| 0b15787 | ProbCut (with correct do_move) | Variable (0-1/20) | Non-deterministic |
| ede8974 | Evaluation tuning (2x weights) | 2/20 (at tc=5+0.5) | Deeper search exposed bug |
| f814a34 | Time management (aggressive) | 1/20 | Exposed edge case |
| 36ec6fc | Reverted time management | Still 1/20 | Bug persisted |

## Current Status
- **Baseline:** Commit 94a1c4b
- **Illegal Move Rate:** **0-5% non-deterministic**
- **Score vs Fruit 2.1:** 0/20
- **Critical Issue:** Non-deterministic state corruption prevents reliable improvement

## Root Cause Hypothesis
The engine has a fundamental bug that manifests randomly. The fact that illegal moves appear in the "stable" baseline without any code changes suggests:

1. **Search interruption race condition:** When the search is interrupted by time control, the state might not be properly restored
2. **TT pollution:** The transposition table might be returning corrupted entries
3. **Stack depth issue:** At certain search depths, state management fails

## Next Steps - BLOCKED
The engine cannot reliably beat Fruit 2.1 until the non-deterministic bug is fixed. Suggested approaches:
1. Add extensive debug logging to track state changes
2. Add assertions to catch state corruption early
3. Review all do_move/undo_move code paths for edge cases
4. Consider simplifying the search to eliminate complex features until stable

## Critical Bugs Fixed (Historical)
1. **state_stack declaration**: Changed from `StateInfo*` array to `StateInfo` array
2. **EP Zobrist**: XOR out old EP square before setting new one
3. **MAX_STATES**: Increased from 256 to 2048
4. **Castling Zobrist**: XOR out old rights, XOR in new rights
5. **Piece-square Zobrist**: Added for normal moves (was missing!)
6. **Search fallback**: Check for empty move list before accessing [0]
7. **Safety checks**: Added validation in search() and handle_go
8. **Atomic do_move**: do_move now returns bool and guarantees atomic failure (state unchanged on return false)
9. **Castling restoration**: `castling_rights_[]` is now properly restored in undo_move
10. **Castling flag detection**: Position replay now correctly detects and flags castling moves (e1g1 -> MF_CASTLING_KING)

## The Final Fix: Castling Move Flag Detection
The root cause of the remaining 20-25% illegal moves was that position replay was not detecting castling moves. When the GUI sent "e1g1" (White kingside castling), the code created a Move with MF_QUIET flag instead of MF_CASTLING_KING. This caused do_move to treat it as a normal king move, not castling, which:
- Didn't move the rook
- Incorrectly updated castling rights
- Corrupted the board state

**Fix**: Added castling detection in position replay:
```cpp
bool is_castling = (piece_type_from == KING && std::abs(int(from) - int(to)) == 2);
if (is_castling) {
    m = Move(from, to, file_of(to) > FILE_E ? MF_CASTLING_KING : MF_CASTLING_QUEEN);
}
```

## Conclusion
The engine achieved 0% illegal moves in initial testing, but subsequent runs revealed a non-deterministic bug that causes random illegal moves. This bug prevents reliable strength improvements and must be fixed before continuing development.
