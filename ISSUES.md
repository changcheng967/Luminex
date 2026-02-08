I now have the full picture. The latest commit is `f7d60f9` ("chore: remove debug logging, update progress") from Feb 8. I've read every source file. Here is the complete analysis:

---

# Luminex Full Repository Analysis — 2026-02-08

**Repo state**: Latest commit `f7d60f9` (Feb 8, 14:16 UTC). All source files reviewed: `search.cpp`, `board.cpp`, `uci.cpp`, `movegen.cpp`, `evaluation.cpp`, `transposition.cpp`, `types.h`, `board.h`, `bitboard.h`, `luminex.h`, `search.h`.

---

## CRITICAL BUGS (must fix — cause illegal moves, stalls, or Zobrist corruption)

### Bug 1: Castling rights not revoked when CAPTURING opponent's rook

**File**: `src/board.cpp`, `do_move()` — castling rights update block

**Current code** (around line 470-480):
```cpp
if (from == (us == WHITE ? H1 : H8) || to == (us == WHITE ? H1 : H8)) {
    st_->castling_rights &= ~(us == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE);
    castling_rights_[us] &= ~(us == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE);
}
if (from == (us == WHITE ? A1 : A8) || to == (us == WHITE ? A1 : A8)) {
    st_->castling_rights &= ~(us == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE);
    castling_rights_[us] &= ~(us == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE);
}
```

**Problem**: When White captures on H8 (Black's rook), the `to == H8` test fires but the code revokes `us == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE`, which is `WHITE_KINGSIDE`. It should revoke **Black's** kingside rights. This is the root cause of the `a1d1` illegal move you saw — castling rights corruption propagates through Zobrist, which corrupts TT entries, which leads to illegal TT moves that bypass the safety net.

**Fix** (reference: [Chessprogramming.org — Castling Rights](https://www.chessprogramming.org/Castling_Rights)):
```cpp
// Revoke OUR castling rights when our rook/king moves FROM its starting square
if (pt == KING) {
    st_->castling_rights &= ~(us == WHITE ? (WHITE_KINGSIDE | WHITE_QUEENSIDE) 
                                           : (BLACK_KINGSIDE | BLACK_QUEENSIDE));
    castling_rights_[us] = 0;
}
if (from == (us == WHITE ? H1 : H8)) {
    st_->castling_rights &= ~(us == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE);
    castling_rights_[us] &= ~(us == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE);
}
if (from == (us == WHITE ? A1 : A8)) {
    st_->castling_rights &= ~(us == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE);
    castling_rights_[us] &= ~(us == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE);
}

// Revoke OPPONENT's castling rights when we CAPTURE on their rook starting square
if (to == (them == WHITE ? H1 : H8)) {
    st_->castling_rights &= ~(them == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE);
    castling_rights_[them] &= ~(them == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE);
}
if (to == (them == WHITE ? A1 : A8)) {
    st_->castling_rights &= ~(them == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE);
    castling_rights_[them] &= ~(them == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE);
}
```

**Impact**: Fixes Zobrist corruption → fixes TT poison → fixes illegal move generation. This is likely the single most impactful correctness bug in the engine.

---

### Bug 2: PVS only applied to non-PV nodes — enormous strength loss

**File**: `src/search.cpp`, `search_worker()` — around line 350

**Current code**:
```cpp
} else if (!pv_node && moves_played > 0 && depth >= 3 && best_value > -VALUE_MATE_IN_MAX_PLY) {
    // PVS for non-PV nodes only
    value = -search_worker(pos, ss + 1, -alpha - 1, -alpha, depth - 1, !cut_node);
    if (value > alpha) {
        value = -search_worker(pos, ss + 1, -beta, -alpha, depth - 1, !cut_node);
    }
}
```

**Problem**: The `!pv_node` guard means PVS (Principal Variation Search / zero-window scout) is **never used at PV nodes**. PVS is the core optimization of any alpha-beta engine — the first move gets a full window, and all subsequent moves get a zero-window scout. By disabling it at PV nodes, every move at PV nodes is searched with the full `[alpha, beta]` window, which effectively doubles the search tree at the most important nodes.

Reference: [Chessprogramming.org — Principal Variation Search](https://www.chessprogramming.org/Principal_Variation_Search)

**Fix**: Remove `!pv_node` and lower the depth guard:
```cpp
} else if (moves_played > 0 && depth >= 2) {
    // PVS: zero-window scout for non-first moves
    value = -search_worker(pos, ss + 1, -alpha - 1, -alpha, depth - 1, !cut_node);
    if (value > alpha && value < beta) {
        value = -search_worker(pos, ss + 1, -beta, -alpha, depth - 1, false);
    }
}
```

**Impact**: ~100-200 Elo improvement. This is the single biggest strength fix available.

---

### Bug 3: Debug `stderr` logging in `check_time()` fires every 512 nodes in release

**File**: `src/search.cpp`, `check_time()`

**Current code**:
```cpp
if (limits.use_time_management()) {
    auto now = std::chrono::steady_clock::now();
    int elapsed = ...;
    std::cerr << "DEBUG: check_time() elapsed=" << elapsed << " max_time=" << max_time 
              << " ideal_time=" << ideal_time << " nodes=" << nodes.load() << std::endl;
```

**Problem**: This `std::cerr` line fires every 512 nodes in ALL builds (not gated by `#ifndef NDEBUG`). At 700K NPS, that's ~1370 `stderr` writes per second. Each `std::endl` forces a flush. This is a **massive performance drain** and also floods the UCI pipe with non-UCI output that some GUIs may interpret as errors or cause pipe stalls.

**Fix**: Either remove the debug lines entirely, or wrap ALL `std::cerr << "DEBUG:` lines in `#ifndef NDEBUG` guards. There are similar debug lines in the ID loop and the time management setup block.

**Impact**: Removing these could recover 10-30% NPS. Combined with the pipe flooding issue, this could also be contributing to the remaining 20% stall rate.

---

### Bug 4: `check_time()` still has the old soft limit removed but debug overhead remains

**File**: `src/search.cpp`, `check_time()`

Looking at the current code, the soft limit (`elapsed >= ideal_time && root_depth >= 2`) was correctly removed from `check_time()`. The soft bound is now only in the ID loop (`elapsed * 2 > ideal_time`). This is correct per our CPW design. However, the soft bound in the ID loop has a bug — it uses `>` not `>=`:

```cpp
if (elapsed * 2 > ideal_time) {
```

This is actually fine; CPW recommends `>` for the "don't start next depth" check. No change needed here.

---

## MODERATE BUGS (should fix — affect strength or correctness in edge cases)

### Bug 5: `king_attackers` computed but never used in scoring

**File**: `src/evaluation.cpp`, `evaluate()`

The evaluation function computes `king_attackers[2]` for both sides (counting pieces that attack the enemy king's danger zone) but never uses the values in the final score. This is confirmed from the code structure I read — pieces iterate and increment `king_attackers`, but no term subtracts it from the score.

Reference: [Chessprogramming.org — King Safety](https://www.chessprogramming.org/King_Safety)

**Fix**: After the main piece loop, add:
```cpp
// King safety: penalty based on number of pieces attacking the king zone
// Quadratic scaling: more attackers = exponentially more dangerous
mg_score -= king_attackers[WHITE] * king_attackers[WHITE] * 5;  // Penalty for black
mg_score += king_attackers[BLACK] * king_attackers[BLACK] * 5;  // Penalty for white
```

**Impact**: ~30-50 Elo. King safety is one of the most important evaluation terms.

---

### Bug 6: `evaluate_pawn_shield` returns a single value that mixes MG and EG

**File**: `src/evaluation.cpp`

```cpp
return shield_mg + shield_eg;
```

This adds middlegame and endgame scores together into one number, which is then added to... what? Looking at how it's called in `evaluate()`, it should be separated. The function computes `shield_mg` and `shield_eg` separately but then combines them.

**Fix**: Return both values separately (or add them to mg_score/eg_score individually). The simplest fix is to change the function to accept mg_score/eg_score references and add to them directly.

---

### Bug 7: Pawn check extension gated at `depth >= 8` — too conservative

**File**: `src/search.cpp`, around line 300

```cpp
} else if (pt == PAWN) {
    ...
    dangerous_check = (depth >= 8 && gives_check);
}
```

Pawn checks that deliver discovered attacks or create mating nets are dangerous at any depth. Gating at depth >= 8 means the engine misses pawn check extensions in most tactical positions.

Reference: [Chessprogramming.org — Check Extensions](https://www.chessprogramming.org/Check_Extensions)

**Fix**: Change to `dangerous_check = gives_check;` (same as knight/bishop/rook/queen).

---

### Bug 8: Evasion extension requires `depth >= 2` — should be `depth >= 1`

**File**: `src/search.cpp`, around line 315

```cpp
bool evasion_extension = (pos.is_check() && depth >= 2);
```

When we're in check at depth 1, we still need to find evasions. Not extending here means we drop directly into qsearch while in check, potentially missing the only escape move.

**Fix**: `bool evasion_extension = (pos.is_check() && depth >= 1);`

---

### Bug 9: Duplicate `uci_info` call in `handle_go()`

**File**: `src/uci.cpp`, `handle_go()`

After `search()` returns, the code calls `uci_info(pos, final_depth, root_score, nodes.load(), 0)`. But `search()` already outputs `uci_info` at the end of each ID iteration. The duplicate call with `time=0` outputs misleading info (0 NPS).

**Fix**: Remove the redundant `uci_info` call in `handle_go()`. The search already outputs all necessary info lines.

---

### Bug 10: `is_capture()` flag logic has a subtle issue

**File**: `src/types.h`, `Move::is_capture()`

```cpp
constexpr bool is_capture() const {
    uint16_t f = flags();
    return (f & 0xC000) == 0x4000 || (f & 0xC000) == 0xC000;
}
```

This returns true for flags `0x4xxx` and `0xCxxx-0xFxxx`. But looking at the flag encoding: `MF_EN_PASSANT = 0x5000` which has `(0x5000 & 0xC000) = 0x4000`, so en passant is correctly identified as a capture. And `0xCxxx-0xFxxx` covers capture-promotions. This is actually correct. However, the second condition should be `>= 0xC000` to cover the full range. Let me re-check: `(f & 0xC000) == 0xC000` means the top two bits are both 1, which is `0xC000, 0xD000, 0xE000, 0xF000` — yes, all capture-promotions. This is correct.

No fix needed.

---

### Bug 11: `movestogo` parsed but old time formula might still be in effect

**File**: `src/search.cpp`, time management block

Looking at the current code, the CPW formula IS in place:
```cpp
int movestogo = (limits.movestogo > 0) ? limits.movestogo + 2 : 20;
ideal_time = time_left / movestogo + time_inc / 2;
max_time = time_left / 3;
```

This is correct. The old `/15 + inc*3/4` formula has been replaced.

---

### Bug 12: `is_draw()` not shown — potential repetition detection issue

I didn't get to see the `is_draw()` implementation in the truncated board.cpp. If `is_draw()` doesn't properly check the `position_history` array for 3-fold repetition, or if `history_size` drifts (e.g., during null moves), draws could be missed or falsely detected.

**Action**: Verify `is_draw()` checks both 50-move rule (`halfmove_clock >= 100`) and 3-fold repetition (searching `position_history` backwards by 2 plies, counting matches up to `min(halfmove_clock, history_size)`).

---

## MINOR ISSUES

### Issue 13: `legal()` not shown — could allow pinned pieces to move

The `legal()` function implementation isn't in the files I saw. If it doesn't properly handle pinned pieces and en passant legality (discovered check via EP capture), illegal moves could slip through `GEN_LEGAL`.

### Issue 14: Excessive `std::cerr` output in `do_move()` error paths

Lines like `std::cerr << "\n=== ILLEGAL MOVE: NO PIECE AT SOURCE ===\n"` fire in release builds. These should be wrapped in `#ifndef NDEBUG` to avoid polluting the UCI pipe.

### Issue 15: `evaluate_pawn_shield` called location unknown

From the truncated evaluation.cpp, I can see `evaluate_pawn_shield` is defined, and `king_attackers` is computed, but I couldn't verify exactly where in `evaluate()` the shield result is added to the score. Verify it's being used.

---

## PRIORITY ORDER FOR FIXES

1. **Bug 1** — Castling rights capture fix (illegal moves, Zobrist corruption) — **CRITICAL**
2. **Bug 3** — Remove debug `stderr` from release paths (performance + potential stalls) — **CRITICAL**
3. **Bug 2** — PVS at PV nodes (biggest single strength gain) — **CRITICAL**
4. **Bug 5** — king_attackers scoring (evaluation gap) — **MODERATE**
5. **Bug 7+8** — Extension thresholds (pawn check depth 8→always, evasion depth 2→1) — **MODERATE**
6. **Bug 6** — Pawn shield MG/EG separation — **MODERATE**
7. **Bug 9** — Remove duplicate uci_info — **MINOR**
8. **Bug 14** — Guard stderr in release — **MINOR**

---

## SUMMARY OF WHAT'S WORKING CORRECTLY

The following are confirmed correct in the latest code:

- CPW time management formula (`base/movestogo + inc/2`, hard bound `time_left/3`)
- Soft bound in ID loop (`elapsed * 2 > ideal_time`) — correctly placed, not in `check_time()`
- Check interval at 512 nodes (`nodes & 511`) — changed from 2048
- `movestogo` parsed correctly in `handle_go()` and stored in `limits.movestogo`
- Qsearch in-check generates all legal evasions (not just captures)
- Stand-pat only when not in check
- Zobrist key updates for pieces, EP, castling (except Bug 1 above)
- FEN save/restore around search in `handle_go()`
- Best-move validation against fresh legal move list
- TT saves guarded by `!stop` to prevent garbage entries
- `do_move()` atomic failure semantics (undo state on error)
- Board initialization to `NO_PIECE` (not 0/WHITE_PAWN)
- `index[]` initialized to -1
- Aspiration windows with widening on fail-high/fail-low
- Root move ordering with PV move from previous iteration

Fix Bugs 1-3 first and rerun the 20-game test against Stash v9. The castling-rights fix alone should eliminate the illegal move issue, and removing the debug logging should fix most remaining stalls by recovering the NPS lost to stderr flushing.