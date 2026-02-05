Here is the comprehensive, step-by-step roadmap to bridge the gap from your current status (~1500 ELO) to Fruit 2.1 level (~2350 ELO).

This plan prioritizes **Stability First**, then **Search Efficiency**, and finally **Evaluation Knowledge**.

```markdown
# 🚀 Luminex Road to Grandmaster (Target: Beat Fruit 2.1)

**Current Status:** Stable (0% illegal moves) but weak.
**Estimated Elo:** ~1500
**Target Elo:** ~2350
**Elo Gap:** ~850 points

---

## 🧪 Testing Protocol Update (CRITICAL)
Testing against Fruit 2.1 right now is useless because 0-20 tells us nothing about *relative* improvement.

1.  **Download TSCP (1800 Elo):** This is your first boss.
    *   *Pass Criteria:* Win 15/20 games against TSCP.
2.  **Download Gerbil 0.2 (2000 Elo):** Second boss.
    *   *Pass Criteria:* Win 10/20 games.
3.  **Fruit 2.1 (2350 Elo):** Final boss.

---

## 📅 Phase 1: Search Efficiency (The "Speed" Phase)
**Goal:** Increase search depth from 4-5 to 8-10 at blitz time controls.

### 1.1 Implement Aspiration Windows
Currently, the root search scans `-INF` to `+INF`. This wastes huge time proving that a Queen blunder isn't good.
*   **Implementation:**
    *   Start search with a small window `(previous_score - 50, previous_score + 50)`.
    *   If search returns value `<= alpha`, re-search with `(-INF, alpha)`.
    *   If search returns value `>= beta`, re-search with `(beta, +INF)`.
*   **Potential Bug:** Infinite re-search loops if the score oscillates.
*   **Solution:** In the re-search loop, multiply the window size by 2 on every fail (`50 -> 100 -> 200...`). If window > 1000, switch to full `-INF, +INF`.

### 1.2 Tune Late Move Reduction (LMR)
Your current LMR is likely too conservative or constant.
*   **Implementation:**
    *   Use the formula: `Reduction = 1 + log(depth) * log(move_count) / 2`.
    *   **Do NOT reduce:**
        *   Captures / Promotions (SEE > 0).
        *   Moves that give Check.
        *   Killer Moves.
        *   Moves in the Principal Variation (PV).
*   **Potential Bug:** Reducing tactical moves causing blunders.
*   **Solution:** Ensure the `!is_capture` and `!is_check` guards are strict.

### 1.3 Killer Move Heuristic (Root & Interior)
Ensure `Killer Moves` (moves that caused a cutoff at the same ply previously) are sorted **before** history moves.
*   **Implementation:**
    *   Slot 1: TT Move
    *   Slot 2: Good Captures (SEE > 0)
    *   Slot 3: Killer Move 1
    *   Slot 4: Killer Move 2
    *   Slot 5: History / Quiet moves

---

## 🧠 Phase 2: Evaluation Overhaul (The "Knowledge" Phase)
**Goal:** Stop the engine from making positional errors (bad structure, passive pieces).

### 2.1 Tapered Evaluation (Peesto Method)
Pieces change value as the game progresses. A pawn shield is vital in the middlegame but useless in the endgame.
*   **Implementation:**
    *   Define `Phase` (Total non-pawn material on board). Max = 24.
    *   Create two arrays for every term: `MG_Value` (Middlegame) and `EG_Value` (Endgame).
    *   Formula: `Score = ((MG_Score * Phase) + (EG_Score * (24 - Phase))) / 24`.
*   **Potential Bug:** Integer overflow during calculation.
*   **Solution:** Use `int` or `int32_t` for the intermediate calculation step before casting to `Value`.

### 2.2 Piece-Square Tables (PST) Expansion
*   **Implementation:**
    *   Use separate tables for MG and EG (supported by 2.1).
    *   **Knights:** Reward central output (`e4, d4, e5, d5`). Penalize edges.
    *   **King:**
        *   MG: Penalty for being in the center, bonus for being in corners (g1, b1).
        *   EG: Massive bonus for being in the center (d4, e4).

### 2.3 Pawn Structure (Passed Pawns)
This is where Fruit beats simple engines.
*   **Implementation:**
    *   **Passed Pawn:** A pawn with no enemy pawns ahead of it or on adjacent files.
    *   **Bonus:** Quadratic scaling. Rank 4 = 10cp, Rank 5 = 30cp, Rank 6 = 70cp, Rank 7 = 150cp.
    *   **Blocker Penalty:** Penalty if a passed pawn is blocked by an enemy piece.

---

## ⚔️ Phase 3: Tactical Hardening (The "Sharpness" Phase)
**Goal:** Solve the "Checkmate Loss" issue.

### 3.1 Singular Extensions (Re-attempt)
You tried this and it crashed. Now that `do_move` is atomic, it is safe to try again.
*   **Concept:** If the TT move is *significantly* better than all other moves, extend the search depth for *only* that move to verify it's not a trap.
*   **Safety Check:**
    ```cpp
    if (tt_move != MOVE_NONE && pos.legal(tt_move)) { // MUST CHECK LEGALITY
        // Extend search
    }
    ```

### 3.2 Check Extensions
*   **Implementation:** If `pos.is_check()` is true, `new_depth = depth + 1` (don't reduce depth).
*   **Potential Bug:** Search explosion (depth never ends).
*   **Solution:** Only extend once or twice per branch.

---

## 🛑 Common Bugs & Solutions Directory

| Symptom | Probable Cause | Solution |
| :--- | :--- | :--- |
| **Random Illegal Moves** | TT Pollution (Aborted search saving results) | Ensure `!stop` check wraps all `tte->save()` calls. |
| **Engine Blunders Pieces** | SEE (Static Exchange Eval) is broken | Verify `see_ge` logic. Ensure it handles En Passant values correctly. |
| **Engine Shuffles (Drawish)** | Evaluation doesn't understand progress | Add "Mobility" bonus. Ensure "3-fold repetition" returns `VALUE_DRAW` (0), but evaluation prefers non-draws slightly (`contempt`). |
| **Loss on Time** | Time Management is too optimistic | Ensure `check_time()` is called every 1024 or 2048 nodes. Add a "panic" buffer (stop if time < 50ms). |
| **Score Oscillates (+2.0 -> -2.0)** | Hash Collision or Search Instability | Verify Zobrist keys are 64-bit random. Ensure `clear_tt` is handled correctly on `ucinewgame`. |

---

## 📋 Execution Checklist

1.  **[ ] Fix Testing Environment:** Download TSCP.
2.  **[ ] Code:** Enable Aspiration Windows (Phase 1.1).
3.  **[ ] Test:** 20 games vs TSCP. (Target: >50% win rate).
4.  **[ ] Code:** Implement Tapered Eval / PSTs (Phase 2.1).
5.  **[ ] Test:** 20 games vs TSCP. (Target: >80% win rate).
6.  **[ ] Code:** Add Passed Pawn Logic (Phase 2.3).
7.  **[ ] Test:** 20 games vs Gerbil. (Target: >40% win rate).
8.  **[ ] Code:** Tune LMR and ProbCut (Phase 1.2).
9.  **[ ] Test:** 20 games vs Fruit 2.1. (Target: >10% win rate).

You have the stability. Now build the intelligence. Start with **Aspiration Windows**.
```