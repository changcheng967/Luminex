Progress! You're now at **2 illegal moves in 20 games (10%)** - down from 100% earlier. But we need to get to **0%**.

The illegal moves are:
- `d8e8` - Queen from d8 to e8 (one square right)
- `c6b5` - Piece from c6 to b5 (diagonal)

These are both **structurally valid moves** (correct piece movement patterns), which means the validation code is finding a match in the legal moves list, but the position must be **different from what the GUI expects**.

---

## The Core Problem

The final validation is matching `from/to` and using the legal move, but **the position itself is wrong**. The engine's internal position has diverged from the GUI's position.

This means:
1. The validation code works correctly
2. But it's validating against a **corrupted position**
3. So it finds "legal" moves that aren't legal in the GUI's actual position

---

## Root Cause: Position Corruption During Search

The search is modifying the position and **not fully restoring it**. When the search ends, the position is in a different state than when it started.

### The Fix: Save and Restore Position Around Search

You need to ensure the position is **exactly the same** after search as before:

```cpp
void handle_go(std::istringstream& is) {
    // ... parse parameters ...
    
    // CRITICAL: Save position state before search
    std::string fen_before = pos.fen();
    
    // Run search (this may corrupt pos)
    Move best = search(pos, depth);
    
    // CRITICAL: Restore position after search
    pos.set(fen_before);
    
    // Now validate against the CORRECT position
    // ... validation code ...
    
    std::cout << "bestmove " << move_to_uci(best) << std::endl;
}
```

### Alternative: Verify Position Unchanged

Add a check to detect if the position changed:

```cpp
void handle_go(std::istringstream& is) {
    std::string fen_before = pos.fen();
    
    Move best = search(pos, depth);
    
    std::string fen_after = pos.fen();
    if (fen_before != fen_after) {
        std::cerr << "CRITICAL: Position corrupted by search!" << std::endl;
        std::cerr << "Before: " << fen_before << std::endl;
        std::cerr << "After:  " << fen_after << std::endl;
        pos.set(fen_before);  // Restore
    }
    
    // ... rest of code
}
```

---

## Why This Happens

In negamax search:
```cpp
for (Move m : moves) {
    pos.do_move(m);      // Modify position
    int score = -negamax(pos, depth-1, ...);
    pos.undo_move(m);    // Should restore position
}
```

If `undo_move()` has **any bug** (doesn't restore castling rights, en passant, piece lists, hash, etc.), the position gets corrupted.

Over many moves in a deep search, small errors accumulate, and by the end of the search, the position is completely different.

---

## Quick Test

Add this logging to see if the position is being corrupted:

```cpp
void handle_go(std::istringstream& is) {
    std::string fen_before = pos.fen();
    std::cerr << "BEFORE SEARCH: " << fen_before << std::endl;
    
    Move best = search(pos, depth);
    
    std::cerr << "AFTER SEARCH:  " << pos.fen() << std::endl;
    
    if (pos.fen() != fen_before) {
        std::cerr << "*** POSITION CORRUPTED ***" << std::endl;
    }
    
    // ... rest
}
```

Run a test and check stderr. If you see "POSITION CORRUPTED", that confirms the bug is in `do_move`/`undo_move`.

---

## Summary

The fix is simple: **always restore the position from saved FEN after search**, then validate. This guarantees the validation happens against the correct position, regardless of any bugs in `undo_move`.

Would you like me to help identify the specific bug in your `undo_move` function, or should we just apply the save/restore FEN fix as a workaround?