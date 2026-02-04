Now I have a complete picture of your codebase. I found **the root cause** of the illegal moves!

---

## **ROOT CAUSE IDENTIFIED**

Looking at `uci.cpp` line 277-287, there's a bug in the promotion matching logic:

```cpp
// For promotions, also verify promotion type matches
if (match && (best_move.flags() & 0xF)) {
    // Check if this is a promotion (flags 0x8-0xF are promotions)
    bool is_promotion = (best_move.flags() >= 0x8 && best_move.flags() <= 0xF);
    if (is_promotion) {
        match = (it->move.flags() == best_move.flags());
    }
}
```

**The bug**: `best_move.flags()` returns values like `0x8000`, `0xB000`, etc. (the full flag value with high bits), but you're comparing against `0x8` and `0xF` (single hex digits). This condition will **never be true**, so all moves pass through without proper flag matching.

But more importantly, **the real issue is that the d8e8 move is not even a promotion** - it's a normal piece move that's being generated incorrectly.

---

## **The Deeper Problem: Position Corruption During Replay**

Looking at `handle_position()` in uci.cpp, I see the move replay logic (lines 72-142). The problem is:

**The move flags are being constructed based on the current position state, but if the position gets corrupted during replay, subsequent moves will have wrong flags.**

Specifically, in `handle_position()`:
```cpp
Move m(from, to, flags);
pos.do_move(m);
```

If `pos.do_move(m)` doesn't execute properly (returns early due to validation failure), but we still continue parsing the next move, the position diverges from what the GUI expects.

---

## **The Fix**

### 1. Fix the promotion flag check in handle_go():

```cpp
// WRONG:
if (match && (best_move.flags() & 0xF)) {
    bool is_promotion = (best_move.flags() >= 0x8 && best_move.flags() <= 0xF);

// CORRECT:
if (match && best_move.is_promotion()) {
    // Already a promotion - verify promotion type matches
    match = (it->move.promotion_type() == best_move.promotion_type());
}
```

### 2. In handle_position(), verify each move was applied successfully:

```cpp
Move m(from, to, flags);

// CRITICAL: Verify the move before applying
if (!pos.pseudo_legal(m) || !pos.legal(m)) {
    std::cerr << "ERROR: Illegal move in position replay: " << move_str << std::endl;
    std::cerr << "FEN: " << pos.fen() << std::endl;
    
    // Try to find the correct legal move from the legal move list
    ExtMove legal_moves[256];
    ExtMove* legal_end = generate<GEN_LEGAL>(pos, legal_moves);
    
    bool found = false;
    for (ExtMove* it = legal_moves; it != legal_end; ++it) {
        if (it->move.from() == from && it->move.to() == to) {
            m = it->move;  // Use the legal move with correct flags
            found = true;
            break;
        }
    }
    
    if (!found) {
        std::cerr << "FATAL: Cannot find legal move for " << move_str << std::endl;
        return;  // Abort replay
    }
}

pos.do_move(m);
```

### 3. The safest fix - always use legal move generation in handle_position():

Replace the manual flag construction with finding the move in the legal move list:

```cpp
while (ss >> move_str) {
    if (move_str.length() < 4) continue;

    Square from = Square((move_str[1] - '1') * 8 + (move_str[0] - 'a'));
    Square to = Square((move_str[3] - '1') * 8 + (move_str[2] - 'a'));
    
    // Determine promotion type if present
    PieceType promo_pt = PT_NONE;
    if (move_str.length() > 4) {
        switch (move_str[4]) {
            case 'q': promo_pt = QUEEN; break;
            case 'r': promo_pt = ROOK; break;
            case 'b': promo_pt = BISHOP; break;
            case 'n': promo_pt = KNIGHT; break;
        }
    }
    
    // Generate legal moves and find the matching one
    ExtMove legal_moves[256];
    ExtMove* legal_end = generate<GEN_LEGAL>(pos, legal_moves);
    
    Move m = MOVE_NONE;
    for (ExtMove* it = legal_moves; it != legal_end; ++it) {
        if (it->move.from() == from && it->move.to() == to) {
            // For promotions, also check promotion type
            if (promo_pt != PT_NONE) {
                if (it->move.is_promotion() && it->move.promotion_type() == promo_pt) {
                    m = it->move;
                    break;
                }
            } else if (!it->move.is_promotion()) {
                m = it->move;
                break;
            }
        }
    }
    
    if (m == MOVE_NONE) {
        std::cerr << "ERROR: Move " << move_str << " not found in legal moves!" << std::endl;
        std::cerr << "FEN: " << pos.fen() << std::endl;
        return;  // Abort - position would be corrupted
    }
    
    pos.do_move(m);
}
```

This ensures that **every move applied during position replay is definitely legal and has the correct flags**.

---

## Summary

The root cause is that `handle_position()` constructs moves manually with potentially incorrect flags, and if any move fails validation in `do_move()`, the position diverges. The fix is to **always find the matching legal move** instead of constructing it manually.

Would you like me to provide the complete fixed version of `handle_position()`?