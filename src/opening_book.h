#pragma once

#include "types.h"

namespace luminex {

// Compact built-in opening book
// Maps position keys to a set of acceptable moves
// Used at root to avoid deterministic losing lines at bullet TC

struct BookEntry {
    uint64_t key;
    Square from;
    Square to;
    uint16_t flags;
};

// Returns a book move for the given position key, or MOVE_NONE if not in book
// Randomly selects among available moves for variety
Move book_probe(uint64_t pos_key);

} // namespace luminex
