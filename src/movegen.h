#pragma once

#include "board.h"
#include "types.h"

namespace luminex {

struct ExtMove {
    Move move;
    Value value;

    constexpr operator Move() const { return move; }
    void operator=(Move m) { move = m; }

    // For sorting
    bool operator<(const ExtMove& other) const { return value > other.value; }
    bool operator>(const ExtMove& other) const { return value < other.value; }
};

class Position;

template<GenType T>
ExtMove* generate(const Position& pos, ExtMove* moveList);

inline ExtMove* generate_legals(const Position& pos, ExtMove* moveList) {
    return generate<GEN_LEGAL>(pos, moveList);
}

// Move ordering values (indexed by PieceType: PAWN=0, KNIGHT=1, ...)
inline Value mvv_lva(PieceType victim, PieceType attacker) {
    constexpr Value victim_score[7] = {100, 200, 300, 400, 500, 600, 0};
    constexpr Value attacker_score[7] = {6, 5, 4, 3, 2, 1, 0};
    return victim_score[victim] * 8 + attacker_score[attacker];
}

} // namespace luminex
