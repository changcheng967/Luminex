// v5_rel.h -- gen768t's relation inputs: attack relations and pawn pairs (nnue/V5-GEN768T.md, section 2).
// Which rows a position has, and which rows a move adds and removes, for one perspective or both. The rules
// and the index layout are the spec's (LAYOUT = 1); nnue.cpp sums the rows into the relation lanes.
#pragma once
#include "types.h"
#include <cstdint>

namespace luminex::v5rel {

// A board as the finder reads it: the mailbox (a Piece per square: 0-5 White P N B R Q K, 6-11 Black, 12 empty)
// and a bitboard per piece, with the occupancy.
struct Bd {
    uint8_t pc[64];
    uint64_t by[13];
    uint64_t occ;
};

constexpr int MAX_ROWS = 96;     // rows one move may add or remove for one perspective; more, and that side is rebuilt
constexpr int MAX_ACTIVE = 256;  // rows a position can have (the spec's bound is 248)

struct RowLists {
    int na[2], ns[2];
    bool overflow[2];
    uint16_t add[2][MAX_ROWS], sub[2][MAX_ROWS];
};

void init();     // builds the tables (once; later calls return at once)
int n_rows();    // 55,960: the 54,472 relation rows, then the 1,488 pawn-pair rows

// The rows the move m adds and removes for each perspective p with need[p], from the board before it.
// ppm: 0 = pawn pairs in contact, 1 = every pawn pair on the same or adjacent files.
void find(const Bd& before, Move m, int ppm, const bool need[2], RowLists& out);

// Every active row of one perspective on a board, in the order they are found; returns their number.
int enumerate(const Bd& b, int persp, int ppm, uint16_t* out);

}  // namespace luminex::v5rel
