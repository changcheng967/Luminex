// v5_rel.h -- V5's relation inputs (nnue/V5-draft/V5-SPEC.md, section 2) for the gate G1 prototype: which rows a
// position has, and which rows a move adds and removes. The rules are v5-relations-ref.py's (the definition) and
// this code is v5_census.h's, which the census checked against a brute force on 18.1M search moves.
// A prototype to measure speed with: the weight table it indexes is zero.
#pragma once
#include "types.h"
#include <cstdint>

namespace luminex::v5rel {

// A board: mailbox (0-5 white P N B R Q K, 6-11 black, 12 empty) and bitboards by piece.
struct Bd {
    uint8_t pc[64];
    uint64_t by[13];
    uint64_t occ;
};

constexpr int MAX_ROWS = 96;          // rows one move may add or remove for one perspective; more: rebuild it
constexpr int MAX_ACTIVE = 256;       // rows a position can have (the spec's bound is 248)

struct RowLists {
    int na[2], ns[2];
    bool overflow[2];
    uint16_t add[2][MAX_ROWS], sub[2][MAX_ROWS];
};

void init();                          // build the tables (once, before any other call)
int n_rows();                         // 55,960: relation rows, then pawn-pair rows

// The rows the move adds and removes for each perspective p with need[p], from the board before it.
// ppm: 0 contact pawn pairs, 1 every pawn pair on the same or adjacent files.
void find(const Bd& before, Move m, int ppm, const bool need[2], RowLists& out);

// Every row of one perspective on a board; returns their number.
int enumerate(const Bd& b, int persp, int ppm, uint16_t* out);

}  // namespace luminex::v5rel
