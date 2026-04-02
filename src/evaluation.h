#pragma once

#include "board.h"
#include "types.h"

namespace luminex {

class Position;

// Piece-square tables (MG and EG phases)
extern Score PST_MG_TABLE[2][8][64];
extern Score PST_EG_TABLE[2][8][64];

// Main evaluation function
Value evaluate(const Position& pos);

// Evaluation scaling
int scale_factor(const Position& pos, Value eg);

// Initialize evaluation tables
void init_evaluation();

} // namespace luminex
