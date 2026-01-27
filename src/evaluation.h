#pragma once

#include "board.h"
#include "types.h"

namespace luminex {

class Position;

// Main evaluation function
Value evaluate(const Position& pos);

// Evaluation scaling
int scale_factor(const Position& pos, Value eg);

// Initialize evaluation tables
void init_evaluation();

} // namespace luminex
