#pragma once

// Luminex Chess Engine - Main header

#include "types.h"
#include "bitboard.h"
#include "board.h"
#include "movegen.h"
#include "evaluation.h"
#include "search.h"
#include "transposition.h"
#include "uci.h"

namespace luminex {

// Engine info
constexpr const char* ENGINE_NAME = "Luminex";
constexpr const char* ENGINE_AUTHOR = "changcheng967";
constexpr const char* ENGINE_VERSION = "2.4.0";

// Initialize engine
void init();

// Perft testing (for debugging)
uint64_t perft(Position& pos, Depth depth);

} // namespace luminex
