#pragma once

#include <array>
#include <cstddef>

namespace luminex {

// Phase 1: 32 tunable parameters (material + mobility + key eval params)
enum ParamIdx : size_t {
    // Material values (10)
    PAWN_VALUE_MG = 0,
    PAWN_VALUE_EG,
    KNIGHT_VALUE_MG,
    KNIGHT_VALUE_EG,
    BISHOP_VALUE_MG,
    BISHOP_VALUE_EG,
    ROOK_VALUE_MG,
    ROOK_VALUE_EG,
    QUEEN_VALUE_MG,
    QUEEN_VALUE_EG,

    // Mobility (16): base + slope for MG/EG, 4 piece types
    KNIGHT_MOB_BASE_MG,
    KNIGHT_MOB_SLOPE_MG,
    KNIGHT_MOB_BASE_EG,
    KNIGHT_MOB_SLOPE_EG,
    BISHOP_MOB_BASE_MG,
    BISHOP_MOB_SLOPE_MG,
    BISHOP_MOB_BASE_EG,
    BISHOP_MOB_SLOPE_EG,
    ROOK_MOB_BASE_MG,
    ROOK_MOB_SLOPE_MG,
    ROOK_MOB_BASE_EG,
    ROOK_MOB_SLOPE_EG,
    QUEEN_MOB_BASE_MG,
    QUEEN_MOB_SLOPE_MG,
    QUEEN_MOB_BASE_EG,
    QUEEN_MOB_SLOPE_EG,

    // Key EvalParams (6)
    BISHOP_PAIR_MG,
    BISHOP_PAIR_EG,
    ROOK_OPEN_FILE_MG,
    ROOK_OPEN_FILE_EG,
    ROOK_SEMI_OPEN_MG,
    ROOK_SEMI_OPEN_EG,

    PARAM_COUNT = 32
};

// Mutable parameter array — defaults match hardcoded values in evaluation.cpp
extern std::array<int, PARAM_COUNT> g_params;

// Initialize g_params with current hardcoded defaults
void init_tuner_params();

// Clear pawn hash table (for tuner between positions)
void clear_pawn_table();

} // namespace luminex
