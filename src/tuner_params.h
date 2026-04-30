#pragma once

#include <array>
#include <cstddef>

namespace luminex {

// 148 tunable parameters: material (10) + mobility tables (132) + eval bonuses (6)
//
// Architecture: mobility uses non-linear lookup tables instead of linear
// base+slope formulas. Each mobility level has an independently tunable
// value, capturing diminishing returns without assuming a functional form.
//
// Knight: 9 entries (mob 0-8)  × 2 phases = 18 params
// Bishop: 14 entries (mob 0-13) × 2 phases = 28 params
// Rook:   15 entries (mob 0-14) × 2 phases = 30 params
// Queen:  28 entries (mob 0-27) × 2 phases = 56 params
//                                        Total = 132 mobility params

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

    // Knight mobility table MG (9 entries, mob 0-8)
    KNIGHT_MOB_MG_0,
    KNIGHT_MOB_MG_1,
    KNIGHT_MOB_MG_2,
    KNIGHT_MOB_MG_3,
    KNIGHT_MOB_MG_4,
    KNIGHT_MOB_MG_5,
    KNIGHT_MOB_MG_6,
    KNIGHT_MOB_MG_7,
    KNIGHT_MOB_MG_8,

    // Knight mobility table EG (9 entries, mob 0-8)
    KNIGHT_MOB_EG_0,
    KNIGHT_MOB_EG_1,
    KNIGHT_MOB_EG_2,
    KNIGHT_MOB_EG_3,
    KNIGHT_MOB_EG_4,
    KNIGHT_MOB_EG_5,
    KNIGHT_MOB_EG_6,
    KNIGHT_MOB_EG_7,
    KNIGHT_MOB_EG_8,

    // Bishop mobility table MG (14 entries, mob 0-13)
    BISHOP_MOB_MG_0,
    BISHOP_MOB_MG_1,
    BISHOP_MOB_MG_2,
    BISHOP_MOB_MG_3,
    BISHOP_MOB_MG_4,
    BISHOP_MOB_MG_5,
    BISHOP_MOB_MG_6,
    BISHOP_MOB_MG_7,
    BISHOP_MOB_MG_8,
    BISHOP_MOB_MG_9,
    BISHOP_MOB_MG_10,
    BISHOP_MOB_MG_11,
    BISHOP_MOB_MG_12,
    BISHOP_MOB_MG_13,

    // Bishop mobility table EG (14 entries, mob 0-13)
    BISHOP_MOB_EG_0,
    BISHOP_MOB_EG_1,
    BISHOP_MOB_EG_2,
    BISHOP_MOB_EG_3,
    BISHOP_MOB_EG_4,
    BISHOP_MOB_EG_5,
    BISHOP_MOB_EG_6,
    BISHOP_MOB_EG_7,
    BISHOP_MOB_EG_8,
    BISHOP_MOB_EG_9,
    BISHOP_MOB_EG_10,
    BISHOP_MOB_EG_11,
    BISHOP_MOB_EG_12,
    BISHOP_MOB_EG_13,

    // Rook mobility table MG (15 entries, mob 0-14)
    ROOK_MOB_MG_0,
    ROOK_MOB_MG_1,
    ROOK_MOB_MG_2,
    ROOK_MOB_MG_3,
    ROOK_MOB_MG_4,
    ROOK_MOB_MG_5,
    ROOK_MOB_MG_6,
    ROOK_MOB_MG_7,
    ROOK_MOB_MG_8,
    ROOK_MOB_MG_9,
    ROOK_MOB_MG_10,
    ROOK_MOB_MG_11,
    ROOK_MOB_MG_12,
    ROOK_MOB_MG_13,
    ROOK_MOB_MG_14,

    // Rook mobility table EG (15 entries, mob 0-14)
    ROOK_MOB_EG_0,
    ROOK_MOB_EG_1,
    ROOK_MOB_EG_2,
    ROOK_MOB_EG_3,
    ROOK_MOB_EG_4,
    ROOK_MOB_EG_5,
    ROOK_MOB_EG_6,
    ROOK_MOB_EG_7,
    ROOK_MOB_EG_8,
    ROOK_MOB_EG_9,
    ROOK_MOB_EG_10,
    ROOK_MOB_EG_11,
    ROOK_MOB_EG_12,
    ROOK_MOB_EG_13,
    ROOK_MOB_EG_14,

    // Queen mobility table MG (28 entries, mob 0-27)
    QUEEN_MOB_MG_0,
    QUEEN_MOB_MG_1,
    QUEEN_MOB_MG_2,
    QUEEN_MOB_MG_3,
    QUEEN_MOB_MG_4,
    QUEEN_MOB_MG_5,
    QUEEN_MOB_MG_6,
    QUEEN_MOB_MG_7,
    QUEEN_MOB_MG_8,
    QUEEN_MOB_MG_9,
    QUEEN_MOB_MG_10,
    QUEEN_MOB_MG_11,
    QUEEN_MOB_MG_12,
    QUEEN_MOB_MG_13,
    QUEEN_MOB_MG_14,
    QUEEN_MOB_MG_15,
    QUEEN_MOB_MG_16,
    QUEEN_MOB_MG_17,
    QUEEN_MOB_MG_18,
    QUEEN_MOB_MG_19,
    QUEEN_MOB_MG_20,
    QUEEN_MOB_MG_21,
    QUEEN_MOB_MG_22,
    QUEEN_MOB_MG_23,
    QUEEN_MOB_MG_24,
    QUEEN_MOB_MG_25,
    QUEEN_MOB_MG_26,
    QUEEN_MOB_MG_27,

    // Queen mobility table EG (28 entries, mob 0-27)
    QUEEN_MOB_EG_0,
    QUEEN_MOB_EG_1,
    QUEEN_MOB_EG_2,
    QUEEN_MOB_EG_3,
    QUEEN_MOB_EG_4,
    QUEEN_MOB_EG_5,
    QUEEN_MOB_EG_6,
    QUEEN_MOB_EG_7,
    QUEEN_MOB_EG_8,
    QUEEN_MOB_EG_9,
    QUEEN_MOB_EG_10,
    QUEEN_MOB_EG_11,
    QUEEN_MOB_EG_12,
    QUEEN_MOB_EG_13,
    QUEEN_MOB_EG_14,
    QUEEN_MOB_EG_15,
    QUEEN_MOB_EG_16,
    QUEEN_MOB_EG_17,
    QUEEN_MOB_EG_18,
    QUEEN_MOB_EG_19,
    QUEEN_MOB_EG_20,
    QUEEN_MOB_EG_21,
    QUEEN_MOB_EG_22,
    QUEEN_MOB_EG_23,
    QUEEN_MOB_EG_24,
    QUEEN_MOB_EG_25,
    QUEEN_MOB_EG_26,
    QUEEN_MOB_EG_27,

    // Key EvalParams (6)
    BISHOP_PAIR_MG,
    BISHOP_PAIR_EG,
    ROOK_OPEN_FILE_MG,
    ROOK_OPEN_FILE_EG,
    ROOK_SEMI_OPEN_MG,
    ROOK_SEMI_OPEN_EG,

    PARAM_COUNT = 148
};

// Mutable parameter array
extern std::array<int, PARAM_COUNT> g_params;

// Initialize g_params with current defaults (linear formula → table values)
void init_tuner_params();

// Clear pawn hash table (for tuner between positions)
void clear_pawn_table();

} // namespace luminex
