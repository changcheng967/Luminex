#pragma once

#include "board.h"
#include "types.h"

namespace luminex {

class Position;

// Piece-square tables (MG and EG phases)
extern Score PST_MG_TABLE[2][8][64];
extern Score PST_EG_TABLE[2][8][64];

// Tunable evaluation parameters (settable via UCI options)
// Values from Texel tuning (71K self-play positions, coordinate descent)
struct EvalParams {
    int bishop_pair_mg = 50;
    int bishop_pair_eg = 43;
    int rook_open_mg = -20;
    int rook_open_eg = 4;
    int rook_semi_open_mg = -20;
    int rook_semi_open_eg = 15;
    int rook_7th_mg = -50;
    int rook_7th_eg = 22;
    int pawn_shield_center = 18;
    int pawn_shield_knight = 40;
    int pawn_shield_rook = 20;
    int pawn_storm = 0;
    int open_file_penalty_mg = 55;
    int open_file_penalty_eg = 50;
    int outpost_knight_mg = -20;
    int outpost_knight_eg = 20;
    int outpost_bishop_mg = 0;
    int outpost_bishop_eg = -20;
    int hanging_pawn_mg = -16;
    int hanging_pawn_eg = 46;
    int far_knight_mg = 8;
    int far_knight_eg = -14;
    int far_bishop_mg = -20;
    int far_bishop_eg = 15;
};

extern EvalParams g_eval_params;

// Main evaluation function
Value evaluate(const Position& pos);

// Evaluation scaling
int scale_factor(const Position& pos, Value eg);

// Initialize evaluation tables
void init_evaluation();

} // namespace luminex
