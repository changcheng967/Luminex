#pragma once

#include "board.h"
#include "types.h"

namespace luminex {

class Position;

// Piece-square tables (MG and EG phases)
extern Score PST_MG_TABLE[2][8][64];
extern Score PST_EG_TABLE[2][8][64];

// Tunable evaluation parameters (settable via UCI options for SPSA tuning)
// Values from SPSA self-play tuning (40 iterations, 800 games)
struct EvalParams {
    int bishop_pair_mg = 30;
    int bishop_pair_eg = 96;
    int rook_open_mg = 29;
    int rook_open_eg = 47;
    int rook_semi_open_mg = 20;
    int rook_semi_open_eg = 15;
    int rook_7th_mg = 30;
    int rook_7th_eg = 17;
    int pawn_shield_center = 11;
    int pawn_shield_knight = 15;
    int pawn_shield_rook = 8;
    int pawn_storm = 9;
    int open_file_penalty_mg = 21;
    int open_file_penalty_eg = 18;
    int outpost_knight_mg = 27;
    int outpost_knight_eg = 17;
    int outpost_bishop_mg = 50;
    int outpost_bishop_eg = 31;
    int hanging_pawn_mg = 8;
    int hanging_pawn_eg = 39;
    int far_knight_mg = 29;
    int far_knight_eg = 8;
    int far_bishop_mg = 5;
    int far_bishop_eg = 3;
};

extern EvalParams g_eval_params;

// Main evaluation function
Value evaluate(const Position& pos);

// Evaluation scaling
int scale_factor(const Position& pos, Value eg);

// Initialize evaluation tables
void init_evaluation();

} // namespace luminex
