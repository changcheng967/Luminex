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
    int bishop_pair_mg = 40;
    int bishop_pair_eg = 100;
    int rook_open_mg = 25;
    int rook_open_eg = 40;
    int rook_semi_open_mg = 15;
    int rook_semi_open_eg = 20;
    int rook_7th_mg = 30;
    int rook_7th_eg = 25;
    int pawn_shield_center = 12;
    int pawn_shield_knight = 15;
    int pawn_shield_rook = 8;
    int pawn_storm = 10;
    int open_file_penalty_mg = 20;
    int open_file_penalty_eg = 15;
    int outpost_knight_mg = 30;
    int outpost_knight_eg = 15;
    int outpost_bishop_mg = 40;
    int outpost_bishop_eg = 25;
    int hanging_pawn_mg = 10;
    int hanging_pawn_eg = 35;
    int far_knight_mg = 15;
    int far_knight_eg = 5;
    int far_bishop_mg = 10;
    int far_bishop_eg = 5;
};

extern EvalParams g_eval_params;

// Main evaluation function
Value evaluate(const Position& pos);

// Evaluation scaling
int scale_factor(const Position& pos, Value eg);

// Initialize evaluation tables
void init_evaluation();

} // namespace luminex
