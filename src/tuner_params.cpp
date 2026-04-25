#include "tuner_params.h"
#include <cstring>

namespace luminex {

std::array<int, PARAM_COUNT> g_params;

void init_tuner_params() {
    // Material values
    g_params[PAWN_VALUE_MG]   = 90;
    g_params[PAWN_VALUE_EG]   = 100;
    g_params[KNIGHT_VALUE_MG] = 320;
    g_params[KNIGHT_VALUE_EG] = 290;
    g_params[BISHOP_VALUE_MG] = 340;
    g_params[BISHOP_VALUE_EG] = 310;
    g_params[ROOK_VALUE_MG]   = 500;
    g_params[ROOK_VALUE_EG]   = 530;
    g_params[QUEEN_VALUE_MG]  = 960;
    g_params[QUEEN_VALUE_EG]  = 940;

    // Knight mobility
    g_params[KNIGHT_MOB_BASE_MG]  = -60;
    g_params[KNIGHT_MOB_SLOPE_MG] = 15;
    g_params[KNIGHT_MOB_BASE_EG]  = -75;
    g_params[KNIGHT_MOB_SLOPE_EG] = 18;

    // Bishop mobility
    g_params[BISHOP_MOB_BASE_MG]  = -48;
    g_params[BISHOP_MOB_SLOPE_MG] = 7;
    g_params[BISHOP_MOB_BASE_EG]  = -56;
    g_params[BISHOP_MOB_SLOPE_EG] = 9;

    // Rook mobility
    g_params[ROOK_MOB_BASE_MG]  = -38;
    g_params[ROOK_MOB_SLOPE_MG] = 5;
    g_params[ROOK_MOB_BASE_EG]  = -48;
    g_params[ROOK_MOB_SLOPE_EG] = 7;

    // Queen mobility
    g_params[QUEEN_MOB_BASE_MG]  = -28;
    g_params[QUEEN_MOB_SLOPE_MG] = 2;
    g_params[QUEEN_MOB_BASE_EG]  = -38;
    g_params[QUEEN_MOB_SLOPE_EG] = 3;

    // Key EvalParams
    g_params[BISHOP_PAIR_MG]    = 40;
    g_params[BISHOP_PAIR_EG]    = 100;
    g_params[ROOK_OPEN_FILE_MG] = 25;
    g_params[ROOK_OPEN_FILE_EG] = 40;
    g_params[ROOK_SEMI_OPEN_MG] = 15;
    g_params[ROOK_SEMI_OPEN_EG] = 20;
}

} // namespace luminex
