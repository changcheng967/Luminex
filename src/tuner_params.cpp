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

    // Knight mobility tables: -60 + 15*i (MG), -75 + 18*i (EG)
    for (int i = 0; i < 9; i++) {
        g_params[KNIGHT_MOB_MG_0 + i] = -60 + 15 * i;
        g_params[KNIGHT_MOB_EG_0 + i] = -75 + 18 * i;
    }

    // Bishop mobility tables: -48 + 7*i (MG), -56 + 9*i (EG)
    for (int i = 0; i < 14; i++) {
        g_params[BISHOP_MOB_MG_0 + i] = -48 + 7 * i;
        g_params[BISHOP_MOB_EG_0 + i] = -56 + 9 * i;
    }

    // Rook mobility tables: -38 + 5*i (MG), -48 + 7*i (EG)
    for (int i = 0; i < 15; i++) {
        g_params[ROOK_MOB_MG_0 + i] = -38 + 5 * i;
        g_params[ROOK_MOB_EG_0 + i] = -48 + 7 * i;
    }

    // Queen mobility tables: -28 + 2*i (MG), -38 + 3*i (EG)
    for (int i = 0; i < 28; i++) {
        g_params[QUEEN_MOB_MG_0 + i] = -28 + 2 * i;
        g_params[QUEEN_MOB_EG_0 + i] = -38 + 3 * i;
    }

    // Key EvalParams
    g_params[BISHOP_PAIR_MG]    = 40;
    g_params[BISHOP_PAIR_EG]    = 100;
    g_params[ROOK_OPEN_FILE_MG] = 25;
    g_params[ROOK_OPEN_FILE_EG] = 40;
    g_params[ROOK_SEMI_OPEN_MG] = 15;
    g_params[ROOK_SEMI_OPEN_EG] = 20;
}

} // namespace luminex
