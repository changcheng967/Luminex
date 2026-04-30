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

    // Knight mobility tables (sqrt-shaped: diminishing returns)
    int knight_mg[] = { -60, -18, -1, 13, 24, 34, 43, 51, 59 };
    int knight_eg[] = { -75, -24, -3, 13, 27, 39, 50, 60, 69 };
    for (int i = 0; i < 9; i++) {
        g_params[KNIGHT_MOB_MG_0 + i] = knight_mg[i];
        g_params[KNIGHT_MOB_EG_0 + i] = knight_eg[i];
    }

    // Bishop mobility tables (sqrt-shaped)
    int bishop_mg[] = { -48, -23, -13, -5, 2, 8, 13, 18, 23, 27, 31, 35, 39, 42 };
    int bishop_eg[] = { -56, -24, -11, -1, 8, 16, 22, 29, 35, 40, 45, 50, 55, 59 };
    for (int i = 0; i < 14; i++) {
        g_params[BISHOP_MOB_MG_0 + i] = bishop_mg[i];
        g_params[BISHOP_MOB_EG_0 + i] = bishop_eg[i];
    }

    // Rook mobility tables (sqrt-shaped)
    int rook_mg[] = { -38, -19, -11, -5, 0, 4, 9, 12, 16, 19, 22, 25, 28, 31, 33 };
    int rook_eg[] = { -48, -22, -11, -3, 4, 10, 16, 21, 26, 30, 34, 38, 42, 46, 49 };
    for (int i = 0; i < 15; i++) {
        g_params[ROOK_MOB_MG_0 + i] = rook_mg[i];
        g_params[ROOK_MOB_EG_0 + i] = rook_eg[i];
    }

    // Queen mobility tables (sqrt-shaped)
    int queen_mg[] = { -28, -18, -14, -11, -8, -6, -4, -2, 0, 2, 4, 5, 7, 8,
                        9, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
    int queen_eg[] = { -38, -22, -15, -10, -6, -2, 1, 4, 7, 10, 13, 15, 17, 20,
                        22, 24, 26, 28, 30, 32, 34, 35, 37, 39, 40, 42, 44, 45 };
    for (int i = 0; i < 28; i++) {
        g_params[QUEEN_MOB_MG_0 + i] = queen_mg[i];
        g_params[QUEEN_MOB_EG_0 + i] = queen_eg[i];
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
