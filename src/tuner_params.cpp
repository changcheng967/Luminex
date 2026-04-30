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

    // Knight mobility tables (tuned)
    const int kn_mg[9] = { -65, -42, -46, -51, -52, -45, -30, -15, 0 };
    const int kn_eg[9] = { -77, -53, 5, -5, -23, -25, -26, -9, 9 };
    for (int i = 0; i < 9; i++) {
        g_params[KNIGHT_MOB_MG_0 + i] = kn_mg[i];
        g_params[KNIGHT_MOB_EG_0 + i] = kn_eg[i];
    }

    // Bishop mobility tables (tuned)
    const int bi_mg[14] = { -60, -73, -62, -71, -68, -73, -66, -59, -52, -45, -38, -31, -23, -17 };
    const int bi_eg[14] = { -40, -63, -66, -89, -68, -71, -62, -53, -44, -35, -26, -17, 101, 1 };
    for (int i = 0; i < 14; i++) {
        g_params[BISHOP_MOB_MG_0 + i] = bi_mg[i];
        g_params[BISHOP_MOB_EG_0 + i] = bi_eg[i];
    }

    // Rook mobility tables (tuned)
    const int ro_mg[15] = { 22, 3, -5, -23, -34, -54, -56, -63, -58, -53, -48, -43, -38, -33, -28 };
    const int ro_eg[15] = { 12, -9, 1, -7, 12, -2, -13, -19, -44, -45, -38, -31, -24, -17, -10 };
    for (int i = 0; i < 15; i++) {
        g_params[ROOK_MOB_MG_0 + i] = ro_mg[i];
        g_params[ROOK_MOB_EG_0 + i] = ro_eg[i];
    }

    // Queen mobility tables (tuned)
    const int qu_mg[28] = {
        -88, -19, -35, -31, -36, -43, -25, -28, -32, -30, -24, -40, -39, -53,
        -60, -58, -8, -54, 0, -50, -44, -46, 73, 40, -40, 23, 16, 26
    };
    const int qu_eg[28] = {
        -8, -35, -86, 18, 34, -67, -13, -10, -10, -11, -68, -65, -62, -31,
        -42, -53, -50, -47, -24, -41, -38, -35, 88, -29, -26, 41, 44, 43
    };
    for (int i = 0; i < 28; i++) {
        g_params[QUEEN_MOB_MG_0 + i] = qu_mg[i];
        g_params[QUEEN_MOB_EG_0 + i] = qu_eg[i];
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
