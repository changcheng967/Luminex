#include "tuner_params.h"
#include <cstring>

namespace luminex {

std::array<int, PARAM_COUNT> g_params;

void init_tuner_params() {
    // Material values: EG ~1.5-2.1x MG
    g_params[PAWN_VALUE_MG]   = 90;
    g_params[PAWN_VALUE_EG]   = 155;
    g_params[KNIGHT_VALUE_MG] = 320;
    g_params[KNIGHT_VALUE_EG] = 520;
    g_params[BISHOP_VALUE_MG] = 340;
    g_params[BISHOP_VALUE_EG] = 590;
    g_params[ROOK_VALUE_MG]   = 500;
    g_params[ROOK_VALUE_EG]   = 1060;
    g_params[QUEEN_VALUE_MG]  = 960;
    g_params[QUEEN_VALUE_EG]  = 1750;

    // Knight mobility: concave curve (steep at low, flat at high)
    {
        static const int mg[9] = { -55, -37, -20, -8, 0, 8, 14, 19, 22 };
        static const int eg[9] = { -65, -42, -22, -6, 5, 15, 25, 33, 38 };
        for (int i = 0; i < 9; i++) {
            g_params[KNIGHT_MOB_MG_0 + i] = mg[i];
            g_params[KNIGHT_MOB_EG_0 + i] = eg[i];
        }
    }

    // Bishop mobility: concave curve
    {
        static const int mg[14] = { -45, -30, -18, -8, 0, 6, 12, 17, 21, 25, 28, 30, 32, 33 };
        static const int eg[14] = { -52, -35, -20, -8, 2, 12, 22, 30, 37, 43, 48, 52, 55, 57 };
        for (int i = 0; i < 14; i++) {
            g_params[BISHOP_MOB_MG_0 + i] = mg[i];
            g_params[BISHOP_MOB_EG_0 + i] = eg[i];
        }
    }

    // Rook mobility: concave, EG steep ramp
    {
        static const int mg[15] = { -35, -22, -12, -4, 2, 8, 14, 20, 25, 30, 34, 38, 41, 44, 46 };
        static const int eg[15] = { -42, -28, -15, -4, 5, 14, 24, 35, 46, 55, 63, 70, 76, 80, 83 };
        for (int i = 0; i < 15; i++) {
            g_params[ROOK_MOB_MG_0 + i] = mg[i];
            g_params[ROOK_MOB_EG_0 + i] = eg[i];
        }
    }

    // Queen mobility: peaks then drops at high mobility (overextended = exposed)
    {
        static const int mg[28] = {
            -25, -18, -13, -9, -5, -2, 1, 3, 5, 7, 8, 9, 10, 10,
            11, 11, 11, 10, 10, 9, 8, 7, 5, 3, 1, -2, -5, -10
        };
        static const int eg[28] = {
            -32, -22, -14, -6, 0, 6, 12, 18, 24, 30, 35, 39, 43, 46,
            48, 50, 51, 52, 52, 51, 50, 48, 45, 42, 38, 33, 27, 20
        };
        for (int i = 0; i < 28; i++) {
            g_params[QUEEN_MOB_MG_0 + i] = mg[i];
            g_params[QUEEN_MOB_EG_0 + i] = eg[i];
        }
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
