#include "src/luminex.h"
#include <iostream>

using namespace luminex;

int main() {
    init_zobrist();
    init_evaluation();

    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    std::cout << "Starting position evaluation breakdown:" << std::endl;

    int white_mg = 0;
    int white_eg = 0;
    int black_mg = 0;
    int black_eg = 0;

    // Count material
    white_mg += 8 * 100;  // 8 pawns
    white_mg += 2 * 320; // 2 knights
    white_mg += 2 * 330; // 2 bishops
    white_mg += 2 * 500; // 2 rooks
    white_mg += 1 * 900; // 1 queen
    white_eg = white_mg;

    black_mg += 8 * 100;
    black_mg += 2 * 320;
    black_mg += 2 * 330;
    black_mg += 2 * 500;
    black_mg += 1 * 900;
    black_eg = black_mg;

    std::cout << "WHITE material MG: " << white_mg << " EG: " << white_eg << std::endl;
    std::cout << "BLACK material MG: " << black_mg << " EG: " << black_eg << std::endl;

    // Check PST for pawns
    for (int sq = 0; sq < 64; sq++) {
        Piece pc = pos.piece_on(Square(sq));
        if (pc != 12) {
            Color c = Color(pc / 6);
            PieceType pt = piece_type_of(pc);
            int mg_val = PST_MG_TABLE[int(c)][int(pt)][sq];
            int eg_val = PST_EG_TABLE[int(c)][int(pt)][sq];
            std::cout << "Square " << sq << " piece=" << int(pc) << " c=" << int(c) << " pt=" << int(pt)
                      << " MG=" << mg_val << " EG=" << eg_val << std::endl;
        }
    }

    std::cout << "Total WHITE MG: " << white_mg << " EG: " << white_eg << std::endl;
    std::cout << "Total BLACK MG: " << black_mg << " EG: " << black_eg << std::endl;
    std::cout << "Net score MG: " << (white_mg - black_mg) << std::endl;
    std::cout << "Net score EG: " << (white_eg - black_eg) << std::endl;

    Value eval = evaluate(pos);
    std::cout << "\nFinal eval (from WHITE perspective): " << eval << std::endl;

    return 0;
}
