#include "src/luminex.h"
#include <iostream>

using namespace luminex;

int main() {
    init_zobrist();
    init_evaluation();

    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    ExtMove moves[MAX_MOVES];
    ExtMove* end = generate<GEN_LEGAL>(pos, moves);

    std::cout << "Generated " << (end - moves) << " legal moves (expected 20)" << std::endl;
    std::cout << std::endl;

    // Print all moves
    for (ExtMove* it = moves; it != end; ++it) {
        Move m = it->move;
        Square from = m.from();
        Square to = m.to();
        char fromFile = 'a' + file_of(from);
        char fromRank = '1' + rank_of(from);
        char toFile = 'a' + file_of(to);
        char toRank = '1' + rank_of(to);

        std::cout << fromFile << fromRank << toFile << toRank;

        if (m.is_castling()) std::cout << " (CASTLING)";
        if (m.is_promotion()) std::cout << " (PROMOTION)";
        if (m.is_capture()) std::cout << " (CAPTURE)";
        if (m.is_en_passant()) std::cout << " (EN_PASSANT)";

        std::cout << std::endl;
    }

    return 0;
}
