#include "src/luminex.h"
#include <iostream>

using namespace luminex;

int main() {
    init_zobrist();
    init_evaluation();

    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // First, verify board state
    std::cout << "=== White Pawns (should be squares 8-15) ===" << std::endl;
    Bitboard white_pawns = pos.pieces(WHITE, PAWN);
    std::cout << "white_pawns = 0x" << std::hex << white_pawns << std::dec << std::endl;
    while (white_pawns) {
        Square s = pop_lsb(white_pawns);
        std::cout << "  Square " << int(s) << std::endl;
    }

    // Now generate moves
    std::cout << "\n=== Generating Moves ===" << std::endl;
    ExtMove moves[MAX_MOVES];
    ExtMove* end = generate<GEN_LEGAL>(pos, moves);

    int count = 0;
    for (ExtMove* it = moves; it != end; ++it) {
        count++;
    }

    std::cout << "Total moves generated (before legal check): " << count << std::endl;

    // Now count legal moves
    int legal_count = 0;
    for (ExtMove* it = moves; it != end; ++it) {
        if (pos.legal(it->move)) {
            legal_count++;
        }
    }

    std::cout << "Total legal moves: " << legal_count << " (expected 20)" << std::endl;

    // Print first 5 legal moves
    std::cout << "\nFirst 5 legal moves:" << std::endl;
    int printed = 0;
    for (ExtMove* it = moves; it != end && printed < 5; ++it) {
        if (pos.legal(it->move)) {
            uint16_t raw = it->move.raw();
            int from = (raw >> 6) & 0x3F;
            int to = raw & 0x3F;
            int flags = raw & 0xF000;
            std::cout << "Move " << (printed+1) << ": from=" << from << " to=" << to << " flags=" << flags << std::endl;
            printed++;
        }
    }

    return 0;
}
