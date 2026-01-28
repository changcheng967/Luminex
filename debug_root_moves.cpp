#include "src/luminex.h"
#include <iostream>

using namespace luminex;

int main() {
    init_zobrist();
    init_evaluation();

    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // Count and list all moves from startpos
    ExtMove moves[MAX_MOVES];
    ExtMove* end = generate<GEN_LEGAL>(pos, moves);
    int move_count = 0;
    for (ExtMove* it = moves; it != end; ++it) {
        if (pos.legal(it->move)) {
            move_count++;
            // Print first 30 moves with raw data
            if (move_count <= 30) {
                uint16_t raw = it->move.raw();
                int from_square = (raw >> 6) & 0x3F;
                int to_square = raw & 0x3F;
                int flags = raw & 0xF000;
                Piece pc = pos.piece_on(Square(from_square));
                std::cout << "Move " << move_count
                          << ": raw=" << raw
                          << " from_sq=" << from_square
                          << " to_sq=" << to_square
                          << " flags=" << flags
                          << " piece=" << int(pc) << std::endl;
            }
        }
    }

    std::cout << "Total legal moves from startpos: " << move_count << " (expected 20)" << std::endl;

    return 0;
}
