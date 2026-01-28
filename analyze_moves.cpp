#include "src/luminex.h"
#include <iostream>

using namespace luminex;

int main() {
    init_zobrist();
    init_evaluation();
    
    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    // Count by piece type
    ExtMove moves[MAX_MOVES];
    ExtMove* end = generate<GEN_LEGAL>(pos, moves);
    
    int counts[12] = {0};  // By piece type
    
    for (ExtMove* it = moves; it != end; ++it) {
        Piece pc = pos.piece_on(it->move.from());
        counts[int(pc)]++;
        std::cout << "Move from " << int(it->move.from()) << " to " << int(it->move.to())
                  << " piece=" << int(pc) << " raw=" << it->move.raw() << std::endl;
    }
    
    std::cout << "\nMove counts by piece:" << std::endl;
    for (int i = 0; i < 12; i++) {
        if (counts[i] > 0) {
            std::cout << "  Piece " << i << ": " << counts[i] << " moves" << std::endl;
        }
    }
    
    return 0;
}
