#include "src/luminex.h"
#include <iostream>

int main() {
    luminex::init_zobrist();
    luminex::init_evaluation();
    
    luminex::Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    luminex::ExtMove moves[luminex::MAX_MOVES];
    luminex::ExtMove* end = luminex::generate<luminex::GEN_LEGAL>(pos, moves);
    
    int legal_count = 0;
    for (luminex::ExtMove* it = moves; it != end; ++it) {
        if (pos.legal(it->move)) {
            legal_count++;
        }
    }
    
    std::cout << "Legal moves: " << legal_count << " (expected 20)" << std::endl;
    return 0;
}
