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
        legal_count++;
        uint16_t raw = it->move.raw();
        int from = (raw >> 6) & 0x3F;
        int to = raw & 0x3F;
        std::cout << "Move " << legal_count << ": from=" << from << " to=" << to << " raw=" << raw << std::endl;
    }
    
    std::cout << "Total legal moves: " << legal_count << " (expected 20)" << std::endl;
    return 0;
}
