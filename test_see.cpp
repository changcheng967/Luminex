#include <iostream>

// Test position: White knight can capture a pawn, but the recapture wins the knight back
// Position after 1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Bxc6 dxc6 5.Nxe5??
// 
// FEN: r1bqkbnr/ppp2ppp/2p5/4N3/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 5
//
// After 5.Nxe5?? (knight captures pawn on e5):
// - White gains 1 pawn
// - Black recaptures with fxe5, winning a knight
// - Net result: white loses a knight for a pawn (-2 pieces)
//
// SEE should return negative for Nxe5

int main() {
    std::cout << "Test position for SEE:" << std::endl;
    std::cout << "FEN: r1bqkbnr/ppp2ppp/2p5/4N3/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 5" << std::endl;
    std::cout << std::endl;
    std::cout << "White knight on e5 can capture pawn on... wait, knight is already on e5" << std::endl;
    std::cout << std::endl;
    std::cout << "Let me create a proper test position:" << std::endl;
    std::cout << "White knight on e4, black pawn on d5" << std::endl;
    std::cout << "If Nxd5, black recaptures with cxd5, winning the knight" << std::endl;
    
    return 0;
}
