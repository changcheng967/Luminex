#include "src/luminex.h"
#include <iostream>

using namespace luminex;

int main() {
    init_evaluation();

    std::cout << "=== PST Values Check ===" << std::endl;

    // Check WHITE pawn at A2 (square 8) - should be positive
    std::cout << "WHITE PAWN at A2 (square 8):" << std::endl;
    std::cout << "  MG: " << PST_MG_TABLE[WHITE][PAWN][8] << std::endl;
    std::cout << "  EG: " << PST_EG_TABLE[WHITE][PAWN][8] << std::endl;

    // Check BLACK pawn at A7 (square 48) - should mirror WHITE's A2
    std::cout << "\nBLACK PAWN at A7 (square 48):" << std::endl;
    std::cout << "  MG: " << PST_MG_TABLE[BLACK][PAWN][48] << std::endl;
    std::cout << "  EG: " << PST_EG_TABLE[BLACK][PAWN][48] << std::endl;

    // For BLACK, A7 (48) should mirror WHITE's A2 (8)
    // Mirror formula: 48 ^ 56 = 8, so BLACK[48] should equal WHITE[8]
    std::cout << "\nWHITE PAWN at square (48 ^ 56 = " << (48 ^ 56) << "):" << std::endl;
    std::cout << "  MG: " << PST_MG_TABLE[WHITE][PAWN][48 ^ 56] << std::endl;

    // Sum all WHITE and BLACK pawn PST values
    int white_sum = 0;
    int black_sum = 0;
    for (int s = 0; s < 64; s++) {
        white_sum += PST_MG_TABLE[WHITE][PAWN][s];
        black_sum += PST_MG_TABLE[BLACK][PAWN][s];
    }
    std::cout << "\nWHITE PAWN PST sum: " << white_sum << std::endl;
    std::cout << "BLACK PAWN PST sum: " << black_sum << std::endl;

    return 0;
}
