#include <iostream>
#include <string>

// Simple square representation
int file = 3; // d = 4th file, 0-indexed
int rank = 4; // 5 = 5th rank, 0-indexed

int main() {
    std::cout << "Knight on d5 (file=" << file << ", rank=" << rank << ")" << std::endl;
    std::cout << "Possible knight moves:" << std::endl;
    
    // Knight moves: (±1, ±2) or (±2, ±1)
    int moves[8][2] = {{1,2}, {1,-2}, {-1,2}, {-1,-2}, {2,1}, {2,-1}, {-2,1}, {-2,-1}};
    
    for (int i = 0; i < 8; i++) {
        int f = file + moves[i][0];
        int r = rank + moves[i][1];
        if (f >= 0 && f < 8 && r >= 0 && r < 8) {
            char file_char = 'a' + f;
            char rank_char = '1' + r;
            std::cout << "  " << file_char << rank_char << std::endl;
        }
    }
    
    std::cout << std::endl;
    std::cout << "Is c6 in this list? Knight from d5 to c6 is:" << std::endl;
    std::cout << "  d5 -> c6: file -1 (d->c), rank +1 (5->6)" << std::endl;
    std::cout << "  That's (-1, +1) which is NOT a knight move!" << std::endl;
    std::cout << "  Knight moves are (±1, ±2) or (±2, ±1), not (±1, ±1)" << std::endl;
    
    return 0;
}
