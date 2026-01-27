#pragma once

#include "types.h"
#include <algorithm>
#include <cstdint>

namespace luminex {

using Bitboard = uint64_t;

constexpr Bitboard BB_EMPTY = 0ULL;
constexpr Bitboard BB_ALL = ~BB_EMPTY;

// Individual squares
constexpr Bitboard BB_SQUARES[64] = {
    0x0000000000000001ULL, 0x0000000000000002ULL, 0x0000000000000004ULL, 0x0000000000000008ULL,
    0x0000000000000010ULL, 0x0000000000000020ULL, 0x0000000000000040ULL, 0x0000000000000080ULL,
    0x0000000000000100ULL, 0x0000000000000200ULL, 0x0000000000000400ULL, 0x0000000000000800ULL,
    0x0000000000001000ULL, 0x0000000000002000ULL, 0x0000000000004000ULL, 0x0000000000008000ULL,
    0x0000000000010000ULL, 0x0000000000020000ULL, 0x0000000000040000ULL, 0x0000000000080000ULL,
    0x0000000000100000ULL, 0x0000000000200000ULL, 0x0000000000400000ULL, 0x0000000000800000ULL,
    0x0000000001000000ULL, 0x0000000002000000ULL, 0x0000000004000000ULL, 0x0000000008000000ULL,
    0x0000000010000000ULL, 0x0000000020000000ULL, 0x0000000040000000ULL, 0x0000000080000000ULL,
    0x0000000100000000ULL, 0x0000000200000000ULL, 0x0000000400000000ULL, 0x0000000800000000ULL,
    0x0000001000000000ULL, 0x0000002000000000ULL, 0x0000004000000000ULL, 0x0000008000000000ULL,
    0x0000010000000000ULL, 0x0000020000000000ULL, 0x0000040000000000ULL, 0x0000080000000000ULL,
    0x0000100000000000ULL, 0x0000200000000000ULL, 0x0000400000000000ULL, 0x0000800000000000ULL,
    0x0001000000000000ULL, 0x0002000000000000ULL, 0x0004000000000000ULL, 0x0008000000000000ULL,
    0x0010000000000000ULL, 0x0020000000000000ULL, 0x0040000000000000ULL, 0x0080000000000000ULL,
    0x0100000000000000ULL, 0x0200000000000000ULL, 0x0400000000000000ULL, 0x0800000000000000ULL,
    0x1000000000000000ULL, 0x2000000000000000ULL, 0x4000000000000000ULL, 0x8000000000000000ULL
};

// Ranks
constexpr Bitboard BB_RANK_1 = 0x00000000000000FFULL;
constexpr Bitboard BB_RANK_2 = 0x000000000000FF00ULL;
constexpr Bitboard BB_RANK_3 = 0x0000000000FF0000ULL;
constexpr Bitboard BB_RANK_4 = 0x00000000FF000000ULL;
constexpr Bitboard BB_RANK_5 = 0x000000FF00000000ULL;
constexpr Bitboard BB_RANK_6 = 0x0000FF0000000000ULL;
constexpr Bitboard BB_RANK_7 = 0x00FF000000000000ULL;
constexpr Bitboard BB_RANK_8 = 0xFF00000000000000ULL;

constexpr Bitboard BB_RANKS[8] = {
    BB_RANK_1, BB_RANK_2, BB_RANK_3, BB_RANK_4,
    BB_RANK_5, BB_RANK_6, BB_RANK_7, BB_RANK_8
};

// Files
constexpr Bitboard BB_FILE_A = 0x0101010101010101ULL;
constexpr Bitboard BB_FILE_B = 0x0202020202020202ULL;
constexpr Bitboard BB_FILE_C = 0x0404040404040404ULL;
constexpr Bitboard BB_FILE_D = 0x0808080808080808ULL;
constexpr Bitboard BB_FILE_E = 0x1010101010101010ULL;
constexpr Bitboard BB_FILE_F = 0x2020202020202020ULL;
constexpr Bitboard BB_FILE_G = 0x4040404040404040ULL;
constexpr Bitboard BB_FILE_H = 0x8080808080808080ULL;

constexpr Bitboard BB_FILES[8] = {
    BB_FILE_A, BB_FILE_B, BB_FILE_C, BB_FILE_D,
    BB_FILE_E, BB_FILE_F, BB_FILE_G, BB_FILE_H
};

// Edges and center
constexpr Bitboard BB_EDGE = BB_FILE_A | BB_FILE_H | BB_RANK_1 | BB_RANK_8;
constexpr Bitboard BB_CENTER = BB_SQUARES[D4] | BB_SQUARES[E4] | BB_SQUARES[D5] | BB_SQUARES[E5];
constexpr Bitboard BB_BIG_CENTER = (BB_RANK_4 | BB_RANK_5) & (BB_FILE_D | BB_FILE_E | BB_FILE_C | BB_FILE_F);

// Diagonals
constexpr Bitboard BB_DIAGONAL_A1H8 = 0x8040201008040201ULL;
constexpr Bitboard BB_DIAGONAL_H1A8 = 0x0102040810204080ULL;

// Corners
constexpr Bitboard BB_CORNERS = BB_SQUARES[A1] | BB_SQUARES[H1] | BB_SQUARES[A8] | BB_SQUARES[H8];

// Bitboard utility functions
constexpr Bitboard square_bb(Square s) { return BB_SQUARES[s]; }
constexpr Square lsb(Bitboard b) { return static_cast<Square>(std::countr_zero(b)); }
constexpr Square msb(Bitboard b) { return static_cast<Square>(63 - std::countl_zero(b)); }
constexpr int popcount(Bitboard b) { return static_cast<int>(std::popcount(b)); }
constexpr Bitboard square_bb(File f, Rank r) { return BB_SQUARES[make_square(f, r)]; }
constexpr Bitboard rank_bb(Rank r) { return BB_RANKS[r]; }
constexpr Bitboard file_bb(File f) { return BB_FILES[f]; }

inline Square pop_lsb(Bitboard& b) {
    Square s = lsb(b);
    b &= b - 1;
    return s;
}

constexpr Bitboard shift_n(Bitboard b) { return b << 8; }
constexpr Bitboard shift_s(Bitboard b) { return b >> 8; }
constexpr Bitboard shift_e(Bitboard b) { return (b << 1) & ~BB_FILE_A; }
constexpr Bitboard shift_w(Bitboard b) { return (b >> 1) & ~BB_FILE_H; }
constexpr Bitboard shift_ne(Bitboard b) { return (b << 9) & ~BB_FILE_A; }
constexpr Bitboard shift_nw(Bitboard b) { return (b << 7) & ~BB_FILE_H; }
constexpr Bitboard shift_se(Bitboard b) { return (b >> 7) & ~BB_FILE_A; }
constexpr Bitboard shift_sw(Bitboard b) { return (b >> 9) & ~BB_FILE_H; }

// Pawn attacks
constexpr Bitboard pawn_attacks_bb(Color c, Square s) {
    return c == WHITE ?
        ((square_bb(s) << 7) & ~BB_FILE_H) | ((square_bb(s) << 9) & ~BB_FILE_A) :
        ((square_bb(s) >> 7) & ~BB_FILE_A) | ((square_bb(s) >> 9) & ~BB_FILE_H);
}

constexpr Bitboard pawn_attacks_bb(Color c, Bitboard b) {
    return c == WHITE ?
        ((b << 7) & ~BB_FILE_H) | ((b << 9) & ~BB_FILE_A) :
        ((b >> 7) & ~BB_FILE_A) | ((b >> 9) & ~BB_FILE_H);
}

// Knight attacks
constexpr Bitboard knight_attacks_bb(Square s) {
    const Bitboard b = square_bb(s);
    return (((b << 17) | (b >> 15)) & ~BB_FILE_A)
         | (((b << 10) | (b >> 6))  & ~(BB_FILE_A | BB_FILE_B))
         | (((b << 6)  | (b >> 10)) & ~(BB_FILE_G | BB_FILE_H))
         | (((b << 15) | (b >> 17)) & ~BB_FILE_H);
}

// King attacks
constexpr Bitboard king_attacks_bb(Square s) {
    const Bitboard b = square_bb(s);
    return ((b << 8) | (b >> 8))
         | ((b << 7) & ~BB_FILE_H) | ((b << 9) & ~BB_FILE_A)
         | ((b >> 7) & ~BB_FILE_A) | ((b >> 9) & ~BB_FILE_H)
         | ((b << 1) & ~BB_FILE_A) | ((b >> 1) & ~BB_FILE_H);
}

// Runtime line computation
inline Bitboard line_bb(Square s1, Square s2) {
    if (s1 == s2) return BB_EMPTY;

    int f1 = file_of(s1);
    int r1 = rank_of(s1);
    int f2 = file_of(s2);
    int r2 = rank_of(s2);

    int df = f2 - f1;
    int dr = r2 - r1;

    Bitboard line = 0;
    Square s = s1;

    // Check if aligned
    if (df == 0 || dr == 0 || (df < 0 ? -df : df) == (dr < 0 ? -dr : dr)) {
        int step_f = df == 0 ? 0 : (df > 0 ? 1 : -1);
        int step_r = dr == 0 ? 0 : (dr > 0 ? 1 : -1);

        do {
            s = Square(s + step_r * 8 + step_f);
            line |= square_bb(s);
        } while (s != s2);
    }

    return line;
}

// Runtime between squares computation
inline Bitboard between_bb(Square s1, Square s2) {
    Bitboard b = line_bb(s1, s2);
    return b & ~(square_bb(s1) | square_bb(s2));
}

// Bishop attacks (uses runtime line computation)
inline Bitboard bishop_attacks_bb(Square s) {
    Bitboard attacks = BB_EMPTY;

    // Scan along diagonals
    for (int d : {-1, 1}) {
        for (int dr : {-1, 1}) {
            Square sq = s;
            while (true) {
                int f = file_of(sq) + d;
                int r = rank_of(sq) + dr;
                if (f < 0 || f > 7 || r < 0 || r > 7) break;
                sq = make_square(File(f), Rank(r));
                attacks |= square_bb(sq);
            }
        }
    }

    return attacks;
}

// Rook attacks
inline Bitboard rook_attacks_bb(Square s) {
    return rank_bb(rank_of(s)) | file_bb(file_of(s));
}

// Queen attacks
inline Bitboard queen_attacks_bb(Square s) {
    return rook_attacks_bb(s) | bishop_attacks_bb(s);
}

// Check if two squares are aligned
inline bool aligned(Square s1, Square s2, Square s3) {
    return line_bb(s1, s2) & square_bb(s3);
}

// Distance between squares
constexpr int distance(Square s1, Square s2) {
    int f = std::abs(int(file_of(s1)) - int(file_of(s2)));
    int r = std::abs(int(rank_of(s1)) - int(rank_of(s2)));
    return std::max(f, r);
}

// Check if more than one bit is set
constexpr bool more_than_one(Bitboard b) {
    return b & (b - 1);
}

} // namespace luminex
