#pragma once

#include <cstdint>
#include <iostream>
#include <cstdlib>

namespace luminex {

// Basic types
using Key = uint64_t;
using Value = int16_t;
using Depth = int;
using Direction = int;

// Forward declarations
class Move;

// Constants
constexpr int MAX_PLY = 246;
constexpr int MAX_MOVES = 256;

// Move generation types
enum GenType : int {
    GEN_LEGAL,
    GEN_CAPTURE,
    GEN_QUIET,
    GEN_QUIET_CHECK,
    GEN_EVASION,
    GEN_NON_EVASION,
    GEN_ALL
};

// Piece types
enum PieceType : uint8_t {
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,
    PT_NONE = 6,
    ALL_PIECES = 7
};

// Colors
enum Color : uint8_t {
    WHITE,
    BLACK,
    NO_COLOR = 2
};

// Castling rights
enum CastlingRight : uint8_t {
    WHITE_KINGSIDE  = 1 << 0,
    WHITE_QUEENSIDE = 1 << 1,
    BLACK_KINGSIDE  = 1 << 2,
    BLACK_QUEENSIDE = 1 << 3
};

// Bound type for transposition table
enum Bound : uint8_t {
    BOUND_NONE,
    BOUND_LOWER,
    BOUND_UPPER,
    BOUND_EXACT = BOUND_LOWER | BOUND_UPPER
};

// File and Rank
enum File : uint8_t { FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_NONE = 8 };
enum Rank : uint8_t { RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_NONE = 8 };

// Square representation
enum Square : uint8_t {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    SQUARE_NONE = 64
};

// Piece encoding
enum Piece : uint8_t {
    WHITE_PAWN = 0, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING,
    BLACK_PAWN = 6, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING,
    NO_PIECE = 12
};

inline Piece make_piece(Color c, PieceType pt) { return static_cast<Piece>((c * 6) + pt); }
inline PieceType piece_type_of(Piece p) { return p == NO_PIECE ? PT_NONE : static_cast<PieceType>(p % 6); }
inline Color color_of_piece(Piece p) { return p == NO_PIECE ? NO_COLOR : static_cast<Color>(p / 6); }

// Move flags
enum MoveFlag : uint16_t {
    MF_QUIET = 0,
    MF_DOUBLE_PAWN = 1 << 0,
    MF_CASTLING_KING = 1 << 1,
    MF_CASTLING_QUEEN = 1 << 2,
    MF_CAPTURE = 1 << 3,
    MF_EN_PASSANT = 1 << 4,
    MF_KNIGHT_PROMO = 1 << 5,
    MF_BISHOP_PROMO = 1 << 6,
    MF_ROOK_PROMO = 1 << 7,
    MF_QUEEN_PROMO = 1 << 8,
    MF_PROMOTION = MF_KNIGHT_PROMO | MF_BISHOP_PROMO | MF_ROOK_PROMO | MF_QUEEN_PROMO
};

// Move representation (16 bits)
class Move {
    uint16_t data_;

public:
    constexpr Move() : data_(0) {}
    constexpr Move(uint16_t data) : data_(data) {}
    constexpr Move(Square from, Square to, uint16_t flags = MF_QUIET)
        : data_(static_cast<uint16_t>(from) << 6 | static_cast<uint16_t>(to) | flags) {}

    constexpr Square from() const { return static_cast<Square>((data_ >> 6) & 0x3F); }
    constexpr Square to() const { return static_cast<Square>(data_ & 0x3F); }
    constexpr uint16_t flags() const { return data_ & 0xF000; }
    constexpr uint16_t raw() const { return data_; }

    constexpr bool is_capture() const { return flags() & MF_CAPTURE; }
    constexpr bool is_promotion() const { return flags() & MF_PROMOTION; }
    constexpr bool is_castling() const { return flags() & (MF_CASTLING_KING | MF_CASTLING_QUEEN); }
    constexpr bool is_en_passant() const { return flags() & MF_EN_PASSANT; }

    constexpr PieceType promotion_type() const {
        if (flags() & MF_KNIGHT_PROMO) return KNIGHT;
        if (flags() & MF_BISHOP_PROMO) return BISHOP;
        if (flags() & MF_ROOK_PROMO) return ROOK;
        if (flags() & MF_QUEEN_PROMO) return QUEEN;
        return PT_NONE;
    }

    constexpr bool operator==(const Move& other) const { return data_ == other.data_; }
    constexpr bool operator!=(const Move& other) const { return data_ != other.data_; }
    constexpr explicit operator bool() const { return data_ != 0; }
};

static_assert(sizeof(Move) == 2, "Move should be 2 bytes");

constexpr Move MOVE_NONE;

// Depth constants
constexpr Depth DEPTH_ZERO = 0;
constexpr Depth DEPTH_MAX = MAX_PLY;
constexpr Depth DEPTH_QS = -1;
constexpr Depth DEPTH_QS_CHECKS = -2;

// Value constants
constexpr Value VALUE_ZERO = 0;
constexpr Value VALUE_DRAW = 0;
constexpr Value VALUE_INFINITE = 30000;
constexpr Value VALUE_MATE = 29000;
constexpr Value VALUE_MATE_IN_MAX_PLY = VALUE_MATE - 256;
constexpr Value VALUE_KNOWN_WIN = 15000;

// Piece values for evaluation
constexpr Value PAWN_VALUE   = 100;
constexpr Value KNIGHT_VALUE = 320;
constexpr Value BISHOP_VALUE = 330;
constexpr Value ROOK_VALUE   = 500;
constexpr Value QUEEN_VALUE  = 900;

// Piece-square table values
constexpr Value PST_MG = 0;
constexpr Value PST_EG = 1;

// Utility functions
constexpr Color operator~(Color c) { return static_cast<Color>(c ^ 1); }
constexpr Square make_square(File f, Rank r) { return static_cast<Square>(r * 8 + f); }
constexpr File file_of(Square s) { return static_cast<File>(s % 8); }
constexpr Rank rank_of(Square s) { return static_cast<Rank>(s / 8); }
constexpr bool is_ok(Square s) { return s < SQUARE_NONE; }

constexpr Square relative_square(Color c, Square s) {
    return static_cast<Square>(s ^ (c * 56));
}

constexpr Rank relative_rank(Color c, Rank r) {
    return static_cast<Rank>(r ^ (c * 7));
}

constexpr Rank relative_rank(Color c, Square s) {
    return relative_rank(c, rank_of(s));
}

constexpr Square flip_rank(Square s) {
    return static_cast<Square>(s ^ 56);
}

constexpr Square flip_file(Square s) {
    return static_cast<Square>(s ^ 7);
}

inline char piece_char(Color c, PieceType pt) {
    constexpr char chars[2][7] = {
        {'P', 'N', 'B', 'R', 'Q', 'K', ' '},
        {'p', 'n', 'b', 'r', 'q', 'k', ' '}
    };
    return chars[c][pt];
}

inline char file_char(File f) { return 'a' + f; }
inline char rank_char(Rank r) { return '1' + r; }

inline std::ostream& operator<<(std::ostream& os, Square s) {
    if (is_ok(s)) os << file_char(file_of(s)) << rank_char(rank_of(s));
    else os << "-";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, Move m) {
    if (m) os << m.from() << m.to();
    else os << "0000";
    return os;
}

using Score = Value;
using Sign = int;

} // namespace luminex
