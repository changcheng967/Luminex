// luminex.h — the Luminex chess engine's one header: every declaration its source files share, each part after
// the ones it needs. The network's SIMD kernels are apart, in nnue_kernels.h: they are compiled once per
// instruction-set level, so they include none of this. CMake reads the version, ENGINE_VERSION, from the end.
#pragma once

// The Microsoft C runtime marks fopen, fscanf and getenv deprecated in favour of its own _s variants, which no
// other platform has. CMake defines this for every Windows target; it is here too for the builds and editors
// that do not take CMake's definitions. Every source file includes this header first.
#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

// =====================================================================================================================
// Types
// =====================================================================================================================

#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <utility>  // std::to_underlying, std::unreachable (C++23)
#include <format>   // std::format, std::formatter (C++20/23)
#include <string>   // std::string (to_uci_string + std::formatter<Move>)

namespace luminex {

// Basic types
using Key = uint64_t;
using Value = int32_t;  // Changed from int16_t to prevent overflow
using Depth = int;
using Direction = int;

// Forward declarations
class Move;

// Constants
constexpr int MAX_PLY = 246;
constexpr int MAX_MOVES = 512;

// Move generation types
enum GenType : int {
    GEN_LEGAL,
    GEN_CAPTURE,
    GEN_QUIET,
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
    PT_NONE = 255,  // Changed from 6 to avoid conflict with piece values
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

inline Piece make_piece(Color c, PieceType pt) { return static_cast<Piece>(std::to_underlying(c) * 6 + std::to_underlying(pt)); }
inline PieceType piece_type_of(Piece p) { return p == NO_PIECE ? PT_NONE : static_cast<PieceType>(std::to_underlying(p) % 6); }
inline Color color_of_piece(Piece p) { return p == NO_PIECE ? NO_COLOR : static_cast<Color>(std::to_underlying(p) / 6); }

// Move flags (16-bit move encoding: from(6) to(6) flags(4))
// Bits 0-5: destination square
// Bits 6-11: source square
// Bits 12-15: flags (direct value, not bitmask)
enum MoveFlag : uint16_t {
    // Bits 15:14 layout
    // 00xx: special quiet moves (00, 01, 02, 03)
    // 01xx: captures (04, 05)
    // 10xx: promotions (06, 07, 08, 09)
    // 11xx: capture + promotions (10, 11, 12, 13)

    MF_QUIET = 0x0000,
    MF_DOUBLE_PAWN = 0x1000,
    MF_CASTLING_KING = 0x2000,
    MF_CASTLING_QUEEN = 0x3000,

    MF_CAPTURE = 0x4000,
    MF_EN_PASSANT = 0x5000,

    MF_PROMO_KNIGHT = 0x8000,
    MF_PROMO_BISHOP = 0x9000,
    MF_PROMO_ROOK = 0xA000,
    MF_PROMO_QUEEN = 0xB000,

    MF_CAPTURE_PROMO_KNIGHT = 0xC000,
    MF_CAPTURE_PROMO_BISHOP = 0xD000,
    MF_CAPTURE_PROMO_ROOK = 0xE000,
    MF_CAPTURE_PROMO_QUEEN = 0xF000
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

    static constexpr Move promotion(Square from, Square to, PieceType pt, bool isCapture = false) {
        uint16_t promo_flag = isCapture ? 0xC000 : 0x8000;  // Base for knight promo
        if (pt == KNIGHT) promo_flag += 0;
        else if (pt == BISHOP) promo_flag += 0x1000;
        else if (pt == ROOK) promo_flag += 0x2000;
        else if (pt == QUEEN) promo_flag += 0x3000;
        return Move(from, to, promo_flag);
    }

    constexpr bool is_capture() const {
        uint16_t f = flags();
        return (f & 0xC000) == 0x4000 || (f & 0xC000) == 0xC000;  // 0x4xxx or 0xCxxx-Fxxx
    }

    constexpr bool is_promotion() const {
        return (flags() & 0xC000) >= 0x8000;  // 0x8xxx to 0xFxxx
    }

    constexpr bool is_castling() const {
        uint16_t f = flags();
        return f == MF_CASTLING_KING || f == MF_CASTLING_QUEEN;
    }

    constexpr bool is_en_passant() const { return flags() == MF_EN_PASSANT; }

    constexpr PieceType promotion_type() const {
        uint16_t f = flags();
        if ((f & 0xC000) < 0x8000) return PT_NONE;  // Not a promotion
        uint16_t low_bits = f & 0x3000;  // Bits 12-13
        switch (low_bits) {
            case 0x0000: return KNIGHT;
            case 0x1000: return BISHOP;
            case 0x2000: return ROOK;
            case 0x3000: return QUEEN;
        }
        std::unreachable();  // low_bits is masked to 2 bits → the four cases above are exhaustive
    }

    friend constexpr bool operator==(const Move&, const Move&) = default;
    constexpr explicit operator bool() const { return data_ != 0; }
};

static_assert(sizeof(Move) == 2, "Move should be 2 bytes");

constexpr Move MOVE_NONE;
constexpr Move MOVE_NULL{0x1F};  // Sentinel distinct from MOVE_NONE (0) for null moves

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
constexpr Value KNIGHT_VALUE = 330;
constexpr Value BISHOP_VALUE = 330;
constexpr Value ROOK_VALUE   = 500;
constexpr Value QUEEN_VALUE  = 1000;

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

// UCI move string (e.g. "e2e4", "e7e8q", "0000" for none). Shared by operator<<
// and the std::formatter<luminex::Move> specialization defined below.
inline std::string to_uci_string(Move m) {
    if (!m) return "0000";
    std::string s;
    s += file_char(file_of(m.from()));
    s += rank_char(rank_of(m.from()));
    s += file_char(file_of(m.to()));
    s += rank_char(rank_of(m.to()));
    if (m.is_promotion()) {
        constexpr char promo_chars[] = {'n', 'b', 'r', 'q'};
        s += promo_chars[(m.flags() & 0x3000) >> 12];
    }
    return s;
}

inline std::ostream& operator<<(std::ostream& os, Move m) {
    return os << to_uci_string(m);
}

using Score = Value;
using Sign = int;

} // namespace luminex

// C++20/23: enable std::format / std::print with luminex::Move (formats as the UCI string).
// Must live in namespace std and be visible before any std::format use of Move.
namespace std {
template <>
struct formatter<luminex::Move> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
    auto format(const luminex::Move& m, format_context& ctx) const {
        return std::format_to(ctx.out(), "{}", luminex::to_uci_string(m));
    }
};
} // namespace std

// =====================================================================================================================
// Bitboards
// =====================================================================================================================

#include <algorithm>
#include <array>
#include <bit>      // std::countr_zero, std::countl_zero, std::popcount (C++20)
#include <cstdint>
#include <utility>  // std::to_underlying (C++23)

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
constexpr Bitboard square_bb(Square s) {
    return (s >= 0 && s < 64) ? BB_SQUARES[s] : BB_EMPTY;
}
// Bit scans and counts from <bit>: constexpr on every compiler, and one
// instruction each with BMI1 and POPCNT (tzcnt, lzcnt, popcnt; x86-64-v3).
constexpr Square lsb(Bitboard b) { return static_cast<Square>(std::countr_zero(b)); }
constexpr Square msb(Bitboard b) { return static_cast<Square>(63 - std::countl_zero(b)); }
constexpr int popcount(Bitboard b) { return std::popcount(b); }
constexpr Bitboard square_bb(File f, Rank r) { return BB_SQUARES[make_square(f, r)]; }
constexpr Bitboard rank_bb(Rank r) { return BB_RANKS[r]; }
constexpr Bitboard file_bb(File f) { return BB_FILES[f]; }

inline Square pop_lsb(Bitboard& b) {
    [[assume(b != 0)]];  // every caller invokes pop_lsb only on a non-empty bitboard
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

// The squares strictly between two squares on one rank, file or diagonal
// (empty for any other pair), built at compile time: there is no init call to
// forget (a featurizer that skipped one generated illegal moves on pins).
inline constexpr std::array<std::array<Bitboard, 64>, 64> BetweenBB = [] {
    std::array<std::array<Bitboard, 64>, 64> t{};
    for (int s1 = 0; s1 < 64; ++s1)
        for (int s2 = 0; s2 < 64; ++s2) {
            const int df = s2 % 8 - s1 % 8, dr = s2 / 8 - s1 / 8;
            const int adf = df < 0 ? -df : df, adr = dr < 0 ? -dr : dr;
            if ((df == 0 && dr == 0) || (df != 0 && dr != 0 && adf != adr)) continue;
            const int step = ((dr > 0) - (dr < 0)) * 8 + ((df > 0) - (df < 0));
            Bitboard b = 0;
            for (int s = s1 + step; s != s2; s += step) b |= Bitboard(1) << s;
            t[s1][s2] = b;
        }
    return t;
}();

constexpr Bitboard between_bb(Square s1, Square s2) {
    return BetweenBB[std::to_underlying(s1)][std::to_underlying(s2)];
}

// Distance between squares
inline int distance(Square s1, Square s2) {
    int f = std::abs(int(file_of(s1)) - int(file_of(s2)));
    int r = std::abs(int(rank_of(s1)) - int(rank_of(s2)));
    return (f > r) ? f : r;
}

// Check if more than one bit is set
constexpr bool more_than_one(Bitboard b) {
    return b & (b - 1);
}

// ============================================================
// Magic Bitboard tables for sliding piece attacks
// ============================================================

struct Magic {
    Bitboard mask;
    Bitboard magic;
    Bitboard* attacks;
    unsigned shift;
};

extern Magic RookMagics[64];
extern Magic BishopMagics[64];
extern Bitboard RookTable[262144];   // Large enough for all rook entries
extern Bitboard BishopTable[65536];  // Large enough for all bishop entries

// Magic bitboard attack lookups - O(1) table lookup
inline Bitboard rook_attacks_bb(Square s, Bitboard occupied) {
    const Magic& m = RookMagics[int(s)];
    return m.attacks[((occupied & m.mask) * m.magic) >> m.shift];
}

inline Bitboard bishop_attacks_bb(Square s, Bitboard occupied) {
    const Magic& m = BishopMagics[int(s)];
    return m.attacks[((occupied & m.mask) * m.magic) >> m.shift];
}

inline Bitboard rook_attacks_bb(Square s) {
    return rank_bb(rank_of(s)) | file_bb(file_of(s));
}

inline Bitboard bishop_attacks_bb(Square s) {
    Bitboard attacks = BB_EMPTY;
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

inline Bitboard queen_attacks_bb(Square s, Bitboard occupied) {
    return rook_attacks_bb(s, occupied) | bishop_attacks_bb(s, occupied);
}

inline Bitboard queen_attacks_bb(Square s) {
    return rook_attacks_bb(s) | bishop_attacks_bb(s);
}

// Legacy compatibility aliases
inline Bitboard bb_rank_attacks(Square s, Bitboard occupied) {
    return rook_attacks_bb(s, occupied);
}

inline Bitboard bb_file_attacks(Square s, Bitboard occupied) {
    return rook_attacks_bb(s, occupied);
}

inline Bitboard bb_diag_attacks(Square s, Bitboard occupied) {
    return bishop_attacks_bb(s, occupied);
}

// Initialize magic bitboard tables - call once at startup
void init_magic_bitboards();

} // namespace luminex

// =====================================================================================================================
// The position
// =====================================================================================================================

#include <string>

namespace luminex {

// Position state for undo
struct StateInfo {
    Key key = 0;
    Key pawn_key = 0;
    Key nonpawn_key[2] = {0, 0};  // Non-pawn hash per color [WHITE, BLACK]
    Key minor_key = 0;            // Minor piece hash (knights + bishops)
    Key major_key = 0;            // Major piece hash (rooks + queens)
    int psq_mg = 0;  // Incremental material+PST middlegame score (white-relative)
    int psq_eg = 0;  // Incremental material+PST endgame score (white-relative)
    Bitboard checkers = 0;
    Bitboard pinned = 0;
    Bitboard block_checkers = 0;
    Square ep_square = SQUARE_NONE;
    int castling_rights = 0;
    int ply = 0;
    Move move = MOVE_NONE;
    PieceType captured_piece = PT_NONE;
    bool move_was_executed = true;  // Tracks if do_move actually executed the move (vs. returning early)
    int halfmove_clock = 0;  // Half-moves since last pawn push or capture (for 50-move rule)
};

class Position {
public:
    Position() : st_(&dummy_state), st_ply(0) {
        // CRITICAL: Initialize board to NO_PIECE, not 0 (which is WHITE_PAWN)
        for (int i = 0; i < SQUARE_NONE; ++i) {
            board[i] = NO_PIECE;
            index[i] = -1;  // -1 means "not in piece_list"
        }
        // Initialize all other state properly
        for (int i = 0; i < ALL_PIECES; ++i) {
            pieces_by_type[i] = 0;
        }
        for (int i = 0; i < NO_COLOR; ++i) {
            pieces_by_color[i] = 0;
            king_square[i] = SQUARE_NONE;
            castling_rights_[i] = 0;
            castling_rook_square_[i][0] = SQUARE_NONE;
            castling_rook_square_[i][1] = SQUARE_NONE;
            castling_path_[i][0] = 0;
            castling_path_[i][1] = 0;
            for (int j = 0; j < ALL_PIECES; ++j) {
                piece_count[i][j] = 0;
                for (int k = 0; k < 16; ++k) {
                    piece_list[i][j][k] = SQUARE_NONE;
                }
            }
        }
        side_to_move_ = WHITE;
        game_ply_ = 0;
        dummy_state.key = 0;
        dummy_state.checkers = 0;
        dummy_state.pinned = 0;
        dummy_state.block_checkers = 0;
        dummy_state.ep_square = SQUARE_NONE;
        dummy_state.castling_rights = 0;
        dummy_state.ply = 0;
    }

    void set(const std::string& fen);
    std::string fen() const;

    Bitboard pieces() const;
    Bitboard pieces(Color c) const;
    Bitboard pieces(PieceType pt) const;
    Bitboard pieces(Color c, PieceType pt) const;
    Bitboard pieces(Color c, PieceType pt1, PieceType pt2) const;

    Piece piece_on(Square s) const;
    PieceType piece_type_on(Square s) const;
    Color color_of_piece(Piece p) const;
    Color side_to_move() const;

    Bitboard checkers() const;
    Bitboard pinned() const;

    Square ep_square() const;
    Square king_sq(Color c) const;
    int state_ply() const { return st_ply; }   // current state-stack index (NNUE indexes its accumulator by this)
    const Piece* mailbox() const { return board; }   // the 64 squares' pieces (NNUE's relation rows read it)
    bool castling_allowed(Color c, CastlingRight cr) const;

    int game_ply() const;
    int halfmove_clock() const;

    Key key() const;
    Key pawn_key() const;
    Key nonpawn_key(Color c) const;
    Key minor_key() const;
    Key major_key() const;
    int psq_mg() const;
    int psq_eg() const;

    bool is_draw() const;
    // The search root's place in the history: a repetition of a position
    // from before the search needs a second earlier occurrence (is_draw)
    void mark_root() { root_history_size_ = history_size; }
    bool is_check() const;

    bool do_move(Move m);  // Returns true if move executed, false if failed
    bool do_move_replay(Move m);  // Lean forward-only board update for the featurizer (no StateInfo/keys/check info)
    void undo_move(Move m);

    void do_null_move();
    void undo_null_move();

    bool see_ge(Move m, Value threshold = VALUE_ZERO) const;
    Value exchange(Square to, Color stm, Bitboard occ, PieceType captured) const;

    bool legal(Move m, bool skip_pseudo = false) const;    bool pseudo_legal(const Move m) const;
    bool capture(Move m) const;
    bool capture_or_promotion(Move m) const;

    Direction pawn_push(Color c) const;
    void assert_consistency(const char* location);  // Debug: check board consistency (made public for uci.cpp)

    Bitboard attackers_to(Square s) const;
    Bitboard attackers_to(Square s, Bitboard occupied) const;

    static constexpr int MAX_MOVES = 256;
    static constexpr int MAX_PLY = 246;

private:
    void put_piece(Color c, PieceType pt, Square s);
    void remove_piece(Square s);
    void move_piece(Square from, Square to);

    void set_check_info(StateInfo* st);
    bool validate_move(Move m, Square from, Square to, Piece moved_pc, PieceType captured) const;

    Bitboard slider_blockers(Bitboard sliders, Bitboard& pinners) const;

    Bitboard pieces_by_type[ALL_PIECES] = {};
    Bitboard pieces_by_color[NO_COLOR] = {};
    Piece board[SQUARE_NONE];  // Initialized in constructor to NO_PIECE
    int piece_count[NO_COLOR][ALL_PIECES] = {};
    Square piece_list[NO_COLOR][ALL_PIECES][16] = {};
    int index[SQUARE_NONE] = {};

    Square king_square[NO_COLOR] = {};

    Color side_to_move_ = WHITE;
    StateInfo* st_ = nullptr;
    int game_ply_ = 0;
    int st_ply = 0;  // Index into state_stack

    static constexpr int MAX_STATES = 2048;  // Must support long games (not just search depth)
    StateInfo state_stack[MAX_STATES] = {};
    StateInfo dummy_state;  // For default initialization

    int castling_rights_[NO_COLOR] = {};
    Square castling_rook_square_[NO_COLOR][2] = {};
    Bitboard castling_path_[NO_COLOR][2] = {};

    // Position history for repetition detection
    // As deep as the state stack: a game plus its search can never outgrow it, so
    // every do_move records its key and every undo_move removes one (at 1024, a
    // game past ~775 plies stopped recording while undo_move kept removing).
    static constexpr int MAX_HISTORY = MAX_STATES;
    Key position_history[MAX_HISTORY];
    int history_size = 0;
    int root_history_size_ = 0;
};

inline Bitboard Position::pieces() const { return pieces_by_color[WHITE] | pieces_by_color[BLACK]; }
inline Bitboard Position::pieces(Color c) const { return pieces_by_color[c]; }
inline Bitboard Position::pieces(PieceType pt) const { return pieces_by_type[pt]; }
inline Bitboard Position::pieces(Color c, PieceType pt) const { return pieces_by_color[c] & pieces_by_type[pt]; }
inline Bitboard Position::pieces(Color c, PieceType pt1, PieceType pt2) const { return pieces_by_color[c] & (pieces_by_type[pt1] | pieces_by_type[pt2]); }

inline Piece Position::piece_on(Square s) const { return is_ok(s) ? board[s] : NO_PIECE; }
inline PieceType Position::piece_type_on(Square s) const { return is_ok(s) ? piece_type_of(board[s]) : PT_NONE; }
inline Color Position::color_of_piece(Piece p) const { return p == NO_PIECE ? NO_COLOR : static_cast<Color>(p / 6); }
inline Color Position::side_to_move() const { return side_to_move_; }

inline Bitboard Position::checkers() const { return st_->checkers; }
inline Bitboard Position::pinned() const { return st_->pinned; }

inline Square Position::ep_square() const { return st_->ep_square; }
inline Square Position::king_sq(Color c) const { return king_square[c]; }
inline bool Position::castling_allowed(Color c, CastlingRight cr) const { return castling_rights_[c] & cr; }

inline int Position::game_ply() const { return game_ply_; }

inline int Position::halfmove_clock() const { return st_->halfmove_clock; }

inline Key Position::key() const { return st_->key; }
inline Key Position::pawn_key() const { return st_->pawn_key; }
inline Key Position::nonpawn_key(Color c) const { return st_->nonpawn_key[int(c)]; }
inline Key Position::minor_key() const { return st_->minor_key; }
inline Key Position::major_key() const { return st_->major_key; }
inline int Position::psq_mg() const { return st_->psq_mg; }
inline int Position::psq_eg() const { return st_->psq_eg; }

inline bool Position::is_check() const { return checkers() != 0; }

// Zobrist hash keys
namespace Zobrist {
    // Match sizes in board.cpp implementation
    extern Key psq[2][8][64];     // piece-square keys [color][pieceType][square]
    extern Key castling[4];       // castling rights keys
    extern Key en_passant[8];     // en passant file keys
    extern Key side;              // side to move key

    void init();  // Initialize Zobrist keys with random values
}

// Zobrist initialization
void init_zobrist();

// Direction constants for pawn push
enum : Direction {
    NORTH = 8,
    EAST = 1,
    SOUTH = -8,
    WEST = -1,
    NORTH_EAST = NORTH + EAST,
    NORTH_WEST = NORTH + WEST,
    SOUTH_EAST = SOUTH + EAST,
    SOUTH_WEST = SOUTH + WEST
};

} // namespace luminex

// =====================================================================================================================
// Move generation
// =====================================================================================================================

namespace luminex {

struct ExtMove {
    Move move;
    Value value;

    constexpr operator Move() const { return move; }
    void operator=(Move m) { move = m; }

    // For sorting
    bool operator<(const ExtMove& other) const { return value > other.value; }
    bool operator>(const ExtMove& other) const { return value < other.value; }
};

class Position;

template<GenType T>
ExtMove* generate(const Position& pos, ExtMove* moveList);

inline ExtMove* generate_legals(const Position& pos, ExtMove* moveList) {
    return generate<GEN_LEGAL>(pos, moveList);
}

// Move ordering values (indexed by PieceType: PAWN=0, KNIGHT=1, ...)
inline Value mvv_lva(PieceType victim, PieceType attacker) {
    constexpr Value victim_score[7] = {100, 200, 300, 400, 500, 600, 0};
    constexpr Value attacker_score[7] = {6, 5, 4, 3, 2, 1, 0};
    return victim_score[victim] * 8 + attacker_score[attacker];
}

} // namespace luminex

// =====================================================================================================================
// Large pages
// =====================================================================================================================

// Memory for the big tables the search reads at random: the
// transposition table and the network's feature weights.
//
// Their pages are 2 MiB where the system allows. A random read into a table
// of tens of megabytes otherwise lands on a 4 KiB page whose address
// translation is rarely cached (the second-level TLB holds a few thousand
// pages: 8 to 16 MB of 4 KiB ones), and the read waits for a page walk. On
// Linux the memory is aligned to 2 MiB and advised MADV_HUGEPAGE before its
// first write, so the faults of that write take transparent huge pages when
// THP is "always" or "madvise" (Ubuntu's default, under which nothing else
// gets them). Elsewhere, or where the system declines, the pages stay 4 KiB
// and nothing else changes.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <type_traits>
#if defined(__linux__)
#include <sys/mman.h>
#endif
#if defined(_WIN32)
#include <malloc.h>
#endif

namespace luminex {

inline constexpr std::size_t LARGE_PAGE = std::size_t(2) << 20;

// The engine builds without exceptions, so a failed allocation ends it, as
// std::vector's did.
[[noreturn]] inline void large_alloc_failed(std::size_t bytes) {
    std::fprintf(stderr, "luminex: cannot allocate %zu MB\n", (bytes + (1u << 20) - 1) >> 20);
    std::abort();
}

// bytes of zeroed memory, aligned to LARGE_PAGE and rounded up to whole large
// pages (on Windows aligned to 64 bytes)
inline void* large_alloc(std::size_t bytes) {
    void* p = nullptr;
#if defined(_WIN32)
    p = _aligned_malloc(bytes, 64);
    if (!p) large_alloc_failed(bytes);
#else
    bytes = (bytes + LARGE_PAGE - 1) / LARGE_PAGE * LARGE_PAGE;
    if (posix_memalign(&p, LARGE_PAGE, bytes) != 0) large_alloc_failed(bytes);
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    madvise(p, bytes, MADV_HUGEPAGE);   // advice: refused, the pages stay 4 KiB
#endif
#endif
    std::memset(p, 0, bytes);
    return p;
}

inline void large_free(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

// n zeroed elements of T in large_alloc memory, not copyable: the part of
// std::vector these tables used, with reset() for resize().
template <class T>
class LargeArray {
    static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>,
                  "the elements are zeroed bytes, never constructed or destroyed");
public:
    LargeArray() = default;
    ~LargeArray() { large_free(p_); }
    LargeArray(const LargeArray&) = delete;
    LargeArray& operator=(const LargeArray&) = delete;

    // n zero elements in place of the old ones, which are freed first; none for n = 0
    void reset(std::size_t n) {
        large_free(p_);
        p_ = nullptr;
        n_ = 0;
        if (n) {
            p_ = static_cast<T*>(large_alloc(n * sizeof(T)));
            n_ = n;
        }
    }
    std::size_t size() const { return n_; }
    bool empty() const { return n_ == 0; }
    T* data() { return p_; }
    const T* data() const { return p_; }
    T& operator[](std::size_t i) { return p_[i]; }
    const T& operator[](std::size_t i) const { return p_[i]; }

private:
    T* p_ = nullptr;
    std::size_t n_ = 0;
};

// An allocator for std::vector whose data starts on a cache line: the
// network's SIMD kernels read the dense weights 32 or 64 bytes at a time,
// and a read that straddles two lines costs two. std::vector's own allocator
// guarantees 16 bytes.
template <class T>
struct CacheAligned {
    using value_type = T;
    static constexpr std::size_t ALIGN = 64;
    CacheAligned() = default;
    template <class U> CacheAligned(const CacheAligned<U>&) noexcept {}
    T* allocate(std::size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(ALIGN))); }
    void deallocate(T* p, std::size_t) noexcept { ::operator delete(p, std::align_val_t(ALIGN)); }
    template <class U> bool operator==(const CacheAligned<U>&) const noexcept { return true; }
    template <class U> bool operator!=(const CacheAligned<U>&) const noexcept { return false; }
};

} // namespace luminex

// =====================================================================================================================
// The transposition table
// =====================================================================================================================

#include <bit>
#include <cstdint>
#include <cstring>
#include <vector>
#if defined(_MSC_VER) && !defined(__SIZEOF_INT128__)
#include <intrin.h>
#endif

namespace luminex {

// The high 64 bits of the 128-bit product a * b.
inline uint64_t mul_hi64(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
    return uint64_t((unsigned __int128)a * b >> 64);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
    return __umulh(a, b);
#else
    // Four 32 x 32 products: the middle column's carry into the high half is
    // the top of the sum of its three 32-bit parts.
    const uint64_t al = uint32_t(a), ah = a >> 32, bl = uint32_t(b), bh = b >> 32;
    const uint64_t ll = al * bl, hl = ah * bl, lh = al * bh;
    return ah * bh + (hl >> 32) + (lh >> 32) + (((ll >> 32) + uint32_t(hl) + uint32_t(lh)) >> 32);
#endif
}

// Transposition table entry
struct TTEntry {
    uint32_t key32;
    uint16_t move16;
    int32_t value_;  // Changed from int16_t to prevent truncation (Value is int32_t)
    int32_t eval_;   // Changed from int16_t to prevent truncation
    uint8_t depth_;
    uint8_t gen_bound;
    uint8_t pv_flag;    // 1 if this was a PV node, 0 if not
    uint8_t search_id_; // which search wrote this entry (cross-search aging)

    Move move() const { return Move(move16); }
    Depth depth() const { return Depth(depth_ - 127); }
    Bound bound() const { return Bound(gen_bound & 0x3); }
    bool is_pv() const { return pv_flag != 0; }
    Value value() const { return Value(value_); }
    Value eval() const { return Value(eval_); }
    uint8_t search_age(uint8_t now) const { return uint8_t(now - search_id_); }

    void save(uint64_t k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t g);
};

// Transposition table cluster (cache line size = 64 bytes)
// Entry size is 20 bytes (3 * 20 = 60), padding = 4 bytes
struct TTCluster {
    TTEntry entry[3];
    char padding[4];
};

static_assert(sizeof(TTCluster) == 64, "TTCluster size must be 64 bytes");

// Transposition table
class TranspositionTable {
public:
    TranspositionTable() : table(), entries(0) {}

    ~TranspositionTable() { resize(0); }

    void resize(size_t mb);
    void clear();
    void new_search();
    // The table is allocated on first use (isready or a search) rather than
    // at startup, so a GUI's Hash setting is the only allocation: zeroing a
    // default 256 MB that is replaced at once cost the start most of a second.
    void ensure_allocated() { if (table.empty()) resize(DEFAULT_MB); }
    static constexpr size_t DEFAULT_MB = 256;

    TTEntry* probe(uint64_t key, bool& found);
    void prefetch(uint64_t key) const;

    // Store policy for probe-returned entries: always store on a miss (the
    // replacement victim was already chosen by probe); on a same-key hit keep
    // the old entry unless this search is at least as deep or the entry is
    // from an older search.
    bool should_store(const TTEntry* tte, bool found, Depth d) const {
        return !found || d >= tte->depth() || (tte->gen_bound & 0xFC) != generation8;
    }

    uint8_t generation() const { return generation8; }
    uint8_t search_id() const { return search_id8; }
    void begin_search() { ++search_id8; }  // once per "go": entry ages by 1/search
    size_t hashfull() const;

private:
    // The cluster that holds key, in a table of any size N. A power of two takes
    // the key's low bits, as the table always has. Any other N takes the high
    // half of rotl(key, 64 - k) * N, 2^k being the next power of two: the
    // rotation puts those same k low bits on top with the key's high bits below
    // them, so the product spreads keys evenly over all N clusters (and equals
    // key & (N - 1) when N is 2^k). key32 checks the key's high 32 bits: the
    // power-of-two index never reads them, the other reads them only to put a
    // key on one side of a boundary between two clusters, which keeps a false
    // match at 1 in 2^32 or rarer. (Past 2^32 clusters, Hash 262,144, index and
    // key32 share bits either way.)
    size_t cluster_of(uint64_t key) const {
        return pow2_ ? size_t(key) & (table.size() - 1)
                     : size_t(mul_hi64(std::rotl(key, rot_), table.size()));
    }

    LargeArray<TTCluster> table;   // 2 MiB pages where the system allows (LargeArray)
    size_t entries;
    int rot_ = 0;                  // 64 - k
    bool pow2_ = true;             // N is a power of two
    uint8_t generation8 = 0;
    uint8_t search_id8 = 0;
};

inline TranspositionTable TT;

} // namespace luminex

// =====================================================================================================================
// The hand-crafted evaluation
// =====================================================================================================================

namespace luminex {

class Position;

// Piece-square tables (MG and EG phases)
extern Score PST_MG_TABLE[2][8][64];
extern Score PST_EG_TABLE[2][8][64];

// Combined piece-value + PST tables for incremental eval
// PSQ_MG[c][pt][sq] = PieceValueMG[pt] + PST_MG_TABLE[c][pt][sq] (white-positive)
extern int PSQ_MG[2][8][64];
extern int PSQ_EG[2][8][64];

// Evaluation parameters (self-engineered from chess first principles)
struct EvalParams {
    int bishop_pair_mg = 40;
    int bishop_pair_eg = 100;
    int rook_open_mg = 25;
    int rook_open_eg = 40;
    int rook_semi_open_mg = 15;
    int rook_semi_open_eg = 20;
    int rook_7th_mg = 30;
    int rook_7th_eg = 25;
    int pawn_shield_center = 12;
    int pawn_shield_knight = 15;
    int pawn_shield_rook = 8;
    int pawn_storm = 10;
    int open_file_penalty_mg = 20;
    int open_file_penalty_eg = 15;
    int outpost_knight_mg = 30;
    int outpost_knight_eg = 15;
    int outpost_bishop_mg = 40;
    int outpost_bishop_eg = 25;
    int hanging_pawn_mg = 10;
    int hanging_pawn_eg = 35;
    int far_knight_mg = 15;
    int far_knight_eg = 5;
    int far_bishop_mg = 10;
    int far_bishop_eg = 5;
};

extern EvalParams g_eval_params;

// Main evaluation function
Value evaluate(const Position& pos, bool tactical_only = false);

// Endgames the material settles. A bare king on one side: mating material against it is a known win
// (VALUE_KNOWN_WIN plus a gradient toward the mate), K+P vs K comes from the exact bitbase, wrong-bishop rook
// pawns with the king in the corner and no mating material are draws. A queen against one rook, bishop or
// knight gets the same gradient at the net's level. Sets value (side-to-move relative) and returns true
// there, false everywhere else. evaluate() returns this value first, and the search takes it as it is: no
// 50-move damping, no correction.
bool known_endgame(const Position& pos, Value& value);

// Evaluation scaling
int scale_factor(const Position& pos, Value eg);

// Initialize evaluation tables
void init_evaluation();

// Sync mutable eval arrays from g_params (for tuner)
void sync_eval_params();

} // namespace luminex

// =====================================================================================================================
// The evaluation's feature index space
// =====================================================================================================================

// The HCE feature-index space shared by evaluation.cpp, luminex-evaltrace (tools.cpp) and the fitted
// coefficients below (gen_fitted.py generated them).
//
// The index space IS the solver space: MG block 0..606, EG block 607..1213.
// The fitted coefficients (next) define FE_MG / FE_EG holding the CURRENT coefficient
// for every feature (engine defaults until a distillation fit is applied).
// Keeping one numbering across engine + tracer + solver means the --verify
// invariants revalidate the engine against whatever fit is applied.

namespace luminex {
namespace feat {
constexpr int PST            = 0;    // 384: pt*64 + relsq (relsq = sq^56 for black)
constexpr int MAT            = 384;  // 6
constexpr int MOB_N          = 390;  // 9
constexpr int MOB_B          = 399;  // 14
constexpr int MOB_R          = 413;  // 15
constexpr int MOB_Q          = 428;  // 28
constexpr int PAWN_DOUBLED   = 456;
constexpr int PAWN_ISOLATED  = 457;
constexpr int PAWN_CONNECTED = 458;  // +7 (r-1)
constexpr int PAWN_PHALANX   = 465;  // +7 (r-1)
constexpr int PAWN_LEVER     = 472;  // +7 (r-1)
constexpr int PAWN_LEVER_C   = 479;
constexpr int PAWN_BACKWARD  = 480;
constexpr int PAWN_CAND      = 481;  // +14: sup*7 + (r-1)
constexpr int PAWN_PASSED    = 495;  // +7 (r-1)
constexpr int PP_CONNECTED   = 502;  // +7 (r-1), value = n_connected
constexpr int PP_ROOK_BEHIND = 509;
constexpr int PP_ENEMY_ROOK  = 510;
constexpr int PP_KING_PROX   = 511;  // value = (their_kdist - our_kdist)
constexpr int PP_UNREACH_S   = 512;  // value = unreachable delta (<=4)
constexpr int PP_UNREACH_B   = 513;  // value = unreachable delta (>4)
constexpr int PP_BLOCKADE    = 514;  // +7 (r-1)
constexpr int PP_STOP_OWN    = 521;
constexpr int PP_STOP_ATT    = 522;
constexpr int PP_SAFE        = 523;  // +7 (r-1)
constexpr int PP_CLEAR       = 530;  // +7 (r-1)
constexpr int OUTPOST_N      = 537;  // +3 (kr-3)
constexpr int TRAPPED_N      = 540;
constexpr int FAR_N          = 541;
constexpr int FAR_B          = 542;
constexpr int BSC            = 543;  // +7 (same-color pawn count, capped 6)
constexpr int BISH_LONG      = 550;
constexpr int OUTPOST_B      = 551;
constexpr int ROOK_OPEN      = 552;
constexpr int ROOK_SEMI      = 553;
constexpr int ROOK_7TH       = 554;
constexpr int ROOK_XRAY_Q    = 555;
constexpr int FAR_R          = 556;
constexpr int ROOK_TRAP_C    = 557;
constexpr int ROOK_TRAP_NC   = 558;
constexpr int ROOK_BLOCKED   = 559;
constexpr int FAR_Q          = 560;
constexpr int SHIELD_R1      = 561;
constexpr int SHIELD_R2      = 562;
constexpr int SHIELD_R3      = 563;
constexpr int OPEN_KFILE     = 564;
constexpr int OPEN_ADJ       = 565;  // value = # adjacent open files
constexpr int STORM          = 566;  // value = sum(pr-3) over storm pawns
constexpr int CASTLED        = 567;
constexpr int NOCASTLE       = 568;
constexpr int THR            = 569;  // +25: attacker(P..Q)*5 + target(P..Q)
constexpr int HANG_PAWN      = 594;
constexpr int HANG_PIECE     = 595;
constexpr int PINNED         = 596;
constexpr int SPACE_OWN      = 597;
constexpr int SPACE_OPP      = 598;
constexpr int IMBAL_CONST    = 599;
constexpr int IMBAL_PAWN     = 600;  // value = pawn count of bishop side
constexpr int BISHOP_PAIR    = 601;
constexpr int KNIGHT_PAWN    = 602;  // value = knights * (total_pawns - 8)
constexpr int TRADEDOWN      = 603;  // EG only, value = diff*simpl/256
constexpr int KS_AU          = 604;  // value = attack-unit danger (pre-flight)
constexpr int KS_FLIGHT1     = 605;
constexpr int KS_FLIGHT2     = 606;
constexpr int ROOK_PAWN      = 607;  // value = rooks * pawns (per side)
constexpr int NN_PAIR        = 608;  // both knights present (per side)
constexpr int NB_PAIR        = 609;  // knight + bishop present (per side)
constexpr int SHIELD_KF_R1   = 610;  // +3 (r-1): own pawn on (king file, rel-rank r)
constexpr int SHIELD_ADJ_R1  = 613;  // +3 (r-1): own pawns on (kf+-1, r), value = count
constexpr int SHIELD_EXT_R1  = 616;  // +3 (r-1): own pawns on (kf+-2, r), value = count
constexpr int HOLE_KF        = 619;  // no own pawn anywhere on king file
constexpr int HOLE_ADJ       = 620;  // value = # adjacent files with no own pawn
constexpr int KS_AUB8        = 621;  // +5: attack_units >= T (T=8,16,24,32,40)
constexpr int VOL_CTX        = 626;  // value = tension x clamped material margin (context eval)
constexpr int OVERLOAD2      = 627;  // own pieces solely defending >= 2 attacked targets
constexpr int OVERLOAD3      = 628;  // own pieces solely defending >= 3 attacked targets
constexpr int RAZOR_HANGAU   = 629;  // value = enemy-hanging-count * min(au,48)/16 (razor class)
constexpr int RAZOR_THRAU    = 630;  // value = our-threat-count * min(au,48)/16
constexpr int RAZOR_MATAU    = 631;  // value = clamp(npm_diff,+-400)/100 * min(au,48)/16 (sac compensation)
constexpr int NPHASE         = 632;
constexpr int TEMPO_MG       = 2 * NPHASE;      // value = +ph / -ph
constexpr int TEMPO_EG       = 2 * NPHASE + 1;  // value = +(24-ph) / -(24-ph)
constexpr int NFEAT          = 2 * NPHASE + 2;
} // namespace feat
} // namespace luminex

// =====================================================================================================================
// The evaluation's fitted coefficients
// =====================================================================================================================

// GENERATED by gen_fitted.py from coefs_razor_iso1e7.txt -- do not edit by hand.
// FE_MG/FE_EG: current coefficient of every eval feature, solver-index space
// (the part before). evaluation.cpp and luminex-evaltrace (tools.cpp) both read these, so
// --verify revalidates the engine against whatever fit is applied here.

namespace luminex {

inline int FE_MG[632] = {
        32,     32,     32,     32,     32,     32,     32,     32,
       -12,     -5,     -3,     -5,     -2,      0,     -3,    -17,
       -14,     -5,     -1,     -3,      2,     -1,     -1,    -13,
       -11,     -6,      4,     12,     14,      5,     -3,     -6,
        -8,      2,      9,     23,     26,     17,      3,      3,
        -5,     15,     19,     28,     30,     27,     20,     17,
        50,     41,     50,     46,     41,     41,     26,     34,
        32,     32,     32,     32,     32,     32,     32,     32,
       -37,    -26,    -12,     -6,     -5,    -17,    -30,    -49,
       -16,     -2,      0,      8,      8,     -2,     -4,    -15,
       -11,      3,     11,     19,     20,     11,      3,    -10,
        -2,      4,     15,     26,     26,     18,     12,      1,
         0,      6,      9,     27,     26,     18,     16,     12,
        -1,     -8,    -11,      6,      5,     -2,      4,     16,
       -10,     -8,      8,    -13,      0,      7,     -6,      8,
       -81,     -7,    -28,    -10,      3,    -26,     12,    -53,
        -5,     13,     -8,     -4,     -7,    -13,      6,    -20,
         0,      5,     14,      5,      8,      8,      7,      0,
        -2,      9,     10,     14,     12,      9,     10,     -2,
        -2,     -4,      6,     13,     12,      3,     -1,     -2,
        -7,      6,     -1,      9,     12,      6,      7,      2,
        -7,      4,    -21,      0,     -2,    -28,     17,     11,
        -8,    -15,     -1,    -28,    -41,     -2,    -38,      2,
       -16,      5,    -24,    -19,     -1,    -47,     24,     11,
       -15,     -7,     -7,      1,     -3,      0,     -2,    -18,
       -18,    -11,     -4,      2,      1,     -3,     -7,    -20,
       -13,     -5,     -4,      1,      2,     -3,      0,    -11,
       -13,     -5,      0,      4,      3,     -2,      1,     -9,
       -10,     -1,      2,      6,      9,      7,     10,     -1,
        -5,      4,      4,     11,     12,     13,     15,      9,
        -6,     -1,      3,      5,      4,      0,      5,      6,
        -5,      5,     -9,    -26,    -27,    -10,     17,     15,
        -3,     -7,     -4,      2,      3,     -9,    -11,     -3,
        -8,     -5,      1,      8,      8,     -2,     -7,     -7,
        -8,     -4,      3,      5,      5,      4,     -2,     -4,
       -11,     -9,     -4,      3,      3,      1,      0,     -7,
       -12,     -6,     -5,      0,      8,      5,      6,      4,
        -8,    -11,     -7,      2,      7,      3,     13,     14,
       -13,    -15,     -8,    -17,    -14,     -5,     -4,     14,
       -12,     -3,     -2,    -55,      7,     13,     19,     19,
        25,     25,     -5,    -35,    -32,     -9,     15,     23,
        24,      7,    -19,    -39,    -39,    -21,      8,     19,
        -8,    -17,    -36,    -56,    -56,    -38,    -20,    -12,
       -15,    -19,    -34,    -55,    -56,    -37,    -22,    -19,
        -5,     -4,    -21,    -41,    -39,    -16,      0,     -2,
         1,     11,     -4,    -22,    -20,      1,     14,      8,
        13,     20,     11,     -3,     -2,     10,     21,     15,
        18,     33,     31,      7,     14,     27,     35,     22,
        85, /*MAT+pawn*/    315, /*MAT+knight*/    337, /*MAT+bishop*/    469, /*MAT+rook*/    882, /*MAT+queen*/      0, /*MAT+king*/    -49,    -36,
       -24,    -13,     -3,      7,     16,     22,     20,    -49,
       -42,    -31,    -25,    -20,    -16,    -13,    -10,     -8,
        -6,     -1,      2,      9,      1,    -31,    -27,    -25,
       -21,    -18,    -17,    -13,     -9,     -6,     -3,     -1,
         0,      2,      3,     -3,    -35,    -30,    -26,    -24,
       -22,    -19,    -16,    -14,    -13,    -11,    -10,    -10,
        -9,     -9,     -8,     -7,     -5,     -3,      1,      5,
         8,     12,     18,     21,     26,     32,     27,     41,
       -12, /*pawn_doubled*/     -9, /*pawn_isolated*/      8,     11,      9,     16,     23,     47,
         0,      4,      7,     15,     22,     36,     43,      0,
        14,      7,     13,     13,      6,     16,      0,      3, /*pawn_lever_c*/
        -4, /*pawn_backward*/     -9,    -16,    -21,    -24,    -11,      0,      0,
        -8,    -22,    -13,     -5,     11,      0,      0,     -5,
        -4,      2,     15,     37,     77,      0,      6,     -3,
       -12,    -17,    -26,    -52,      0,      8, /*pp_rook_behind*/    -28, /*pp_enemy_rook*/      0, /*pp_king_prox*/
         0, /*pp_unreach_s*/      0, /*pp_unreach_b*/     -5,    -10,    -10,     -8,    -13,    -38,
         0,    -11, /*pp_stop_own*/     -6, /*pp_stop_att*/     -3,     -8,    -14,    -14,     -9,
        13,      0,      0,      0,      0,      0,      0,      0,
         0,     15,     24,     39,      1, /*trapped_n*/    -16, /*far_n*/     -9, /*far_b*/      5,
        10,      7,      2,     -2,     -6,     -8,      7, /*bish_long*/     25, /*outpost_b*/
         7, /*rook_open*/      9, /*rook_semi*/      4, /*rook_7th*/     12, /*rook_xray_q*/    -10, /*far_r*/    -10, /*rook_trap_c*/    -39, /*rook_trap_nc*/     -6, /*rook_blocked*/
        -8, /*far_q*/     18, /*shield_r1*/     10, /*shield_r2*/      7, /*shield_r3*/    -38, /*open_kfile*/    -21, /*open_adj*/     -7, /*storm*/     25, /*castled*/
       -15, /*nocastle*/     -3,     59,     66,     97,    101,      5,     -7,
        23,     43,     54,      6,     22,    139,     48,     49,
         2,     18,     21,     68,     72,      2,     11,     15,
         7,    -21,      2, /*hang_pawn*/     23, /*hang_piece*/     13, /*pinned*/      2, /*space_own*/     -3, /*space_opp*/     17, /*imbal_const*/
        -4, /*imbal_pawn*/     25, /*bishop_pair*/      0, /*knight_pawn*/      0, /*tradedown*/      0, /*ks_au*/     48, /*ks_flight1*/     11, /*ks_flight2*/      0, /*rook_pawn*/
         0, /*nn_pair*/      0, /*nb_pair*/      0, /*shield_kf_r1*/      0, /*shield_kf_r2*/      0, /*shield_kf_r3*/      0, /*shield_adj_r1*/      0, /*shield_adj_r2*/      0, /*shield_adj_r3*/
         0, /*shield_ext_r1*/      0, /*shield_ext_r2*/      0, /*shield_ext_r3*/      0, /*hole_kf*/      0, /*hole_adj*/      0, /*ks_aub8*/      0, /*ks_aub16*/      0, /*ks_aub24*/
         0, /*ks_aub32*/      0, /*ks_aub40*/      0,      0,      0,     -4,      0,    -15,
};

inline int FE_EG[632] = {
         9,      9,      9,      9,      9,      9,      9,      9,
        -3,      0,      7,     -2,      5,      8,      3,     -9,
        -6,     -2,      1,     -1,      2,      3,      0,     -4,
        -4,      3,      5,      3,      4,      6,      4,      2,
         5,     16,     17,     20,     21,     17,     17,     14,
        20,     43,     45,     47,     47,     46,     42,     35,
        71,     78,     83,     78,     77,     81,     77,     70,
         9,      9,      9,      9,      9,      9,      9,      9,
       -37,     -7,     -8,     -2,     -1,     -7,     -7,    -32,
       -18,     -3,     -2,      4,      4,     -1,     -1,    -10,
        -5,     -2,     -2,      7,      8,      0,      1,      0,
         0,      3,      5,     12,     14,      6,      8,      6,
         0,      2,      5,     13,     14,      8,      9,      6,
        -2,      0,      3,      7,      9,      6,      4,      3,
        -8,     -1,      1,     11,     10,      1,     -1,     -4,
       -43,    -14,     -8,     -4,     -2,     -6,    -14,    -41,
        -6,     -2,     -2,     -5,     -8,     -4,     -8,    -12,
        -5,     -6,      1,      0,     -1,     -3,     -9,    -11,
        -3,      5,      0,      5,      5,     -2,      1,     -5,
        -5,     -2,      2,      3,      3,      0,     -1,     -6,
        -4,      1,      2,      4,      5,      4,      4,     -1,
        -1,      2,      2,      4,      5,      5,      5,      4,
        -2,     -1,      5,      6,      6,      3,      1,      0,
        -9,     -2,      0,     -3,     -5,     -1,     -2,     -7,
       -10,     -7,     -4,     -6,     -6,     -6,     -7,    -11,
       -13,     -9,     -3,     -2,     -2,     -6,    -13,    -16,
        -3,      0,      2,      1,      0,      2,      0,     -5,
         3,      9,     10,      8,      9,     11,      9,      1,
        10,     13,     13,     13,     13,     16,     14,      9,
        15,     16,     18,     17,     19,     21,     18,     14,
        11,     17,     16,     17,     17,     16,     18,     12,
        12,     17,     19,     19,     21,     20,     18,     13,
       -23,    -15,    -16,    -21,    -23,    -33,    -29,    -24,
       -17,    -14,     -9,    -10,    -12,    -18,    -26,    -18,
        -8,      1,      0,      0,      1,      3,      4,     -3,
        -2,      1,      5,     10,     12,     12,     11,     15,
        -1,      4,     11,     17,     22,     18,     20,     23,
         0,      4,     10,     20,     21,     16,     15,     24,
         4,      6,     15,     20,     23,     15,      6,     19,
       -12,     -1,      3,     14,     14,      6,      2,      1,
       -26,    -18,    -22,     -9,    -10,    -14,    -34,    -33,
       -10,     -7,     -3,      5,      6,     -3,     -7,    -13,
        -8,      0,      5,     12,     11,      5,     -2,     -9,
         1,      9,     15,     20,     20,     14,      8,      0,
         9,     17,     20,     25,     25,     21,     16,      7,
         7,     18,     22,     27,     25,     22,     18,      6,
        -2,     17,     18,     22,     22,     17,     18,     -1,
       -18,     12,     13,     13,     13,     14,     15,    -13,
        99, /*MAT+pawn*/    287, /*MAT+knight*/    306, /*MAT+bishop*/    520, /*MAT+rook*/    922, /*MAT+queen*/      0, /*MAT+king*/    -56,    -31,
       -19,     -9,      2,     14,     25,     36,     47,    -50,
       -40,    -34,    -28,    -21,    -14,     -6,      0,      7,
        12,     21,     25,     35,     38,    -32,    -28,    -23,
       -17,    -14,    -11,     -7,     -2,      5,     12,     17,
        22,     26,     29,     38,   -104,    -71,    -53,    -46,
       -38,    -32,    -29,    -26,    -24,    -23,    -20,    -17,
       -13,     -8,     -4,      1,      7,     11,     16,     19,
        22,     24,     30,     31,     36,     40,     32,     39,
       -16, /*pawn_doubled*/    -14, /*pawn_isolated*/      5,      9,      3,      6,     16,     16,
         0,      4,      9,     17,     40,     78,    136,      0,
         0,      0,      0,      0,      0,      0,      0,      0, /*pawn_lever_c*/
         0, /*pawn_backward*/     23,     13,      7,     21,     28,      0,      0,
        10,      8,     20,     41,     75,      0,      0,     18,
        24,     42,     70,    116,    192,      0,      8,     15,
        23,     25,     20,     18,      0,     26, /*pp_rook_behind*/    -23, /*pp_enemy_rook*/      8, /*pp_king_prox*/
        16, /*pp_unreach_s*/     20, /*pp_unreach_b*/      2,     -7,    -16,    -32,    -54,    -86,
         0,     -1, /*pp_stop_own*/     -5, /*pp_stop_att*/      8,      6,      7,     10,     15,
        29,      0,     10,     10,     13,     18,     24,     39,
         0,      5,      8,     13,     14, /*trapped_n*/     -6, /*far_n*/     -8, /*far_b*/     25,
        21,     15,      9,      4,      1,      4,     10, /*bish_long*/     16, /*outpost_b*/
        24, /*rook_open*/     14, /*rook_semi*/     15, /*rook_7th*/     20, /*rook_xray_q*/     -7, /*far_r*/    -10, /*rook_trap_c*/      8, /*rook_trap_nc*/     -6, /*rook_blocked*/
        -2, /*far_q*/      0, /*shield_r1*/      0, /*shield_r2*/      0, /*shield_r3*/      0, /*open_kfile*/      0, /*open_adj*/      0, /*storm*/      1, /*castled*/
        -1, /*nocastle*/    -31,     66,     90,    128,    360,     12,    -17,
        26,     63,    254,     10,     23,     86,     70,    246,
         6,     22,     25,     61,    168,     -3,      6,      6,
         5,    -57,     31, /*hang_pawn*/     25, /*hang_piece*/      2, /*pinned*/      0, /*space_own*/      0, /*space_opp*/     36, /*imbal_const*/
        -5, /*imbal_pawn*/     73, /*bishop_pair*/      0, /*knight_pawn*/      0, /*tradedown*/      1, /*ks_au*/     23, /*ks_flight1*/      7, /*ks_flight2*/      0, /*rook_pawn*/
         0, /*nn_pair*/      0, /*nb_pair*/      0, /*shield_kf_r1*/      0, /*shield_kf_r2*/      0, /*shield_kf_r3*/      0, /*shield_adj_r1*/      0, /*shield_adj_r2*/      0, /*shield_adj_r3*/
         0, /*shield_ext_r1*/      0, /*shield_ext_r2*/      0, /*shield_ext_r3*/      0, /*hole_kf*/      0, /*hole_adj*/      0, /*ks_aub8*/      0, /*ks_aub16*/      0, /*ks_aub24*/
         0, /*ks_aub32*/      0, /*ks_aub40*/      0,      0,      0,    -10,      1,      2,
};

} // namespace luminex

// =====================================================================================================================
// The evaluation's tuner parameters
// =====================================================================================================================

#include <array>
#include <cstddef>

namespace luminex {

// 148 tunable parameters: material (10) + mobility tables (132) + eval bonuses (6)
//
// Architecture: mobility uses non-linear lookup tables instead of linear
// base+slope formulas. Each mobility level has an independently tunable
// value, capturing diminishing returns without assuming a functional form.
//
// Knight: 9 entries (mob 0-8)  × 2 phases = 18 params
// Bishop: 14 entries (mob 0-13) × 2 phases = 28 params
// Rook:   15 entries (mob 0-14) × 2 phases = 30 params
// Queen:  28 entries (mob 0-27) × 2 phases = 56 params
//                                        Total = 132 mobility params

enum ParamIdx : size_t {
    // Material values (10)
    PAWN_VALUE_MG = 0,
    PAWN_VALUE_EG,
    KNIGHT_VALUE_MG,
    KNIGHT_VALUE_EG,
    BISHOP_VALUE_MG,
    BISHOP_VALUE_EG,
    ROOK_VALUE_MG,
    ROOK_VALUE_EG,
    QUEEN_VALUE_MG,
    QUEEN_VALUE_EG,

    // Knight mobility table MG (9 entries, mob 0-8)
    KNIGHT_MOB_MG_0,
    KNIGHT_MOB_MG_1,
    KNIGHT_MOB_MG_2,
    KNIGHT_MOB_MG_3,
    KNIGHT_MOB_MG_4,
    KNIGHT_MOB_MG_5,
    KNIGHT_MOB_MG_6,
    KNIGHT_MOB_MG_7,
    KNIGHT_MOB_MG_8,

    // Knight mobility table EG (9 entries, mob 0-8)
    KNIGHT_MOB_EG_0,
    KNIGHT_MOB_EG_1,
    KNIGHT_MOB_EG_2,
    KNIGHT_MOB_EG_3,
    KNIGHT_MOB_EG_4,
    KNIGHT_MOB_EG_5,
    KNIGHT_MOB_EG_6,
    KNIGHT_MOB_EG_7,
    KNIGHT_MOB_EG_8,

    // Bishop mobility table MG (14 entries, mob 0-13)
    BISHOP_MOB_MG_0,
    BISHOP_MOB_MG_1,
    BISHOP_MOB_MG_2,
    BISHOP_MOB_MG_3,
    BISHOP_MOB_MG_4,
    BISHOP_MOB_MG_5,
    BISHOP_MOB_MG_6,
    BISHOP_MOB_MG_7,
    BISHOP_MOB_MG_8,
    BISHOP_MOB_MG_9,
    BISHOP_MOB_MG_10,
    BISHOP_MOB_MG_11,
    BISHOP_MOB_MG_12,
    BISHOP_MOB_MG_13,

    // Bishop mobility table EG (14 entries, mob 0-13)
    BISHOP_MOB_EG_0,
    BISHOP_MOB_EG_1,
    BISHOP_MOB_EG_2,
    BISHOP_MOB_EG_3,
    BISHOP_MOB_EG_4,
    BISHOP_MOB_EG_5,
    BISHOP_MOB_EG_6,
    BISHOP_MOB_EG_7,
    BISHOP_MOB_EG_8,
    BISHOP_MOB_EG_9,
    BISHOP_MOB_EG_10,
    BISHOP_MOB_EG_11,
    BISHOP_MOB_EG_12,
    BISHOP_MOB_EG_13,

    // Rook mobility table MG (15 entries, mob 0-14)
    ROOK_MOB_MG_0,
    ROOK_MOB_MG_1,
    ROOK_MOB_MG_2,
    ROOK_MOB_MG_3,
    ROOK_MOB_MG_4,
    ROOK_MOB_MG_5,
    ROOK_MOB_MG_6,
    ROOK_MOB_MG_7,
    ROOK_MOB_MG_8,
    ROOK_MOB_MG_9,
    ROOK_MOB_MG_10,
    ROOK_MOB_MG_11,
    ROOK_MOB_MG_12,
    ROOK_MOB_MG_13,
    ROOK_MOB_MG_14,

    // Rook mobility table EG (15 entries, mob 0-14)
    ROOK_MOB_EG_0,
    ROOK_MOB_EG_1,
    ROOK_MOB_EG_2,
    ROOK_MOB_EG_3,
    ROOK_MOB_EG_4,
    ROOK_MOB_EG_5,
    ROOK_MOB_EG_6,
    ROOK_MOB_EG_7,
    ROOK_MOB_EG_8,
    ROOK_MOB_EG_9,
    ROOK_MOB_EG_10,
    ROOK_MOB_EG_11,
    ROOK_MOB_EG_12,
    ROOK_MOB_EG_13,
    ROOK_MOB_EG_14,

    // Queen mobility table MG (28 entries, mob 0-27)
    QUEEN_MOB_MG_0,
    QUEEN_MOB_MG_1,
    QUEEN_MOB_MG_2,
    QUEEN_MOB_MG_3,
    QUEEN_MOB_MG_4,
    QUEEN_MOB_MG_5,
    QUEEN_MOB_MG_6,
    QUEEN_MOB_MG_7,
    QUEEN_MOB_MG_8,
    QUEEN_MOB_MG_9,
    QUEEN_MOB_MG_10,
    QUEEN_MOB_MG_11,
    QUEEN_MOB_MG_12,
    QUEEN_MOB_MG_13,
    QUEEN_MOB_MG_14,
    QUEEN_MOB_MG_15,
    QUEEN_MOB_MG_16,
    QUEEN_MOB_MG_17,
    QUEEN_MOB_MG_18,
    QUEEN_MOB_MG_19,
    QUEEN_MOB_MG_20,
    QUEEN_MOB_MG_21,
    QUEEN_MOB_MG_22,
    QUEEN_MOB_MG_23,
    QUEEN_MOB_MG_24,
    QUEEN_MOB_MG_25,
    QUEEN_MOB_MG_26,
    QUEEN_MOB_MG_27,

    // Queen mobility table EG (28 entries, mob 0-27)
    QUEEN_MOB_EG_0,
    QUEEN_MOB_EG_1,
    QUEEN_MOB_EG_2,
    QUEEN_MOB_EG_3,
    QUEEN_MOB_EG_4,
    QUEEN_MOB_EG_5,
    QUEEN_MOB_EG_6,
    QUEEN_MOB_EG_7,
    QUEEN_MOB_EG_8,
    QUEEN_MOB_EG_9,
    QUEEN_MOB_EG_10,
    QUEEN_MOB_EG_11,
    QUEEN_MOB_EG_12,
    QUEEN_MOB_EG_13,
    QUEEN_MOB_EG_14,
    QUEEN_MOB_EG_15,
    QUEEN_MOB_EG_16,
    QUEEN_MOB_EG_17,
    QUEEN_MOB_EG_18,
    QUEEN_MOB_EG_19,
    QUEEN_MOB_EG_20,
    QUEEN_MOB_EG_21,
    QUEEN_MOB_EG_22,
    QUEEN_MOB_EG_23,
    QUEEN_MOB_EG_24,
    QUEEN_MOB_EG_25,
    QUEEN_MOB_EG_26,
    QUEEN_MOB_EG_27,

    // Key EvalParams (6)
    BISHOP_PAIR_MG,
    BISHOP_PAIR_EG,
    ROOK_OPEN_FILE_MG,
    ROOK_OPEN_FILE_EG,
    ROOK_SEMI_OPEN_MG,
    ROOK_SEMI_OPEN_EG,

    PARAM_COUNT = 148
};

// Mutable parameter array
extern std::array<int, PARAM_COUNT> g_params;

// Initialize g_params with current defaults (linear formula → table values)
void init_tuner_params();

// Clear pawn hash table (for tuner between positions)
void clear_pawn_table();

} // namespace luminex

// =====================================================================================================================
// NNUE
// =====================================================================================================================

// Optional NNUE evaluation for Luminex.
// Toggleable at runtime via the UCI option "UseNNUE". When disabled (default)
// or when no net is loaded, the engine falls back to the pure HCE eval, so NNUE
// is fully optional and non-destructive. Net file format written by
// Luminex-NNUE's trainer/luminex_nnue_train.py (header LNN1 + float32 weights).
#include <string>
#include <vector>

namespace luminex::nnue {

// The cap on L1 width (NNUE_L1_MAX) and the kernels' interface are in
// nnue_kernels.h.

// Network shipped with each release, by its real name: CMake reads it from the
// file name of the network .github/scripts/ci.sh pins (pin_url), so the default
// NNUEFile follows every net swap. Release binaries carry the network inside
// them (CMake LUMINEX_EMBED_PINNED); a build without it loads the file of this
// name from the working directory or next to the engine binary.
#ifndef LUMINEX_DEFAULT_NET
#error "LUMINEX_DEFAULT_NET comes from CMake: the pinned network's file name in .github/scripts/ci.sh (pin_url)"
#endif
inline constexpr const char* DEFAULT_NET = LUMINEX_DEFAULT_NET;

// Load a .nnue file. Returns true on success. A missing file or bad header
// keeps the current net; a file that fails part-way leaves no net loaded.
bool load(const std::string& path);

// Load the network built into the binary, under the same rules as load().
// False when the build carries none.
bool load_embedded();

// Whether the binary carries a built-in network.
bool has_embedded();

// The last load's report ("info string NNUE evaluation using ...", then any
// warning of that load), once: empty when it was taken or no net is loaded.
// The UCI layer prints it at the next command that uses the net (go, eval,
// ...). A load happens at setoption or isready, where a UCI client reads
// any line but readyok as unexpected output (python-chess, and so
// lichess-bot, warns on each).
std::string take_report();

// True iff a net is loaded AND the user enabled UseNNUE. evaluate() checks this.
bool enabled();

// Set the UseNNUE flag (wired from the UCI option handler).
void set_enabled(bool on);

// The NNUE eval for the position, in engine eval units (centipawns).
// Reads the incremental accumulator (must be current — see refresh/update).
// Only valid when enabled() is true.
Value evaluate(const Position& pos);

// Whether a net is currently loaded (independent of the enabled flag).
bool loaded();

// Runtime L1 width of the loaded net (0 if none loaded).
int l1();

// Print profiling counters (evals/updates/refreshes + time each) to stderr.
void print_stats();

// gen768t (LNT8) debugging, for the relrows and lanes UCI commands. relation_rows: the position's relation and
// pawn-pair rows for White's and Black's perspective, sorted, one "relrows <side> <n> <rows...>" line each
// (ppm: pawn pairs in contact 0, all 1; needs no network). lane_dump: the lanes the forward pass reads, per
// perspective ("lanes <side> <L1 values>", the relation term included below T, then "rel <side> <T values>"),
// brought current first.
std::string relation_rows(const Position& pos, int ppm);
std::string lane_dump(const Position& pos);

// ---- incremental accumulator hooks (no-ops when !loaded(); cheap branch otherwise) ----
// Full rebuild of the CURRENT state's accumulator from the board. Call from Position::set.
void refresh(Position& pos);

// Apply the move's feature deltas to the current (child) state's accumulator.
// The parent→child copy is done in board.cpp's do_move copy block; this applies the delta.
//   m        : the move
//   moved    : the Piece that moved (read from board[from] before mutation)
//   captured : captured PieceType (PT_NONE if none) = st->captured_piece
// Call from do_move AFTER board mutation + st_ advance.
void update(Position& pos, Move m, Piece moved, PieceType captured);

// Null move: board unchanged, so the child accumulator is just a copy of the parent's.
// Call from do_null_move AFTER st_ advance.
void copy_for_null(Position& pos);

// ---- SIMD kernel levels ----
// Whether the binary picks its kernels at run time (the universal x86-64
// build, CMake LUMINEX_DISPATCH). Only then do the levels below offer a choice.
bool simd_dispatch();

// The kernels' instruction-set level in use: avx512vnni, avx512, avxvnni,
// avx2 or base on x86-64; dotprod or base on ARM.
const char* simd_level();

// The levels this build carries, best first (the values of the SimdLevel
// option after auto).
std::vector<std::string> simd_levels();

// The SimdLevel option: "auto" for the best level the CPU supports, or a
// level's name, which gives the best supported level at or below it. Returns
// what to report: the level now in use, and why if it is not the one asked for.
std::string set_simd_level(const std::string& request);

// nnuecheck: the evaluations of a fixed set of games under every level the
// CPU supports, which must agree exactly, and their digest for comparing
// builds. Fills report (lines, the last "nnuecheck ok" or "nnuecheck FAILED:
// ..."); false on a difference or without an int8 network.
bool check_simd(std::string& report);

} // namespace luminex::nnue

// =====================================================================================================================
// Search
// =====================================================================================================================

#include <atomic>
#include <cstdint>

namespace luminex {

// Search stack
class Stack {
public:
    Move pv[64];  // Reduced from MAX_PLY + 7 to save stack space
    int ply = 0;
    Move current_move = MOVE_NONE;
    Move excluded_move = MOVE_NONE;
    Piece moved_piece = NO_PIECE;  // Piece that moved at this ply
    Value static_eval = VALUE_ZERO;
    int move_count = 0;
    int reduction = 0;  // LMR reduction applied at this ply (for hindsight adjustment)
    int forced_ply_count = 0;  // Consecutive plies in forcing sequence (checks/low mobility)
    bool improving = false;
    bool improving_deep = false;  // 4-ply eval trajectory (sustained improvement)
    Stack* previous = nullptr;
};

// Search limits
struct Limits {
    int time[2] = {};
    int inc[2] = {};
    int npmsec = 0;
    int movetime = 0;
    int depth = 0;  // Default 0 to allow time management
    int nodes = 0;
    int movestogo = 0;  // UCI movestogo parameter for time management
    uint64_t mate = 0;
    bool infinite = false;
    bool ponder = false;

    bool use_time_management() const { return !movetime && !depth && !nodes && !mate && !infinite; }
};

// Search parameters
struct SearchParams {
    int contempt = 0;
    int multi_pv = 1;
    bool show_wdl = false;      // UCI "UCI_ShowWDL": append wdl W D L to info
    int move_overhead_ms = 30;  // UCI "Move Overhead": latency safety margin (ms)
    // UCI "TimeSchedule", sudden death with an increment: 0 = the old
    // reserve formula; 1 and 2 = schedules solved by dynamic programming over
    // SPRT game lengths (1: every move worth the same, 2: early moves worth
    // more, the default: +72 Elo against the old formula at 10+0.1, schedule
    // 1 +32). 1 and 2 also move at once with a single legal reply and never
    // start an iteration that cannot finish before the hard limit.
    int time_schedule = 2;
    // UCI "TimeStability": 0 = the old soft-limit factors for best-move
    // stability; 1 = factors measured on recorded searches (more time right
    // after the best move changes, much less once it has held or the game
    // is decided), the default: +8.3 Elo against 0 at 10+0.1
    int time_stability = 1;
    // UCI "TimeZeroInc", sudden death without an increment: 0 = the old
    // formula; 1 = the clock above a reserve (1 s, or 30 move overheads)
    // spread over the value-weighted horizon (value_horizon() in search.cpp),
    // the default: +5.6 Elo against 0 at 10+0
    int time_zero_inc = 1;
    // UCI "TimeMovesToGo", controls with moves to go: 0 = the old formula
    // (time_left / movestogo); 1 = the clock above a small reserve spread over
    // the value-weighted horizon to the last move before the next control,
    // the default: +11.0 Elo against 0 at 40 moves in 10 s
    int time_moves_to_go = 1;
    // UCI "TimeLag", sudden death without an increment: 1 = the schedule also
    // pays for what the clock loses on each move beyond the engine's own
    // thinking (the connection's lag, measured from the clock: lag_estimate()
    // in search.cpp), the default; 0 = it does not. Without measurable lag the
    // two give the same budgets.
    int time_lag = 1;
};

// Search statistics for debugging
struct SearchStats {
    // Node counts
    int64_t main_nodes = 0;
    int64_t qs_nodes = 0;

    // TT stats
    int64_t tt_probes = 0;
    int64_t tt_hits = 0;
    int64_t tt_cutoffs = 0;
    int64_t tt_hit_but_shallow = 0;  // TT hit but depth < current, no cutoff

    // Eval cache stats
    int64_t eval_cache_hits_main = 0;
    int64_t eval_cache_misses_main = 0;
    int64_t eval_cache_hits_qs = 0;
    int64_t eval_cache_misses_qs = 0;

    // Qsearch depth stats
    int64_t qs_stand_pat_cutoffs = 0;   // eval >= beta, no captures searched
    int64_t qs_delta_prunes = 0;        // delta pruning cutoffs
    int64_t qs_cap_searched = 0;        // nodes where at least 1 capture was searched
    int64_t qs_cap_cutoffs = 0;         // nodes where a capture caused cutoff
    int64_t qs_no_moves = 0;            // nodes with no legal captures (all pruned or none)

    // Pruning stats
    int64_t null_move_prunes = 0;
    int64_t futility_prunes = 0;
    int64_t rev_futility_prunes = 0;
    int64_t razoring_prunes = 0;
    int64_t probcut_prunes = 0;
    int64_t delta_prunes_qs = 0;
    int64_t lmp_prunes = 0;
    int64_t history_prunes = 0;

    // LMR stats
    int64_t lmr_total = 0;
    int64_t lmr_researches = 0;
    int64_t lmr_improved = 0;

    // Extension stats
    int64_t singular_extensions = 0;
    int64_t check_extensions = 0;
    int64_t recapture_extensions = 0;

    // Move ordering stats
    int64_t captures_generated = 0;
    int64_t captures_searched = 0;
    int64_t quiets_generated = 0;
    int64_t quiets_searched = 0;
    int64_t first_move_cutoffs = 0;    // Cutoff on first non-TT move searched
    int64_t iir_reductions = 0;

    int max_depth_reached = 0;

    // PMG phase statistics
    int64_t pmg_tt_cutoffs = 0;      // TT move caused cutoff
    int64_t pmg_capture_cutoffs = 0; // Capture caused cutoff (quiets never generated)
    int64_t pmg_quiet_generated = 0; // Had to generate quiet moves
    int64_t pmg_quiet_cutoffs = 0;   // Quiet move caused cutoff
};

extern thread_local SearchStats g_stats;

class Position;
class TranspositionTable;

// Root search
Move search(Position& pos, Limits& limits);

// Rebuild the tables derived from SPSAParams (call after changing a param)
void init_search_params();

// Correction history - clear on new game
void clear_correction_history();

// Search history - clear on new game
void clear_search_history();

// The clock's lag measurement - clear on new game
void clear_clock_lag();

// UCI info output; lower_bound marks the score as a lower bound
void uci_info(const Position& pos, int depth, Value score, uint64_t nodes, int time,
              int multipv = 1, Move seed_move = MOVE_NONE, bool lower_bound = false);

// Search globals
extern Limits limits;
extern SearchParams params;
extern std::atomic<uint64_t> nodes;
extern std::atomic<bool> stop;
extern int root_depth;
extern Value root_score;
extern int num_threads; // Default 1, SMP only helps at longer TC
extern std::atomic<bool> ponder_mode;
extern std::atomic<bool> ponderhit_received;

// Called by the search thread once the reply (bestmove) is ready, in this
// order: a ponder or infinite search's reply waits for ponderhit or stop; then
// a played move's clock is recorded for the next own move's lag sample.
void hold_reply(const Limits& lim);
void finish_clock(const Limits& lim);
// After hold_reply, for a search's reply (not a book move's): the line a
// reply after a ponderhit needs when none was printed since the hit.
void report_after_hit(const Position& root);

} // namespace luminex

// =====================================================================================================================
// SPSA search constants
// =====================================================================================================================

// Runtime-tunable search constants for SPSA optimization.
// Every default but the shallow null-move pair is the rounded final value of
// OpenBench tune 45 (SPSA on those 27 fields at once, 80,000 games at 10+0.1,
// run on main 5545f07's search); the pair is the setting #41's SPRT tested.
// Each field is also a UCI spin option (see uci.cpp), so OpenBench SPSA can
// tune it directly. A spsa_params.txt in the working directory (the first 7
// integers, one per line, in field order) is read once at startup; UCI options
// set afterwards take precedence. Missing file -> defaults.
#include <cstdio>
#include <string>

struct SPSAParams {
    // LMR reduction scale (reduction = scale^2 * ln(depth) * ln(moves) / 1024)
    int lmr_scale_quiet  = 30;
    // Null-move pruning reduction base: R = nmp_base + depth / 3 + min(margin / 200, 3)
    int nmp_base         = 6;
    // Razoring: prune if eval + razor_base + depth^2 * razor_coeff < alpha
    int razor_base       = 285;
    int razor_coeff      = 57;
    // Reverse futility (static null move): prune if eval - rev_fut_coeff * depth >= beta
    int rev_fut_coeff    = 90;
    // Aspiration window initial half-size
    int aspiration_delta = 41;
    // Singular extension margin: sBeta = tt_value - depth * _singular_margin / 320
    int _singular_margin = 201;

    static constexpr int COUNT = 7;  // fields read from and written to spsa_params.txt

    // Reverse futility: extra margin when neither side's eval trend helps us
    int rfp_not_improving = 26;
    // Futility (quiet moves, depth <= 5): prune if eval + depth * fut_coeff
    // + (improving ? fut_improving : fut_base) < alpha
    int fut_coeff        = 134;
    int fut_base         = 101;
    int fut_improving    = 27;
    // Late move pruning, in hundredths of a move: the count is
    // (lmp_base + 100 if improving + depth^2 * lmp_coeff) / 100
    int lmp_base         = 233;
    int lmp_coeff        = 94;
    // History pruning (quiets, depth <= 4): prune if combined history
    // < hist_prune_base - hist_prune_coeff * (depth - 1)
    int hist_prune_base  = 896;
    int hist_prune_coeff = 5050;
    // ProbCut: rbeta = beta + probcut_margin
    int probcut_margin   = 132;
    // Null-move R: the depth term is depth * 100 / nmp_depth_div, the eval
    // term min((eval - beta) / nmp_eval_div, 3)
    int nmp_depth_div    = 289;
    int nmp_eval_div     = 214;
    // LMR: history adjustment = clamp(history / lmr_hist_div, -3, 3); the
    // cut-node and ttPv adjustments are in 1024ths of a ply
    int lmr_hist_div     = 3894;
    int lmr_cut_node     = 1021;
    int lmr_ttpv         = 2101;
    // Double singular extension when the verification fails by more than
    // depth * dext_margin / 16
    int dext_margin      = 16;
    // History bonus: min(quad * d^2 + lin * d - offset, 8192); the malus is
    // its negative
    int hist_bonus_quad  = 21;
    int hist_bonus_lin   = 206;
    int hist_bonus_offset = 125;
    // Qsearch skips captures with SEE below qs_see_coeff * depth (depth <= 0)
    int qs_see_coeff     = 50;
    // Correction history update weight: min(corr_weight_max, depth + 1) / 256
    int corr_weight_max  = 15;

    // Null-move pruning at depth <= nmp_shallow_depth needs the static eval
    // at least nmp_shallow_margin above beta (deeper: at beta)
    int nmp_shallow_margin = 57;
    int nmp_shallow_depth  = 5;

    // A main-search result replaces a same-key entry when its depth plus
    // this reaches the entry's (0 kept only results as deep or deeper, so a
    // shallower search of the same position lost its best move)
    int tt_store_slack = 3;
    // No extension once a path is this many quarters of the iteration's
    // depth long (0 = no cap): a line of checks and singular extensions
    // could otherwise keep the depth from falling
    int ext_path_cap = 8;

    // Under measurement (0 = off): each bit switches one search heuristic off
    // (the self-knockout campaign; the bits are listed at KoMask's uses)
    int ko_mask = 0;      // derived: ko_mask_opt | the Ko<b> options (refresh_ko)
    int ko_mask_opt = 0;  // the KoMask option
    // Ko<b> > 0 switches bit b off (the knockout factorial: OpenBench's SPSA flips them all at once)
    int ko0 = 0, ko1 = 0, ko2 = 0, ko3 = 0, ko4 = 0, ko5 = 0, ko6 = 0, ko7 = 0, ko8 = 0, ko9 = 0, ko10 = 0, ko11 = 0, ko12 = 0, ko13 = 0, ko14 = 0, ko15 = 0, ko16 = 0, ko17 = 0, ko18 = 0;
    void refresh_ko() {
        ko_mask = ko_mask_opt
            | ((ko0 > 0) << 0)
            | ((ko1 > 0) << 1)
            | ((ko2 > 0) << 2)
            | ((ko3 > 0) << 3)
            | ((ko4 > 0) << 4)
            | ((ko5 > 0) << 5)
            | ((ko6 > 0) << 6)
            | ((ko7 > 0) << 7)
            | ((ko8 > 0) << 8)
            | ((ko9 > 0) << 9)
            | ((ko10 > 0) << 10)
            | ((ko11 > 0) << 11)
            | ((ko12 > 0) << 12)
            | ((ko13 > 0) << 13)
            | ((ko14 > 0) << 14)
            | ((ko15 > 0) << 15)
            | ((ko16 > 0) << 16)
            | ((ko17 > 0) << 17)
            | ((ko18 > 0) << 18);
    }

    // Load from file (COUNT ints, one per line, in field order). Returns true if loaded.
    bool load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "r");
        if (!f) return false;
        int* p[COUNT] = {&lmr_scale_quiet, &nmp_base, &razor_base, &razor_coeff,
                         &rev_fut_coeff, &aspiration_delta, &_singular_margin};
        int v[COUNT];
        for (int i = 0; i < COUNT; ++i)
            if (std::fscanf(f, "%d", &v[i]) != 1) { std::fclose(f); return false; }
        std::fclose(f);
        for (int i = 0; i < COUNT; ++i) *p[i] = v[i];
        return true;
    }

    // Save to file (for SPSA loop to write perturbed values)
    void save(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "w");
        if (!f) return;
        std::fprintf(f, "%d\n%d\n%d\n%d\n%d\n%d\n%d\n",
                     lmr_scale_quiet, nmp_base, razor_base, razor_coeff,
                     rev_fut_coeff, aspiration_delta, _singular_margin);
        std::fclose(f);
    }

    // Global instance
    static SPSAParams& get() {
        static SPSAParams instance;
        return instance;
    }
};

// The UCI spin options, one per field. uci.cpp prints them and clamps values
// set through them to [min, max], which also keeps every divisor above zero.
struct SPSAOption {
    const char* name;
    int SPSAParams::* field;
    int min, max;
};

inline constexpr SPSAOption SPSA_OPTIONS[] = {
    {"LmrScale",        &SPSAParams::lmr_scale_quiet,    20,    70},
    {"NmpBase",         &SPSAParams::nmp_base,            2,     6},
    {"RazorBase",       &SPSAParams::razor_base,        150,   500},
    {"RazorCoeff",      &SPSAParams::razor_coeff,        30,   120},
    {"RfpCoeff",        &SPSAParams::rev_fut_coeff,      60,   160},
    {"AspirationDelta", &SPSAParams::aspiration_delta,   20,   100},
    {"SingularMargin",  &SPSAParams::_singular_margin,  100,   400},
    {"RfpNotImproving", &SPSAParams::rfp_not_improving,   0,   100},
    {"FutCoeff",        &SPSAParams::fut_coeff,          50,   300},
    {"FutBase",         &SPSAParams::fut_base,            0,   250},
    {"FutImproving",    &SPSAParams::fut_improving,       0,   200},
    {"LmpBase",         &SPSAParams::lmp_base,           50,   600},
    {"LmpCoeff",        &SPSAParams::lmp_coeff,          25,   300},
    {"HistPruneBase",   &SPSAParams::hist_prune_base, -4000,  4000},
    {"HistPruneCoeff",  &SPSAParams::hist_prune_coeff, 1000, 12000},
    {"ProbcutMargin",   &SPSAParams::probcut_margin,     50,   400},
    {"NmpDepthDiv",     &SPSAParams::nmp_depth_div,     150,   800},
    {"NmpEvalDiv",      &SPSAParams::nmp_eval_div,       50,   600},
    {"LmrHistDiv",      &SPSAParams::lmr_hist_div,     1024, 16384},
    {"LmrCutNode",      &SPSAParams::lmr_cut_node,        0,  3072},
    {"LmrTtPv",         &SPSAParams::lmr_ttpv,            0,  4096},
    {"DoubleExtMargin", &SPSAParams::dext_margin,         0,    64},
    {"HistBonusQuad",   &SPSAParams::hist_bonus_quad,     0,    64},
    {"HistBonusLin",    &SPSAParams::hist_bonus_lin,      0,   400},
    {"HistBonusOffset", &SPSAParams::hist_bonus_offset,   0,   400},
    {"QsSeeCoeff",      &SPSAParams::qs_see_coeff,        0,   200},
    {"CorrWeightMax",   &SPSAParams::corr_weight_max,     1,   128},
    {"NmpShallowMargin", &SPSAParams::nmp_shallow_margin, 0,  300},
    {"NmpShallowDepth", &SPSAParams::nmp_shallow_depth,   0,   12},
    {"TtStoreSlack",    &SPSAParams::tt_store_slack,      0,     8},
    {"ExtPathCap",      &SPSAParams::ext_path_cap,        0,    16},
    {"KoMask",          &SPSAParams::ko_mask_opt,         0, 1048575},
    {"Ko0",             &SPSAParams::ko0,                  -10000, 10000},
    {"Ko1",             &SPSAParams::ko1,                  -10000, 10000},
    {"Ko2",             &SPSAParams::ko2,                  -10000, 10000},
    {"Ko3",             &SPSAParams::ko3,                  -10000, 10000},
    {"Ko4",             &SPSAParams::ko4,                  -10000, 10000},
    {"Ko5",             &SPSAParams::ko5,                  -10000, 10000},
    {"Ko6",             &SPSAParams::ko6,                  -10000, 10000},
    {"Ko7",             &SPSAParams::ko7,                  -10000, 10000},
    {"Ko8",             &SPSAParams::ko8,                  -10000, 10000},
    {"Ko9",             &SPSAParams::ko9,                  -10000, 10000},
    {"Ko10",            &SPSAParams::ko10,                 -10000, 10000},
    {"Ko11",            &SPSAParams::ko11,                 -10000, 10000},
    {"Ko12",            &SPSAParams::ko12,                 -10000, 10000},
    {"Ko13",            &SPSAParams::ko13,                 -10000, 10000},
    {"Ko14",            &SPSAParams::ko14,                 -10000, 10000},
    {"Ko15",            &SPSAParams::ko15,                 -10000, 10000},
    {"Ko16",            &SPSAParams::ko16,                 -10000, 10000},
    {"Ko17",            &SPSAParams::ko17,                 -10000, 10000},
    {"Ko18",            &SPSAParams::ko18,                 -10000, 10000},
};

inline const SPSAOption* find_spsa_option(const std::string& name) {
    for (const SPSAOption& o : SPSA_OPTIONS)
        if (name == o.name) return &o;
    return nullptr;
}

// =====================================================================================================================
// The search thread
// =====================================================================================================================

// A thread for the search, started with a stack of SEARCH_THREAD_STACK bytes whatever the platform's default
// for secondary threads is (1 MiB on Windows, 512 KiB on macOS). A search thread's entry holds a Position by
// value (about 220 KB: the state stack and the game history) and every search frame holds 4 KiB move lists;
// with 512 KiB the first search overflows. The size is a reservation: pages are committed only as the search
// touches them. If the stack cannot be had, the job runs on a default thread rather than not at all.

#include <cstddef>
#include <functional>
#include <memory>
#include <thread>
#ifndef _WIN32
#include <pthread.h>
#endif

namespace luminex {

inline constexpr std::size_t SEARCH_THREAD_STACK = std::size_t(32) * 1024 * 1024;

class SearchThread {
public:
    SearchThread() = default;
    explicit SearchThread(std::function<void()> job) { start(std::move(job)); }
    SearchThread(const SearchThread&) = delete;
    SearchThread& operator=(const SearchThread&) = delete;
    SearchThread(SearchThread&& other) noexcept { take(other); }
    SearchThread& operator=(SearchThread&& other) noexcept {
        if (this != &other) { join(); take(other); }
        return *this;
    }
    ~SearchThread() { join(); }

    bool joinable() const { return running_ || fallback_.joinable(); }

    // join() and start() are in search.cpp, the one file that includes windows.h: its macros (small, near, far,
    // interface, ERROR, ...) would collide with names in the others.
    void join();

private:
    using Job = std::function<void()>;

    static void run(void* job) {
        std::unique_ptr<Job> owned(static_cast<Job*>(job));
        (*owned)();
    }

    void start(Job job);

    void take(SearchThread& other) {
#ifdef _WIN32
        handle_ = other.handle_;
        other.handle_ = nullptr;
#else
        thread_ = other.thread_;
#endif
        running_ = other.running_;
        other.running_ = false;
        fallback_ = std::move(other.fallback_);
    }

#ifdef _WIN32
    void* handle_ = nullptr;   // a HANDLE
#else
    pthread_t thread_{};
#endif
    bool running_ = false;
    std::thread fallback_;
};

} // namespace luminex

// =====================================================================================================================
// UCI
// =====================================================================================================================

namespace luminex {

class Position;

// UCI loop - main entry point
void uci_loop();

// luminex nnuecheck [NETFILE]: load NETFILE (empty: the default network) and
// check that every SIMD kernel level evaluates alike; true when they do.
bool run_nnuecheck(const std::string& net_file);

// UCI command handlers
void handle_uci();
void handle_isready();
void handle_ucinewgame();
void handle_position(Position& pos, const std::string& cmd);
void handle_go(Position& pos, const std::string& cmd);
void handle_setoption(const std::string& cmd);
void handle_stop();
void handle_quit();

// UCI output
void uci_send(const char* msg, ...);

// Thread-safe output for use by search thread
void uci_safe_output(const std::string& msg);

// Debug logging for search thread
void uci_debug_log(const char* format, ...);

} // namespace luminex

// =====================================================================================================================
// The engine
// =====================================================================================================================

namespace luminex {

// Engine info
constexpr const char* ENGINE_NAME = "Luminex";
constexpr const char* ENGINE_AUTHOR = "changcheng967";
constexpr const char* ENGINE_VERSION = "6.6.0";

// Initialize engine
void init();

// Perft testing (for debugging)
uint64_t perft(Position& pos, Depth depth);

} // namespace luminex
