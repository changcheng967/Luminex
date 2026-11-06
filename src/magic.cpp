#include "luminex.h"

namespace luminex {

// Precomputed line and between tables
Bitboard LineBB[64][64];
Bitboard BetweenBB[64][64];

// Magic bitboard tables
Magic RookMagics[64];
Magic BishopMagics[64];
Bitboard RookTable[262144];   // Large enough for all rook entries
Bitboard BishopTable[65536];  // Large enough for all bishop entries

namespace {

// Compute attack mask for a rook (edges excluded)
Bitboard compute_rook_mask(Square s) {
    Bitboard mask = 0;
    int f = file_of(s), r = rank_of(s);
    for (int ff = f + 1; ff < 7; ++ff) mask |= square_bb(make_square(File(ff), Rank(r)));
    for (int ff = f - 1; ff > 0; --ff) mask |= square_bb(make_square(File(ff), Rank(r)));
    for (int rr = r + 1; rr < 7; ++rr) mask |= square_bb(make_square(File(f), Rank(rr)));
    for (int rr = r - 1; rr > 0; --rr) mask |= square_bb(make_square(File(f), Rank(rr)));
    return mask;
}

// Compute attack mask for a bishop (edges excluded)
Bitboard compute_bishop_mask(Square s) {
    Bitboard mask = 0;
    int f = file_of(s), r = rank_of(s);
    for (int ff = f + 1, rr = r + 1; ff < 7 && rr < 7; ++ff, ++rr)
        mask |= square_bb(make_square(File(ff), Rank(rr)));
    for (int ff = f - 1, rr = r + 1; ff > 0 && rr < 7; --ff, ++rr)
        mask |= square_bb(make_square(File(ff), Rank(rr)));
    for (int ff = f + 1, rr = r - 1; ff < 7 && rr > 0; ++ff, --rr)
        mask |= square_bb(make_square(File(ff), Rank(rr)));
    for (int ff = f - 1, rr = r - 1; ff > 0 && rr > 0; --ff, --rr)
        mask |= square_bb(make_square(File(ff), Rank(rr)));
    return mask;
}

// Compute rook attacks for given occupancy (slow but correct)
Bitboard slow_rook_attacks(Square s, Bitboard occupied) {
    Bitboard attacks = 0;
    int f = file_of(s), r = rank_of(s);
    for (int ff = f + 1; ff <= 7; ++ff) {
        Square sq = make_square(File(ff), Rank(r));
        attacks |= square_bb(sq);
        if (occupied & square_bb(sq)) break;
    }
    for (int ff = f - 1; ff >= 0; --ff) {
        Square sq = make_square(File(ff), Rank(r));
        attacks |= square_bb(sq);
        if (occupied & square_bb(sq)) break;
    }
    for (int rr = r + 1; rr <= 7; ++rr) {
        Square sq = make_square(File(f), Rank(rr));
        attacks |= square_bb(sq);
        if (occupied & square_bb(sq)) break;
    }
    for (int rr = r - 1; rr >= 0; --rr) {
        Square sq = make_square(File(f), Rank(rr));
        attacks |= square_bb(sq);
        if (occupied & square_bb(sq)) break;
    }
    return attacks;
}

// Compute bishop attacks for given occupancy (slow but correct)
Bitboard slow_bishop_attacks(Square s, Bitboard occupied) {
    Bitboard attacks = 0;
    int f = file_of(s), r = rank_of(s);
    for (int ff = f + 1, rr = r + 1; ff <= 7 && rr <= 7; ++ff, ++rr) {
        Square sq = make_square(File(ff), Rank(rr));
        attacks |= square_bb(sq);
        if (occupied & square_bb(sq)) break;
    }
    for (int ff = f - 1, rr = r + 1; ff >= 0 && rr <= 7; --ff, ++rr) {
        Square sq = make_square(File(ff), Rank(rr));
        attacks |= square_bb(sq);
        if (occupied & square_bb(sq)) break;
    }
    for (int ff = f + 1, rr = r - 1; ff <= 7 && rr >= 0; ++ff, --rr) {
        Square sq = make_square(File(ff), Rank(rr));
        attacks |= square_bb(sq);
        if (occupied & square_bb(sq)) break;
    }
    for (int ff = f - 1, rr = r - 1; ff >= 0 && rr >= 0; --ff, --rr) {
        Square sq = make_square(File(ff), Rank(rr));
        attacks |= square_bb(sq);
        if (occupied & square_bb(sq)) break;
    }
    return attacks;
}

// Magic multipliers, one per square. They are the numbers the deterministic
// per-square search (seeded xorshift, sparse candidates, verified over every
// occupancy subset) used to find at every start, which took seconds; the
// attack tables they build are identical.
constexpr Bitboard RookMagicNumbers[64] = {
    0x4080004000102080ULL, 0x8040004020001000ULL, 0x4A00084082001020ULL, 0x2900100005000920ULL,
    0x0600200422003008ULL, 0xE200100802000481ULL, 0x0200082281140A00ULL, 0x4200042301408204ULL,
    0x2020801020804000ULL, 0x0020802000904000ULL, 0x1000808020001000ULL, 0x01A0040081004200ULL,
    0x1002800400080080ULL, 0x0402000804020010ULL, 0xC011002900140200ULL, 0x00010020810016C2ULL,
    0x2010208010804002ULL, 0x4090004000402000ULL, 0x0020010010204101ULL, 0x0001010008100020ULL,
    0x0000808004000800ULL, 0x0094818002000400ULL, 0x10000400C8010210ULL, 0x4000020021008044ULL,
    0xB000800080204000ULL, 0x0040200040005000ULL, 0x060816C100200102ULL, 0x0880240900100100ULL,
    0x0625000500080090ULL, 0x040C000480800200ULL, 0x0400304400110882ULL, 0xBB402A4200010194ULL,
    0x8830400020800080ULL, 0x0060100040400020ULL, 0x0000200080801000ULL, 0x0290040811002100ULL,
    0x0080040801001100ULL, 0x4802800400800200ULL, 0x0005080204000110ULL, 0x2120009042000419ULL,
    0x0301008200460024ULL, 0x0090004020004000ULL, 0x08A0008010008020ULL, 0x20202201400A0010ULL,
    0x0901000408010010ULL, 0x000A000804010100ULL, 0x0180021190040028ULL, 0x1034208041120014ULL,
    0x4080004000200040ULL, 0x2000802100400100ULL, 0xC000600150008280ULL, 0x0080100008008280ULL,
    0x1040080004008080ULL, 0x4004004100020040ULL, 0x0028800100020080ULL, 0x8000800063000180ULL,
    0x8081002043120182ULL, 0x100B810140041021ULL, 0x0104102001010841ULL, 0x8B4200C038201016ULL,
    0x400A000408102002ULL, 0x0C02000108100482ULL, 0x1011004082000421ULL, 0x0001000031408601ULL,
};

constexpr Bitboard BishopMagicNumbers[64] = {
    0x8A10010801004208ULL, 0x01A00808A3104600ULL, 0xA290008200540400ULL, 0x0491040080402800ULL,
    0x18C40504003014A0ULL, 0x008A0A0220002001ULL, 0x2002209220300182ULL, 0x1840250100904040ULL,
    0x610040A212020200ULL, 0x4000488210840100ULL, 0x004010A08083000CULL, 0xA000044400804040ULL,
    0x00010404220041C0ULL, 0x00200A0910080000ULL, 0x900A040409082A40ULL, 0xC040044042101000ULL,
    0x1020003220011902ULL, 0x1009041002008401ULL, 0x0031020208070900ULL, 0x0042001040114000ULL,
    0x7402001420211484ULL, 0x0001000080414002ULL, 0x0002200048020850ULL, 0x0028440A86189004ULL,
    0x1244208090600104ULL, 0x2110242811110200ULL, 0x0004020010082442ULL, 0x1904002104020880ULL,
    0x1001010000104000ULL, 0x00040020D100A000ULL, 0x2808408001140100ULL, 0x0004008400220124ULL,
    0x0010307000260400ULL, 0x043A022208109040ULL, 0x4400211000010400ULL, 0x0882008021020200ULL,
    0x0900920200040108ULL, 0x0308020420041000ULL, 0x8821122204208800ULL, 0x8420810041010400ULL,
    0x0204020840000411ULL, 0x001100822010100AULL, 0x8140140024100800ULL, 0x1000002058000900ULL,
    0x4000080104002044ULL, 0x0911860802001540ULL, 0x0020020210402210ULL, 0x0002280241020290ULL,
    0x0841084202202000ULL, 0x002A044402090208ULL, 0x4800620042087020ULL, 0x0C00001042020580ULL,
    0x0400204090444000ULL, 0x10004048C141030CULL, 0xA422440400841000ULL, 0x0002280104108000ULL,
    0x2502410410022200ULL, 0x0400002484500840ULL, 0x0810000110880410ULL, 0x0819200008420210ULL,
    0x0008000040050104ULL, 0x6008041002100105ULL, 0x08080A1810540040ULL, 0x8102100202004200ULL,
};

// Initialize magic tables for one piece type
void init_magics(Magic magics[], Bitboard table[], bool is_rook) {
    Bitboard* table_ptr = table;

    for (int sq = 0; sq < 64; ++sq) {
        magics[sq].mask = is_rook ? compute_rook_mask(Square(sq)) : compute_bishop_mask(Square(sq));
        magics[sq].attacks = table_ptr;

        int bits = popcount(magics[sq].mask);
        magics[sq].shift = 64 - bits;
        int size = 1 << bits;

        Bitboard magic = is_rook ? RookMagicNumbers[sq] : BishopMagicNumbers[sq];
        magics[sq].magic = magic;

        // Now fill the table using the found magic
        Bitboard occ = 0;
        do {
            Bitboard attacks = is_rook ? slow_rook_attacks(Square(sq), occ)
                                       : slow_bishop_attacks(Square(sq), occ);

            unsigned index = unsigned((occ * magic) >> magics[sq].shift);
            table_ptr[index] = attacks;

            occ = (occ - magics[sq].mask) & magics[sq].mask;
        } while (occ);

        table_ptr += size;
    }
}

} // anonymous namespace

void init_magic_bitboards() {
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    init_magics(RookMagics, RookTable, true);
    init_magics(BishopMagics, BishopTable, false);
}

void init_line_tables() {
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    for (int s1 = 0; s1 < 64; ++s1) {
        for (int s2 = 0; s2 < 64; ++s2) {
            int f1 = file_of(Square(s1)), r1 = rank_of(Square(s1));
            int f2 = file_of(Square(s2)), r2 = rank_of(Square(s2));
            int df = f2 - f1, dr = r2 - r1;

            if (df == 0 && dr == 0) {
                LineBB[s1][s2] = BB_EMPTY;
                BetweenBB[s1][s2] = BB_EMPTY;
                continue;
            }

            bool is_aligned = (df == 0 || dr == 0 ||
                              (df < 0 ? -df : df) == (dr < 0 ? -dr : dr));

            if (!is_aligned) {
                LineBB[s1][s2] = BB_EMPTY;
                BetweenBB[s1][s2] = BB_EMPTY;
                continue;
            }

            Bitboard line = 0;
            int step_f = df == 0 ? 0 : (df > 0 ? 1 : -1);
            int step_r = dr == 0 ? 0 : (dr > 0 ? 1 : -1);

            Square s = Square(s1);
            while (s != Square(s2)) {
                s = Square(s + step_r * 8 + step_f);
                line |= square_bb(s);
            }
            line |= square_bb(Square(s1));
            LineBB[s1][s2] = line;
            BetweenBB[s1][s2] = line & ~(square_bb(Square(s1)) | square_bb(Square(s2)));
        }
    }
}

} // namespace luminex
