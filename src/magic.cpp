#include "luminex.h"

#if defined(__BMI2__) && defined(__x86_64__)
#include <immintrin.h>
#endif

namespace luminex {

// Precomputed line and between tables
Bitboard LineBB[64][64];
Bitboard BetweenBB[64][64];

// Magic bitboard tables
Magic RookMagics[64];
Magic BishopMagics[64];
Bitboard RookTable[262144];   // Large enough for all rook entries
Bitboard BishopTable[65536];  // Large enough for all bishop entries

#if defined(__BMI2__) && defined(__x86_64__)
Bitboard RookPEXT[64][4096];
Bitboard BishopPEXT[64][512];
#endif

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

// PRNG for magic number search
class PRNG {
    uint64_t s;
public:
    explicit PRNG(uint64_t seed) : s(seed) {}
    uint64_t next64() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 2685821657736338717ULL;
    }
    Bitboard sparse64() { return next64() & next64() & next64(); }
};

// Verify a magic number works for all occupancy subsets
bool verify_magic(Square sq, Bitboard mask, Bitboard magic, int shift, Bitboard* table, int size,
                  bool is_rook) {
    // Use a temporary used[] array
    auto* used = new bool[size];
    std::memset(used, 0, size * sizeof(bool));

    Bitboard occ = 0;
    bool ok = true;
    do {
        Bitboard attacks = is_rook ? slow_rook_attacks(sq, occ) : slow_bishop_attacks(sq, occ);
        unsigned idx = unsigned((occ * magic) >> shift);

        if (used[idx]) {
            if (table[idx] != attacks) { ok = false; break; }
        } else {
            used[idx] = true;
            table[idx] = attacks;
        }

        occ = (occ - mask) & mask;
    } while (occ);

    delete[] used;
    return ok;
}

// Search for a magic number for a given square
Bitboard search_magic(Square sq, Bitboard mask, bool is_rook) {
    int bits = popcount(mask);
    int size = 1 << bits;
    unsigned shift = 64 - bits;

    auto* table = new Bitboard[size];

    PRNG rng(uint64_t(0x12345678) + uint64_t(sq) * 0x11111111);

    for (int tries = 0; tries < 100000000; ++tries) {
        Bitboard magic = rng.sparse64();

        if (popcount((magic * mask) & 0xFF00000000000000ULL) < 6) continue;

        if (verify_magic(sq, mask, magic, shift, table, size, is_rook)) {
            delete[] table;
            return magic;
        }
    }

    delete[] table;
    return 0; // Should never happen
}

// Initialize magic tables for one piece type
void init_magics(Magic magics[], Bitboard table[], bool is_rook) {
    Bitboard* table_ptr = table;

    for (int sq = 0; sq < 64; ++sq) {
        magics[sq].mask = is_rook ? compute_rook_mask(Square(sq)) : compute_bishop_mask(Square(sq));
        magics[sq].attacks = table_ptr;

        int bits = popcount(magics[sq].mask);
        magics[sq].shift = 64 - bits;
        int size = 1 << bits;

        // Search for magic number
        Bitboard magic = search_magic(Square(sq), magics[sq].mask, is_rook);
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

#if defined(__BMI2__) && defined(__x86_64__)
    // Build PEXT tables from magic masks using slow attack generation
    for (int sq = 0; sq < 64; ++sq) {
        Bitboard mask = RookMagics[sq].mask;
        Bitboard occ = 0;
        do {
            RookPEXT[sq][_pext_u64(occ, mask)] = slow_rook_attacks(Square(sq), occ);
            occ = (occ - mask) & mask;
        } while (occ);
    }
    for (int sq = 0; sq < 64; ++sq) {
        Bitboard mask = BishopMagics[sq].mask;
        Bitboard occ = 0;
        do {
            BishopPEXT[sq][_pext_u64(occ, mask)] = slow_bishop_attacks(Square(sq), occ);
            occ = (occ - mask) & mask;
        } while (occ);
    }
#endif
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
