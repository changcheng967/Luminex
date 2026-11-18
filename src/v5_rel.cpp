// v5_rel.cpp -- see v5_rel.h. Built as its own translation unit so that the prototype can compile it at another
// instruction-set level than the rest (CMake V5_REL_MARCH), as the release build compiles everything but the
// network kernels for x86-64-v2.
#include "v5_rel.h"
#include "bitboard.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace luminex::v5rel {

using u64 = uint64_t;
namespace {
constexpr int NOPC = 12;
constexpr u64 RANKS27 = 0x00FFFFFFFFFFFF00ULL;

inline int pop(u64& b) {
    const int s = __builtin_ctzll(b);
    b &= b - 1;
    return s;
}
inline int popc(u64 b) { return __builtin_popcountll(b); }
inline u64 bit(int s) { return 1ULL << s; }
inline bool same_diag(int x, int y) { return std::abs((x & 7) - (y & 7)) == std::abs((x >> 3) - (y >> 3)); }

// Tables, generated from the rules (no constant is typed in).
struct Tables {
    u64 below[64], beyond[64][64], files3[8];
    u64 e_pup[64], e_pdn[64], e_n[64], e_k[64], e_d[64], e_o[64];   // empty-board attack sets; "up" = toward rank 8
    u64 patt[2][64];                                                // squares a pawn of colour c on s attacks
    int16_t blk[2][6][2][5];                                        // [attacker side][type][victim side][type] -> block
    int n_blk = 0, n_rel = 0, n_pp = 0;
    bool reduced = false;        // LUMINEX_REL_SET=1: only attacks on the other colour, pawn attackers and pawn pairs
    u64 mask[128][64];
    int boff[128][64];                                              // the block's base + the square's offset
    int16_t pp[96][96];
    struct Cls { int16_t blk; uint8_t swap, unordered; } cls[2][12][12];   // [perspective][attacker piece][victim piece]

    static int step(int s, int df, int dr) {
        const int f = (s & 7) + df, r = (s >> 3) + dr;
        return (f < 0 || f > 7 || r < 0 || r > 7) ? -1 : r * 8 + f;
    }
    static u64 pawn_facing(int x, int sv) {      // squares from which a pawn of side sv (0 = own, moving up) attacks x
        u64 m = 0;
        for (int df = -1; df <= 1; df += 2) {
            const int t = step(x, df, sv == 0 ? -1 : 1);
            if (t >= 0) m |= bit(t);
        }
        return m & RANKS27;
    }
    u64 block_mask(int sa, int ta, int sv, int tv, int x) const {
        if (ta == 5 && sa == 0) return 0;        // the own king's attacks: the king-bucketed piece-squares carry them
        if (ta == 0 && tv == 0) return 0;        // pawn on pawn: the pawn-pair group
        u64 m;
        switch (ta) {
            case 0:  m = sa == 0 ? e_pup[x] : e_pdn[x]; break;
            case 1:  m = e_n[x]; break;
            case 2:  m = e_d[x]; break;
            case 3:  m = e_o[x]; break;
            case 4:  m = tv == 2 ? e_o[x] : tv == 3 ? e_d[x] : (e_d[x] | e_o[x]); break;
            default: m = e_k[x]; break;
        }
        if (tv == 0) {
            m &= RANKS27;
            if (ta == 2 || ta == 4) m &= ~pawn_facing(x, sv);
        }
        if (ta == tv && ta >= 1 && ta <= 4) {
            if (sa == sv) m &= ~(bit(x) | below[x]);
            else if (sa == 1) return 0;
        }
        return m;
    }
    explicit Tables(bool reduced_set) : reduced(reduced_set) {
        static const int KN[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
        static const int KD[8][2] = {{1, 1}, {-1, 1}, {-1, -1}, {1, -1}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        std::memset(beyond, 0, sizeof beyond);
        for (int s = 0; s < 64; ++s) {
            below[s] = bit(s) - 1;
            e_pup[s] = e_pdn[s] = e_n[s] = e_k[s] = e_d[s] = e_o[s] = 0;
            patt[0][s] = patt[1][s] = 0;
            for (int df = -1; df <= 1; df += 2) {
                int t = step(s, df, 1);
                if (t >= 0) { patt[0][s] |= bit(t); if (s >= 8 && s < 56) e_pup[s] |= bit(t); }
                t = step(s, df, -1);
                if (t >= 0) { patt[1][s] |= bit(t); if (s >= 8 && s < 56) e_pdn[s] |= bit(t); }
            }
            for (const auto& d : KN) {
                const int t = step(s, d[0], d[1]);
                if (t >= 0) e_n[s] |= bit(t);
            }
            for (int i = 0; i < 8; ++i) {
                int ray[8], n = 0;
                for (int t = step(s, KD[i][0], KD[i][1]); t >= 0; t = step(t, KD[i][0], KD[i][1])) ray[n++] = t;
                if (n) e_k[s] |= bit(ray[0]);
                for (int a = 0; a < n; ++a) {
                    (i < 4 ? e_d[s] : e_o[s]) |= bit(ray[a]);
                    for (int c = a + 1; c < n; ++c) beyond[s][ray[a]] |= bit(ray[c]);
                }
            }
        }
        for (int f = 0; f < 8; ++f) {
            files3[f] = 0;
            for (int g = std::max(0, f - 1); g <= std::min(7, f + 1); ++g) files3[f] |= 0x0101010101010101ULL << g;
        }
        for (int sa = 0; sa < 2; ++sa)
            for (int ta = 0; ta < 6; ++ta)
                for (int sv = 0; sv < 2; ++sv)
                    for (int tv = 0; tv < 5; ++tv) {
                        int size = 0;
                        for (int x = 0; x < 64; ++x) size += popc(block_mask(sa, ta, sv, tv, x));
                        if (!size) { blk[sa][ta][sv][tv] = -1; continue; }
                        const int b = n_blk++;
                        blk[sa][ta][sv][tv] = int16_t(b);
                        int acc = 0;
                        for (int x = 0; x < 64; ++x) {
                            mask[b][x] = block_mask(sa, ta, sv, tv, x);
                            boff[b][x] = n_rel + acc;
                            acc += popc(mask[b][x]);
                        }
                        n_rel += size;
                    }
        for (int i = 0; i < 96; ++i)
            for (int j = 0; j < 96; ++j) {
                pp[i][j] = -1;
                if (j <= i) continue;
                const int qi = i % 48, qj = j % 48;
                if (qi == qj || std::abs(((qi + 8) & 7) - ((qj + 8) & 7)) > 1) continue;
                pp[i][j] = int16_t(n_pp++);
            }
        for (int persp = 0; persp < 2; ++persp)
            for (int pa = 0; pa < 12; ++pa)
                for (int pv = 0; pv < 12; ++pv) {
                    Cls& c = cls[persp][pa][pv];
                    const int ta = pa % 6, tv = pv % 6;
                    int sa = (pa / 6) != persp, sv = (pv / 6) != persp;
                    c.swap = c.unordered = 0;
                    c.blk = -1;
                    if (tv == 5) continue;
                    if (ta == tv && ta >= 1 && ta <= 4) {
                        if (sa == sv) c.unordered = 1;
                        else if (sa == 1) { c.swap = 1; std::swap(sa, sv); }
                    }
                    c.blk = blk[sa][ta][sv][tv];
                    if (reduced && sa == sv && ta != 0) c.blk = -1;   // the reduced set: no piece on its own side's pieces
                }
    }
};
const Tables* g_t = nullptr;

inline void put(Bd& b, int s, int p) {
    b.pc[s] = uint8_t(p);
    b.by[p] |= bit(s);
    b.occ |= bit(s);
}
inline void take(Bd& b, int s) {
    b.by[b.pc[s]] &= ~bit(s);
    b.occ &= ~bit(s);
    b.pc[s] = NOPC;
}

// Is the attack x -> y (known to hold on the board) a member of the canonical set?
inline bool rel_ok(const Tables& t, int pa, int x, int pv, int y) {
    const int ta = pa % 6, tv = pv % 6;
    if (tv == 5) return false;
    if (ta == 0 && tv == 0) return false;
    if (ta == tv && ta >= 1 && ta <= 4) return x < y;
    if (ta == 4 && tv == 2 && same_diag(x, y)) return false;
    if (ta == 4 && tv == 3 && !same_diag(x, y)) return false;
    if ((ta == 2 || ta == 4) && tv == 0 && (t.patt[pv >= 6][y] & bit(x))) return false;
    return true;
}

inline int rel_row(const Tables& t, int p, int ox, int x, int y, int px, int py) {
    const Tables::Cls& c = t.cls[p][px][py];
    if (c.blk < 0) return -1;
    int xo = x ^ ox, yo = y ^ ox;
    if (c.swap || (c.unordered && xo > yo)) std::swap(xo, yo);
    return t.boff[c.blk][xo] + popc(t.mask[c.blk][xo] & (bit(yo) - 1));
}
inline int pair_row(const Tables& t, int p, int ox, int s1, int s2, int p1, int p2) {
    int i = ((p1 / 6) != p) * 48 + (s1 ^ ox) - 8, j = ((p2 / 6) != p) * 48 + (s2 ^ ox) - 8;
    if (i > j) std::swap(i, j);
    return t.n_rel + t.pp[i][j];
}
inline int orient_of(const Bd& b, int p) {       // XOR 56 for Black, XOR 7 when that side's king is on files a-d
    return (p ? 56 : 0) ^ ((__builtin_ctzll(b.by[p ? 11 : 5]) & 7) < 4 ? 7 : 0);
}

struct Finder {
    Bd& b;
    const Tables& t;
    const int ppm;
    const int ox[2];
    const bool need[2];
    RowLists& o;

    void emit(int sign, int p, int row) {
        if (sign > 0) {
            if (o.na[p] < MAX_ROWS) o.add[p][o.na[p]++] = uint16_t(row); else o.overflow[p] = true;
        } else {
            if (o.ns[p] < MAX_ROWS) o.sub[p][o.ns[p]++] = uint16_t(row); else o.overflow[p] = true;
        }
    }
    void rel(int sign, int x, int y, int px, int py) {
        for (int p = 0; p < 2; ++p) {
            if (!need[p]) continue;
            const int row = rel_row(t, p, ox[p], x, y, px, py);
            if (row >= 0) emit(sign, p, row);
        }
    }
    void pair(int sign, int s1, int s2) {
        if (s1 > s2) std::swap(s1, s2);
        for (int p = 0; p < 2; ++p)
            if (need[p]) emit(sign, p, pair_row(t, p, ox[p], s1, s2, b.pc[s1], b.pc[s2]));
    }
    // Every feature the piece on z takes part in, as attacker, victim or pawn of a pair.
    // dz, oz: the diagonal and line attack sets from z (they do not depend on what stands on z).
    void piece(int z, int sign, u64 dz, u64 oz) {
        const int pz = b.pc[z], tz = pz % 6;
        const u64 pawns = b.by[0] | b.by[6], knights = b.by[1] | b.by[7], bishops = b.by[2] | b.by[8],
                  rooks = b.by[3] | b.by[9], queens = b.by[4] | b.by[10], kings = b.by[5] | b.by[11];
        const u64 pawn_att = (t.patt[1][z] & b.by[0]) | (t.patt[0][z] & b.by[6]);   // pawns attacking z
        const u64 own_att = tz == 0 ? t.patt[pz >= 6][z] : 0;                        // what a pawn on z attacks
        const u64 lo = t.below[z];
        // the reduced set: a piece's relations with the other colour only (a pawn attacker keeps both)
        const u64 white = b.by[0] | b.by[1] | b.by[2] | b.by[3] | b.by[4] | b.by[5];
        const u64 other = pz < 6 ? ~white : white;
        u64 v;
        switch (tz) {                    // z as attacker: victims, with every rule of the canonical set as a mask
            case 0:  v = own_att & b.occ & ~pawns; break;
            case 1:  v = t.e_n[z] & b.occ & ~(knights & lo); break;
            case 2:  v = dz & b.occ & ~(bishops & lo) & ~pawn_att; break;
            case 3:  v = oz & b.occ & ~(rooks & lo); break;
            case 4:  v = ((dz & ~bishops) | (oz & ~rooks)) & b.occ & ~(queens & lo) & ~pawn_att; break;
            default: v = t.e_k[z] & b.occ; break;
        }
        v &= ~kings;
        if (t.reduced && tz != 0) v &= other;
        while (v) {
            const int y = pop(v);
            rel(sign, z, y, pz, b.pc[y]);
        }
        if (tz != 5) {                   // z as victim: attackers
            const u64 nA = t.e_n[z] & knights, kA = t.e_k[z] & kings;
            const u64 dB = dz & bishops, dQ = dz & queens, oR = oz & rooks, oQ = oz & queens;
            u64 a;
            switch (tz) {
                case 0:  a = nA | kA | ((dB | dQ) & ~own_att) | oR | oQ; break;
                case 1:  a = pawn_att | (nA & lo) | kA | dB | dQ | oR | oQ; break;
                case 2:  a = pawn_att | nA | kA | (dB & lo) | oR | oQ; break;
                case 3:  a = pawn_att | nA | kA | dB | dQ | (oR & lo); break;
                default: a = pawn_att | nA | kA | dB | oR | ((dQ | oQ) & lo); break;
            }
            if (t.reduced) a &= other | pawns;
            while (a) {
                const int x = pop(a);
                rel(sign, x, z, b.pc[x], pz);
            }
        }
        if (tz == 0)
            for (u64 c = ppm ? (pawns & t.files3[z & 7] & ~bit(z)) : ((own_att & pawns) | pawn_att); c;) pair(sign, z, pop(c));
    }
    // With z empty: (slider -> the first piece beyond z) for every slider whose line reaches z.
    void through(int z, int sign, u64 dz, u64 oz) {
        const u64 queens = b.by[4] | b.by[10];
        const u64 ends = (dz | oz) & b.occ;
        for (u64 s = (dz & (b.by[2] | b.by[8] | queens)) | (oz & (b.by[3] | b.by[9] | queens)); s;) {
            const int x = pop(s);
            const u64 w = ends & t.beyond[x][z];
            if (!w) continue;
            const int y = __builtin_ctzll(w);
            if (rel_ok(t, b.pc[x], x, b.pc[y], y)) rel(sign, x, y, b.pc[x], b.pc[y]);
        }
    }
    void remove(int z) {
        const u64 dz = bishop_attacks_bb(Square(z), b.occ), oz = rook_attacks_bb(Square(z), b.occ);
        piece(z, -1, dz, oz);
        take(b, z);
        through(z, +1, dz, oz);
    }
    void add(int z, int p) {
        const u64 dz = bishop_attacks_bb(Square(z), b.occ), oz = rook_attacks_bb(Square(z), b.occ);
        through(z, -1, dz, oz);
        put(b, z, p);
        piece(z, +1, dz, oz);
    }
    void replace(int z, int p) {
        const u64 dz = bishop_attacks_bb(Square(z), b.occ), oz = rook_attacks_bb(Square(z), b.occ);
        piece(z, -1, dz, oz);
        take(b, z);
        put(b, z, p);
        piece(z, +1, dz, oz);
    }
    void move(Move m) {
        const int from = int(m.from()), to = int(m.to()), pm = b.pc[from], us = pm / 6;
        if (m.is_castling()) {
            const bool kingside = (to & 7) == 6;
            const int rank = from & 56, rf = rank + (kingside ? 7 : 0), rt = rank + (kingside ? 5 : 3);
            remove(from);
            remove(rf);
            add(to, us * 6 + 5);
            add(rt, us * 6 + 3);
            return;
        }
        const int np = m.is_promotion() ? us * 6 + int(m.promotion_type()) : pm;
        remove(from);
        if (m.is_en_passant()) {
            remove(to + (us == 0 ? -8 : 8));
            add(to, np);
        } else if (b.pc[to] != NOPC) replace(to, np);
        else add(to, np);
    }
};
}  // namespace

void init() {
    const char* e = std::getenv("LUMINEX_REL_SET");
    static const Tables t(e && std::atoi(e) == 1);
    g_t = &t;
}
int n_rows() { return g_t->n_rel + g_t->n_pp; }

__attribute__((noinline)) void find(const Bd& before, Move m, int ppm, const bool need[2], RowLists& out) {
    Bd b = before;
    out.na[0] = out.na[1] = out.ns[0] = out.ns[1] = 0;
    out.overflow[0] = out.overflow[1] = false;
    Finder f{b, *g_t, ppm, {orient_of(before, 0), orient_of(before, 1)}, {need[0], need[1]}, out};
    f.move(m);
}

__attribute__((noinline)) int enumerate(const Bd& b, int persp, int ppm, uint16_t* out) {
    const Tables& t = *g_t;
    const int ox = orient_of(b, persp);
    const u64 pawns = b.by[0] | b.by[6];
    int n = 0;
    for (u64 o = b.occ; o;) {
        const int s = pop(o), p = b.pc[s];
        u64 att;
        switch (p % 6) {
            case 0:  att = t.patt[p >= 6][s]; break;
            case 1:  att = t.e_n[s]; break;
            case 2:  att = bishop_attacks_bb(Square(s), b.occ); break;
            case 3:  att = rook_attacks_bb(Square(s), b.occ); break;
            case 4:  att = bishop_attacks_bb(Square(s), b.occ) | rook_attacks_bb(Square(s), b.occ); break;
            default: att = t.e_k[s]; break;
        }
        for (u64 v = att & b.occ; v;) {
            const int y = pop(v);
            if (!rel_ok(t, p, s, b.pc[y], y)) continue;
            const int row = rel_row(t, persp, ox, s, y, p, b.pc[y]);
            if (row >= 0 && n < MAX_ACTIVE) out[n++] = uint16_t(row);
        }
        if (p % 6 == 0) {
            const u64 partners = ppm ? (pawns & t.files3[s & 7]) : ((t.patt[p >= 6][s] & pawns) | (t.patt[1][s] & b.by[0]) | (t.patt[0][s] & b.by[6]));
            for (u64 c = partners & ~(bit(s) | t.below[s]); c;) {
                const int y = pop(c);
                if (n < MAX_ACTIVE) out[n++] = uint16_t(pair_row(t, persp, ox, s, y, p, b.pc[y]));
            }
        }
    }
    return n;
}

}  // namespace luminex::v5rel
