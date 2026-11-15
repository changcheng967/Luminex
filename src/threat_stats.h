// threat_stats.h -- LUMINEX_THREAT_STATS: the cost model of V5's threat inputs, measured inside the search.
// A relation is a piece on one square attacking a non-king piece of either colour on another square
// (sliders stop at the first piece; attacks on own pieces, i.e. defences, count too). Per evaluated
// position the number of relations is what a threat accumulator sums; per move the number that change
// (added or removed) is what an incremental update costs. A king move across the d/e file boundary flips
// that side's oriented (mirrored) frame, so every threat feature of that perspective changes; those moves
// are counted apart. Sampled every 8th call; main-thread counters, cumulative over the process.
#pragma once
#include "board.h"
#include "movegen.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace luminex::threat_stats {

inline bool on() {
    static const bool v = std::getenv("LUMINEX_THREAT_STATS") != nullptr;
    return v;
}

constexpr int MAX_REL = 512;
constexpr int MAX_DEPTH = 256;   // per-ply buffers for the relations before a move

// The position's relations as sorted keys: attacker square | victim square << 6 | attacker piece << 12
// | victim piece << 16.
inline int relations(const Position& pos, uint32_t* out) {
    int n = 0;
    const Bitboard occ = pos.pieces();
    const Bitboard victims = occ & ~pos.pieces(KING);
    for (Bitboard b = occ; b;) {
        const Square s = pop_lsb(b);
        const Piece p = pos.piece_on(s);
        Bitboard att;
        switch (piece_type_of(p)) {
            case PAWN:   att = pawn_attacks_bb(color_of_piece(p), s); break;
            case KNIGHT: att = knight_attacks_bb(s); break;
            case BISHOP: att = bishop_attacks_bb(s, occ); break;
            case ROOK:   att = rook_attacks_bb(s, occ); break;
            case QUEEN:  att = queen_attacks_bb(s, occ); break;
            default:     att = king_attacks_bb(s); break;
        }
        for (Bitboard t = att & victims; t && n < MAX_REL;) {
            const Square v = pop_lsb(t);
            out[n++] = uint32_t(s) | uint32_t(v) << 6 | uint32_t(p) << 12 | uint32_t(pos.piece_on(v)) << 16;
        }
    }
    std::sort(out, out + n);
    return n;
}

struct Stats {
    uint64_t ticks = 0;
    uint64_t pos_n = 0, pos_sum = 0, pos_hist[9] = {};          // relations a position: 0-9, 10-19, ..., 80+
    uint64_t mv_n[5] = {}, mv_sum[5] = {};                      // changes a move: quiet, capture, king, promotion, ep
    uint64_t mv_hist[8] = {};                                   // changes: 0-4, 5-9, 10-14, 15-19, 20-29, 30-39, 40-59, 60+
    uint64_t flips = 0;                                         // king moves across files d/e
    uint64_t enum_ns = 0, enum_n = 0;                           // a full enumeration's time
};
inline Stats& stats() {
    static thread_local Stats s;
    return s;
}

inline bool tick() { return on() && (++stats().ticks & 7) == 0; }

inline int timed_relations(const Position& pos, uint32_t* out) {
    const auto t0 = std::chrono::steady_clock::now();
    const int n = relations(pos, out);
    stats().enum_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    ++stats().enum_n;
    return n;
}

inline void position(const Position& pos) {
    uint32_t r[MAX_REL];
    const int n = timed_relations(pos, r);
    Stats& s = stats();
    ++s.pos_n;
    s.pos_sum += uint64_t(n);
    ++s.pos_hist[std::min(n / 10, 8)];
}

// The kind of a move, read before it is made: 0 quiet, 1 capture, 2 king (castling too), 3 promotion,
// 4 en passant; flip = a king crossing the d/e file boundary.
inline int kind(const Position& pos, Move m, bool& flip) {
    const Piece p = pos.piece_on(m.from());
    flip = false;
    if (piece_type_of(p) == KING) {
        flip = ((int(m.from()) & 7) < 4) != ((int(m.to()) & 7) < 4);
        return 2;
    }
    if (m.is_en_passant()) return 4;
    if (m.is_promotion()) return 3;
    return m.is_capture() ? 1 : 0;
}

// After the move: the changes between the relations before (sorted) and the position's now.
inline void move(const Position& after, int k, bool flip, const uint32_t* before, int nb) {
    uint32_t a[MAX_REL];
    const int na = relations(after, a);
    int i = 0, j = 0, changes = 0;
    while (i < nb && j < na) {
        if (before[i] == a[j]) { ++i; ++j; }
        else if (before[i] < a[j]) { ++i; ++changes; }
        else { ++j; ++changes; }
    }
    changes += (nb - i) + (na - j);
    Stats& s = stats();
    ++s.mv_n[k];
    s.mv_sum[k] += uint64_t(changes);
    const int b = changes < 20 ? changes / 5 : changes < 30 ? 4 : changes < 40 ? 5 : changes < 60 ? 6 : 7;
    ++s.mv_hist[b];
    if (flip) ++s.flips;
}

inline std::string report() {
    const Stats& s = stats();
    uint64_t mn = 0, ms = 0;
    for (int k = 0; k < 5; ++k) { mn += s.mv_n[k]; ms += s.mv_sum[k]; }
    auto avg = [](uint64_t a, uint64_t b) { return b ? double(a) / double(b) : 0.0; };
    char buf[1024];
    std::snprintf(buf, sizeof buf,
        "info string THREATS positions %llu: %.1f relations each (0-9/10-19/.../80+: %llu/%llu/%llu/%llu/%llu/%llu/%llu/%llu/%llu)"
        " | moves %llu: %.1f changes each; quiet %.1f, capture %.1f, king %.1f, promotion %.1f, ep %.1f"
        " (shares %.1f/%.1f/%.1f/%.2f/%.2f%%); 0-4/5-9/10-14/15-19/20-29/30-39/40-59/60+: %llu/%llu/%llu/%llu/%llu/%llu/%llu/%llu"
        " | king moves across d/e: %.2f%% of moves | enumeration %.0f ns\n",
        (unsigned long long)s.pos_n, avg(s.pos_sum, s.pos_n),
        (unsigned long long)s.pos_hist[0], (unsigned long long)s.pos_hist[1], (unsigned long long)s.pos_hist[2],
        (unsigned long long)s.pos_hist[3], (unsigned long long)s.pos_hist[4], (unsigned long long)s.pos_hist[5],
        (unsigned long long)s.pos_hist[6], (unsigned long long)s.pos_hist[7], (unsigned long long)s.pos_hist[8],
        (unsigned long long)mn, avg(ms, mn), avg(s.mv_sum[0], s.mv_n[0]), avg(s.mv_sum[1], s.mv_n[1]),
        avg(s.mv_sum[2], s.mv_n[2]), avg(s.mv_sum[3], s.mv_n[3]), avg(s.mv_sum[4], s.mv_n[4]),
        100.0 * avg(s.mv_n[0], mn), 100.0 * avg(s.mv_n[1], mn), 100.0 * avg(s.mv_n[2], mn), 100.0 * avg(s.mv_n[3], mn),
        100.0 * avg(s.mv_n[4], mn),
        (unsigned long long)s.mv_hist[0], (unsigned long long)s.mv_hist[1], (unsigned long long)s.mv_hist[2],
        (unsigned long long)s.mv_hist[3], (unsigned long long)s.mv_hist[4], (unsigned long long)s.mv_hist[5],
        (unsigned long long)s.mv_hist[6], (unsigned long long)s.mv_hist[7],
        100.0 * avg(s.flips, mn), avg(s.enum_ns, s.enum_n));
    return std::string(buf);
}

}  // namespace luminex::threat_stats
