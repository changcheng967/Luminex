// ============================================================================
// luminex-evaltrace — sparse feature decomposition of the HCE evaluation.
//
// Purpose: HCE distillation tuning. Reads FENs from stdin (one per line;
// "fen\tstm\ttarget" lines also accepted, extra fields ignored), emits ONE
// line per position:
//
//   <phase> <sf> <stm> <mirror_eval_white> <idx>:<val> <idx>:<val> ...
//
//   phase : 0..24 game phase (same formula as evaluate())
//   sf    : scale_factor (EG block enters the blend as (24-ph)/24 * sf/32)
//   stm   : 1 = white to move
//   mirror: int-exact mirror of evaluate()'s RETURN (stm-relative). The fit
//           flips to white perspective: y_white = mirror * (stm ? 1 : -1);
//           features are white-perspective so the flip applies to y only.
//   idx:val: sparse white-perspective feature counts. idx < NPHASE = MG
//           block, [NPHASE, 2*NPHASE) = EG block, then TEMPO_MG/TEMPO_EG.
//
// FIDELITY CONTRACT: mirrors evaluate() (evaluation.cpp) term-for-term with
// the SAME integer operations in the SAME order, so the mirror accumulator
// is bit-identical to the engine. Every add site also emits a feature;
// --verify checks BOTH invariants over the input:
//   (a) mirror == evaluate(pos) exactly            (structure is faithful)
//   (b) dot(features, current coefs) == mirror <=2 (features are complete)
//
// Positions hitting the non-linear early returns (KPK bitbase, KXK) are not
// linear-representable: printed as "<ph> <sf> <stm> G" so the fit drops them.
// ============================================================================
#include "luminex.h"
#include "evaluation.h"
#include "eval_feat.h"
#include "eval_fitted.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace luminex {

namespace etrace {
int* piece_value_mg();  int* piece_value_eg();
int* knight_mob_mg();   int* knight_mob_eg();
int* bishop_mob_mg();   int* bishop_mob_eg();
int* rook_mob_mg();     int* rook_mob_eg();
int* queen_mob_mg();    int* queen_mob_eg();
const int* pawn_threat_mg();   const int* pawn_threat_eg();
const int* knight_threat_mg(); const int* knight_threat_eg();
const int* bishop_threat_mg(); const int* bishop_threat_eg();
const int* rook_threat_mg();   const int* rook_threat_eg();
const int* queen_threat_mg();  const int* queen_threat_eg();
const int* phalanx_mg(); const int* phalanx_eg();
const int* cand_mg();    const int* cand_eg();
const int* passed_mg();  const int* passed_eg();
}


struct Tracer {
    int f[feat::NFEAT] = {0};
    int touched[feat::NFEAT];
    bool listed[feat::NFEAT] = {false};
    int nt = 0;
    int mmg = 0, meg = 0;   // int-exact mirror of evaluate()'s accumulators

    // Membership is tracked explicitly: a feature whose running sum passes
    // through 0 (e.g. MAT+KNIGHT: +1 white, -1 black, -1 black) must NOT be
    // re-appended to touched[] — duplicates double-count in reconstruct().
    inline void add(int idx, int v) {
        if (v == 0) return;
        if (!listed[idx]) { listed[idx] = true; touched[nt++] = idx; }
        f[idx] += v;
    }
    void reset() {
        for (int i = 0; i < nt; ++i) { f[touched[i]] = 0; listed[touched[i]] = false; }
        nt = 0; mmg = 0; meg = 0;
    }
};

struct TraceOut { int ph = 0, sf = 32, stm = 1, mirror_white = 0; bool linear = true; };

// ============================================================================
// --solve mode: closed-form ridge accumulation, in-process.
//
// Row model (cp space, white perspective):
//   y    = target * (stm ? 1 : -1)              [SF eval, white-relative]
//   r    = y - tempo                            [tempo is fixed, not a fit param]
//   x_j  = f_j * ph / 24                (j in MG block)
//        = f_j * (24-ph) * sf / 768    (j in EG block; = (24-ph)/24 * sf/32)
//   predict: sum_j c_j x_j ~= r  ->  normal equations A = X'X, b = X'r over all rows.
// TEMPO features are excluded from the parameter vector. Writes binary
// <out>.bin = [int64 N][double A[NPF][NPF]][double b[NPF]] with NPF = 2*NPHASE.
// ============================================================================
constexpr int NSOLVE = 2 * feat::NPHASE;   // MG block (0..606) + EG block (607..1213)
struct SolverAcc {
    std::vector<double> A, b;     // A row-major NSOLVE*NSOLVE
    long long n = 0;
    double sum_r = 0, sum_r2 = 0; // train-side target stats: R^2 = 1 - SSE/ (sum_r2 - sum_r^2/n)
    SolverAcc() : A((size_t)NSOLVE * NSOLVE, 0.0), b(NSOLVE, 0.0) {}
    void add(const Tracer& t, int ph, int sf, double r, int* sidx, double* sxv, int& snz) {
        // build the weighted sparse row (MG then EG slots in solver space)
        snz = 0;
        double wmg = ph / 24.0;
        double weg = (24 - ph) * sf / 768.0;
        for (int i = 0; i < t.nt; ++i) {
            int idx = t.touched[i];
            int v = t.f[idx];
            if (v == 0) continue;
            // solver index space = feature index space (MG 0..606, EG 607..1213)
            if (idx < feat::NPHASE)          { sidx[snz] = idx; sxv[snz] = v * wmg; ++snz; }
            else if (idx < 2 * feat::NPHASE) { sidx[snz] = idx; sxv[snz] = v * weg; ++snz; }
        }
        // touched[] is in ADD order, not sorted. The outer product below only
        // fills A[min][max] (true upper triangle); an unsorted row would drop
        // every product whose later-added index is numerically smaller — the
        // mirror at flush would then overwrite them. Sort (idx,val) pairs first.
        for (int i = 1; i < snz; ++i) {
            int ti = sidx[i]; double tv = sxv[i];
            int j = i - 1;
            while (j >= 0 && sidx[j] > ti) { sidx[j + 1] = sidx[j]; sxv[j + 1] = sxv[j]; --j; }
            sidx[j + 1] = ti; sxv[j + 1] = tv;
        }
        for (int i = 0; i < snz; ++i) {
            int j = sidx[i];
            double xi = sxv[i];
            b[j] += xi * r;
            double* row = &A[(size_t)j * NSOLVE];
            for (int k = i; k < snz; ++k) row[sidx[k]] += xi * sxv[k];   // upper triangle
        }
        sum_r += r; sum_r2 += r * r;
        ++n;
    }
};

// Mirror of evaluate_pawns (engine caches in pawn_table; values identical).
static void trace_pawns(const Position& pos, Tracer& t, int32_t& mg_out, int32_t& eg_out) {
    using namespace feat;
    const int* PVMG = etrace::piece_value_mg();
    const int* PVEG = etrace::piece_value_eg();
    const int* PHMG = etrace::phalanx_mg();
    const int* PHEG = etrace::phalanx_eg();
    const int* CAMG = etrace::cand_mg();
    const int* CAEG = etrace::cand_eg();
    const int* PAMG = etrace::passed_mg();
    const int* PAEG = etrace::passed_eg();
    int32_t mg = 0;
    int32_t eg = 0;

    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color c = Color(c_idx);
        Color them = Color(c_idx ^ 1);
        int sign = (c == WHITE) ? 1 : -1;
        Bitboard our_pawns = pos.pieces(c, PAWN);
        Bitboard their_pawns = pos.pieces(them, PAWN);

        int file_count[8] = {0};
        Bitboard tmp = our_pawns;
        while (tmp) { file_count[file_of(pop_lsb(tmp))]++; }

        Bitboard supported_by_adj = pawn_attacks_bb(c, our_pawns);

        Bitboard pawns = our_pawns;
        while (pawns) {
            Square sq = pop_lsb(pawns);
            File f = file_of(sq);
            Rank r = relative_rank(c, sq);
            int relsq = (c == WHITE) ? int(sq) : (int(sq) ^ 56);

            // Material + PST
            mg += sign * (PVMG[PAWN] + PST_MG_TABLE[int(c)][int(PAWN)][int(sq)]);
            eg += sign * (PVEG[PAWN] + PST_EG_TABLE[int(c)][int(PAWN)][int(sq)]);
            t.add(PST + int(PAWN) * 64 + relsq, sign);
            t.add(PST + int(PAWN) * 64 + relsq + NPHASE, sign);
            t.add(MAT + int(PAWN), sign);
            t.add(MAT + int(PAWN) + NPHASE, sign);

            // Doubled pawn
            if (file_count[f] > 1) {
                mg += sign * FE_MG[PAWN_DOUBLED];
                eg += sign * FE_EG[PAWN_DOUBLED];
                t.add(PAWN_DOUBLED, sign); t.add(PAWN_DOUBLED + NPHASE, sign);
            }

            // Isolated pawn
            bool left = (f > FILE_A && file_count[f - 1] > 0);
            bool right = (f < FILE_H && file_count[f + 1] > 0);
            if (!left && !right) {
                mg += sign * FE_MG[PAWN_ISOLATED];
                eg += sign * FE_EG[PAWN_ISOLATED];
                t.add(PAWN_ISOLATED, sign); t.add(PAWN_ISOLATED + NPHASE, sign);
            }

            // Connected pawn (protected by another pawn)
            if (square_bb(sq) & supported_by_adj) {
                mg += sign * FE_MG[PAWN_CONNECTED + r - 1];
                eg += sign * FE_EG[PAWN_CONNECTED + r - 1];
                t.add(PAWN_CONNECTED + (r - 1), sign);
                t.add(PAWN_CONNECTED + (r - 1) + NPHASE, sign);
            }

            // Phalanx
            {
                Bitboard neighbors = our_pawns & (file_bb(File(f - 1)) | file_bb(File(f + 1)));
                Bitboard same_rank = neighbors & rank_bb(rank_of(sq));
                if (same_rank) {
                    if (r >= RANK_2 && r <= RANK_7) {
                        mg += sign * PHMG[r];
                        eg += sign * PHEG[r];
                        t.add(PAWN_PHALANX + (r - 1), sign);
                        t.add(PAWN_PHALANX + (r - 1) + NPHASE, sign);
                    }
                }
            }

            // Lever
            {
                Bitboard lever_targets = pawn_attacks_bb(c, sq) & their_pawns;
                if (lever_targets) {
                    mg += sign * FE_MG[PAWN_LEVER + r - 1];
                    t.add(PAWN_LEVER + (r - 1), sign);
                    if (f >= FILE_C && f <= FILE_F) {
                        mg += sign * FE_MG[PAWN_LEVER_C];
                        t.add(PAWN_LEVER_C, sign);
                    }
                }
            }

            // Backward pawn
            {
                Square push = relative_square(c, make_square(f, Rank(r + 1)));
                if (push < SQUARE_NONE && r >= RANK_2 && r <= RANK_5) {
                    bool push_attacked = (pawn_attacks_bb(them, their_pawns) & square_bb(push)) != 0;
                    if (push_attacked) {
                        Bitboard our_pawn_attacks = pawn_attacks_bb(c, our_pawns);
                        bool defended = (our_pawn_attacks & square_bb(push)) != 0;
                        if (!defended) {
                            mg += sign * FE_MG[PAWN_BACKWARD];
                            eg += sign * FE_EG[PAWN_BACKWARD];
                            t.add(PAWN_BACKWARD, sign); t.add(PAWN_BACKWARD + NPHASE, sign);
                        }
                    }
                }
            }

            // Passed / candidate detection
            Bitboard ahead = 0;
            for (int rr = r + 1; rr <= RANK_7; ++rr) {
                Square rsq = relative_square(c, make_square(f, Rank(rr)));
                ahead |= square_bb(rsq);
                if (f > FILE_A) ahead |= square_bb(relative_square(c, make_square(File(f - 1), Rank(rr))));
                if (f < FILE_H) ahead |= square_bb(relative_square(c, make_square(File(f + 1), Rank(rr))));
            }

            if ((ahead & their_pawns) && r >= RANK_2 && r <= RANK_6) {
                Bitboard stoppers = ahead & their_pawns;
                Bitboard threats = their_pawns & pawn_attacks_bb(c, sq);
                Bitboard support = our_pawns & pawn_attacks_bb(them, sq);
                Bitboard push_sq_bb = relative_square(c, make_square(f, Rank(r + 1)));
                Bitboard push_threats = their_pawns & pawn_attacks_bb(c, push_sq_bb);
                Bitboard push_support = our_pawns & pawn_attacks_bb(them, push_sq_bb);
                Bitboard leftovers = stoppers & ~(threats | push_threats);
                bool supported = popcount(support) >= popcount(threats);
                if (!leftovers && popcount(push_support) >= popcount(push_threats)) {
                    int sup = supported ? 1 : 0;
                    mg += sign * CAMG[sup * 8 + r];
                    eg += sign * CAEG[sup * 8 + r];
                    t.add(PAWN_CAND + sup * 7 + (r - 1), sign);
                    t.add(PAWN_CAND + sup * 7 + (r - 1) + NPHASE, sign);
                }
            }

            if (!(ahead & their_pawns)) {
                mg += sign * PAMG[r];
                eg += sign * PAEG[r];
                t.add(PAWN_PASSED + (r - 1), sign);
                t.add(PAWN_PASSED + (r - 1) + NPHASE, sign);
            }
        }
    }

    mg_out = mg;
    eg_out = eg;
    t.mmg += mg;
    t.meg += eg;
}

// Full mirror of evaluate(pos, tactical_only=false). Emits features into t,
// keeps t.mmg/t.meg int-exact, returns phase/scale/mirror.
static TraceOut trace_eval(const Position& pos, Tracer& t) {
    using namespace feat;
    TraceOut out;
    out.stm = (pos.side_to_move() == WHITE) ? 1 : 0;

    // --- non-linear gates (KPK bitbase / KXK): not linear-representable ---
    if (popcount(pos.pieces(PAWN)) == 1
        && (pos.pieces(KNIGHT) | pos.pieces(BISHOP) | pos.pieces(ROOK) | pos.pieces(QUEEN)) == 0) {
        out.linear = false;
        return out;
    }
    for (int strong = 0; strong < 2; ++strong) {
        Color c = Color(strong);
        Color weak = Color(strong ^ 1);
        int strong_material = popcount(pos.pieces(c)) - 1;
        int weak_material = popcount(pos.pieces(weak)) - 1;
        if (strong_material >= 1 && weak_material == 0) { out.linear = false; return out; }
    }

    int mg_score = 0;
    int eg_score = 0;

    Square ksq_arr[2] = {pos.king_sq(WHITE), pos.king_sq(BLACK)};
    int bishop_count[2] = {0, 0};

    Bitboard occupied = pos.pieces();

    Bitboard attacks_by[2][7] = {};
    Bitboard all_attacks[2] = {};

    for (int c = 0; c < 2; ++c) {
        attacks_by[c][PAWN] = pawn_attacks_bb(Color(c), pos.pieces(Color(c), PAWN));
        attacks_by[c][KING] = king_attacks_bb(ksq_arr[c]);
        all_attacks[c] = attacks_by[c][PAWN] | attacks_by[c][KING];
    }

    {
        int32_t pawn_mg = 0, pawn_eg = 0;
        trace_pawns(pos, t, pawn_mg, pawn_eg);
        mg_score += pawn_mg;
        eg_score += pawn_eg;
    }

    const int* PVMG = etrace::piece_value_mg();
    const int* PVEG = etrace::piece_value_eg();

    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color c = Color(c_idx);
        Color them = Color(c_idx ^ 1);
        int sign = (c == WHITE) ? 1 : -1;
        Bitboard our_pawns = pos.pieces(c, PAWN);
        Bitboard their_pawns = pos.pieces(them, PAWN);

        Bitboard mob_area = ~pawn_attacks_bb(them, their_pawns);

        // ----------------- passed pawn extras -----------------
        {
        Bitboard passed_pawns_bb = BB_EMPTY;
        {
            Bitboard pawns = our_pawns;
            while (pawns) {
                Square sq = pop_lsb(pawns);
                File f = file_of(sq);
                Rank r = relative_rank(c, sq);
                Bitboard ahead = 0;
                for (int rr = r + 1; rr <= RANK_7; ++rr) {
                    Square rsq = relative_square(c, make_square(f, Rank(rr)));
                    ahead |= square_bb(rsq);
                    if (f > FILE_A) ahead |= square_bb(relative_square(c, make_square(File(f - 1), Rank(rr))));
                    if (f < FILE_H) ahead |= square_bb(relative_square(c, make_square(File(f + 1), Rank(rr))));
                }
                if (!(ahead & their_pawns)) passed_pawns_bb |= square_bb(sq);
            }
        }

        Bitboard pp = passed_pawns_bb;
        while (pp) {
            Square sq = pop_lsb(pp);
            File f = file_of(sq);
            Rank r = relative_rank(c, sq);

            Bitboard ahead_file = 0;
            for (int rr = r + 1; rr <= RANK_7; ++rr) {
                ahead_file |= square_bb(relative_square(c, make_square(f, Rank(rr))));
            }

            int mg_passer = 0;
            int eg_passer = 0;

            Bitboard adjacent_files = BB_EMPTY;
            if (f > FILE_A) adjacent_files |= file_bb(File(f - 1));
            if (f < FILE_H) adjacent_files |= file_bb(File(f + 1));
            Bitboard connected = passed_pawns_bb & adjacent_files & rank_bb(rank_of(sq));
            if (connected) {
                int n_connected = popcount(connected);
                mg_passer += n_connected * FE_MG[PP_CONNECTED + r - 1];
                eg_passer += n_connected * FE_EG[PP_CONNECTED + r - 1];
                t.add(PP_CONNECTED + (r - 1), sign * n_connected);
                t.add(PP_CONNECTED + (r - 1) + NPHASE, sign * n_connected);
            }

            Bitboard behind = file_bb(f) & pos.pieces(c, ROOK);
            if (behind) {
                Square rsq2 = lsb(behind);
                bool rook_behind = (c == WHITE) ? (rank_of(rsq2) < rank_of(sq)) : (rank_of(rsq2) > rank_of(sq));
                if (rook_behind) {
                    mg_passer += FE_MG[PP_ROOK_BEHIND];
                    eg_passer += FE_EG[PP_ROOK_BEHIND];
                    t.add(PP_ROOK_BEHIND, sign); t.add(PP_ROOK_BEHIND + NPHASE, sign);
                }
            }

            Bitboard enemy_behind = file_bb(f) & pos.pieces(them, ROOK);
            if (enemy_behind) {
                Square ersq = lsb(enemy_behind);
                bool enemy_rook_behind = (c == WHITE) ? (rank_of(ersq) < rank_of(sq)) : (rank_of(ersq) > rank_of(sq));
                if (enemy_rook_behind) {
                    mg_passer += FE_MG[PP_ENEMY_ROOK];
                    eg_passer += FE_EG[PP_ENEMY_ROOK];
                    t.add(PP_ENEMY_ROOK, sign); t.add(PP_ENEMY_ROOK + NPHASE, sign);
                }
            }

            Square our_ksq = ksq_arr[c_idx];
            Square their_ksq = ksq_arr[c_idx ^ 1];
            Square promo_sq = relative_square(c, make_square(f, RANK_8));
            int our_kdist = distance(our_ksq, promo_sq);
            int their_kdist = distance(their_ksq, promo_sq);
            eg_passer += (their_kdist - our_kdist) * FE_EG[PP_KING_PROX];
            t.add(PP_KING_PROX + NPHASE, sign * (their_kdist - our_kdist));

            if (their_kdist > our_kdist + (c == pos.side_to_move() ? 0 : 1)) {
                int unreachable = their_kdist - our_kdist;
                eg_passer += unreachable * (unreachable > 4 ? FE_EG[PP_UNREACH_B]
                                                           : FE_EG[PP_UNREACH_S]);
                if (unreachable > 4) t.add(PP_UNREACH_B + NPHASE, sign * unreachable);
                else                 t.add(PP_UNREACH_S + NPHASE, sign * unreachable);
            }

            {
                Square stop_sq = relative_square(c, make_square(f, Rank(r + 1)));
                bool stop_enemy = (pos.pieces(them) & square_bb(stop_sq)) != 0;
                bool stop_own   = (pos.pieces(c)    & square_bb(stop_sq)) != 0;
                bool stop_attacked_by_enemy = (pos.attackers_to(stop_sq) & pos.pieces(them)) != 0;
                bool enemy_on_path = (ahead_file & pos.pieces(them)) != 0;

                if (stop_enemy) {
                    mg_passer += FE_MG[PP_BLOCKADE + r - 1];
                    eg_passer += FE_EG[PP_BLOCKADE + r - 1];
                    t.add(PP_BLOCKADE + (r - 1), sign); t.add(PP_BLOCKADE + (r - 1) + NPHASE, sign);
                } else if (stop_own) {
                    mg_passer += FE_MG[PP_STOP_OWN];
                    eg_passer += FE_EG[PP_STOP_OWN];
                    t.add(PP_STOP_OWN, sign); t.add(PP_STOP_OWN + NPHASE, sign);
                } else if (stop_attacked_by_enemy) {
                    mg_passer += FE_MG[PP_STOP_ATT];
                    eg_passer += FE_EG[PP_STOP_ATT];
                    t.add(PP_STOP_ATT, sign); t.add(PP_STOP_ATT + NPHASE, sign);
                } else {
                    mg_passer += FE_MG[PP_SAFE + r - 1];
                    eg_passer += FE_EG[PP_SAFE + r - 1];
                    t.add(PP_SAFE + (r - 1), sign); t.add(PP_SAFE + (r - 1) + NPHASE, sign);
                    if (!enemy_on_path) {
                        eg_passer += FE_EG[PP_CLEAR + r - 1];
                        t.add(PP_CLEAR + (r - 1) + NPHASE, sign);
                    }
                }
            }

            mg_score += sign * mg_passer;
            eg_score += sign * eg_passer;
        }
        }

        // ----------------- knights -----------------
        Bitboard knights = pos.pieces(c, KNIGHT);
        while (knights) {
            Square sq = pop_lsb(knights);
            int relsq = (c == WHITE) ? int(sq) : (int(sq) ^ 56);
            mg_score += sign * (PVMG[KNIGHT] + PST_MG_TABLE[int(c)][int(KNIGHT)][int(sq)]);
            eg_score += sign * (PVEG[KNIGHT] + PST_EG_TABLE[int(c)][int(KNIGHT)][int(sq)]);
            t.add(PST + int(KNIGHT) * 64 + relsq, sign);
            t.add(PST + int(KNIGHT) * 64 + relsq + NPHASE, sign);
            t.add(MAT + int(KNIGHT), sign);
            t.add(MAT + int(KNIGHT) + NPHASE, sign);

            Bitboard attacks = knight_attacks_bb(sq);
            attacks_by[c][KNIGHT] |= attacks;
            all_attacks[c] |= attacks;

            int mob = popcount(attacks & mob_area & ~pos.pieces(c));
            mob = std::min(mob, 8);
            mg_score += sign * etrace::knight_mob_mg()[mob];
            eg_score += sign * etrace::knight_mob_eg()[mob];
            t.add(MOB_N + mob, sign); t.add(MOB_N + mob + NPHASE, sign);

            Rank kr = relative_rank(c, sq);
            if (kr >= RANK_4 && kr <= RANK_6) {
                if (pawn_attacks_bb(c, our_pawns) & square_bb(sq)) {
                    Bitboard enemy_pawn_attacks = pawn_attacks_bb(them, their_pawns);
                    if (!(enemy_pawn_attacks & square_bb(sq))) {
                        mg_score += sign * FE_MG[OUTPOST_N + kr - 3];
                        eg_score += sign * FE_EG[OUTPOST_N + kr - 3];
                        t.add(OUTPOST_N + (kr - 3), sign);
                        t.add(OUTPOST_N + (kr - 3) + NPHASE, sign);
                    }
                }
            }

            if (mob == 0) {
                mg_score += sign * FE_MG[TRAPPED_N];
                eg_score += sign * FE_EG[TRAPPED_N];
                t.add(TRAPPED_N, sign); t.add(TRAPPED_N + NPHASE, sign);
            }

            if (distance(sq, ksq_arr[c_idx]) > 3) {
                mg_score += sign * FE_MG[FAR_N];
                eg_score += sign * FE_EG[FAR_N];
                t.add(FAR_N, sign); t.add(FAR_N + NPHASE, sign);
            }
        }

        // ----------------- bishops -----------------
        Bitboard bishops = pos.pieces(c, BISHOP);
        bishop_count[c_idx] = popcount(bishops);
        while (bishops) {
            Square sq = pop_lsb(bishops);
            int relsq = (c == WHITE) ? int(sq) : (int(sq) ^ 56);
            mg_score += sign * (PVMG[BISHOP] + PST_MG_TABLE[int(c)][int(BISHOP)][int(sq)]);
            eg_score += sign * (PVEG[BISHOP] + PST_EG_TABLE[int(c)][int(BISHOP)][int(sq)]);
            t.add(PST + int(BISHOP) * 64 + relsq, sign);
            t.add(PST + int(BISHOP) * 64 + relsq + NPHASE, sign);
            t.add(MAT + int(BISHOP), sign);
            t.add(MAT + int(BISHOP) + NPHASE, sign);

            Bitboard attacks = bb_diag_attacks(sq, occupied);
            attacks_by[c][BISHOP] |= attacks;
            all_attacks[c] |= attacks;

            int mob = popcount(attacks & mob_area & ~pos.pieces(c));
            mob = std::min(mob, 13);
            mg_score += sign * etrace::bishop_mob_mg()[mob];
            eg_score += sign * etrace::bishop_mob_eg()[mob];
            t.add(MOB_B + mob, sign); t.add(MOB_B + mob + NPHASE, sign);

            {
                static constexpr Bitboard BB_DARK_SQ = 0x55AA55AA55AA55AAULL;
                static constexpr Bitboard BB_LIGHT_SQ = 0xAA55AA55AA55AA55ULL;
                bool is_dark = ((int(sq) + rank_of(sq)) % 2) == 0;
                Bitboard same_color_bb = is_dark ? BB_DARK_SQ : BB_LIGHT_SQ;
                int same_color_pawns = popcount(our_pawns & same_color_bb);
                same_color_pawns = std::min(same_color_pawns, 6);
                mg_score += sign * FE_MG[BSC + same_color_pawns];
                eg_score += sign * FE_EG[BSC + same_color_pawns];
                t.add(BSC + same_color_pawns, sign);
                t.add(BSC + same_color_pawns + NPHASE, sign);
            }

            if (popcount(attacks & BB_CENTER) >= 2) {
                mg_score += sign * FE_MG[BISH_LONG];
                eg_score += sign * FE_EG[BISH_LONG];
                t.add(BISH_LONG, sign); t.add(BISH_LONG + NPHASE, sign);
            }

            {
                Rank br = relative_rank(c, sq);
                if (br >= RANK_4 && br <= RANK_6) {
                    if ((pawn_attacks_bb(c, our_pawns) & square_bb(sq)) &&
                        !(pawn_attacks_bb(them, their_pawns) & square_bb(sq))) {
                        mg_score += sign * FE_MG[OUTPOST_B];
                        eg_score += sign * FE_EG[OUTPOST_B];
                        t.add(OUTPOST_B, sign); t.add(OUTPOST_B + NPHASE, sign);
                    }
                }
            }

            if (distance(sq, ksq_arr[c_idx]) > 3) {
                mg_score += sign * FE_MG[FAR_B];
                eg_score += sign * FE_EG[FAR_B];
                t.add(FAR_B, sign); t.add(FAR_B + NPHASE, sign);
            }
        }

        // ----------------- rooks -----------------
        Bitboard rooks = pos.pieces(c, ROOK);
        while (rooks) {
            Square sq = pop_lsb(rooks);
            int relsq = (c == WHITE) ? int(sq) : (int(sq) ^ 56);
            mg_score += sign * (PVMG[ROOK] + PST_MG_TABLE[int(c)][int(ROOK)][int(sq)]);
            eg_score += sign * (PVEG[ROOK] + PST_EG_TABLE[int(c)][int(ROOK)][int(sq)]);
            t.add(PST + int(ROOK) * 64 + relsq, sign);
            t.add(PST + int(ROOK) * 64 + relsq + NPHASE, sign);
            t.add(MAT + int(ROOK), sign);
            t.add(MAT + int(ROOK) + NPHASE, sign);

            Bitboard attacks = rook_attacks_bb(sq, occupied ^ pos.pieces(c, ROOK) ^ pos.pieces(c, QUEEN));
            attacks_by[c][ROOK] |= attacks;
            all_attacks[c] |= attacks;

            int mob = popcount(attacks & mob_area & ~pos.pieces(c));
            mob = std::min(mob, 14);
            mg_score += sign * etrace::rook_mob_mg()[mob];
            eg_score += sign * etrace::rook_mob_eg()[mob];
            t.add(MOB_R + mob, sign); t.add(MOB_R + mob + NPHASE, sign);

            File f = file_of(sq);
            if (!(pos.pieces(PAWN) & file_bb(f))) {
                mg_score += sign * FE_MG[ROOK_OPEN];
                eg_score += sign * FE_EG[ROOK_OPEN];
                t.add(ROOK_OPEN, sign); t.add(ROOK_OPEN + NPHASE, sign);
            } else if (!(our_pawns & file_bb(f))) {
                mg_score += sign * FE_MG[ROOK_SEMI];
                eg_score += sign * FE_EG[ROOK_SEMI];
                t.add(ROOK_SEMI, sign); t.add(ROOK_SEMI + NPHASE, sign);
            }

            Rank rr = relative_rank(c, rank_of(sq));
            if (rr == RANK_7) {
                mg_score += sign * FE_MG[ROOK_7TH];
                eg_score += sign * FE_EG[ROOK_7TH];
                t.add(ROOK_7TH, sign); t.add(ROOK_7TH + NPHASE, sign);
            }

            if (file_bb(f) & pos.pieces(them, QUEEN)) {
                mg_score += sign * FE_MG[ROOK_XRAY_Q];
                eg_score += sign * FE_EG[ROOK_XRAY_Q];
                t.add(ROOK_XRAY_Q, sign); t.add(ROOK_XRAY_Q + NPHASE, sign);
            }

            if (distance(sq, ksq_arr[c_idx]) > 3) {
                mg_score += sign * FE_MG[FAR_R];
                eg_score += sign * FE_EG[FAR_R];
                t.add(FAR_R, sign); t.add(FAR_R + NPHASE, sign);
            }

            if (rr <= RANK_2) {
                Square ksq = ksq_arr[c_idx];
                int fdist = std::abs(file_of(sq) - file_of(ksq));
                bool king_between = (c == WHITE)
                    ? (rank_of(ksq) <= rank_of(sq) && fdist <= 1)
                    : (rank_of(ksq) >= rank_of(sq) && fdist <= 1);
                if (king_between && mob <= 4) {
                    CastlingRight ks_cr = c == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE;
                    CastlingRight qs_cr = c == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE;
                    bool has_castling = pos.castling_allowed(c, ks_cr) || pos.castling_allowed(c, qs_cr);
                    if (has_castling) {
                        mg_score += sign * FE_MG[ROOK_TRAP_C];
                        eg_score += sign * FE_EG[ROOK_TRAP_C];
                        t.add(ROOK_TRAP_C, sign); t.add(ROOK_TRAP_C + NPHASE, sign);
                    } else {
                        mg_score += sign * FE_MG[ROOK_TRAP_NC];
                        eg_score += sign * FE_EG[ROOK_TRAP_NC];
                        t.add(ROOK_TRAP_NC, sign); t.add(ROOK_TRAP_NC + NPHASE, sign);
                    }
                }
            }

            {
                Bitboard ahead_pawns;
                if (c == WHITE) {
                    Bitboard below_and_same = 0;
                    for (int r = RANK_1; r <= rank_of(sq); ++r)
                        below_and_same |= rank_bb(Rank(r));
                    ahead_pawns = our_pawns & file_bb(f) & ~below_and_same;
                } else {
                    Bitboard above_and_same = 0;
                    for (int r = rank_of(sq); r <= RANK_8; ++r)
                        above_and_same |= rank_bb(Rank(r));
                    ahead_pawns = our_pawns & file_bb(f) & ~above_and_same;
                }
                if (ahead_pawns) {
                    mg_score += sign * FE_MG[ROOK_BLOCKED];
                    eg_score += sign * FE_EG[ROOK_BLOCKED];
                    t.add(ROOK_BLOCKED, sign); t.add(ROOK_BLOCKED + NPHASE, sign);
                }
            }
        }

        // ----------------- queens -----------------
        Bitboard queens = pos.pieces(c, QUEEN);
        while (queens) {
            Square sq = pop_lsb(queens);
            int relsq = (c == WHITE) ? int(sq) : (int(sq) ^ 56);
            mg_score += sign * (PVMG[QUEEN] + PST_MG_TABLE[int(c)][int(QUEEN)][int(sq)]);
            eg_score += sign * (PVEG[QUEEN] + PST_EG_TABLE[int(c)][int(QUEEN)][int(sq)]);
            t.add(PST + int(QUEEN) * 64 + relsq, sign);
            t.add(PST + int(QUEEN) * 64 + relsq + NPHASE, sign);
            t.add(MAT + int(QUEEN), sign);
            t.add(MAT + int(QUEEN) + NPHASE, sign);

            Bitboard attacks = bb_diag_attacks(sq, occupied ^ pos.pieces(c, BISHOP))
                             | rook_attacks_bb(sq, occupied ^ pos.pieces(c, ROOK));
            attacks_by[c][QUEEN] |= attacks;
            all_attacks[c] |= attacks;

            int mob = popcount(attacks & mob_area & ~pos.pieces(c));
            mob = std::min(mob, 27);
            mg_score += sign * etrace::queen_mob_mg()[mob];
            eg_score += sign * etrace::queen_mob_eg()[mob];
            t.add(MOB_Q + mob, sign); t.add(MOB_Q + mob + NPHASE, sign);

            if (distance(sq, ksq_arr[c_idx]) > 3) {
                mg_score += sign * FE_MG[FAR_Q];
                eg_score += sign * FE_EG[FAR_Q];
                t.add(FAR_Q, sign); t.add(FAR_Q + NPHASE, sign);
            }
        }

        // ----------------- king -----------------
        Square ksq = ksq_arr[c_idx];
        {
            int relsq = (c == WHITE) ? int(ksq) : (int(ksq) ^ 56);
            mg_score += sign * (PVMG[KING] + PST_MG_TABLE[int(c)][int(KING)][int(ksq)]);
            eg_score += sign * (PVEG[KING] + PST_EG_TABLE[int(c)][int(KING)][int(ksq)]);
            t.add(PST + int(KING) * 64 + relsq, sign);
            t.add(PST + int(KING) * 64 + relsq + NPHASE, sign);
        }

        {
        Rank krank = rank_of(ksq);
        File kfile = file_of(ksq);
        bool on_back = (c == WHITE && krank <= RANK_2) || (c == BLACK && krank >= RANK_7);
        if (on_back) {
            for (int r = 1; r <= 3; ++r) {
                int feat_id = (r == 1) ? SHIELD_R1 : (r == 2) ? SHIELD_R2 : SHIELD_R3;
                int weight = (r == 1) ? FE_MG[SHIELD_R1]
                             : (r == 2) ? FE_MG[SHIELD_R2]
                             : FE_MG[SHIELD_R3];
                for (int df = -1; df <= 1; ++df) {
                    File sf = File(int(kfile) + df);
                    if (sf < FILE_A || sf > FILE_H) continue;
                    Square shield_sq = relative_square(c, make_square(sf, Rank(r)));
                    if (our_pawns & square_bb(shield_sq)) {
                        mg_score += sign * weight;
                        t.add(feat_id, sign);
                    }
                }
            }

            Bitboard all_pawns = pos.pieces(PAWN);
            if (!(all_pawns & file_bb(kfile))) {
                mg_score += sign * FE_MG[OPEN_KFILE];
                t.add(OPEN_KFILE, sign);
            }
            int adj_open = 0;
            if (kfile > FILE_A && !(all_pawns & file_bb(File(kfile - 1)))) {
                mg_score += sign * FE_MG[OPEN_ADJ];
                adj_open++;
            }
            if (kfile < FILE_H && !(all_pawns & file_bb(File(kfile + 1)))) {
                mg_score += sign * FE_MG[OPEN_ADJ];
                adj_open++;
            }
            if (adj_open) t.add(OPEN_ADJ, sign * adj_open);

            int storm_sum = 0;
            for (int df = -1; df <= 1; ++df) {
                File storm_file = File(int(kfile) + df);
                if (storm_file < FILE_A || storm_file > FILE_H) continue;
                Bitboard enemy_file_pawns = their_pawns & file_bb(storm_file);
                while (enemy_file_pawns) {
                    Square psq = pop_lsb(enemy_file_pawns);
                    Rank pr = relative_rank(c, rank_of(psq));
                    if (pr >= RANK_4) {
                        mg_score += sign * (pr - 3) * FE_MG[STORM];
                        storm_sum += (pr - 3);
                    }
                }
            }
            if (storm_sum) t.add(STORM, sign * storm_sum);
        }

        bool castled = false;
        if (c == WHITE && krank == RANK_1 && (kfile == FILE_G || kfile == FILE_C)) castled = true;
        if (c == BLACK && krank == RANK_8 && (kfile == FILE_G || kfile == FILE_C)) castled = true;
        if (castled) {
            mg_score += sign * FE_MG[CASTLED]; eg_score += sign * FE_EG[CASTLED];
            t.add(CASTLED, sign); t.add(CASTLED + NPHASE, sign);
        }

        CastlingRight ks_cr = c == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE;
        CastlingRight qs_cr = c == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE;
        if (!pos.castling_allowed(c, ks_cr) && !pos.castling_allowed(c, qs_cr) && !castled) {
            mg_score += sign * FE_MG[NOCASTLE];
            eg_score += sign * FE_EG[NOCASTLE];
            t.add(NOCASTLE, sign); t.add(NOCASTLE + NPHASE, sign);
        }
        }

        // ----------------- threats -----------------
        {
            Bitboard their_pieces = pos.pieces(them);

            Bitboard threats = their_pieces & attacks_by[c][PAWN];
            while (threats) {
                PieceType pt = piece_type_of(pos.piece_on(pop_lsb(threats)));
                if (pt < KING) {
                    mg_score += sign * etrace::pawn_threat_mg()[pt];
                    eg_score += sign * etrace::pawn_threat_eg()[pt];
                    t.add(THR + 0 * 5 + int(pt), sign);
                    t.add(THR + 0 * 5 + int(pt) + NPHASE, sign);
                }
            }
            threats = their_pieces & attacks_by[c][KNIGHT];
            while (threats) {
                PieceType pt = piece_type_of(pos.piece_on(pop_lsb(threats)));
                if (pt < KING) {
                    mg_score += sign * etrace::knight_threat_mg()[pt];
                    eg_score += sign * etrace::knight_threat_eg()[pt];
                    t.add(THR + 1 * 5 + int(pt), sign);
                    t.add(THR + 1 * 5 + int(pt) + NPHASE, sign);
                }
            }
            threats = their_pieces & attacks_by[c][BISHOP];
            while (threats) {
                PieceType pt = piece_type_of(pos.piece_on(pop_lsb(threats)));
                if (pt < KING) {
                    mg_score += sign * etrace::bishop_threat_mg()[pt];
                    eg_score += sign * etrace::bishop_threat_eg()[pt];
                    t.add(THR + 2 * 5 + int(pt), sign);
                    t.add(THR + 2 * 5 + int(pt) + NPHASE, sign);
                }
            }
            threats = their_pieces & attacks_by[c][ROOK];
            while (threats) {
                PieceType pt = piece_type_of(pos.piece_on(pop_lsb(threats)));
                if (pt < KING) {
                    mg_score += sign * etrace::rook_threat_mg()[pt];
                    eg_score += sign * etrace::rook_threat_eg()[pt];
                    t.add(THR + 3 * 5 + int(pt), sign);
                    t.add(THR + 3 * 5 + int(pt) + NPHASE, sign);
                }
            }
            threats = their_pieces & attacks_by[c][QUEEN];
            while (threats) {
                PieceType pt = piece_type_of(pos.piece_on(pop_lsb(threats)));
                if (pt < KING) {
                    mg_score += sign * etrace::queen_threat_mg()[pt];
                    eg_score += sign * etrace::queen_threat_eg()[pt];
                    t.add(THR + 4 * 5 + int(pt), sign);
                    t.add(THR + 4 * 5 + int(pt) + NPHASE, sign);
                }
            }
            Bitboard hanging_pawns = pos.pieces(them, PAWN) & ~all_attacks[them] & all_attacks[c];
            if (hanging_pawns) {
                mg_score += sign * FE_MG[HANG_PAWN] * popcount(hanging_pawns);
                eg_score += sign * FE_EG[HANG_PAWN] * popcount(hanging_pawns);
                t.add(HANG_PAWN, sign * popcount(hanging_pawns));
                t.add(HANG_PAWN + NPHASE, sign * popcount(hanging_pawns));
            }
            Bitboard hanging_pieces = (pos.pieces(them, KNIGHT) | pos.pieces(them, BISHOP)
                                     | pos.pieces(them, ROOK) | pos.pieces(them, QUEEN))
                                     & ~all_attacks[them] & all_attacks[c];
            if (hanging_pieces) {
                mg_score += sign * FE_MG[HANG_PIECE] * popcount(hanging_pieces);
                eg_score += sign * FE_EG[HANG_PIECE] * popcount(hanging_pieces);
                t.add(HANG_PIECE, sign * popcount(hanging_pieces));
                t.add(HANG_PIECE + NPHASE, sign * popcount(hanging_pieces));
            }
            {
                Square their_ksq = ksq_arr[c_idx ^ 1];
                Bitboard our_sliders = (pos.pieces(c, ROOK, QUEEN) | pos.pieces(c, BISHOP, QUEEN));
                Bitboard enemy_pinned = BB_EMPTY;
                while (our_sliders) {
                    Square sniper = pop_lsb(our_sliders);
                    Bitboard ray_sniper = (piece_type_of(pos.piece_on(sniper)) == BISHOP)
                        ? bb_diag_attacks(their_ksq, Bitboard(0)) : rook_attacks_bb(their_ksq, Bitboard(0));
                    if (!(ray_sniper & square_bb(sniper))) continue;
                    Bitboard between = between_bb(their_ksq, sniper) & occupied;
                    if (between && !more_than_one(between) && (between & pos.pieces(them))) {
                        enemy_pinned |= between;
                    }
                }
                if (enemy_pinned) {
                    int pin_count = popcount(enemy_pinned);
                    mg_score += sign * pin_count * FE_MG[PINNED];
                    eg_score += sign * pin_count * FE_EG[PINNED];
                    t.add(PINNED, sign * pin_count);
                    t.add(PINNED + NPHASE, sign * pin_count);
                }
            }
        }
    }

    // ----------------- early eval exit -----------------
    {
        int quick_phase = popcount(pos.pieces(KNIGHT)) + popcount(pos.pieces(BISHOP))
                        + popcount(pos.pieces(ROOK)) * 2 + popcount(pos.pieces(QUEEN)) * 4;
        quick_phase = std::min(24, quick_phase);
        int quick_score = (mg_score * quick_phase + eg_score * (24 - quick_phase)) / 24;
        if (quick_score > 500 || quick_score < -500) {
            int sf = scale_factor(pos, Value(eg_score));
            int eg_scaled = eg_score * sf / 32;
            int score = (mg_score * quick_phase + eg_scaled * (24 - quick_phase)) / 24;
            int tempo = (15 * quick_phase + 5 * (24 - quick_phase)) / 24;
            score += (pos.side_to_move() == WHITE) ? tempo : -tempo;
            int mirror = pos.side_to_move() == WHITE ? score : -score;
            if (getenv("ETRACE_DEBUG")) fprintf(stderr, "DBG[quick] ph=%d sf=%d mg=%d eg=%d egsc=%d tempo=%d mirror=%d\n", quick_phase, sf, t.mmg, t.meg, eg_scaled, tempo, mirror);
            out.ph = quick_phase; out.sf = sf;
            out.mirror_white = mirror;
            t.add(TEMPO_MG, (pos.side_to_move() == WHITE) ? quick_phase : -quick_phase);
            t.add(TEMPO_EG, (pos.side_to_move() == WHITE) ? (24 - quick_phase) : -(24 - quick_phase));
            return out;
        }
    }

    // ----------------- space -----------------
    {
        int space[2] = {0, 0};
        int pawn_count[2] = {0, 0};
        for (int c = 0; c < 2; ++c) {
            Bitboard our_pawns = pos.pieces(Color(c), PAWN);
            pawn_count[c] = popcount(our_pawns);
            Bitboard our_pawn_attacks = pawn_attacks_bb(Color(c), our_pawns);
            Bitboard their_pawn_attacks = pawn_attacks_bb(Color(c ^ 1), pos.pieces(Color(c ^ 1), PAWN));
            Bitboard space_area = (BB_FILE_C | BB_FILE_D | BB_FILE_E | BB_FILE_F)
                                 & (our_pawn_attacks & ~their_pawn_attacks);
            space[c] = popcount(space_area);
        }
        if (pawn_count[WHITE] > pawn_count[BLACK]) {
            mg_score += space[WHITE] * FE_MG[SPACE_OWN] + space[BLACK] * FE_MG[SPACE_OPP];
            t.add(SPACE_OWN, space[WHITE]);
            t.add(SPACE_OPP, space[BLACK]);
        } else if (pawn_count[BLACK] > pawn_count[WHITE]) {
            mg_score -= space[BLACK] * FE_MG[SPACE_OWN] + space[WHITE] * FE_MG[SPACE_OPP];
            t.add(SPACE_OWN, -space[BLACK]);
            t.add(SPACE_OPP, -space[WHITE]);
        }
    }

    // ----------------- material imbalance -----------------
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color c = Color(c_idx);
        int sign = (c_idx == 0) ? 1 : -1;
        int pawns = popcount(pos.pieces(c, PAWN));
        int bishops = bishop_count[c_idx];
        if (bishops >= 1) {
            mg_score += sign * (FE_MG[IMBAL_CONST] + pawns * FE_MG[IMBAL_PAWN]);
            eg_score += sign * (FE_EG[IMBAL_CONST] + pawns * FE_EG[IMBAL_PAWN]);
            t.add(IMBAL_CONST, sign); t.add(IMBAL_CONST + NPHASE, sign);
            t.add(IMBAL_PAWN, sign * pawns); t.add(IMBAL_PAWN + NPHASE, sign * pawns);
        }
    }

    if (bishop_count[WHITE] >= 2) {
        mg_score += FE_MG[BISHOP_PAIR]; eg_score += FE_EG[BISHOP_PAIR];
        t.add(BISHOP_PAIR, 1); t.add(BISHOP_PAIR + NPHASE, 1);
    }
    if (bishop_count[BLACK] >= 2) {
        mg_score -= FE_MG[BISHOP_PAIR]; eg_score -= FE_EG[BISHOP_PAIR];
        t.add(BISHOP_PAIR, -1); t.add(BISHOP_PAIR + NPHASE, -1);
    }

    {
        int w_pawns = popcount(pos.pieces(WHITE, PAWN));
        int b_pawns = popcount(pos.pieces(BLACK, PAWN));
        int total_pawns = w_pawns + b_pawns;
        int knight_adj = total_pawns - 8;
        int w_knights = popcount(pos.pieces(WHITE, KNIGHT));
        int b_knights = popcount(pos.pieces(BLACK, KNIGHT));
        if (knight_adj > 0) {
            mg_score += w_knights * knight_adj * FE_MG[KNIGHT_PAWN];
            mg_score -= b_knights * knight_adj * FE_MG[KNIGHT_PAWN];
            eg_score += w_knights * knight_adj * FE_EG[KNIGHT_PAWN];
            eg_score -= b_knights * knight_adj * FE_EG[KNIGHT_PAWN];
            t.add(KNIGHT_PAWN, (w_knights - b_knights) * knight_adj);
            t.add(KNIGHT_PAWN + NPHASE, (w_knights - b_knights) * knight_adj);
        }
    }

    {
        int w_np_mat = popcount(pos.pieces(WHITE, KNIGHT)) * PVEG[KNIGHT]
                     + popcount(pos.pieces(WHITE, BISHOP)) * PVEG[BISHOP]
                     + popcount(pos.pieces(WHITE, ROOK))   * PVEG[ROOK]
                     + popcount(pos.pieces(WHITE, QUEEN))  * PVEG[QUEEN];
        int b_np_mat = popcount(pos.pieces(BLACK, KNIGHT)) * PVEG[KNIGHT]
                     + popcount(pos.pieces(BLACK, BISHOP)) * PVEG[BISHOP]
                     + popcount(pos.pieces(BLACK, ROOK))   * PVEG[ROOK]
                     + popcount(pos.pieces(BLACK, QUEEN))  * PVEG[QUEEN];
        int nonpawn_pieces = popcount(pos.pieces(KNIGHT)) + popcount(pos.pieces(BISHOP))
                           + popcount(pos.pieces(ROOK))   + popcount(pos.pieces(QUEEN));
        int simplification = 14 - nonpawn_pieces;
        if (simplification > 0) {
            int diff = w_np_mat - b_np_mat;
            eg_score += FE_EG[TRADEDOWN] * (diff * simplification / 256);
            t.add(TRADEDOWN + NPHASE, diff * simplification / 256);
        }
    }

    // ----------------- king safety -----------------
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color c = Color(c_idx);
        Color them = Color(c_idx ^ 1);
        int sign = (c_idx == 0) ? 1 : -1;
        Square our_ksq = ksq_arr[c_idx];

        Bitboard king_zone = king_attacks_bb(our_ksq) | square_bb(our_ksq);
        Rank fwd = (c == WHITE) ? Rank(rank_of(our_ksq) + 1) : Rank(rank_of(our_ksq) - 1);
        if (fwd >= RANK_1 && fwd <= RANK_8) {
            for (File f = FILE_A; f <= FILE_H; f = File(f + 1)) {
                king_zone |= square_bb(make_square(f, fwd));
            }
        }

        Bitboard enemy_pieces_near = pos.pieces(them) & king_zone;
        enemy_pieces_near &= ~(pos.pieces(them, PAWN) | pos.pieces(them, KING));
        if (!enemy_pieces_near) continue;

        int attack_units = 0;
        int attacker_count = 0;

        Bitboard our_pawn_attacks = pawn_attacks_bb(c, pos.pieces(c, PAWN));
        Bitboard safe = ~our_pawn_attacks;

        Bitboard king_diag_rays = bb_diag_attacks(our_ksq, Bitboard(0));
        Bitboard king_orth_rays = rook_attacks_bb(our_ksq, Bitboard(0));

        Bitboard enemy_pawns = pos.pieces(them, PAWN);
        Bitboard ep = enemy_pawns;
        while (ep) {
            Square psq = pop_lsb(ep);
            if (pawn_attacks_bb(them, square_bb(psq)) & king_zone) {
                attack_units += 2;
                attacker_count++;
            }
        }

        Bitboard enemy_kn = pos.pieces(them, KNIGHT);
        while (enemy_kn) {
            Square ksq2 = pop_lsb(enemy_kn);
            Bitboard kn_attacks = knight_attacks_bb(ksq2);
            if (kn_attacks & king_zone) {
                attack_units += 4;
                attacker_count++;
            }
            Bitboard kn_checks = kn_attacks & king_attacks_bb(our_ksq);
            if (kn_checks & safe) attack_units += 10;
        }

        Bitboard enemy_bi = pos.pieces(them, BISHOP);
        while (enemy_bi) {
            Square bsq = pop_lsb(enemy_bi);
            Bitboard bi_attacks = bishop_attacks_bb(bsq, occupied);
            if (bi_attacks & king_zone) {
                attack_units += 3;
                attacker_count++;
            }
            Bitboard bi_checks = bi_attacks & king_diag_rays;
            if (bi_checks & safe) attack_units += 8;
        }

        Bitboard enemy_ro = pos.pieces(them, ROOK);
        while (enemy_ro) {
            Square rsq = pop_lsb(enemy_ro);
            Bitboard ro_attacks = rook_attacks_bb(rsq, occupied);
            if (ro_attacks & king_zone) {
                attack_units += 5;
                attacker_count++;
            }
            Bitboard ro_checks = ro_attacks & king_orth_rays;
            if (ro_checks & safe) attack_units += 12;
        }

        Bitboard enemy_qu = pos.pieces(them, QUEEN);
        while (enemy_qu) {
            Square qsq = pop_lsb(enemy_qu);
            Bitboard qu_attacks = queen_attacks_bb(qsq, occupied);
            if (qu_attacks & king_zone) {
                attack_units += 7;
                attacker_count++;
            }
            Bitboard qu_checks = qu_attacks & (king_diag_rays | king_orth_rays);
            if (qu_checks & safe) attack_units += 16;
        }

        if (attacker_count >= 2) {
            int au_pos = std::max(0, attack_units);
            int au_danger_mg = (au_pos * au_pos) / 8;
            int au_danger_eg = au_pos / 2;

            if (!enemy_qu) au_danger_mg = au_danger_mg / 4;
            if (!enemy_qu && !enemy_ro) au_danger_mg = au_danger_mg / 4;

            Bitboard king_moves = king_attacks_bb(our_ksq);
            Bitboard safe_king_squares = king_moves & ~pos.pieces(c) & ~all_attacks[them];
            int flight = popcount(safe_king_squares);
            int flight1 = 0, flight2 = 0;
            int flight_mg = (flight <= 1) ? FE_MG[KS_FLIGHT1]
                          : (flight == 2) ? FE_MG[KS_FLIGHT2] : 0;
            int flight_eg = (flight <= 1) ? FE_EG[KS_FLIGHT1]
                          : (flight == 2) ? FE_EG[KS_FLIGHT2] : 0;
            if (flight <= 1) flight1 = 1;
            else if (flight == 2) flight2 = 1;

            mg_score -= sign * (FE_MG[KS_AU] * au_danger_mg + flight_mg);
            eg_score -= sign * (FE_EG[KS_AU] * au_danger_eg + flight_eg);
            t.add(KS_AU, -sign * au_danger_mg);
            t.add(KS_AU + NPHASE, -sign * au_danger_eg);
            if (flight1) { t.add(KS_FLIGHT1, -sign); t.add(KS_FLIGHT1 + NPHASE, -sign); }
            if (flight2) { t.add(KS_FLIGHT2, -sign); t.add(KS_FLIGHT2 + NPHASE, -sign); }
        }
    }

    // ----------------- blend -----------------
    int phase = popcount(pos.pieces(KNIGHT)) + popcount(pos.pieces(BISHOP))
              + popcount(pos.pieces(ROOK)) * 2 + popcount(pos.pieces(QUEEN)) * 4;
    phase = std::min(24, phase);

    int sf = scale_factor(pos, Value(eg_score));
    int eg_scaled = eg_score * sf / 32;
    int score = (mg_score * phase + eg_scaled * (24 - phase)) / 24;

    int tempo = (15 * phase + 5 * (24 - phase)) / 24;
    score += (pos.side_to_move() == WHITE) ? tempo : -tempo;

    int mirror = pos.side_to_move() == WHITE ? score : -score;
    if (getenv("ETRACE_DEBUG")) fprintf(stderr, "RAW N=%d B=%d R=%d Q=%d | ", popcount(pos.pieces(KNIGHT)), popcount(pos.pieces(BISHOP)), popcount(pos.pieces(ROOK)), popcount(pos.pieces(QUEEN)));
    if (getenv("ETRACE_DEBUG")) fprintf(stderr, "DBG ph=%d sf=%d mg=%d eg=%d egsc=%d tempo=%d mirror=%d\n", phase, sf, mg_score, eg_score, eg_scaled, tempo, mirror);
    out.ph = phase; out.sf = sf;
    out.mirror_white = mirror;
    t.add(TEMPO_MG, (pos.side_to_move() == WHITE) ? phase : -phase);
    t.add(TEMPO_EG, (pos.side_to_move() == WHITE) ? (24 - phase) : -(24 - phase));
    return out;
}

// ============================================================================
// Current-engine coefficients (for --verify reconstruction). Every value is
// read from the engine's own tables/params via etrace/g_eval_params so the
// check can never drift from the engine.
// ============================================================================
static double CMG[feat::NPHASE];
static double CEG[feat::NPHASE];

static void init_coefs() {
    // FE_MG/FE_EG ARE the engine's current coefficients (evaluation.cpp reads
    // them directly), so the reconstruction check compares against the same
    // numbers the engine uses — by construction, no drift possible.
    for (int i = 0; i < feat::NPHASE; ++i) {
        CMG[i] = FE_MG[i];
        CEG[i] = FE_EG[i];
    }
}

// White-perspective reconstruction: dot(features, coefs) blended like the engine.
// Returns the WHITE-perspective eval; compare against mirror * stm_sign.
static double reconstruct(const Tracer& t, int ph, int sf) {
    double mg = 0.0, eg = 0.0;
    for (int i = 0; i < t.nt; ++i) {
        int idx = t.touched[i];
        if (idx < feat::NPHASE)             mg += CMG[idx] * t.f[idx];
        else if (idx < 2 * feat::NPHASE)    eg += CEG[idx - feat::NPHASE] * t.f[idx];
    }
    double eg_scaled = std::trunc(eg * sf / 32.0);
    double score = (mg * ph + eg_scaled * (24 - ph)) / 24.0;
    double tempo = (15.0 * t.f[feat::TEMPO_MG] + 5.0 * t.f[feat::TEMPO_EG]) / 24.0;
    if (getenv("ETRACE_DEBUG")) fprintf(stderr, "DBG[rec] mg_rec=%.1f eg_rec=%.1f egsc=%.1f tempo=%.1f total=%.1f\n", mg, eg, eg_scaled, tempo, score + tempo);
    return score + tempo;
}

} // namespace luminex

int main(int argc, char** argv) {
    bool verify = false, dump_coefs = false, solve = false, score = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--verify") == 0) verify = true;
        if (std::strcmp(argv[i], "--dump-coefs") == 0) dump_coefs = true;
        if (std::strcmp(argv[i], "--solve") == 0) solve = true;
        if (std::strcmp(argv[i], "--score") == 0) score = true;
    }

    luminex::init_magic_bitboards();   // attack tables (set()/eval depend on these)
    luminex::init_line_tables();       // BetweenBB/LineBB for pin detection
    luminex::init_zobrist();
    luminex::init_evaluation();
    luminex::init_coefs();

    if (dump_coefs) {
        // solver-space single coefs: line j = "j value" with j<607 -> CMG[j], j>=607 -> CEG[j-607]
        for (int j = 0; j < luminex::NSOLVE; ++j)
            std::printf("%d %.6f\n", j, j < luminex::feat::NPHASE ? luminex::CMG[j]
                                                              : luminex::CEG[j - luminex::feat::NPHASE]);
        return 0;
    }

    luminex::Position pos;
    std::string line, fen;
    long long n = 0, gated = 0, bad_fen = 0;
    long long mirror_mismatch = 0, rec_fail = 0;
    double max_rec_diff = 0.0, sum_rec_diff = 0.0;

    // --score state: score fen\tstm\ttarget rows under candidate coefs (arg 1) AND the
    // current-engine coefs (baseline) in one pass. Prints R^2 / RMSE / MAE for both.
    double sc_cand[luminex::NSOLVE], sc_base[luminex::NSOLVE];
    bool sc_ok = false;
    long long sc_n = 0;
    double sc_ss_tot = 0, sc_ssr_c = 0, sc_ssr_b = 0, sc_abs_c = 0, sc_abs_b = 0, sc_sum_r = 0;
    if (score) {
        const char* cf = std::getenv("NNUE_SCORE_COEFS");
        if (!cf) { std::fprintf(stderr, "--score needs NNUE_SCORE_COEFS=<coefs file>\n"); return 2; }
        FILE* f = std::fopen(cf, "r");
        if (!f) { std::fprintf(stderr, "cannot read %s\n", cf); return 2; }
        for (int i = 0; i < luminex::NSOLVE; ++i)
            if (std::fscanf(f, "%*d %lf", &sc_cand[i]) != 1) {
                std::fprintf(stderr, "coefs file too short at %d\n", i); return 2;
            }
        std::fclose(f);
        for (int i = 0; i < luminex::feat::NPHASE; ++i) {
            sc_base[i] = luminex::CMG[i];
            sc_base[i + luminex::feat::NPHASE] = luminex::CEG[i];
        }
        if (getenv("ETRACE_DEBUG")) {
            double mx = 0; int amx = 0;
            for (int i = 0; i < luminex::NSOLVE; ++i) {
                double d = sc_base[i] - sc_cand[i];
                if (std::fabs(d) > mx) { mx = std::fabs(d); amx = i; }
            }
            std::fprintf(stderr, "DBG[score-fill] cand[0]=%.3f base[0]=%.3f cand[607]=%.3f base[607]=%.3f maxdiff=%.4f at %d\n",
                         sc_cand[0], sc_base[0], sc_cand[607], sc_base[607], mx, amx);
        }
        sc_ok = true;
    }

    // --solve state: normal-equation accumulator + holdout stream (1-in-K by FEN hash).
    // sidx/sxv must hold ANY row's nonzero count: worst case ~NFEAT (every touched
    // feature distinct) — 256 overflowed silently on complex positions.
    // NNUE_TARGET_MAX=cp drops rows whose white-perspective target exceeds +-cp (the
    // gamepack eval tail runs to +-9096; a handful of near-mate targets would
    // otherwise dominate the squared error of a cp-space linear fit).
    luminex::SolverAcc acc;
    FILE* solve_out = nullptr;
    FILE* holdout = nullptr;
    long long n_hold = 0;
    double tmax = 1e18;
    if (const char* tm = std::getenv("NNUE_TARGET_MAX")) tmax = std::atof(tm);
    long long n_tmax = 0;
    int sidx[luminex::feat::NFEAT];
    double sxv[luminex::feat::NFEAT];
    if (solve) {
        const char* so = std::getenv("NNUE_SOLVE_OUT");
        if (!so) { std::fprintf(stderr, "--solve needs NNUE_SOLVE_OUT=<prefix>\n"); return 2; }
        std::string bin = std::string(so) + ".bin";
        solve_out = std::fopen(bin.c_str(), "wb");
        if (!solve_out) { std::fprintf(stderr, "cannot write %s\n", bin.c_str()); return 2; }
        const char* hf = std::getenv("NNUE_SOLVE_HOLDOUT");
        if (hf) {
            holdout = std::fopen(hf, "w");
            if (!holdout) { std::fprintf(stderr, "cannot write %s\n", hf); return 2; }
        }
    }

    while (std::getline(std::cin, line)) {
        size_t tab = line.find('\t');
        fen = (tab == std::string::npos) ? line : line.substr(0, tab);
        while (!fen.empty() && (fen.back() == '\r' || fen.back() == '\n')) fen.pop_back();
        if (fen.empty()) continue;

        // Cheap validity guard: a FEN has >= 4 space-separated fields. (Built
        // with -fno-exceptions; set() on garbage is UB, so pre-filter instead.)
        int fields = 1;
        for (char ch : fen) if (ch == ' ') fields++;
        if (fields < 4) { bad_fen++; continue; }
        pos.set(fen);

        luminex::Tracer t;
        t.reset();
        luminex::TraceOut out = luminex::trace_eval(pos, t);
        n++;

        // Echo the target field (if present) so trace lines are self-contained for the
        // solver: "<ph> <sf> <stm> <mirror> <target> idx:val ...". Dump format is
        // "fen\tstm\ttarget" -> target is the 3rd field; a lone "fen\tX" echo X.
        std::string target = "NA";
        if (tab != std::string::npos) {
            size_t t2 = line.find('\t', tab + 1);
            size_t t3 = (t2 == std::string::npos) ? std::string::npos : line.find('\t', t2 + 1);
            if (t2 != std::string::npos && t3 != std::string::npos)
                target = line.substr(t2 + 1, t3 - t2 - 1);          // fen\tstm\tTARGET\t...
            else if (t2 != std::string::npos)
                target = line.substr(t2 + 1);                        // fen\tstm\tTARGET(eol)
            else
                target = line.substr(tab + 1);                       // fen\tTARGET
        }

        if (!out.linear) {
            gated++;
            if (!verify && !solve) std::printf("%d %d %d G %s\n", out.ph, out.sf, out.stm, target.c_str());
            continue;
        }

        if (score) {
            double tgt = (target != "NA") ? std::strtod(target.c_str(), nullptr) : 0.0;
            double y_white = out.stm ? tgt : -tgt;
            if (std::fabs(y_white) > tmax) { n_tmax++; continue; }
            double tempo = (15.0 * t.f[luminex::feat::TEMPO_MG] + 5.0 * t.f[luminex::feat::TEMPO_EG]) / 24.0;
            double r = y_white - tempo;
            double wmg = out.ph / 24.0;
            double weg = (24 - out.ph) * out.sf / 768.0;
            double pc = 0.0, pb = 0.0;
            for (int i = 0; i < t.nt; ++i) {
                int idx = t.touched[i];
                int v = t.f[idx];
                if (v == 0) continue;
                // TEMPO features (idx >= 2*NPHASE) are accounted explicitly via the
                // tempo term above and have no solver slot -- scoring them read past
                // the end of sc_cand/sc_base (stack garbage, silently poisoned R2).
                if (idx < luminex::feat::NPHASE) { pc += sc_cand[idx] * v * wmg; pb += sc_base[idx] * v * wmg; }
                else if (idx < luminex::NSOLVE)  { pc += sc_cand[idx] * v * weg; pb += sc_base[idx] * v * weg; }
            }
            sc_n++;
            sc_sum_r += r;
            sc_ss_tot += r * r;
            sc_ssr_c += (r - pc) * (r - pc); sc_abs_c += std::fabs(r - pc);
            sc_ssr_b += (r - pb) * (r - pb); sc_abs_b += std::fabs(r - pb);
            continue;
        }

        if (solve) {
            // holdout: 1-in-10 by FEN hash, written verbatim for --score passes
            if (holdout) {
                uint32_t h = 2166136261u;
                for (char ch : fen) { h ^= (uint8_t)ch; h *= 16777619u; }
                if (h % 10 == 0) { std::fprintf(holdout, "%s\n", line.c_str()); n_hold++; continue; }
            }
            double tgt = (target != "NA") ? std::strtod(target.c_str(), nullptr) : 0.0;
            double y_white = out.stm ? tgt : -tgt;
            if (std::fabs(y_white) > tmax) { n_tmax++; continue; }
            double tempo = (15.0 * t.f[luminex::feat::TEMPO_MG] + 5.0 * t.f[luminex::feat::TEMPO_EG]) / 24.0;
            int snz = 0;
            acc.add(t, out.ph, out.sf, y_white - tempo, sidx, sxv, snz);
            continue;
        }

        if (verify) {
            luminex::Value eng = luminex::evaluate(pos, false);
            if (eng != out.mirror_white) {
                mirror_mismatch++;
                if (mirror_mismatch <= 10)
                    std::fprintf(stderr, "MIRROR MISMATCH fen=%s engine=%d mirror=%d\n",
                                 fen.c_str(), int(eng), out.mirror_white);
            }
            int stm_sign = out.stm ? 1 : -1;
            double rec = luminex::reconstruct(t, out.ph, out.sf);
            double diff = std::fabs(rec - double(out.mirror_white) * stm_sign);
            sum_rec_diff += diff;
            if (diff > max_rec_diff) max_rec_diff = diff;
            if (diff > 2.0) {
                rec_fail++;
                if (rec_fail <= 10)
                    std::fprintf(stderr, "RECONSTRUCT FAIL fen=%s diff=%.2f mirror=%d rec=%.1f\n",
                                 fen.c_str(), diff, out.mirror_white, rec);
            }
        } else {
            std::sort(t.touched, t.touched + t.nt);
            std::printf("%d %d %d %d %s", out.ph, out.sf, out.stm, out.mirror_white, target.c_str());
            for (int i = 0; i < t.nt; ++i) {
                int idx = t.touched[i];
                if (t.f[idx] == 0) continue;   // net-zero features carry no info
                std::printf(" %d:%d", idx, t.f[idx]);
            }
            std::printf("\n");
        }

        if (verify && n % 1000000 == 0)
            std::fprintf(stderr, "  ... %lldM pos: mirror_mismatch=%lld rec_fail=%lld max_diff=%.2f\n",
                         n / 1000000, mirror_mismatch, rec_fail, max_rec_diff);
    }

    if (score) {
        double mean_r = sc_sum_r / std::max(1LL, sc_n);
        double var_r = sc_ss_tot / std::max(1LL, sc_n) - mean_r * mean_r;
        double r2_c = 1.0 - (sc_ssr_c / std::max(1LL, sc_n)) / std::max(1e-9, var_r);
        double r2_b = 1.0 - (sc_ssr_b / std::max(1LL, sc_n)) / std::max(1e-9, var_r);
        std::fprintf(stderr, "SCORE %lld rows (dropped %lld >|%.0f|cp) | cand: R2=%.4f RMSE=%.2f MAE=%.2f | base: R2=%.4f RMSE=%.2f MAE=%.2f\n",
                     sc_n, n_tmax, tmax, r2_c, std::sqrt(sc_ssr_c / std::max(1LL, sc_n)), sc_abs_c / std::max(1LL, sc_n),
                     r2_b, std::sqrt(sc_ssr_b / std::max(1LL, sc_n)), sc_abs_b / std::max(1LL, sc_n));
        return 0;
    }

    if (solve) {
        // mirror A to full symmetric (accumulated upper triangle only)
        constexpr int NS_ = luminex::NSOLVE;
        for (int j = 0; j < NS_; ++j)
            for (int k = 0; k < j; ++k)
                acc.A[(size_t)j * NS_ + k] = acc.A[(size_t)k * NS_ + j];
        std::fwrite(&acc.n, sizeof(long long), 1, solve_out);
        std::fwrite(acc.A.data(), sizeof(double), (size_t)NS_ * NS_, solve_out);
        std::fwrite(acc.b.data(), sizeof(double), NS_, solve_out);
        std::fwrite(&acc.sum_r, sizeof(double), 1, solve_out);    // tail v2: target stats
        std::fwrite(&acc.sum_r2, sizeof(double), 1, solve_out);
        std::fclose(solve_out);
        if (holdout) std::fclose(holdout);
        std::fprintf(stderr, "SOLVE: %lld rows accumulated, %lld held out, %lld dropped >|%.0f|cp\n",
                     acc.n, n_hold, n_tmax, tmax);
        return 0;
    }

    if (verify) {
        std::fprintf(stderr, "VERIFY: %lld pos (%lld gated, %lld bad fen)\n", n, gated, bad_fen);
        std::fprintf(stderr, "  mirror==evaluate : %lld mismatches\n", mirror_mismatch);
        std::fprintf(stderr, "  reconstruction   : %lld fails (>2cp), max diff %.2f, mean %.3f\n",
                     rec_fail, max_rec_diff, sum_rec_diff / std::max(1LL, n - gated));
        return (mirror_mismatch == 0 && rec_fail == 0) ? 0 : 1;
    }
    return 0;
}
