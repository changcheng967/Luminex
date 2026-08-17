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
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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
constexpr int NPHASE         = 607;
constexpr int TEMPO_MG       = 2 * NPHASE;      // value = +ph / -ph
constexpr int TEMPO_EG       = 2 * NPHASE + 1;  // value = +(24-ph) / -(24-ph)
constexpr int NFEAT          = 2 * NPHASE + 2;
} // namespace feat

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
                mg -= sign * 12;
                eg -= sign * 20;
                t.add(PAWN_DOUBLED, sign); t.add(PAWN_DOUBLED + NPHASE, sign);
            }

            // Isolated pawn
            bool left = (f > FILE_A && file_count[f - 1] > 0);
            bool right = (f < FILE_H && file_count[f + 1] > 0);
            if (!left && !right) {
                mg -= sign * 15;
                eg -= sign * 20;
                t.add(PAWN_ISOLATED, sign); t.add(PAWN_ISOLATED + NPHASE, sign);
            }

            // Connected pawn (protected by another pawn)
            if (square_bb(sq) & supported_by_adj) {
                int connected_bonus = 5 + r * 3;
                mg += sign * connected_bonus;
                eg += sign * (3 + r * 2);
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
                    int lever_bonus = 4 + r * 2;
                    mg += sign * lever_bonus;
                    t.add(PAWN_LEVER + (r - 1), sign);
                    if (f >= FILE_C && f <= FILE_F) {
                        mg += sign * 3;
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
                            mg -= sign * 12;
                            eg -= sign * 15;
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
    const EvalParams& EP = g_eval_params;

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
                mg_passer += n_connected * 15;
                eg_passer += n_connected * (10 + r * 8);
                t.add(PP_CONNECTED + (r - 1), sign * n_connected);
                t.add(PP_CONNECTED + (r - 1) + NPHASE, sign * n_connected);
            }

            Bitboard behind = file_bb(f) & pos.pieces(c, ROOK);
            if (behind) {
                Square rsq2 = lsb(behind);
                bool rook_behind = (c == WHITE) ? (rank_of(rsq2) < rank_of(sq)) : (rank_of(rsq2) > rank_of(sq));
                if (rook_behind) {
                    mg_passer += 20;
                    eg_passer += 30;
                    t.add(PP_ROOK_BEHIND, sign); t.add(PP_ROOK_BEHIND + NPHASE, sign);
                }
            }

            Bitboard enemy_behind = file_bb(f) & pos.pieces(them, ROOK);
            if (enemy_behind) {
                Square ersq = lsb(enemy_behind);
                bool enemy_rook_behind = (c == WHITE) ? (rank_of(ersq) < rank_of(sq)) : (rank_of(ersq) > rank_of(sq));
                if (enemy_rook_behind) {
                    mg_passer -= 10;
                    eg_passer -= 15;
                    t.add(PP_ENEMY_ROOK, sign); t.add(PP_ENEMY_ROOK + NPHASE, sign);
                }
            }

            Square our_ksq = ksq_arr[c_idx];
            Square their_ksq = ksq_arr[c_idx ^ 1];
            Square promo_sq = relative_square(c, make_square(f, RANK_8));
            int our_kdist = distance(our_ksq, promo_sq);
            int their_kdist = distance(their_ksq, promo_sq);
            eg_passer += (their_kdist - our_kdist) * 8;
            t.add(PP_KING_PROX + NPHASE, sign * (their_kdist - our_kdist));

            if (their_kdist > our_kdist + (c == pos.side_to_move() ? 0 : 1)) {
                int unreachable = their_kdist - our_kdist;
                eg_passer += unreachable * (unreachable > 4 ? 25 : 15);
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
                    int blk = r * r;
                    mg_passer -= blk / 3;
                    eg_passer -= blk * 2;
                    t.add(PP_BLOCKADE + (r - 1), sign); t.add(PP_BLOCKADE + (r - 1) + NPHASE, sign);
                } else if (stop_own) {
                    mg_passer -= 6;
                    eg_passer -= 10;
                    t.add(PP_STOP_OWN, sign); t.add(PP_STOP_OWN + NPHASE, sign);
                } else if (stop_attacked_by_enemy) {
                    mg_passer -= 4;
                    eg_passer -= 8;
                    t.add(PP_STOP_ATT, sign); t.add(PP_STOP_ATT + NPHASE, sign);
                } else {
                    mg_passer += 2 + r;
                    eg_passer += 4 + r * 2;
                    t.add(PP_SAFE + (r - 1), sign); t.add(PP_SAFE + (r - 1) + NPHASE, sign);
                    if (!enemy_on_path) {
                        eg_passer += 8 + r * 3;
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
                        mg_score += sign * (EP.outpost_knight_mg + (kr - 3) * 5);
                        eg_score += sign * (EP.outpost_knight_eg + (kr - 3) * 3);
                        t.add(OUTPOST_N + (kr - 3), sign);
                        t.add(OUTPOST_N + (kr - 3) + NPHASE, sign);
                    }
                }
            }

            if (mob == 0) {
                mg_score -= sign * 10;
                eg_score -= sign * 5;
                t.add(TRAPPED_N, sign); t.add(TRAPPED_N + NPHASE, sign);
            }

            if (distance(sq, ksq_arr[c_idx]) > 3) {
                mg_score -= sign * EP.far_knight_mg;
                eg_score -= sign * EP.far_knight_eg;
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
                static constexpr int BscMG[7] = { 20, 15, 10, 5, 0, -8, -15 };
                static constexpr int BscEG[7] = { 40, 30, 20, 10, 0, -10, -20 };
                mg_score += sign * BscMG[same_color_pawns];
                eg_score += sign * BscEG[same_color_pawns];
                t.add(BSC + same_color_pawns, sign);
                t.add(BSC + same_color_pawns + NPHASE, sign);
            }

            if (popcount(attacks & BB_CENTER) >= 2) {
                mg_score += sign * 12;
                eg_score += sign * 20;
                t.add(BISH_LONG, sign); t.add(BISH_LONG + NPHASE, sign);
            }

            {
                Rank br = relative_rank(c, sq);
                if (br >= RANK_4 && br <= RANK_6) {
                    if ((pawn_attacks_bb(c, our_pawns) & square_bb(sq)) &&
                        !(pawn_attacks_bb(them, their_pawns) & square_bb(sq))) {
                        mg_score += sign * EP.outpost_bishop_mg;
                        eg_score += sign * EP.outpost_bishop_eg;
                        t.add(OUTPOST_B, sign); t.add(OUTPOST_B + NPHASE, sign);
                    }
                }
            }

            if (distance(sq, ksq_arr[c_idx]) > 3) {
                mg_score -= sign * EP.far_bishop_mg;
                eg_score -= sign * EP.far_bishop_eg;
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
                mg_score += sign * EP.rook_open_mg;
                eg_score += sign * EP.rook_open_eg;
                t.add(ROOK_OPEN, sign); t.add(ROOK_OPEN + NPHASE, sign);
            } else if (!(our_pawns & file_bb(f))) {
                mg_score += sign * EP.rook_semi_open_mg;
                eg_score += sign * EP.rook_semi_open_eg;
                t.add(ROOK_SEMI, sign); t.add(ROOK_SEMI + NPHASE, sign);
            }

            Rank rr = relative_rank(c, rank_of(sq));
            if (rr == RANK_7) {
                mg_score += sign * EP.rook_7th_mg;
                eg_score += sign * EP.rook_7th_eg;
                t.add(ROOK_7TH, sign); t.add(ROOK_7TH + NPHASE, sign);
            }

            if (file_bb(f) & pos.pieces(them, QUEEN)) {
                mg_score += sign * 15;
                eg_score += sign * 5;
                t.add(ROOK_XRAY_Q, sign); t.add(ROOK_XRAY_Q + NPHASE, sign);
            }

            if (distance(sq, ksq_arr[c_idx]) > 3) {
                mg_score -= sign * 8;
                eg_score -= sign * 5;
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
                        mg_score += sign * (-8);
                        eg_score += sign * (-12);
                        t.add(ROOK_TRAP_C, sign); t.add(ROOK_TRAP_C + NPHASE, sign);
                    } else {
                        mg_score += sign * (-55);
                        eg_score += sign * (-25);
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
                    mg_score += sign * (-8);
                    eg_score += sign * (-8);
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
                mg_score -= sign * 8;
                eg_score -= sign * 3;
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
                int weight = (r == 1) ? EP.pawn_shield_knight
                             : (r == 2) ? EP.pawn_shield_center
                             : EP.pawn_shield_rook;
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
                mg_score -= sign * EP.open_file_penalty_mg;
                t.add(OPEN_KFILE, sign);
            }
            int adj_open = 0;
            if (kfile > FILE_A && !(all_pawns & file_bb(File(kfile - 1)))) {
                mg_score -= sign * EP.open_file_penalty_eg;
                adj_open++;
            }
            if (kfile < FILE_H && !(all_pawns & file_bb(File(kfile + 1)))) {
                mg_score -= sign * EP.open_file_penalty_eg;
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
                        int storm_danger = (pr - 3) * EP.pawn_storm;
                        mg_score -= sign * storm_danger;
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
            mg_score += sign * 30; eg_score += sign * 10;
            t.add(CASTLED, sign); t.add(CASTLED + NPHASE, sign);
        }

        CastlingRight ks_cr = c == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE;
        CastlingRight qs_cr = c == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE;
        if (!pos.castling_allowed(c, ks_cr) && !pos.castling_allowed(c, qs_cr) && !castled) {
            mg_score -= sign * 25;
            eg_score -= sign * 10;
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
                mg_score += sign * EP.hanging_pawn_mg * popcount(hanging_pawns);
                eg_score += sign * EP.hanging_pawn_eg * popcount(hanging_pawns);
                t.add(HANG_PAWN, sign * popcount(hanging_pawns));
                t.add(HANG_PAWN + NPHASE, sign * popcount(hanging_pawns));
            }
            Bitboard hanging_pieces = (pos.pieces(them, KNIGHT) | pos.pieces(them, BISHOP)
                                     | pos.pieces(them, ROOK) | pos.pieces(them, QUEEN))
                                     & ~all_attacks[them] & all_attacks[c];
            if (hanging_pieces) {
                mg_score += sign * 25 * popcount(hanging_pieces);
                eg_score += sign * 15 * popcount(hanging_pieces);
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
                    mg_score += sign * pin_count * 20;
                    eg_score += sign * pin_count * 10;
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
            mg_score += space[WHITE] * 4 - space[BLACK] * 2;
            t.add(SPACE_OWN, space[WHITE]);
            t.add(SPACE_OPP, space[BLACK]);
        } else if (pawn_count[BLACK] > pawn_count[WHITE]) {
            mg_score -= space[BLACK] * 4 - space[WHITE] * 2;
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
            mg_score += sign * (25 - pawns * 2);
            eg_score += sign * (45 - pawns * 3);
            t.add(IMBAL_CONST, sign); t.add(IMBAL_CONST + NPHASE, sign);
            t.add(IMBAL_PAWN, sign * pawns); t.add(IMBAL_PAWN + NPHASE, sign * pawns);
        }
    }

    if (bishop_count[WHITE] >= 2) {
        mg_score += g_eval_params.bishop_pair_mg; eg_score += g_eval_params.bishop_pair_eg;
        t.add(BISHOP_PAIR, 1); t.add(BISHOP_PAIR + NPHASE, 1);
    }
    if (bishop_count[BLACK] >= 2) {
        mg_score -= g_eval_params.bishop_pair_mg; eg_score -= g_eval_params.bishop_pair_eg;
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
            mg_score += w_knights * knight_adj * 2;
            mg_score -= b_knights * knight_adj * 2;
            eg_score += w_knights * knight_adj;
            eg_score -= b_knights * knight_adj;
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
            eg_score += diff * simplification / 256;
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
            int danger_mg = (au_pos * au_pos) / 8;
            int danger_eg = au_pos / 2;

            if (!enemy_qu) danger_mg = danger_mg / 4;
            if (!enemy_qu && !enemy_ro) danger_mg = danger_mg / 4;

            Bitboard king_moves = king_attacks_bb(our_ksq);
            Bitboard safe_king_squares = king_moves & ~pos.pieces(c) & ~all_attacks[them];
            int flight = popcount(safe_king_squares);
            int flight1 = 0, flight2 = 0;
            if (flight <= 1) { danger_mg += 80; danger_eg += 40; flight1 = 1; }
            else if (flight == 2) { danger_mg += 30; danger_eg += 15; flight2 = 1; }

            mg_score -= sign * danger_mg;
            eg_score -= sign * danger_eg;
            // KS_AU feature: the pre-flight attack-unit danger (per phase).
            int au_danger_mg = (au_pos * au_pos) / 8;
            if (!enemy_qu) au_danger_mg = au_danger_mg / 4;
            if (!enemy_qu && !enemy_ro) au_danger_mg = au_danger_mg / 4;
            t.add(KS_AU, -sign * au_danger_mg);
            t.add(KS_AU + NPHASE, -sign * (au_pos / 2));
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
    using namespace feat;
    const EvalParams& EP = g_eval_params;
    const int* PVMG = etrace::piece_value_mg();
    const int* PVEG = etrace::piece_value_eg();

    for (int i = 0; i < NPHASE; ++i) { CMG[i] = 0.0; CEG[i] = 0.0; }

    for (int pt = 0; pt < 6; ++pt) { CMG[MAT + pt] = PVMG[pt]; CEG[MAT + pt] = PVEG[pt]; }
    for (int pt = 0; pt < 6; ++pt)
        for (int s = 0; s < 64; ++s) {
            CMG[PST + pt * 64 + s] = PST_MG_TABLE[WHITE][pt][s];
            CEG[PST + pt * 64 + s] = PST_EG_TABLE[WHITE][pt][s];
        }
    for (int i = 0; i < 9;  ++i) { CMG[MOB_N + i] = etrace::knight_mob_mg()[i]; CEG[MOB_N + i] = etrace::knight_mob_eg()[i]; }
    for (int i = 0; i < 14; ++i) { CMG[MOB_B + i] = etrace::bishop_mob_mg()[i]; CEG[MOB_B + i] = etrace::bishop_mob_eg()[i]; }
    for (int i = 0; i < 15; ++i) { CMG[MOB_R + i] = etrace::rook_mob_mg()[i];   CEG[MOB_R + i] = etrace::rook_mob_eg()[i]; }
    for (int i = 0; i < 28; ++i) { CMG[MOB_Q + i] = etrace::queen_mob_mg()[i];  CEG[MOB_Q + i] = etrace::queen_mob_eg()[i]; }

    CMG[PAWN_DOUBLED] = -12; CEG[PAWN_DOUBLED] = -20;
    CMG[PAWN_ISOLATED] = -15; CEG[PAWN_ISOLATED] = -20;
    for (int r = 1; r <= 6; ++r) { CMG[PAWN_CONNECTED + r - 1] = 5 + r * 3; CEG[PAWN_CONNECTED + r - 1] = 3 + r * 2; }
    for (int r = 1; r <= 6; ++r) { CMG[PAWN_PHALANX + r - 1] = etrace::phalanx_mg()[r]; CEG[PAWN_PHALANX + r - 1] = etrace::phalanx_eg()[r]; }
    for (int r = 1; r <= 6; ++r) { CMG[PAWN_LEVER + r - 1] = 4 + r * 2; CEG[PAWN_LEVER + r - 1] = 0; }
    CMG[PAWN_LEVER_C] = 3; CEG[PAWN_LEVER_C] = 0;
    CMG[PAWN_BACKWARD] = -12; CEG[PAWN_BACKWARD] = -15;
    for (int sup = 0; sup < 2; ++sup)
        for (int r = 1; r <= 6; ++r) {
            CMG[PAWN_CAND + sup * 7 + r - 1] = etrace::cand_mg()[sup * 8 + r];
            CEG[PAWN_CAND + sup * 7 + r - 1] = etrace::cand_eg()[sup * 8 + r];
        }
    for (int r = 1; r <= 6; ++r) { CMG[PAWN_PASSED + r - 1] = etrace::passed_mg()[r]; CEG[PAWN_PASSED + r - 1] = etrace::passed_eg()[r]; }

    for (int r = 1; r <= 6; ++r) { CMG[PP_CONNECTED + r - 1] = 15; CEG[PP_CONNECTED + r - 1] = 10 + r * 8; }
    CMG[PP_ROOK_BEHIND] = 20; CEG[PP_ROOK_BEHIND] = 30;
    CMG[PP_ENEMY_ROOK] = -10; CEG[PP_ENEMY_ROOK] = -15;
    CMG[PP_KING_PROX] = 0;    CEG[PP_KING_PROX] = 8;
    CMG[PP_UNREACH_S] = 0;    CEG[PP_UNREACH_S] = 15;
    CMG[PP_UNREACH_B] = 0;    CEG[PP_UNREACH_B] = 25;
    for (int r = 1; r <= 6; ++r) { CMG[PP_BLOCKADE + r - 1] = -(r * r / 3); CEG[PP_BLOCKADE + r - 1] = -(r * r * 2); }
    CMG[PP_STOP_OWN] = -6; CEG[PP_STOP_OWN] = -10;
    CMG[PP_STOP_ATT] = -4; CEG[PP_STOP_ATT] = -8;
    for (int r = 1; r <= 6; ++r) { CMG[PP_SAFE + r - 1] = 2 + r; CEG[PP_SAFE + r - 1] = 4 + r * 2; }
    for (int r = 1; r <= 6; ++r) { CMG[PP_CLEAR + r - 1] = 0; CEG[PP_CLEAR + r - 1] = 8 + r * 3; }

    for (int k = 0; k < 3; ++k) { // kr = RANK_4..RANK_6 (0-based 3,4,5); engine adds (kr-3)*{5,3}
        CMG[OUTPOST_N + k] = EP.outpost_knight_mg + k * 5;
        CEG[OUTPOST_N + k] = EP.outpost_knight_eg + k * 3;
    }
    CMG[TRAPPED_N] = -10; CEG[TRAPPED_N] = -5;
    CMG[FAR_N] = -EP.far_knight_mg; CEG[FAR_N] = -EP.far_knight_eg;
    CMG[FAR_B] = -EP.far_bishop_mg; CEG[FAR_B] = -EP.far_bishop_eg;
    {
        static constexpr int BscMG[7] = { 20, 15, 10, 5, 0, -8, -15 };
        static constexpr int BscEG[7] = { 40, 30, 20, 10, 0, -10, -20 };
        for (int i = 0; i < 7; ++i) { CMG[BSC + i] = BscMG[i]; CEG[BSC + i] = BscEG[i]; }
    }
    CMG[BISH_LONG] = 12; CEG[BISH_LONG] = 20;
    CMG[OUTPOST_B] = EP.outpost_bishop_mg; CEG[OUTPOST_B] = EP.outpost_bishop_eg;
    CMG[ROOK_OPEN] = EP.rook_open_mg; CEG[ROOK_OPEN] = EP.rook_open_eg;
    CMG[ROOK_SEMI] = EP.rook_semi_open_mg; CEG[ROOK_SEMI] = EP.rook_semi_open_eg;
    CMG[ROOK_7TH] = EP.rook_7th_mg; CEG[ROOK_7TH] = EP.rook_7th_eg;
    CMG[ROOK_XRAY_Q] = 15; CEG[ROOK_XRAY_Q] = 5;
    CMG[FAR_R] = -8; CEG[FAR_R] = -5;
    CMG[ROOK_TRAP_C] = -8; CEG[ROOK_TRAP_C] = -12;
    CMG[ROOK_TRAP_NC] = -55; CEG[ROOK_TRAP_NC] = -25;
    CMG[ROOK_BLOCKED] = -8; CEG[ROOK_BLOCKED] = -8;
    CMG[FAR_Q] = -8; CEG[FAR_Q] = -3;
    CMG[SHIELD_R1] = EP.pawn_shield_knight; CEG[SHIELD_R1] = 0;
    CMG[SHIELD_R2] = EP.pawn_shield_center; CEG[SHIELD_R2] = 0;
    CMG[SHIELD_R3] = EP.pawn_shield_rook;   CEG[SHIELD_R3] = 0;
    CMG[OPEN_KFILE] = -EP.open_file_penalty_mg; CEG[OPEN_KFILE] = 0;
    CMG[OPEN_ADJ] = -EP.open_file_penalty_eg;   CEG[OPEN_ADJ] = 0; // EG param, added to MG score (engine quirk)
    CMG[STORM] = -EP.pawn_storm; CEG[STORM] = 0;
    CMG[CASTLED] = 30; CEG[CASTLED] = 10;
    CMG[NOCASTLE] = -25; CEG[NOCASTLE] = -10;
    const int* thr_mg[5] = { etrace::pawn_threat_mg(), etrace::knight_threat_mg(), etrace::bishop_threat_mg(), etrace::rook_threat_mg(), etrace::queen_threat_mg() };
    const int* thr_eg[5] = { etrace::pawn_threat_eg(), etrace::knight_threat_eg(), etrace::bishop_threat_eg(), etrace::rook_threat_eg(), etrace::queen_threat_eg() };
    for (int a = 0; a < 5; ++a)
        for (int p = 0; p < 5; ++p) { CMG[THR + a * 5 + p] = thr_mg[a][p]; CEG[THR + a * 5 + p] = thr_eg[a][p]; }
    CMG[HANG_PAWN] = EP.hanging_pawn_mg; CEG[HANG_PAWN] = EP.hanging_pawn_eg;
    CMG[HANG_PIECE] = 25; CEG[HANG_PIECE] = 15;
    CMG[PINNED] = 20; CEG[PINNED] = 10;
    CMG[SPACE_OWN] = 4; CEG[SPACE_OWN] = 0;
    CMG[SPACE_OPP] = -2; CEG[SPACE_OPP] = 0;
    CMG[IMBAL_CONST] = 25; CEG[IMBAL_CONST] = 45;
    CMG[IMBAL_PAWN] = -2; CEG[IMBAL_PAWN] = -3;
    CMG[BISHOP_PAIR] = EP.bishop_pair_mg; CEG[BISHOP_PAIR] = EP.bishop_pair_eg;
    CMG[KNIGHT_PAWN] = 2; CEG[KNIGHT_PAWN] = 1;
    CMG[TRADEDOWN] = 0; CEG[TRADEDOWN] = 1;
    CMG[KS_AU] = 1; CEG[KS_AU] = 1;
    CMG[KS_FLIGHT1] = 80; CEG[KS_FLIGHT1] = 40;
    CMG[KS_FLIGHT2] = 30; CEG[KS_FLIGHT2] = 15;
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
    bool verify = false, dump_coefs = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--verify") == 0) verify = true;
        if (std::strcmp(argv[i], "--dump-coefs") == 0) dump_coefs = true;
    }

    luminex::init_magic_bitboards();   // attack tables (set()/eval depend on these)
    luminex::init_line_tables();       // BetweenBB/LineBB for pin detection
    luminex::init_zobrist();
    luminex::init_evaluation();
    luminex::init_coefs();

    if (dump_coefs) {
        for (int i = 0; i < luminex::feat::NFEAT; ++i) {
            double mg = (i < luminex::feat::NPHASE) ? luminex::CMG[i]
                       : (i < 2 * luminex::feat::NPHASE) ? luminex::CMG[i - luminex::feat::NPHASE] : 0.0;
            double eg = (i < luminex::feat::NPHASE) ? luminex::CEG[i]
                       : (i < 2 * luminex::feat::NPHASE) ? luminex::CEG[i - luminex::feat::NPHASE] : 0.0;
            std::printf("%d %.1f %.1f\n", i, mg, eg);
        }
        return 0;
    }

    luminex::Position pos;
    std::string line, fen;
    long long n = 0, gated = 0, bad_fen = 0;
    long long mirror_mismatch = 0, rec_fail = 0;
    double max_rec_diff = 0.0, sum_rec_diff = 0.0;

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

        if (!out.linear) {
            gated++;
            if (!verify) std::printf("%d %d %d G\n", out.ph, out.sf, out.stm);
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
            std::printf("%d %d %d %d", out.ph, out.sf, out.stm, out.mirror_white);
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

    if (verify) {
        std::fprintf(stderr, "VERIFY: %lld pos (%lld gated, %lld bad fen)\n", n, gated, bad_fen);
        std::fprintf(stderr, "  mirror==evaluate : %lld mismatches\n", mirror_mismatch);
        std::fprintf(stderr, "  reconstruction   : %lld fails (>2cp), max diff %.2f, mean %.3f\n",
                     rec_fail, max_rec_diff, sum_rec_diff / std::max(1LL, n - gated));
        return (mirror_mismatch == 0 && rec_fail == 0) ? 0 : 1;
    }
    return 0;
}
