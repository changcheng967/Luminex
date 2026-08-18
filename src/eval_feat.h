// eval_feat.h — shared HCE feature-index space for evaluation.cpp, eval_trace.cpp,
// and the generated fitted-parameter header (eval_fitted.h).
//
// The index space IS the solver space: MG block 0..606, EG block 607..1213.
// eval_fitted.h defines FE_MG[607] / FE_EG[607] holding the CURRENT coefficient
// for every feature (engine defaults until a distillation fit is applied).
// Keeping one numbering across engine + tracer + solver means the --verify
// invariants revalidate the engine against whatever fit is applied.
#pragma once

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
constexpr int NPHASE         = 610;
constexpr int TEMPO_MG       = 2 * NPHASE;      // value = +ph / -ph
constexpr int TEMPO_EG       = 2 * NPHASE + 1;  // value = +(24-ph) / -(24-ph)
constexpr int NFEAT          = 2 * NPHASE + 2;
} // namespace feat
} // namespace luminex
