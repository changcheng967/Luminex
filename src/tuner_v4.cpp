// tuner_v4.cpp — Phase 1 PMG-style JOINT Adam tuner.
// Tunes PST (768) + mobility tables (132) + eval bonuses (6) = 906 params TOGETHER,
// via analytic gradients. All terms are LINEAR in their params → one gradient pass
// computes all 906 partials simultaneously (same cost as one eval pass).
// This solves v3's failure: tuning PST alone unbalanced the eval. Tuning ALL terms
// jointly lets the optimizer find the balanced optimum.
#define _CRT_SECURE_NO_WARNINGS
#include "luminex.h"
#include "board.h"
#include "movegen.h"
#include "evaluation.h"
#include "bitboard.h"
#include "tuner_params.h"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace luminex {
std::atomic<uint64_t> nodes{0};
std::atomic<bool> stop{false};
int root_depth = 0; Value root_score = VALUE_ZERO;
int num_threads = 1; Move previous_root_best = MOVE_NONE;
std::atomic<bool> ponder_mode{false}, ponderhit_received{false};
Limits limits; SearchParams params; SearchStats g_stats;
}
using namespace luminex;

// ===== Parameter layout: 906 total =====
// [  0..767]  PST: phase*384 + pt*64 + sq   (phase 0=MG 1=EG; pt 0-5; sq 0-63)
// [768..899]  Mobility: knight(9×2) bishop(14×2) rook(15×2) queen(28×2) = 132
// [900..905]  Bonuses: BishopPairMG/EG, RookOpenMG/EG, RookSemiMG/EG
constexpr int N_PARAMS = 906;
constexpr int PST_BASE = 0;        // 768
constexpr int MOB_BASE = 768;      // 132
constexpr int BONUS_BASE = 900;    // 6

// Mobility sub-layout within [768..899]
// KNIGHT: MG [768..776], EG [777..785]     (9 each)
// BISHOP: MG [786..799], EG [800..813]     (14 each)
// ROOK:   MG [814..828], EG [829..843]     (15 each)
// QUEEN:  MG [844..871], EG [872..899]     (28 each)
struct MobRange { int mg_off, eg_off, max_mob; };
constexpr MobRange MOB_RANGES[6] = {
    {0, 0, 0},                          // PAWN (no mobility)
    {0, 9, 8},                           // KNIGHT
    {18, 32, 13},                        // BISHOP
    {46, 61, 14},                        // ROOK
    {76, 104, 27},                       // QUEEN
    {0, 0, 0},                           // KING (no mobility table)
};

// ===== Tunable state =====
static double theta[N_PARAMS];
static double adam_m[N_PARAMS], adam_v[N_PARAMS];
static double K = 1.0, LR = 0.1;
static double ADAM_B1 = 0.9, ADAM_B2 = 0.999, EPS = 1e-8;

// ===== Dataset =====
struct Entry { std::string fen; double result; };
static std::vector<Entry> dataset;

// ===== Init: snapshot current values =====
static void init_params() {
    // PST
    for (int pt = 0; pt < 6; pt++)
        for (int sq = 0; sq < 64; sq++) {
            theta[pt*64+sq]         = (double)PST_MG_TABLE[WHITE][pt][sq];
            theta[384+pt*64+sq]     = (double)PST_EG_TABLE[WHITE][pt][sq];
        }
    // Mobility (from g_params, already initialized by init_tuner_params + sync_eval_params)
    int mob_param_ids[][2] = {  // {g_params enum base, count}
        {KNIGHT_MOB_MG_0, 9}, {KNIGHT_MOB_EG_0, 9},
        {BISHOP_MOB_MG_0, 14}, {BISHOP_MOB_EG_0, 14},
        {ROOK_MOB_MG_0, 15}, {ROOK_MOB_EG_0, 15},
        {QUEEN_MOB_MG_0, 28}, {QUEEN_MOB_EG_0, 28},
    };
    int offset = 0;
    for (auto& [base, cnt] : mob_param_ids) {
        for (int i = 0; i < cnt; i++)
            theta[MOB_BASE + offset + i] = (double)g_params[base + i];
        offset += cnt;
    }
    // Bonuses
    int bonus_ids[] = {BISHOP_PAIR_MG, BISHOP_PAIR_EG, ROOK_OPEN_FILE_MG,
                        ROOK_OPEN_FILE_EG, ROOK_SEMI_OPEN_MG, ROOK_SEMI_OPEN_EG};
    for (int i = 0; i < 6; i++)
        theta[BONUS_BASE + i] = (double)g_params[bonus_ids[i]];
}

// ===== Apply: write tuned values back to engine =====
static void apply_params() {
    // PST → PST_MG/EG_TABLE → init_evaluation (mirrors BLACK + rebuilds PSQ)
    for (int pt = 0; pt < 6; pt++)
        for (int sq = 0; sq < 64; sq++) {
            PST_MG_TABLE[WHITE][pt][sq] = (Score)std::round(theta[pt*64+sq]);
            PST_EG_TABLE[WHITE][pt][sq] = (Score)std::round(theta[384+pt*64+sq]);
        }
    init_evaluation();

    // Mobility → g_params → sync_eval_params
    int mob_param_ids[][2] = {
        {KNIGHT_MOB_MG_0, 9}, {KNIGHT_MOB_EG_0, 9},
        {BISHOP_MOB_MG_0, 14}, {BISHOP_MOB_EG_0, 14},
        {ROOK_MOB_MG_0, 15}, {ROOK_MOB_EG_0, 15},
        {QUEEN_MOB_MG_0, 28}, {QUEEN_MOB_EG_0, 28},
    };
    int offset = 0;
    for (auto& [base, cnt] : mob_param_ids) {
        for (int i = 0; i < cnt; i++)
            g_params[base + i] = (int)std::round(theta[MOB_BASE + offset + i]);
        offset += cnt;
    }
    // Bonuses → g_params
    int bonus_ids[] = {BISHOP_PAIR_MG, BISHOP_PAIR_EG, ROOK_OPEN_FILE_MG,
                        ROOK_OPEN_FILE_EG, ROOK_SEMI_OPEN_MG, ROOK_SEMI_OPEN_EG};
    for (int i = 0; i < 6; i++)
        g_params[bonus_ids[i]] = (int)std::round(theta[BONUS_BASE + i]);

    sync_eval_params();  // copies g_params → static mobility arrays + g_eval_params
    clear_pawn_table();
}

// ===== Loss + analytic gradient in ONE pass =====
static double compute_loss_and_gradient(double* grad) {
    double total_loss = 0.0;
    int n = (int)dataset.size();

    #pragma omp parallel
    {
        double local_grad[N_PARAMS];
        std::memset(local_grad, 0, sizeof(local_grad));
        double local_loss = 0.0;

        #pragma omp for schedule(dynamic, 512)
        for (int i = 0; i < n; i++) {
            Position pos;
            pos.set(dataset[i].fen);

            int eval_cp = (int)evaluate(pos, false);
            double sig = 1.0 / (1.0 + std::exp(-eval_cp / (400.0 * K)));
            double R = dataset[i].result;  // game result 0/0.5/1
            double err = sig - R;
            local_loss += err * err;

            double signal = 2.0 * err * sig * (1.0 - sig) / (400.0 * K);

            int phase = popcount(pos.pieces(KNIGHT)) + popcount(pos.pieces(BISHOP))
                      + popcount(pos.pieces(ROOK)) * 2 + popcount(pos.pieces(QUEEN)) * 4;
            if (phase > 24) phase = 24;
            double mg_w = (double)phase / 24.0;
            double eg_w = 1.0 - mg_w;

            int stm_sign = (pos.side_to_move() == WHITE) ? 1 : -1;

            // Precompute per-color mobility areas
            Bitboard occupied = pos.pieces();
            Bitboard wp = pos.pieces(WHITE, PAWN), bp = pos.pieces(BLACK, PAWN);
            Bitboard mob_area_w = ~pawn_attacks_bb(BLACK, bp);
            Bitboard mob_area_b = ~pawn_attacks_bb(WHITE, wp);
            Bitboard w_pc = pos.pieces(WHITE), b_pc = pos.pieces(BLACK);
            int w_bishops = popcount(pos.pieces(WHITE, BISHOP));
            int b_bishops = popcount(pos.pieces(BLACK, BISHOP));
            Bitboard all_pawns = pos.pieces(PAWN);

            // Iterate board: scatter gradients for PST + mobility + bonuses
            for (int sq = 0; sq < 64; sq++) {
                Piece p = pos.piece_on(Square(sq));
                if (p == NO_PIECE) continue;
                Color c = (p < 6) ? WHITE : BLACK;
                PieceType pt = PieceType(p % 6);
                if (pt >= 6) continue;

                int rel_sq = (c == WHITE) ? sq : (sq ^ 56);
                int sgn = (c == WHITE) ? 1 : -1;
                Bitboard our_pieces = (c == WHITE) ? w_pc : b_pc;
                Bitboard mob_area = (c == WHITE) ? mob_area_w : mob_area_b;

                // --- PST gradient ---
                local_grad[pt*64 + rel_sq]         += sgn * stm_sign * mg_w * signal;
                local_grad[384 + pt*64 + rel_sq]   += sgn * stm_sign * eg_w * signal;

                // --- Mobility gradient (knight/bishop/rook/queen only) ---
                const auto& mr = MOB_RANGES[pt];
                if (mr.max_mob > 0) {
                    int mob = 0;
                    Bitboard attacks = 0;
                    switch (pt) {
                        case KNIGHT:
                            attacks = knight_attacks_bb(Square(sq));
                            break;
                        case BISHOP:
                            attacks = bishop_attacks_bb(Square(sq), occupied);
                            break;
                        case ROOK:
                            attacks = rook_attacks_bb(Square(sq),
                                occupied ^ pos.pieces(c, ROOK) ^ pos.pieces(c, QUEEN));
                            break;
                        case QUEEN:
                            attacks = bishop_attacks_bb(Square(sq), occupied ^ pos.pieces(c, BISHOP))
                                     | rook_attacks_bb(Square(sq), occupied ^ pos.pieces(c, ROOK));
                            break;
                    }
                    mob = popcount(attacks & mob_area & ~our_pieces);
                    if (mob > mr.max_mob) mob = mr.max_mob;

                    local_grad[MOB_BASE + mr.mg_off + mob] += sgn * stm_sign * mg_w * signal;
                    local_grad[MOB_BASE + mr.eg_off + mob] += sgn * stm_sign * eg_w * signal;
                }

                // --- Rook open/semi-open file bonus gradient ---
                if (pt == ROOK) {
                    File f = file_of(Square(sq));
                    Bitboard fbb = file_bb(f);
                    bool is_open = !(all_pawns & fbb);
                    bool is_semi = !is_open && !((c == WHITE ? wp : bp) & fbb);
                    if (is_open) {
                        local_grad[BONUS_BASE + 2] += sgn * stm_sign * mg_w * signal; // RookOpenMG
                        local_grad[BONUS_BASE + 3] += sgn * stm_sign * eg_w * signal; // RookOpenEG
                    } else if (is_semi) {
                        local_grad[BONUS_BASE + 4] += sgn * stm_sign * mg_w * signal; // RookSemiMG
                        local_grad[BONUS_BASE + 5] += sgn * stm_sign * eg_w * signal; // RookSemiEG
                    }
                }
            }

            // --- Bishop pair bonus gradient (checked once per position, not per piece) ---
            {
                int pair_indicator = (w_bishops >= 2 ? 1 : 0) - (b_bishops >= 2 ? 1 : 0);
                if (pair_indicator != 0) {
                    local_grad[BONUS_BASE + 0] += stm_sign * mg_w * pair_indicator * signal;
                    local_grad[BONUS_BASE + 1] += stm_sign * eg_w * pair_indicator * signal;
                }
            }
        }

        #pragma omp critical
        {
            total_loss += local_loss;
            for (int i = 0; i < N_PARAMS; i++) grad[i] += local_grad[i];
        }
    }
    double scale = 1.0 / n;
    for (int i = 0; i < N_PARAMS; i++) grad[i] *= scale;
    return total_loss * scale;
}

// ===== Adam step =====
static void adam_step(const double* grad, int iter) {
    double bc1 = 1.0 - std::pow(ADAM_B1, iter);
    double bc2 = 1.0 - std::pow(ADAM_B2, iter);
    for (int i = 0; i < N_PARAMS; i++) {
        adam_m[i] = ADAM_B1 * adam_m[i] + (1.0 - ADAM_B1) * grad[i];
        adam_v[i] = ADAM_B2 * adam_v[i] + (1.0 - ADAM_B2) * grad[i] * grad[i];
        double mhat = adam_m[i] / bc1;
        double vhat = adam_v[i] / bc2;
        theta[i] -= LR * mhat / (std::sqrt(vhat) + EPS);
    }
}

// ===== Dataset loading =====
static bool load_dataset(const char* path) {
    FILE* f = std::fopen(path, "r");
    if (!f) { std::fprintf(stderr, "Cannot open %s\n", path); return false; }
    char line[1024];
    while (std::fgets(line, sizeof(line), f)) {
        int len = (int)std::strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
        if (len == 0) continue;
        char* sep = std::strrchr(line, '|');
        if (!sep) continue;
        *sep = 0;
        double r = std::atof(sep + 1);  // SF eval_cp (any integer) or game result (0/0.5/1)
        dataset.push_back({std::string(line), r});
    }
    std::fclose(f);
    std::printf("Loaded %d positions from %s\n", (int)dataset.size(), path);
    return !dataset.empty();
}

// ===== Main =====
int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s dataset.txt [K] [iters] [lr]\n", argv[0]);
        return 1;
    }

    init_magic_bitboards();
    init_line_tables();
    init_zobrist();
    init_evaluation();
    init_tuner_params();
    sync_eval_params();
    clear_pawn_table();

    if (argc >= 3) K = std::atof(argv[2]);
    int max_iters = (argc >= 4) ? std::atoi(argv[3]) : 200;
    if (argc >= 5) LR = std::atof(argv[4]);

    setvbuf(stdout, nullptr, _IONBF, 0);
    if (!load_dataset(argv[1])) return 1;

    init_params();
    apply_params();

    std::printf("=== v4 JOINT Adam tuner: %d params (PST 768 + Mob 132 + Bonus 6) ===\n", N_PARAMS);
    std::printf("%d positions, K=%.2f lr=%.3f iters=%d\n\n", (int)dataset.size(), K, LR, max_iters);

    double grad[N_PARAMS];
    double loss = compute_loss_and_gradient(grad);
    std::printf("Initial loss: %.6f\n", loss);

    for (int iter = 1; iter <= max_iters; iter++) {
        adam_step(grad, iter);
        apply_params();
        loss = compute_loss_and_gradient(grad);
        std::printf("Iter %3d: loss=%.6f\n", iter, loss);
    }

    // Output tuned values
    std::printf("\n=== TUNED PST ===\n");
    for (int phase = 0; phase < 2; phase++) {
        for (int pt = 0; pt < 6; pt++) {
            const char* pn[] = {"PAWN","KNIGHT","BISHOP","ROOK","QUEEN","KING"};
            const char* ph = (phase == 0) ? "MG" : "EG";
            std::printf("\n// %s %s\n{\n", pn[pt], ph);
            for (int r = 7; r >= 0; r--) {
                std::printf("    ");
                for (int f = 0; f < 8; f++)
                    std::printf("%4.0f", theta[phase*384 + pt*64 + r*8 + f]);
                std::printf("\n");
            }
            std::printf("}\n");
        }
    }
    std::printf("\n=== TUNED MOBILITY ===\n");
    const char* mn[] = {"KnightMG","KnightEG","BishopMG","BishopEG","RookMG","RookEG","QueenMG","QueenEG"};
    int ms[] = {9,9,14,14,15,15,28,28};
    int off = 0;
    for (int t = 0; t < 8; t++) {
        std::printf("%s: {", mn[t]);
        for (int i = 0; i < ms[t]; i++) std::printf("%s%.0f", i ? "," : "", theta[MOB_BASE+off+i]);
        std::printf("}\n");
        off += ms[t];
    }
    std::printf("\n=== TUNED BONUSES ===\n");
    std::printf("BISHOP_PAIR_MG=%.0f EG=%.0f\n", theta[BONUS_BASE], theta[BONUS_BASE+1]);
    std::printf("ROOK_OPEN_MG=%.0f EG=%.0f\n", theta[BONUS_BASE+2], theta[BONUS_BASE+3]);
    std::printf("ROOK_SEMI_MG=%.0f EG=%.0f\n", theta[BONUS_BASE+4], theta[BONUS_BASE+5]);
    return 0;
}
