// tuner_v3.cpp — PST Adam tuner with ANALYTIC gradients.
#define _CRT_SECURE_NO_WARNINGS
// Tunes 768 PST params (6 pieces × 64 squares × MG/EG) via Adam, using the fact
// that eval is LINEAR in PST values → ∂eval/∂PST[pt][sq] = piece-on-square indicator.
// One gradient pass = one eval pass (O(N) regardless of param count) → ~1000× faster
// than coordinate descent. NO regularization (overfit avoided via data scale, per the
// proven Texel/Ethereal/Arasan approach). Material/mobility/bonuses stay fixed.
#include "luminex.h"
#include "board.h"
#include "movegen.h"
#include "evaluation.h"
#include "bitboard.h"
#include "tuner_params.h"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace luminex {
// Stubs needed by board/eval linking
std::atomic<uint64_t> nodes{0};
std::atomic<bool> stop{false};
int root_depth = 0;
Value root_score = VALUE_ZERO;
int num_threads = 1;
Move previous_root_best = MOVE_NONE;
std::atomic<bool> ponder_mode{false};
std::atomic<bool> ponderhit_received{false};
Limits limits;
SearchParams params;
SearchStats g_stats;
}

using namespace luminex;

// ---- Dataset ----
struct Entry { std::string fen; double result; };
static std::vector<Entry> dataset;

// ---- PST params: 768 = 6 piece_types × 64 squares × 2 phases ----
// Index: phase*384 + pt*64 + sq   (phase: 0=MG, 1=EG; pt: 0-5; sq: 0-63 WHITE perspective)
static double pst[768];  // float params (accumulate fractional Adam updates; round on apply)
static double adam_m[768], adam_v[768];

// ---- Hyperparams ----
static double K = 1.0;
static double LR = 0.5;       // learning rate (PST values are integers; round after update)
static double ADAM_B1 = 0.9, ADAM_B2 = 0.999, EPS = 1e-8;

// ---- PST management ----
static void init_pst() {
    for (int pt = 0; pt < 6; pt++)
        for (int sq = 0; sq < 64; sq++) {
            pst[pt*64+sq]       = (double)PST_MG_TABLE[WHITE][pt][sq];
            pst[384+pt*64+sq]   = (double)PST_EG_TABLE[WHITE][pt][sq];
        }
}

static void apply_pst() {
    for (int pt = 0; pt < 6; pt++)
        for (int sq = 0; sq < 64; sq++) {
            PST_MG_TABLE[WHITE][pt][sq] = (Score)std::round(pst[pt*64+sq]);
            PST_EG_TABLE[WHITE][pt][sq] = (Score)std::round(pst[384+pt*64+sq]);
        }
    init_evaluation();
}

// ---- Loss + analytic gradient in ONE pass ----
static double compute_loss_and_gradient(double* grad) {
    double total_loss = 0.0;
    int n = (int)dataset.size();

    #pragma omp parallel
    {
        double local_grad[768];
        std::memset(local_grad, 0, sizeof(local_grad));
        double local_loss = 0.0;

        #pragma omp for schedule(dynamic, 512)
        for (int i = 0; i < n; i++) {
            Position pos;
            pos.set(dataset[i].fen);

            int eval_cp = (int)evaluate(pos, false);
            double sig = 1.0 / (1.0 + std::exp(-eval_cp / (400.0 * K)));
            double R = dataset[i].result;
            double err = sig - R;
            local_loss += err * err;

            // d(loss_i)/d(eval_cp) = 2*err * sig*(1-sig) / (400*K)
            double signal = 2.0 * err * sig * (1.0 - sig) / (400.0 * K);

            // Phase (same formula as evaluate())
            int phase = popcount(pos.pieces(KNIGHT)) + popcount(pos.pieces(BISHOP))
                      + popcount(pos.pieces(ROOK)) * 2 + popcount(pos.pieces(QUEEN)) * 4;
            if (phase > 24) phase = 24;
            double mg_w = (double)phase / 24.0;
            double eg_w = 1.0 - mg_w;

            // Scatter gradient to active PST params (iterate board)
            // CRITICAL: eval returns side_to_move-relative (negated for BLACK-to-move).
            // The gradient must account for this: ∂eval/∂PST includes stm_sign.
            int stm_sign = (pos.side_to_move() == WHITE) ? 1 : -1;
            for (int sq = 0; sq < 64; sq++) {
                Piece p = pos.piece_on(Square(sq));
                if (p == NO_PIECE) continue;
                Color c = (p < 6) ? WHITE : BLACK;  // WHITE_PAWN=0..WHITE_KING=5, BLACK=6..11
                PieceType pt = PieceType(p % 6);
                if (pt >= 6) continue;

                int rel_sq = (c == WHITE) ? sq : (sq ^ 56);
                int sgn = (c == WHITE) ? 1 : -1;

                local_grad[pt * 64 + rel_sq]       += sgn * stm_sign * mg_w * signal;  // MG
                local_grad[384 + pt * 64 + rel_sq] += sgn * stm_sign * eg_w * signal;  // EG
            }
        }

        #pragma omp critical
        {
            total_loss += local_loss;
            for (int i = 0; i < 768; i++) grad[i] += local_grad[i];
        }
    }
    double scale = 1.0 / n;
    for (int i = 0; i < 768; i++) grad[i] *= scale;
    return total_loss * scale;
}

// ---- Adam step ----
static void adam_step(const double* grad, int iter) {
    double bc1 = 1.0 - std::pow(ADAM_B1, iter);
    double bc2 = 1.0 - std::pow(ADAM_B2, iter);
    for (int i = 0; i < 768; i++) {
        adam_m[i] = ADAM_B1 * adam_m[i] + (1.0 - ADAM_B1) * grad[i];
        adam_v[i] = ADAM_B2 * adam_v[i] + (1.0 - ADAM_B2) * grad[i] * grad[i];
        double mhat = adam_m[i] / bc1;
        double vhat = adam_v[i] / bc2;
        double update = LR * mhat / (std::sqrt(vhat) + EPS);
        pst[i] -= update;  // accumulate fractional (round only on apply_pst)
    }
}

// ---- Dataset loading ----
static bool load_dataset(const char* path) {
    FILE* f = std::fopen(path, "r");
    if (!f) { std::fprintf(stderr, "Cannot open %s\n", path); return false; }
    char line[1024];
    int count = 0;
    while (std::fgets(line, sizeof(line), f)) {
        int len = (int)std::strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
        if (len == 0) continue;
        char* sep = std::strrchr(line, '|');
        if (!sep) continue;
        *sep = 0;
        double r = std::atof(sep + 1);
        if (r != 0.0 && r != 0.5 && r != 1.0) continue;
        dataset.push_back({std::string(line), r});
        count++;
    }
    std::fclose(f);
    std::printf("Loaded %d positions from %s\n", count, path);
    return count > 0;
}

// ---- Main ----
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
    clear_pawn_table();  // clear pawn hash (declared in tuner_params.h)

    if (argc >= 3) K = std::atof(argv[2]);
    int max_iters = (argc >= 4) ? std::atoi(argv[3]) : 200;
    if (argc >= 5) LR = std::atof(argv[4]);

    setvbuf(stdout, nullptr, _IONBF, 0);

    if (!load_dataset(argv[1])) return 1;

    init_pst();       // snapshot current PST into pst[]
    apply_pst();      // apply to eval (should be identity on first call)

    std::printf("PST Adam tuner: 768 params, %d positions, K=%.2f lr=%.3f iters=%d\n\n",
                (int)dataset.size(), K, LR, max_iters);

    double grad[768];
    double best_loss = compute_loss_and_gradient(grad);
    std::printf("Initial loss: %.6f\n", best_loss);

    for (int iter = 1; iter <= max_iters; iter++) {
        adam_step(grad, iter);
        apply_pst();
        double loss = compute_loss_and_gradient(grad);
        std::printf("Iter %3d: loss=%.6f\n", iter, loss);
        if (iter % 20 == 0) {
            // Print sample tuned PST values to check sanity
            std::printf("  KnightMG: ");
            for (int sq = 0; sq < 64; sq += 8)
                std::printf("%.0f ", pst[KNIGHT*64+sq+27]); // center-ish squares
            std::printf("\n");
        }
    }

    // Final output: the tuned PST tables in C++ format
    std::printf("\n=== TUNED PST (WHITE perspective) ===\n");
    for (int phase = 0; phase < 2; phase++) {
        const char* ph = (phase == 0) ? "MG" : "EG";
        for (int pt = 0; pt < 6; pt++) {
            const char* pn[] = {"PAWN","KNIGHT","BISHOP","ROOK","QUEEN","KING"};
            std::printf("\n// %s %s\n", pn[pt], ph);
            std::printf("{\n");
            for (int r = 7; r >= 0; r--) {
                std::printf("    ");
                for (int f = 0; f < 8; f++)
                    std::printf("%4.0f", pst[phase*384 + pt*64 + r*8 + f]);
                std::printf("\n");
            }
            std::printf("}\n");
        }
    }
    return 0;
}
