// Texel Tuner for Luminex
// Optimizes eval parameters by minimizing sigmoid(eval/K) vs game outcome
// Uses coordinate descent: try each parameter ±step, keep best MSE
//
// Build: cmake -DBUILD_TUNER=ON ...
// Usage: luminex-tuner dataset.txt [options]
// Dataset format: FEN|result (one per line, result = 1.0/0.0/0.5)

#include "luminex.h"
#include "tuner_params.h"
#include "board.h"
#include "evaluation.h"
#include "bitboard.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace luminex {

// Global stubs needed by board/eval linking
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

} // namespace luminex

using namespace luminex;

struct TunerEntry {
    std::string fen;
    double result; // 1.0, 0.0, 0.5
};

static std::vector<TunerEntry> dataset;
static double K = 1.13; // Default K factor

double sigmoid(int eval_cp, double k) {
    return 1.0 / (1.0 + std::exp(-eval_cp / (400.0 * k)));
}

double compute_mse(double k) {
    double total_error = 0.0;
    int n = (int)dataset.size();

#ifdef _OPENMP
    #pragma omp parallel for reduction(+:total_error)
#endif
    for (int i = 0; i < n; ++i) {
        Position pos;
        pos.set(dataset[i].fen);
        int eval_cp = (int)evaluate(pos, false);
        double sig = sigmoid(eval_cp, k);
        double err = sig - dataset[i].result;
        total_error += err * err;
    }

    return total_error / n;
}

double compute_mse_current() {
    return compute_mse(K);
}

double optimize_K() {
    printf("Optimizing K factor...\n");
    double best_k = 0.4;
    double best_mse = compute_mse(best_k);

    for (double k = 0.4; k <= 2.0; k += 0.05) {
        double mse = compute_mse(k);
        printf("  K=%.2f  MSE=%.6f\n", k, mse);
        if (mse < best_mse) {
            best_mse = mse;
            best_k = k;
        }
    }

    // Fine-tune around best
    for (double k = best_k - 0.05; k <= best_k + 0.05; k += 0.01) {
        if (k < 0.1) continue;
        double mse = compute_mse(k);
        if (mse < best_mse) {
            best_mse = mse;
            best_k = k;
        }
    }

    printf("Best K=%.2f  MSE=%.6f\n\n", best_k, best_mse);
    return best_k;
}

bool load_dataset(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Cannot open dataset: %s\n", filename);
        return false;
    }

    char line[1024];
    int count = 0;
    while (fgets(line, sizeof(line), f)) {
        // Strip newline
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
        if (len == 0) continue;

        // Find last '|' separator
        char* sep = strrchr(line, '|');
        if (!sep) continue;

        *sep = 0;
        double result = atof(sep + 1);
        if (result != 0.0 && result != 0.5 && result != 1.0) continue;

        dataset.push_back({std::string(line), result});
        count++;
    }
    fclose(f);
    printf("Loaded %d positions from %s\n", count, filename);
    return count > 0;
}

void print_params() {
    printf("\n=== Tuned Parameters ===\n");
    printf("// Material values\n");
    printf("static constexpr int PieceValueMG[8] = {%d, %d, %d, %d, %d, 0, 0, 0};\n",
           g_params[PAWN_VALUE_MG], g_params[KNIGHT_VALUE_MG], g_params[BISHOP_VALUE_MG],
           g_params[ROOK_VALUE_MG], g_params[QUEEN_VALUE_MG]);
    printf("static constexpr int PieceValueEG[8] = {%d, %d, %d, %d, %d, 0, 0, 0};\n",
           g_params[PAWN_VALUE_EG], g_params[KNIGHT_VALUE_EG], g_params[BISHOP_VALUE_EG],
           g_params[ROOK_VALUE_EG], g_params[QUEEN_VALUE_EG]);

    printf("\n// Knight mobility\n");
    printf("static constexpr int KnightMobBaseMG = %d;\n", g_params[KNIGHT_MOB_BASE_MG]);
    printf("static constexpr int KnightMobSlopeMG = %d;\n", g_params[KNIGHT_MOB_SLOPE_MG]);
    printf("static constexpr int KnightMobBaseEG = %d;\n", g_params[KNIGHT_MOB_BASE_EG]);
    printf("static constexpr int KnightMobSlopeEG = %d;\n", g_params[KNIGHT_MOB_SLOPE_EG]);

    printf("\n// Bishop mobility\n");
    printf("static constexpr int BishopMobBaseMG = %d;\n", g_params[BISHOP_MOB_BASE_MG]);
    printf("static constexpr int BishopMobSlopeMG = %d;\n", g_params[BISHOP_MOB_SLOPE_MG]);
    printf("static constexpr int BishopMobBaseEG = %d;\n", g_params[BISHOP_MOB_BASE_EG]);
    printf("static constexpr int BishopMobSlopeEG = %d;\n", g_params[BISHOP_MOB_SLOPE_EG]);

    printf("\n// Rook mobility\n");
    printf("static constexpr int RookMobBaseMG = %d;\n", g_params[ROOK_MOB_BASE_MG]);
    printf("static constexpr int RookMobSlopeMG = %d;\n", g_params[ROOK_MOB_SLOPE_MG]);
    printf("static constexpr int RookMobBaseEG = %d;\n", g_params[ROOK_MOB_BASE_EG]);
    printf("static constexpr int RookMobSlopeEG = %d;\n", g_params[ROOK_MOB_SLOPE_EG]);

    printf("\n// Queen mobility\n");
    printf("static constexpr int QueenMobBaseMG = %d;\n", g_params[QUEEN_MOB_BASE_MG]);
    printf("static constexpr int QueenMobSlopeMG = %d;\n", g_params[QUEEN_MOB_SLOPE_MG]);
    printf("static constexpr int QueenMobBaseEG = %d;\n", g_params[QUEEN_MOB_BASE_EG]);
    printf("static constexpr int QueenMobSlopeEG = %d;\n", g_params[QUEEN_MOB_SLOPE_EG]);

    printf("\n// Key EvalParams\n");
    printf("bishop_pair_mg = %d;\n", g_params[BISHOP_PAIR_MG]);
    printf("bishop_pair_eg = %d;\n", g_params[BISHOP_PAIR_EG]);
    printf("rook_open_mg = %d;\n", g_params[ROOK_OPEN_FILE_MG]);
    printf("rook_open_eg = %d;\n", g_params[ROOK_OPEN_FILE_EG]);
    printf("rook_semi_open_mg = %d;\n", g_params[ROOK_SEMI_OPEN_MG]);
    printf("rook_semi_open_eg = %d;\n", g_params[ROOK_SEMI_OPEN_EG]);
}

int main_tuner(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s dataset.txt [K_value] [max_iterations]\n", argv[0]);
        return 1;
    }

    // Initialize engine
    init_magic_bitboards();
    init_evaluation();
    init_tuner_params();
    sync_eval_params();

    // Load dataset
    if (!load_dataset(argv[1])) return 1;

    // K value
    if (argc >= 3) {
        K = atof(argv[2]);
        printf("Using K=%.2f from command line\n", K);
    } else {
        K = optimize_K();
    }

    int max_iterations = 100;
    if (argc >= 4) max_iterations = atoi(argv[3]);

    printf("Starting coordinate descent: %d parameters, %d positions, K=%.2f\n",
           (int)PARAM_COUNT, (int)dataset.size(), K);
    printf("Max iterations: %d\n\n", max_iterations);

    double best_mse = compute_mse_current();
    printf("Initial MSE: %.6f\n\n", best_mse);

    int step = 2;
    int no_improve_count = 0;

    // Parameter bounds: {min, max} — prevent degenerate solutions
    struct Bounds { int lo, hi; };
    Bounds bounds[PARAM_COUNT] = {
        {80,110},   // PAWN_VALUE_MG
        {80,120},   // PAWN_VALUE_EG
        {280,360},  // KNIGHT_VALUE_MG
        {250,340},  // KNIGHT_VALUE_EG
        {300,380},  // BISHOP_VALUE_MG
        {270,350},  // BISHOP_VALUE_EG
        {440,560},  // ROOK_VALUE_MG
        {470,590},  // ROOK_VALUE_EG
        {850,1100}, // QUEEN_VALUE_MG
        {830,1050}, // QUEEN_VALUE_EG
        {-130,-30}, // KNIGHT_MOB_BASE_MG
        {5,30},     // KNIGHT_MOB_SLOPE_MG (must be positive)
        {-130,-30}, // KNIGHT_MOB_BASE_EG
        {5,35},     // KNIGHT_MOB_SLOPE_EG (must be positive)
        {-100,-20}, // BISHOP_MOB_BASE_MG
        {3,15},     // BISHOP_MOB_SLOPE_MG (must be positive)
        {-100,-20}, // BISHOP_MOB_BASE_EG
        {4,18},     // BISHOP_MOB_SLOPE_EG (must be positive)
        {-80,-10},  // ROOK_MOB_BASE_MG
        {2,12},     // ROOK_MOB_SLOPE_MG (must be positive)
        {-80,-10},  // ROOK_MOB_BASE_EG
        {3,14},     // ROOK_MOB_SLOPE_EG (must be positive)
        {-60,-10},  // QUEEN_MOB_BASE_MG
        {1,8},      // QUEEN_MOB_SLOPE_MG (must be positive)
        {-80,-10},  // QUEEN_MOB_BASE_EG
        {1,8},      // QUEEN_MOB_SLOPE_EG (must be positive)
        {0,100},    // BISHOP_PAIR_MG
        {40,200},   // BISHOP_PAIR_EG
        {5,60},     // ROOK_OPEN_FILE_MG
        {10,80},    // ROOK_OPEN_FILE_EG
        {0,40},     // ROOK_SEMI_OPEN_MG
        {0,50},     // ROOK_SEMI_OPEN_EG
    };

    for (int iter = 1; iter <= max_iterations; ++iter) {
        bool improved = false;
        auto iter_start = std::chrono::steady_clock::now();

        for (size_t p = 0; p < PARAM_COUNT; ++p) {
            int original = g_params[p];
            int lo = bounds[p].lo, hi = bounds[p].hi;

            int plus_val = std::min(original + step, hi);
            int minus_val = std::max(original - step, lo);
            if (plus_val == original && minus_val == original) continue;

            double mse_plus = 1e9, mse_minus = 1e9;

            if (plus_val != original) {
                g_params[p] = plus_val;
                sync_eval_params();
                mse_plus = compute_mse_current();
            }

            if (minus_val != original) {
                g_params[p] = minus_val;
                sync_eval_params();
                mse_minus = compute_mse_current();
            }

            // Keep best
            if (mse_plus < best_mse && mse_plus <= mse_minus) {
                g_params[p] = plus_val;
                best_mse = mse_plus;
                improved = true;
            } else if (mse_minus < best_mse) {
                g_params[p] = minus_val;
                best_mse = mse_minus;
                improved = true;
            } else {
                g_params[p] = original;
            }
            sync_eval_params();
        }

        auto iter_end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(iter_end - iter_start).count();

        printf("Iter %3d: MSE=%.6f  step=%d  time=%.1fs\n", iter, best_mse, step, elapsed);

        if (!improved) {
            no_improve_count++;
            step = std::max(1, step / 2);
            if (step == 1 && no_improve_count >= 3) {
                printf("Converged after %d iterations.\n", iter);
                break;
            }
        } else {
            no_improve_count = 0;
        }
    }

    print_params();
    printf("\nFinal MSE: %.6f (K=%.2f)\n", best_mse, K);
    return 0;
}

int main(int argc, char** argv) {
    return main_tuner(argc, argv);
}
