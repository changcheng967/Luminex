// Texel Tuner for Luminex
// Direct parameter optimization: tunes ALL 148 eval parameters
// (material values, individual mobility entries, eval bonuses)
// using coordinate descent with step-halving.
//
// Build: cmake -DBUILD_TUNER=ON ...
// Usage: luminex-tuner dataset.txt [K_value] [max_iterations]

#include "luminex.h"
#include "tuner_params.h"
#include "board.h"
#include "movegen.h"
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

// Per-parameter tuning configuration
struct ParamConfig {
    int init_step;
    int min_val;
    int max_val;
};

static ParamConfig param_cfg[PARAM_COUNT];

static void init_param_configs() {
    // Material values: wider range, larger steps
    for (int i = 0; i < 10; i++) {
        param_cfg[i] = {8, 50, 1500};
    }
    // Mobility tables: moderate range
    for (int i = 10; i < 142; i++) {
        param_cfg[i] = {4, -150, 150};
    }
    // Eval bonuses: moderate range
    for (int i = 142; i < 148; i++) {
        param_cfg[i] = {4, -100, 300};
    }
}

struct TunerEntry {
    std::string fen;
    double result;
};

static std::vector<TunerEntry> dataset;
static double K = 1.13;

double sigmoid(int eval_cp, double k) {
    return 1.0 / (1.0 + std::exp(-eval_cp / (400.0 * k)));
}

double compute_mse(double k) {
    double total_error = 0.0;
    int n = (int)dataset.size();

#ifdef _OPENMP
    #pragma omp parallel for reduction(+:total_error) schedule(dynamic, 256)
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
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
        if (len == 0) continue;

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

    // Material values
    printf("\n// Material values\n");
    printf("PAWN_VALUE_MG   = %d\n", g_params[PAWN_VALUE_MG]);
    printf("PAWN_VALUE_EG   = %d\n", g_params[PAWN_VALUE_EG]);
    printf("KNIGHT_VALUE_MG = %d\n", g_params[KNIGHT_VALUE_MG]);
    printf("KNIGHT_VALUE_EG = %d\n", g_params[KNIGHT_VALUE_EG]);
    printf("BISHOP_VALUE_MG = %d\n", g_params[BISHOP_VALUE_MG]);
    printf("BISHOP_VALUE_EG = %d\n", g_params[BISHOP_VALUE_EG]);
    printf("ROOK_VALUE_MG   = %d\n", g_params[ROOK_VALUE_MG]);
    printf("ROOK_VALUE_EG   = %d\n", g_params[ROOK_VALUE_EG]);
    printf("QUEEN_VALUE_MG  = %d\n", g_params[QUEEN_VALUE_MG]);
    printf("QUEEN_VALUE_EG  = %d\n", g_params[QUEEN_VALUE_EG]);

    // Knight mobility
    printf("\nstatic int KnightMobilityMG[9] = {%d", g_params[KNIGHT_MOB_MG_0]);
    for (int i = 1; i < 9; i++) printf(", %d", g_params[KNIGHT_MOB_MG_0 + i]);
    printf("};\n");
    printf("static int KnightMobilityEG[9] = {%d", g_params[KNIGHT_MOB_EG_0]);
    for (int i = 1; i < 9; i++) printf(", %d", g_params[KNIGHT_MOB_EG_0 + i]);
    printf("};\n");

    // Bishop mobility
    printf("\nstatic int BishopMobilityMG[14] = {%d", g_params[BISHOP_MOB_MG_0]);
    for (int i = 1; i < 14; i++) printf(", %d", g_params[BISHOP_MOB_MG_0 + i]);
    printf("};\n");
    printf("static int BishopMobilityEG[14] = {%d", g_params[BISHOP_MOB_EG_0]);
    for (int i = 1; i < 14; i++) printf(", %d", g_params[BISHOP_MOB_EG_0 + i]);
    printf("};\n");

    // Rook mobility
    printf("\nstatic int RookMobilityMG[15] = {%d", g_params[ROOK_MOB_MG_0]);
    for (int i = 1; i < 15; i++) printf(", %d", g_params[ROOK_MOB_MG_0 + i]);
    printf("};\n");
    printf("static int RookMobilityEG[15] = {%d", g_params[ROOK_MOB_EG_0]);
    for (int i = 1; i < 15; i++) printf(", %d", g_params[ROOK_MOB_EG_0 + i]);
    printf("};\n");

    // Queen mobility
    printf("\nstatic int QueenMobilityMG[28] = {\n    %d", g_params[QUEEN_MOB_MG_0]);
    for (int i = 1; i < 28; i++) {
        if (i % 14 == 0) printf(",\n    %d", g_params[QUEEN_MOB_MG_0 + i]);
        else printf(", %d", g_params[QUEEN_MOB_MG_0 + i]);
    }
    printf("};\n");
    printf("static int QueenMobilityEG[28] = {\n    %d", g_params[QUEEN_MOB_EG_0]);
    for (int i = 1; i < 28; i++) {
        if (i % 14 == 0) printf(",\n    %d", g_params[QUEEN_MOB_EG_0 + i]);
        else printf(", %d", g_params[QUEEN_MOB_EG_0 + i]);
    }
    printf("};\n");

    // Eval bonuses
    printf("\n// Eval bonuses\n");
    printf("BISHOP_PAIR_MG    = %d\n", g_params[BISHOP_PAIR_MG]);
    printf("BISHOP_PAIR_EG    = %d\n", g_params[BISHOP_PAIR_EG]);
    printf("ROOK_OPEN_FILE_MG = %d\n", g_params[ROOK_OPEN_FILE_MG]);
    printf("ROOK_OPEN_FILE_EG = %d\n", g_params[ROOK_OPEN_FILE_EG]);
    printf("ROOK_SEMI_OPEN_MG = %d\n", g_params[ROOK_SEMI_OPEN_MG]);
    printf("ROOK_SEMI_OPEN_EG = %d\n", g_params[ROOK_SEMI_OPEN_EG]);
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
    init_param_configs();

    // Unbuffer stdout for redirected output
    setvbuf(stdout, nullptr, _IONBF, 0);

    // Load dataset
    if (!load_dataset(argv[1])) return 1;

    // K value
    if (argc >= 3) {
        K = atof(argv[2]);
        printf("Using K=%.2f from command line\n", K);
    } else {
        K = optimize_K();
    }

    int max_iterations = 500;
    if (argc >= 4) max_iterations = atoi(argv[3]);

    printf("Direct parameter tuner: %d parameters, %d positions, K=%.2f\n",
           PARAM_COUNT, (int)dataset.size(), K);
    printf("Max iterations: %d\n\n", max_iterations);

    // Per-parameter step sizes (independent halving)
    int steps[PARAM_COUNT];
    for (int i = 0; i < PARAM_COUNT; i++) {
        steps[i] = param_cfg[i].init_step;
    }

    double best_mse = compute_mse(K);
    printf("Initial MSE: %.6f\n\n", best_mse);

    int no_improve_global = 0;

    for (int iter = 1; iter <= max_iterations; ++iter) {
        bool any_improved = false;
        auto iter_start = std::chrono::steady_clock::now();

        // Coordinate descent: try +step and -step for each parameter
        for (int p = 0; p < PARAM_COUNT; p++) {
            if (steps[p] == 0) continue;

            int original = g_params[p];
            int lo = param_cfg[p].min_val;
            int hi = param_cfg[p].max_val;

            int plus_val  = std::min(original + steps[p], hi);
            int minus_val = std::max(original - steps[p], lo);

            double mse_plus = 1e18, mse_minus = 1e18;

            // Try +
            if (plus_val != original) {
                g_params[p] = plus_val;
                sync_eval_params();
                mse_plus = compute_mse(K);
            }
            // Try -
            if (minus_val != original) {
                g_params[p] = minus_val;
                sync_eval_params();
                mse_minus = compute_mse(K);
            }

            // Accept best
            if (mse_plus < best_mse && mse_plus <= mse_minus) {
                g_params[p] = plus_val;
                sync_eval_params();
                best_mse = mse_plus;
                any_improved = true;
            } else if (mse_minus < best_mse) {
                g_params[p] = minus_val;
                sync_eval_params();
                best_mse = mse_minus;
                any_improved = true;
            } else {
                g_params[p] = original;
                sync_eval_params();
                // Halve this parameter's step since it didn't help
                steps[p] = std::max(1, steps[p] / 2);
            }
        }

        auto iter_end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(iter_end - iter_start).count();

        int active_params = 0;
        for (int i = 0; i < PARAM_COUNT; i++) if (steps[i] > 1) active_params++;

        printf("Iter %3d: MSE=%.6f  active=%d  time=%.1fs\n",
               iter, best_mse, active_params, elapsed);

        if (!any_improved) {
            no_improve_global++;
            if (no_improve_global >= 5) {
                printf("Converged after %d iterations.\n", iter);
                break;
            }
        } else {
            no_improve_global = 0;
        }
    }

    print_params();
    printf("\nFinal MSE: %.6f (K=%.2f)\n", best_mse, K);
    return 0;
}

int main(int argc, char** argv) {
    return main_tuner(argc, argv);
}
