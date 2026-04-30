// Texel Tuner for Luminex
// Parametric model: score(mob) = a * sqrt(mob) + c
// 16 parameters (a, c per piece per phase) with guaranteed monotonicity
//
// Build: cmake -DBUILD_TUNER=ON ...
// Usage: luminex-tuner dataset.txt [K_value] [max_iterations]

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

// Table metadata: maps formula params to g_params entries
struct TableInfo {
    size_t param_start;
    int max_mob;
};

static const TableInfo tables[] = {
    {KNIGHT_MOB_MG_0, 8},
    {KNIGHT_MOB_EG_0, 8},
    {BISHOP_MOB_MG_0, 13},
    {BISHOP_MOB_EG_0, 13},
    {ROOK_MOB_MG_0, 14},
    {ROOK_MOB_EG_0, 14},
    {QUEEN_MOB_MG_0, 27},
    {QUEEN_MOB_EG_0, 27},
};
static const int NUM_TABLES = 8;

// Formula params: score(mob) = round(a * sqrt(mob) + c)
// a > 0 guarantees monotonicity; sqrt guarantees diminishing returns
static int formula_a[NUM_TABLES];
static int formula_c[NUM_TABLES];

// Initialize from linear formula fit (eval-equivalent starting point)
void init_formula_params() {
    // Fitted to match linear formula endpoints: a * sqrt(max_mob) + c ≈ linear_max
    formula_a[0] = 42; formula_c[0] = -60; // Knight MG: -60+15*i
    formula_a[1] = 51; formula_c[1] = -75; // Knight EG: -75+18*i
    formula_a[2] = 25; formula_c[2] = -48; // Bishop MG: -48+7*i
    formula_a[3] = 32; formula_c[3] = -56; // Bishop EG: -56+9*i
    formula_a[4] = 19; formula_c[4] = -38; // Rook MG: -38+5*i
    formula_a[5] = 26; formula_c[5] = -48; // Rook EG: -48+7*i
    formula_a[6] = 10; formula_c[6] = -28; // Queen MG: -28+2*i
    formula_a[7] = 16; formula_c[7] = -38; // Queen EG: -38+3*i
}

void generate_tables() {
    for (int t = 0; t < NUM_TABLES; t++) {
        for (int mob = 0; mob <= tables[t].max_mob; mob++) {
            double val = formula_a[t] * std::sqrt((double)mob) + formula_c[t];
            g_params[tables[t].param_start + mob] = (int)std::round(val);
        }
    }
    sync_eval_params();
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

void print_tables() {
    printf("\n=== Tuned Tables (from a*sqrt(mob)+c) ===\n");

    const char* names[] = {"Knight MG", "Knight EG", "Bishop MG", "Bishop EG",
                           "Rook MG", "Rook EG", "Queen MG", "Queen EG"};
    for (int t = 0; t < NUM_TABLES; t++) {
        printf("\n// %s: a=%d, c=%d\n", names[t], formula_a[t], formula_c[t]);
        int sz = tables[t].max_mob + 1;
        printf("static int %s[%d] = {%d", names[t], sz,
               g_params[tables[t].param_start]);
        for (int i = 1; i < sz; i++) {
            if (i % 14 == 0) printf(",\n    %d", g_params[tables[t].param_start + i]);
            else printf(", %d", g_params[tables[t].param_start + i]);
        }
        printf("};\n");
    }

    // Also print as C arrays matching evaluation.cpp format
    printf("\n=== evaluation.cpp format ===\n");

    printf("static int KnightMobilityMG[9] = {%d", g_params[KNIGHT_MOB_MG_0]);
    for (int i = 1; i < 9; i++) printf(", %d", g_params[KNIGHT_MOB_MG_0 + i]);
    printf("};\n");
    printf("static int KnightMobilityEG[9] = {%d", g_params[KNIGHT_MOB_EG_0]);
    for (int i = 1; i < 9; i++) printf(", %d", g_params[KNIGHT_MOB_EG_0 + i]);
    printf("};\n");

    printf("\nstatic int BishopMobilityMG[14] = {%d", g_params[BISHOP_MOB_MG_0]);
    for (int i = 1; i < 14; i++) printf(", %d", g_params[BISHOP_MOB_MG_0 + i]);
    printf("};\n");
    printf("static int BishopMobilityEG[14] = {%d", g_params[BISHOP_MOB_EG_0]);
    for (int i = 1; i < 14; i++) printf(", %d", g_params[BISHOP_MOB_EG_0 + i]);
    printf("};\n");

    printf("\nstatic int RookMobilityMG[15] = {%d", g_params[ROOK_MOB_MG_0]);
    for (int i = 1; i < 15; i++) printf(", %d", g_params[ROOK_MOB_MG_0 + i]);
    printf("};\n");
    printf("static int RookMobilityEG[15] = {%d", g_params[ROOK_MOB_EG_0]);
    for (int i = 1; i < 15; i++) printf(", %d", g_params[ROOK_MOB_EG_0 + i]);
    printf("};\n");

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
    init_formula_params();
    generate_tables();

    // Load dataset
    if (!load_dataset(argv[1])) return 1;

    // K value
    if (argc >= 3) {
        K = atof(argv[2]);
        printf("Using K=%.2f from command line\n", K);
    } else {
        K = optimize_K();
    }

    int max_iterations = 300;
    if (argc >= 4) max_iterations = atoi(argv[3]);

    printf("Parametric tuner: score(mob) = a*sqrt(mob) + c\n");
    printf("16 parameters, %d positions, K=%.2f\n", (int)dataset.size(), K);
    printf("Max iterations: %d\n\n", max_iterations);

    double best_mse = compute_mse_current();
    printf("Initial MSE: %.6f\n\n", best_mse);

    // Bounds for formula params
    // a: [1, 120] — must be positive for monotonicity
    // c: [-120, 120]
    struct { int lo, hi; } bounds_a[NUM_TABLES], bounds_c[NUM_TABLES];
    for (int t = 0; t < NUM_TABLES; t++) {
        bounds_a[t] = {1, 120};
        bounds_c[t] = {-120, 120};
    }

    int step_a = 4;
    int step_c = 4;
    int no_improve_count = 0;

    for (int iter = 1; iter <= max_iterations; ++iter) {
        bool improved = false;
        auto iter_start = std::chrono::steady_clock::now();

        // Tune all 'a' params
        for (int t = 0; t < NUM_TABLES; t++) {
            int original_a = formula_a[t];
            int lo = bounds_a[t].lo, hi = bounds_a[t].hi;

            int plus_a = std::min(original_a + step_a, hi);
            int minus_a = std::max(original_a - step_a, lo);

            double mse_plus = 1e9, mse_minus = 1e9;

            if (plus_a != original_a) {
                formula_a[t] = plus_a;
                generate_tables();
                mse_plus = compute_mse_current();
            }
            if (minus_a != original_a) {
                formula_a[t] = minus_a;
                generate_tables();
                mse_minus = compute_mse_current();
            }

            if (mse_plus < best_mse && mse_plus <= mse_minus) {
                formula_a[t] = plus_a;
                generate_tables();
                best_mse = mse_plus;
                improved = true;
            } else if (mse_minus < best_mse) {
                formula_a[t] = minus_a;
                generate_tables();
                best_mse = mse_minus;
                improved = true;
            } else {
                formula_a[t] = original_a;
                generate_tables();
            }
        }

        // Tune all 'c' params
        for (int t = 0; t < NUM_TABLES; t++) {
            int original_c = formula_c[t];
            int lo = bounds_c[t].lo, hi = bounds_c[t].hi;

            int plus_c = std::min(original_c + step_c, hi);
            int minus_c = std::max(original_c - step_c, lo);

            double mse_plus = 1e9, mse_minus = 1e9;

            if (plus_c != original_c) {
                formula_c[t] = plus_c;
                generate_tables();
                mse_plus = compute_mse_current();
            }
            if (minus_c != original_c) {
                formula_c[t] = minus_c;
                generate_tables();
                mse_minus = compute_mse_current();
            }

            if (mse_plus < best_mse && mse_plus <= mse_minus) {
                formula_c[t] = plus_c;
                generate_tables();
                best_mse = mse_plus;
                improved = true;
            } else if (mse_minus < best_mse) {
                formula_c[t] = minus_c;
                generate_tables();
                best_mse = mse_minus;
                improved = true;
            } else {
                formula_c[t] = original_c;
                generate_tables();
            }
        }

        auto iter_end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(iter_end - iter_start).count();

        printf("Iter %3d: MSE=%.6f  step=%d  time=%.1fs\n", iter, best_mse, step_a, elapsed);

        if (!improved) {
            no_improve_count++;
            step_a = std::max(1, step_a / 2);
            step_c = std::max(1, step_c / 2);
            if (step_a == 1 && no_improve_count >= 5) {
                printf("Converged after %d iterations.\n", iter);
                break;
            }
        } else {
            no_improve_count = 0;
        }
    }

    print_tables();
    printf("\nFinal MSE: %.6f (K=%.2f)\n", best_mse, K);
    return 0;
}

int main(int argc, char** argv) {
    return main_tuner(argc, argv);
}
