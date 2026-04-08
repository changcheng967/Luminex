/*
 * texel_tune.cpp - Texel tuning utility for Luminex eval parameters
 *
 * Optimizes the 24 eval parameters in g_eval_params to minimize
 * sigmoid error against a quiet-labeled EPD dataset.
 *
 * Usage: ./texel_tune quiet-labeled.epd
 *
 * EPD format per line:
 *   rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 c9 "1/2-1/2";
 *   (FEN fields + [result] where result is 0, 1/2, or 1)
 *
 * Algorithm: Gradient descent with momentum on the sigmoid error function.
 * E = sum_i (result_i - sigmoid(eval_i))^2
 */

#include "board.h"
#include "evaluation.h"
#include "luminex.h"
#include "bitboard.h"
#include "transposition.h"
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>

using namespace luminex;

struct TexelPosition {
    char fen[256];
    double result; // 0.0, 0.5, or 1.0
};

static EvalParams tune_params;

// Parameter metadata for tuning
struct ParamInfo {
    const char* name;
    int* ptr;        // Pointer to parameter in EvalParams
    int default_val;
    int min_val;
    int max_val;
};

static const ParamInfo PARAM_INFO[] = {
    {"BishopPairMG", &tune_params.bishop_pair_mg, 30, -100, 200},
    {"BishopPairEG", &tune_params.bishop_pair_eg, 96, -100, 300},
    {"RookOpenMG", &tune_params.rook_open_mg, 29, -50, 150},
    {"RookOpenEG", &tune_params.rook_open_eg, 47, -50, 150},
    {"RookSemiOpenMG", &tune_params.rook_semi_open_mg, 20, -50, 100},
    {"RookSemiOpenEG", &tune_params.rook_semi_open_eg, 15, -50, 100},
    {"Rook7thMG", &tune_params.rook_7th_mg, 30, -50, 100},
    {"Rook7thEG", &tune_params.rook_7th_eg, 17, -50, 150},
    {"PawnShieldCenter", &tune_params.pawn_shield_center, 11, -20, 40},
    {"PawnShieldKnight", &tune_params.pawn_shield_knight, 15, -20, 40},
    {"PawnShieldRook", &tune_params.pawn_shield_rook, 8, -20, 30},
    {"PawnStorm", &tune_params.pawn_storm, 9, 0, 30},
    {"OpenFilePenaltyMG", &tune_params.open_file_penalty_mg, 21, -50, 100},
    {"OpenFilePenaltyEG", &tune_params.open_file_penalty_eg, 18, -50, 100},
    {"OutpostKnightMG", &tune_params.outpost_knight_mg, 27, -20, 60},
    {"OutpostKnightEG", &tune_params.outpost_knight_eg, 17, -20, 40},
    {"OutpostBishopMG", &tune_params.outpost_bishop_mg, 50, -20, 100},
    {"OutpostBishopEG", &tune_params.outpost_bishop_eg, 31, -20, 60},
    {"HangingPawnMG", &tune_params.hanging_pawn_mg, 8, -20, 40},
    {"HangingPawnEG", &tune_params.hanging_pawn_eg, 39, -20, 80},
    {"FarKnightMG", &tune_params.far_knight_mg, 29, -20, 60},
    {"FarKnightEG", &tune_params.far_knight_eg, 8, -20, 40},
    {"FarBishopMG", &tune_params.far_bishop_mg, 5, -20, 30},
    {"FarBishopEG", &tune_params.far_bishop_eg, 3, -20, 30},
};

static constexpr int N_PARAMS = sizeof(PARAM_INFO) / sizeof(PARAM_INFO[0]);

double sigmoid(double eval_cp, double k) {
    return 1.0 / (1.0 + std::exp(-k * eval_cp / 400.0));
}

double compute_error(const std::vector<TexelPosition>& positions, double k) {
    g_eval_params = tune_params;  // Sync current tuning params
    double total = 0.0;
    for (const auto& pos : positions) {
        Position board;
        board.set(pos.fen);
        Value eval = evaluate(board);
        double pred = sigmoid(static_cast<double>(eval), k);
        double err = pos.result - pred;
        total += err * err;
    }
    return total / static_cast<double>(positions.size());
}

// Find optimal K value using a subset for speed
double find_optimal_k(const std::vector<TexelPosition>& positions) {
    double best_k = 1.0;
    double best_error = 1e9;

    // Use every 10th position for K search (much faster)
    std::vector<TexelPosition> subset;
    subset.reserve(positions.size() / 10 + 1);
    for (size_t i = 0; i < positions.size(); i += 10) {
        subset.push_back(positions[i]);
    }

    // Search K from 0.5 to 3.0 with fine steps
    for (double k = 0.5; k <= 3.0; k += 0.05) {
        double err = 0.0;
        for (const auto& pos : subset) {
            Position board;
            board.set(pos.fen);
            Value eval = evaluate(board);
            double pred = sigmoid(static_cast<double>(eval), k);
            double e = pos.result - pred;
            err += e * e;
        }
        err /= static_cast<double>(subset.size());
        if (err < best_error) {
            best_error = err;
            best_k = k;
        }
    }
    return best_k;
}

int main(int argc, char* argv[]) {
    // Force unbuffered output for nohup
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    if (argc < 2) {
        printf("Usage: %s <quiet-labeled.epd> [iterations]\n", argv[0]);
        return 1;
    }

    int n_iter = (argc > 2) ? std::atoi(argv[2]) : 50;

    // Initialize all engine subsystems
    init_magic_bitboards();
    init_line_tables();
    init_zobrist();
    init_evaluation();
    TT.resize(256);
    TT.clear();

    // Load positions
    std::vector<TexelPosition> positions;
    positions.reserve(800000);
    std::ifstream file(argv[1]);
    std::string line;
    int loaded = 0;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#' || line[0] == '[') continue;

        // Parse: find the result in quotes at the end
        // Format: ... c9 "1/2-1/2";
        auto quote_start = line.find('"');
        auto quote_end = line.rfind('"');
        if (quote_start == std::string::npos || quote_end == quote_start) continue;

        std::string result_str = line.substr(quote_start + 1, quote_end - quote_start - 1);
        double result = -1.0;

        if (result_str == "1-0" || result_str == "1") result = 1.0;
        else if (result_str == "0-1" || result_str == "0") result = 0.0;
        else if (result_str == "1/2-1/2" || result_str == "1/2") result = 0.5;

        if (result < 0) continue;

        // Extract FEN: everything before the result annotation
        std::string fen_part = line.substr(0, quote_start);
        // Trim trailing whitespace
        while (!fen_part.empty() && (fen_part.back() == ' ' || fen_part.back() == ';'))
            fen_part.pop_back();

        // FEN might have 4, 5, or 6 space-separated fields
        // EPD format: piece_placement color castle enpassant [halfmove] [fullmove]
        // Accept 4+ fields (at minimum: position, color, castling, enpassant)
        int spaces = 0;
        for (char c : fen_part) if (c == ' ') spaces++;
        if (spaces < 3) continue;

        // Ensure the FEN ends with 0 1 if only 4 fields (add default halfmove/fullmove)
        if (spaces == 3) {
            fen_part += " 0 1";
        } else if (spaces == 4) {
            fen_part += " 1";
        }

        TexelPosition tp;
        std::strncpy(tp.fen, fen_part.c_str(), sizeof(tp.fen) - 1);
        tp.fen[sizeof(tp.fen) - 1] = '\0';
        tp.result = result;
        positions.push_back(tp);
        loaded++;
    }
    file.close();

    printf("Loaded %d positions\n", loaded);
    if (positions.empty()) {
        printf("No valid positions. Exiting.\n");
        return 1;
    }

    // Subsample to ~100K positions for faster tuning (every Nth position)
    int target_size = 100000;
    if ((int)positions.size() > target_size) {
        int step = (int)positions.size() / target_size;
        std::vector<TexelPosition> subsampled;
        subsampled.reserve(target_size);
        for (int i = 0; i < (int)positions.size(); i += step) {
            subsampled.push_back(positions[i]);
        }
        positions = std::move(subsampled);
        printf("Subsampled to %d positions (step=%d)\n", (int)positions.size(), step);
    }

    // Initialize params to defaults
    tune_params = g_eval_params;

    // Use fixed K=1.5 (standard for HCE engines)
    // Dynamic K optimization tends to find degenerate low-K solutions
    double k = 1.5;
    printf("Using fixed K: %.3f\n", k);

    // Compute initial error
    double error = compute_error(positions, k);
    printf("Initial error: %.6f\n\n", error);

    // Coordinate descent: for each parameter, scan its range and keep the best value
    // Much more robust than gradient descent on flat error surfaces
    int total_changed = 0;

    for (int outer = 0; outer < n_iter; outer++) {
        int round_changed = 0;

        printf("--- Round %d ---\n", outer);

        for (int i = 0; i < N_PARAMS; i++) {
            int* ptr = PARAM_INFO[i].ptr;
            int old_val = *ptr;
            int range = PARAM_INFO[i].max_val - PARAM_INFO[i].min_val;
            if (range == 0) range = 1;
            int step = std::max(1, range / 30);  // ~30 values per parameter for finer resolution

            int best_val = old_val;
            double best_err = compute_error(positions, k);

            for (int v = PARAM_INFO[i].min_val; v <= PARAM_INFO[i].max_val; v += step) {
                if (v == old_val) continue;  // Skip current value (already measured)
                *ptr = v;
                g_eval_params = tune_params;
                double err = compute_error(positions, k);
                if (err < best_err) {
                    best_err = err;
                    best_val = v;
                }
            }

            // Restore to best value
            *ptr = best_val;
            g_eval_params = tune_params;

            if (best_val != old_val) {
                round_changed++;
                total_changed++;
                printf("  %s: %d -> %d (err: %.6f -> %.6f)\n",
                       PARAM_INFO[i].name, old_val, best_val, error, best_err);
                error = best_err;
            }

            // Periodically report
            if ((i + 1) % 6 == 0 || i == N_PARAMS - 1) {
                printf("  [scan %d/%d] current_error=%.6f round_changed=%d\n",
                       i + 1, N_PARAMS, error, round_changed);
            }
        }

        g_eval_params = tune_params;
        error = compute_error(positions, k);

        printf("Round %d done: error=%.6f changed=%d total_changed=%d\n",
               outer, error, round_changed, total_changed);

        // Stop if no changes (converged)
        if (round_changed == 0) {
            printf("Converged (no changes in round %d)\n", outer);
            break;
        }
    }

    // Final output
    printf("\n=== FINAL TUNED VALUES ===\n");
    for (int i = 0; i < N_PARAMS; i++) {
        int val = *PARAM_INFO[i].ptr;
        int def = PARAM_INFO[i].default_val;
        if (val != def) {
            printf("setoption name %s value %d  # was %d, delta %d\n",
                   PARAM_INFO[i].name, val, def, val - def);
        }
    }

    printf("\nFinal error: %.6f (was %.6f)\n", error,
           compute_error(positions, k) - error + error);

    return 0;
}
