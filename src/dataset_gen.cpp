// Quick dataset generator: plays self-play games at fast TC
// and extracts quiet positions (not in check, after move 10)
// Usage: luminex-dataset-gen [num_games] [output_file]
// Output format: FEN|result (one per line)

#include "luminex.h"
#include "board.h"
#include "movegen.h"
#include "evaluation.h"
#include "bitboard.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

namespace luminex {
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

static std::mt19937 rng(42);

// Play a game and collect quiet positions
// Returns result: 1.0 = white wins, 0.0 = black wins, 0.5 = draw
double play_game(Position& pos, std::vector<std::string>& positions_out) {
    positions_out.clear();
    std::vector<std::string> game_positions;

    for (int move_num = 0; move_num < 200; ++move_num) {
        // Checkmate/stalemate
        ExtMove moves[MAX_MOVES];
        ExtMove* end = generate<GEN_LEGAL>(pos, moves);
        if (end == moves) {
            return pos.is_check() ? (pos.side_to_move() == WHITE ? 0.0 : 1.0) : 0.5;
        }

        // Simple eval-based move selection for variety
        int best_idx = 0;
        int best_score = -999999;
        for (ExtMove* it = moves; it != end; ++it) {
            if (!pos.do_move(it->move)) continue;
            int score = -((int)evaluate(pos, false));
            pos.undo_move(it->move);
            if (score > best_score) { best_score = score; best_idx = (int)(it - moves); }
        }

        // With 20% chance, pick a random move instead for variety
        int chosen = best_idx;
        if (std::uniform_int_distribution<int>(0, 4)(rng) == 0) {
            chosen = std::uniform_int_distribution<int>(0, (int)(end - moves) - 1)(rng);
        }

        // Record position if: after move 10 (10 plies), not in check, enough material
        if (move_num >= 10 && !pos.is_check()) {
            Bitboard all = pos.pieces();
            int pieces = popcount(all) - 2;
            if (pieces >= 4) {
                game_positions.push_back(pos.fen());
            }
        }

        // Make the chosen move
        if (!pos.do_move(moves[chosen].move)) break;
    }

    // Game didn't end - evaluate and assign result
    int final_eval = (int)evaluate(pos, false);
    double result = 0.5;
    if (final_eval > 300) result = 1.0;
    else if (final_eval < -300) result = 0.0;

    // Save 1 in every 3 positions
    for (size_t i = 0; i < game_positions.size(); i += 3) {
        positions_out.push_back(game_positions[i]);
    }
    return result;
}

int main(int argc, char** argv) {
    int num_games = argc >= 2 ? atoi(argv[1]) : 1000;
    const char* output = argc >= 3 ? argv[2] : "dataset.txt";

    init_magic_bitboards();
    init_evaluation();

    FILE* f = fopen(output, "w");
    if (!f) {
        fprintf(stderr, "Cannot open %s for writing\n", output);
        return 1;
    }

    printf("Generating dataset: %d games -> %s\n", num_games, output);

    int total_positions = 0;
    int white_wins = 0, black_wins = 0, draws = 0;

    for (int g = 0; g < num_games; ++g) {
        Position pos;
        pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        std::vector<std::string> positions;
        double result = play_game(pos, positions);

        if (result == 1.0) white_wins++;
        else if (result == 0.0) black_wins++;
        else draws++;

        for (const auto& fen : positions) {
            // Flip result for black-to-move positions
            double pos_result = result;
            // FEN tells us whose turn it is
            size_t turn_pos = fen.find(" b ");
            if (turn_pos != std::string::npos) {
                pos_result = 1.0 - pos_result;
            }
            fprintf(f, "%s|%.1f\n", fen.c_str(), pos_result);
            total_positions++;
        }

        if ((g + 1) % 100 == 0) {
            printf("Game %d/%d: %d positions (W:%d B:%d D:%d)\n",
                   g + 1, num_games, total_positions, white_wins, black_wins, draws);
        }
    }

    fclose(f);
    printf("Done! %d positions written to %s\n", total_positions, output);
    printf("Results: W:%d B:%d D:%d\n", white_wins, black_wins, draws);
    return 0;
}
