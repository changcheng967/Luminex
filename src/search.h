#pragma once

#include "board.h"
#include "types.h"
#include <atomic>
#include <cstdint>

namespace luminex {

// Search stack
class Stack {
public:
    Move pv[64];  // Reduced from MAX_PLY + 7 to save stack space
    int ply = 0;
    Move current_move = MOVE_NONE;
    Move excluded_move = MOVE_NONE;
    Piece moved_piece = NO_PIECE;  // Piece that moved at this ply
    Value static_eval = VALUE_ZERO;
    int move_count = 0;
    int reduction = 0;  // LMR reduction applied at this ply (for hindsight adjustment)
    bool improving = false;
    Stack* previous = nullptr;
};

// Search limits
struct Limits {
    int time[2] = {};
    int inc[2] = {};
    int npmsec = 0;
    int movetime = 0;
    int depth = 0;  // Default 0 to allow time management
    int nodes = 0;
    int movestogo = 0;  // UCI movestogo parameter for time management
    uint64_t mate = 0;
    bool infinite = false;
    bool ponder = false;

    bool use_time_management() const { return !movetime && !depth && !nodes && !mate && !infinite; }
};

// Search parameters
struct SearchParams {
    int contempt = 0;
    int multi_pv = 1;
};

// Search statistics for debugging
struct SearchStats {
    // Node counts
    int64_t main_nodes = 0;
    int64_t qs_nodes = 0;

    // TT stats
    int64_t tt_probes = 0;
    int64_t tt_hits = 0;
    int64_t tt_cutoffs = 0;
    int64_t tt_hit_but_shallow = 0;  // TT hit but depth < current, no cutoff

    // Eval cache stats
    int64_t eval_cache_hits_main = 0;
    int64_t eval_cache_misses_main = 0;
    int64_t eval_cache_hits_qs = 0;
    int64_t eval_cache_misses_qs = 0;

    // Qsearch depth stats
    int64_t qs_stand_pat_cutoffs = 0;   // eval >= beta, no captures searched
    int64_t qs_delta_prunes = 0;        // delta pruning cutoffs
    int64_t qs_cap_searched = 0;        // nodes where at least 1 capture was searched
    int64_t qs_cap_cutoffs = 0;         // nodes where a capture caused cutoff
    int64_t qs_no_moves = 0;            // nodes with no legal captures (all pruned or none)

    // Pruning stats
    int64_t null_move_prunes = 0;
    int64_t futility_prunes = 0;
    int64_t rev_futility_prunes = 0;
    int64_t razoring_prunes = 0;
    int64_t probcut_prunes = 0;
    int64_t delta_prunes_qs = 0;
    int64_t lmp_prunes = 0;
    int64_t history_prunes = 0;

    // LMR stats
    int64_t lmr_total = 0;
    int64_t lmr_researches = 0;
    int64_t lmr_improved = 0;

    // Extension stats
    int64_t singular_extensions = 0;
    int64_t check_extensions = 0;
    int64_t recapture_extensions = 0;

    // Move ordering stats
    int64_t captures_generated = 0;
    int64_t captures_searched = 0;
    int64_t quiets_generated = 0;
    int64_t quiets_searched = 0;
    int64_t first_move_cutoffs = 0;    // Cutoff on first non-TT move searched
    int64_t iir_reductions = 0;

    int max_depth_reached = 0;

    // PMG phase statistics
    int64_t pmg_tt_cutoffs = 0;      // TT move caused cutoff
    int64_t pmg_capture_cutoffs = 0; // Capture caused cutoff (quiets never generated)
    int64_t pmg_quiet_generated = 0; // Had to generate quiet moves
    int64_t pmg_quiet_cutoffs = 0;   // Quiet move caused cutoff
};

extern SearchStats g_stats;

class Position;
class TranspositionTable;

// Root search
Move search(Position& pos, Limits& limits);

// Correction history - clear on new game
void clear_correction_history();

// UCI info output
void uci_info(const Position& pos, int depth, Value score, uint64_t nodes, int time);

// Search globals
extern Limits limits;
extern SearchParams params;
extern std::atomic<uint64_t> nodes;
extern std::atomic<bool> stop;
extern int root_depth;
extern Value root_score;
extern int num_threads; // Default 1, SMP only helps at longer TC
extern Move previous_root_best; // Previous iteration's best move for ordering
extern std::atomic<bool> ponder_mode;
extern std::atomic<bool> ponderhit_received;

} // namespace luminex
