#define _CRT_SECURE_NO_WARNINGS
#include "luminex.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <sstream>
#include <thread>
#include <vector>

namespace luminex {

// Forward declaration - defined in uci.cpp
extern bool check_for_stop_command();
extern void uci_debug_log(const char* format, ...);

// Global volatile stop flag for immediate response
// This is set by the main thread when "stop" is received
volatile bool g_stop_requested = false;

// Search globals
Limits limits;
SearchParams params;
std::atomic<uint64_t> nodes;
std::atomic<bool> stop;
int root_depth;
Value root_score;

// Lazy SMP thread management
int num_threads = 1;
Move previous_root_best = MOVE_NONE;

namespace {

// Per-thread search state for lazy SMP
constexpr int MAX_PLY_PLUS_6 = MAX_PLY + 6;

struct SearchWorker {
    Stack stack[MAX_PLY_PLUS_6];
    Move killers[MAX_PLY][2] = {};
    int history[12][64] = {};
    int capture_history[12][64][7] = {};
    Move counter_move_table[12][64] = {};
};

// Thread-local pointer to current thread's search state
static thread_local SearchWorker* worker = nullptr;

// Shared between threads (lazy SMP tolerates races in these heuristic tables)
// Counter-move history: [prev_piece][prev_to][piece][to]
int counter_moves[12][64][12][64];

// Continuation history (2-ply): [piece2][to2][piece1][to1]
// Tracks how good a move is given OUR previous move (2 plies back)
int continuation_history[12][64][12][64];

// ============================================================
// CORRECTION HISTORY — Simplified: Pawn-only
// Index by pawn structure hash (most reliable signal)
// ============================================================
struct PawnCorrectionHistory {
    static constexpr int CORRECTION_SIZE = 16384;
    static constexpr int CORRECTION_LIMIT = 512;

    int table[2][CORRECTION_SIZE];

    void clear() {
        std::memset(table, 0, sizeof(table));
    }

    void age() {
        for (int c = 0; c < 2; ++c) {
            for (int i = 0; i < CORRECTION_SIZE; ++i) table[c][i] /= 2;
        }
    }

    static void gravity_update(int& val, int bonus) {
        val += bonus - val * std::abs(bonus) / CORRECTION_LIMIT;
    }

    void update(Color c, uint64_t pawn_key, Value eval_diff, Depth depth) {
        int bonus = eval_diff * depth * 16 / 128;
        bonus = std::max(-CORRECTION_LIMIT, std::min(CORRECTION_LIMIT, bonus));
        gravity_update(table[c][pawn_key & (CORRECTION_SIZE - 1)], bonus);
    }

    Value correct(Color c, uint64_t pawn_key, Value static_eval) const {
        int corr = table[c][pawn_key & (CORRECTION_SIZE - 1)];
        return static_eval + corr * 66 / 512;
    }
};

PawnCorrectionHistory pawn_correction_history;

// Lazy SMP helper thread management (inside anonymous namespace for internal linkage)
static std::vector<std::thread> helper_threads;
static std::atomic<bool> helpers_running{false};



struct EvalCacheEntry {
    uint64_t key;
    int32_t value;  // Changed from int16_t to match Value type
};
constexpr int EVAL_CACHE_SIZE = 524288;  // 512K entries for better hit rate
EvalCacheEntry eval_cache[EVAL_CACHE_SIZE];

// Thread-local node counter to avoid atomic overhead on every node
static thread_local uint64_t local_nodes = 0;
static thread_local uint64_t last_reported_nodes = 0;

// Compute pawn structure hash key (same as evaluate_pawns uses internally)
inline uint64_t compute_pawn_key(const Position& pos) {
    uint64_t pawn_key = 0;
    for (int c = 0; c < 2; ++c) {
        Bitboard pb = pos.pieces(Color(c), PAWN);
        while (pb) {
            Square sq = pop_lsb(pb);
            pawn_key ^= Zobrist::psq[c][int(PAWN)][int(sq)];
        }
    }
    return pawn_key;
}

inline Value eval_cached(const Position& pos) {
    uint64_t key = pos.key();
    uint32_t idx = uint32_t(key) & (EVAL_CACHE_SIZE - 1);

    if (eval_cache[idx].key == key) {
        return Value(eval_cache[idx].value);
    }

    Value eval = evaluate(pos);
    eval_cache[idx].key = key;
    eval_cache[idx].value = int32_t(eval);
    return eval;
}

[[maybe_unused]] inline void clear_eval_cache() {
    std::memset(eval_cache, 0, sizeof(eval_cache));
}

// Reduction constants
constexpr int futility_margin(int depth, bool improving) {
    // Depth-dependent futility margins (slightly tighter for more pruning)
    int base = 130 * depth + 50;

    if (improving) base -= 20;
    else base += 25;

    return base;
}

// Search start time for time management
static std::chrono::steady_clock::time_point search_start;
static int ideal_time = 0;  // Ideal time to use for this search
static int max_time = 0;    // Maximum time to use

// Check time - returns true if time limit exceeded
bool check_time() {
    // Check volatile stop flag first for fastest response
    if (g_stop_requested) {
        return true;
    }

    // Check for stop command from GUI (stdin polling)
    if (check_for_stop_command()) {
        return true;
    }

    if (stop.load(std::memory_order_relaxed)) {
        return true;
    }

    if (limits.nodes && nodes >= uint64_t(limits.nodes)) {
        stop = true;
        return true;
    }

    if (limits.movetime) {
        auto now = std::chrono::steady_clock::now();
        int elapsed = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
            now - search_start).count());
        if (elapsed >= limits.movetime) {
            stop = true;
            return true;
        }
    }

    if (limits.use_time_management()) {
        auto now = std::chrono::steady_clock::now();
        int elapsed = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
            now - search_start).count());
        // Hard limit: stop immediately when max_time is exceeded
        if (elapsed >= max_time) {
            stop = true;
            return true;
        }
    }

    // SAFETY: For depth-only searches (no time limit), add a 30-minute maximum
    if (!limits.movetime && !limits.use_time_management() && limits.nodes == 0) {
        auto now = std::chrono::steady_clock::now();
        int elapsed = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
            now - search_start).count());
        if (elapsed >= 1800000) {  // 30 minutes
            stop = true;
            return true;
        }
    }

    return false;
}

// Quiescence search
Value qsearch(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth) {
    // Check for max ply to prevent stack overflow
    if (ss->ply >= MAX_PLY) {
        return eval_cached(pos);
    }

    // Search captures to depth -4 to avoid horizon effect
    if (depth < -4) {
        return eval_cached(pos);
    }

    // Memory barrier for ensure we see the latest stop flag
    if (g_stop_requested || stop.load(std::memory_order_relaxed)) {
        return VALUE_ZERO;
    }

    ++local_nodes;

    // Check for draw
    if (pos.is_draw()) {
        // Draw score randomization: tiny noise to avoid search instabilities on draws (Stash-style)
        Value draw_noise = (local_nodes & 2) - 1;
        return VALUE_DRAW + draw_noise - (pos.side_to_move() == WHITE ? params.contempt / 2 : -params.contempt / 2);
    }

    // Evaluate position
    bool in_check = pos.is_check();
    Value eval = VALUE_ZERO;

    if (!in_check) {
        // Stand pat only when NOT in check
        eval = eval_cached(pos);
        if (eval >= beta) {
            return beta;
        }
        if (eval > alpha) {
            alpha = eval;
        }
    }

    // Delta pruning: if best possible capture can't raise alpha, skip captures
    if (!in_check) {
        Value delta = 200 + QUEEN_VALUE;  // Assume best case: can capture a queen
        if (eval + delta < alpha) {
            return alpha;
        }
    }

    // Generate moves: captures normally, but ALL evasions when in check
    // CRITICAL: When in check, we must consider king moves and blocking moves, not just captures
    ExtMove moves[MAX_MOVES];
    ExtMove* end;
    if (in_check) {
        end = generate<GEN_LEGAL>(pos, moves);  // All legal moves when in check
    } else {
        end = generate<GEN_CAPTURE>(pos, moves);  // Only captures when not in check
    }

    int moves_searched = 0;

    // Score captures in qsearch by MVV-LVA for better ordering
    if (!in_check) {
        for (ExtMove* it = moves; it != end; ++it) {
            if (it->move.is_capture()) {
                PieceType captured = pos.piece_type_on(it->move.to());
                PieceType attacker = pos.piece_type_on(it->move.from());
                static constexpr int pv[] = {100, 320, 330, 500, 900, 20000, 0};
                it->value = (captured != PT_NONE ? pv[captured] : 0) * 10 - pv[attacker];
                if (it->move.is_promotion()) it->value += pv[it->move.promotion_type()];
            } else {
                it->value = 0;
            }
        }
        // Sort captures by value
        std::sort(moves, end, [](const ExtMove& a, const ExtMove& b) {
            return a.value > b.value;
        });
    }

    for (ExtMove* it = moves; it != end; ++it) {
        // FIX: Check for stop at top of move loop for faster response
        if (stop.load(std::memory_order_relaxed)) break;

        if (in_check) {
            // For evasions, legal() check already done by GEN_LEGAL
        } else if (!pos.legal(it->move, true)) {
            continue;
        }

        // Skip losing captures with negative SEE (but keep queen promotions)
        // Exception: when in check, try all evasions regardless of SEE
        if (!in_check && !it->move.is_promotion() && !pos.see_ge(it->move, VALUE_ZERO)) {
            continue;
        }

        ss->current_move = it->move;
        ss->moved_piece = pos.piece_on(it->move.from());

        if (!pos.do_move(it->move)) {
            // CRITICAL: do_move failed - atomic failure, no state change
            // Do NOT call undo_move - do_move already guarantees state is unchanged
            continue;
        }

        // This move passed do_move - count it as searched
        moves_searched++;

        Value value = -qsearch(pos, ss + 1, -beta, -alpha, depth - 1);

        // CRITICAL: Check stop immediately after recursive call
        if (stop.load(std::memory_order_relaxed)) {
            pos.undo_move(it->move);
            return VALUE_ZERO;
        }

        pos.undo_move(it->move);

        if (value >= beta) {
            return value;  // fail-soft
        }
        if (value > alpha) {
            alpha = value;
        }
    }

    // Checkmate detection - if in check and no legal evasions
    if (in_check && moves_searched == 0) {
        return -VALUE_MATE + ss->ply;
    }

    return alpha;
}

// Main search function (internal worker)
[[maybe_unused]] static Value search_worker(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cut_node) {
    // Save original alpha for proper bound determination
    Value original_alpha = alpha;
    Move best_move_found = MOVE_NONE;

    // Check for max ply to prevent stack overflow
    if (ss->ply >= MAX_PLY) {
        return eval_cached(pos);
    }

    // Memory barrier to ensure we see the latest stop flag
    if (g_stop_requested || stop.load(std::memory_order_relaxed)) {
        return VALUE_ZERO;
    }

    // Check for draw
    if (pos.is_draw()) {
        Value draw_noise = (local_nodes & 2) - 1;
        return VALUE_DRAW + draw_noise - (pos.side_to_move() == WHITE ? params.contempt / 2 : -params.contempt / 2);
    }

    ++local_nodes;

    // Check time every 1024 nodes for better time control
    // Use local_nodes for cheap counting, flush to atomic periodically
    if ((local_nodes - last_reported_nodes) >= 1024) {
        nodes.fetch_add(local_nodes - last_reported_nodes, std::memory_order_relaxed);
        last_reported_nodes = local_nodes;
        if (check_time()) {
            return VALUE_ZERO;
        }
    }

    const bool pv_node = (beta - alpha > 1);

    // Mate distance pruning: if we already found a forced mate,
    // we can prune positions that can't produce a shorter mate
    alpha = std::max(Value(-VALUE_MATE + ss->ply), alpha);
    beta = std::min(Value(VALUE_MATE - ss->ply - 1), beta);
    if (alpha >= beta) {
        return alpha;
    }

    // Prevent stale PV data from being saved to TT on fail-low nodes
    if (ss->ply < 64) ss->pv[ss->ply] = MOVE_NONE;

    // Check stop every node for instant response
    if (g_stop_requested || stop.load(std::memory_order_relaxed)) {
        return VALUE_ZERO;
    }

    // Quiescence search at depth 0
    if (depth <= 0) {
        return qsearch(pos, ss, alpha, beta, 0);
    }

    // Transposition table lookup
    bool found;
    TTEntry* tte = TT.probe(pos.key(), found);

    // Validate TT move with FULL legal check before using it
    Move tt_move = MOVE_NONE;
    if (found) {
        Move m = tte->move();
        // Full validation: bounds check + legal
        if (m && m.from() < SQUARE_NONE && m.to() < SQUARE_NONE && pos.legal(m)) {
            tt_move = m;
        }
    }
    Value tt_value = found ? tte->value() : VALUE_ZERO;
    Depth tt_depth = found ? tte->depth() : DEPTH_ZERO;

    // TT cutoff: require exact or greater depth for safety
    // IIR naturally improves TT population without aggressive margins
    if (!pv_node && found && tt_depth >= depth &&
        (tt_value >= beta ? (tte->bound() & BOUND_LOWER) : (tte->bound() & BOUND_UPPER))) {
        return tt_value;
    }

    // Static evaluation
    Value eval = VALUE_ZERO;
    if (!pos.is_check()) {
        eval = eval_cached(pos);
        // Apply pawn correction history
        Color stm = pos.side_to_move();
        uint64_t pawn_key = compute_pawn_key(pos);
        eval = pawn_correction_history.correct(stm, pawn_key, eval);
    }
    ss->static_eval = eval;

    // Compute improving flag: position is improving if eval is better than 2 plies ago
    ss->improving = (ss->ply >= 2 && eval > (ss - 2)->static_eval);

    // Opponent worsening: our eval is better than opponent's eval from 1 ply ago
    // This means the opponent's last move didn't help them
    bool opponent_worsening = (ss->ply >= 1 && eval > -(ss - 1)->static_eval);

    // Internal Iterative Reduction (IIR): reduce depth by 1 when no TT move available
    if (tt_move == MOVE_NONE && depth >= 4) {
        depth--;
    }

    // Futility pruning - use improving for better pruning decisions
    // Depth <= 6 for balance between pruning and tactical accuracy
    // More aggressive when opponent is worsening (wider margin)
    if (!pv_node && !pos.is_check() && depth <= 6 && eval - futility_margin(depth, ss->improving || opponent_worsening) >= beta) {
        return eval;
    }

    // Reverse futility pruning (static null move): if eval is far above beta, prune immediately
    if (!pv_node && !pos.is_check() && depth <= 8 && eval - 100 * depth - ((ss->improving || opponent_worsening) ? 0 : 30) >= beta) {
        return eval;
    }

    // Razoring: at low depths, if eval is far below alpha, try qsearch to confirm
    if (!pv_node && !pos.is_check() && depth <= 3) {
        Value razor_margin = 300 + depth * depth * 60;
        if (eval + razor_margin < alpha) {
            // Try quiescence search to confirm the position is really losing
            Value qsearch_value = qsearch(pos, ss, alpha - 1, alpha, 0);
            if (qsearch_value <= alpha) {
                return qsearch_value;  // Confirmed losing, prune
            }
        }
    }

    // Null move pruning (Stash-style)
    int piece_count = popcount(pos.pieces()) - popcount(pos.pieces(PAWN)) - 2;
    bool null_move_ok = !pv_node && !pos.is_check() && depth >= 2 && piece_count >= 1 &&
                          eval >= beta && ss->ply >= 1 && !ss->excluded_move;

    if (null_move_ok) {
        pos.do_null_move();

        int R = 3 + (depth > 6 ? 1 : 0) + (depth > 12 ? 1 : 0);

        if (piece_count < 4) R -= 1;

        R = std::max(2, std::min(R, depth - 1));
        Value null_value = -search_worker(pos, ss + 1, -beta, -beta + 1, depth - R, !cut_node);
        pos.undo_null_move();

        if (null_value >= beta) {
            // Distrust mate scores from null move
            if (null_value >= VALUE_MATE_IN_MAX_PLY) null_value = beta;

            // Verification search at high depth
            if (piece_count < 4 && depth >= 6) {
                Value verify = search_worker(pos, ss, beta - 1, beta, (depth - R) * 3 / 4, !cut_node);
                if (verify >= beta) return null_value;
            } else {
                return null_value;
            }
        }
    }

    // ProbCut - if a capture is obviously good enough, verify with shallow search
    if (!pv_node && depth >= 5 && !pos.is_check() && ss->ply >= 2) {
        Value rbeta = std::min(beta + 175, VALUE_INFINITE - 200);
        int rdepth = depth - 3;

        // Try winning captures (SEE > 0)
        ExtMove probcut_moves[MAX_MOVES];
        ExtMove* probcut_end = generate<GEN_CAPTURE>(pos, probcut_moves);

        for (ExtMove* it = probcut_moves; it != probcut_end; ++it) {
            // FIX: Check for stop at top of move loop for faster response
            if (stop.load(std::memory_order_relaxed)) break;

            Move m = it->move;

            // Skip losing captures
            if (!pos.see_ge(m, Value(1))) continue;

            // CRITICAL FIX: Check move is legal before do_move
            // This prevents analyzing positions where our king is in check
            if (!pos.legal(m)) continue;

            if (!pos.do_move(m)) continue;

            // Shallow search with reduced beta
            Value value = -search_worker(pos, ss + 1, -rbeta, -rbeta + 1, rdepth, !cut_node);

            pos.undo_move(m);

            if (value >= rbeta) {
                // Shallow search confirms this move is very good - prune
                return value;
            }
        }
    }

    // ================================================================
    // INNOVATION: "Phased Move Generation" (PMG)
    // Traditional engines generate ALL moves upfront, score them all,
    // then search. This wastes huge amounts of work: most positions
    // cause cutoff on the TT move or a capture, making quiet move
    // generation + scoring completely wasted.
    //
    // Our approach: search in strict priority phases, only generating
    // the next phase if no cutoff occurred:
    //   Phase 1: TT move (free - already known)
    //   Phase 2: Captures + promotions (GEN_CAPTURE)
    //   Phase 3: Quiet moves (GEN_QUIET) - only if needed
    //
    // This is a novel design because most engines that split generation
    // still pre-generate everything. We truly defer quiet generation.
    // In positions where a capture causes cutoff (very common), we
    // never even touch the quiet move generator.
    // ================================================================

    Value best_value = -VALUE_INFINITE;
    int moves_played = 0;

    // Track quiet moves for history gravity (penalizing non-cutoff moves)
    Move quiets_searched[64];
    int quiet_count = 0;

    // Internal Iterative Deepening: if no TT move at PV nodes, search at reduced depth
    // to populate TT with a good move for ordering
    if (pv_node && tt_move == MOVE_NONE && depth >= 4) {
        search_worker(pos, ss, alpha, beta, depth - 2, false);
        TTEntry* iid_tte = TT.probe(pos.key(), found);
        if (found) {
            Move iid_move = iid_tte->move();
            if (iid_move && iid_move.from() < SQUARE_NONE && iid_move.to() < SQUARE_NONE && pos.legal(iid_move)) {
                tt_move = iid_move;
            }
        }
    }

    // Singular extension: check if TT move is significantly better than alternatives
    int singular_extension = 0;
    if (tt_move != MOVE_NONE && !pv_node && found && tt_depth >= depth - 3 && depth >= 8 &&
        (tte->bound() & BOUND_LOWER) && abs(tt_value) < VALUE_KNOWN_WIN) {
        Value sBeta = Value(tt_value - depth * 2);
        ss->excluded_move = tt_move;
        Value singular_value = search_worker(pos, ss, sBeta - 1, sBeta, (depth - 1) / 2, cut_node);
        ss->excluded_move = MOVE_NONE;
        if (singular_value < sBeta) {
            singular_extension = 1;
            // Double extension: if move is EXTREMELY singular (fails by a lot),
            // extend even more - inspired by Stockfish's double extension
            if (singular_value < sBeta - depth) {
                singular_extension = 2;
            }
        } else if (singular_value >= beta) {
            // Multi-cut: other moves are also good enough to beat beta
            // Return immediately without searching further
            return singular_value;
        } else if (tt_value >= beta) {
            // TT move is NOT singular but expected to beat beta
            // Negative extension: reduce depth for non-singular expected cutoff
            singular_extension = -2;
        } else if (cut_node) {
            singular_extension = -1;
        }
    }

    // Helper lambda: compute LMR reduction for a move (takes gives_chk to avoid recomputation)
    auto compute_reduction = [&](Move m, int mp, bool gives_chk) -> int {
        // Logarithmic LMR formula with history-based adjustments
        int reduction = int(1.35 + std::log(double(std::max(depth - 1, 1))) * std::log(double(std::max(mp, 1))) / 2.75);
        if (!ss->improving && !opponent_worsening) reduction += 1;
        if (cut_node) reduction += 1;

        // TT move gets less reduction
        if (m == tt_move) reduction -= 1;

        // History-based adjustment: combine plain + counter + continuation
        Piece pc = pos.piece_on(m.from());
        if (pc != NO_PIECE) {
            int history_score = worker->history[int(pc)][int(m.to())];

            // Counter-move history (1-ply)
            if (ss->ply >= 1 && (ss - 1)->current_move != MOVE_NONE && (ss - 1)->moved_piece != NO_PIECE) {
                Move prev_move = (ss - 1)->current_move;
                Piece prev_pc = (ss - 1)->moved_piece;
                history_score += counter_moves[int(prev_pc)][int(prev_move.to())][int(pc)][int(m.to())];
            }

            // Continuation history (2-ply)
            if (ss->ply >= 2 && (ss - 2)->current_move != MOVE_NONE && (ss - 2)->moved_piece != NO_PIECE) {
                Move prev2_move = (ss - 2)->current_move;
                Piece prev2_pc = (ss - 2)->moved_piece;
                history_score += continuation_history[int(prev2_pc)][int(prev2_move.to())][int(pc)][int(m.to())];
            }

            // History influence on reduction
            reduction -= history_score / 8192;
        }

        // Reduce less for killer moves
        if (m == worker->killers[ss->ply][0]) reduction -= 1;

        // Counter-move bonus
        if (ss->ply >= 1 && (ss - 1)->current_move != MOVE_NONE && (ss - 1)->moved_piece != NO_PIECE) {
            Move prev_move = (ss - 1)->current_move;
            Piece prev_pc = (ss - 1)->moved_piece;
            if (m == worker->counter_move_table[int(prev_pc)][int(prev_move.to())]) reduction -= 1;
        }

        // Check-giving moves: reduce less (use pre-computed value)
        if (gives_chk) reduction -= 1;

        // Promotion moves: reduce less
        if (m.is_promotion()) reduction -= 1;

        return std::max(1, std::min(reduction, depth - 2));
    };

    // Helper lambda: compute check extension for a move
    auto gives_check = [&](Move m) -> bool {
        Square opp_ksq = pos.king_sq(Color(pos.side_to_move() ^ 1));
        PieceType pt = piece_type_of(pos.piece_on(m.from()));

        if (pt == PAWN) {
            return (pawn_attacks_bb(pos.side_to_move(), m.to()) & square_bb(opp_ksq)) != 0;
        } else if (pt == KNIGHT) {
            return (knight_attacks_bb(m.to()) & square_bb(opp_ksq)) != 0;
        } else if (pt == BISHOP) {
            return (bishop_attacks_bb(m.to(), pos.pieces() ^ square_bb(m.from())) & square_bb(opp_ksq)) != 0;
        } else if (pt == ROOK) {
            return (rook_attacks_bb(m.to(), pos.pieces() ^ square_bb(m.from())) & square_bb(opp_ksq)) != 0;
        } else if (pt == QUEEN) {
            Bitboard occ_no_from = pos.pieces() ^ square_bb(m.from());
            return (bishop_attacks_bb(m.to(), occ_no_from) & square_bb(opp_ksq)) != 0
                || (rook_attacks_bb(m.to(), occ_no_from) & square_bb(opp_ksq)) != 0;
        }
        return false;
    };

    // Helper lambda: search a single move, return true if beta cutoff
    auto search_move = [&](Move m, bool is_quiet) -> bool {
        if (stop.load(std::memory_order_relaxed)) return false;

        // Skip excluded move (used by singular extension)
        if (m == ss->excluded_move) return false;

        // Save captured piece type before do_move (for capture history update)
        PieceType captured_pt = m.is_capture() ? pos.piece_type_on(m.to()) : PT_NONE;

        // Skip pseudo_legal check: moves from generator are already pseudo-legal
        if (!pos.legal(m, true)) return false;  // Skip pseudo_legal check for generated moves

        // SEE-based capture pruning with depth-scaled margin
        if (!pv_node && ss->ply > 0 && m.is_capture() && !m.is_promotion() && depth <= 5) {
            int see_margin = depth * 80;
            if (!pos.see_ge(m, Value(-see_margin))) {
                return false;
            }
        }

        // Continuation pruning: skip quiet moves with very poor history at low depth
        // Worth ~10 ELO (from Ethereal). Uses combined counter+continuation history.
        if (is_quiet && !pv_node && depth <= 4 && moves_played > 0) {
            Piece pc = pos.piece_on(m.from());
            if (pc != NO_PIECE) {
                int hist_score = worker->history[int(pc)][int(m.to())];
                if (ss->ply >= 1 && (ss - 1)->current_move != MOVE_NONE && (ss - 1)->moved_piece != NO_PIECE) {
                    Move prev_move = (ss - 1)->current_move;
                    Piece prev_pc = (ss - 1)->moved_piece;
                    hist_score += counter_moves[int(prev_pc)][int(prev_move.to())][int(pc)][int(m.to())];
                }
                if (ss->ply >= 2 && (ss - 2)->current_move != MOVE_NONE && (ss - 2)->moved_piece != NO_PIECE) {
                    Move prev2_move = (ss - 2)->current_move;
                    Piece prev2_pc = (ss - 2)->moved_piece;
                    hist_score += continuation_history[int(prev2_pc)][int(prev2_move.to())][int(pc)][int(m.to())];
                }
                // Prune if history is very negative (move has historically been bad)
                if (hist_score < 783 - 4872 * (depth - 1)) {
                    return false;
                }
            }
        }

        // SEE-based quiet move pruning at shallow depth
        if (is_quiet && !pv_node && ss->ply > 0 && depth <= 3) {
            if (!pos.see_ge(m, Value(-20 * depth * depth))) {
                return false;
            }
        }

        // Late move pruning: prune quiet moves after examining a reasonable number
        // Improving positions can tolerate more pruning (more likely to recover)
        int lmp_base = ss->improving ? 3 : 2;
        int lmp_threshold = lmp_base + depth * depth;
        if (is_quiet && !pv_node && ss->ply > 0 && depth <= 6 && moves_played >= lmp_threshold) {
            return false;
        }

        // Futility pruning: skip quiet moves that can't improve alpha
        if (is_quiet && !pv_node && ss->ply > 0 && !pos.is_check() && depth <= 5) {
            int margin = depth * 150 + (ss->improving ? 30 : 100);
            if (eval + margin < alpha) {
                return false;
            }
        }

        // Late Move Reduction: reduce moves that are unlikely to be best
        // Apply LMR to quiet moves and losing captures (SEE < 0)
        // But NEVER reduce moves that give check — they are forcing
        Depth new_depth = depth - 1;
        bool gives_chk = gives_check(m);
        bool is_losing_capture = m.is_capture() && !m.is_promotion() && depth >= 1 &&
                                  !pos.see_ge(m, Value(-depth * 80));
        bool do_lmr = depth >= 3 && moves_played >= 3 &&
                      (is_quiet || is_losing_capture) && !gives_chk;

        if (do_lmr) {
            int reduction = compute_reduction(m, moves_played, gives_chk);
            // Reduce less in PV nodes
            if (pv_node) reduction = std::max(1, reduction - 1);
            new_depth = depth - 1 - reduction;
            if (new_depth < 1) new_depth = 1;
        }

        // Extensions
        int ext_count = 0;
        if (m == tt_move) {
            ext_count += singular_extension;
        }
        if (gives_chk) {
            ext_count++;
        }
        // Recapture extension: recapturing on the same square is often forced/tactical
        if (ss->ply >= 1 && (ss - 1)->current_move != MOVE_NONE && m.to() == (ss - 1)->current_move.to()) {
            if (m.is_capture()) {
                ext_count++;
            }
        }
        new_depth += ext_count;

        // Store current move and moved piece for counter-move history
        ss->current_move = m;
        ss->moved_piece = pos.piece_on(m.from());

        if (!pos.do_move(m)) {
            return false;
        }

        Value value;
        if (do_lmr && new_depth > 0) {
            // LMR: reduced zero-window search
            value = -search_worker(pos, ss + 1, -alpha - 1, -alpha, new_depth, !cut_node);
            if (stop.load(std::memory_order_relaxed)) {
                pos.undo_move(m);
                return false;
            }
            // Deeper/shallower re-search adjustment
            // Only apply when we have a valid best_value (not first move)
            if (value > alpha) {
                if (best_value > -VALUE_INFINITE + 100) {
                    bool do_deeper = value > best_value + 48;
                    bool do_shallower = value < best_value + 9;
                    int re_depth = depth - 1 + (do_deeper ? 1 : 0) - (do_shallower ? 1 : 0);
                    re_depth = std::max(1, re_depth);
                    value = -search_worker(pos, ss + 1, -beta, -alpha, re_depth, !cut_node);
                } else {
                    value = -search_worker(pos, ss + 1, -beta, -alpha, depth - 1, !cut_node);
                }
                if (stop.load(std::memory_order_relaxed)) {
                    pos.undo_move(m);
                    return false;
                }
            }
        } else if (moves_played > 0) {
            // PVS: zero-window scout search for non-first moves
            value = -search_worker(pos, ss + 1, -alpha - 1, -alpha, depth - 1, !cut_node);
            if (stop.load(std::memory_order_relaxed)) {
                pos.undo_move(m);
                return false;
            }
            // Re-search with full window if scout found improvement
            if (value > alpha && value < beta) {
                value = -search_worker(pos, ss + 1, -beta, -alpha, depth - 1, false);
                if (stop.load(std::memory_order_relaxed)) {
                    pos.undo_move(m);
                    return false;
                }
            }
        } else {
            // First move: full window search
            value = -search_worker(pos, ss + 1, -beta, -alpha, depth - 1, false);
            if (stop.load(std::memory_order_relaxed)) {
                pos.undo_move(m);
                return false;
            }
        }

        pos.undo_move(m);

        moves_played++;

        // Track quiet moves for history gravity
        if (is_quiet && quiet_count < 64) {
            quiets_searched[quiet_count++] = m;
        }

        if (value > best_value) {
            best_value = value;
            best_move_found = m;

            if (value > alpha) {
                alpha = value;

                // Update PV (guard against array overflow)
                if (ss->ply < 63) {
                    ss->pv[ss->ply] = m;
                    int i = 0;
                    for (; (ss + 1)->pv[(ss + 1)->ply + i] != MOVE_NONE && ss->ply + 1 + i < 63; ++i) {
                        ss->pv[ss->ply + 1 + i] = (ss + 1)->pv[(ss + 1)->ply + i];
                    }
                    ss->pv[ss->ply + 1 + i] = MOVE_NONE;
                }

                if (value >= beta) {
                // Beta cutoff - update killers, history and counter-move history
                if (is_quiet && ss->ply < MAX_PLY) {
                    if (m != worker->killers[ss->ply][0]) {
                        worker->killers[ss->ply][1] = worker->killers[ss->ply][0];
                        worker->killers[ss->ply][0] = m;
                    }
                    Piece pc = ss->moved_piece;
                    if (pc != NO_PIECE) {
                        // Stockfish-style stat_bonus: non-linear scaling with depth
                        int bonus = depth > 15 ? -8 : 19 * depth * depth + 155 * depth - 132;
                        // Gravity formula for plain history
                        int& h = worker->history[int(pc)][int(m.to())];
                        h += bonus - h * abs(bonus) / 32768;

                        // Counter-move history (1-ply: opponent's last move)
                        if (ss->ply >= 1 && (ss - 1)->current_move != MOVE_NONE && (ss - 1)->moved_piece != NO_PIECE) {
                            Move prev_move = (ss - 1)->current_move;
                            Piece prev_pc = (ss - 1)->moved_piece;
                            int& cm = counter_moves[int(prev_pc)][int(prev_move.to())][int(pc)][int(m.to())];
                            cm += bonus - cm * abs(bonus) / 32768;
                            worker->counter_move_table[int(prev_pc)][int(prev_move.to())] = m;
                        }

                        // Continuation history (2-ply: our own previous move)
                        if (ss->ply >= 2 && (ss - 2)->current_move != MOVE_NONE && (ss - 2)->moved_piece != NO_PIECE) {
                            Move prev2_move = (ss - 2)->current_move;
                            Piece prev2_pc = (ss - 2)->moved_piece;
                            int& ch = continuation_history[int(prev2_pc)][int(prev2_move.to())][int(pc)][int(m.to())];
                            ch += bonus - ch * abs(bonus) / 32768;
                        }
                    }
                    // Capture history update for capture moves that caused cutoff
                    if (!is_quiet && ss->ply < MAX_PLY && captured_pt != PT_NONE) {
                        int cap_bonus = depth > 15 ? -8 : 19 * depth * depth + 155 * depth - 132;
                        int& ch = worker->capture_history[int(ss->moved_piece)][int(m.to())][int(captured_pt)];
                        ch += cap_bonus - ch * std::abs(cap_bonus) / 32768;
                    }
                    // History gravity malus for non-cutoff quiet moves (use stat_bonus formula)
                    int malus = depth > 15 ? 8 : -(19 * depth * depth + 155 * depth - 132);
                    for (int i = 0; i < quiet_count - 1; ++i) {
                        Move qm = quiets_searched[i];
                        Piece qpc = pos.piece_on(qm.from());
                        if (qpc != NO_PIECE) {
                            int& h = worker->history[int(qpc)][int(qm.to())];
                            h += malus - h * abs(malus) / 32768;
                            if (ss->ply >= 1 && (ss - 1)->current_move != MOVE_NONE && (ss - 1)->moved_piece != NO_PIECE) {
                                Move prev_move = (ss - 1)->current_move;
                                Piece prev_pc = (ss - 1)->moved_piece;
                                int& cm = counter_moves[int(prev_pc)][int(prev_move.to())][int(qpc)][int(qm.to())];
                                cm += malus - cm * abs(malus) / 32768;
                            }
                            if (ss->ply >= 2 && (ss - 2)->current_move != MOVE_NONE && (ss - 2)->moved_piece != NO_PIECE) {
                                Move prev2_move = (ss - 2)->current_move;
                                Piece prev2_pc = (ss - 2)->moved_piece;
                                int& ch = continuation_history[int(prev2_pc)][int(prev2_move.to())][int(qpc)][int(qm.to())];
                                ch += malus - ch * abs(malus) / 32768;
                            }
                        }
                    }
                }
                if (!stop.load(std::memory_order_relaxed)) {
                    tte->save(pos.key(), value, false, BOUND_LOWER, depth, m, eval, TT.generation());
                }
                return true;  // beta cutoff
                }
            }
        }
        return false;  // no cutoff
    };

    // ========================================
    // PHASE 1: TT move (already known, free)
    // ========================================
    if (tt_move != MOVE_NONE) {
        if (search_move(tt_move, false)) {
            // TT move caused cutoff - early return
            if (!stop.load(std::memory_order_relaxed)) {
                // best_value already set inside search_move
                return best_value;
            }
        }
    }

    // ========================================
    // PHASE 2: Captures + promotions
    // ========================================
    {
        ExtMove captures[MAX_MOVES];
        ExtMove* cap_end = generate<GEN_CAPTURE>(pos, captures);

        // Score captures: MVV-LVA + SEE classification
        static constexpr int piece_value[] = {100, 320, 330, 500, 900, 20000, 0};
        for (ExtMove* it = captures; it != cap_end; ++it) {
            Move m = it->move;
            int score = 0;

            if (m == tt_move) {
                score = -1;  // Already searched in Phase 1, skip
            } else if (m.is_promotion()) {
                score = 1800000 + m.promotion_type() * 10000;
            } else {
                PieceType captured = pos.piece_type_on(m.to());
                PieceType attacker = pos.piece_type_on(m.from());
                int mvv_lva = piece_value[captured] * 10 - piece_value[attacker];
                // Add capture history for better ordering
                int cap_hist = worker->capture_history[int(pos.piece_on(m.from()))][int(m.to())][int(captured)];
                if (pos.see_ge(m, VALUE_ZERO)) {
                    score = 1500000 + mvv_lva + cap_hist;
                } else {
                    score = 500000 + mvv_lva;
                }
            }
            it->value = score;
        }

        // Pick-best through captures
        for (ExtMove* it = captures; it != cap_end; ++it) {
            if (stop.load(std::memory_order_relaxed)) break;

            // Pick best remaining
            {
                ExtMove* best = it;
                for (ExtMove* jt = it + 1; jt != cap_end; ++jt) {
                    if (jt->value > best->value) best = jt;
                }
                if (best != it) {
                    ExtMove tmp = *it;
                    *it = *best;
                    *best = tmp;
                }
            }

            // Skip TT move (already searched in Phase 1)
            if (it->value == -1) continue;

            if (search_move(it->move, false)) {
                if (!stop.load(std::memory_order_relaxed)) {
                    return best_value;
                }
            }
        }
    }

    // ========================================
    // PHASE 3: Quiet moves (only if no cutoff from captures)
    // This is the key savings: we never generate quiet moves
    // when a capture already caused beta cutoff.
    // ========================================
    {
        ExtMove quiets[MAX_MOVES];
        ExtMove* quiet_end = generate<GEN_QUIET>(pos, quiets);

        // Score quiets: killers + counter-moves + history + escape-aware
        for (ExtMove* it = quiets; it != quiet_end; ++it) {
            Move m = it->move;
            int score = 0;

            if (m == tt_move) {
                score = -1;  // Already searched
            } else {
                Piece pc = pos.piece_on(m.from());
                if (pc != NO_PIECE) {
                    score = worker->history[int(pc)][int(m.to())];
                    if (ss->ply >= 1 && (ss - 1)->current_move != MOVE_NONE && (ss - 1)->moved_piece != NO_PIECE) {
                        Move prev_move = (ss - 1)->current_move;
                        Piece prev_pc = (ss - 1)->moved_piece;
                        score += counter_moves[int(prev_pc)][int(prev_move.to())][int(pc)][int(m.to())];
                    }
                    // Continuation history (2-ply)
                    if (ss->ply >= 2 && (ss - 2)->current_move != MOVE_NONE && (ss - 2)->moved_piece != NO_PIECE) {
                        Move prev2_move = (ss - 2)->current_move;
                        Piece prev2_pc = (ss - 2)->moved_piece;
                        score += continuation_history[int(prev2_pc)][int(prev2_move.to())][int(pc)][int(m.to())];
                    }
                }

                // Previous iteration's root best move bonus
                if (m == previous_root_best) score += 70000;

                // Killer moves bonus ON TOP of history
                if (m == worker->killers[ss->ply][0]) score += 60000;
                else if (m == worker->killers[ss->ply][1]) score += 50000;
                else if (ss->ply >= 1 && (ss - 1)->current_move != MOVE_NONE && (ss - 1)->moved_piece != NO_PIECE) {
                    Move prev_move = (ss - 1)->current_move;
                    Piece prev_pc = (ss - 1)->moved_piece;
                    if (m == worker->counter_move_table[int(prev_pc)][int(prev_move.to())]) {
                        score += 40000;
                    }
                }

                // INNOVATION: "Escape-Aware Ordering" - pieces under pawn attack move first
                Piece moved_piece = pos.piece_on(m.from());
                if (moved_piece != NO_PIECE && piece_type_of(moved_piece) != PAWN) {
                    Color them = Color(pos.side_to_move() ^ 1);
                    if (pawn_attacks_bb(them, pos.pieces(them, PAWN)) & square_bb(m.from())) {
                        score += 15000;
                    }
                }
            }
            it->value = score;
        }

        // Pick-best through quiets
        for (ExtMove* it = quiets; it != quiet_end; ++it) {
            if (stop.load(std::memory_order_relaxed)) break;

            // Pick best remaining
            {
                ExtMove* best = it;
                for (ExtMove* jt = it + 1; jt != quiet_end; ++jt) {
                    if (jt->value > best->value) best = jt;
                }
                if (best != it) {
                    ExtMove tmp = *it;
                    *it = *best;
                    *best = tmp;
                }
            }

            // Skip TT move (already searched in Phase 1)
            if (it->value == -1) continue;

            if (search_move(it->move, true)) {
                if (!stop.load(std::memory_order_relaxed)) {
                    return best_value;
                }
            }
        }
    }

    // Checkmate/stalemate detection: if no moves were played, there are no legal moves
    if (moves_played == 0 && !stop.load(std::memory_order_relaxed)) {
        if (pos.is_check()) {
            return -VALUE_MATE + ss->ply;
        }
        return VALUE_DRAW - (pos.side_to_move() == WHITE ? params.contempt / 2 : -params.contempt / 2);
    }

    // CRITICAL FIX: Only save to TT if search completed fully
    // If search was aborted (stop=true), don't save garbage/incomplete scores
    if (!stop.load(std::memory_order_relaxed)) {
        // Update pawn correction history when search completed
        // Only update when: not in check, eval is meaningful
        if (!pos.is_check() && depth >= 1 && abs(best_value) < VALUE_KNOWN_WIN) {
            Color stm = pos.side_to_move();
            Value raw_eval = eval_cached(pos);  // Get uncorrected eval
            Value diff = best_value - raw_eval;
            if (diff != 0) {
                uint64_t pawn_key = compute_pawn_key(pos);
                pawn_correction_history.update(stm, pawn_key, diff, depth);
            }
        }

        Bound bound;
        if (best_value >= beta) bound = BOUND_LOWER;
        else if (best_value > original_alpha) bound = BOUND_EXACT;  // PV node improved alpha
        else bound = BOUND_UPPER;
        // Use best_move_found instead of ss->pv[ss->ply] to avoid stale PV corruption
        tte->save(pos.key(), best_value, pv_node, bound, depth, best_move_found, eval, TT.generation());
    }

    return best_value;
}

} // namespace

// Helper thread function for lazy SMP
// Each helper runs its own iterative deepening, populating the shared TT
// thread_id is used to offset start depth to reduce TT contention
static void helper_thread_func(Position pos_copy, int thread_id) {
    // Create per-thread search worker
    SearchWorker* w = new SearchWorker();
    worker = w;
    local_nodes = 0;
    last_reported_nodes = 0;

    // Initialize stack
    for (int i = 0; i < MAX_PLY_PLUS_6; ++i) {
        w->stack[i].ply = i - 1;
        w->stack[i].current_move = MOVE_NONE;
        w->stack[i].moved_piece = NO_PIECE;
        w->stack[i].previous = i > 0 ? &w->stack[i - 1] : nullptr;
        w->stack[i].pv[0] = MOVE_NONE;
    }

    // Simple iterative deepening - no UCI output, no complex time management
    // Offset start depth by thread_id to reduce TT contention between threads
    Move prev_best = MOVE_NONE;
    Value prev_score = -VALUE_INFINITE;
    int effective_depth = (limits.depth == 0) ? MAX_PLY : limits.depth;
    int start_depth = 1 + thread_id;

    for (int d = start_depth; d <= effective_depth; ++d) {
        if (stop.load(std::memory_order_relaxed) || g_stop_requested) break;

        // Check time periodically
        if ((local_nodes & 1023) == 0) {
            nodes.fetch_add(local_nodes - last_reported_nodes, std::memory_order_relaxed);
            last_reported_nodes = local_nodes;
            if (check_time()) break;
        }

        // Aspiration window (same as main thread)
        Value alpha = -VALUE_INFINITE;
        Value beta = VALUE_INFINITE;
        if (d >= 4 && prev_score > -VALUE_KNOWN_WIN && prev_score < VALUE_KNOWN_WIN) {
            alpha = prev_score - 30;
            beta = prev_score + 30;
        }

        // Generate moves
        ExtMove moves[MAX_MOVES];
        ExtMove* end = generate<GEN_LEGAL>(pos_copy, moves);

        // Simple move ordering
        for (ExtMove* it = moves; it != end; ++it) {
            Move m = it->move;
            int score = 0;
            if (m == prev_best) score = 2000000;
            else if (m.is_capture()) {
                PieceType captured = pos_copy.piece_type_on(m.to());
                PieceType attacker = pos_copy.piece_type_on(m.from());
                static constexpr int pv[] = {100, 320, 330, 500, 900, 20000, 0};
                score = 1000000 + (captured != PT_NONE ? pv[captured] : 0) * 10 - pv[attacker];
            } else if (m.is_promotion()) {
                score = 900000 + m.promotion_type() * 10000;
            } else {
                Piece pc = pos_copy.piece_on(m.from());
                if (pc != NO_PIECE) score = w->history[int(pc)][int(m.to())];
            }
            it->value = score;
        }

        std::sort(moves, end, [](const ExtMove& a, const ExtMove& b) {
            return a.value > b.value;
        });

        Value depth_best_value = -VALUE_INFINITE;
        Move depth_best_move = MOVE_NONE;

        for (ExtMove* it = moves; it != end; ++it) {
            if (stop.load(std::memory_order_relaxed) || g_stop_requested) break;
            if ((local_nodes & 1023) == 0) {
                nodes.fetch_add(local_nodes - last_reported_nodes, std::memory_order_relaxed);
            last_reported_nodes = local_nodes;
                if (check_time()) break;
            }

            if (!pos_copy.do_move(it->move)) continue;

            Value value = -search_worker(pos_copy, w->stack + 1, -beta, -alpha, d - 1, false);

            pos_copy.undo_move(it->move);

            if (stop.load(std::memory_order_relaxed)) break;

            if (value > depth_best_value) {
                depth_best_value = value;
                depth_best_move = it->move;
            }
            if (value > alpha) {
                alpha = value;
            }
            if (value >= beta) break;
        }

        if (stop.load(std::memory_order_relaxed)) break;

        // Handle aspiration fail
        if (depth_best_move != MOVE_NONE) {
            if (depth_best_value <= alpha || depth_best_value >= beta) {
                // Fail - re-search with full window
                alpha = -VALUE_INFINITE;
                beta = VALUE_INFINITE;
                depth_best_value = -VALUE_INFINITE;
                depth_best_move = MOVE_NONE;

                for (ExtMove* it = moves; it != end; ++it) {
                    if (stop.load(std::memory_order_relaxed) || g_stop_requested) break;

                    if (!pos_copy.do_move(it->move)) continue;
                    Value value = -search_worker(pos_copy, w->stack + 1, -beta, -alpha, d - 1, false);
                    pos_copy.undo_move(it->move);

                    if (stop.load(std::memory_order_relaxed)) break;

                    if (value > depth_best_value) {
                        depth_best_value = value;
                        depth_best_move = it->move;
                    }
                    if (value > alpha) alpha = value;
                    if (value >= beta) break;
                }

                if (stop.load(std::memory_order_relaxed)) break;
            }

            // Update prev for next depth
            prev_best = depth_best_move;
            prev_score = depth_best_value;

            // Save to TT (shared with main thread)
            if (!stop.load(std::memory_order_relaxed)) {
                bool found;
                TTEntry* tte = TT.probe(pos_copy.key(), found);
                tte->save(pos_copy.key(), depth_best_value, true, BOUND_EXACT, d, depth_best_move, VALUE_ZERO, TT.generation());
            }
        }
    }

    // Flush final node count
    nodes.fetch_add(local_nodes - last_reported_nodes, std::memory_order_relaxed);
    last_reported_nodes = local_nodes;

    delete w;
    worker = nullptr;
}

// Root search with iterative deepening
Move search(Position& pos, Limits& lim) {
    limits = lim;
    g_stop_requested = false;
    stop = false;
    nodes = 0;
    local_nodes = 0;
    last_reported_nodes = 0;

    // Track search start time for time management
    search_start = std::chrono::steady_clock::now();

    Move best_move = MOVE_NONE;
    Value best_value = -VALUE_INFINITE;
    root_score = best_value;
    previous_root_best = MOVE_NONE;

    // Check if we have any legal moves at all
    ExtMove initial_moves[MAX_MOVES];
    ExtMove* initial_end = generate<GEN_LEGAL>(pos, initial_moves);

    if (initial_moves == initial_end) {
        // No legal moves for us - we are checkmated or stalemated
        bool is_checkmate = pos.is_check();
        if (is_checkmate) {
            // We are checkmated
            uci_safe_output("info depth 1 score mate 0 nodes 0 nps 0\n");
        } else {
            // We are stalemated
            uci_safe_output("info depth 1 score cp 0 nodes 0 nps 0\n");
        }

        return MOVE_NONE;  // No move to make
    }

    // Create main thread search worker
    SearchWorker* main_worker = new SearchWorker();
    worker = main_worker;

    // Initialize search stack
    // CRITICAL: stack[0] is unused (ply -1), stack[1] is ply 0 for root search
    for (int i = 0; i < MAX_PLY_PLUS_6; ++i) {
        worker->stack[i].ply = i - 1;  // stack[0].ply = -1, stack[1].ply = 0, etc.
        worker->stack[i].current_move = MOVE_NONE;
        worker->stack[i].moved_piece = NO_PIECE;
        worker->stack[i].previous = i > 0 ? &worker->stack[i - 1] : nullptr;
        worker->stack[i].pv[0] = MOVE_NONE;
    }

    // No opening book - rely on search for best opening moves

    // Launch helper threads for lazy SMP
    helpers_running = true;
    for (int i = 1; i < num_threads; ++i) {
        helper_threads.emplace_back(helper_thread_func, pos, i);
    }

    // Time management — Third Generation: Stability + Score Progression
    if (limits.use_time_management()) {
        Color us = pos.side_to_move();
        int time_left = limits.time[int(us)];
        int time_inc = limits.inc[int(us)];

        if (time_left < 0) time_left = 0;
        if (time_inc < 0) time_inc = 0;

        int overhead = 30;

        if (limits.movestogo > 0) {
            int mtg = std::max(1, limits.movestogo);
            ideal_time = time_left / mtg + time_inc - overhead;
            max_time = time_left / std::max(1, mtg / 2) + time_inc - overhead;
        } else {
            // Sudden death: simple fraction + increment
            int mtg = 25;
            int opt_time = (time_left + time_inc * mtg * 3 / 4) / mtg;
            ideal_time = std::min(opt_time, time_left / 2);
            max_time = std::min(time_left - overhead, std::max(ideal_time * 3, time_inc + 50));
        }

        ideal_time = std::max(10, ideal_time);
        max_time = std::max(ideal_time, max_time);
        ideal_time = std::min(ideal_time, time_left - overhead);
        max_time = std::min(max_time, time_left - overhead);
    } else {
        ideal_time = 0;
        max_time = 0;
    }

    // Initialize killers (clear for new search)
    for (int i = 0; i < MAX_PLY; ++i) {
        worker->killers[i][0] = MOVE_NONE;
        worker->killers[i][1] = MOVE_NONE;
    }

    // Age history tables (gravity formula handles intra-search decay,
    // so we only need gentle inter-search aging)
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 64; ++j) {
            worker->history[i][j] /= 2;
        }
    }

    // Counter-move and continuation history: don't age at all.
    // The gravity formula (bonus - h * abs(bonus) / 16384) already handles
    // natural decay within each search. Aging 1M+ entries is too expensive
    // and provides marginal benefit over gravity-based decay.

    // Age pawn correction history between searches
    pawn_correction_history.age();

    // Iterative deepening
    // When depth=0, search until time runs out (tournament time control)
    // When depth>0, search to that specific depth
    // Always start from depth 1 for proper iterative deepening
    int effective_depth = (limits.depth == 0) ? MAX_PLY : limits.depth;
    int start_depth = 1;

    for (root_depth = start_depth; root_depth <= effective_depth; ++root_depth) {
        // Check stop at the very start of each depth iteration
        if (g_stop_requested || stop.load(std::memory_order_relaxed)) {
            break;
        }

        // Time management: stop before starting a new depth if we've used too much time
        // Always complete depth 1, then be aggressive about stopping
        if (limits.use_time_management() && root_depth > 1) {
            auto now = std::chrono::steady_clock::now();
            int elapsed = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                now - search_start).count());
            if (elapsed > ideal_time) {
                break;
            }
        }
        if (limits.movetime) {
            auto now = std::chrono::steady_clock::now();
            int elapsed = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                now - search_start).count());
            if (elapsed >= (limits.movetime * 7) / 10) {
                break;
            }
        }

        // Check time before starting a new depth
        check_time();
        if (stop.load(std::memory_order_relaxed)) break;

        // Aspiration window - use previous depth's score
        // Start with very wide window for stability
        Value alpha = -VALUE_INFINITE;
        Value beta = VALUE_INFINITE;
        int aspiration_delta = 30;

        if (root_depth >= 4 && best_value > -VALUE_KNOWN_WIN && best_value < VALUE_KNOWN_WIN) {
            alpha = std::max(Value(-VALUE_INFINITE), Value(best_value - aspiration_delta));
            beta = std::min(Value(VALUE_INFINITE), Value(best_value + aspiration_delta));
        }

        // Generate moves
        ExtMove moves[MAX_MOVES];
        ExtMove* end = generate<GEN_LEGAL>(pos, moves);

        // Order moves at root for better efficiency
        for (ExtMove* it = moves; it != end; ++it) {
            Move m = it->move;
            int score = 0;

            // CRITICAL: Prioritize previous iteration's best move (PV move)
            if (root_depth > 1 && m == best_move) {
                score = 2000000;  // Highest priority - previous PV move
            }
            // Prioritize winning captures (MVV-LVA)
            else if (m.is_capture()) {
                PieceType captured = pos.piece_type_on(m.to());
                PieceType attacker = pos.piece_type_on(m.from());
                static constexpr int piece_value[] = {100, 320, 330, 500, 900, 20000, 0};
                score = 1000000 + piece_value[captured] * 10 - piece_value[attacker];
            }
            // Promotions
            else if (m.is_promotion()) {
                score = 900000 + m.promotion_type() * 10000;
            }
            // Quiet moves: use history + killer ordering (was missing before!)
            else {
                Piece pc = pos.piece_on(m.from());
                if (pc != NO_PIECE) {
                    score = worker->history[int(pc)][int(m.to())];
                    if (m == worker->killers[0][0]) score += 500000;
                    else if (m == worker->killers[0][1]) score += 400000;
                }
            }

            it->value = score;
        }

        // Sort moves by score (only need to sort once per depth)
        std::sort(moves, end, [](const ExtMove& a, const ExtMove& b) {
            return a.value > b.value;
        });

        // Clean aspiration loop with root PVS
        Value depth_best_value = -VALUE_INFINITE;
        Move depth_best_move = MOVE_NONE;

        int aspiration_attempts = 0;
        while (true) {
            aspiration_attempts++;
            // Reset on each aspiration iteration to avoid stale values
            depth_best_value = -VALUE_INFINITE;
            depth_best_move = MOVE_NONE;
            if (aspiration_attempts > 5) {
                alpha = -VALUE_INFINITE;
                beta = VALUE_INFINITE;
            }

            // Search all root moves with current bounds
            int root_moves_searched = 0;
            Value running_alpha = alpha;  // PVS window only, NOT the aspiration bound

            for (ExtMove* it = moves; it != end; ++it) {
                if (g_stop_requested || stop.load(std::memory_order_relaxed)) break;
                if (check_time()) break;

                if (!pos.do_move(it->move)) continue;

                // Check stop immediately after do_move
                if (g_stop_requested || stop.load(std::memory_order_relaxed)) {
                    pos.undo_move(it->move);
                    break;
                }

                Value value;
                // PVS: first move gets full window, rest get zero-window scout
                if (root_moves_searched == 0) {
                    value = -search_worker(pos, worker->stack + 1, -beta, -running_alpha,
                                           root_depth - 1, false);
                } else {
                    // Zero-window scout search
                    value = -search_worker(pos, worker->stack + 1, -running_alpha - 1, -running_alpha,
                                           root_depth - 1, true);
                    // CRITICAL: Check stop immediately after recursive call
                    if (g_stop_requested || stop.load(std::memory_order_relaxed)) {
                        pos.undo_move(it->move);
                        break;
                    }
                    // Re-search with full window only if scout found something better
                    if (value > running_alpha && value < beta) {
                        value = -search_worker(pos, worker->stack + 1, -beta, -running_alpha,
                                               root_depth - 1, false);
                    }
                }

                // CRITICAL: Check stop after recursive search
                if (stop.load(std::memory_order_relaxed)) {
                    pos.undo_move(it->move);
                    break;
                }

                pos.undo_move(it->move);
                root_moves_searched++;

                if (check_time()) break;

                if (value > depth_best_value) {
                    depth_best_value = value;
                    depth_best_move = it->move;
                }
                if (value > running_alpha) {
                    running_alpha = value;
                }
                if (value >= beta) break;  // Beta cutoff
            }

            // Check if time ran out during search
            if (stop.load(std::memory_order_relaxed)) {
                // Use whatever we have from this iteration
                // depth_best_value and depth_best_move already set
                break;
            }

            // Aspiration window check uses ORIGINAL alpha (not running_alpha)
            if (depth_best_value < alpha) {
                // Fail low - widen downward (lower alpha only, keep beta)
                alpha = std::max(Value(-VALUE_INFINITE),
                                 Value(depth_best_value - aspiration_delta));
                aspiration_delta += aspiration_delta / 2;
            } else if (depth_best_value >= beta) {
                // Fail high - widen upward (raise beta only, keep alpha)
                beta = std::min(Value(VALUE_INFINITE),
                                Value(depth_best_value + aspiration_delta));
                aspiration_delta += aspiration_delta / 2;
            } else {
                // Score inside window - accept result
                // depth_best_value and depth_best_move already set
                break;
            }

            // Safety: if delta is huge, use full window
            if (aspiration_delta > 1200) {
                alpha = -VALUE_INFINITE;
                beta = VALUE_INFINITE;
            }
        }

        // Check time after all moves searched
        check_time();
        if (stop.load(std::memory_order_relaxed)) break;

        // Update overall best with this depth's result
        if (depth_best_move != MOVE_NONE) {
            best_value = depth_best_value;
            best_move = depth_best_move;
            previous_root_best = depth_best_move;
        }

        // Save root position to TT for PV extraction
        if (depth_best_move != MOVE_NONE && !stop.load(std::memory_order_relaxed)) {
            bool root_found;
            TTEntry* root_tte = TT.probe(pos.key(), root_found);
            root_tte->save(pos.key(), depth_best_value, true, BOUND_EXACT,
                           root_depth, depth_best_move, VALUE_ZERO, TT.generation());
        }

        root_score = depth_best_value;

        // Flush local node count to atomic for accurate reporting
        nodes.fetch_add(local_nodes - last_reported_nodes, std::memory_order_relaxed);
        last_reported_nodes = local_nodes;

        // Calculate elapsed time for NPS
        auto search_end = std::chrono::steady_clock::now();
        int time_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(search_end - search_start).count());

        // Send UCI info
        uci_info(pos, root_depth, depth_best_value, local_nodes, time_ms);
    }

    // Fallback: if best_move is still MOVE_NONE, use the first legal move we found
    if (best_move == MOVE_NONE) {
        if (initial_end > initial_moves) {
            best_move = initial_moves[0].move;
        } else {
            return MOVE_NONE;  // No legal moves
        }
    }

    // SAFETY: Verify the piece on the from-square belongs to side to move
    if (best_move != MOVE_NONE) {
        Piece pc = pos.piece_on(best_move.from());
        if (pc == NO_PIECE || color_of_piece(pc) != pos.side_to_move()) {
            // Move references wrong-color piece - fall back
            ExtMove fallback_moves[MAX_MOVES];
            ExtMove* fallback_end = generate<GEN_LEGAL>(pos, fallback_moves);
            best_move = (fallback_end > fallback_moves) ? fallback_moves[0].move : MOVE_NONE;
        }
    }

    // Final safety: verify best_move is actually in the legal move list
    if (best_move != MOVE_NONE) {
        ExtMove verify_moves[MAX_MOVES];
        ExtMove* verify_end = generate<GEN_LEGAL>(pos, verify_moves);
        bool found_in_legal = false;
        for (ExtMove* vit = verify_moves; vit != verify_end; ++vit) {
            if (vit->move.raw() == best_move.raw()) {
                found_in_legal = true;
                break;
            }
        }
        if (!found_in_legal) {
            // Fallback to first legal move
            if (verify_end > verify_moves) {
                best_move = verify_moves[0].move;
            } else {
                best_move = MOVE_NONE;
            }
        }
    }

    // Signal and stop helper threads
    stop.store(true, std::memory_order_relaxed);
    for (auto& t : helper_threads) {
        if (t.joinable()) t.join();
    }
    helper_threads.clear();
    helpers_running = false;

    // Clean up main thread search worker
    delete worker;
    worker = nullptr;

    return best_move;
}

void uci_info([[maybe_unused]] const Position& pos, int depth, Value score, uint64_t node_count, int time_ms) {
    std::ostringstream oss;
    oss << "info depth " << depth << " score ";

    // Handle mate scores
    // Note: -VALUE_INFINITE is not a mate score, it means no valid search result
    if (score >= VALUE_MATE_IN_MAX_PLY && score < VALUE_INFINITE) {
        // Mate in N moves (convert plies to moves)
        int mate_in = (VALUE_MATE - score + 1) / 2;
        oss << "mate " << mate_in;
    } else if (score <= -VALUE_MATE_IN_MAX_PLY && score > -VALUE_INFINITE) {
        // Being mated in N moves
        int mate_in = -(VALUE_MATE + score + 1) / 2;
        oss << "mate " << mate_in;
    } else {
        // Normal score in centipawns
        // Score is already from side-to-move's perspective (evaluate() returns side-to-move relative)
        int score_cp = score * 100 / PAWN_VALUE;
        oss << "cp " << score_cp;
    }

    oss << " nodes " << node_count
        << " nps " << (time_ms > 0 ? node_count * 1000 / time_ms : 0);

    // Extract PV from TT
    oss << " pv";
    Position pv_pos = pos;
    Move pv_moves[64];
    int pv_count = 0;
    uint64_t pv_keys[64];

    for (int i = 0; i < 64; ++i) {
        bool found;
        TTEntry* pv_tte = TT.probe(pv_pos.key(), found);
        if (!found || pv_tte->move() == MOVE_NONE) break;

        Move m = pv_tte->move();
        // Validate move is legal
        if (!pv_pos.legal(m)) break;

        // Cycle detection
        for (int j = 0; j < pv_count; ++j) {
            if (pv_keys[j] == pv_pos.key()) { pv_count = -1; break; }
        }
        if (pv_count < 0) break;

        pv_keys[pv_count] = pv_pos.key();
        pv_moves[pv_count++] = m;
        oss << " " << m;
        if (!pv_pos.do_move(m)) break;
    }

    // Undo PV moves to restore position
    for (int i = pv_count - 1; i >= 0; --i) {
        pv_pos.undo_move(pv_moves[i]);
    }

    oss << "\n";

    // Thread-safe output
    uci_safe_output(oss.str());
}

} // namespace luminex
