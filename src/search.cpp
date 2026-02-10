#define _CRT_SECURE_NO_WARNINGS
#include "luminex.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>

namespace luminex {

// Forward declaration - defined in uci.cpp
extern bool check_for_stop_command();

// Search globals
Limits limits;
SearchParams params;
std::atomic<uint64_t> nodes;
std::atomic<bool> stop;
int root_depth;
Value root_score;

namespace {

// Stack to store search states
constexpr int MAX_PLY_PLUS_6 = MAX_PLY + 6;
Stack stack[MAX_PLY_PLUS_6];

// Killer moves: moves that caused beta cutoffs at each ply
Move killers[MAX_PLY][2];

// History heuristic: [piece][to_square]
int history[12][64];

// Counter-move history: [prev_piece][prev_to][piece][to]
int counter_moves[12][64][12][64];



// Evaluation cache - store eval results to avoid recomputing expensive evals
struct EvalCacheEntry {
    uint64_t key;
    int32_t value;  // Changed from int16_t to match Value type
};
constexpr int EVAL_CACHE_SIZE = 8192;  // Power of 2 for fast indexing
EvalCacheEntry eval_cache[EVAL_CACHE_SIZE];

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

inline void clear_eval_cache() {
    std::memset(eval_cache, 0, sizeof(eval_cache));
}

// Reduction constants
constexpr int futility_margin(int depth, bool improving) {
    // Reduced futility margins for better tactical vision
    // Lowered by ~20% to prune less and search more positions
    int base = 100;  // Was 120
    if (depth == 1) base = 125;  // Was 150
    else if (depth == 2) base = 160;  // Was 200
    else if (depth == 3) base = 200;  // Was 250
    else base = 225 + (depth - 3) * 40;  // Was 280 + 50, depth >= 4

    // Adjust based on improving flag
    if (improving) base -= 30;  // Was 40
    else base += 50;  // Was 60

    return base;
}

// LMR reduction computation - balanced for accuracy
inline int lmr_reduction(int depth, int moves_played, bool improving, bool pv_node) {
    // Start with no base reduction
    int reduction = 0;

    // Progressive reduction based on move count
    if (moves_played >= 4) reduction += 1;
    if (moves_played >= 8) reduction += 1;
    if (moves_played >= 12) reduction += 1;

    // Node type and improvement
    if (!pv_node) reduction += 1;
    if (!improving) reduction += 1;

    // Depth-based: more reduction at deeper depths
    if (depth >= 6) reduction += 1;
    if (depth >= 10) reduction += 1;
    if (depth >= 15) reduction += 1;  // Extra reduction at very high depth
    if (depth >= 22) reduction += 1;  // Maximum reduction for extreme depths

    // Cap reduction - don't reduce too much
    if (reduction > depth - 2) reduction = depth - 2;
    if (reduction < 0) reduction = 0;

    return reduction;
}

// Search start time for time management
static std::chrono::steady_clock::time_point search_start;
static int ideal_time = 0;  // Ideal time to use for this search
static int max_time = 0;    // Maximum time to use

// Check time - returns true if time limit exceeded
bool check_time() {
    if (stop.load(std::memory_order_relaxed)) {
        return true;
    }

    if (limits.nodes && nodes >= uint64_t(limits.nodes)) {
        stop = true;
        return true;
    }

    if (limits.movetime) {
        auto now = std::chrono::steady_clock::now();
        int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - search_start).count();
        if (elapsed >= (limits.movetime - 50)) {
            stop = true;
            return true;
        }
    }

    if (limits.use_time_management()) {
        auto now = std::chrono::steady_clock::now();
        int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - search_start).count();
        if (elapsed >= ideal_time) {
            stop = true;
            return true;
        }
        if (elapsed >= max_time - 50) {
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

    // Don't search captures beyond a certain depth - limited to -8
    // Allows deeper tactical resolution while preventing unbounded recursion
    if (depth < -8) {
        return eval_cached(pos);
    }

    if (stop.load(std::memory_order_relaxed)) {
        return VALUE_ZERO;
    }

    ++nodes;

    // Check time very frequently to catch timeout during deep searches
    // Check every 64 nodes to prevent long-running searches from going over time
    if ((nodes & 63) == 0) {
        check_time();
    }

    // Check for draw
    if (pos.is_draw()) {
        // Apply contempt: if positive, prefer avoiding draws
        // From root perspective, positive contempt means "I want to avoid draws"
        // So we return VALUE_DRAW - contempt/2 for root player
        return VALUE_DRAW - (pos.side_to_move() == WHITE ? params.contempt / 2 : -params.contempt / 2);
    }

    // Evaluate position
    bool in_check = pos.is_check();

    if (!in_check) {
        // Stand pat only when NOT in check
        Value eval = evaluate(pos);
        if (eval >= beta) {
            return beta;
        }
        if (eval > alpha) {
            alpha = eval;
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

    for (ExtMove* it = moves; it != end; ++it) {
        if (in_check) {
            // For evasions, legal() check already done by GEN_LEGAL
        } else if (!pos.legal(it->move)) {
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

    if (stop.load(std::memory_order_relaxed)) {
        return VALUE_ZERO;
    }

    // Check for draw
    if (pos.is_draw()) {
        // Apply contempt: if positive, prefer avoiding draws
        // From root perspective, positive contempt means "I want to avoid draws"
        // So we return VALUE_DRAW - contempt/2 for root player
        return VALUE_DRAW - (pos.side_to_move() == WHITE ? params.contempt / 2 : -params.contempt / 2);
    }

    ++nodes;

    const bool pv_node = (beta - alpha > 1);

    // Prevent stale PV data from being saved to TT on fail-low nodes
    ss->pv[ss->ply] = MOVE_NONE;

    // Check time very frequently to catch timeout during deep searches
    // Check every 64 nodes to prevent long-running searches from going over time
    if ((nodes & 63) == 0) {
        check_time();
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
    }
    ss->static_eval = eval;

    // Compute improving flag: position is improving if eval is better than 2 plies ago
    ss->improving = (ss->ply >= 2 && eval > (ss - 2)->static_eval);

    // Internal Iterative Reduction (IIR): reduce depth by 1 when no TT move available
    // Much cheaper than IID (which does a full sub-search), improves move ordering naturally
    if (tt_move == MOVE_NONE && depth >= 4) {
        depth--;
    }

    // Futility pruning - use improving for better pruning decisions
    // Depth <= 5 for balance between pruning and tactical accuracy
    if (!pv_node && !pos.is_check() && depth <= 5 && eval - futility_margin(depth, ss->improving) >= beta) {
        return eval;
    }

    // Razoring: at low depths, if eval is far below alpha, try qsearch to confirm
    if (!pv_node && !pos.is_check() && depth <= 3) {
        Value razor_margin = 400 + depth * 100;  // More aggressive margin
        if (eval + razor_margin < alpha) {
            // Try quiescence search to confirm the position is really losing
            Value qsearch_value = qsearch(pos, ss, alpha - 1, alpha, 0);
            if (qsearch_value <= alpha) {
                return qsearch_value;  // Confirmed losing, prune
            }
        }
    }

    // Null move pruning - moderate reduction
    int piece_count = popcount(pos.pieces()) - popcount(pos.pieces(PAWN)) - 2;  // Exclude kings and pawns
    bool null_move_ok = !pv_node && !pos.is_check() && depth >= 2 && piece_count >= 4 &&
                          eval >= beta && ss->ply >= 1;  // Don't do at root

    if (null_move_ok) {
        pos.do_null_move();

        // Adaptive null move reduction: deeper at higher depths
        // R=3 at depth 2-5, R=4 at depth 6-11, R=5 at depth 12+, etc.
        int R = 3 + depth / 6;
        Value null_value = -search_worker(pos, ss + 1, -beta, -beta + 1, depth - R, !cut_node);

        pos.undo_null_move();

        if (null_value >= beta) {
            // Don't return mate scores from null move pruning
            if (null_value >= VALUE_MATE_IN_MAX_PLY) {
                null_value = beta;
            }
            return null_value;
        }
    }

    // ProbCut - ENABLED with legality check
    // If a capture is obviously good enough to cause beta cutoff, verify with shallow search
    // FIXED: Added pos.legal(m) check to prevent analyzing illegal positions
    if (!pv_node && depth >= 5 && !pos.is_check() && ss->ply >= 2) {
        Value rbeta = std::min(beta + 200, VALUE_INFINITE - 200);
        int rdepth = depth - 3;

        // Try winning captures (SEE > 0)
        ExtMove probcut_moves[MAX_MOVES];
        ExtMove* probcut_end = generate<GEN_CAPTURE>(pos, probcut_moves);

        for (ExtMove* it = probcut_moves; it != probcut_end; ++it) {
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

    // Generate moves - always generate all legal moves
    // Removed capture_only optimization which was blind to quiet threats deep in the tree
    ExtMove moves[MAX_MOVES];
    ExtMove* end = generate<GEN_LEGAL>(pos, moves);

    if (moves == end) {
        // No legal moves
        if (pos.is_check()) {
            return -VALUE_MATE + ss->ply;
        }
        // Stalemate - apply contempt
        return VALUE_DRAW - (pos.side_to_move() == WHITE ? params.contempt / 2 : -params.contempt / 2);
    }

    // Score moves for ordering
    for (ExtMove* it = moves; it != end; ++it) {
        Move m = it->move;
        int score = 0;

        // TT move gets highest priority
        if (m == tt_move) {
            score = 2000000;
        }
        // Promotions
        else if (m.is_promotion()) {
            score = 1800000 + m.promotion_type() * 10000;
        }
        // Captures - fast MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
        else if (m.is_capture()) {
            PieceType captured = pos.piece_type_on(m.to());
            PieceType attacker = pos.piece_type_on(m.from());

            // MVV-LVA scoring: capture value * 10 - attacker value
            // This orders PxQ > NxQ > BxQ > ... > PxQ
            // Within same victim, prefer cheaper attacker
            static constexpr int piece_value[] = {0, 100, 320, 330, 500, 900, 0};
            score = 1000000 + piece_value[captured] * 10 - piece_value[attacker];
        }
        // Killer moves and history for quiet moves
        else if (ss->ply < MAX_PLY) {
            if (m == killers[ss->ply][0]) score = 60000;
            else if (m == killers[ss->ply][1]) score = 50000;
            // Counter-move history and history heuristic
            else {
                Piece pc = pos.piece_on(m.from());
                if (pc != NO_PIECE) {
                    score = history[int(pc)][int(m.to())];

                    // Add counter-move history bonus if we have a previous move
                    if (ss->ply >= 2 && (ss - 1)->current_move != MOVE_NONE && (ss - 1)->moved_piece != NO_PIECE) {
                        Move prev_move = (ss - 1)->current_move;
                        Piece prev_pc = (ss - 1)->moved_piece;
                        score += counter_moves[int(prev_pc)][int(prev_move.to())][int(pc)][int(m.to())];
                    }
                }
            }
        }

        it->value = score;
    }

    // Sort by value (descending)
    std::sort(moves, end, [](const ExtMove& a, const ExtMove& b) {
        return a.value > b.value;
    });

    Value best_value = -VALUE_INFINITE;
    int moves_played = 0;

    // Singular extension DISABLED for speed - expensive feature
    bool tt_move_is_singular = false;
    /*
    if (depth >= 6 && pv_node && tt_move != MOVE_NONE && found &&
        (tte->bound() & BOUND_LOWER) && tt_depth >= depth - 3) {
        // ... singular extension code disabled ...
    }
    */

    for (ExtMove* it = moves; it != end; ++it) {
        Move m = it->move;

        // SEE-based capture pruning: skip captures that lose material
        // More conservative than before - use SEE to evaluate actual exchange
        if (!pv_node && ss->ply > 0 && m.is_capture() && !m.is_promotion() && depth <= 5) {
            // Skip captures with negative SEE (losing material)
            // Use depth-dependent margin: -depth * 80 allows "equal" captures at low depth
            if (!pos.see_ge(m, Value(-depth * 80))) {
                continue;
            }
        }

        // Late move pruning: standard threshold formula 3 + depth²
        // At depth 4: 3+16=19 moves, not 4. This prevents pruning good moves too early.
        if (!pv_node && ss->ply > 0 && moves_played >= 3 + depth * depth &&
            !m.is_capture() && !m.is_promotion() && !m.is_castling()) {
            continue;  // Skip very late quiet moves
        }

        // Futility pruning: skip quiet moves that can't improve alpha
        // Balanced - apply at depth <= 4
        if (!pv_node && ss->ply > 0 && !pos.is_check() && depth <= 4 &&
            !m.is_capture() && !m.is_promotion() && !m.is_castling()) {
            // Moderate futility margin
            int margin = depth * 200 + (ss->improving ? 0 : 100);

            // Check if move is futile (eval + margin < alpha)
            if (eval + margin < alpha) {
                continue;
            }
        }

        // Late Move Reduction (LMR)
        Depth new_depth = depth - 1;
        // LMR: apply from move 4 at depth 2+ for quiet moves only (standard threshold)
        bool do_lmr = !pv_node && depth >= 2 && moves_played >= 3 && !m.is_capture() && !m.is_promotion() && !m.is_castling();

        if (do_lmr) {
            // Use unified LMR reduction function
            int reduction = lmr_reduction(depth, moves_played, ss->improving, pv_node);

            // Additional cut node reduction
            if (cut_node) reduction += 1;

            // History-based adjustment: reduce less for moves with good history
            Piece pc = pos.piece_on(m.from());
            int history_score = 0;
            if (pc != NO_PIECE) {
                history_score = history[int(pc)][int(m.to())];

                // Add counter-move history if available
                if (ss->ply >= 2 && (ss - 1)->current_move != MOVE_NONE && (ss - 1)->moved_piece != NO_PIECE) {
                    Move prev_move = (ss - 1)->current_move;
                    Piece prev_pc = (ss - 1)->moved_piece;
                    history_score += counter_moves[int(prev_pc)][int(prev_move.to())][int(pc)][int(m.to())];
                }
            }

            // Adjust reduction based on history: good history = less reduction
            if (history_score > 2000) {
                reduction -= 1;
            } else if (history_score < -1000) {
                // Bad history = more reduction
                reduction += 1;
            }

            // Limit reduction
            if (reduction > depth - 2) reduction = depth - 2;
            if (reduction < 1) reduction = 1;  // Minimum reduction

            new_depth = depth - 1 - reduction;
            if (new_depth < 1) new_depth = 1;
        }

        // CRITICAL: Check evasion extension - when we're IN check, extend to find defensive moves
        // This is the highest priority extension - being mated is the worst outcome
        bool evasion_extension = (pos.is_check() && depth >= 2);

        // Check extension: extend by one ply if move gives check (helps find tactical sequences)
        // LIMITED: Only extend at high depths and for dangerous checks to prevent explosion
        bool gives_check = false;
        bool dangerous_check = false;
        if (depth >= 3 && !pos.is_check()) {  // Lowered from 4 to 3 for better tactics
            PieceType pt = pos.piece_type_on(m.from());
            Square to = m.to();
            Color opponent = Color(int(pos.side_to_move()) ^ 1);
            Square king_sq = pos.king_sq(opponent);

            if (pt == KNIGHT) {
                gives_check = (knight_attacks_bb(to) & square_bb(king_sq)) != 0;
                // Knight checks are often dangerous
                dangerous_check = gives_check;
            } else if (pt == BISHOP) {
                gives_check = (bb_diag_attacks(to, pos.pieces()) & square_bb(king_sq)) != 0;
                dangerous_check = gives_check;
            } else if (pt == ROOK) {
                gives_check = ((bb_rank_attacks(to, pos.pieces()) | bb_file_attacks(to, pos.pieces())) & square_bb(king_sq)) != 0;
                // Rook checks are dangerous
                dangerous_check = gives_check;
            } else if (pt == QUEEN) {
                gives_check = (queen_attacks_bb(to, pos.pieces()) & square_bb(king_sq)) != 0;
                // Queen checks are very dangerous
                dangerous_check = gives_check;
            } else if (pt == PAWN) {
                Bitboard pawn_attacks = 0;
                Bitboard pb = square_bb(to);
                if (pos.side_to_move() == WHITE) {
                    if (file_of(to) > FILE_A) pawn_attacks |= shift_nw(pb);
                    if (file_of(to) < FILE_H) pawn_attacks |= shift_ne(pb);
                } else {
                    if (file_of(to) > FILE_A) pawn_attacks |= shift_sw(pb);
                    if (file_of(to) < FILE_H) pawn_attacks |= shift_se(pb);
                }
                gives_check = (pawn_attacks & square_bb(king_sq)) != 0;
                // Pawn checks are less dangerous, only extend if deep
                dangerous_check = (depth >= 8 && gives_check);
            }
        }

        // Single extension: at most one extension per move to prevent explosion
        bool extension = false;

        // Evasion extension: HIGHEST priority - we're being attacked!
        if (evasion_extension) {
            extension = true;
        }
        // Singular extension: high priority (disabled for now)
        else if (m == tt_move && tt_move_is_singular) {
            extension = true;
        }
        // Check extension: ONLY for dangerous checks (was all checks)
        else if (dangerous_check) {
            extension = true;
        }

        if (extension) {
            new_depth++;
        }

        // Store current move and moved piece for counter-move history
        ss->current_move = m;
        ss->moved_piece = pos.piece_on(m.from());

        if (!pos.do_move(m)) {
            // CRITICAL: do_move failed - atomic failure, no state change
            // Do NOT call undo_move - do_move already guarantees state is unchanged
            continue;
        }

        Value value;
        if (do_lmr && new_depth > 0) {
            // Search with reduced depth first
            value = -search_worker(pos, ss + 1, -alpha - 1, -alpha, new_depth, !cut_node);
            if (value > alpha) {
                // Failed high, re-search at full depth
                value = -search_worker(pos, ss + 1, -beta, -alpha, depth - 1, !cut_node);
            }
        } else if (!pv_node && moves_played > 0 && depth >= 3 && best_value > -VALUE_MATE_IN_MAX_PLY) {
            // PVS for non-PV nodes only - use zero-window scout for moves after the first
            value = -search_worker(pos, ss + 1, -alpha - 1, -alpha, depth - 1, !cut_node);
            if (value > alpha) {
                value = -search_worker(pos, ss + 1, -beta, -alpha, depth - 1, !cut_node);
            }
        } else {
            // Full window search
            value = -search_worker(pos, ss + 1, -beta, -alpha, depth - 1, !cut_node);
        }

        pos.undo_move(m);

        moves_played++;

        if (value > best_value) {
            best_value = value;
            best_move_found = m;

            if (value > alpha) {
                alpha = value;

                // Update PV
                ss->pv[ss->ply] = it->move;
                ss->pv[ss->ply + 1] = MOVE_NONE;
            }

            if (value >= beta) {
                // Beta cutoff - update killers, history and counter-move history
                if (!m.is_capture() && ss->ply < MAX_PLY) {
                    // Update killer moves
                    if (m != killers[ss->ply][0]) {
                        killers[ss->ply][1] = killers[ss->ply][0];
                        killers[ss->ply][0] = m;
                    }
                    // Update history
                    Piece pc = ss->moved_piece;
                    if (pc != NO_PIECE) {
                        history[int(pc)][int(m.to())] += depth * depth;

                        // Overflow protection: if entry exceeds threshold, halve all history
                        if (history[int(pc)][int(m.to())] > 8000) {
                            for (int i = 0; i < 12; ++i)
                                for (int j = 0; j < 64; ++j)
                                    history[i][j] /= 2;
                        }

                        // Update counter-move history
                        if (ss->ply >= 1 && (ss - 1)->current_move != MOVE_NONE && (ss - 1)->moved_piece != NO_PIECE) {
                            Move prev_move = (ss - 1)->current_move;
                            Piece prev_pc = (ss - 1)->moved_piece;
                            int& cm = counter_moves[int(prev_pc)][int(prev_move.to())][int(pc)][int(m.to())];
                            cm += depth * depth;

                            // Clip individual counter-move entry to avoid expensive full table halving
                            if (cm > 8000) cm = 8000;
                        }
                    }
                }
                // CRITICAL FIX: Only save to TT if search wasn't aborted
                if (!stop.load(std::memory_order_relaxed)) {
                    tte->save(pos.key(), value, false, BOUND_LOWER, depth, it->move, eval, TT.generation());
                }
                return value;  // fail-soft
            }
        }
    }

    // CRITICAL FIX: Only save to TT if search completed fully
    // If search was aborted (stop=true), don't save garbage/incomplete scores
    if (!stop.load(std::memory_order_relaxed)) {
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

// Root search with iterative deepening
Move search(Position& pos, Limits& lim) {
    limits = lim;
    stop = false;
    nodes = 0;

    // Track search start time for time management
    search_start = std::chrono::steady_clock::now();

    Move best_move = MOVE_NONE;
    Value best_value = -VALUE_INFINITE;
    root_score = best_value;

    // Check if we have any legal moves at all
    ExtMove initial_moves[MAX_MOVES];
    ExtMove* initial_end = generate<GEN_LEGAL>(pos, initial_moves);

    if (initial_moves == initial_end) {
        // No legal moves for us - we are checkmated or stalemated
        bool is_checkmate = pos.is_check();
        if (is_checkmate) {
            // We are checkmated
            std::cout << "info depth 1 score mate 0 nodes 0 nps 0" << std::endl;
            std::cout.flush();
        } else {
            // We are stalemated
            std::cout << "info depth 1 score cp 0 nodes 0 nps 0" << std::endl;
            std::cout.flush();
        }

        return MOVE_NONE;  // No move to make
    }

    // Initialize search stack
    // CRITICAL: stack[0] is unused (ply -1), stack[1] is ply 0 for root search
    for (int i = 0; i < MAX_PLY_PLUS_6; ++i) {
        stack[i].ply = i - 1;  // stack[0].ply = -1, stack[1].ply = 0, etc.
        stack[i].current_move = MOVE_NONE;
        stack[i].moved_piece = NO_PIECE;
        stack[i].previous = i > 0 ? &stack[i - 1] : nullptr;
        stack[i].pv[0] = MOVE_NONE;
    }

    TT.new_search();
    clear_eval_cache();

    // Initialize time management for tournament time controls
    // Using CPW formula: base/20 + inc/2
    if (limits.use_time_management()) {
        Color us = pos.side_to_move();
        int time_left = limits.time[int(us)];
        int time_inc = limits.inc[int(us)];

        // Safety: clamp negatives
        if (time_left < 0) time_left = 0;
        if (time_inc < 0) time_inc = 0;

        // CPW formula: base/20 + inc/2
        int movestogo = (limits.movestogo > 0) ? limits.movestogo + 2 : 20;
        // The +2 is the CPW safety buffer: assume slightly more moves than told

        ideal_time = time_left / movestogo + time_inc / 2;

        // Hard bound: never use more than 1/3 of remaining time
        max_time = time_left / 3;

        // Overhead buffer: account for communication latency (50ms)
        const int overhead = 50;
        ideal_time = std::max(1, ideal_time - overhead);
        max_time = std::max(1, max_time - overhead);

        // Never exceed remaining time minus overhead
        max_time = std::min(max_time, std::max(1, time_left - overhead));
        ideal_time = std::min(ideal_time, max_time);

        // Absolute floor: always have at least 10ms to search
        ideal_time = std::max(10, ideal_time);
        max_time = std::max(10, max_time);
    } else {
        ideal_time = 0;
        max_time = 0;
    }

    // Initialize killers (clear for new search)
    for (int i = 0; i < MAX_PLY; ++i) {
        killers[i][0] = MOVE_NONE;
        killers[i][1] = MOVE_NONE;
    }

    // Age history tables (divide by 8) instead of clearing
    // This preserves valuable information between searches
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 64; ++j) {
            history[i][j] /= 8;
        }
    }

    // Age counter-move history
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 64; ++j) {
            for (int k = 0; k < 12; ++k) {
                for (int l = 0; l < 64; ++l) {
                    counter_moves[i][j][k][l] /= 8;
                }
            }
        }
    }

    // Iterative deepening
    // When depth=0, search until time runs out (tournament time control)
    // When depth>0, search to that specific depth
    // Always start from depth 1 for proper iterative deepening
    int effective_depth = (limits.depth == 0) ? MAX_PLY : limits.depth;
    // For time-controlled games, cap depth at 7 to prevent timeout
    // TODO: Fix the root cause of depth 8+ explosion and remove this cap
    if (limits.use_time_management() && effective_depth > 7) {
        effective_depth = 7;
    }
    int start_depth = 1;

    // Track previous score for aspiration windows
    Value previous_score = VALUE_ZERO;

    for (root_depth = start_depth; root_depth <= effective_depth; ++root_depth) {
        // Time management: stop before starting a new depth if we've used significant time
        // Be VERY conservative - depth N can take 10x longer than depth N-1
        if (limits.use_time_management()) {
            auto now = std::chrono::steady_clock::now();
            int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - search_start).count();
            // Stop if we've used more than 1/8 of ideal time
            if (elapsed > ideal_time / 8) {
                break;
            }
        }
        if (limits.movetime) {
            auto now = std::chrono::steady_clock::now();
            int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - search_start).count();
            if (elapsed >= (limits.movetime * 7) / 10) {
                break;
            }
        }

        // Check time before starting a new depth
        check_time();
        if (stop.load(std::memory_order_relaxed)) break;

        // Aspiration window - ENABLED for faster root searches
        // Use small window around previous score, widen only if fails
        int aspiration_delta = 50;  // Initial window size
        Value alpha;
        Value beta;

        if (root_depth > 1 && previous_score > -VALUE_INFINITE + 5000 && previous_score < VALUE_INFINITE - 5000) {
            // Use aspiration window around previous score
            alpha = std::max(Value(-VALUE_INFINITE), Value(previous_score - aspiration_delta));
            beta = std::min(Value(VALUE_INFINITE), Value(previous_score + aspiration_delta));
        } else {
            // First depth or no previous score - use full window
            alpha = -VALUE_INFINITE;
            beta = VALUE_INFINITE;
            aspiration_delta = 1000;  // Skip aspiration on next iteration
        }

        // Generate moves
        ExtMove moves[MAX_MOVES];
        ExtMove* end = generate<GEN_LEGAL>(pos, moves);

        // Order moves at root for better efficiency
        for (ExtMove* it = moves; it != end; ++it) {
            Move m = it->move;
            int score = 0;

            // CRITICAL: Prioritize previous iteration's best move (PV move)
            // This improves search stability and helps maintain good lines
            if (root_depth > 1 && m == best_move) {
                score = 2000000;  // Highest priority - previous PV move
            }
            // Prioritize winning captures
            else if (m.is_capture()) {
                PieceType captured = pos.piece_type_on(m.to());
                Value cap_value = 0;
                if (captured == PAWN) cap_value = PAWN_VALUE;
                else if (captured == KNIGHT) cap_value = KNIGHT_VALUE;
                else if (captured == BISHOP) cap_value = BISHOP_VALUE;
                else if (captured == ROOK) cap_value = ROOK_VALUE;
                else if (captured == QUEEN) cap_value = QUEEN_VALUE;

                score = 1000000 + int(cap_value);
            }
            // Checks
            else if (!pos.is_check()) {
                PieceType pt = pos.piece_type_on(m.from());
                Square to = m.to();
                Color opponent = Color(int(pos.side_to_move()) ^ 1);
                Square king_sq = pos.king_sq(opponent);

                bool gives_check = false;
                if (pt == KNIGHT) {
                    gives_check = (knight_attacks_bb(to) & square_bb(king_sq)) != 0;
                } else if (pt == BISHOP) {
                    gives_check = (bb_diag_attacks(to, pos.pieces()) & square_bb(king_sq)) != 0;
                } else if (pt == ROOK) {
                    gives_check = ((bb_rank_attacks(to, pos.pieces()) | bb_file_attacks(to, pos.pieces())) & square_bb(king_sq)) != 0;
                } else if (pt == QUEEN) {
                    gives_check = (queen_attacks_bb(to, pos.pieces()) & square_bb(king_sq)) != 0;
                } else if (pt == PAWN) {
                    Bitboard pawn_attacks = 0;
                    Bitboard pb = square_bb(to);
                    if (pos.side_to_move() == WHITE) {
                        if (file_of(to) > FILE_A) pawn_attacks |= shift_nw(pb);
                        if (file_of(to) < FILE_H) pawn_attacks |= shift_ne(pb);
                    } else {
                        if (file_of(to) > FILE_A) pawn_attacks |= shift_sw(pb);
                        if (file_of(to) < FILE_H) pawn_attacks |= shift_se(pb);
                    }
                    gives_check = (pawn_attacks & square_bb(king_sq)) != 0;
                }

                if (gives_check) {
                    score = 500000;
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
            if (aspiration_attempts > 5) {
                alpha = -VALUE_INFINITE;
                beta = VALUE_INFINITE;
            }

            // Search all root moves with current bounds
            int root_moves_searched = 0;
            Value running_alpha = alpha;  // PVS window only, NOT the aspiration bound

            for (ExtMove* it = moves; it != end; ++it) {
                if (stop.load(std::memory_order_relaxed)) break;
                if (check_time()) break;

                if (!pos.do_move(it->move)) continue;

                Value value;
                // PVS: first move gets full window, rest get zero-window scout
                if (root_moves_searched == 0) {
                    value = -search_worker(pos, stack + 1, -beta, -running_alpha,
                                           root_depth - 1, false);
                } else {
                    // Zero-window scout search
                    value = -search_worker(pos, stack + 1, -running_alpha - 1, -running_alpha,
                                           root_depth - 1, true);
                    // Re-search with full window only if scout found something better
                    if (value > running_alpha && value < beta) {
                        value = -search_worker(pos, stack + 1, -beta, -running_alpha,
                                               root_depth - 1, false);
                    }
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
            if (depth_best_value <= alpha) {
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
        }

        // Update previous_score for next depth's aspiration window
        if (depth_best_value > -VALUE_INFINITE + 5000 && depth_best_value < VALUE_INFINITE - 5000) {
            previous_score = depth_best_value;
        }
        root_score = depth_best_value;

        // Calculate elapsed time for NPS
        auto search_end = std::chrono::steady_clock::now();
        int time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(search_end - search_start).count();

        // Send UCI info
        uci_info(pos, root_depth, depth_best_value, nodes.load(), time_ms);
    }

    // Fallback: if best_move is still MOVE_NONE, use the first legal move we found
    if (best_move == MOVE_NONE) {
        if (initial_end > initial_moves) {
            best_move = initial_moves[0].move;
        } else {
            return MOVE_NONE;  // No legal moves
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

    // Print search statistics before returning
    // Debug logging removed - statistics confirmed TT efficiency improved 4-5x with PVS fix

    return best_move;
}

void uci_info([[maybe_unused]] const Position& pos, int depth, Value score, uint64_t node_count, int time_ms) {
    std::cout << "info depth " << depth << " score ";

    // Handle mate scores
    // Note: -VALUE_INFINITE is not a mate score, it means no valid search result
    if (score >= VALUE_MATE_IN_MAX_PLY && score < VALUE_INFINITE) {
        // Mate in N moves (convert plies to moves)
        int mate_in = (VALUE_MATE - score + 1) / 2;
        std::cout << "mate " << mate_in;
    } else if (score <= -VALUE_MATE_IN_MAX_PLY && score > -VALUE_INFINITE) {
        // Being mated in N moves
        int mate_in = -(VALUE_MATE + score + 1) / 2;
        std::cout << "mate " << mate_in;
    } else {
        // Normal score in centipawns
        // Score is already from side-to-move's perspective (evaluate() returns side-to-move relative)
        int score_cp = score * 100 / PAWN_VALUE;
        std::cout << "cp " << score_cp;
    }

    std::cout << " nodes " << node_count
              << " nps " << (time_ms > 0 ? node_count * 1000 / time_ms : 0)
              << "\n";
    std::cout.flush();
}

} // namespace luminex
