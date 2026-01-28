#include "luminex.h"
#include <algorithm>
#include <chrono>

namespace luminex {

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

// Reduction constants
constexpr int futility_margin(int depth, bool improving) {
    return depth * (150 + improving * 50);
}

// Search start time for time management
static std::chrono::steady_clock::time_point search_start;

// Check time
void check_time() {
    if (limits.nodes && nodes >= uint64_t(limits.nodes)) {
        stop = true;
        return;
    }

    // Check movetime limit
    if (limits.movetime) {
        auto now = std::chrono::steady_clock::now();
        int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - search_start).count();
        if (elapsed >= limits.movetime) {
            stop = true;
        }
    }
}

// Quiescence search
Value qsearch(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth) {
    // Check for max ply to prevent stack overflow
    if (ss->ply >= MAX_PLY) {
        return evaluate(pos);
    }

    // Don't search captures beyond a certain depth
    if (depth < -2) {
        return evaluate(pos);
    }

    if (stop.load(std::memory_order_relaxed)) {
        return VALUE_ZERO;
    }

    ++nodes;

    // Check for draw
    if (pos.is_draw()) {
        return VALUE_DRAW;
    }

    // Evaluate position
    Value eval = evaluate(pos);

    // Stand pat
    if (eval >= beta) {
        return beta;
    }
    if (eval > alpha) {
        alpha = eval;
    }

    // Only generate captures at depth 0 or depth -1
    if (depth < -1) {
        return alpha;
    }

    // Generate and search captures
    ExtMove moves[MAX_MOVES];
    ExtMove* end = generate<GEN_CAPTURE>(pos, moves);

    for (ExtMove* it = moves; it != end; ++it) {
        if (!pos.legal(it->move)) continue;

        ss->current_move = it->move;
        ss->moved_piece = pos.piece_on(it->move.from());

        pos.do_move(it->move);
        Value value = -qsearch(pos, ss + 1, -beta, -alpha, depth - 1);
        pos.undo_move(it->move);

        if (value >= beta) {
            return beta;
        }
        if (value > alpha) {
            alpha = value;
        }
    }

    return alpha;
}

// Main search function (internal worker)
[[maybe_unused]] static Value search_worker(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cut_node) {
    // Check for max ply to prevent stack overflow
    if (ss->ply >= MAX_PLY) {
        return evaluate(pos);
    }

    if (stop.load(std::memory_order_relaxed)) {
        return VALUE_ZERO;
    }

    // Check for draw
    if (pos.is_draw()) {
        return VALUE_DRAW;
    }

    ++nodes;

    const bool pv_node = (beta - alpha > 1);

    // Check time every 1024 nodes
    if ((nodes & 1023) == 0) {
        check_time();
    }

    // Quiescence search at depth 0
    if (depth <= 0) {
        return qsearch(pos, ss, alpha, beta, 0);
    }

    // Transposition table lookup
    bool found;
    TTEntry* tte = TT.probe(pos.key(), found);

    Move tt_move = found ? tte->move() : MOVE_NONE;
    Value tt_value = found ? tte->value() : VALUE_ZERO;
    Depth tt_depth = found ? tte->depth() : DEPTH_ZERO;

    if (!pv_node && found && tt_depth >= depth &&
        (tt_value >= beta ? (tte->bound() & BOUND_LOWER) : (tte->bound() & BOUND_UPPER))) {
        return tt_value;
    }

    // Static evaluation
    Value eval = VALUE_ZERO;
    if (!pos.is_check()) {
        eval = evaluate(pos);
    }
    ss->static_eval = eval;

    // Compute improving flag: position is improving if eval is better than 2 plies ago
    ss->improving = (ss->ply >= 2 && eval > (ss - 2)->static_eval);

    // Futility pruning - use improving for better pruning decisions
    if (!pv_node && !pos.is_check() && depth <= 3 && eval - futility_margin(depth, ss->improving) >= beta) {
        return eval;
    }

    // Null move pruning
    if (!pv_node && !pos.is_check() && depth >= 3 && eval >= beta) {
        pos.do_null_move();

        Value null_value = -search_worker(pos, ss + 1, -beta, -beta + 1, depth - 3, !cut_node);

        pos.undo_null_move();

        if (null_value >= beta) {
            return beta;
        }
    }

    // Generate moves
    ExtMove moves[MAX_MOVES];
    ExtMove* end = generate<GEN_LEGAL>(pos, moves);

    if (moves == end) {
        // No legal moves
        if (pos.is_check()) {
            return -VALUE_MATE + ss->ply;
        }
        return VALUE_DRAW;
    }

    // Score moves for ordering
    for (ExtMove* it = moves; it != end; ++it) {
        Move m = it->move;
        int score = 0;

        // TT move gets highest priority
        if (m == tt_move) {
            score = 1000000;
        }
        // Promotions
        else if (m.is_promotion()) {
            score = 900000 + m.promotion_type() * 10000;
        }
        // Captures - use SEE for better ordering
        else if (m.is_capture()) {
            // Winning captures first, then losing captures
            if (pos.see_ge(m, VALUE_ZERO)) {
                // Winning capture: score by captured piece value
                PieceType captured = pos.piece_type_on(m.to());
                Value cap_value = 0;
                if (captured == PAWN) cap_value = PAWN_VALUE;
                else if (captured == KNIGHT) cap_value = KNIGHT_VALUE;
                else if (captured == BISHOP) cap_value = BISHOP_VALUE;
                else if (captured == ROOK) cap_value = ROOK_VALUE;
                else if (captured == QUEEN) cap_value = QUEEN_VALUE;
                score = 200000 + cap_value;
            } else {
                // Losing capture: lower priority
                score = -100000;
            }
        }
        // Killer moves
        else if (ss->ply < MAX_PLY) {
            if (m == killers[ss->ply][0]) score = 50000;
            else if (m == killers[ss->ply][1]) score = 40000;
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

    for (ExtMove* it = moves; it != end; ++it) {
        Move m = it->move;

        // Capture pruning: skip losing captures at low depths
        if (!pv_node && depth <= 4 && m.is_capture() && !m.is_promotion()) {
            // Skip losing captures (negative SEE) when not at root
            if (ss->ply > 0 && !pos.see_ge(m, VALUE_ZERO)) {
                continue;
            }
        }

        // Late move pruning (futility pruning): skip quiet moves that can't improve alpha
        if (!pv_node && depth <= 3 && ss->ply > 0 && !pos.is_check() &&
            !m.is_capture() && !m.is_promotion() && !m.is_castling()) {
            // Futility margin: depth * 100 centipawns
            int margin = depth * 100;

            // Check if move is futile (eval + margin < alpha)
            if (eval + margin < alpha) {
                continue;
            }
        }

        // Late Move Reduction (LMR)
        Depth new_depth = depth - 1;
        bool do_lmr = !pv_node && depth >= 3 && moves_played >= 3 && !m.is_capture() && !m.is_promotion() && !m.is_castling();

        if (do_lmr) {
            // More aggressive reduction formula using improving flag
            // Base reduction + additional reduction for late moves
            int reduction = 1 + (moves_played - 3) / 3;

            // Increase reduction for cut nodes
            if (cut_node) reduction += 1;

            // Increase reduction at higher depths
            if (depth >= 6) reduction += 1;

            // Reduce more when position is not improving
            if (!ss->improving) reduction += 1;

            // Limit reduction
            if (reduction > depth / 2) reduction = depth / 2;
            if (reduction > 4) reduction = 4;  // Maximum reduction

            new_depth = depth - 1 - reduction;
            if (new_depth < 1) new_depth = 1;
        }

        // Store current move and moved piece for counter-move history
        ss->current_move = m;
        ss->moved_piece = pos.piece_on(m.from());

        pos.do_move(m);

        Value value;
        if (do_lmr && new_depth > 0) {
            // Search with reduced depth first
            value = -search_worker(pos, ss + 1, -alpha - 1, -alpha, new_depth, !cut_node);
            if (value > alpha) {
                // Failed high, re-search at full depth
                value = -search_worker(pos, ss + 1, -beta, -alpha, depth - 1, !cut_node);
            }
        } else if (!pv_node && moves_played > 0 && depth >= 3 && best_value > -VALUE_MATE_IN_MAX_PLY) {
            // Late moves with narrow window
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

                        // Update counter-move history
                        if (ss->ply >= 1 && (ss - 1)->current_move != MOVE_NONE && (ss - 1)->moved_piece != NO_PIECE) {
                            Move prev_move = (ss - 1)->current_move;
                            Piece prev_pc = (ss - 1)->moved_piece;
                            counter_moves[int(prev_pc)][int(prev_move.to())][int(pc)][int(m.to())] += depth * depth;
                        }
                    }
                }
                tte->save(pos.key(), value, false, BOUND_LOWER, depth, it->move, eval, TT.generation());
                return beta;
            }
        }
    }

    // Save to TT
    Bound bound = best_value >= beta ? BOUND_LOWER : BOUND_UPPER;
    tte->save(pos.key(), best_value, pv_node, bound, depth, ss->pv[0], eval, TT.generation());

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
    ExtMove test_moves[MAX_MOVES];
    ExtMove* test_end = generate<GEN_LEGAL>(pos, test_moves);
    if (test_moves == test_end) {
        // No legal moves for us - we are checkmated or stalemated
        if (pos.is_check()) {
            // We are checkmated
            std::cout << "info depth 1 score mate 0 nodes 0 nps 0" << std::endl;
        } else {
            // We are stalemated
            std::cout << "info depth 1 score cp 0 nodes 0 nps 0" << std::endl;
        }
        return MOVE_NONE;  // No move to make
    }

    // Initialize search stack
    for (int i = 0; i < MAX_PLY_PLUS_6; ++i) {
        stack[i].ply = i;
        stack[i].current_move = MOVE_NONE;
        stack[i].moved_piece = NO_PIECE;
        stack[i].previous = i > 0 ? &stack[i - 1] : nullptr;
        stack[i].pv[0] = MOVE_NONE;
    }

    TT.new_search();

    // Initialize killers and history (clear for new search)
    for (int i = 0; i < MAX_PLY; ++i) {
        killers[i][0] = MOVE_NONE;
        killers[i][1] = MOVE_NONE;
    }
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 64; ++j) {
            history[i][j] = 0;
        }
    }
    // Initialize counter-move history
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 64; ++j) {
            for (int k = 0; k < 12; ++k) {
                for (int l = 0; l < 64; ++l) {
                    counter_moves[i][j][k][l] = 0;
                }
            }
        }
    }

    // Iterative deepening
    for (root_depth = 1; root_depth <= limits.depth; ++root_depth) {
        // Check time before starting a new depth (for movetime)
        check_time();
        if (stop.load(std::memory_order_relaxed)) break;

        // Aspiration window: use narrow window after depth 4
        Value alpha, beta;
        Value delta = Value(PAWN_VALUE);  // Initial window size (one pawn)

        if (root_depth >= 5 && best_value > -VALUE_MATE_IN_MAX_PLY && best_value < VALUE_MATE_IN_MAX_PLY) {
            alpha = best_value - delta;
            beta = best_value + delta;
        } else {
            alpha = -VALUE_INFINITE;
            beta = VALUE_INFINITE;
        }

        // Re-search with widening window until we get a result inside the window
        while (true) {
            Value depth_best_value = -VALUE_INFINITE;
            Move depth_best_move = MOVE_NONE;

            // Generate moves
            ExtMove moves[MAX_MOVES];
            ExtMove* end = generate<GEN_LEGAL>(pos, moves);

            for (ExtMove* it = moves; it != end; ++it) {
                if (stop.load(std::memory_order_relaxed)) {
                    break;
                }

                pos.do_move(it->move);
                Value value = -search_worker(pos, stack + 1, -beta, -alpha, root_depth - 1, false);
                pos.undo_move(it->move);

                if (value > depth_best_value) {
                    depth_best_value = value;
                    depth_best_move = it->move;
                }

                if (value > alpha) {
                    alpha = value;
                }

                if (value >= beta) {
                    break; // Beta cutoff
                }
            }

            if (stop.load(std::memory_order_relaxed)) break;

            // Check if we failed high or low and need to re-search
            if (depth_best_value >= beta) {
                // Failed high - widen window and re-search
                beta += delta;
                delta *= 2;
                if (beta > VALUE_INFINITE) beta = VALUE_INFINITE;
                continue;  // Re-search
            } else if (depth_best_value <= alpha) {
                // This shouldn't happen with the current structure, but handle it
                // Actually, we update alpha during search, so this check needs to be against original bounds
                // For now, just continue normally
            }

            // Search completed successfully
            // Update overall best with this depth's result
            // Use >= so we update best_move even when score stays same (e.g., mate in 1)
            if (depth_best_value >= best_value) {
                best_value = depth_best_value;
                best_move = depth_best_move;
            }
            root_score = depth_best_value;

            // Calculate elapsed time for NPS
            auto search_end = std::chrono::steady_clock::now();
            int time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(search_end - search_start).count();

            // Send UCI info
            uci_info(pos, root_depth, depth_best_value, nodes.load(), time_ms);

            break;  // Done with this depth
        }

        if (stop.load(std::memory_order_relaxed)) break;
    }

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
        int score_cp = score * 100 / PAWN_VALUE;
        std::cout << "cp " << score_cp;
    }

    std::cout << " nodes " << node_count
              << " nps " << (time_ms > 0 ? node_count * 1000 / time_ms : 0)
              << std::endl;
}

} // namespace luminex
