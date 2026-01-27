#include "luminex.h"
#include <algorithm>

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

// Reduction constants
constexpr int futility_margin(int depth, bool improving) {
    return depth * (150 + improving * 50);
}

// Check time
void check_time() {
    if (limits.nodes && nodes >= uint64_t(limits.nodes)) {
        stop = true;
    }
}

// Quiescence search
Value qsearch(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth) {
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

    // Generate and search captures
    ExtMove moves[MAX_MOVES];
    ExtMove* end = generate<GEN_CAPTURE>(pos, moves);

    // Sort by MVV-LVA (simplified)
    std::sort(moves, end, [&pos](const ExtMove& a, const ExtMove& b) {
        return mvv_lva(pos.piece_type_on(b.move.to()), KNIGHT) >
               mvv_lva(pos.piece_type_on(a.move.to()), KNIGHT);
    });

    for (ExtMove* it = moves; it != end; ++it) {
        if (!pos.legal(it->move)) continue;

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
    if (stop.load(std::memory_order_relaxed)) {
        return VALUE_ZERO;
    }

    // Check for draw
    if (pos.is_draw()) {
        return VALUE_DRAW;
    }

    ++nodes;

    const bool pv_node = (beta - alpha > 1);
    const bool root_node = (ss->ply == 0);

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

    // Futility pruning
    if (!pv_node && !pos.is_check() && depth <= 3 && eval - futility_margin(depth, false) >= beta) {
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

    // Move ordering
    // 1. TT move first
    // 2. Captures
    // 3. Quiet moves

    if (tt_move && root_node) {
        std::rotate(moves, std::find_if(moves, end, [tt_move](const ExtMove& em) {
            return em.move == tt_move;
        }), end);
    }

    std::sort(moves, end, [&pos, tt_move](const ExtMove& a, const ExtMove& b) {
        if (a.move == tt_move) return true;
        if (b.move == tt_move) return false;

        bool a_capture = pos.capture_or_promotion(a.move);
        bool b_capture = pos.capture_or_promotion(b.move);

        if (a_capture && !b_capture) return true;
        if (!a_capture && b_capture) return false;

        return a.move.raw() < b.move.raw();
    });

    Value best_value = -VALUE_INFINITE;

    for (ExtMove* it = moves; it != end; ++it) {
        pos.do_move(it->move);

        Value value = -search_worker(pos, ss + 1, -beta, -alpha, depth - 1, !cut_node);

        pos.undo_move(it->move);

        if (value > best_value) {
            best_value = value;

            if (value > alpha) {
                alpha = value;

                // Update PV
                ss->pv[ss->ply] = it->move;
                ss->pv[ss->ply + 1] = MOVE_NONE;
            }

            if (value >= beta) {
                // Beta cutoff
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

    Move best_move = MOVE_NONE;
    Value best_value = -VALUE_INFINITE;
    root_score = best_value;

    // Initialize search stack
    for (int i = 0; i < MAX_PLY_PLUS_6; ++i) {
        stack[i].ply = i;
        stack[i].current_move = MOVE_NONE;
        stack[i].previous = i > 0 ? &stack[i - 1] : nullptr;
        stack[i].pv[0] = MOVE_NONE;
    }

    TT.new_search();

    // Iterative deepening
    for (root_depth = 1; root_depth <= limits.depth; ++root_depth) {
        // Search at current depth
        ExtMove moves[MAX_MOVES];
        ExtMove* end = generate<GEN_LEGAL>(pos, moves);

        Value alpha = -VALUE_INFINITE;
        Value beta = VALUE_INFINITE;

        for (ExtMove* it = moves; it != end; ++it) {
            if (stop.load(std::memory_order_relaxed)) break;

            pos.do_move(it->move);
            Value value = -search_worker(pos, stack + 1, -beta, -alpha, root_depth - 1, false);
            pos.undo_move(it->move);

            if (value > best_value) {
                best_value = value;
                best_move = it->move;
                root_score = best_value;
            }

            if (value > alpha) {
                alpha = value;
            }

            if (value >= beta) {
                break; // Beta cutoff
            }
        }

        // Send UCI info
        uci_info(pos, root_depth, best_value, nodes.load(), 0);

        if (stop.load(std::memory_order_relaxed)) break;
    }

    return best_move;
}

void uci_info([[maybe_unused]] const Position& pos, int depth, Value score, uint64_t node_count, int time_ms) {
    int score_cp = score * 100 / PAWN_VALUE;

    std::cout << "info depth " << depth
              << " score " << (score > 0 ? "cp " : "cp ") << score_cp
              << " nodes " << node_count
              << " nps " << (time_ms > 0 ? node_count * 1000 / time_ms : 0)
              << std::endl;
}

} // namespace luminex
