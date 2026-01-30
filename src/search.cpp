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
    // More aggressive futility margins for deeper search
    // Based on Stockfish and other strong engines
    int base = 100;
    if (depth == 1) base = 120;
    else if (depth == 2) base = 180;
    else if (depth == 3) base = 250;
    else base = 320;  // depth >= 4

    // Adjust based on improving flag
    if (improving) base -= 40;
    else base += 80;

    return base * depth;
}

// LMR reduction computation
inline int lmr_reduction(int depth, int moves_played, bool improving, bool pv_node) {
    // More aggressive LMR for non-PV nodes and late moves
    int reduction = 2 + (moves_played - 3) / 3;  // More aggressive base

    if (!pv_node) reduction += 2;
    if (!improving) reduction += 1;
    if (depth >= 8) reduction += depth / 4;

    // Limit reduction to prevent over-reduction
    if (reduction > depth - 2) reduction = depth - 2;
    if (reduction < 1) reduction = 1;

    return reduction;
}

// Search start time for time management
static std::chrono::steady_clock::time_point search_start;
static int ideal_time = 0;  // Ideal time to use for this search
static int max_time = 0;    // Maximum time to use

// Check time - returns true if time limit exceeded
bool check_time() {
    if (limits.nodes && nodes >= uint64_t(limits.nodes)) {
        stop = true;
        return true;
    }

    // Check movetime limit - use buffer to avoid exceeding time
    if (limits.movetime) {
        auto now = std::chrono::steady_clock::now();
        int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - search_start).count();
        // Use more aggressive buffer for better performance
        int limit = (limits.movetime * 9) / 10;
        if (elapsed >= limit) {
            stop = true;
            return true;
        }
    }

    // Check tournament time control (wtime/btime)
    if (limits.use_time_management()) {
        auto now = std::chrono::steady_clock::now();
        int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - search_start).count();

        // Stop if we've used max time (hard limit)
        if (elapsed >= max_time) {
            stop = true;
            return true;
        }

        // Consider stopping if we've used ideal time and depth is sufficient
        if (elapsed >= ideal_time && root_depth >= 6 && !stop) {
            // Check if we can safely stop (score is stable, not in tactical position)
            // Use more time in complex positions (low root_depth or high score changes)
            static Value last_root_score = -VALUE_INFINITE;
            static int stable_depth_count = 0;

            if (root_score == last_root_score || root_depth >= 12) {
                stable_depth_count++;
            } else {
                stable_depth_count = 0;
            }
            last_root_score = root_score;

            // Stop if score is stable for 3+ depths and we're past depth 9
            // Or if we've used most of max_time (90%)
            if ((stable_depth_count >= 3 && root_depth >= 9) || elapsed >= max_time * 9 / 10) {
                stop = true;
                return true;
            }
        }
    }

    return false;
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

    // Check time every 64 nodes
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

        // Skip losing captures with negative SEE (but keep queen promotions)
        if (!it->move.is_promotion() && !pos.see_ge(it->move, VALUE_ZERO)) {
            continue;
        }

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
        // Apply contempt: if positive, prefer avoiding draws
        // From root perspective, positive contempt means "I want to avoid draws"
        // So we return VALUE_DRAW - contempt/2 for root player
        return VALUE_DRAW - (pos.side_to_move() == WHITE ? params.contempt / 2 : -params.contempt / 2);
    }

    ++nodes;

    const bool pv_node = (beta - alpha > 1);

    // Check time every 64 nodes for better time control
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

    // Internal iterative deepening: if we don't have a TT move and depth is high enough,
    // do a shallow search to find a good move for move ordering
    if (tt_move == MOVE_NONE && depth >= 4 && !pv_node && !pos.is_check()) {
        // Search at reduced depth to get a move for ordering
        search_worker(pos, ss, alpha, beta, depth - 2, cut_node);
        // Re-probe TT to get the move from the shallow search
        tte = TT.probe(pos.key(), found);
        if (found) {
            tt_move = tte->move();
        }
    }

    // Futility pruning - use improving for better pruning decisions
    if (!pv_node && !pos.is_check() && depth <= 3 && eval - futility_margin(depth, ss->improving) >= beta) {
        return eval;
    }

    // Razoring: at very low depths, if eval is far below alpha, try qsearch to confirm
    if (!pv_node && !pos.is_check() && depth <= 2) {
        Value razor_margin = 300 + depth * 50;  // Margin increases with depth
        if (eval + razor_margin < alpha) {
            // Try quiescence search to confirm the position is really losing
            Value qsearch_value = qsearch(pos, ss, alpha - 1, alpha, 0);
            if (qsearch_value <= alpha) {
                return qsearch_value;  // Confirmed losing, prune
            }
        }
    }

    // Null move pruning - skip if we're in check, or if we have few pieces, or if eval is much worse than beta
    int piece_count = popcount(pos.pieces()) - popcount(pos.pieces(PAWN)) - 2;  // Exclude kings and pawns
    bool null_move_ok = !pv_node && !pos.is_check() && depth >= 3 && piece_count >= 3 &&
                          eval >= beta && ss->ply >= 1;  // Don't do at root

    // Additional check: don't use null move if we have very few pieces (potential zugzwang)
    if (null_move_ok && piece_count < 4) {
        null_move_ok = false;
    }

    if (null_move_ok) {
        pos.do_null_move();

        Value null_value = -search_worker(pos, ss + 1, -beta, -beta + 1, depth - 3, !cut_node);

        pos.undo_null_move();

        if (null_value >= beta) {
            // Don't return mate scores from null move pruning
            if (null_value >= VALUE_MATE_IN_MAX_PLY) {
                null_value = beta;
            }
            return null_value;
        }
    }

    // ProbCut: if beta is high, try a shallow search with reduced threshold
    // If we can't beat beta - margin with shallow search, we can prune
    if (!pv_node && depth >= 5 && !pos.is_check() && ss->ply >= 2) {
        Value probcut_beta = beta + 200;  // Threshold margin
        if (eval >= probcut_beta) {
            // Try captures that might beat the threshold
            ExtMove probcut_moves[MAX_MOVES];
            ExtMove* probcut_end = generate<GEN_LEGAL>(pos, probcut_moves);

            for (ExtMove* it = probcut_moves; it != probcut_end; ++it) {
                Move m = it->move;
                if (m.is_capture()) {
                    PieceType captured = pos.piece_type_on(m.to());
                    if (captured == PT_NONE) continue;

                    // Quick SEE check for promising captures
                    if (!pos.see_ge(m, Value(probcut_beta - eval))) continue;

                    pos.do_move(m);
                    Value value = -search_worker(pos, ss + 1, -probcut_beta, -probcut_beta + 1, depth - 4, !cut_node);
                    pos.undo_move(m);

                    if (value >= probcut_beta) {
                        return value;
                    }
                }
            }
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
        // Captures - order by captured piece value
        else if (m.is_capture()) {
            PieceType captured = pos.piece_type_on(m.to());
            Value cap_value = 0;
            if (captured == PAWN) cap_value = PAWN_VALUE;
            else if (captured == KNIGHT) cap_value = KNIGHT_VALUE;
            else if (captured == BISHOP) cap_value = BISHOP_VALUE;
            else if (captured == ROOK) cap_value = ROOK_VALUE;
            else if (captured == QUEEN) cap_value = QUEEN_VALUE;

            // Check if this is a winning capture with SEE
            bool good_capture = pos.see_ge(m, VALUE_ZERO);

            if (good_capture) {
                score = 1000000 + int(cap_value);
                // Bonus for capturing with a less valuable piece
                PieceType mover = pos.piece_type_on(m.from());
                if (int(captured) > int(mover) && mover != PT_NONE) {
                    score += 50000;
                }
            } else {
                score = -100000 + int(cap_value);  // Losing captures get lower priority
            }
        }
        // Checks - prioritize moves that give check (useful for tactical lines)
        else if (ss->ply < MAX_PLY && !pos.is_check()) {
            // Check if this move gives check without doing a full do_move
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
                // Pawn checks: pawn attacks the king from its destination square
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
                score = 800000;  // Checks get high priority
            } else {
                // Killer moves
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
        }
        // Killer moves
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

    // Singular extension: check if TT move is singular (much better than all alternatives)
    bool tt_move_is_singular = false;
    if (depth >= 6 && pv_node && tt_move != MOVE_NONE && found &&
        (tte->bound() & BOUND_LOWER) && tt_depth >= depth - 3) {
        Value tt_value = tte->value();
        // Try to refute the TT move by searching other moves at reduced depth
        Value singular_beta = tt_value - 50;  // Threshold for singularity
        bool failed_high = false;

        for (ExtMove* it = moves; it != end; ++it) {
            if (it->move == tt_move) continue;

            Move m = it->move;
            // Skip illegal moves - do_move will handle this but we can skip obviously bad ones
            if (!pos.pseudo_legal(m)) continue;

            pos.do_move(m);
            Value value = -search_worker(pos, ss + 1, -singular_beta - 1, -singular_beta, depth / 2 - 2, !cut_node);
            pos.undo_move(m);

            if (value > singular_beta) {
                failed_high = true;  // Found a refutation, TT move is not singular
                break;
            }
        }

        if (!failed_high) {
            tt_move_is_singular = true;  // TT move is singular
        }
    }

    for (ExtMove* it = moves; it != end; ++it) {
        // Check time very frequently during move loop
        if ((moves_played & 1) == 0) {
            check_time();
            if (stop.load(std::memory_order_relaxed)) break;
        }
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

        // Check extension: extend by one ply if move gives check (helps find tactical sequences)
        bool gives_check = false;
        if (depth >= 2 && !pos.is_check()) {
            PieceType pt = pos.piece_type_on(m.from());
            Square to = m.to();
            Color opponent = Color(int(pos.side_to_move()) ^ 1);
            Square king_sq = pos.king_sq(opponent);

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
        }

        // Single extension: at most one extension per move to prevent explosion
        bool extension = false;

        // Singular extension: highest priority
        if (m == tt_move && tt_move_is_singular) {
            extension = true;
        }
        // Check extension (only if not singular extended)
        else if (gives_check && depth >= 3) {
            extension = true;
        }

        if (extension && depth < 8) {
            new_depth++;
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
            std::cout.flush();
        } else {
            // We are stalemated
            std::cout << "info depth 1 score cp 0 nodes 0 nps 0" << std::endl;
            std::cout.flush();
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

    // Initialize time management for tournament time controls
    if (limits.use_time_management()) {
        Color us = pos.side_to_move();
        int time_left = limits.time[int(us)];
        int time_inc = limits.inc[int(us)];

        // Use a fraction of remaining time based on game phase
        // In middle game (more pieces), use more time
        // In endgame (fewer pieces), use less time per move
        int piece_count = popcount(pos.pieces());
        double time_fraction = 0.10;  // Base: 10% of remaining time

        if (piece_count > 28) {
            time_fraction = 0.15;  // Opening: more time for important decisions
        } else if (piece_count < 16) {
            time_fraction = 0.08;  // Endgame: less time needed (simpler positions)
        }

        ideal_time = int(time_left * time_fraction) + time_inc;
        max_time = int(time_left * 0.75);  // Never use more than 75% at once

        // Minimum time to ensure some thinking
        if (ideal_time < 400) ideal_time = 400;
        // Maximum time to avoid time forfeits
        if (max_time > time_left - 3000) max_time = time_left - 3000;
        if (ideal_time > max_time) ideal_time = max_time;
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
    for (root_depth = 1; limits.depth == 0 || root_depth <= limits.depth; ++root_depth) {
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

            // Order moves at root for better efficiency
            for (ExtMove* it = moves; it != end; ++it) {
                Move m = it->move;
                int score = 0;

                // Prioritize winning captures
                if (m.is_capture()) {
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

            // Sort moves by score
            std::sort(moves, end, [](const ExtMove& a, const ExtMove& b) {
                return a.value > b.value;
            });

            for (ExtMove* it = moves; it != end; ++it) {
                if (stop.load(std::memory_order_relaxed)) {
                    break;
                }

                // Check time before each root move (especially for movetime)
                if (check_time()) {
                    break;  // Time limit exceeded
                }

                pos.do_move(it->move);
                Value value = -search_worker(pos, stack + 1, -beta, -alpha, root_depth - 1, false);
                pos.undo_move(it->move);

                // Check time after each root move
                if (check_time()) {
                    break;  // Time limit exceeded
                }

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

            // Check time after all moves searched (before re-search or next depth)
            check_time();
            if (stop.load(std::memory_order_relaxed)) break;

            // Check if we failed high or low and need to re-search
            Value original_alpha = alpha - delta + Value(PAWN_VALUE);  // Reconstruct original alpha
            if (depth_best_value >= beta) {
                // Failed high - widen window and re-search
                beta += delta;
                delta *= 2;
                if (beta > VALUE_INFINITE) beta = VALUE_INFINITE;
                continue;  // Re-search
            } else if (depth_best_value <= original_alpha) {
                // Failed low - widen window downward and re-search
                alpha -= delta;
                delta *= 2;
                if (alpha < -VALUE_INFINITE) alpha = -VALUE_INFINITE;
                continue;  // Re-search
            }

            // Search completed successfully
            // Update overall best with this depth's result
            // Only update if we found a valid move (not MOVE_NONE)
            if (depth_best_move != MOVE_NONE && depth_best_value >= best_value) {
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

    // Fallback: if best_move is still MOVE_NONE but we have legal moves, pick the first one
    if (best_move == MOVE_NONE) {
        ExtMove moves[MAX_MOVES];
        ExtMove* end = generate<GEN_LEGAL>(pos, moves);
        if (moves != end) {
            best_move = moves[0].move;  // Use first legal move as fallback
        }
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
              << "\n";
    std::cout.flush();
}

} // namespace luminex
