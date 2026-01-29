#include "luminex.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <sstream>

namespace luminex {

// Zobrist keys
namespace Zobrist {

Key psq[2][8][64]; // piece-square keys
Key en_passant[8]; // en passant file keys
Key castling[4];   // castling rights keys
Key side;          // side to move

void init() {
    // Simple initialization using splitmix64
    uint64_t x = 0x9e3779b97f4a7c15ULL;

    auto next = [&]() {
        x += 0x9e3779b97f4a7c15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    };

    for (int c = 0; c < 2; ++c) {
        for (int pt = 0; pt < 8; ++pt) {
            for (int s = 0; s < 64; ++s) {
                psq[c][pt][s] = next();
            }
        }
    }

    for (int f = 0; f < 8; ++f) {
        en_passant[f] = next();
    }

    for (int i = 0; i < 4; ++i) {
        castling[i] = next();
    }

    side = next();
}

} // namespace Zobrist

// State info allocator
constexpr int STATE_ALLOC_SIZE = 128;

StateInfo* state_stack[STATE_ALLOC_SIZE];
int state_stack_top = 0;

StateInfo* new_state_info() {
    if (state_stack_top > 0) {
        return state_stack[--state_stack_top];
    }
    return new StateInfo();
}

void free_state_info(StateInfo* st) {
    if (state_stack_top < STATE_ALLOC_SIZE) {
        state_stack[state_stack_top++] = st;
    } else {
        delete st;
    }
}

// Position implementation

void Position::set(const std::string& fen) {
    // Clear all arrays
    for (int i = 0; i < SQUARE_NONE; ++i) {
        board[i] = NO_PIECE;
        index[i] = 0;
    }
    for (int i = 0; i < ALL_PIECES; ++i) {
        pieces_by_type[i] = 0;
    }
    for (int i = 0; i < NO_COLOR; ++i) {
        pieces_by_color[i] = 0;
        for (int j = 0; j < ALL_PIECES; ++j) {
            piece_count[i][j] = 0;
        }
        for (int j = 0; j < ALL_PIECES; ++j) {
            for (int k = 0; k < 16; ++k) {
                piece_list[i][j][k] = SQUARE_NONE;
            }
        }
    }

    std::fill_n(castling_rights_, NO_COLOR, 0);

    std::istringstream ss(fen);
    std::string token;

    // Piece placement
    ss >> token;
    int sq = 56; // A8

    for (char c : token) {
        if (c == '/') {
            sq -= 16; // Move to start of next rank
            continue;
        }

        if (std::isdigit(c)) {
            sq += (c - '0'); // Skip empty squares
            continue;
        }

        Color color = islower(c) ? BLACK : WHITE;
        PieceType pt = PT_NONE;

        switch (toupper(c)) {
            case 'P': pt = PAWN; break;
            case 'N': pt = KNIGHT; break;
            case 'B': pt = BISHOP; break;
            case 'R': pt = ROOK; break;
            case 'Q': pt = QUEEN; break;
            case 'K': pt = KING; break;
        }

        put_piece(color, pt, Square(sq));
        sq++;
    }

    // Side to move
    ss >> token;
    side_to_move_ = (token == "w") ? WHITE : BLACK;

    // Castling rights
    ss >> token;
    for (char ch : token) {
        switch (ch) {
            case 'K': castling_rights_[WHITE] |= WHITE_KINGSIDE; break;
            case 'Q': castling_rights_[WHITE] |= WHITE_QUEENSIDE; break;
            case 'k': castling_rights_[BLACK] |= BLACK_KINGSIDE; break;
            case 'q': castling_rights_[BLACK] |= BLACK_QUEENSIDE; break;
        }
    }

    // Initialize castling rook squares
    castling_rook_square_[WHITE][0] = H1;
    castling_rook_square_[WHITE][1] = A1;
    castling_rook_square_[BLACK][0] = H8;
    castling_rook_square_[BLACK][1] = A8;

    castling_path_[WHITE][0] = 0;
    castling_path_[WHITE][1] = 0;
    castling_path_[BLACK][0] = 0;
    castling_path_[BLACK][1] = 0;

    // Initialize state stack
    st_ply = 0;
    st_ = &state_stack[0];
    st_->key = 0;
    st_->ply = 0;
    st_->ep_square = SQUARE_NONE;
    st_->castling_rights = 0;
    st_->checkers = 0;
    st_->pinned = 0;
    st_->block_checkers = 0;
    st_->move = MOVE_NONE;
    st_->captured_piece = PT_NONE;

    // En passant square
    ss >> token;
    st_->ep_square = (token == "-") ? SQUARE_NONE : Square((token[1] - '1') * 8 + (token[0] - 'a'));

    // Game ply
    ss >> token; // halfmove clock (ignored for now)
    ss >> token; // fullmove number (ignored for now)

    st_->castling_rights = 0;
    for (Color c : {WHITE, BLACK}) {
        st_->castling_rights |= castling_rights_[c];
    }

    set_check_info(st_);

    // Compute position key (start with 0, add side key only if BLACK to move)
    Key k = (side_to_move_ == BLACK) ? Zobrist::side : 0;

    Bitboard b = pieces();
    while (b) {
        Square s = pop_lsb(b);
        Piece pc = board[s];
        Color c = Color(pc / 6);
        PieceType pt = piece_type_of(pc);
        k ^= Zobrist::psq[int(c)][int(pt)][int(s)];
    }

    if (st_->ep_square != SQUARE_NONE) {
        k ^= Zobrist::en_passant[file_of(st_->ep_square)];
    }

    for (Color c : {WHITE, BLACK}) {
        for (int i = 0; i < 2; ++i) {
            if (castling_rights_[c] & (WHITE_KINGSIDE << i)) {
                k ^= Zobrist::castling[(c << 1) | i];
            }
        }
    }

    st_->key = k;

    game_ply_ = 0;
}

std::string Position::fen() const {
    std::ostringstream ss;

    // Piece placement
    int empty = 0;
    for (int r = int(RANK_8); r >= int(RANK_1); --r) {
        for (int f = int(FILE_A); f <= int(FILE_H); ++f) {
            Square s = make_square(File(f), Rank(r));
            Piece pc = board[s];

            if (pc == NO_PIECE) {
                ++empty;
            } else {
                if (empty > 0) {
                    ss << empty;
                    empty = 0;
                }
                Color c = Color(pc / 6);
                PieceType pt = piece_type_of(pc);
                ss << piece_char(c, pt);
            }
        }
        if (empty > 0) {
            ss << empty;
            empty = 0;
        }
        if (r > int(RANK_1)) {
            ss << '/';
        }
    }

    // Side to move
    ss << (side_to_move_ == WHITE ? " w " : " b ");

    // Castling rights
    bool any_castle = false;
    if (castling_rights_[WHITE] & WHITE_KINGSIDE) { ss << 'K'; any_castle = true; }
    if (castling_rights_[WHITE] & WHITE_QUEENSIDE) { ss << 'Q'; any_castle = true; }
    if (castling_rights_[BLACK] & BLACK_KINGSIDE) { ss << 'k'; any_castle = true; }
    if (castling_rights_[BLACK] & BLACK_QUEENSIDE) { ss << 'q'; any_castle = true; }
    if (!any_castle) ss << '-';

    // En passant
    ss << ' ' << (st_->ep_square == SQUARE_NONE ? "-" : std::string(1, file_char(file_of(st_->ep_square))) + std::string(1, rank_char(rank_of(st_->ep_square))));

    // Halfmove and fullmove clocks
    ss << " 0 1";

    return ss.str();
}

void Position::put_piece(Color c, PieceType pt, Square s) {
    board[s] = make_piece(c, pt);
    pieces_by_type[pt] |= square_bb(s);
    pieces_by_color[c] |= square_bb(s);

    int count = piece_count[int(c)][int(pt)];
    piece_list[int(c)][int(pt)][count] = s;
    index[s] = count;
    piece_count[int(c)][int(pt)]++;

    if (pt == KING) {
        king_square[int(c)] = s;
    }
}

void Position::remove_piece(Square s) {
    Piece pc = board[s];
    Color c = Color(pc / 6);
    PieceType pt = piece_type_of(pc);

    pieces_by_type[pt] ^= square_bb(s);
    pieces_by_color[c] ^= square_bb(s);

    int& count = piece_count[int(c)][int(pt)];
    int idx = index[s];
    Square last_sq = piece_list[int(c)][int(pt)][count - 1];

    piece_list[int(c)][int(pt)][idx] = last_sq;
    index[last_sq] = idx;
    piece_list[int(c)][int(pt)][count - 1] = SQUARE_NONE;
    --count;

    board[s] = NO_PIECE;
    index[s] = 0;
}

void Position::move_piece(Square from, Square to) {
    Piece pc = board[from];
    if (pc == NO_PIECE) return;
    Color c = Color(pc / 6);
    PieceType pt = piece_type_of(pc);
    if (int(c) > 1) return;
    pieces_by_color[c] ^= square_bb(from) | square_bb(to);
    pieces_by_type[pt] ^= square_bb(from) | square_bb(to);
    board[from] = NO_PIECE;
    board[to] = pc;
    int idx = index[from];
    if (idx == 0) {
        int count = piece_count[int(c)][int(pt)];
        for (int i = 0; i < count; ++i) {
            if (piece_list[int(c)][int(pt)][i] == from) {
                idx = i;
                break;
            }
        }
    }
    piece_list[int(c)][int(pt)][idx] = to;
    index[to] = idx;
    index[from] = 0;
    if (pt == KING) {
        king_square[int(c)] = to;
    }
}

Bitboard Position::attackers_to(Square s) const {
    return attackers_to(s, pieces());
}

Bitboard Position::attackers_to(Square s, Bitboard occupied) const {
    return (pawn_attacks_bb(WHITE, s) & pieces(BLACK, PAWN))
         | (pawn_attacks_bb(BLACK, s) & pieces(WHITE, PAWN))
         | (knight_attacks_bb(s) & pieces(KNIGHT))
         | (king_attacks_bb(s) & pieces(KING))
         | ((bb_rank_attacks(s, occupied) | bb_file_attacks(s, occupied)) & (pieces(ROOK) | pieces(QUEEN)))
         | ((bb_diag_attacks(s, occupied)) & (pieces(BISHOP) | pieces(QUEEN)));
}

Bitboard Position::slider_blockers([[maybe_unused]] Bitboard sliders, Bitboard& pinners) const {
    Bitboard blockers = BB_EMPTY;
    pinners = BB_EMPTY;

    Square ksq = king_square[side_to_move_];

    // Snipers are sliders that attack our king when one piece is removed
    Bitboard snipers = 0;

    snipers |= (pieces(Color(side_to_move_ ^ 1), BISHOP, QUEEN) & BB_DIAGONAL_A1H8) & line_bb(ksq, Square(ksq ^ 56));
    snipers |= (pieces(Color(side_to_move_ ^ 1), BISHOP, QUEEN) & BB_DIAGONAL_H1A8) & line_bb(ksq, ksq);
    snipers |= (pieces(Color(side_to_move_ ^ 1), ROOK, QUEEN) & rank_bb(rank_of(ksq))) & line_bb(ksq, ksq);
    snipers |= (pieces(Color(side_to_move_ ^ 1), ROOK, QUEEN) & file_bb(file_of(ksq))) & line_bb(ksq, ksq);

    Bitboard occupancy = pieces() ^ snipers;

    while (snipers) {
        Square sniper_sq = pop_lsb(snipers);
        Bitboard b = between_bb(ksq, sniper_sq) & occupancy;

        if (b && !more_than_one(b)) {
            blockers |= b;
            if (b & pieces(side_to_move_)) {
                pinners |= square_bb(sniper_sq);
            }
        }
    }

    return blockers;
}

void Position::set_check_info(StateInfo* si) {
    si->checkers = 0;
    si->pinned = 0;
    si->block_checkers = 0;

    Square ksq = king_square[side_to_move_];
    si->block_checkers = slider_blockers(pieces(Color(side_to_move_ ^ 1)), si->pinned);

    // Pawn checks
    Bitboard pawns = pieces(Color(side_to_move_ ^ 1), PAWN);
    if (pawns) {
        Bitboard pawn_checks = pawn_attacks_bb(Color(side_to_move_ ^ 1), ksq) & pawns;
        si->checkers |= pawn_checks;
    }

    // Knight checks
    Bitboard knights = pieces(Color(side_to_move_ ^ 1), KNIGHT);
    if (knights) {
        Bitboard knight_checks = knight_attacks_bb(ksq) & knights;
        si->checkers |= knight_checks;
    }

    // Bishop/Queen diagonal checks
    Color them = Color(side_to_move_ ^ 1);
    Bitboard bishop_queens = pieces(them, BISHOP, QUEEN);
    if (bishop_queens) {
        Bitboard diag_attacks = bb_diag_attacks(ksq, pieces());
        si->checkers |= diag_attacks & bishop_queens;
    }

    // Rook/Queen straight checks
    Bitboard rook_queens = pieces(Color(side_to_move_ ^ 1), ROOK, QUEEN);
    if (rook_queens) {
        Bitboard straight_attacks = (bb_rank_attacks(ksq, pieces()) | bb_file_attacks(ksq, pieces()));
        si->checkers |= straight_attacks & rook_queens;
    }

    // King checks (adjacent kings not possible in legal chess)
}

void Position::do_move(Move m) {
    Square from = m.from();
    Square to = m.to();
    Piece pc = board[from];

    // Flag to track if this move should be executed
    bool valid_move = true;

    if (pc == NO_PIECE) {
        valid_move = false;
    }

    Color us = Color(pc / 6);
    Color them = Color(us ^ 1);  // Switch color by XORing with 1
    PieceType pt = piece_type_of(pc);

    if (int(us) > 1) {
        valid_move = false;
    }

    // ALWAYS increment state stack index and save state (even for invalid moves)
    // This ensures undo_move can run without causing st_ply underflow
    st_ply++;
    StateInfo& next_st = state_stack[st_ply];

    // Save state for undo - copy current state to next slot in state_stack
    next_st.key = st_->key;
    next_st.checkers = st_->checkers;
    next_st.pinned = st_->pinned;
    next_st.block_checkers = st_->block_checkers;
    next_st.ep_square = st_->ep_square;
    next_st.castling_rights = st_->castling_rights;
    next_st.ply = st_->ply;
    next_st.move = m;
    next_st.captured_piece = piece_type_on(to);
    next_st.move_was_executed = valid_move;  // Track if move was actually executed

    st_ = &next_st;
    ++game_ply_;

    // Only execute the move if it's valid
    if (!valid_move) {
        next_st.move_was_executed = false;
        return;
    }

    // Handle capture
    PieceType captured = piece_type_on(to);
    if (captured != PT_NONE) {
        if (captured == KING) {
            // "Capturing" the king means checkmate
            // Remove the king from the board so the checkmate detection works
            remove_piece(to);
            st_->key ^= Zobrist::psq[int(them)][int(KING)][int(to)];
        } else {
            // Normal capture
            remove_piece(to);
            st_->key ^= Zobrist::psq[int(them)][int(captured)][int(to)];
        }
    }

    // Handle promotion
    if (m.is_promotion()) {
        remove_piece(from);
        PieceType promoted = m.promotion_type();
        put_piece(us, promoted, to);
        st_->key ^= Zobrist::psq[int(us)][int(promoted)][int(to)];
    } else {
        // Move the piece
        move_piece(from, to);
    }

    // Handle castling
    if (m.is_castling()) {
        Square rfrom, rto;
        if (to == (us == WHITE ? G1 : G8)) {
            rfrom = us == WHITE ? H1 : H8;
            rto = us == WHITE ? F1 : F8;
        } else {
            rfrom = us == WHITE ? A1 : A8;
            rto = us == WHITE ? D1 : D8;
        }
        move_piece(rfrom, rto);
    }

    // Handle en passant
    if (m.is_en_passant()) {
        Square cap_sq = Square(to - (us == WHITE ? 8 : -8));
        remove_piece(cap_sq);
        st_->key ^= Zobrist::psq[int(them)][int(PAWN)][int(cap_sq)];
    }

    // Update en passant square (check piece type before move)
    st_->ep_square = SQUARE_NONE;
    if (pt == PAWN && std::abs(int(rank_of(to)) - int(rank_of(from))) == 2) {
        st_->ep_square = Square((from + to) / 2);
        st_->key ^= Zobrist::en_passant[file_of(st_->ep_square)];
    }

    // Update castling rights
    // Remove castling rights if rook moves or is captured
    if (pt == KING) {
        st_->castling_rights &= ~(us == WHITE ? (WHITE_KINGSIDE | WHITE_QUEENSIDE) : (BLACK_KINGSIDE | BLACK_QUEENSIDE));
    }
    if (from == (us == WHITE ? H1 : H8) || to == (us == WHITE ? H1 : H8)) {
        st_->castling_rights &= ~(us == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE);
    }
    if (from == (us == WHITE ? A1 : A8) || to == (us == WHITE ? A1 : A8)) {
        st_->castling_rights &= ~(us == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE);
    }

    // Switch side
    st_->key ^= Zobrist::side;
    side_to_move_ = them;

    // Update check info
    set_check_info(st_);
}

void Position::undo_move(Move m) {
    if (st_ply <= 0) {
        return;
    }

    // Check if the move was actually executed (vs. being invalid and returning early)
    if (!st_->move_was_executed) {
        // Move was not executed, just decrement st_ply and restore state
        st_ply--;
        st_ = &state_stack[st_ply];
        --game_ply_;
        return;
    }

    Square from = m.from();
    Square to = m.to();
    Color us = Color(side_to_move_ ^ 1);  // side that made the move (opposite of current side)
    Color them_color = side_to_move_;  // Save this BEFORE changing side_to_move_
    PieceType captured = st_->captured_piece;
    bool is_capture = m.is_capture();

    // Restore side to move FIRST
    side_to_move_ = us;

    // Handle promotion
    if (m.is_promotion()) {
        remove_piece(to);
        put_piece(us, PAWN, from);
    } else {
        // For both captures and non-captures, move the piece back
        move_piece(to, from);
    }

    // Handle castling
    if (m.is_castling()) {
        Square rfrom, rto;
        if (to == (us == WHITE ? G1 : G8)) {
            rfrom = us == WHITE ? H1 : H8;
            rto = us == WHITE ? F1 : F8;
        } else {
            rfrom = us == WHITE ? A1 : A8;
            rto = us == WHITE ? D1 : D8;
        }
        move_piece(rto, rfrom);
    }

    // Handle en passant
    if (m.is_en_passant()) {
        Square cap_sq = Square(to - (us == WHITE ? 8 : -8));
        put_piece(them_color, PAWN, cap_sq);
    }

    // Handle capture - restore the captured piece at 'to'
    // Note: En passant is handled separately above, so skip it here
    // At this point, square 'to' is empty (we moved the piece back to 'from'),
    // so we can safely use put_piece to restore the captured piece
    if (is_capture && !m.is_en_passant() && captured != PT_NONE) {
        put_piece(them_color, captured, to);
    }

    // Decrement state stack index and restore state pointer
    st_ply--;
    st_ = &state_stack[st_ply];
    --game_ply_;
}

void Position::do_null_move() {
    // Increment state stack index
    st_ply++;
    StateInfo& next_st = state_stack[st_ply];

    // Copy current state
    next_st.key = st_->key;
    next_st.checkers = st_->checkers;
    next_st.pinned = st_->pinned;
    next_st.block_checkers = st_->block_checkers;
    next_st.ep_square = st_->ep_square;
    next_st.castling_rights = st_->castling_rights;
    next_st.ply = st_->ply;
    next_st.move = MOVE_NONE;
    next_st.captured_piece = PT_NONE;

    st_ = &next_st;

    if (st_->ep_square != SQUARE_NONE) {
        st_->key ^= Zobrist::en_passant[file_of(st_->ep_square)];
    }

    st_->ep_square = SQUARE_NONE;
    st_->key ^= Zobrist::side;
    side_to_move_ = Color(side_to_move_ ^ 1);

    st_->ply = 0;
    ++game_ply_;
}

void Position::undo_null_move() {
    // Decrement state stack index and restore state pointer
    st_ply--;
    st_ = &state_stack[st_ply];
    side_to_move_ = Color(side_to_move_ ^ 1);
    --game_ply_;
}

bool Position::see_ge(Move m, Value threshold) const {
    // Implement Static Exchange Evaluation (SEE)
    Square from = m.from();
    Square to = m.to();

    // Piece value array (index by PieceType)
    constexpr Value piece_value[] = {
        0,              // PT_NONE
        PAWN_VALUE,     // PAWN
        KNIGHT_VALUE,   // KNIGHT
        BISHOP_VALUE,   // BISHOP
        ROOK_VALUE,     // ROOK
        QUEEN_VALUE,    // QUEEN
        0,              // KING (not used in SEE)
    };

    // Assume the move is made
    Bitboard occupied = pieces();
    occupied &= ~square_bb(from);

    // Get the value of the piece we're capturing (or promoting)
    Value gain = piece_value[piece_type_on(to)];
    if (m.is_promotion()) {
        gain += piece_value[m.promotion_type()] - piece_value[PAWN];
    }

    // If it's an en passant capture, remove the captured pawn
    if (m.is_en_passant()) {
        Square cap_sq = Square(to - (side_to_move_ == WHITE ? 8 : -8));
        occupied &= ~square_bb(cap_sq);
    }

    // Early exit if gain already meets threshold
    if (gain >= threshold) return true;

    // Get attackers for the target square
    Bitboard attackers = attackers_to(to, occupied);

    // Side to move after the initial capture
    Color stm = Color(side_to_move_ ^ 1);

    while (attackers) {
        // Get the least valuable attacker for the current side
        Bitboard stm_attackers = attackers & pieces(stm);
        if (!stm_attackers) break;  // No more attackers for this side

        // Find least valuable piece
        PieceType pt;
        Bitboard pca;
        if ((pca = stm_attackers & pieces(stm, PAWN))) {
            pt = PAWN;
        } else if ((pca = stm_attackers & pieces(stm, KNIGHT))) {
            pt = KNIGHT;
        } else if ((pca = stm_attackers & pieces(stm, BISHOP))) {
            pt = BISHOP;
        } else if ((pca = stm_attackers & pieces(stm, ROOK))) {
            pt = ROOK;
        } else if ((pca = stm_attackers & pieces(stm, QUEEN))) {
            pt = QUEEN;
        } else {
            pt = KING;  // King is the last resort
            pca = stm_attackers & pieces(stm, KING);
        }

        // Remove this attacker from the board
        Square sq = lsb(pca);
        occupied &= ~square_bb(sq);

        // Switch sides
        stm = Color(stm ^ 1);

        // Update gain (subtract value of captured piece, add value of recapture)
        gain = -gain - piece_value[pt];

        // Check if we've met the threshold
        if (gain >= threshold) return true;

        // Update attackers (sliders may have new attacks after piece removal)
        attackers = attackers_to(to, occupied);
    }

    return gain >= threshold;
}

bool Position::legal(Move m) const {
    if (!m) return false;

    Square from = m.from();
    Square to = m.to();

    Color us = side_to_move_;
    Color them = Color(us ^ 1);
    Square ksq = king_square[us];

    // King moves: check destination and distance to opponent king
    if (piece_type_on(from) == KING) {
        Square opp_king = king_square[them];
        if (distance(to, opp_king) <= 1) {
            return false;  // Kings cannot be adjacent
        }
        // King cannot move to attacked square
        Bitboard attacks = attackers_to(to, pieces() ^ square_bb(from));
        if (attacks & pieces(them)) {
            return false;
        }
        // Castling needs special handling
        if (m.is_castling()) {
            if (to == (us == WHITE ? G1 : G8)) {
                // Kingside castling: check f1/g1 squares
                Square s1 = Square((us == WHITE ? F1 : F8));
                Square s2 = Square((us == WHITE ? G1 : G8));
                if ((pieces() & (square_bb(s1) | square_bb(s2))) != 0) return false;
                if ((attackers_to(s1) & pieces(them)) != 0) return false;
                if ((attackers_to(s2) & pieces(them)) != 0) return false;
            } else {
                // Queenside castling: check b1/c1/d1 squares
                Square s1 = Square((us == WHITE ? C1 : C8));
                Square s2 = Square((us == WHITE ? D1 : D8));
                Square s3 = Square((us == WHITE ? B1 : B8));
                if ((pieces() & (square_bb(s1) | square_bb(s2) | square_bb(s3))) != 0) return false;
                if ((attackers_to(s1) & pieces(them)) != 0) return false;
                if ((attackers_to(s2) & pieces(them)) != 0) return false;
            }
        }
        return true;
    }

    // For non-king moves, we need to verify that the move doesn't leave king in check
    // Calculate the occupancy after the move
    Bitboard occ_after = pieces();
    // Use XOR for both: removes piece from 'from', places piece on 'to'
    // If 'to' has an enemy piece (capture), it's also removed - perfect for our check calculation
    occ_after ^= square_bb(from);
    occ_after ^= square_bb(to);

    // For en passant, also remove the captured pawn (different square)
    if (m.is_en_passant()) {
        Square cap_sq = Square(to - (us == WHITE ? 8 : -8));
        occ_after ^= square_bb(cap_sq);
    }

    // Check if king is under attack after the move
    // Slider attacks (queen, rook, bishop)
    Bitboard enemy_sliders = pieces(them, QUEEN) | pieces(them, ROOK) | pieces(them, BISHOP);

    // Check queen attacks (all directions)
    Bitboard queen_attacks = queen_attacks_bb(ksq, occ_after);
    if (queen_attacks & enemy_sliders & pieces(them, QUEEN)) {
        return false;  // Queen would attack our king
    }

    // Check rook attacks (rank/file)
    Bitboard rook_attacks = bb_rank_attacks(ksq, occ_after) | bb_file_attacks(ksq, occ_after);
    if (rook_attacks & (pieces(them, ROOK) | pieces(them, QUEEN))) {
        return false;  // Rook or queen would attack our king
    }

    // Check bishop attacks (diagonals)
    Bitboard bishop_attacks = bb_diag_attacks(ksq, occ_after);
    if (bishop_attacks & (pieces(them, BISHOP) | pieces(them, QUEEN))) {
        return false;  // Bishop or queen would attack our king
    }

    // Check knight attacks
    Bitboard knight_attacks = knight_attacks_bb(ksq);
    if (knight_attacks & pieces(them, KNIGHT)) {
        return false;  // Knight would attack our king
    }

    // Check pawn attacks
    Bitboard enemy_pawns = pieces(them, PAWN);
    Bitboard pawn_attacks = 0;
    while (enemy_pawns) {
        Square psq = pop_lsb(enemy_pawns);
        Bitboard pb = square_bb(psq);
        if (us == WHITE) {
            // Black pawns attack diagonally upward (from white's perspective)
            if (file_of(psq) > FILE_A) pawn_attacks |= shift_nw(pb);
            if (file_of(psq) < FILE_H) pawn_attacks |= shift_ne(pb);
        } else {
            // White pawns attack diagonally downward (from black's perspective)
            if (file_of(psq) > FILE_A) pawn_attacks |= shift_sw(pb);
            if (file_of(psq) < FILE_H) pawn_attacks |= shift_se(pb);
        }
    }
    if (pawn_attacks & square_bb(ksq)) {
        return false;  // Pawn would attack our king
    }

    // If we were in check, verify the move actually escapes
    if (is_check()) {
        Bitboard checkers_b = checkers();

        // If this is en passant, verify it captures the checking piece or resolves check
        if (m.is_en_passant()) {
            Square cap_sq = Square(to - (us == WHITE ? 8 : -8));
            // If the checker is not the captured pawn, en passant doesn't help
            if (!(checkers() & square_bb(cap_sq))) {
                return false;
            }
        }

        // If capturing a checker, that's fine
        if (checkers_b & square_bb(to)) {
            return true;
        }

        // If single checker, verify we block it
        if (popcount(checkers_b) == 1) {
            Square checker_sq = pop_lsb(checkers_b);
            if (between_bb(ksq, checker_sq) & square_bb(to)) {
                return true;  // Valid blocking move
            }
        }

        // Double check can only be resolved by king move
        return false;
    }

    return true;
}

bool Position::pseudo_legal(const Move m) const {
    Square from = m.from();
    Square to = m.to();
    PieceType pt = piece_type_on(from);

    if (pt == PT_NONE) return false;
    if (Color(board[from] / 6) != side_to_move_) return false;
    if (pieces(side_to_move_) & square_bb(to)) return false;

    // Castling
    if (m.is_castling()) {
        if (pt != KING) return false;
        // TODO: Check castling rights
        return true;
    }

    // En passant
    if (m.is_en_passant()) {
        if (pt != PAWN) return false;
        if (to != st_->ep_square) return false;
        return true;
    }

    // Pawn moves
    if (pt == PAWN) {
        Rank r2 = relative_rank(side_to_move_, to);
        Rank r1 = relative_rank(side_to_move_, from);

        if (m.is_promotion()) {
            if (r2 != RANK_8) return false;
        } else if (r2 == RANK_8) {
            return false;
        }

        Direction d = pawn_push(side_to_move_);

        // Single pawn push
        if (to == Square(from + d)) {
            return piece_on(to) == NO_PIECE;
        }

        // Double pawn push
        if (to == Square(from + d * 2)) {
            if (r1 != RANK_2) return false;
            return piece_on(to) == NO_PIECE && piece_on(Square(from + d)) == NO_PIECE;
        }

        // Pawn capture
        if (to == Square(from + d + WEST) || to == Square(from + d + EAST)) {
            if (piece_on(to) != NO_PIECE) {
                return Color(piece_on(to) / 6) == Color(side_to_move_ ^ 1);
            }
            return false;
        }

        return false;
    }

    // Normal piece moves
    Bitboard attacks = BB_EMPTY;
    switch (pt) {
        case KNIGHT: attacks = knight_attacks_bb(from); break;
        case BISHOP: attacks = bb_diag_attacks(from, pieces()); break;
        case ROOK: attacks = bb_rank_attacks(from, pieces()) | bb_file_attacks(from, pieces()); break;
        case QUEEN: attacks = queen_attacks_bb(from, pieces()); break;
        case KING: attacks = king_attacks_bb(from); break;
        default: break;
    }

    return (attacks & square_bb(to)) != 0;
}

bool Position::capture(Move m) const {
    return m.is_capture() || m.is_promotion();
}

bool Position::capture_or_promotion(Move m) const {
    return m.is_capture() || m.is_promotion();
}

// Sliding attack helpers using simple ray casting
Bitboard bb_rank_attacks(Square s, Bitboard occupied) {
    Bitboard attacks = BB_EMPTY;
    Rank r = rank_of(s);
    File f = file_of(s);

    // Scan east
    for (int ff = int(f) + 1; ff <= int(FILE_H); ++ff) {
        Square sq = make_square(File(ff), r);
        attacks |= square_bb(sq);
        if (occupied & square_bb(sq)) break;
    }

    // Scan west
    for (int ff = int(f) - 1; ff >= int(FILE_A); --ff) {
        Square sq = make_square(File(ff), r);
        attacks |= square_bb(sq);
        if (occupied & square_bb(sq)) break;
    }

    return attacks;
}

Bitboard bb_file_attacks(Square s, Bitboard occupied) {
    Bitboard attacks = BB_EMPTY;
    Rank r = rank_of(s);
    File f = file_of(s);

    // Scan north
    for (int rr = int(r) + 1; rr <= int(RANK_8); ++rr) {
        Square sq = make_square(f, Rank(rr));
        attacks |= square_bb(sq);
        if (occupied & square_bb(sq)) break;
    }

    // Scan south
    for (int rr = int(r) - 1; rr >= int(RANK_1); --rr) {
        Square sq = make_square(f, Rank(rr));
        attacks |= square_bb(sq);
        if (occupied & square_bb(sq)) break;
    }

    return attacks;
}

Bitboard bb_diag_attacks(Square s, Bitboard occupied) {
    Bitboard attacks = BB_EMPTY;
    Rank r = rank_of(s);
    File f = file_of(s);

    // Scan northeast
    for (int d = 1; d <= 7; ++d) {
        int ff = int(f) + d;
        int rr = int(r) + d;
        if (ff > int(FILE_H) || rr > int(RANK_8)) break;
        Square sq = make_square(File(ff), Rank(rr));
        attacks |= square_bb(sq);
        if (occupied & square_bb(sq)) break;
    }

    // Scan northwest
    for (int d = 1; d <= 7; ++d) {
        int ff = int(f) - d;
        int rr = int(r) + d;
        if (ff < int(FILE_A) || rr > int(RANK_8)) break;
        Square sq = make_square(File(ff), Rank(rr));
        attacks |= square_bb(sq);
        if (occupied & square_bb(sq)) break;
    }

    // Scan southeast
    for (int d = 1; d <= 7; ++d) {
        int ff = int(f) + d;
        int rr = int(r) - d;
        if (ff > int(FILE_H) || rr < int(RANK_1)) break;
        Square sq = make_square(File(ff), Rank(rr));
        attacks |= square_bb(sq);
        if (occupied & square_bb(sq)) break;
    }

    // Scan southwest
    for (int d = 1; d <= 7; ++d) {
        int ff = int(f) - d;
        int rr = int(r) - d;
        if (ff < int(FILE_A) || rr < int(RANK_1)) break;
        Square sq = make_square(File(ff), Rank(rr));
        attacks |= square_bb(sq);
        if (occupied & square_bb(sq)) break;
    }

    return attacks;
}

bool Position::is_draw() const {
    // TODO: Implement proper draw detection
    return false;
}

void Position::set_castling_right([[maybe_unused]] Color c, [[maybe_unused]] Square rfrom) {
    // TODO: Implement castling right setup
}

void init_zobrist() {
    Zobrist::init();
}

Direction Position::pawn_push(Color c) const {
    return c == WHITE ? NORTH : SOUTH;
}

} // namespace luminex
