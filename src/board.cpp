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
    if (pc == NO_PIECE) {
        return;  // Safety: don't corrupt state if source is empty
    }
    Color c = Color(pc / 6);
    PieceType pt = piece_type_of(pc);

    pieces_by_color[c] ^= square_bb(from) | square_bb(to);
    pieces_by_type[pt] ^= square_bb(from) | square_bb(to);

    board[from] = NO_PIECE;
    board[to] = pc;

    int idx = index[from];
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

    snipers |= (pieces(~side_to_move_, BISHOP, QUEEN) & BB_DIAGONAL_A1H8) & line_bb(ksq, Square(ksq ^ 56));
    snipers |= (pieces(~side_to_move_, BISHOP, QUEEN) & BB_DIAGONAL_H1A8) & line_bb(ksq, ksq);
    snipers |= (pieces(~side_to_move_, ROOK, QUEEN) & rank_bb(rank_of(ksq))) & line_bb(ksq, ksq);
    snipers |= (pieces(~side_to_move_, ROOK, QUEEN) & file_bb(file_of(ksq))) & line_bb(ksq, ksq);

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
    si->block_checkers = slider_blockers(pieces(~side_to_move_), si->pinned);

    // Pawn checks
    Bitboard pawns = pieces(~side_to_move_, PAWN);
    if (pawns) {
        Bitboard pawn_checks = pawn_attacks_bb(~side_to_move_, ksq) & pawns;
        si->checkers |= pawn_checks;
    }

    // Knight checks
    Bitboard knights = pieces(~side_to_move_, KNIGHT);
    if (knights) {
        Bitboard knight_checks = knight_attacks_bb(ksq) & knights;
        si->checkers |= knight_checks;
    }

    // Bishop/Queen diagonal checks
    Color them = ~side_to_move_;
    Bitboard bishop_queens = pieces(them, BISHOP, QUEEN);
    if (bishop_queens) {
        Bitboard diag_attacks = bb_diag_attacks(ksq, pieces());
        si->checkers |= diag_attacks & bishop_queens;
    }

    // Rook/Queen straight checks
    Bitboard rook_queens = pieces(~side_to_move_, ROOK, QUEEN);
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
    Color us = Color(pc / 6);
    Color them = Color(us ^ 1);  // Switch color by XORing with 1
    PieceType pt = piece_type_of(pc);

    // Increment state stack index - check bounds to prevent overflow
    if (st_ply >= MAX_STATES - 1) {
        // State stack full - should not happen in normal search
    } else {
        st_ply++;
    }
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

    st_ = &next_st;
    ++game_ply_;

    // Handle capture
    PieceType captured = piece_type_on(to);
    if (captured != PT_NONE) {
        remove_piece(to);
        st_->key ^= Zobrist::psq[int(them)][int(captured)][int(to)];
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
    Square from = m.from();
    Square to = m.to();
    Color us = Color(side_to_move_ ^ 1);  // side that made the move (opposite of current side)
    Color them = side_to_move_;

    // Restore side to move
    side_to_move_ = us;

    // Handle promotion
    if (m.is_promotion()) {
        remove_piece(to);
        put_piece(us, PAWN, from);
    } else {
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
        put_piece(them, PAWN, cap_sq);
    }

    // Handle capture - only restore if captured_piece is valid (not zero and not PT_NONE)
    PieceType captured = st_->captured_piece;
    if (captured != PT_NONE && captured != PieceType(0) && !m.is_en_passant()) {
        put_piece(them, captured, to);
    }

    // Decrement state stack index and restore state pointer
    st_ply = std::max(st_ply - 1, 0);
    st_ = &state_stack[st_ply];
    --game_ply_;
}

void Position::do_null_move() {
    // Increment state stack index - check bounds to prevent overflow
    if (st_ply >= MAX_STATES - 1) {
        // State stack full - should not happen in normal search
    } else {
        st_ply++;
    }
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
    st_ply = std::max(st_ply - 1, 0);
    st_ = &state_stack[st_ply];
    side_to_move_ = Color(side_to_move_ ^ 1);
    --game_ply_;
}

bool Position::see_ge([[maybe_unused]] Move m, [[maybe_unused]] Value threshold) const {
    // TODO: Implement SEE
    return true;
}

bool Position::legal(Move m) const {
    // TODO: Implement full legality check
    if (!m) return false;

    Square from = m.from();
    Square to = m.to();

    Color us = side_to_move_;
    Color them = ~us;

    // King cannot be captured
    if (piece_type_on(to) == KING) {
        return false;
    }

    // If our king is not in check, all pseudo-legal moves are legal
    // except castling and en passant which need special handling
    if (!is_check()) {
        if (m.is_castling()) {
            // TODO: Verify castling legality
            return true;
        }
        if (m.is_en_passant()) {
            // TODO: Verify en passant legality
            return true;
        }
        return true;
    }

    // In check - only moves that escape check are legal
    Square ksq = king_square[us];

    // King move
    if (piece_type_on(from) == KING) {
        // King cannot move to attacked square
        Bitboard attacks = attackers_to(to, pieces() ^ square_bb(from));
        return !(attacks & pieces(them));
    }

    // En passant capture can uncover check
    if (m.is_en_passant()) {
        // TODO: Handle en passant
        return true;
    }

    // Normal move must be a blocking move or capture of checking piece
    if (!aligned(from, to, ksq)) {
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
                return Color(piece_on(to) / 6) == ~side_to_move_;
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
        case QUEEN: attacks = queen_attacks_bb(from); break;
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

bool Position::see_gen([[maybe_unused]] Bitboard stmAttackers, [[maybe_unused]] Bitboard occupied) const {
    // TODO: Implement SEE generation
    return true;
}

void init_zobrist() {
    Zobrist::init();
}

Direction Position::pawn_push(Color c) const {
    return c == WHITE ? NORTH : SOUTH;
}

} // namespace luminex
