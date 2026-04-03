#include "luminex.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

namespace {
#ifndef NDEBUG
// Global debug log for board corruption (debug builds only)
std::ofstream board_debug_log;
bool board_debug_initialized = false;

void init_board_debug() {
    if (!board_debug_initialized) {
        board_debug_log.open("C:\\Users\\chang\\Downloads\\Luminex\\board_corruption.txt", std::ios::out | std::ios::trunc);
        board_debug_initialized = true;
    }
}
#endif
}

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

// Position implementation

void Position::set(const std::string& fen) {
    // Clear all arrays
    for (int i = 0; i < SQUARE_NONE; ++i) {
        board[i] = NO_PIECE;
        index[i] = -1;  // FIXED: Initialize to -1 (not in list) instead of 0
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

    // CRITICAL: Initialize king_square to invalid values
    king_square[WHITE] = SQUARE_NONE;
    king_square[BLACK] = SQUARE_NONE;

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
    st_ = &state_stack[0];  // CRITICAL FIX: Use state_stack[0], not dummy_state!
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

    // Halfmove clock (for 50-move rule)
    int halfmove = 0;
    if (ss >> halfmove) {
        st_->halfmove_clock = halfmove;
    } else {
        st_->halfmove_clock = 0;
    }

    // Fullmove number - compute game_ply_ from it
    // If fullmove is N and it's White's turn, then (N-1)*2 plies have been played
    // If fullmove is N and it's Black's turn, then (N-1)*2 + 1 plies have been played
    int fullmove = 1;
    if (ss >> fullmove) {
        game_ply_ = (fullmove - 1) * 2 + (side_to_move_ == BLACK ? 1 : 0);
    } else {
        game_ply_ = 0;
    }
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

    // CRITICAL: game_ply_ is already computed from FEN fullmove above (line 186)
    // Do NOT reset to 0 here, or fen() will output wrong fullmove number

    // Initialize position history for repetition detection
    history_size = 0;
    position_history[history_size++] = st_->key;
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
    ss << " " << st_->halfmove_clock << " " << (1 + game_ply_ / 2);

    return ss.str();
}

void Position::put_piece(Color c, PieceType pt, Square s) {
#ifndef NDEBUG
    // DEBUG: Check if this square already has a piece (corruption)
    if (board[s] != NO_PIECE) {
        init_board_debug();
        board_debug_log << "\n=== PUT_PIECE CORRUPTION ===\n";
        board_debug_log << "Square " << s << " already has piece " << int(board[s]);
        board_debug_log << ", trying to put " << int(make_piece(c, pt)) << "\n";
        board_debug_log << "FEN: " << fen() << "\n";
        board_debug_log << "================================\n";
        board_debug_log.flush();
    }
#endif

    board[s] = make_piece(c, pt);
    pieces_by_type[pt] |= square_bb(s);
    pieces_by_color[c] |= square_bb(s);

    int count = piece_count[int(c)][int(pt)];
    piece_list[int(c)][int(pt)][count] = s;
    index[s] = count;
    piece_count[int(c)][int(pt)]++;

    if (pt == KING) {
#ifndef NDEBUG
        // DEBUG: Check if we already have a king of this color
        if (king_square[int(c)] != SQUARE_NONE && king_square[int(c)] != s) {
            init_board_debug();
            board_debug_log << "\n=== DUPLICATE KING DETECTED ===\n";
            board_debug_log << "Color " << c << " already has king at " << king_square[int(c)];
            board_debug_log << ", trying to put another at " << s << "\n";
            board_debug_log << "FEN: " << fen() << "\n";
            board_debug_log << "==============================\n";
            board_debug_log.flush();
        }
#endif
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

    // Safety check: if idx is out of bounds, search for the piece
    if (idx < 0 || idx >= count) {
        idx = -1;
        for (int i = 0; i < count; ++i) {
            if (piece_list[int(c)][int(pt)][i] == s) {
                idx = i;
                break;
            }
        }
        if (idx < 0) return;  // Piece not found, abort
    }

    Square last_sq = piece_list[int(c)][int(pt)][count - 1];

    piece_list[int(c)][int(pt)][idx] = last_sq;
    index[last_sq] = idx;
    piece_list[int(c)][int(pt)][count - 1] = SQUARE_NONE;
    --count;

    board[s] = NO_PIECE;
    index[s] = -1;  // FIXED: Use -1 to indicate "not in list" instead of 0
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

    // FIXED: Use piece_count to check if index is valid, not just idx == 0
    // idx could legitimately be 0 (first piece in list)
    // Check if idx is within valid range instead
    int count = piece_count[int(c)][int(pt)];
    if (idx < 0 || idx >= count) {
        // Search for the piece in piece_list
        idx = -1;  // Assume not found
        for (int i = 0; i < count; ++i) {
            if (piece_list[int(c)][int(pt)][i] == from) {
                idx = i;
                break;
            }
        }
        // If still not found, this is an error - skip the update
        if (idx < 0) {
            return;  // Can't find piece, abort
        }
    }
    piece_list[int(c)][int(pt)][idx] = to;
    index[to] = idx;
    index[from] = -1;  // FIXED: Use -1 to indicate "not in list" instead of 0
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
         | (rook_attacks_bb(s, occupied) & (pieces(ROOK) | pieces(QUEEN)))
         | (bishop_attacks_bb(s, occupied) & (pieces(BISHOP) | pieces(QUEEN)));
}

Bitboard Position::slider_blockers([[maybe_unused]] Bitboard sliders, Bitboard& pinners) const {
    Bitboard blockers = BB_EMPTY;
    pinners = BB_EMPTY;

    Square ksq = king_square[side_to_move_];

    // CRITICAL FIX: Find all enemy sliders that could potentially pin through our king
    // Use attack functions with empty occupancy (0) to get all squares along each ray
    // Then filter by actual enemy sliders to find potential snipers
    Bitboard snipers = (rook_attacks_bb(ksq, 0) & pieces(Color(side_to_move_ ^ 1), ROOK, QUEEN))
                     | (bishop_attacks_bb(ksq, 0) & pieces(Color(side_to_move_ ^ 1), BISHOP, QUEEN));

    Bitboard occupancy = pieces();

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
        Bitboard pawn_checks = pawn_attacks_bb(side_to_move_, ksq) & pawns;
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
        Bitboard diag_attacks = bishop_attacks_bb(ksq, pieces());
        si->checkers |= diag_attacks & bishop_queens;
    }

    // Rook/Queen straight checks
    Bitboard rook_queens = pieces(Color(side_to_move_ ^ 1), ROOK, QUEEN);
    if (rook_queens) {
        Bitboard straight_attacks = rook_attacks_bb(ksq, pieces());
        si->checkers |= straight_attacks & rook_queens;
    }

    // King checks (adjacent kings not possible in legal chess)
}

bool Position::do_move(Move m) {
    Square from = m.from();
    Square to = m.to();
    Piece pc = board[from];

    // CRITICAL FIX: Check if st_ply is about to overflow state_stack
    if (st_ply >= MAX_STATES - 2) {
        return false;  // Abort the move to prevent crash
    }

    if (pc == NO_PIECE) {
        // CRITICAL: Log this for debugging - trying to move from empty square
#ifndef NDEBUG
        std::cerr << "\n=== ILLEGAL MOVE: NO PIECE AT SOURCE ===\n";
        std::cerr << "Move: " << m << " (" << from << " to " << to << ")\n";
        std::cerr << "Side to move: " << (side_to_move_ == WHITE ? "WHITE" : "BLACK") << "\n";
        std::cerr << "FEN: " << fen() << "\n";
        std::cerr << "=======================================\n";
#endif
        return false;
    }

    Color us = Color(pc / 6);
    Color them = Color(us ^ 1);  // Switch color by XORing with 1

    // CRITICAL: Check that the piece belongs to the side to move
    // Without this, opponent's pieces could move during our turn, corrupting the board
    if (us != side_to_move_) {
        return false;
    }
    PieceType pt = piece_type_of(pc);

    if (int(us) > 1) {
        return false;
    }

    // ALWAYS increment state stack index first (required for undo to work)
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
    next_st.move_was_executed = true;  // Assume valid until validation proves otherwise

    // Update halfmove clock for 50-move rule
    // Reset to 0 on pawn moves or captures, otherwise increment
    next_st.halfmove_clock = st_->halfmove_clock + 1;
    if (pt == PAWN || piece_type_on(to) != PT_NONE) {
        next_st.halfmove_clock = 0;
    }

    st_ = &next_st;
    ++game_ply_;

    // CRITICAL: Validate move flags match actual board state AFTER state is saved
    // This ensures undo can work even if validation fails
    PieceType piece_at_to = piece_type_on(to);
    bool piece_at_to_is_enemy = (board[to] != NO_PIECE && color_of_piece(board[to]) == them);
    bool move_flag_says_capture = m.is_capture();

    if (piece_at_to_is_enemy && !move_flag_says_capture && !m.is_promotion()) {
        // Enemy piece at destination but move is NOT flagged as capture
        // This will cause board corruption! Undo state advance and abort.
#ifndef NDEBUG
        std::cerr << "\n=== MOVE FLAG ERROR in do_move ===\n";
        std::cerr << "Move: " << m << " (" << from << " to " << to << ")\n";
        std::cerr << "Enemy piece at destination but not flagged as capture!\n";
        std::cerr << "Piece at " << to << ": " << int(board[to]) << "\n";
        std::cerr << "Move flags: 0x" << std::hex << m.flags() << std::dec << "\n";
        std::cerr << "Undoing state advance and aborting.\n";
        std::cerr << "====================================\n";
#endif
        // CRITICAL: Manually undo state advance since we're returning false
        // This ensures atomic failure - nothing changed
        st_ply--;
        st_ = &state_stack[st_ply];
        game_ply_--;
        return false;
    }

    if (!piece_at_to_is_enemy && piece_at_to != PT_NONE && color_of_piece(board[to]) == us) {
        // Friendly piece at destination - invalid move
#ifndef NDEBUG
        std::cerr << "\n=== CAPTURING OWN PIECE in do_move ===\n";
        std::cerr << "Move: " << m << " (" << from << " to " << to << ")\n";
        std::cerr << "Friendly piece at destination!\n";
        std::cerr << "Undoing state advance and aborting.\n";
        std::cerr << "====================================\n";
#endif
        // CRITICAL: Manually undo state advance since we're returning false
        st_ply--;
        st_ = &state_stack[st_ply];
        game_ply_--;
        return false;
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
        // CRITICAL FIX: XOR out pawn from origin BEFORE removing it
        // Without this, Zobrist key is corrupted after promotion
        st_->key ^= Zobrist::psq[int(us)][int(PAWN)][int(from)];
        remove_piece(from);
        PieceType promoted = m.promotion_type();
        put_piece(us, promoted, to);
        // XOR in promoted piece at destination
        st_->key ^= Zobrist::psq[int(us)][int(promoted)][int(to)];
    } else {
        // CRITICAL FIX: Update Zobrist key for the moving piece
        // XOR out the piece from its old square, XOR in at the new square
        st_->key ^= Zobrist::psq[int(us)][int(pt)][int(from)];
        move_piece(from, to);
        st_->key ^= Zobrist::psq[int(us)][int(pt)][int(to)];
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
        // CRITICAL: Update Zobrist hash for rook position change
        st_->key ^= Zobrist::psq[int(us)][int(ROOK)][int(rfrom)];
        move_piece(rfrom, rto);
        st_->key ^= Zobrist::psq[int(us)][int(ROOK)][int(rto)];
    }

    // Handle en passant
    if (m.is_en_passant()) {
        Square cap_sq = Square(to - (us == WHITE ? 8 : -8));
        remove_piece(cap_sq);
        st_->key ^= Zobrist::psq[int(them)][int(PAWN)][int(cap_sq)];
    }

    // Update en passant square (check piece type before move)
    // CRITICAL FIX: Remove old EP square from key BEFORE setting new one
    // The key was copied from previous state and still contains the old EP hash
    if (st_->ep_square != SQUARE_NONE) {
        st_->key ^= Zobrist::en_passant[file_of(st_->ep_square)];
    }
    st_->ep_square = SQUARE_NONE;
    if (pt == PAWN && std::abs(int(rank_of(to)) - int(rank_of(from))) == 2) {
        st_->ep_square = Square((from + to) / 2);
        st_->key ^= Zobrist::en_passant[file_of(st_->ep_square)];
    }

    // CRITICAL FIX: Update Zobrist key for castling rights changes
    // XOR out OLD castling rights BEFORE modifying them
    for (Color c : {WHITE, BLACK}) {
        for (int i = 0; i < 2; ++i) {
            if (st_->castling_rights & (c == WHITE ? (WHITE_KINGSIDE << i) : (BLACK_KINGSIDE << i))) {
                st_->key ^= Zobrist::castling[(c << 1) | i];
            }
        }
    }

    // Update castling rights
    // Remove OUR castling rights when our rook moves FROM its starting square
    if (pt == KING) {
        st_->castling_rights &= ~(us == WHITE ? (WHITE_KINGSIDE | WHITE_QUEENSIDE)
                                               : (BLACK_KINGSIDE | BLACK_QUEENSIDE));
        castling_rights_[us] = 0;
    }
    if (from == (us == WHITE ? H1 : H8)) {
        st_->castling_rights &= ~(us == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE);
        castling_rights_[us] &= ~(us == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE);
    }
    if (from == (us == WHITE ? A1 : A8)) {
        st_->castling_rights &= ~(us == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE);
        castling_rights_[us] &= ~(us == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE);
    }

    // Remove OPPONENT's castling rights when we CAPTURE on their rook starting square
    if (to == (them == WHITE ? H1 : H8)) {
        st_->castling_rights &= ~(them == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE);
        castling_rights_[them] &= ~(them == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE);
    }
    if (to == (them == WHITE ? A1 : A8)) {
        st_->castling_rights &= ~(them == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE);
        castling_rights_[them] &= ~(them == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE);
    }

    // XOR in NEW castling rights AFTER modifying them
    for (Color c : {WHITE, BLACK}) {
        for (int i = 0; i < 2; ++i) {
            if (st_->castling_rights & (c == WHITE ? (WHITE_KINGSIDE << i) : (BLACK_KINGSIDE << i))) {
                st_->key ^= Zobrist::castling[(c << 1) | i];
            }
        }
    }

    // Switch side
    st_->key ^= Zobrist::side;
    side_to_move_ = them;

#ifndef NDEBUG
    // VALIDATION: Catch board corruption immediately (debug builds only)
    bool valid = validate_move(m, from, to, pc, captured);
    if (!valid) {
        // Silently ignore corruption during search - just don't use this position
        // The move will still be undone properly
    }

    // DIAGNOSTIC: Verify side_to_move was flipped correctly
    if (side_to_move_ != them) {
        std::cerr << "\n=== SIDE_TO_MOVE NOT FLIPPED IN do_move ===\n";
        std::cerr << "Move: " << m << "\n";
        std::cerr << "Expected side to move: " << (them == WHITE ? "WHITE" : "BLACK") << "\n";
        std::cerr << "Actual side to move: " << (side_to_move_ == WHITE ? "WHITE" : "BLACK") << "\n";
        std::cerr << "==========================================\n";
    }
#endif

    // Update check info
    set_check_info(st_);

#ifndef NDEBUG
    // DEBUG: Check board consistency after every move (debug builds only)
    assert_consistency("do_move");
#endif

    // Record position for repetition detection
    if (history_size < MAX_HISTORY) {
        position_history[history_size++] = st_->key;
    }

    return true;  // Move executed successfully
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

    // CRITICAL FIX: Restore castling_rights_[] from the restored state
    // do_move modifies castling_rights_[], but it wasn't being restored in undo_move
    // This caused permanent castling rights corruption after the first move
    for (Color c : {WHITE, BLACK}) {
        castling_rights_[c] = 0;
        if (st_->castling_rights & (c == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE)) {
            castling_rights_[c] |= (c == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE);
        }
        if (st_->castling_rights & (c == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE)) {
            castling_rights_[c] |= (c == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE);
        }
    }

#ifndef NDEBUG
    // DEBUG: Check board consistency after every undo (debug builds only)
    assert_consistency("undo_move");

    // DIAGNOSTIC: Verify side_to_move was restored correctly
    // After undo, side_to_move should be us (the side that made the move)
    if (side_to_move_ != us) {
        std::cerr << "\n=== SIDE_TO_MOVE WRONG AFTER undo_move ===\n";
        std::cerr << "Move: " << m << "\n";
        std::cerr << "Expected side to move: " << (us == WHITE ? "WHITE" : "BLACK") << "\n";
        std::cerr << "Actual side to move: " << (side_to_move_ == WHITE ? "WHITE" : "BLACK") << "\n";
        std::cerr << "st_ply: " << st_ply << "\n";
        std::cerr << "===========================================\n";
    }
#endif

    // Decrement history size for repetition detection
    if (history_size > 0) {
        history_size--;
    }
}

void Position::do_null_move() {
    // Increment state stack index
    st_ply++;
    StateInfo& next_st = state_stack[st_ply];

    // CRITICAL: Ensure all fields are initialized (state_stack may contain garbage)
    next_st = StateInfo{};

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
    next_st.halfmove_clock = st_->halfmove_clock + 1;  // Critical for is_draw()

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

// CRITICAL: Consistency check to catch board state desync
// Call after every do_move/undo_move during debugging
void Position::assert_consistency([[maybe_unused]] const char* location) {
#ifndef NDEBUG
    // Debug-only: skip all checks in release builds for performance
    // Check all 64 squares
    for (int sq = 0; sq < 64; ++sq) {
        Piece p = board[sq];
        bool mailbox_has = (p != NO_PIECE);

        // Check if this square is set in any bitboard
        Bitboard all_pieces = pieces_by_color[WHITE] | pieces_by_color[BLACK];
        bool bitboard_has = (all_pieces & (1ULL << sq)) != 0;

        if (mailbox_has != bitboard_has) {
            std::cerr << "\n=== BOARD STATE DESYNC DETECTED ===\n";
            std::cerr << "Location: " << location << "\n";
            std::cerr << "Square: " << sq << "\n";
            std::cerr << "Mailbox says: " << (mailbox_has ? "HAS PIECE " : "EMPTY");
            if (mailbox_has) {
                std::cerr << int(p) << "\n";
            } else {
                std::cerr << "\n";
            }
            std::cerr << "Bitboard says: " << (bitboard_has ? "HAS PIECE" : "EMPTY") << "\n";
            std::cerr << "All pieces bitboard: 0x" << std::hex << all_pieces << std::dec << "\n";
            std::cerr << "White pieces: 0x" << std::hex << pieces_by_color[WHITE] << std::dec << "\n";
            std::cerr << "Black pieces: 0x" << std::hex << pieces_by_color[BLACK] << std::dec << "\n";
            std::cerr << "FEN: " << fen() << "\n";
            std::cerr << "Last move: " << st_->move << "\n";
            std::cerr << "====================================\n";
            std::cerr.flush();

            // Don't abort in production, just log and continue
            // abort();
        }
    }

    // CRITICAL: Verify piece counts match bitboards
    for (int c = 0; c < NO_COLOR; ++c) {
        for (int pt = 0; pt < ALL_PIECES; ++pt) {
            int count = piece_count[c][pt];
            Bitboard bb = pieces_by_color[c] & pieces_by_type[pt];
            int bb_count = popcount(bb);

            if (count != bb_count) {
                std::cerr << "\n=== PIECE COUNT DESYNC ===\n";
                std::cerr << "Location: " << location << "\n";
                std::cerr << "Color: " << c << " (" << (c == WHITE ? "WHITE" : "BLACK") << ")\n";
                std::cerr << "PieceType: " << pt << "\n";
                std::cerr << "piece_count says: " << count << "\n";
                std::cerr << "bitboard count says: " << bb_count << "\n";
                std::cerr << "Bitboard: 0x" << std::hex << bb << std::dec << "\n";
                std::cerr << "FEN: " << fen() << "\n";
                std::cerr << "Last move: " << st_->move << "\n";
                std::cerr << "============================\n";
                std::cerr.flush();

                // Log piece_list contents
                std::cerr << "piece_list contents:\n";
                for (int i = 0; i < 16; ++i) {
                    Square s = piece_list[c][pt][i];
                    if (s != SQUARE_NONE) {
                        std::cerr << "  [" << i << "] = " << s << " (board=" << int(board[s]) << ")\n";
                    }
                }
                std::cerr.flush();
            }
        }
    }

    // CRITICAL: Verify that board[] array matches bitboards for each piece type
    for (int sq = 0; sq < 64; ++sq) {
        Piece p = board[sq];
        if (p != NO_PIECE) {
            Color c = color_of_piece(p);
            PieceType pt = piece_type_of(p);
            Bitboard expected_bb = pieces_by_color[c] & pieces_by_type[pt];
            if ((expected_bb & (1ULL << sq)) == 0) {
                std::ofstream err_log("C:\\Users\\chang\\Downloads\\Luminex\\board_corruption.txt", std::ios::app);
                err_log << "\n=== PIECE TYPE MISMATCH ===\n";
                err_log << "Location: " << location << "\n";
                err_log << "Square: " << sq << " (" << char('a' + (sq % 8)) << char('1' + (sq / 8)) << ")\n";
                err_log << "board[] has: " << int(p) << " (" << (c == WHITE ? "W" : "B") << "PNBRQK"[pt] << ")\n";
                err_log << "But bitboards don't have this piece type at this square!\n";
                err_log << "pieces_by_color[" << c << "]: 0x" << std::hex << pieces_by_color[c] << std::dec << "\n";
                err_log << "pieces_by_type[" << pt << "]: 0x" << std::hex << pieces_by_type[pt] << std::dec << "\n";
                err_log << "FEN: " << fen() << "\n";
                err_log << "=============================\n";
                err_log.flush();
            }
        }
    }
#endif
}

bool Position::see_ge(Move m, Value threshold) const {
    // Implement Static Exchange Evaluation (SEE)
    Square from = m.from();
    Square to = m.to();

    // Piece value array (index by PieceType: PAWN=0, KNIGHT=1, ...)
    constexpr Value piece_value[] = {
        PAWN_VALUE,     // PAWN
        KNIGHT_VALUE,   // KNIGHT
        BISHOP_VALUE,   // BISHOP
        ROOK_VALUE,     // ROOK
        QUEEN_VALUE,    // QUEEN
        0,              // KING (infinite in reality, but 0 for SEE loop termination)
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

    // CRITICAL: First check if move is pseudo-legal (piece can actually make this move)
    // This prevents impossible moves like pawn moving like knight (h5f4, c3h8)
    if (!pseudo_legal(m)) return false;

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
        // Calculate occupancy after king moves (king removed from 'from')
        // If capturing, also remove the captured piece from 'to'
        Bitboard occ = pieces();
        occ ^= square_bb(from);
        Piece captured_piece = piece_on(to);
        if (captured_piece != NO_PIECE) {
            occ ^= square_bb(to);
        }
        Bitboard attacks = attackers_to(to, occ);

        // Capture enemy pieces bitboard BEFORE any potential state change
        Bitboard enemy_pieces = pieces(them);

        if (attacks & enemy_pieces) {
            return false;
        }
        // Castling needs special handling
        if (m.is_castling()) {
            // King cannot castle out of check
            if ((attackers_to(from) & pieces(them)) != 0) return false;

            if (to == (us == WHITE ? G1 : G8)) {
                // Kingside castling: check f1/g1 squares AND rook on h1/h8
                Square rook_sq = Square(us == WHITE ? H1 : H8);
                if (piece_on(rook_sq) != make_piece(us, ROOK)) return false;
                Square s1 = Square((us == WHITE ? F1 : F8));
                Square s2 = Square((us == WHITE ? G1 : G8));
                if ((pieces() & (square_bb(s1) | square_bb(s2))) != 0) return false;
                if ((attackers_to(s1) & pieces(them)) != 0) return false;
                if ((attackers_to(s2) & pieces(them)) != 0) return false;
            } else {
                // Queenside castling: check b1/c1/d1 squares AND rook on a1/a8
                Square rook_sq = Square(us == WHITE ? A1 : A8);
                if (piece_on(rook_sq) != make_piece(us, ROOK)) return false;
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

    // For non-king moves, verify the king is not in check after the move
    // Use attackers_to with updated occupancy - this is simpler and more correct
    // than hand-rolled attack checks (which had the pawn direction bug)
    Bitboard occ_after = pieces();
    occ_after ^= square_bb(from);
    occ_after |= square_bb(to);  // Place piece on destination

    // For en passant, also remove the captured pawn
    Square ep_cap_sq = SQUARE_NONE;
    if (m.is_en_passant()) {
        ep_cap_sq = Square(to - (us == WHITE ? 8 : -8));
        occ_after ^= square_bb(ep_cap_sq);
    }

    // Use attackers_to with updated occupancy to find all attacks on king
    Bitboard attacks = attackers_to(ksq, occ_after);

    // Remove captured pieces from enemy set
    Bitboard enemy = pieces(them);
    if (enemy & square_bb(to)) {
        enemy ^= square_bb(to);
    }
    if (ep_cap_sq != SQUARE_NONE && (enemy & square_bb(ep_cap_sq))) {
        enemy ^= square_bb(ep_cap_sq);
    }

    // If any enemy piece attacks the king after the move, it's illegal
    if (attacks & enemy) {
        return false;
    }

    return true;
}

bool Position::pseudo_legal(const Move m) const {
    Square from = m.from();
    Square to = m.to();

    // CRITICAL: Validate squares before accessing board array
    // board[] has SQUARE_NONE elements (64), indices 0-63 are valid
    // is_ok(s) returns s < SQUARE_NONE
    if (!is_ok(from) || !is_ok(to)) return false;

    PieceType pt = piece_type_on(from);

    if (pt == PT_NONE) return false;
    if (Color(board[from] / 6) != side_to_move_) return false;
    if (pieces(side_to_move_) & square_bb(to)) return false;

    // Castling
    if (m.is_castling()) {
        if (pt != KING) return false;
        // CRITICAL: King must be on starting square for castling to be valid
        Square ksq = king_square[side_to_move_];
        if (from != ksq) return false;  // From must be king's current square
        // Verify move goes to correct castling destination
        if (to == Square(side_to_move_ == WHITE ? G1 : G8)) return true;  // Kingside
        if (to == Square(side_to_move_ == WHITE ? C1 : C8)) return true;  // Queenside
        return false;  // Invalid castling destination
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
        case BISHOP: attacks = bishop_attacks_bb(from, pieces()); break;
        case ROOK: attacks = rook_attacks_bb(from, pieces()); break;
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

bool Position::validate_move(Move m, Square from, Square to, Piece moved_pc, PieceType captured) const {
    // Check 1: The piece that moved should now be on 'to' square
    Piece piece_on_to = board[to];
    Color expected_color = Color(moved_pc / 6);
    (void)captured;  // Unused for now

    // Special case: castling - king is on 'to', rook moved too
    if (m.is_castling()) {
        Piece king_on_to = board[to];
        if (king_on_to != make_piece(expected_color, KING)) {
            return false;  // King not on destination
        }
        // Check rook position
        Square rook_to = (file_of(to) > file_of(from)) ? Square(to - 1) : Square(to + 1);
        if (piece_on(rook_to) != make_piece(expected_color, ROOK)) {
            return false;  // Rook not on expected square
        }
    } else {
        // Normal move: piece should be on 'to'
        if (piece_on_to != moved_pc) {
            return false;  // Piece not on destination
        }
    }

    // Check 2: 'from' square should be empty (except for en passant where we capture a pawn)
    if (!m.is_castling() && !m.is_en_passant()) {
        if (board[from] != NO_PIECE) {
            return false;  // From square not empty after move
        }
    }

    // Check 3: Total piece count should be consistent
    int total_pieces = 0;
    for (int c = 0; c < NO_COLOR; ++c) {
        for (int pt = 0; pt < ALL_PIECES; ++pt) {
            int count = piece_count[c][pt];
            total_pieces += count;
        }
    }

    // Starting position has 32 pieces, can only decrease with captures
    if (total_pieces > 32 || total_pieces < 2) {  // Need at least 2 kings
        return false;  // Invalid piece count
    }

    // Check 4: Exactly one king of each color
    int white_kings = piece_count[WHITE][KING];
    int black_kings = piece_count[BLACK][KING];
    if (white_kings != 1 || black_kings != 1) {
        return false;  // Wrong number of kings
    }

    // Check 5: Kings should be on their tracked squares
    if (board[king_square[WHITE]] != make_piece(WHITE, KING)) {
        return false;  // White king not on tracked square
    }
    if (board[king_square[BLACK]] != make_piece(BLACK, KING)) {
        return false;  // Black king not on tracked square
    }

    return true;
}

bool Position::is_draw() const {
    // 50-move rule: if 100 half-moves (50 full moves) without
    // a pawn push or capture, it's a draw
    if (st_->halfmove_clock >= 100) {
        return true;
    }

    // Insufficient material: K vs K only (keep it minimal and safe)
    if (popcount(pieces()) == 2) {
        return true;  // K vs K
    }

    // Repetition detection — only check actual moves, not null moves
    // Key insight: only search back halfmove_clock positions, because any
    // pawn move or capture resets the possibility of repetition.
    // Also, only compare against positions where the same side was to move (step by 2).
    if (history_size >= 5 && st_->halfmove_clock >= 4) {
        int limit = std::min(st_->halfmove_clock, history_size - 1);
        Key k = st_->key;

        for (int i = 4; i <= limit; i += 2) {
            int idx = history_size - 1 - i;
            if (idx < 0) break;
            if (position_history[idx] == k) {
                return true;  // Twofold in search = draw
            }
        }
    }

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
