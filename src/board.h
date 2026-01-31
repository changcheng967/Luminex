#pragma once

#include "bitboard.h"
#include "types.h"
#include <string>

namespace luminex {

// Position state for undo
struct StateInfo {
    Key key = 0;
    Bitboard checkers = 0;
    Bitboard pinned = 0;
    Bitboard block_checkers = 0;
    Square ep_square = SQUARE_NONE;
    int castling_rights = 0;
    int ply = 0;
    Move move = MOVE_NONE;
    PieceType captured_piece = PT_NONE;
    bool move_was_executed = true;  // Tracks if do_move actually executed the move (vs. returning early)
};

class Position {
public:
    Position() : st_(&dummy_state), st_ply(0) {
        dummy_state.key = 0;
        dummy_state.checkers = 0;
        dummy_state.pinned = 0;
        dummy_state.block_checkers = 0;
        dummy_state.ep_square = SQUARE_NONE;
        dummy_state.castling_rights = 0;
        dummy_state.ply = 0;
    }

    void set(const std::string& fen);
    std::string fen() const;

    Bitboard pieces() const;
    Bitboard pieces(Color c) const;
    Bitboard pieces(PieceType pt) const;
    Bitboard pieces(Color c, PieceType pt) const;
    Bitboard pieces(Color c, PieceType pt1, PieceType pt2) const;

    Piece piece_on(Square s) const;
    PieceType piece_type_on(Square s) const;
    Color color_of_piece(Piece p) const;
    Color side_to_move() const;

    Bitboard checkers() const;
    Bitboard pinned() const;
    Bitboard blockers_for_king(Color c) const;

    Square ep_square() const;
    Square king_sq(Color c) const;
    bool castling_allowed(Color c, CastlingRight cr) const;

    int game_ply() const;
    void set_game_ply(int ply);

    Key key() const;

    bool is_draw() const;
    bool is_check() const;

    void do_move(Move m);
    void undo_move(Move m);

    void do_null_move();
    void undo_null_move();

    bool see_ge(Move m, Value threshold = VALUE_ZERO) const;

    bool legal(Move m) const;
    bool pseudo_legal(const Move m) const;
    bool capture(Move m) const;
    bool capture_or_promotion(Move m) const;

    Direction pawn_push(Color c) const;

    static constexpr int MAX_MOVES = 256;
    static constexpr int MAX_PLY = 246;

private:
    void put_piece(Color c, PieceType pt, Square s);
    void remove_piece(Square s);
    void move_piece(Square from, Square to);

    void set_castling_right(Color c, Square rfrom);
    void set_check_info(StateInfo* st);
    bool validate_move(Move m, Square from, Square to, Piece moved_pc, PieceType captured) const;
    void assert_consistency(const char* location);  // Debug: check board consistency

    bool see_gen(Bitboard stmAttackers, Bitboard occupied) const;

    Bitboard attackers_to(Square s) const;
    Bitboard attackers_to(Square s, Bitboard occupied) const;
    Bitboard slider_blockers(Bitboard sliders, Bitboard& pinners) const;

    Bitboard pieces_by_type[ALL_PIECES] = {};
    Bitboard pieces_by_color[NO_COLOR] = {};
    Piece board[SQUARE_NONE] = {};
    int piece_count[NO_COLOR][ALL_PIECES] = {};
    Square piece_list[NO_COLOR][ALL_PIECES][16] = {};
    int index[SQUARE_NONE] = {};

    Square king_square[NO_COLOR] = {};

    Color side_to_move_ = WHITE;
    StateInfo* st_ = nullptr;
    int game_ply_ = 0;
    int st_ply = 0;  // Index into state_stack

    static constexpr int MAX_STATES = MAX_PLY + 10;
    StateInfo state_stack[MAX_STATES] = {};
    StateInfo dummy_state;  // For default initialization

    int castling_rights_[NO_COLOR] = {};
    Square castling_rook_square_[NO_COLOR][2] = {};
    Bitboard castling_path_[NO_COLOR][2] = {};
};

inline Bitboard Position::pieces() const { return pieces_by_color[WHITE] | pieces_by_color[BLACK]; }
inline Bitboard Position::pieces(Color c) const { return pieces_by_color[c]; }
inline Bitboard Position::pieces(PieceType pt) const { return pieces_by_type[pt]; }
inline Bitboard Position::pieces(Color c, PieceType pt) const { return pieces_by_color[c] & pieces_by_type[pt]; }
inline Bitboard Position::pieces(Color c, PieceType pt1, PieceType pt2) const { return pieces_by_color[c] & (pieces_by_type[pt1] | pieces_by_type[pt2]); }

inline Piece Position::piece_on(Square s) const { return is_ok(s) ? board[s] : NO_PIECE; }
inline PieceType Position::piece_type_on(Square s) const { return piece_type_of(board[s]); }
inline Color Position::color_of_piece(Piece p) const { return p == NO_PIECE ? NO_COLOR : static_cast<Color>(p / 6); }
inline Color Position::side_to_move() const { return side_to_move_; }

inline Bitboard Position::checkers() const { return st_->checkers; }
inline Bitboard Position::pinned() const { return st_->pinned; }
inline Bitboard Position::blockers_for_king(Color) const { return st_->block_checkers; }

inline Square Position::ep_square() const { return st_->ep_square; }
inline Square Position::king_sq(Color c) const { return king_square[c]; }
inline bool Position::castling_allowed(Color c, CastlingRight cr) const { return castling_rights_[c] & cr; }

inline int Position::game_ply() const { return game_ply_; }
inline void Position::set_game_ply(int ply) { game_ply_ = ply; }

inline Key Position::key() const { return st_->key; }

inline bool Position::is_check() const { return checkers() != 0; }

// Sliding attack helpers
Bitboard bb_rank_attacks(Square s, Bitboard occupied);
Bitboard bb_file_attacks(Square s, Bitboard occupied);
Bitboard bb_diag_attacks(Square s, Bitboard occupied);

// Queen attacks with blocking pieces
inline Bitboard queen_attacks_bb(Square s, Bitboard occupied) {
    return (bb_rank_attacks(s, occupied) | bb_file_attacks(s, occupied) | bb_diag_attacks(s, occupied));
}

// Zobrist initialization
void init_zobrist();

// Direction constants for pawn push
enum : Direction {
    NORTH = 8,
    EAST = 1,
    SOUTH = -8,
    WEST = -1,
    NORTH_EAST = NORTH + EAST,
    NORTH_WEST = NORTH + WEST,
    SOUTH_EAST = SOUTH + EAST,
    SOUTH_WEST = SOUTH + WEST
};

} // namespace luminex
