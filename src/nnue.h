// nnue.h — Optional NNUE evaluation for Luminex.
// Toggleable at runtime via the UCI option "UseNNUE". When disabled (default)
// or when no net is loaded, the engine falls back to the pure HCE eval, so NNUE
// is fully optional and non-destructive. Net file format written by
// nnue/luminex_nnue_train.py (header LNN1 + float32 weights).
#pragma once
#include "luminex.h"

namespace luminex::nnue {

// Compile-time cap on L1 width (the runtime L1, read from the net, may be smaller).
inline constexpr int NNUE_L1_MAX = 512;

// Load a .nnue file. Returns true on success. Safe to call multiple times.
bool load(const std::string& path);

// True iff a net is loaded AND the user enabled UseNNUE. evaluate() checks this.
bool enabled();

// Set the UseNNUE flag (wired from the UCI option handler).
void set_enabled(bool on);

// Hybrid mode: NNUE for qsearch leaves only, HCE for internal nodes.
void set_hybrid(bool on);
bool hybrid();

// Skip accumulator updates (used during qsearch in hybrid mode — the position is
// restored after qsearch, so the accumulator stays correct without updates).
void set_skip_updates(bool on);

// The NNUE eval for the position, in engine eval units (centipawns).
// Reads the incremental accumulator (must be current — see refresh/update).
// Only valid when enabled() is true.
Value evaluate(const Position& pos);

// Whether a net is currently loaded (independent of the enabled flag).
bool loaded();

// Runtime L1 width of the loaded net (0 if none loaded).
int l1();

// Print profiling counters (evals/updates/refreshes + time each) to stderr.
void print_stats();

// ---- incremental accumulator hooks (no-ops when !loaded(); cheap branch otherwise) ----
// Full rebuild of the CURRENT state's accumulator from the board. Call from Position::set.
void refresh(Position& pos);

// Apply the move's feature deltas to the current (child) state's accumulator.
// The parent→child copy is done in board.cpp's do_move copy block; this applies the delta.
//   m        : the move
//   moved    : the Piece that moved (read from board[from] before mutation)
//   captured : captured PieceType (PT_NONE if none) = st->captured_piece
// Call from do_move AFTER board mutation + st_ advance.
void update(Position& pos, Move m, Piece moved, PieceType captured);

// Null move: board unchanged, so the child accumulator is just a copy of the parent's.
// Call from do_null_move AFTER st_ advance.
void copy_for_null(Position& pos);

} // namespace luminex::nnue
