// nnue.h — Optional NNUE evaluation for Luminex.
// Toggleable at runtime via the UCI option "UseNNUE". When disabled (default)
// or when no net is loaded, the engine falls back to the pure HCE eval, so NNUE
// is fully optional and non-destructive. Net file format written by
// nnue/luminex_nnue_train.py (header LNN1 + float32 weights).
#pragma once
#include "luminex.h"
#include <string>
#include <vector>

namespace luminex::nnue {

// The cap on L1 width (NNUE_L1_MAX) and the kernels' interface are in
// nnue_kernels.h.

// Network shipped with each release, by its real name: CMake reads it from the
// file name of the asset .github/scripts/fetch-net.sh pins, so the default
// NNUEFile follows every net swap. Release binaries carry the network inside
// them (CMake LUMINEX_EMBED_PINNED); a build without it loads the file of this
// name from the working directory or next to the engine binary.
#ifndef LUMINEX_DEFAULT_NET
#error "LUMINEX_DEFAULT_NET comes from CMake: the pinned network's file name in .github/scripts/fetch-net.sh"
#endif
inline constexpr const char* DEFAULT_NET = LUMINEX_DEFAULT_NET;

// Load a .nnue file. Returns true on success. A missing file or bad header
// keeps the current net; a file that fails part-way leaves no net loaded.
bool load(const std::string& path);

// Load the network built into the binary, under the same rules as load().
// False when the build carries none.
bool load_embedded();

// Whether the binary carries a built-in network.
bool has_embedded();

// True iff a net is loaded AND the user enabled UseNNUE. evaluate() checks this.
bool enabled();

// Set the UseNNUE flag (wired from the UCI option handler).
void set_enabled(bool on);

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

// gen768t (LNT8) debugging, for the relrows and lanes UCI commands. relation_rows: the position's relation and
// pawn-pair rows for White's and Black's perspective, sorted, one "relrows <side> <n> <rows...>" line each
// (ppm: pawn pairs in contact 0, all 1; needs no network). lane_dump: the lanes the forward pass reads, per
// perspective ("lanes <side> <L1 values>", the relation term included below T, then "rel <side> <T values>"),
// brought current first.
std::string relation_rows(const Position& pos, int ppm);
std::string lane_dump(const Position& pos);

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

// ---- SIMD kernel levels ----
// Whether the binary picks its kernels at run time (the universal x86-64
// build, CMake LUMINEX_DISPATCH). Only then do the levels below offer a choice.
bool simd_dispatch();

// The kernels' instruction-set level in use: avx512vnni, avx512, avxvnni,
// avx2 or base on x86-64; dotprod or base on ARM.
const char* simd_level();

// The levels this build carries, best first (the values of the SimdLevel
// option after auto).
std::vector<std::string> simd_levels();

// The SimdLevel option: "auto" for the best level the CPU supports, or a
// level's name, which gives the best supported level at or below it. Returns
// what to report: the level now in use, and why if it is not the one asked for.
std::string set_simd_level(const std::string& request);

// nnuecheck: the evaluations of a fixed set of games under every level the
// CPU supports, which must agree exactly, and their digest for comparing
// builds. Fills report (lines, the last "nnuecheck ok" or "nnuecheck FAILED:
// ..."); false on a difference or without an int8 network.
bool check_simd(std::string& report);

} // namespace luminex::nnue
