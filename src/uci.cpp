#ifdef _MSC_VER
#pragma warning(disable: 4459) // variable shadowing - intentional for pos parameter
#endif
#include "luminex.h"
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <chrono>
#include <cstdarg>
#include <expected>
#include <format>
#include <iterator>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <atomic>
#include <memory>
#include <mutex>
#include <cstdio>
#include <fstream>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX   // no min and max macros: they would break std::min and std::max
#endif
#include <windows.h>
#include <process.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

// =====================================================================================================================
// The opening book (its code is after the UCI loop's)
// =====================================================================================================================

#include <cstdint>
#include <string>

namespace luminex {

class Position;

// Polyglot opening book reader
class Book {
public:
    bool open(const char* path);
    void close();

    // Probe book for current position. Returns UCI move string or empty.
    std::string probe(const Position& pos);

    bool is_open() const { return file_ != nullptr; }

private:
    struct Entry {
        uint64_t key;
        uint16_t move;
        uint16_t weight;
        uint32_t learn;
    };

    Entry read_entry(int index);
    int find_pos(uint64_t key);

    FILE* file_ = nullptr;
    int size_ = 0;
};

// Global book instance
extern Book g_book;
extern std::string g_book_path;

// Compute Polyglot hash for a position
uint64_t compute_polyglot_key(const Position& pos);

} // namespace luminex

namespace luminex {

// Global volatile stop flag - defined in search.cpp
extern volatile bool g_stop_requested;

// Per-root-move perft counts; defined at the end of this file (used by "go perft")
uint64_t perft_divide(Position& pos, Depth depth);

// Position for UCI
static Position pos;
int g_search_depth = 0;
int g_nodes_per_move = 0;  // NodesPerMove: >0 = fixed-node search (SPSA tuning, no clock forfeits)

// Search thread management: one search thread at a time, on a large stack (SearchThread, luminex.h)
static SearchThread search_thread;
static std::mutex io_mutex;

// Debug logging
static FILE* dbglog = nullptr;

// NNUE network selection. The net loads lazily at isready/go so that a GUI's
// NNUEFile setoption (sent between uci and isready) wins over the default.
static std::string g_net_option;      // NNUEFile value; empty = DEFAULT_NET
static bool g_net_attempted = false;  // load already tried for this setting
static bool g_use_nnue = true;

// Directory of the running binary (with trailing separator), or "" if unknown.
static std::string executable_dir() {
    std::string path;
#if defined(_WIN32)
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) path.assign(buf, n);
#elif defined(__APPLE__)
    char buf[4096];
    uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) == 0) path = buf;
#elif defined(__linux__)
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) path.assign(buf, size_t(n));
#endif
    size_t cut = path.find_last_of("/\\");
    return cut == std::string::npos ? std::string() : path.substr(0, cut + 1);
}

static bool file_readable(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

// A net name resolves as given (relative to the working directory), then
// next to the engine binary. Returns "" if neither exists.
static std::string resolve_net_path(const std::string& name) {
    if (name.empty()) return {};
    if (file_readable(name)) return name;
    bool absolute = name[0] == '/' || name[0] == '\\' || (name.size() > 1 && name[1] == ':');
    if (!absolute) {
        std::string dir = executable_dir();
        if (!dir.empty() && file_readable(dir + name)) return dir + name;
    }
    return {};
}

// Check for stop command - simply check the atomic flag
bool check_for_stop_command() {
    return stop.load(std::memory_order_seq_cst);
}

// Thread-safe output
static void safe_output(const std::string& msg) {
    std::lock_guard<std::mutex> lock(io_mutex);
    if (dbglog) { fprintf(dbglog, "SEND: [%s]\n", msg.c_str()); fflush(dbglog); }
    std::cout << msg << std::flush;
}

// Thread-safe output for search.cpp (exports the mutex functionality)
void uci_safe_output(const std::string& msg) {
    safe_output(msg);
}

// Debug log function for search.cpp
void uci_debug_log(const char* format, ...) {
    if (dbglog) {
        va_list args;
        va_start(args, format);
        vfprintf(dbglog, format, args);
        va_end(args);
        fflush(dbglog);
    }
}

// Forward declaration
static void search_worker(Position pos_copy, Limits lim);

// Wait for search thread to finish
static void wait_for_search_thread() {
    if (search_thread.joinable()) search_thread.join();
}

// End the search, if one runs, and wait for it. Both stop flags are set: a
// reply that waits for ponderhit or stop (hold_reply) is released too, for a
// GUI that sends position without stop.
static void end_search() {
    g_stop_requested = true;
    stop.store(true, std::memory_order_seq_cst);
    wait_for_search_thread();
    stop.store(false, std::memory_order_seq_cst);
    g_stop_requested = false;
}

// Launch the search on its own thread with a large stack, on every platform: the
// default for secondary threads (1 MiB on Windows, 512 KiB on macOS) is too small
static void launch_search_thread(const Position& pos, const Limits& lim) {
    wait_for_search_thread();
    // The copy of the position (~230 KB) is made on the heap, not on this thread's
    // stack: the UCI thread may have as little as the search threads used to
    auto copy = std::make_shared<Position>(pos);
    search_thread = SearchThread([copy, lim] { search_worker(*copy, lim); });
}

// Worker function that runs search in a separate thread
static void search_worker(Position pos_copy, Limits lim) {
    // The search runs in its own thread → its thread-local NNUE accumulator is empty.
    // Seed it here (the main thread's Position::set refreshed only the main thread's).
    if (nnue::loaded()) nnue::refresh(pos_copy);
    if (dbglog) { fprintf(dbglog, "SEARCH_START: depth=%d time[W]=%d time[B]=%d movetime=%d ponder=%d\n",
        lim.depth, lim.time[0], lim.time[1], lim.movetime, lim.ponder); fflush(dbglog); }

    // Probe opening book first
    std::string book_move_str = g_book.probe(pos_copy);
    Move best_move = MOVE_NONE;

    if (!book_move_str.empty()) {
        // Convert UCI string to Move
        Square from = Square((book_move_str[1] - '1') * 8 + (book_move_str[0] - 'a'));
        Square to = Square((book_move_str[3] - '1') * 8 + (book_move_str[2] - 'a'));
        PieceType promo_pt = PT_NONE;
        if (book_move_str.length() > 4) {
            switch (book_move_str[4]) {
                case 'q': promo_pt = QUEEN; break;
                case 'r': promo_pt = ROOK; break;
                case 'b': promo_pt = BISHOP; break;
                case 'n': promo_pt = KNIGHT; break;
            }
        }
        ExtMove legal_moves[MAX_MOVES];
        ExtMove* legal_end = generate<GEN_LEGAL>(pos_copy, legal_moves);
        for (ExtMove* it = legal_moves; it != legal_end; ++it) {
            if (it->move.from() == from && it->move.to() == to) {
                if (promo_pt != PT_NONE) {
                    if (it->move.is_promotion() && it->move.promotion_type() == promo_pt) {
                        best_move = it->move;
                        break;
                    }
                } else if (!it->move.is_promotion()) {
                    best_move = it->move;
                    break;
                }
            }
        }
    }

    const bool from_book = best_move != MOVE_NONE;
    if (!from_book)
        best_move = search(pos_copy, lim);

    // A ponder or infinite search replies only on ponderhit or stop, however
    // early its move was ready (a book move, a forced mate, the depth limit)
    hold_reply(lim);
    if (!from_book)
        report_after_hit(pos_copy);
    finish_clock(lim);

    if (dbglog) { fprintf(dbglog, "SEARCH_DONE: about to send bestmove\n"); fflush(dbglog); }

    // Extract ponder move from TT (opponent's expected response)
    Move ponder_move = MOVE_NONE;
    if (best_move != MOVE_NONE) {
        if (pos_copy.do_move(best_move)) {
            bool found;
            TTEntry* tte = TT.probe(pos_copy.key(), found);
            if (found && tte->move() != MOVE_NONE) {
                // The TT move may come from a colliding position: print it only
                // if it is a legal reply here.
                Move pm = tte->move();
                ExtMove replies[MAX_MOVES];
                ExtMove* replies_end = generate<GEN_LEGAL>(pos_copy, replies);
                for (ExtMove* it = replies; it != replies_end; ++it)
                    if (it->move == pm) { ponder_move = pm; break; }
            }
        }
    }

    // Output best move directly from search thread (thread-safe)
    std::string line;
    if (best_move != MOVE_NONE) {
        line = std::format("bestmove {}", best_move);  // uses std::formatter<luminex::Move>
        if (ponder_move != MOVE_NONE) {
            std::format_to(std::back_inserter(line), " ponder {}", ponder_move);
        }
        line += "\n";
    } else {
        line = "bestmove 0000\n";
    }
    safe_output(line);

    ponder_mode.store(false, std::memory_order_relaxed);
}

// Load the network file `name` (resolved as resolve_net_path does), saying
// why when it cannot.
static bool load_net_file(const std::string& name) {
    std::string path = resolve_net_path(name);
    if (path.empty()) {
        safe_output("info string NNUEFile " + name + " not found\n");
        return false;
    }
    if (!nnue::load(path)) {
        safe_output("info string NNUE network " + path + " failed to load\n");
        return false;
    }
    return true;
}

// The default network: the one built into the binary, or for a build without
// one, the DEFAULT_NET file in the working directory or next to the binary.
static bool load_default_net() {
    if (nnue::has_embedded()) {
        if (nnue::load_embedded()) return true;
        safe_output("info string the built-in NNUE network failed to load\n");
    }
    std::string path = resolve_net_path(nnue::DEFAULT_NET);
    if (path.empty()) return false;
    if (!nnue::load(path)) {
        safe_output("info string NNUE network " + path + " failed to load\n");
        return false;
    }
    return true;
}

// Load the configured net unless that was already tried for the current
// setting; force reloads even when a net is loaded (a new NNUEFile value).
// NNUEFile empty or at its default name selects the default network; any
// other value names a file. A file that will not load leaves the net in use
// in place, or the default when there is none yet. Search state learned
// under the previous eval is cleared on every successful load.
static void ensure_net_loaded(bool force = false) {
    if (!g_use_nnue) return;
    if (!force && (g_net_attempted || nnue::loaded())) return;
    g_net_attempted = true;

    const bool wants_default = g_net_option.empty() || g_net_option == nnue::DEFAULT_NET;
    bool ok = !wants_default && load_net_file(g_net_option);
    if (!ok && (wants_default || !nnue::loaded())) ok = load_default_net();
    if (ok) {
        TT.clear();
        TT.new_search();
        clear_correction_history();
        clear_search_history();
        nnue::refresh(pos);  // pos was set before this net existed
        return;
    }
    if (!nnue::loaded())
        safe_output("info string no NNUE network loaded, using the classical evaluation\n");
}

// The net's load report, printed by the commands that use the net (go, eval,
// nnuecheck, ...) rather than at the load: see nnue::take_report
static void print_net_report() {
    if (const std::string report = nnue::take_report(); !report.empty())
        safe_output(report);
}

// luminex nnuecheck [NETFILE]: the nnuecheck command from the command line,
// with NETFILE or the default network (the built-in one, or the file of its
// name) loaded as NNUEFile would load it. True when every level agrees.
bool run_nnuecheck(const std::string& net_file) {
    g_net_option = net_file;
    g_net_attempted = false;
    ensure_net_loaded(true);
    print_net_report();
    std::string report;
    const bool ok = nnue::check_simd(report);
    safe_output(report + "\n");
    return ok;
}

void handle_uci() {
    init_evaluation();
    // Development builds report their build id (CMake's LUMINEX_BUILD_ID,
    // e.g. dev-20261112-d4774f1a) in place of the version
#ifdef LUMINEX_BUILD_ID
    safe_output("id name " + std::string(ENGINE_NAME) + " " + LUMINEX_BUILD_ID + "\n");
#else
    safe_output("id name " + std::string(ENGINE_NAME) + " " + ENGINE_VERSION + "\n");
#endif
    safe_output("id author " + std::string(ENGINE_AUTHOR) + "\n");
    safe_output("option name Hash type spin default 256 min 1 max 1048576\n");
    safe_output("option name Threads type spin default 1 min 1 max 64\n");
    safe_output("option name MultiPV type spin default 1 min 1 max 256\n");
    safe_output("option name Ponder type check default false\n");
    safe_output("option name Contempt type spin default 0 min -1000 max 1000\n");
    safe_output("option name Clear Hash type button\n");
    safe_output("option name BookFile type string default <empty>\n");
    safe_output("option name Move Overhead type spin default 30 min 0 max 5000\n");
    safe_output("option name TimeSchedule type spin default 2 min 0 max 2\n");
    safe_output("option name TimeStability type spin default 1 min 0 max 1\n");
    safe_output("option name TimeZeroInc type spin default 1 min 0 max 1\n");
    safe_output("option name TimeMovesToGo type spin default 1 min 0 max 1\n");
    safe_output("option name TimeLag type spin default 1 min 0 max 1\n");
    safe_output("option name SyzygyPath type string default <empty>\n");
    safe_output("option name Skill Level type spin default 20 min 0 max 20\n");
    safe_output("option name UCI_ShowWDL type check default false\n");
    safe_output("option name DatagenInfo type check default false\n");
    safe_output("option name SearchDepth type spin default 0 min 0 max 30\n");
    safe_output("option name NodesPerMove type spin default 0 min 0 max 100000000\n");
    // Search constants (SPSAParams, luminex.h), exposed for OpenBench SPSA tuning
    {
        const SPSAParams& sp = SPSAParams::get();
        for (const SPSAOption& o : SPSA_OPTIONS)
            safe_output(std::format("option name {} type spin default {} min {} max {}\n",
                                    o.name, sp.*o.field, o.min, o.max));
    }
    safe_output("option name UCI_Elo type spin default 1320 min 1320 max 3190\n");
    safe_output("option name UCI_LimitStrength type check default false\n");
    // UCI_Chess960 is deliberately not advertised: castling is implemented
    // for standard chess only.
    // NNUE evaluation; without a loadable net the engine uses the HCE and says so.
    safe_output("option name UseNNUE type check default true\n");
    safe_output(std::string("option name NNUEFile type string default ") + nnue::DEFAULT_NET + "\n");
    // The universal x86-64 build picks its network kernels for the CPU; this
    // forces a lower level, for testing (one the CPU lacks gives the best below it).
    if (nnue::simd_dispatch()) {
        std::string opt = "option name SimdLevel type combo default auto var auto";
        for (const std::string& level : nnue::simd_levels()) opt += " var " + level;
        safe_output(opt + "\n");
    }
    // Eval parameters (self-engineered defaults)
    safe_output("option name BishopPairMG type spin default 40 min -100 max 200\n");
    safe_output("option name BishopPairEG type spin default 100 min -100 max 300\n");
    safe_output("option name RookOpenMG type spin default 25 min -50 max 150\n");
    safe_output("option name RookOpenEG type spin default 40 min -50 max 150\n");
    safe_output("option name RookSemiOpenMG type spin default 15 min -50 max 100\n");
    safe_output("option name RookSemiOpenEG type spin default 20 min -50 max 100\n");
    safe_output("option name Rook7thMG type spin default 30 min -50 max 100\n");
    safe_output("option name Rook7thEG type spin default 25 min -50 max 150\n");
    safe_output("option name PawnShieldCenter type spin default 12 min -20 max 40\n");
    safe_output("option name PawnShieldKnight type spin default 15 min -20 max 40\n");
    safe_output("option name PawnShieldRook type spin default 8 min -20 max 30\n");
    safe_output("option name PawnStorm type spin default 10 min 0 max 30\n");
    safe_output("option name OpenFilePenaltyMG type spin default 20 min -50 max 100\n");
    safe_output("option name OpenFilePenaltyEG type spin default 15 min -50 max 100\n");
    safe_output("option name OutpostKnightMG type spin default 30 min -20 max 60\n");
    safe_output("option name OutpostKnightEG type spin default 15 min -20 max 40\n");
    safe_output("option name OutpostBishopMG type spin default 40 min -20 max 100\n");
    safe_output("option name OutpostBishopEG type spin default 25 min -20 max 60\n");
    safe_output("option name HangingPawnMG type spin default 10 min -20 max 50\n");
    safe_output("option name HangingPawnEG type spin default 35 min -20 max 100\n");
    safe_output("option name FarKnightMG type spin default 15 min -50 max 60\n");
    safe_output("option name FarKnightEG type spin default 5 min -50 max 60\n");
    safe_output("option name FarBishopMG type spin default 10 min -50 max 40\n");
    safe_output("option name FarBishopEG type spin default 5 min -50 max 40\n");
    safe_output("uciok\n");
}

void handle_isready() {
    safe_output("readyok\n");
}

void handle_ucinewgame() {
    TT.clear();
    TT.new_search();
    clear_correction_history();
    clear_search_history();
    clear_clock_lag();
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

void handle_position(Position& pos, const std::string& cmd) {
    size_t pos_idx = cmd.find("position");
    size_t moves_idx = cmd.find("moves");

    std::string fen;

    if (moves_idx != std::string::npos) {
        fen = cmd.substr(pos_idx + 8, moves_idx - pos_idx - 9);
    } else {
        fen = cmd.substr(pos_idx + 8);
    }

    size_t start = fen.find_first_not_of(" \t\r\n");
    if (start != std::string::npos) {
        fen = fen.substr(start);
    }

    size_t end = fen.find_last_not_of(" \t\r\n");
    if (end != std::string::npos) {
        fen = fen.substr(0, end + 1);
    }

    if (fen == "startpos") {
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    } else if (fen.size() >= 3 && fen.substr(0, 3) == "fen") {
        size_t fen_start = fen.find_first_not_of(" \t", 3);
        if (fen_start != std::string::npos) {
            fen = fen.substr(fen_start);
        } else {
            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        }
    }

    // A FEN without both kings is illegal and would crash king-centric code
    // (set_check_info, NNUE features); reject instead of dying.
    {
        std::string board = fen.substr(0, fen.find(' '));
        size_t wk = std::count(board.begin(), board.end(), 'K');
        size_t bk = std::count(board.begin(), board.end(), 'k');
        if (wk != 1 || bk != 1) {
            safe_output("info string invalid FEN (kings " + std::to_string(wk) +
                        "/" + std::to_string(bk) + ") - position unchanged\n");
            return;
        }
    }

    pos.set(fen);

    if (moves_idx != std::string::npos) {
        std::string moves_str = cmd.substr(moves_idx + 5);   // + 6 is past the end of "... moves"
        std::istringstream ss(moves_str);
        std::string move_str;

        while (ss >> move_str) {
            if (move_str.length() < 4) continue;

            Square from = Square((move_str[1] - '1') * 8 + (move_str[0] - 'a'));
            Square to = Square((move_str[3] - '1') * 8 + (move_str[2] - 'a'));

            PieceType promo_pt = PT_NONE;
            if (move_str.length() > 4) {
                switch (std::tolower(static_cast<unsigned char>(move_str[4]))) {
                    case 'q': promo_pt = QUEEN; break;
                    case 'r': promo_pt = ROOK; break;
                    case 'b': promo_pt = BISHOP; break;
                    case 'n': promo_pt = KNIGHT; break;
                }
            }

            ExtMove legal_moves[MAX_MOVES];
            ExtMove* legal_end = generate<GEN_LEGAL>(pos, legal_moves);
            Move matched = MOVE_NONE;
            for (ExtMove* lit = legal_moves; lit != legal_end; ++lit) {
                // Castling may also be written as the king taking its own rook
                // (e1h1, e8a8): the rook's corner is one file past g, two before c
                const bool rook_square_castle = lit->move.is_castling()
                    && to == Square(lit->move.to() + (file_of(lit->move.to()) == 6 ? 1 : -2));
                if (lit->move.from() == from && (lit->move.to() == to || rook_square_castle)) {
                    if (promo_pt != PT_NONE) {
                        if (lit->move.is_promotion() && lit->move.promotion_type() == promo_pt) {
                            matched = lit->move;
                            break;
                        }
                    } else if (!lit->move.is_promotion()) {
                        matched = lit->move;
                        break;
                    }
                }
            }

            // Stop at the first unplayable move and keep the position reached;
            // resetting to the start position would search a different game.
            if (matched == MOVE_NONE || !pos.do_move(matched)) {
                safe_output("info string illegal move " + move_str +
                            " in position command; it and later moves were ignored\n");
                return;
            }
        }
    }
}

void handle_go(Position& pos, const std::string& cmd) {
    // A "go" while a search runs (outside the protocol, but tools send it) ends
    // that search first: waiting for it to end by itself would block this reader
    // for ever behind a "go infinite", so no "stop" could be read.
    stop.store(true, std::memory_order_seq_cst);
    g_stop_requested = true;
    wait_for_search_thread();
    // The stop flags are cleared here, before the search thread starts, not by
    // the search itself: a "stop" read between the launch and the search's first
    // line would otherwise be erased, and a "go infinite" would never end.
    stop.store(false, std::memory_order_seq_cst);
    g_stop_requested = false;

    // "go perft <depth>": per-root-move node counts, then the total. No
    // search state is touched, so no TT reset and no search thread.
    {
        std::istringstream ps(cmd);
        std::string t;
        ps >> t;  // "go"
        if (ps >> t && t == "perft") {
            int d = 0;
            ps >> d;
            if (d >= 1 && d <= 8) {
                uint64_t total = perft_divide(pos, Depth(d));
                safe_output("total " + std::to_string(total) + "\n");
            } else {
                safe_output("info string go perft needs a depth 1-8\n");
            }
            return;
        }
    }

    // A GUI may send go without isready; the net must be in place either way.
    ensure_net_loaded();
    print_net_report();

    TT.new_search();

    std::istringstream ss(cmd);
    std::string token;
    limits = Limits();

    while (ss >> token) {
        if (token == "depth") ss >> limits.depth;
        else if (token == "nodes") ss >> limits.nodes;
        else if (token == "movetime") ss >> limits.movetime;
        else if (token == "infinite") limits.infinite = true;
        else if (token == "ponder") limits.ponder = true;
        else if (token == "wtime") ss >> limits.time[WHITE];
        else if (token == "btime") ss >> limits.time[BLACK];
        else if (token == "winc") ss >> limits.inc[WHITE];
        else if (token == "binc") ss >> limits.inc[BLACK];
        else if (token == "movestogo") ss >> limits.movestogo;
    }

    // SearchDepth UCI option: if set (>0), ignore ALL time controls and search to that
    // exact depth. This enables truly equal-depth matches (both engines same depth,
    // pure eval-quality comparison — no NPS advantage).
    if (g_search_depth > 0) {
        limits.depth = g_search_depth;
        limits.time[WHITE] = 0; limits.time[BLACK] = 0;
        limits.inc[WHITE] = 0; limits.inc[BLACK] = 0;
        limits.movetime = 0; limits.infinite = false;
    }

    // NodesPerMove: if set (>0), ignore ALL time controls and search exactly N nodes.
    // Eliminates clock-forfeiture from CPU contention during SPSA tuning.
    if (g_nodes_per_move > 0) {
        limits.nodes = g_nodes_per_move;
        limits.time[WHITE] = 0; limits.time[BLACK] = 0;
        limits.inc[WHITE] = 0; limits.inc[BLACK] = 0;
        limits.movetime = 0; limits.infinite = false;
    }

    // "go" without any time controls or depth: search indefinitely until "stop"
    if (limits.depth == 0 && !limits.infinite && limits.movetime == 0 &&
        limits.time[WHITE] == 0 && limits.time[BLACK] == 0 && limits.nodes == 0) {
        limits.infinite = true;
    }

    // Set ponder mode if go ponder. A ponderhit read while no search ran (one
    // that crossed an early reply) must not count for this search.
    ponderhit_received.store(false, std::memory_order_relaxed);
    ponder_mode.store(limits.ponder, std::memory_order_relaxed);

    // Launch search thread with large stack to prevent stack overflow
    launch_search_thread(pos, limits);
}

// A spin option's value as std::stoi reads it (leading blanks, a sign, digits;
// anything after them ignored), or why it is not one.
static std::expected<int, std::string> parse_spin(const std::string& value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || errno == ERANGE || parsed > INT_MAX || parsed < INT_MIN)
        return std::unexpected("'" + value + "' is not an integer");
    return int(parsed);
}

void handle_setoption(const std::string& cmd) {
    std::istringstream ss(cmd);
    std::string token;
    ss >> token;
    ss >> token;

    std::string name;
    std::string value;

    while (ss >> token && token != "value") {
        if (!name.empty()) name += " ";
        name += token;
    }

    while (ss >> token) {
        if (!value.empty()) value += " ";
        value += token;
    }

    // Every option but these is a spin, parsed once here: a value that is not an
    // int is reported and the option left as it was (release builds have no
    // exceptions, so std::stoi on such a value used to end the engine).
    const bool text_option = name == "Ponder" || name == "Clear Hash" || name == "BookFile" ||
        name == "SyzygyPath" || name == "UCI_ShowWDL" || name == "DatagenInfo" || name == "UCI_LimitStrength" ||
        name == "UseNNUE" || name == "NNUEFile" || name == "SimdLevel" ||
        name == "UCI_Chess960" || name == "Analysis" || name == "UCI_AnalyseMode";
    int v = 0;
    if (!text_option) {
        const std::expected<int, std::string> spin = parse_spin(value);
        if (!spin) {
            safe_output("info string option " + name + ": " + spin.error() + "; left unchanged\n");
            return;
        }
        v = *spin;
    }

    if (name == "SyzygyPath" || name == "Skill Level" ||
        name == "UCI_Elo" || name == "UCI_LimitStrength" ||
        name == "UCI_Chess960" || name == "Analysis") { /* accepted — UCI GUI/bot compatibility */ }
    else if (name == "MultiPV") {
        int n = v;
        if (n < 1) n = 1;
        if (n > 256) n = 256;
        params.multi_pv = n;
    }
    else if (name == "UCI_ShowWDL") {
        params.show_wdl = (value == "true");
    }
    else if (name == "DatagenInfo") {
        g_datagen_info = (value == "true");
    }
    else if (name == "Move Overhead") {
        int ov = v;
        if (ov < 0) ov = 0;
        if (ov > 5000) ov = 5000;
        params.move_overhead_ms = ov;
    }
    else if (name == "TimeSchedule") {
        params.time_schedule = std::clamp(v, 0, 2);
    }
    else if (name == "TimeStability") {
        params.time_stability = std::clamp(v, 0, 1);
    }
    else if (name == "TimeZeroInc") {
        params.time_zero_inc = std::clamp(v, 0, 1);
    }
    else if (name == "TimeMovesToGo") {
        params.time_moves_to_go = std::clamp(v, 0, 1);
    }
    else if (name == "TimeLag") {
        params.time_lag = std::clamp(v, 0, 1);
    }
    else if (name == "UseNNUE") {
        g_use_nnue = (value == "true");
        nnue::set_enabled(g_use_nnue);
    }
    else if (name == "NNUEFile") {
        // Every value loads now, the empty one too: it selects the default
        // network, which must replace a file loaded under an earlier value.
        g_net_option = (value == "<empty>") ? std::string() : value;
        g_net_attempted = false;
        ensure_net_loaded(true);
    }
    else if (name == "SimdLevel") {
        // The network's dense layers are rearranged for the new kernels, so
        // no search may be reading them.
        end_search();
        std::string level = value;
        std::transform(level.begin(), level.end(), level.begin(),
                       [](unsigned char ch) { return char(std::tolower(ch)); });
        safe_output("info string " + nnue::set_simd_level(level) + "\n");
    }
    else if (name == "SearchDepth") { g_search_depth = v; }
    else if (name == "NodesPerMove") { g_nodes_per_move = v; }
    else if (const SPSAOption* o = find_spsa_option(name)) {
        SPSAParams::get().*o->field = std::clamp(v, o->min, o->max);
        if (o->field == &SPSAParams::lmr_scale_quiet)
            init_search_params();  // the reduction table is derived from it
    }
    else if (name == "Contempt" || name == "UCI_AnalyseMode") {
        if (name == "Contempt") {
            params.contempt = v;
        }
    } else if (name == "BishopPairMG") { g_eval_params.bishop_pair_mg = v;
    } else if (name == "BishopPairEG") { g_eval_params.bishop_pair_eg = v;
    } else if (name == "RookOpenMG") { g_eval_params.rook_open_mg = v;
    } else if (name == "RookOpenEG") { g_eval_params.rook_open_eg = v;
    } else if (name == "RookSemiOpenMG") { g_eval_params.rook_semi_open_mg = v;
    } else if (name == "RookSemiOpenEG") { g_eval_params.rook_semi_open_eg = v;
    } else if (name == "Rook7thMG") { g_eval_params.rook_7th_mg = v;
    } else if (name == "Rook7thEG") { g_eval_params.rook_7th_eg = v;
    } else if (name == "PawnShieldCenter") { g_eval_params.pawn_shield_center = v;
    } else if (name == "PawnShieldKnight") { g_eval_params.pawn_shield_knight = v;
    } else if (name == "PawnShieldRook") { g_eval_params.pawn_shield_rook = v;
    } else if (name == "PawnStorm") { g_eval_params.pawn_storm = v;
    } else if (name == "OpenFilePenaltyMG") { g_eval_params.open_file_penalty_mg = v;
    } else if (name == "OpenFilePenaltyEG") { g_eval_params.open_file_penalty_eg = v;
    } else if (name == "OutpostKnightMG") { g_eval_params.outpost_knight_mg = v;
    } else if (name == "OutpostKnightEG") { g_eval_params.outpost_knight_eg = v;
    } else if (name == "OutpostBishopMG") { g_eval_params.outpost_bishop_mg = v;
    } else if (name == "OutpostBishopEG") { g_eval_params.outpost_bishop_eg = v;
    } else if (name == "HangingPawnMG") { g_eval_params.hanging_pawn_mg = v;
    } else if (name == "HangingPawnEG") { g_eval_params.hanging_pawn_eg = v;
    } else if (name == "FarKnightMG") { g_eval_params.far_knight_mg = v;
    } else if (name == "FarKnightEG") { g_eval_params.far_knight_eg = v;
    } else if (name == "FarBishopMG") { g_eval_params.far_bishop_mg = v;
    } else if (name == "FarBishopEG") { g_eval_params.far_bishop_eg = v;
    } else if (name == "Threads") {
        int t = v;
        if (t < 1) t = 1;
        if (t > 64) t = 64;
        num_threads = t;
    } else if (name == "Clear Hash") {
        TT.clear();
    } else if (name == "BookFile") {
        g_book_path = value;
        if (!value.empty() && value != "<empty>") {
            if (!g_book.open(value.c_str())) {
                g_book.close();
            }
        } else {
            g_book.close();
        }
    } else if (name == "Hash") {
        // The declared bounds: a negative value used to wrap to an enormous size.
        TT.resize(size_t(std::clamp(v, 1, 1048576)));
    }
}

void handle_quit() {
    g_stop_requested = true;
    stop.store(true, std::memory_order_seq_cst);
}

void uci_loop() {
    // Debug logging disabled
    // dbglog = fopen("C:\\Users\\chang\\Downloads\\Luminex\\luminex_debug.log", "w");

    std::string line;

    while (std::getline(std::cin, line)) {
        // Log received command
        if (dbglog) { fprintf(dbglog, "RECV: [%s]\n", line.c_str()); fflush(dbglog); }

        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) continue;

        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;

        if (cmd == "uci") {
            handle_uci();
        } else if (cmd == "isready") {
            ensure_net_loaded();
            TT.ensure_allocated();
            safe_output("readyok\n");
        } else if (cmd == "ucinewgame") {
            end_search();
            handle_ucinewgame();
        } else if (cmd == "position") {
            end_search();
            handle_position(pos, line);
        } else if (cmd == "go") {
            handle_go(pos, line);
        } else if (cmd == "setoption") {
            handle_setoption(line);
        } else if (cmd == "stop") {
            if (dbglog) { fprintf(dbglog, "STOP_HANDLER: setting stop=true\n"); fflush(dbglog); }
            g_stop_requested = true;  // Set volatile flag first for immediate visibility
            stop.store(true, std::memory_order_seq_cst);
            std::atomic_thread_fence(std::memory_order_seq_cst);  // Ensure visibility
            // Do NOT join here - search thread will see flag and terminate
        } else if (cmd == "ponderhit") {
            // Opponent played the expected move - switch from ponder to normal mode
            ponderhit_received.store(true, std::memory_order_relaxed);
        } else if (cmd == "quit") {
            g_stop_requested = true;
            stop.store(true, std::memory_order_seq_cst);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            wait_for_search_thread();
            if (nnue::loaded()) nnue::print_stats();
            break;
        } else if (cmd == "d") {
            safe_output(pos.fen() + "\n");
        } else if (cmd == "nnuecheck") {
            // Every kernel level the CPU supports must evaluate a fixed set of
            // games alike (nnue::check_simd); not during a search.
            end_search();
            ensure_net_loaded();
            print_net_report();
            std::string report;
            nnue::check_simd(report);
            safe_output(report + "\n");
        } else if (cmd == "eval") {
            // Static eval of the current position (NNUE if enabled, else HCE),
            // side-to-move-relative cp. Used to validate the NNUE inference
            // against the Python trainer's eval.
            ensure_net_loaded();
            print_net_report();
            Value v = evaluate(pos, false);
            std::string out = "eval cp " + std::to_string(int(v));
            safe_output(out + "\n");
        } else if (cmd == "relrows") {
            // relrows [0|1]: gen768t's relation rows of the current position (debugging; 1 = all pawn pairs)
            int ppm = 0;
            ss >> ppm;
            safe_output(nnue::relation_rows(pos, ppm));
        } else if (cmd == "lanes") {
            // the feature-layer lanes the forward pass reads for the current position (debugging)
            ensure_net_loaded();
            print_net_report();
            safe_output(nnue::lane_dump(pos));
        } else if (cmd.rfind("evalbench", 0) == 0) {
            // evalbench <N>: raw evaluate() throughput on the current position
            // (roofline probe — measures the eval kernel ceiling, no search).
            ensure_net_loaded();
            print_net_report();
            long n = atol(cmd.c_str() + 10);
            if (n <= 0) n = 1000000;
            auto t0 = std::chrono::steady_clock::now();
            Value acc = VALUE_ZERO;
            for (long i = 0; i < n; ++i) acc = Value(acc + evaluate(pos, false));
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - t0).count();
            char buf[128];
            std::snprintf(buf, sizeof buf, "evalbench %ld in %lldms = %.0f evals/s (sink %d)\n",
                          n, static_cast<long long>(ms), ms ? n * 1000.0 / ms : 0.0, int(acc) & 1);
            safe_output(buf);
        }
    }

    // End of input: a search still running (go infinite) is stopped, not waited for
    stop.store(true, std::memory_order_seq_cst);
    g_stop_requested = true;
    wait_for_search_thread();

    if (dbglog) { fclose(dbglog); }
}

void uci_send(const char* msg, ...) {
    va_list args;
    va_start(args, msg);
    std::vprintf(msg, args);
    va_end(args);
    std::cout.flush();
}

} // namespace luminex

// =====================================================================================================================
// The opening book
// =====================================================================================================================

#include <bit>       // std::byteswap, std::endian (C++23)
#include <cstdio>    // std::fread
#include <cstddef>   // std::size_t
#include <cstdlib>

namespace luminex {

Book g_book;
std::string g_book_path;

// Polyglot Zobrist random keys (public domain, from Polyglot specification)
static const uint64_t POLYGLOT_RANDOM[781] = {
    0x9D39247E33776D41ULL, 0x2AF7398005AAA5C7ULL, 0x44DB015024623547ULL, 0x9C15F73E62A76AE2ULL,
    0x75834465489C0C89ULL, 0x3290AC3A203001BFULL, 0x0FBBAD1F61042279ULL, 0xE83A908FF2FB60CAULL,
    0x0D7E765D58755C10ULL, 0x1A083822CEAFE02DULL, 0x9605D5F0E25EC3B0ULL, 0xD021FF5CD13A2ED5ULL,
    0x40BDF15D4A672E32ULL, 0x011355146FD56395ULL, 0x5DB4832046F3D9E5ULL, 0x239F8B2D7FF719CCULL,
    0x05D1A1AE85B49AA1ULL, 0x679F848F6E8FC971ULL, 0x7449BBFF801FED0BULL, 0x7D11CDB1C3B7ADF0ULL,
    0x82C7709E781EB7CCULL, 0xF3218F1C9510786CULL, 0x331478F3AF51BBE6ULL, 0x4BB38DE5E7219443ULL,
    0xAA649C6EBCFD50FCULL, 0x8DBD98A352AFD40BULL, 0x87D2074B81D79217ULL, 0x19F3C751D3E92AE1ULL,
    0xB4AB30F062B19ABFULL, 0x7B0500AC42047AC4ULL, 0xC9452CA81A09D85DULL, 0x24AA6C514DA27500ULL,
    0x4C9F34427501B447ULL, 0x14A68FD73C910841ULL, 0xA71B9B83461CBD93ULL, 0x03488B95B0F1850FULL,
    0x637B2B34FF93C040ULL, 0x09D1BC9A3DD90A94ULL, 0x3575668334A1DD3BULL, 0x735E2B97A4C45A23ULL,
    0x18727070F1BD400BULL, 0x1FCBACD259BF02E7ULL, 0xD310A7C2CE9B6555ULL, 0xBF983FE0FE5D8244ULL,
    0x9F74D14F7454A824ULL, 0x51EBDC4AB9BA3035ULL, 0x5C82C505DB9AB0FAULL, 0xFCF7FE8A3430B241ULL,
    0x3253A729B9BA3DDEULL, 0x8C74C368081B3075ULL, 0xB9BC6C87167C33E7ULL, 0x7EF48F2B83024E20ULL,
    0x11D505D4C351BD7FULL, 0x6568FCA92C76A243ULL, 0x4DE0B0F40F32A7B8ULL, 0x96D693460CC37E5DULL,
    0x42E240CB63689F2FULL, 0x6D2BDCDAE2919661ULL, 0x42880B0236E4D951ULL, 0x5F0F4A5898171BB6ULL,
    0x39F890F579F92F88ULL, 0x93C5B5F47356388BULL, 0x63DC359D8D231B78ULL, 0xEC16CA8AEA98AD76ULL,
    0x5355F900C2A82DC7ULL, 0x07FB9F855A997142ULL, 0x5093417AA8A7ED5EULL, 0x7BCBC38DA25A7F3CULL,
    0x19FC8A768CF4B6D4ULL, 0x637A7780DECFC0D9ULL, 0x8249A47AEE0E41F7ULL, 0x79AD695501E7D1E8ULL,
    0x14ACBAF4777D5776ULL, 0xF145B6BECCDEA195ULL, 0xDABF2AC8201752FCULL, 0x24C3C94DF9C8D3F6ULL,
    0xBB6E2924F03912EAULL, 0x0CE26C0B95C980D9ULL, 0xA49CD132BFBF7CC4ULL, 0xE99D662AF4243939ULL,
    0x27E6AD7891165C3FULL, 0x8535F040B9744FF1ULL, 0x54B3F4FA5F40D873ULL, 0x72B12C32127FED2BULL,
    0xEE954D3C7B411F47ULL, 0x9A85AC909A24EAA1ULL, 0x70AC4CD9F04F21F5ULL, 0xF9B89D3E99A075C2ULL,
    0x87B3E2B2B5C907B1ULL, 0xA366E5B8C54F48B8ULL, 0xAE4A9346CC3F7CF2ULL, 0x1920C04D47267BBDULL,
    0x87BF02C6B49E2AE9ULL, 0x092237AC237F3859ULL, 0xFF07F64EF8ED14D0ULL, 0x8DE8DCA9F03CC54EULL,
    0x9C1633264DB49C89ULL, 0xB3F22C3D0B0B38EDULL, 0x390E5FB44D01144BULL, 0x5BFEA5B4712768E9ULL,
    0x1E1032911FA78984ULL, 0x9A74ACB964E78CB3ULL, 0x4F80F7A035DAFB04ULL, 0x6304D09A0B3738C4ULL,
    0x2171E64683023A08ULL, 0x5B9B63EB9CEFF80CULL, 0x506AACF489889342ULL, 0x1881AFC9A3A701D6ULL,
    0x6503080440750644ULL, 0xDFD395339CDBF4A7ULL, 0xEF927DBCF00C20F2ULL, 0x7B32F7D1E03680ECULL,
    0xB9FD7620E7316243ULL, 0x05A7E8A57DB91B77ULL, 0xB5889C6E15630A75ULL, 0x4A750A09CE9573F7ULL,
    0xCF464CEC899A2F8AULL, 0xF538639CE705B824ULL, 0x3C79A0FF5580EF7FULL, 0xEDE6C87F8477609DULL,
    0x799E81F05BC93F31ULL, 0x86536B8CF3428A8CULL, 0x97D7374C60087B73ULL, 0xA246637CFF328532ULL,
    0x043FCAE60CC0EBA0ULL, 0x920E449535DD359EULL, 0x70EB093B15B290CCULL, 0x73A1921916591CBDULL,
    0x56436C9FE1A1AA8DULL, 0xEFAC4B70633B8F81ULL, 0xBB215798D45DF7AFULL, 0x45F20042F24F1768ULL,
    0x930F80F4E8EB7462ULL, 0xFF6712FFCFD75EA1ULL, 0xAE623FD67468AA70ULL, 0xDD2C5BC84BC8D8FCULL,
    0x7EED120D54CF2DD9ULL, 0x22FE545401165F1CULL, 0xC91800E98FB99929ULL, 0x808BD68E6AC10365ULL,
    0xDEC468145B7605F6ULL, 0x1BEDE3A3AEF53302ULL, 0x43539603D6C55602ULL, 0xAA969B5C691CCB7AULL,
    0xA87832D392EFEE56ULL, 0x65942C7B3C7E11AEULL, 0xDED2D633CAD004F6ULL, 0x21F08570F420E565ULL,
    0xB415938D7DA94E3CULL, 0x91B859E59ECB6350ULL, 0x10CFF333E0ED804AULL, 0x28AED140BE0BB7DDULL,
    0xC5CC1D89724FA456ULL, 0x5648F680F11A2741ULL, 0x2D255069F0B7DAB3ULL, 0x9BC5A38EF729ABD4ULL,
    0xEF2F054308F6A2BCULL, 0xAF2042F5CC5C2858ULL, 0x480412BAB7F5BE2AULL, 0xAEF3AF4A563DFE43ULL,
    0x19AFE59AE451497FULL, 0x52593803DFF1E840ULL, 0xF4F076E65F2CE6F0ULL, 0x11379625747D5AF3ULL,
    0xBCE5D2248682C115ULL, 0x9DA4243DE836994FULL, 0x066F70B33FE09017ULL, 0x4DC4DE189B671A1CULL,
    0x51039AB7712457C3ULL, 0xC07A3F80C31FB4B4ULL, 0xB46EE9C5E64A6E7CULL, 0xB3819A42ABE61C87ULL,
    0x21A007933A522A20ULL, 0x2DF16F761598AA4FULL, 0x763C4A1371B368FDULL, 0xF793C46702E086A0ULL,
    0xD7288E012AEB8D31ULL, 0xDE336A2A4BC1C44BULL, 0x0BF692B38D079F23ULL, 0x2C604A7A177326B3ULL,
    0x4850E73E03EB6064ULL, 0xCFC447F1E53C8E1BULL, 0xB05CA3F564268D99ULL, 0x9AE182C8BC9474E8ULL,
    0xA4FC4BD4FC5558CAULL, 0xE755178D58FC4E76ULL, 0x69B97DB1A4C03DFEULL, 0xF9B5B7C4ACC67C96ULL,
    0xFC6A82D64B8655FBULL, 0x9C684CB6C4D24417ULL, 0x8EC97D2917456ED0ULL, 0x6703DF9D2924E97EULL,
    0xC547F57E42A7444EULL, 0x78E37644E7CAD29EULL, 0xFE9A44E9362F05FAULL, 0x08BD35CC38336615ULL,
    0x9315E5EB3A129ACEULL, 0x94061B871E04DF75ULL, 0xDF1D9F9D784BA010ULL, 0x3BBA57B68871B59DULL,
    0xD2B7ADEEDED1F73FULL, 0xF7A255D83BC373F8ULL, 0xD7F4F2448C0CEB81ULL, 0xD95BE88CD210FFA7ULL,
    0x336F52F8FF4728E7ULL, 0xA74049DAC312AC71ULL, 0xA2F61BB6E437FDB5ULL, 0x4F2A5CB07F6A35B3ULL,
    0x87D380BDA5BF7859ULL, 0x16B9F7E06C453A21ULL, 0x7BA2484C8A0FD54EULL, 0xF3A678CAD9A2E38CULL,
    0x39B0BF7DDE437BA2ULL, 0xFCAF55C1BF8A4424ULL, 0x18FCF680573FA594ULL, 0x4C0563B89F495AC3ULL,
    0x40E087931A00930DULL, 0x8CFFA9412EB642C1ULL, 0x68CA39053261169FULL, 0x7A1EE967D27579E2ULL,
    0x9D1D60E5076F5B6FULL, 0x3810E399B6F65BA2ULL, 0x32095B6D4AB5F9B1ULL, 0x35CAB62109DD038AULL,
    0xA90B24499FCFAFB1ULL, 0x77A225A07CC2C6BDULL, 0x513E5E634C70E331ULL, 0x4361C0CA3F692F12ULL,
    0xD941ACA44B20A45BULL, 0x528F7C8602C5807BULL, 0x52AB92BEB9613989ULL, 0x9D1DFA2EFC557F73ULL,
    0x722FF175F572C348ULL, 0x1D1260A51107FE97ULL, 0x7A249A57EC0C9BA2ULL, 0x04208FE9E8F7F2D6ULL,
    0x5A110C6058B920A0ULL, 0x0CD9A497658A5698ULL, 0x56FD23C8F9715A4CULL, 0x284C847B9D887AAEULL,
    0x04FEABFBBDB619CBULL, 0x742E1E651C60BA83ULL, 0x9A9632E65904AD3CULL, 0x881B82A13B51B9E2ULL,
    0x506E6744CD974924ULL, 0xB0183DB56FFC6A79ULL, 0x0ED9B915C66ED37EULL, 0x5E11E86D5873D484ULL,
    0xF678647E3519AC6EULL, 0x1B85D488D0F20CC5ULL, 0xDAB9FE6525D89021ULL, 0x0D151D86ADB73615ULL,
    0xA865A54EDCC0F019ULL, 0x93C42566AEF98FFBULL, 0x99E7AFEABE000731ULL, 0x48CBFF086DDF285AULL,
    0x7F9B6AF1EBF78BAFULL, 0x58627E1A149BBA21ULL, 0x2CD16E2ABD791E33ULL, 0xD363EFF5F0977996ULL,
    0x0CE2A38C344A6EEDULL, 0x1A804AADB9CFA741ULL, 0x907F30421D78C5DEULL, 0x501F65EDB3034D07ULL,
    0x37624AE5A48FA6E9ULL, 0x957BAF61700CFF4EULL, 0x3A6C27934E31188AULL, 0xD49503536ABCA345ULL,
    0x088E049589C432E0ULL, 0xF943AEE7FEBF21B8ULL, 0x6C3B8E3E336139D3ULL, 0x364F6FFA464EE52EULL,
    0xD60F6DCEDC314222ULL, 0x56963B0DCA418FC0ULL, 0x16F50EDF91E513AFULL, 0xEF1955914B609F93ULL,
    0x565601C0364E3228ULL, 0xECB53939887E8175ULL, 0xBAC7A9A18531294BULL, 0xB344C470397BBA52ULL,
    0x65D34954DAF3CEBDULL, 0xB4B81B3FA97511E2ULL, 0xB422061193D6F6A7ULL, 0x071582401C38434DULL,
    0x7A13F18BBEDC4FF5ULL, 0xBC4097B116C524D2ULL, 0x59B97885E2F2EA28ULL, 0x99170A5DC3115544ULL,
    0x6F423357E7C6A9F9ULL, 0x325928EE6E6F8794ULL, 0xD0E4366228B03343ULL, 0x565C31F7DE89EA27ULL,
    0x30F5611484119414ULL, 0xD873DB391292ED4FULL, 0x7BD94E1D8E17DEBCULL, 0xC7D9F16864A76E94ULL,
    0x947AE053EE56E63CULL, 0xC8C93882F9475F5FULL, 0x3A9BF55BA91F81CAULL, 0xD9A11FBB3D98008EULL,
    0x0FD22063EDC29FCAULL, 0xB3F256D8ACA0B0B9ULL, 0xB03031A8B4516E84ULL, 0x35DD37D5871448AFULL,
    0xE9F6082B05542E4EULL, 0xEBFAFA33D7254B59ULL, 0x9255ABB50D532280ULL, 0xB9AB4CE57F2D34F3ULL,
    0x693501D628297551ULL, 0xC62C58F97DD949BFULL, 0xCD454F8F19C5126AULL, 0xBBE83F4ECC2BDECBULL,
    0xDC842B7E2819E230ULL, 0xBA89142E007503B8ULL, 0xA3BC941D0A5061CBULL, 0xE9F6760E32CD8021ULL,
    0x09C7E552BC76492FULL, 0x852F54934DA55CC9ULL, 0x8107FCCF064FCF56ULL, 0x098954D51FFF6580ULL,
    0x23B70EDB1955C4BFULL, 0xC330DE426430F69DULL, 0x4715ED43E8A45C0AULL, 0xA8D7E4DAB780A08DULL,
    0x0572B974F03CE0BBULL, 0xB57D2E985E1419C7ULL, 0xE8D9ECBE2CF3D73FULL, 0x2FE4B17170E59750ULL,
    0x11317BA87905E790ULL, 0x7FBF21EC8A1F45ECULL, 0x1725CABFCB045B00ULL, 0x964E915CD5E2B207ULL,
    0x3E2B8BCBF016D66DULL, 0xBE7444E39328A0ACULL, 0xF85B2B4FBCDE44B7ULL, 0x49353FEA39BA63B1ULL,
    0x1DD01AAFCD53486AULL, 0x1FCA8A92FD719F85ULL, 0xFC7C95D827357AFAULL, 0x18A6A990C8B35EBDULL,
    0xCCCB7005C6B9C28DULL, 0x3BDBB92C43B17F26ULL, 0xAA70B5B4F89695A2ULL, 0xE94C39A54A98307FULL,
    0xB7A0B174CFF6F36EULL, 0xD4DBA84729AF48ADULL, 0x2E18BC1AD9704A68ULL, 0x2DE0966DAF2F8B1CULL,
    0xB9C11D5B1E43A07EULL, 0x64972D68DEE33360ULL, 0x94628D38D0C20584ULL, 0xDBC0D2B6AB90A559ULL,
    0xD2733C4335C6A72FULL, 0x7E75D99D94A70F4DULL, 0x6CED1983376FA72BULL, 0x97FCAACBF030BC24ULL,
    0x7B77497B32503B12ULL, 0x8547EDDFB81CCB94ULL, 0x79999CDFF70902CBULL, 0xCFFE1939438E9B24ULL,
    0x829626E3892D95D7ULL, 0x92FAE24291F2B3F1ULL, 0x63E22C147B9C3403ULL, 0xC678B6D860284A1CULL,
    0x5873888850659AE7ULL, 0x0981DCD296A8736DULL, 0x9F65789A6509A440ULL, 0x9FF38FED72E9052FULL,
    0xE479EE5B9930578CULL, 0xE7F28ECD2D49EECDULL, 0x56C074A581EA17FEULL, 0x5544F7D774B14AEFULL,
    0x7B3F0195FC6F290FULL, 0x12153635B2C0CF57ULL, 0x7F5126DBBA5E0CA7ULL, 0x7A76956C3EAFB413ULL,
    0x3D5774A11D31AB39ULL, 0x8A1B083821F40CB4ULL, 0x7B4A38E32537DF62ULL, 0x950113646D1D6E03ULL,
    0x4DA8979A0041E8A9ULL, 0x3BC36E078F7515D7ULL, 0x5D0A12F27AD310D1ULL, 0x7F9D1A2E1EBE1327ULL,
    0xDA3A361B1C5157B1ULL, 0xDCDD7D20903D0C25ULL, 0x36833336D068F707ULL, 0xCE68341F79893389ULL,
    0xAB9090168DD05F34ULL, 0x43954B3252DC25E5ULL, 0xB438C2B67F98E5E9ULL, 0x10DCD78E3851A492ULL,
    0xDBC27AB5447822BFULL, 0x9B3CDB65F82CA382ULL, 0xB67B7896167B4C84ULL, 0xBFCED1B0048EAC50ULL,
    0xA9119B60369FFEBDULL, 0x1FFF7AC80904BF45ULL, 0xAC12FB171817EEE7ULL, 0xAF08DA9177DDA93DULL,
    0x1B0CAB936E65C744ULL, 0xB559EB1D04E5E932ULL, 0xC37B45B3F8D6F2BAULL, 0xC3A9DC228CAAC9E9ULL,
    0xF3B8B6675A6507FFULL, 0x9FC477DE4ED681DAULL, 0x67378D8ECCEF96CBULL, 0x6DD856D94D259236ULL,
    0xA319CE15B0B4DB31ULL, 0x073973751F12DD5EULL, 0x8A8E849EB32781A5ULL, 0xE1925C71285279F5ULL,
    0x74C04BF1790C0EFEULL, 0x4DDA48153C94938AULL, 0x9D266D6A1CC0542CULL, 0x7440FB816508C4FEULL,
    0x13328503DF48229FULL, 0xD6BF7BAEE43CAC40ULL, 0x4838D65F6EF6748FULL, 0x1E152328F3318DEAULL,
    0x8F8419A348F296BFULL, 0x72C8834A5957B511ULL, 0xD7A023A73260B45CULL, 0x94EBC8ABCFB56DAEULL,
    0x9FC10D0F989993E0ULL, 0xDE68A2355B93CAE6ULL, 0xA44CFE79AE538BBEULL, 0x9D1D84FCCE371425ULL,
    0x51D2B1AB2DDFB636ULL, 0x2FD7E4B9E72CD38CULL, 0x65CA5B96B7552210ULL, 0xDD69A0D8AB3B546DULL,
    0x604D51B25FBF70E2ULL, 0x73AA8A564FB7AC9EULL, 0x1A8C1E992B941148ULL, 0xAAC40A2703D9BEA0ULL,
    0x764DBEAE7FA4F3A6ULL, 0x1E99B96E70A9BE8BULL, 0x2C5E9DEB57EF4743ULL, 0x3A938FEE32D29981ULL,
    0x26E6DB8FFDF5ADFEULL, 0x469356C504EC9F9DULL, 0xC8763C5B08D1908CULL, 0x3F6C6AF859D80055ULL,
    0x7F7CC39420A3A545ULL, 0x9BFB227EBDF4C5CEULL, 0x89039D79D6FC5C5CULL, 0x8FE88B57305E2AB6ULL,
    0xA09E8C8C35AB96DEULL, 0xFA7E393983325753ULL, 0xD6B6D0ECC617C699ULL, 0xDFEA21EA9E7557E3ULL,
    0xB67C1FA481680AF8ULL, 0xCA1E3785A9E724E5ULL, 0x1CFC8BED0D681639ULL, 0xD18D8549D140CAEAULL,
    0x4ED0FE7E9DC91335ULL, 0xE4DBF0634473F5D2ULL, 0x1761F93A44D5AEFEULL, 0x53898E4C3910DA55ULL,
    0x734DE8181F6EC39AULL, 0x2680B122BAA28D97ULL, 0x298AF231C85BAFABULL, 0x7983EED3740847D5ULL,
    0x66C1A2A1A60CD889ULL, 0x9E17E49642A3E4C1ULL, 0xEDB454E7BADC0805ULL, 0x50B704CAB602C329ULL,
    0x4CC317FB9CDDD023ULL, 0x66B4835D9EAFEA22ULL, 0x219B97E26FFC81BDULL, 0x261E4E4C0A333A9DULL,
    0x1FE2CCA76517DB90ULL, 0xD7504DFA8816EDBBULL, 0xB9571FA04DC089C8ULL, 0x1DDC0325259B27DEULL,
    0xCF3F4688801EB9AAULL, 0xF4F5D05C10CAB243ULL, 0x38B6525C21A42B0EULL, 0x36F60E2BA4FA6800ULL,
    0xEB3593803173E0CEULL, 0x9C4CD6257C5A3603ULL, 0xAF0C317D32ADAA8AULL, 0x258E5A80C7204C4BULL,
    0x8B889D624D44885DULL, 0xF4D14597E660F855ULL, 0xD4347F66EC8941C3ULL, 0xE699ED85B0DFB40DULL,
    0x2472F6207C2D0484ULL, 0xC2A1E7B5B459AEB5ULL, 0xAB4F6451CC1D45ECULL, 0x63767572AE3D6174ULL,
    0xA59E0BD101731A28ULL, 0x116D0016CB948F09ULL, 0x2CF9C8CA052F6E9FULL, 0x0B090A7560A968E3ULL,
    0xABEEDDB2DDE06FF1ULL, 0x58EFC10B06A2068DULL, 0xC6E57A78FBD986E0ULL, 0x2EAB8CA63CE802D7ULL,
    0x14A195640116F336ULL, 0x7C0828DD624EC390ULL, 0xD74BBE77E6116AC7ULL, 0x804456AF10F5FB53ULL,
    0xEBE9EA2ADF4321C7ULL, 0x03219A39EE587A30ULL, 0x49787FEF17AF9924ULL, 0xA1E9300CD8520548ULL,
    0x5B45E522E4B1B4EFULL, 0xB49C3B3995091A36ULL, 0xD4490AD526F14431ULL, 0x12A8F216AF9418C2ULL,
    0x001F837CC7350524ULL, 0x1877B51E57A764D5ULL, 0xA2853B80F17F58EEULL, 0x993E1DE72D36D310ULL,
    0xB3598080CE64A656ULL, 0x252F59CF0D9F04BBULL, 0xD23C8E176D113600ULL, 0x1BDA0492E7E4586EULL,
    0x21E0BD5026C619BFULL, 0x3B097ADAF088F94EULL, 0x8D14DEDB30BE846EULL, 0xF95CFFA23AF5F6F4ULL,
    0x3871700761B3F743ULL, 0xCA672B91E9E4FA16ULL, 0x64C8E531BFF53B55ULL, 0x241260ED4AD1E87DULL,
    0x106C09B972D2E822ULL, 0x7FBA195410E5CA30ULL, 0x7884D9BC6CB569D8ULL, 0x0647DFEDCD894A29ULL,
    0x63573FF03E224774ULL, 0x4FC8E9560F91B123ULL, 0x1DB956E450275779ULL, 0xB8D91274B9E9D4FBULL,
    0xA2EBEE47E2FBFCE1ULL, 0xD9F1F30CCD97FB09ULL, 0xEFED53D75FD64E6BULL, 0x2E6D02C36017F67FULL,
    0xA9AA4D20DB084E9BULL, 0xB64BE8D8B25396C1ULL, 0x70CB6AF7C2D5BCF0ULL, 0x98F076A4F7A2322EULL,
    0xBF84470805E69B5FULL, 0x94C3251F06F90CF3ULL, 0x3E003E616A6591E9ULL, 0xB925A6CD0421AFF3ULL,
    0x61BDD1307C66E300ULL, 0xBF8D5108E27E0D48ULL, 0x240AB57A8B888B20ULL, 0xFC87614BAF287E07ULL,
    0xEF02CDD06FFDB432ULL, 0xA1082C0466DF6C0AULL, 0x8215E577001332C8ULL, 0xD39BB9C3A48DB6CFULL,
    0x2738259634305C14ULL, 0x61CF4F94C97DF93DULL, 0x1B6BACA2AE4E125BULL, 0x758F450C88572E0BULL,
    0x959F587D507A8359ULL, 0xB063E962E045F54DULL, 0x60E8ED72C0DFF5D1ULL, 0x7B64978555326F9FULL,
    0xFD080D236DA814BAULL, 0x8C90FD9B083F4558ULL, 0x106F72FE81E2C590ULL, 0x7976033A39F7D952ULL,
    0xA4EC0132764CA04BULL, 0x733EA705FAE4FA77ULL, 0xB4D8F77BC3E56167ULL, 0x9E21F4F903B33FD9ULL,
    0x9D765E419FB69F6DULL, 0xD30C088BA61EA5EFULL, 0x5D94337FBFAF7F5BULL, 0x1A4E4822EB4D7A59ULL,
    0x6FFE73E81B637FB3ULL, 0xDDF957BC36D8B9CAULL, 0x64D0E29EEA8838B3ULL, 0x08DD9BDFD96B9F63ULL,
    0x087E79E5A57D1D13ULL, 0xE328E230E3E2B3FBULL, 0x1C2559E30F0946BEULL, 0x720BF5F26F4D2EAAULL,
    0xB0774D261CC609DBULL, 0x443F64EC5A371195ULL, 0x4112CF68649A260EULL, 0xD813F2FAB7F5C5CAULL,
    0x660D3257380841EEULL, 0x59AC2C7873F910A3ULL, 0xE846963877671A17ULL, 0x93B633ABFA3469F8ULL,
    0xC0C0F5A60EF4CDCFULL, 0xCAF21ECD4377B28CULL, 0x57277707199B8175ULL, 0x506C11B9D90E8B1DULL,
    0xD83CC2687A19255FULL, 0x4A29C6465A314CD1ULL, 0xED2DF21216235097ULL, 0xB5635C95FF7296E2ULL,
    0x22AF003AB672E811ULL, 0x52E762596BF68235ULL, 0x9AEBA33AC6ECC6B0ULL, 0x944F6DE09134DFB6ULL,
    0x6C47BEC883A7DE39ULL, 0x6AD047C430A12104ULL, 0xA5B1CFDBA0AB4067ULL, 0x7C45D833AFF07862ULL,
    0x5092EF950A16DA0BULL, 0x9338E69C052B8E7BULL, 0x455A4B4CFE30E3F5ULL, 0x6B02E63195AD0CF8ULL,
    0x6B17B224BAD6BF27ULL, 0xD1E0CCD25BB9C169ULL, 0xDE0C89A556B9AE70ULL, 0x50065E535A213CF6ULL,
    0x9C1169FA2777B874ULL, 0x78EDEFD694AF1EEDULL, 0x6DC93D9526A50E68ULL, 0xEE97F453F06791EDULL,
    0x32AB0EDB696703D3ULL, 0x3A6853C7E70757A7ULL, 0x31865CED6120F37DULL, 0x67FEF95D92607890ULL,
    0x1F2B1D1F15F6DC9CULL, 0xB69E38A8965C6B65ULL, 0xAA9119FF184CCCF4ULL, 0xF43C732873F24C13ULL,
    0xFB4A3D794A9A80D2ULL, 0x3550C2321FD6109CULL, 0x371F77E76BB8417EULL, 0x6BFA9AAE5EC05779ULL,
    0xCD04F3FF001A4778ULL, 0xE3273522064480CAULL, 0x9F91508BFFCFC14AULL, 0x049A7F41061A9E60ULL,
    0xFCB6BE43A9F2FE9BULL, 0x08DE8A1C7797DA9BULL, 0x8F9887E6078735A1ULL, 0xB5B4071DBFC73A66ULL,
    0x230E343DFBA08D33ULL, 0x43ED7F5A0FAE657DULL, 0x3A88A0FBBCB05C63ULL, 0x21874B8B4D2DBC4FULL,
    0x1BDEA12E35F6A8C9ULL, 0x53C065C6C8E63528ULL, 0xE34A1D250E7A8D6BULL, 0xD6B04D3B7651DD7EULL,
    0x5E90277E7CB39E2DULL, 0x2C046F22062DC67DULL, 0xB10BB459132D0A26ULL, 0x3FA9DDFB67E2F199ULL,
    0x0E09B88E1914F7AFULL, 0x10E8B35AF3EEAB37ULL, 0x9EEDECA8E272B933ULL, 0xD4C718BC4AE8AE5FULL,
    0x81536D601170FC20ULL, 0x91B534F885818A06ULL, 0xEC8177F83F900978ULL, 0x190E714FADA5156EULL,
    0xB592BF39B0364963ULL, 0x89C350C893AE7DC1ULL, 0xAC042E70F8B383F2ULL, 0xB49B52E587A1EE60ULL,
    0xFB152FE3FF26DA89ULL, 0x3E666E6F69AE2C15ULL, 0x3B544EBE544C19F9ULL, 0xE805A1E290CF2456ULL,
    0x24B33C9D7ED25117ULL, 0xE74733427B72F0C1ULL, 0x0A804D18B7097475ULL, 0x57E3306D881EDB4FULL,
    0x4AE7D6A36EB5DBCBULL, 0x2D8D5432157064C8ULL, 0xD1E649DE1E7F268BULL, 0x8A328A1CEDFE552CULL,
    0x07A3AEC79624C7DAULL, 0x84547DDC3E203C94ULL, 0x990A98FD5071D263ULL, 0x1A4FF12616EEFC89ULL,
    0xF6F7FD1431714200ULL, 0x30C05B1BA332F41CULL, 0x8D2636B81555A786ULL, 0x46C9FEB55D120902ULL,
    0xCCEC0A73B49C9921ULL, 0x4E9D2827355FC492ULL, 0x19EBB029435DCB0FULL, 0x4659D2B743848A2CULL,
    0x963EF2C96B33BE31ULL, 0x74F85198B05A2E7DULL, 0x5A0F544DD2B1FB18ULL, 0x03727073C2E134B1ULL,
    0xC7F6AA2DE59AEA61ULL, 0x352787BAA0D7C22FULL, 0x9853EAB63B5E0B35ULL, 0xABBDCDD7ED5C0860ULL,
    0xCF05DAF5AC8D77B0ULL, 0x49CAD48CEBF4A71EULL, 0x7A4C10EC2158C4A6ULL, 0xD9E92AA246BF719EULL,
    0x13AE978D09FE5557ULL, 0x730499AF921549FFULL, 0x4E4B705B92903BA4ULL, 0xFF577222C14F0A3AULL,
    0x55B6344CF97AAFAEULL, 0xB862225B055B6960ULL, 0xCAC09AFBDDD2CDB4ULL, 0xDAF8E9829FE96B5FULL,
    0xB5FDFC5D3132C498ULL, 0x310CB380DB6F7503ULL, 0xE87FBB46217A360EULL, 0x2102AE466EBB1148ULL,
    0xF8549E1A3AA5E00DULL, 0x07A69AFDCC42261AULL, 0xC4C118BFE78FEAAEULL, 0xF9F4892ED96BD438ULL,
    0x1AF3DBE25D8F45DAULL, 0xF5B4B0B0D2DEEEB4ULL, 0x962ACEEFA82E1C84ULL, 0x046E3ECAAF453CE9ULL,
    0xF05D129681949A4CULL, 0x964781CE734B3C84ULL, 0x9C2ED44081CE5FBDULL, 0x522E23F3925E319EULL,
    0x177E00F9FC32F791ULL, 0x2BC60A63A6F3B3F2ULL, 0x222BBFAE61725606ULL, 0x486289DDCC3D6780ULL,
    0x7DC7785B8EFDFC80ULL, 0x8AF38731C02BA980ULL, 0x1FAB64EA29A2DDF7ULL, 0xE4D9429322CD065AULL,
    0x9DA058C67844F20CULL, 0x24C0E332B70019B0ULL, 0x233003B5A6CFE6ADULL, 0xD586BD01C5C217F6ULL,
    0x5E5637885F29BC2BULL, 0x7EBA726D8C94094BULL, 0x0A56A5F0BFE39272ULL, 0xD79476A84EE20D06ULL,
    0x9E4C1269BAA4BF37ULL, 0x17EFEE45B0DEE640ULL, 0x1D95B0A5FCF90BC6ULL, 0x93CBE0B699C2585DULL,
    0x65FA4F227A2B6D79ULL, 0xD5F9E858292504D5ULL, 0xC2B5A03F71471A6FULL, 0x59300222B4561E00ULL,
    0xCE2F8642CA0712DCULL, 0x7CA9723FBB2E8988ULL, 0x2785338347F2BA08ULL, 0xC61BB3A141E50E8CULL,
    0x150F361DAB9DEC26ULL, 0x9F6A419D382595F4ULL, 0x64A53DC924FE7AC9ULL, 0x142DE49FFF7A7C3DULL,
    0x0C335248857FA9E7ULL, 0x0A9C32D5EAE45305ULL, 0xE6C42178C4BBB92EULL, 0x71F1CE2490D20B07ULL,
    0xF1BCC3D275AFE51AULL, 0xE728E8C83C334074ULL, 0x96FBF83A12884624ULL, 0x81A1549FD6573DA5ULL,
    0x5FA7867CAF35E149ULL, 0x56986E2EF3ED091BULL, 0x917F1DD5F8886C61ULL, 0xD20D8C88C8FFE65FULL,
    0x31D71DCE64B2C310ULL, 0xF165B587DF898190ULL, 0xA57E6339DD2CF3A0ULL, 0x1EF6E6DBB1961EC9ULL,
    0x70CC73D90BC26E24ULL, 0xE21A6B35DF0C3AD7ULL, 0x003A93D8B2806962ULL, 0x1C99DED33CB890A1ULL,
    0xCF3145DE0ADD4289ULL, 0xD0E4427A5514FB72ULL, 0x77C621CC9FB3A483ULL, 0x67A34DAC4356550BULL,
    0xF8D626AAAF278509ULL
};

// Polyglot piece mapping: Luminex piece -> Polyglot piece index
// Luminex: WHITE_PAWN=0..WHITE_KING=5, BLACK_PAWN=6..BLACK_KING=11
// Polyglot: white_pawn=1, white_knight=3, white_bishop=5, white_rook=7, white_queen=9, white_king=11
//           black_pawn=0, black_knight=2, black_bishop=4, black_rook=6, black_queen=8, black_king=10
static const int POLYGLOT_PIECE[12] = {
    1, 3, 5, 7, 9, 11,   // WHITE
    0, 2, 4, 6, 8, 10    // BLACK
};

uint64_t compute_polyglot_key(const Position& pos) {
    uint64_t key = 0;

    for (int sq = 0; sq < 64; sq++) {
        Piece p = pos.piece_on(Square(sq));
        if (p != NO_PIECE) {
            int kind = POLYGLOT_PIECE[int(p)];
            int file = sq & 7;
            int row = sq >> 3;
            key ^= POLYGLOT_RANDOM[64 * kind + 8 * row + file];
        }
    }

    // Castling
    if (pos.castling_allowed(WHITE, WHITE_KINGSIDE))  key ^= POLYGLOT_RANDOM[768 + 0];
    if (pos.castling_allowed(WHITE, WHITE_QUEENSIDE)) key ^= POLYGLOT_RANDOM[768 + 1];
    if (pos.castling_allowed(BLACK, BLACK_KINGSIDE))  key ^= POLYGLOT_RANDOM[768 + 2];
    if (pos.castling_allowed(BLACK, BLACK_QUEENSIDE)) key ^= POLYGLOT_RANDOM[768 + 3];

    // En passant (only if a pawn can actually capture)
    Square ep = pos.ep_square();
    if (ep != SQUARE_NONE) {
        key ^= POLYGLOT_RANDOM[772 + (ep & 7)];
    }

    // Side to move: Polyglot hashes white's turn as XOR of Random[780]
    // (opposite of python-chess which XORs for BLACK)
    if (pos.side_to_move() == WHITE) key ^= POLYGLOT_RANDOM[780];

    return key;
}

Book::Entry Book::read_entry(int index) {
    fseek(file_, index * 16, SEEK_SET);
    Entry e{};  // zero-initialised (key=0 signals "not found" on truncated read)
    // Polyglot book entries are stored big-endian. Read raw and byte-swap on
    // little-endian hosts (all current CI/cloud targets); pass through on big-endian.
    uint64_t k = 0; uint16_t mv = 0, wt = 0; uint32_t ln = 0;
    std::size_t got = 0;
    got += std::fread(&k,  8, 1, file_);
    got += std::fread(&mv, 2, 1, file_);
    got += std::fread(&wt, 2, 1, file_);
    got += std::fread(&ln, 4, 1, file_);
    if (got != 4) return e;  // truncated record / EOF
    if constexpr (std::endian::native == std::endian::little) {
        e.key    = std::byteswap(k);
        e.move   = std::byteswap(mv);
        e.weight = std::byteswap(wt);
        e.learn  = std::byteswap(ln);
    } else {
        e.key = k; e.move = mv; e.weight = wt; e.learn = ln;
    }
    return e;
}

int Book::find_pos(uint64_t key) {
    if (size_ == 0) return 0;
    int left = 0, right = size_ - 1;
    while (left < right) {
        int mid = (left + right) / 2;
        if (key <= read_entry(mid).key) right = mid;
        else left = mid + 1;
    }
    Entry e = read_entry(left);
    return (e.key == key) ? left : size_;
}

bool Book::open(const char* path) {
    close();
    file_ = fopen(path, "rb");
    if (!file_) return false;
    fseek(file_, 0, SEEK_END);
    size_ = ftell(file_) / 16;
    return size_ > 0;
}

void Book::close() {
    if (file_) { fclose(file_); file_ = nullptr; }
    size_ = 0;
}

std::string Book::probe(const Position& pos) {
    if (!file_ || size_ == 0) return "";

    uint64_t key = compute_polyglot_key(pos);
    int pos_idx = find_pos(key);

    uint16_t best_move = 0;
    int best_score = 0;

    for (int i = pos_idx; i < size_; i++) {
        Entry e = read_entry(i);
        if (e.key != key) break;
        if (e.weight == 0) continue;
        best_score += e.weight;
        if ((rand() % best_score) < e.weight) best_move = e.move;
    }

    if (best_move == 0) return "";

    // Decode Polyglot move to UCI string
    int to_file   = best_move & 7;
    int to_row    = (best_move >> 3) & 7;
    int from_file = (best_move >> 6) & 7;
    int from_row  = (best_move >> 9) & 7;
    int promo     = (best_move >> 12) & 7;

    std::string uci;
    uci += char('a' + from_file);
    uci += char('1' + from_row);
    uci += char('a' + to_file);
    uci += char('1' + to_row);
    if (promo) uci += "nbrq"[promo - 1];
    return uci;
}

} // namespace luminex

// =====================================================================================================================
// main
// =====================================================================================================================

#include <chrono>
#include <iostream>
#include <cstdio>
#if defined(LUMINEX_STACKTRACE)
#include <csignal>
#include <stacktrace>
#include <string>
#endif

namespace luminex {

void init() {
    init_magic_bitboards();
    init_zobrist();
    init_evaluation();
    // SPSA overrides from the working directory; UCI options set later win
    SPSAParams::get().load("spsa_params.txt");
    init_search_params();
    // The transposition table (default TranspositionTable::DEFAULT_MB = 256 MB,
    // adopted 2026-08-19: self-play +16.7 LOS 89% over 128 MB) is allocated
    // on first use, so a GUI's Hash setting replaces it before it is touched.
}

// Perft for testing
uint64_t perft(Position& pos, Depth depth) {
    if (depth == 0) return 1;

    ExtMove moves[MAX_MOVES];
    ExtMove* end = generate<GEN_LEGAL>(pos, moves);

    uint64_t count = 0;
    for (ExtMove* it = moves; it != end; ++it) {
        if (!pos.do_move(it->move)) {
            // Should never happen with legal move generation
            // Do NOT call undo_move - do_move guarantees atomic failure
            continue;
        }
        count += perft(pos, depth - 1);
        pos.undo_move(it->move);
    }

    return count;
}

// Perft divide - shows perft result for each move
uint64_t perft_divide(Position& pos, Depth depth) {
    ExtMove moves[MAX_MOVES];
    ExtMove* end = generate<GEN_LEGAL>(pos, moves);

    uint64_t total = 0;
    for (ExtMove* it = moves; it != end; ++it) {
        if (!pos.do_move(it->move)) {
            // Should never happen with legal move generation
            // Do NOT call undo_move - do_move guarantees atomic failure
            continue;
        }
        uint64_t count = (depth == 1) ? 1 : perft(pos, depth - 1);
        pos.undo_move(it->move);

        std::cout << it->move << ": " << count << std::endl;
        total += count;
    }
    return total;
}

#if defined(LUMINEX_STACKTRACE)
// A debugging aid, never in a release (cmake -DLUMINEX_STACKTRACE=ON): a crash
// prints the crashing thread's stack to stderr, then the default action ends
// the process as before. Formatting a trace is not async-signal-safe, which a
// process that is already dying can afford; a stack overflow, whose handler
// would need the very stack that ran out, is not traced.
static void crash_trace(int sig) {
    std::signal(sig, SIG_DFL);
    const std::string trace = std::to_string(std::stacktrace::current());
    std::fprintf(stderr, "luminex: signal %d, stack:\n%s\n", sig, trace.c_str());
    std::fflush(stderr);
    std::raise(sig);
}
#endif

} // namespace luminex

// genfens N seed S book <path|None> [net PATH] [plies K] [maxplies K2] [dfrc P]
//         [dfrcplies D] [dfrcmaxplies D2] [maxscore M] [nodes X]
// OpenBench's DATAGEN workloads ask the engine for N opening positions and read
// the lines "info string genfens <fen>"; fastchess then plays the games. Each
// opening is a random line of the book (or the start position) or, P percent
// of the time, a Double Fischer Random start with its castling rights cleared
// (castling here is standard chess only), followed by K to K2 random legal
// plies (default 8 to 9, so either side may move first; D to D2 after a DFRC
// start, by default the same). An opening with no
// legal move, or whose X-node search scores beyond M centipawns either way, is
// drawn again. The search uses the network at PATH (OpenBench's client keeps
// the workload's network at Networks/<id> beside the engine's working
// directory); without one it would be the HCE's, which passed half of the
// openings 05q scores beyond 300. The seed fixes every choice. Exactly N lines
// are printed: after 200 tries per line the filter is dropped rather than
// stall the client.
static int run_genfens(const std::string& command) {
    using namespace luminex;
    std::istringstream in(command);
    std::string tok, book = "None";
    long long n = 0;
    uint64_t seed = 0;
    std::string net;
    int plies = 8, max_plies = -1, dfrc = 10, max_score = 300, filter_nodes = 10000;
    int dfrc_plies = -1, dfrc_max_plies = -1;
    in >> tok >> n;
    while (in >> tok) {
        if (tok == "seed") in >> seed;
        else if (tok == "book") in >> book;
        else if (tok == "net") in >> net;
        else if (tok == "plies") in >> plies;
        else if (tok == "maxplies") in >> max_plies;
        else if (tok == "dfrcplies") in >> dfrc_plies;
        else if (tok == "dfrcmaxplies") in >> dfrc_max_plies;
        else if (tok == "dfrc") in >> dfrc;
        else if (tok == "maxscore") in >> max_score;
        else if (tok == "nodes") in >> filter_nodes;
    }
    plies = std::clamp(plies, 0, 40);
    max_plies = max_plies < 0 ? plies + 1 : std::clamp(max_plies, plies, 40);
    dfrc_plies = dfrc_plies < 0 ? plies : std::clamp(dfrc_plies, 0, 40);
    dfrc_max_plies = dfrc_max_plies < 0 ? std::max(dfrc_plies, max_plies) : std::clamp(dfrc_max_plies, dfrc_plies, 40);
    if (!net.empty() && !load_net_file(net)) {
        std::cerr << "genfens: the network " << net << " did not load" << std::endl;
        return 1;
    }
    dfrc = std::clamp(dfrc, 0, 100);
    filter_nodes = std::clamp(filter_nodes, 100, 10000000);
    std::vector<std::string> lines;
    if (book != "None" && book != "none") {
        std::ifstream f(book);
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream fields(line);
            std::string board, side, castling, ep;
            if (fields >> board >> side >> castling >> ep && std::count(board.begin(), board.end(), '/') == 7)
                lines.push_back(board + " " + side + " " + castling + " " + ep + " 0 1");
        }
        if (lines.empty()) {
            std::cerr << "genfens: no positions in book " << book << std::endl;
            return 1;
        }
    }
    uint64_t state = seed ^ 0x6a09e667f3bcc909ULL;
    auto next = [&state]() {   // splitmix64
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    };
    auto below = [&next](uint64_t k) { return k ? next() % k : 0; };
    // A Double Fischer Random start: each side's back rank drawn on its own
    // (bishops on opposite colours, the king between the rooks), no castling
    auto dfrc_start = [&]() {
        std::string ranks[2];
        for (std::string& r : ranks) {
            r = std::string(8, ' ');
            r[2 * below(4)] = 'b';
            r[2 * below(4) + 1] = 'b';
            auto put = [&r, &below](char piece) {
                int empty = 0;
                for (char c : r) empty += c == ' ';
                int k = int(below(uint64_t(empty)));
                for (char& c : r)
                    if (c == ' ' && k-- == 0) { c = piece; return; }
            };
            put('q');
            put('n');
            put('n');
            const char rest[3] = {'r', 'k', 'r'};
            int i = 0;
            for (char& c : r)
                if (c == ' ') c = rest[i++];
        }
        std::string white = ranks[0];
        for (char& c : white) c = char(std::toupper(c));
        return ranks[1] + "/pppppppp/8/8/8/8/PPPPPPPP/" + white + " w - - 0 1";
    };
    int status = 0;
    // Searches run on a SearchThread, as every go does: a Position holds about
    // 230 KB, and the main thread has 1 MiB on Windows
    SearchThread job([&] {
        TT.resize(16);
        for (long long produced = 0; produced < n; ++produced) {
            for (int attempt = 0;; ++attempt) {
                const bool filtered = attempt < 200;
                Position p;
                const bool use_dfrc = int(below(100)) < dfrc;
                p.set(use_dfrc ? dfrc_start()
                               : lines.empty() ? std::string("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
                                               : lines[below(lines.size())]);
                bool dead = false;
                const int lo = use_dfrc ? dfrc_plies : plies, hi = use_dfrc ? dfrc_max_plies : max_plies;
                const int k = lo + int(below(uint64_t(hi - lo + 1)));
                for (int i = 0; i < k && !dead; ++i) {
                    ExtMove moves[MAX_MOVES];
                    ExtMove* end = generate<GEN_LEGAL>(p, moves);
                    if (end == moves || !p.do_move(moves[below(uint64_t(end - moves))].move)) dead = true;
                }
                ExtMove moves[MAX_MOVES];
                if (dead || generate<GEN_LEGAL>(p, moves) == moves) continue;
                if (filtered) {
                    TT.clear();
                    TT.new_search();
                    clear_correction_history();
                    clear_search_history();
                    Position s = p;
                    Limits lim;
                    lim.nodes = filter_nodes;
                    stop = false;
                    search(s, lim);
                    if (std::abs(int(root_score)) > max_score) continue;
                }
                std::cout << "info string genfens " << p.fen() << std::endl;
                break;
            }
        }
    });
    job.join();
    return status;
}

int main(int argc, char* argv[]) {
    using namespace luminex;
#if defined(LUMINEX_STACKTRACE)
    for (int sig : {SIGSEGV, SIGABRT, SIGFPE, SIGILL}) std::signal(sig, crash_trace);
#endif

    // CRITICAL: Unbuffered stdout for immediate UCI response
    // Keep sync_with_stdio TRUE for proper getline/stdin interaction on Windows
    setvbuf(stdout, NULL, _IONBF, 0);
    std::ios_base::sync_with_stdio(true);

    // Initialize engine
    init();

    // Check for debug mode
    if (argc > 1 && std::string(argv[1]) == "debug") {
        Position pos;
        pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        std::cout << "FEN: " << pos.fen() << std::endl;
        std::cout << "piece_on(A1): " << int(pos.piece_on(A1)) << std::endl;
        std::cout << "piece_on(A8): " << int(pos.piece_on(A8)) << std::endl;
        std::cout << "piece_on(A2): " << int(pos.piece_on(A2)) << std::endl;
        std::cout << "piece_on(H8): " << int(pos.piece_on(H8)) << std::endl;
        return 0;
    }

    // Check for eval mode
    if (argc > 1 && std::string(argv[1]) == "eval") {
        Position pos;
        if (argc > 2) {
            // Use provided FEN
            std::string fen;
            for (int i = 2; i < argc; ++i) {
                if (i > 2) fen += " ";
                fen += argv[i];
            }
            pos.set(fen);
        } else {
            pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        }
        std::cout << "FEN: " << pos.fen() << std::endl;
        std::cout << "Static eval: " << evaluate(pos) << std::endl;
        return 0;
    }

    // Check for bench mode
    if (argc > 1 && std::string(argv[1]) == "bench") {
        Position pos;
        pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        using namespace std::chrono;
        auto start = high_resolution_clock::now();

        uint64_t p1 = perft(pos, 1);
        std::cout << "Perft 1: " << p1 << " (expected 20)" << std::endl;

        uint64_t p2 = perft(pos, 2);
        std::cout << "Perft 2: " << p2 << " (expected 400)" << std::endl;

        uint64_t p3 = perft(pos, 3);
        std::cout << "Perft 3: " << p3 << " (expected 8902)" << std::endl;

        uint64_t p4 = perft(pos, 4);
        std::cout << "Perft 4: " << p4 << " (expected 197281)" << std::endl;

        auto end = high_resolution_clock::now();
        auto ms = duration_cast<milliseconds>(end - start).count();
        std::cout << "Time: " << ms << "ms" << std::endl;

        if (p1 == 20 && p2 == 400 && p3 == 8902 && p4 == 197281) {
            std::cout << "ALL PERFT TESTS PASSED" << std::endl;
        } else {
            std::cout << "PERFT TESTS FAILED!" << std::endl;
            return 1;
        }

        // OpenBench bench contract, two lines, printed LAST so the client's
        // reverse-order parser picks them over any search info lines:
        //   "Nodes searched" = the PERFT total — integer-only, identical on
        //     every ISA, which is what makes it the cross-worker determinism
        //     check (a search-bench node count would differ between
        //     AVX2/AVX-512/ARM builds and get workers blacklisted).
        //   "NPS" = measured SEARCH speed from a short depth-limited probe.
        //     Bench runs net-less (HCE eval), so this is a speed probe, not a
        //     strength measurement — the client reports it as the machine's
        //     engine speed, which drives time-control scaling per worker.
        {
            Position probe;
            probe.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            Limits lim;
            lim.depth = 11;
            // Allocate the hash table before the clock starts: the first search
            // would otherwise allocate it inside the timed probe and report a
            // speed that includes zeroing 256 MB, which OpenBench then scales
            // every time control by.
            TT.ensure_allocated();
            stop = false;   // search() leaves the stop flags to its caller
            auto t0 = high_resolution_clock::now();
            search(probe, lim);
            auto sms = duration_cast<milliseconds>(high_resolution_clock::now() - t0).count();
            const uint64_t total_nodes = p1 + p2 + p3 + p4;
            std::cout << "Nodes searched " << total_nodes << std::endl;
            if (sms > 0)
                std::cout << "NPS " << (nodes.load() * 1000ULL / (uint64_t)sms) << std::endl;
        }
        return 0;
    }

    // OpenBench's DATAGEN: "genfens N seed S book <path|None> ..." as one argument
    if (argc > 1 && std::string(argv[1]).rfind("genfens", 0) == 0)
        return run_genfens(argv[1]);

    // nnuecheck [NETFILE]: every SIMD kernel level must evaluate alike (CI
    // runs it on each release binary, and under Intel SDE for the levels the
    // runner's CPU lacks); the exit status says whether they did.
    if (argc > 1 && std::string(argv[1]) == "nnuecheck")
        return run_nnuecheck(argc > 2 ? argv[2] : "") ? 0 : 1;

    // Check for divide mode
    if (argc > 1 && std::string(argv[1]) == "divide") {
        int depth = (argc > 2) ? std::atoi(argv[2]) : 3;
        Position pos;
        pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        std::cout << "Divide perft depth " << depth << ":" << std::endl;
        uint64_t total = perft_divide(pos, depth);
        std::cout << "Total: " << total << " nodes" << std::endl;
        return 0;
    }

    // Enter UCI loop
    uci_loop();

    return 0;
}
