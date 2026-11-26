// nnue.cpp — NNUE evaluation with an incremental accumulator.
//
// A full feature-transformer rebuild per eval is ~15x slower than this design
// (measured 135K nps on the first prototype). The accumulator lives in a
// per-thread stack (acc_stack[], indexed by Position::st_ply()) and is updated
// incrementally on make_move, so evaluate() just reads it. The debug self-check
// validates it against the full-refresh reference exactly.
//
// The accumulator lives in a thread_local global (NOT in StateInfo) — putting it in
// StateInfo bloated state_stack[2048] to ~8MB and overflowed the stack (Position is
// stack-local). thread_local also makes it safe across Luminex's helper threads.
//
// HalfKAv2_hm feature indexer mirrored EXACTLY from Luminex-NNUE's trainer/luminex_nnue_train.py.
//
// The loops over lanes and the forward pass are the kernels of
// nnue_kernels.h. Built as usual they are compiled into this file for the
// instruction-set level the compiler targets and called directly; the
// universal x86-64 build (CMake LUMINEX_DISPATCH) carries one set per level,
// each compiled on its own, and this file calls the best set the CPU supports
// (see "Kernel levels" below).
#include "luminex.h"
#include "nnue_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <istream>
#include <memory>
#include <streambuf>
#include <string>
#include <thread>
#include <vector>
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>   // _mm_prefetch needs xmmintrin even without AVX2 (Debug builds)
#endif

// =====================================================================================================================
// The network built into the binary (CMake LUMINEX_EMBED_NET; the code is at the end of this file)
// =====================================================================================================================

#include <cstddef>

namespace luminex {

// The bytes of the network file built into this binary, or {nullptr, 0} when
// the build carries none (the CMake default, and OpenBench's Makefile build).
struct EmbeddedNet {
    const unsigned char* data;
    std::size_t size;
};

EmbeddedNet embedded_net();

} // namespace luminex

// =====================================================================================================================
// gen768t's relation inputs (the code is after the network's)
// =====================================================================================================================

// Attack relations and pawn pairs (docs/NNUE.md, V5-GEN768T section 2). Which rows a position has, and which
// rows a move adds and removes, for one perspective or both. The rules and the index layout (LAYOUT = 1) are
// below; the network code sums the rows into the relation lanes.
//
// Frame. Squares are a1 = 0 ... h8 = 63. For perspective P with its own king on k:
// - o(s) = s XOR (56 if P is Black) XOR (7 if k is on files a-d), group 1's orientation;
// - a piece's side is own (colour P) or their;
// - own pawns capture toward rank 8 of the oriented board, their pawns toward rank 1.
//
// Group 2: which attacks are features. A relation (a on x) -> (v on y) holds when a attacks square y and v stands
// on it: a pawn attacks its two capture squares, a knight or king its jumps, a bishop, rook or queen each of its
// lines up to and including the first occupied square. a and v may be of one colour. A relation is not a feature
// when another input always carries it:
// 1. v is a king.
// 2. a is the perspective's own king (group 1's king bucket and v's piece-square carry it exactly).
// 3. a and v are both pawns (group 3).
// 4. Mutual attacks are told once:
//    - a and v of one type among N, B, R, Q: on one side, one feature per unordered pair; across sides, the feature
//      whose attacker is the own piece.
//    - Q -> B along a diagonal (the feature is B -> Q); Q -> R along a rank or file (the feature is R -> Q).
//    - B -> P or Q -> P when that pawn attacks the bishop's or queen's square (the feature is P -> B or P -> Q).
//
// Group 2: index. Attacker class a = (side, type), victim class v = (side, type), oriented squares x' = o(x), y' =
// o(y). Before indexing, a pair of one type on one side is ordered x' < y', and a pair of one type with a their
// attacker and an own victim swaps roles.
//
// E(a)[x'] is the set of squares class a attacks from x' on an empty board (pawns stand on ranks 2-7 only).
// G(a->v)[x'] is E(a)[x'] restricted as follows:
// - v a pawn: ranks 2-7 only;
// - Q -> B: rank and file lines only; Q -> R: diagonals only;
// - B -> P and Q -> P: without the squares from which that pawn attacks x' (an own pawn: the two squares diagonally
//   below x'; their pawn: the two diagonally above);
// - one type, one side: only y' > x'.
//
// index = base(a->v) + off(a->v)[x'] + popcount( G(a->v)[x'] & (bit(y') - 1) )
// off(a->v)[x'] = sum of |G(a->v)[u]| over u < x'
// base(a->v)    = sum of the sizes of all earlier blocks
//
// Block order: attacker side (own, their), attacker type (P, N, B, R, Q, K), victim side (own, their), victim type
// (P, N, B, R, Q); a block with no feature is skipped. Block sizes (rows):
//
//   attacker \ victim  own P  own N  own B  own R  own Q  their P  their N  their B  their R  their Q    rows
//   own P                  -     84     84     84     84        -       84       84       84       84     672
//   own N                284    168    336    336    336      284      336      336      336      336   3,088
//   own B                364    560    280    560    560      364      560      560      560      560   4,928
//   own R                672    896    896    448    896      672      896      896      896      896   8,064
//   own Q              1,036  1,456    896    560    728    1,036    1,456      896      560    1,456  10,080
//   their P                -     84     84     84     84        -       84       84       84       84     672
//   their N              284      -    336    336    336      284      168      336      336      336   2,752
//   their B              364    560      -    560    560      364      560      280      560      560   4,368
//   their R              672    896    896      -    896      672      896      896      448      896   7,168
//   their Q            1,036  1,456    896    560      -    1,036    1,456      896      560      728   8,624
//   their K              348    420    420    420    420      348      420      420      420      420   4,056
//
// Total 54,472; every row is reached by exactly one (class pair, x', y').
//
// Group 3: pawn pairs. A pawn's id is 48 * side + o(s) - 8 (0-95). A pair is two pawns on different squares whose
// files differ by at most one: 480 same-file and 1,008 adjacent-file rows.
//
// index = 54472 + rank of (lower id, higher id) among all valid pairs in lexicographic order
//
// - PPM = 0: a pair is active only when one pawn attacks the other's square: an own pawn defending an own pawn (70
//   rows), their pawn defending their pawn (70), an own and their pawn attacking each other (70).
// - PPM = 1: every pair on the board is active.
//
// What a move changes. Three steps on single squares; each emits removals (-) and additions (+) of physical
// features, and each perspective maps them to its own indices (a relation that is not a feature in that perspective
// is skipped).
//
// remove(z):     - every relation the piece on z attacks in, - every relation it is the victim of,
//                - its pawn pairs; empty z;
//                + (s -> w) for every slider s whose line reaches z, w the first piece beyond z
// add(z, p):     - (s -> w) for every slider s whose line reaches z, w the first piece beyond z;
//                put p on z;
//                + every relation p attacks in, + every relation it is the victim of, + its pawn pairs
// replace(z, p): - the old piece's relations and pawn pairs; put p on z; + the new piece's
//                (the occupancy is unchanged, so no line changes)
//
// A slider's line reaches z when z is on one of its lines with only empty squares between. Every + and - applies
// the rules above (kings as victims, mutual attacks told once, pawn on pawn as a pair).
//
//   Move                            Steps, in order
//   Quiet                           remove(from), add(to, piece)
//   Capture                         remove(from), replace(to, piece)
//   En passant                      remove(from), remove(captured pawn's square), add(to, pawn)
//   Promotion                       remove(from), then add(to, promoted piece) or replace(to, promoted piece)
//   Castling                        remove(king from), remove(rook from), add(king to, king), add(rook to, rook)
//   Null move                       nothing
//   Own king crosses the d/e files  the mover's perspective: every index changes, groups 2 and 3 are
//                                   rebuilt from the board; the other perspective: the steps above
//
// A feature a move both adds and removes may be cancelled or applied twice: the sums are exact either way.
//
// Conformance. White's view, then Black's: active features of groups 2 and 3, and the sum of their indices.
//
//   active features / sum of indices: PPM = 0 White, Black; PPM = 1 White, Black
//   start position:
//       33 / 923,020, 33 / 923,020; 69 / 2,897,986, 69 / 2,897,986
//   r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -:
//       47 / 1,268,017, 47 / 1,275,960; 77 / 2,913,373, 77 / 2,925,714
//   2kr3r/ppp2ppp/2n1bn2/2b1p3/4P3/2NP1N2/PPPB1PPP/2KR1B1R b - - (both kings mirrored):
//       34 / 762,273, 35 / 866,598; 63 / 2,353,560, 64 / 2,457,869
//   8/5pk1/6p1/8/3R4/6P1/r4PK1/8 w - -:
//       5 / 250,781, 5 / 229,138; 9 / 469,911, 9 / 448,268
//
// Single features, in the third position unless named:
//
//   Feature                                             View   Oriented    Block               base + off + rank
//   Ne5 attacks pawn d7 (second position)               White  e5 -> d7    own N -> their P    2132 + 168 + 6 = 2306
//   the same                                            Black  e4 -> d2    their N -> own P    27504 + 116 + 0 = 27620
//   Bc5 attacks pawn f2                                 White  f5 -> c2    their B -> own P    30256 + 228 + 0 = 30484
//   the same                                            Black  f4 -> c7    own B -> their P    6084 + 152 + 6 = 6242
//   Rd1 defends Bd2                                     White  e1 -> e2    own R -> own B      10256 + 56 + 7 = 10319
//   rooks d8 and h8 see each other                      Black  a1, e1      own R -> own R      11152 + 0 + 3 = 11155
//   the same                                            White  a8, e8      their R -> their R  40448 + 420 + 3 = 40871
//   pawns e4 and d5 attack each other (after 1. e4 d5)  White  ids 20, 75  pawn pair           54472 + 577 = 55049
//   the same                                            Black  ids 19, 76  pawn pair           54472 + 552 = 55024
#include <cstdint>

namespace luminex::v5rel {

// A board as the finder reads it: the mailbox (a Piece per square: 0-5 White P N B R Q K, 6-11 Black, 12 empty)
// and a bitboard per piece, with the occupancy.
struct Bd {
    uint8_t pc[64];
    uint64_t by[13];
    uint64_t occ;
};

constexpr int MAX_ROWS = 96;     // rows one move may add or remove for one perspective; more, and that side is rebuilt
constexpr int MAX_ACTIVE = 256;  // rows a position can have (the spec's bound is 248)

struct RowLists {
    int na[2], ns[2];
    bool overflow[2];
    uint16_t add[2][MAX_ROWS], sub[2][MAX_ROWS];
};

void init();     // builds the tables (once; later calls return at once)
int n_rows();    // 55,960: the 54,472 relation rows, then the 1,488 pawn-pair rows

// The rows the move m adds and removes for each perspective p with need[p], from the board before it.
// ppm: 0 = pawn pairs in contact, 1 = every pawn pair on the same or adjacent files.
void find(const Bd& before, Move m, int ppm, const bool need[2], RowLists& out);

// Every active row of one perspective on a board, in the order they are found; returns their number.
int enumerate(const Bd& b, int persp, int ppm, uint16_t* out);

}  // namespace luminex::v5rel

namespace luminex::nnue {

// ---- profiling counters (data-driven: find the NPS bottleneck from numbers) ----
namespace { struct Stat { long long n = 0, cyc = 0; }; }

static Stat st_eval, st_update, st_refresh, st_king_refresh, st_incremental;
static Stat st_catchup;   // make_current: bringing a ply up to date before an evaluation
static Stat st_rel_find, st_rel_apply, st_rel_rebuild;   // gen768t: rows found, rows applied, lanes rebuilt
static long long g_rel_rows = 0, g_rel_plies = 0, g_rel_checks = 0, g_psq_checked = 0, g_psq_mismatch = 0;
static long long g_act_nz = 0, g_act_tot = 0;   // activation sparsity (non-zero L1 outputs after SCReLU)
static long long g_sat_events = 0;              // exact accumulator leaving int16 range (self-check only)
static bool g_profile = false;
static bool g_profile_inited = false;
static inline bool profile_on() {
    if (!g_profile_inited) { g_profile = (std::getenv("NNUE_PROFILE") != nullptr); g_profile_inited = true; }
    return g_profile;
}
static inline long long rdtsc() {
#if defined(__x86_64__) || defined(__i386__)
    unsigned a, d; __asm__ __volatile__("rdtsc" : "=a"(a), "=d"(d));
    return ((long long)d << 32) | a;
#else
    return 0;
#endif
}
void print_stats() {
    if (!profile_on()) return;
    std::fprintf(stderr, "NNUE profile: evals=%lld (%.1fM cyc, %.0f cyc/eval) | updates=%lld (%.1fM cyc, %.0f cyc/upd) | king-refreshes=%lld (%.0f cyc/kr) | incrementals=%lld (%.0f cyc/inc) | refreshes=%lld (%.1fM cyc) | L1-act density=%.1f%% | sat_events=%lld | catch-ups=%lld (%.1fM cyc)\n",
                 st_eval.n, st_eval.cyc/1000000.0, st_eval.n ? double(st_eval.cyc)/st_eval.n : 0.0,
                 st_update.n, st_update.cyc/1000000.0, st_update.n ? double(st_update.cyc)/st_update.n : 0.0,
                 st_king_refresh.n, st_king_refresh.n ? double(st_king_refresh.cyc)/st_king_refresh.n : 0.0,
                 st_incremental.n, st_incremental.n ? double(st_incremental.cyc)/st_incremental.n : 0.0,
                 st_refresh.n, st_refresh.cyc/1000000.0,
                 g_act_tot ? 100.0*g_act_nz/g_act_tot : 0.0,
                 g_sat_events,
                 st_catchup.n, st_catchup.cyc/1000000.0);
    if (g_rel_plies || st_rel_rebuild.n || g_rel_checks)
        std::fprintf(stderr, "NNUE relation lanes: rows found for %lld plies (%.0f cyc/ply) | applied in %lld walks (%.0f cyc/walk): "
                     "%lld perspective-plies, %.2f rows each | rebuilds=%lld (%.0f cyc) | checks=%lld | piece-square lanes checked=%lld, "
                     "not exact=%lld\n",
                     st_rel_find.n, st_rel_find.n ? double(st_rel_find.cyc)/st_rel_find.n : 0.0,
                     st_rel_apply.n, st_rel_apply.n ? double(st_rel_apply.cyc)/st_rel_apply.n : 0.0, g_rel_plies,
                     g_rel_plies ? double(g_rel_rows)/g_rel_plies : 0.0, st_rel_rebuild.n,
                     st_rel_rebuild.n ? double(st_rel_rebuild.cyc)/st_rel_rebuild.n : 0.0, g_rel_checks,
                     g_psq_checked, g_psq_mismatch);
}

constexpr int NUM_SQ = 64;
constexpr int NUM_PLANES = 768;              // 64 sq * 12 piece types
constexpr int NUM_INPUTS = NUM_PLANES * 32;  // 24576 per half (32 king buckets)
constexpr int OUT_SCALE = 300;               // matches trainer's output *300
constexpr int NNUE_MAX_PLY = 2048;           // MUST match Position::MAX_STATES (luminex.h)

static const int KingBuckets[NUM_SQ] = {
    -1, -1, -1, -1, 31, 30, 29, 28,
    -1, -1, -1, -1, 27, 26, 25, 24,
    -1, -1, -1, -1, 23, 22, 21, 20,
    -1, -1, -1, -1, 19, 18, 17, 16,
    -1, -1, -1, -1, 15, 14, 13, 12,
    -1, -1, -1, -1, 11, 10,  9,  8,
    -1, -1, -1, -1,  7,  6,  5,  4,
    -1, -1, -1, -1,  3,  2,  1,  0,
};

static inline int orient(bool white_pov, int sq, int ksq) {
    int kfile = ksq & 7;
    return ((kfile < 4) ? 7 : 0) ^ (white_pov ? 0 : 56) ^ sq;
}

static inline int halfka_idx(bool white_pov, int king_sq, int sq, Piece p) {
    int pt = static_cast<int>(piece_type_of(p));               // 0..5 (PAWN..KING)
    bool is_white_piece = (color_of_piece(p) == WHITE);
    int p_idx = pt * 2 + (is_white_piece != white_pov ? 1 : 0);
    int o_ksq = orient(white_pov, king_sq, king_sq);
    return orient(white_pov, sq, king_sq) + p_idx * NUM_SQ + KingBuckets[o_ksq] * NUM_PLANES;
}

static int g_L1 = 0, g_L2 = 0, g_L3 = 0;
// FT weights stored as int16 (half the memory traffic vs float → fewer L3 misses on
// feature delta). The accumulator stays full float32 (no SCReLU precision loss).
static constexpr float FT_WSCALE = NNUE_FT_SCALE;
[[maybe_unused]] static constexpr float FT_WINV = 1.0f / FT_WSCALE;   // the debug self-check's; the kernels have their own
// In 2 MiB pages where the system allows (LargeArray): the updates read its
// rows at random.
static LargeArray<int16_t> ft_w;
static std::vector<int32_t> ft_b_i32;   // integer-domain FT bias (b*FT_WSCALE)
// DOSL linear head (Step-0): per-feature scalar weights over the SAME HalfKAv2_hg
// indices; accumulator keeps one int32 per perspective (O(1) qsearch stand-pat).
// Weights stored in 1/32 cp units: a queen feature (~900cp) still fits int16.

static std::vector<float> ft_b;   // (L1,)
static std::vector<float> l2_w, l3_w, out_w;   // float dense weights (LNN1 nets)
static bool g_int8 = false;
static bool g_loaded = false;
// The last load's report, until take_report() hands it to the UCI layer
static std::string g_report;
// gen768t (LNT8 nets, docs/NNUE.md, V5-GEN768T): attack-relation and pawn-pair rows, int8 in steps of
// 2^g_rel_rs on the feature grid, summed into int16 relation lanes of their own and added into the first
// g_rel_T lanes of the feature layer at evaluation (RelLanes). g_rel_T = 0 for every other format.
static int g_rel_T = 0, g_rel_rs = 0, g_rel_ppm = 0;
static LargeArray<int8_t> g_rel_w;   // [rows][T], one row per relation or pawn-pair feature
// LUMINEX_REL_CHECK (read at load): 1 = every evaluation rebuilds the relation lanes from the board and
// aborts on a difference; 2 = also compares every piece-square lane with its exact int32 sum and counts the
// differences (print_stats); 3 = also evaluates through the spec's formula for the relation term in plain
// integers and aborts unless the output equals the kernel's. A test, not a timing.
static int g_rel_check = 0;

// The dense tail after the feature transformer: L2/L3/out biases, and for int8
// nets the int8 weights with their scales (the FT stays float on disk). LNN1
// and LNI8 nets have one tail; LNB8 and LNR8 nets have one per output bucket,
// chosen by the number of pieces on the board (bucket_of). LNI8 and LNB8 hold
// one weight scale per layer, LNR8 one per row (output neuron) of L2 and L3.
// The dense weights start on cache lines (LargeArray): the kernels read them
// 32 or 64 bytes at a time.
using WeightBytes = std::vector<int8_t, CacheAligned<int8_t>>;
struct DenseTail {
    std::vector<int8_t> l2_w_file, l3_w_file;   // L2 and L3 as the file stores them (output-major)
    WeightBytes l2_w_i8, l3_w_i8;       // the same, arranged for the kernels in use (arrange_tails)
    WeightBytes out_w_i8;
    std::vector<uint8_t> l2_slot;   // L2Layout::Pairs: each input chunk's slot (pair_chunks)
    int l2_pairs = 0;
    std::vector<float> l2_b, l3_b;
    // A dot of int8 weights (w x s) with uint8 activations (a x 127) becomes
    // the float sum when multiplied by 1 / (s x 127): one factor per output.
    std::vector<float> ics2, ics3;
    float icso = 1.0f;
    float out_b = 0.0f;
    TailView view;   // what the kernels read: the vectors above, and the layouts they got
};
constexpr int MAX_BUCKETS = 16;
static std::vector<DenseTail> g_tails(1);
static int g_buckets = 1;

// Output bucket: (pieces on the board, kings included, minus 1) x buckets / 32,
// so 8 buckets split 2-4, 5-8, ..., 29-32 pieces. The trainer uses the same rule.
static inline int bucket_of(const Position& pos) {
    if (g_buckets == 1) return 0;
    int b = (popcount(pos.pieces()) - 1) * g_buckets / 32;
    return b < 0 ? 0 : (b >= g_buckets ? g_buckets - 1 : b);
}

static bool g_enabled = true;
static uint32_t g_net_id = 0;   // incremented by every successful load(); keys the refresh cache

// Per-thread accumulator stack, indexed by Position::st_ply(). v[perspective][neuron].
// Cache-line aligned for clean loads during the SCReLU/quant8 pass.
// int16 accumulator: exact integer sums as long as every lane's sum stays inside
// int16, which on the x2048 grid (FT_WSCALE) is +-16.0 window units against the
// 11.5 of the largest sum measured on games (gen768s_03q; 12.9 on search leaves).
// The updates wrap, so a partial sum that leaves int16 within one update cannot
// change the result; the grid, not the SCReLU clip, is what keeps the lanes exact.
// Halves copy/update/eval traffic vs the int32 accumulator.
// Padded so consecutive plies' entries sit 7,168 bytes apart (3,072 past a 4 KB
// multiple): a child built from its parent then never stores to an address whose
// low 12 bits match a parent load just ahead (4,160 bytes apart they matched 64
// bytes on, which store-to-load forwarding treats as a possible dependence on
// AMD Zen 2, Software Optimization Guide 56305 section 2.12). Exact. The cost is
// memory: 2,048 entries (NNUE_MAX_PLY) x 7,168 bytes = 14.7 MB a thread, 6.2 MB
// more than at 4,160 bytes (about 376 MiB more at 64 threads). The padding is
// zeroed, so copying a whole Accumulator never reads indeterminate bytes, but
// nothing should copy one: a copy moves its 3 KB of padding too. Every use is
// a reference, and the null move copies the lanes only.
struct alignas(64) Accumulator { int16_t v[2][NNUE_L1_MAX]; int32_t lin[2]; char pad_[7168 - 2 * NNUE_L1_MAX * 2 - 8] = {}; };
static_assert(sizeof(Accumulator) % 4096 == 3072, "accumulator stride: see above");

// Heap-allocated per thread, NOT a thread_local C-array: an 8MB thread_local array
// reserves 8MB of *static TLS* for every thread and overflows the stack at creation.
// The vector object is ~24 bytes of TLS; its 8MB buffer lives on the heap and is freed
// when the thread exits (RAII).
static inline Accumulator* nnue_acc_stack() {
    thread_local std::vector<Accumulator> store(NNUE_MAX_PLY);
    return store.data();
}

// Lazy updates. do_move only records what its move changes; evaluate() brings
// its own ply up to date first (make_current), so plies whose subtree never
// evaluates are never computed. One record per state ply, beside the
// accumulator stack; every do_move rewrites its own ply's record, so walking
// back from the current ply only ever reads the current line's moves.
struct PlyDelta {
    Move move;
    Piece moved = NO_PIECE;        // the moving piece before the move (a pawn for promotions)
    PieceType captured = PT_NONE;
    uint8_t ksq[2] = {0, 0};       // both kings' squares after the move
    int8_t rebuild = -1;           // the side whose own king moved: rebuilt there, not updated
    bool null_move = false;
    bool current[2] = {false, false};
    bool relc[2] = {false, false};      // gen768t: this ply's relation lanes are computed, per side
    bool relflip[2] = {false, false};   // gen768t: this side's own king crossed the d/e files here
};
// The board after a king move, which the deferred rebuild of the mover's side reads.
struct KingBoard { Bitboard pieces[2][6]; };

static inline PlyDelta* nnue_deltas() {
    thread_local std::vector<PlyDelta> store(NNUE_MAX_PLY);
    return store.data();
}
static inline KingBoard* nnue_king_boards() {
    thread_local std::vector<KingBoard> store(NNUE_MAX_PLY);
    return store.data();
}

// gen768t: per ply, both sides' relation lanes (T int16 each) and the board after the ply's move (a move's rows
// are found from the board before it). Allocated on a thread's first use, for the loaded net's T.
constexpr int REL_WALK = 16;   // plies whose rows one catch-up finds ahead of the piece-square work, and prefetches
static inline int16_t* rel_lane_stack() {
    thread_local std::vector<int16_t> store;
    const size_t need = size_t(NNUE_MAX_PLY) * 2 * size_t(g_rel_T);
    if (store.size() < need) store.assign(need, 0);
    return store.data();
}
static inline int16_t* rel_side(int16_t* stack, int ply, int p) { return stack + (size_t(ply) * 2 + size_t(p)) * size_t(g_rel_T); }
static inline v5rel::Bd* rel_boards() {
    thread_local std::vector<v5rel::Bd> store(NNUE_MAX_PLY);
    return store.data();
}
static inline v5rel::RowLists* rel_buf() {
    thread_local std::vector<v5rel::RowLists> store(REL_WALK);
    return store.data();
}

bool loaded() { return g_loaded; }
bool enabled() { return g_loaded && g_enabled; }
void set_enabled(bool on) { g_enabled = on; }
int l1() { return g_L1; }

// ---- Kernel levels ----
// The kernel set in use (g_k). A build without LUMINEX_DISPATCH has one, for
// the level its compiler targets, and calls it directly. The universal x86-64
// build has every level's, best first, and uses the best one the CPU and its
// OS support, or the best at or below the one the SimdLevel option asks for.
// The choice is made once, before the first network is arranged for it, and
// again only when SimdLevel changes.
#if defined(LUMINEX_NNUE_DISPATCH)
#if !defined(__GNUC__) && !defined(__clang__)
#error "LUMINEX_DISPATCH needs GCC or Clang (clang-cl included): cpuid and xgetbv below are GNU inline assembly"
#endif
static const Kernels* const LEVELS[] = {
    &k_avx512vnni::kernels, &k_avx512::kernels, &k_avxvnni::kernels, &k_avx2::kernels, &k_base::kernels,
};
static const Kernels* g_k = &k_base::kernels;
static bool g_k_chosen = false;
#define NNUE_K(fn) (g_k->fn)
#else
static const Kernels* const LEVELS[] = { &k_native::kernels };
static const Kernels* const g_k = &k_native::kernels;
#define NNUE_K(fn) (k_native::fn)
#endif
constexpr int NUM_LEVELS = int(sizeof(LEVELS) / sizeof(LEVELS[0]));

#if defined(LUMINEX_NNUE_DISPATCH)
static int g_request = -1;   // SimdLevel: an index into LEVELS, -1 for auto
namespace {
// cpuid(leaf, subleaf) -> eax, ebx, ecx, edx
void cpuid(uint32_t leaf, uint32_t sub, uint32_t r[4]) {
    __asm__ __volatile__("cpuid" : "=a"(r[0]), "=b"(r[1]), "=c"(r[2]), "=d"(r[3]) : "a"(leaf), "c"(sub));
}
// XCR0: the register state the OS saves on a context switch.
uint64_t xcr0() {
    uint32_t lo, hi;
    __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (uint64_t(hi) << 32) | lo;
}

// Which LEVELS this CPU and its OS can run. A level's file is compiled for a
// whole -march level, and the compiler may use any instruction of it there,
// so a level needs every feature of its x86-64 level, not only the ones the
// kernels name.
void detect_levels(bool (&ok)[NUM_LEVELS]) {
    uint32_t r[4];
    cpuid(0, 0, r);
    const uint32_t max_leaf = r[0];
    cpuid(0x80000000u, 0, r);
    const uint32_t max_ext = r[0];
    uint32_t l1[4] = {}, l7[4] = {}, l71[4] = {}, e1[4] = {};
    if (max_leaf >= 1) cpuid(1, 0, l1);
    if (max_leaf >= 7) {
        cpuid(7, 0, l7);
        if (l7[0] >= 1) cpuid(7, 1, l71);   // eax of subleaf 0: the last subleaf
    }
    if (max_ext >= 0x80000001u) cpuid(0x80000001u, 0, e1);
    auto bit = [](uint32_t reg, int b) { return ((reg >> b) & 1u) != 0; };
    // The YMM and ZMM registers work only where the OS saves them: XCR0 bits
    // 1-2 (SSE, AVX) and 5-7 (opmask, ZMM_Hi256, Hi16_ZMM). xgetbv exists
    // when OSXSAVE (leaf 1 ecx bit 27) says so.
    const uint64_t xcr = bit(l1[2], 27) ? xcr0() : 0;
    const bool ymm = (xcr & 0x6) == 0x6;
    const bool zmm = ymm && (xcr & 0xE0) == 0xE0;
    // x86-64-v2, -v3 and -v4 as the x86-64 psABI defines them.
    const bool v2 = bit(l1[2], 0) && bit(l1[2], 9) && bit(l1[2], 13) && bit(l1[2], 19)   // SSE3 SSSE3 CX16 SSE4.1
                 && bit(l1[2], 20) && bit(l1[2], 23) && bit(e1[2], 0);                 // SSE4.2 POPCNT LAHF
    const bool v3 = v2 && ymm && bit(l1[2], 12) && bit(l1[2], 22) && bit(l1[2], 28)    // FMA MOVBE AVX
                 && bit(l1[2], 29) && bit(l7[1], 3) && bit(l7[1], 5) && bit(l7[1], 8) // F16C BMI1 AVX2 BMI2
                 && bit(e1[2], 5);                                                     // LZCNT
    const bool v4 = v3 && zmm && bit(l7[1], 16) && bit(l7[1], 17) && bit(l7[1], 28)    // AVX512 F DQ CD
                 && bit(l7[1], 30) && bit(l7[1], 31);                                  // AVX512 BW VL
    ok[0] = v4 && bit(l7[2], 11);    // avx512vnni: + AVX512_VNNI (leaf 7 ecx bit 11)
    ok[1] = v4;                      // avx512
    ok[2] = v3 && bit(l71[0], 4);    // avxvnni: + AVX-VNNI (leaf 7 subleaf 1 eax bit 4)
    ok[3] = v3;                      // avx2
    ok[4] = true;                    // base: no more than the rest of the engine needs
}
} // namespace
#endif

// Which LEVELS the CPU supports (all of them in a build with one level).
static const bool* levels_supported() {
    static bool ok[NUM_LEVELS];
    static bool known = false;
    if (!known) {
#if defined(LUMINEX_NNUE_DISPATCH)
        detect_levels(ok);
#else
        for (bool& b : ok) b = true;
#endif
        known = true;
    }
    return ok;
}

// The best supported level at or below the requested one (-1: the best).
static int best_level(int request) {
    const bool* ok = levels_supported();
    for (int i = request < 0 ? 0 : request; i < NUM_LEVELS; ++i)
        if (ok[i]) return i;
    return NUM_LEVELS - 1;
}

static void choose_kernels() {
#if defined(LUMINEX_NNUE_DISPATCH)
    if (!g_k_chosen) {
        g_k = LEVELS[best_level(g_request)];
        g_k_chosen = true;
    }
#endif
}

static void arrange_tails();

bool simd_dispatch() {
#if defined(LUMINEX_NNUE_DISPATCH)
    return true;
#else
    return false;
#endif
}

const char* simd_level() {
    choose_kernels();
    return g_k->name;
}

std::vector<std::string> simd_levels() {
    std::vector<std::string> names;
    for (const Kernels* k : LEVELS) names.push_back(k->name);
    return names;
}

std::string set_simd_level(const std::string& request) {
    int want = -1;
    if (request != "auto") {
        want = -2;
        for (int i = 0; i < NUM_LEVELS; ++i)
            if (request == LEVELS[i]->name) want = i;
        if (want == -2) return "SimdLevel " + request + " is not a level of this build; SIMD level " + simd_level();
    }
    const int got = best_level(want);
#if defined(LUMINEX_NNUE_DISPATCH)
    g_request = want;
    g_k = LEVELS[got];
    g_k_chosen = true;
#endif
    if (g_loaded) arrange_tails();   // the network's dense layers, in the new kernels' layout
    std::string msg = std::string("SIMD level ") + g_k->name;
    if (want >= 0 && got != want) msg += " (this CPU cannot run " + request + ")";
    return msg;
}

// Pair L2's 32-input chunks for L2Layout::Pairs. Chunk c's int16 lane j holds inputs 32c + 2j and 32c + 2j + 1:
// two products of an activation in [0, NNUE_ACT_MAX] and an int8 weight. Two chunks may share an int16 sum when,
// for every output and lane, NNUE_ACT_MAX times the four weights' |w| summed fits int16: then the sum is exact
// whatever the activations. The pairs are a matching in the graph of such chunk pairs: the free chunk with the
// fewest free partners pairs first, with its partner of fewest; then two chunks left over take the halves of a
// pair that suit them, until none do. A chunk left single is still exact (the kernel adds it alone), only
// slower. Returns the pair count; order gets the chunks slot by slot, the pairs first, single chunks after.
static int pair_chunks(const std::vector<int8_t>& w, int L2, int total_in, std::vector<uint8_t>& order) {
    const int n = total_in / 32, lim = 32767 / NNUE_ACT_MAX;
    const size_t m = size_t(L2) * 16;
    std::vector<int> s(size_t(n) * m);   // s[c * m + o * 16 + j]: chunk c's lane j |w| sum for output o
    for (int o = 0; o < L2; ++o)
        for (int c = 0; c < n; ++c)
            for (int j = 0; j < 16; ++j) {
                const int8_t* p = &w[size_t(o) * total_in + 32 * c + 2 * j];
                s[c * m + size_t(o) * 16 + j] = std::abs(int(p[0])) + std::abs(int(p[1]));
            }
    std::vector<char> ok(size_t(n) * n, 0);
    for (int c = 0; c < n; ++c)
        for (int d = c + 1; d < n; ++d) {
            bool fits = true;
            for (size_t i = 0; i < m && fits; ++i) fits = s[c * m + i] + s[d * m + i] <= lim;
            ok[size_t(c) * n + d] = ok[size_t(d) * n + c] = fits;
        }
    std::vector<int> mate(n, -1);
    auto partners = [&](int c) {
        int k = 0;
        for (int d = 0; d < n; ++d) k += (d != c && mate[d] < 0 && ok[size_t(c) * n + d]);
        return k;
    };
    for (;;) {
        int u = -1, ku = 0;
        for (int c = 0; c < n; ++c)
            if (mate[c] < 0)
                if (const int k = partners(c); k > 0 && (u < 0 || k < ku)) { u = c; ku = k; }
        if (u < 0) break;
        int v = -1, kv = 0;
        for (int d = 0; d < n; ++d)
            if (d != u && mate[d] < 0 && ok[size_t(u) * n + d])
                if (const int k = partners(d); v < 0 || k < kv) { v = d; kv = k; }
        mate[u] = v; mate[v] = u;
    }
    for (bool more = true; more;) {
        more = false;
        for (int u = 0; u < n && !more; ++u)
            for (int v = u + 1; v < n && !more && mate[u] < 0; ++v)
                for (int a = 0; a < n && !more && mate[v] < 0; ++a) {
                    const int b = mate[a];
                    if (b < 0 || !ok[size_t(u) * n + a] || !ok[size_t(v) * n + b]) continue;
                    mate[u] = a; mate[a] = u; mate[v] = b; mate[b] = v;
                    more = true;
                }
    }
    order.clear();
    for (int c = 0; c < n; ++c)
        if (mate[c] > c) { order.push_back(uint8_t(c)); order.push_back(uint8_t(mate[c])); }
    const int pairs = int(order.size()) / 2;
    for (int c = 0; c < n; ++c)
        if (mate[c] < 0) order.push_back(uint8_t(c));
    return pairs;
}

// Arrange every tail's L2 and L3 weights the way the kernels in use read them,
// from the file's output-major order, and point the kernels' view at them.
static void arrange_tails() {
    const Kernels& k = *g_k;
    const int L1 = g_L1, L2 = g_L2, L3 = g_L3;
    const int total_in = 2 * L1;
    for (DenseTail& t : g_tails) {
        TailView v;
        if (g_int8) {
            t.l2_slot.clear();
            t.l2_pairs = 0;
            const bool by_input = k.l2_layout == L2Layout::ByInput4 && L2 == 16;
            const bool batched = k.l3_batched && L2 == 16 && L3 % 4 == 0;
            if (batched) {
                // Batched-L3 layout: dpbusd lane (o*4+j) accumulates h2-slice j against
                // l3_w[o] slice j — one interleaved row per 4 outputs, 2 dpbusd for all 32.
                t.l3_w_i8.assign(t.l3_w_file.size(), 0);
                for (int o = 0; o < L3; ++o)
                    for (int j = 0; j < 4; ++j)
                        std::memcpy(&t.l3_w_i8[(static_cast<size_t>(o) * 4 + j) * 4], &t.l3_w_file[static_cast<size_t>(o) * L2 + j * 4], 4);
            } else {
                t.l3_w_i8.assign(t.l3_w_file.begin(), t.l3_w_file.end());
            }
            if (by_input) {
                // Input-chunk L2 layout: block c (64 bytes) holds, for each of the 16
                // outputs, its 4 weights for inputs 4c..4c+3, so one VPDPBUSD of a
                // broadcast 4-byte input chunk updates all 16 output sums at once.
                t.l2_w_i8.assign(t.l2_w_file.size(), 0);
                for (int c = 0; c < total_in / 4; ++c)
                    for (int o = 0; o < L2; ++o)
                        std::memcpy(&t.l2_w_i8[(static_cast<size_t>(c) * L2 + o) * 4],
                                    &t.l2_w_file[static_cast<size_t>(o) * total_in + c * 4], 4);
            } else if (k.l2_layout == L2Layout::Pairs) {
                // Paired chunks (pair_chunks), four outputs a pass: for each pass, pair k's 4 x 2 rows of 32
                // weights (output by output, its two chunks in slot order), then each single chunk's 4 rows.
                const int nslot = total_in / 32;
                std::vector<uint8_t> order;
                t.l2_pairs = pair_chunks(t.l2_w_file, L2, total_in, order);
                t.l2_slot.assign(nslot, 0);
                for (int s = 0; s < nslot; ++s) t.l2_slot[order[s]] = uint8_t(s);
                t.l2_w_i8.assign(t.l2_w_file.size(), 0);
                size_t at = 0;
                auto put = [&](int o, int c) {
                    std::memcpy(&t.l2_w_i8[at], &t.l2_w_file[static_cast<size_t>(o) * total_in + 32 * c], 32);
                    at += 32;
                };
                for (int q = 0; q < L2; q += 4) {
                    for (int p = 0; p < t.l2_pairs; ++p)
                        for (int o = q; o < q + 4; ++o) { put(o, order[2 * p]); put(o, order[2 * p + 1]); }
                    for (int s = 2 * t.l2_pairs; s < nslot; ++s)
                        for (int o = q; o < q + 4; ++o) put(o, order[s]);
                }
            } else if (k.l2_layout == L2Layout::Chunk32) {
                // AVX2 SF-style chunk access. Old: [output][chunk] → New: [chunk][output]
                // (each chunk's L2 weight rows contiguous)
                t.l2_w_i8.assign(t.l2_w_file.size(), 0);
                for (int c = 0; c < total_in / 32; ++c)
                    for (int o = 0; o < L2; ++o)
                        std::memcpy(&t.l2_w_i8[(static_cast<size_t>(c) * L2 + o) * 32],
                                    &t.l2_w_file[static_cast<size_t>(o) * total_in + c * 32], 32);
            } else {
                t.l2_w_i8.assign(t.l2_w_file.begin(), t.l2_w_file.end());
            }
            v.l2_w = t.l2_w_i8.data();
            v.l3_w = t.l3_w_i8.data();
            v.out_w = t.out_w_i8.data();
            v.l2_by_input = by_input;
            v.l3_batched = batched;
            v.l2_slot = t.l2_slot.empty() ? nullptr : t.l2_slot.data();
            v.l2_pairs = t.l2_pairs;
        } else {
            v.l2_wf = l2_w.data();
            v.l3_wf = l3_w.data();
            v.out_wf = out_w.data();
        }
        v.l2_b = t.l2_b.data();
        v.l3_b = t.l3_b.data();
        v.ics2 = t.ics2.data(); v.ics3 = t.ics3.data(); v.icso = t.icso;
        v.out_b = t.out_b;
        t.view = v;
    }
}

static bool read_tensor(std::istream& f, std::vector<float>& v, int expected) {
    int32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);
    if (n != expected) { std::fprintf(stderr, "nnue: size mismatch %d != %d\n", n, expected); return false; }
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()), sizeof(float) * n);
    return f.good();
}

// l2_w(i8), l2_b(i32), l3_w(i8), l3_b(i32), out_w(i8), s_out(f32), out_b(i32).

// Read a network from f, a file or the copy built into the binary; name is
// what the report calls it.
static bool load_stream(std::istream& f, const std::string& name) {
    char magic[4];
    f.read(magic, 4);
    std::string mg(magic, 4);
    // LNN1: float tail. LNI8: int8 tail. LNB8: int8 tails, one per output
    // bucket; its header carries the bucket count after the four sizes. LNR8:
    // LNB8 with a scale per row of L2 and L3 instead of one per layer. LNT8
    // (gen768t, docs/NNUE.md, V5-GEN768T), little-endian, every tensor after its
    // int32 element count as in LNR8: "LNT8"; int32 L1, L2, L3, NI, NB; int32
    // T (relation lanes, a multiple of 64), NR (55,960 rows), FS (the feature
    // grid is x 2^FS), RS (a relation step is 2^RS units of it), PPM (pawn
    // pairs: 0 in contact, 1 all), LAYOUT (1: the v5rel index); int16
    // psq[NI][L1] (feature-major, x 2^FS); int32 psq_bias[L1]; int8 rel[NR][T];
    // then the NB tails and any tagged sections exactly as in LNR8.
    const bool lnt8 = (mg == "LNT8");
    if (mg != "LNN1" && mg != "LNI8" && mg != "LNB8" && mg != "LNR8" && !lnt8) { std::fprintf(stderr, "nnue: bad magic\n"); return false; }
    const bool rows = (mg == "LNR8" || lnt8);
    int32_t hdr[4];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (!f) { std::fprintf(stderr, "nnue: truncated header\n"); return false; }
    int L1 = hdr[0], L2 = hdr[1], L3 = hdr[2], NI = hdr[3];
    int32_t NB = 1;
    if (mg == "LNB8" || mg == "LNR8" || lnt8) {
        f.read(reinterpret_cast<char*>(&NB), 4);
        if (!f || NB < 1 || NB > MAX_BUCKETS) { std::fprintf(stderr, "nnue: bucket count %d out of range\n", NB); return false; }
    }
    // LNT8: T (relation lanes), NR (rows), FS (the feature grid is x 2^FS), RS (a relation step is 2^RS units of
    // it), PPM (pawn pairs: 0 in contact, 1 all on the same or adjacent files), LAYOUT (1: the v5rel index)
    int32_t rh[6] = {0, 0, 0, 0, 0, 0};
    if (lnt8) {
        f.read(reinterpret_cast<char*>(rh), sizeof(rh));
        if (!f) { std::fprintf(stderr, "nnue: truncated LNT8 header\n"); return false; }
        v5rel::init();
        if ((1 << std::clamp(rh[2], 0, 30)) != int(NNUE_FT_SCALE)) {
            std::fprintf(stderr, "nnue: LNT8 feature grid x2^%d, this engine's is x%d\n", rh[2], int(NNUE_FT_SCALE));
            return false;
        }
        if (rh[5] != 1 || rh[1] != v5rel::n_rows()) {
            std::fprintf(stderr, "nnue: LNT8 layout %d with %d rows; this engine knows layout 1 with %d rows\n", rh[5], rh[1], v5rel::n_rows());
            return false;
        }
        if (rh[3] < 0 || rh[3] > 6 || rh[4] < 0 || rh[4] > 1) {
            std::fprintf(stderr, "nnue: LNT8 relation step 2^%d or pawn-pair mode %d out of range\n", rh[3], rh[4]);
            return false;
        }
    }
    if (NI != NUM_INPUTS) { std::fprintf(stderr, "nnue: NUM_INPUTS mismatch %d\n", NI); return false; }
    if (L1 <= 0 || L1 > NNUE_L1_MAX) { std::fprintf(stderr, "nnue: L1=%d out of range (max %d)\n", L1, NNUE_L1_MAX); return false; }
    // The SIMD loops step L1 in 32-lane blocks and L2/L3 in 8-lane blocks, and
    // the small VNNI dots read at most 64 bytes.
    if (L1 % 32 != 0 || L2 <= 0 || L2 > 64 || L2 % 8 != 0 || L3 <= 0 || L3 > 64 || L3 % 8 != 0) {
        std::fprintf(stderr, "nnue: unsupported layer sizes L1=%d L2=%d L3=%d\n", L1, L2, L3);
        return false;
    }
    if (lnt8 && (rh[0] <= 0 || rh[0] % 64 != 0 || rh[0] > L1)) {
        std::fprintf(stderr, "nnue: LNT8 relation lanes T=%d: a multiple of 64 up to L1=%d\n", rh[0], L1);
        return false;
    }
    // From here on the globals are overwritten; a failure part-way must not
    // leave a half-loaded net marked usable.
    g_loaded = false;
    g_int8 = (mg != "LNN1");
    g_tails.assign(NB, DenseTail{});
    g_buckets = NB;
    g_L1 = L1; g_L2 = L2; g_L3 = L3;
    g_rel_T = 0;   // an LNT8 net sets it again once it has loaded whole
    long long clamped = 0;
    if (lnt8) {
        // The feature layer as the engine holds it: int16 rows, feature-major, on the grid; the bias in int32.
        auto read_count = [&](int32_t expected, const char* what) -> bool {
            int32_t n = 0;
            f.read(reinterpret_cast<char*>(&n), 4);
            if (!f || n != expected) { std::fprintf(stderr, "nnue: LNT8 %s size %d != %d\n", what, n, expected); return false; }
            return true; };
        if (!read_count(NUM_INPUTS * L1, "piece-square table")) return false;
        ft_w.reset(static_cast<size_t>(NUM_INPUTS) * L1);
        f.read(reinterpret_cast<char*>(ft_w.data()), std::streamsize(sizeof(int16_t) * size_t(NUM_INPUTS) * size_t(L1)));
        if (!read_count(L1, "bias")) return false;
        ft_b_i32.resize(L1);
        f.read(reinterpret_cast<char*>(ft_b_i32.data()), std::streamsize(sizeof(int32_t) * size_t(L1)));
        ft_b.resize(L1);
        for (int l = 0; l < L1; ++l) ft_b[l] = float(ft_b_i32[l]) / FT_WSCALE;
        if (!read_count(rh[1] * rh[0], "relation table")) return false;
        g_rel_w.reset(size_t(rh[1]) * size_t(rh[0]));
        f.read(reinterpret_cast<char*>(g_rel_w.data()), std::streamsize(size_t(rh[1]) * size_t(rh[0])));
        if (!f) { std::fprintf(stderr, "nnue: truncated LNT8 tables\n"); return false; }
    } else {
    std::vector<float> ft_w_raw;
    if (!read_tensor(f, ft_w_raw, L1 * NUM_INPUTS)) return false;
    ft_w.reset(static_cast<size_t>(NUM_INPUTS) * L1);
    // int16 at x2048 holds |w| < 16.0; clamp instead of letting the cast wrap.
    for (int i = 0; i < NUM_INPUTS; ++i)
        for (int l = 0; l < L1; ++l) {
            long long q = std::llround(ft_w_raw[l * NUM_INPUTS + i] * FT_WSCALE);
            if (q > 32767 || q < -32767) { q = q > 0 ? 32767 : -32767; ++clamped; }
            ft_w[i * L1 + l] = (int16_t)q;
        }
    if (!read_tensor(f, ft_b, L1)) return false;
    ft_b_i32.resize(L1);
    for (int l = 0; l < L1; ++l) ft_b_i32[l] = std::lround(ft_b[l] * FT_WSCALE);
    }
    if (g_int8) {
        // LNI8 / LNB8: per tail, int8 weights + float bias + float scale per layer.
        // LNR8: the L2 and L3 scales are float tensors, one per row.
        auto read_i8 = [&](auto& v, int expected) -> bool {
            int32_t n = 0; f.read(reinterpret_cast<char*>(&n), 4);
            if (n != expected) { std::fprintf(stderr, "nnue: i8 size %d != %d\n", n, expected); return false; }
            v.resize(n); f.read(reinterpret_cast<char*>(v.data()), n); return f.good(); };
        // L2 and L3 are kept as stored; arrange_tails lays them out for the kernels.
        // The factor 1 / (s x 127) of each output, computed once here with the
        // operations the evaluation used to repeat (the same floats, bit for bit).
        auto read_scales = [&](std::vector<float>& ics, int rows_n, bool per_row) -> bool {
            std::vector<float> sc;
            if (per_row) { if (!read_tensor(f, sc, rows_n)) return false; }
            else { float one = 1.0f; f.read(reinterpret_cast<char*>(&one), 4); sc.assign(rows_n, one); }
            ics.resize(rows_n);
            for (int o = 0; o < rows_n; ++o) {
                const float cs = sc[o] * 127.0f;
                ics[o] = 1.0f / cs;
            }
            return f.good(); };
        for (DenseTail& t : g_tails) {
        if (!read_i8(t.l2_w_file, L2 * 2 * L1)) return false;
        if (!read_tensor(f, t.l2_b, L2)) return false;
        if (!read_scales(t.ics2, L2, rows)) return false;
        if (!read_i8(t.l3_w_file, L3 * L2)) return false;
        if (!read_tensor(f, t.l3_b, L3)) return false;
        if (!read_scales(t.ics3, L3, rows)) return false;
        if (!read_i8(t.out_w_i8, L3)) return false;
        std::vector<float> ob;
        if (!read_tensor(f, ob, 1)) return false;
        t.out_b = ob[0];
        {   // the output layer has one row: its scale is a single float in every format
            std::vector<float> ico;
            if (!read_scales(ico, 1, false)) return false;
            t.icso = ico[0];
        }
        if (!f) { std::fprintf(stderr, "nnue: truncated dense tail\n"); return false; }
        }
    } else {
        DenseTail& t = g_tails[0];
        if (!read_tensor(f, l2_w, L2 * 2 * L1)) return false;
        if (!read_tensor(f, t.l2_b, L2)) return false;
        if (!read_tensor(f, l3_w, L3 * L2)) return false;
        if (!read_tensor(f, t.l3_b, L3)) return false;
        std::vector<float> out_w_raw;
        if (!read_tensor(f, out_w_raw, L3)) return false;
        out_w = out_w_raw;
        int32_t ob_n;
        f.read(reinterpret_cast<char*>(&ob_n), 4);
        if (ob_n != 1) { std::fprintf(stderr, "nnue: out.bias size %d\n", ob_n); return false; }
        f.read(reinterpret_cast<char*>(&t.out_b), sizeof(float));
    }
    // Optional DOSL linear head (older nets simply end here).
    {
        char tag[4] = {0,0,0,0};
        std::streampos save = f.tellg();
        f.read(tag, 4);
        if (f.gcount() == 4 && tag[0]=='L' && tag[1]=='I' && tag[2]=='N' && tag[3]=='H') {
            // Skip the LINH section (removed feature): read size, skip data
            int32_t n = 0; f.read(reinterpret_cast<char*>(&n), 4);
            if (n > 0 && n <= NUM_INPUTS) {
                f.seekg(static_cast<long>(n) * 4, std::ios::cur); // skip float weights
            }
        } else {
            f.seekg(save);   // not a LINH section — rewind (older net format)
        }
    }
    choose_kernels();
    arrange_tails();
    if (lnt8) {
        g_rel_T = rh[0];
        g_rel_rs = rh[3];
        g_rel_ppm = rh[4];
        const char* chk = std::getenv("LUMINEX_REL_CHECK");
        g_rel_check = chk ? std::atoi(chk) : 0;
    }
    g_loaded = true;
    ++g_net_id;   // refresh-cache entries of the previous net are stale
    // The report: an info string (stdout carries UCI traffic) ending with the
    // kernels' instruction-set level (SimdLevel), then any warning. It is kept
    // for take_report(), not printed here.
    if (lnt8)
        g_report = std::format("info string NNUE evaluation using {} (gen768t: int8 L1={} L2={} L3={}, {} output buckets, "
                               "a scale per row; relation lanes T={}, {} rows, step 2^{}, pawn pairs {}{}), SIMD level {}\n",
                               name, L1, L2, L3, NB, g_rel_T, rh[1], g_rel_rs, g_rel_ppm ? "all" : "in contact",
                               g_rel_check ? ", LUMINEX_REL_CHECK on" : "", g_k->name);
    else if (NB > 1)
        g_report = std::format("info string NNUE evaluation using {} (int8 L1={} L2={} L3={}, {} output buckets{}), SIMD level {}\n",
                               name, L1, L2, L3, NB, rows ? ", a scale per row" : "", g_k->name);
    else
        g_report = std::format("info string NNUE evaluation using {} ({} L1={} L2={} L3={}), SIMD level {}\n",
                               name, g_int8 ? "int8" : "float", L1, L2, L3, g_k->name);
    if (clamped > 0)
        g_report += std::format("info string NNUE warning: {} feature weights exceed the int16 range and were clamped\n",
                                clamped);
    return true;
}

bool load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "nnue: cannot open %s\n", path.c_str()); return false; }
    return load_stream(f, path);
}

namespace {
// Read-only stream buffer over bytes already in memory, seekable because the
// loader probes for an optional trailing section and rewinds when it is absent.
class MemoryBuffer : public std::streambuf {
public:
    MemoryBuffer(const unsigned char* data, std::size_t size) {
        char* p = reinterpret_cast<char*>(const_cast<unsigned char*>(data));
        setg(p, p, p + size);
    }

protected:
    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override {
        if (!(which & std::ios_base::in)) return pos_type(off_type(-1));
        char* from = dir == std::ios_base::beg ? eback() : dir == std::ios_base::cur ? gptr() : egptr();
        if (off < eback() - from || off > egptr() - from) return pos_type(off_type(-1));
        setg(eback(), from + off, egptr());
        return pos_type(off_type(gptr() - eback()));
    }
    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
        return seekoff(off_type(pos), std::ios_base::beg, which);
    }
};
} // namespace

bool has_embedded() { return embedded_net().size > 0; }

std::string take_report() {
    std::string report = std::exchange(g_report, std::string());
    return g_loaded ? report : std::string();   // a load that failed part-way left no net to report
}

bool load_embedded() {
    const EmbeddedNet net = embedded_net();
    if (net.size == 0) return false;
    MemoryBuffer buf(net.data, net.size);
    std::istream f(&buf);
#ifdef LUMINEX_EMBED_NAME
    return load_stream(f, std::string("built-in ") + LUMINEX_EMBED_NAME);   // the built-in file's own name
#else
    return load_stream(f, std::string("built-in ") + DEFAULT_NET);
#endif
}

static inline const int16_t* feature_row(bool white_pov, int ksq, int sq, Piece piece) {
    return &ft_w[static_cast<size_t>(halfka_idx(white_pov, ksq, sq, piece)) * g_L1];
}

// Rebuild side p's lanes from the board: the bias, then every piece's row in
// square order, in wrapping int16 arithmetic (refresh_lanes): the exact sum
// whenever it fits int16.
static void refresh_perspective(const Position& pos, Accumulator& a, int p) {
    bool white_pov = (p == 0);
    int ksq = static_cast<int>(pos.king_sq(white_pov ? WHITE : BLACK));
    const int16_t* rows[NUM_SQ];
    int n = 0;
    for (int sq = 0; sq < NUM_SQ; ++sq) {
        Piece pc = pos.piece_on(Square(sq));
        if (pc == NO_PIECE) continue;
        rows[n++] = feature_row(white_pov, ksq, sq, pc);
    }
    NNUE_K(refresh_lanes)(a.v[p], ft_b_i32.data(), rows, n, g_L1);
}

// Refresh cache: per perspective and own-king square, the lanes of the board
// that king square last saw. A king move then applies only the pieces that
// differ from that board instead of rebuilding from every piece. The cached
// sums are exact int32, saturated only on the copy into the int16
// accumulator, so an entry reused for a whole game never drifts. Entries
// built under a different net (net_id) start over from the bias.
struct alignas(64) RefreshEntry {
    int32_t acc[NNUE_L1_MAX];
    Bitboard pieces[2][6];   // [color][piece type] of the board acc describes
    uint32_t net_id = 0;
};

static inline RefreshEntry* refresh_cache() {
    thread_local std::vector<RefreshEntry> store(2 * NUM_SQ);
    return store.data();
}

// Rebuild side p from the board `pieces` with its king on ksq, through the
// refresh cache. The entry's int32 lanes are the exact sum for the board it
// holds, so after the diff they are the exact sum for `pieces`, saturated once
// here: the result does not depend on what the entry held before, which is
// what lets a rebuild wait until an evaluation needs it.
static void refresh_from_board(const Bitboard (&pieces)[2][6], int ksq, Accumulator& a, int p) {
    bool white_pov = (p == 0);
    RefreshEntry& e = refresh_cache()[p * NUM_SQ + ksq];
    if (e.net_id != g_net_id) {
        std::memcpy(e.acc, ft_b_i32.data(), sizeof(int32_t) * size_t(g_L1));
        std::memset(e.pieces, 0, sizeof(e.pieces));
        e.net_id = g_net_id;
    }
    const int16_t* gone_rows[NUM_SQ];
    const int16_t* added_rows[NUM_SQ];
    int ngone = 0, nadded = 0;
    for (int c = 0; c < 2; ++c)
        for (int pt = PAWN; pt <= KING; ++pt) {
            Bitboard now = pieces[c][pt];
            Bitboard was = e.pieces[c][pt];
            if (now == was) continue;
            Piece pc = make_piece(Color(c), PieceType(pt));
            for (Bitboard gone = was & ~now; gone; )
                gone_rows[ngone++] = feature_row(white_pov, ksq, pop_lsb(gone), pc);
            for (Bitboard added = now & ~was; added; )
                added_rows[nadded++] = feature_row(white_pov, ksq, pop_lsb(added), pc);
            e.pieces[c][pt] = now;
        }
    NNUE_K(refresh_cached)(e.acc, gone_rows, ngone, added_rows, nadded, a.v[p], g_L1);
}

// Sides [p_begin, p_end) of `child` from those of `parent` and the move
// recorded in d (a ply that rebuilds none of these sides): the rows and step
// order the eager update used, so the lanes are bit-identical to it. The move
// is decoded once for both sides.
static void apply_delta(const PlyDelta& d, const Accumulator& parent, Accumulator& child,
                        int p_begin, int p_end) {
    if (d.null_move) {
        for (int p = p_begin; p < p_end; ++p)
            std::memcpy(child.v[p], parent.v[p], sizeof(int16_t) * size_t(g_L1));
        return;
    }
    const Move m = d.move;
    const Square from = m.from(), to = m.to();
    const Color us = color_of_piece(d.moved), them = ~us;
    const PieceType moved_pt = piece_type_of(d.moved);
    const bool promo = m.is_promotion(), ep = m.is_en_passant(), castling = m.is_castling();
    const Piece from_pc = make_piece(us, promo ? PAWN : moved_pt);
    const Piece to_pc   = make_piece(us, promo ? PieceType(m.promotion_type()) : moved_pt);
    Square rfrom = SQUARE_NONE, rto = SQUARE_NONE, cap_sq = to;
    Piece cap_pc = NO_PIECE;
    if (castling) {
        const bool kingside = to == (us == WHITE ? Square(G1) : Square(G8));
        rfrom = kingside ? (us == WHITE ? Square(H1) : Square(H8)) : (us == WHITE ? Square(A1) : Square(A8));
        rto   = kingside ? (us == WHITE ? Square(F1) : Square(F8)) : (us == WHITE ? Square(D1) : Square(D8));
    } else if (ep) {
        cap_sq = Square(to - (us == WHITE ? 8 : -8));
        cap_pc = make_piece(them, PAWN);
    } else if (d.captured != PT_NONE) {
        cap_pc = make_piece(them, d.captured);
    }
    const Piece rook = make_piece(us, ROOK);
    // Per side: + the moved piece on its new square, - it on the old one, then
    // - the captured piece, or for castling - and + the rook.
    DeltaRows jobs[2];
    int njobs = 0;
    for (int p = p_begin; p < p_end; ++p) {
        const bool white_pov = (p == 0);
        const int ksq = d.ksq[p];
        DeltaRows& r = jobs[njobs++];
        r.parent = parent.v[p];
        r.child = child.v[p];
        r.add0 = feature_row(white_pov, ksq, to, to_pc);
        r.sub0 = feature_row(white_pov, ksq, from, from_pc);
        r.sub1 = nullptr;
        r.add1 = nullptr;
        if (castling) {
            r.sub1 = feature_row(white_pov, ksq, rfrom, rook);
            r.add1 = feature_row(white_pov, ksq, rto, rook);
        } else if (cap_pc != NO_PIECE) {
            r.sub1 = feature_row(white_pov, ksq, cap_sq, cap_pc);
        }
    }
    NNUE_K(update_lanes)(jobs, njobs, g_L1);
}

// Bring both sides of the accumulator at pos's ply up to date. Each side walks
// back to the latest ply that is computed or rebuilds that side (its own king
// moved there) and rebuilds that one from its board snapshot if needed; then
// the later plies are replayed in order, both sides together where both need
// them, and marked computed so siblings reuse them.
// ---- gen768t: the relation lanes ----
// A ply's relation lanes come from its parent's and the rows its move adds and removes, which the finder reads
// off the board before the move (the board after each move is kept per ply). A side whose own king crossed the
// d/e files at a ply has every index changed there, so that side is rebuilt from that ply's board. Like the
// piece-square lanes they are brought current when an evaluation needs them, and the rows of the plies a
// catch-up replays are found first and prefetched, so that they arrive while the piece-square lanes are worked.

static_assert(sizeof(Piece) == 1 && NO_PIECE == 12, "the finder reads the mailbox as bytes, 12 for an empty square");
static inline void rel_snapshot(const Position& pos, v5rel::Bd& b) {
    std::memcpy(b.pc, pos.mailbox(), 64);
    for (int c = 0; c < 2; ++c)
        for (int pt = PAWN; pt <= KING; ++pt) b.by[c * 6 + pt] = pos.pieces(Color(c), PieceType(pt));
    b.by[12] = 0;
    b.occ = pos.pieces();
}

static inline void rel_apply(const int16_t* parent, int16_t* child, const uint16_t* add, int na, const uint16_t* sub, int ns) {
    const int8_t* ra[v5rel::MAX_ACTIVE];
    const int8_t* rs[v5rel::MAX_ROWS];
    const int8_t* const W = g_rel_w.data();
    const size_t T = size_t(g_rel_T);
    for (int i = 0; i < na; ++i) ra[i] = W + size_t(add[i]) * T;
    for (int i = 0; i < ns; ++i) rs[i] = W + size_t(sub[i]) * T;
    NNUE_K(rel_rows)(parent, child, ra, na, rs, ns, g_rel_T);
}

static inline void rel_prefetch(const uint16_t* rows, int n) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    const char* const W = reinterpret_cast<const char*>(g_rel_w.data());
    const size_t T = size_t(g_rel_T);
    for (int i = 0; i < n; ++i)
        for (size_t o = 0; o < T; o += 64) _mm_prefetch(W + size_t(rows[i]) * T + o, _MM_HINT_T0);
#else
    (void)rows; (void)n;
#endif
}

// One side's relation lanes from the board alone.
static void rel_rebuild(const v5rel::Bd& b, int p, int16_t* lanes) {
    alignas(64) static const int16_t zero[NNUE_L1_MAX] = {};
    uint16_t rows[v5rel::MAX_ACTIVE];
    const int n = v5rel::enumerate(b, p, g_rel_ppm, rows);
    rel_apply(zero, lanes, rows, n, nullptr, 0);
}

struct RelPlan {
    bool todo[2], cold[2];
    int start[2];
    int lo;          // the first ply whose rows are needed
    bool buffered;   // the rows of plies lo..ply sit in rel_buf()
};

// Before the piece-square work: which plies each side's relation lanes need; find their rows and prefetch them.
static void rel_plan(int ply, const PlyDelta* ds, RelPlan& rp, bool pr) {
    rp.lo = ply + 1;
    rp.buffered = false;
    for (int p = 0; p < 2; ++p) {
        rp.todo[p] = !ds[ply].relc[p];
        rp.cold[p] = false;
        rp.start[p] = ply;
        if (!rp.todo[p]) continue;
        int j = ply;
        while (j > 0 && !ds[j].relc[p] && !ds[j].relflip[p]) --j;
        if (!ds[j].relc[p] && !ds[j].relflip[p]) { rp.cold[p] = true; continue; }   // nothing below computed on this thread
        rp.start[p] = j;
        rp.lo = std::min(rp.lo, j + 1);
    }
    if (rp.lo > ply || ply - rp.lo + 1 > REL_WALK) return;
    const long long t0 = pr ? rdtsc() : 0;
    v5rel::RowLists* const buf = rel_buf();
    const v5rel::Bd* const boards = rel_boards();
    for (int i = rp.lo; i <= ply; ++i) {
        if (ds[i].null_move) continue;
        const bool need[2] = {rp.todo[0] && !rp.cold[0] && i > rp.start[0], rp.todo[1] && !rp.cold[1] && i > rp.start[1]};
        v5rel::RowLists& L = buf[i - rp.lo];
        v5rel::find(boards[i - 1], ds[i].move, g_rel_ppm, need, L);
        for (int p = 0; p < 2; ++p)
            if (need[p]) { rel_prefetch(L.add[p], L.na[p]); rel_prefetch(L.sub[p], L.ns[p]); }
    }
    rp.buffered = true;
    if (pr) { st_rel_find.cyc += rdtsc() - t0; st_rel_find.n += ply - rp.lo + 1; }
}

// After the piece-square work: bring this ply's relation lanes up to date.
static void rel_finish(const Position& pos, int ply, PlyDelta* ds, const RelPlan& rp, bool pr) {
    int16_t* const R = rel_lane_stack();
    v5rel::Bd* const boards = rel_boards();
    for (int p = 0; p < 2; ++p) {
        if (!rp.todo[p]) continue;
        if (rp.cold[p]) {
            const long long t0 = pr ? rdtsc() : 0;
            rel_snapshot(pos, boards[ply]);
            rel_rebuild(boards[ply], p, rel_side(R, ply, p));
            ds[ply].relc[p] = true;
            if (pr) { st_rel_rebuild.cyc += rdtsc() - t0; st_rel_rebuild.n++; }
            continue;
        }
        const int j = rp.start[p];
        if (!ds[j].relc[p]) {   // this side's king crossed d/e at j: every row changed
            const long long t0 = pr ? rdtsc() : 0;
            rel_rebuild(boards[j], p, rel_side(R, j, p));
            ds[j].relc[p] = true;
            if (pr) { st_rel_rebuild.cyc += rdtsc() - t0; st_rel_rebuild.n++; }
        }
        const long long t0 = pr ? rdtsc() : 0;
        for (int i = j + 1; i <= ply; ++i) {
            if (ds[i].null_move) {
                std::memcpy(rel_side(R, i, p), rel_side(R, i - 1, p), sizeof(int16_t) * size_t(g_rel_T));
            } else {
                v5rel::RowLists local;
                const v5rel::RowLists* L;
                if (rp.buffered) L = &rel_buf()[i - rp.lo];
                else {   // a walk longer than the buffer: find the rows here
                    const bool need[2] = {p == 0, p == 1};
                    v5rel::find(boards[i - 1], ds[i].move, g_rel_ppm, need, local);
                    L = &local;
                }
                if (L->overflow[p]) rel_rebuild(boards[i], p, rel_side(R, i, p));
                else rel_apply(rel_side(R, i - 1, p), rel_side(R, i, p), L->add[p], L->na[p], L->sub[p], L->ns[p]);
                if (pr) { g_rel_rows += L->na[p] + L->ns[p]; ++g_rel_plies; }
            }
            ds[i].relc[p] = true;
        }
        if (pr) { st_rel_apply.cyc += rdtsc() - t0; st_rel_apply.n++; }
    }
}

// LUMINEX_REL_CHECK: the lanes kept move by move against a rebuild from the board (relation lanes, exact: abort on
// a difference) and, at 2, every piece-square lane against its exact int32 sum (counted, for print_stats).
// LUMINEX_REL_CHECK=3: the kernel's combination against the spec's formula, lane by lane in plain integers, both
// through the same tail: the outputs must be the same float. Out of line, so evaluate() stays as it was.
[[gnu::noinline]] static void rel_check_output(const Position& pos, const int16_t* acc_stm, const int16_t* acc_nstm,
                                               const RelLanes& rel, const TailView& tail, float ov) {
    alignas(64) int16_t cs[NNUE_L1_MAX], cn[NNUE_L1_MAX];
    for (int l = 0; l < g_L1; ++l) {
        cs[l] = acc_stm[l];
        cn[l] = acc_nstm[l];
        if (l < g_rel_T) {
            cs[l] = int16_t(std::clamp(int(acc_stm[l]) + std::clamp<int>(rel.stm[l], -NNUE_REL_L, NNUE_REL_L) * (1 << g_rel_rs), -32768, 32767));
            cn[l] = int16_t(std::clamp(int(acc_nstm[l]) + std::clamp<int>(rel.nstm[l], -NNUE_REL_L, NNUE_REL_L) * (1 << g_rel_rs), -32768, 32767));
        }
    }
    const float ref = NNUE_K(forward_int8)(cs, cn, tail, g_L1, g_L2, g_L3, nullptr);
    if (ref != ov) {
        std::fprintf(stderr, "NNUE RELATION COMBINATION MISMATCH kernel=%.9g formula=%.9g fen=%s\n", ov, ref, pos.fen().c_str());
        std::abort();
    }
}

static void rel_check(const Position& pos) {
    v5rel::Bd b;
    rel_snapshot(pos, b);
    const int ply = pos.state_ply();
    int16_t* const R = rel_lane_stack();
    for (int p = 0; p < 2; ++p) {
        alignas(64) int16_t want[NNUE_L1_MAX];
        rel_rebuild(b, p, want);
        const int16_t* have = rel_side(R, ply, p);
        if (std::memcmp(want, have, sizeof(int16_t) * size_t(g_rel_T)) != 0) {
            int l = 0;
            while (want[l] == have[l]) ++l;
            std::fprintf(stderr, "NNUE RELATION LANES MISMATCH ply=%d side=%d lane=%d rebuilt=%d kept=%d fen=%s\n",
                         ply, p, l, want[l], have[l], pos.fen().c_str());
            std::abort();
        }
    }
    ++g_rel_checks;
    if (g_rel_check < 2) return;
    const Accumulator& a = nnue_acc_stack()[ply];
    for (int p = 0; p < 2; ++p) {
        const bool white_pov = (p == 0);
        const int ksq = static_cast<int>(pos.king_sq(white_pov ? WHITE : BLACK));
        std::vector<int32_t> ref(ft_b_i32.begin(), ft_b_i32.end());
        for (int sq = 0; sq < NUM_SQ; ++sq) {
            const Piece pc = pos.piece_on(Square(sq));
            if (pc == NO_PIECE) continue;
            const int16_t* row = &ft_w[static_cast<size_t>(halfka_idx(white_pov, ksq, sq, pc)) * g_L1];
            for (int l = 0; l < g_L1; ++l) ref[l] += row[l];
        }
        for (int l = 0; l < g_L1; ++l)
            if (ref[l] >= -32768 && ref[l] <= 32767) { ++g_psq_checked; g_psq_mismatch += (ref[l] != a.v[p][l]); }
    }
}

static void make_current(const Position& pos) {
    const int ply = pos.state_ply();
    PlyDelta* const ds = nnue_deltas();
    if (ds[ply].current[0] && ds[ply].current[1] && (!g_rel_T || (ds[ply].relc[0] && ds[ply].relc[1]))) return;
    const bool pr = profile_on();
    const long long t0 = pr ? rdtsc() : 0;
    Accumulator* const acc_stack = nnue_acc_stack();
    RelPlan relp;
    if (g_rel_T) rel_plan(ply, ds, relp, pr);
    int start[2];
    for (int p = 0; p < 2; ++p) {
        int j = ply;
        while (j > 0 && !ds[j].current[p] && ds[j].rebuild != p) --j;
        if (!ds[j].current[p]) {
            if (ds[j].rebuild != p) {
                // Nothing below was ever computed on this thread (no refresh
                // since the net loaded): rebuild this ply from the board.
                refresh_perspective(pos, acc_stack[ply], p);
                ds[ply].current[p] = true;
                j = ply;
            } else {
                const long long kr_t0 = pr ? rdtsc() : 0;
                refresh_from_board(nnue_king_boards()[j].pieces, ds[j].ksq[p], acc_stack[j], p);
                ds[j].current[p] = true;
                if (pr) { st_king_refresh.cyc += rdtsc() - kr_t0; st_king_refresh.n++; }
            }
        }
        start[p] = j;
    }
    for (int i = std::min(start[0], start[1]) + 1; i <= ply; ++i) {
        const int p_begin = start[0] < i ? 0 : 1;
        const int p_end   = start[1] < i ? 2 : 1;
        const long long inc_t0 = pr ? rdtsc() : 0;
        apply_delta(ds[i], acc_stack[i - 1], acc_stack[i], p_begin, p_end);
        for (int p = p_begin; p < p_end; ++p) ds[i].current[p] = true;
        if (pr) { st_incremental.cyc += rdtsc() - inc_t0; st_incremental.n += p_end - p_begin; }
    }
    if (g_rel_T) rel_finish(pos, ply, ds, relp, pr);
    if (pr) { st_catchup.cyc += rdtsc() - t0; st_catchup.n++; }
}

void refresh(Position& pos) {
    if (!g_loaded) return;
    bool pr = profile_on();
    long long t0 = pr ? rdtsc() : 0;
    Accumulator* const acc_stack = nnue_acc_stack();
    Accumulator& a = acc_stack[pos.state_ply()];
    refresh_perspective(pos, a, 0);
    refresh_perspective(pos, a, 1);
    PlyDelta& d = nnue_deltas()[pos.state_ply()];
    d.null_move = false;
    d.rebuild = -1;
    d.current[0] = d.current[1] = true;
    if (g_rel_T) {
        v5rel::Bd& b = rel_boards()[pos.state_ply()];
        rel_snapshot(pos, b);
        int16_t* const R = rel_lane_stack();
        rel_rebuild(b, 0, rel_side(R, pos.state_ply(), 0));
        rel_rebuild(b, 1, rel_side(R, pos.state_ply(), 1));
        d.relc[0] = d.relc[1] = true;
        d.relflip[0] = d.relflip[1] = false;
    }
    if (pr) { st_refresh.cyc += rdtsc() - t0; st_refresh.n++; }
}

void copy_for_null(Position& pos) {
    if (!g_loaded) return;
    // The board is unchanged: the child's lanes will be a copy of the parent's,
    // made when an evaluation needs them.
    PlyDelta& d = nnue_deltas()[pos.state_ply()];
    d.null_move = true;
    d.rebuild = -1;
    d.current[0] = d.current[1] = false;
    if (g_rel_T) {
        v5rel::Bd* const boards = rel_boards();
        boards[pos.state_ply()] = boards[pos.state_ply() - 1];
        d.relc[0] = d.relc[1] = false;
        d.relflip[0] = d.relflip[1] = false;
    }
}

void update(Position& pos, Move m, Piece moved, PieceType captured) {
    if (!g_loaded) return;

    bool pr = profile_on();
    long long t0 = pr ? rdtsc() : 0;
    const int ply = pos.state_ply();
    const Color us = color_of_piece(moved);
    const PieceType moved_pt = piece_type_of(moved);

    // Record the move; make_current applies it when an evaluation needs this ply.
    PlyDelta& d = nnue_deltas()[ply];
    d.move = m;
    d.moved = moved;
    d.captured = captured;
    d.ksq[0] = uint8_t(pos.king_sq(WHITE));
    d.ksq[1] = uint8_t(pos.king_sq(BLACK));
    d.rebuild = moved_pt == KING ? int8_t(us == WHITE ? 0 : 1) : int8_t(-1);
    d.null_move = false;
    d.current[0] = d.current[1] = false;
    if (g_rel_T) {
        rel_snapshot(pos, rel_boards()[ply]);
        d.relc[0] = d.relc[1] = false;
        d.relflip[0] = d.relflip[1] = false;
        if (moved_pt == KING && ((int(m.from()) & 7) < 4) != ((int(m.to()) & 7) < 4)) d.relflip[us == WHITE ? 0 : 1] = true;
    }
    if (d.rebuild >= 0) {
        KingBoard& kb = nnue_king_boards()[ply];
        for (int c = 0; c < 2; ++c)
            for (int pt = PAWN; pt <= KING; ++pt)
                kb.pieces[c][pt] = pos.pieces(Color(c), PieceType(pt));
    }

    // Prefetch the feature weight rows the delta reads: a child evaluated soon
    // (most qsearch nodes) then finds them in cache. x86 only: other targets
    // have no prefetch here.
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    {
        const bool promo = m.is_promotion();
        const PieceType from_pt = promo ? PAWN : moved_pt;
        const PieceType to_pt   = promo ? PieceType(m.promotion_type()) : moved_pt;
        const Square from = m.from(), to = m.to();
        int wk = static_cast<int>(pos.king_sq(WHITE));
        int bk = static_cast<int>(pos.king_sq(BLACK));
        Piece dep_piece = make_piece(us, from_pt);
        Piece arr_piece = make_piece(us, to_pt);
        for (int i = 0; i < 4; ++i) {  // up to 4 prefetches per perspective
            int idx = -1;
            if (i == 0) idx = halfka_idx(true,  wk, from, dep_piece);
            else if (i == 1) idx = halfka_idx(true,  wk, to, arr_piece);
            else if (i == 2) idx = halfka_idx(false, bk, from, dep_piece);
            else idx = halfka_idx(false, bk, to, arr_piece);
            _mm_prefetch((const char*)&ft_w[static_cast<size_t>(idx) * g_L1], _MM_HINT_T0);
        }
    }
#endif
    if (pr) { st_update.cyc += rdtsc() - t0; st_update.n++; }
}


Value evaluate(const Position& pos) {
    // Bring this ply's accumulator up to date first (lazy updates); its cost is
    // timed apart from st_eval, so cyc/eval stays the forward pass alone.
    make_current(pos);
    bool pr = profile_on();
    long long t0 = pr ? rdtsc() : 0;
    Accumulator* const acc_stack = nnue_acc_stack();
#if !defined(NDEBUG) || defined(NNUE_FORCE_SELFCHECK)
    // Self-check (debug builds, or release with -DNNUE_FORCE_SELFCHECK): recompute the
    // accumulator from scratch and compare to the incrementally-maintained one. Catches
    // any bug in update()'s feature deltas immediately.
    if (g_loaded) {
        const Accumulator& inc = acc_stack[pos.state_ply()];
        for (int p = 0; p < 2; ++p) {
            bool white_pov = (p == 0);
            int ksq = static_cast<int>(pos.king_sq(white_pov ? WHITE : BLACK));
            for (int l = 0; l < g_L1; ++l) {
                int32_t ref = ft_b_i32[l];
                for (int sq = 0; sq < NUM_SQ; ++sq) {
                    Piece pc = pos.piece_on(Square(sq));
                    if (pc == NO_PIECE) continue;
                    int idx = halfka_idx(white_pov, ksq, sq, pc);
                    // the loop computed idx and discarded it — the "reference"
                    // was the bias alone, so any real position aborted here
                    ref += ft_w[(size_t)idx * g_L1 + l];
                }
                // The lane must be the exact sum whenever that fits int16: the
                // updates wrap, so a partial sum that left int16 on the way
                // cannot have changed it. g_sat_events counts sums that do not
                // fit (none on the x2048 grid so far); such a lane wraps until
                // its sum is back in range, and is not compared.
                if (ref < -32768 || ref > 32767) { ++g_sat_events; continue; }
                if (inc.v[p][l] != ref) {
                    std::fprintf(stderr, "NNUE ACC MISMATCH ply=%d p=%d l=%d ref=%d inc=%d\n",
                                 pos.state_ply(), p, l, int(ref), int(inc.v[p][l]));
                    std::abort();
                }
            }
        }
    }
#endif
    const Accumulator& a = acc_stack[pos.state_ply()];
    bool stm_white = (pos.side_to_move() == WHITE);
    const int16_t* acc_stm  = stm_white ? a.v[0] : a.v[1];
    const int16_t* acc_nstm = stm_white ? a.v[1] : a.v[0];
    RelLanes rel{};
    const RelLanes* relp = nullptr;
    if (g_rel_T) {
        int16_t* const R = rel_lane_stack();
        rel.stm  = rel_side(R, pos.state_ply(), stm_white ? 0 : 1);
        rel.nstm = rel_side(R, pos.state_ply(), stm_white ? 1 : 0);
        rel.T = g_rel_T;
        rel.rs = g_rel_rs;
        relp = &rel;
#if !defined(NDEBUG) || defined(NNUE_FORCE_SELFCHECK)
        rel_check(pos);   // debug builds check every evaluation, as the piece-square self-check above does
#else
        if (g_rel_check) rel_check(pos);
#endif
    }
    // This position's dense tail (the only one unless the net has output
    // buckets), as the kernels read it.
    const TailView& tail = g_tails[bucket_of(pos)].view;
    float ov;
    if (g_int8) {
        ov = relp ? NNUE_K(forward_int8_rel)(acc_stm, acc_nstm, *relp, tail, g_L1, g_L2, g_L3, pr ? &g_act_nz : nullptr)
                  : NNUE_K(forward_int8)(acc_stm, acc_nstm, tail, g_L1, g_L2, g_L3, pr ? &g_act_nz : nullptr);
        if (pr) g_act_tot += 2 * g_L1;
        if (relp && g_rel_check >= 3) rel_check_output(pos, acc_stm, acc_nstm, rel, tail, ov);
    } else {
        ov = NNUE_K(forward_float)(acc_stm, acc_nstm, tail, g_L1, g_L2, g_L3);
    }
    Value v = static_cast<Value>(std::llround(ov * OUT_SCALE));
    if (pr) { st_eval.cyc += rdtsc() - t0; st_eval.n++; }
    return v;
}

// ---- gen768t debugging: the relrows and lanes UCI commands ----

std::string relation_rows(const Position& pos, int ppm) {
    v5rel::init();
    v5rel::Bd b;
    rel_snapshot(pos, b);
    std::string out;
    for (int p = 0; p < 2; ++p) {
        uint16_t rows[v5rel::MAX_ACTIVE];
        const int n = v5rel::enumerate(b, p, ppm, rows);
        std::sort(rows, rows + n);
        out += "relrows " + std::to_string(p) + " " + std::to_string(n);
        for (int i = 0; i < n; ++i) out += " " + std::to_string(rows[i]);
        out += "\n";
    }
    return out;
}

std::string lane_dump(const Position& pos) {
    if (!g_loaded) return "lanes: no network loaded\n";
    make_current(pos);
    const Accumulator& a = nnue_acc_stack()[pos.state_ply()];
    int16_t* const R = g_rel_T ? rel_lane_stack() : nullptr;
    std::string out;
    for (int p = 0; p < 2; ++p) {
        out += "lanes " + std::to_string(p);
        for (int l = 0; l < g_L1; ++l) {
            int v = a.v[p][l];
            if (l < g_rel_T)
                v = std::clamp(v + std::clamp<int>(rel_side(R, pos.state_ply(), p)[l], -NNUE_REL_L, NNUE_REL_L) * (1 << g_rel_rs), -32768, 32767);
            out += " " + std::to_string(v);
        }
        out += "\n";
        if (g_rel_T) {
            out += "rel " + std::to_string(p);
            for (int l = 0; l < g_rel_T; ++l) out += " " + std::to_string(rel_side(R, pos.state_ply(), p)[l]);
            out += "\n";
        }
    }
    return out;
}

// ---- nnuecheck: the same evaluations at every kernel level ----
//
// A fixed set of games from a few positions, by moves a seeded generator
// picks, evaluated along the way as a search evaluates: after most moves (the
// plies left out make make_current catch up over several), after a sibling
// move tried and taken back, after a null move. Every level the CPU supports
// plays the same games, each in a fresh thread (its own accumulators and
// refresh cache), and their evaluations must agree exactly. The digest of the
// evaluations compares builds: x86-64 builds of the same network print the
// same one.
namespace {

const char* const CHECK_FENS[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "8/PPP4k/8/8/8/8/4Kppp/8 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
};
constexpr int CHECK_GAMES = 6;     // per position
constexpr int CHECK_PLIES = 100;   // per game, at most

uint64_t splitmix64(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

struct CheckEval {
    int32_t value;
    uint8_t fen, game, kind;   // kind: 0 after a move, 1 after a sibling, 2 after a null move
    uint16_t ply;
};
const char* const CHECK_KINDS[] = {"a move", "a sibling move", "a null move"};

std::vector<CheckEval> check_games() {
    std::vector<CheckEval> out;
    auto pos = std::make_unique<Position>();
    for (int f = 0; f < int(std::size(CHECK_FENS)); ++f)
        for (int g = 0; g < CHECK_GAMES; ++g) {
            uint64_t seed = 0x6C756D696E6578ull + uint64_t(f) * 1000 + uint64_t(g);
            pos->set(CHECK_FENS[f]);
            auto record = [&](int ply, int kind) {
                out.push_back({int32_t(nnue::evaluate(*pos)), uint8_t(f), uint8_t(g), uint8_t(kind), uint16_t(ply)});
            };
            record(0, 0);
            for (int ply = 1; ply <= CHECK_PLIES; ++ply) {
                ScoredMove moves[MAX_MOVES];
                const int n = int(generate<MOVES_LEGAL>(*pos, moves) - moves);
                if (n == 0) break;
                const uint64_t r = splitmix64(seed);
                if (n > 1 && (r & 3) == 0) {
                    const Move s = moves[(r >> 8) % uint64_t(n)].move;
                    if (pos->do_move(s)) { record(ply, 1); pos->undo_move(s); }
                }
                if ((r >> 2) % 7 == 0 && !pos->is_check()) {
                    pos->do_null_move();
                    record(ply, 2);
                    pos->undo_null_move();
                }
                const Move m = moves[(r >> 16) % uint64_t(n)].move;
                if (!pos->do_move(m)) break;
                if ((r >> 40) % 4 != 0) record(ply, 0);
            }
        }
    return out;
}

// FNV-1a over the evaluations' little-endian bytes.
uint64_t check_digest(const std::vector<CheckEval>& evals) {
    uint64_t h = 0xCBF29CE484222325ull;
    for (const CheckEval& e : evals) {
        const uint32_t v = uint32_t(e.value);
        for (int i = 0; i < 4; ++i) { h ^= (v >> (8 * i)) & 0xFF; h *= 0x100000001B3ull; }
    }
    return h;
}

#if defined(LUMINEX_NNUE_DISPATCH)
// The base level's software bias add (no FMA) against a level's hardware FMA:
// random operands, and operands whose sum in double lands exactly halfway
// between two floats while the exact sum does not, where rounding the double
// would go the wrong way about half the time (naive_wrong counts it).
struct RoundingCheck {
    long long cases = 0, halfway = 0, naive_wrong = 0;
    std::string mismatch;
};
RoundingCheck check_bias_rounding(const Kernels& soft, const Kernels& hard) {
    RoundingCheck res;
    auto bits = [](float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; };
    auto test = [&](float b, int32_t d, float s) {
        ++res.cases;
        const float want = hard.bias_add(b, d, s), got = soft.bias_add(b, d, s);
        if (bits(want) != bits(got) && res.mismatch.empty()) {
            char buf[200];
            std::snprintf(buf, sizeof buf, "bias_add(%a, %d, %a) is %a on %s, %a on %s",
                          double(b), int(d), double(s), double(want), hard.name, double(got), soft.name);
            res.mismatch = buf;
        }
    };
    uint64_t seed = 0x5EED;
    for (int i = 0; i < 1000000; ++i) {
        const uint64_t r1 = splitmix64(seed), r2 = splitmix64(seed);
        const float mb = float((r1 & 0x7FFFFF) | 0x800000), ms = float((r2 & 0x7FFFFF) | 0x800000);
        const float b = std::ldexp((r1 >> 63) ? -mb : mb, int((r1 >> 24) % 34) - 53);   // |b| in [2^-30, 2^4)
        const float s = std::ldexp(ms, -int((r2 >> 24) % 21) - 23);                      // s in [2^-20, 1)
        const int32_t d = int32_t((r2 >> 32) & 0x3FFFFFF) - 0x2000000;                   // |d| <= 2^25
        test(b, d, s);
    }
    // a * c = 2^47 +- m with 0 < m < 2^18 makes float(a) * c * 2^-71 = 2^-24
    // +- less than 2^-53, so 1 + it (or 1 + 2^-23 + it) rounds in double to
    // the midpoint of two floats; scaled and mirrored.
    for (uint64_t k = 0; k < 40000; ++k) {
        const uint64_t a = (1ull << 23) + (k * 2654435761ull) % (1ull << 23);
        for (int up = 0; up < 2; ++up) {
            const uint64_t c = up ? ((1ull << 47) + a - 1) / a : (1ull << 47) / a;
            const uint64_t prod = a * c;
            const uint64_t m = up ? prod - (1ull << 47) : (1ull << 47) - prod;
            if (m == 0 || m >= (1ull << 18) || c > (1ull << 24)) continue;
            for (float b0 : {1.0f, 1.0f + 0x1p-23f})
                for (int e : {-12, 0, 9})
                    for (int sign : {1, -1}) {
                        const float b = std::ldexp(sign * b0, e);
                        const int32_t d = sign * int32_t(a);
                        const float s = std::ldexp(float(c), e - 71);
                        const double r = double(float(d)) * double(s) + double(b);
                        uint64_t u;
                        std::memcpy(&u, &r, sizeof u);
                        if ((u & 0x1FFFFFFFull) == 0x10000000ull) {
                            ++res.halfway;
                            if (bits(float(r)) != bits(hard.bias_add(b, d, s))) ++res.naive_wrong;
                        }
                        test(b, d, s);
                    }
        }
    }
    return res;
}
#endif

} // namespace

bool check_simd(std::string& report) {
    if (!g_loaded) { report = "nnuecheck FAILED: no network loaded"; return false; }
    if (!g_int8) {
        report = "nnuecheck FAILED: a float (LNN1) network; its levels sum in different orders, only int8 networks evaluate alike";
        return false;
    }
    choose_kernels();
    [[maybe_unused]] const Kernels* const in_use = g_k;
    const bool* ok = levels_supported();
    std::vector<int> levels;
    for (int i = 0; i < NUM_LEVELS; ++i)
        if (ok[i]) levels.push_back(i);
    std::vector<std::vector<CheckEval>> runs;
    for ([[maybe_unused]] int i : levels) {
#if defined(LUMINEX_NNUE_DISPATCH)
        g_k = LEVELS[i];
#endif
        arrange_tails();
        std::vector<CheckEval> evals;
        std::thread worker([&evals] { evals = check_games(); });
        worker.join();
        runs.push_back(std::move(evals));
    }
#if defined(LUMINEX_NNUE_DISPATCH)
    g_k = in_use;
#endif
    arrange_tails();

    std::string names;
    for (int i : levels) names += std::string(names.empty() ? "" : " ") + LEVELS[i]->name;
    report = "nnuecheck: levels " + names + " (in use: " + g_k->name + ")\n";
    char buf[320];
    std::snprintf(buf, sizeof buf, "nnuecheck: %zu evaluations per level, digest %016llx\n",
                  runs[0].size(), (unsigned long long)check_digest(runs[0]));
    report += buf;
    std::string failure;
    for (size_t k = 1; k < runs.size() && failure.empty(); ++k) {
        const std::vector<CheckEval>& want = runs[0];
        const std::vector<CheckEval>& got = runs[k];
        for (size_t j = 0; j < want.size() && failure.empty(); ++j)
            if (j >= got.size() || got[j].value != want[j].value) {
                const CheckEval& e = want[j];
                std::snprintf(buf, sizeof buf,
                              "%s differs from %s at evaluation %zu (position %d, game %d, ply %d, after %s): %d vs %d",
                              LEVELS[levels[k]]->name, LEVELS[levels[0]]->name, j, int(e.fen) + 1, int(e.game) + 1,
                              int(e.ply), CHECK_KINDS[e.kind], j < got.size() ? int(got[j].value) : 0, int(e.value));
                failure = buf;
            }
        if (failure.empty() && got.size() != want.size())
            failure = std::string(LEVELS[levels[k]]->name) + " made a different number of evaluations";
    }
#if defined(LUMINEX_NNUE_DISPATCH)
    // The base level's bias add is software; test it against hardware FMA
    // where the CPU has one (every level above base).
    if (failure.empty() && levels.size() > 1) {
        const Kernels& soft = *LEVELS[NUM_LEVELS - 1];
        const Kernels& hard = *LEVELS[levels[levels.size() - 2]];
        const RoundingCheck rc = check_bias_rounding(soft, hard);
        std::snprintf(buf, sizeof buf,
                      "nnuecheck: %s's bias rounding against %s's FMA: %lld operand sets, %lld halfway in double (%lld of them wrong if rounded twice)\n",
                      soft.name, hard.name, rc.cases, rc.halfway, rc.naive_wrong);
        report += buf;
        if (!rc.mismatch.empty()) failure = rc.mismatch;
        else if (rc.halfway == 0 || rc.naive_wrong == 0) failure = "the halfway rounding cases did not come out halfway";
    }
#endif
    if (!failure.empty()) {
        report += "nnuecheck FAILED: " + failure;
        return false;
    }
    report += "nnuecheck ok";
    return true;
}

} // namespace luminex::nnue

// =====================================================================================================================
// gen768t's relation inputs
// =====================================================================================================================

// The rules are those of the reference script the V5 design work checked everything
// against (v5-relations-ref.py, on python-chess). This is the code the V5 speed gate measured: a census checked
// its change-finder against a brute force on 18.1M search moves, and its rows matched the trainer's on game and
// search-tree positions.
#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>

namespace luminex::v5rel {

using u64 = uint64_t;
namespace {
constexpr int NOPC = 12;
constexpr u64 RANKS27 = 0x00FFFFFFFFFFFF00ULL;

inline int pop(u64& b) {
    const int s = std::countr_zero(b);
    b &= b - 1;
    return s;
}
inline int popc(u64 b) { return std::popcount(b); }
inline u64 bit(int s) { return 1ULL << s; }
inline bool same_diag(int x, int y) { return std::abs((x & 7) - (y & 7)) == std::abs((x >> 3) - (y >> 3)); }

// Tables, generated from the rules (no constant is typed in).
struct Tables {
    u64 below[64], beyond[64][64], files3[8];
    u64 e_pup[64], e_pdn[64], e_n[64], e_k[64], e_d[64], e_o[64];   // empty-board attack sets; "up" = toward rank 8
    u64 patt[2][64];                                                // squares a pawn of colour c on s attacks
    int16_t blk[2][6][2][5];                                        // [attacker side][type][victim side][type] -> block
    int n_blk = 0, n_rel = 0, n_pp = 0;
    u64 mask[128][64];
    int boff[128][64];                                              // the block's base + the square's offset
    int16_t pp[96][96];
    struct Cls { int16_t blk; uint8_t swap, unordered; } cls[2][12][12];   // [perspective][attacker piece][victim piece]

    static int step(int s, int df, int dr) {
        const int f = (s & 7) + df, r = (s >> 3) + dr;
        return (f < 0 || f > 7 || r < 0 || r > 7) ? -1 : r * 8 + f;
    }
    static u64 pawn_facing(int x, int sv) {      // squares from which a pawn of side sv (0 = own, moving up) attacks x
        u64 m = 0;
        for (int df = -1; df <= 1; df += 2) {
            const int t = step(x, df, sv == 0 ? -1 : 1);
            if (t >= 0) m |= bit(t);
        }
        return m & RANKS27;
    }
    u64 block_mask(int sa, int ta, int sv, int tv, int x) const {
        if (ta == 5 && sa == 0) return 0;        // the own king's attacks: the king-bucketed piece-squares carry them
        if (ta == 0 && tv == 0) return 0;        // pawn on pawn: the pawn-pair group
        u64 m;
        switch (ta) {
            case 0:  m = sa == 0 ? e_pup[x] : e_pdn[x]; break;
            case 1:  m = e_n[x]; break;
            case 2:  m = e_d[x]; break;
            case 3:  m = e_o[x]; break;
            case 4:  m = tv == 2 ? e_o[x] : tv == 3 ? e_d[x] : (e_d[x] | e_o[x]); break;
            default: m = e_k[x]; break;
        }
        if (tv == 0) {
            m &= RANKS27;
            if (ta == 2 || ta == 4) m &= ~pawn_facing(x, sv);
        }
        if (ta == tv && ta >= 1 && ta <= 4) {
            if (sa == sv) m &= ~(bit(x) | below[x]);
            else if (sa == 1) return 0;
        }
        return m;
    }
    Tables() {
        static const int KN[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
        static const int KD[8][2] = {{1, 1}, {-1, 1}, {-1, -1}, {1, -1}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        std::memset(beyond, 0, sizeof beyond);
        for (int s = 0; s < 64; ++s) {
            below[s] = bit(s) - 1;
            e_pup[s] = e_pdn[s] = e_n[s] = e_k[s] = e_d[s] = e_o[s] = 0;
            patt[0][s] = patt[1][s] = 0;
            for (int df = -1; df <= 1; df += 2) {
                int t = step(s, df, 1);
                if (t >= 0) { patt[0][s] |= bit(t); if (s >= 8 && s < 56) e_pup[s] |= bit(t); }
                t = step(s, df, -1);
                if (t >= 0) { patt[1][s] |= bit(t); if (s >= 8 && s < 56) e_pdn[s] |= bit(t); }
            }
            for (const auto& d : KN) {
                const int t = step(s, d[0], d[1]);
                if (t >= 0) e_n[s] |= bit(t);
            }
            for (int i = 0; i < 8; ++i) {
                int ray[8], n = 0;
                for (int t = step(s, KD[i][0], KD[i][1]); t >= 0; t = step(t, KD[i][0], KD[i][1])) ray[n++] = t;
                if (n) e_k[s] |= bit(ray[0]);
                for (int a = 0; a < n; ++a) {
                    (i < 4 ? e_d[s] : e_o[s]) |= bit(ray[a]);
                    for (int c = a + 1; c < n; ++c) beyond[s][ray[a]] |= bit(ray[c]);
                }
            }
        }
        for (int f = 0; f < 8; ++f) {
            files3[f] = 0;
            for (int g = std::max(0, f - 1); g <= std::min(7, f + 1); ++g) files3[f] |= 0x0101010101010101ULL << g;
        }
        for (int sa = 0; sa < 2; ++sa)
            for (int ta = 0; ta < 6; ++ta)
                for (int sv = 0; sv < 2; ++sv)
                    for (int tv = 0; tv < 5; ++tv) {
                        int size = 0;
                        for (int x = 0; x < 64; ++x) size += popc(block_mask(sa, ta, sv, tv, x));
                        if (!size) { blk[sa][ta][sv][tv] = -1; continue; }
                        const int b = n_blk++;
                        blk[sa][ta][sv][tv] = int16_t(b);
                        int acc = 0;
                        for (int x = 0; x < 64; ++x) {
                            mask[b][x] = block_mask(sa, ta, sv, tv, x);
                            boff[b][x] = n_rel + acc;
                            acc += popc(mask[b][x]);
                        }
                        n_rel += size;
                    }
        for (int i = 0; i < 96; ++i)
            for (int j = 0; j < 96; ++j) {
                pp[i][j] = -1;
                if (j <= i) continue;
                const int qi = i % 48, qj = j % 48;
                if (qi == qj || std::abs(((qi + 8) & 7) - ((qj + 8) & 7)) > 1) continue;
                pp[i][j] = int16_t(n_pp++);
            }
        for (int persp = 0; persp < 2; ++persp)
            for (int pa = 0; pa < 12; ++pa)
                for (int pv = 0; pv < 12; ++pv) {
                    Cls& c = cls[persp][pa][pv];
                    const int ta = pa % 6, tv = pv % 6;
                    int sa = (pa / 6) != persp, sv = (pv / 6) != persp;
                    c.swap = c.unordered = 0;
                    c.blk = -1;
                    if (tv == 5) continue;
                    if (ta == tv && ta >= 1 && ta <= 4) {
                        if (sa == sv) c.unordered = 1;
                        else if (sa == 1) { c.swap = 1; std::swap(sa, sv); }
                    }
                    c.blk = blk[sa][ta][sv][tv];
                }
    }
};
const Tables* g_t = nullptr;

inline void put(Bd& b, int s, int p) {
    b.pc[s] = uint8_t(p);
    b.by[p] |= bit(s);
    b.occ |= bit(s);
}
inline void take(Bd& b, int s) {
    b.by[b.pc[s]] &= ~bit(s);
    b.occ &= ~bit(s);
    b.pc[s] = NOPC;
}

// Is the attack x -> y (known to hold on the board) a member of the canonical set?
inline bool rel_ok(const Tables& t, int pa, int x, int pv, int y) {
    const int ta = pa % 6, tv = pv % 6;
    if (tv == 5) return false;
    if (ta == 0 && tv == 0) return false;
    if (ta == tv && ta >= 1 && ta <= 4) return x < y;
    if (ta == 4 && tv == 2 && same_diag(x, y)) return false;
    if (ta == 4 && tv == 3 && !same_diag(x, y)) return false;
    if ((ta == 2 || ta == 4) && tv == 0 && (t.patt[pv >= 6][y] & bit(x))) return false;
    return true;
}

inline int rel_row(const Tables& t, int p, int ox, int x, int y, int px, int py) {
    const Tables::Cls& c = t.cls[p][px][py];
    if (c.blk < 0) return -1;
    int xo = x ^ ox, yo = y ^ ox;
    if (c.swap || (c.unordered && xo > yo)) std::swap(xo, yo);
    return t.boff[c.blk][xo] + popc(t.mask[c.blk][xo] & (bit(yo) - 1));
}
inline int pair_row(const Tables& t, int p, int ox, int s1, int s2, int p1, int p2) {
    int i = ((p1 / 6) != p) * 48 + (s1 ^ ox) - 8, j = ((p2 / 6) != p) * 48 + (s2 ^ ox) - 8;
    if (i > j) std::swap(i, j);
    return t.n_rel + t.pp[i][j];
}
inline int orient_of(const Bd& b, int p) {       // XOR 56 for Black, XOR 7 when that side's king is on files a-d
    return (p ? 56 : 0) ^ ((std::countr_zero(b.by[p ? 11 : 5]) & 7) < 4 ? 7 : 0);
}

struct Finder {
    Bd& b;
    const Tables& t;
    const int ppm;
    const int ox[2];
    const bool need[2];
    RowLists& o;

    void emit(int sign, int p, int row) {
        if (sign > 0) {
            if (o.na[p] < MAX_ROWS) o.add[p][o.na[p]++] = uint16_t(row); else o.overflow[p] = true;
        } else {
            if (o.ns[p] < MAX_ROWS) o.sub[p][o.ns[p]++] = uint16_t(row); else o.overflow[p] = true;
        }
    }
    void rel(int sign, int x, int y, int px, int py) {
        for (int p = 0; p < 2; ++p) {
            if (!need[p]) continue;
            const int row = rel_row(t, p, ox[p], x, y, px, py);
            if (row >= 0) emit(sign, p, row);
        }
    }
    void pair(int sign, int s1, int s2) {
        if (s1 > s2) std::swap(s1, s2);
        for (int p = 0; p < 2; ++p)
            if (need[p]) emit(sign, p, pair_row(t, p, ox[p], s1, s2, b.pc[s1], b.pc[s2]));
    }
    // Every feature the piece on z takes part in, as attacker, victim or pawn of a pair.
    // dz, oz: the diagonal and line attack sets from z (they do not depend on what stands on z).
    void piece(int z, int sign, u64 dz, u64 oz) {
        const int pz = b.pc[z], tz = pz % 6;
        const u64 pawns = b.by[0] | b.by[6], knights = b.by[1] | b.by[7], bishops = b.by[2] | b.by[8],
                  rooks = b.by[3] | b.by[9], queens = b.by[4] | b.by[10], kings = b.by[5] | b.by[11];
        const u64 pawn_att = (t.patt[1][z] & b.by[0]) | (t.patt[0][z] & b.by[6]);   // pawns attacking z
        const u64 own_att = tz == 0 ? t.patt[pz >= 6][z] : 0;                        // what a pawn on z attacks
        const u64 lo = t.below[z];
        u64 v;
        switch (tz) {                    // z as attacker: victims, with every rule of the canonical set as a mask
            case 0:  v = own_att & b.occ & ~pawns; break;
            case 1:  v = t.e_n[z] & b.occ & ~(knights & lo); break;
            case 2:  v = dz & b.occ & ~(bishops & lo) & ~pawn_att; break;
            case 3:  v = oz & b.occ & ~(rooks & lo); break;
            case 4:  v = ((dz & ~bishops) | (oz & ~rooks)) & b.occ & ~(queens & lo) & ~pawn_att; break;
            default: v = t.e_k[z] & b.occ; break;
        }
        v &= ~kings;
        while (v) {
            const int y = pop(v);
            rel(sign, z, y, pz, b.pc[y]);
        }
        if (tz != 5) {                   // z as victim: attackers
            const u64 nA = t.e_n[z] & knights, kA = t.e_k[z] & kings;
            const u64 dB = dz & bishops, dQ = dz & queens, oR = oz & rooks, oQ = oz & queens;
            u64 a;
            switch (tz) {
                case 0:  a = nA | kA | ((dB | dQ) & ~own_att) | oR | oQ; break;
                case 1:  a = pawn_att | (nA & lo) | kA | dB | dQ | oR | oQ; break;
                case 2:  a = pawn_att | nA | kA | (dB & lo) | oR | oQ; break;
                case 3:  a = pawn_att | nA | kA | dB | dQ | (oR & lo); break;
                default: a = pawn_att | nA | kA | dB | oR | ((dQ | oQ) & lo); break;
            }
            while (a) {
                const int x = pop(a);
                rel(sign, x, z, b.pc[x], pz);
            }
        }
        if (tz == 0)
            for (u64 c = ppm ? (pawns & t.files3[z & 7] & ~bit(z)) : ((own_att & pawns) | pawn_att); c;) pair(sign, z, pop(c));
    }
    // With z empty: (slider -> the first piece beyond z) for every slider whose line reaches z.
    void through(int z, int sign, u64 dz, u64 oz) {
        const u64 queens = b.by[4] | b.by[10];
        const u64 ends = (dz | oz) & b.occ;
        for (u64 s = (dz & (b.by[2] | b.by[8] | queens)) | (oz & (b.by[3] | b.by[9] | queens)); s;) {
            const int x = pop(s);
            const u64 w = ends & t.beyond[x][z];
            if (!w) continue;
            const int y = std::countr_zero(w);
            if (rel_ok(t, b.pc[x], x, b.pc[y], y)) rel(sign, x, y, b.pc[x], b.pc[y]);
        }
    }
    void remove(int z) {
        const u64 dz = bishop_attacks_bb(Square(z), b.occ), oz = rook_attacks_bb(Square(z), b.occ);
        piece(z, -1, dz, oz);
        take(b, z);
        through(z, +1, dz, oz);
    }
    void add(int z, int p) {
        const u64 dz = bishop_attacks_bb(Square(z), b.occ), oz = rook_attacks_bb(Square(z), b.occ);
        through(z, -1, dz, oz);
        put(b, z, p);
        piece(z, +1, dz, oz);
    }
    void replace(int z, int p) {
        const u64 dz = bishop_attacks_bb(Square(z), b.occ), oz = rook_attacks_bb(Square(z), b.occ);
        piece(z, -1, dz, oz);
        take(b, z);
        put(b, z, p);
        piece(z, +1, dz, oz);
    }
    void move(Move m) {
        const int from = int(m.from()), to = int(m.to()), pm = b.pc[from], us = pm / 6;
        if (m.is_castling()) {
            const bool kingside = (to & 7) == 6;
            const int rank = from & 56, rf = rank + (kingside ? 7 : 0), rt = rank + (kingside ? 5 : 3);
            remove(from);
            remove(rf);
            add(to, us * 6 + 5);
            add(rt, us * 6 + 3);
            return;
        }
        const int np = m.is_promotion() ? us * 6 + int(m.promotion_type()) : pm;
        remove(from);
        if (m.is_en_passant()) {
            remove(to + (us == 0 ? -8 : 8));
            add(to, np);
        } else if (b.pc[to] != NOPC) replace(to, np);
        else add(to, np);
    }
};
}  // namespace

void init() {
    static const Tables t;
    g_t = &t;
}
int n_rows() { return g_t->n_rel + g_t->n_pp; }

void find(const Bd& before, Move m, int ppm, const bool need[2], RowLists& out) {
    Bd b = before;
    out.na[0] = out.na[1] = out.ns[0] = out.ns[1] = 0;
    out.overflow[0] = out.overflow[1] = false;
    Finder f{b, *g_t, ppm, {orient_of(before, 0), orient_of(before, 1)}, {need[0], need[1]}, out};
    f.move(m);
}

int enumerate(const Bd& b, int persp, int ppm, uint16_t* out) {
    const Tables& t = *g_t;
    const int ox = orient_of(b, persp);
    const u64 pawns = b.by[0] | b.by[6];
    int n = 0;
    for (u64 o = b.occ; o;) {
        const int s = pop(o), p = b.pc[s];
        u64 att;
        switch (p % 6) {
            case 0:  att = t.patt[p >= 6][s]; break;
            case 1:  att = t.e_n[s]; break;
            case 2:  att = bishop_attacks_bb(Square(s), b.occ); break;
            case 3:  att = rook_attacks_bb(Square(s), b.occ); break;
            case 4:  att = bishop_attacks_bb(Square(s), b.occ) | rook_attacks_bb(Square(s), b.occ); break;
            default: att = t.e_k[s]; break;
        }
        for (u64 v = att & b.occ; v;) {
            const int y = pop(v);
            if (!rel_ok(t, p, s, b.pc[y], y)) continue;
            const int row = rel_row(t, persp, ox, s, y, p, b.pc[y]);
            if (row >= 0 && n < MAX_ACTIVE) out[n++] = uint16_t(row);
        }
        if (p % 6 == 0) {
            const u64 partners = ppm ? (pawns & t.files3[s & 7]) : ((t.patt[p >= 6][s] & pawns) | (t.patt[1][s] & b.by[0]) | (t.patt[0][s] & b.by[6]));
            for (u64 c = partners & ~(bit(s) | t.below[s]); c;) {
                const int y = pop(c);
                if (n < MAX_ACTIVE) out[n++] = uint16_t(pair_row(t, persp, ox, s, y, p, b.pc[y]));
            }
        }
    }
    return n;
}

}  // namespace luminex::v5rel

// =====================================================================================================================
// The network built into the binary (last: on Windows it includes windows.h)
// =====================================================================================================================

// The network file built into the binary.
//
// CMake's LUMINEX_EMBED_NET puts a network file inside the executable, so a
// release plays with NNUE on its own instead of needing the .nnue beside it.
// MSVC and clang-cl link the file as an RCDATA resource (luminex_net.rc,
// written by CMake); GCC and Clang assemble it in with .incbin. Builds without
// the option carry no network and embedded_net() reports none.

#if defined(LUMINEX_EMBED_NET_RC)

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace luminex {

EmbeddedNet embedded_net() {
    // A resource stays mapped with the image: nothing to copy or free.
    HRSRC res = FindResourceA(nullptr, "LUMINEX_NET", MAKEINTRESOURCEA(10));  // 10 = RT_RCDATA
    HGLOBAL mem = res ? LoadResource(nullptr, res) : nullptr;
    const void* data = mem ? LockResource(mem) : nullptr;
    if (!data) return {nullptr, 0};
    return {static_cast<const unsigned char*>(data), std::size_t(SizeofResource(nullptr, res))};
}

} // namespace luminex

#elif defined(LUMINEX_EMBED_NET_FILE)

#include <cstdint>

// Assembler symbols carry the platform's C prefix (an underscore on macOS).
#define LUMINEX_STR_(x) #x
#define LUMINEX_STR(x) LUMINEX_STR_(x)
#define LUMINEX_SYM(name) LUMINEX_STR(__USER_LABEL_PREFIX__) #name

#if defined(__APPLE__)
#define LUMINEX_NET_SECTION ".const_data\n"
#elif defined(_WIN32)
#define LUMINEX_NET_SECTION ".section .rdata,\"dr\"\n"
#else
#define LUMINEX_NET_SECTION ".section .rodata\n"
#endif

__asm__(LUMINEX_NET_SECTION
        ".globl " LUMINEX_SYM(luminex_net_begin) "\n"
        ".globl " LUMINEX_SYM(luminex_net_end) "\n"
        ".p2align 6\n"
        LUMINEX_SYM(luminex_net_begin) ":\n"
        ".incbin \"" LUMINEX_EMBED_NET_FILE "\"\n"
        LUMINEX_SYM(luminex_net_end) ":\n"
        ".text\n");

extern "C" const unsigned char luminex_net_begin[];
extern "C" const unsigned char luminex_net_end[];

namespace luminex {

EmbeddedNet embedded_net() {
    const auto begin = reinterpret_cast<std::uintptr_t>(luminex_net_begin);
    const auto end = reinterpret_cast<std::uintptr_t>(luminex_net_end);
    return {luminex_net_begin, std::size_t(end - begin)};
}

} // namespace luminex

#else

namespace luminex {

EmbeddedNet embedded_net() { return {nullptr, 0}; }

} // namespace luminex

#endif
