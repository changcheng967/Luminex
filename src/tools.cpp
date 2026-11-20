// tools.cpp — the training data tools: one program per section, each its own executable, built from this file
// with its macro by CMake (the BUILD_ options):
//   LUMINEX_TOOL_FEATURIZE      luminex-featurize   (BUILD_FEATURIZER)
//   LUMINEX_TOOL_EVALTRACE      luminex-evaltrace   (BUILD_EVALTRACE)
//   LUMINEX_TOOL_LC0PACK        luminex-lc0pack     (BUILD_LC0PACK, with frame_merge)
//   LUMINEX_TOOL_FRAME_MERGE    frame_merge         (BUILD_LC0PACK)
//   LUMINEX_TOOL_VERIFY_FRAMES  verify_frames       (BUILD_VERIFY)
// The first three link the engine's board.cpp, evaluation.cpp and nnue.cpp; the last two stand alone.

#if defined(LUMINEX_TOOL_FEATURIZE)

// luminex-featurize — Game-Pack (.xz) -> HalfKAv2_hm feature .npy cache for NNUE training.
//
// Replays the Game-Pack produced by colab_pack_pgn.py and emits the SAME
// feat_{w,b,s,t}.npy layout that c500_featurize.py produces, so the output drops straight
// into luminex_nnue_train.py's .npy path (NNUE_NPY_DIR).
//
// CORRECTNESS GUARANTEES (the #1 constraint — no silent data corruption):
//   1. Reuses Luminex's OWN tested Position + generate<GEN_LEGAL> (perft + competitive play).
//   2. halfka_idx/orient/KingBuckets copied VERBATIM from nnue.cpp — which is itself
//      "mirrored EXACTLY from luminex_nnue_train.py" (confirmed bit-identical to the trainer).
//   3. Legal moves sorted by (from, to, promo) — the SAME key python-chess used when encoding
//      the move index. PieceType order N<B<R<Q matches python-chess's 2<3<4<5 (relative order
//      identical; a (from,to) is either all-4-promotions or a single non-promo, never mixed).
//   4. EVERY move index is validated: idx < numLegal, else FATAL. Any movegen divergence
//      (en-passant, castling-through-check, promo) aborts immediately rather than corrupting.
//   5. Evals output STM-relative (white_rel if stm==WHITE else -white_rel): matches the
//      engine's evaluate() which returns the net output WITHOUT stm negation, so negamax
//      (which needs stm-relative leaves) stays correct — same convention as the 280M data.
//
// USAGE (cloud):
//   xz -dc gamepack.xz | ./luminex-featurize --out-dir /tmp/feat --max-pos 500000000 --threads 8
//   (xz decompresses at ~19M pos/s; this replays+featurizes far faster than training consumes.)
//
// Output (into --out-dir): feat_w.npy feat_b.npy feat_s.npy feat_t.npy featurized_meta.txt
//   feat_w / feat_b : int16 [N, 32], HalfKAv2 indices, padded with NUM_INPUTS=24576
//   feat_s          : float32 [N], stm (1.0 = white to move, 0.0 = black)
//   feat_t          : float32 [N], eval in centipawns (STM-relative)
#include "luminex.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>   // std::sort
#include <sys/stat.h>  // mkdir
#include <mutex>
#include <deque>
#include <map>
#include <condition_variable>
#include <malloc.h>   // mallopt (job-buffer reuse)
#include <signal.h>    // SIGSEGV handler
#include <execinfo.h>  // backtrace
#include <sys/mman.h>  // mmap (lazy file load -> fits in 16 GB RAM)
#include <fcntl.h>     // open
#include <unistd.h>    // fstat, close
#ifdef _WIN32
  #include <direct.h>
  #define mkdir(path,mode) _mkdir(path)
#endif

// On Linux, pwrite gives lock-free concurrent writes to one fd at distinct offsets.
#if defined(__linux__) || defined(__unix__)
  #include <unistd.h>
  #include <fcntl.h>
  #define HAVE_PWRITE 1
#else
  #define HAVE_PWRITE 0
#endif

namespace luminex { }   // board/movegen/magic/nnue reference no search-only globals here.
using namespace luminex;

// ============================================================================
// HalfKAv2_hm feature indexer — copied VERBATIM from nnue.cpp (== trainer).
// ============================================================================
// --compact (fen-eval mode): LCR1 records of 32 bytes instead of the 140-byte
// feature records. The GPU-resident trainer (resident_train.py) keeps every
// position in VRAM in this form and computes the HalfKAv2_hm indices itself.
struct CompactRec {
    uint64_t occ;       // occupied squares, bit = square (A1 = 0)
    uint8_t  pcs[16];   // Piece (0..11) of the k-th occupied square, ascending; k even = low nibble
    int16_t  target;    // side-to-move-relative centipawns, clamped to +-32767
    uint8_t  flags;     // bit 0: white to move; bits 1-2: 0 quiet, 1 capture-best, 2 in-check
    uint8_t  npieces;   // popcount(occ)
    uint32_t fenhash;   // crc32 of the FEN text: a deterministic train/validation split
};
static_assert(sizeof(CompactRec) == 32, "LCR1 records are 32 bytes");
static bool g_compact = false;
static size_t g_rec_bytes = 140;

constexpr int NUM_SQ = 64;
constexpr int NUM_PLANES = NUM_SQ * 12;            // 768
constexpr int NUM_INPUTS = NUM_PLANES * 32;        // 24576 per half
constexpr int MAX_PIECES = 32;
static const int KingBuckets[NUM_SQ] = {
    -1,-1,-1,-1,31,30,29,28, -1,-1,-1,-1,27,26,25,24,
    -1,-1,-1,-1,23,22,21,20, -1,-1,-1,-1,19,18,17,16,
    -1,-1,-1,-1,15,14,13,12, -1,-1,-1,-1,11,10, 9, 8,
    -1,-1,-1,-1, 7, 6, 5, 4, -1,-1,-1,-1, 3, 2, 1, 0
};
static inline int orient(bool white_pov, int sq, int ksq) {
    int kfile = ksq & 7;
    return ((kfile < 4) ? 7 : 0) ^ (white_pov ? 0 : 56) ^ sq;
}
// p_idx: pt*2 — Luminex PieceType is PAWN=0..KING=5 (0-indexed), so pt*2 == trainer's
// (piece_type-1)*2 (python-chess 1-indexed). Own pieces take the lower of each pair.
static inline int halfka_idx(bool white_pov, int king_sq, int sq, Piece p) {
    int pt = static_cast<int>(piece_type_of(p));               // 0..5 (PAWN..KING)
    bool is_white_piece = (color_of_piece(p) == WHITE);
    int p_idx = pt * 2 + (is_white_piece != white_pov ? 1 : 0);
    int o_ksq = orient(white_pov, king_sq, king_sq);
    return orient(white_pov, sq, king_sq) + p_idx * NUM_SQ + KingBuckets[o_ksq] * NUM_PLANES;
}

// ============================================================================
// Move sort key — MUST match python-chess: (from, to, promotion|0).
// python-chess promotion ints: N=2 B=3 R=4 Q=5, None=0. Luminex PieceType: N=1 B=2 R=3 Q=4.
// Relative order is identical, and a (from,to) is never a mix of promo/non-promo, so using
// the raw PieceType value (N=1..Q=4, non-promo=0) yields the same ordering.
// ============================================================================
static inline uint32_t move_sort_key(Move m) {
    uint32_t promo = m.is_promotion() ? (uint32_t)m.promotion_type() : 0;  // 0,1,2,3,4
    return ((uint32_t)m.from() << 12) | ((uint32_t)m.to() << 4) | promo;
}

// ============================================================================
// Read entire stdin (or --input file) into a buffer.
// ============================================================================
static std::vector<uint8_t> read_all(std::FILE* f) {
    std::vector<uint8_t> buf;
    buf.reserve(1 << 28);  // 256 MB initial
    uint8_t chunk[1 << 20];
    size_t n;
    while ((n = std::fread(chunk, 1, sizeof(chunk), f)) > 0) buf.insert(buf.end(), chunk, chunk + n);
    return buf;
}

// ============================================================================
// .npy writer (format v1.0). Writes header with the given dtype/shape, returns the
// header byte length so callers can compute data offsets.
// ============================================================================
static size_t write_npy_header(std::FILE* f, const char* dtype, int ndims, const long* shape) {
    char dict[128];
    int dlen;
    if (ndims == 1)
        dlen = std::snprintf(dict, sizeof(dict), "{'descr': '%s', 'fortran_order': False, 'shape': (%ld,),}", dtype, shape[0]);
    else
        dlen = std::snprintf(dict, sizeof(dict), "{'descr': '%s', 'fortran_order': False, 'shape': (%ld, %ld),}", dtype, shape[0], shape[1]);
    // Pad header (incl. trailing \n) so (10 + header_len) % 64 == 0.
    size_t hlen = dlen + 1;          // +1 for '\n'
    size_t total = 10 + hlen;
    size_t pad = (64 - (total % 64)) % 64;
    hlen += pad;
    std::fwrite("\x93NUMPY", 1, 6, f);
    uint8_t ver[2] = {1, 0};
    std::fwrite(ver, 1, 2, f);
    uint8_t hlen_le[2] = {(uint8_t)(hlen & 0xFF), (uint8_t)((hlen >> 8) & 0xFF)};
    std::fwrite(hlen_le, 1, 2, f);
    std::fwrite(dict, 1, dlen, f);
    for (size_t i = 0; i < pad; ++i) std::fputc(' ', f);
    std::fputc('\n', f);
    return 10 + hlen;
}

// SIGSEGV/SIGABRT backtrace handler — prints the crash location so we can find
// the featurizer bug without gdb (not installed on the build/C500 hosts).
static void crash_handler(int sig) {
    void *bt[32];
    int n = backtrace(bt, 32);
    std::fprintf(stderr, "\n=== CRASH (signal %d) backtrace ===\n", sig);
    backtrace_symbols_fd(bt, n, 2);   // 2 = stderr fd
    std::fprintf(stderr, "=== end backtrace ===\n");
    _exit(1);
}

// CRC-32 (IEEE, reflected) over the FEN bytes — matches Python's zlib.crc32.
// Used by --skip-hash-frac to hold out a validation subset at the stream level:
// positions whose FEN hash falls below the threshold are dropped from TRAIN so
// the same positions can serve as a leak-free validation set (visit-multiplicity
// repeats of a held-out position are dropped together — same FEN, same hash).
static uint32_t crc32_ieee(const char* p, size_t n) {
    static uint32_t table[256]; static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        init = true;
    }
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) c = table[(c ^ (unsigned char)p[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

// --fen-eval mode: read "fen<tab>cp_white_rel" lines from stdin (one position per
// line, e.g. from Gigafish), featurize each HalfKAv2 independently, emit the SAME
// 140-byte records as --stream so the trainer consumes them identically. No game
// replay (positions are independent). cp is white-relative; target flipped to stm-rel.
// Positions are independent, so this parallelizes over worker threads (the old
// single-thread path parsed a full FEN per position and ran ~6x slower than the
// game-replay mode); output order is preserved via sequence-numbered jobs.
static bool fen_eval_one(Position& pos, const char* ls, const char* le, uint32_t skip_thresh,
                         std::vector<char>& out) {
    const char* tab = nullptr;
    for (const char* p = le - 1; p >= ls && !tab; --p)
        if (*p == '\t') tab = p;
    if (!tab) return false;
    // Column counting decides the format: a 3-column line is fen/cp/quiet
    // (0 quiet, 1 capture-best, 2 in-check); a 2-column line is fen/cp and
    // the flag derives from the FEN below. Reading "one column past the last
    // tab if non-empty" misparses 2-column lines — their last tab IS the cp
    // column, and that bug fed centipawn values in as quiet flags.
    int ntabs = 0;
    for (const char* p = ls; p < le && ntabs < 2; ++p)
        if (*p == '\t') ++ntabs;
    const char* cp_col = (ntabs >= 2) ? nullptr : tab;   // 2-col: cp at last tab
    // for 3-col: find the FIRST tab (cp), last tab is the quiet flag
    const char* tab1 = nullptr;
    if (ntabs >= 2) {
        for (const char* p = ls; p < le && !tab1; ++p)
            if (*p == '\t') tab1 = p;
        cp_col = tab1;
    }
    int quiet = (ntabs >= 2) ? atoi(tab + 1) : 0;
    if (skip_thresh && crc32_ieee(ls, (size_t)(cp_col - ls)) < skip_thresh) return false;
    char fbuf[128];
    size_t flen = (size_t)(cp_col - ls);
    if (flen >= sizeof fbuf) return false;
    std::memcpy(fbuf, ls, flen); fbuf[flen] = '\0';
    float cp = (float)atoi(cp_col + 1);   // cp is an int (centipawns); atoi never throws
    pos.set(fbuf);
    // quiet flag: column value if present, else derived from the FEN —
    // in-check positions carry pure label noise for a static eval.
    if (quiet == 0 && pos.checkers()) quiet = 2;
    if (g_compact) {
        Bitboard occ = pos.pieces();
        if (popcount(occ) > MAX_PIECES) return false;
        CompactRec r;
        std::memset(&r, 0, sizeof r);
        r.occ = occ;
        int k = 0;
        for (Bitboard bb = occ; bb; ++k) {
            Square sq = pop_lsb(bb);
            r.pcs[k >> 1] |= (uint8_t)((int)pos.piece_on(sq) << (4 * (k & 1)));
        }
        bool white = (pos.side_to_move() == WHITE);
        float t = white ? cp : -cp;   // white-relative column -> stm-relative, as below
        r.target = (int16_t)std::max(-32767.0f, std::min(32767.0f, t));
        r.flags = (uint8_t)((white ? 1 : 0) | ((quiet & 3) << 1));
        r.npieces = (uint8_t)k;
        r.fenhash = crc32_ieee(ls, (size_t)(cp_col - ls));
        const char* rp = reinterpret_cast<const char*>(&r);
        out.insert(out.end(), rp, rp + sizeof r);
        return true;
    }
    int wk = (int)pos.king_sq(WHITE), bk = (int)pos.king_sq(BLACK);
    int16_t wfeat[MAX_PIECES], bfeat[MAX_PIECES]; int nw = 0, nb = 0;
#ifdef FAST_HALFKA
    int xor_w  = ((wk & 7) < 4 ? 7 : 0);
    int xor_b  = ((bk & 7) < 4 ? 7 : 0) ^ 56;
    int bucket_w = KingBuckets[xor_w ^ wk] * NUM_PLANES;
    int bucket_b = KingBuckets[xor_b ^ bk] * NUM_PLANES;
    Bitboard occ = pos.pieces();
    while (occ && nw < MAX_PIECES) {
        int sq = (int)pop_lsb(occ);
        Piece pc = pos.piece_on(Square(sq));
        int pt = (int)piece_type_of(pc);
        bool is_w = (color_of_piece(pc) == WHITE);
        wfeat[nw++] = (int16_t)((xor_w ^ sq) + (pt * 2 + (is_w ? 0 : 1)) * NUM_SQ + bucket_w);
        bfeat[nb++] = (int16_t)((xor_b ^ sq) + (pt * 2 + (is_w ? 1 : 0)) * NUM_SQ + bucket_b);
    }
#else
    for (int sq = 0; sq < NUM_SQ && nw < MAX_PIECES; ++sq) {
        Piece pc = pos.piece_on(Square(sq));
        if (pc == NO_PIECE) continue;
        wfeat[nw++] = (int16_t)halfka_idx(true,  wk, sq, pc);
        bfeat[nb++] = (int16_t)halfka_idx(false, bk, sq, pc);
    }
#endif
    for (int i = nw; i < MAX_PIECES; ++i) wfeat[i] = (int16_t)NUM_INPUTS;
    for (int i = nb; i < MAX_PIECES; ++i) bfeat[i] = (int16_t)NUM_INPUTS;
    bool stm_white = (pos.side_to_move() == WHITE);
    // TSV contract: the cp column is WHITE-relative and the trainer wants
    // stm-relative targets — negate for black to move. (A dump tool once wrote
    // the binpack's raw stm-relative score; the double negation flipped half
    // the targets and collapsed a run. The trainer's stm-split correlation
    // gate now catches this class before training starts.)
    float target = stm_white ? cp : -cp;
    float stm = stm_white ? 1.0f : 0.0f;
    // quiet flag rides in the result channel (unused, always 0, for fen-eval
    // data): 0 quiet, 1 capture-best, 2 in-check. Round-3 loss weights by it.
    float result = static_cast<float>(quiet);
    char buf[140];
    std::memcpy(buf, wfeat, 64); std::memcpy(buf + 64, bfeat, 64);
    std::memcpy(buf + 128, &stm, 4); std::memcpy(buf + 132, &target, 4);
    std::memcpy(buf + 136, &result, 4);
    out.insert(out.end(), buf, buf + 140);
    return true;
}

static void fen_eval_mode(long max_pos, double skip_hash_frac, int nthreads) {
    // Job buffers are multi-MB; keep them on the heap instead of per-allocation
    // mmap/munmap (measured 40s of system time churning page faults otherwise).
    mallopt(M_MMAP_THRESHOLD, 64 * 1024 * 1024);
    crc32_ieee("", 0);   // build the CRC table here: the worker threads only read it
    const uint32_t skip_thresh = (uint32_t)(skip_hash_frac * 4294967296.0);
    if (skip_hash_frac > 0.0)
        std::fprintf(stderr, "fen-eval: skipping FEN hashes below %u (%.4f%% held out)\n",
                     skip_thresh, skip_hash_frac * 100.0);
    long n = 0;
    const auto t0 = std::chrono::steady_clock::now();
    long next_prog = 5000000;
    auto progress = [&](bool force) {
        if (n >= next_prog || force) {
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "fen-eval: %ldM positions, %.0fK pos/s\n",
                         n / 1000000, n / 1000.0 / (sec > 1e-9 ? sec : 1e-9));
            next_prog = n + 5000000;
        }
    };
    if (nthreads <= 1) {
        Position pos1;
        std::string line;
        while (std::getline(std::cin, line)) {
            if (max_pos && n >= max_pos) break;
            std::vector<char> out;
            if (fen_eval_one(pos1, line.data(), line.data() + line.size(), skip_thresh, out)) {
                std::fwrite(out.data(), 1, out.size(), stdout);
                ++n;
            }
            progress(false);
        }
    } else {
        // Unordered parallel path: the trainer reshuffles every epoch, so output
        // order carries zero information — workers write completed blocks
        // directly under a mutex. (An earlier ordered job-queue version spent
        // most of its wall time serializing on sequence numbers.)
        struct Block { std::string text; };
        std::deque<Block*> ready;
        std::mutex m, wm;
        std::condition_variable cv_work;
        bool eof = false;
        auto worker = [&]() {
            Position wpos;
            std::vector<char> out;
            for (;;) {
                Block* b;
                {
                    std::unique_lock<std::mutex> lk(m);
                    cv_work.wait(lk, [&] { return !ready.empty() || eof; });
                    if (ready.empty()) return;
                    b = ready.front(); ready.pop_front();
                }
                out.clear();
                size_t st = 0;
                while (st < b->text.size()) {
                    size_t e = b->text.find('\n', st);
                    if (e == std::string::npos) e = b->text.size();
                    if (e > st) fen_eval_one(wpos, b->text.data() + st, b->text.data() + e, skip_thresh, out);
                    st = e + 1;
                }
                {
                    std::lock_guard<std::mutex> lw(wm);
                    std::fwrite(out.data(), 1, out.size(), stdout);
                    n += (long)(out.size() / g_rec_bytes);
                    progress(false);
                }
                delete b;
            }
        };
        std::vector<std::thread> pool;
        for (int i = 0; i < nthreads; ++i) pool.emplace_back(worker);
        std::string carry;
        char rbuf[1 << 22];
        while (true) {
            size_t got = std::fread(rbuf, 1, sizeof rbuf, stdin);
            if (got == 0) break;
            size_t last_nl = 0, nl = std::string::npos;
            // find last newline in the read chunk
            for (size_t i = got; i > 0; --i) if (rbuf[i-1] == '\n') { nl = i - 1; break; }
            std::string* block;
            if (carry.empty()) {
                if (nl == std::string::npos) { carry.append(rbuf, got); continue; }
                block = new std::string(rbuf, nl + 1);
                carry.append(rbuf + nl + 1, got - nl - 1);
            } else {
                block = new std::string(std::move(carry));
                carry.clear();
                if (nl == std::string::npos) { block->append(rbuf, got); }
                else { block->append(rbuf, nl + 1); carry.append(rbuf + nl + 1, got - nl - 1); }
            }
            {
                std::lock_guard<std::mutex> lk(m);
                ready.push_back(reinterpret_cast<Block*>(block));
                cv_work.notify_one();
            }
            // bounded queue: wait if workers are far behind
            while (true) {
                size_t rq;
                { std::lock_guard<std::mutex> lk(m); rq = ready.size(); }
                if (rq < (size_t)(4 * nthreads)) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
        {
            std::lock_guard<std::mutex> lk(m);
            eof = true;
            cv_work.notify_all();
        }
        for (auto& t : pool) t.join();
    }
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "fen-eval: featurized %ld positions in %.0fs (%.0fK pos/s)\n",
                 n, sec, n / 1000.0 / (sec > 1e-9 ? sec : 1e-9));
}

int main(int argc, char** argv) {
    signal(SIGSEGV, crash_handler);
    signal(SIGABRT, crash_handler);
    const char* out_dir = "/tmp/feat";
    const char* in_path = nullptr;          // nullptr => stdin
    const char* out_path = nullptr;         // --output (fen-eval): records to this file, not stdout
    long max_pos = 0;                        // 0 = no limit
    bool stream_mode = false;                // --stream: emit 140-byte records to stdout (no .npy)
    bool fen_eval = false;                   // --fen-eval: read "fen<tab>cp" lines, emit 140-byte records
    double skip_hash_frac = 0.0;             // --skip-hash-frac F: drop lines with crc32(fen)/2^32 < F (val holdout)
    int nthreads = (int)std::thread::hardware_concurrency();
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* def){ return (i + 1 < argc) ? argv[++i] : def; };
        if      (a == "--out-dir") out_dir = next(out_dir);
        else if (a == "--input")   in_path = next("");
        else if (a == "--max-pos") max_pos = std::atol(next("0"));
        else if (a == "--threads") nthreads = std::atoi(next("0"));
        else if (a == "--stream")  stream_mode = true;
        else if (a == "--fen-eval") fen_eval = true;
        else if (a == "--skip-hash-frac") skip_hash_frac = std::atof(next("0"));
        else if (a == "--compact") { g_compact = true; g_rec_bytes = sizeof(CompactRec); }
        else if (a == "--output")  out_path = next("");
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }

    init_magic_bitboards();
    if (fen_eval) {
        // --input / --output read and write files directly instead of stdin/stdout.
        if (in_path && *in_path && !std::freopen(in_path, "rb", stdin)) { std::perror(in_path); return 1; }
        if (out_path && *out_path && !std::freopen(out_path, "wb", stdout)) { std::perror(out_path); return 1; }
        fen_eval_mode(max_pos, skip_hash_frac, nthreads);
        return 0;
    }

    if (!stream_mode) mkdir(out_dir, 0755);   // create output dir if missing (fopen "wb" won't)

    // Load the frame. For a FILE we mmap it (lazy page-in -> low RAM, fits in 16 GB,
    // lets us test on a small box); for stdin (pipe, e.g. the C500's `xz -dc | featurize`)
    // we must read_all into RAM (the pipe isn't seekable/mmap-able).
    std::vector<uint8_t> frame_vec;
    uint8_t* frame = nullptr; size_t frame_size = 0;
    if (in_path) {
        int fd = open(in_path, O_RDONLY);
        if (fd < 0) { std::perror("open input"); return 1; }
        struct stat _st;
        if (fstat(fd, &_st)) { std::perror("fstat"); return 1; }
        if (S_ISREG(_st.st_mode)) {           // regular file -> mmap (lazy, low RAM)
            frame_size = _st.st_size;
            void* m = mmap(nullptr, frame_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (m == MAP_FAILED) { std::perror("mmap"); return 1; }
            frame = (uint8_t*)m; close(fd);
            std::fprintf(stderr, "featurize: mmap'd %s (%zu bytes)\n", in_path, frame_size);
        } else {                               // pipe / /dev/stdin -> read_all into RAM
            std::FILE* ff = fdopen(fd, "rb");
            frame_vec = read_all(ff);
            frame = frame_vec.data(); frame_size = frame_vec.size();
            std::fclose(ff);
            std::fprintf(stderr, "featurize: read %s (%zu bytes)\n", in_path, frame_size);
        }
    } else {
        std::fprintf(stderr, "featurize: reading frame from stdin...\n");
        frame_vec = read_all(stdin);
        frame = frame_vec.data();
        frame_size = frame_vec.size();
        std::fprintf(stderr, "  frame: %zu bytes\n", frame_size);
    }

    // Parse the 3 length-prefixed blocks IN-PLACE — const pointers into `frame`, NO copy.
    // A 15B+ pack decompresses to ~30+ GB; the old code duplicated it into 3 block vectors,
    // doubling peak RAM (~60 GB) and OOMing the C500. One backing buffer, three regions.
    size_t p = 0;
    auto rd_u64 = [&]() -> uint64_t {
        uint64_t v = 0; for (int i = 0; i < 8; ++i) v |= (uint64_t)frame[p + i] << (8 * i); p += 8; return v;
    };
    uint64_t hdr_n = rd_u64(); const uint8_t* hdr = frame + p; p += hdr_n;
    uint64_t mv_n  = rd_u64(); const uint8_t* mv  = frame + p; p += mv_n;
    uint64_t ev_n  = rd_u64(); const uint8_t* ev  = frame + p; p += ev_n;

    // Decode header. v2: [u32 GP2_MAGIC][u32 nfens][fens][10-byte entries
    // (+int8 white-rel result)]. v1: [u32 nfens][fens][9-byte entries] — the
    // magic is larger than any fen count, so the first u32 discriminates.
    size_t hp = 0;
    auto rd_u32 = [&]() { uint32_t v=0; for(int i=0;i<4;++i) v |= (uint32_t)hdr[hp+i]<<i*8; hp+=4; return v; };
    auto rd_u16 = [&]() { uint16_t v=0; for(int i=0;i<2;++i) v |= (uint16_t)hdr[hp+i]<<i*8; hp+=2; return v; };
    auto rd_s16 = [&]() { int16_t v=(int16_t)rd_u16(); return v; };
    constexpr uint32_t GP2_MAGIC = 0x47503200;
    bool has_result = false;
    if (hdr_n >= 4) {
        uint32_t first;
        std::memcpy(&first, hdr, 4);
        if (first == GP2_MAGIC) { hp = 4; has_result = true; }
    }
    const int ent_w = has_result ? 10 : 9;
    uint32_t nfens = rd_u32();
    std::vector<std::string> fens(nfens);
    for (uint32_t i = 0; i < nfens; ++i) {
        uint16_t fl = rd_u16();
        fens[i].assign((const char*)hdr + hp, fl);
        hp += fl;
    }
    struct Game { uint32_t fen_idx; uint16_t n_pos; uint8_t stm; int16_t start_eval; int8_t result; long mv_off; long ev_off; long out_off; };
    std::vector<Game> games;
    long mv_acc = 0;
    while (hp + ent_w <= hdr_n) {
        Game g;
        g.n_pos = (uint16_t)hdr[hp] | ((uint16_t)hdr[hp+1] << 8); hp += 2;   // u16 n_pos (no 255 cap)
        g.stm   = hdr[hp++];
        g.fen_idx = rd_u32();
        g.start_eval = rd_s16();
        g.result = has_result ? (int8_t)hdr[hp++] : 0;
        g.mv_off = mv_acc;
#ifdef RAW_MOVES
        mv_acc += (long)g.n_pos * 2;   // raw mode stores 2 bytes/pos (Move::raw)
#else
        mv_acc += g.n_pos;             // index mode: 1 byte/pos
#endif
        g.ev_off = 0;   // computed by the scan below (eval entries are VARIABLE-length)
        games.push_back(g);
    }

    // Eval block has VARIABLE-length entries (1 byte normal delta, 3 bytes for the 0x80 escape).
    // So per-game offsets cannot be computed from n_pos alone — precompute them with one
    // sequential scan, so threads can later decode each game independently from its start offset.
    {
        long ev_ptr = 0;
        for (auto& g : games) {
            g.ev_off = ev_ptr;
            long ndelta = (g.n_pos > 0) ? g.n_pos - 1 : 0;   // plies 1..n-1
            for (long i = 0; i < ndelta; ++i) {
                if (ev_ptr < (long)ev_n && ev[ev_ptr] == 0x80) ev_ptr += 3;  // escape
                else ev_ptr += 1;
            }
        }
    }

    // Compute output offsets (prefix sum of n_pos), honoring --max-pos.
    long total_pos = 0;
    for (auto& g : games) {
        if (max_pos && total_pos >= max_pos) { g.out_off = -1; continue; }
        long take = g.n_pos;
        if (max_pos && total_pos + take > max_pos) take = max_pos - total_pos;
        g.out_off = total_pos;
        total_pos += take;
        // NOTE: we still consume the full g.n_pos from mv/ev blocks below even if we cap output;
        // simpler to just stop emitting once max_pos reached (handled per-ply).
    }
    std::fprintf(stderr, "  games=%zu  total_positions=%ld  (max_pos cap=%ld)\n",
                 games.size(), total_pos, max_pos);
    if (total_pos == 0) { std::fprintf(stderr, "no positions\n"); return 1; }

    // Open 4 output .npy files + write headers (skipped in --stream mode: emit to stdout instead).
    long shape2[2] = { total_pos, MAX_PIECES };
    long shape1[1] = { total_pos };
    std::FILE* fw=nullptr,*fb=nullptr,*fs=nullptr,*ft=nullptr;
    size_t hw=0,hb=0,hs=0,ht=0;
    int fd_w=-1,fd_b=-1,fd_s=-1,fd_t=-1;
    if (!stream_mode) {
        auto open_npy = [&](const char* name, const char* dtype, int ndims, const long* shape) {
            std::string path = std::string(out_dir) + "/" + name;
            std::FILE* f = std::fopen(path.c_str(), "wb");
            if (!f) { std::perror(path.c_str()); std::exit(1); }
            size_t hlen = write_npy_header(f, dtype, ndims, shape);
            return std::make_pair(f, hlen);
        };
        auto a = open_npy("feat_w.npy", "<i2", 2, shape2); fw=a.first; hw=a.second;
        auto b = open_npy("feat_b.npy", "<i2", 2, shape2); fb=b.first; hb=b.second;
        auto c = open_npy("feat_s.npy", "<f4", 1, shape1); fs=c.first; hs=c.second;
        auto d = open_npy("feat_t.npy", "<f4", 1, shape1); ft=d.first; ht=d.second;
#if HAVE_PWRITE
        fd_w = fileno(fw); fd_b = fileno(fb); fd_s = fileno(fs); fd_t = fileno(ft);
#endif
    }
    if (!HAVE_PWRITE) nthreads = 1;
    std::fprintf(stderr, "  threads=%d%s%s\n", nthreads, HAVE_PWRITE ? "" : " (no pwrite -> single thread)",
                 stream_mode ? " [STREAM->stdout]" : "");

    // Per-thread output buffers (write whole rows via one pwrite/fwrite each).
    constexpr size_t WROW = MAX_PIECES * sizeof(int16_t);  // 64
    std::mutex out_mtx;   // guards stdout in --stream mode
    // Optional FEN+target dump (for data-integrity diagnosis): NNUE_DUMP_FENS=file.
    // NNUE_DUMP_RATE=K keeps a hash-based 1-in-K sample (position-stable: the hash keys on
    // fen_idx/game/ply, so the sample is identical under any thread split). Output line:
    //   fen \t stm_white \t target   (target is stm-RELATIVE; y_white = stm ? target : -target)
    // NNUE_DUMP_NOFEAT=1 = dump-only fast path: replay + eval decode, skip feature compute
    // and record emission entirely (a 14.5B-position gamepack -> 25M FENs in minutes).
    FILE* dump_fens = nullptr;
    long dump_rate = 1;
    if (const char* dp = std::getenv("NNUE_DUMP_FENS")) dump_fens = std::fopen(dp, "w");
    if (const char* dr = std::getenv("NNUE_DUMP_RATE")) {
        long r = std::atol(dr);
        if (r > 0) dump_rate = r;
    }
    bool dump_only = std::getenv("NNUE_DUMP_NOFEAT") != nullptr && dump_fens;
    auto write_row = [&](long row, const int16_t* w, const int16_t* b, float s, float t) {
        if (stream_mode) return;   // stream emit is batched in the worker (below)
        long ow = hw + row * WROW, ob = hb + row * WROW;
        long os = hs + row * sizeof(float), ot = ht + row * sizeof(float);
#if HAVE_PWRITE
        pwrite(fd_w, w, WROW, ow);
        pwrite(fd_b, b, WROW, ob);
        pwrite(fd_s, &s, 4, os);
        pwrite(fd_t, &t, 4, ot);
#else
        std::fseek(fw, (long)ow, SEEK_SET); std::fwrite(w, 1, WROW, fw);
        std::fseek(fb, (long)ob, SEEK_SET); std::fwrite(b, 1, WROW, fb);
        std::fseek(fs, (long)os, SEEK_SET); std::fwrite(&s, 1, 4, fs);
        std::fseek(ft, (long)ot, SEEK_SET); std::fwrite(&t, 1, 4, ft);
#endif
    };

    std::atomic<long> emitted{0};
    std::atomic<long> errors{0};
    auto t0 = std::chrono::steady_clock::now();
#ifdef PROFILE
    // Per-thread stage timers: setup(pos.set FEN) + move + feat + write.
    std::vector<std::array<double,4>> prof(nthreads);
    for (auto& p : prof) p = {0,0,0,0};
#endif

    auto worker = [&](int tid) {
        Position pos;
        ExtMove list[MAX_MOVES];
        std::vector<char> tbuf;   // per-thread stream buffer (140 bytes/record)
        tbuf.reserve(140 * 2048); // no reallocation; flush every 2048 records
        long local_n = 0;         // positions buffered since last global update (batch the atomic)
        long last_emitted_m = -1; // for the periodic progress print
#ifdef PROFILE
        double tm=0, tf=0, tw=0, tset=0;
        std::chrono::steady_clock::time_point _ts;
        #define _TICK() _ts = std::chrono::steady_clock::now()
        #define _TOCK(x) x += std::chrono::duration<double>(std::chrono::steady_clock::now()-_ts).count()
#else
        #define _TICK()
        #define _TOCK(x)
#endif
        for (size_t gi = tid; gi < games.size(); gi += nthreads) {
            Game& g = games[gi];
            if (g.out_off < 0) continue;
            if (tid == 0 && (gi % 50000 == 0))
                std::fprintf(stderr, "  [prog] game %zu/%zu (emitted %ld)\n", gi, games.size(), emitted.load());
#ifdef PROFILE
                _TICK();
#endif
            pos.set(fens[g.fen_idx]);   // FENs come from python-chess via the encoder — valid

            long mv_p = g.mv_off;
            long ev_p = g.ev_off;
            int16_t prev_eval = g.start_eval;   // white-relative
            bool first = true;
#ifdef PROFILE
                _TOCK(tset);
#endif

            for (int ply = 0; ply < g.n_pos; ++ply) {
#ifdef PROFILE
                _TICK();
#endif
#ifdef RAW_MOVES
                // Direct-move: read the raw 16-bit Move (from/to/flags) written by encode and
                // replay it. ZERO generate<GEN_LEGAL>, ZERO sort -> ~5-10x faster featurize.
                // The move is exactly legal (encode parsed it via generate<GEN_LEGAL>), so
                // do_move cannot diverge from the index path -> bit-identical features.
                uint16_t raw = (uint16_t)mv[mv_p] | ((uint16_t)mv[mv_p + 1] << 8); mv_p += 2;
                Move mvmove(raw);
#ifdef REPLAY_MOVE
                if (!pos.do_move_replay(mvmove)) { std::atomic_fetch_add(&errors, 1L); break; }
#else
                if (!pos.do_move(mvmove)) { std::atomic_fetch_add(&errors, 1L); break; }
#endif
#else
                // Decode the move index -> legal move.
                uint8_t idx = mv[mv_p++];
                ExtMove* end = generate<GEN_LEGAL>(pos, list);
                int nlegal = (int)(end - list);
                // Sort by (from,to,promo) to match python-chess's indexing order.
                std::sort(list, end, [](const ExtMove& a, const ExtMove& b){
                    return move_sort_key(a.move) < move_sort_key(b.move);
                });
                if (idx >= nlegal) {
                    // Movegen divergence vs python-chess on this game. SKIP it (break the
                    // ply loop -> next game) instead of aborting the whole run. A handful of
                    // divergences costs a few positions, not 15B. Log the first 20 so we can
                    // see the offending FENs and gauge how often it happens.
                    std::atomic_fetch_add(&errors, 1L);
                    if (errors.load() <= 20)
                        std::fprintf(stderr, "SKIP: game %zu ply %d idx %u >= nlegal %d (fen %s)\n",
                                     gi, ply, idx, nlegal, fens[g.fen_idx].c_str());
                    break;
                }
                Move mv = list[idx].move;
                if (!pos.do_move(mv)) { std::atomic_fetch_add(&errors, 1L); break; }
#endif
#ifdef PROFILE
                _TOCK(tm); _TICK();
#endif

                // Decode the white-relative eval for THIS resulting position.
                int16_t eq;
                if (first) { eq = g.start_eval; first = false; }
                else {
                    int8_t d = (int8_t)ev[ev_p++];      // signed delta byte
                    if (d == (int8_t)0x80) {                 // escape: 0x80 + s16 absolute
                        int16_t absval = (int16_t)((uint16_t)ev[ev_p] | ((uint16_t)ev[ev_p+1] << 8));
                        ev_p += 2; eq = absval;
                    } else { eq = (int16_t)(prev_eval + d); }
                }
                prev_eval = eq;

                long row = g.out_off + ply;
                if (row >= total_pos) break;                 // --max-pos cap

                // FEN+target dump: sampled replay positions for the HCE distillation fit.
                if (dump_fens) {
                    uint32_t h = (uint32_t)(g.fen_idx * 2654435761u) ^ (uint32_t)(gi * 40503u)
                               ^ (uint32_t)(ply * 2246822519u);
                    h ^= h >> 13; h *= 2654435761u; h ^= h >> 16;
                    if (h % (uint32_t)dump_rate == 0) {
                        bool dw = (pos.side_to_move() == WHITE);
                        float dt = dw ? (float)eq : (float)(-eq);
                        std::lock_guard<std::mutex> lk(out_mtx);
                        std::fprintf(dump_fens, "%s\t%d\t%.1f\n", pos.fen().c_str(), (int)dw, (double)dt);
                    }
                    if (dump_only) continue;   // skip features + record emission
                }

                // Compute HalfKAv2 features of the resulting position.
                int wk = (int)pos.king_sq(WHITE), bk = (int)pos.king_sq(BLACK);
                int16_t wfeat[MAX_PIECES], bfeat[MAX_PIECES];
                int nw = 0, nb = 0;
#ifdef FAST_HALFKA
                // Precompute the 4 king-dependent constants (constant for all pieces in
                // THIS position), then iterate ONLY occupied squares via pop_lsb. Per-piece
                // work drops to ~1 xor + 2 adds (vs a full halfka_idx call). pop_lsb yields
                // squares in increasing order -> bit-identical to the 0..63 scan.
                int xor_w  = ((wk & 7) < 4 ? 7 : 0);
                int xor_b  = ((bk & 7) < 4 ? 7 : 0) ^ 56;
                int bucket_w = KingBuckets[xor_w ^ wk] * NUM_PLANES;
                int bucket_b = KingBuckets[xor_b ^ bk] * NUM_PLANES;
                Bitboard occ = pos.pieces();
                while (occ && nw < MAX_PIECES) {
                    int sq = (int)pop_lsb(occ);
                    Piece pc = pos.piece_on(Square(sq));
                    int pt = (int)piece_type_of(pc);
                    bool is_w = (color_of_piece(pc) == WHITE);
                    wfeat[nw++] = (int16_t)((xor_w ^ sq) + (pt * 2 + (is_w ? 0 : 1)) * NUM_SQ + bucket_w);
                    bfeat[nb++] = (int16_t)((xor_b ^ sq) + (pt * 2 + (is_w ? 1 : 0)) * NUM_SQ + bucket_b);
                }
#else
                for (int sq = 0; sq < NUM_SQ && nw < MAX_PIECES; ++sq) {
                    Piece pc = pos.piece_on(Square(sq));
                    if (pc == NO_PIECE) continue;
                    wfeat[nw++] = (int16_t)halfka_idx(true,  wk, sq, pc);
                    bfeat[nb++] = (int16_t)halfka_idx(false, bk, sq, pc);
                }
#endif
                for (int i = nw; i < MAX_PIECES; ++i) wfeat[i] = (int16_t)NUM_INPUTS;
                for (int i = nb; i < MAX_PIECES; ++i) bfeat[i] = (int16_t)NUM_INPUTS;
#ifdef PROFILE
                _TOCK(tf); _TICK();
#endif

                // STM-relative eval target (engine returns net output without stm negation).
                bool stm_white = (pos.side_to_move() == WHITE);
                float target = stm_white ? (float)eq : (float)(-eq);
                float stm = stm_white ? 1.0f : 0.0f;
                if (stream_mode) {
                    // Pack a 140-byte record (w[32] b[32] stm target result) and batch into
                    // the thread buffer; flush under the stdout mutex. Order across threads
                    // doesn't matter — the trainer shuffles a chunk anyway. Backpressure:
                    // fwrite blocks when the stdout pipe is full, throttling featurize to
                    // the trainer's consume rate. result is WHITE-relative (v1 frames: 0).
                    float result = (float)g.result;
                    char buf[140];
                    std::memcpy(buf, wfeat, 64); std::memcpy(buf + 64, bfeat, 64);
                    std::memcpy(buf + 128, &stm, 4); std::memcpy(buf + 132, &target, 4);
                    std::memcpy(buf + 136, &result, 4);
                    tbuf.insert(tbuf.end(), buf, buf + 140);
                    local_n++;
                    if (tbuf.size() >= 140 * 2048) {
                        std::lock_guard<std::mutex> lk(out_mtx);
                        std::fwrite(tbuf.data(), 1, tbuf.size(), stdout);
                        tbuf.resize(0);          // keep capacity (no realloc)
                        emitted.fetch_add(local_n); local_n = 0;   // batched atomic
                    }
                } else {
                    write_row(row, wfeat, bfeat, stm, target);
                    local_n++;
                }
#ifdef PROFILE
                _TOCK(tw);
#endif

                if (tid == 0) {
                    long e = emitted.load() + local_n;
                    long em = e / 1000000;
                    if (em != last_emitted_m) {
                        last_emitted_m = em;
                        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                        std::fprintf(stderr, "  %ldM pos (%.1fM/s)\n", em, e / dt / 1e6);
                    }
                }
            }
        }
        if (stream_mode && !tbuf.empty()) {   // flush this thread's leftover
            std::lock_guard<std::mutex> lk(out_mtx);
            std::fwrite(tbuf.data(), 1, tbuf.size(), stdout);
            tbuf.resize(0);
            emitted.fetch_add(local_n); local_n = 0;
        } else if (!stream_mode && local_n) {
            emitted.fetch_add(local_n); local_n = 0;
        }
#ifdef PROFILE
        prof[tid] = {tset, tm, tf, tw};
#endif
    };

    std::vector<std::thread> th;
    for (int t = 0; t < nthreads; ++t) th.emplace_back(worker, t);
    for (auto& t : th) t.join();
#ifdef PROFILE
    {
        double se=0,m=0,f=0,w=0;
        for (auto& p : prof) { se+=p[0]; m+=p[1]; f+=p[2]; w+=p[3]; }
        double tot=se+m+f+w;
        if (tot>0) std::fprintf(stderr, "[PROFILE] setup=%.2fs (%.0f%%)  move=%.2fs (%.0f%%)  feat=%.2fs (%.0f%%)  write=%.2fs (%.0f%%)  [sum=%.2fs x nthreads]\n",
            se,100*se/tot, m,100*m/tot, f,100*f/tot, w,100*w/tot, tot);
    }
#endif

    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (stream_mode) {
        std::fflush(stdout);   // ensure all streamed records are flushed to the pipe
    } else {
        std::fclose(fw); std::fclose(fb); std::fclose(fs); std::fclose(ft);
        // meta.txt (first line = N, as c500_featurize.py writes).
        std::string metap = std::string(out_dir) + "/featurized_meta.txt";
        std::FILE* mf = std::fopen(metap.c_str(), "w");
        if (mf) { std::fprintf(mf, "%ld\n", total_pos); std::fclose(mf); }
    }

    std::fprintf(stderr, "\nDONE: %ld positions in %.1fs (%.2fM pos/s, %d threads)\n",
                 emitted.load(), dt, emitted.load() / dt / 1e6, nthreads);
    std::fprintf(stderr, "skipped games (movegen divergence): %ld\n", errors.load());
    return 0;
}

#elif defined(LUMINEX_TOOL_EVALTRACE)

// ============================================================================
// luminex-evaltrace — sparse feature decomposition of the HCE evaluation.
//
// Purpose: HCE distillation tuning. Reads FENs from stdin (one per line;
// "fen\tstm\ttarget" lines also accepted, extra fields ignored), emits ONE
// line per position:
//
//   <phase> <sf> <stm> <mirror_eval_white> <idx>:<val> <idx>:<val> ...
//
//   phase : 0..24 game phase (same formula as evaluate())
//   sf    : scale_factor (EG block enters the blend as (24-ph)/24 * sf/32)
//   stm   : 1 = white to move
//   mirror: int-exact mirror of evaluate()'s RETURN (stm-relative). The fit
//           flips to white perspective: y_white = mirror * (stm ? 1 : -1);
//           features are white-perspective so the flip applies to y only.
//   idx:val: sparse white-perspective feature counts. idx < NPHASE = MG
//           block, [NPHASE, 2*NPHASE) = EG block, then TEMPO_MG/TEMPO_EG.
//
// FIDELITY CONTRACT: mirrors evaluate() (evaluation.cpp) term-for-term with
// the SAME integer operations in the SAME order, so the mirror accumulator
// is bit-identical to the engine. Every add site also emits a feature;
// --verify checks BOTH invariants over the input:
//   (a) mirror == evaluate(pos) exactly            (structure is faithful)
//   (b) dot(features, current coefs) == mirror <=2 (features are complete)
//
// Positions hitting the non-linear early returns (known endgames) are not
// linear-representable: printed as "<ph> <sf> <stm> G" so the fit drops them.
// ============================================================================
#include "luminex.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace luminex {

namespace etrace {
int* piece_value_mg();  int* piece_value_eg();
int* knight_mob_mg();   int* knight_mob_eg();
int* bishop_mob_mg();   int* bishop_mob_eg();
int* rook_mob_mg();     int* rook_mob_eg();
int* queen_mob_mg();    int* queen_mob_eg();
const int* pawn_threat_mg();   const int* pawn_threat_eg();
const int* knight_threat_mg(); const int* knight_threat_eg();
const int* bishop_threat_mg(); const int* bishop_threat_eg();
const int* rook_threat_mg();   const int* rook_threat_eg();
const int* queen_threat_mg();  const int* queen_threat_eg();
const int* phalanx_mg(); const int* phalanx_eg();
const int* cand_mg();    const int* cand_eg();
const int* passed_mg();  const int* passed_eg();
}


struct Tracer {
    int f[feat::NFEAT] = {0};
    int touched[feat::NFEAT];
    bool listed[feat::NFEAT] = {false};
    int nt = 0;
    int mmg = 0, meg = 0;   // int-exact mirror of evaluate()'s accumulators

    // Membership is tracked explicitly: a feature whose running sum passes
    // through 0 (e.g. MAT+KNIGHT: +1 white, -1 black, -1 black) must NOT be
    // re-appended to touched[] — duplicates double-count in reconstruct().
    inline void add(int idx, int v) {
        if (v == 0) return;
        if (!listed[idx]) { listed[idx] = true; touched[nt++] = idx; }
        f[idx] += v;
    }
    void reset() {
        for (int i = 0; i < nt; ++i) { f[touched[i]] = 0; listed[touched[i]] = false; }
        nt = 0; mmg = 0; meg = 0;
    }
};

struct TraceOut { int ph = 0, sf = 32, stm = 1, mirror_white = 0, tension = 0; bool linear = true; };

// ============================================================================
// --solve mode: closed-form ridge accumulation, in-process.
//
// Row model (cp space, white perspective):
//   y    = target * (stm ? 1 : -1)              [SF eval, white-relative]
//   r    = y - tempo                            [tempo is fixed, not a fit param]
//   x_j  = f_j * ph / 24                (j in MG block)
//        = f_j * (24-ph) * sf / 768    (j in EG block; = (24-ph)/24 * sf/32)
//   predict: sum_j c_j x_j ~= r  ->  normal equations A = X'X, b = X'r over all rows.
// TEMPO features are excluded from the parameter vector. Writes binary
// <out>.bin = [int64 N][double A[NPF][NPF]][double b[NPF]] with NPF = 2*NPHASE.
// ============================================================================
constexpr int NSOLVE = 2 * feat::NPHASE;   // MG block (0..606) + EG block (607..1213)
struct SolverAcc {
    std::vector<double> A, b;     // A row-major NSOLVE*NSOLVE
    long long n = 0;
    double sum_r = 0, sum_r2 = 0; // train-side target stats: R^2 = 1 - SSE/ (sum_r2 - sum_r^2/n)
    SolverAcc() : A((size_t)NSOLVE * NSOLVE, 0.0), b(NSOLVE, 0.0) {}
    void add(const Tracer& t, int ph, int sf, double r, int* sidx, double* sxv, int& snz) {
        // build the weighted sparse row (MG then EG slots in solver space)
        snz = 0;
        double wmg = ph / 24.0;
        double weg = (24 - ph) * sf / 768.0;
        for (int i = 0; i < t.nt; ++i) {
            int idx = t.touched[i];
            int v = t.f[idx];
            if (v == 0) continue;
            // solver index space = feature index space (MG 0..606, EG 607..1213)
            if (idx < feat::NPHASE)          { sidx[snz] = idx; sxv[snz] = v * wmg; ++snz; }
            else if (idx < 2 * feat::NPHASE) { sidx[snz] = idx; sxv[snz] = v * weg; ++snz; }
        }
        // touched[] is in ADD order, not sorted. The outer product below only
        // fills A[min][max] (true upper triangle); an unsorted row would drop
        // every product whose later-added index is numerically smaller — the
        // mirror at flush would then overwrite them. Sort (idx,val) pairs first.
        for (int i = 1; i < snz; ++i) {
            int ti = sidx[i]; double tv = sxv[i];
            int j = i - 1;
            while (j >= 0 && sidx[j] > ti) { sidx[j + 1] = sidx[j]; sxv[j + 1] = sxv[j]; --j; }
            sidx[j + 1] = ti; sxv[j + 1] = tv;
        }
        for (int i = 0; i < snz; ++i) {
            int j = sidx[i];
            double xi = sxv[i];
            b[j] += xi * r;
            double* row = &A[(size_t)j * NSOLVE];
            for (int k = i; k < snz; ++k) row[sidx[k]] += xi * sxv[k];   // upper triangle
        }
        sum_r += r; sum_r2 += r * r;
        ++n;
    }
};

static TraceOut trace_eval(const Position& pos, Tracer& t);   // fwd (def below)

// ============================================================================
// --texel mode: logistic training (outcomes or sigma(eval) labels).
//
// Rows on stdin: "fen\tz" with z in [0,1] (P(white scores)). Model:
//   p = sigmoid((sum_j c_j x_j) / SCALE)   with the SAME blended sparse rows
//   as --solve (x_j includes ph/sf weighting; tempo excluded).
// Adam with mini-batches, L2 pull toward c0 (gauge anchor). Trains the
// decision-relevant regime naturally: confidently-wrong rows get the
// largest gradients, unlike uniform squared error.
// Env: NNUE_TEXEL_C0 (init/anchor coefs, required), NNUE_TEXEL_EPOCHS,
//      NNUE_TEXEL_LR, NNUE_TEXEL_SCALE, NNUE_TEXEL_L2, NNUE_TEXEL_BATCH.
// Output: coefs file (arg after --texel), same "j value" format as --solve.
// ============================================================================
constexpr int NTX = 2 * feat::NPHASE;

static int build_sparse_row(const Tracer& t, int ph, int sf, int* sidx, double* sxv) {
    int snz = 0;
    double wmg = ph / 24.0;
    double weg = (24 - ph) * sf / 768.0;
    for (int i = 0; i < t.nt; ++i) {
        int idx = t.touched[i];
        int v = t.f[idx];
        if (v == 0) continue;
        if (idx < feat::NPHASE)          { sidx[snz] = idx; sxv[snz] = v * wmg; ++snz; }
        else if (idx < 2 * feat::NPHASE) { sidx[snz] = idx; sxv[snz] = v * weg; ++snz; }
    }
    for (int i = 1; i < snz; ++i) {          // sort by index (stable enough)
        int ti = sidx[i]; double tv = sxv[i];
        int j = i - 1;
        while (j >= 0 && sidx[j] > ti) { sidx[j + 1] = sidx[j]; sxv[j + 1] = sxv[j]; --j; }
        sidx[j + 1] = ti; sxv[j + 1] = tv;
    }
    return snz;
}

static int texel_train(const char* out_path, const char* c0_path) {
    auto envd = [](const char* k, double d) {
        const char* s = std::getenv(k);
        return s ? std::atof(s) : d;
    };
    int epochs = (int)envd("NNUE_TEXEL_EPOCHS", 40);
    double lr = envd("NNUE_TEXEL_LR", 0.001);
    double scale = envd("NNUE_TEXEL_SCALE", 400.0);
    double l2 = envd("NNUE_TEXEL_L2", 1e-7);
    int batch = (int)envd("NNUE_TEXEL_BATCH", 16384);
    int holdevery = (int)envd("NNUE_TEXEL_HOLDOUT", 0);   // every Nth row: eval-only
    double wsharp = envd("NNUE_TEXEL_WSHARP", 0.0);       // tension row-weight (0 = off)

    std::vector<double> c0(NTX, 0.0);
    {
        std::FILE* f = std::fopen(c0_path, "r");
        if (!f) { std::fprintf(stderr, "texel: cannot open c0 %s\n", c0_path); return 2; }
        int j; double v;
        while (std::fscanf(f, "%d %lf", &j, &v) == 2)
            if (j >= 0 && j < NTX) c0[j] = v;
        std::fclose(f);
    }
    std::vector<double> c = c0, m(NTX, 0.0), vv(NTX, 0.0), grad(NTX, 0.0);
    int adam_t = 0;

    // Load rows: fen \t z
    std::vector<std::pair<std::string, double>> rows;
    std::string line;
    long long bad = 0;
    while (std::getline(std::cin, line)) {
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        char* endp = nullptr;
        double z = std::strtod(line.c_str() + tab + 1, &endp);
        if (!endp || endp == line.c_str() + tab + 1 || z < 0.0 || z > 1.0) { ++bad; continue; }
        rows.emplace_back(line.substr(0, tab), z);
    }
    std::fprintf(stderr, "texel: %zu rows (%lld bad)\n", rows.size(), bad);

    Position pos;
    std::vector<int> sidx(feat::NFEAT);
    std::vector<double> sxv(feat::NFEAT);
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8;

    for (int ep = 1; ep <= epochs; ++ep) {
        double loss_sum = 0.0, hold_sum = 0.0; long long loss_n = 0, hold_n = 0; int in_batch = 0;
        size_t row_i = 0;
        std::fill(grad.begin(), grad.end(), 0.0);
        for (auto& [fen, z] : rows) {
            ++row_i;
            bool is_holdout = holdevery > 0 && (row_i % holdevery) == 0;
            // Cheap validity guard (built -fno-exceptions; set() on garbage is UB)
            int fields = 1;
            for (char ch : fen) if (ch == ' ') fields++;
            if (fields < 4) continue;
            pos.set(fen);
            Tracer t;
            TraceOut o = trace_eval(pos, t);
            int snz = build_sparse_row(t, o.ph, o.sf, sidx.data(), sxv.data());
            double score = 0.0;
            for (int i = 0; i < snz; ++i) score += c[sidx[i]] * sxv[i];
            double p = 1.0 / (1.0 + std::exp(-score / scale));
            double pc = std::min(std::max(p, 1e-12), 1.0 - 1e-12);
            double lrow = -(z * std::log(pc) + (1 - z) * std::log(1 - pc));
            if (is_holdout) { hold_sum += lrow; ++hold_n; continue; }
            loss_sum += lrow;
            ++loss_n;
            double g = (p - z) / scale;              // dLoss/dscore
            if (wsharp > 0.0) g *= 1.0 + wsharp * std::min(60, o.tension) / 30.0;
            for (int i = 0; i < snz; ++i) grad[sidx[i]] += g * sxv[i];
            if (++in_batch == batch) {
                adam_t++;
                double bc1 = 1 - std::pow(b1, adam_t), bc2 = 1 - std::pow(b2, adam_t);
                for (int j = 0; j < NTX; ++j) {
                    double gj = grad[j] / batch + l2 * (c[j] - c0[j]);
                    m[j] = b1 * m[j] + (1 - b1) * gj;
                    vv[j] = b2 * vv[j] + (1 - b2) * gj * gj;
                    c[j] -= lr * (m[j] / bc1) / (std::sqrt(vv[j] / bc2) + eps);
                }
                std::fill(grad.begin(), grad.end(), 0.0);
                in_batch = 0;
            }
        }
        if (in_batch > 0) {   // flush trailing partial batch
            adam_t++;
            double bc1 = 1 - std::pow(b1, adam_t), bc2 = 1 - std::pow(b2, adam_t);
            for (int j = 0; j < NTX; ++j) {
                double gj = grad[j] / in_batch + l2 * (c[j] - c0[j]);
                m[j] = b1 * m[j] + (1 - b1) * gj;
                vv[j] = b2 * vv[j] + (1 - b2) * gj * gj;
                c[j] -= lr * (m[j] / bc1) / (std::sqrt(vv[j] / bc2) + eps);
            }
        }
        std::fprintf(stderr, "texel: epoch %d/%d  logloss=%.6f  holdout=%.6f (n=%lld)  max|c-c0|=%.4f  mean|c-c0|=%.5f\n",
                     ep, epochs, loss_n ? loss_sum / loss_n : 0.0,
                     hold_n ? hold_sum / hold_n : -1.0, hold_n,
                     [&]{ double mx = 0; for (int j = 0; j < NTX; ++j) mx = std::max(mx, std::abs(c[j] - c0[j])); return mx; }(),
                     [&]{ double s = 0; for (int j = 0; j < NTX; ++j) s += std::abs(c[j] - c0[j]); return s / NTX; }());
    }
    std::FILE* fo = std::fopen(out_path, "w");
    if (!fo) { std::fprintf(stderr, "texel: cannot write %s\n", out_path); return 2; }
    for (int j = 0; j < NTX; ++j) std::fprintf(fo, "%d %.6f\n", j, c[j]);
    std::fclose(fo);
    std::fprintf(stderr, "texel: wrote %s\n", out_path);
    return 0;
}

// Mirror of evaluate_pawns (engine caches in pawn_table; values identical).
static void trace_pawns(const Position& pos, Tracer& t, int32_t& mg_out, int32_t& eg_out) {
    using namespace feat;
    const int* PVMG = etrace::piece_value_mg();
    const int* PVEG = etrace::piece_value_eg();
    const int* PHMG = etrace::phalanx_mg();
    const int* PHEG = etrace::phalanx_eg();
    const int* CAMG = etrace::cand_mg();
    const int* CAEG = etrace::cand_eg();
    const int* PAMG = etrace::passed_mg();
    const int* PAEG = etrace::passed_eg();
    int32_t mg = 0;
    int32_t eg = 0;

    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color c = Color(c_idx);
        Color them = Color(c_idx ^ 1);
        int sign = (c == WHITE) ? 1 : -1;
        Bitboard our_pawns = pos.pieces(c, PAWN);
        Bitboard their_pawns = pos.pieces(them, PAWN);

        int file_count[8] = {0};
        Bitboard tmp = our_pawns;
        while (tmp) { file_count[file_of(pop_lsb(tmp))]++; }

        Bitboard supported_by_adj = pawn_attacks_bb(c, our_pawns);

        Bitboard pawns = our_pawns;
        while (pawns) {
            Square sq = pop_lsb(pawns);
            File f = file_of(sq);
            Rank r = relative_rank(c, sq);
            int relsq = (c == WHITE) ? int(sq) : (int(sq) ^ 56);

            // Material + PST
            mg += sign * (PVMG[PAWN] + PST_MG_TABLE[int(c)][int(PAWN)][int(sq)]);
            eg += sign * (PVEG[PAWN] + PST_EG_TABLE[int(c)][int(PAWN)][int(sq)]);
            t.add(PST + int(PAWN) * 64 + relsq, sign);
            t.add(PST + int(PAWN) * 64 + relsq + NPHASE, sign);
            t.add(MAT + int(PAWN), sign);
            t.add(MAT + int(PAWN) + NPHASE, sign);

            // Doubled pawn
            if (file_count[f] > 1) {
                mg += sign * FE_MG[PAWN_DOUBLED];
                eg += sign * FE_EG[PAWN_DOUBLED];
                t.add(PAWN_DOUBLED, sign); t.add(PAWN_DOUBLED + NPHASE, sign);
            }

            // Isolated pawn
            bool left = (f > FILE_A && file_count[f - 1] > 0);
            bool right = (f < FILE_H && file_count[f + 1] > 0);
            if (!left && !right) {
                mg += sign * FE_MG[PAWN_ISOLATED];
                eg += sign * FE_EG[PAWN_ISOLATED];
                t.add(PAWN_ISOLATED, sign); t.add(PAWN_ISOLATED + NPHASE, sign);
            }

            // Connected pawn (protected by another pawn)
            if (square_bb(sq) & supported_by_adj) {
                mg += sign * FE_MG[PAWN_CONNECTED + r - 1];
                eg += sign * FE_EG[PAWN_CONNECTED + r - 1];
                t.add(PAWN_CONNECTED + (r - 1), sign);
                t.add(PAWN_CONNECTED + (r - 1) + NPHASE, sign);
            }

            // Phalanx
            {
                Bitboard neighbors = our_pawns & (file_bb(File(f - 1)) | file_bb(File(f + 1)));
                Bitboard same_rank = neighbors & rank_bb(rank_of(sq));
                if (same_rank) {
                    if (r >= RANK_2 && r <= RANK_7) {
                        mg += sign * PHMG[r];
                        eg += sign * PHEG[r];
                        t.add(PAWN_PHALANX + (r - 1), sign);
                        t.add(PAWN_PHALANX + (r - 1) + NPHASE, sign);
                    }
                }
            }

            // Lever
            {
                Bitboard lever_targets = pawn_attacks_bb(c, sq) & their_pawns;
                if (lever_targets) {
                    mg += sign * FE_MG[PAWN_LEVER + r - 1];
                    t.add(PAWN_LEVER + (r - 1), sign);
                    if (f >= FILE_C && f <= FILE_F) {
                        mg += sign * FE_MG[PAWN_LEVER_C];
                        t.add(PAWN_LEVER_C, sign);
                    }
                }
            }

            // Backward pawn
            {
                Square push = relative_square(c, make_square(f, Rank(r + 1)));
                if (push < SQUARE_NONE && r >= RANK_2 && r <= RANK_5) {
                    bool push_attacked = (pawn_attacks_bb(them, their_pawns) & square_bb(push)) != 0;
                    if (push_attacked) {
                        Bitboard our_pawn_attacks = pawn_attacks_bb(c, our_pawns);
                        bool defended = (our_pawn_attacks & square_bb(push)) != 0;
                        if (!defended) {
                            mg += sign * FE_MG[PAWN_BACKWARD];
                            eg += sign * FE_EG[PAWN_BACKWARD];
                            t.add(PAWN_BACKWARD, sign); t.add(PAWN_BACKWARD + NPHASE, sign);
                        }
                    }
                }
            }

            // Passed / candidate detection
            Bitboard ahead = 0;
            for (int rr = r + 1; rr <= RANK_7; ++rr) {
                Square rsq = relative_square(c, make_square(f, Rank(rr)));
                ahead |= square_bb(rsq);
                if (f > FILE_A) ahead |= square_bb(relative_square(c, make_square(File(f - 1), Rank(rr))));
                if (f < FILE_H) ahead |= square_bb(relative_square(c, make_square(File(f + 1), Rank(rr))));
            }

            if ((ahead & their_pawns) && r >= RANK_2 && r <= RANK_6) {
                Bitboard stoppers = ahead & their_pawns;
                Bitboard threats = their_pawns & pawn_attacks_bb(c, sq);
                Bitboard support = our_pawns & pawn_attacks_bb(them, sq);
                Bitboard push_sq_bb = relative_square(c, make_square(f, Rank(r + 1)));
                Bitboard push_threats = their_pawns & pawn_attacks_bb(c, push_sq_bb);
                Bitboard push_support = our_pawns & pawn_attacks_bb(them, push_sq_bb);
                Bitboard leftovers = stoppers & ~(threats | push_threats);
                bool supported = popcount(support) >= popcount(threats);
                if (!leftovers && popcount(push_support) >= popcount(push_threats)) {
                    int sup = supported ? 1 : 0;
                    mg += sign * CAMG[sup * 8 + r];
                    eg += sign * CAEG[sup * 8 + r];
                    t.add(PAWN_CAND + sup * 7 + (r - 1), sign);
                    t.add(PAWN_CAND + sup * 7 + (r - 1) + NPHASE, sign);
                }
            }

            if (!(ahead & their_pawns)) {
                mg += sign * PAMG[r];
                eg += sign * PAEG[r];
                t.add(PAWN_PASSED + (r - 1), sign);
                t.add(PAWN_PASSED + (r - 1) + NPHASE, sign);
            }
        }
    }

    mg_out = mg;
    eg_out = eg;
    t.mmg += mg;
    t.meg += eg;
}

// Full mirror of evaluate(pos, tactical_only=false). Emits features into t,
// keeps t.mmg/t.meg int-exact, returns phase/scale/mirror.
static TraceOut trace_eval(const Position& pos, Tracer& t) {
    using namespace feat;
    TraceOut out;
    out.stm = (pos.side_to_move() == WHITE) ? 1 : 0;

    // --- non-linear gate (known endgames: KPK bitbase, bare king): not linear-representable ---
    Value known;
    if (known_endgame(pos, known)) { out.linear = false; return out; }

    int mg_score = 0;
    int eg_score = 0;

    Square ksq_arr[2] = {pos.king_sq(WHITE), pos.king_sq(BLACK)};
    int bishop_count[2] = {0, 0};

    Bitboard occupied = pos.pieces();

    Bitboard attacks_by[2][7] = {};
    Bitboard all_attacks[2] = {};

    for (int c = 0; c < 2; ++c) {
        attacks_by[c][PAWN] = pawn_attacks_bb(Color(c), pos.pieces(Color(c), PAWN));
        attacks_by[c][KING] = king_attacks_bb(ksq_arr[c]);
        all_attacks[c] = attacks_by[c][PAWN] | attacks_by[c][KING];
    }

    {
        int32_t pawn_mg = 0, pawn_eg = 0;
        trace_pawns(pos, t, pawn_mg, pawn_eg);
        mg_score += pawn_mg;
        eg_score += pawn_eg;
    }

    const int* PVMG = etrace::piece_value_mg();
    const int* PVEG = etrace::piece_value_eg();

    // Razor-interaction inputs (E2b): filled in the threat loop, consumed by
    // the king-safety section — mirrors evaluation.cpp.
    int thr_cnt_arr[2]  = {0, 0};
    int hang_cnt_arr[2] = {0, 0};

    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color c = Color(c_idx);
        Color them = Color(c_idx ^ 1);
        int sign = (c == WHITE) ? 1 : -1;
        Bitboard our_pawns = pos.pieces(c, PAWN);
        Bitboard their_pawns = pos.pieces(them, PAWN);

        Bitboard mob_area = ~pawn_attacks_bb(them, their_pawns);

        // ----------------- passed pawn extras -----------------
        {
        Bitboard passed_pawns_bb = BB_EMPTY;
        {
            Bitboard pawns = our_pawns;
            while (pawns) {
                Square sq = pop_lsb(pawns);
                File f = file_of(sq);
                Rank r = relative_rank(c, sq);
                Bitboard ahead = 0;
                for (int rr = r + 1; rr <= RANK_7; ++rr) {
                    Square rsq = relative_square(c, make_square(f, Rank(rr)));
                    ahead |= square_bb(rsq);
                    if (f > FILE_A) ahead |= square_bb(relative_square(c, make_square(File(f - 1), Rank(rr))));
                    if (f < FILE_H) ahead |= square_bb(relative_square(c, make_square(File(f + 1), Rank(rr))));
                }
                if (!(ahead & their_pawns)) passed_pawns_bb |= square_bb(sq);
            }
        }

        Bitboard pp = passed_pawns_bb;
        while (pp) {
            Square sq = pop_lsb(pp);
            File f = file_of(sq);
            Rank r = relative_rank(c, sq);

            Bitboard ahead_file = 0;
            for (int rr = r + 1; rr <= RANK_7; ++rr) {
                ahead_file |= square_bb(relative_square(c, make_square(f, Rank(rr))));
            }

            int mg_passer = 0;
            int eg_passer = 0;

            Bitboard adjacent_files = BB_EMPTY;
            if (f > FILE_A) adjacent_files |= file_bb(File(f - 1));
            if (f < FILE_H) adjacent_files |= file_bb(File(f + 1));
            Bitboard connected = passed_pawns_bb & adjacent_files & rank_bb(rank_of(sq));
            if (connected) {
                int n_connected = popcount(connected);
                mg_passer += n_connected * FE_MG[PP_CONNECTED + r - 1];
                eg_passer += n_connected * FE_EG[PP_CONNECTED + r - 1];
                t.add(PP_CONNECTED + (r - 1), sign * n_connected);
                t.add(PP_CONNECTED + (r - 1) + NPHASE, sign * n_connected);
            }

            Bitboard behind = file_bb(f) & pos.pieces(c, ROOK);
            if (behind) {
                Square rsq2 = lsb(behind);
                bool rook_behind = (c == WHITE) ? (rank_of(rsq2) < rank_of(sq)) : (rank_of(rsq2) > rank_of(sq));
                if (rook_behind) {
                    mg_passer += FE_MG[PP_ROOK_BEHIND];
                    eg_passer += FE_EG[PP_ROOK_BEHIND];
                    t.add(PP_ROOK_BEHIND, sign); t.add(PP_ROOK_BEHIND + NPHASE, sign);
                }
            }

            Bitboard enemy_behind = file_bb(f) & pos.pieces(them, ROOK);
            if (enemy_behind) {
                Square ersq = lsb(enemy_behind);
                bool enemy_rook_behind = (c == WHITE) ? (rank_of(ersq) < rank_of(sq)) : (rank_of(ersq) > rank_of(sq));
                if (enemy_rook_behind) {
                    mg_passer += FE_MG[PP_ENEMY_ROOK];
                    eg_passer += FE_EG[PP_ENEMY_ROOK];
                    t.add(PP_ENEMY_ROOK, sign); t.add(PP_ENEMY_ROOK + NPHASE, sign);
                }
            }

            Square our_ksq = ksq_arr[c_idx];
            Square their_ksq = ksq_arr[c_idx ^ 1];
            Square promo_sq = relative_square(c, make_square(f, RANK_8));
            int our_kdist = distance(our_ksq, promo_sq);
            int their_kdist = distance(their_ksq, promo_sq);
            eg_passer += (their_kdist - our_kdist) * FE_EG[PP_KING_PROX];
            t.add(PP_KING_PROX + NPHASE, sign * (their_kdist - our_kdist));

            if (their_kdist > our_kdist + (c == pos.side_to_move() ? 0 : 1)) {
                int unreachable = their_kdist - our_kdist;
                eg_passer += unreachable * (unreachable > 4 ? FE_EG[PP_UNREACH_B]
                                                           : FE_EG[PP_UNREACH_S]);
                if (unreachable > 4) t.add(PP_UNREACH_B + NPHASE, sign * unreachable);
                else                 t.add(PP_UNREACH_S + NPHASE, sign * unreachable);
            }

            {
                Square stop_sq = relative_square(c, make_square(f, Rank(r + 1)));
                bool stop_enemy = (pos.pieces(them) & square_bb(stop_sq)) != 0;
                bool stop_own   = (pos.pieces(c)    & square_bb(stop_sq)) != 0;
                bool stop_attacked_by_enemy = (pos.attackers_to(stop_sq) & pos.pieces(them)) != 0;
                bool enemy_on_path = (ahead_file & pos.pieces(them)) != 0;

                if (stop_enemy) {
                    mg_passer += FE_MG[PP_BLOCKADE + r - 1];
                    eg_passer += FE_EG[PP_BLOCKADE + r - 1];
                    t.add(PP_BLOCKADE + (r - 1), sign); t.add(PP_BLOCKADE + (r - 1) + NPHASE, sign);
                } else if (stop_own) {
                    mg_passer += FE_MG[PP_STOP_OWN];
                    eg_passer += FE_EG[PP_STOP_OWN];
                    t.add(PP_STOP_OWN, sign); t.add(PP_STOP_OWN + NPHASE, sign);
                } else if (stop_attacked_by_enemy) {
                    mg_passer += FE_MG[PP_STOP_ATT];
                    eg_passer += FE_EG[PP_STOP_ATT];
                    t.add(PP_STOP_ATT, sign); t.add(PP_STOP_ATT + NPHASE, sign);
                } else {
                    mg_passer += FE_MG[PP_SAFE + r - 1];
                    eg_passer += FE_EG[PP_SAFE + r - 1];
                    t.add(PP_SAFE + (r - 1), sign); t.add(PP_SAFE + (r - 1) + NPHASE, sign);
                    if (!enemy_on_path) {
                        eg_passer += FE_EG[PP_CLEAR + r - 1];
                        t.add(PP_CLEAR + (r - 1) + NPHASE, sign);
                    }
                }
            }

            mg_score += sign * mg_passer;
            eg_score += sign * eg_passer;
        }
        }

        // ----------------- knights -----------------
        Bitboard knights = pos.pieces(c, KNIGHT);
        while (knights) {
            Square sq = pop_lsb(knights);
            int relsq = (c == WHITE) ? int(sq) : (int(sq) ^ 56);
            mg_score += sign * (PVMG[KNIGHT] + PST_MG_TABLE[int(c)][int(KNIGHT)][int(sq)]);
            eg_score += sign * (PVEG[KNIGHT] + PST_EG_TABLE[int(c)][int(KNIGHT)][int(sq)]);
            t.add(PST + int(KNIGHT) * 64 + relsq, sign);
            t.add(PST + int(KNIGHT) * 64 + relsq + NPHASE, sign);
            t.add(MAT + int(KNIGHT), sign);
            t.add(MAT + int(KNIGHT) + NPHASE, sign);

            Bitboard attacks = knight_attacks_bb(sq);
            attacks_by[c][KNIGHT] |= attacks;
            all_attacks[c] |= attacks;

            int mob = popcount(attacks & mob_area & ~pos.pieces(c));
            mob = std::min(mob, 8);
            mg_score += sign * etrace::knight_mob_mg()[mob];
            eg_score += sign * etrace::knight_mob_eg()[mob];
            t.add(MOB_N + mob, sign); t.add(MOB_N + mob + NPHASE, sign);

            Rank kr = relative_rank(c, sq);
            if (kr >= RANK_4 && kr <= RANK_6) {
                if (pawn_attacks_bb(c, our_pawns) & square_bb(sq)) {
                    Bitboard enemy_pawn_attacks = pawn_attacks_bb(them, their_pawns);
                    if (!(enemy_pawn_attacks & square_bb(sq))) {
                        mg_score += sign * FE_MG[OUTPOST_N + kr - 3];
                        eg_score += sign * FE_EG[OUTPOST_N + kr - 3];
                        t.add(OUTPOST_N + (kr - 3), sign);
                        t.add(OUTPOST_N + (kr - 3) + NPHASE, sign);
                    }
                }
            }

            if (mob == 0) {
                mg_score += sign * FE_MG[TRAPPED_N];
                eg_score += sign * FE_EG[TRAPPED_N];
                t.add(TRAPPED_N, sign); t.add(TRAPPED_N + NPHASE, sign);
            }

            if (distance(sq, ksq_arr[c_idx]) > 3) {
                mg_score += sign * FE_MG[FAR_N];
                eg_score += sign * FE_EG[FAR_N];
                t.add(FAR_N, sign); t.add(FAR_N + NPHASE, sign);
            }
        }

        // ----------------- bishops -----------------
        Bitboard bishops = pos.pieces(c, BISHOP);
        bishop_count[c_idx] = popcount(bishops);
        while (bishops) {
            Square sq = pop_lsb(bishops);
            int relsq = (c == WHITE) ? int(sq) : (int(sq) ^ 56);
            mg_score += sign * (PVMG[BISHOP] + PST_MG_TABLE[int(c)][int(BISHOP)][int(sq)]);
            eg_score += sign * (PVEG[BISHOP] + PST_EG_TABLE[int(c)][int(BISHOP)][int(sq)]);
            t.add(PST + int(BISHOP) * 64 + relsq, sign);
            t.add(PST + int(BISHOP) * 64 + relsq + NPHASE, sign);
            t.add(MAT + int(BISHOP), sign);
            t.add(MAT + int(BISHOP) + NPHASE, sign);

            Bitboard attacks = bb_diag_attacks(sq, occupied);
            attacks_by[c][BISHOP] |= attacks;
            all_attacks[c] |= attacks;

            int mob = popcount(attacks & mob_area & ~pos.pieces(c));
            mob = std::min(mob, 13);
            mg_score += sign * etrace::bishop_mob_mg()[mob];
            eg_score += sign * etrace::bishop_mob_eg()[mob];
            t.add(MOB_B + mob, sign); t.add(MOB_B + mob + NPHASE, sign);

            {
                static constexpr Bitboard BB_DARK_SQ = 0x55AA55AA55AA55AAULL;
                static constexpr Bitboard BB_LIGHT_SQ = 0xAA55AA55AA55AA55ULL;
                bool is_dark = ((int(sq) + rank_of(sq)) % 2) == 0;
                Bitboard same_color_bb = is_dark ? BB_DARK_SQ : BB_LIGHT_SQ;
                int same_color_pawns = popcount(our_pawns & same_color_bb);
                same_color_pawns = std::min(same_color_pawns, 6);
                mg_score += sign * FE_MG[BSC + same_color_pawns];
                eg_score += sign * FE_EG[BSC + same_color_pawns];
                t.add(BSC + same_color_pawns, sign);
                t.add(BSC + same_color_pawns + NPHASE, sign);
            }

            if (popcount(attacks & BB_CENTER) >= 2) {
                mg_score += sign * FE_MG[BISH_LONG];
                eg_score += sign * FE_EG[BISH_LONG];
                t.add(BISH_LONG, sign); t.add(BISH_LONG + NPHASE, sign);
            }

            {
                Rank br = relative_rank(c, sq);
                if (br >= RANK_4 && br <= RANK_6) {
                    if ((pawn_attacks_bb(c, our_pawns) & square_bb(sq)) &&
                        !(pawn_attacks_bb(them, their_pawns) & square_bb(sq))) {
                        mg_score += sign * FE_MG[OUTPOST_B];
                        eg_score += sign * FE_EG[OUTPOST_B];
                        t.add(OUTPOST_B, sign); t.add(OUTPOST_B + NPHASE, sign);
                    }
                }
            }

            if (distance(sq, ksq_arr[c_idx]) > 3) {
                mg_score += sign * FE_MG[FAR_B];
                eg_score += sign * FE_EG[FAR_B];
                t.add(FAR_B, sign); t.add(FAR_B + NPHASE, sign);
            }
        }

        // ----------------- rooks -----------------
        Bitboard rooks = pos.pieces(c, ROOK);
        while (rooks) {
            Square sq = pop_lsb(rooks);
            int relsq = (c == WHITE) ? int(sq) : (int(sq) ^ 56);
            mg_score += sign * (PVMG[ROOK] + PST_MG_TABLE[int(c)][int(ROOK)][int(sq)]);
            eg_score += sign * (PVEG[ROOK] + PST_EG_TABLE[int(c)][int(ROOK)][int(sq)]);
            t.add(PST + int(ROOK) * 64 + relsq, sign);
            t.add(PST + int(ROOK) * 64 + relsq + NPHASE, sign);
            t.add(MAT + int(ROOK), sign);
            t.add(MAT + int(ROOK) + NPHASE, sign);

            Bitboard attacks = rook_attacks_bb(sq, occupied ^ pos.pieces(c, ROOK) ^ pos.pieces(c, QUEEN));
            attacks_by[c][ROOK] |= attacks;
            all_attacks[c] |= attacks;

            int mob = popcount(attacks & mob_area & ~pos.pieces(c));
            mob = std::min(mob, 14);
            mg_score += sign * etrace::rook_mob_mg()[mob];
            eg_score += sign * etrace::rook_mob_eg()[mob];
            t.add(MOB_R + mob, sign); t.add(MOB_R + mob + NPHASE, sign);

            File f = file_of(sq);
            if (!(pos.pieces(PAWN) & file_bb(f))) {
                mg_score += sign * FE_MG[ROOK_OPEN];
                eg_score += sign * FE_EG[ROOK_OPEN];
                t.add(ROOK_OPEN, sign); t.add(ROOK_OPEN + NPHASE, sign);
            } else if (!(our_pawns & file_bb(f))) {
                mg_score += sign * FE_MG[ROOK_SEMI];
                eg_score += sign * FE_EG[ROOK_SEMI];
                t.add(ROOK_SEMI, sign); t.add(ROOK_SEMI + NPHASE, sign);
            }

            Rank rr = relative_rank(c, rank_of(sq));
            if (rr == RANK_7) {
                mg_score += sign * FE_MG[ROOK_7TH];
                eg_score += sign * FE_EG[ROOK_7TH];
                t.add(ROOK_7TH, sign); t.add(ROOK_7TH + NPHASE, sign);
            }

            if (file_bb(f) & pos.pieces(them, QUEEN)) {
                mg_score += sign * FE_MG[ROOK_XRAY_Q];
                eg_score += sign * FE_EG[ROOK_XRAY_Q];
                t.add(ROOK_XRAY_Q, sign); t.add(ROOK_XRAY_Q + NPHASE, sign);
            }

            if (distance(sq, ksq_arr[c_idx]) > 3) {
                mg_score += sign * FE_MG[FAR_R];
                eg_score += sign * FE_EG[FAR_R];
                t.add(FAR_R, sign); t.add(FAR_R + NPHASE, sign);
            }

            if (rr <= RANK_2) {
                Square ksq = ksq_arr[c_idx];
                int fdist = std::abs(file_of(sq) - file_of(ksq));
                bool king_between = (c == WHITE)
                    ? (rank_of(ksq) <= rank_of(sq) && fdist <= 1)
                    : (rank_of(ksq) >= rank_of(sq) && fdist <= 1);
                if (king_between && mob <= 4) {
                    CastlingRight ks_cr = c == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE;
                    CastlingRight qs_cr = c == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE;
                    bool has_castling = pos.castling_allowed(c, ks_cr) || pos.castling_allowed(c, qs_cr);
                    if (has_castling) {
                        mg_score += sign * FE_MG[ROOK_TRAP_C];
                        eg_score += sign * FE_EG[ROOK_TRAP_C];
                        t.add(ROOK_TRAP_C, sign); t.add(ROOK_TRAP_C + NPHASE, sign);
                    } else {
                        mg_score += sign * FE_MG[ROOK_TRAP_NC];
                        eg_score += sign * FE_EG[ROOK_TRAP_NC];
                        t.add(ROOK_TRAP_NC, sign); t.add(ROOK_TRAP_NC + NPHASE, sign);
                    }
                }
            }

            {
                Bitboard ahead_pawns;
                if (c == WHITE) {
                    Bitboard below_and_same = 0;
                    for (int r = RANK_1; r <= rank_of(sq); ++r)
                        below_and_same |= rank_bb(Rank(r));
                    ahead_pawns = our_pawns & file_bb(f) & ~below_and_same;
                } else {
                    Bitboard above_and_same = 0;
                    for (int r = rank_of(sq); r <= RANK_8; ++r)
                        above_and_same |= rank_bb(Rank(r));
                    ahead_pawns = our_pawns & file_bb(f) & ~above_and_same;
                }
                if (ahead_pawns) {
                    mg_score += sign * FE_MG[ROOK_BLOCKED];
                    eg_score += sign * FE_EG[ROOK_BLOCKED];
                    t.add(ROOK_BLOCKED, sign); t.add(ROOK_BLOCKED + NPHASE, sign);
                }
            }
        }

        // ----------------- queens -----------------
        Bitboard queens = pos.pieces(c, QUEEN);
        while (queens) {
            Square sq = pop_lsb(queens);
            int relsq = (c == WHITE) ? int(sq) : (int(sq) ^ 56);
            mg_score += sign * (PVMG[QUEEN] + PST_MG_TABLE[int(c)][int(QUEEN)][int(sq)]);
            eg_score += sign * (PVEG[QUEEN] + PST_EG_TABLE[int(c)][int(QUEEN)][int(sq)]);
            t.add(PST + int(QUEEN) * 64 + relsq, sign);
            t.add(PST + int(QUEEN) * 64 + relsq + NPHASE, sign);
            t.add(MAT + int(QUEEN), sign);
            t.add(MAT + int(QUEEN) + NPHASE, sign);

            Bitboard attacks = bb_diag_attacks(sq, occupied ^ pos.pieces(c, BISHOP))
                             | rook_attacks_bb(sq, occupied ^ pos.pieces(c, ROOK));
            attacks_by[c][QUEEN] |= attacks;
            all_attacks[c] |= attacks;

            int mob = popcount(attacks & mob_area & ~pos.pieces(c));
            mob = std::min(mob, 27);
            mg_score += sign * etrace::queen_mob_mg()[mob];
            eg_score += sign * etrace::queen_mob_eg()[mob];
            t.add(MOB_Q + mob, sign); t.add(MOB_Q + mob + NPHASE, sign);

            if (distance(sq, ksq_arr[c_idx]) > 3) {
                mg_score += sign * FE_MG[FAR_Q];
                eg_score += sign * FE_EG[FAR_Q];
                t.add(FAR_Q, sign); t.add(FAR_Q + NPHASE, sign);
            }
        }

        // ----------------- king -----------------
        Square ksq = ksq_arr[c_idx];
        {
            int relsq = (c == WHITE) ? int(ksq) : (int(ksq) ^ 56);
            mg_score += sign * (PVMG[KING] + PST_MG_TABLE[int(c)][int(KING)][int(ksq)]);
            eg_score += sign * (PVEG[KING] + PST_EG_TABLE[int(c)][int(KING)][int(ksq)]);
            t.add(PST + int(KING) * 64 + relsq, sign);
            t.add(PST + int(KING) * 64 + relsq + NPHASE, sign);
        }

        {
        Rank krank = rank_of(ksq);
        File kfile = file_of(ksq);
        bool on_back = (c == WHITE && krank <= RANK_2) || (c == BLACK && krank >= RANK_7);
        if (on_back) {
            for (int r = 1; r <= 3; ++r) {
                int feat_id = (r == 1) ? SHIELD_R1 : (r == 2) ? SHIELD_R2 : SHIELD_R3;
                int weight = (r == 1) ? FE_MG[SHIELD_R1]
                             : (r == 2) ? FE_MG[SHIELD_R2]
                             : FE_MG[SHIELD_R3];
                for (int df = -1; df <= 1; ++df) {
                    File sf = File(int(kfile) + df);
                    if (sf < FILE_A || sf > FILE_H) continue;
                    Square shield_sq = relative_square(c, make_square(sf, Rank(r)));
                    if (our_pawns & square_bb(shield_sq)) {
                        mg_score += sign * weight;
                        t.add(feat_id, sign);
                    }
                }
            }

            // Per-file pawn shield (mirrors evaluation.cpp): king-file vs
            // adjacent vs next-adjacent shelter resolved per rank band, plus
            // own-pawn-missing HOLE flags.
            {
                int kf_cnt[3] = {0, 0, 0}, adj_cnt[3] = {0, 0, 0}, ext_cnt[3] = {0, 0, 0};
                for (int df = -2; df <= 2; ++df) {
                    File sf = File(int(kfile) + df);
                    if (sf < FILE_A || sf > FILE_H) continue;
                    for (int r = 1; r <= 3; ++r) {
                        if (!(our_pawns & square_bb(relative_square(c, make_square(sf, Rank(r)))))) continue;
                        if (df == 0)      ++kf_cnt[r - 1];
                        else if (df == -1 || df == 1) ++adj_cnt[r - 1];
                        else              ++ext_cnt[r - 1];
                    }
                }
                int hole_kf = (our_pawns & file_bb(kfile)) ? 0 : 1;
                int hole_adj = 0;
                if (kfile > FILE_A && !(our_pawns & file_bb(File(kfile - 1)))) ++hole_adj;
                if (kfile < FILE_H && !(our_pawns & file_bb(File(kfile + 1)))) ++hole_adj;
                for (int r = 1; r <= 3; ++r) {
                    int kf_j = SHIELD_KF_R1 + r - 1, adj_j = SHIELD_ADJ_R1 + r - 1, ext_j = SHIELD_EXT_R1 + r - 1;
                    mg_score += sign * kf_cnt[r - 1]  * FE_MG[kf_j];
                    eg_score += sign * kf_cnt[r - 1]  * FE_EG[kf_j];
                    mg_score += sign * adj_cnt[r - 1] * FE_MG[adj_j];
                    eg_score += sign * adj_cnt[r - 1] * FE_EG[adj_j];
                    mg_score += sign * ext_cnt[r - 1] * FE_MG[ext_j];
                    eg_score += sign * ext_cnt[r - 1] * FE_EG[ext_j];
                    if (kf_cnt[r - 1])  { t.add(kf_j,  sign * kf_cnt[r - 1]);  t.add(kf_j  + NPHASE, sign * kf_cnt[r - 1]); }
                    if (adj_cnt[r - 1]) { t.add(adj_j, sign * adj_cnt[r - 1]); t.add(adj_j + NPHASE, sign * adj_cnt[r - 1]); }
                    if (ext_cnt[r - 1]) { t.add(ext_j, sign * ext_cnt[r - 1]); t.add(ext_j + NPHASE, sign * ext_cnt[r - 1]); }
                }
                if (hole_kf)  { mg_score += sign * FE_MG[HOLE_KF];  eg_score += sign * FE_EG[HOLE_KF];  t.add(HOLE_KF, sign);  t.add(HOLE_KF + NPHASE, sign); }
                if (hole_adj) { mg_score += sign * hole_adj * FE_MG[HOLE_ADJ]; eg_score += sign * hole_adj * FE_EG[HOLE_ADJ]; t.add(HOLE_ADJ, sign * hole_adj); t.add(HOLE_ADJ + NPHASE, sign * hole_adj); }
            }

            Bitboard all_pawns = pos.pieces(PAWN);
            if (!(all_pawns & file_bb(kfile))) {
                mg_score += sign * FE_MG[OPEN_KFILE];
                t.add(OPEN_KFILE, sign);
            }
            int adj_open = 0;
            if (kfile > FILE_A && !(all_pawns & file_bb(File(kfile - 1)))) {
                mg_score += sign * FE_MG[OPEN_ADJ];
                adj_open++;
            }
            if (kfile < FILE_H && !(all_pawns & file_bb(File(kfile + 1)))) {
                mg_score += sign * FE_MG[OPEN_ADJ];
                adj_open++;
            }
            if (adj_open) t.add(OPEN_ADJ, sign * adj_open);

            int storm_sum = 0;
            for (int df = -1; df <= 1; ++df) {
                File storm_file = File(int(kfile) + df);
                if (storm_file < FILE_A || storm_file > FILE_H) continue;
                Bitboard enemy_file_pawns = their_pawns & file_bb(storm_file);
                while (enemy_file_pawns) {
                    Square psq = pop_lsb(enemy_file_pawns);
                    Rank pr = relative_rank(c, rank_of(psq));
                    if (pr >= RANK_4) {
                        mg_score += sign * (pr - 3) * FE_MG[STORM];
                        storm_sum += (pr - 3);
                    }
                }
            }
            if (storm_sum) t.add(STORM, sign * storm_sum);
        }

        bool castled = false;
        if (c == WHITE && krank == RANK_1 && (kfile == FILE_G || kfile == FILE_C)) castled = true;
        if (c == BLACK && krank == RANK_8 && (kfile == FILE_G || kfile == FILE_C)) castled = true;
        if (castled) {
            mg_score += sign * FE_MG[CASTLED]; eg_score += sign * FE_EG[CASTLED];
            t.add(CASTLED, sign); t.add(CASTLED + NPHASE, sign);
        }

        CastlingRight ks_cr = c == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE;
        CastlingRight qs_cr = c == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE;
        if (!pos.castling_allowed(c, ks_cr) && !pos.castling_allowed(c, qs_cr) && !castled) {
            mg_score += sign * FE_MG[NOCASTLE];
            eg_score += sign * FE_EG[NOCASTLE];
            t.add(NOCASTLE, sign); t.add(NOCASTLE + NPHASE, sign);
        }
        }

        // ----------------- threats -----------------
        {
            Bitboard their_pieces = pos.pieces(them);

            thr_cnt_arr[c_idx] = popcount(their_pieces & all_attacks[c]);

            Bitboard threats = their_pieces & attacks_by[c][PAWN];
            while (threats) {
                PieceType pt = piece_type_of(pos.piece_on(pop_lsb(threats)));
                if (pt < KING) {
                    mg_score += sign * etrace::pawn_threat_mg()[pt];
                    eg_score += sign * etrace::pawn_threat_eg()[pt];
                    t.add(THR + 0 * 5 + int(pt), sign);
                    t.add(THR + 0 * 5 + int(pt) + NPHASE, sign);
                }
            }
            threats = their_pieces & attacks_by[c][KNIGHT];
            while (threats) {
                PieceType pt = piece_type_of(pos.piece_on(pop_lsb(threats)));
                if (pt < KING) {
                    mg_score += sign * etrace::knight_threat_mg()[pt];
                    eg_score += sign * etrace::knight_threat_eg()[pt];
                    t.add(THR + 1 * 5 + int(pt), sign);
                    t.add(THR + 1 * 5 + int(pt) + NPHASE, sign);
                }
            }
            threats = their_pieces & attacks_by[c][BISHOP];
            while (threats) {
                PieceType pt = piece_type_of(pos.piece_on(pop_lsb(threats)));
                if (pt < KING) {
                    mg_score += sign * etrace::bishop_threat_mg()[pt];
                    eg_score += sign * etrace::bishop_threat_eg()[pt];
                    t.add(THR + 2 * 5 + int(pt), sign);
                    t.add(THR + 2 * 5 + int(pt) + NPHASE, sign);
                }
            }
            threats = their_pieces & attacks_by[c][ROOK];
            while (threats) {
                PieceType pt = piece_type_of(pos.piece_on(pop_lsb(threats)));
                if (pt < KING) {
                    mg_score += sign * etrace::rook_threat_mg()[pt];
                    eg_score += sign * etrace::rook_threat_eg()[pt];
                    t.add(THR + 3 * 5 + int(pt), sign);
                    t.add(THR + 3 * 5 + int(pt) + NPHASE, sign);
                }
            }
            threats = their_pieces & attacks_by[c][QUEEN];
            while (threats) {
                PieceType pt = piece_type_of(pos.piece_on(pop_lsb(threats)));
                if (pt < KING) {
                    mg_score += sign * etrace::queen_threat_mg()[pt];
                    eg_score += sign * etrace::queen_threat_eg()[pt];
                    t.add(THR + 4 * 5 + int(pt), sign);
                    t.add(THR + 4 * 5 + int(pt) + NPHASE, sign);
                }
            }
            Bitboard hanging_pawns = pos.pieces(them, PAWN) & ~all_attacks[them] & all_attacks[c];
            if (hanging_pawns) {
                mg_score += sign * FE_MG[HANG_PAWN] * popcount(hanging_pawns);
                eg_score += sign * FE_EG[HANG_PAWN] * popcount(hanging_pawns);
                t.add(HANG_PAWN, sign * popcount(hanging_pawns));
                t.add(HANG_PAWN + NPHASE, sign * popcount(hanging_pawns));
            }
            Bitboard hanging_pieces = (pos.pieces(them, KNIGHT) | pos.pieces(them, BISHOP)
                                     | pos.pieces(them, ROOK) | pos.pieces(them, QUEEN))
                                     & ~all_attacks[them] & all_attacks[c];
            if (hanging_pieces) {
                hang_cnt_arr[c_idx] = popcount(hanging_pieces);
                mg_score += sign * FE_MG[HANG_PIECE] * hang_cnt_arr[c_idx];
                eg_score += sign * FE_EG[HANG_PIECE] * hang_cnt_arr[c_idx];
                t.add(HANG_PIECE, sign * hang_cnt_arr[c_idx]);
                t.add(HANG_PIECE + NPHASE, sign * hang_cnt_arr[c_idx]);
            }
            {
                Square their_ksq = ksq_arr[c_idx ^ 1];
                Bitboard our_sliders = (pos.pieces(c, ROOK, QUEEN) | pos.pieces(c, BISHOP, QUEEN));
                Bitboard enemy_pinned = BB_EMPTY;
                while (our_sliders) {
                    Square sniper = pop_lsb(our_sliders);
                    Bitboard ray_sniper = (piece_type_of(pos.piece_on(sniper)) == BISHOP)
                        ? bb_diag_attacks(their_ksq, Bitboard(0)) : rook_attacks_bb(their_ksq, Bitboard(0));
                    if (!(ray_sniper & square_bb(sniper))) continue;
                    Bitboard between = between_bb(their_ksq, sniper) & occupied;
                    if (between && !more_than_one(between) && (between & pos.pieces(them))) {
                        enemy_pinned |= between;
                    }
                }
                if (enemy_pinned) {
                    int pin_count = popcount(enemy_pinned);
                    mg_score += sign * pin_count * FE_MG[PINNED];
                    eg_score += sign * pin_count * FE_EG[PINNED];
                    t.add(PINNED, sign * pin_count);
                    t.add(PINNED + NPHASE, sign * pin_count);
                }
            }
        }
    }

    // ----------------- early eval exit -----------------
    {
        int quick_phase = popcount(pos.pieces(KNIGHT)) + popcount(pos.pieces(BISHOP))
                        + popcount(pos.pieces(ROOK)) * 2 + popcount(pos.pieces(QUEEN)) * 4;
        quick_phase = std::min(24, quick_phase);
        int quick_score = (mg_score * quick_phase + eg_score * (24 - quick_phase)) / 24;
        if (quick_score > 500 || quick_score < -500) {
            int sf = scale_factor(pos, Value(eg_score));
            int eg_scaled = eg_score * sf / 32;
            int score = (mg_score * quick_phase + eg_scaled * (24 - quick_phase)) / 24;
            int tempo = (15 * quick_phase + 5 * (24 - quick_phase)) / 24;
            score += (pos.side_to_move() == WHITE) ? tempo : -tempo;
            int mirror = pos.side_to_move() == WHITE ? score : -score;
            if (getenv("ETRACE_DEBUG")) fprintf(stderr, "DBG[quick] ph=%d sf=%d mg=%d eg=%d egsc=%d tempo=%d mirror=%d\n", quick_phase, sf, t.mmg, t.meg, eg_scaled, tempo, mirror);
            out.ph = quick_phase; out.sf = sf;
            out.mirror_white = mirror;
            t.add(TEMPO_MG, (pos.side_to_move() == WHITE) ? quick_phase : -quick_phase);
            t.add(TEMPO_EG, (pos.side_to_move() == WHITE) ? (24 - quick_phase) : -(24 - quick_phase));
            return out;
        }
    }

    // ----------------- space -----------------
    {
        int space[2] = {0, 0};
        int pawn_count[2] = {0, 0};
        for (int c = 0; c < 2; ++c) {
            Bitboard our_pawns = pos.pieces(Color(c), PAWN);
            pawn_count[c] = popcount(our_pawns);
            Bitboard our_pawn_attacks = pawn_attacks_bb(Color(c), our_pawns);
            Bitboard their_pawn_attacks = pawn_attacks_bb(Color(c ^ 1), pos.pieces(Color(c ^ 1), PAWN));
            Bitboard space_area = (BB_FILE_C | BB_FILE_D | BB_FILE_E | BB_FILE_F)
                                 & (our_pawn_attacks & ~their_pawn_attacks);
            space[c] = popcount(space_area);
        }
        if (pawn_count[WHITE] > pawn_count[BLACK]) {
            mg_score += space[WHITE] * FE_MG[SPACE_OWN] + space[BLACK] * FE_MG[SPACE_OPP];
            t.add(SPACE_OWN, space[WHITE]);
            t.add(SPACE_OPP, space[BLACK]);
        } else if (pawn_count[BLACK] > pawn_count[WHITE]) {
            mg_score -= space[BLACK] * FE_MG[SPACE_OWN] + space[WHITE] * FE_MG[SPACE_OPP];
            t.add(SPACE_OWN, -space[BLACK]);
            t.add(SPACE_OPP, -space[WHITE]);
        }
    }

    // ----------------- material imbalance -----------------
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color c = Color(c_idx);
        int sign = (c_idx == 0) ? 1 : -1;
        int pawns = popcount(pos.pieces(c, PAWN));
        int bishops = bishop_count[c_idx];
        if (bishops >= 1) {
            mg_score += sign * (FE_MG[IMBAL_CONST] + pawns * FE_MG[IMBAL_PAWN]);
            eg_score += sign * (FE_EG[IMBAL_CONST] + pawns * FE_EG[IMBAL_PAWN]);
            t.add(IMBAL_CONST, sign); t.add(IMBAL_CONST + NPHASE, sign);
            t.add(IMBAL_PAWN, sign * pawns); t.add(IMBAL_PAWN + NPHASE, sign * pawns);
        }
    }

    if (bishop_count[WHITE] >= 2) {
        mg_score += FE_MG[BISHOP_PAIR]; eg_score += FE_EG[BISHOP_PAIR];
        t.add(BISHOP_PAIR, 1); t.add(BISHOP_PAIR + NPHASE, 1);
    }
    if (bishop_count[BLACK] >= 2) {
        mg_score -= FE_MG[BISHOP_PAIR]; eg_score -= FE_EG[BISHOP_PAIR];
        t.add(BISHOP_PAIR, -1); t.add(BISHOP_PAIR + NPHASE, -1);
    }

    // Imbalance polynomial (mirrors evaluation.cpp): pairwise piece-count interactions
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color c = Color(c_idx);
        int sign = (c_idx == 0) ? 1 : -1;
        int rooks = popcount(pos.pieces(c, ROOK));
        int pawns = popcount(pos.pieces(c, PAWN));
        int knights = popcount(pos.pieces(c, KNIGHT));
        int rp = rooks * pawns;
        if (rp) { mg_score += sign * rp * FE_MG[ROOK_PAWN]; eg_score += sign * rp * FE_EG[ROOK_PAWN]; }
        if (rp) { t.add(ROOK_PAWN, sign * rp); t.add(ROOK_PAWN + NPHASE, sign * rp); }
        if (knights >= 2) { mg_score += sign * FE_MG[NN_PAIR]; eg_score += sign * FE_EG[NN_PAIR]; t.add(NN_PAIR, sign); t.add(NN_PAIR + NPHASE, sign); }
        if (knights >= 1 && bishop_count[c_idx] >= 1) { mg_score += sign * FE_MG[NB_PAIR]; eg_score += sign * FE_EG[NB_PAIR]; t.add(NB_PAIR, sign); t.add(NB_PAIR + NPHASE, sign); }

        // Overloaded defenders (mirrors evaluation.cpp rung 8)
        {
            Color them = Color(c_idx ^ 1);
            Bitboard attacked_ours = pos.pieces(c) & all_attacks[them];
            attacked_ours &= ~(pos.pieces(c, PAWN) | pos.pieces(c, KING));
            int loads[64] = {0};
            Bitboard scan = attacked_ours;
            while (scan) {
                Square sq = pop_lsb(scan);
                Bitboard defs = pos.attackers_to(sq) & pos.pieces(c) & ~pos.pieces(c, KING);
                if (popcount(defs) == 1) ++loads[int(lsb(defs))];
            }
            int over2 = 0, over3 = 0;
            for (int s = 0; s < 64; ++s) {
                if (loads[s] >= 3) ++over3;
                else if (loads[s] == 2) ++over2;
            }
            if (over2) {
                mg_score -= sign * over2 * FE_MG[OVERLOAD2]; eg_score -= sign * over2 * FE_EG[OVERLOAD2];
                t.add(OVERLOAD2, -sign * over2); t.add(OVERLOAD2 + NPHASE, -sign * over2);
            }
            if (over3) {
                mg_score -= sign * over3 * FE_MG[OVERLOAD3]; eg_score -= sign * over3 * FE_EG[OVERLOAD3];
                t.add(OVERLOAD3, -sign * over3); t.add(OVERLOAD3 + NPHASE, -sign * over3);
            }
        }
    }

    {
        int w_pawns = popcount(pos.pieces(WHITE, PAWN));
        int b_pawns = popcount(pos.pieces(BLACK, PAWN));
        int total_pawns = w_pawns + b_pawns;
        int knight_adj = total_pawns - 8;
        int w_knights = popcount(pos.pieces(WHITE, KNIGHT));
        int b_knights = popcount(pos.pieces(BLACK, KNIGHT));
        if (knight_adj > 0) {
            mg_score += w_knights * knight_adj * FE_MG[KNIGHT_PAWN];
            mg_score -= b_knights * knight_adj * FE_MG[KNIGHT_PAWN];
            eg_score += w_knights * knight_adj * FE_EG[KNIGHT_PAWN];
            eg_score -= b_knights * knight_adj * FE_EG[KNIGHT_PAWN];
            t.add(KNIGHT_PAWN, (w_knights - b_knights) * knight_adj);
            t.add(KNIGHT_PAWN + NPHASE, (w_knights - b_knights) * knight_adj);
        }
    }

    {
        int w_np_mat = popcount(pos.pieces(WHITE, KNIGHT)) * PVEG[KNIGHT]
                     + popcount(pos.pieces(WHITE, BISHOP)) * PVEG[BISHOP]
                     + popcount(pos.pieces(WHITE, ROOK))   * PVEG[ROOK]
                     + popcount(pos.pieces(WHITE, QUEEN))  * PVEG[QUEEN];
        int b_np_mat = popcount(pos.pieces(BLACK, KNIGHT)) * PVEG[KNIGHT]
                     + popcount(pos.pieces(BLACK, BISHOP)) * PVEG[BISHOP]
                     + popcount(pos.pieces(BLACK, ROOK))   * PVEG[ROOK]
                     + popcount(pos.pieces(BLACK, QUEEN))  * PVEG[QUEEN];
        int nonpawn_pieces = popcount(pos.pieces(KNIGHT)) + popcount(pos.pieces(BISHOP))
                           + popcount(pos.pieces(ROOK))   + popcount(pos.pieces(QUEEN));
        int simplification = 14 - nonpawn_pieces;
        if (simplification > 0) {
            int diff = w_np_mat - b_np_mat;
            eg_score += FE_EG[TRADEDOWN] * (diff * simplification / 256);
            t.add(TRADEDOWN + NPHASE, diff * simplification / 256);
        }
    }

    // ----------------- king safety -----------------
    int ks_au_arr[2] = {0, 0};  // per-side attack units (context eval input)
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color c = Color(c_idx);
        Color them = Color(c_idx ^ 1);
        int sign = (c_idx == 0) ? 1 : -1;
        Square our_ksq = ksq_arr[c_idx];

        Bitboard king_zone = king_attacks_bb(our_ksq) | square_bb(our_ksq);
        Rank fwd = (c == WHITE) ? Rank(rank_of(our_ksq) + 1) : Rank(rank_of(our_ksq) - 1);
        if (fwd >= RANK_1 && fwd <= RANK_8) {
            for (File f = FILE_A; f <= FILE_H; f = File(f + 1)) {
                king_zone |= square_bb(make_square(f, fwd));
            }
        }

        Bitboard enemy_pieces_near = pos.pieces(them) & king_zone;
        enemy_pieces_near &= ~(pos.pieces(them, PAWN) | pos.pieces(them, KING));
        if (!enemy_pieces_near) continue;

        int attack_units = 0;
        int attacker_count = 0;

        Bitboard our_pawn_attacks = pawn_attacks_bb(c, pos.pieces(c, PAWN));
        Bitboard safe = ~our_pawn_attacks;

        Bitboard king_diag_rays = bb_diag_attacks(our_ksq, Bitboard(0));
        Bitboard king_orth_rays = rook_attacks_bb(our_ksq, Bitboard(0));

        Bitboard enemy_pawns = pos.pieces(them, PAWN);
        Bitboard ep = enemy_pawns;
        while (ep) {
            Square psq = pop_lsb(ep);
            if (pawn_attacks_bb(them, square_bb(psq)) & king_zone) {
                attack_units += 2;
                attacker_count++;
            }
        }

        Bitboard enemy_kn = pos.pieces(them, KNIGHT);
        while (enemy_kn) {
            Square ksq2 = pop_lsb(enemy_kn);
            Bitboard kn_attacks = knight_attacks_bb(ksq2);
            if (kn_attacks & king_zone) {
                attack_units += 4;
                attacker_count++;
            }
            Bitboard kn_checks = kn_attacks & king_attacks_bb(our_ksq);
            if (kn_checks & safe) attack_units += 10;
        }

        Bitboard enemy_bi = pos.pieces(them, BISHOP);
        while (enemy_bi) {
            Square bsq = pop_lsb(enemy_bi);
            Bitboard bi_attacks = bishop_attacks_bb(bsq, occupied);
            if (bi_attacks & king_zone) {
                attack_units += 3;
                attacker_count++;
            }
            Bitboard bi_checks = bi_attacks & king_diag_rays;
            if (bi_checks & safe) attack_units += 8;
        }

        Bitboard enemy_ro = pos.pieces(them, ROOK);
        while (enemy_ro) {
            Square rsq = pop_lsb(enemy_ro);
            Bitboard ro_attacks = rook_attacks_bb(rsq, occupied);
            if (ro_attacks & king_zone) {
                attack_units += 5;
                attacker_count++;
            }
            Bitboard ro_checks = ro_attacks & king_orth_rays;
            if (ro_checks & safe) attack_units += 12;
        }

        Bitboard enemy_qu = pos.pieces(them, QUEEN);
        while (enemy_qu) {
            Square qsq = pop_lsb(enemy_qu);
            Bitboard qu_attacks = queen_attacks_bb(qsq, occupied);
            if (qu_attacks & king_zone) {
                attack_units += 7;
                attacker_count++;
            }
            Bitboard qu_checks = qu_attacks & (king_diag_rays | king_orth_rays);
            if (qu_checks & safe) attack_units += 16;
        }

        if (attacker_count >= 2) {
            int au_pos = std::max(0, attack_units);
            ks_au_arr[c_idx] = au_pos;
            int au_danger_mg = (au_pos * au_pos) / 8;
            int au_danger_eg = au_pos / 2;

            if (!enemy_qu) au_danger_mg = au_danger_mg / 4;
            if (!enemy_qu && !enemy_ro) au_danger_mg = au_danger_mg / 4;

            Bitboard king_moves = king_attacks_bb(our_ksq);
            Bitboard safe_king_squares = king_moves & ~pos.pieces(c) & ~all_attacks[them];
            int flight = popcount(safe_king_squares);
            int flight1 = 0, flight2 = 0;
            int flight_mg = (flight <= 1) ? FE_MG[KS_FLIGHT1]
                          : (flight == 2) ? FE_MG[KS_FLIGHT2] : 0;
            int flight_eg = (flight <= 1) ? FE_EG[KS_FLIGHT1]
                          : (flight == 2) ? FE_EG[KS_FLIGHT2] : 0;
            if (flight <= 1) flight1 = 1;
            else if (flight == 2) flight2 = 1;

            mg_score -= sign * (FE_MG[KS_AU] * au_danger_mg + flight_mg);
            eg_score -= sign * (FE_EG[KS_AU] * au_danger_eg + flight_eg);
            t.add(KS_AU, -sign * au_danger_mg);
            t.add(KS_AU + NPHASE, -sign * au_danger_eg);
            if (flight1) { t.add(KS_FLIGHT1, -sign); t.add(KS_FLIGHT1 + NPHASE, -sign); }
            if (flight2) { t.add(KS_FLIGHT2, -sign); t.add(KS_FLIGHT2 + NPHASE, -sign); }
            // Danger-curve shape buckets (mirrors evaluation.cpp): au >= 8/16/24/32/40
            for (int b = 0; b < 5; ++b) {
                if (au_pos >= 8 * (b + 1)) {
                    mg_score -= sign * FE_MG[KS_AUB8 + b];
                    eg_score -= sign * FE_EG[KS_AUB8 + b];
                    t.add(KS_AUB8 + b, -sign);
                    t.add(KS_AUB8 + b + NPHASE, -sign);
                }
            }

            // Razor-class interactions (mirrors evaluation.cpp E2b)
            {
                int att = c_idx ^ 1;
                int au_b = std::min(au_pos, 48) / 16;
                if (au_b > 0) {
                    int hang_v = hang_cnt_arr[att] * au_b;
                    int thr_v  = thr_cnt_arr[att] * au_b;
                    int margin = 0;
                    for (int pt = PAWN; pt <= QUEEN; ++pt) {
                        margin += PVMG[pt]
                            * (popcount(pos.pieces(them, PieceType(pt)))
                             - popcount(pos.pieces(c, PieceType(pt))));
                    }
                    margin = std::max(-400, std::min(400, margin));
                    int mat_v = margin / 100 * au_b;
                    if (hang_v) {
                        mg_score -= sign * FE_MG[RAZOR_HANGAU] * hang_v;
                        eg_score -= sign * FE_EG[RAZOR_HANGAU] * hang_v;
                        t.add(RAZOR_HANGAU, -sign * hang_v);
                        t.add(RAZOR_HANGAU + NPHASE, -sign * hang_v);
                    }
                    if (thr_v) {
                        mg_score -= sign * FE_MG[RAZOR_THRAU] * thr_v;
                        eg_score -= sign * FE_EG[RAZOR_THRAU] * thr_v;
                        t.add(RAZOR_THRAU, -sign * thr_v);
                        t.add(RAZOR_THRAU + NPHASE, -sign * thr_v);
                    }
                    if (mat_v) {
                        mg_score -= sign * FE_MG[RAZOR_MATAU] * mat_v;
                        eg_score -= sign * FE_EG[RAZOR_MATAU] * mat_v;
                        t.add(RAZOR_MATAU, -sign * mat_v);
                        t.add(RAZOR_MATAU + NPHASE, -sign * mat_v);
                    }
                }
            }
        }
    }

    // Context eval (mirrors evaluation.cpp rung 5a): volatility x margin.
    out.tension = std::min(60, ks_au_arr[0] + ks_au_arr[1]);
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color c = Color(c_idx);
        int sign = (c_idx == 0) ? 1 : -1;
        Color them = Color(c_idx ^ 1);
        int mat_margin = 0;
        for (int pt = PAWN; pt <= QUEEN; ++pt) {
            mat_margin += PVMG[pt]
                * (popcount(pos.pieces(c, PieceType(pt))) - popcount(pos.pieces(them, PieceType(pt))));
        }
        int margin_term = std::max(-600, std::min(600, mat_margin));
        // Per-side danger (mirrors evaluation.cpp): ahead -> own king's au;
        // behind -> opponent's au.
        int au_relevant = (margin_term > 0) ? ks_au_arr[c_idx] : ks_au_arr[c_idx ^ 1];
        if (margin_term == 0 || au_relevant == 0) continue;
        int v = std::min(60, au_relevant) * margin_term / 900;  // |v| <= 40
        if (getenv("VOLCTX_DEBUG")) fprintf(stderr, "VOLCTX m=%d au=%d v=%d\n", margin_term, au_relevant, v);
        mg_score -= sign * FE_MG[VOL_CTX] * v;
        t.add(VOL_CTX, -sign * v);
    }

    // ----------------- blend -----------------
    int phase = popcount(pos.pieces(KNIGHT)) + popcount(pos.pieces(BISHOP))
              + popcount(pos.pieces(ROOK)) * 2 + popcount(pos.pieces(QUEEN)) * 4;
    phase = std::min(24, phase);

    int sf = scale_factor(pos, Value(eg_score));
    int eg_scaled = eg_score * sf / 32;
    int score = (mg_score * phase + eg_scaled * (24 - phase)) / 24;

    int tempo = (15 * phase + 5 * (24 - phase)) / 24;
    score += (pos.side_to_move() == WHITE) ? tempo : -tempo;

    int mirror = pos.side_to_move() == WHITE ? score : -score;
    if (getenv("ETRACE_DEBUG")) fprintf(stderr, "RAW N=%d B=%d R=%d Q=%d | ", popcount(pos.pieces(KNIGHT)), popcount(pos.pieces(BISHOP)), popcount(pos.pieces(ROOK)), popcount(pos.pieces(QUEEN)));
    if (getenv("ETRACE_DEBUG")) fprintf(stderr, "DBG ph=%d sf=%d mg=%d eg=%d egsc=%d tempo=%d mirror=%d\n", phase, sf, mg_score, eg_score, eg_scaled, tempo, mirror);
    out.ph = phase; out.sf = sf;
    out.mirror_white = mirror;
    t.add(TEMPO_MG, (pos.side_to_move() == WHITE) ? phase : -phase);
    t.add(TEMPO_EG, (pos.side_to_move() == WHITE) ? (24 - phase) : -(24 - phase));
    return out;
}

// ============================================================================
// Current-engine coefficients (for --verify reconstruction). Every value is
// read from the engine's own tables/params via etrace/g_eval_params so the
// check can never drift from the engine.
// ============================================================================
static double CMG[feat::NPHASE];
static double CEG[feat::NPHASE];

static void init_coefs() {
    // FE_MG/FE_EG ARE the engine's current coefficients (evaluation.cpp reads
    // them directly), so the reconstruction check compares against the same
    // numbers the engine uses — by construction, no drift possible.
    for (int i = 0; i < feat::NPHASE; ++i) {
        CMG[i] = FE_MG[i];
        CEG[i] = FE_EG[i];
    }
}

// White-perspective reconstruction: dot(features, coefs) blended like the engine.
// Returns the WHITE-perspective eval; compare against mirror * stm_sign.
static double reconstruct(const Tracer& t, int ph, int sf) {
    double mg = 0.0, eg = 0.0;
    for (int i = 0; i < t.nt; ++i) {
        int idx = t.touched[i];
        if (idx < feat::NPHASE)             mg += CMG[idx] * t.f[idx];
        else if (idx < 2 * feat::NPHASE)    eg += CEG[idx - feat::NPHASE] * t.f[idx];
    }
    double eg_scaled = std::trunc(eg * sf / 32.0);
    double score = (mg * ph + eg_scaled * (24 - ph)) / 24.0;
    double tempo = (15.0 * t.f[feat::TEMPO_MG] + 5.0 * t.f[feat::TEMPO_EG]) / 24.0;
    if (getenv("ETRACE_DEBUG")) fprintf(stderr, "DBG[rec] mg_rec=%.1f eg_rec=%.1f egsc=%.1f tempo=%.1f total=%.1f\n", mg, eg, eg_scaled, tempo, score + tempo);
    return score + tempo;
}

} // namespace luminex

int main(int argc, char** argv) {
    bool verify = false, dump_coefs = false, solve = false, score = false, texel = false;
    bool dump_rows = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--verify") == 0) verify = true;
        if (std::strcmp(argv[i], "--dump-coefs") == 0) dump_coefs = true;
        if (std::strcmp(argv[i], "--solve") == 0) solve = true;
        if (std::strcmp(argv[i], "--score") == 0) score = true;
        if (std::strcmp(argv[i], "--texel") == 0) texel = true;
        if (std::strcmp(argv[i], "--dump-rows") == 0) dump_rows = true;
    }

    luminex::init_magic_bitboards();   // attack tables (set()/eval depend on these)
    luminex::init_zobrist();
    luminex::init_evaluation();
    luminex::init_coefs();

    if (dump_coefs) {
        // solver-space single coefs: line j = "j value" with j<607 -> CMG[j], j>=607 -> CEG[j-607]
        for (int j = 0; j < luminex::NSOLVE; ++j)
            std::printf("%d %.6f\n", j, j < luminex::feat::NPHASE ? luminex::CMG[j]
                                                              : luminex::CEG[j - luminex::feat::NPHASE]);
        return 0;
    }

    if (texel) {
        const char* c0p = std::getenv("NNUE_TEXEL_C0");
        const char* outp = nullptr;
        for (int i = 1; i < argc; ++i)
            if (std::strcmp(argv[i], "--texel") != 0) outp = argv[i];
        if (!c0p || !outp) {
            std::fprintf(stderr, "usage: NNUE_TEXEL_C0=<coefs> %s --texel <out.coefs> < rows\n", argv[0]);
            return 2;
        }
        return luminex::texel_train(outp, c0p);
    }

    luminex::Position pos;
    std::string line, fen;
    long long n = 0, gated = 0, bad_fen = 0;
    long long mirror_mismatch = 0, rec_fail = 0;
    long long dr_n = 0;
    double max_rec_diff = 0.0, sum_rec_diff = 0.0;

    // --score state: score fen\tstm\ttarget rows under candidate coefs (arg 1) AND the
    // current-engine coefs (baseline) in one pass. Prints R^2 / RMSE / MAE for both.
    double sc_cand[luminex::NSOLVE], sc_base[luminex::NSOLVE];
    bool sc_ok = false;
    long long sc_n = 0;
    double sc_ss_tot = 0, sc_ssr_c = 0, sc_ssr_b = 0, sc_abs_c = 0, sc_abs_b = 0, sc_sum_r = 0;
    if (score) {
        const char* cf = std::getenv("NNUE_SCORE_COEFS");
        if (!cf) { std::fprintf(stderr, "--score needs NNUE_SCORE_COEFS=<coefs file>\n"); return 2; }
        FILE* f = std::fopen(cf, "r");
        if (!f) { std::fprintf(stderr, "cannot read %s\n", cf); return 2; }
        for (int i = 0; i < luminex::NSOLVE; ++i)
            if (std::fscanf(f, "%*d %lf", &sc_cand[i]) != 1) {
                std::fprintf(stderr, "coefs file too short at %d\n", i); return 2;
            }
        std::fclose(f);
        for (int i = 0; i < luminex::feat::NPHASE; ++i) {
            sc_base[i] = luminex::CMG[i];
            sc_base[i + luminex::feat::NPHASE] = luminex::CEG[i];
        }
        if (getenv("ETRACE_DEBUG")) {
            double mx = 0; int amx = 0;
            for (int i = 0; i < luminex::NSOLVE; ++i) {
                double d = sc_base[i] - sc_cand[i];
                if (std::fabs(d) > mx) { mx = std::fabs(d); amx = i; }
            }
            std::fprintf(stderr, "DBG[score-fill] cand[0]=%.3f base[0]=%.3f cand[607]=%.3f base[607]=%.3f maxdiff=%.4f at %d\n",
                         sc_cand[0], sc_base[0], sc_cand[607], sc_base[607], mx, amx);
        }
        sc_ok = true;
    }

    // --solve state: normal-equation accumulator + holdout stream (1-in-K by FEN hash).
    // sidx/sxv must hold ANY row's nonzero count: worst case ~NFEAT (every touched
    // feature distinct) — 256 overflowed silently on complex positions.
    // NNUE_TARGET_MAX=cp drops rows whose white-perspective target exceeds +-cp (the
    // gamepack eval tail runs to +-9096; a handful of near-mate targets would
    // otherwise dominate the squared error of a cp-space linear fit).
    luminex::SolverAcc acc;
    FILE* solve_out = nullptr;
    FILE* holdout = nullptr;
    long long n_hold = 0;
    double tmax = 1e18;
    if (const char* tm = std::getenv("NNUE_TARGET_MAX")) tmax = std::atof(tm);
    long long n_tmax = 0;
    int sidx[luminex::feat::NFEAT];
    double sxv[luminex::feat::NFEAT];
    if (solve) {
        const char* so = std::getenv("NNUE_SOLVE_OUT");
        if (!so) { std::fprintf(stderr, "--solve needs NNUE_SOLVE_OUT=<prefix>\n"); return 2; }
        std::string bin = std::string(so) + ".bin";
        solve_out = std::fopen(bin.c_str(), "wb");
        if (!solve_out) { std::fprintf(stderr, "cannot write %s\n", bin.c_str()); return 2; }
        const char* hf = std::getenv("NNUE_SOLVE_HOLDOUT");
        if (hf) {
            holdout = std::fopen(hf, "w");
            if (!holdout) { std::fprintf(stderr, "cannot write %s\n", hf); return 2; }
        }
    }

    while (std::getline(std::cin, line)) {
        size_t tab = line.find('\t');
        fen = (tab == std::string::npos) ? line : line.substr(0, tab);
        while (!fen.empty() && (fen.back() == '\r' || fen.back() == '\n')) fen.pop_back();
        if (fen.empty()) continue;

        // Cheap validity guard: a FEN has >= 4 space-separated fields. (Built
        // with -fno-exceptions; set() on garbage is UB, so pre-filter instead.)
        int fields = 1;
        for (char ch : fen) if (ch == ' ') fields++;
        if (fields < 4) { bad_fen++; continue; }
        pos.set(fen);

        luminex::Tracer t;
        t.reset();
        luminex::TraceOut out = luminex::trace_eval(pos, t);
        n++;

        // Echo the target field (if present) so trace lines are self-contained for the
        // solver: "<ph> <sf> <stm> <mirror> <target> idx:val ...". Dump format is
        // "fen\tstm\ttarget" -> target is the 3rd field; a lone "fen\tX" echo X.
        std::string target = "NA";
        if (tab != std::string::npos) {
            size_t t2 = line.find('\t', tab + 1);
            size_t t3 = (t2 == std::string::npos) ? std::string::npos : line.find('\t', t2 + 1);
            if (t2 != std::string::npos && t3 != std::string::npos)
                target = line.substr(t2 + 1, t3 - t2 - 1);          // fen\tstm\tTARGET\t...
            else if (t2 != std::string::npos)
                target = line.substr(t2 + 1);                        // fen\tstm\tTARGET(eol)
            else
                target = line.substr(tab + 1);                       // fen\tTARGET
        }

        if (!out.linear) {
            gated++;
            if (!verify && !solve) std::printf("%d %d %d G %s\n", out.ph, out.sf, out.stm, target.c_str());
            continue;
        }

        // --dump-rows: emit one line per position, "y ph sf idx:val ...", for
        // offline model-class experiments (GBM oracle vs linear). y is the
        // white-perspective target after the NNUE_TARGET_MAX clip; idx runs
        // over the raw feature space 0..2*NPHASE-1 (MG+EG blocks, TEMPO
        // excluded); val is the raw feature value (no ph/sf weighting).
        if (dump_rows) {
            double tgt = (target != "NA") ? std::strtod(target.c_str(), nullptr) : 0.0;
            double y_white = out.stm ? tgt : -tgt;
            if (std::fabs(y_white) > tmax) { n_tmax++; continue; }
            std::printf("%.1f %d %d", y_white, out.ph, out.sf);
            for (int i = 0; i < t.nt; ++i) {
                int idx = t.touched[i];
                if (idx >= luminex::NSOLVE) continue;   // TEMPO features
                if (t.f[idx] == 0) continue;
                std::printf(" %d:%d", idx, t.f[idx]);
            }
            std::printf("\n");
            dr_n++;
            if (dr_n % 1000000 == 0)
                std::fprintf(stderr, "  ... %lldM rows dumped (%lld clipped)\n", dr_n / 1000000, n_tmax);
            continue;
        }

        if (score) {
            double tgt = (target != "NA") ? std::strtod(target.c_str(), nullptr) : 0.0;
            double y_white = out.stm ? tgt : -tgt;
            if (std::fabs(y_white) > tmax) { n_tmax++; continue; }
            double tempo = (15.0 * t.f[luminex::feat::TEMPO_MG] + 5.0 * t.f[luminex::feat::TEMPO_EG]) / 24.0;
            double r = y_white - tempo;
            double wmg = out.ph / 24.0;
            double weg = (24 - out.ph) * out.sf / 768.0;
            double pc = 0.0, pb = 0.0;
            for (int i = 0; i < t.nt; ++i) {
                int idx = t.touched[i];
                int v = t.f[idx];
                if (v == 0) continue;
                // TEMPO features (idx >= 2*NPHASE) are accounted explicitly via the
                // tempo term above and have no solver slot -- scoring them read past
                // the end of sc_cand/sc_base (stack garbage, silently poisoned R2).
                if (idx < luminex::feat::NPHASE) { pc += sc_cand[idx] * v * wmg; pb += sc_base[idx] * v * wmg; }
                else if (idx < luminex::NSOLVE)  { pc += sc_cand[idx] * v * weg; pb += sc_base[idx] * v * weg; }
            }
            sc_n++;
            sc_sum_r += r;
            sc_ss_tot += r * r;
            sc_ssr_c += (r - pc) * (r - pc); sc_abs_c += std::fabs(r - pc);
            sc_ssr_b += (r - pb) * (r - pb); sc_abs_b += std::fabs(r - pb);
            continue;
        }

        if (solve) {
            // holdout: 1-in-10 by FEN hash, written verbatim for --score passes
            if (holdout) {
                uint32_t h = 2166136261u;
                for (char ch : fen) { h ^= (uint8_t)ch; h *= 16777619u; }
                if (h % 10 == 0) { std::fprintf(holdout, "%s\n", line.c_str()); n_hold++; continue; }
            }
            double tgt = (target != "NA") ? std::strtod(target.c_str(), nullptr) : 0.0;
            double y_white = out.stm ? tgt : -tgt;
            if (std::fabs(y_white) > tmax) { n_tmax++; continue; }
            double tempo = (15.0 * t.f[luminex::feat::TEMPO_MG] + 5.0 * t.f[luminex::feat::TEMPO_EG]) / 24.0;
            int snz = 0;
            acc.add(t, out.ph, out.sf, y_white - tempo, sidx, sxv, snz);
            continue;
        }

        if (verify) {
            luminex::Value eng = luminex::evaluate(pos, false);
            if (eng != out.mirror_white) {
                mirror_mismatch++;
                if (mirror_mismatch <= 10)
                    std::fprintf(stderr, "MIRROR MISMATCH fen=%s engine=%d mirror=%d\n",
                                 fen.c_str(), int(eng), out.mirror_white);
            }
            int stm_sign = out.stm ? 1 : -1;
            double rec = luminex::reconstruct(t, out.ph, out.sf);
            double diff = std::fabs(rec - double(out.mirror_white) * stm_sign);
            sum_rec_diff += diff;
            if (diff > max_rec_diff) max_rec_diff = diff;
            if (diff > 2.0) {
                rec_fail++;
                if (rec_fail <= 10)
                    std::fprintf(stderr, "RECONSTRUCT FAIL fen=%s diff=%.2f mirror=%d rec=%.1f\n",
                                 fen.c_str(), diff, out.mirror_white, rec);
            }
        } else {
            std::sort(t.touched, t.touched + t.nt);
            std::printf("%d %d %d %d %s", out.ph, out.sf, out.stm, out.mirror_white, target.c_str());
            for (int i = 0; i < t.nt; ++i) {
                int idx = t.touched[i];
                if (t.f[idx] == 0) continue;   // net-zero features carry no info
                std::printf(" %d:%d", idx, t.f[idx]);
            }
            std::printf("\n");
        }

        if (verify && n % 1000000 == 0)
            std::fprintf(stderr, "  ... %lldM pos: mirror_mismatch=%lld rec_fail=%lld max_diff=%.2f\n",
                         n / 1000000, mirror_mismatch, rec_fail, max_rec_diff);
    }

    if (score) {
        double mean_r = sc_sum_r / std::max(1LL, sc_n);
        double var_r = sc_ss_tot / std::max(1LL, sc_n) - mean_r * mean_r;
        double r2_c = 1.0 - (sc_ssr_c / std::max(1LL, sc_n)) / std::max(1e-9, var_r);
        double r2_b = 1.0 - (sc_ssr_b / std::max(1LL, sc_n)) / std::max(1e-9, var_r);
        std::fprintf(stderr, "SCORE %lld rows (dropped %lld >|%.0f|cp) | cand: R2=%.4f RMSE=%.2f MAE=%.2f | base: R2=%.4f RMSE=%.2f MAE=%.2f\n",
                     sc_n, n_tmax, tmax, r2_c, std::sqrt(sc_ssr_c / std::max(1LL, sc_n)), sc_abs_c / std::max(1LL, sc_n),
                     r2_b, std::sqrt(sc_ssr_b / std::max(1LL, sc_n)), sc_abs_b / std::max(1LL, sc_n));
        return 0;
    }

    if (solve) {
        // mirror A to full symmetric (accumulated upper triangle only)
        constexpr int NS_ = luminex::NSOLVE;
        for (int j = 0; j < NS_; ++j)
            for (int k = 0; k < j; ++k)
                acc.A[(size_t)j * NS_ + k] = acc.A[(size_t)k * NS_ + j];
        std::fwrite(&acc.n, sizeof(long long), 1, solve_out);
        std::fwrite(acc.A.data(), sizeof(double), (size_t)NS_ * NS_, solve_out);
        std::fwrite(acc.b.data(), sizeof(double), NS_, solve_out);
        std::fwrite(&acc.sum_r, sizeof(double), 1, solve_out);    // tail v2: target stats
        std::fwrite(&acc.sum_r2, sizeof(double), 1, solve_out);
        std::fclose(solve_out);
        if (holdout) std::fclose(holdout);
        std::fprintf(stderr, "SOLVE: %lld rows accumulated, %lld held out, %lld dropped >|%.0f|cp\n",
                     acc.n, n_hold, n_tmax, tmax);
        return 0;
    }

    if (verify) {
        std::fprintf(stderr, "VERIFY: %lld pos (%lld gated, %lld bad fen)\n", n, gated, bad_fen);
        std::fprintf(stderr, "  mirror==evaluate : %lld mismatches\n", mirror_mismatch);
        std::fprintf(stderr, "  reconstruction   : %lld fails (>2cp), max diff %.2f, mean %.3f\n",
                     rec_fail, max_rec_diff, sum_rec_diff / std::max(1LL, n - gated));
        return (mirror_mismatch == 0 && rec_fail == 0) ? 0 : 1;
    }

    if (dump_rows) {
        std::fprintf(stderr, "DUMP-ROWS: %lld rows written, %lld clipped >|%.0f|cp, %lld gated, %lld bad fen\n",
                     dr_n, n_tmax, tmax, gated, bad_fen);
        return 0;
    }
    return 0;
}

#elif defined(LUMINEX_TOOL_LC0PACK)

// luminex-lc0pack — Leela v6 training chunks (.tar of .gz) -> Luminex GAMEPACK frame.
// Byte-compatible with luminex-encode output (featurize --stream consumes unchanged);
// format v2 prepends the GP2 magic and adds a 10th entry byte: the game result
// (white-relative int8 from result_q, taken from the final record) for WDL-mix
// training. v1 files (no magic, 9-byte entries) still featurize — result reads 0.
//
// Per game (one .gz chunk = one game, records in ply order):
//   - start FEN reconstructed from record 0 (planes + castling/ep/50 fields)
//   - moves decoded from played_idx (policy index, un-transformed)
//   - evals from root_q (stm-rel Q) -> white-rel cp via 800*atanh, quant 8
//   - replayed with our movegen; the played move's index in the sorted legal
//     list is the mv byte (same convention featurize --stream decodes)
// Games whose replay diverges (960-castling edge cases, decode mismatch) are
// skipped and counted — packer and trainer consume the SAME FEN, so any game
// that packs is guaranteed to featurize cleanly.
#include "luminex.h"
// Generated from LeelaChessZero/lc0 src/neural/encoder.cc kMoveStrs (policy-index order).
// Each entry: from(2) to(2) [promo(1)]. Index in this array == policy/NN index.
static const char* kNNMoves[1858] = {
    "a1b1", "a1c1", "a1d1", "a1e1", "a1f1", "a1g1", "a1h1", "a1a2",
    "a1b2", "a1c2", "a1a3", "a1b3", "a1c3", "a1a4", "a1d4", "a1a5",
    "a1e5", "a1a6", "a1f6", "a1a7", "a1g7", "a1a8", "a1h8", "b1a1",
    "b1c1", "b1d1", "b1e1", "b1f1", "b1g1", "b1h1", "b1a2", "b1b2",
    "b1c2", "b1d2", "b1a3", "b1b3", "b1c3", "b1d3", "b1b4", "b1e4",
    "b1b5", "b1f5", "b1b6", "b1g6", "b1b7", "b1h7", "b1b8", "c1a1",
    "c1b1", "c1d1", "c1e1", "c1f1", "c1g1", "c1h1", "c1a2", "c1b2",
    "c1c2", "c1d2", "c1e2", "c1a3", "c1b3", "c1c3", "c1d3", "c1e3",
    "c1c4", "c1f4", "c1c5", "c1g5", "c1c6", "c1h6", "c1c7", "c1c8",
    "d1a1", "d1b1", "d1c1", "d1e1", "d1f1", "d1g1", "d1h1", "d1b2",
    "d1c2", "d1d2", "d1e2", "d1f2", "d1b3", "d1c3", "d1d3", "d1e3",
    "d1f3", "d1a4", "d1d4", "d1g4", "d1d5", "d1h5", "d1d6", "d1d7",
    "d1d8", "e1a1", "e1b1", "e1c1", "e1d1", "e1f1", "e1g1", "e1h1",
    "e1c2", "e1d2", "e1e2", "e1f2", "e1g2", "e1c3", "e1d3", "e1e3",
    "e1f3", "e1g3", "e1b4", "e1e4", "e1h4", "e1a5", "e1e5", "e1e6",
    "e1e7", "e1e8", "f1a1", "f1b1", "f1c1", "f1d1", "f1e1", "f1g1",
    "f1h1", "f1d2", "f1e2", "f1f2", "f1g2", "f1h2", "f1d3", "f1e3",
    "f1f3", "f1g3", "f1h3", "f1c4", "f1f4", "f1b5", "f1f5", "f1a6",
    "f1f6", "f1f7", "f1f8", "g1a1", "g1b1", "g1c1", "g1d1", "g1e1",
    "g1f1", "g1h1", "g1e2", "g1f2", "g1g2", "g1h2", "g1e3", "g1f3",
    "g1g3", "g1h3", "g1d4", "g1g4", "g1c5", "g1g5", "g1b6", "g1g6",
    "g1a7", "g1g7", "g1g8", "h1a1", "h1b1", "h1c1", "h1d1", "h1e1",
    "h1f1", "h1g1", "h1f2", "h1g2", "h1h2", "h1f3", "h1g3", "h1h3",
    "h1e4", "h1h4", "h1d5", "h1h5", "h1c6", "h1h6", "h1b7", "h1h7",
    "h1a8", "h1h8", "a2a1", "a2b1", "a2c1", "a2b2", "a2c2", "a2d2",
    "a2e2", "a2f2", "a2g2", "a2h2", "a2a3", "a2b3", "a2c3", "a2a4",
    "a2b4", "a2c4", "a2a5", "a2d5", "a2a6", "a2e6", "a2a7", "a2f7",
    "a2a8", "a2g8", "b2a1", "b2b1", "b2c1", "b2d1", "b2a2", "b2c2",
    "b2d2", "b2e2", "b2f2", "b2g2", "b2h2", "b2a3", "b2b3", "b2c3",
    "b2d3", "b2a4", "b2b4", "b2c4", "b2d4", "b2b5", "b2e5", "b2b6",
    "b2f6", "b2b7", "b2g7", "b2b8", "b2h8", "c2a1", "c2b1", "c2c1",
    "c2d1", "c2e1", "c2a2", "c2b2", "c2d2", "c2e2", "c2f2", "c2g2",
    "c2h2", "c2a3", "c2b3", "c2c3", "c2d3", "c2e3", "c2a4", "c2b4",
    "c2c4", "c2d4", "c2e4", "c2c5", "c2f5", "c2c6", "c2g6", "c2c7",
    "c2h7", "c2c8", "d2b1", "d2c1", "d2d1", "d2e1", "d2f1", "d2a2",
    "d2b2", "d2c2", "d2e2", "d2f2", "d2g2", "d2h2", "d2b3", "d2c3",
    "d2d3", "d2e3", "d2f3", "d2b4", "d2c4", "d2d4", "d2e4", "d2f4",
    "d2a5", "d2d5", "d2g5", "d2d6", "d2h6", "d2d7", "d2d8", "e2c1",
    "e2d1", "e2e1", "e2f1", "e2g1", "e2a2", "e2b2", "e2c2", "e2d2",
    "e2f2", "e2g2", "e2h2", "e2c3", "e2d3", "e2e3", "e2f3", "e2g3",
    "e2c4", "e2d4", "e2e4", "e2f4", "e2g4", "e2b5", "e2e5", "e2h5",
    "e2a6", "e2e6", "e2e7", "e2e8", "f2d1", "f2e1", "f2f1", "f2g1",
    "f2h1", "f2a2", "f2b2", "f2c2", "f2d2", "f2e2", "f2g2", "f2h2",
    "f2d3", "f2e3", "f2f3", "f2g3", "f2h3", "f2d4", "f2e4", "f2f4",
    "f2g4", "f2h4", "f2c5", "f2f5", "f2b6", "f2f6", "f2a7", "f2f7",
    "f2f8", "g2e1", "g2f1", "g2g1", "g2h1", "g2a2", "g2b2", "g2c2",
    "g2d2", "g2e2", "g2f2", "g2h2", "g2e3", "g2f3", "g2g3", "g2h3",
    "g2e4", "g2f4", "g2g4", "g2h4", "g2d5", "g2g5", "g2c6", "g2g6",
    "g2b7", "g2g7", "g2a8", "g2g8", "h2f1", "h2g1", "h2h1", "h2a2",
    "h2b2", "h2c2", "h2d2", "h2e2", "h2f2", "h2g2", "h2f3", "h2g3",
    "h2h3", "h2f4", "h2g4", "h2h4", "h2e5", "h2h5", "h2d6", "h2h6",
    "h2c7", "h2h7", "h2b8", "h2h8", "a3a1", "a3b1", "a3c1", "a3a2",
    "a3b2", "a3c2", "a3b3", "a3c3", "a3d3", "a3e3", "a3f3", "a3g3",
    "a3h3", "a3a4", "a3b4", "a3c4", "a3a5", "a3b5", "a3c5", "a3a6",
    "a3d6", "a3a7", "a3e7", "a3a8", "a3f8", "b3a1", "b3b1", "b3c1",
    "b3d1", "b3a2", "b3b2", "b3c2", "b3d2", "b3a3", "b3c3", "b3d3",
    "b3e3", "b3f3", "b3g3", "b3h3", "b3a4", "b3b4", "b3c4", "b3d4",
    "b3a5", "b3b5", "b3c5", "b3d5", "b3b6", "b3e6", "b3b7", "b3f7",
    "b3b8", "b3g8", "c3a1", "c3b1", "c3c1", "c3d1", "c3e1", "c3a2",
    "c3b2", "c3c2", "c3d2", "c3e2", "c3a3", "c3b3", "c3d3", "c3e3",
    "c3f3", "c3g3", "c3h3", "c3a4", "c3b4", "c3c4", "c3d4", "c3e4",
    "c3a5", "c3b5", "c3c5", "c3d5", "c3e5", "c3c6", "c3f6", "c3c7",
    "c3g7", "c3c8", "c3h8", "d3b1", "d3c1", "d3d1", "d3e1", "d3f1",
    "d3b2", "d3c2", "d3d2", "d3e2", "d3f2", "d3a3", "d3b3", "d3c3",
    "d3e3", "d3f3", "d3g3", "d3h3", "d3b4", "d3c4", "d3d4", "d3e4",
    "d3f4", "d3b5", "d3c5", "d3d5", "d3e5", "d3f5", "d3a6", "d3d6",
    "d3g6", "d3d7", "d3h7", "d3d8", "e3c1", "e3d1", "e3e1", "e3f1",
    "e3g1", "e3c2", "e3d2", "e3e2", "e3f2", "e3g2", "e3a3", "e3b3",
    "e3c3", "e3d3", "e3f3", "e3g3", "e3h3", "e3c4", "e3d4", "e3e4",
    "e3f4", "e3g4", "e3c5", "e3d5", "e3e5", "e3f5", "e3g5", "e3b6",
    "e3e6", "e3h6", "e3a7", "e3e7", "e3e8", "f3d1", "f3e1", "f3f1",
    "f3g1", "f3h1", "f3d2", "f3e2", "f3f2", "f3g2", "f3h2", "f3a3",
    "f3b3", "f3c3", "f3d3", "f3e3", "f3g3", "f3h3", "f3d4", "f3e4",
    "f3f4", "f3g4", "f3h4", "f3d5", "f3e5", "f3f5", "f3g5", "f3h5",
    "f3c6", "f3f6", "f3b7", "f3f7", "f3a8", "f3f8", "g3e1", "g3f1",
    "g3g1", "g3h1", "g3e2", "g3f2", "g3g2", "g3h2", "g3a3", "g3b3",
    "g3c3", "g3d3", "g3e3", "g3f3", "g3h3", "g3e4", "g3f4", "g3g4",
    "g3h4", "g3e5", "g3f5", "g3g5", "g3h5", "g3d6", "g3g6", "g3c7",
    "g3g7", "g3b8", "g3g8", "h3f1", "h3g1", "h3h1", "h3f2", "h3g2",
    "h3h2", "h3a3", "h3b3", "h3c3", "h3d3", "h3e3", "h3f3", "h3g3",
    "h3f4", "h3g4", "h3h4", "h3f5", "h3g5", "h3h5", "h3e6", "h3h6",
    "h3d7", "h3h7", "h3c8", "h3h8", "a4a1", "a4d1", "a4a2", "a4b2",
    "a4c2", "a4a3", "a4b3", "a4c3", "a4b4", "a4c4", "a4d4", "a4e4",
    "a4f4", "a4g4", "a4h4", "a4a5", "a4b5", "a4c5", "a4a6", "a4b6",
    "a4c6", "a4a7", "a4d7", "a4a8", "a4e8", "b4b1", "b4e1", "b4a2",
    "b4b2", "b4c2", "b4d2", "b4a3", "b4b3", "b4c3", "b4d3", "b4a4",
    "b4c4", "b4d4", "b4e4", "b4f4", "b4g4", "b4h4", "b4a5", "b4b5",
    "b4c5", "b4d5", "b4a6", "b4b6", "b4c6", "b4d6", "b4b7", "b4e7",
    "b4b8", "b4f8", "c4c1", "c4f1", "c4a2", "c4b2", "c4c2", "c4d2",
    "c4e2", "c4a3", "c4b3", "c4c3", "c4d3", "c4e3", "c4a4", "c4b4",
    "c4d4", "c4e4", "c4f4", "c4g4", "c4h4", "c4a5", "c4b5", "c4c5",
    "c4d5", "c4e5", "c4a6", "c4b6", "c4c6", "c4d6", "c4e6", "c4c7",
    "c4f7", "c4c8", "c4g8", "d4a1", "d4d1", "d4g1", "d4b2", "d4c2",
    "d4d2", "d4e2", "d4f2", "d4b3", "d4c3", "d4d3", "d4e3", "d4f3",
    "d4a4", "d4b4", "d4c4", "d4e4", "d4f4", "d4g4", "d4h4", "d4b5",
    "d4c5", "d4d5", "d4e5", "d4f5", "d4b6", "d4c6", "d4d6", "d4e6",
    "d4f6", "d4a7", "d4d7", "d4g7", "d4d8", "d4h8", "e4b1", "e4e1",
    "e4h1", "e4c2", "e4d2", "e4e2", "e4f2", "e4g2", "e4c3", "e4d3",
    "e4e3", "e4f3", "e4g3", "e4a4", "e4b4", "e4c4", "e4d4", "e4f4",
    "e4g4", "e4h4", "e4c5", "e4d5", "e4e5", "e4f5", "e4g5", "e4c6",
    "e4d6", "e4e6", "e4f6", "e4g6", "e4b7", "e4e7", "e4h7", "e4a8",
    "e4e8", "f4c1", "f4f1", "f4d2", "f4e2", "f4f2", "f4g2", "f4h2",
    "f4d3", "f4e3", "f4f3", "f4g3", "f4h3", "f4a4", "f4b4", "f4c4",
    "f4d4", "f4e4", "f4g4", "f4h4", "f4d5", "f4e5", "f4f5", "f4g5",
    "f4h5", "f4d6", "f4e6", "f4f6", "f4g6", "f4h6", "f4c7", "f4f7",
    "f4b8", "f4f8", "g4d1", "g4g1", "g4e2", "g4f2", "g4g2", "g4h2",
    "g4e3", "g4f3", "g4g3", "g4h3", "g4a4", "g4b4", "g4c4", "g4d4",
    "g4e4", "g4f4", "g4h4", "g4e5", "g4f5", "g4g5", "g4h5", "g4e6",
    "g4f6", "g4g6", "g4h6", "g4d7", "g4g7", "g4c8", "g4g8", "h4e1",
    "h4h1", "h4f2", "h4g2", "h4h2", "h4f3", "h4g3", "h4h3", "h4a4",
    "h4b4", "h4c4", "h4d4", "h4e4", "h4f4", "h4g4", "h4f5", "h4g5",
    "h4h5", "h4f6", "h4g6", "h4h6", "h4e7", "h4h7", "h4d8", "h4h8",
    "a5a1", "a5e1", "a5a2", "a5d2", "a5a3", "a5b3", "a5c3", "a5a4",
    "a5b4", "a5c4", "a5b5", "a5c5", "a5d5", "a5e5", "a5f5", "a5g5",
    "a5h5", "a5a6", "a5b6", "a5c6", "a5a7", "a5b7", "a5c7", "a5a8",
    "a5d8", "b5b1", "b5f1", "b5b2", "b5e2", "b5a3", "b5b3", "b5c3",
    "b5d3", "b5a4", "b5b4", "b5c4", "b5d4", "b5a5", "b5c5", "b5d5",
    "b5e5", "b5f5", "b5g5", "b5h5", "b5a6", "b5b6", "b5c6", "b5d6",
    "b5a7", "b5b7", "b5c7", "b5d7", "b5b8", "b5e8", "c5c1", "c5g1",
    "c5c2", "c5f2", "c5a3", "c5b3", "c5c3", "c5d3", "c5e3", "c5a4",
    "c5b4", "c5c4", "c5d4", "c5e4", "c5a5", "c5b5", "c5d5", "c5e5",
    "c5f5", "c5g5", "c5h5", "c5a6", "c5b6", "c5c6", "c5d6", "c5e6",
    "c5a7", "c5b7", "c5c7", "c5d7", "c5e7", "c5c8", "c5f8", "d5d1",
    "d5h1", "d5a2", "d5d2", "d5g2", "d5b3", "d5c3", "d5d3", "d5e3",
    "d5f3", "d5b4", "d5c4", "d5d4", "d5e4", "d5f4", "d5a5", "d5b5",
    "d5c5", "d5e5", "d5f5", "d5g5", "d5h5", "d5b6", "d5c6", "d5d6",
    "d5e6", "d5f6", "d5b7", "d5c7", "d5d7", "d5e7", "d5f7", "d5a8",
    "d5d8", "d5g8", "e5a1", "e5e1", "e5b2", "e5e2", "e5h2", "e5c3",
    "e5d3", "e5e3", "e5f3", "e5g3", "e5c4", "e5d4", "e5e4", "e5f4",
    "e5g4", "e5a5", "e5b5", "e5c5", "e5d5", "e5f5", "e5g5", "e5h5",
    "e5c6", "e5d6", "e5e6", "e5f6", "e5g6", "e5c7", "e5d7", "e5e7",
    "e5f7", "e5g7", "e5b8", "e5e8", "e5h8", "f5b1", "f5f1", "f5c2",
    "f5f2", "f5d3", "f5e3", "f5f3", "f5g3", "f5h3", "f5d4", "f5e4",
    "f5f4", "f5g4", "f5h4", "f5a5", "f5b5", "f5c5", "f5d5", "f5e5",
    "f5g5", "f5h5", "f5d6", "f5e6", "f5f6", "f5g6", "f5h6", "f5d7",
    "f5e7", "f5f7", "f5g7", "f5h7", "f5c8", "f5f8", "g5c1", "g5g1",
    "g5d2", "g5g2", "g5e3", "g5f3", "g5g3", "g5h3", "g5e4", "g5f4",
    "g5g4", "g5h4", "g5a5", "g5b5", "g5c5", "g5d5", "g5e5", "g5f5",
    "g5h5", "g5e6", "g5f6", "g5g6", "g5h6", "g5e7", "g5f7", "g5g7",
    "g5h7", "g5d8", "g5g8", "h5d1", "h5h1", "h5e2", "h5h2", "h5f3",
    "h5g3", "h5h3", "h5f4", "h5g4", "h5h4", "h5a5", "h5b5", "h5c5",
    "h5d5", "h5e5", "h5f5", "h5g5", "h5f6", "h5g6", "h5h6", "h5f7",
    "h5g7", "h5h7", "h5e8", "h5h8", "a6a1", "a6f1", "a6a2", "a6e2",
    "a6a3", "a6d3", "a6a4", "a6b4", "a6c4", "a6a5", "a6b5", "a6c5",
    "a6b6", "a6c6", "a6d6", "a6e6", "a6f6", "a6g6", "a6h6", "a6a7",
    "a6b7", "a6c7", "a6a8", "a6b8", "a6c8", "b6b1", "b6g1", "b6b2",
    "b6f2", "b6b3", "b6e3", "b6a4", "b6b4", "b6c4", "b6d4", "b6a5",
    "b6b5", "b6c5", "b6d5", "b6a6", "b6c6", "b6d6", "b6e6", "b6f6",
    "b6g6", "b6h6", "b6a7", "b6b7", "b6c7", "b6d7", "b6a8", "b6b8",
    "b6c8", "b6d8", "c6c1", "c6h1", "c6c2", "c6g2", "c6c3", "c6f3",
    "c6a4", "c6b4", "c6c4", "c6d4", "c6e4", "c6a5", "c6b5", "c6c5",
    "c6d5", "c6e5", "c6a6", "c6b6", "c6d6", "c6e6", "c6f6", "c6g6",
    "c6h6", "c6a7", "c6b7", "c6c7", "c6d7", "c6e7", "c6a8", "c6b8",
    "c6c8", "c6d8", "c6e8", "d6d1", "d6d2", "d6h2", "d6a3", "d6d3",
    "d6g3", "d6b4", "d6c4", "d6d4", "d6e4", "d6f4", "d6b5", "d6c5",
    "d6d5", "d6e5", "d6f5", "d6a6", "d6b6", "d6c6", "d6e6", "d6f6",
    "d6g6", "d6h6", "d6b7", "d6c7", "d6d7", "d6e7", "d6f7", "d6b8",
    "d6c8", "d6d8", "d6e8", "d6f8", "e6e1", "e6a2", "e6e2", "e6b3",
    "e6e3", "e6h3", "e6c4", "e6d4", "e6e4", "e6f4", "e6g4", "e6c5",
    "e6d5", "e6e5", "e6f5", "e6g5", "e6a6", "e6b6", "e6c6", "e6d6",
    "e6f6", "e6g6", "e6h6", "e6c7", "e6d7", "e6e7", "e6f7", "e6g7",
    "e6c8", "e6d8", "e6e8", "e6f8", "e6g8", "f6a1", "f6f1", "f6b2",
    "f6f2", "f6c3", "f6f3", "f6d4", "f6e4", "f6f4", "f6g4", "f6h4",
    "f6d5", "f6e5", "f6f5", "f6g5", "f6h5", "f6a6", "f6b6", "f6c6",
    "f6d6", "f6e6", "f6g6", "f6h6", "f6d7", "f6e7", "f6f7", "f6g7",
    "f6h7", "f6d8", "f6e8", "f6f8", "f6g8", "f6h8", "g6b1", "g6g1",
    "g6c2", "g6g2", "g6d3", "g6g3", "g6e4", "g6f4", "g6g4", "g6h4",
    "g6e5", "g6f5", "g6g5", "g6h5", "g6a6", "g6b6", "g6c6", "g6d6",
    "g6e6", "g6f6", "g6h6", "g6e7", "g6f7", "g6g7", "g6h7", "g6e8",
    "g6f8", "g6g8", "g6h8", "h6c1", "h6h1", "h6d2", "h6h2", "h6e3",
    "h6h3", "h6f4", "h6g4", "h6h4", "h6f5", "h6g5", "h6h5", "h6a6",
    "h6b6", "h6c6", "h6d6", "h6e6", "h6f6", "h6g6", "h6f7", "h6g7",
    "h6h7", "h6f8", "h6g8", "h6h8", "a7a1", "a7g1", "a7a2", "a7f2",
    "a7a3", "a7e3", "a7a4", "a7d4", "a7a5", "a7b5", "a7c5", "a7a6",
    "a7b6", "a7c6", "a7b7", "a7c7", "a7d7", "a7e7", "a7f7", "a7g7",
    "a7h7", "a7a8", "a7b8", "a7c8", "b7b1", "b7h1", "b7b2", "b7g2",
    "b7b3", "b7f3", "b7b4", "b7e4", "b7a5", "b7b5", "b7c5", "b7d5",
    "b7a6", "b7b6", "b7c6", "b7d6", "b7a7", "b7c7", "b7d7", "b7e7",
    "b7f7", "b7g7", "b7h7", "b7a8", "b7b8", "b7c8", "b7d8", "c7c1",
    "c7c2", "c7h2", "c7c3", "c7g3", "c7c4", "c7f4", "c7a5", "c7b5",
    "c7c5", "c7d5", "c7e5", "c7a6", "c7b6", "c7c6", "c7d6", "c7e6",
    "c7a7", "c7b7", "c7d7", "c7e7", "c7f7", "c7g7", "c7h7", "c7a8",
    "c7b8", "c7c8", "c7d8", "c7e8", "d7d1", "d7d2", "d7d3", "d7h3",
    "d7a4", "d7d4", "d7g4", "d7b5", "d7c5", "d7d5", "d7e5", "d7f5",
    "d7b6", "d7c6", "d7d6", "d7e6", "d7f6", "d7a7", "d7b7", "d7c7",
    "d7e7", "d7f7", "d7g7", "d7h7", "d7b8", "d7c8", "d7d8", "d7e8",
    "d7f8", "e7e1", "e7e2", "e7a3", "e7e3", "e7b4", "e7e4", "e7h4",
    "e7c5", "e7d5", "e7e5", "e7f5", "e7g5", "e7c6", "e7d6", "e7e6",
    "e7f6", "e7g6", "e7a7", "e7b7", "e7c7", "e7d7", "e7f7", "e7g7",
    "e7h7", "e7c8", "e7d8", "e7e8", "e7f8", "e7g8", "f7f1", "f7a2",
    "f7f2", "f7b3", "f7f3", "f7c4", "f7f4", "f7d5", "f7e5", "f7f5",
    "f7g5", "f7h5", "f7d6", "f7e6", "f7f6", "f7g6", "f7h6", "f7a7",
    "f7b7", "f7c7", "f7d7", "f7e7", "f7g7", "f7h7", "f7d8", "f7e8",
    "f7f8", "f7g8", "f7h8", "g7a1", "g7g1", "g7b2", "g7g2", "g7c3",
    "g7g3", "g7d4", "g7g4", "g7e5", "g7f5", "g7g5", "g7h5", "g7e6",
    "g7f6", "g7g6", "g7h6", "g7a7", "g7b7", "g7c7", "g7d7", "g7e7",
    "g7f7", "g7h7", "g7e8", "g7f8", "g7g8", "g7h8", "h7b1", "h7h1",
    "h7c2", "h7h2", "h7d3", "h7h3", "h7e4", "h7h4", "h7f5", "h7g5",
    "h7h5", "h7f6", "h7g6", "h7h6", "h7a7", "h7b7", "h7c7", "h7d7",
    "h7e7", "h7f7", "h7g7", "h7f8", "h7g8", "h7h8", "a8a1", "a8h1",
    "a8a2", "a8g2", "a8a3", "a8f3", "a8a4", "a8e4", "a8a5", "a8d5",
    "a8a6", "a8b6", "a8c6", "a8a7", "a8b7", "a8c7", "a8b8", "a8c8",
    "a8d8", "a8e8", "a8f8", "a8g8", "a8h8", "b8b1", "b8b2", "b8h2",
    "b8b3", "b8g3", "b8b4", "b8f4", "b8b5", "b8e5", "b8a6", "b8b6",
    "b8c6", "b8d6", "b8a7", "b8b7", "b8c7", "b8d7", "b8a8", "b8c8",
    "b8d8", "b8e8", "b8f8", "b8g8", "b8h8", "c8c1", "c8c2", "c8c3",
    "c8h3", "c8c4", "c8g4", "c8c5", "c8f5", "c8a6", "c8b6", "c8c6",
    "c8d6", "c8e6", "c8a7", "c8b7", "c8c7", "c8d7", "c8e7", "c8a8",
    "c8b8", "c8d8", "c8e8", "c8f8", "c8g8", "c8h8", "d8d1", "d8d2",
    "d8d3", "d8d4", "d8h4", "d8a5", "d8d5", "d8g5", "d8b6", "d8c6",
    "d8d6", "d8e6", "d8f6", "d8b7", "d8c7", "d8d7", "d8e7", "d8f7",
    "d8a8", "d8b8", "d8c8", "d8e8", "d8f8", "d8g8", "d8h8", "e8e1",
    "e8e2", "e8e3", "e8a4", "e8e4", "e8b5", "e8e5", "e8h5", "e8c6",
    "e8d6", "e8e6", "e8f6", "e8g6", "e8c7", "e8d7", "e8e7", "e8f7",
    "e8g7", "e8a8", "e8b8", "e8c8", "e8d8", "e8f8", "e8g8", "e8h8",
    "f8f1", "f8f2", "f8a3", "f8f3", "f8b4", "f8f4", "f8c5", "f8f5",
    "f8d6", "f8e6", "f8f6", "f8g6", "f8h6", "f8d7", "f8e7", "f8f7",
    "f8g7", "f8h7", "f8a8", "f8b8", "f8c8", "f8d8", "f8e8", "f8g8",
    "f8h8", "g8g1", "g8a2", "g8g2", "g8b3", "g8g3", "g8c4", "g8g4",
    "g8d5", "g8g5", "g8e6", "g8f6", "g8g6", "g8h6", "g8e7", "g8f7",
    "g8g7", "g8h7", "g8a8", "g8b8", "g8c8", "g8d8", "g8e8", "g8f8",
    "g8h8", "h8a1", "h8h1", "h8b2", "h8h2", "h8c3", "h8h3", "h8d4",
    "h8h4", "h8e5", "h8h5", "h8f6", "h8g6", "h8h6", "h8f7", "h8g7",
    "h8h7", "h8a8", "h8b8", "h8c8", "h8d8", "h8e8", "h8f8", "h8g8",
    "a7a8q", "a7a8r", "a7a8b", "a7b8q", "a7b8r", "a7b8b", "b7a8q", "b7a8r",
    "b7a8b", "b7b8q", "b7b8r", "b7b8b", "b7c8q", "b7c8r", "b7c8b", "c7b8q",
    "c7b8r", "c7b8b", "c7c8q", "c7c8r", "c7c8b", "c7d8q", "c7d8r", "c7d8b",
    "d7c8q", "d7c8r", "d7c8b", "d7d8q", "d7d8r", "d7d8b", "d7e8q", "d7e8r",
    "d7e8b", "e7d8q", "e7d8r", "e7d8b", "e7e8q", "e7e8r", "e7e8b", "e7f8q",
    "e7f8r", "e7f8b", "f7e8q", "f7e8r", "f7e8b", "f7f8q", "f7f8r", "f7f8b",
    "f7g8q", "f7g8r", "f7g8b", "g7f8q", "g7f8r", "g7f8b", "g7g8q", "g7g8r",
    "g7g8b", "g7h8q", "g7h8r", "g7h8b", "h7g8q", "h7g8r", "h7g8b", "h7h8q",
    "h7h8r", "h7h8b",
};

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <array>
#include <condition_variable>
#include <unordered_map>
#include <zlib.h>

using namespace luminex;

#pragma pack(push, 1)
struct V6Rec {
    uint32_t version, input_format;
    float probabilities[1858];
    uint64_t planes[104];
    uint8_t castling_us_ooo, castling_us_oo, castling_them_ooo, castling_them_oo;
    uint8_t side_to_move_or_enpassant;
    uint8_t rule50_count;
    uint8_t invariance_info;
    uint8_t dummy;
    float root_q, best_q, root_d, best_d, root_m, best_m, plies_left;
    float result_q, result_d, played_q, played_d, played_m, orig_q, orig_d, orig_m;
    uint32_t visits;
    uint16_t played_idx, best_idx;
    float policy_kld, q_st;
};
#pragma pack(pop)
static_assert(sizeof(V6Rec) == 8356, "record size mismatch");

// ---- transform machinery (lc0 encoder.cc semantics, combined self-inverse map) ----
// bit0 flip (file), bit1 mirror (rank), bit2 transpose. Transform(square):
//   rank flips if (mirror|transpose), file flips if (flip|transpose).
struct SqMap { uint8_t to[64]; };
static SqMap g_map[8];
static void init_maps() {
    for (int t = 0; t < 8; ++t) {
        bool rf = (t & 0x2) || (t & 0x4);   // mirror|transpose -> rank flip
        bool ff = (t & 0x1) || (t & 0x4);   // flip|transpose   -> file flip
        for (int sq = 0; sq < 64; ++sq) {
            int f = sq & 7, r = sq >> 3;
            if (rf) r ^= 7;
            if (ff) f ^= 7;
            g_map[t].to[sq] = (uint8_t)(r * 8 + f);
        }
    }
}
static inline uint64_t map_bb(uint64_t b, const SqMap& m) {
    uint64_t r = 0;
    while (b) { int sq = __builtin_ctzll(b); b &= b - 1; r |= 1ULL << m.to[sq]; }
    return r;
}

// ---- policy index -> our Move (from/to/promo), with un-transform ----
static int g_idx_from[1858], g_idx_to[1858], g_idx_promo[1858]; // PieceType N=1 B=2 R=3 Q=4
static void init_move_table() {
    for (int i = 0; i < 1858; ++i) {
        const char* s = kNNMoves[i];
        g_idx_from[i] = (s[0] - 'a') + (s[1] - '1') * 8;
        g_idx_to[i]   = (s[2] - 'a') + (s[3] - '1') * 8;
        g_idx_promo[i] = s[4] ? (s[4] == 'n' ? 1 : s[4] == 'b' ? 2 : s[4] == 'r' ? 3 : 4) : 0;
    }
}

// (index-mode sort key no longer needed in RAW mode — kept out deliberately)

struct Shared {
    std::mutex mtx;
    std::unordered_map<std::string, uint32_t> fen_dict;
    std::vector<std::string> fen_list;
    uint32_t get_fen_idx(const std::string& fen) {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = fen_dict.find(fen);
        if (it != fen_dict.end()) return it->second;
        uint32_t idx = (uint32_t)fen_list.size();
        fen_dict[fen] = idx; fen_list.push_back(fen);
        return idx;
    }
};

// v2 header magic — larger than any possible fen count, so the featurizer can
// tell v2 (magic + 10-byte entries) from v1 (bare nfens + 9-byte entries).
static constexpr uint32_t GP2_MAGIC = 0x47503200;
struct PerThread {
    int tid = 0;
    std::vector<uint8_t> game_entries;
    std::vector<uint8_t> ev_bytes;
    std::vector<uint16_t> mv_raw;   // RAW mode: 16-bit moves (featurize replays without legal-gen)
    uint64_t games = 0, pos = 0, skipped = 0;
};

static double q_to_cp(float q) {
    if (std::isnan(q)) q = 0.0f;          // safety: treat NaN as draw
    if (q > 0.99999f) q = 0.99999f;       // clamp, DON'T skip — decided positions
    if (q < -0.99999f) q = -0.99999f;     // are exactly what we need (fat tail!)
    return 800.0 * 0.5 * std::log((1.0 + q) / (1.0 - q));
}

static inline bool rec_stm_black(const V6Rec& r) {
    return r.input_format >= 3 ? (r.invariance_info & 0x80) != 0
                               : (r.side_to_move_or_enpassant != 0);
}

// (from_pt_is_king removed — was only used by the deleted fast path)

// Decode a record's 12 piece bitboards into TRUE white-perspective placement.
// Stored planes are bit-reversed per byte (writer applies ReverseBitsInBytes).
// IFMT=1 (classical): stm-perspective (ours=stm, ranks mirrored for black);
//   stm = side_to_move_or_enpassant byte. IFMT>=3: white-perspective + invariance
//   transform (bits 0-2), stm = bit 7.
static void decode_boards(const V6Rec& r, uint64_t out[12]) {
    static uint64_t rev[256]; static bool init = false;
    if (!init) { for (int x = 0; x < 256; ++x) { uint64_t v = 0; for (int j = 0; j < 8; ++j) if (x >> j & 1) v |= 1ULL << (7 - j); rev[x] = v; } init = true; }
    for (int i = 0; i < 12; ++i) {
        uint64_t b = r.planes[i];
        b = (rev[b & 0xFF]) | (rev[(b >> 8) & 0xFF] << 8) | (rev[(b >> 16) & 0xFF] << 16) |
            (rev[(b >> 24) & 0xFF] << 24) | (rev[(b >> 32) & 0xFF] << 32) |
            (rev[(b >> 40) & 0xFF] << 40) | (rev[(b >> 48) & 0xFF] << 48) |
            (rev[(b >> 56) & 0xFF] << 56);
        out[i] = b;
    }
    if (r.input_format >= 3) {
        const SqMap& m = g_map[r.invariance_info & 0x7];
        for (int i = 0; i < 12; ++i) out[i] = map_bb(out[i], m);
    } else if (rec_stm_black(r)) {
        for (int i = 0; i < 12; ++i) out[i] = __builtin_bswap64(out[i]);
        for (int p = 0; p < 6; ++p) std::swap(out[p], out[6 + p]);
    }
}


// Reconstruct the true FEN of a record.
// IFMT=1 (classical): planes are SIDE-TO-MOVE perspective (ours=stm, ranks mirrored
//   for black); stm = side_to_move_or_enpassant byte (0=white 1=black); no transforms.
// IFMT>=3 (canonical): planes white-perspective; invariance bits 0-2 transform,
//   bit 7 = stm.
static std::string rec_to_fen(const V6Rec& r) {
    uint64_t bb[12];
    decode_boards(r, bb);
    bool black = rec_stm_black(r);
    static const char W[6] = {'P','N','B','R','Q','K'};
    static const char B[6] = {'p','n','b','r','q','k'};
    std::string fen;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            int sq = rank * 8 + file;
            char pc = 0;
            for (int p = 0; p < 6 && !pc; ++p) {
                if ((bb[p] >> sq) & 1) pc = W[p];
                else if ((bb[6 + p] >> sq) & 1) pc = B[p];
            }
            if (pc) { if (empty) { fen += std::to_string(empty); empty = 0; } fen += pc; }
            else ++empty;
        }
        if (empty) fen += std::to_string(empty);
        if (rank) fen += '/';
    }
    bool black2 = black;   // stm determined above per format
    fen += black2 ? " b " : " w ";
    // castling: classical stores us/them = stm-relative; canonical = white-relative
    bool us_is_white = (!black2) || (r.input_format >= 3);
    std::string cast;
    uint64_t wr = bb[3], br = bb[9];
    uint64_t wk = bb[5], bk = bb[11];
    auto file_at = [](uint64_t b, int rank) -> int {
        if (!b) return -1;
        int sq = __builtin_ctzll(b); if ((sq >> 3) != rank) return -1; return sq & 7;
    };
    bool std_w = file_at(wk, 0) == 4, std_b = file_at(bk, 7) == 4;
    auto us_oo  = us_is_white ? r.castling_us_oo   : r.castling_them_oo;
    auto us_ooo = us_is_white ? r.castling_us_ooo  : r.castling_them_ooo;
    auto th_oo  = us_is_white ? r.castling_them_oo : r.castling_us_oo;
    auto th_ooo = us_is_white ? r.castling_them_ooo: r.castling_us_ooo;
    if (std_w && us_oo  && (wr & (1ULL << 7)))  cast += 'K';
    if (std_w && us_ooo && (wr & (1ULL << 0)))  cast += 'Q';
    if (std_b && th_oo  && (br & (1ULL << 63))) cast += 'k';
    if (std_b && th_ooo && (br & (1ULL << 56))) cast += 'q';
    fen += cast.empty() ? "-" : cast;
    fen += " -";
    fen += " " + std::to_string((int)r.rule50_count) + " 1";
    return fen;
}

static void process_game(const std::vector<uint8_t>& dec, Shared& sh, PerThread& pt, int quant) {
    size_t n = dec.size() / sizeof(V6Rec);
    if (n < 4) { pt.skipped++; return; }   // accept even very short games (was 8)            // too short to be worth packing
    const V6Rec* recs = reinterpret_cast<const V6Rec*>(dec.data());

    // evals first: all must be valid
    std::vector<int> eqs(n);
    for (size_t i = 0; i < n; ++i) {
        double cp = q_to_cp(recs[i].root_q);
        // q_to_cp now clamps (no sentinel) — all positions included
        if (cp > 1500) cp = 1500; if (cp < -1500) cp = -1500;
        bool stm_white2 = !rec_stm_black(recs[i]);
        int white_cp = (int)llround(stm_white2 ? cp : -cp);
        int q = white_cp / quant; if (white_cp % quant != 0 && white_cp < 0) q -= 1;
        eqs[i] = q * quant;                           // quantized white-relative
    }

    std::string fen = rec_to_fen(recs[0]);
    static std::atomic<int> dbg_games{0};
    bool debug = (dbg_games.fetch_add(1) < 2) && getenv("LC0PACK_DEBUG");
    if (debug) fprintf(stderr, "[dbg] game: fen='%s' n=%zu IFMT=%u\n", fen.c_str(), n, recs[0].input_format);
    Position pos;
    pos.set(fen.c_str());
    std::vector<uint16_t> mvs; mvs.reserve(n);
    // LAZY: only 2 boards in memory (192B, always L1 cache). decode next per step.
    uint64_t cur[12], nxt[12];
    decode_boards(recs[0], cur);
    uint64_t cur_w = 0, cur_b = 0;
    for (int p = 0; p < 6; ++p) { cur_w |= cur[p]; cur_b |= cur[6 + p]; }

    for (size_t i = 0; i + 1 < n; ++i) {
        // decode next board (lazy: one decode per step, not upfront)
        decode_boards(recs[i + 1], nxt);
        uint64_t nxt_w = 0, nxt_b = 0;
        for (int p = 0; p < 6; ++p) { nxt_w |= nxt[p]; nxt_b |= nxt[6 + p]; }

        bool wtm = !rec_stm_black(recs[i]);
        uint64_t ma = wtm ? cur_w : cur_b, ma1 = wtm ? nxt_w : nxt_b;
        uint64_t gone = ma & ~ma1, appeared = ma1 & ~ma;
        uint64_t opp = wtm ? cur_b : cur_w;

        Move chosen = MOVE_NONE;

        if (__builtin_popcountll(gone) == 2 && __builtin_popcountll(appeared) == 2) {
            // CASTLING: king AND rook both moved. Get king from/to from king bitboards.
            uint64_t kb = cur[wtm ? 5 : 11], ka = nxt[wtm ? 5 : 11];
            int kfrom = __builtin_ctzll(kb), kto = __builtin_ctzll(ka);
            // kingside = king at G file (6), queenside = C file (2)
            uint16_t flag = ((kto & 7) == 6) ? 0x2000 : 0x3000;  // MF_CASTLING_KING/QUEEN
            chosen = Move((uint16_t)((kfrom << 6) | kto | flag));
        }
        else if (__builtin_popcountll(gone) == 1 && __builtin_popcountll(appeared) == 1) {
            int fs = __builtin_ctzll(gone), ts = __builtin_ctzll(appeared);
            // piece type at from/to (our engine: PAWN=0 KNIGHT=1 BISHOP=2 ROOK=3 QUEEN=4 KING=5)
            int from_pt = -1, to_pt = -1;
            for (int p = 0; p < 6; ++p) {
                if ((cur[(wtm ? p : 6 + p)] >> fs) & 1) from_pt = p;
                if ((nxt[(wtm ? p : 6 + p)] >> ts) & 1) to_pt = p;
            }
            bool is_capture = (opp >> ts) & 1;
            uint16_t raw;
            if (from_pt == 0 && to_pt >= 1 && to_pt <= 4 && (ts >> 3) == (wtm ? 7 : 0)) {
                // PROMOTION: base 0x8000 quiet / 0xC000 capture, + (piece-1)*0x1000
                raw = ((uint16_t)fs << 6) | (uint16_t)ts
                    | (uint16_t)((is_capture ? 0xC000 : 0x8000) + ((to_pt - 1) * 0x1000));
            } else if (from_pt == 0 && abs((ts & 7) - (fs & 7)) == 1 && !is_capture
                       && (ts >> 3) != (wtm ? 7 : 0)
                       && !((cur_w | cur_b) >> ts & 1)) {
                // EN PASSANT: pawn diagonal to empty square (not promotion rank)
                raw = ((uint16_t)fs << 6) | (uint16_t)ts | 0x5000;  // MF_EN_PASSANT
            } else if (from_pt == 0 && (ts & 7) == (fs & 7)
                       && abs((ts >> 3) - (fs >> 3)) == 2) {
                // DOUBLE PAWN PUSH: same file, 2 ranks
                raw = ((uint16_t)fs << 6) | (uint16_t)ts | 0x1000;  // MF_DOUBLE_PAWN
            } else if (is_capture) {
                raw = ((uint16_t)fs << 6) | (uint16_t)ts | 0x4000;  // MF_CAPTURE
            } else {
                raw = ((uint16_t)fs << 6) | (uint16_t)ts;           // MF_QUIET
            }
            chosen = Move(raw);
        }

        // APPLY DIRECTLY — encoding verified (0 divergence on V8). Single application.
        bool applied = false;
        if (chosen != MOVE_NONE) {
            if (pos.do_move_replay(chosen)) {
                applied = true;
                mvs.push_back(chosen.raw());
            } else {
                chosen = MOVE_NONE;
            }
        }

        if (!applied) {
            // Fallback: legal-gen + full scan (castling edge cases, rare failures)
            ExtMove list[MAX_MOVES];
            ExtMove* end = generate<GEN_LEGAL>(pos, list);
            int nlegal = (int)(end - list);
            for (int k = 0; k < nlegal; ++k) {
                Position trial = pos;
                if (!trial.do_move_replay(list[k].move)) continue;
                bool ok = true;
                for (int p = 0; p < 6 && ok; ++p) {
                    if (trial.pieces((Color)WHITE, (PieceType)p) != nxt[p] ||
                        trial.pieces((Color)BLACK, (PieceType)p) != nxt[6 + p]) ok = false;
                }
                if (ok) { chosen = list[k].move; break; }
            }
            if (chosen == MOVE_NONE) { pt.skipped++; return; }
            mvs.push_back(chosen.raw());
            pos.do_move_replay(chosen);
        }

        // advance: cur ← nxt (lazy decoding — next iteration decodes the new nxt)
        memcpy(cur, nxt, sizeof(cur));
        cur_w = nxt_w; cur_b = nxt_b;
    }

    // emit gamepack structures
    uint32_t fidx = sh.get_fen_idx(fen);
    uint16_t np = (uint16_t)(n - 1);   // evals attach to after-move boards: drop board-0's eval
    int16_t start_eval = (int16_t)eqs[1];
    float rq = recs[n - 1].result_q;   // white-relative game result (constant per game)
    uint8_t result = rq > 0.5f ? 1 : (rq < -0.5f ? (uint8_t)0xFF : 0);   // int8 -1/0/+1
    pt.game_entries.push_back((uint8_t)(np & 0xFF));
    pt.game_entries.push_back((uint8_t)((np >> 8) & 0xFF));
    pt.game_entries.push_back(rec_stm_black(recs[0]) ? 0 : 1);
    pt.game_entries.insert(pt.game_entries.end(), (uint8_t*)&fidx, (uint8_t*)&fidx + 4);
    pt.game_entries.insert(pt.game_entries.end(), (uint8_t*)&start_eval, (uint8_t*)&start_eval + 2);
    pt.game_entries.push_back(result);
    pt.mv_raw.insert(pt.mv_raw.end(), mvs.begin(), mvs.end());
    for (size_t i = 2; i < n; ++i) {
        int d = eqs[i] - eqs[i - 1];
        if (d >= -127 && d <= 127) { int8_t b = (int8_t)d; pt.ev_bytes.push_back((uint8_t)b); }
        else {
            pt.ev_bytes.push_back(0x80);
            int16_t abs = (int16_t)eqs[i];
            pt.ev_bytes.insert(pt.ev_bytes.end(), (uint8_t*)&abs, (uint8_t*)&abs + 2);
        }
    }
    pt.games++; pt.pos += n;
}

static bool gz_decompress(const uint8_t* data, size_t size, std::vector<uint8_t>& out) {
    z_stream zs{};
    if (inflateInit2(&zs, 16 + MAX_WBITS) != Z_OK) return false;
    // Read ISIZE from gzip footer (last 4 bytes) for EXACT pre-allocation.
    // Average chunk decompresses to ~970KB; the old size*40=1MB caused 30-40%
    // of chunks to trigger the slow realloc+retry path.
    size_t est;
    if (size >= 8) {
        uint32_t isize = (uint32_t)data[size-4] | ((uint32_t)data[size-3] << 8)
                       | ((uint32_t)data[size-2] << 16) | ((uint32_t)data[size-1] << 24);
        est = (isize > 0 && isize < (1u<<30)) ? isize : size * 80;
    } else {
        est = size * 80;
    }
    out.resize(est);
    zs.next_in = const_cast<Bytef*>(data); zs.avail_in = size;
    zs.next_out = out.data(); zs.avail_out = out.size();
    int rc = inflate(&zs, Z_FINISH);
    if (rc == Z_STREAM_END) {
        out.resize(out.size() - zs.avail_out);
        inflateEnd(&zs);
        return true;
    }
    if (rc == Z_BUF_ERROR && est <= (size * 80)) {
        // ISIZE was wrong (multi-member?) — fallback to generous size and retry
        size_t used = out.size() - zs.avail_out;
        out.resize(size * 80);
        zs.next_out = out.data() + used; zs.avail_out = out.size() - used;
        rc = inflate(&zs, Z_FINISH);
        if (rc == Z_STREAM_END) {
            out.resize(out.size() - zs.avail_out);
            inflateEnd(&zs);
            return true;
        }
    }
    inflateEnd(&zs);
    return false;
}

struct ChunkFile { std::vector<uint8_t> gz; };

// Process ONE tar file at a time (memory-safe: peak = 1 tar ~100MB + its chunks)
static void process_tar(const std::string& path, Shared& sh, PerThread& pt,
                        int quant, std::atomic<size_t>& next_id,
                        std::vector<std::vector<ChunkFile>>& all_chunks,
                        std::mutex& chunks_mtx) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) return;
    std::vector<uint8_t> tar; { std::vector<uint8_t> buf(1 << 20); size_t n; while ((n = fread(buf.data(), 1, buf.size(), fp)) > 0) tar.insert(tar.end(), buf.begin(), buf.begin() + n); }
    fclose(fp);

    // extract chunks from this tar
    std::vector<ChunkFile> chunks;
    if (tar.size() >= 2 && tar[0] == 0x1f && tar[1] == 0x8b) {
        chunks.push_back({std::move(tar)});
    } else if (tar.size() > 512) {
        size_t off = 0;
        while (off + 512 <= tar.size()) {
            const uint8_t* hdr = tar.data() + off;
            if (hdr[0] == 0) break;
            size_t sz = 0; for (int k = 0; k < 11; ++k) sz = sz * 8 + (hdr[124 + k] - '0');
            size_t nd = off + 512 + sz;
            if (sz > 0 && nd <= tar.size()) chunks.push_back({std::vector<uint8_t>(tar.begin() + off + 512, tar.begin() + nd)});
            off = (nd + 511) & ~size_t(511);
        }
    }

    // hand chunks to the global pool (workers pick them up)
    {
        std::lock_guard<std::mutex> lk(chunks_mtx);
        all_chunks.push_back(std::move(chunks));
    }
}

int main(int argc, char** argv) {
    int nthreads = (int)std::thread::hardware_concurrency(); if (nthreads > 8) nthreads = 8;
    int quant = 8;
    const char* out_path = nullptr;
    std::vector<std::string> inputs;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--threads") nthreads = std::max(1, atoi(next()));
        else if (a == "--quant") quant = atoi(next());
        else if (a == "-o" || a == "--out") out_path = next();
        else inputs.push_back(a);
    }
    if (inputs.empty() || !out_path) { fprintf(stderr, "usage: lc0pack --out frame.raw [--threads N] tars...\n"); return 2; }
    init_magic_bitboards();
    init_maps();
    init_move_table();

    // In-memory expansion (BATCH ≤ 40 tars = ~4GB peak — safe for 14GB machines).
    // For larger batches, invoke multiple times; the gamepack format is per-invocation.
    std::vector<ChunkFile> chunks;
    for (auto& f : inputs) {
        FILE* fp = fopen(f.c_str(), "rb");
        if (!fp) continue;
        std::vector<uint8_t> tar; { std::vector<uint8_t> buf(1 << 20); size_t n; while ((n = fread(buf.data(), 1, buf.size(), fp)) > 0) tar.insert(tar.end(), buf.begin(), buf.begin() + n); }
        fclose(fp);
        if (tar.size() >= 2 && tar[0] == 0x1f && tar[1] == 0x8b) {
            chunks.push_back({std::move(tar)});
        } else if (tar.size() > 512) {
            size_t off = 0;
            while (off + 512 <= tar.size()) {
                const uint8_t* hdr = tar.data() + off;
                if (hdr[0] == 0) break;
                size_t sz = 0; for (int k = 0; k < 11; ++k) sz = sz * 8 + (hdr[124 + k] - '0');
                size_t nd = off + 512 + sz;
                if (sz > 0 && nd <= tar.size()) chunks.push_back({std::vector<uint8_t>(tar.begin() + off + 512, tar.begin() + nd)});
                off = (nd + 511) & ~size_t(511);
            }
        }
        tar.clear(); tar.shrink_to_fit();   // free this tar before reading the next
    }
    fprintf(stderr, "lc0pack: %zu chunks from %zu files\n", chunks.size(), inputs.size());

    Shared sh;
    std::vector<PerThread> pts(nthreads);
    std::atomic<size_t> next{0};

    auto worker = [&](int tid) {
        PerThread& pt = pts[tid];
        std::vector<uint8_t> dec;
        while (true) {
            size_t i = next.fetch_add(1);
            if (i >= chunks.size()) break;
            if (gz_decompress(chunks[i].gz.data(), chunks[i].gz.size(), dec))
                process_game(dec, sh, pt, quant);
            else pt.skipped++;
        }
    };
    std::vector<std::thread> ths;
    for (int t = 0; t < nthreads; ++t) ths.emplace_back(worker, t);
    for (auto& t : ths) t.join();

    // assemble: [u64 hdr_n][hdr][u64 mv_n][mv][u64 ev_n][ev]
    std::vector<uint8_t> hdr;
    hdr.insert(hdr.end(), (const uint8_t*)&GP2_MAGIC, (const uint8_t*)&GP2_MAGIC + 4);
    uint32_t nfens = (uint32_t)sh.fen_list.size();
    hdr.insert(hdr.end(), (uint8_t*)&nfens, (uint8_t*)&nfens + 4);
    for (auto& f : sh.fen_list) {
        uint16_t fl = (uint16_t)f.size();
        hdr.insert(hdr.end(), (uint8_t*)&fl, (uint8_t*)&fl + 2);
        hdr.insert(hdr.end(), f.begin(), f.end());
    }
    uint64_t sz_hdr = hdr.size(), sz_mv = 0, sz_ev = 0;
    for (auto& pt : pts) { sz_mv += pt.mv_raw.size() * 2; sz_ev += pt.ev_bytes.size(); }
    // game entries appended AFTER fens (thread order)
    for (auto& pt : pts) hdr.insert(hdr.end(), pt.game_entries.begin(), pt.game_entries.end());
    sz_hdr = hdr.size();

    FILE* out = fopen(out_path, "wb");
    if (!out) { fprintf(stderr, "cannot write %s\n", out_path); return 1; }
    fwrite(&sz_hdr, 8, 1, out); fwrite(hdr.data(), 1, hdr.size(), out);
    fwrite(&sz_mv, 8, 1, out);
    for (auto& pt : pts) fwrite(pt.mv_raw.data(), 2, pt.mv_raw.size(), out);
    fwrite(&sz_ev, 8, 1, out);
    for (auto& pt : pts) fwrite(pt.ev_bytes.data(), 1, pt.ev_bytes.size(), out);
    fclose(out);
    uint64_t games = 0, posn = 0, skip = 0;
    for (auto& pt : pts) { games += pt.games; posn += pt.pos; skip += pt.skipped; }
    fprintf(stderr, "lc0pack: %llu games %llu positions (%llu skipped games), fens=%u, out=%s\n",
            (unsigned long long)games, (unsigned long long)posn, (unsigned long long)skip, nfens, out_path);
    return 0;
}

#elif defined(LUMINEX_TOOL_FRAME_MERGE)

// frame_merge — merge N gamepack frames into one denser frame.
//
// Frame format (written by luminex-lc0pack, read by luminex-featurize):
//   [u64 hdr_len][hdr][u64 mv_len][mv][u64 ev_len][ev]
//   hdr = u32 nfens, { u16 len + fen bytes } * nfens, then 9-byte game entries:
//         u16 np (LE) | u8 stm_white | u32 fidx (LE) | i16 start_eval (LE)
//   mv   = np u16 raw moves per game, entries in order
//   ev   = per game (np-1) eval deltas: int8, or 0x80 + i16 absolute (3 bytes)
//
// Per-game data is fully self-contained (start_eval anchors each game's delta
// chain), so a merge is: concat FEN tables with fidx remapping by cumulative
// offsets, concat game entries, concat mv and ev streams — order preserving.
// The tool WALKS every game's eval stream to prove the parse consumed exactly
// ev_len bytes (the same walk the featurizer performs); any mismatch aborts
// before writing, so a corrupt input can never produce a corrupt output.
//
// usage: frame_merge -o merged.raw frameA.raw frameB.raw [more...]
// (frames on disk are .zst/.xz compressed; decompress before, recompress after)

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

static bool read_u64(FILE* f, uint64_t& v) {
    return fread(&v, 8, 1, f) == 1;
}
static bool read_exact(FILE* f, std::vector<uint8_t>& buf, uint64_t n) {
    buf.resize((size_t)n);
    return n == 0 || fread(buf.data(), 1, (size_t)n, f) == n;
}

struct Frame {
    std::vector<uint8_t> hdr;   // nfens + fen table + game entries (remappable)
    std::vector<uint8_t> mv;
    std::vector<uint8_t> ev;
    uint32_t nfens = 0;
    size_t n_entries = 0;
    uint64_t total_np = 0;      // sum of np across games (u16 move count check)
};

static bool load_frame(const char* path, Frame& fr, uint64_t& games_out) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "frame_merge: cannot open %s\n", path); return false; }
    uint64_t hl = 0, ml = 0, el = 0;
    bool ok = read_u64(f, hl) && read_exact(f, fr.hdr, hl)
           && read_u64(f, ml) && read_exact(f, fr.mv, ml)
           && read_u64(f, el) && read_exact(f, fr.ev, el);
    // trailing bytes after ev = corruption (featurize would misparse the next
    // section boundary in a concatenated stream context)
    uint8_t extra;
    if (ok && fread(&extra, 1, 1, f) == 1) { fprintf(stderr, "frame_merge: %s has trailing bytes\n", path); ok = false; }
    fclose(f);
    if (!ok) { fprintf(stderr, "frame_merge: %s truncated/invalid\n", path); return false; }

    // ---- parse header: fen table, then 9-byte entries ----
    size_t p = 0;
    if (fr.hdr.size() < 4) { fprintf(stderr, "frame_merge: %s hdr too small\n", path); return false; }
    memcpy(&fr.nfens, fr.hdr.data() + p, 4); p += 4;
    for (uint32_t i = 0; i < fr.nfens; ++i) {
        if (p + 2 > fr.hdr.size()) { fprintf(stderr, "frame_merge: %s fen table overrun\n", path); return false; }
        uint16_t fl; memcpy(&fl, fr.hdr.data() + p, 2); p += 2;
        if (p + fl > fr.hdr.size()) { fprintf(stderr, "frame_merge: %s fen string overrun\n", path); return false; }
        p += fl;
    }
    if ((fr.hdr.size() - p) % 9 != 0) { fprintf(stderr, "frame_merge: %s entry block not 9-byte aligned\n", path); return false; }
    fr.n_entries = (fr.hdr.size() - p) / 9;
    if (fr.mv.size() % 2 != 0) { fprintf(stderr, "frame_merge: %s mv length odd\n", path); return false; }

    // ---- walk entries: verify fidx range, np sum vs mv_len, and that the
    // delta-eval walk consumes EXACTLY ev_len bytes (the featurizer's read) ----
    uint64_t np_sum = 0, ev_pos = 0;
    for (size_t e = 0; e < fr.n_entries; ++e) {
        const uint8_t* ent = fr.hdr.data() + p + e * 9;
        uint16_t np; memcpy(&np, ent, 2);
        uint32_t fidx; memcpy(&fidx, ent + 3, 4);
        if (fidx >= fr.nfens) { fprintf(stderr, "frame_merge: %s entry %zu fidx %u >= nfens %u\n", path, e, fidx, fr.nfens); return false; }
        if (np == 0) continue;   // start board only — no moves, no evals
        np_sum += np;
        for (uint32_t d = 0; d + 1 < np; ++d) {
            if (ev_pos >= fr.ev.size()) { fprintf(stderr, "frame_merge: %s eval stream underrun (entry %zu)\n", path, e); return false; }
            if (fr.ev[ev_pos] == 0x80) ev_pos += 3; else ev_pos += 1;
        }
    }
    if (np_sum * 2 != fr.mv.size()) { fprintf(stderr, "frame_merge: %s move count %llu != entries %llu\n", path, (unsigned long long)(fr.mv.size() / 2), (unsigned long long)np_sum); return false; }
    if (ev_pos != fr.ev.size()) { fprintf(stderr, "frame_merge: %s eval stream length mismatch (%llu used of %llu)\n", path, (unsigned long long)ev_pos, (unsigned long long)fr.ev.size()); return false; }
    fr.total_np = np_sum;
    games_out = fr.n_entries;
    return true;
}

int main(int argc, char** argv) {
    std::string out_path;
    std::vector<std::string> inputs;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-o" || a == "--out") out_path = argv[++i];
        else inputs.push_back(a);
    }
    if (inputs.empty() || out_path.empty()) {
        fprintf(stderr, "usage: frame_merge -o merged.raw frameA.raw frameB.raw [more...]\n");
        return 2;
    }

    std::vector<uint8_t> fens, entries, mv, ev;   // hdr = [nfens][fens][entries]
    uint64_t nfens_total = 0, games_total = 0, np_total = 0;
    uint32_t fens_so_far = 0;
    for (auto& path : inputs) {
        Frame fr; uint64_t g = 0;
        if (!load_frame(path.c_str(), fr, g)) return 1;
        size_t entry_off;
        {   // recompute where the entry block starts in this frame's hdr
            size_t p = 4;
            for (uint32_t i = 0; i < fr.nfens; ++i) { uint16_t fl; memcpy(&fl, fr.hdr.data() + p, 2); p += 2 + fl; }
            entry_off = p;
        }
        // remap every entry's fidx into the merged FEN table's coordinate space
        for (size_t e = 0; e < fr.n_entries; ++e) {
            uint8_t* ent = fr.hdr.data() + entry_off + e * 9;
            uint32_t fidx; memcpy(&fidx, ent + 3, 4);
            uint32_t nf = fidx + fens_so_far;   // both < 2^32: nfens total stays tiny
            memcpy(ent + 3, &nf, 4);
        }
        // the format is [ALL fens][ALL entries] — keep the sections separate
        fens.insert(fens.end(), fr.hdr.begin() + 4, fr.hdr.begin() + entry_off);
        entries.insert(entries.end(), fr.hdr.begin() + entry_off, fr.hdr.end());
        mv.insert(mv.end(), fr.mv.begin(), fr.mv.end());
        ev.insert(ev.end(), fr.ev.begin(), fr.ev.end());
        fens_so_far += fr.nfens; nfens_total += fr.nfens; games_total += g; np_total += fr.total_np;
    }

    // ---- write merged frame: [nfens][fens][entries][mv][ev] ----
    uint32_t nf = (uint32_t)nfens_total;
    FILE* o = fopen(out_path.c_str(), "wb");
    if (!o) { fprintf(stderr, "frame_merge: cannot write %s\n", out_path.c_str()); return 1; }
    uint64_t hl = 4 + (uint64_t)fens.size() + entries.size(), ml = mv.size(), el = ev.size();
    fwrite(&hl, 8, 1, o);
    fwrite(&nf, 4, 1, o);
    fwrite(fens.data(), 1, fens.size(), o);
    fwrite(entries.data(), 1, entries.size(), o);
    fwrite(&ml, 8, 1, o); fwrite(mv.data(), 1, mv.size(), o);
    fwrite(&el, 8, 1, o); fwrite(ev.data(), 1, ev.size(), o);
    fclose(o);
    fprintf(stderr, "frame_merge: %zu frames -> %s | games=%llu fens=%llu moves=%llu (%.1fM pos)\n",
            inputs.size(), out_path.c_str(), (unsigned long long)games_total,
            (unsigned long long)nfens_total, (unsigned long long)(ml / 2),
            (double)np_total / 1e6);
    return 0;
}

#elif defined(LUMINEX_TOOL_VERIFY_FRAMES)

// verify_frames — integrity check for lc0pack gamepack frames, mirroring the
// FEATURIZER's parser exactly (the consumer is the spec):
//   frame: [u64 hdr_n][hdr][u64 mv_n][mv][u64 ev_n][ev]
//   hdr:   u32 nfens; nfens x (u16 len + fen bytes); then 9-byte game entries
//          [u16 n_pos][u8 stm][u32 fen_idx][i16 start_eval] while >=9 bytes remain
//   mv:    2 bytes/pos (RAW mode), per-game offsets from entries (mv_n may have
//          small orphan tail from anomalously-encoded games the consumer skips)
//   ev:    per game (n_pos-1) codes: int8 delta, or 0x80 + i16 abs
// Fails on: xz corruption, size-closure, fen/entry field violations, move-flag
// nibble 6/7 (unused in our MoveFlag enum), eval-range violations >0.1%.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
  #define popen  _popen   // MSVC POSIX names
  #define pclose _pclose
#endif

typedef struct {
    uint64_t games, positions, evals;
    uint64_t h[5];        // <=100, <=300, <=1000, >1000, zero
    uint64_t range_bad, mv_left, ev_left, slack_bytes;
} Stats;

static int verify_one(const char* path, Stats* st, char* err, size_t errsz) {
    char cmd[4096];
    snprintf(cmd, sizeof cmd, "xz -dc '%s'", path);
    FILE* p = popen(cmd, "r");
    if (!p) { snprintf(err, errsz, "popen failed"); return 1; }
    size_t cap = 1 << 22, len = 0;
    unsigned char* buf = (unsigned char*)malloc(cap);
    if (!buf) { pclose(p); snprintf(err, errsz, "oom"); return 1; }
    size_t r;
    while ((r = fread(buf + len, 1, cap - len, p)) > 0) {
        len += r;
        if (len == cap) {
            cap *= 2;
            unsigned char* nb = (unsigned char*)realloc(buf, cap);
            if (!nb) { free(buf); pclose(p); snprintf(err, errsz, "oom"); return 1; }
            buf = nb;
        }
    }
    int pc = pclose(p);
    if (pc != 0) { free(buf); snprintf(err, errsz, "xz failed rc=%d (corrupt/truncated)", pc); return 1; }

    uint64_t hdr_n, mv_n, ev_n;
    if (len < 8) { free(buf); snprintf(err, errsz, "too small"); return 1; }
    memcpy(&hdr_n, buf, 8);
    if (8 + hdr_n + 8 > len) { free(buf); snprintf(err, errsz, "hdr_n %llu overflows", (unsigned long long)hdr_n); return 1; }
    memcpy(&mv_n, buf + 8 + hdr_n, 8);
    if (8 + hdr_n + 8 + mv_n + 8 > len) { free(buf); snprintf(err, errsz, "mv_n %llu overflows", (unsigned long long)mv_n); return 1; }
    memcpy(&ev_n, buf + 8 + hdr_n + 8 + mv_n, 8);
    if (8 + hdr_n + 8 + mv_n + 8 + ev_n != len) {
        free(buf);
        snprintf(err, errsz, "size closure: 8+%llu+8+%llu+8+%llu != %zu",
                 (unsigned long long)hdr_n, (unsigned long long)mv_n,
                 (unsigned long long)ev_n, len);
        return 1;
    }
    const unsigned char* h  = buf + 8;
    const unsigned char* mv = buf + 8 + hdr_n + 8;
    const unsigned char* ev = buf + 8 + hdr_n + 8 + mv_n + 8;

    uint32_t nfens; memcpy(&nfens, h, 4);
    size_t off = 4;
    for (uint32_t i = 0; i < nfens; i++) {
        if (off + 2 > hdr_n) { free(buf); snprintf(err, errsz, "fen table overflow"); return 1; }
        uint16_t fl; memcpy(&fl, h + off, 2);
        off += 2 + fl;
        if (off > hdr_n) { free(buf); snprintf(err, errsz, "fen table overflow"); return 1; }
    }
    size_t games_bytes = hdr_n - off;
    size_t ngames = games_bytes / 9;
    st->slack_bytes = games_bytes % 9;
    if (mv_n % 2 != 0) { free(buf); snprintf(err, errsz, "mv_n odd"); return 1; }

    size_t ecur = 0, mcur = 0;
    uint64_t range_bad = 0;
    for (size_t g = 0; g < ngames; g++) {
        const unsigned char* e = h + off + g * 9;
        uint16_t np;   memcpy(&np, e, 2);
        uint8_t  stm = e[2];
        uint32_t fidx; memcpy(&fidx, e + 3, 4);
        int16_t  se;   memcpy(&se, e + 7, 2);
        if (np == 0)       { free(buf); snprintf(err, errsz, "game %zu np=0", g); return 1; }
        if (stm > 1)       { free(buf); snprintf(err, errsz, "game %zu stm=%u", g, stm); return 1; }
        if (fidx >= nfens) { free(buf); snprintf(err, errsz, "game %zu fidx OOB", g); return 1; }
        if (se < -5000 || se > 5000) range_bad++;
        for (int k = 0; k < (int)np; k++) {
            uint16_t m; memcpy(&m, mv + mcur, 2); mcur += 2;
            uint16_t fn = (m >> 12) & 0xF;
            if (fn == 6 || fn == 7) { free(buf); snprintf(err, errsz, "game %zu move %04X bad flag", g, m); return 1; }
        }
        int curev = se;
        int a = curev < 0 ? -curev : curev;
        st->h[a <= 100 ? 0 : a <= 300 ? 1 : a <= 1000 ? 2 : 3]++;
        if (curev == 0) st->h[4]++;
        for (int k = 0; k < (int)np - 1; k++) {
            if (ecur >= ev_n) { free(buf); snprintf(err, errsz, "ev overrun game %zu", g); return 1; }
            uint8_t c = ev[ecur];
            if (c == 0x80) {
                if (ecur + 3 > ev_n) { free(buf); snprintf(err, errsz, "ev abs overrun game %zu", g); return 1; }
                int16_t abs16; memcpy(&abs16, ev + ecur + 1, 2);
                curev = abs16; ecur += 3;
            } else {
                curev += (int8_t)c; ecur += 1;
            }
            if (curev < -5000 || curev > 5000) range_bad++;
            a = curev < 0 ? -curev : curev;
            st->h[a <= 100 ? 0 : a <= 300 ? 1 : a <= 1000 ? 2 : 3]++;
            if (curev == 0) st->h[4]++;
        }
        st->games++;
        st->positions += (uint64_t)np;   // consumer emits n_pos positions per game
    }
    if (mcur > mv_n) { free(buf); snprintf(err, errsz, "move overrun %zu > %llu", mcur, (unsigned long long)mv_n); return 1; }
    if (ecur > ev_n) { free(buf); snprintf(err, errsz, "ev walk overrun %zu > %llu", ecur, (unsigned long long)ev_n); return 1; }
    st->mv_left = mv_n - mcur;
    st->ev_left = ev_n - ecur;
    st->evals = st->positions;           // one eval per emitted position (start_eval + np-1 deltas)
    st->range_bad = range_bad;
    free(buf);
    return 0;
}

int main(int argc, char** argv) {
    Stats tot; memset(&tot, 0, sizeof tot);
    int bad = 0;
    for (int i = 1; i < argc; i++) {
        Stats st; memset(&st, 0, sizeof st);
        char err[256] = "";
        if (verify_one(argv[i], &st, err, sizeof err)) {
            printf("FAIL %s: %s\n", argv[i], err);
            bad++;
        } else {
            uint64_t e = st.h[0] + st.h[1] + st.h[2] + st.h[3];
            int range_ok = st.range_bad * 1000 <= e;
            if (range_ok) {
                printf("PASS %s: %llu games %lluM pos | <=100 %.1f%% 100-300 %.1f%% 300-1000 %.1f%% >1000 %.1f%% | rbad %llu mvleft %llu evleft %llu\n",
                       argv[i], (unsigned long long)st.games, (unsigned long long)(st.positions / 1000000),
                       100.0 * st.h[0] / e, 100.0 * st.h[1] / e, 100.0 * st.h[2] / e, 100.0 * st.h[3] / e,
                       (unsigned long long)st.range_bad, (unsigned long long)st.mv_left, (unsigned long long)st.ev_left);
                tot.games += st.games; tot.positions += st.positions;
                for (int k = 0; k < 5; k++) tot.h[k] += st.h[k];
                tot.range_bad += st.range_bad; tot.mv_left += st.mv_left; tot.ev_left += st.ev_left;
                tot.slack_bytes += st.slack_bytes;
            } else {
                printf("FAIL %s: eval range violations %llu / %llu (>0.1%%)\n", argv[i],
                       (unsigned long long)st.range_bad, (unsigned long long)e);
                bad++;
            }
        }
        fflush(stdout);
    }
    uint64_t e = tot.h[0] + tot.h[1] + tot.h[2] + tot.h[3];
    printf("TOTAL %d frames: %llu games, %llu positions, %llu evals | <=100cp %.2f%% | 100-300 %.2f%% | 300-1000 %.2f%% | >1000cp %.2f%% | zero %.2f%% | BAD %d\n",
           argc - 1, (unsigned long long)tot.games, (unsigned long long)tot.positions,
           (unsigned long long)e,
           100.0 * tot.h[0] / e, 100.0 * tot.h[1] / e, 100.0 * tot.h[2] / e,
           100.0 * tot.h[3] / e, 100.0 * tot.h[4] / e, bad);
    printf("AGGREGATE: range_bad %llu | orphan mv bytes %llu | orphan ev bytes %llu | slack bytes %llu\n",
           (unsigned long long)tot.range_bad, (unsigned long long)tot.mv_left,
           (unsigned long long)tot.ev_left, (unsigned long long)tot.slack_bytes);
    return bad ? 2 : 0;
}

#else
#error "tools.cpp builds one tool at a time: define one LUMINEX_TOOL_ macro (CMake does)"
#endif
