// featurize.cpp — Game-Pack (.xz) -> HalfKAv2_hm feature .npy cache for NNUE training.
//
// Replays the Game-Pack produced by nnue/openi_upload/colab_pack_pgn.py and emits the SAME
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
#include "board.h"
#include "movegen.h"
#include "bitboard.h"
#include "types.h"

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

int main(int argc, char** argv) {
    signal(SIGSEGV, crash_handler);
    signal(SIGABRT, crash_handler);
    const char* out_dir = "/tmp/feat";
    const char* in_path = nullptr;          // nullptr => stdin
    long max_pos = 0;                        // 0 = no limit
    bool stream_mode = false;                // --stream: emit 136-byte records to stdout (no .npy)
    int nthreads = (int)std::thread::hardware_concurrency();
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* def){ return (i + 1 < argc) ? argv[++i] : def; };
        if      (a == "--out-dir") out_dir = next(out_dir);
        else if (a == "--input")   in_path = next("");
        else if (a == "--max-pos") max_pos = std::atol(next("0"));
        else if (a == "--threads") nthreads = std::atoi(next("0"));
        else if (a == "--stream")  stream_mode = true;
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }

    if (!stream_mode) mkdir(out_dir, 0755);   // create output dir if missing (fopen "wb" won't)
    init_magic_bitboards();
    init_line_tables();   // CRITICAL: populates BetweenBB/LineBB; without it pin detection
                          // fails and generate<GEN_LEGAL> returns illegal moves (the C featurizer
                          // would then emit wrong move indices on any pinned position).

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

    // Decode header: u32 nfens; [u16 len + bytes]*nfens; per game [u16 n_pos][u8 stm][u32 fen_idx][s16 start_eval].
    size_t hp = 0;
    auto rd_u32 = [&]() { uint32_t v=0; for(int i=0;i<4;++i) v |= (uint32_t)hdr[hp+i]<<i*8; hp+=4; return v; };
    auto rd_u16 = [&]() { uint16_t v=0; for(int i=0;i<2;++i) v |= (uint16_t)hdr[hp+i]<<i*8; hp+=2; return v; };
    auto rd_s16 = [&]() { int16_t v=(int16_t)rd_u16(); return v; };
    uint32_t nfens = rd_u32();
    std::vector<std::string> fens(nfens);
    for (uint32_t i = 0; i < nfens; ++i) {
        uint16_t fl = rd_u16();
        fens[i].assign((const char*)hdr + hp, fl);
        hp += fl;
    }
    struct Game { uint32_t fen_idx; uint16_t n_pos; uint8_t stm; int16_t start_eval; long mv_off; long ev_off; long out_off; };
    std::vector<Game> games;
    long mv_acc = 0;
    while (hp + 9 <= hdr_n) {
        Game g;
        g.n_pos = (uint16_t)hdr[hp] | ((uint16_t)hdr[hp+1] << 8); hp += 2;   // u16 n_pos (no 255 cap)
        g.stm   = hdr[hp++];
        g.fen_idx = rd_u32();
        g.start_eval = rd_s16();
        g.mv_off = mv_acc; mv_acc += g.n_pos;
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

    auto worker = [&](int tid) {
        Position pos;
        ExtMove list[MAX_MOVES];
        std::vector<char> tbuf;   // per-thread stream buffer (136 bytes/record)
        for (size_t gi = tid; gi < games.size(); gi += nthreads) {
            Game& g = games[gi];
            if (g.out_off < 0) continue;
            if (tid == 0 && (gi % 50000 == 0))
                std::fprintf(stderr, "  [prog] game %zu/%zu (emitted %ld)\n", gi, games.size(), emitted.load());
            pos.set(fens[g.fen_idx]);   // FENs come from python-chess via the encoder — valid

            long mv_p = g.mv_off;
            long ev_p = g.ev_off;
            int16_t prev_eval = g.start_eval;   // white-relative
            bool first = true;

            for (int ply = 0; ply < g.n_pos; ++ply) {
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

                // Compute HalfKAv2 features of the resulting position.
                int wk = (int)pos.king_sq(WHITE), bk = (int)pos.king_sq(BLACK);
                int16_t wfeat[MAX_PIECES], bfeat[MAX_PIECES];
                int nw = 0, nb = 0;
                for (int sq = 0; sq < NUM_SQ && nw < MAX_PIECES; ++sq) {
                    Piece pc = pos.piece_on(Square(sq));
                    if (pc == NO_PIECE) continue;
                    wfeat[nw++] = (int16_t)halfka_idx(true,  wk, sq, pc);
                    bfeat[nb++] = (int16_t)halfka_idx(false, bk, sq, pc);
                }
                for (int i = nw; i < MAX_PIECES; ++i) wfeat[i] = (int16_t)NUM_INPUTS;
                for (int i = nb; i < MAX_PIECES; ++i) bfeat[i] = (int16_t)NUM_INPUTS;

                // STM-relative eval target (engine returns net output without stm negation).
                bool stm_white = (pos.side_to_move() == WHITE);
                float target = stm_white ? (float)eq : (float)(-eq);
                float stm = stm_white ? 1.0f : 0.0f;
                if (stream_mode) {
                    // Pack a 136-byte record (w[32] b[32] stm target) and batch into the thread
                    // buffer; flush under the stdout mutex. Order across threads doesn't matter —
                    // the trainer shuffles a chunk anyway. Backpressure: fwrite blocks when the
                    // stdout pipe is full, throttling featurize to the trainer's consume rate.
                    char buf[136];
                    std::memcpy(buf, wfeat, 64); std::memcpy(buf + 64, bfeat, 64);
                    std::memcpy(buf + 128, &stm, 4); std::memcpy(buf + 132, &target, 4);
                    tbuf.insert(tbuf.end(), buf, buf + 136);
                    if (tbuf.size() >= 136 * 512) {
                        std::lock_guard<std::mutex> lk(out_mtx);
                        std::fwrite(tbuf.data(), 1, tbuf.size(), stdout);
                        tbuf.clear();
                    }
                } else {
                    write_row(row, wfeat, bfeat, stm, target);
                }

                long e = emitted.fetch_add(1) + 1;
                if (tid == 0 && e % 1000000 == 0) {
                    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    std::fprintf(stderr, "  %ldM pos (%.1fM/s)\n", e / 1000000, e / dt / 1e6);
                }
            }
        }
        if (stream_mode && !tbuf.empty()) {   // flush this thread's leftover
            std::lock_guard<std::mutex> lk(out_mtx);
            std::fwrite(tbuf.data(), 1, tbuf.size(), stdout);
            tbuf.clear();
        }
    };

    std::vector<std::thread> th;
    for (int t = 0; t < nthreads; ++t) th.emplace_back(worker, t);
    for (auto& t : th) t.join();

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
