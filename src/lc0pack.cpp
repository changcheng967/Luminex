// lc0pack.cpp — Leela v6 training chunks (.tar of .gz) -> Luminex GAMEPACK frame.
// Byte-compatible with luminex-encode output (featurize --stream consumes unchanged).
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
#include "board.h"
#include "movegen.h"
#include "bitboard.h"
#include "types.h"
#include "lc0_move_table.h"

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
    pt.game_entries.push_back((uint8_t)(np & 0xFF));
    pt.game_entries.push_back((uint8_t)((np >> 8) & 0xFF));
    pt.game_entries.push_back(rec_stm_black(recs[0]) ? 0 : 1);
    pt.game_entries.insert(pt.game_entries.end(), (uint8_t*)&fidx, (uint8_t*)&fidx + 4);
    pt.game_entries.insert(pt.game_entries.end(), (uint8_t*)&start_eval, (uint8_t*)&start_eval + 2);
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
    init_line_tables();
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
