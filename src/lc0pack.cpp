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
    if (!(q > -0.99999f && q < 0.99999f) || std::isnan(q)) return 1e9; // sentinel: skip
    return 800.0 * 0.5 * std::log((1.0 + q) / (1.0 - q));
}

static inline bool rec_stm_black(const V6Rec& r) {
    return r.input_format >= 3 ? (r.invariance_info & 0x80) != 0
                               : (r.side_to_move_or_enpassant != 0);
}

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
    if (n < 8) { pt.skipped++; return; }            // too short to be worth packing
    const V6Rec* recs = reinterpret_cast<const V6Rec*>(dec.data());

    // evals first: all must be valid
    std::vector<int> eqs(n);
    for (size_t i = 0; i < n; ++i) {
        double cp = q_to_cp(recs[i].root_q);
        if (cp > 1e8) { pt.skipped++; return; }
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
    ExtMove list[MAX_MOVES];
    std::vector<uint16_t> mvs; mvs.reserve(n);
    // precompute each record's true placement as 12 bitboards (white-perspective)
    std::vector<std::array<uint64_t, 12>> boards(n);
    for (size_t i = 0; i < n; ++i) decode_boards(recs[i], boards[i].data());

    for (size_t i = 0; i + 1 < n; ++i) {
        ExtMove* end = generate<GEN_LEGAL>(pos, list);
        int nlegal = (int)(end - list);
        // board-delta move derivation: find the legal whose application yields the
        // next record's placement (immune to policy-index space mysteries)
        Move chosen = MOVE_NONE;
        for (int k = 0; k < nlegal; ++k) {
            Position trial = pos;
            if (!trial.do_move_replay(list[k].move)) continue;
            bool ok = true;
            for (int pt = 0; pt < 6 && ok; ++pt) {
                if (trial.pieces((Color)WHITE, (PieceType)pt) != boards[i + 1][pt] ||
                    trial.pieces((Color)BLACK, (PieceType)pt) != boards[i + 1][6 + pt]) ok = false;
            }
            if (ok) { chosen = list[k].move; break; }
        }
        if (debug && i < 3) fprintf(stderr, "[dbg] ply %zu: board-match=%s nlegal=%d\n", i, chosen == MOVE_NONE ? "FAIL" : "ok", nlegal);
        if (chosen == MOVE_NONE) { pt.skipped++; return; }   // divergence
        mvs.push_back(chosen.raw());
        if (!pos.do_move_replay(chosen)) { pt.skipped++; return; }
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
    out.clear();
    zs.next_in = const_cast<Bytef*>(data); zs.avail_in = size;
    std::vector<uint8_t> buf(1 << 22);
    int rc = Z_OK;
    while (rc != Z_STREAM_END) {
        zs.next_out = buf.data(); zs.avail_out = buf.size();
        rc = inflate(&zs, Z_NO_FLUSH);
        if (rc != Z_OK && rc != Z_STREAM_END) { inflateEnd(&zs); return false; }
        out.insert(out.end(), buf.data(), buf.data() + (buf.size() - zs.avail_out));
        if (zs.avail_in == 0 && rc != Z_STREAM_END) break;
    }
    inflateEnd(&zs);
    return rc == Z_STREAM_END;
}

struct ChunkFile { std::vector<uint8_t> gz; };

static std::vector<ChunkFile> expand_tar(const std::string& path) {
    std::vector<ChunkFile> out;
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) return out;
    std::vector<uint8_t> tar; { std::vector<uint8_t> buf(1 << 20); size_t n; while ((n = fread(buf.data(), 1, buf.size(), fp)) > 0) tar.insert(tar.end(), buf.begin(), buf.begin() + n); }
    fclose(fp);
    if (tar.size() >= 2 && tar[0] == 0x1f && tar[1] == 0x8b) { out.push_back({std::move(tar)}); return out; }
    size_t off = 0;
    while (off + 512 <= tar.size()) {
        const uint8_t* hdr = tar.data() + off;
        if (hdr[0] == 0) break;
        size_t sz = 0; for (int k = 0; k < 11; ++k) sz = sz * 8 + (hdr[124 + k] - '0');
        size_t nd = off + 512 + sz;
        if (sz > 0 && nd <= tar.size()) out.push_back({std::vector<uint8_t>(tar.begin() + off + 512, tar.begin() + nd)});
        off = (nd + 511) & ~size_t(511);
    }
    return out;
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

    // expand all tars up front (memory: ~1 tar at a time is the intended usage)
    std::vector<ChunkFile> chunks;
    for (auto& f : inputs) {
        auto cs = expand_tar(f);
        for (auto& c : cs) chunks.push_back(std::move(c));
    }
    fprintf(stderr, "lc0pack: %zu chunks from %zu files\n", chunks.size(), inputs.size());

    Shared sh;
    std::vector<PerThread> pts(nthreads);
    std::atomic<size_t> next{0};
    std::vector<uint8_t> dec;
    auto worker = [&](int tid) {
        PerThread& pt = pts[tid];
        std::vector<uint8_t> dec;   // PER-THREAD buffer (shared one = data race)
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
