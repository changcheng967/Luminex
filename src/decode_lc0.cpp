// decode_lc0.cpp — Leela v6 training chunks (.tar of .gz, headerless 8356B records)
// -> "fen\tcp" lines on stdout, for luminex-featurize --fen-eval.
//
// Format (verified against LeelaChessZero/lc0 src/trainingdata/trainingdata_v6.h):
//   record = 8356 bytes packed:
//     u32 version, u32 input_format, float probabilities[1858], u64 planes[104],
//     u8 castling_us_ooo/oo, castling_them_ooo/oo, side_to_move_or_enpassant,
//     u8 rule50, u8 invariance_info, u8 dummy,
//     float root_q best_q root_d best_d root_m best_m plies_left result_q result_d
//           played_q played_d played_m orig_q orig_d orig_m,
//     u32 visits, u16 played_idx best_idx, float policy_kld q_st
//   planes[0..11] = P N B R Q K "ours" then "theirs" (u64 bitboards, a1=bit0)
//   canonical formats (input_format >= 3): planes are WHITE-perspective, possibly
//   transformed; invariance_info bits: 0=flip(files) 1=mirror(ranks) 2=transpose
//   3=best_q proven 7=side-to-move(1=black). Transforms are involutions -> re-apply
//   to recover the true position.
//   label: root_q is side-to-move Q in [-1,1] -> cp = 111.1 * logit(Q).
//
// Usage: decode_lc0 [--q best|root] file.tar [more.tar | dir of .gz ...]
//   (also accepts bare .gz chunk files). Multi-threaded over files.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <zlib.h>

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

static std::atomic<long> g_out{0}, g_skip{0};
static std::mutex g_out_mtx;
static bool g_use_best_q = false;

static inline uint64_t flip_files(uint64_t b) {   // mirror each rank byte's bits
    static uint64_t rev[256]; static bool init = false;
    if (!init) { for (int i = 0; i < 256; ++i) { uint64_t r = 0; for (int j = 0; j < 8; ++j) if (i >> j & 1) r |= 1ULL << (7 - j); rev[i] = r; } init = true; }
    return (rev[b & 0xFF]) | (rev[(b >> 8) & 0xFF] << 8) | (rev[(b >> 16) & 0xFF] << 16) |
           (rev[(b >> 24) & 0xFF] << 24) | (rev[(b >> 32) & 0xFF] << 32) |
           (rev[(b >> 40) & 0xFF] << 40) | (rev[(b >> 48) & 0xFF] << 48) |
           (rev[(b >> 56) & 0xFF] << 56);
}
static inline uint64_t mirror_ranks(uint64_t b) { // swap byte order
    return __builtin_bswap64(b);
}
static inline uint64_t transpose_bb(uint64_t b) { // (file,rank) -> (rank,file)
    uint64_t r = 0;
    for (int sq = 0; sq < 64; ++sq) if (b >> sq & 1) {
        int f = sq & 7, rk = sq >> 3;
        r |= 1ULL << (f * 8 + rk);
    }
    return r;
}

static void emit_record(const V6Rec& r, std::string& out) {
    float q = g_use_best_q ? r.best_q : r.root_q;
    if (!(q > -0.99999f && q < 0.99999f) || std::isnan(q)) { g_skip++; return; }
    // Faithful sigmoid-space map: t = 800*atanh(q) makes sigma(t/400) == (1+q)/2
    // exactly — the trainer's loss operates in sigmoid space, so this preserves
    // Leela's Q ordering losslessly. The 111.1 display convention compresses
    // decisive positions (q=0.998 -> 383cp); this map sends them to the +-1500
    // clamp where material swings belong.
    double cp = 800.0 * 0.5 * std::log((1.0 + q) / (1.0 - q));
    if (cp > 1500) cp = 1500; if (cp < -1500) cp = -1500;

    uint64_t bb[12];
    for (int i = 0; i < 12; ++i) bb[i] = r.planes[i];
    bool canonical = r.input_format >= 3;
    if (canonical) {
        // un-apply the recorded transforms (each is its own inverse)
        if (r.invariance_info & 0x4) for (auto& b : bb) b = transpose_bb(b);
        if (r.invariance_info & 0x2) for (auto& b : bb) b = mirror_ranks(b);
        if (r.invariance_info & 0x1) for (auto& b : bb) b = flip_files(b);
    } else {
        // non-canonical: planes are side-to-move perspective; if black to move
        // (aux bit in old formats is not available in v6 fields) — v6 non-canonical
        // is not expected in current data; treat as white-perspective.
    }
    // canonical planes are WHITE-perspective: bb[0..5]=white P..K, bb[6..11]=black.
    static const char W[6] = {'P','N','B','R','Q','K'};
    static const char B[6] = {'p','n','b','r','q','k'};
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            int sq = rank * 8 + file;
            char pc = 0;
            for (int p = 0; p < 6; ++p) {
                if ((bb[p] >> sq) & 1) { pc = W[p]; break; }
                if ((bb[6 + p] >> sq) & 1) { pc = B[p]; break; }
            }
            if (pc) { if (empty) { out += std::to_string(empty); empty = 0; } out += pc; }
            else ++empty;
        }
        if (empty) out += std::to_string(empty);
        if (rank) out += '/';
    }
    // side to move: canonical formats keep it in invariance bit 7 (1 = black)
    bool black = canonical ? (r.invariance_info & 0x80) != 0
                           : (r.side_to_move_or_enpassant != 0);
    out += black ? " b " : " w ";
    // castling from structured fields ("us"=white in canonical); safest minimal form
    std::string cast;
    if (r.castling_us_oo) cast += 'K';
    if (r.castling_us_ooo) cast += 'Q';
    if (r.castling_them_oo) cast += 'k';
    if (r.castling_them_ooo) cast += 'q';
    out += cast.empty() ? "-" : cast;
    out += " - 0 1\t";
    out += std::to_string((long long)llround(cp));
    out += '\n';
    if (out.size() > (1 << 22)) {
        std::lock_guard<std::mutex> lk(g_out_mtx);
        fwrite(out.data(), 1, out.size(), stdout);
        out.clear();
    }
    g_out++;
}

static void process_gz_buffer(const std::vector<uint8_t>& raw, std::string& out) {
    // raw = full decompressed chunk: concatenated 8356B records
    size_t n = raw.size() / sizeof(V6Rec);
    for (size_t i = 0; i < n; ++i)
        emit_record(reinterpret_cast<const V6Rec&>(raw[i * sizeof(V6Rec)]), out);
}

static bool gz_decompress(const uint8_t* data, size_t size, std::vector<uint8_t>& out) {
    z_stream zs{}; 
    if (inflateInit2(&zs, 16 + MAX_WBITS) != Z_OK) return false;
    out.clear(); out.reserve(size * 6);
    zs.next_in = const_cast<Bytef*>(data); zs.avail_in = size;
    std::vector<uint8_t> buf(1 << 22);
    int rc = Z_OK;
    while (rc != Z_STREAM_END) {
        zs.next_out = buf.data(); zs.avail_out = buf.size();
        rc = inflate(&zs, Z_NO_FLUSH);
        if (rc != Z_OK && rc != Z_STREAM_END) { inflateEnd(&zs); return false; }
        out.insert(out.end(), buf.data(), buf.data() + (buf.size() - zs.avail_out));
        if (zs.avail_in == 0 && rc != Z_STREAM_END) { inflateEnd(&zs); return false; }
    }
    inflateEnd(&zs);
    return true;
}

struct Job { std::string name; std::vector<uint8_t> data; };

int main(int argc, char** argv) {
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--q" && i + 1 < argc) { g_use_best_q = (strcmp(argv[++i], "best") == 0); continue; }
        files.push_back(a);
    }
    if (files.empty()) { fprintf(stderr, "usage: decode_lc0 [--q best|root] files...(tar/gz)\n"); return 2; }

    std::vector<Job> jobs; jobs.reserve(4096);
    // expand: tars -> member gz files; gz files -> themselves
    for (auto& f : files) {
        FILE* fp = fopen(f.c_str(), "rb");
        if (!fp) { fprintf(stderr, "skip %s (open)\n", f.c_str()); continue; }
        std::vector<uint8_t> tar;
        { std::vector<uint8_t> buf(1 << 20); size_t n; while ((n = fread(buf.data(), 1, buf.size(), fp)) > 0) tar.insert(tar.end(), buf.begin(), buf.begin() + n); }
        fclose(fp);
        if (tar.size() >= 2 && tar[0] == 0x1f && tar[1] == 0x8b) {
            jobs.push_back({f, std::move(tar)});           // a gz chunk directly
        } else if (tar.size() > 512) {                      // a tar of gz chunks
            size_t off = 0;
            while (off + 512 <= tar.size()) {
                const uint8_t* hdr = tar.data() + off;
                if (hdr[0] == 0) break;
                char name[100]; memcpy(name, hdr, 99); name[99] = 0;
                size_t sz = 0;
                for (int k = 0; k < 11; ++k) sz = sz * 8 + (hdr[124 + k] - '0');
                size_t nd = off + 512 + sz;
                if (sz > 0 && nd <= tar.size() && name[0] != '\0')
                    jobs.push_back({std::string(name), std::vector<uint8_t>(tar.begin() + off + 512, tar.begin() + nd)});
                off = (nd + 511) & ~size_t(511);
            }
        }
    }
    fprintf(stderr, "decode_lc0: %zu chunks, %lld MB\n", jobs.size(),
            (long long)(jobs.empty() ? 0 : [&]{ long long s=0; for (auto&j : jobs) s += j.data.size(); return s; }() / 1048576));

    std::atomic<size_t> next{0};
    auto worker = [&] {
        std::string out; out.reserve(1 << 22);
        std::vector<uint8_t> dec;
        while (true) {
            size_t i = next.fetch_add(1);
            if (i >= jobs.size()) break;
            if (gz_decompress(jobs[i].data.data(), jobs[i].data.size(), dec))
                process_gz_buffer(dec, out);
        }
        std::lock_guard<std::mutex> lk(g_out_mtx);
        fwrite(out.data(), 1, out.size(), stdout);
    };
    unsigned nt = std::thread::hardware_concurrency(); if (nt > 8) nt = 8; if (nt < 1) nt = 1;
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nt; ++t) th.emplace_back(worker);
    for (auto& t : th) t.join();
    fflush(stdout);
    fprintf(stderr, "decode_lc0: emitted %ld positions (skipped %ld)\n", g_out.load(), g_skip.load());
    return 0;
}
