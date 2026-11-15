// binfeed.cpp — training binpacks held in host RAM and decoded on the CPU into LCR1 records,
// for resident_train.py's BinpackFeeder (NNUE_BINPACKS).
//
// A binpack costs ~2.6 bytes a position against an LCR1 record's 32, so all of t91's BT4
// months (8.56B usable positions, 26.8 GB of binpacks) fit in host RAM, where 32-byte records
// capped the trainer at ~6.1B (2.3B in VRAM + 3.8B in host RAM). The files are read once into
// memory and their chunks ("BINP", uint32 size, data: self-contained, ~1 MiB each) are decoded
// with nnue-pytorch's binpack reader, as binpack2lcr.cpp does: the records are binpack2lcr's,
// field for field (check_binfeed.py compares the two on a month's file).
//
// Three uses:
//  * the index pass (bf_index): every chunk decoded once, to count its training positions and
//    collect the held-out ones (poshash below the threshold: the trainer's validation split by
//    hash, NNUE_VAL_FRAC). The counts and held-out records are cached next to each binpack
//    (<file>.bfx, keyed by the file's size and a hash of its bytes), so a second run skips it;
//  * the stream (bf_stream_start, bf_next): pass after pass, the chunks in a pseudo-random
//    order keyed by (seed, pass), each decoded by one of several threads and handed out in that
//    order, so the stream is the same for a given seed however the threads run. Held-out and
//    VALUE_NONE positions are left out, so each pass yields every training position once;
//  * bf_bench and bf_sort, for check_binfeed.py: the decode rate at a thread count, and a
//    canonical order of records (the hash of the sorted records is the check's fingerprint).
//
// Build (against a checkout of official-stockfish/nnue-pytorch, as binpack2lcr):
//   g++ -O3 -march=native -std=c++20 -shared -fPIC -pthread -I <nnue-pytorch>/data_loader/cpp \
//       binfeed.cpp -o libbinfeed.so
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "lib/binpack.h"

namespace {

struct CompactRec {   // src/featurize.cpp CompactRec, as binpack2lcr.cpp writes it
    std::uint64_t occ;
    std::uint8_t pcs[16];
    std::int16_t target;
    std::uint8_t flags;
    std::uint8_t npieces;
    std::uint32_t poshash;
};
static_assert(sizeof(CompactRec) == 32, "LCR1 records are 32 bytes");

constexpr int VALUE_NONE = 32002;   // a producer's "skip this position" (binpack2lcr.cpp)
constexpr std::uint64_t PIECE = 64ull << 20;   // bytes per read and per hashed piece

inline std::uint64_t mix64(std::uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

inline std::uint64_t splitmix(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// The chunk order of a pass: a 4-round Feistel bijection on an even number of bits >= log2(n),
// keyed by (seed, pass), walking cycles past values >= n (feeder.cpp's construction).
struct ChunkOrder {
    std::uint64_t ks[4];
    unsigned h;
    std::uint64_t mask, n;

    ChunkOrder(std::uint64_t seed, std::uint64_t pass, std::uint64_t n_) : n(n_) {
        unsigned bits = 2;
        while (bits < 64 && (1ull << bits) < n) ++bits;
        bits += bits & 1;
        h = bits / 2;
        mask = (1ull << h) - 1;
        std::uint64_t s = seed * 0xD1B54A32D192ED03ULL ^ (pass + 1) * 0x9E3779B97F4A7C15ULL ^ 0xB1AF0EEDULL;
        for (auto& k : ks) k = splitmix(s);
    }
    std::uint64_t round(std::uint64_t r, std::uint64_t k) const {
        std::uint64_t x = (r ^ k) * 0x9E3779B97F4A7C15ULL;
        x ^= x >> 29;
        x *= 0xBF58476D1CE4E5B9ULL;
        x ^= x >> 32;
        return x & mask;
    }
    std::uint64_t perm(std::uint64_t x) const {
        std::uint64_t left = x >> h, right = x & mask;
        for (auto k : ks) {
            const std::uint64_t next = left ^ round(right, k);
            left = right;
            right = next;
        }
        return (left << h) | right;
    }
    std::uint64_t at(std::uint64_t i) const {
        std::uint64_t y = perm(i);
        while (y >= n) y = perm(y);
        return y;
    }
};

std::uint64_t hash_bytes(const std::uint8_t* p, std::size_t n, std::uint64_t h) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        std::uint64_t w;
        std::memcpy(&w, p + i, 8);
        h = (h ^ w) * 0x9E3779B97F4A7C15ULL;
        h ^= h >> 31;
    }
    for (; i < n; ++i) h = (h ^ p[i]) * 0x100000001B3ULL;
    return mix64(h ^ n);
}

struct Counts {
    std::uint32_t train = 0, val = 0, marked = 0;
};

struct Stats {   // over the usable (training + held-out) positions, as binpack2lcr's DONE line
    std::int64_t quiet = 0, capture = 0, check = 0, win = 0, draw = 0, loss = 0;
    void add(const Stats& o) {
        quiet += o.quiet; capture += o.capture; check += o.check;
        win += o.win; draw += o.draw; loss += o.loss;
    }
};

struct File {
    std::string path;
    std::uint8_t* data = nullptr;
    std::size_t size = 0;
    std::uint64_t hash = 0;
    std::size_t c0 = 0, c1 = 0;   // its chunks: [c0, c1) of Feeder::chunks
    bool cached = false;
    Stats stats;
};

struct Chunk {
    std::uint32_t file;
    std::uint32_t size;
    std::uint64_t off;   // offset of the chunk's data (after the 8-byte header) in the file
};

struct Slot {
    std::int64_t ticket = -1;
    bool ready = false;
    std::vector<CompactRec> recs;
    std::size_t pos = 0;
};

struct Feeder {
    std::vector<File> files;
    std::vector<Chunk> chunks;
    std::vector<Counts> counts;                  // per chunk (index pass or cache)
    std::vector<std::vector<CompactRec>> val;    // per chunk: its held-out records, in order
    std::uint64_t thr = 0;                       // held out: poshash < thr (2^32 holds out every position)
    bool use_cache = true;
    bool indexed = false;
    std::size_t active = 0;                      // the stream's chunks: [0, active) (bf_select)
    std::string error;

    // the stream
    std::uint64_t seed = 0;
    std::vector<std::thread> workers;
    std::vector<Slot> ring;
    std::mutex mu;
    std::condition_variable cv_prod, cv_cons;
    std::atomic<std::int64_t> next_ticket{0};
    std::int64_t consumed = 0;                   // the ticket the consumer reads (all below are handed out)
    bool stop = false;
    std::atomic<int> failed{0};
    // waits that found their condition true when a timed wait ran out, i.e. a wakeup that never
    // came (or came within the last instant): each costs WAIT_MS instead of hanging the stream
    std::atomic<std::int64_t> late_prod{0}, late_cons{0};
};

constexpr int WAIT_MS = 20;   // every wait re-checks its condition this often, woken or not

// cv.wait(lk, ready) that also wakes every WAIT_MS: a wakeup that never arrives (a lost futex
// wake in the platform's condition variable) then costs WAIT_MS instead of the stream, and is
// counted in `late`. The protocol itself loses none: every change of a condition is made under
// mu and followed by a notify.
template <class Ready>
void wait_until_ready(std::condition_variable& cv, std::unique_lock<std::mutex>& lk, Ready ready,
                      std::atomic<std::int64_t>& late) {
    while (!ready())
        if (cv.wait_for(lk, std::chrono::milliseconds(WAIT_MS)) == std::cv_status::timeout && ready())
            ++late;
}

std::mutex g_error_mu;
std::string g_error;   // the last failure's message (bf_last_error)

void set_error(std::string s) {
    std::lock_guard<std::mutex> lk(g_error_mu);
    g_error = std::move(s);
}

// One chunk -> records, binpack2lcr.cpp's conversion record for record. Positions scored
// VALUE_NONE are counted and left out; the rest go to `val` (if given) when their hash is below
// thr, else to `train` (if given). False on a position of more than 32 pieces.
bool decode_chunk(const std::uint8_t* p, std::uint32_t size, std::uint64_t thr, std::vector<unsigned char>& buf,
                  std::vector<CompactRec>* train, std::vector<CompactRec>* val, Counts& cnt, Stats* st) {
    buf.assign(p, p + size);
    binpack::ChunkReader reader;
    while (reader.hasNext(buf)) {
        const auto e = reader.next(buf);
        if (e.score == VALUE_NONE) { ++cnt.marked; continue; }
        CompactRec r;
        std::memset(&r, 0, sizeof r);
        r.occ = e.pos.piecesBB().bits();
        int k = 0;
        for (std::uint64_t bb = r.occ; bb; bb &= bb - 1, ++k) {
            const int sq = __builtin_ctzll(bb);
            const auto pc = e.pos.pieceAt(chess::Square(sq));
            const int code = static_cast<int>(chess::ordinal(pc.type())) + 6 * static_cast<int>(chess::ordinal(pc.color()));
            if (k < 32) r.pcs[k >> 1] |= static_cast<std::uint8_t>(code << (4 * (k & 1)));
        }
        if (k > 32) return false;
        const bool white = e.pos.sideToMove() == chess::Color::White;
        const int q = !e.pos.checkers().isEmpty() ? 2 : (e.isCapturingMove() ? 1 : 0);
        const int res = e.result > 0 ? 2 : (e.result < 0 ? 0 : 1);
        r.target = e.score;
        r.flags = static_cast<std::uint8_t>((white ? 1 : 0) | (q << 1) | (1 << 3) | (res << 4));
        r.npieces = static_cast<std::uint8_t>(k);
        std::uint64_t lo, hi;
        std::memcpy(&lo, r.pcs, 8);
        std::memcpy(&hi, r.pcs + 8, 8);
        r.poshash = static_cast<std::uint32_t>(mix64(r.occ ^ mix64(lo ^ mix64(hi ^ (white ? 1 : 0)))) >> 32);
        if (r.poshash < thr) {
            ++cnt.val;
            if (val) val->push_back(r);
        } else {
            ++cnt.train;
            if (train) train->push_back(r);
        }
        if (st) {
            st->quiet += q == 0; st->capture += q == 1; st->check += q == 2;
            st->loss += res == 0; st->draw += res == 1; st->win += res == 2;
        }
    }
    return true;
}

// Runs fn(i) for i in [0, n) on `threads` threads, each taking the next i.
template <typename Fn>
void parallel_for(std::size_t n, int threads, Fn fn) {
    std::atomic<std::size_t> next{0};
    auto run = [&] {
        for (std::size_t i; (i = next.fetch_add(1)) < n;) fn(i);
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < threads; ++t) pool.emplace_back(run);
    run();
    for (auto& th : pool) th.join();
}

// The cache next to a binpack: what the index pass found in it.
struct CacheHeader {
    char magic[8];
    std::uint64_t file_size, file_hash, thr, nchunks, nval;
    Stats stats;
};
constexpr char CACHE_MAGIC[8] = {'L', 'U', 'X', 'B', 'F', 'X', '1', 0};

std::string cache_path(const File& f) { return f.path + ".bfx"; }

bool read_cache(Feeder& F, File& f) {
    std::FILE* in = std::fopen(cache_path(f).c_str(), "rb");
    if (!in) return false;
    CacheHeader hd;
    const std::size_t nc = f.c1 - f.c0;
    bool ok = std::fread(&hd, sizeof hd, 1, in) == 1 && std::memcmp(hd.magic, CACHE_MAGIC, 8) == 0
              && hd.file_size == f.size && hd.file_hash == f.hash && hd.thr == F.thr && hd.nchunks == nc;
    std::vector<Counts> cnt(nc);
    std::vector<CompactRec> recs;
    if (ok) ok = std::fread(cnt.data(), sizeof(Counts), nc, in) == nc;
    std::uint64_t total = 0;
    for (const auto& c : cnt) total += c.val;
    if (ok) ok = total == hd.nval;
    if (ok) {
        recs.resize(hd.nval);
        ok = std::fread(recs.data(), sizeof(CompactRec), hd.nval, in) == hd.nval;
    }
    std::fclose(in);
    if (!ok) return false;
    std::size_t o = 0;
    for (std::size_t i = 0; i < nc; ++i) {
        F.counts[f.c0 + i] = cnt[i];
        F.val[f.c0 + i].assign(recs.begin() + o, recs.begin() + o + cnt[i].val);
        o += cnt[i].val;
    }
    f.stats = hd.stats;
    f.cached = true;
    return true;
}

bool write_cache(const Feeder& F, const File& f) {
    const std::string path = cache_path(f), tmp = path + ".tmp";
    std::FILE* out = std::fopen(tmp.c_str(), "wb");
    if (!out) return false;
    CacheHeader hd;
    std::memcpy(hd.magic, CACHE_MAGIC, 8);
    hd.file_size = f.size;
    hd.file_hash = f.hash;
    hd.thr = F.thr;
    hd.nchunks = f.c1 - f.c0;
    hd.nval = 0;
    for (std::size_t c = f.c0; c < f.c1; ++c) hd.nval += F.counts[c].val;
    hd.stats = f.stats;
    bool ok = std::fwrite(&hd, sizeof hd, 1, out) == 1
              && std::fwrite(F.counts.data() + f.c0, sizeof(Counts), hd.nchunks, out) == hd.nchunks;
    for (std::size_t c = f.c0; ok && c < f.c1; ++c)
        ok = std::fwrite(F.val[c].data(), sizeof(CompactRec), F.val[c].size(), out) == F.val[c].size();
    ok = std::fclose(out) == 0 && ok;
    if (ok) ok = std::rename(tmp.c_str(), path.c_str()) == 0;
    if (!ok) std::remove(tmp.c_str());
    return ok;
}

void worker(Feeder* F) {
    std::vector<unsigned char> buf;
    const std::uint64_t n = F->active;
    const std::int64_t R = static_cast<std::int64_t>(F->ring.size());
    std::uint64_t order_pass = ~0ull;
    ChunkOrder order(F->seed, 0, n);
    for (;;) {
        const std::int64_t t = F->next_ticket.fetch_add(1);
        Slot& s = F->ring[t % R];
        {
            std::unique_lock<std::mutex> lk(F->mu);   // wait until the slot's last ticket is handed out
            wait_until_ready(F->cv_prod, lk, [&] { return F->stop || t < F->consumed + R; }, F->late_prod);
            if (F->stop) return;
        }
        const std::uint64_t pass = static_cast<std::uint64_t>(t) / n;
        if (pass != order_pass) {
            order = ChunkOrder(F->seed, pass, n);
            order_pass = pass;
        }
        const Chunk& c = F->chunks[order.at(static_cast<std::uint64_t>(t) % n)];
        s.recs.clear();
        s.pos = 0;
        Counts cnt;
        if (!decode_chunk(F->files[c.file].data + c.off, c.size, F->thr, buf, &s.recs, nullptr, cnt, nullptr))
            F->failed = 1;
        {
            std::lock_guard<std::mutex> lk(F->mu);
            s.ticket = t;
            s.ready = true;
        }
        F->cv_cons.notify_all();
        if (F->failed) return;
    }
}

void stop_stream(Feeder* F) {
    {
        std::lock_guard<std::mutex> lk(F->mu);
        F->stop = true;
    }
    F->cv_prod.notify_all();
    F->cv_cons.notify_all();
    for (auto& w : F->workers) w.join();
    F->workers.clear();
}

}   // namespace

extern "C" {

const char* bf_last_error() {
    static thread_local std::string copy;
    std::lock_guard<std::mutex> lk(g_error_mu);
    copy = g_error;
    return copy.c_str();
}

// Reads the binpacks (paths separated by '\n') into memory, `threads` pieces at a time, and
// finds their chunks. thr: positions whose hash is below it are held out (2^32: all of them).
// use_cache: read <file>.bfx where it matches, and write it after the index pass.
void* bf_open(const char* paths, std::uint64_t thr, int threads, int use_cache) {
    auto* F = new Feeder;
    F->thr = thr;
    F->use_cache = use_cache != 0;
    if (threads < 1) threads = 1;
    for (const char* p = paths; *p;) {
        const char* e = std::strchr(p, '\n');
        const std::size_t len = e ? static_cast<std::size_t>(e - p) : std::strlen(p);
        if (len) F->files.push_back(File{std::string(p, len)});
        p += len + (e ? 1 : 0);
    }
    struct Piece { std::size_t file, off, len; };
    std::vector<Piece> pieces;
    std::vector<int> fds;
    for (std::size_t i = 0; i < F->files.size(); ++i) {
        File& f = F->files[i];
        const int fd = open(f.path.c_str(), O_RDONLY);
        struct stat st;
        if (fd < 0 || fstat(fd, &st) != 0) {
            set_error("cannot open " + f.path);
            for (int d : fds) close(d);
            delete F;
            return nullptr;
        }
        fds.push_back(fd);
        f.size = static_cast<std::size_t>(st.st_size);
        f.data = static_cast<std::uint8_t*>(std::malloc(f.size ? f.size : 1));
        if (!f.data) {
            set_error("out of memory for " + f.path);
            for (int d : fds) close(d);
            delete F;
            return nullptr;
        }
        for (std::size_t off = 0; off < f.size; off += PIECE) pieces.push_back({i, off, std::min<std::size_t>(PIECE, f.size - off)});
    }
    std::vector<std::uint64_t> ph(pieces.size());
    std::atomic<int> bad{0};
    parallel_for(pieces.size(), threads, [&](std::size_t k) {
        const Piece& pc = pieces[k];
        std::uint8_t* dst = F->files[pc.file].data + pc.off;
        std::size_t done = 0;
        while (done < pc.len) {
            const ssize_t r = pread(fds[pc.file], dst + done, pc.len - done, static_cast<off_t>(pc.off + done));
            if (r <= 0) { bad = 1; return; }
            done += static_cast<std::size_t>(r);
        }
        ph[k] = hash_bytes(dst, pc.len, 0x243F6A8885A308D3ULL ^ pc.off);
    });
    for (int d : fds) close(d);
    if (bad) {
        set_error("short read");
        delete F;
        return nullptr;
    }
    for (std::size_t k = 0; k < pieces.size(); ++k) {
        File& f = F->files[pieces[k].file];
        f.hash = mix64(f.hash ^ ph[k]) + pieces[k].off;
    }
    for (std::size_t i = 0; i < F->files.size(); ++i) {
        File& f = F->files[i];
        f.c0 = F->chunks.size();
        std::size_t off = 0;
        while (off + 8 <= f.size) {
            const std::uint8_t* h = f.data + off;
            if (std::memcmp(h, "BINP", 4) != 0) {
                set_error(f.path + ": no chunk header at byte " + std::to_string(off));
                delete F;
                return nullptr;
            }
            const std::uint32_t n = h[4] | (h[5] << 8) | (h[6] << 16) | (static_cast<std::uint32_t>(h[7]) << 24);
            if (off + 8 + n > f.size) {
                set_error(f.path + ": the chunk at byte " + std::to_string(off) + " runs past the end of the file");
                delete F;
                return nullptr;
            }
            F->chunks.push_back(Chunk{static_cast<std::uint32_t>(i), n, off + 8});
            off += 8 + n;
        }
        if (off != f.size) {
            set_error(f.path + ": " + std::to_string(f.size - off) + " stray bytes after the last chunk");
            delete F;
            return nullptr;
        }
        f.c1 = F->chunks.size();
    }
    F->counts.assign(F->chunks.size(), Counts{});
    F->val.assign(F->chunks.size(), {});
    if (F->use_cache)
        for (auto& f : F->files) read_cache(*F, f);
    F->active = F->chunks.size();
    return F;
}

// The index pass over every chunk of the files without a cache. Returns the number of caches
// written (-1 on a malformed position, -2 when a cache could not be written: not an error).
int bf_index(void* h, int threads) {
    auto* F = static_cast<Feeder*>(h);
    std::vector<std::size_t> todo;
    for (const auto& f : F->files)
        if (!f.cached)
            for (std::size_t c = f.c0; c < f.c1; ++c) todo.push_back(c);
    std::vector<Stats> cstat(F->chunks.size());
    std::atomic<int> bad{0};
    parallel_for(todo.size(), std::max(1, threads), [&](std::size_t k) {
        thread_local std::vector<unsigned char> buf;
        const std::size_t c = todo[k];
        const Chunk& ch = F->chunks[c];
        Counts cnt;
        if (!decode_chunk(F->files[ch.file].data + ch.off, ch.size, F->thr, buf, nullptr, &F->val[c], cnt, &cstat[c]))
            bad = 1;
        F->counts[c] = cnt;
    });
    if (bad) {
        set_error("a position with more than 32 pieces");
        return -1;
    }
    int written = 0;
    bool unwritten = false;
    for (auto& f : F->files) {
        if (f.cached) continue;
        for (std::size_t c = f.c0; c < f.c1; ++c) f.stats.add(cstat[c]);
        if (F->use_cache) {
            if (write_cache(*F, f)) ++written;
            else unwritten = true;
        }
    }
    F->indexed = true;
    return unwritten ? -2 : written;
}

// Totals: out[0..] = files, chunks, bytes, training, held out, marked skip, cached files,
// quiet, capture, check, won, drawn, lost (side to move), over every chunk of every file; and
// the stream's chunks (bf_select).
void bf_info(void* h, std::int64_t* out) {
    auto* F = static_cast<Feeder*>(h);
    std::int64_t bytes = 0, tr = 0, va = 0, mk = 0, cached = 0;
    Stats st;
    for (const auto& f : F->files) {
        bytes += static_cast<std::int64_t>(f.size);
        cached += f.cached;
        st.add(f.stats);
    }
    for (const auto& c : F->counts) { tr += c.train; va += c.val; mk += c.marked; }
    const std::int64_t v[] = {static_cast<std::int64_t>(F->files.size()), static_cast<std::int64_t>(F->chunks.size()),
                              bytes, tr, va, mk, cached, st.quiet, st.capture, st.check, st.win, st.draw, st.loss,
                              static_cast<std::int64_t>(F->active)};
    std::memcpy(out, v, sizeof v);
}

// Per file: out[4 * i + 0..3] = training, held out and marked skip positions, and 1 if the
// counts came from the file's cache.
void bf_file_counts(void* h, std::int64_t* out) {
    auto* F = static_cast<Feeder*>(h);
    for (std::size_t i = 0; i < F->files.size(); ++i) {
        std::int64_t tr = 0, va = 0, mk = 0;
        for (std::size_t c = F->files[i].c0; c < F->files[i].c1; ++c) {
            tr += F->counts[c].train; va += F->counts[c].val; mk += F->counts[c].marked;
        }
        out[4 * i] = tr; out[4 * i + 1] = va; out[4 * i + 2] = mk; out[4 * i + 3] = F->files[i].cached;
    }
}

// The stream's chunks: the first ones in file order while their training positions stay within
// max_pos (0 = every chunk; at least one chunk). Returns the training positions in a pass.
std::int64_t bf_select(void* h, std::int64_t max_pos) {
    auto* F = static_cast<Feeder*>(h);
    std::int64_t n = 0;
    std::size_t a = 0;
    for (; a < F->chunks.size(); ++a) {
        if (max_pos > 0 && a > 0 && n + F->counts[a].train > max_pos) break;
        n += F->counts[a].train;
    }
    F->active = a;
    return n;
}

// The held-out records of the stream's chunks, in chunk order: the count, and a copy into out.
std::int64_t bf_val_count(void* h) {
    auto* F = static_cast<Feeder*>(h);
    std::int64_t n = 0;
    for (std::size_t c = 0; c < F->active; ++c) n += static_cast<std::int64_t>(F->val[c].size());
    return n;
}

void bf_val_copy(void* h, std::uint8_t* out) {
    auto* F = static_cast<Feeder*>(h);
    for (std::size_t c = 0; c < F->active; ++c) {
        std::memcpy(out, F->val[c].data(), F->val[c].size() * sizeof(CompactRec));
        out += F->val[c].size() * sizeof(CompactRec);
    }
}

// Starts the stream: `threads` decoders ahead of the consumer by at most `ring` chunks.
int bf_stream_start(void* h, std::uint64_t seed, int threads, int ring) {
    auto* F = static_cast<Feeder*>(h);
    std::int64_t n = 0;
    for (std::size_t c = 0; c < F->active; ++c) n += F->counts[c].train;
    if (!F->workers.empty() || n == 0) {   // no training position: bf_next would wait forever
        set_error(F->workers.empty() ? "the stream's chunks hold no training position (run bf_index first)"
                                     : "the stream is running");
        return -1;
    }
    F->seed = seed;
    F->stop = false;
    F->failed = 0;
    F->next_ticket = 0;
    F->consumed = 0;
    F->late_prod = 0;
    F->late_cons = 0;
    F->ring.assign(static_cast<std::size_t>(std::max(ring, std::max(threads, 1) + 2)), Slot{});
    for (int t = 0; t < std::max(threads, 1); ++t) F->workers.emplace_back(worker, F);
    return 0;
}

// The stream's next `count` records into out (blocking). pass_out: the pass of the last one.
// Returns count, or -1 if a decoder failed (a malformed position) or the stream was stopped.
std::int64_t bf_next(void* h, std::uint8_t* out, std::int64_t count, std::int64_t* pass_out) {
    auto* F = static_cast<Feeder*>(h);
    const std::int64_t R = static_cast<std::int64_t>(F->ring.size());
    std::int64_t got = 0;
    while (got < count) {
        Slot& s = F->ring[F->consumed % R];
        {
            std::unique_lock<std::mutex> lk(F->mu);
            wait_until_ready(F->cv_cons, lk, [&] { return F->stop || F->failed || (s.ready && s.ticket == F->consumed); },
                             F->late_cons);
            if (F->stop || F->failed) {
                set_error(F->failed ? "a position with more than 32 pieces" : "the stream was stopped");
                return -1;
            }
        }
        const std::size_t take = std::min<std::size_t>(s.recs.size() - s.pos, static_cast<std::size_t>(count - got));
        std::memcpy(out + 32 * got, s.recs.data() + s.pos, take * sizeof(CompactRec));
        s.pos += take;
        got += static_cast<std::int64_t>(take);
        if (pass_out) *pass_out = F->consumed / static_cast<std::int64_t>(F->active);
        if (s.pos == s.recs.size()) {
            {
                std::lock_guard<std::mutex> lk(F->mu);
                s.ready = false;
                ++F->consumed;
            }
            F->cv_prod.notify_all();
        }
    }
    return got;
}

void bf_stream_stop(void* h) { stop_stream(static_cast<Feeder*>(h)); }

// The stream's state, for a stall report: out[12] = consumed ticket, next ticket, ring slots,
// slots ready, the consumed slot's ticket and ready flag, late producer and consumer waits,
// failed, stopped, decode threads, active chunks.
void bf_state(void* h, std::int64_t* out) {
    auto* F = static_cast<Feeder*>(h);
    std::lock_guard<std::mutex> lk(F->mu);
    const std::int64_t R = static_cast<std::int64_t>(F->ring.size());
    std::int64_t ready = 0;
    for (const auto& s : F->ring) ready += s.ready;
    const Slot* s = R ? &F->ring[F->consumed % R] : nullptr;
    const std::int64_t v[12] = {F->consumed, F->next_ticket.load(), R, ready, s ? s->ticket : -1, s ? s->ready : 0,
                                F->late_prod.load(), F->late_cons.load(), F->failed.load(), F->stop,
                                static_cast<std::int64_t>(F->workers.size()), static_cast<std::int64_t>(F->active)};
    std::memcpy(out, v, sizeof v);
}

void bf_close(void* h) {
    auto* F = static_cast<Feeder*>(h);
    stop_stream(F);
    for (auto& f : F->files) std::free(f.data);
    delete F;
}

// Decodes chunks [0, nchunks) (all if <= 0) on `threads` threads into per-thread buffers, as the
// stream does, and returns the seconds it took; out = entries decoded (marked skip included),
// training positions, held out.
double bf_bench(void* h, int threads, std::int64_t nchunks, std::int64_t* out) {
    auto* F = static_cast<Feeder*>(h);
    const std::size_t n = nchunks > 0 ? std::min<std::size_t>(static_cast<std::size_t>(nchunks), F->chunks.size()) : F->chunks.size();
    std::atomic<std::int64_t> entries{0}, tr{0}, va{0};
    const auto t0 = std::chrono::steady_clock::now();
    parallel_for(n, std::max(1, threads), [&](std::size_t c) {
        thread_local std::vector<unsigned char> buf;
        thread_local std::vector<CompactRec> recs;
        recs.clear();
        const Chunk& ch = F->chunks[c];
        Counts cnt;
        decode_chunk(F->files[ch.file].data + ch.off, ch.size, F->thr, buf, &recs, nullptr, cnt, nullptr);
        entries += cnt.train + cnt.val + cnt.marked;
        tr += cnt.train;
        va += cnt.val;
    });
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    out[0] = entries; out[1] = tr; out[2] = va;
    return sec;
}

// Sorts n 32-byte records into byte order (memcmp), in parallel: a canonical form, so two sets
// of records are the same multiset exactly when their sorted bytes are equal.
void bf_sort(std::uint8_t* data, std::int64_t n, int threads) {
    struct R32 { std::uint8_t b[32]; };
    auto* r = reinterpret_cast<R32*>(data);
    auto less = [](const R32& a, const R32& b) { return std::memcmp(a.b, b.b, 32) < 0; };
    int parts = 1;
    while (parts * 2 <= std::max(1, threads) && n / (parts * 2) > 4096) parts *= 2;
    std::vector<std::int64_t> cut(parts + 1);
    for (int i = 0; i <= parts; ++i) cut[i] = n * i / parts;
    std::vector<std::thread> pool;
    for (int i = 0; i < parts; ++i) pool.emplace_back([&, i] { std::sort(r + cut[i], r + cut[i + 1], less); });
    for (auto& t : pool) t.join();
    for (int w = 1; w < parts; w *= 2) {
        pool.clear();
        for (int i = 0; i + w < parts; i += 2 * w)
            pool.emplace_back([&, i, w] {
                std::inplace_merge(r + cut[i], r + cut[i + w], r + cut[std::min(i + 2 * w, parts)], less);
            });
        for (auto& t : pool) t.join();
    }
}

}   // extern "C"
