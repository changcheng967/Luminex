// binpack2lcr.cpp — convert a training binpack straight into LCR1 records
// (32 bytes each, the format resident_train.py keeps in VRAM), nothing dropped.
//
// A binpack is a sequence of self-contained chunks ("BINP", uint32 size, data),
// so the chunks are decoded in parallel: thread k converts the k-th contiguous
// run of chunks into OUT.kkk.lcr, which keeps the binpack's order across the
// parts. Every position is written: mates and other large scores raw (the
// trainer clips them in its loss), every ply, in-check and capture positions
// flagged as sf_binpack_dump.cpp flags them.
//
// Record (src/featurize.cpp CompactRec): occupancy u64 | the pieces of the
// occupied squares in ascending order as 4-bit codes (src/types.h Piece:
// WHITE_PAWN = 0 .. BLACK_KING = 11) | i16 side-to-move-relative score |
// u8 flags (bit 0 white to move, bits 1-2: 0 quiet, 1 capture-best,
// 2 in-check, bit 3 game result known, bits 4-5 the game's result for the side
// to move + 1: 0 loss, 1 draw, 2 win) | u8 piece count | u32 position hash
// (the validation split; the featurizer path stores a crc32 of the FEN there
// instead). Readers mask the bits they use, so records written before the
// result bits existed read as "result unknown" (bit 3 clear).
//
// Build (against a checkout of official-stockfish/nnue-pytorch):
//   g++ -O2 -std=c++20 -pthread -I <nnue-pytorch>/data_loader/cpp \
//       -o binpack2lcr binpack2lcr.cpp
// Usage:
//   ./binpack2lcr in.binpack OUT [threads]      -> OUT.000.lcr, OUT.001.lcr, ...
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "lib/binpack.h"

struct CompactRec {
    std::uint64_t occ;
    std::uint8_t pcs[16];
    std::int16_t target;
    std::uint8_t flags;
    std::uint8_t npieces;
    std::uint32_t poshash;
};
static_assert(sizeof(CompactRec) == 32, "LCR1 records are 32 bytes");

static inline std::uint64_t mix64(std::uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s in.binpack OUT [threads]  (writes OUT.000.lcr ...)\n", argv[0]);
        return 1;
    }
    const std::string out = argv[2];
    int threads = argc > 3 ? std::atoi(argv[3]) : static_cast<int>(std::thread::hardware_concurrency());
    if (threads < 1) threads = 1;
    const auto t0 = std::chrono::steady_clock::now();

    const int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { std::perror(argv[1]); return 1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { std::perror("fstat"); return 1; }
    const std::size_t size = static_cast<std::size_t>(st.st_size);
    const auto* base = static_cast<const unsigned char*>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (base == MAP_FAILED) { std::perror("mmap"); return 1; }

    std::vector<std::pair<std::size_t, std::uint32_t>> chunks;   // (data offset, size)
    std::size_t off = 0;
    while (off + 8 <= size) {
        const unsigned char* h = base + off;
        if (std::memcmp(h, "BINP", 4) != 0) {
            std::fprintf(stderr, "FAIL: no chunk header at byte %zu\n", off);
            return 1;
        }
        const std::uint32_t n = h[4] | (h[5] << 8) | (h[6] << 16) | (static_cast<std::uint32_t>(h[7]) << 24);
        if (off + 8 + n > size) {
            std::fprintf(stderr, "FAIL: chunk at byte %zu runs past the end of the file\n", off);
            return 1;
        }
        chunks.emplace_back(off + 8, n);
        off += 8 + n;
    }
    if (off != size) {
        std::fprintf(stderr, "FAIL: %zu stray bytes after the last chunk\n", size - off);
        return 1;
    }
    if (threads > static_cast<int>(chunks.size())) threads = static_cast<int>(chunks.size());

    std::atomic<long long> total{0}, quiet{0}, capture{0}, check{0}, wins{0}, draws{0}, losses{0};
    std::atomic<int> failed{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            const std::size_t c0 = chunks.size() * t / threads, c1 = chunks.size() * (t + 1) / threads;
            char name[4096];
            std::snprintf(name, sizeof name, "%s.%03d.lcr", out.c_str(), t);
            std::FILE* f = std::fopen(name, "wb");
            if (!f) { std::perror(name); failed = 1; return; }
            std::vector<char> obuf(1 << 24);
            std::setvbuf(f, obuf.data(), _IOFBF, obuf.size());
            std::vector<unsigned char> chunk;
            long long n = 0, nq = 0, nc = 0, ni = 0, nw = 0, nd = 0, nl = 0;
            for (std::size_t c = c0; c < c1; ++c) {
                const unsigned char* p = base + chunks[c].first;
                chunk.assign(p, p + chunks[c].second);
                binpack::ChunkReader reader;
                while (reader.hasNext(chunk)) {
                    const auto e = reader.next(chunk);
                    CompactRec r;
                    std::memset(&r, 0, sizeof r);
                    r.occ = e.pos.piecesBB().bits();
                    int k = 0;
                    for (std::uint64_t bb = r.occ; bb; bb &= bb - 1, ++k) {
                        const int sq = __builtin_ctzll(bb);
                        const auto pc = e.pos.pieceAt(chess::Square(sq));
                        const int code = static_cast<int>(chess::ordinal(pc.type()))
                                       + 6 * static_cast<int>(chess::ordinal(pc.color()));
                        if (k < 32) r.pcs[k >> 1] |= static_cast<std::uint8_t>(code << (4 * (k & 1)));
                    }
                    if (k > 32) { failed = 2; continue; }   // not a chess position; never seen
                    const bool white = e.pos.sideToMove() == chess::Color::White;
                    const int q = !e.pos.checkers().isEmpty() ? 2 : (e.isCapturingMove() ? 1 : 0);
                    const int res = e.result > 0 ? 2 : (e.result < 0 ? 0 : 1);   // side to move's game result + 1
                    r.target = e.score;                       // side-to-move relative already
                    r.flags = static_cast<std::uint8_t>((white ? 1 : 0) | (q << 1) | (1 << 3) | (res << 4));
                    r.npieces = static_cast<std::uint8_t>(k);
                    std::uint64_t lo, hi;
                    std::memcpy(&lo, r.pcs, 8);
                    std::memcpy(&hi, r.pcs + 8, 8);
                    r.poshash = static_cast<std::uint32_t>(mix64(r.occ ^ mix64(lo ^ mix64(hi ^ (white ? 1 : 0)))) >> 32);
                    std::fwrite(&r, sizeof r, 1, f);
                    ++n;
                    nq += q == 0; nc += q == 1; ni += q == 2;
                    nl += res == 0; nd += res == 1; nw += res == 2;
                }
            }
            if (std::fclose(f) != 0) failed = 3;
            total += n; quiet += nq; capture += nc; check += ni;
            wins += nw; draws += nd; losses += nl;
        });
    }
    for (auto& t : pool) t.join();
    munmap(const_cast<unsigned char*>(base), size);
    close(fd);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (failed) {
        std::fprintf(stderr, "FAIL: code %d\n", failed.load());
        return 1;
    }
    std::fprintf(stderr, "DONE %lld positions (quiet %lld, capture %lld, check %lld; side to move "
                         "won %lld, drew %lld, lost %lld) from %zu chunks in %.1f s (%.1fM positions/s, %d threads)\n",
                 total.load(), quiet.load(), capture.load(), check.load(), wins.load(), draws.load(),
                 losses.load(), chunks.size(), sec, total.load() / 1e6 / (sec > 0 ? sec : 1), threads);
    return 0;
}
