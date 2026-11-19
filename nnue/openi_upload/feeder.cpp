// One training batch for resident_train.py's PoolFeeder, prepared in C++ so the GIL is free
// for the whole batch and several threads share the work. The order is KeyedShuffle's, bit
// for bit: the same 4-round Feistel network keyed by (seed, pass), walking cycles past values
// >= n. Positions below ng are VRAM-pool indices; the rest are host-pool rows, copied straight
// into the caller's (pinned) buffer. Both keep the order they have in the pass, as the numpy
// path does. Build: g++ -O3 -march=native -shared -fPIC -pthread feeder.cpp -o libfeeder.so
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

static inline uint64_t mix(uint64_t r, uint64_t k, uint64_t mask) {
    uint64_t x = (r ^ k) * 0x9E3779B97F4A7C15ULL;
    x ^= x >> 29;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 32;
    return x & mask;
}

static inline uint64_t feistel(uint64_t x, const uint64_t* ks, unsigned h, uint64_t mask) {
    uint64_t left = x >> h, right = x & mask;
    for (int i = 0; i < 4; i++) {
        uint64_t next = left ^ mix(right, ks[i], mask);
        left = right;
        right = next;
    }
    return (left << h) | right;
}

// Writes the VRAM-pool indices to gi_out and the host rows (32 bytes each) to rows_out, both
// in pass order; scratch holds count uint64. Returns the number of VRAM-pool indices.
extern "C" int64_t feed_batch(const uint8_t* host, int64_t ng, int64_t n, const uint64_t* ks,
                              unsigned h, uint64_t mask, int64_t start, int64_t count,
                              int64_t* gi_out, uint8_t* rows_out, uint64_t* scratch, int threads) {
    if (threads < 1) threads = 1;
    std::vector<int64_t> before(threads + 1, 0);
    auto order = [&](int t) {
        int64_t lo = count * t / threads, hi = count * (t + 1) / threads, c = 0;
        for (int64_t i = lo; i < hi; i++) {
            uint64_t y = feistel(uint64_t(start + i), ks, h, mask);
            while (y >= uint64_t(n)) y = feistel(y, ks, h, mask);
            scratch[i] = y;
            c += y < uint64_t(ng);
        }
        before[t + 1] = c;
    };
    auto gather = [&](int t) {
        int64_t lo = count * t / threads, hi = count * (t + 1) / threads;
        int64_t g = before[t], r = lo - before[t];
        for (int64_t i = lo; i < hi; i++) {
            uint64_t y = scratch[i];
            if (y < uint64_t(ng)) gi_out[g++] = int64_t(y);
            else std::memcpy(rows_out + 32 * r++, host + 32 * (y - uint64_t(ng)), 32);
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < threads; t++) pool.emplace_back(order, t);
    order(0);
    for (auto& th : pool) th.join();
    for (int t = 0; t < threads; t++) before[t + 1] += before[t];
    pool.clear();
    for (int t = 1; t < threads; t++) pool.emplace_back(gather, t);
    gather(0);
    for (auto& th : pool) th.join();
    return before[threads];
}
