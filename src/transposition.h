#pragma once

#include "types.h"
#include <bit>
#include <cstdint>
#include <cstring>
#include <vector>
#include "large_pages.h"
#if defined(_MSC_VER) && !defined(__SIZEOF_INT128__)
#include <intrin.h>
#endif

namespace luminex {

// The high 64 bits of the 128-bit product a * b.
inline uint64_t mul_hi64(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
    return uint64_t((unsigned __int128)a * b >> 64);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
    return __umulh(a, b);
#else
    // Four 32 x 32 products: the middle column's carry into the high half is
    // the top of the sum of its three 32-bit parts.
    const uint64_t al = uint32_t(a), ah = a >> 32, bl = uint32_t(b), bh = b >> 32;
    const uint64_t ll = al * bl, hl = ah * bl, lh = al * bh;
    return ah * bh + (hl >> 32) + (lh >> 32) + (((ll >> 32) + uint32_t(hl) + uint32_t(lh)) >> 32);
#endif
}

// Transposition table entry
struct TTEntry {
    uint32_t key32;
    uint16_t move16;
    int32_t value_;  // Changed from int16_t to prevent truncation (Value is int32_t)
    int32_t eval_;   // Changed from int16_t to prevent truncation
    uint8_t depth_;
    uint8_t gen_bound;
    uint8_t pv_flag;    // 1 if this was a PV node, 0 if not
    uint8_t search_id_; // which search wrote this entry (cross-search aging)

    Move move() const { return Move(move16); }
    Depth depth() const { return Depth(depth_ - 127); }
    Bound bound() const { return Bound(gen_bound & 0x3); }
    bool is_pv() const { return pv_flag != 0; }
    Value value() const { return Value(value_); }
    Value eval() const { return Value(eval_); }
    uint8_t search_age(uint8_t now) const { return uint8_t(now - search_id_); }

    void save(uint64_t k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t g);
};

// Transposition table cluster (cache line size = 64 bytes)
// Entry size is 20 bytes (3 * 20 = 60), padding = 4 bytes
struct TTCluster {
    TTEntry entry[3];
    char padding[4];
};

static_assert(sizeof(TTCluster) == 64, "TTCluster size must be 64 bytes");

// Transposition table
class TranspositionTable {
public:
    TranspositionTable() : table(), entries(0) {}

    ~TranspositionTable() { resize(0); }

    void resize(size_t mb);
    void clear();
    void new_search();
    // The table is allocated on first use (isready or a search) rather than
    // at startup, so a GUI's Hash setting is the only allocation: zeroing a
    // default 256 MB that is replaced at once cost the start most of a second.
    void ensure_allocated() { if (table.empty()) resize(DEFAULT_MB); }
    static constexpr size_t DEFAULT_MB = 256;

    TTEntry* probe(uint64_t key, bool& found);
    void prefetch(uint64_t key) const;

    // Store policy for probe-returned entries: always store on a miss (the
    // replacement victim was already chosen by probe); on a same-key hit keep
    // the old entry unless this search is at least as deep or the entry is
    // from an older search.
    bool should_store(const TTEntry* tte, bool found, Depth d) const {
        return !found || d >= tte->depth() || (tte->gen_bound & 0xFC) != generation8;
    }


    uint8_t generation() const { return generation8; }
    uint8_t search_id() const { return search_id8; }
    void begin_search() { ++search_id8; }  // once per "go": entry ages by 1/search
    size_t hashfull() const;

private:
    // The cluster that holds key, in a table of any size N. A power of two takes
    // the key's low bits, as the table always has. Any other N takes the high
    // half of rotl(key, 64 - k) * N, 2^k being the next power of two: the
    // rotation puts those same k low bits on top with the key's high bits below
    // them, so the product spreads keys evenly over all N clusters (and equals
    // key & (N - 1) when N is 2^k). key32 checks the key's high 32 bits: the
    // power-of-two index never reads them, the other reads them only to put a
    // key on one side of a boundary between two clusters, which keeps a false
    // match at 1 in 2^32 or rarer. (Past 2^32 clusters, Hash 262,144, index and
    // key32 share bits either way.)
    size_t cluster_of(uint64_t key) const {
        return pow2_ ? size_t(key) & (table.size() - 1)
                     : size_t(mul_hi64(std::rotl(key, rot_), table.size()));
    }

    LargeArray<TTCluster> table;   // 2 MiB pages where the system allows (large_pages.h)
    size_t entries;
    int rot_ = 0;                  // 64 - k
    bool pow2_ = true;             // N is a power of two
    uint8_t generation8 = 0;
    uint8_t search_id8 = 0;
};

inline TranspositionTable TT;

} // namespace luminex
