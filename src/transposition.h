#pragma once

#include "types.h"
#include <cstdint>
#include <cstring>
#include <vector>

namespace luminex {

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

    TTEntry* probe(uint64_t key, bool& found);
    void write(uint64_t key, Value v, bool pv, Bound b, Depth d, Move m, Value ev);
    void prefetch(uint64_t key) const;

    uint8_t generation() const { return generation8; }
    uint8_t search_id() const { return search_id8; }
    void begin_search() { ++search_id8; }  // once per "go": entry ages by 1/search
    size_t hashfull() const;

private:
    std::vector<TTCluster> table;
    size_t entries;
    uint8_t generation8 = 0;
    uint8_t search_id8 = 0;
};

inline TranspositionTable TT;

} // namespace luminex
