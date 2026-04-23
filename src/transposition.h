#pragma once

#include "types.h"
#include <cstdint>
#include <cstring>
#include <vector>

namespace luminex {

// Transposition table entry (16 bytes — 4 per cache line)
struct TTEntry {
    uint32_t key32;
    uint16_t move16;
    int16_t value_;
    int16_t eval_;
    uint8_t depth_;
    uint8_t gen_bound;
    uint8_t pv_flag;

    Move move() const { return Move(move16); }
    Depth depth() const { return Depth(depth_ - 127); }
    Bound bound() const { return Bound(gen_bound & 0x3); }
    bool is_pv() const { return pv_flag != 0; }
    Value value() const { return Value(value_); }
    Value eval() const { return Value(eval_); }

    void save(uint64_t k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t g);
};

// 4 entries x 16 bytes = 64 bytes (exactly one cache line)
struct TTCluster {
    TTEntry entry[4];
};

static_assert(sizeof(TTCluster) == 64, "TTCluster size must be 64 bytes");
static_assert(sizeof(TTEntry) == 16, "TTEntry size must be 16 bytes");

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
    size_t hashfull() const;

private:
    std::vector<TTCluster> table;
    size_t entries;
    uint8_t generation8 = 0;
};

inline TranspositionTable TT;

} // namespace luminex
