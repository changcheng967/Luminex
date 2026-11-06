#include "luminex.h"
#include <cstring>
#include <algorithm>

namespace luminex {

void TTEntry::save(uint64_t k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t g) {
    // A store without a move (a fail-low node, a qsearch node with no capture
    // worth recording) keeps the move already saved for this position: that
    // move is still the best ordering hint the position has.
    const bool same_position = depth_ != 0 && key32 == uint32_t(k >> 32);
    if (m != MOVE_NONE || !same_position)
        move16 = uint16_t(m.raw());
    key32 = uint32_t(k >> 32);
    value_ = int32_t(v);
    eval_ = int32_t(ev);
    // Stored as d + 127 for d in [0, 128]: qsearch entries keep depth 0, so a
    // main-search node (depth >= 1) never takes a cutoff from a qsearch bound.
    // depth_ == 0 is the empty slot left by clear(); 128 caps the uint8 range.
    depth_ = uint8_t(std::clamp<Depth>(d, 0, 128) + 127);
    // g arrives pre-shifted (generation8 advances in steps of 4, keeping the
    // low 2 bits free for the bound); shifting again here would desynchronize
    // save and probe encodings.
    gen_bound = uint8_t(g | b);
    pv_flag = pv ? 1 : 0;
    search_id_ = TT.search_id();
}

void TranspositionTable::resize(size_t mb) {
    table.clear();

    // Round size down to power of 2 for fast bitwise AND indexing
    size_t new_size = mb > 0 ? (mb * 1024 * 1024) / sizeof(TTCluster) : 0;
    if (new_size > 0) {
        // Find largest power of 2 <= new_size
        new_size--;
        new_size |= new_size >> 1;
        new_size |= new_size >> 2;
        new_size |= new_size >> 4;
        new_size |= new_size >> 8;
        new_size |= new_size >> 16;
        if constexpr (sizeof(size_t) > 4) new_size |= new_size >> 32;
        new_size++;
        // Minimum size of 1024 to avoid tiny tables
        if (new_size < 1024) new_size = 1024;
    }
    entries = new_size * 3;

    if (new_size > 0) {
        table.resize(new_size);  // value-initialized: already zero
    }
}

void TranspositionTable::clear() {
    if (entries == 0) return;

    std::memset(table.data(), 0, table.size() * sizeof(TTCluster));
}

void TranspositionTable::new_search() {
    generation8 += 4;
}

TTEntry* TranspositionTable::probe(uint64_t key, bool& found) {
    if (table.empty()) {
        static TTEntry dummy;
        found = false;
        return &dummy;
    }

    // Use bitwise AND instead of modulo - much faster (20-30% TT speedup)
    size_t idx = (size_t)key & (table.size() - 1);
    TTEntry* entry = &table[idx].entry[0];

    // depth_ == 0 marks an empty slot (clear() zeroes the table); stored
    // entries are depth + 127 >= 127.
    for (int i = 0; i < 3; ++i, ++entry) {
        if (entry->depth_ != 0 && entry->key32 == uint32_t(key >> 32)) {
            if ((entry->gen_bound & 0xFC) != generation8) {
                entry->gen_bound = uint8_t(generation8 | (entry->gen_bound & 0x3));
            }
            found = true;
            // Self-cleaning cross-search aging: age the STORED depth byte so the
            // entry drops out of both the cutoff check and the depth-preferred
            // replacement ranking (a read-time-only discount leaves stale deep
            // entries squatting in slots they can never be trusted for). Clamp at
            // 127 (depth 0) so an aged entry never reads as an empty slot.
            if (entry->search_id_ != search_id8) {
                int age = int(uint8_t(search_id8 - entry->search_id_));
                entry->depth_ = uint8_t(std::max(127, int(entry->depth_) - 2 * age));
                entry->search_id_ = search_id8;
            }
            return entry;
        }
    }

    // Find the least valuable entry to replace. Old-generation entries are
    // discounted first (they lose authority to anything current); within the
    // same generation the shallowest depth goes.
    TTEntry* replace = &table[idx].entry[0];
    for (int i = 1; i < 3; ++i) {
        TTEntry* e = &table[idx].entry[i];
        if (e->depth_ - ((e->gen_bound & 0xFC) == generation8 ? 0 : 127) <
            replace->depth_ - ((replace->gen_bound & 0xFC) == generation8 ? 0 : 127)) {
            replace = e;
        }
    }

    found = false;
    return replace;
}

size_t TranspositionTable::hashfull() const {
    if (entries == 0) return 0;

    int cnt = 0;
    for (size_t i = 0; i < std::min<size_t>(1000, table.size()); ++i) {
        for (int j = 0; j < 3; ++j) {
            if ((table[i].entry[j].gen_bound & 0xFC) == generation8) {
                ++cnt;
            }
        }
    }

    return cnt * 1000 / (std::min(size_t(1000), table.size()) * 3);
}

void TranspositionTable::prefetch(uint64_t key) const {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(&table[(size_t)key & (table.size() - 1)], 0, 3);
#endif
}

} // namespace luminex
