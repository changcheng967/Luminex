#include "luminex.h"
#include <cstring>

namespace luminex {

void TTEntry::save(uint64_t k, Value v, bool, Bound b, Depth d, Move m, Value ev, uint8_t g) {
    key16 = uint16_t(k >> 48);
    move16 = uint16_t(m.raw());
    value_ = int16_t(v);
    eval_ = int16_t(ev);
    depth_ = uint8_t(d + 127);
    gen_bound = uint8_t((g << 2) | b);
}

void TranspositionTable::resize(size_t mb) {
    table.clear();

    size_t new_size = mb > 0 ? (mb * 1024 * 1024) / sizeof(TTCluster) : 0;
    entries = new_size * 3;

    if (new_size > 0) {
        table.resize(new_size);
    }

    clear();
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

    TTEntry* entry = &table[(size_t)key % table.size()].entry[0];

    for (int i = 0; i < 3; ++i, ++entry) {
        if (entry->key16 == uint16_t(key >> 48) || entry->depth_ == 127) {
            if (entry->depth_ == 127 || (entry->gen_bound & 0xFC) != generation8) {
                entry->gen_bound = uint8_t(generation8 | (entry->gen_bound & 0x3));
            }
            found = (entry->depth_ != 127);
            return entry;
        }
    }

    // Find entry with lowest depth
    TTEntry* replace = &table[(size_t)key % table.size()].entry[0];
    for (int i = 1; i < 3; ++i) {
        TTEntry* e = &table[(size_t)key % table.size()].entry[i];
        if (e->depth_ - ((e->gen_bound & 0xFC) == generation8 ? 127 : 0) <
            replace->depth_ - ((replace->gen_bound & 0xFC) == generation8 ? 127 : 0)) {
            replace = e;
        }
    }

    found = false;
    return replace;
}

void TranspositionTable::write(uint64_t key, Value v, bool pv, Bound b, Depth d, Move m, Value ev) {
    if (table.empty()) return;

    TTEntry* entry = &table[(size_t)key % table.size()].entry[0];

    // Find an entry to replace
    for (int i = 0; i < 3; ++i, ++entry) {
        if (entry->key16 == uint16_t(key >> 48) || entry->depth_ == 127 ||
            (entry->gen_bound & 0xFC) != generation8 || d >= entry->depth()) {
            break;
        }
    }

    // Don't overwrite more valuable entries
    if (b != BOUND_EXACT && entry->key16 == uint16_t(key >> 48)) {
        if (entry->depth_ != 127 &&
            (b == BOUND_LOWER ? entry->bound() == BOUND_UPPER : entry->bound() == BOUND_LOWER)) {
            // Don't overwrite with opposite bound at same or lower depth
            if (d <= entry->depth()) {
                return;
            }
        }
    }

    entry->save(key, v, pv, b, d, m, ev, generation8);
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

} // namespace luminex
