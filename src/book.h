#pragma once

#include "board.h"
#include "types.h"
#include <cstdint>
#include <string>

namespace luminex {

class Position;

// Polyglot opening book reader
class Book {
public:
    bool open(const char* path);
    void close();

    // Probe book for current position. Returns UCI move string or empty.
    std::string probe(const Position& pos);

    bool is_open() const { return file_ != nullptr; }

private:
    struct Entry {
        uint64_t key;
        uint16_t move;
        uint16_t weight;
        uint32_t learn;
    };

    Entry read_entry(int index);
    int find_pos(uint64_t key);

    FILE* file_ = nullptr;
    int size_ = 0;
};

// Global book instance
extern Book g_book;
extern std::string g_book_path;

// Compute Polyglot hash for a position
uint64_t compute_polyglot_key(const Position& pos);

} // namespace luminex
