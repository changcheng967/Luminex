#include "book.h"
#include "board.h"
#include "movegen.h"
#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

namespace luminex {

struct BookEntry {
    Key key;
    uint16_t weight;
    char uci[5]; // e.g. "e2e4\0"
};

static std::vector<BookEntry> g_book;
static std::mt19937 g_rng;

// Convert UCI string to Move for a given position
static Move uci_to_move(const Position& pos, const char* uci) {
    if (!uci[0] || !uci[1] || !uci[2] || !uci[3]) return MOVE_NONE;

    File ff = static_cast<File>(uci[0] - 'a');
    Rank fr = static_cast<Rank>(uci[1] - '1');
    File tf = static_cast<File>(uci[2] - 'a');
    Rank tr = static_cast<Rank>(uci[3] - '1');

    if (ff > FILE_H || fr > RANK_8 || tf > FILE_H || tr > RANK_8) return MOVE_NONE;

    Square from = static_cast<Square>(ff + fr * 8);
    Square to   = static_cast<Square>(tf + tr * 8);

    PieceType promo = PT_NONE;
    if (uci[4]) {
        switch (uci[4]) {
            case 'q': promo = QUEEN;  break;
            case 'r': promo = ROOK;   break;
            case 'b': promo = BISHOP; break;
            case 'n': promo = KNIGHT; break;
            default: return MOVE_NONE;
        }
    }

    ExtMove moves[MAX_MOVES];
    ExtMove* end = generate<GEN_LEGAL>(pos, moves);
    for (ExtMove* it = moves; it != end; ++it) {
        if (it->move.from() != from || it->move.to() != to) continue;
        if (promo != PT_NONE) {
            if (it->move.is_promotion() && it->move.promotion_type() == promo)
                return it->move;
        } else if (!it->move.is_promotion()) {
            return it->move;
        }
    }
    return MOVE_NONE;
}

struct BookLine {
    int weight;
    const char* moves[16]; // null-terminated
};

static const BookLine book_lines[] = {
    // === As White ===

    // Ruy Lopez Morphy Defense (w=5)
    {5, {"e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6",
         "e1g1", "f8e7", "f1e1", "b7b5", "a4b3", "d7d6", "c2c3", "e8g8"}},

    // Italian Giuoco Piano (w=4)
    {4, {"e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6",
         "d2d3", "e8g8", "e1g1", nullptr}},

    // Alapin Sicilian (w=3)
    {3, {"e2e4", "c7c5", "c2c3", "g8f6", "e2e3", "d7d5", nullptr}},

    // Caro-Kann Exchange (w=2)
    {2, {"e2e4", "c7c6", "d2d4", "d7d5", "e4d5", "c6d5", "c1f4", nullptr}},

    // QGD Classical (w=5)
    {5, {"d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7",
         "e2e3", "e8g8", "g1f3", "b8d7", nullptr}},

    // Nimzo-Indian Rubinstein (w=3)
    {3, {"d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "d1c2", "e8g8",
         "a2a3", "f8e7", nullptr}},

    // London System (w=3)
    {3, {"d2d4", "d7d5", "g1f3", "g8f6", "e2e3", "c7c5", "c1f4", nullptr}},

    // English (w=2)
    {2, {"c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6", "g2g3", "f8b4", nullptr}},

    // === As Black vs 1.e4 ===

    // Caro-Kann Classical (w=5)
    {5, {"e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5d4", nullptr}},

    // Caro-Kann Advance (w=3)
    {3, {"e2e4", "c7c6", "d2d4", "d7d5", "e4e5", "c8f5", nullptr}},

    // Sicilian Taimanov (w=2)
    {2, {"e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "g1f3", "e7e6", nullptr}},

    // 1...e5 Italian (w=2)
    {2, {"e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", nullptr}},

    // === As Black vs 1.d4 ===

    // QGD Tartakower (w=5)
    {5, {"d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "g1f3", "f8e7",
         "c1f4", "e8g8", nullptr}},

    // Nimzo-Indian (w=4)
    {4, {"d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "e8g8",
         "f1d3", "d7d5", nullptr}},

    // === As Black vs 1.c4 ===

    // Reversed Sicilian (w=3)
    {3, {"c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6", "g2g3", "f8b4", nullptr}},

    // 1...Nf6 (w=3)
    {3, {"c2c4", "g8f6", "b1c3", "e7e5", "g1f3", "b8c6", nullptr}},

    // === As Black vs 1.Nf3 ===

    // 1...d5 (w=4)
    {4, {"g1f3", "d7d5", "g2g3", "g8f6", "f1g2", "e7e6", "e1g1", "f8e7", nullptr}},

    // 1...Nf6 (w=3)
    {3, {"g1f3", "g8f6", "g2g3", "e7e5", "f1g2", "b8c6", nullptr}},
};

void init_book() {
    g_book.clear();

    Position pos;
    int line_count = sizeof(book_lines) / sizeof(book_lines[0]);

    for (int i = 0; i < line_count; i++) {
        const BookLine& line = book_lines[i];
        pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        for (int j = 0; j < 16 && line.moves[j]; j++) {
            Key key = pos.key();
            Move m = uci_to_move(pos, line.moves[j]);
            if (m == MOVE_NONE) break;

            BookEntry entry;
            entry.key = key;
            entry.weight = static_cast<uint16_t>(line.weight);
            std::memcpy(entry.uci, line.moves[j], 5);

            g_book.push_back(entry);

            if (!pos.do_move(m)) break;
        }
    }

    // Sort by key for binary search
    struct BookComp {
        bool operator()(const BookEntry& e, Key k) const { return e.key < k; }
        bool operator()(Key k, const BookEntry& e) const { return k < e.key; }
    };
    std::sort(g_book.begin(), g_book.end(),
              [](const BookEntry& a, const BookEntry& b) { return a.key < b.key; });

    // Seed RNG
    std::random_device rd;
    g_rng.seed(rd());

    printf("Book initialized: %zu entries from %d lines\n", g_book.size(), line_count);
}

Move book_probe(const Position& pos) {
    if (g_book.empty()) return MOVE_NONE;

    Key key = pos.key();

    // Binary search for range of entries matching this key
    struct BookComp {
        bool operator()(const BookEntry& e, Key k) const { return e.key < k; }
        bool operator()(Key k, const BookEntry& e) const { return k < e.key; }
    };
    auto range = std::equal_range(g_book.begin(), g_book.end(), key, BookComp{});

    // Count matches and sum weights
    int total_weight = 0;
    for (auto it = range.first; it != range.second; ++it)
        total_weight += it->weight;

    if (total_weight == 0) return MOVE_NONE;

    // Weighted random selection
    std::uniform_int_distribution<int> dist(0, total_weight - 1);
    int target = dist(g_rng);
    int accum = 0;

    const char* selected_uci = nullptr;
    for (auto it = range.first; it != range.second; ++it) {
        accum += it->weight;
        if (accum > target) {
            selected_uci = it->uci;
            break;
        }
    }

    if (!selected_uci) selected_uci = range.first->uci;

    // Convert UCI string to Move and verify legality
    Move m = uci_to_move(pos, selected_uci);
    return m;
}

void book_new_game() {
    std::random_device rd;
    g_rng.seed(rd());
}

} // namespace luminex
