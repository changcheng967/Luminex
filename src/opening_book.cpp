#include "opening_book.h"
#include "board.h"
#include <cstdlib>
#include <ctime>

namespace luminex {

// ============================================================
// Built-in Opening Book
//
// Problem: at bullet TC (1+0.01), Luminex reaches only ~depth 10
// in the opening. This causes it to play the same line every game:
//   As White: 1.Nf3 d5 2.Nc3 d4 3.Ne4 f5 → loses 60/100 games
//   As Black vs 1.d4: ...Nf6 2.Bf4 → loses 52/100 games
// Together these account for 91% of all losses.
//
// Solution: compact book that provides variety and avoids known-bad
// lines. All moves are standard opening theory — no engine values.
//
// Book is keyed by position hash, with multiple moves per position
// for variety. A random selection is made among available moves.
// ============================================================

struct BookMove {
    Square from;
    Square to;
    uint16_t flags;
};

struct BookNode {
    uint64_t key;
    BookMove moves[4];
    int num_moves;
};

// Book is populated at init time from FEN sequences
// This avoids hardcoding hash values that depend on Zobrist tables
static BookNode book_table[64];
static int book_size = 0;

static void add_entry(uint64_t key, const BookMove* mvs, int count) {
    if (book_size >= 64) return;
    book_table[book_size].key = key;
    book_table[book_size].num_moves = count;
    for (int i = 0; i < count; i++)
        book_table[book_size].moves[i] = mvs[i];
    book_size++;
}

static void init_book(Position& pos) {
    // === Startpos ===
    {
        uint64_t k = pos.key();
        // e4, d4, c4 (avoid Nf3 which leads to A06 disaster)
        BookMove mvs[] = {
            {E2, E4, MF_DOUBLE_PAWN},   // 1.e4
            {D2, D4, MF_DOUBLE_PAWN},   // 1.d4
            {C2, C4, MF_DOUBLE_PAWN},   // 1.c4
        };
        add_entry(k, mvs, 3);
    }

    // === After 1.e4 ===
    {
        pos.do_move(Move(E2, E4, MF_DOUBLE_PAWN));
        uint64_t k = pos.key();
        BookMove mvs[] = {
            {E7, E5, MF_DOUBLE_PAWN},   // 1...e5
            {C7, C5, MF_DOUBLE_PAWN},   // 1...c5
            {E7, E6, MF_QUIET},   // 1...e6
            {C7, C6, MF_QUIET},   // 1...c6
        };
        add_entry(k, mvs, 4);
        {
            pos.do_move(Move(E7, E5, MF_DOUBLE_PAWN));
            uint64_t k2 = pos.key();
            BookMove mvs2[] = {
                {G1, F3, MF_QUIET},   // 2.Nf3
            };
            add_entry(k2, mvs2, 1);

            pos.do_move(Move(G8, F6, MF_QUIET)); // 2...Nf6
            uint64_t k3 = pos.key();
            BookMove mvs3[] = {
                {F1, C4, MF_QUIET},   // 3.Bc4
                {F1, B5, MF_QUIET},   // 3.Bb5
                {D2, D4, MF_DOUBLE_PAWN},   // 3.d4 (Scotch)
            };
            add_entry(k3, mvs3, 3);
            pos.undo_move(Move(G8, F6, MF_QUIET));

            pos.do_move(Move(B8, C6, MF_QUIET)); // 2...Nc6
            uint64_t k4 = pos.key();
            BookMove mvs4[] = {
                {F1, B5, MF_QUIET},   // 3.Bb5 (Ruy Lopez)
                {F1, C4, MF_QUIET},   // 3.Bc4 (Italian)
                {D2, D4, MF_DOUBLE_PAWN},   // 3.d4 (Scotch)
            };
            add_entry(k4, mvs4, 3);
            pos.undo_move(Move(B8, C6, MF_QUIET));

            pos.undo_move(Move(E7, E5, MF_DOUBLE_PAWN));
        }

        // === After 1.e4 c5 ===
        {
            pos.do_move(Move(C7, C5, MF_DOUBLE_PAWN));
            uint64_t k2 = pos.key();
            BookMove mvs2[] = {
                {G1, F3, MF_QUIET},   // 2.Nf3
                {C2, C3, MF_QUIET},   // 2.c3 (Alapin)
            };
            add_entry(k2, mvs2, 2);

            pos.do_move(Move(D7, D6, MF_QUIET)); // 2...d6
            uint64_t k3 = pos.key();
            BookMove mvs3[] = {
                {D2, D4, MF_DOUBLE_PAWN},   // 3.d4
            };
            add_entry(k3, mvs3, 1);
            pos.undo_move(Move(D7, D6, MF_QUIET));

            pos.do_move(Move(B8, C6, MF_QUIET)); // 2...Nc6
            uint64_t k4 = pos.key();
            BookMove mvs4[] = {
                {D2, D4, MF_DOUBLE_PAWN},   // 3.d4
                {F1, B5, MF_QUIET},   // 3.Bb5 (Rossolimo)
            };
            add_entry(k4, mvs4, 2);
            pos.undo_move(Move(B8, C6, MF_QUIET));

            pos.undo_move(Move(C7, C5, MF_DOUBLE_PAWN));
        }

        // === After 1.e4 e6 ===
        {
            pos.do_move(Move(E7, E6, MF_QUIET));
            uint64_t k2 = pos.key();
            BookMove mvs2[] = {
                {D2, D4, MF_DOUBLE_PAWN},   // 2.d4
            };
            add_entry(k2, mvs2, 1);
            pos.undo_move(Move(E7, E6, MF_QUIET));
        }

        // === After 1.e4 c6 ===
        {
            pos.do_move(Move(C7, C6, MF_QUIET));
            uint64_t k2 = pos.key();
            BookMove mvs2[] = {
                {D2, D4, MF_DOUBLE_PAWN},   // 2.d4
            };
            add_entry(k2, mvs2, 1);
            pos.undo_move(Move(C7, C6, MF_QUIET));
        }

        pos.undo_move(Move(E2, E4, MF_DOUBLE_PAWN));
    }

    // === After 1.d4 ===
    {
        pos.do_move(Move(D2, D4, MF_DOUBLE_PAWN));
        uint64_t k = pos.key();
        // As Black: play d5, e6, c5 (avoid Nf6 which leads to London trouble)
        BookMove mvs[] = {
            {D7, D5, MF_DOUBLE_PAWN},   // 1...d5
            {E7, E6, MF_QUIET},   // 1...e6
            {C7, C5, MF_DOUBLE_PAWN},   // 1...c5
        };
        add_entry(k, mvs, 3);

        // === After 1.d4 d5 ===
        {
            pos.do_move(Move(D7, D5, MF_DOUBLE_PAWN));
            uint64_t k2 = pos.key();
            BookMove mvs2[] = {
                {C2, C4, MF_DOUBLE_PAWN},   // 2.c4
                {G1, F3, MF_QUIET},   // 2.Nf3
            };
            add_entry(k2, mvs2, 2);

            pos.do_move(Move(C2, C4, MF_DOUBLE_PAWN)); // 2.c4
            uint64_t k3 = pos.key();
            BookMove mvs3[] = {
                {E7, E6, MF_QUIET},   // 2...e6 (1 rank, not double)
                {C7, C6, MF_QUIET},   // 2...c6 (1 rank, not double)
            };
            add_entry(k3, mvs3, 2);
            pos.undo_move(Move(C2, C4, MF_DOUBLE_PAWN));

            pos.undo_move(Move(D7, D5, MF_DOUBLE_PAWN));
        }

        // === After 1.d4 e6 ===
        {
            pos.do_move(Move(E7, E6, MF_QUIET));
            uint64_t k2 = pos.key();
            BookMove mvs2[] = {
                {C2, C4, MF_DOUBLE_PAWN},   // 2.c4
            };
            add_entry(k2, mvs2, 1);
            pos.undo_move(Move(E7, E6, MF_QUIET));
        }

        // === After 1.d4 c5 ===
        {
            pos.do_move(Move(C7, C5, MF_DOUBLE_PAWN));
            uint64_t k2 = pos.key();
            BookMove mvs2[] = {
                {D4, C5, MF_CAPTURE},  // 2.dxc5
                {E2, E3, MF_QUIET},    // 2.e3
            };
            add_entry(k2, mvs2, 2);
            pos.undo_move(Move(C7, C5, MF_DOUBLE_PAWN));
        }

        pos.undo_move(Move(D2, D4, MF_DOUBLE_PAWN));
    }

    // === After 1.c4 ===
    {
        pos.do_move(Move(C2, C4, MF_DOUBLE_PAWN));
        uint64_t k = pos.key();
        BookMove mvs[] = {
            {E7, E5, MF_DOUBLE_PAWN},   // 1...e5
            {C7, C5, MF_DOUBLE_PAWN},   // 1...c5
            {E7, E6, MF_QUIET},   // 1...e6
            {G7, G6, MF_QUIET},   // 1...g6
        };
        add_entry(k, mvs, 4);

        pos.do_move(Move(E7, E5, MF_DOUBLE_PAWN));
        uint64_t k2 = pos.key();
        BookMove mvs2[] = {
            {G1, F3, MF_QUIET},   // 2.Nf3
            {B1, C3, MF_QUIET},   // 2.Nc3
        };
        add_entry(k2, mvs2, 2);
        pos.undo_move(Move(E7, E5, MF_DOUBLE_PAWN));

        pos.undo_move(Move(C2, C4, MF_DOUBLE_PAWN));
    }
}

static bool book_initialized = false;

Move book_probe(uint64_t pos_key) {
    if (!book_initialized) {
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        Position pos;
        pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        init_book(pos);
        book_initialized = true;
    }

    // Linear scan (book is small, ~30 entries)
    for (int i = 0; i < book_size; i++) {
        if (book_table[i].key == pos_key) {
            int idx = std::rand() % book_table[i].num_moves;
            const BookMove& bm = book_table[i].moves[idx];
            return Move(bm.from, bm.to, bm.flags);
        }
    }
    return MOVE_NONE;
}

} // namespace luminex
