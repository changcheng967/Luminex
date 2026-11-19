// v5_rel.h -- gen768t's relation inputs: attack relations and pawn pairs (nnue/V5-GEN768T.md, section 2).
// Which rows a position has, and which rows a move adds and removes, for one perspective or both. The rules
// and the index layout (LAYOUT = 1) are below; nnue.cpp sums the rows into the relation lanes.
//
// Frame. Squares are a1 = 0 ... h8 = 63. For perspective P with its own king on k:
// - o(s) = s XOR (56 if P is Black) XOR (7 if k is on files a-d), group 1's orientation;
// - a piece's side is own (colour P) or their;
// - own pawns capture toward rank 8 of the oriented board, their pawns toward rank 1.
//
// Group 2: which attacks are features. A relation (a on x) -> (v on y) holds when a attacks square y and v stands
// on it: a pawn attacks its two capture squares, a knight or king its jumps, a bishop, rook or queen each of its
// lines up to and including the first occupied square. a and v may be of one colour. A relation is not a feature
// when another input always carries it:
// 1. v is a king.
// 2. a is the perspective's own king (group 1's king bucket and v's piece-square carry it exactly).
// 3. a and v are both pawns (group 3).
// 4. Mutual attacks are told once:
//    - a and v of one type among N, B, R, Q: on one side, one feature per unordered pair; across sides, the feature
//      whose attacker is the own piece.
//    - Q -> B along a diagonal (the feature is B -> Q); Q -> R along a rank or file (the feature is R -> Q).
//    - B -> P or Q -> P when that pawn attacks the bishop's or queen's square (the feature is P -> B or P -> Q).
//
// Group 2: index. Attacker class a = (side, type), victim class v = (side, type), oriented squares x' = o(x), y' =
// o(y). Before indexing, a pair of one type on one side is ordered x' < y', and a pair of one type with a their
// attacker and an own victim swaps roles.
//
// E(a)[x'] is the set of squares class a attacks from x' on an empty board (pawns stand on ranks 2-7 only).
// G(a->v)[x'] is E(a)[x'] restricted as follows:
// - v a pawn: ranks 2-7 only;
// - Q -> B: rank and file lines only; Q -> R: diagonals only;
// - B -> P and Q -> P: without the squares from which that pawn attacks x' (an own pawn: the two squares diagonally
//   below x'; their pawn: the two diagonally above);
// - one type, one side: only y' > x'.
//
// index = base(a->v) + off(a->v)[x'] + popcount( G(a->v)[x'] & (bit(y') - 1) )
// off(a->v)[x'] = sum of |G(a->v)[u]| over u < x'
// base(a->v)    = sum of the sizes of all earlier blocks
//
// Block order: attacker side (own, their), attacker type (P, N, B, R, Q, K), victim side (own, their), victim type
// (P, N, B, R, Q); a block with no feature is skipped. Block sizes (rows):
//
//   attacker \ victim  own P  own N  own B  own R  own Q  their P  their N  their B  their R  their Q    rows
//   own P                  -     84     84     84     84        -       84       84       84       84     672
//   own N                284    168    336    336    336      284      336      336      336      336   3,088
//   own B                364    560    280    560    560      364      560      560      560      560   4,928
//   own R                672    896    896    448    896      672      896      896      896      896   8,064
//   own Q              1,036  1,456    896    560    728    1,036    1,456      896      560    1,456  10,080
//   their P                -     84     84     84     84        -       84       84       84       84     672
//   their N              284      -    336    336    336      284      168      336      336      336   2,752
//   their B              364    560      -    560    560      364      560      280      560      560   4,368
//   their R              672    896    896      -    896      672      896      896      448      896   7,168
//   their Q            1,036  1,456    896    560      -    1,036    1,456      896      560      728   8,624
//   their K              348    420    420    420    420      348      420      420      420      420   4,056
//
// Total 54,472; every row is reached by exactly one (class pair, x', y').
//
// Group 3: pawn pairs. A pawn's id is 48 * side + o(s) - 8 (0-95). A pair is two pawns on different squares whose
// files differ by at most one: 480 same-file and 1,008 adjacent-file rows.
//
// index = 54472 + rank of (lower id, higher id) among all valid pairs in lexicographic order
//
// - PPM = 0: a pair is active only when one pawn attacks the other's square: an own pawn defending an own pawn (70
//   rows), their pawn defending their pawn (70), an own and their pawn attacking each other (70).
// - PPM = 1: every pair on the board is active.
//
// What a move changes. Three steps on single squares; each emits removals (-) and additions (+) of physical
// features, and each perspective maps them to its own indices (a relation that is not a feature in that perspective
// is skipped).
//
// remove(z):     - every relation the piece on z attacks in, - every relation it is the victim of,
//                - its pawn pairs; empty z;
//                + (s -> w) for every slider s whose line reaches z, w the first piece beyond z
// add(z, p):     - (s -> w) for every slider s whose line reaches z, w the first piece beyond z;
//                put p on z;
//                + every relation p attacks in, + every relation it is the victim of, + its pawn pairs
// replace(z, p): - the old piece's relations and pawn pairs; put p on z; + the new piece's
//                (the occupancy is unchanged, so no line changes)
//
// A slider's line reaches z when z is on one of its lines with only empty squares between. Every + and - applies
// the rules above (kings as victims, mutual attacks told once, pawn on pawn as a pair).
//
//   Move                            Steps, in order
//   Quiet                           remove(from), add(to, piece)
//   Capture                         remove(from), replace(to, piece)
//   En passant                      remove(from), remove(captured pawn's square), add(to, pawn)
//   Promotion                       remove(from), then add(to, promoted piece) or replace(to, promoted piece)
//   Castling                        remove(king from), remove(rook from), add(king to, king), add(rook to, rook)
//   Null move                       nothing
//   Own king crosses the d/e files  the mover's perspective: every index changes, groups 2 and 3 are
//                                   rebuilt from the board; the other perspective: the steps above
//
// A feature a move both adds and removes may be cancelled or applied twice: the sums are exact either way.
//
// Conformance. White's view, then Black's: active features of groups 2 and 3, and the sum of their indices.
//
//   active features / sum of indices: PPM = 0 White, Black; PPM = 1 White, Black
//   start position:
//       33 / 923,020, 33 / 923,020; 69 / 2,897,986, 69 / 2,897,986
//   r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -:
//       47 / 1,268,017, 47 / 1,275,960; 77 / 2,913,373, 77 / 2,925,714
//   2kr3r/ppp2ppp/2n1bn2/2b1p3/4P3/2NP1N2/PPPB1PPP/2KR1B1R b - - (both kings mirrored):
//       34 / 762,273, 35 / 866,598; 63 / 2,353,560, 64 / 2,457,869
//   8/5pk1/6p1/8/3R4/6P1/r4PK1/8 w - -:
//       5 / 250,781, 5 / 229,138; 9 / 469,911, 9 / 448,268
//
// Single features, in the third position unless named:
//
//   Feature                                             View   Oriented    Block               base + off + rank
//   Ne5 attacks pawn d7 (second position)               White  e5 -> d7    own N -> their P    2132 + 168 + 6 = 2306
//   the same                                            Black  e4 -> d2    their N -> own P    27504 + 116 + 0 = 27620
//   Bc5 attacks pawn f2                                 White  f5 -> c2    their B -> own P    30256 + 228 + 0 = 30484
//   the same                                            Black  f4 -> c7    own B -> their P    6084 + 152 + 6 = 6242
//   Rd1 defends Bd2                                     White  e1 -> e2    own R -> own B      10256 + 56 + 7 = 10319
//   rooks d8 and h8 see each other                      Black  a1, e1      own R -> own R      11152 + 0 + 3 = 11155
//   the same                                            White  a8, e8      their R -> their R  40448 + 420 + 3 = 40871
//   pawns e4 and d5 attack each other (after 1. e4 d5)  White  ids 20, 75  pawn pair           54472 + 577 = 55049
//   the same                                            Black  ids 19, 76  pawn pair           54472 + 552 = 55024
#pragma once
#include "types.h"
#include <cstdint>

namespace luminex::v5rel {

// A board as the finder reads it: the mailbox (a Piece per square: 0-5 White P N B R Q K, 6-11 Black, 12 empty)
// and a bitboard per piece, with the occupancy.
struct Bd {
    uint8_t pc[64];
    uint64_t by[13];
    uint64_t occ;
};

constexpr int MAX_ROWS = 96;     // rows one move may add or remove for one perspective; more, and that side is rebuilt
constexpr int MAX_ACTIVE = 256;  // rows a position can have (the spec's bound is 248)

struct RowLists {
    int na[2], ns[2];
    bool overflow[2];
    uint16_t add[2][MAX_ROWS], sub[2][MAX_ROWS];
};

void init();     // builds the tables (once; later calls return at once)
int n_rows();    // 55,960: the 54,472 relation rows, then the 1,488 pawn-pair rows

// The rows the move m adds and removes for each perspective p with need[p], from the board before it.
// ppm: 0 = pawn pairs in contact, 1 = every pawn pair on the same or adjacent files.
void find(const Bd& before, Move m, int ppm, const bool need[2], RowLists& out);

// Every active row of one perspective on a board, in the order they are found; returns their number.
int enumerate(const Bd& b, int persp, int ppm, uint16_t* out);

}  // namespace luminex::v5rel
