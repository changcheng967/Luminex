// sf_binpack_dump.cpp — dump an SF-relabel binpack (v6 training data format)
// to a filtered "fen<TAB>cp" TSV on stdout, for NNUE_FENEVAL_FILE training.
//
// Build (against a checkout of official-stockfish/nnue-pytorch):
//   g++ -O2 -std=c++17 -I <nnue-pytorch>/data_loader/cpp \
//       -o sf_binpack_dump sf_binpack_dump.cpp -lzstd
//
// Usage:
//   ./sf_binpack_dump test80-full.binpack            [max_positions]  > out.tsv
//
// Filter (same as the sf_train_50m recipe): |score| < 800 cp, 8 <= ply <= 200.
// Score convention and FEN format come straight from TrainingDataEntry /
// chess::Position::fen() in the official headers — identical to the original
// 50M harvest, verified by byte-comparing sample heads against sf_train_50m.
#include <cstdio>
#include <cstdlib>
#include <string>

#include "lib/binpack.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s in.binpack [max_positions]\n", argv[0]);
        return 1;
    }
    const long long maxPos = (argc > 2) ? std::atoll(argv[2]) : 0;

    binpack::CompressedTrainingDataEntryReader reader(argv[1]);

    long long scanned = 0, kept = 0;
    char line[256];
    while (reader.hasNext()) {
        const auto e = reader.next();
        ++scanned;

        if (e.score < -800 || e.score > 800) continue;
        if (e.ply < 8 || e.ply > 200) continue;

        const std::string fen = e.pos.fen();
        // TSV contract: the cp column is WHITE-relative. The binpack's score is
        // SIDE-TO-MOVE-relative (nnue-pytorch's convention), so negate it when
        // black is to move. Writing it raw flipped half the training targets
        // once and collapsed a full run.
        const auto sp1 = fen.find(' ');
        const bool black_to_move = (sp1 != std::string::npos
                                    && fen.size() > sp1 + 1 && fen[sp1 + 1] == 'b');
        const int cp_out = black_to_move ? -static_cast<int>(e.score)
                                         : static_cast<int>(e.score);
        // Quiet flags for round-3 training: the engine consumes eval at quiet
        // nodes only (qsearch resolves checks and captures), so the loss should
        // weight positions by the same classes. Both derive from the stored
        // move: in-check = the mover was in check; capture-best = the binpack's
        // best move takes material. Third column: 0 quiet, 1 capture, 2 check.
        int quiet = 0;
        if (!e.pos.checkers().isEmpty()) quiet = 2;
        else if (e.isCapturingMove()) quiet = 1;
        const int n = std::snprintf(line, sizeof line, "%s\t%d\t%d\n",
                                    fen.c_str(), cp_out, quiet);
        std::fwrite(line, 1, n, stdout);
        ++kept;

        if ((scanned % 10'000'000) == 0)
            std::fprintf(stderr, "scanned %lldM, kept %lldM\n",
                         scanned / 1'000'000, kept / 1'000'000);
        if (maxPos && kept >= maxPos) break;
    }
    std::fprintf(stderr, "DONE scanned=%lld kept=%lld\n", scanned, kept);
    return 0;
}
