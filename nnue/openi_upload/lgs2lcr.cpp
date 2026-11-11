// lgs2lcr.cpp — convert one Leela T91 LGS shard (xushawn/t91-lgs, format v1, docs/FORMAT.md)
// into LCR1 training records by replaying every standard-start game with nnue-pytorch's chess
// library. Records are built exactly as binpack2lcr.cpp builds them (pieces, side to move,
// quiet/capture/check class, game result bits, position hash); the score is a centipawn value
// from best_q (score_cp) and the result from tb_result, as the t91 binpacks carry them.
//
// Usage: ./lgs2lcr SHARD.lgs OUT.lcr lc0_policy_moves.txt [--debug DEBUG.bin] [--score best|root]
//   --debug writes, per record (55 bytes), the 25-byte key (occ, pcs, side to move), six f32
//   values (best_q, best_d, root_q, root_d, orig_q, played_q), i8 result_q, i8 tb_result,
//   u16 best_idx and u16 played_idx, for checking against binpack records.
//   Chess960 and FEN starts (about 4% of T91 games) are skipped.
//
// Build: g++ -O2 -std=c++20 -I <nnue-pytorch>/data_loader/cpp -o lgs2lcr lgs2lcr.cpp -lzstd
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <zstd.h>

#include "lib/binpack.h"

struct CompactRec {
    std::uint64_t occ;
    std::uint8_t pcs[16];
    std::int16_t target;
    std::uint8_t flags;
    std::uint8_t npieces;
    std::uint32_t poshash;
};
static_assert(sizeof(CompactRec) == 32, "LCR1 records are 32 bytes");

static inline std::uint64_t mix64(std::uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

struct Col {
    std::string name;
    int kind = 0, dtype = 0, filter = 0, base = 0;
    std::uint64_t raw_len = 0, comp_len = 0, offset = 0;
    std::vector<std::uint8_t> raw;
    bool done = false;
};

static int dtype_size(int d) {
    switch (d) {
        case 2: case 3: case 7: return 2;
        case 4: case 5: case 8: return 4;
        case 6: return 8;
        default: return 1;
    }
}

[[noreturn]] static void fail(const char* msg) { std::fprintf(stderr, "FAIL: %s\n", msg); std::exit(1); }

template <typename T> static T rd(const std::uint8_t* p) { T v; std::memcpy(&v, p, sizeof(T)); return v; }

static void decode(std::vector<Col>& cols, int i, const std::uint8_t* payload) {
    Col& c = cols[i];
    if (c.done) return;
    std::vector<std::uint8_t> z(c.raw_len);
    if (c.raw_len) {
        const std::size_t got = ZSTD_decompress(z.data(), c.raw_len, payload + c.offset, c.comp_len);
        if (ZSTD_isError(got) || got != c.raw_len) fail(("zstd frame of column " + c.name).c_str());
    }
    if (c.filter == 1 || c.filter == 2) {   // byte shuffle: W planes of N bytes -> N values of W bytes
        const int w = dtype_size(c.dtype);
        const std::size_t n = c.raw_len / w;
        std::vector<std::uint8_t> r(c.raw_len);
        for (int k = 0; k < w; ++k)
            for (std::size_t j = 0; j < n; ++j) r[j * w + k] = z[k * n + j];
        z.swap(r);
    }
    if (c.filter == 2) {                    // then XOR with the base column's raw bytes
        decode(cols, c.base, payload);
        const auto& b = cols[c.base].raw;
        if (b.size() != z.size()) fail(("xor base length of column " + c.name).c_str());
        for (std::size_t j = 0; j < z.size(); ++j) z[j] ^= b[j];
    }
    c.raw.swap(z);
    c.done = true;
}

// centipawns, side to move, from a q in [-1, 1] (W - L): lc0's conversion 660.6 q / (1 - 0.9751875 q^10)
// and +-26624 once |q| >= 0.975. Checked against binpack2lcr records of the same February games
// (t91 binpacks, 20,802 unique endgame positions): exact on 97.3% of ordinary scores; the rest of the
// big values are tablebase wins the binpacks scale by distance, all far past the trainer's 1500 clip.
static std::int16_t score_cp(float q) {
    if (!std::isfinite(q)) return 0;
    const double x = static_cast<double>(q);
    if (std::fabs(x) >= 0.975) return static_cast<std::int16_t>(x > 0 ? 26624 : -26624);
    const double cp = 660.6 * x / (1.0 - 0.9751875 * std::pow(x, 10));
    return static_cast<std::int16_t>(std::max(-26624.0, std::min(26624.0, std::round(cp))));
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s SHARD.lgs OUT.lcr lc0_policy_moves.txt [--debug DEBUG.bin] [--score best|root]\n", argv[0]);
        return 1;
    }
    std::string debug_path, score_field = "best";
    for (int i = 4; i + 1 < argc; i += 2) {
        if (std::string(argv[i]) == "--debug") debug_path = argv[i + 1];
        else if (std::string(argv[i]) == "--score") score_field = argv[i + 1];
    }
    // the 1858 policy moves, side to move's frame: 'e2e4', 'a7a8q'
    std::vector<std::array<int, 3>> moves;
    {
        FILE* f = std::fopen(argv[3], "r");
        if (!f) fail("open the policy move list");
        char line[32];
        while (std::fgets(line, sizeof line, f)) {
            const std::size_t n = std::strlen(line);
            if (n < 4) continue;
            const int from = (line[0] - 'a') + 8 * (line[1] - '1'), to = (line[2] - 'a') + 8 * (line[3] - '1');
            const int promo = (n > 4 && line[4] >= 'a' && line[4] <= 'z') ? line[4] : 0;
            moves.push_back({from, to, promo});
        }
        std::fclose(f);
        if (moves.size() != 1858) fail("the policy move list does not have 1858 moves");
    }
    std::vector<std::uint8_t> buf;
    {
        FILE* f = std::fopen(argv[1], "rb");
        if (!f) fail("open the shard");
        std::fseek(f, 0, SEEK_END);
        buf.resize(static_cast<std::size_t>(std::ftell(f)));
        std::fseek(f, 0, SEEK_SET);
        if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) fail("read the shard");
        std::fclose(f);
    }
    if (buf.size() < 24 || std::memcmp(buf.data(), "LGS1", 4) != 0) fail("not an LGS1 shard");
    if (std::memcmp(buf.data() + buf.size() - 4, "LGSE", 4) != 0) fail("no LGSE trailer");
    const std::uint64_t idx = rd<std::uint64_t>(buf.data() + buf.size() - 12);
    if (std::memcmp(buf.data() + idx, "LGSI", 4) != 0) fail("no LGSI index");
    const std::uint32_t n_blocks = rd<std::uint32_t>(buf.data() + idx + 4);

    FILE* out = std::fopen(argv[2], "wb");
    if (!out) fail("open the output");
    FILE* dbg = debug_path.empty() ? nullptr : std::fopen(debug_path.c_str(), "wb");
    long long n_games = 0, n_skipped_games = 0, n_rec = 0, n_bad = 0, nq = 0, nc = 0, ni = 0;

    for (std::uint32_t b = 0; b < n_blocks; ++b) {
        const std::uint64_t off = rd<std::uint64_t>(buf.data() + idx + 8 + 16ull * b);
        const std::uint8_t* p = buf.data() + off;
        if (std::memcmp(p, "LGSB", 4) != 0) fail("block magic");
        const std::uint32_t ng = rd<std::uint32_t>(p + 4), np = rd<std::uint32_t>(p + 8), nc_cols = rd<std::uint32_t>(p + 12);
        std::vector<Col> cols(nc_cols);
        std::uint64_t poff = 0;
        for (std::uint32_t i = 0; i < nc_cols; ++i) {
            const std::uint8_t* e = p + 16 + 40ull * i;
            Col& c = cols[i];
            c.name.assign(reinterpret_cast<const char*>(e), strnlen(reinterpret_cast<const char*>(e), 16));
            c.kind = e[16]; c.dtype = e[17]; c.filter = e[18]; c.base = e[19];
            c.raw_len = rd<std::uint64_t>(e + 20); c.comp_len = rd<std::uint64_t>(e + 28);
            c.offset = poff; poff += c.comp_len;
        }
        const std::uint8_t* payload = p + 16 + 40ull * nc_cols;
        std::map<std::string, int> at;
        for (std::uint32_t i = 0; i < nc_cols; ++i) at[cols[i].name] = static_cast<int>(i);
        auto need = [&](const char* name, int dtype) -> const std::uint8_t* {
            auto it = at.find(name);
            if (it == at.end()) fail((std::string("missing column ") + name).c_str());
            decode(cols, it->second, payload);
            if (cols[it->second].dtype != dtype) fail((std::string("dtype of ") + name).c_str());
            return cols[it->second].raw.data();
        };
        const auto* g_npos = need("n_pos", 2);
        const auto* g_sw = need("start_w", 2);
        const auto* g_sb = need("start_b", 2);
        const auto* g_res = need("result", 1);
        const auto* played = need("played_idx", 2);
        const auto* best_idx = need("best_idx", 2);
        const auto* p_result_q = need("result_q", 1);
        const auto* p_tb_result = need("tb_result", 1);
        const auto* best_q = need("best_q", 8);
        const auto* best_d = need("best_d", 8);
        const auto* root_q = need("root_q", 8);
        const auto* root_d = need("root_d", 8);
        const auto* orig_q = need("orig_q", 8);
        const auto* played_q = need("played_q", 8);
        const auto* score_q = score_field == "root" ? root_q : best_q;

        std::uint64_t row = 0;
        for (std::uint32_t g = 0; g < ng; ++g) {
            const std::uint16_t npos = rd<std::uint16_t>(g_npos + 2ull * g);
            const std::uint16_t sw = rd<std::uint16_t>(g_sw + 2ull * g), sb = rd<std::uint16_t>(g_sb + 2ull * g);
            const int res_white = static_cast<std::int8_t>(g_res[g]);
            ++n_games;
            if (sw != 518 || sb != 518) { ++n_skipped_games; row += npos; continue; }
            chess::Position pos = chess::Position::startPosition();
            for (std::uint16_t k = 0; k < npos; ++k, ++row) {
                const bool white = pos.sideToMove() == chess::Color::White;
                // the played move, from the side to move's frame
                const std::uint16_t pi = rd<std::uint16_t>(played + 2ull * row);
                if (pi >= moves.size()) { ++n_bad; row += npos - k; break; }
                int from = moves[pi][0], to = moves[pi][1];
                const int promo = moves[pi][2];
                if (!white) { from ^= 56; to ^= 56; }
                chess::Move mv{};
                bool found = false;
                chess::movegen::forEachLegalMove(pos, [&](chess::Move m) {
                    if (found || static_cast<int>(chess::ordinal(m.from)) != from || static_cast<int>(chess::ordinal(m.to)) != to) return;
                    if (m.type == chess::MoveType::Promotion) {
                        const auto pt = m.promotedPiece.type();
                        const char want = promo ? static_cast<char>(promo) : 'n';
                        const char have = pt == chess::PieceType::Queen ? 'q' : pt == chess::PieceType::Rook ? 'r'
                                        : pt == chess::PieceType::Bishop ? 'b' : 'n';
                        if (have != want) return;
                    } else if (promo) {
                        return;
                    }
                    mv = m; found = true;
                });
                if (!found) { ++n_bad; row += npos - k; break; }
                // the record, as binpack2lcr writes it
                CompactRec r{};
                r.occ = pos.piecesBB().bits();
                int n = 0;
                for (std::uint64_t bb = r.occ; bb; bb &= bb - 1, ++n) {
                    const int sq = __builtin_ctzll(bb);
                    const auto pc = pos.pieceAt(chess::Square(sq));
                    const int code = static_cast<int>(chess::ordinal(pc.type())) + 6 * static_cast<int>(chess::ordinal(pc.color()));
                    if (n < 32) r.pcs[n >> 1] |= static_cast<std::uint8_t>(code << (4 * (n & 1)));
                }
                const bool capture = pos.pieceAt(mv.to) != chess::Piece::none() && pos.pieceAt(mv.to).color() != pos.pieceAt(mv.from).color();
                const int q = !pos.checkers().isEmpty() ? 2 : (capture ? 1 : 0);
                // the result as the binpacks carry it: lc0's Syzygy-rescored result where known (it matched
                // 99.83% of the same games' binpack results, the raw game result 99.28%), side to move's view
                const int tbr = static_cast<std::int8_t>(p_tb_result[row]);
                const int res_stm = tbr != 127 ? tbr : (white ? res_white : -res_white);
                const int res = res_stm > 0 ? 2 : (res_stm < 0 ? 0 : 1);
                const float sq_ = rd<float>(score_q + 4ull * row);
                r.target = score_cp(sq_);
                r.flags = static_cast<std::uint8_t>((white ? 1 : 0) | (q << 1) | (1 << 3) | (res << 4));
                r.npieces = static_cast<std::uint8_t>(n);
                std::uint64_t lo, hi;
                std::memcpy(&lo, r.pcs, 8);
                std::memcpy(&hi, r.pcs + 8, 8);
                r.poshash = static_cast<std::uint32_t>(mix64(r.occ ^ mix64(lo ^ mix64(hi ^ (white ? 1 : 0)))) >> 32);
                std::fwrite(&r, sizeof r, 1, out);
                ++n_rec; nq += q == 0; nc += q == 1; ni += q == 2;
                if (dbg) {
                    std::uint8_t key[25];
                    std::memcpy(key, &r.occ, 8); std::memcpy(key + 8, r.pcs, 16); key[24] = white ? 1 : 0;
                    std::fwrite(key, 1, 25, dbg);
                    const float v[6] = {rd<float>(best_q + 4ull * row), rd<float>(best_d + 4ull * row), rd<float>(root_q + 4ull * row),
                                        rd<float>(root_d + 4ull * row), rd<float>(orig_q + 4ull * row), rd<float>(played_q + 4ull * row)};
                    std::fwrite(v, 4, 6, dbg);
                    const std::int8_t rq = static_cast<std::int8_t>(p_result_q[row]), tb = static_cast<std::int8_t>(p_tb_result[row]);
                    const std::uint16_t bi = rd<std::uint16_t>(best_idx + 2ull * row);
                    std::fwrite(&rq, 1, 1, dbg); std::fwrite(&tb, 1, 1, dbg);
                    std::fwrite(&bi, 2, 1, dbg); std::fwrite(&pi, 2, 1, dbg);
                }
                pos.doMove(mv);
            }
        }
        if (row != np) {
            std::fprintf(stderr, "note: block %u rows %llu, header n_pos %u\n", b, static_cast<unsigned long long>(row), np);
        }
    }
    std::fclose(out);
    if (dbg) std::fclose(dbg);
    std::fprintf(stderr, "DONE %lld records from %lld games (%lld non-standard starts skipped, %lld games cut at an undecodable move); "
                 "quiet %lld, capture %lld, check %lld\n", n_rec, n_games, n_skipped_games, n_bad, nq, nc, ni);
    return 0;
}
