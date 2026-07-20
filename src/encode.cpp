// encode.cpp — PGN (.pgn/.pgn.gz) -> Game-Pack raw frame (pipe to xz for the .xz).
//
// MULTI-THREADED (one pass, NO merge): N threads each process a shard of the input PGN
// files, writing per-thread mv/ev temp files + per-thread game-entry buffers. A single
// shared FEN dictionary (mutex-protected, one global fen_idx space) means NO fen_idx
// remap is needed at assembly — eliminating the alignment bug the old multi-frame merger
// had. Assembly concatenates per-thread state in a fixed thread order so game_entries,
// mv, and ev stay perfectly aligned.
//
// Frame (one xz stream over): [len_hdr:u64][hdr][len_mv:u64][mv][len_ev:u64][ev]
//   hdr = u32 nfens; [u16 len + bytes]*nfens; per game [n_pos:u16][stm:u8][fen_idx:u32][start_eval:s16]
//   mv  = per ply u8 index into the SORTED legal list (from,to,promo)
//   ev  = per game (n_pos-1) deltas: s8, or 0x80 + s16-abs escape
// n_pos is u16 (max 65535 plies) — NO 255 cap, so long games no longer shift the mv stream.
// Eval stored = WHITE-relative (mover->white), quantized to QUANT cp. featurize flips to stm-rel.
//
// USAGE:  ./luminex-encode [--quant 8] [--threads 8] [-o frame.raw] f1.pgn.gz f2.pgn.gz ...
//         ./luminex-encode ... | xz -T0 -6 > gamepack.xz
#include "board.h"
#include "movegen.h"
#include "bitboard.h"
#include "types.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <climits>
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <fstream>

#ifdef _WIN32
  #define popen  _popen
  #define pclose _pclose
#else
  #include <unistd.h>   // getpid
#endif

namespace luminex { }
using namespace luminex;

static inline uint32_t move_sort_key(Move m) {
    uint32_t promo = m.is_promotion() ? (uint32_t)m.promotion_type() : 0;
    return ((uint32_t)m.from() << 12) | ((uint32_t)m.to() << 4) | promo;
}

// ============================================================================
// PGN streaming reader -> events: 1 token, 2 comment(in tok), 3 header(hkey/hval), 0 EOF.
// ============================================================================
struct PgnReader {
    FILE* f; int peek = -2;
    std::string tok, hkey, hval;
    explicit PgnReader(FILE* f_) : f(f_) {}
    int get() { if (peek != -2) { int c = peek; peek = -2; return c; } return fgetc(f); }
    int next() {
        tok.clear();
        int c;
        while ((c = get()) != EOF) {
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') continue;
            if (c == '{') {                                   // comment -> preceding move's eval
                tok.clear();
                while ((c = get()) != '}' && c != EOF) tok.push_back((char)c);
                return 2;
            }
            if (c == '(') {                                   // variation: skip (nested)
                int depth = 1;
                while (depth > 0 && (c = get()) != EOF) {
                    if (c == '(') depth++;
                    else if (c == ')') depth--;
                    else if (c == '{') while ((c = get()) != '}' && c != EOF);
                }
                continue;
            }
            if (c == '$') {                                   // NAG: skip token
                while ((c = get()) != EOF && c != ' ' && c != '\t' && c != '\n' && c != '\r');
                continue;
            }
            if (c == '[') {                                   // header [Key "Value"]
                hkey.clear(); hval.clear();
                int d = get();
                while (d != EOF && d != ' ' && d != '\t' && d != '"' && d != ']') { hkey.push_back((char)d); d = get(); }
                while (d == ' ' || d == '\t') d = get();
                if (d == '"') {
                    d = get();
                    while (d != EOF && d != '"' && d != ']') { hval.push_back((char)d); d = get(); }
                    while (d != EOF && d != ']') d = get();
                } else { while (d != EOF && d != ']') d = get(); }
                return 3;
            }
            // movetext token (SAN / move number / result)
            tok.clear(); tok.push_back((char)c);
            while ((c = get()) != EOF && c != ' ' && c != '\t' && c != '\n' && c != '\r'
                   && c != '{' && c != '(' && c != '$' && c != '[') tok.push_back((char)c);
            if (c == '{' || c == '(' || c == '$' || c == '[') peek = c;
            return 1;
        }
        return 0;
    }
};

static bool is_result(const std::string& s) {
    return s == "1-0" || s == "0-1" || s == "1/2-1/2" || s == "*";
}
static bool is_move_number(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) if (!(c >= '0' && c <= '9') && c != '.') return false;
    return true;
}

// SAN -> Move (resolved against pos's legal moves). Fills list with the legal moves
// (unsorted) and outputs nlegal. Returns false if unparseable/ambiguous. Caller sorts list
// once and finds the index — avoids a second generate<GEN_LEGAL> per move (~2x faster).
static bool parse_san(Position& pos, const std::string& san0, ExtMove* list, int& nlegal, Move& out) {
    std::string s = san0;
    while (!s.empty() && (s.back() == '+' || s.back() == '#' || s.back() == '!'
                          || s.back() == '?' || s.back() == ' ')) s.pop_back();
    if (s.empty()) return false;
    ExtMove* end = generate<GEN_LEGAL>(pos, list);
    nlegal = (int)(end - list);
    bool ck = (s == "O-O" || s == "0-0"), cq = (s == "O-O-O" || s == "0-0-0");
    if (ck || cq) {
        for (int i = 0; i < nlegal; ++i) {
            Move m = list[i].move;
            if (m.is_castling()) { int dx = (int)m.to() - (int)m.from();
                if ((ck && dx == 2) || (cq && dx == -2)) { out = m; return true; } }
        }
        return false;
    }
    size_t i = 0, n = s.size();
    PieceType ptype = PAWN;
    if (s[0] == 'K' || s[0] == 'Q' || s[0] == 'R' || s[0] == 'B' || s[0] == 'N') {
        switch (s[0]) { case 'K': ptype = KING; break; case 'Q': ptype = QUEEN; break;
            case 'R': ptype = ROOK; break; case 'B': ptype = BISHOP; break; case 'N': ptype = KNIGHT; break; }
        i = 1;
    }
    PieceType promo = PT_NONE;
    size_t eq = s.find('=');
    if (eq != std::string::npos && eq + 1 < n) {
        switch (s[eq + 1]) { case 'Q': promo = QUEEN; break; case 'R': promo = ROOK; break;
            case 'B': promo = BISHOP; break; case 'N': promo = KNIGHT; break; default: return false; }
        n = eq;
    }
    if (n < i + 2) return false;
    char df = s[n - 2], dr = s[n - 1];
    if (df < 'a' || df > 'h' || dr < '1' || dr > '8') return false;
    int dest = (df - 'a') + (dr - '1') * 8;
    int from_file = -1, from_rank = -1;
    for (; i < n - 2; ++i) {
        if (s[i] >= 'a' && s[i] <= 'h') from_file = s[i] - 'a';
        else if (s[i] >= '1' && s[i] <= '8') from_rank = s[i] - '1';
        else if (s[i] == 'x') {}
        else return false;
    }
    Move found = MOVE_NONE; int matches = 0;
    for (int k = 0; k < nlegal; ++k) {
        Move m = list[k].move;
        if ((int)m.to() != dest) continue;
        if (piece_type_of(pos.piece_on(m.from())) != ptype) continue;
        if (m.is_promotion()) { if (promo == PT_NONE || m.promotion_type() != promo) continue; }
        else { if (promo != PT_NONE) continue; }
        if (from_file >= 0 && (m.from() & 7) != from_file) continue;
        if (from_rank >= 0 && (m.from() >> 3) != from_rank) continue;
        found = m; ++matches;
    }
    if (matches != 1) return false;
    out = found; return true;
}

static const int EVAL_NONE = INT_MIN;
static int eval_from_comment(const std::string& c) {
    if (c.empty()) return EVAL_NONE;
    size_t p = c.find("%eval");
    if (p != std::string::npos) {
        size_t j = p + 5; while (j < c.size() && (c[j] == ' ' || c[j] == '\t')) ++j;
        bool neg = false, mate = false;
        if (j < c.size() && (c[j] == '-' || c[j] == '+')) { neg = (c[j] == '-'); ++j; }
        if (j < c.size() && c[j] == '#') { mate = true; ++j; }
        if (j < c.size() && (c[j] == '-' || c[j] == '+')) { neg = (c[j] == '-'); ++j; }
        std::string num;
        while (j < c.size() && (c[j] == '.' || (c[j] >= '0' && c[j] <= '9'))) num.push_back(c[j++]);
        if (num.empty()) return EVAL_NONE;
        double v = atof(num.c_str());
        if (mate) { int mv = (int)llround(v); return (32000 - std::abs(mv) * 8) * (neg ? -1 : 1); }
        if (std::abs(v) > 320) return neg ? -32000 : 32000;
        return (int)llround(v * 100) * (neg ? -1 : 1);
    }
    for (size_t k = 0; k + 1 < c.size(); ++k) {
        size_t s = k; bool neg = false;
        if (c[s] == '-' || c[s] == '+') { neg = (c[s] == '-'); ++s; }
        if (s >= c.size() || !(c[s] >= '0' && c[s] <= '9')) continue;
        if (k > 0 && c[k - 1] >= '0' && c[k - 1] <= '9') continue;
        size_t e = s;
        while (e < c.size() && ((c[e] >= '0' && c[e] <= '9') || c[e] == '.')) ++e;
        if (e >= c.size() || c[e] != '/') continue;
        double v = atof(c.substr(s, e - s).c_str());
        if (std::abs(v) > 320) return neg ? -32000 : 32000;
        return (int)llround(v * 100) * (neg ? -1 : 1);
    }
    return EVAL_NONE;
}

// ============================================================================
// Shared state across threads: ONE global FEN dictionary (mutex-protected). Because every
// game's fen_idx is allocated from this single dict, no remap is needed at assembly time —
// which is what eliminates the FEN<->moves misalignment the old multi-frame merger had.
// ============================================================================
struct Shared {
    std::mutex mtx;
    std::unordered_map<std::string, uint32_t> fen_dict;
    std::vector<std::string> fen_list;
    uint32_t get_fen_idx(const std::string& fen) {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = fen_dict.find(fen);
        if (it != fen_dict.end()) return it->second;
        uint32_t idx = (uint32_t)fen_list.size();
        fen_dict[fen] = idx;
        fen_list.push_back(fen);
        return idx;
    }
};

struct PerThread {
    int tid = 0;
    std::vector<uint8_t> game_entries;   // per game: [n_pos:u16][stm:u8][fen_idx:u32][start_eval:s16]
    std::string mv_path, ev_path;        // per-thread mv/ev temp files
    uint64_t games = 0, pos = 0, bad_san = 0;
};

static void process_files(const std::vector<std::string>& files, Shared& sh, PerThread& pt, int quant) {
    std::string tmppre = "/tmp/_enc_" + std::to_string((long)getpid()) + "_" + std::to_string(pt.tid);
    pt.mv_path = tmppre + "_mv.tmp";
    pt.ev_path = tmppre + "_ev.tmp";
    FILE* fmv = fopen(pt.mv_path.c_str(), "wb");
    FILE* fev = fopen(pt.ev_path.c_str(), "wb");
    if (!fmv || !fev) { fprintf(stderr, "T%d: cannot open temp files in /tmp\n", pt.tid); return; }

    Position pos;
    ExtMove list[MAX_MOVES];

    for (const auto& path : files) {
        bool gz = (path.size() > 3 && path.substr(path.size() - 3) == ".gz");
        FILE* f = gz ? popen(("gunzip -c " + path).c_str(), "r") : fopen(path.c_str(), "rb");
        if (!f) { fprintf(stderr, "T%d: cannot open %s\n", pt.tid, path.c_str()); continue; }
        PgnReader rd(f);

        std::string cur_fen;
        bool have_fen = false, game_set = false, start_stm_white = true, in_movetext = false, skip_game = false;
        bool pending = false;
        uint8_t pending_idx = 0; bool pending_stm_white = false; int pending_eval = EVAL_NONE;
        bool first_ply = true; int16_t prev_eq = 0, start_eval = 0; int game_npos = 0;

        auto emit_pend = [&]() -> bool {
            if (!pending || pending_eval == EVAL_NONE) return false;
            bool mover_white = !pending_stm_white;
            int eq_raw = mover_white ? pending_eval : -pending_eval;
            // Floor-division quantize to match python's (wr // QUANT) * QUANT.
            int q = eq_raw / quant;
            if (eq_raw % quant != 0 && eq_raw < 0) q -= 1;
            int16_t eq = (int16_t)(q * quant);
            if (first_ply) { start_eval = eq; first_ply = false; }
            else {
                int d = (int)eq - (int)prev_eq;
                if (d >= -127 && d <= 127) { int8_t b = (int8_t)d; fwrite(&b, 1, 1, fev); }
                else { int8_t esc = -128; fwrite(&esc, 1, 1, fev); fwrite(&eq, 2, 1, fev); }
            }
            prev_eq = eq;
            fputc((int)pending_idx, fmv);
            game_npos++; pt.pos++;
            return true;
        };
        auto flush_game = [&]() {
            emit_pend();
            if (game_set && game_npos > 0) {
                std::string fen = have_fen ? cur_fen : std::string("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                uint32_t fidx = sh.get_fen_idx(fen);
                uint16_t np = (uint16_t)game_npos;   // u16: no 255 cap (max 65535 plies)
                pt.game_entries.push_back((uint8_t)(np & 0xFF));
                pt.game_entries.push_back((uint8_t)((np >> 8) & 0xFF));
                pt.game_entries.push_back(start_stm_white ? 1 : 0);
                pt.game_entries.insert(pt.game_entries.end(), (uint8_t*)&fidx, (uint8_t*)&fidx + 4);
                pt.game_entries.insert(pt.game_entries.end(), (uint8_t*)&start_eval, (uint8_t*)&start_eval + 2);
                pt.games++;
            }
            game_set = false; have_fen = false; in_movetext = false;
            pending = false; pending_eval = EVAL_NONE; first_ply = true; game_npos = 0;
        };

        int ev;
        while ((ev = rd.next()) != 0) {
            if (ev == 3) {                                  // header
                skip_game = false;                              // new game -> stop skipping
                if (in_movetext) { flush_game(); }          // new game starts
                if (rd.hkey == "FEN") { cur_fen = rd.hval; have_fen = true; }
                continue;
            }
            if (ev == 2) { pending_eval = eval_from_comment(rd.tok); continue; }  // comment
            // ev == 1: token
            const std::string& tok = rd.tok;
            if (skip_game) continue;                        // skip rest of a diverged game until next header
            if (is_result(tok)) { flush_game(); continue; }
            if (is_move_number(tok)) continue;
            // SAN move
            if (emit_pend() == false && pending && pending_eval == EVAL_NONE) {
                flush_game();                               // previous move had no eval -> truncate
            }
            pending_eval = EVAL_NONE;
            if (!game_set) {
                pos.set(have_fen ? cur_fen : std::string("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));
                start_stm_white = (pos.side_to_move() == WHITE);
                game_set = true; first_ply = true; game_npos = 0; prev_eq = 0;
                pending = false;
            }
            in_movetext = true;
            Move mv; int nlegal;
            if (!parse_san(pos, tok, list, nlegal, mv)) {
                pt.bad_san++;
                if (pt.bad_san <= 20) fprintf(stderr, "T%d BADSAN: '%s' fen='%s'\n", pt.tid, tok.c_str(), pos.fen().c_str());
                flush_game();
                skip_game = true;   // stop the cascade: skip until the next game header
                continue;
            }
            std::sort(list, list + nlegal, [](const ExtMove& a, const ExtMove& b){ return move_sort_key(a.move) < move_sort_key(b.move); });
            uint8_t idx = 255;
            for (int k = 0; k < nlegal; ++k) if (list[k].move == mv) { idx = (uint8_t)k; break; }
            pos.do_move(mv);
            pending = true; pending_idx = idx;
            pending_stm_white = (pos.side_to_move() == WHITE);
        }
        flush_game();                                       // EOF: last game
        if (gz) pclose(f); else fclose(f);
    }
    fclose(fmv); fclose(fev);
}

int main(int argc, char** argv) {
    int quant = 8;
    int nthreads = (int)std::thread::hardware_concurrency();
    if (nthreads < 1) nthreads = 1;
    const char* out_path = nullptr;
    std::vector<std::string> inputs;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](){ return (i + 1 < argc) ? argv[++i] : ""; };
        if      (a == "--quant") quant = atoi(next());
        else if (a == "--threads") nthreads = std::max(1, atoi(next()));
        else if (a == "-o" || a == "--out") out_path = next();
        else if (a == "--filelist") { std::string fl = next(); std::ifstream in(fl); std::string line; while (std::getline(in, line)) { if (!line.empty()) inputs.push_back(line); } }
        else if (!a.empty() && a[0] == '-') { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
        else inputs.push_back(a);
    }
    if (inputs.empty()) { fprintf(stderr, "usage: luminex-encode [--quant N] [--threads N] [-o frame.raw] f.pgn[.gz]...\n"); return 1; }
    init_magic_bitboards();
    init_line_tables();   // CRITICAL: pin detection needs BetweenBB/LineBB populated.

    // Partition input files round-robin into nthreads shards.
    if (nthreads > (int)inputs.size()) nthreads = (int)inputs.size();
    std::vector<std::vector<std::string>> shards(nthreads);
    for (size_t i = 0; i < inputs.size(); ++i) shards[i % nthreads].push_back(inputs[i]);

    Shared sh;
    std::vector<PerThread> pts(nthreads);
    std::vector<std::thread> ths;
    for (int t = 0; t < nthreads; ++t) {
        pts[t].tid = t;
        ths.emplace_back(process_files, std::cref(shards[t]), std::ref(sh), std::ref(pts[t]), quant);
    }
    for (auto& th : ths) th.join();

    // ---- assemble hdr block: [u32 nfens][fens][game_entries (thread 0..N-1 in order)] ----
    uint32_t nfens = (uint32_t)sh.fen_list.size();
    std::vector<uint8_t> hdr_buf;
    hdr_buf.insert(hdr_buf.end(), (uint8_t*)&nfens, (uint8_t*)&nfens + 4);
    for (auto& fen : sh.fen_list) {
        uint16_t fl = (uint16_t)fen.size();
        hdr_buf.insert(hdr_buf.end(), (uint8_t*)&fl, (uint8_t*)&fl + 2);
        hdr_buf.insert(hdr_buf.end(), fen.begin(), fen.end());
    }
    for (int t = 0; t < nthreads; ++t)
        hdr_buf.insert(hdr_buf.end(), pts[t].game_entries.begin(), pts[t].game_entries.end());
    uint64_t sz_hdr = hdr_buf.size();

    // ---- total mv/ev sizes (sum of per-thread temp files) ----
    auto fsize = [](const std::string& p)->uint64_t {
        FILE* f = fopen(p.c_str(), "rb"); if (!f) return 0;
        fseek(f, 0, SEEK_END); uint64_t s = (uint64_t)ftell(f); fclose(f); return s;
    };
    uint64_t sz_mv = 0, sz_ev = 0;
    for (int t = 0; t < nthreads; ++t) { sz_mv += fsize(pts[t].mv_path); sz_ev += fsize(pts[t].ev_path); }

    // ---- emit frame: [u64 sz_hdr][hdr][u64 sz_mv][mv (thread 0..N-1)][u64 sz_ev][ev] ----
    std::string real_out = out_path ? std::string(out_path) : "";
    std::string tmp_out = real_out + ".tmp";
    FILE* out = real_out.empty() ? stdout : fopen(tmp_out.c_str(), "wb");
    if (!out) { fprintf(stderr, "cannot open output\n"); return 1; }
    fwrite(&sz_hdr, 8, 1, out); fwrite(hdr_buf.data(), 1, hdr_buf.size(), out);
    fwrite(&sz_mv, 8, 1, out);
    { uint8_t buf[1 << 20]; for (int t = 0; t < nthreads; ++t) { FILE* f = fopen(pts[t].mv_path.c_str(), "rb"); if (!f) continue; size_t r; while ((r = fread(buf, 1, sizeof(buf), f)) > 0) fwrite(buf, 1, r, out); fclose(f); } }
    fwrite(&sz_ev, 8, 1, out);
    { uint8_t buf[1 << 20]; for (int t = 0; t < nthreads; ++t) { FILE* f = fopen(pts[t].ev_path.c_str(), "rb"); if (!f) continue; size_t r; while ((r = fread(buf, 1, sizeof(buf), f)) > 0) fwrite(buf, 1, r, out); fclose(f); } }
    fflush(out);
    if (!real_out.empty()) { fclose(out); if (std::rename(tmp_out.c_str(), real_out.c_str()) != 0) { fprintf(stderr, "rename failed\n"); return 1; } }

    for (int t = 0; t < nthreads; ++t) { remove(pts[t].mv_path.c_str()); remove(pts[t].ev_path.c_str()); }

    uint64_t tg = 0, tp = 0, tb = 0;
    for (auto& p : pts) { tg += p.games; tp += p.pos; tb += p.bad_san; }
    fprintf(stderr, "encode: games=%llu pos=%llu unique_fens=%u bad_san=%llu threads=%d  frame=%llu bytes (hdr=%llu mv=%llu ev=%llu)\n",
            (unsigned long long)tg, (unsigned long long)tp, nfens, (unsigned long long)tb, nthreads,
            (unsigned long long)(24 + sz_hdr + sz_mv + sz_ev),
            (unsigned long long)sz_hdr, (unsigned long long)sz_mv, (unsigned long long)sz_ev);
    return 0;
}
