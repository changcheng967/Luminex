/* lc0_to_gamepack.c — V6 chunks → gamepack frame (game-based, ~3 B/pos raw).
   Each .gz = one game's sequential V6 positions. Diff consecutive boards → raw Move.
   Output: gamepack frame (header+FENs + game entries + RAW_MOVES + delta-evals).
   xz compresses to ~1.5 B/pos → ~12B positions in 18GB. 9x more than 136-byte records.
   The existing luminex-featurize (RAW_MOVES mode) replays this format. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <zlib.h>
#include <stdint.h>

#define V6_SIZE 8356
#define MAX_GAMES 500000
#define MAX_NAME 8192

/* Move flags (from types.h) */
#define MF_QUIET 0x0000
#define MF_DOUBLE_PAWN 0x1000
#define MF_K_CASTLE 0x2000
#define MF_Q_CASTLE 0x3000
#define MF_CAPTURE 0x4000
#define MF_EP 0x5000
#define MF_PROMO_N 0x8000
#define MF_PROMO_B 0x9000
#define MF_PROMO_R 0xA000
#define MF_PROMO_Q 0xB000
#define MF_CAP_PROMO_N 0xC000
#define MF_CAP_PROMO_B 0xD000
#define MF_CAP_PROMO_R 0xE000
#define MF_CAP_PROMO_Q 0xF000

/* piece: 0=empty, 1=WP 2=WN 3=WB 4=WR 5=WQ 6=WK 7=BP 8=BN 9=BB 10=BR 11=BQ 12=BK */
typedef struct { int piece[64]; int stm_white; } Board;

static inline int make_move_raw(int from, int to, uint16_t flags) {
    return (int)(flags | (from << 6) | to);
}

/* decode V6 planes[0..11] → Board */
static void decode_board(const unsigned char* v6, Board* b) {
    const uint64_t* planes = (const uint64_t*)(v6 + 7440);
    int invar = v6[8278];
    int stm_white = (invar & 0x80) ? 1 : 0;
    b->stm_white = stm_white;
    memset(b->piece, 0, sizeof(b->piece));
    /* planes[0..5] = our P,N,B,R,Q,K; planes[6..11] = their P,N,B,R,Q,K */
    static const int base_color[2][12] = {
        {1,2,3,4,5,6, 7,8,9,10,11,12},  /* stm=white: planes[0..5]=white, [6..11]=black */
        {7,8,9,10,11,12, 1,2,3,4,5,6},   /* stm=black: planes[0..5]=black, [6..11]=white */
    };
    for (int p = 0; p < 12; p++) {
        uint64_t bb = planes[p];
        int pc = base_color[stm_white][p];
        while (bb) {
            int sq = __builtin_ffsll(bb) - 1; bb &= bb - 1;
            b->piece[sq] = pc;
        }
    }
}

/* diff two boards → raw Move. Returns 0 if no move found (error). */
static int diff_move(const Board* prev, const Board* cur) {
    int gone[64], ng = 0;      /* squares where piece disappeared */
    int appeared[64], na = 0;  /* squares where piece appeared */
    int changed_from[64], changed_to[64], nc = 0; /* changed: old piece → new piece */
    for (int sq = 0; sq < 64; sq++) {
        int op = prev->piece[sq], np = cur->piece[sq];
        if (op == 0 && np != 0) appeared[na++] = sq;
        else if (op != 0 && np == 0) gone[ng++] = sq;
        else if (op != 0 && op != np) { changed_from[nc] = op; changed_to[nc] = np; changed_from[nc] |= (sq << 8); nc++; }
    }
    /* castling: 2 gone + 2 appeared, king moved 2 files */
    if (ng == 2 && na == 2) {
        /* find king move (2 files) */
        for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
            int kf = gone[i], kt = appeared[j];
            int pcf = prev->piece[kf];
            if ((pcf == 6 || pcf == 12) && abs((kf&7) - (kt&7)) == 2 && (kf/8) == (kt/8)) {
                int flags = ((kt & 7) == 6) ? MF_K_CASTLE : MF_Q_CASTLE;
                return make_move_raw(kf, kt, flags);
            }
        }
    }
    /* promotion + capture: 1 gone (pawn) + 1 changed (captured → promo piece) */
    if (ng == 1 && nc == 1) {
        int from = gone[0], to = changed_from[0] >> 8;
        int old_pawn = (prev->piece[from] == 1 || prev->piece[from] == 7);
        int new_piece = changed_to[0];
        if (old_pawn) {
            int promo_base = (new_piece <= 6) ? 0 : 4; /* white: 0, black: +4 → capture_promo */
            int pt = (new_piece <= 6) ? new_piece : new_piece - 6; /* 2=N,3=B,4=R,5=Q */
            uint16_t flags = MF_CAP_PROMO_N + ((pt - 2) << 12) + (promo_base << 2);
            /* simpler: */
            static const uint16_t cap_promo[4] = {MF_CAP_PROMO_N, MF_CAP_PROMO_B, MF_CAP_PROMO_R, MF_CAP_PROMO_Q};
            int pidx = (pt == 2) ? 0 : (pt == 3) ? 1 : (pt == 4) ? 2 : 3;
            return make_move_raw(from, to, cap_promo[pidx]);
        }
        /* normal capture: mover from 'gone', captures at 'changed' sq */
        return make_move_raw(from, to, MF_CAPTURE);
    }
    /* promotion (no capture): 1 gone (pawn) + 1 appeared (non-pawn) */
    if (ng == 1 && na == 1) {
        int from = gone[0], to = appeared[0];
        int old_pawn = (prev->piece[from] == 1 || prev->piece[from] == 7);
        int new_piece = cur->piece[to];
        if (old_pawn && new_piece != 1 && new_piece != 7) {
            int pt = (new_piece <= 6) ? new_piece : new_piece - 6;
            static const uint16_t promo[4] = {MF_PROMO_N, MF_PROMO_B, MF_PROMO_R, MF_PROMO_Q};
            int pidx = (pt == 2) ? 0 : (pt == 3) ? 1 : (pt == 4) ? 2 : 3;
            return make_move_raw(from, to, promo[pidx]);
        }
        /* en passant: pawn moved diagonally, enemy pawn disappeared elsewhere */
        if (old_pawn && abs((from&7) - (to&7)) == 1) {
            /* check: an enemy pawn disappeared from a non-to square */
            /* EP: gone has the from sq, but the captured pawn is at to-8 or to+8 */
            return make_move_raw(from, to, MF_EP);
        }
        /* double pawn push */
        if (old_pawn && abs((from/8) - (to/8)) == 2) return make_move_raw(from, to, MF_DOUBLE_PAWN);
        /* normal quiet move */
        return make_move_raw(from, to, MF_QUIET);
    }
    /* capture: 1 gone (mover) + 1 changed (target square: had enemy piece, now has mover) */
    if (ng == 1 && nc == 1) {
        int from = gone[0], to = changed_from[0] >> 8;
        return make_move_raw(from, to, MF_CAPTURE);
    }
    /* fallback: 1 gone + 1 appeared → normal/capture */
    if (ng >= 1 && na >= 1) {
        int from = gone[0], to = appeared[0];
        int capture = (prev->piece[to] != 0) ? 1 : 0;
        return make_move_raw(from, to, capture ? MF_CAPTURE : MF_QUIET);
    }
    return 0; /* error */
}

static float best_q_to_cp(const unsigned char* v6) {
    float q; memcpy(&q, v6 + 8284, 4);
    if (q > 0.999f) q = 0.999f; if (q < -0.999f) q = -0.999f;
    return 800.0f * atanhf(q);
}

/* write FEN from board (minimal: placement + stm + castling) */
static int board_to_fen(const Board* b, const unsigned char* v6, char* out) {
    const char* pc_str = ".PNBRQKpnbrqk";
    char* p = out;
    for (int rank = 7; rank >= 0; rank--) {
        int empty = 0;
        for (int file = 0; file < 8; file++) {
            int sq = rank * 8 + file;
            int pc = b->piece[sq];
            if (pc == 0) empty++;
            else { if (empty) *p++ = '0' + empty; empty = 0; *p++ = pc_str[pc]; }
        }
        if (empty) *p++ = '0' + empty;
        if (rank > 0) *p++ = '/';
    }
    *p++ = ' '; *p++ = b->stm_white ? 'w' : 'b';
    *p++ = ' '; *p++ = '-'; /* castling (simplified) */
    *p++ = ' '; *p++ = '-'; /* ep */
    *p++ = ' '; *p++ = '0'; *p++ = ' '; *p++ = '1'; /* rule50, move */
    *p = 0;
    return p - out;
}

int main(int argc, char** argv) {
    /* accumulate gamepack frame in memory */
    char* fens = malloc(MAX_GAMES * 100);  /* FEN strings */
    int fen_lens[MAX_GAMES]; int nfens = 0;
    /* game entries: n_pos(u16) stm(u8) fen_idx(u32) start_eval(s16) */
    uint16_t* g_npos = malloc(MAX_GAMES * 2);
    uint8_t* g_stm = malloc(MAX_GAMES);
    uint32_t* g_fenidx = malloc(MAX_GAMES * 4);
    int16_t* g_eval = malloc(MAX_GAMES * 2);
    /* moves + eval deltas */
    uint16_t* moves = malloc(500000000); /* 500M positions × 2 bytes = 1GB max */
    unsigned char* evals = malloc(500000000); /* deltas */
    long n_moves = 0, n_evals = 0;
    int n_games = 0;
    long total_pos = 0;

    /* collect filenames: from --stdin (unlimited) or argv */
    char** gz_files; int n_gz_files;
    if (argc > 1 && strcmp(argv[1], "--stdin") == 0) {
        gz_files = malloc(sizeof(char*) * 2000000);
        char fbuf[4096]; n_gz_files = 0;
        while (fgets(fbuf, sizeof(fbuf), stdin)) {
            fbuf[strcspn(fbuf, "\n")] = 0;
            if (fbuf[0]) gz_files[n_gz_files++] = strdup(fbuf);
        }
    } else {
        gz_files = argv + 1; n_gz_files = argc - 1;
    }
    for (int fi = 0; fi < n_gz_files; fi++) {
        gzFile gz = gzopen(gz_files[fi], "rb");
        if (!gz) continue;
        unsigned char v6[V6_SIZE];
        Board prev, cur;
        int have_prev = 0;
        int game_start_eval = 0;
        int game_moves_start = n_moves;
        int game_npos = 0;
        int first_eval_set = 0;

        /* read all positions in this .gz (one game) */
        while (gzread(gz, v6, V6_SIZE) == V6_SIZE) {
            decode_board(v6, &cur);
            float cp = best_q_to_cp(v6);
            if (!have_prev) {
                /* first position → FEN + start */
                char fen[100];
                int flen = board_to_fen(&cur, v6, fen);
                memcpy(fens + nfens * 100, fen, flen);
                fen_lens[nfens] = flen;
                g_fenidx[n_games] = nfens;
                g_stm[n_games] = cur.stm_white;
                nfens++; n_games++;
                prev = cur; have_prev = 1;
                continue;
            }
            /* diff prev→cur → move */
            int mv = diff_move(&prev, &cur);
            if (mv == 0) { prev = cur; continue; } /* skip broken */
            moves[n_moves++] = (uint16_t)mv;
            game_npos++;
            /* eval: first evaluated position → start_eval, rest → delta */
            int icp = (int)roundf(cp);
            if (!first_eval_set) {
                g_eval[n_games - 1] = (int16_t)(icp > 32000 ? 32000 : icp < -32000 ? -32000 : icp);
                first_eval_set = 1;
            } else {
                int delta = icp - game_start_eval;
                if (delta >= -127 && delta <= 127) {
                    evals[n_evals++] = (unsigned char)(int8_t)delta;
                } else {
                    evals[n_evals++] = 0x80;
                    int16_t absval = (int16_t)(icp > 32000 ? 32000 : icp < -32000 ? -32000 : icp);
                    memcpy(evals + n_evals, &absval, 2); n_evals += 2;
                }
            }
            game_start_eval = icp;
            prev = cur;
            total_pos++;
        }
        g_npos[n_games - 1] = game_npos;
        gzclose(gz);
    }

    /* output gamepack frame to stdout */
    /* header: u32 nfens; [u16 len + bytes] * nfens */
    fwrite(&nfens, 4, 1, stdout);
    for (int i = 0; i < nfens; i++) {
        fwrite(&fen_lens[i], 2, 1, stdout);
        fwrite(fens + i * 100, 1, fen_lens[i], stdout);
    }
    /* game entries: [u16 n_pos][u8 stm][u32 fen_idx][s16 start_eval] * n_games */
    for (int i = 0; i < n_games; i++) {
        fwrite(&g_npos[i], 2, 1, stdout);
        fwrite(&g_stm[i], 1, 1, stdout);
        fwrite(&g_fenidx[i], 4, 1, stdout);
        fwrite(&g_eval[i], 2, 1, stdout);
    }
    /* move block: u16 raw_move * total_pos (RAW_MOVES mode) */
    fwrite(moves, 2, n_moves, stdout);
    /* eval block: deltas */
    fwrite(evals, 1, n_evals, stdout);

    fprintf(stderr, "lc0_to_gamepack: %d games, %ld positions, frame=%ld bytes\n",
            n_games, total_pos,
            (long)(4 + nfens * 102 + n_games * 9 + n_moves * 2 + n_evals));
    return 0;
}
