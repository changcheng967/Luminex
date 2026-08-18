/* lc0_to_gamepack_idx.c — V6 → gamepack with INDEX-mode moves (1 byte/move).
   Includes minimal C movegen for legal move generation + sorting.
   Saves 1 byte/position vs RAW_MOVES mode: ~0.9 B/pos xz vs 1.44. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <zlib.h>
#include <stdint.h>

#define V6_SIZE 8356
#define MAX_GAMES 500000
#define MAX_MOVES 240   /* max legal moves in any chess position */

/* piece: 0=empty, 1=WP 2=WN 3=WB 4=WR 5=WQ 6=WK 7=BP 8=BN 9=BB 10=BR 11=BQ 12=BK */
#define IS_WHITE(p) ((p) >= 1 && (p) <= 6)
#define IS_BLACK(p) ((p) >= 7 && (p) <= 12)
#define IS_EMPTY(p) ((p) == 0)
#define PIECE_TYPE(p) ((p) <= 6 ? (p) : (p) - 6)  /* 1=P..6=K */
#define COLOR(p) ((p) <= 6 ? 0 : 1)  /* 0=white, 1=black */

/* knight offsets */
static const int knight_dr[] = {-2,-2,-1,-1,1,1,2,2};
static const int knight_df[] = {-1,1,-2,2,-2,2,-1,1};
/* king offsets */
static const int king_dr[] = {-1,-1,-1,0,0,1,1,1};
static const int king_df[] = {-1,0,1,-1,1,-1,0,1};
/* bishop/rook/queen directions */
static const int bidiag_dr[] = {-1,-1,1,1};
static const int bidiag_df[] = {-1,1,-1,1};
static const int rook_dr[] = {-1,1,0,0};
static const int rook_df[] = {0,0,-1,1};

#define SQ(r,f) ((r)*8+(f))
#define RANK(s) ((s)/8)
#define FILE(s) ((s)%8)
#define INB(r,f) ((r)>=0&&(r)<8&&(f)>=0&&(f)<8)

/* Move flags (matching types.h) */
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

typedef struct { int board[64]; int wk_sq, bk_sq; int stm_white; int ep_sq; int castle; } Pos;

/* is square (r,f) attacked by side `by_white`? */
static int is_attacked(const Pos* p, int sq, int by_white) {
    int r = RANK(sq), f = FILE(sq);
    /* pawn attacks */
    if (by_white) {
        /* white pawns attack from r-1 */
        int pr = r - 1;
        if (pr >= 0) {
            if (f > 0 && p->board[SQ(pr,f-1)] == 1) return 1;
            if (f < 7 && p->board[SQ(pr,f+1)] == 1) return 1;
        }
    } else {
        int pr = r + 1;
        if (pr < 8) {
            if (f > 0 && p->board[SQ(pr,f-1)] == 7) return 1;
            if (f < 7 && p->board[SQ(pr,f+1)] == 7) return 1;
        }
    }
    /* knight */
    int nt = by_white ? 2 : 8;
    for (int i = 0; i < 8; i++) {
        int nr = r + knight_dr[i], nf = f + knight_df[i];
        if (INB(nr,nf) && p->board[SQ(nr,nf)] == nt) return 1;
    }
    /* king */
    int kt = by_white ? 6 : 12;
    for (int i = 0; i < 8; i++) {
        int nr = r + king_dr[i], nf = f + king_df[i];
        if (INB(nr,nf) && p->board[SQ(nr,nf)] == kt) return 1;
    }
    /* bishop/queen (diagonal) */
    int bt = by_white ? 3 : 9, qt = by_white ? 5 : 11;
    for (int d = 0; d < 4; d++) {
        int nr = r + bidiag_dr[d], nf = f + bidiag_df[d];
        while (INB(nr,nf)) {
            int pc = p->board[SQ(nr,nf)];
            if (pc) { if (pc == bt || pc == qt) return 1; break; }
            nr += bidiag_dr[d]; nf += bidiag_df[d];
        }
    }
    /* rook/queen (orthogonal) */
    int rt = by_white ? 4 : 10;
    for (int d = 0; d < 4; d++) {
        int nr = r + rook_dr[d], nf = f + rook_df[d];
        while (INB(nr,nf)) {
            int pc = p->board[SQ(nr,nf)];
            if (pc) { if (pc == rt || pc == qt) return 1; break; }
            nr += rook_dr[d]; nf += rook_df[d];
        }
    }
    return 0;
}

/* generate pseudo-legal moves, return count. moves[] = (from, to, promo_type) */
typedef struct { int from, to, promo; } SMove;
static int gen_pseudo(const Pos* p, SMove* moves) {
    int n = 0;
    int us_white = p->stm_white;
    int them_white = !us_white;
    int our_min = us_white ? 1 : 7, our_max = us_white ? 6 : 12;

    for (int sq = 0; sq < 64; sq++) {
        int pc = p->board[sq];
        if (pc < our_min || pc > our_max) continue;
        int pt = PIECE_TYPE(pc);
        int r = RANK(sq), f = FILE(sq);

        if (pt == 1) { /* pawn */
            int dir = us_white ? 1 : -1;
            int nr = r + dir;
            /* push */
            if (INB(nr,f) && p->board[SQ(nr,f)] == 0) {
                if (nr == 0 || nr == 7) { /* promotion */
                    for (int pr = 2; pr <= 5; pr++) { moves[n++] = (SMove){sq, SQ(nr,f), pr}; }
                } else {
                    moves[n++] = (SMove){sq, SQ(nr,f), 0};
                    /* double push */
                    int nr2 = r + 2*dir;
                    if ((us_white ? r == 1 : r == 6) && p->board[SQ(nr2,f)] == 0)
                        moves[n++] = (SMove){sq, SQ(nr2,f), 0};
                }
            }
            /* captures */
            for (int df = -1; df <= 1; df += 2) {
                int nf = f + df;
                if (!INB(nr,nf)) continue;
                int tq = SQ(nr,nf);
                int tp = p->board[tq];
                if (tp && COLOR(tp) == them_white) {
                    if (nr == 0 || nr == 7) {
                        for (int pr = 2; pr <= 5; pr++) { moves[n++] = (SMove){sq, tq, pr}; }
                    } else {
                        moves[n++] = (SMove){sq, tq, 0};
                    }
                }
                /* en passant */
                if (tq == p->ep_sq && tp == 0) {
                    moves[n++] = (SMove){sq, tq, 0};
                }
            }
        } else if (pt == 2) { /* knight */
            for (int i = 0; i < 8; i++) {
                int nr = r + knight_dr[i], nf = f + knight_df[i];
                if (!INB(nr,nf)) continue;
                int tq = SQ(nr,nf);
                if (IS_EMPTY(p->board[tq]) || COLOR(p->board[tq]) == them_white)
                    moves[n++] = (SMove){sq, tq, 0};
            }
        } else if (pt == 6) { /* king */
            for (int i = 0; i < 8; i++) {
                int nr = r + king_dr[i], nf = f + king_df[i];
                if (!INB(nr,nf)) continue;
                int tq = SQ(nr,nf);
                if (IS_EMPTY(p->board[tq]) || COLOR(p->board[tq]) == them_white)
                    moves[n++] = (SMove){sq, tq, 0};
            }
            /* castling */
            if (us_white && sq == SQ(0,4) && !is_attacked(p, SQ(0,4), 0)) {
                if ((p->castle & 1) && p->board[SQ(0,5)]==0 && p->board[SQ(0,6)]==0
                    && p->board[SQ(0,7)]==4 && !is_attacked(p,SQ(0,5),0) && !is_attacked(p,SQ(0,6),0))
                    moves[n++] = (SMove){sq, SQ(0,6), 0};
                if ((p->castle & 2) && p->board[SQ(0,3)]==0 && p->board[SQ(0,2)]==0 && p->board[SQ(0,1)]==0
                    && p->board[SQ(0,0)]==4 && !is_attacked(p,SQ(0,3),0) && !is_attacked(p,SQ(0,2),0))
                    moves[n++] = (SMove){sq, SQ(0,2), 0};
            }
            if (!us_white && sq == SQ(7,4) && !is_attacked(p, SQ(7,4), 1)) {
                if ((p->castle & 4) && p->board[SQ(7,5)]==0 && p->board[SQ(7,6)]==0
                    && p->board[SQ(7,7)]==10 && !is_attacked(p,SQ(7,5),1) && !is_attacked(p,SQ(7,6),1))
                    moves[n++] = (SMove){sq, SQ(7,6), 0};
                if ((p->castle & 8) && p->board[SQ(7,3)]==0 && p->board[SQ(7,2)]==0 && p->board[SQ(7,1)]==0
                    && p->board[SQ(7,0)]==10 && !is_attacked(p,SQ(7,3),1) && !is_attacked(p,SQ(7,2),1))
                    moves[n++] = (SMove){sq, SQ(7,2), 0};
            }
        } else { /* bishop(3), rook(4), queen(5) */
            const int *dr, *df; int ndir;
            if (pt == 3) { dr = bidiag_dr; df = bidiag_df; ndir = 4; }
            else if (pt == 4) { dr = rook_dr; df = rook_df; ndir = 4; }
            else { /* queen: both */
                for (int d = 0; d < 4; d++) {
                    int nr = r+bidiag_dr[d], nf = f+bidiag_df[d];
                    while (INB(nr,nf)) {
                        int tq = SQ(nr,nf), tp = p->board[tq];
                        if (tp == 0) { moves[n++] = (SMove){sq,tq,0}; }
                        else { if (COLOR(tp)==them_white) moves[n++] = (SMove){sq,tq,0}; break; }
                        nr += bidiag_dr[d]; nf += bidiag_df[d];
                    }
                }
                dr = rook_dr; df = rook_df; ndir = 4;
            }
            for (int d = 0; d < ndir; d++) {
                int nr = r+dr[d], nf = f+df[d];
                while (INB(nr,nf)) {
                    int tq = SQ(nr,nf), tp = p->board[tq];
                    if (tp == 0) { moves[n++] = (SMove){sq,tq,0}; }
                    else { if (COLOR(tp)==them_white) moves[n++] = (SMove){sq,tq,0}; break; }
                    nr += dr[d]; nf += df[d];
                }
            }
        }
    }
    return n;
}

/* make/undo for legality check */
static int make_move_check(Pos* p, SMove m) {
    int from = m.from, to = m.to;
    int moved = p->board[from];
    int captured = p->board[to];
    /* ep capture */
    int ep_cap_sq = -1;
    if (PIECE_TYPE(moved) == 1 && to == p->ep_sq && captured == 0) {
        ep_cap_sq = p->stm_white ? to - 8 : to + 8;
        captured = p->board[ep_cap_sq];
        p->board[ep_cap_sq] = 0;
    }
    p->board[to] = (m.promo ? (p->stm_white ? m.promo : m.promo + 6) : moved);
    p->board[from] = 0;
    /* castling: move rook too */
    if (PIECE_TYPE(moved) == 6 && abs(FILE(from) - FILE(to)) == 2) {
        if (FILE(to) == 6) { int rs=SQ(RANK(to),7), rd=SQ(RANK(to),5); p->board[rd]=p->board[rs]; p->board[rs]=0; }
        else { int rs=SQ(RANK(to),0), rd=SQ(RANK(to),3); p->board[rd]=p->board[rs]; p->board[rs]=0; }
    }
    int king_sq = p->stm_white ? p->wk_sq : p->bk_sq;
    if (PIECE_TYPE(moved) == 6) king_sq = to;
    int in_check = is_attacked(p, king_sq, !p->stm_white);
    /* undo */
    p->board[from] = moved; p->board[to] = (ep_cap_sq >= 0) ? 0 : captured;
    if (m.promo) p->board[from] = moved; /* restore pawn on undo */
    if (ep_cap_sq >= 0) p->board[ep_cap_sq] = captured;
    if (PIECE_TYPE(moved) == 6 && abs(FILE(from) - FILE(to)) == 2) {
        if (FILE(to) == 6) { int rs=SQ(RANK(to),7), rd=SQ(RANK(to),5); p->board[rs]=p->board[rd]; p->board[rd]=0; }
        else { int rs=SQ(RANK(to),0), rd=SQ(RANK(to),3); p->board[rs]=p->board[rd]; p->board[rd]=0; }
    }
    return !in_check; /* return 1 if legal (not in check) */
}

/* generate legal moves, sort by (from<<12)|(to<<4)|promo, return count */
static int gen_legal_sorted(const Pos* p, SMove* sorted) {
    SMove pseudo[MAX_MOVES];
    int np = gen_pseudo(p, pseudo);
    int nlegal = 0;
    for (int i = 0; i < np; i++) {
        if (make_move_check((Pos*)p, pseudo[i]))
            sorted[nlegal++] = pseudo[i];
    }
    /* sort by (from<<12)|(to<<4)|promo — insertion sort (nlegal < 80 typically) */
    for (int i = 1; i < nlegal; i++) {
        SMove tmp = sorted[i]; int j = i - 1;
        uint32_t ki = ((uint32_t)tmp.from << 12) | ((uint32_t)tmp.to << 4) | tmp.promo;
        while (j >= 0) {
            uint32_t kj = ((uint32_t)sorted[j].from << 12) | ((uint32_t)sorted[j].to << 4) | sorted[j].promo;
            if (kj > ki) { sorted[j+1] = sorted[j]; j--; } else break;
        }
        sorted[j+1] = tmp;
    }
    return nlegal;
}

/* find index of (from,to,promo) in sorted list, or -1 */
static int find_index(const SMove* sorted, int n, int from, int to, int promo) {
    for (int i = 0; i < n; i++)
        if (sorted[i].from == from && sorted[i].to == to && sorted[i].promo == promo)
            return i;
    return -1;
}

/* ===== V6 decode + board diff (from lc0_to_gamepack.c) ===== */
typedef struct { int piece[64]; int stm_white; } Board;
/* undo Lc0's data-augmentation transforms to get canonical board orientation */
static void undo_transform(int board[64], int transform) {
    if (transform & 0x01) { int t[64]; for(int s=0;s<64;s++) t[s^56]=board[s]; memcpy(board,t,256); } /* flip (vertical) */
    if (transform & 0x02) { int t[64]; for(int s=0;s<64;s++) t[s^7]=board[s]; memcpy(board,t,256); }  /* mirror (horizontal) */
    if (transform & 0x04) { int t[64]; for(int s=0;s<64;s++) t[(s%8)*8+(s/8)]=board[s]; memcpy(board,t,256); } /* transpose */
}
static void decode_board(const unsigned char* v6, Board* b) {
    const uint64_t* planes = (const uint64_t*)(v6 + 7440);
    int infmt = *(const uint32_t*)(v6 + 4);
    if (infmt == 3) b->stm_white = (v6[8278] >> 7) & 1;
    else b->stm_white = v6[8276] & 1;  /* input_format=1: stm at offset 8276 */
    memset(b->piece, 0, 64*sizeof(int));
    /* FIXED orientation: planes[0..5] always same color, planes[6..11] always other */
    static const int bc[12] = {1,2,3,4,5,6, 7,8,9,10,11,12};
    for (int p = 0; p < 12; p++) {
        uint64_t bb = planes[p];
        while (bb) { int sq = __builtin_ffsll(bb)-1; bb &= bb-1; b->piece[sq] = bc[p]; }
    }
}
static int diff_move(const Board* prev, const Board* cur, int* from, int* to, int* promo) {
    int gone[64], ng=0, app[64], na=0, chg[64], nc=0;
    for (int sq=0;sq<64;sq++){int o=prev->piece[sq],n=cur->piece[sq];if(!o&&n)app[na++]=sq;else if(o&&!n)gone[ng++]=sq;else if(o&&o!=n)chg[nc++]=sq;}
    *promo = 0;
    if (ng==2 && na==2) { /* castling */
        for(int i=0;i<2;i++)for(int j=0;j<2;j++){int kf=gone[i],kt=app[j];if((prev->piece[kf]==6||prev->piece[kf]==12)&&abs((kf&7)-(kt&7))==2&&(kf/8)==(kt/8)){*from=kf;*to=kt;return 1;}}}
    if (ng==1 && nc==1) { /* capture or promo-capture */
        *from=gone[0]; *to=chg[0]; int op=prev->piece[*from];
        if((op==1||op==7)&&cur->piece[*to]!=1&&cur->piece[*to]!=7)*promo=PIECE_TYPE(cur->piece[*to]);return 1;}
    if (ng==1 && na==1) { /* normal/ep/promo/double */
        *from=gone[0]; *to=app[0]; int op=prev->piece[*from];
        if((op==1||op==7)&&cur->piece[*to]!=1&&cur->piece[*to]!=7)*promo=PIECE_TYPE(cur->piece[*to]);
        return 1;}
    return 0;
}
static float best_q_to_cp(const unsigned char* v6) {
    float q; memcpy(&q, v6+8284, 4);
    if(q>0.999f)q=0.999f; if(q<-0.999f)q=-0.999f; return 800.0f*atanhf(q);
}
static int board_to_fen(const Board* b, char* out) {
    const char* s=".PNBRQKpnbrqk"; char* p=out;
    for(int r=7;r>=0;r--){int e=0;for(int f=0;f<8;f++){int pc=b->piece[r*8+f];if(!pc)e++;else{if(e)*p++='0'+e;e=0;*p++=s[pc];}}if(e)*p++='0'+e;if(r>0)*p++='/';}
    *p++=' ';*p++=b->stm_white?'w':'b';*p++=' ';*p++='-';*p++=' ';*p++='-';*p++=' ';*p++='0';*p++=' ';*p++='1';*p=0;return p-out;
}

int main(int argc, char** argv) {
    char* fens=malloc(MAX_GAMES*100); int fen_lens[MAX_GAMES]; int nfens=0;
    uint16_t* g_npos=malloc(MAX_GAMES*2); uint8_t* g_stm=malloc(MAX_GAMES);
    uint32_t* g_fenidx=malloc(MAX_GAMES*4); int16_t* g_eval=malloc(MAX_GAMES*2);
    uint8_t* moves=malloc(500000000); /* 1 byte per move (INDEX mode) */
    unsigned char* evals=malloc(500000000);
    long n_moves=0, n_evals=0; int n_games=0; long total_pos=0;
    int fallback_count=0; /* games where movegen failed → 2-byte fallback */

    char** gz_files; int n_gz_files;
    if (argc > 1 && strcmp(argv[1], "--stdin") == 0) {
        gz_files=malloc(sizeof(char*)*2000000); char fb[4096]; n_gz_files=0;
        while(fgets(fb,sizeof(fb),stdin)){fb[strcspn(fb,"\n")]=0;if(fb[0])gz_files[n_gz_files++]=strdup(fb);}
    } else { gz_files=argv+1; n_gz_files=argc-1; }

    for (int fi=0; fi<n_gz_files; fi++) {
        gzFile gz=gzopen(gz_files[fi],"rb"); if(!gz) continue;
        unsigned char v6[V6_SIZE]; Board prev,cur; int have_prev=0;
        int game_start_eval=0, game_npos=0, first_eval_set=0;
        Pos mpos; /* for movegen */

        while (gzread(gz,v6,V6_SIZE)==V6_SIZE) {
            decode_board(v6,&cur); float cp=best_q_to_cp(v6);
            if (!have_prev) {
                char fen[100]; int flen=board_to_fen(&cur,fen);
                memcpy(fens+nfens*100,fen,flen); fen_lens[nfens]=flen;
                g_fenidx[n_games]=nfens; g_stm[n_games]=cur.stm_white;
                nfens++; n_games++; prev=cur; have_prev=1;
                /* set up movegen pos */
                memcpy(mpos.board, cur.piece, 64*sizeof(int));
                mpos.wk_sq=0; mpos.bk_sq=0;
                for(int s=0;s<64;s++){if(cur.piece[s]==6)mpos.wk_sq=s;if(cur.piece[s]==12)mpos.bk_sq=s;}
                mpos.stm_white=cur.stm_white; mpos.ep_sq=-1; mpos.castle=0;
                /* extract castling from V6 */
                mpos.castle = ((v6[8272]&1)?1:0)|((v6[8273]&1)?2:0)|((v6[8274]&1)?4:0)|((v6[8275]&1)?8:0);
                continue;
            }
            int from, to, promo;
            if (!diff_move(&prev,&cur,&from,&to,&promo)) { prev=cur; continue; }

            /* generate legal moves at prev position, find index */
            SMove legal[MAX_MOVES];
            int nlegal = gen_legal_sorted(&mpos, legal);
            int idx = find_index(legal, nlegal, from, to, promo);

            if (idx >= 0 && idx < 256) {
                moves[n_moves++] = (uint8_t)idx;
            } else {
                /* movegen failed: store raw 2-byte as fallback (encode index=255 + raw) */
                moves[n_moves++] = 255;
                uint16_t raw = (uint16_t)((from<<6)|to); /* simplified */
                moves[n_moves++] = (uint8_t)(raw & 0xFF);
                moves[n_moves++] = (uint8_t)(raw >> 8);
                fallback_count++;
            }

            game_npos++;
            int icp=(int)roundf(cp);
            if (!first_eval_set) {
                g_eval[n_games-1]=(int16_t)(icp>32000?32000:icp<-32000?-32000:icp);
                first_eval_set=1;
            } else {
                int delta=icp-game_start_eval;
                if(delta>=-127&&delta<=127) evals[n_evals++]=(unsigned char)(int8_t)delta;
                else { evals[n_evals++]=0x80; int16_t av=(int16_t)(icp>32000?32000:icp<-32000?-32000:icp); memcpy(evals+n_evals,&av,2); n_evals+=2; }
            }
            game_start_eval=icp; prev=cur; total_pos++;

            /* update movegen pos for next iteration */
            memcpy(mpos.board, cur.piece, 64*sizeof(int));
            for(int s=0;s<64;s++){if(cur.piece[s]==6)mpos.wk_sq=s;if(cur.piece[s]==12)mpos.bk_sq=s;}
            mpos.stm_white=cur.stm_white;
        }
        g_npos[n_games-1]=game_npos; gzclose(gz);
    }

    /* output gamepack frame */
    fwrite(&nfens,4,1,stdout);
    for(int i=0;i<nfens;i++){fwrite(&fen_lens[i],2,1,stdout);fwrite(fens+i*100,1,fen_lens[i],stdout);}
    for(int i=0;i<n_games;i++){fwrite(&g_npos[i],2,1,stdout);fwrite(&g_stm[i],1,1,stdout);fwrite(&g_fenidx[i],4,1,stdout);fwrite(&g_eval[i],2,1,stdout);}
    fwrite(moves,1,n_moves,stdout);
    fwrite(evals,1,n_evals,stdout);
    fprintf(stderr,"lc0_to_gamepack_idx: %d games, %ld positions, %d fallbacks, frame=%ld bytes\n",
            n_games, total_pos, fallback_count,
            (long)(4+nfens*102+n_games*9+n_moves+n_evals));
    return 0;
}
