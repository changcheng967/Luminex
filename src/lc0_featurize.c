/* lc0_featurize.c — FAST V6 chunk → 136-byte HalfKA records converter.
   Reads .gz V6 chunk files (args), decodes board from planes[0..11], computes
   HalfKAv2 features directly (same as featurize.cpp FAST_HALFKA), packs records,
   writes to stdout. ~5-10M positions/s (C, no Python overhead).
   Usage: lc0_featurize chunk1.gz chunk2.gz ... > records.bin
   Then: xz -dc records.bin | trainer  (or pipe directly) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <zlib.h>
#include <stdint.h>

#define NUM_SQ 64
#define NUM_PLANES 768
#define NUM_INPUTS 24576
#define MAX_PIECES 32
#define V6_SIZE 8356

static const int KingBuckets[64] = {
    -1,-1,-1,-1,31,30,29,28,-1,-1,-1,-1,27,26,25,24,
    -1,-1,-1,-1,23,22,21,20,-1,-1,-1,-1,19,18,17,16,
    -1,-1,-1,-1,15,14,13,12,-1,-1,-1,-1,11,10, 9, 8,
    -1,-1,-1,-1, 7, 6, 5, 4,-1,-1,-1,-1, 3, 2, 1, 0
};

int main(int argc, char** argv) {
    /* batch buffer for fast writes */
    enum { BUFSZ = 136 * 8192 };
    char* buf = malloc(BUFSZ);
    int buf_n = 0;
    long total = 0;

    for (int fi = 1; fi < argc; fi++) {
        gzFile gz = gzopen(argv[fi], "rb");
        if (!gz) { fprintf(stderr, "skip %s\n", argv[fi]); continue; }
        unsigned char v6[V6_SIZE];
        while (gzread(gz, v6, V6_SIZE) == V6_SIZE) {
            const uint64_t* planes = (const uint64_t*)(v6 + 7440);
            int stm_white = (v6[8278] & 0x80) ? 1 : 0;
            float best_q; memcpy(&best_q, v6 + 8284, 4);

            /* kings: planes[5]=our K, planes[11]=their K */
            int our_k = __builtin_ffsll(planes[5]) - 1;
            int their_k = __builtin_ffsll(planes[11]) - 1;
            int wk = stm_white ? our_k : their_k;
            int bk = stm_white ? their_k : our_k;

            int xor_w  = ((wk & 7) < 4) ? 7 : 0;
            int xor_b  = (((bk & 7) < 4) ? 7 : 0) ^ 56;
            int bucket_w = KingBuckets[xor_w ^ wk] * NUM_PLANES;
            int bucket_b = KingBuckets[xor_b ^ bk] * NUM_PLANES;

            int16_t wfeat[MAX_PIECES], bfeat[MAX_PIECES];
            int nw = 0, nb = 0;
            for (int pt = 0; pt < 6; pt++) {
                /* our pieces (color = stm) */
                uint64_t bb = planes[pt];
                int ow = stm_white;  /* is our piece white? */
                while (bb && nw < MAX_PIECES) {
                    int sq = __builtin_ffsll(bb) - 1; bb &= bb - 1;
                    wfeat[nw++] = (int16_t)((xor_w ^ sq) + (pt*2 + (ow?0:1)) * NUM_SQ + bucket_w);
                    bfeat[nb++] = (int16_t)((xor_b ^ sq) + (pt*2 + (ow?1:0)) * NUM_SQ + bucket_b);
                }
                /* their pieces (color = !stm) */
                bb = planes[pt + 6];
                int tw = !stm_white;
                while (bb && nw < MAX_PIECES) {
                    int sq = __builtin_ffsll(bb) - 1; bb &= bb - 1;
                    wfeat[nw++] = (int16_t)((xor_w ^ sq) + (pt*2 + (tw?0:1)) * NUM_SQ + bucket_w);
                    bfeat[nb++] = (int16_t)((xor_b ^ sq) + (pt*2 + (tw?1:0)) * NUM_SQ + bucket_b);
                }
            }
            for (int i = nw; i < MAX_PIECES; i++) wfeat[i] = NUM_INPUTS;
            for (int i = nb; i < MAX_PIECES; i++) bfeat[i] = NUM_INPUTS;

            float q = best_q;
            if (q > 0.999f) q = 0.999f; if (q < -0.999f) q = -0.999f;
            float target = 800.0f * atanhf(q);
            float stm = stm_white ? 1.0f : 0.0f;

            char* r = buf + buf_n;
            memcpy(r, wfeat, 64); memcpy(r+64, bfeat, 64);
            memcpy(r+128, &stm, 4); memcpy(r+132, &target, 4);
            buf_n += 136; total++;
            if (buf_n >= BUFSZ - 136) { fwrite(buf, 1, buf_n, stdout); buf_n = 0; }
        }
        gzclose(gz);
    }
    if (buf_n) fwrite(buf, 1, buf_n, stdout);
    free(buf);
    fprintf(stderr, "lc0_featurize: %ld positions\n", total);
    return 0;
}
