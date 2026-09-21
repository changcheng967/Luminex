// verify_frames.c — integrity check for lc0pack gamepack frames, mirroring the
// FEATURIZER's parser exactly (the consumer is the spec):
//   frame: [u64 hdr_n][hdr][u64 mv_n][mv][u64 ev_n][ev]
//   hdr:   u32 nfens; nfens x (u16 len + fen bytes); then 9-byte game entries
//          [u16 n_pos][u8 stm][u32 fen_idx][i16 start_eval] while >=9 bytes remain
//   mv:    2 bytes/pos (RAW mode), per-game offsets from entries (mv_n may have
//          small orphan tail from anomalously-encoded games the consumer skips)
//   ev:    per game (n_pos-1) codes: int8 delta, or 0x80 + i16 abs
// Fails on: xz corruption, size-closure, fen/entry field violations, move-flag
// nibble 6/7 (unused in our MoveFlag enum), eval-range violations >0.1%.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct {
    uint64_t games, positions, evals;
    uint64_t h[5];        // <=100, <=300, <=1000, >1000, zero
    uint64_t range_bad, mv_left, ev_left, slack_bytes;
} Stats;

static int verify_one(const char* path, Stats* st, char* err, size_t errsz) {
    char cmd[4096];
    snprintf(cmd, sizeof cmd, "xz -dc '%s'", path);
    FILE* p = popen(cmd, "r");
    if (!p) { snprintf(err, errsz, "popen failed"); return 1; }
    size_t cap = 1 << 22, len = 0;
    unsigned char* buf = malloc(cap);
    if (!buf) { pclose(p); snprintf(err, errsz, "oom"); return 1; }
    size_t r;
    while ((r = fread(buf + len, 1, cap - len, p)) > 0) {
        len += r;
        if (len == cap) {
            cap *= 2;
            unsigned char* nb = realloc(buf, cap);
            if (!nb) { free(buf); pclose(p); snprintf(err, errsz, "oom"); return 1; }
            buf = nb;
        }
    }
    int pc = pclose(p);
    if (pc != 0) { free(buf); snprintf(err, errsz, "xz failed rc=%d (corrupt/truncated)", pc); return 1; }

    uint64_t hdr_n, mv_n, ev_n;
    if (len < 8) { free(buf); snprintf(err, errsz, "too small"); return 1; }
    memcpy(&hdr_n, buf, 8);
    if (8 + hdr_n + 8 > len) { free(buf); snprintf(err, errsz, "hdr_n %llu overflows", (unsigned long long)hdr_n); return 1; }
    memcpy(&mv_n, buf + 8 + hdr_n, 8);
    if (8 + hdr_n + 8 + mv_n + 8 > len) { free(buf); snprintf(err, errsz, "mv_n %llu overflows", (unsigned long long)mv_n); return 1; }
    memcpy(&ev_n, buf + 8 + hdr_n + 8 + mv_n, 8);
    if (8 + hdr_n + 8 + mv_n + 8 + ev_n != len) {
        free(buf);
        snprintf(err, errsz, "size closure: 8+%llu+8+%llu+8+%llu != %zu",
                 (unsigned long long)hdr_n, (unsigned long long)mv_n,
                 (unsigned long long)ev_n, len);
        return 1;
    }
    const unsigned char* h  = buf + 8;
    const unsigned char* mv = buf + 8 + hdr_n + 8;
    const unsigned char* ev = buf + 8 + hdr_n + 8 + mv_n + 8;

    uint32_t nfens; memcpy(&nfens, h, 4);
    size_t off = 4;
    for (uint32_t i = 0; i < nfens; i++) {
        if (off + 2 > hdr_n) { free(buf); snprintf(err, errsz, "fen table overflow"); return 1; }
        uint16_t fl; memcpy(&fl, h + off, 2);
        off += 2 + fl;
        if (off > hdr_n) { free(buf); snprintf(err, errsz, "fen table overflow"); return 1; }
    }
    size_t games_bytes = hdr_n - off;
    size_t ngames = games_bytes / 9;
    st->slack_bytes = games_bytes % 9;
    if (mv_n % 2 != 0) { free(buf); snprintf(err, errsz, "mv_n odd"); return 1; }

    size_t ecur = 0, mcur = 0;
    uint64_t range_bad = 0;
    for (size_t g = 0; g < ngames; g++) {
        const unsigned char* e = h + off + g * 9;
        uint16_t np;   memcpy(&np, e, 2);
        uint8_t  stm = e[2];
        uint32_t fidx; memcpy(&fidx, e + 3, 4);
        int16_t  se;   memcpy(&se, e + 7, 2);
        if (np == 0)       { free(buf); snprintf(err, errsz, "game %zu np=0", g); return 1; }
        if (stm > 1)       { free(buf); snprintf(err, errsz, "game %zu stm=%u", g, stm); return 1; }
        if (fidx >= nfens) { free(buf); snprintf(err, errsz, "game %zu fidx OOB", g); return 1; }
        if (se < -5000 || se > 5000) range_bad++;
        for (int k = 0; k < (int)np; k++) {
            uint16_t m; memcpy(&m, mv + mcur, 2); mcur += 2;
            uint16_t fn = (m >> 12) & 0xF;
            if (fn == 6 || fn == 7) { free(buf); snprintf(err, errsz, "game %zu move %04X bad flag", g, m); return 1; }
        }
        int curev = se;
        int a = curev < 0 ? -curev : curev;
        st->h[a <= 100 ? 0 : a <= 300 ? 1 : a <= 1000 ? 2 : 3]++;
        if (curev == 0) st->h[4]++;
        for (int k = 0; k < (int)np - 1; k++) {
            if (ecur >= ev_n) { free(buf); snprintf(err, errsz, "ev overrun game %zu", g); return 1; }
            uint8_t c = ev[ecur];
            if (c == 0x80) {
                if (ecur + 3 > ev_n) { free(buf); snprintf(err, errsz, "ev abs overrun game %zu", g); return 1; }
                int16_t abs16; memcpy(&abs16, ev + ecur + 1, 2);
                curev = abs16; ecur += 3;
            } else {
                curev += (int8_t)c; ecur += 1;
            }
            if (curev < -5000 || curev > 5000) range_bad++;
            a = curev < 0 ? -curev : curev;
            st->h[a <= 100 ? 0 : a <= 300 ? 1 : a <= 1000 ? 2 : 3]++;
            if (curev == 0) st->h[4]++;
        }
        st->games++;
        st->positions += (uint64_t)np;   // consumer emits n_pos positions per game
    }
    if (mcur > mv_n) { free(buf); snprintf(err, errsz, "move overrun %zu > %llu", mcur, (unsigned long long)mv_n); return 1; }
    if (ecur > ev_n) { free(buf); snprintf(err, errsz, "ev walk overrun %zu > %llu", ecur, (unsigned long long)ev_n); return 1; }
    st->mv_left = mv_n - mcur;
    st->ev_left = ev_n - ecur;
    st->evals = st->positions;           // one eval per emitted position (start_eval + np-1 deltas)
    st->range_bad = range_bad;
    free(buf);
    return 0;
}

int main(int argc, char** argv) {
    Stats tot; memset(&tot, 0, sizeof tot);
    int bad = 0;
    for (int i = 1; i < argc; i++) {
        Stats st; memset(&st, 0, sizeof st);
        char err[256] = "";
        if (verify_one(argv[i], &st, err, sizeof err)) {
            printf("FAIL %s: %s\n", argv[i], err);
            bad++;
        } else {
            uint64_t e = st.h[0] + st.h[1] + st.h[2] + st.h[3];
            int range_ok = st.range_bad * 1000 <= e;
            if (range_ok) {
                printf("PASS %s: %llu games %lluM pos | <=100 %.1f%% 100-300 %.1f%% 300-1000 %.1f%% >1000 %.1f%% | rbad %llu mvleft %llu evleft %llu\n",
                       argv[i], (unsigned long long)st.games, (unsigned long long)(st.positions / 1000000),
                       100.0 * st.h[0] / e, 100.0 * st.h[1] / e, 100.0 * st.h[2] / e, 100.0 * st.h[3] / e,
                       (unsigned long long)st.range_bad, (unsigned long long)st.mv_left, (unsigned long long)st.ev_left);
                tot.games += st.games; tot.positions += st.positions;
                for (int k = 0; k < 5; k++) tot.h[k] += st.h[k];
                tot.range_bad += st.range_bad; tot.mv_left += st.mv_left; tot.ev_left += st.ev_left;
                tot.slack_bytes += st.slack_bytes;
            } else {
                printf("FAIL %s: eval range violations %llu / %llu (>0.1%%)\n", argv[i],
                       (unsigned long long)st.range_bad, (unsigned long long)e);
                bad++;
            }
        }
        fflush(stdout);
    }
    uint64_t e = tot.h[0] + tot.h[1] + tot.h[2] + tot.h[3];
    printf("TOTAL %d frames: %llu games, %llu positions, %llu evals | <=100cp %.2f%% | 100-300 %.2f%% | 300-1000 %.2f%% | >1000cp %.2f%% | zero %.2f%% | BAD %d\n",
           argc - 1, (unsigned long long)tot.games, (unsigned long long)tot.positions,
           (unsigned long long)e,
           100.0 * tot.h[0] / e, 100.0 * tot.h[1] / e, 100.0 * tot.h[2] / e,
           100.0 * tot.h[3] / e, 100.0 * tot.h[4] / e, bad);
    printf("AGGREGATE: range_bad %llu | orphan mv bytes %llu | orphan ev bytes %llu | slack bytes %llu\n",
           (unsigned long long)tot.range_bad, (unsigned long long)tot.mv_left,
           (unsigned long long)tot.ev_left, (unsigned long long)tot.slack_bytes);
    return bad ? 2 : 0;
}
