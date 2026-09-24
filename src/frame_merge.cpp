// frame_merge — merge N gamepack frames into one denser frame.
//
// Frame format (written by lc0pack.cpp, read by luminex-featurize):
//   [u64 hdr_len][hdr][u64 mv_len][mv][u64 ev_len][ev]
//   hdr = u32 nfens, { u16 len + fen bytes } * nfens, then 9-byte game entries:
//         u16 np (LE) | u8 stm_white | u32 fidx (LE) | i16 start_eval (LE)
//   mv   = np u16 raw moves per game, entries in order
//   ev   = per game (np-1) eval deltas: int8, or 0x80 + i16 absolute (3 bytes)
//
// Per-game data is fully self-contained (start_eval anchors each game's delta
// chain), so a merge is: concat FEN tables with fidx remapping by cumulative
// offsets, concat game entries, concat mv and ev streams — order preserving.
// The tool WALKS every game's eval stream to prove the parse consumed exactly
// ev_len bytes (the same walk the featurizer performs); any mismatch aborts
// before writing, so a corrupt input can never produce a corrupt output.
//
// usage: frame_merge -o merged.raw frameA.raw frameB.raw [more...]
// (frames on disk are .zst/.xz compressed; decompress before, recompress after)

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

static bool read_u64(FILE* f, uint64_t& v) {
    return fread(&v, 8, 1, f) == 1;
}
static bool read_exact(FILE* f, std::vector<uint8_t>& buf, uint64_t n) {
    buf.resize((size_t)n);
    return n == 0 || fread(buf.data(), 1, (size_t)n, f) == n;
}

struct Frame {
    std::vector<uint8_t> hdr;   // nfens + fen table + game entries (remappable)
    std::vector<uint8_t> mv;
    std::vector<uint8_t> ev;
    uint32_t nfens = 0;
    size_t n_entries = 0;
    uint64_t total_np = 0;      // sum of np across games (u16 move count check)
};

static bool load_frame(const char* path, Frame& fr, uint64_t& games_out) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "frame_merge: cannot open %s\n", path); return false; }
    uint64_t hl = 0, ml = 0, el = 0;
    bool ok = read_u64(f, hl) && read_exact(f, fr.hdr, hl)
           && read_u64(f, ml) && read_exact(f, fr.mv, ml)
           && read_u64(f, el) && read_exact(f, fr.ev, el);
    // trailing bytes after ev = corruption (featurize would misparse the next
    // section boundary in a concatenated stream context)
    uint8_t extra;
    if (ok && fread(&extra, 1, 1, f) == 1) { fprintf(stderr, "frame_merge: %s has trailing bytes\n", path); ok = false; }
    fclose(f);
    if (!ok) { fprintf(stderr, "frame_merge: %s truncated/invalid\n", path); return false; }

    // ---- parse header: fen table, then 9-byte entries ----
    size_t p = 0;
    if (fr.hdr.size() < 4) { fprintf(stderr, "frame_merge: %s hdr too small\n", path); return false; }
    memcpy(&fr.nfens, fr.hdr.data() + p, 4); p += 4;
    for (uint32_t i = 0; i < fr.nfens; ++i) {
        if (p + 2 > fr.hdr.size()) { fprintf(stderr, "frame_merge: %s fen table overrun\n", path); return false; }
        uint16_t fl; memcpy(&fl, fr.hdr.data() + p, 2); p += 2;
        if (p + fl > fr.hdr.size()) { fprintf(stderr, "frame_merge: %s fen string overrun\n", path); return false; }
        p += fl;
    }
    if ((fr.hdr.size() - p) % 9 != 0) { fprintf(stderr, "frame_merge: %s entry block not 9-byte aligned\n", path); return false; }
    fr.n_entries = (fr.hdr.size() - p) / 9;
    if (fr.mv.size() % 2 != 0) { fprintf(stderr, "frame_merge: %s mv length odd\n", path); return false; }

    // ---- walk entries: verify fidx range, np sum vs mv_len, and that the
    // delta-eval walk consumes EXACTLY ev_len bytes (the featurizer's read) ----
    uint64_t np_sum = 0, ev_pos = 0;
    for (size_t e = 0; e < fr.n_entries; ++e) {
        const uint8_t* ent = fr.hdr.data() + p + e * 9;
        uint16_t np; memcpy(&np, ent, 2);
        uint32_t fidx; memcpy(&fidx, ent + 3, 4);
        if (fidx >= fr.nfens) { fprintf(stderr, "frame_merge: %s entry %zu fidx %u >= nfens %u\n", path, e, fidx, fr.nfens); return false; }
        if (np == 0) continue;   // start board only — no moves, no evals
        np_sum += np;
        for (uint32_t d = 0; d + 1 < np; ++d) {
            if (ev_pos >= fr.ev.size()) { fprintf(stderr, "frame_merge: %s eval stream underrun (entry %zu)\n", path, e); return false; }
            if (fr.ev[ev_pos] == 0x80) ev_pos += 3; else ev_pos += 1;
        }
    }
    if (np_sum * 2 != fr.mv.size()) { fprintf(stderr, "frame_merge: %s move count %llu != entries %llu\n", path, (unsigned long long)(fr.mv.size() / 2), (unsigned long long)np_sum); return false; }
    if (ev_pos != fr.ev.size()) { fprintf(stderr, "frame_merge: %s eval stream length mismatch (%llu used of %llu)\n", path, (unsigned long long)ev_pos, (unsigned long long)fr.ev.size()); return false; }
    fr.total_np = np_sum;
    games_out = fr.n_entries;
    return true;
}

int main(int argc, char** argv) {
    std::string out_path;
    std::vector<std::string> inputs;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-o" || a == "--out") out_path = argv[++i];
        else inputs.push_back(a);
    }
    if (inputs.empty() || out_path.empty()) {
        fprintf(stderr, "usage: frame_merge -o merged.raw frameA.raw frameB.raw [more...]\n");
        return 2;
    }

    std::vector<uint8_t> fens, entries, mv, ev;   // hdr = [nfens][fens][entries]
    uint64_t nfens_total = 0, games_total = 0, np_total = 0;
    uint32_t fens_so_far = 0;
    for (auto& path : inputs) {
        Frame fr; uint64_t g = 0;
        if (!load_frame(path.c_str(), fr, g)) return 1;
        size_t entry_off;
        {   // recompute where the entry block starts in this frame's hdr
            size_t p = 4;
            for (uint32_t i = 0; i < fr.nfens; ++i) { uint16_t fl; memcpy(&fl, fr.hdr.data() + p, 2); p += 2 + fl; }
            entry_off = p;
        }
        // remap every entry's fidx into the merged FEN table's coordinate space
        for (size_t e = 0; e < fr.n_entries; ++e) {
            uint8_t* ent = fr.hdr.data() + entry_off + e * 9;
            uint32_t fidx; memcpy(&fidx, ent + 3, 4);
            uint32_t nf = fidx + fens_so_far;   // both < 2^32: nfens total stays tiny
            memcpy(ent + 3, &nf, 4);
        }
        // the format is [ALL fens][ALL entries] — keep the sections separate
        fens.insert(fens.end(), fr.hdr.begin() + 4, fr.hdr.begin() + entry_off);
        entries.insert(entries.end(), fr.hdr.begin() + entry_off, fr.hdr.end());
        mv.insert(mv.end(), fr.mv.begin(), fr.mv.end());
        ev.insert(ev.end(), fr.ev.begin(), fr.ev.end());
        fens_so_far += fr.nfens; nfens_total += fr.nfens; games_total += g; np_total += fr.total_np;
    }

    // ---- write merged frame: [nfens][fens][entries][mv][ev] ----
    uint32_t nf = (uint32_t)nfens_total;
    FILE* o = fopen(out_path.c_str(), "wb");
    if (!o) { fprintf(stderr, "frame_merge: cannot write %s\n", out_path.c_str()); return 1; }
    uint64_t hl = 4 + (uint64_t)fens.size() + entries.size(), ml = mv.size(), el = ev.size();
    fwrite(&hl, 8, 1, o);
    fwrite(&nf, 4, 1, o);
    fwrite(fens.data(), 1, fens.size(), o);
    fwrite(entries.data(), 1, entries.size(), o);
    fwrite(&ml, 8, 1, o); fwrite(mv.data(), 1, mv.size(), o);
    fwrite(&el, 8, 1, o); fwrite(ev.data(), 1, ev.size(), o);
    fclose(o);
    fprintf(stderr, "frame_merge: %zu frames -> %s | games=%llu fens=%llu moves=%llu (%.1fM pos)\n",
            inputs.size(), out_path.c_str(), (unsigned long long)games_total,
            (unsigned long long)nfens_total, (unsigned long long)(ml / 2),
            (double)np_total / 1e6);
    return 0;
}
