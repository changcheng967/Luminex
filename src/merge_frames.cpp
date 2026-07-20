// merge_frames.cpp — combine N Game-Pack frames (from parallel encoder runs) into one.
//
// Frame format (each input): [u64 len_hdr][hdr][u64 len_mv][mv][u64 len_ev][ev]
//   hdr = [u32 nfens][u16 len+bytes per fen][per game: u8 n_pos, u8 stm, u32 fen_idx, s16 start_eval]
//
// Merge: union all FENs into one dict (dedup), remap each game's fen_idx to the unified index,
// concatenate mv and ev blocks in input order. Streams mv/ev from the input files (only hdr
// data is held in RAM) so it scales to billions of positions.
//
// Usage: ./merge_frames -o merged.raw frame1.raw frame2.raw ...
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

int main(int argc, char** argv) {
    const char* out_path = nullptr;
    std::vector<std::string> inputs;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-o" && i + 1 < argc) out_path = argv[++i];
        else if (!a.empty() && a[0] != '-') inputs.push_back(a);
    }
    if (!out_path || inputs.empty()) { fprintf(stderr, "usage: merge_frames -o out.raw f1.raw ...\n"); return 1; }

    std::unordered_map<std::string, uint32_t> uni_dict;
    std::vector<std::string> uni_fens;
    std::vector<uint8_t> uni_game_entries;          // all games, fen_idx remapped (8 bytes each)

    struct Frame { std::string path; uint64_t mv_off, mv_len, ev_off, ev_len; };
    std::vector<Frame> frames;
    uint64_t total_mv = 0, total_ev = 0, total_games = 0, total_pos = 0;

    // Pass 1: parse each frame's hdr, build unified FEN dict + remapped game entries.
    for (auto& path : inputs) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) { fprintf(stderr, "skip (cannot open) %s\n", path.c_str()); continue; }
        uint64_t len_hdr, len_mv, len_ev;
        if (fread(&len_hdr, 8, 1, f) != 1 || len_hdr > (uint64_t)1e11) {
            fprintf(stderr, "skip (bad frame) %s\n", path.c_str()); fclose(f); continue; }
        std::vector<uint8_t> hdr(len_hdr);
        if (fread(hdr.data(), 1, len_hdr, f) != len_hdr) {
            fprintf(stderr, "skip (short hdr) %s\n", path.c_str()); fclose(f); continue; }
        if (fread(&len_mv, 8, 1, f) != 1) { fprintf(stderr, "skip (no mv) %s\n", path.c_str()); fclose(f); continue; }
        uint64_t mv_off = ftell(f); fseek(f, (long)len_mv, SEEK_CUR);
        if (fread(&len_ev, 8, 1, f) != 1) { fprintf(stderr, "skip (no ev) %s\n", path.c_str()); fclose(f); continue; }
        uint64_t ev_off = ftell(f);
        fclose(f);

        // parse hdr: u32 nfens, fens, then game entries
        size_t p = 0;
        uint32_t nfens; memcpy(&nfens, hdr.data() + p, 4); p += 4;
        std::vector<uint32_t> local_to_uni(nfens);
        for (uint32_t i = 0; i < nfens; ++i) {
            uint16_t fl; memcpy(&fl, hdr.data() + p, 2); p += 2;
            std::string fen((const char*)hdr.data() + p, fl); p += fl;
            auto it = uni_dict.find(fen);
            uint32_t uid;
            if (it == uni_dict.end()) { uid = (uint32_t)uni_fens.size(); uni_dict[fen] = uid; uni_fens.push_back(fen); }
            else uid = it->second;
            local_to_uni[i] = uid;
        }
        // game entries: rest of hdr (8 bytes each: n_pos, stm, u32 fen_idx, s16 start_eval)
        size_t n_entries = (len_hdr - p) / 8;
        for (size_t g = 0; g < n_entries; ++g) {
            const uint8_t* e = hdr.data() + p + g * 8;
            uint8_t n_pos = e[0], stm = e[1];
            uint32_t local_fidx; memcpy(&local_fidx, e + 2, 4);
            int16_t start_eval; memcpy(&start_eval, e + 6, 2);
            uint32_t uid = (local_fidx < nfens) ? local_to_uni[local_fidx] : 0;
            uni_game_entries.push_back(n_pos);
            uni_game_entries.push_back(stm);
            uni_game_entries.insert(uni_game_entries.end(), (uint8_t*)&uid, (uint8_t*)&uid + 4);
            uni_game_entries.insert(uni_game_entries.end(), (uint8_t*)&start_eval, (uint8_t*)&start_eval + 2);
            total_games++; total_pos += n_pos;
        }
        frames.push_back({path, mv_off, len_mv, ev_off, len_ev});
        total_mv += len_mv; total_ev += len_ev;
    }

    // Build unified hdr: u32 nfens + fens + game_entries.
    std::vector<uint8_t> uni_hdr;
    uint32_t uni_nfens = (uint32_t)uni_fens.size();
    uni_hdr.insert(uni_hdr.end(), (uint8_t*)&uni_nfens, (uint8_t*)&uni_nfens + 4);
    for (auto& fen : uni_fens) {
        uint16_t fl = (uint16_t)fen.size();
        uni_hdr.insert(uni_hdr.end(), (uint8_t*)&fl, (uint8_t*)&fl + 2);
        uni_hdr.insert(uni_hdr.end(), fen.begin(), fen.end());
    }
    uni_hdr.insert(uni_hdr.end(), uni_game_entries.begin(), uni_game_entries.end());

    // Pass 2: write framed output, streaming mv/ev from each input.
    FILE* out = fopen(out_path, "wb");
    if (!out) { fprintf(stderr, "cannot open output\n"); return 1; }
    uint64_t len_hdr = uni_hdr.size();
    fwrite(&len_hdr, 8, 1, out); fwrite(uni_hdr.data(), 1, uni_hdr.size(), out);
    fwrite(&total_mv, 8, 1, out);
    uint8_t buf[1 << 20];
    for (auto& fr : frames) {
        FILE* f = fopen(fr.path.c_str(), "rb");
        fseek(f, (long)fr.mv_off, SEEK_SET);
        for (uint64_t left = fr.mv_len; left > 0;) { size_t r = fread(buf, 1, std::min<uint64_t>(sizeof(buf), left), f); fwrite(buf, 1, r, out); left -= r; }
        fclose(f);
    }
    fwrite(&total_ev, 8, 1, out);
    for (auto& fr : frames) {
        FILE* f = fopen(fr.path.c_str(), "rb");
        fseek(f, (long)fr.ev_off, SEEK_SET);
        for (uint64_t left = fr.ev_len; left > 0;) { size_t r = fread(buf, 1, std::min<uint64_t>(sizeof(buf), left), f); fwrite(buf, 1, r, out); left -= r; }
        fclose(f);
    }
    fflush(out); fclose(out);

    fprintf(stderr, "merge: %zu frames | games=%llu pos=%llu unique_fens=%u | out=%s (%llu B hdr, %llu mv, %llu ev)\n",
            frames.size(), (unsigned long long)total_games, (unsigned long long)total_pos, uni_nfens,
            out_path, (unsigned long long)len_hdr, (unsigned long long)total_mv, (unsigned long long)total_ev);
    return 0;
}
