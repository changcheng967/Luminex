// spsa_params.h — Runtime-tunable search constants for SPSA optimization.
// Defaults match the hand-set values that produced ~2470 Elo. Override via
// a spsa_params.txt file (12 whitespace-separated ints, one per line) placed
// next to the engine binary. Missing file → use defaults (no-op).
#pragma once
#include <cstdio>
#include <string>

struct SPSAParams {
    // LMR reduction scales (multiply the log(depth)*log(move_count) base)
    int lmr_scale_quiet = 40;
    int lmr_scale_noisy = 24;
    // Futility margin: base = futility_coeff * depth + futility_offset
    int futility_coeff   = 130;
    int futility_offset  = 50;
    // Null-move pruning reduction: R = nmp_base + (depth>nmp_thresh1?1:0) + (depth>nmp_thresh2?1:0)
    int nmp_base         = 3;
    int nmp_thresh1      = 5;
    int nmp_thresh2      = 12;
    // Razoring: prune if eval + razor_base + depth² * razor_coeff < alpha
    int razor_base       = 300;
    int razor_coeff      = 60;
    // Reverse futility (static null move): prune if eval - rev_fut_coeff * depth > beta
    int rev_fut_coeff    = 100;
    // Aspiration window initial half-size
    int aspiration_delta = 50;

    // Load from file (12 ints, one per line). Returns true if loaded.
    bool load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "r");
        if (!f) return false;
        int* p[] = {&lmr_scale_quiet, &lmr_scale_noisy, &futility_coeff,
                    &futility_offset, &nmp_base, &nmp_thresh1, &nmp_thresh2,
                    &razor_base, &razor_coeff, &rev_fut_coeff, &aspiration_delta,
                    &_singular_margin};  // 12th param
        for (int i = 0; i < 12; ++i)
            if (std::fscanf(f, "%d", p[i]) != 1) { std::fclose(f); return false; }
        std::fclose(f);
        return true;
    }

    // Save to file (for SPSA loop to write perturbed values)
    void save(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "w");
        if (!f) return;
        std::fprintf(f, "%d\n%d\n%d\n%d\n%d\n%d\n%d\n%d\n%d\n%d\n%d\n%d\n",
                     lmr_scale_quiet, lmr_scale_noisy, futility_coeff,
                     futility_offset, nmp_base, nmp_thresh1, nmp_thresh2,
                     razor_base, razor_coeff, rev_fut_coeff, aspiration_delta,
                     _singular_margin);
        std::fclose(f);
    }

    // Singular extension margin (beta offset for singularity test)
    int _singular_margin = 200;  // placeholder — will be wired into search.cpp

    // Global instance (loaded once at startup)
    static SPSAParams& get() {
        static SPSAParams instance;
        return instance;
    }
};
