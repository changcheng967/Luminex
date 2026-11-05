// spsa_params.h — Runtime-tunable search constants for SPSA optimization.
// Defaults are the SPSA-converged values for the quiet-trained net (v6.3.0,
// tuned via self-play perturbation pairs, 61 iterations x 120 games).
// Each field is also a UCI spin option (see uci.cpp), so OpenBench SPSA can
// tune it directly. A spsa_params.txt in the working directory (7 integers,
// one per line, in field order) is read once at startup; UCI options set
// afterwards take precedence. Missing file -> defaults.
#pragma once
#include <cstdio>
#include <string>

struct SPSAParams {
    // LMR reduction scale (reduction = scale^2 * ln(depth) * ln(moves) / 1024)
    int lmr_scale_quiet  = 38;
    // Null-move pruning reduction base: R = nmp_base + depth / 3 + min(margin / 200, 3)
    int nmp_base         = 3;
    // Razoring: prune if eval + razor_base + depth^2 * razor_coeff < alpha
    int razor_base       = 317;
    int razor_coeff      = 56;
    // Reverse futility (static null move): prune if eval - rev_fut_coeff * depth >= beta
    int rev_fut_coeff    = 103;
    // Aspiration window initial half-size
    int aspiration_delta = 48;
    // Singular extension margin: sBeta = tt_value - depth * _singular_margin / 320
    int _singular_margin = 194;

    static constexpr int COUNT = 7;

    // Load from file (7 ints, one per line, in field order). Returns true if loaded.
    bool load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "r");
        if (!f) return false;
        int* p[COUNT] = {&lmr_scale_quiet, &nmp_base, &razor_base, &razor_coeff,
                         &rev_fut_coeff, &aspiration_delta, &_singular_margin};
        int v[COUNT];
        for (int i = 0; i < COUNT; ++i)
            if (std::fscanf(f, "%d", &v[i]) != 1) { std::fclose(f); return false; }
        std::fclose(f);
        for (int i = 0; i < COUNT; ++i) *p[i] = v[i];
        return true;
    }

    // Save to file (for SPSA loop to write perturbed values)
    void save(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "w");
        if (!f) return;
        std::fprintf(f, "%d\n%d\n%d\n%d\n%d\n%d\n%d\n",
                     lmr_scale_quiet, nmp_base, razor_base, razor_coeff,
                     rev_fut_coeff, aspiration_delta, _singular_margin);
        std::fclose(f);
    }

    // Global instance
    static SPSAParams& get() {
        static SPSAParams instance;
        return instance;
    }
};
