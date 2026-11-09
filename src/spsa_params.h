// spsa_params.h — Runtime-tunable search constants for SPSA optimization.
// Every default but the shallow null-move pair is the rounded final value of
// OpenBench tune 45 (SPSA on those 27 fields at once, 80,000 games at 10+0.1,
// run on main 5545f07's search); the pair is the setting #41's SPRT tested.
// Each field is also a UCI spin option (see uci.cpp), so OpenBench SPSA can
// tune it directly. A spsa_params.txt in the working directory (the first 7
// integers, one per line, in field order) is read once at startup; UCI options
// set afterwards take precedence. Missing file -> defaults.
#pragma once
#include <cstdio>
#include <string>

struct SPSAParams {
    // LMR reduction scale (reduction = scale^2 * ln(depth) * ln(moves) / 1024)
    int lmr_scale_quiet  = 31;
    // Null-move pruning reduction base: R = nmp_base + depth / 3 + min(margin / 200, 3)
    int nmp_base         = 5;
    // Razoring: prune if eval + razor_base + depth^2 * razor_coeff < alpha
    int razor_base       = 301;
    int razor_coeff      = 56;
    // Reverse futility (static null move): prune if eval - rev_fut_coeff * depth >= beta
    int rev_fut_coeff    = 88;
    // Aspiration window initial half-size
    int aspiration_delta = 40;
    // Singular extension margin: sBeta = tt_value - depth * _singular_margin / 320
    int _singular_margin = 190;

    static constexpr int COUNT = 7;  // fields read from and written to spsa_params.txt

    // Reverse futility: extra margin when neither side's eval trend helps us
    int rfp_not_improving = 28;
    // Futility (quiet moves, depth <= 5): prune if eval + depth * fut_coeff
    // + (improving ? fut_improving : fut_base) < alpha
    int fut_coeff        = 138;
    int fut_base         = 102;
    int fut_improving    = 27;
    // Late move pruning, in hundredths of a move: the count is
    // (lmp_base + 100 if improving + depth^2 * lmp_coeff) / 100
    int lmp_base         = 223;
    int lmp_coeff        = 96;
    // History pruning (quiets, depth <= 4): prune if combined history
    // < hist_prune_base - hist_prune_coeff * (depth - 1)
    int hist_prune_base  = 866;
    int hist_prune_coeff = 5034;
    // ProbCut: rbeta = beta + probcut_margin
    int probcut_margin   = 140;
    // Null-move R: the depth term is depth * 100 / nmp_depth_div, the eval
    // term min((eval - beta) / nmp_eval_div, 3)
    int nmp_depth_div    = 283;
    int nmp_eval_div     = 207;
    // LMR: history adjustment = clamp(history / lmr_hist_div, -3, 3); the
    // cut-node and ttPv adjustments are in 1024ths of a ply
    int lmr_hist_div     = 4014;
    int lmr_cut_node     = 1006;
    int lmr_ttpv         = 2003;
    // Double singular extension when the verification fails by more than
    // depth * dext_margin / 16
    int dext_margin      = 17;
    // History bonus: min(quad * d^2 + lin * d - offset, 8192); the malus is
    // its negative
    int hist_bonus_quad  = 20;
    int hist_bonus_lin   = 185;
    int hist_bonus_offset = 126;
    // Qsearch skips captures with SEE below qs_see_coeff * depth (depth <= 0)
    int qs_see_coeff     = 53;
    // Correction history update weight: min(corr_weight_max, depth + 1) / 256
    int corr_weight_max  = 13;

    // Null-move pruning at depth <= nmp_shallow_depth needs the static eval
    // at least nmp_shallow_margin above beta (deeper: at beta)
    int nmp_shallow_margin = 60;
    int nmp_shallow_depth  = 5;

    // Load from file (COUNT ints, one per line, in field order). Returns true if loaded.
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

// The UCI spin options, one per field. uci.cpp prints them and clamps values
// set through them to [min, max], which also keeps every divisor above zero.
struct SPSAOption {
    const char* name;
    int SPSAParams::* field;
    int min, max;
};

inline constexpr SPSAOption SPSA_OPTIONS[] = {
    {"LmrScale",        &SPSAParams::lmr_scale_quiet,    20,    70},
    {"NmpBase",         &SPSAParams::nmp_base,            2,     6},
    {"RazorBase",       &SPSAParams::razor_base,        150,   500},
    {"RazorCoeff",      &SPSAParams::razor_coeff,        30,   120},
    {"RfpCoeff",        &SPSAParams::rev_fut_coeff,      60,   160},
    {"AspirationDelta", &SPSAParams::aspiration_delta,   20,   100},
    {"SingularMargin",  &SPSAParams::_singular_margin,  100,   400},
    {"RfpNotImproving", &SPSAParams::rfp_not_improving,   0,   100},
    {"FutCoeff",        &SPSAParams::fut_coeff,          50,   300},
    {"FutBase",         &SPSAParams::fut_base,            0,   250},
    {"FutImproving",    &SPSAParams::fut_improving,       0,   200},
    {"LmpBase",         &SPSAParams::lmp_base,           50,   600},
    {"LmpCoeff",        &SPSAParams::lmp_coeff,          25,   300},
    {"HistPruneBase",   &SPSAParams::hist_prune_base, -4000,  4000},
    {"HistPruneCoeff",  &SPSAParams::hist_prune_coeff, 1000, 12000},
    {"ProbcutMargin",   &SPSAParams::probcut_margin,     50,   400},
    {"NmpDepthDiv",     &SPSAParams::nmp_depth_div,     150,   800},
    {"NmpEvalDiv",      &SPSAParams::nmp_eval_div,       50,   600},
    {"LmrHistDiv",      &SPSAParams::lmr_hist_div,     1024, 16384},
    {"LmrCutNode",      &SPSAParams::lmr_cut_node,        0,  3072},
    {"LmrTtPv",         &SPSAParams::lmr_ttpv,            0,  4096},
    {"DoubleExtMargin", &SPSAParams::dext_margin,         0,    64},
    {"HistBonusQuad",   &SPSAParams::hist_bonus_quad,     0,    64},
    {"HistBonusLin",    &SPSAParams::hist_bonus_lin,      0,   400},
    {"HistBonusOffset", &SPSAParams::hist_bonus_offset,   0,   400},
    {"QsSeeCoeff",      &SPSAParams::qs_see_coeff,        0,   200},
    {"CorrWeightMax",   &SPSAParams::corr_weight_max,     1,   128},
    {"NmpShallowMargin", &SPSAParams::nmp_shallow_margin, 0,  300},
    {"NmpShallowDepth", &SPSAParams::nmp_shallow_depth,   0,   12},
};

inline const SPSAOption* find_spsa_option(const std::string& name) {
    for (const SPSAOption& o : SPSA_OPTIONS)
        if (name == o.name) return &o;
    return nullptr;
}
