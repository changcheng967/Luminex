// nnue_kernels.h — the network's SIMD kernels, one set per instruction-set
// level, and the table nnue.cpp calls them through.
//
// The kernels are the loops over accumulator lanes and the dense tail: a
// side's lanes rebuilt from the board, a child's lanes from its parent's, the
// forward pass. nnue_kernels.inc has their code once for every level. With
// CMake's LUMINEX_DISPATCH off (the default, and OpenBench's build) nnue.cpp
// includes it and calls the kernels of the level the compiler targets
// directly. With it on, one file per level (src/nnue_k_<level>.cpp) compiles
// it with that level's instruction-set flags and exports its table; nnue.cpp,
// compiled for x86-64-v2, picks the best table the CPU supports at run time,
// so one x86-64 binary runs every CPU at its own speed.
//
// This header is included by those per-level files, so it holds plain data
// and declarations only: nothing with a constructor, no inline function.
#pragma once
#include <cstdint>

namespace luminex::nnue {

// Compile-time cap on L1 width (the runtime L1, read from the net, may be
// smaller). The accumulator lanes and the dense input are sized from it.
inline constexpr int NNUE_L1_MAX = 1024;
// The widest L2 and L3 a net may have (load() checks it); the dense layers'
// buffers are sized from it.
inline constexpr int NNUE_DENSE_MAX = 64;
// Feature weights and accumulator lanes are int16 at this scale: SCReLU's
// window [0, 1] is lanes [0, 8192].
inline constexpr float NNUE_FT_SCALE = 8192.0f;

// How a kernel set wants L2's int8 weights (2*L1 inputs x L2 outputs)
// arranged; the file stores them output-major.
enum class L2Layout : uint8_t {
    OutputMajor,   // as stored: row o holds output o's 2*L1 weights
    Chunk32,       // AVX2: per 32-input chunk, the L2 outputs' 32 weights each
    ByInput4,      // AVX-512 VNNI with L2 = 16: per 4 inputs, the 16 outputs' 4 weights each
};

// One dense tail as the kernels read it, pointing into nnue.cpp's DenseTail:
// int8 weights in the layout the kernel set asked for (l2_by_input and
// l3_batched say which it got), float biases, the factors that scale each output's dot;
// for LNN1 nets the float weights instead.
struct TailView {
    const int8_t* l2_w = nullptr;
    const int8_t* l3_w = nullptr;
    const int8_t* out_w = nullptr;
    const float* l2_b = nullptr;
    const float* l3_b = nullptr;
    const float* l2_wf = nullptr;   // LNN1 float weights
    const float* l3_wf = nullptr;
    const float* out_wf = nullptr;
    // 1 / (s x 127) for each output of L2 and L3 (one scale per row in LNR8 nets, the layer's in
    // LNI8 / LNB8), and for the output
    const float* ics2 = nullptr;
    const float* ics3 = nullptr;
    float icso = 1.0f;
    float out_b = 0.0f;
    bool l2_by_input = false;
    bool l3_batched = false;
};

// One side of a child accumulator, from its parent's lanes and the feature
// rows its move changes: parent + add0 - sub0, then - sub1 and + add1 where
// given (a capture has sub1, castling both), saturating int16 after every step.
struct DeltaRows {
    const int16_t* parent;
    int16_t* child;
    const int16_t* add0;
    const int16_t* sub0;
    const int16_t* sub1;   // or nullptr
    const int16_t* add1;   // or nullptr (only given with sub1)
};

// A kernel set. n is the lane count (L1, a multiple of 32).
struct Kernels {
    const char* name;      // the level: avx512vnni, avx512, avxvnni, avx2, base, dotprod
    L2Layout l2_layout;
    bool l3_batched;       // reads L3 interleaved for batched VNNI when L2 = 16 and L3 % 4 = 0
    // lanes = the int16-clamped bias, then + rows[0], + rows[1], ... in order,
    // saturating after every row
    void (*refresh_lanes)(int16_t* lanes, const int32_t* bias, const int16_t* const* rows, int nrows, int n);
    // jobs[0..njobs) as DeltaRows describes
    void (*update_lanes)(const DeltaRows* jobs, int njobs, int n);
    // exact int32 sums: entry -= sub rows, entry += add rows; lanes = entry
    // saturated to int16
    void (*refresh_cached)(int32_t* entry, const int16_t* const* sub, int nsub,
                           const int16_t* const* add, int nadd, int16_t* lanes, int n);
    // The forward pass of an int8 tail (LNI8/LNB8) or a float one (LNN1) from
    // the side to move's and the other side's lanes: the output before the
    // x300 and rounding. act_nonzero, if given, gains the number of nonzero
    // L1 activations (NNUE_PROFILE).
    float (*forward_int8)(const int16_t* stm, const int16_t* nstm, const TailView& t,
                          int L1, int L2, int L3, long long* act_nonzero);
    float (*forward_float)(const int16_t* stm, const int16_t* nstm, const TailView& t,
                           int L1, int L2, int L3);
    // b + d * s rounded once, as one FMA: how the int8 tail adds a layer's
    // bias to its integer dot d scaled by s. Exposed so nnuecheck can test the
    // base level's software version against hardware FMA.
    float (*bias_add)(float b, int32_t d, float s);
};

#if defined(LUMINEX_NNUE_DISPATCH)
// The universal x86-64 build's kernel sets, best first; each is defined in
// src/nnue_k_<level>.cpp.
namespace k_avx512vnni { extern const Kernels kernels; }   // x86-64-v4 + AVX512-VNNI
namespace k_avx512     { extern const Kernels kernels; }   // x86-64-v4 (AVX-512 F/BW/CD/DQ/VL)
namespace k_avxvnni    { extern const Kernels kernels; }   // x86-64-v3 + AVX-VNNI
namespace k_avx2       { extern const Kernels kernels; }   // x86-64-v3 (AVX2, BMI2, FMA)
namespace k_base       { extern const Kernels kernels; }   // what the rest of the engine targets
#endif

} // namespace luminex::nnue
