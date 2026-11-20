// nnue_kernels.h — the network's SIMD kernels, written once for every
// instruction-set level, and the table nnue.cpp calls them through.
//
// The kernels are the loops over accumulator lanes and the dense tail: a
// side's lanes rebuilt from the board, a child's lanes from its parent's, the
// forward pass. Their declarations come first, then their code. With CMake's
// LUMINEX_DISPATCH off (the default, and OpenBench's build) nnue.cpp includes
// both and calls the kernels of the level the compiler targets directly. With
// it on, nnue.cpp includes the declarations only, and CMake compiles this file
// once per level with that level's instruction-set flags (through the
// nnue_k_<level>.cpp it writes into the build folder: LK_NAMESPACE and
// LK_EXPECT, then this file); each exports its table, and nnue.cpp, compiled
// for x86-64-v2, picks the best table the CPU supports at run time, so one
// x86-64 binary runs every CPU at its own speed.
//
// The per-level compilations include none of the engine's code, and the
// declarations are plain data: nothing with a constructor, no inline function.
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
// window [0, 1] is lanes [0, 2048], and int16 holds sums to +-16.0 window
// units. At x8192 int16 ended at 4.0, below the sums real nets reach (11.5
// on gen768s_03q), and the saturating updates drifted: 12.7% of a search's
// evaluations were not the net's output. The updates now wrap instead of
// saturating, so a lane is exact whenever its sum fits (DeltaRows). The
// trainer's QAT rounds to the same grid (FT_GRID in Luminex-NNUE's
// trainer/resident_train.py; that repository's CI checks the two agree on
// every push there and daily against this file on main).
inline constexpr float NNUE_FT_SCALE = 2048.0f;

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
// given (a capture has sub1, castling both), in wrapping int16 arithmetic: the
// exact sum whenever it fits int16, whatever a partial sum did on the way.
struct DeltaRows {
    const int16_t* parent;
    int16_t* child;
    const int16_t* add0;
    const int16_t* sub0;
    const int16_t* sub1;   // or nullptr
    const int16_t* add1;   // or nullptr (only given with sub1)
};

// gen768t's relation lanes for the forward pass (docs/nnue/V5-GEN768T.md, section 3): lane l < T of each
// side's piece-square lanes is read as the saturating int16 sum lane + clamp(rel[l], -NNUE_REL_L,
// NNUE_REL_L) * 2^rs. SCReLU reads a lane only through clamp(a, 0, 1), so a sum that saturates activates
// as the exact one would. rs is at most 6, so the shifted term itself fits int16.
inline constexpr int NNUE_REL_L = 511;
struct RelLanes {
    const int16_t* stm;    // the side to move's relation lanes, T of them
    const int16_t* nstm;   // the other side's
    int T;                 // a multiple of 64, at most L1
    int rs;                // a relation step is 2^rs units of the feature grid
};

// A kernel set. n is the lane count (L1, a multiple of 32).
struct Kernels {
    const char* name;      // the level: avx512vnni, avx512, avxvnni, avx2, base, dotprod
    L2Layout l2_layout;
    bool l3_batched;       // reads L3 interleaved for batched VNNI when L2 = 16 and L3 % 4 = 0
    // lanes = the int16-clamped bias, then + rows[0], + rows[1], ... in
    // wrapping int16 arithmetic (exact whenever the sum fits)
    void (*refresh_lanes)(int16_t* lanes, const int32_t* bias, const int16_t* const* rows, int nrows, int n);
    // jobs[0..njobs) as DeltaRows describes
    void (*update_lanes)(const DeltaRows* jobs, int njobs, int n);
    // exact int32 sums: entry -= sub rows, entry += add rows; lanes = entry
    // saturated to int16
    void (*refresh_cached)(int32_t* entry, const int16_t* const* sub, int nsub,
                           const int16_t* const* add, int nadd, int16_t* lanes, int n);
    // The forward pass of an int8 tail (LNI8/LNB8/LNR8) or a float one (LNN1)
    // from the side to move's and the other side's lanes: the output before
    // the x300 and rounding. act_nonzero, if given, gains the number of
    // nonzero L1 activations (NNUE_PROFILE).
    float (*forward_int8)(const int16_t* stm, const int16_t* nstm, const TailView& t,
                          int L1, int L2, int L3, long long* act_nonzero);
    float (*forward_float)(const int16_t* stm, const int16_t* nstm, const TailView& t,
                           int L1, int L2, int L3);
    // b + d * s rounded once, as one FMA: how the int8 tail adds a layer's
    // bias to its integer dot d scaled by s. Exposed so nnuecheck can test the
    // base level's software version against hardware FMA.
    float (*bias_add)(float b, int32_t d, float s);
    // gen768t's relation lanes (T of them, a multiple of 64): child = parent + the
    // int8 rows add[0..na) - the rows sub[0..ns), widened to int16 (the sums never
    // overflow: at most 248 rows of |q| <= 127 are active in any position)
    void (*rel_rows)(const int16_t* parent, int16_t* child, const int8_t* const* add, int na,
                     const int8_t* const* sub, int ns, int T);
    // forward_int8 for an LNT8 net: lanes 0..T-1 read with the relation term
    // (RelLanes). A separate entry, so every other format runs the forward pass
    // it always ran.
    float (*forward_int8_rel)(const int16_t* stm, const int16_t* nstm, const RelLanes& rel, const TailView& t,
                              int L1, int L2, int L3, long long* act_nonzero);
};

#if defined(LUMINEX_NNUE_DISPATCH)
// The universal x86-64 build's kernel sets, best first; each is defined by
// this file's code compiled for its level.
namespace k_avx512vnni { extern const Kernels kernels; }   // x86-64-v4 + AVX512-VNNI
namespace k_avx512     { extern const Kernels kernels; }   // x86-64-v4 (AVX-512 F/BW/CD/DQ/VL)
namespace k_avxvnni    { extern const Kernels kernels; }   // x86-64-v3 + AVX-VNNI
namespace k_avx2       { extern const Kernels kernels; }   // x86-64-v3 (AVX2, BMI2, FMA)
namespace k_base       { extern const Kernels kernels; }   // what the rest of the engine targets
#endif

} // namespace luminex::nnue

#if !defined(LUMINEX_NNUE_DISPATCH) || defined(LK_NAMESPACE)

// The kernels' code, written once for every instruction-set level (the
// declarations above say what each computes).
//
// Two ways in:
// - LUMINEX_DISPATCH off (the default, and OpenBench's build): nnue.cpp
//   includes this file, and the kernels are those of the level the compiler
//   targets (namespace k_native), called directly.
// - LUMINEX_DISPATCH on (the universal x86-64 build): CMake compiles this file
//   once per level, under that level's namespace (LK_NAMESPACE) and with its
//   instruction-set flags, and each exports the table at the end. nnue.cpp,
//   compiled for x86-64-v2, calls the table of the best level the CPU supports.
//
// A per-level compilation is for CPUs the rest of the program must not
// assume, so nothing in it may run before main (no constructors, no dynamic
// initialization: the table is constinit), and it must not define an inline
// function another file also defines: the linker keeps one copy of such a
// function for the whole program, and this file's copy could then run on a CPU
// without its instructions. So everything here except the table is internal to
// the including file (an unnamed namespace), and the code calls no standard
// library function but memcpy and lrintf, which are the C library's.
//
// Every level computes the same evaluation, bit for bit, for int8 nets:
// - The integer dot products are exact everywhere. AVX2's vpmaddubsw adds
//   pairs of uint8 x int8 products with int16 saturation, which would differ
//   from VNNI's vpdpbusd (no intermediate saturation) for activations up to
//   255; but the activations are quantized to [0, 127] (ACT_MAX below), so a
//   pair is at most 2 * 127 * 128 = 32512 in magnitude and never saturates.
//   The int32 sums cannot overflow (2*L1 * 127 * 128 < 2^25), so every order
//   of adding them gives the same total.
// - The float steps between layers are the same operations in the same order
//   at every level. The one place compilers differ is the bias add,
//   b + d * s: GCC and Clang contract it into one FMA (one rounding) wherever
//   the target has FMA, so builds without FMA rounded it twice. The dispatch
//   build makes it explicit at every level, one rounding (bias_add below):
//   hardware FMA from AVX2 up, an exact software version at the base level,
//   which matches the contracted builds (OpenBench's included).
// - The accumulator lanes add and subtract rows with wrapping int16 arithmetic
//   at every level, the SIMD ones and the portable scalar loops (ARM, and x86
//   builds without AVX2 when LUMINEX_DISPATCH is off) alike, so the lanes are
//   the same everywhere bit for bit, and exact whenever the sum fits int16
//   ("accumulator lanes" below). They used to saturate after every row on
//   x86 and clamp once on ARM, which differed wherever a lane saturated (the
//   two digests nnuecheck showed).
// Float (LNN1) nets sum in a different order per level and are not exact
// across levels.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#define LK_X86 1
#endif
#if defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#endif

// ---- the level, from what the compiler targets ----
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
#define LK_AVX512 1            // AVX-512 accumulator and activation kernels
#endif
#if defined(LK_AVX512) && defined(__AVX512VNNI__)
#define LK_AVX512VNNI 1        // VPDPBUSD dense layers on zmm
#endif
#if defined(__AVX2__)
#define LK_AVX2 1              // AVX2 kernels (every x86-64-v3 CPU, which includes FMA)
#endif
#if defined(LK_AVX2) && defined(__AVXVNNI__) && !defined(LK_AVX512VNNI)
#define LK_AVXVNNI 1           // AVX2 dense layers with VEX VPDPBUSD
#endif
#if defined(__ARM_FEATURE_DOTPROD)
#define LK_DOTPROD 1
#endif
#if defined(LK_AVX512VNNI)
#define LK_NAME "avx512vnni"
#define LK_ID 0
#elif defined(LK_AVX512)
#define LK_NAME "avx512"
#define LK_ID 1
#elif defined(LK_AVXVNNI)
#define LK_NAME "avxvnni"
#define LK_ID 2
#elif defined(LK_AVX2)
#define LK_NAME "avx2"
#define LK_ID 3
#elif defined(LK_DOTPROD)
#define LK_NAME "dotprod"
#define LK_ID 5
#else
#define LK_NAME "base"
#define LK_ID 4
#endif
#if defined(LK_EXPECT) && LK_EXPECT != LK_ID
#error "this kernel file was compiled without its level's instruction-set flags (see LUMINEX_DISPATCH in CMakeLists.txt)"
#endif
#if defined(LK_NAMESPACE)
#define LK_DISPATCH 1          // a per-level file of the universal build
#else
#define LK_NAMESPACE k_native
#endif
#if defined(LK_DISPATCH) && defined(LK_X86) && !defined(LK_AVX2)
#define LK_SSE2_LANES 1        // the universal build's base level: SSE2 lane kernels
#endif

namespace luminex::nnue::LK_NAMESPACE {
namespace {

constexpr float FT_WINV = 1.0f / NNUE_FT_SCALE;
// The largest quantized activation. vpmaddubsw is exact only while two
// products fit int16: 2 * ACT_MAX * 128 <= 32767.
constexpr int ACT_MAX = 127;
static_assert(2 * ACT_MAX * 128 <= 32767, "AVX2's vpmaddubsw would saturate and differ from VNNI");

inline float clip01(float x) { float c = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); return c * c; }

// A quantized activation: x in [0, 127] rounded to the nearest integer, ties
// to even (the default rounding mode), as cvtps rounds on the SIMD levels.
inline uint8_t round_u8(float x) {
#if defined(LK_DISPATCH) && defined(LK_X86)
    return uint8_t(_mm_cvtss_si32(_mm_set_ss(x)));   // the instruction lrintf is, without the call
#else
    return uint8_t(std::lrintf(x));
#endif
}

// ---- the bias add: b + d * s ----
#if defined(LK_DISPATCH) && defined(LK_AVX2)
// One rounding, by the FMA every level from AVX2 up has.
inline float bias_add(float b, int32_t d, float s) { return __builtin_fmaf(float(d), s, b); }
#elif defined(LK_DISPATCH)
// One rounding without FMA hardware, the same as an FMA's. The product of the
// two 24-bit significands is exact in double, so p + z rounds once to double
// and then to float; those two roundings give the FMA's result except when
// the double lands exactly halfway between two floats, the one case where the
// double's own rounding decides the float's. There the exact error of the
// double sum (TwoSum) tells which side the true sum is on, and the double
// steps one unit toward it. Exact for finite inputs whose sum is zero or a
// normal float, which every bias add of a network is.
inline float bias_add(float b, int32_t d, float s) {
    const double p = double(float(d)) * double(s);
    const double z = double(b);
    double r = p + z;
    uint64_t u;
    std::memcpy(&u, &r, sizeof u);
    if ((u & 0x1FFFFFFFull) == 0x10000000ull) {   // halfway: 1 then 28 zeros below float precision
        const double t = r - p;
        const double err = (p - (r - t)) + (z - t);
        if (err != 0.0) {
            u += (err > 0.0) == (r > 0.0) ? uint64_t(1) : ~uint64_t(0);
            std::memcpy(&r, &u, sizeof r);
        }
    }
    return float(r);
}
#else
// LUMINEX_DISPATCH off: as the code always was, rounded as the compiler
// contracts it (one FMA wherever the target has FMA, under GCC's and Clang's
// default contraction).
inline float bias_add(float b, int32_t d, float s) { return b + float(d) * s; }
#endif

#if defined(LK_AVX2)
// Horizontal sum of 8 floats -> scalar.
inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
// SCReLU on 8 floats: clamp to [0,1], then square.
inline __m256 screlu8(__m256 x, __m256 z, __m256 one) {
    __m256 c = _mm256_min_ps(_mm256_max_ps(x, z), one);
    return _mm256_mul_ps(c, c);
}
// Quantize 8 floats ([0,1]) -> 8 uint8 ([0,127]), stored contiguously at dst.
inline void quant8(const float* h, uint8_t* dst, __m256 scale127) {
    __m256i i32 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(h), scale127));
    __m128i lo = _mm256_castsi256_si128(i32);
    __m128i hi = _mm256_extracti128_si256(i32, 1);
    __m128i i16 = _mm_packs_epi32(lo, hi);        // 8 int16 (in-order)
    __m128i i8 = _mm_packus_epi16(i16, i16);      // 16 uint8 (low 8 = ours)
    _mm_storel_epi64((__m128i*)dst, i8);          // store low 8 bytes
}
// 32 uint8 x int8 products, summed in fours into 8 int32 lanes and added to acc.
inline __m256i dpbusd256(__m256i acc, __m256i a, __m256i w) {
#if defined(LK_AVXVNNI)
    return _mm256_dpbusd_avx_epi32(acc, a, w);    // VEX VPDPBUSD: one instruction
#else
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i p16 = _mm256_maddubs_epi16(a, w);     // a(uint8) x w(int8) -> 16 int16, exact (see ACT_MAX)
    return _mm256_add_epi32(acc, _mm256_madd_epi16(p16, ones));
#endif
}
// int8 dot product over n bytes: sum(w_signed[i] * a_unsigned[i]) -> int32. Handles
// any n (SIMD for multiples of 32, scalar tail for the remainder, e.g. L2=16 inputs).
inline int32_t dot_i8(const int8_t* w, const uint8_t* a, int n) {
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    for (; i + 32 <= n; i += 32)
        acc = dpbusd256(acc, _mm256_loadu_si256((const __m256i*)(a + i)), _mm256_loadu_si256((const __m256i*)(w + i)));
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i s = _mm_hadd_epi32(_mm_add_epi32(lo, hi), _mm_setzero_si128());
    int32_t sum = _mm_cvtsi128_si32(_mm_hadd_epi32(s, s));
    for (; i < n; ++i) sum += (int32_t)w[i] * (int32_t)a[i];
    return sum;
}
#endif
#if defined(LK_AVX512VNNI)
// VPDPBUSD: 64 uint8 × int8 → 16 int32 accumulate in ONE instruction.
// ~6x throughput vs AVX2 (2x width + 3x fewer instructions per MAC chunk).
inline int32_t dot_i8_vnni(const int8_t* w, const uint8_t* a, int n) {
    // 4 independent accumulator chains: a single chain serializes on vpdpbusd
    // latency (~4-5 cyc), so a 1024-byte L2 dot alone costs ~70+ cycles of pure
    // stalls. Integer adds are associative -> the result is bit-identical.
    __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512(),
            acc2 = _mm512_setzero_si512(), acc3 = _mm512_setzero_si512();
    int i = 0;
    for (; i + 256 <= n; i += 256) {
        acc0 = _mm512_dpbusd_epi32(acc0, _mm512_loadu_si512((const __m512i*)(a + i)),       _mm512_loadu_si512((const __m512i*)(w + i)));
        acc1 = _mm512_dpbusd_epi32(acc1, _mm512_loadu_si512((const __m512i*)(a + i + 64)),  _mm512_loadu_si512((const __m512i*)(w + i + 64)));
        acc2 = _mm512_dpbusd_epi32(acc2, _mm512_loadu_si512((const __m512i*)(a + i + 128)), _mm512_loadu_si512((const __m512i*)(w + i + 128)));
        acc3 = _mm512_dpbusd_epi32(acc3, _mm512_loadu_si512((const __m512i*)(a + i + 192)), _mm512_loadu_si512((const __m512i*)(w + i + 192)));
    }
    for (; i + 64 <= n; i += 64)
        acc0 = _mm512_dpbusd_epi32(acc0, _mm512_loadu_si512((const __m512i*)(a + i)), _mm512_loadu_si512((const __m512i*)(w + i)));
    __m512i acc = _mm512_add_epi32(_mm512_add_epi32(acc0, acc1), _mm512_add_epi32(acc2, acc3));
    int32_t sum = _mm512_reduce_add_epi32(acc);
    for (; i < n; ++i) sum += (int32_t)w[i] * (int32_t)a[i];
    return sum;
}
// Small-n dot (n <= 64): one VPDPBUSD over zero-extended inputs, then reduce.
// For L3 (n=L2=16) and out (n=L3=32) this replaces ~512 scalar int muls with a
// single VNNI instruction — the dominant eval cost after L2.
inline int32_t dot_i8_vnni_small(const int8_t* w, const uint8_t* a, int n) {
    __m512i av, wv;
    if (n <= 16) {
        av = _mm512_zextsi128_si512(_mm_loadu_si128((const __m128i*)a));
        wv = _mm512_zextsi128_si512(_mm_loadu_si128((const __m128i*)w));
    } else if (n <= 32) {
        av = _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*)a));
        wv = _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*)w));
    } else {  // 32 < n <= 64: load full 64 bytes (caller guarantees bounds).
        av = _mm512_loadu_si512((const void*)a);
        wv = _mm512_loadu_si512((const void*)w);
    }
    __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), av, wv);
    return _mm512_reduce_add_epi32(acc);
}
#endif

#if defined(LK_SSE2_LANES)
// 16 uint8 x int8 products, summed in fours into 4 int32 lanes and added to acc.
inline __m128i dpbusd128(__m128i acc, __m128i a, __m128i w) {
#if defined(__SSSE3__)
    __m128i p16 = _mm_maddubs_epi16(a, w);        // exact, as dpbusd256's (see ACT_MAX)
    return _mm_add_epi32(acc, _mm_madd_epi16(p16, _mm_set1_epi16(1)));
#else
    // SSE2: widen both to int16 (zero- and sign-extended), then pmaddwd
    const __m128i zero = _mm_setzero_si128();
    const __m128i alo = _mm_unpacklo_epi8(a, zero), ahi = _mm_unpackhi_epi8(a, zero);
    const __m128i wlo = _mm_srai_epi16(_mm_unpacklo_epi8(w, w), 8), whi = _mm_srai_epi16(_mm_unpackhi_epi8(w, w), 8);
    return _mm_add_epi32(acc, _mm_add_epi32(_mm_madd_epi16(alo, wlo), _mm_madd_epi16(ahi, whi)));
#endif
}
inline int32_t hsum128(__m128i v) {
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0x4E));   // + the other 64-bit half
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0xB1));   // + the neighbouring lane
    return _mm_cvtsi128_si32(v);
}
// int8 dot product over n bytes, 16 at a time, a scalar tail for the rest.
inline int32_t dot_i8_128(const int8_t* w, const uint8_t* a, int n) {
    __m128i acc = _mm_setzero_si128();
    int i = 0;
    for (; i + 16 <= n; i += 16)
        acc = dpbusd128(acc, _mm_loadu_si128((const __m128i*)(a + i)), _mm_loadu_si128((const __m128i*)(w + i)));
    int32_t sum = hsum128(acc);
    for (; i < n; ++i) sum += (int32_t)w[i] * (int32_t)a[i];
    return sum;
}
#endif

// ---- accumulator lanes ----
// The lanes are kept modulo 2^16: rows are added and subtracted with wrapping
// int16 arithmetic at every level, so a lane equals the exact sum of the bias
// and its rows whenever that sum fits int16 (+-16.0 on the x2048 grid), in
// whatever order the rows came and however far a partial sum strayed on the
// way. Saturating after every row, as the lanes did before, clipped a partial
// sum that left int16 even when the final sum fit, and the clipped lane then
// stayed wrong for the rest of the line. A sum that does not fit wraps, and the
// lane is exact again as soon as the sum is back in range (none of the shipped
// or gate nets has a lane past 12.9). The cached refresh's int32 entries are
// exact and saturate only on their copy into the lanes (store_saturated).

// Add one feature row to a perspective's lanes (wrapping int16).
inline void add_row(int16_t* acc, const int16_t* w, int n) {
#if defined(LK_AVX512)
    for (int l = 0; l < n; l += 32)
        _mm512_storeu_si512((__m512i*)(acc + l), _mm512_add_epi16(
            _mm512_loadu_si512((const __m512i*)(acc + l)),
            _mm512_loadu_si512((const __m512i*)(w + l))));
#elif defined(LK_AVX2)
    for (int l = 0; l < n; l += 16)
        _mm256_storeu_si256((__m256i*)(acc + l), _mm256_add_epi16(
            _mm256_loadu_si256((const __m256i*)(acc + l)),
            _mm256_loadu_si256((const __m256i*)(w + l))));
#elif defined(LK_SSE2_LANES)
    for (int l = 0; l < n; l += 8)
        _mm_storeu_si128((__m128i*)(acc + l), _mm_add_epi16(
            _mm_loadu_si128((const __m128i*)(acc + l)),
            _mm_loadu_si128((const __m128i*)(w + l))));
#else
    // int arithmetic, then the conversion to int16_t keeps it modulo 2^16 (C++20)
    for (int l = 0; l < n; ++l) acc[l] = int16_t(int(acc[l]) + int(w[l]));
#endif
}

// One perspective of a child accumulator, written in a single pass from the
// parent's lanes: +add0 -sub0, then -sub1 (S1), then +add1 (A1), all wrapping
// int16, so every level (and the portable scalar build) gives the same lanes
// bit for bit, and the exact sum whenever it fits.
template <bool S1, bool A1>
inline void delta_rows(const int16_t* parent, int16_t* child, const int16_t* add0,
                       const int16_t* sub0, const int16_t* sub1, const int16_t* add1, int n) {
#if defined(LK_AVX512)
    for (int l = 0; l < n; l += 32) {
        __m512i v = _mm512_loadu_si512((const __m512i*)(parent + l));
        v = _mm512_add_epi16(v, _mm512_loadu_si512((const __m512i*)(add0 + l)));
        v = _mm512_sub_epi16(v, _mm512_loadu_si512((const __m512i*)(sub0 + l)));
        if constexpr (S1) v = _mm512_sub_epi16(v, _mm512_loadu_si512((const __m512i*)(sub1 + l)));
        if constexpr (A1) v = _mm512_add_epi16(v, _mm512_loadu_si512((const __m512i*)(add1 + l)));
        _mm512_storeu_si512((__m512i*)(child + l), v);
    }
#elif defined(LK_AVX2)
    for (int l = 0; l < n; l += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(parent + l));
        v = _mm256_add_epi16(v, _mm256_loadu_si256((const __m256i*)(add0 + l)));
        v = _mm256_sub_epi16(v, _mm256_loadu_si256((const __m256i*)(sub0 + l)));
        if constexpr (S1) v = _mm256_sub_epi16(v, _mm256_loadu_si256((const __m256i*)(sub1 + l)));
        if constexpr (A1) v = _mm256_add_epi16(v, _mm256_loadu_si256((const __m256i*)(add1 + l)));
        _mm256_storeu_si256((__m256i*)(child + l), v);
    }
#elif defined(LK_SSE2_LANES)
    for (int l = 0; l < n; l += 8) {
        __m128i v = _mm_loadu_si128((const __m128i*)(parent + l));
        v = _mm_add_epi16(v, _mm_loadu_si128((const __m128i*)(add0 + l)));
        v = _mm_sub_epi16(v, _mm_loadu_si128((const __m128i*)(sub0 + l)));
        if constexpr (S1) v = _mm_sub_epi16(v, _mm_loadu_si128((const __m128i*)(sub1 + l)));
        if constexpr (A1) v = _mm_add_epi16(v, _mm_loadu_si128((const __m128i*)(add1 + l)));
        _mm_storeu_si128((__m128i*)(child + l), v);
    }
#else
    for (int l = 0; l < n; ++l) {
        int v = int(parent[l]) + int(add0[l]) - int(sub0[l]);
        if constexpr (S1) v -= int(sub1[l]);
        if constexpr (A1) v += int(add1[l]);
        child[l] = int16_t(v);   // modulo 2^16, as the SIMD levels wrap
    }
#endif
}

// Lanes of an empty board: the FT bias, clamped to int16.
inline void init_bias(int16_t* acc, const int32_t* bias, int n) {
#if defined(LK_AVX512)
    {   // vectorized clamped bias init (was 512 scalar stores per refresh);
        // inputs pre-clamped to int16 range so signed saturation == exact
        const __m512i lo32 = _mm512_set1_epi32(-32768), hi32 = _mm512_set1_epi32(32767);
        for (int l = 0; l < n; l += 32) {
            __m512i b0 = _mm512_min_epi32(hi32, _mm512_max_epi32(lo32, _mm512_loadu_si512((const __m512i*)&bias[l])));
            __m512i b1 = _mm512_min_epi32(hi32, _mm512_max_epi32(lo32, _mm512_loadu_si512((const __m512i*)&bias[l + 16])));
            _mm256_storeu_si256((__m256i*)(acc + l),      _mm512_cvtsepi32_epi16(b0));
            _mm256_storeu_si256((__m256i*)(acc + l + 16), _mm512_cvtsepi32_epi16(b1));
        }
    }
#else
    for (int l = 0; l < n; ++l)
        acc[l] = int16_t(bias[l] < -32768 ? -32768 : (bias[l] > 32767 ? 32767 : bias[l]));
#endif
}

inline void add_row_i32(int32_t* acc, const int16_t* w, int n) {
#if defined(LK_AVX512)
    for (int l = 0; l < n; l += 16)
        _mm512_storeu_si512((__m512i*)(acc + l), _mm512_add_epi32(
            _mm512_loadu_si512((const __m512i*)(acc + l)),
            _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i*)(w + l)))));
#elif defined(LK_AVX2)
    for (int l = 0; l < n; l += 8)
        _mm256_storeu_si256((__m256i*)(acc + l), _mm256_add_epi32(
            _mm256_loadu_si256((const __m256i*)(acc + l)),
            _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(w + l)))));
#else
    for (int l = 0; l < n; ++l) acc[l] += w[l];
#endif
}
inline void sub_row_i32(int32_t* acc, const int16_t* w, int n) {
#if defined(LK_AVX512)
    for (int l = 0; l < n; l += 16)
        _mm512_storeu_si512((__m512i*)(acc + l), _mm512_sub_epi32(
            _mm512_loadu_si512((const __m512i*)(acc + l)),
            _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i*)(w + l)))));
#elif defined(LK_AVX2)
    for (int l = 0; l < n; l += 8)
        _mm256_storeu_si256((__m256i*)(acc + l), _mm256_sub_epi32(
            _mm256_loadu_si256((const __m256i*)(acc + l)),
            _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(w + l)))));
#else
    for (int l = 0; l < n; ++l) acc[l] -= w[l];
#endif
}
// int32 lanes -> int16 lanes with signed saturation.
inline void store_saturated(int16_t* dst, const int32_t* src, int n) {
#if defined(LK_AVX512)
    for (int l = 0; l < n; l += 16)
        _mm256_storeu_si256((__m256i*)(dst + l),
            _mm512_cvtsepi32_epi16(_mm512_loadu_si512((const __m512i*)(src + l))));
#elif defined(LK_AVX2)
    for (int l = 0; l < n; l += 16) {
        __m256i lo = _mm256_loadu_si256((const __m256i*)(src + l));
        __m256i hi = _mm256_loadu_si256((const __m256i*)(src + l + 8));
        // packs works per 128-bit half: reorder the 64-bit quarters back
        __m256i p = _mm256_permute4x64_epi64(_mm256_packs_epi32(lo, hi), 0xD8);
        _mm256_storeu_si256((__m256i*)(dst + l), p);
    }
#else
    for (int l = 0; l < n; ++l) dst[l] = int16_t(src[l] < -32768 ? -32768 : (src[l] > 32767 ? 32767 : src[l]));
#endif
}

// ---- the table's entries ----
void refresh_lanes(int16_t* lanes, const int32_t* bias, const int16_t* const* rows, int nrows, int n) {
    init_bias(lanes, bias, n);
    for (int i = 0; i < nrows; ++i) add_row(lanes, rows[i], n);
}

void update_lanes(const DeltaRows* jobs, int njobs, int n) {
    for (int j = 0; j < njobs; ++j) {
        const DeltaRows& r = jobs[j];
        if (r.add1)
            delta_rows<true, true>(r.parent, r.child, r.add0, r.sub0, r.sub1, r.add1, n);
        else if (r.sub1)
            delta_rows<true, false>(r.parent, r.child, r.add0, r.sub0, r.sub1, nullptr, n);
        else
            delta_rows<false, false>(r.parent, r.child, r.add0, r.sub0, nullptr, nullptr, n);
    }
}

// The int32 sums are exact, so the order of the rows does not matter.
void refresh_cached(int32_t* entry, const int16_t* const* sub, int nsub,
                    const int16_t* const* add, int nadd, int16_t* lanes, int n) {
    for (int i = 0; i < nsub; ++i) sub_row_i32(entry, sub[i], n);
    for (int i = 0; i < nadd; ++i) add_row_i32(entry, add[i], n);
    store_saturated(lanes, entry, n);
}

// gen768t (LNT8 nets): child = parent + the add rows - the sub rows on the T relation lanes, the int8 rows
// widened to int16. Plain int16 adds: at most 248 rows of |q| <= 127 are ever active, so no lane overflows and
// the order of the rows does not matter.
void rel_rows(const int16_t* parent, int16_t* child, const int8_t* const* add, int na,
              const int8_t* const* sub, int ns, int T) {
#if defined(LK_AVX512)
    for (int l = 0; l < T; l += 32) {
        __m512i v = _mm512_loadu_si512((const void*)(parent + l));
        for (int i = 0; i < na; ++i) v = _mm512_add_epi16(v, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(add[i] + l))));
        for (int i = 0; i < ns; ++i) v = _mm512_sub_epi16(v, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(sub[i] + l))));
        _mm512_storeu_si512((void*)(child + l), v);
    }
#elif defined(LK_AVX2)
    for (int l = 0; l < T; l += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(parent + l));
        for (int i = 0; i < na; ++i) v = _mm256_add_epi16(v, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(add[i] + l))));
        for (int i = 0; i < ns; ++i) v = _mm256_sub_epi16(v, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(sub[i] + l))));
        _mm256_storeu_si256((__m256i*)(child + l), v);
    }
#elif defined(LK_SSE2_LANES)
    // SSE2 has no sign-extending byte load: each byte doubled into a word and shifted back down
    for (int l = 0; l < T; l += 8) {
        __m128i v = _mm_loadu_si128((const __m128i*)(parent + l));
        for (int i = 0; i < na; ++i) {
            const __m128i b = _mm_loadl_epi64((const __m128i*)(add[i] + l));
            v = _mm_add_epi16(v, _mm_srai_epi16(_mm_unpacklo_epi8(b, b), 8));
        }
        for (int i = 0; i < ns; ++i) {
            const __m128i b = _mm_loadl_epi64((const __m128i*)(sub[i] + l));
            v = _mm_sub_epi16(v, _mm_srai_epi16(_mm_unpacklo_epi8(b, b), 8));
        }
        _mm_storeu_si128((__m128i*)(child + l), v);
    }
#elif defined(LK_DOTPROD)
    for (int l = 0; l < T; l += 8) {
        int16x8_t v = vld1q_s16(parent + l);
        for (int i = 0; i < na; ++i) v = vaddq_s16(v, vmovl_s8(vld1_s8(add[i] + l)));
        for (int i = 0; i < ns; ++i) v = vsubq_s16(v, vmovl_s8(vld1_s8(sub[i] + l)));
        vst1q_s16(child + l, v);
    }
#else
    for (int l = 0; l < T; ++l) {
        int v = parent[l];
        for (int i = 0; i < na; ++i) v += add[i][l];
        for (int i = 0; i < ns; ++i) v -= sub[i][l];
        child[l] = int16_t(v);
    }
#endif
}

// gen768t (LNT8 nets): out = the saturating int16 sum psq + clamp(rel, -L, L) * 2^rs, lane by lane, for
// lanes 0..T-1 (RelLanes). Integer steps only, so every level computes the same lanes; rs <= 6 keeps the
// shifted term inside int16.
void combine_rel(int16_t* out, const int16_t* psq, const int16_t* rel, int T, int rs) {
#if defined(LK_AVX2)
    const __m256i lo = _mm256_set1_epi16(int16_t(-NNUE_REL_L)), hi = _mm256_set1_epi16(int16_t(NNUE_REL_L));
    const __m128i sh = _mm_cvtsi32_si128(rs);
    for (int l = 0; l < T; l += 16) {
        const __m256i r = _mm256_sll_epi16(_mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256((const __m256i*)(rel + l)), lo), hi), sh);
        _mm256_storeu_si256((__m256i*)(out + l), _mm256_adds_epi16(_mm256_loadu_si256((const __m256i*)(psq + l)), r));
    }
#elif defined(LK_SSE2_LANES)
    const __m128i lo = _mm_set1_epi16(int16_t(-NNUE_REL_L)), hi = _mm_set1_epi16(int16_t(NNUE_REL_L));
    const __m128i sh = _mm_cvtsi32_si128(rs);
    for (int l = 0; l < T; l += 8) {
        const __m128i r = _mm_sll_epi16(_mm_min_epi16(_mm_max_epi16(_mm_loadu_si128((const __m128i*)(rel + l)), lo), hi), sh);
        _mm_storeu_si128((__m128i*)(out + l), _mm_adds_epi16(_mm_loadu_si128((const __m128i*)(psq + l)), r));
    }
#elif defined(LK_DOTPROD)
    const int16x8_t lo = vdupq_n_s16(int16_t(-NNUE_REL_L)), hi = vdupq_n_s16(int16_t(NNUE_REL_L));
    const int16x8_t sh = vdupq_n_s16(int16_t(rs));
    for (int l = 0; l < T; l += 8) {
        const int16x8_t r = vshlq_s16(vminq_s16(vmaxq_s16(vld1q_s16(rel + l), lo), hi), sh);
        vst1q_s16(out + l, vqaddq_s16(vld1q_s16(psq + l), r));
    }
#else
    for (int l = 0; l < T; ++l) {
        const int r = std::clamp<int>(rel[l], -NNUE_REL_L, NNUE_REL_L) * (1 << rs);
        out[l] = int16_t(std::clamp(int(psq[l]) + r, -32768, 32767));
    }
#endif
}

// REL (an LNT8 net): lanes 0..T-1 with the relation term, read from rel_comb instead of the accumulator. T is
// a multiple of 64, so every block of lanes each path below reads lies on one side of it. Without REL the
// lanes are read where they are, so every other format runs the code it always ran.
template <bool REL>
float forward_int8_impl(const int16_t* acc_stm, const int16_t* acc_nstm, const RelLanes* rel, const TailView& t,
                        int L1, int L2, int L3, long long* act_nonzero) {
    [[maybe_unused]] alignas(64) int16_t rel_comb[REL ? 2 : 1][REL ? NNUE_L1_MAX : 1];
    [[maybe_unused]] int relT = 0;
    if constexpr (REL) {
        relT = rel->T;
        combine_rel(rel_comb[0], acc_stm, rel->stm, relT, rel->rs);
        combine_rel(rel_comb[1], acc_nstm, rel->nstm, relT, rel->rs);
    } else {
        (void)rel;
    }
    auto src = [&](int side, int l) -> const int16_t* {
        if constexpr (REL) {
            if (l < relT) return rel_comb[side] + l;
        }
        return (side == 0 ? acc_stm : acc_nstm) + l;
    };
#if defined(LK_AVX2)
    // ---- FUSED SCReLU+quant8 (no intermediate h[] — saves 8KB L1 traffic/eval) ----
    uint8_t h_i8[2 * NNUE_L1_MAX];
    const __m256 sc127 = _mm256_set1_ps(127.0f);  // shared for quant8 (L3/out)
#if defined(LK_AVX512)
    // AVX-512, 64 lanes per pass. Clamping to the SCReLU window
    // [0, NNUE_FT_SCALE] in int16 and then rounding fl(a*a * 127/SCALE^2)
    // gives the same bytes as the float clamp-square-scale below: a*a is at
    // most 2^22, exact in a float, and scaling by a power of two never
    // rounds, so both round the same product. Packing 4 vectors at a time
    // replaces two narrowing converts per 16 lanes.
    if (L1 % 64 == 0) {
        const __m512i lo16 = _mm512_setzero_si512(), hi16 = _mm512_set1_epi16(int16_t(NNUE_FT_SCALE));
        const __m512 k = _mm512_set1_ps(127.0f / (NNUE_FT_SCALE * NNUE_FT_SCALE));   // 127 * 2^-22, exact
        // packs/packus work inside 128-bit lanes; this dword permute (a 4x4
        // transpose of 4-byte groups) restores input order
        const __m512i order = _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
        auto act16 = [&](__m256i v) {   // 16 clamped int16 -> 16 int32 in [0, 127]
            const __m512 f = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(v));
            return _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_mul_ps(f, f), k));
        };
        for (int side = 0; side < 2; ++side) {
            uint8_t* out = h_i8 + side * L1;
            for (int l = 0; l < L1; l += 64) {
                const int16_t* in = src(side, l);
                const __m512i x0 = _mm512_min_epi16(_mm512_max_epi16(_mm512_loadu_si512(in), lo16), hi16);
                const __m512i x1 = _mm512_min_epi16(_mm512_max_epi16(_mm512_loadu_si512(in + 32), lo16), hi16);
                const __m512i q0 = act16(_mm512_castsi512_si256(x0)), q1 = act16(_mm512_extracti64x4_epi64(x0, 1));
                const __m512i q2 = act16(_mm512_castsi512_si256(x1)), q3 = act16(_mm512_extracti64x4_epi64(x1, 1));
                const __m512i b = _mm512_packus_epi16(_mm512_packs_epi32(q0, q1), _mm512_packs_epi32(q2, q3));
                _mm512_storeu_si512(out + l, _mm512_permutexvar_epi32(order, b));
            }
        }
    } else {
    const __m512 z16 = _mm512_setzero_ps(), one16 = _mm512_set1_ps(1.0f), sc16 = _mm512_set1_ps(127.0f);
    const __m512 winv16 = _mm512_set1_ps(FT_WINV);
    for (int l = 0; l < L1; l += 16) {
        __m512 s = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i*)src(0, l)))), winv16);
        __m512 c = _mm512_min_ps(_mm512_max_ps(s, z16), one16);
        __m512 sq = _mm512_mul_ps(c, c);
        __m512i i32 = _mm512_cvtps_epi32(_mm512_mul_ps(sq, sc16));
        // Pack 16 int32 → 16 uint8: int32→int16→uint8
        __m256i i16 = _mm512_cvtepi32_epi16(i32);  // AVX512F: 16 int32 → 16 int16
        __m128i i8 = _mm256_cvtepi16_epi8(i16);     // AVX512BW+VL: 16 int16 → 16 int8
        // c=clip(s,0,1) -> c² in [0,1] -> ×127 -> [0,127] exactly: no clamp needed.
        _mm_storeu_si128((__m128i*)(h_i8 + l), i8);
        // nstm
        s = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i*)src(1, l)))), winv16);
        c = _mm512_min_ps(_mm512_max_ps(s, z16), one16);
        sq = _mm512_mul_ps(c, c);
        i32 = _mm512_cvtps_epi32(_mm512_mul_ps(sq, sc16));
        i16 = _mm512_cvtepi32_epi16(i32);
        i8 = _mm256_cvtepi16_epi8(i16);
        _mm_storeu_si128((__m128i*)(h_i8 + L1 + l), i8);
    }
    }
#else
    const __m256 z = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f), winv8 = _mm256_set1_ps(FT_WINV);
    for (int l = 0; l < L1; l += 8) {
        __m256 s = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)src(0, l)))), winv8);
        __m256 c = _mm256_min_ps(_mm256_max_ps(s, z), one);
        __m256 sq = _mm256_mul_ps(c, c);
        __m256i i32 = _mm256_cvtps_epi32(_mm256_mul_ps(sq, sc127));
        __m128i lo = _mm256_castsi256_si128(i32), hi = _mm256_extracti128_si256(i32, 1);
        _mm_storel_epi64((__m128i*)(h_i8 + l), _mm_packus_epi16(_mm_packs_epi32(lo, hi), _mm_setzero_si128()));
        s = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)src(1, l)))), winv8);
        c = _mm256_min_ps(_mm256_max_ps(s, z), one);
        sq = _mm256_mul_ps(c, c);
        i32 = _mm256_cvtps_epi32(_mm256_mul_ps(sq, sc127));
        lo = _mm256_castsi256_si128(i32); hi = _mm256_extracti128_si256(i32, 1);
        _mm_storel_epi64((__m128i*)(h_i8 + L1 + l), _mm_packus_epi16(_mm_packs_epi32(lo, hi), _mm_setzero_si128()));
    }
#endif
    if (act_nonzero) { long long nz = 0; for (int i = 0; i < 2 * L1; ++i) nz += (h_i8[i] != 0); *act_nonzero += nz; }
    // Each output's dot is scaled by its own factor 1 / (s x 127), computed once at load (TailView).
    float h2[NNUE_DENSE_MAX];
#if defined(LK_AVX512VNNI)
    if (t.l2_by_input) {
        // One VPDPBUSD per 4 inputs adds their products to all 16 output sums
        // (lane o = output o), so no per-output horizontal sum is needed.
        // Integer adds associate: each sum equals the per-output dot exactly.
        const int n4 = 2 * L1 / 4;
        const int8_t* w = t.l2_w;
        auto chunk = [&](int c) {
            int32_t v; std::memcpy(&v, h_i8 + 4 * c, 4);
            return _mm512_set1_epi32(v);
        };
        __m512i s0 = _mm512_setzero_si512(), s1 = s0, s2 = s0, s3 = s0;
        int c = 0;
        for (; c + 4 <= n4; c += 4) {
            s0 = _mm512_dpbusd_epi32(s0, chunk(c + 0), _mm512_loadu_si512(w + 64 * (c + 0)));
            s1 = _mm512_dpbusd_epi32(s1, chunk(c + 1), _mm512_loadu_si512(w + 64 * (c + 1)));
            s2 = _mm512_dpbusd_epi32(s2, chunk(c + 2), _mm512_loadu_si512(w + 64 * (c + 2)));
            s3 = _mm512_dpbusd_epi32(s3, chunk(c + 3), _mm512_loadu_si512(w + 64 * (c + 3)));
        }
        for (; c < n4; ++c)
            s0 = _mm512_dpbusd_epi32(s0, chunk(c), _mm512_loadu_si512(w + 64 * c));
        alignas(64) int32_t d[16];
        _mm512_store_si512(d, _mm512_add_epi32(_mm512_add_epi32(s0, s1), _mm512_add_epi32(s2, s3)));
        for (int o = 0; o < 16; ++o) h2[o] = clip01(bias_add(t.l2_b[o], d[o], t.ics2[o]));
    } else
    // VPDPBUSD: 64 MACs/instruction. Per-output dot (output-major weights, no reorder).
    for (int o = 0; o < L2; ++o)
        h2[o] = clip01(bias_add(t.l2_b[o], dot_i8_vnni(&t.l2_w[static_cast<size_t>(o) * 2 * L1], h_i8, 2 * L1), t.ics2[o]));
#else
    // AVX2 SF-style chunk L2 (chunk-major weights, reordered at load); with
    // AVX-VNNI each chunk's dot is one VPDPBUSD.
    const int total_in = 2 * L1;
    const int num_chunks = total_in / 32;
    for (int pass = 0; pass < (L2 + 7) / 8; ++pass) {
        int base_o = pass * 8;
        int n_out = L2 - base_o < 8 ? L2 - base_o : 8;
        __m256i acc8[8];
        for (int i = 0; i < 8; ++i) acc8[i] = _mm256_setzero_si256();
        for (int c = 0; c < num_chunks; ++c) {
            __m256i hv = _mm256_loadu_si256((const __m256i*)(h_i8 + c * 32));
            const int8_t* wp = &t.l2_w[(static_cast<size_t>(c) * L2 + base_o) * 32];
            for (int o = 0; o < n_out; ++o)
                acc8[o] = dpbusd256(acc8[o], hv, _mm256_loadu_si256((const __m256i*)(wp + o * 32)));
        }
        for (int o = 0; o < n_out; ++o) {
            __m128i lo = _mm256_castsi256_si128(acc8[o]);
            __m128i hi = _mm256_extracti128_si256(acc8[o], 1);
            __m128i sv = _mm_hadd_epi32(_mm_add_epi32(lo, hi), _mm_setzero_si128());
            int32_t d = _mm_cvtsi128_si32(_mm_hadd_epi32(sv, sv));
            h2[base_o + o] = clip01(bias_add(t.l2_b[base_o + o], d, t.ics2[base_o + o]));
        }
    }
#endif // AVX2 SF-L2 vs AVX512 VNNI L2
    // L3/out (shared — uses VNNI when available, AVX2 otherwise)
    uint8_t h2_i8[NNUE_DENSE_MAX];
    for (int l = 0; l < L2; l += 8) quant8(h2 + l, h2_i8 + l, sc127);
    float h3[NNUE_DENSE_MAX];
    uint8_t h3_i8[NNUE_DENSE_MAX];
#if defined(LK_AVX512VNNI)
    if (t.l3_batched) {
        // All L3 dots via 8 dpbusd + lane-group reduction (bit-exact: integer
        // adds associate). a512: every 4-lane group holds all 16 h2 bytes.
        const __m128i h2_128 = _mm_loadu_si128((const __m128i*)h2_i8);
        const __m512i av = _mm512_broadcast_i32x4(h2_128);   // replicate 16B into each lane group
        const __m512i pairswap = _mm512_setr_epi32(1,0,3,2, 5,4,7,6, 9,8,11,10, 13,12,15,14);
        const __m512i groupswap = _mm512_setr_epi32(2,3,0,1, 6,7,4,5, 10,11,8,9, 14,15,12,13);
        // Butterfly leaves each group's sum in ALL its lanes; the four output
        // dots sit at lanes {0,4,8,12} — gather them into 0..3 before extract.
        const __m512i pick0408 = _mm512_setr_epi32(0,4,8,12, 0,4,8,12, 0,4,8,12, 0,4,8,12);
        int o = 0;
        for (int v = 0; v < L3 / 4; ++v, o += 4) {
            __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), av,
                            _mm512_loadu_si512((const __m512i*)&t.l3_w[v * 64]));
            __m512i t1 = _mm512_add_epi32(acc, _mm512_permutexvar_epi32(pairswap, acc));
            __m512i t2 = _mm512_add_epi32(t1, _mm512_permutexvar_epi32(groupswap, t1));
            // t2 lane 4k = dot for output (v*4 + k)
            __m128i sums = _mm512_castsi512_si128(_mm512_permutexvar_epi32(pick0408, t2));
            h3[o+0] = clip01(bias_add(t.l3_b[o+0], _mm_extract_epi32(sums, 0), t.ics3[o+0]));
            h3[o+1] = clip01(bias_add(t.l3_b[o+1], _mm_extract_epi32(sums, 1), t.ics3[o+1]));
            h3[o+2] = clip01(bias_add(t.l3_b[o+2], _mm_extract_epi32(sums, 2), t.ics3[o+2]));
            h3[o+3] = clip01(bias_add(t.l3_b[o+3], _mm_extract_epi32(sums, 3), t.ics3[o+3]));
        }
    } else
    for (int o = 0; o < L3; ++o)
        h3[o] = clip01(bias_add(t.l3_b[o], dot_i8_vnni_small(&t.l3_w[static_cast<size_t>(o) * L2], h2_i8, L2), t.ics3[o]));
    for (int l = 0; l < L3; l += 8) quant8(h3 + l, h3_i8 + l, sc127);
    return bias_add(t.out_b, dot_i8_vnni_small(t.out_w, h3_i8, L3), t.icso);
#else
    for (int o = 0; o < L3; ++o)
        h3[o] = clip01(bias_add(t.l3_b[o], dot_i8(&t.l3_w[static_cast<size_t>(o) * L2], h2_i8, L2), t.ics3[o]));
    for (int l = 0; l < L3; l += 8) quant8(h3 + l, h3_i8 + l, sc127);
    return bias_add(t.out_b, dot_i8(t.out_w, h3_i8, L3), t.icso);
#endif
#elif defined(LK_SSE2_LANES)
    // The universal build's base level: the portable path's operations, four
    // lanes at a time in SSE2, and the dense dots in 16-byte blocks (SSSE3's
    // pmaddubsw where the base targets x86-64-v2, SSE2 widening otherwise).
    uint8_t h_i8[2 * NNUE_L1_MAX];
    {
        const __m128 z = _mm_setzero_ps(), one = _mm_set1_ps(1.0f), winv = _mm_set1_ps(FT_WINV);
        const __m128 sc = _mm_set1_ps(127.0f);
        for (int side = 0; side < 2; ++side) {
            uint8_t* out = h_i8 + side * L1;
            for (int l = 0; l < L1; l += 8) {
                const __m128i v = _mm_loadu_si128((const __m128i*)src(side, l));
                // int16 -> int32 sign-extended, -> float, x 2^-13 (exact), clamp to [0, 1],
                // square, x 127, round to nearest even: the scalar path's steps per lane
                __m128 flo = _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v, v), 16)), winv);
                __m128 fhi = _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v, v), 16)), winv);
                flo = _mm_min_ps(_mm_max_ps(flo, z), one);
                fhi = _mm_min_ps(_mm_max_ps(fhi, z), one);
                const __m128i qlo = _mm_cvtps_epi32(_mm_mul_ps(_mm_mul_ps(flo, flo), sc));
                const __m128i qhi = _mm_cvtps_epi32(_mm_mul_ps(_mm_mul_ps(fhi, fhi), sc));
                _mm_storel_epi64((__m128i*)(out + l), _mm_packus_epi16(_mm_packs_epi32(qlo, qhi), _mm_setzero_si128()));
            }
        }
    }
    if (act_nonzero) { long long nz = 0; for (int i = 0; i < 2 * L1; ++i) nz += (h_i8[i] != 0); *act_nonzero += nz; }
    const int total_in = 2 * L1;
    float h2[NNUE_DENSE_MAX];
    // L2, output-major weights, four outputs per pass sharing each block of
    // activations (L2 is a multiple of 8, 2*L1 of 64).
    for (int o = 0; o < L2; o += 4) {
        const int8_t* w = &t.l2_w[static_cast<size_t>(o) * total_in];
        __m128i a0 = _mm_setzero_si128(), a1 = a0, a2 = a0, a3 = a0;
        for (int i = 0; i < total_in; i += 16) {
            const __m128i a = _mm_loadu_si128((const __m128i*)(h_i8 + i));
            a0 = dpbusd128(a0, a, _mm_loadu_si128((const __m128i*)(w + i)));
            a1 = dpbusd128(a1, a, _mm_loadu_si128((const __m128i*)(w + total_in + i)));
            a2 = dpbusd128(a2, a, _mm_loadu_si128((const __m128i*)(w + 2 * total_in + i)));
            a3 = dpbusd128(a3, a, _mm_loadu_si128((const __m128i*)(w + 3 * total_in + i)));
        }
        h2[o + 0] = clip01(bias_add(t.l2_b[o + 0], hsum128(a0), t.ics2[o + 0]));
        h2[o + 1] = clip01(bias_add(t.l2_b[o + 1], hsum128(a1), t.ics2[o + 1]));
        h2[o + 2] = clip01(bias_add(t.l2_b[o + 2], hsum128(a2), t.ics2[o + 2]));
        h2[o + 3] = clip01(bias_add(t.l2_b[o + 3], hsum128(a3), t.ics2[o + 3]));
    }
    uint8_t h2_i8[NNUE_DENSE_MAX];
    for (int o = 0; o < L2; ++o) h2_i8[o] = round_u8(h2[o] * 127.0f);
    float h3[NNUE_DENSE_MAX];
    for (int o = 0; o < L3; ++o)
        h3[o] = clip01(bias_add(t.l3_b[o], dot_i8_128(&t.l3_w[static_cast<size_t>(o) * L2], h2_i8, L2), t.ics3[o]));
    uint8_t h3_i8[NNUE_DENSE_MAX];
    for (int o = 0; o < L3; ++o) h3_i8[o] = round_u8(h3[o] * 127.0f);
    return bias_add(t.out_b, dot_i8_128(t.out_w, h3_i8, L3), t.icso);
#elif defined(LK_DOTPROD)
    uint8_t h_i8[2 * NNUE_L1_MAX];
    // SCReLU + quantize: acc int16 → float → clip01 → square → ×127 → uint8
    // 8 elements per iteration via NEON
    for (int l = 0; l < L1; l += 8) {
        // stm half
        int16x8_t v = vld1q_s16(src(0, l));
        int32x4_t lo = vmovl_s16(vget_low_s16(v));
        int32x4_t hi = vmovl_s16(vget_high_s16(v));
        float32x4_t flo = vcvtq_f32_s32(lo);
        float32x4_t fhi = vcvtq_f32_s32(hi);
        flo = vmulq_n_f32(flo, FT_WINV);
        fhi = vmulq_n_f32(fhi, FT_WINV);
        flo = vminq_f32(vmaxq_f32(flo, vdupq_n_f32(0.0f)), vdupq_n_f32(1.0f));
        fhi = vminq_f32(vmaxq_f32(fhi, vdupq_n_f32(0.0f)), vdupq_n_f32(1.0f));
        flo = vmulq_f32(flo, flo);
        fhi = vmulq_f32(fhi, fhi);
        flo = vmulq_n_f32(flo, 127.0f);
        fhi = vmulq_n_f32(fhi, 127.0f);
        // vcvtnq = round-to-nearest-even, matching _mm*_cvtps_epi32 on x86;
        // vcvtq truncates and drifts up to one quantization step per lane.
        int32x4_t ilo = vcvtnq_s32_f32(flo);
        int32x4_t ihi = vcvtnq_s32_f32(fhi);
        int16x8_t i16 = vcombine_s16(vqmovn_s32(ilo), vqmovn_s32(ihi));
        uint8x8_t u8 = vqmovun_s16(i16);
        vst1_u8(h_i8 + l, u8);
        // nstm half
        v = vld1q_s16(src(1, l));
        lo = vmovl_s16(vget_low_s16(v));
        hi = vmovl_s16(vget_high_s16(v));
        flo = vcvtq_f32_s32(lo);
        fhi = vcvtq_f32_s32(hi);
        flo = vmulq_n_f32(flo, FT_WINV);
        fhi = vmulq_n_f32(fhi, FT_WINV);
        flo = vminq_f32(vmaxq_f32(flo, vdupq_n_f32(0.0f)), vdupq_n_f32(1.0f));
        fhi = vminq_f32(vmaxq_f32(fhi, vdupq_n_f32(0.0f)), vdupq_n_f32(1.0f));
        flo = vmulq_f32(flo, flo);
        fhi = vmulq_f32(fhi, fhi);
        flo = vmulq_n_f32(flo, 127.0f);
        fhi = vmulq_n_f32(fhi, 127.0f);
        ilo = vcvtnq_s32_f32(flo);
        ihi = vcvtnq_s32_f32(fhi);
        i16 = vcombine_s16(vqmovn_s32(ilo), vqmovn_s32(ihi));
        u8 = vqmovun_s16(i16);
        vst1_u8(h_i8 + L1 + l, u8);
    }
    if (act_nonzero) { long long nz = 0; for (int i = 0; i < 2 * L1; ++i) nz += (h_i8[i] != 0); *act_nonzero += nz; }
    const int total_in = 2 * L1;
    float h2[NNUE_DENSE_MAX];
    // L2: output-major dots via SDOT (int8 weights × uint8 activations)
    for (int o = 0; o < L2; ++o) {
        int32x4_t acc = vdupq_n_s32(0);
        const int8_t* w = &t.l2_w[static_cast<size_t>(o) * total_in];
        for (int i = 0; i < total_in; i += 16) {
            int8x16_t wv = vld1q_s8(w + i);
            uint8x16_t hv = vld1q_u8(h_i8 + i);
            acc = vdotq_s32(acc, wv, hv);
        }
        int32_t dot = vaddvq_s32(acc);
        float s = bias_add(t.l2_b[o], dot, t.ics2[o]);
        float c = s < 0 ? 0 : (s > 1 ? 1 : s);
        h2[o] = c * c;  // SCReLU: square after clamp, matching the x86 path
    }
    uint8_t h2_i8[NNUE_DENSE_MAX];
    for (int o = 0; o < L2; ++o) h2_i8[o] = round_u8(h2[o] * 127.0f);
    float h3[NNUE_DENSE_MAX];
    // L3: same SDOT pattern over L2 inputs
    for (int o = 0; o < L3; ++o) {
        int32x4_t acc = vdupq_n_s32(0);
        const int8_t* w = &t.l3_w[static_cast<size_t>(o) * L2];
        for (int i = 0; i < L2; i += 16) {
            int8x16_t wv = vld1q_s8(w + i);
            uint8x16_t hv = vld1q_u8(h2_i8 + i);
            acc = vdotq_s32(acc, wv, hv);
        }
        int32_t dot = vaddvq_s32(acc);
        float s = bias_add(t.l3_b[o], dot, t.ics3[o]);
        float c = s < 0 ? 0 : (s > 1 ? 1 : s);
        h3[o] = c * c;  // SCReLU: square after clamp, matching the x86 path
    }
    uint8_t h3_i8[NNUE_DENSE_MAX];
    for (int o = 0; o < L3; ++o) h3_i8[o] = round_u8(h3[o] * 127.0f);
    // output layer: L3=32 is a multiple of 16, no tail handling needed
    int32x4_t acc = vdupq_n_s32(0);
    for (int i = 0; i < L3; i += 16)
        acc = vdotq_s32(acc, vld1q_s8(t.out_w + i), vld1q_u8(h3_i8 + i));
    int32_t dot = vaddvq_s32(acc);
    return bias_add(t.out_b, dot, t.icso);
#else
    // Portable int8 path: the universal x86-64 build's base level, ARM without
    // the dot product, any other CPU.
    uint8_t h_i8[2 * NNUE_L1_MAX];
    for (int l = 0; l < L1; ++l) {
        float s  = *src(0, l)  * FT_WINV; float c  = s  < 0 ? 0 : (s  > 1 ? 1 : s);
        h_i8[l]        = round_u8(c * c * 127.0f);
        float s2 = *src(1, l) * FT_WINV; float c2 = s2 < 0 ? 0 : (s2 > 1 ? 1 : s2);
        h_i8[L1 + l]   = round_u8(c2 * c2 * 127.0f);
    }
    if (act_nonzero) { long long nz = 0; for (int i = 0; i < 2 * L1; ++i) nz += (h_i8[i] != 0); *act_nonzero += nz; }
    const int total_in = 2 * L1;
    float h2[NNUE_DENSE_MAX];
    for (int o = 0; o < L2; ++o) {
        int32_t acc = 0;
        const int8_t* w = &t.l2_w[static_cast<size_t>(o) * total_in];
        for (int i = 0; i < total_in; ++i) acc += w[i] * h_i8[i];
        float s = bias_add(t.l2_b[o], acc, t.ics2[o]);
        float c = s < 0 ? 0 : (s > 1 ? 1 : s);
        h2[o] = c * c;  // SCReLU: square after clamp, matching the x86 path
    }
    uint8_t h2_i8[NNUE_DENSE_MAX];
    for (int o = 0; o < L2; ++o) h2_i8[o] = round_u8(h2[o] * 127.0f);
    float h3[NNUE_DENSE_MAX];
    for (int o = 0; o < L3; ++o) {
        int32_t acc = 0;
        const int8_t* w = &t.l3_w[static_cast<size_t>(o) * L2];
        for (int i = 0; i < L2; ++i) acc += w[i] * h2_i8[i];
        float s = bias_add(t.l3_b[o], acc, t.ics3[o]);
        float c = s < 0 ? 0 : (s > 1 ? 1 : s);
        h3[o] = c * c;  // SCReLU: square after clamp, matching the x86 path
    }
    uint8_t h3_i8[NNUE_DENSE_MAX];
    for (int o = 0; o < L3; ++o) h3_i8[o] = round_u8(h3[o] * 127.0f);
    int32_t acc = 0;
    for (int i = 0; i < L3; ++i) acc += t.out_w[i] * h3_i8[i];
    return bias_add(t.out_b, acc, t.icso);
#endif
}

float forward_int8(const int16_t* acc_stm, const int16_t* acc_nstm, const TailView& t,
                   int L1, int L2, int L3, long long* act_nonzero) {
    return forward_int8_impl<false>(acc_stm, acc_nstm, nullptr, t, L1, L2, L3, act_nonzero);
}

float forward_int8_rel(const int16_t* acc_stm, const int16_t* acc_nstm, const RelLanes& rel, const TailView& t,
                       int L1, int L2, int L3, long long* act_nonzero) {
    return forward_int8_impl<true>(acc_stm, acc_nstm, &rel, t, L1, L2, L3, act_nonzero);
}

// ---- float L2/L3/out path (LNN1 nets) ----
float forward_float(const int16_t* acc_stm, const int16_t* acc_nstm, const TailView& t,
                    int L1, int L2, int L3) {
    float h[2 * NNUE_L1_MAX];
    float h2[NNUE_DENSE_MAX];
    float h3[NNUE_DENSE_MAX];
#if defined(LK_AVX2)
    const __m256 z = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f), winv8 = _mm256_set1_ps(FT_WINV);
    for (int l = 0; l < L1; l += 8) {
        _mm256_storeu_ps(h + l,      screlu8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(acc_stm + l)))), winv8), z, one));
        _mm256_storeu_ps(h + L1 + l, screlu8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(acc_nstm + l)))), winv8), z, one));
    }
    for (int o = 0; o < L2; ++o) {
        const float* w = &t.l2_wf[o * 2 * L1];
        __m256 accv = _mm256_setzero_ps();
        for (int i = 0; i < 2 * L1; i += 8)
            accv = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(h + i), accv);
        h2[o] = clip01(hsum256(accv) + t.l2_b[o]);
    }
    // L3: SIMD dot products over L2 inputs.
    for (int o = 0; o < L3; ++o) {
        const float* w = &t.l3_wf[o * L2];
        __m256 accv = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= L2; i += 8)
            accv = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(h2 + i), accv);
        float s = hsum256(accv) + t.l3_b[o];
        for (; i < L2; ++i) s += w[i] * h2[i];
        h3[o] = clip01(s);
    }
#else
    for (int l = 0; l < L1; ++l) { h[l] = clip01(acc_stm[l] * FT_WINV); h[L1 + l] = clip01(acc_nstm[l] * FT_WINV); }
    for (int o = 0; o < L2; ++o) {
        float s = t.l2_b[o]; const float* w = &t.l2_wf[o * 2 * L1];
        for (int i = 0; i < 2 * L1; ++i) s += w[i] * h[i];
        h2[o] = clip01(s);
    }
    for (int o = 0; o < L3; ++o) {
        float s = t.l3_b[o]; const float* w = &t.l3_wf[o * L2];
        for (int i = 0; i < L2; ++i) s += w[i] * h2[i];
        h3[o] = clip01(s);
    }
#endif
    float ov = t.out_b;
    for (int i = 0; i < L3; ++i) ov += t.out_wf[i] * h3[i];
    return ov;
}

} // namespace

#if defined(LK_AVX512VNNI)
#define LK_L2_LAYOUT L2Layout::ByInput4
#define LK_L3_BATCHED true
#elif defined(LK_AVX2)
#define LK_L2_LAYOUT L2Layout::Chunk32
#define LK_L3_BATCHED false
#else
#define LK_L2_LAYOUT L2Layout::OutputMajor
#define LK_L3_BATCHED false
#endif

#if defined(LK_DISPATCH)
constinit const Kernels kernels = {
#else
constexpr Kernels kernels = {
#endif
    LK_NAME, LK_L2_LAYOUT, LK_L3_BATCHED,
    &refresh_lanes, &update_lanes, &refresh_cached,
    &forward_int8, &forward_float, &bias_add,
    &rel_rows, &forward_int8_rel,
};

} // namespace luminex::nnue::LK_NAMESPACE

#undef LK_X86
#undef LK_AVX512
#undef LK_AVX512VNNI
#undef LK_AVX2
#undef LK_AVXVNNI
#undef LK_DOTPROD
#undef LK_NAME
#undef LK_ID
#undef LK_DISPATCH
#undef LK_SSE2_LANES
#undef LK_L2_LAYOUT
#undef LK_L3_BATCHED

#endif  // !LUMINEX_NNUE_DISPATCH || LK_NAMESPACE
