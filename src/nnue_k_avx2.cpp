// nnue_k_avx2.cpp — the universal x86-64 build's network kernels (CMake
// LUMINEX_DISPATCH) for x86-64-v3 CPUs (AVX2, BMI2, FMA: Intel Haswell and
// later, AMD Excavator and Zen). CMake compiles this file with
// -march=x86-64-v3.
#define LK_NAMESPACE k_avx2
#define LK_EXPECT 3
#include "nnue_kernels.inc"
