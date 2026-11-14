// nnue_k_avx512vnni.cpp — the universal x86-64 build's network kernels
// (CMake LUMINEX_DISPATCH) for x86-64-v4 CPUs with AVX512-VNNI: Cascade Lake,
// Ice Lake, Sapphire Rapids and later Xeons, Zen 4 and later. CMake compiles
// this file with -march=x86-64-v4 -mavx512vnni; nnue.cpp calls its table only
// when the CPU and the OS support all of it.
#define LK_NAMESPACE k_avx512vnni
#define LK_EXPECT 0
#include "nnue_kernels.inc"
