// nnue_k_avx512.cpp — the universal x86-64 build's network kernels (CMake
// LUMINEX_DISPATCH) for x86-64-v4 CPUs without AVX512-VNNI, such as
// Skylake-SP and Skylake-X: AVX-512 accumulator and activation kernels, the
// AVX2 dense layers. CMake compiles this file with -march=x86-64-v4.
#define LK_NAMESPACE k_avx512
#define LK_EXPECT 1
#include "nnue_kernels.inc"
