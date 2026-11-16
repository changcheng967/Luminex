// nnue_k_base.cpp — the universal x86-64 build's network kernels (CMake
// LUMINEX_DISPATCH) for every CPU the engine runs on: compiled for what the
// rest of the engine targets (x86-64-v2 on Linux, x86-64 with POPCNT on
// Windows): SSE2 kernels, with SSSE3's pmaddubsw for the dense layers where
// the target has it.
#define LK_NAMESPACE k_base
#define LK_EXPECT 4
#include "nnue_kernels.inc"
