// nnue_k_avxvnni.cpp — the universal x86-64 build's network kernels (CMake
// LUMINEX_DISPATCH) for AVX2 CPUs with AVX-VNNI and no AVX-512: Alder Lake,
// Raptor Lake, Meteor Lake, Arrow Lake (CCI's Core Ultra 7 265T), Lunar Lake.
// The AVX2 kernels, with each dense-layer dot product one VEX VPDPBUSD instead
// of VPMADDUBSW, VPMADDWD and VPADDD. CMake compiles this file with
// -march=x86-64-v3 -mavxvnni.
#define LK_NAMESPACE k_avxvnni
#define LK_EXPECT 2
#include "nnue_kernels.inc"
