// embedded_net.h — the network file built into the binary (CMake LUMINEX_EMBED_NET).
#pragma once
#include <cstddef>

namespace luminex {

// The bytes of the network file built into this binary, or {nullptr, 0} when
// the build carries none (the CMake default, and OpenBench's Makefile build).
struct EmbeddedNet {
    const unsigned char* data;
    std::size_t size;
};

EmbeddedNet embedded_net();

} // namespace luminex
