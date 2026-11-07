// embedded_net.cpp — the network file built into the binary.
//
// CMake's LUMINEX_EMBED_NET puts a network file inside the executable, so a
// release plays with NNUE on its own instead of needing the .nnue beside it.
// MSVC and clang-cl link the file as an RCDATA resource (luminex_net.rc,
// written by CMake); GCC and Clang assemble it in with .incbin. Builds without
// the option carry no network and embedded_net() reports none.
#include "embedded_net.h"

#if defined(LUMINEX_EMBED_NET_RC)

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace luminex {

EmbeddedNet embedded_net() {
    // A resource stays mapped with the image: nothing to copy or free.
    HRSRC res = FindResourceA(nullptr, "LUMINEX_NET", MAKEINTRESOURCEA(10));  // 10 = RT_RCDATA
    HGLOBAL mem = res ? LoadResource(nullptr, res) : nullptr;
    const void* data = mem ? LockResource(mem) : nullptr;
    if (!data) return {nullptr, 0};
    return {static_cast<const unsigned char*>(data), std::size_t(SizeofResource(nullptr, res))};
}

} // namespace luminex

#elif defined(LUMINEX_EMBED_NET_FILE)

#include <cstdint>

// Assembler symbols carry the platform's C prefix (an underscore on macOS).
#define LUMINEX_STR_(x) #x
#define LUMINEX_STR(x) LUMINEX_STR_(x)
#define LUMINEX_SYM(name) LUMINEX_STR(__USER_LABEL_PREFIX__) #name

#if defined(__APPLE__)
#define LUMINEX_NET_SECTION ".const_data\n"
#elif defined(_WIN32)
#define LUMINEX_NET_SECTION ".section .rdata,\"dr\"\n"
#else
#define LUMINEX_NET_SECTION ".section .rodata\n"
#endif

__asm__(LUMINEX_NET_SECTION
        ".globl " LUMINEX_SYM(luminex_net_begin) "\n"
        ".globl " LUMINEX_SYM(luminex_net_end) "\n"
        ".p2align 6\n"
        LUMINEX_SYM(luminex_net_begin) ":\n"
        ".incbin \"" LUMINEX_EMBED_NET_FILE "\"\n"
        LUMINEX_SYM(luminex_net_end) ":\n"
        ".text\n");

extern "C" const unsigned char luminex_net_begin[];
extern "C" const unsigned char luminex_net_end[];

namespace luminex {

EmbeddedNet embedded_net() {
    const auto begin = reinterpret_cast<std::uintptr_t>(luminex_net_begin);
    const auto end = reinterpret_cast<std::uintptr_t>(luminex_net_end);
    return {luminex_net_begin, std::size_t(end - begin)};
}

} // namespace luminex

#else

namespace luminex {

EmbeddedNet embedded_net() { return {nullptr, 0}; }

} // namespace luminex

#endif
