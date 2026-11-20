cmake_minimum_required(VERSION 3.20)
# Single source of truth for the version: src/luminex.h (ENGINE_VERSION).
# Bump the version THERE only — CMake reads it so project(), installers, and
# the UCI id line can never disagree again (the v6.2.0 release shipped with
# binaries reporting 6.1.0 because the string lived in two places).
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/luminex.h" _LUX_HDR)
string(REGEX MATCH "ENGINE_VERSION = \"([0-9]+\\.[0-9]+\\.[0-9]+)\"" _LUX_VER "${_LUX_HDR}")
if(NOT CMAKE_MATCH_1)
    message(FATAL_ERROR "cannot parse ENGINE_VERSION from src/luminex.h")
endif()
# The oldest macOS the binary runs on. With no target, clang builds for the
# macOS it runs on, so v6.6.0's first macOS binary needed macOS 26, the
# runner's. 13.3 is the oldest that has what the engine uses: std::format
# (the UCI info lines) formats floating point through std::to_chars, which
# macOS ships from 13.3. Every Apple silicon Mac runs macOS 13. project()
# reads the target, so it comes first.
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.3" CACHE STRING "Oldest macOS the binary runs on")
project(Luminex VERSION ${CMAKE_MATCH_1} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# One x86-64 binary for every CPU, as the releases ship: the engine compiled
# for x86-64-v2 (x86-64 with POPCNT under clang-cl), and the network kernels
# (src/nnue_kernels.h) compiled once per instruction-set level, each with its
# level's flags; at run time nnue.cpp uses the best level the CPU and its OS
# support (cpuid and xgetbv), or a lower one the SimdLevel option asks for.
# LUMINEX_MARCH and LUMINEX_MSVC_ARCH do not apply. Off, the default and
# OpenBench's build, the kernels are compiled into nnue.cpp for LUMINEX_MARCH
# and called directly.
option(LUMINEX_DISPATCH "One x86-64 binary that picks its network kernels for the CPU at run time" OFF)
if(LUMINEX_DISPATCH)
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64)$")
        message(FATAL_ERROR "LUMINEX_DISPATCH builds for x86-64, not ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(FATAL_ERROR "LUMINEX_DISPATCH needs GCC, Clang or clang-cl (-T ClangCL): the kernel files take GCC-style instruction-set flags")
    endif()
endif()

# Detect CPU features
include(CheckCXXCompilerFlag)

# Helper: check if a compile flag works
function(try_add_flag flag)
    string(REPLACE "=" "_" flag_var ${flag})
    string(REPLACE "-" "_" flag_var ${flag_var})
    check_cxx_compiler_flag(${flag} HAS_${flag_var})
    if(HAS_${flag_var})
        add_compile_options(${flag})
    endif()
endfunction()

# Compiler flags for maximum performance
if(MSVC)
    # The C and C++ runtime linked into the .exe (/MT), so it runs on a Windows
    # without the Visual C++ Redistributable; v6.6.0's first Windows binaries
    # needed it (MSVCP140, VCRUNTIME140 and VCRUNTIME140_1 DLLs).
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_compile_options(/W4 /Zi)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/Od /RTC1)
    else()
        # Aggressive optimization
        add_compile_options(/O2 /Ob2 /Ot /Oi /GL /GS-)
        add_link_options(/LTCG)
    endif()

    # Instruction set: an explicit level for the CPUs the binary is meant for.
    # This used to probe /arch:AVX512, then AVX2, then SSE4_2 and keep the newest
    # the compiler accepted, which tests the compiler rather than the target CPU.
    # The probe's HAS_ names never matched either, so every accepted flag was
    # added: MSVC kept the last valid one (AVX2) and clang-cl, whose last one
    # (SSE4_2) is not a valid x64 level, dropped them all and built for SSE2.
    # The release build is LUMINEX_DISPATCH, which takes no /arch.
    set(LUMINEX_MSVC_ARCH "AVX2" CACHE STRING "/arch level for MSVC and clang-cl: AVX512, AVX2 or SSE2 (the x64 default, no flag)")
    if(NOT LUMINEX_MSVC_ARCH MATCHES "^(AVX512|AVX2|SSE2)$")
        message(FATAL_ERROR "LUMINEX_MSVC_ARCH must be AVX512, AVX2 or SSE2, not '${LUMINEX_MSVC_ARCH}'")
    endif()
    if(LUMINEX_DISPATCH)
        # The x64 default (SSE2) with POPCNT, as the compatibility build ran:
        # MSVC emits popcnt for __popcnt64 at any level, clang-cl only with it.
        add_compile_options(-mpopcnt)
    elseif(NOT CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT LUMINEX_MSVC_ARCH STREQUAL "SSE2")
        add_compile_options(/arch:${LUMINEX_MSVC_ARCH})
    endif()
else()
    # GCC/Clang
    add_compile_options(-Wall -Wextra)
    add_compile_options(-fno-fast-math -funroll-loops -fomit-frame-pointer -fno-strict-aliasing)

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O0 -g)
    else()
        add_compile_options(-O3 -fno-exceptions -fno-rtti)

        # Instruction set. Local builds default to the build machine (native) and
        # CI to x86-64-v3 (AVX2 + BMI2); LUMINEX_MARCH overrides both for
        # portable binaries. The static featurizer is built on one machine and
        # shipped to others, where a native build SIGILLs (a Hygon Zen-1 pod
        # lacks AVX2). The release builds are LUMINEX_DISPATCH: x86-64-v2 here.
        # A -march in CMAKE_CXX_FLAGS cannot do this: the options added here
        # come later on the command line and win, which is how the v6.3.0
        # "SSE4.2" Linux binary came out byte-identical to the AVX2 one.
        # Apple silicon gets armv8.4-a: Apple clang accepts -march=native there
        # but builds without the dot-product extension, and every Apple silicon
        # Mac has Armv8.4 with it, which the NEON network kernels need.
        if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
            set(_lux_march_default "armv8.4-a")
        elseif(CI)
            set(_lux_march_default "x86-64-v3")
        else()
            set(_lux_march_default "native")
        endif()
        set(LUMINEX_MARCH "${_lux_march_default}" CACHE STRING "-march level (native, x86-64-v2, x86-64-v3, ...)")
        # The universal build's engine runs on any x86-64-v2 CPU; its kernel
        # files carry the higher levels (below).
        if(LUMINEX_DISPATCH)
            set(_lux_march "x86-64-v2")
        else()
            set(_lux_march "${LUMINEX_MARCH}")
        endif()
        try_add_flag(-march=${_lux_march})
        string(REPLACE "-" "_" _lux_march_ok "HAS__march_${_lux_march}")
        if(NOT ${_lux_march_ok} AND NOT _lux_march STREQUAL "native")
            message(FATAL_ERROR "the compiler rejects -march=${_lux_march}")
        endif()
        if(_lux_march STREQUAL "native")
            try_add_flag(-mtune=native)
        endif()

        # Link-time optimization (safe on GCC/Clang)
        try_add_flag(-flto)
        if(HAS__flto)
            add_link_options(-flto)
        endif()

        # clang on Linux: no PIE, linked by lld. clang 23 + lld + no PIE is the
        # fastest build measured, +25% game speed and +16-22% bench over gcc 14
        # on the molab SPRT workers (2026-11-17): a PIE reaches every table
        # through position-independent addressing. lld also links clang's LTO.
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
            try_add_flag(-fno-pie)
            add_link_options(-no-pie)
            include(CheckLinkerFlag)
            check_linker_flag(CXX "-fuse-ld=lld" HAS_LLD)
            if(HAS_LLD)
                add_link_options(-fuse-ld=lld)
            endif()
        endif()
    endif()
endif()

# The network: nnue.cpp, and in the universal build the kernels once per
# instruction-set level, best first, each with the flags of its level: the
# file written here for a level compiles src/nnue_kernels.h in that level's
# namespace (LK_NAMESPACE; LK_EXPECT is the level's index, which the kernels
# check against the level the flags give). They are compiled without LTO, so
# the link keeps each one's code for its own level, and without FMA
# contraction, so every level rounds alike (the bias adds are FMAs, written
# out: see src/nnue_kernels.h).
set(LUMINEX_NNUE_SOURCES src/nnue.cpp)
if(LUMINEX_DISPATCH)
    add_compile_definitions(LUMINEX_NNUE_DISPATCH)
    set(_lux_k_avx512vnni -march=x86-64-v4 -mavx512vnni)
    set(_lux_k_avx512 -march=x86-64-v4)
    set(_lux_k_avxvnni -march=x86-64-v3 -mavxvnni)
    set(_lux_k_avx2 -march=x86-64-v3)
    set(_lux_k_base "")   # the engine's own level
    if(MSVC)   # clang-cl
        set(_lux_k_common /clang:-ffp-contract=off)
    else()
        set(_lux_k_common -ffp-contract=off -fno-lto)
    endif()
    set(_lux_id 0)
    foreach(_lux_level IN ITEMS avx512vnni avx512 avxvnni avx2 base)
        set(_lux_src "${CMAKE_CURRENT_BINARY_DIR}/nnue_k_${_lux_level}.cpp")
        file(CONFIGURE OUTPUT "${_lux_src}" CONTENT
             "#define LK_NAMESPACE k_${_lux_level}\n#define LK_EXPECT ${_lux_id}\n#include \"nnue_kernels.h\"\n")
        set(_lux_opts ${_lux_k_${_lux_level}} ${_lux_k_common})
        list(APPEND LUMINEX_NNUE_SOURCES ${_lux_src})
        set_source_files_properties(${_lux_src} PROPERTIES COMPILE_OPTIONS "${_lux_opts}")
        math(EXPR _lux_id "${_lux_id} + 1")
    endforeach()
    message(STATUS "Universal x86-64 build: the engine for x86-64-v2, network kernels for avx512vnni, avx512, avxvnni, avx2 and base")
endif()

# Source files: the position, the evaluations and the network, which the
# tools link too (LUMINEX_CORE_SOURCES), then the search and the UCI loop.
set(LUMINEX_CORE_SOURCES
    src/board.cpp
    src/evaluation.cpp
    ${LUMINEX_NNUE_SOURCES}
)
set(SOURCES
    src/uci.cpp
    src/search.cpp
    ${LUMINEX_CORE_SOURCES}
)
set(HEADERS
    src/luminex.h
    src/nnue_kernels.h
)

add_executable(luminex ${SOURCES} ${HEADERS})

target_include_directories(luminex PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# A network built into the binary, so it plays with NNUE on its own instead of
# needing the .nnue file beside it; the release builds carry the shipped net.
# The NNUEFile option still loads any other file. MSVC and clang-cl link it as
# a resource, GCC and Clang assemble it in (the end of src/nnue.cpp). Empty, the
# default, builds without one, as OpenBench's Makefile build does.
#
# The shipped network's name has one source: the file name of the network
# .github/scripts/ci.sh pins (its pin_url= line), the network's real name
# (luminex_<architecture>_<NN>..., as in docs/NNUE.md). CMake reads it here, so the
# default NNUEFile, the file ci.sh fetch-net saves, the name a build reports and
# the file a release attaches all follow a net swap with no other edit, and no
# build can name a network it does not carry.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/.github/scripts/ci.sh" _lux_url REGEX "^pin_url=")
string(REGEX REPLACE "^pin_url=.*/" "" LUMINEX_NET_NAME "${_lux_url}")
if(NOT LUMINEX_NET_NAME MATCHES "^luminex_[A-Za-z0-9_]+\\.nnue$")
    message(FATAL_ERROR "ci.sh's pin_url= line names no luminex_*.nnue network: '${_lux_url}'")
endif()
add_compile_definitions("LUMINEX_DEFAULT_NET=\"${LUMINEX_NET_NAME}\"")
set(LUMINEX_EMBED_NET "" CACHE FILEPATH "Network file to build into the binary (empty: none)")
option(LUMINEX_EMBED_PINNED "Build in the pinned network, net/<its real name> as ci.sh fetch-net saves it" OFF)
if(LUMINEX_EMBED_PINNED AND NOT LUMINEX_EMBED_NET)
    set(LUMINEX_EMBED_NET "net/${LUMINEX_NET_NAME}")
endif()
if(LUMINEX_EMBED_NET)
    get_filename_component(_lux_net "${LUMINEX_EMBED_NET}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${_lux_net}")
        message(FATAL_ERROR "LUMINEX_EMBED_NET: ${_lux_net} does not exist")
    endif()
    # The built-in network reports its own file name, whatever file is built in.
    get_filename_component(_lux_net_name "${_lux_net}" NAME)
    target_compile_definitions(luminex PRIVATE "LUMINEX_EMBED_NAME=\"${_lux_net_name}\"")
    if(MSVC)
        enable_language(RC)
        set(_lux_rc "${CMAKE_CURRENT_BINARY_DIR}/luminex_net.rc")
        file(CONFIGURE OUTPUT "${_lux_rc}" CONTENT "LUMINEX_NET RCDATA \"@_lux_net@\"\n" @ONLY)
        target_sources(luminex PRIVATE "${_lux_rc}")
        target_compile_definitions(luminex PRIVATE LUMINEX_EMBED_NET_RC)
        set_source_files_properties("${_lux_rc}" PROPERTIES OBJECT_DEPENDS "${_lux_net}")
    else()
        target_compile_definitions(luminex PRIVATE "LUMINEX_EMBED_NET_FILE=\"${_lux_net}\"")
        # .incbin is invisible to the compiler's dependency scan.
        set_source_files_properties(src/nnue.cpp PROPERTIES OBJECT_DEPENDS "${_lux_net}")
    endif()
    message(STATUS "Building in the network ${_lux_net}")
endif()

# A development build (the rolling luminex-dev pre-release that build.yml
# publishes from every push to main) names itself after its tag, as in
# "id name Luminex dev-20261112-d4774f1a", so a result can be traced to its
# commit. Empty, the default, keeps the version from src/luminex.h.
set(LUMINEX_BUILD_ID "" CACHE STRING "Development build name reported in place of the version (empty: the version)")
if(LUMINEX_BUILD_ID)
    if(NOT LUMINEX_BUILD_ID MATCHES "^[A-Za-z0-9._-]+$")
        message(FATAL_ERROR "LUMINEX_BUILD_ID: letters, digits, '.', '_' and '-' only")
    endif()
    target_compile_definitions(luminex PRIVATE "LUMINEX_BUILD_ID=\"${LUMINEX_BUILD_ID}\"")
    message(STATUS "Development build ${LUMINEX_BUILD_ID}")
endif()

# Optimize for the BUILD machine. The NNUE AVX-512/VNNI paths only compile in
# with the right -march level; measured +12% nps on Xeon 2.6GHz (688K -> 770K).
option(LUMINEX_NATIVE "Optimize for the build machine (-march=native)" OFF)
if(LUMINEX_NATIVE AND LUMINEX_DISPATCH)
    message(FATAL_ERROR "LUMINEX_NATIVE builds for this machine, LUMINEX_DISPATCH for every x86-64 CPU: choose one")
elseif(LUMINEX_NATIVE)
    target_compile_options(luminex PRIVATE -march=native)
endif()

# Platform specific libraries
if(WIN32)
    target_link_libraries(luminex PRIVATE)
endif()

# Link pthreads explicitly: glibc >= 2.34 folds it into libc so CI never
# noticed it was missing, but older sysroots (conda-forge toolchains target
# 2.28) keep it separate and std::thread fails at link time without this.
find_package(Threads REQUIRED)
target_link_libraries(luminex PRIVATE Threads::Threads)

# A crash prints the crashing thread's stack (std::stacktrace). Off in every
# release; libstdc++ keeps <stacktrace> in libstdc++exp (GCC 14 and later).
option(LUMINEX_STACKTRACE "Print a stack trace when the engine crashes (debugging aid)" OFF)
if(LUMINEX_STACKTRACE)
    target_compile_definitions(luminex PRIVATE LUMINEX_STACKTRACE)
    if(NOT MSVC)
        target_link_libraries(luminex PRIVATE stdc++exp)
    endif()
endif()

# === The training data tools (src/tools.cpp, one section and one executable each) ===
option(BUILD_FEATURIZER "Build Game-Pack NNUE featurizer" OFF)
option(BUILD_EVALTRACE "Build HCE eval trace tool" OFF)
option(BUILD_LC0PACK "Build Leela-chunk -> gamepack converter (needs zlib)" OFF)
option(BUILD_VERIFY "Build gamepack frame verifier" OFF)

# A tool: tools.cpp compiled with its macro (LUMINEX_TOOL_<TOOL>), and with
# CORE the position, evaluation and network sources it links.
function(luminex_tool name tool)
    cmake_parse_arguments(_t "CORE" "" "" ${ARGN})
    if(_t_CORE)
        add_executable(${name} src/tools.cpp ${LUMINEX_CORE_SOURCES})
    else()
        add_executable(${name} src/tools.cpp)
    endif()
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(${name} PRIVATE LUMINEX_TOOL_${tool} _CRT_SECURE_NO_WARNINGS)
endfunction()

if(BUILD_FEATURIZER)
    luminex_tool(luminex-featurize FEATURIZE CORE)
    # RAW_MOVES: gamepack frames store u16 raw moves (lc0pack format, the only
    # live data path). Without this the featurizer would read u8 move indices
    # and silently misparse every frame.
    target_compile_definitions(luminex-featurize PRIVATE RAW_MOVES)
endif()

if(BUILD_EVALTRACE)
    luminex_tool(luminex-evaltrace EVALTRACE CORE)
endif()

if(BUILD_LC0PACK)
    find_package(ZLIB REQUIRED)
    luminex_tool(luminex-lc0pack LC0PACK CORE)
    target_link_libraries(luminex-lc0pack PRIVATE ZLIB::ZLIB)
    luminex_tool(frame_merge FRAME_MERGE)
endif()

if(BUILD_VERIFY)
    luminex_tool(verify_frames VERIFY_FRAMES)
endif()
