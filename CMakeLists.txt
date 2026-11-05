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
project(Luminex VERSION ${CMAKE_MATCH_1} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
    # Release builds pass AVX2 (modern) or SSE2 (compatibility).
    set(LUMINEX_MSVC_ARCH "AVX2" CACHE STRING "/arch level for MSVC and clang-cl: AVX512, AVX2 or SSE2 (the x64 default, no flag)")
    if(NOT LUMINEX_MSVC_ARCH MATCHES "^(AVX512|AVX2|SSE2)$")
        message(FATAL_ERROR "LUMINEX_MSVC_ARCH must be AVX512, AVX2 or SSE2, not '${LUMINEX_MSVC_ARCH}'")
    endif()
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT LUMINEX_MSVC_ARCH STREQUAL "SSE2")
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
        # lacks AVX2), and the release's compatibility builds use x86-64-v2.
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
        try_add_flag(-march=${LUMINEX_MARCH})
        string(REPLACE "-" "_" _lux_march_ok "HAS__march_${LUMINEX_MARCH}")
        if(NOT ${_lux_march_ok} AND NOT LUMINEX_MARCH STREQUAL "native")
            message(FATAL_ERROR "the compiler rejects -march=${LUMINEX_MARCH}")
        endif()
        if(LUMINEX_MARCH STREQUAL "native")
            try_add_flag(-mtune=native)
        endif()

        # Link-time optimization (safe on GCC/Clang)
        try_add_flag(-flto)
        if(HAS__flto)
            add_link_options(-flto)
        endif()
    endif()
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/uci.cpp
    src/board.cpp
    src/movegen.cpp
    src/search.cpp
    src/evaluation.cpp
    src/transposition.cpp
    src/magic.cpp
    src/tuner_params.cpp
    src/book.cpp
    src/nnue.cpp
)

set(HEADERS
    src/luminex.h
    src/uci.h
    src/board.h
    src/movegen.h
    src/search.h
    src/evaluation.h
    src/transposition.h
    src/bitboard.h
    src/types.h
    src/book.h
)

add_executable(luminex ${SOURCES} ${HEADERS})

target_include_directories(luminex PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Optimize for the BUILD machine. The NNUE AVX-512/VNNI paths only compile in
# with the right -march level; measured +12% nps on Xeon 2.6GHz (688K -> 770K).
option(LUMINEX_NATIVE "Optimize for the build machine (-march=native)" OFF)
if(LUMINEX_NATIVE)
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

# === Tuner target ===
option(BUILD_FEATURIZER "Build Game-Pack NNUE featurizer" OFF)
option(BUILD_EVALTRACE "Build HCE eval trace tool" OFF)
option(BUILD_LC0PACK "Build Leela-chunk -> gamepack converter (needs zlib)" OFF)
option(BUILD_VERIFY "Build gamepack frame verifier" OFF)

if(BUILD_FEATURIZER)
    add_executable(luminex-featurize
        src/featurize.cpp
        src/tuner_params.cpp
        src/board.cpp
        src/movegen.cpp
        src/evaluation.cpp
        src/nnue.cpp
        src/magic.cpp
        src/transposition.cpp
    )
    target_include_directories(luminex-featurize PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    # RAW_MOVES: gamepack frames store u16 raw moves (lc0pack format, the only
    # live data path). Without this the featurizer would read u8 move indices
    # and silently misparse every frame.
    target_compile_definitions(luminex-featurize PRIVATE RAW_MOVES _CRT_SECURE_NO_WARNINGS)
endif()

if(BUILD_EVALTRACE)
    add_executable(luminex-evaltrace
        src/eval_trace.cpp
        src/tuner_params.cpp
        src/board.cpp
        src/movegen.cpp
        src/evaluation.cpp
        src/nnue.cpp
        src/magic.cpp
        src/transposition.cpp
    )
    target_include_directories(luminex-evaltrace PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(luminex-evaltrace PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()



if(BUILD_LC0PACK)
    find_package(ZLIB REQUIRED)
    add_executable(luminex-lc0pack
        src/lc0pack.cpp
        src/tuner_params.cpp
        src/board.cpp
        src/movegen.cpp
        src/evaluation.cpp
        src/nnue.cpp
        src/magic.cpp
        src/transposition.cpp
    )
    target_include_directories(luminex-lc0pack PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(luminex-lc0pack PRIVATE ZLIB::ZLIB)
    target_compile_definitions(luminex-lc0pack PRIVATE _CRT_SECURE_NO_WARNINGS)
    add_executable(frame_merge src/frame_merge.cpp)
endif()

if(BUILD_VERIFY)
    enable_language(C)
    add_executable(verify_frames src/verify_frames.c)
    target_compile_definitions(verify_frames PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
