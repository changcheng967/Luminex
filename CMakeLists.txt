cmake_minimum_required(VERSION 3.20)
project(Luminex VERSION 5.17.0 LANGUAGES CXX)

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

    # CPU feature detection: try highest first, fall back
    # AVX512 → AVX2 → SSE4.2 → default
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        try_add_flag(/arch:AVX512)
        if(NOT HAS__arch_AVX512)
            try_add_flag(/arch:AVX2)
            if(NOT HAS__arch_AVX2)
                try_add_flag(/arch:SSE4_2)
            endif()
        endif()
    endif()
else()
    # GCC/Clang
    add_compile_options(-Wall -Wextra)
    add_compile_options(-fno-fast-math -funroll-loops -fomit-frame-pointer -fno-strict-aliasing)

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O0 -g)
    else()
        add_compile_options(-O3 -fno-exceptions -fno-rtti)

        # CPU-specific optimizations
        if(NOT CI)
            # -march=native enables all CPU features (BMI2, AVX2, AVX512, etc.)
            try_add_flag(-march=native)
            try_add_flag(-mtune=native)
        else()
            # CI: target common baseline
            try_add_flag(-march=x86-64-v3)  # AVX2 + BMI2 baseline
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

# Platform specific libraries
if(WIN32)
    target_link_libraries(luminex PRIVATE)
endif()

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
    )
    target_include_directories(luminex-lc0pack PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(luminex-lc0pack PRIVATE ZLIB::ZLIB)
    target_compile_definitions(luminex-lc0pack PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

if(BUILD_VERIFY)
    enable_language(C)
    add_executable(verify_frames src/verify_frames.c)
    target_compile_definitions(verify_frames PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
