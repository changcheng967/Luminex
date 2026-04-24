cmake_minimum_required(VERSION 3.20)
project(Luminex VERSION 5.13.0 LANGUAGES CXX)

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
)

add_executable(luminex ${SOURCES} ${HEADERS})

target_include_directories(luminex PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Platform specific libraries
if(WIN32)
    target_link_libraries(luminex PRIVATE)
endif()

