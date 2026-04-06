cmake_minimum_required(VERSION 3.20)
project(Luminex VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler flags for maximum performance
if(MSVC)
    add_compile_options(/W4 /Zi)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/Od /RTC1)
    else()
        add_compile_options(/O2 /Ob2 /Ot /Oi /GL)
        add_link_options(/LTCG)
    endif()
    # AVX2 support (more compatible than AVX-512)
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/arch:AVX2)
    endif()
else()
    add_compile_options(-Wall -Wextra -O2)
    add_compile_options(-fno-fast-math -funroll-loops -fomit-frame-pointer -fno-strict-aliasing)
    # Only use -march=native for local builds (CI doesn't support it)
    if(NOT CI)
        add_compile_options(-march=native -mtune=native)
    endif()
    # add_link_options(-flto)  # Disabled: suspected crash cause
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
