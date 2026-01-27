cmake_minimum_required(VERSION 3.20)
project(Luminex VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler flags for maximum performance
if(MSVC)
    add_compile_options(/W4 /WX /O2 /Ob2 /Ot /Oi /GL)
    add_link_options(/LTCG)
    # AVX-512 support
    add_compile_options(/arch:AVX512)
else()
    add_compile_options(-Wall -Wextra -Werror -O3 -march=native -mtune=native)
    add_compile_options(-ffast-math -funroll-loops -fomit-frame-pointer)
    add_link_options(-flto)
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
