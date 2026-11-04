# OpenBench build shim for Luminex.
#
# Copy this file to the ROOT of the Luminex repository (next to CMakeLists.txt).
# OpenBench clients invoke, from the directory given by EngineConfig.build_path:
#
#     make -j EXE=<binary-name> [CC=<cc>] [CXX=<cxx>] [EVALFILE=<net>]
#
# and then expect the binary to exist at <build_path>/<binary-name>.
#
# Luminex upstream builds with CMake only, so this wraps it. Note OpenBench's
# makefile_command() passes no target, no CPU flags and no system name: the ISA
# is decided here, per worker, which is why MARCH defaults to a fixed level
# rather than "native" (see the note below).

EXE       ?= luminex
CC        ?= gcc
CXX       ?= g++
BUILD_DIR ?= build-openbench

# Why not -march=native: OpenBench verifies that every worker produces the same
# bench node count. Floating point in the HCE path and any vectorised reduction
# can change with ISA level, so a heterogeneous farm built with "native" can
# produce mismatched benches and get blacklisted. A fixed level is reproducible.
# Override per-test if you know every worker is identical hardware.
#
# Arch-dependent: the OpenBench server itself is an Oracle Ampere A1 (aarch64).
# Passing -march=x86-64-v3 there does not fail the build — CMakeLists.txt:70
# routes it through try_add_flag(), which silently drops flags the compiler
# rejects — so you get an unoptimised ARM binary with no warning. Pick per
# architecture instead.
UNAME_M   := $(shell uname -m)
ifeq ($(UNAME_M),aarch64)
  MARCH   ?= armv8-a
else
  # Auto-detect AVX-512: the 512-bit VPDPBUSD (VNNI) doubles int8
  # dot-product throughput over v3's 256-bit version. cascadelake is the
  # minimum Intel target that includes AVX-512 + VNNI together; the binary
  # runs on any x86 CPU with those features (Intel Cascade Lake+, AMD
  # Zen 4+). Falls back to v3 (AVX2) without AVX-512. Override: MARCH=...
  HAS_AVX512 := $(shell grep -q avx512f /proc/cpuinfo 2>/dev/null && echo yes)
  ifeq ($(HAS_AVX512),yes)
    MARCH   ?= cascadelake
  else
    MARCH   ?= x86-64-v3
  endif
endif

# The engine is C++23 (<format>): needs g++ 13+ or clang 18+ with libc++.
# Fail with one clear line instead of a page of template errors on the
# older g++ that many worker images ship by default (e.g. 9.4 on AI Studio).
GCC_MAJOR := $(shell $(CXX) -dumpversion 2>/dev/null | cut -d. -f1)
IS_CLANG  := $(shell $(CXX) --version 2>/dev/null | head -1 | grep -ci clang)
ifneq ($(IS_CLANG),1)
ifeq ($(shell test $(GCC_MAJOR) -ge 13 2>/dev/null && echo ok),)
  $(error Luminex needs g++ 13+ (C++23 <format>). Found $(CXX) $(shell $(CXX) -dumpversion 2>/dev/null). Install g++-13 and rerun with CXX=g++-13)
endif
endif

# EVALFILE is accepted and ignored: Luminex loads .nnue at runtime through the
# EvalFile UCI option, not by embedding it at build time. The OpenBench client
# is patched to inject that option (see ../LUMINEX_INTEGRATION.md).
EVALFILE  ?=

.PHONY: all clean

all:
	@cmake -S . -B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_C_COMPILER=$(CC) \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DLUMINEX_MARCH=$(MARCH) \
		-DLUMINEX_NATIVE=OFF
	@cmake --build $(BUILD_DIR) --target luminex -j
	@cp -f $(BUILD_DIR)/luminex ./$(EXE)
	@echo "built $(EXE)"

clean:
	@rm -rf $(BUILD_DIR) $(EXE)
