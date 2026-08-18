CC      ?= cc
CFLAGS  ?= -O2 -g -std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion
CPPFLAGS = -Iinclude -Ithird_party/blake3
LDFLAGS ?=
LDLIBS   = -lm

# Real semantic embeddings need llama.cpp. Point LLAMA_DIR at a checkout with
# a built libllama.so; without it the library still builds, sigil_embedder_llama()
# returns NULL, and the semantic test is skipped rather than silently passing
# against the non-semantic fallback.
LLAMA_DIR ?= $(HOME)/llama.cpp

ifneq ($(wildcard $(LLAMA_DIR)/build/bin/libllama.so),)
  HAVE_LLAMA := 1
  CPPFLAGS += -DSIGIL_WITH_LLAMA -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include
  LLAMA_LDFLAGS = -L$(LLAMA_DIR)/build/bin -lllama -lggml -lggml-base \
                  -Wl,-rpath,$(LLAMA_DIR)/build/bin
endif

# OpenVINO is the preferred backend on Intel hardware: 515 paragraphs/s on an
# Arc Pro B50 against 96 for llama.cpp on the CPU, and llama.cpp is additionally
# wrong on Battlemage above ~4B parameters. Optional in exactly the same way --
# without it sigil_embedder_openvino() returns NULL.
#
#   pip install openvino "optimum-intel[openvino]"
#   optimum-cli export openvino -m sentence-transformers/all-MiniLM-L6-v2 \
#       --task feature-extraction /models/minilm
# Prefer the native C++ runtime archive. The pip wheel worked, but linking a
# server against a library that lives inside a Python venv means the venv's
# lifetime governs the binary's (one deleted scratch venv already produced an
# unrunnable sigilfs), its TBB loads plugins with RTLD_DEEPBIND (which blocks
# ASan outright), and it ships no unversioned .so. The archive install has
# none of those problems and no Python anywhere in the chain.
OPENVINO_DIR ?= $(firstword $(wildcard /mnt/bulk/openvino-native/current) \
                            /opt/intel/openvino)

# Two layouts. The toolkit/archive installs under runtime/; the pip wheel puts
# headers at include/ and ships libopenvino.so.<ver> with no unversioned
# symlink, so -lopenvino does not resolve and the versioned file is linked
# directly.
#
# --disable-new-dtags makes the rpath DT_RPATH rather than DT_RUNPATH, so it
# beats LD_LIBRARY_PATH: this machine's oneAPI environment exports a 2023-era
# libtbb.so.12 there, and OpenVINO 2026 picking that up at run time is exactly
# the kind of silent substitution this build must not allow.
ifneq ($(wildcard $(OPENVINO_DIR)/runtime/include/openvino/openvino.hpp),)
  HAVE_OPENVINO := 1
  OV_INC = -I$(OPENVINO_DIR)/runtime/include
  OV_LIB = -L$(OPENVINO_DIR)/runtime/lib/intel64 -lopenvino \
           -Wl,--disable-new-dtags \
           -Wl,-rpath,$(OPENVINO_DIR)/runtime/lib/intel64 \
           -Wl,-rpath,$(OPENVINO_DIR)/runtime/3rdparty/tbb/lib
else ifneq ($(wildcard $(OPENVINO_DIR)/include/openvino/openvino.hpp),)
  HAVE_OPENVINO := 1
  OV_INC = -I$(OPENVINO_DIR)/include
  OV_SO  = $(firstword $(wildcard $(OPENVINO_DIR)/libs/libopenvino.so.*))
  OV_LIB = $(OV_SO) -Wl,-rpath,$(OPENVINO_DIR)/libs
endif

ifdef HAVE_OPENVINO
  OVOBJ = src/embed_openvino.o
  OV_CPPFLAGS = -DSIGIL_WITH_OPENVINO
  # Global, not just for the C++ rule. embed_llama.c carries the
  # not-built stub for sigil_embedder_openvino() behind
  # #ifndef SIGIL_WITH_OPENVINO; without the macro there, both the stub
  # and the real definition land in libsigil.a and the linker silently
  # picks the stub -- which returns NULL, so the server reports "cannot
  # load model" and no OpenVINO error, because none ever ran.
  CPPFLAGS += -DSIGIL_WITH_OPENVINO
endif

# Default model for the semantic test.
MODEL ?= $(HOME)/models/all-MiniLM-L6-v2-f16.gguf

# BLAKE3 portable only: sigil has its own SIMD dispatch, and a second
# independent CPU-detection layer is a liability rather than a speedup.
CPPFLAGS += -DBLAKE3_NO_AVX512 -DBLAKE3_NO_AVX2 -DBLAKE3_NO_SSE41 \
            -DBLAKE3_NO_SSE2 -DBLAKE3_USE_NEON=0

BLAKE3 = third_party/blake3/blake3.c third_party/blake3/blake3_dispatch.c \
         third_party/blake3/blake3_portable.c

SRC  = $(BLAKE3) src/sigil.c src/trit.c src/store.c src/scan_scalar.c \
       src/scan_x86.c src/scan_sse.c src/scan_neon.c src/scan_generic.c src/scan_range.c \
       src/simhash.c src/embed_llama.c src/utf8_repair.c src/split.c \
       src/veccache.c
OBJ  = $(SRC:.c=.o) $(OVOBJ)
LIB  = libsigil.a

# The one C++ translation unit. Kept out of the C sources so libsigil stays
# buildable with a C compiler alone when OpenVINO is absent.
src/embed_openvino.o: src/embed_openvino.cpp include/sigil_embed.h
	$(CXX) -std=c++17 -O2 -Iinclude $(OV_CPPFLAGS) $(OV_INC) -c $< -o $@

TESTS = test/differential test/bench test/bench_mt test/semantic test/eval

CORPUS ?= test/data/corpus.txt

# sigilfs is built with plan9port's 9c/9l, not the system compiler: it links
# lib9p and libtab, both of which want plan9port's libc. libsigil stays plain
# C11 and dependency-free; store.c is the only file that sees both worlds.
# The source tree rather than /usr/local/plan9: the installed copy may predate
# the lib9p 64-bit wstat fix. See docs/PLAN9PORT-BUG.md.
PLAN9 ?= $(HOME)/Repo/plan9port
# The canonical libtab checkout, not a vendored snapshot. This used to
# point at objective-9c/libtab, which was a stale copy: it predated the
# text-encoding fix and the streaming writer this code now depends on.
LIBTAB_SRC ?= $(HOME)/Repo/libtab

.PHONY: all check check-semantic eval corpus bench bench-mt clean sbom sigilfs prop sanitize fuzz mutate oom segments

all: $(LIB)

$(LIB): $(OBJ)
	$(AR) rcs $@ $^

%.o: %.c include/sigil.h include/sigil_embed.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

# Vendored third-party code builds without sigil's stricter warnings: it is
# not ours to fix, and patching it would complicate re-vendoring upstream.
third_party/blake3/%.o: third_party/blake3/%.c
	$(CC) -O2 -g -std=c11 -Wall $(CPPFLAGS) -c $< -o $@

test/differential: test/differential.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) -o $@ $(LDFLAGS) $(LDLIBS)

test/bench: test/bench.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) -o $@ $(LDFLAGS) $(LDLIBS)

# Threading lives in the benchmark, not the library: libsigil stays
# single-threaded and free of a pthreads dependency.
test/bench_mt: test/bench_mt.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) -o $@ $(LDFLAGS) $(LDLIBS) -lpthread

test/semantic: test/semantic.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) -o $@ $(LDFLAGS) $(LLAMA_LDFLAGS) $(LDLIBS)

test/eval: test/eval.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) -o $@ $(LDFLAGS) $(LLAMA_LDFLAGS) $(LDLIBS)

test/veccache: test/veccache.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) -Iinclude -o $@ $< $(LIB) $(LLAMA_LDFLAGS) $(OV_LIB) $(LDLIBS) -lstdc++

check: test/differential test/unit test/bridge test/oom test/veccache test/segments cmd/sigilfs
	./test/differential
	./test/unit
	./test/bridge
	./test/oom
	./test/veccache
	./test/segments
	sh test/store.sh

cmd/sigilfs:
	$(MAKE) -C cmd

THEFT ?= $(HOME)/Repo/libtab/tests/vendor/theft

# -lstdc++ because libsigil.a carries embed_openvino.o when OpenVINO is
# found, and that object needs the C++ runtime even when this test never
# calls it.
test/bridge: test/bridge.c cmd/bridge.o $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) -Iinclude -o $@ $< cmd/bridge.o $(LIB) $(LLAMA_LDFLAGS) $(OV_LIB) $(LDLIBS) -lstdc++

cmd/bridge.o: cmd/bridge.c include/sigil.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -Iinclude -c $< -o $@

test/unit: test/unit.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) -Iinclude -o $@ $< $(LIB) $(LLAMA_LDFLAGS) $(OV_LIB) $(LDLIBS) -lstdc++

# Property tests over generated operation sequences, compared against a plain
# model. unit.c checks what someone thought to write down; these check what
# must hold for every sequence, and report a shrunk counterexample when it
# does not. theft is vendored in the libtab checkout.
test/prop: test/prop.c $(LIB)
	$(CC) -std=c99 -O1 -g -Iinclude -I$(THEFT)/inc -o $@ $< $(LIB) \
		$(THEFT)/build/libtheft.a $(LLAMA_LDFLAGS) $(OV_LIB) $(LDLIBS) -lpthread

prop: test/prop
	./test/prop

# Allocation-failure paths, via ld --wrap. "Needs an out-of-memory condition"
# is not a reason to leave error handling untested; it is a reason to inject
# the failure deterministically.
# Growth without copying, per features/store_growth.feature. Wraps
# posix_memalign rather than malloc: the field arrays are aligned for the
# AVX2 loads, so wrapping malloc would measure nothing while appearing to
# work.
test/segments: test/segments.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) -Iinclude -o $@ $< $(LIB) \
		-Wl,--wrap=posix_memalign,--wrap=free \
		$(LLAMA_LDFLAGS) $(OV_LIB) $(LDLIBS) -lstdc++

segments: test/segments
	./test/segments

test/oom: test/oom.c cmd/bridge.o $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) -Iinclude -o $@ $< cmd/bridge.o $(LIB) 		-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc 		$(LLAMA_LDFLAGS) $(OV_LIB) $(LDLIBS) -lstdc++

oom: test/oom
	./test/oom

# ASan + UBSan over both suites. libsigil only: OpenVINO dlopens TBB with
# RTLD_DEEPBIND, which the sanitizer runtime refuses, and the logic worth
# checking is all on this side of that boundary anyway.
# Derived from SRC, never a second copy of it. As a hand-maintained list
# this silently fell behind three times -- utf8_repair.c, split.c and
# veccache.c each joined the library and the unit suite without being
# added here, and the target had been failing to link since. A sanitizer
# run that does not build is a sanitizer run nobody is doing.
SANSRC = $(SRC)
SANFLAGS = -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
           -Iinclude -Ithird_party/blake3 -DBLAKE3_NO_AVX512 -DBLAKE3_NO_AVX2 \
           -DBLAKE3_NO_SSE41 -DBLAKE3_NO_SSE2 -DBLAKE3_USE_NEON=0

sanitize:
	@mkdir -p build/san
	$(CC) $(SANFLAGS) -c test/unit.c -o build/san/unit.o
	$(CC) $(SANFLAGS) $(SANSRC) build/san/unit.o -o build/san/unit -lm
	ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1 ./build/san/unit
	$(CC) $(SANFLAGS) -std=c99 -I$(THEFT)/inc -c test/prop.c -o build/san/prop.o
	$(CC) $(SANFLAGS) $(SANSRC) build/san/prop.o $(THEFT)/build/libtheft.a \
		-o build/san/prop -lpthread -lm
	ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1 ./build/san/prop

# libFuzzer over the paths that eat untrusted bytes: the paragraph splitter,
# the hash, the trit decoder, the scan. Needs a clang whose sanitizer runtimes
# are installed -- clang-18 here; the newer build in PATH ships without them.
FUZZCC ?= clang-18
FUZZTIME ?= 120

test/fuzz_sigil: test/fuzz_sigil.c
	$(FUZZCC) -O1 -g -fsanitize=fuzzer,address,undefined -Iinclude \
		-Ithird_party/blake3 -DBLAKE3_NO_AVX512 -DBLAKE3_NO_AVX2 \
		-DBLAKE3_NO_SSE41 -DBLAKE3_NO_SSE2 -DBLAKE3_USE_NEON=0 \
		$< $(BLAKE3) src/sigil.c src/trit.c src/store.c \
		src/scan_scalar.c src/scan_x86.c src/scan_sse.c src/scan_neon.c \
		src/scan_generic.c src/scan_range.c src/simhash.c src/split.c -o $@

# The cache loader parses a file it did not necessarily write: a crash
# leaves a partial line, and recovery is the worst moment for an
# out-of-bounds read.
test/fuzz_veccache: test/fuzz_veccache.c src/veccache.c
	$(FUZZCC) -O1 -g -fsanitize=fuzzer,address,undefined -Iinclude \
		$< src/veccache.c -o $@

fuzz-veccache: test/fuzz_veccache
	./test/fuzz_veccache -max_total_time=$(FUZZTIME) test/fuzz-corpus-vc

fuzz: test/fuzz_sigil
	@mkdir -p test/fuzz-corpus
	./test/fuzz_sigil -max_total_time=$(FUZZTIME) test/fuzz-corpus

# Prove the tests can fail. A suite that passes is not evidence until you
# have watched it fail.
mutate: test/unit test/prop
	sh test/mutate.sh

# The test that separates a semantic filesystem from a hashing one.
check-semantic: test/semantic
ifeq ($(HAVE_LLAMA),1)
	@if [ -f "$(MODEL)" ]; then \
		./test/semantic "$(MODEL)"; \
	else \
		echo "SKIP: model not found at $(MODEL)"; \
		echo "  convert one with llama.cpp/convert_hf_to_gguf.py"; \
		exit 1; \
	fi
else
	@echo "SKIP: libllama.so not found under $(LLAMA_DIR)"
	@echo "  set LLAMA_DIR=/path/to/llama.cpp"
	@exit 1
endif

# Retrieval quality on a standard corpus. Needs tools/fetch-corpus.py run first.
eval: test/eval
	@if [ ! -f "$(CORPUS)" ]; then \
		echo "no corpus at $(CORPUS) — fetch one first:"; \
		echo "  python3 -m venv ~/.sigil-eval"; \
		echo "  ~/.sigil-eval/bin/pip install datasets"; \
		echo "  ~/.sigil-eval/bin/python tools/fetch-corpus.py"; \
		exit 1; \
	fi
	./test/eval "$(MODEL)" "$(CORPUS)"

corpus:
	@echo "python3 -m venv ~/.sigil-eval && \\"
	@echo "  ~/.sigil-eval/bin/pip install datasets && \\"
	@echo "  ~/.sigil-eval/bin/python tools/fetch-corpus.py"

bench: test/bench
	./test/bench

bench-mt: test/bench_mt
	./test/bench_mt

sbom:
	./tools/gen-sbom.sh > sbom.spdx.json
	@echo "wrote sbom.spdx.json"

# Built separately from `all` so a missing plan9port never breaks the library.
sigilfs:
	@if [ ! -x "$(PLAN9)/bin/9c" ]; then \
		echo "need plan9port: set PLAN9=/path/to/plan9port"; exit 1; \
	fi
	@if [ ! -d "$(LIBTAB_SRC)" ]; then \
		echo "need libtab sources: set LIBTAB_SRC=/path/to/libtab"; exit 1; \
	fi
	$(MAKE) -C cmd PLAN9=$(PLAN9) LIBTAB_SRC=$(LIBTAB_SRC)

clean:
	rm -f $(OBJ) $(LIB) $(TESTS)
	-$(MAKE) -C cmd clean 2>/dev/null || true
