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
       src/simhash.c src/embed_llama.c
OBJ  = $(SRC:.c=.o)
LIB  = libsigil.a

TESTS = test/differential test/bench test/bench_mt test/semantic test/eval

CORPUS ?= test/data/corpus.txt

# sigilfs is built with plan9port's 9c/9l, not the system compiler: it links
# lib9p and libtab, both of which want plan9port's libc. libsigil stays plain
# C11 and dependency-free; store.c is the only file that sees both worlds.
PLAN9 ?= $(HOME)/Repo/plan9port
LIBTAB_SRC ?= $(HOME)/Repo/objective-9c/libtab

.PHONY: all check check-semantic eval corpus bench bench-mt clean sbom sigilfs

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

check: test/differential
	./test/differential

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
