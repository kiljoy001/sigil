CC      ?= cc
CFLAGS  ?= -O2 -g -std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion
CPPFLAGS = -Iinclude
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

SRC  = src/sigil.c src/trit.c src/store.c src/scan_scalar.c src/scan_avx2.c \
       src/simhash.c src/embed_llama.c
OBJ  = $(SRC:.c=.o)
LIB  = libsigil.a

TESTS = test/differential test/bench test/semantic test/eval

CORPUS ?= test/data/corpus.txt

.PHONY: all check check-semantic eval corpus bench clean sbom

all: $(LIB)

$(LIB): $(OBJ)
	$(AR) rcs $@ $^

%.o: %.c include/sigil.h include/sigil_embed.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

test/differential: test/differential.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) -o $@ $(LDFLAGS) $(LDLIBS)

test/bench: test/bench.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) -o $@ $(LDFLAGS) $(LDLIBS)

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

sbom:
	./tools/gen-sbom.sh > sbom.spdx.json
	@echo "wrote sbom.spdx.json"

clean:
	rm -f $(OBJ) $(LIB) $(TESTS)
