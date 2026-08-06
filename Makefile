CC      ?= cc
CFLAGS  ?= -O2 -g -std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion
CPPFLAGS = -Iinclude
LDFLAGS ?=

SRC  = src/sigil.c src/trit.c src/store.c src/scan_scalar.c src/scan_avx2.c
OBJ  = $(SRC:.c=.o)
LIB  = libsigil.a

TESTS = test/differential test/bench

.PHONY: all check bench clean sbom

all: $(LIB)

$(LIB): $(OBJ)
	$(AR) rcs $@ $^

%.o: %.c include/sigil.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

test/differential: test/differential.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) -o $@ $(LDFLAGS)

test/bench: test/bench.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) -o $@ $(LDFLAGS)

check: test/differential
	./test/differential

bench: test/bench
	./test/bench

sbom:
	./tools/gen-sbom.sh > sbom.spdx.json
	@echo "wrote sbom.spdx.json"

clean:
	rm -f $(OBJ) $(LIB) $(TESTS)
