# sigilfs: built with plan9port's 9c/9l.
#
# libthread owns main, so sigilfs defines threadmain. Link order matters:
# libtab pulls in lib9pclient, which needs symbols from libthread.

# Default to the source tree, not /usr/local/plan9: the installed copy may be
# older and lack the lib9p 64-bit wstat fix. See docs/PLAN9PORT-BUG.md.
PLAN9      ?= $(HOME)/Repo/plan9port
LIBTAB_SRC ?= $(HOME)/Repo/objective-9c/libtab
MONOCYPHER ?= $(HOME)/Repo/objective-9c

CC9 = $(PLAN9)/bin/9c
LD9 = $(PLAN9)/bin/9l

# Two compilers on purpose. Everything here is plan9port C except bridge.c,
# which is the only file that sees libsigil: plan9port's libc.h and C11's
# stdint.h define the same typedefs and collide, so the two worlds meet
# through plain-typed functions and an opaque pointer.
SRC  = sigilfs.c fs.c ctl.c store.c trace.c persist.c index.c
OBJ  = $(SRC:.c=.o) bridge.o

SIGIL_ROOT = ..

# libsigil.a carries the llama.cpp embedder when built with SIGIL_WITH_LLAMA,
# so sigilfs needs those shared objects at link time. Leave LLAMA_DIR unset to
# build without an embedder: records then get the byte-shingle fallback, which
# is not semantic, and /stats says so.
LLAMA_DIR ?= $(HOME)/llama.cpp
ifneq ($(wildcard $(LLAMA_DIR)/build/bin/libllama.so),)
  LLAMA_LIBS = -L$(LLAMA_DIR)/build/bin -lllama -lggml -lggml-base \
               -Wl,-rpath,$(LLAMA_DIR)/build/bin -lstdc++ -lm
endif

TABSRC = $(wildcard $(LIBTAB_SRC)/*.c) $(MONOCYPHER)/monocypher.c
TABOBJ = $(notdir $(TABSRC:.c=.o))

all: sigilfs

sigilfs: $(OBJ) libtab.a $(SIGIL_ROOT)/libsigil.a
	$(LD9) -o $@ $(OBJ) libtab.a $(SIGIL_ROOT)/libsigil.a \
		-l9p -l9pclient -lthread $(LLAMA_LIBS)

# System compiler, C11, no plan9port headers.
bridge.o: bridge.c $(SIGIL_ROOT)/include/sigil.h
	cc -O2 -std=c11 -Wall -Wextra -c bridge.c -o $@ \
		-I$(SIGIL_ROOT)/include -I$(SIGIL_ROOT)/third_party/blake3

$(SIGIL_ROOT)/libsigil.a:
	$(MAKE) -C $(SIGIL_ROOT) libsigil.a

# Vendored build of libtab from source: it ships no library target of its own.
libtab.a: $(TABSRC)
	@for f in $(TABSRC); do \
		$(CC9) -I$(LIBTAB_SRC) -I$(MONOCYPHER) -c $$f || exit 1; \
	done
	ar rcs $@ $(TABOBJ)

%.o: %.c sigilfs.h
	$(CC9) -I$(LIBTAB_SRC) -c $<

clean:
	rm -f *.o *.a sigilfs

.PHONY: all clean
