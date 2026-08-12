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
SRC  = sigilfs.c fs.c ctl.c store.c trace.c persist.c index.c similar.c
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

# Same for OpenVINO. Default to the native C++ runtime archive (see the
# top-level Makefile for why the pip wheel is no longer the default);
# OPENVINO_LIB=<path to libopenvino.so.<ver>> still overrides for a wheel.
# DT_RPATH (--disable-new-dtags) so the oneAPI environment's stale
# LD_LIBRARY_PATH TBB cannot substitute itself at run time.
OPENVINO_NATIVE ?= /mnt/bulk/openvino-native/current
ifneq ($(OPENVINO_LIB),)
  OV_LIBS = $(OPENVINO_LIB) -Wl,--disable-new-dtags \
            -Wl,-rpath,$(dir $(OPENVINO_LIB)) -lstdc++ -lm
else ifneq ($(wildcard $(OPENVINO_NATIVE)/runtime/lib/intel64/libopenvino.so),)
  OV_LIBS = -L$(OPENVINO_NATIVE)/runtime/lib/intel64 -lopenvino \
            -Wl,--disable-new-dtags \
            -Wl,-rpath,$(OPENVINO_NATIVE)/runtime/lib/intel64 \
            -Wl,-rpath,$(OPENVINO_NATIVE)/runtime/3rdparty/tbb/lib \
            -lstdc++ -lm
endif

# Exclude stub_9pclient.c: it is a shim for callers that link libauth but not
# lib9pclient (py-libtab), and it defines fsopen/fscreate/fswrite -- the same
# names as this server's 9P callbacks in fs.c. sigilfs links the real
# lib9pclient, so the stub both collides and is unnecessary.
TABSRC = $(filter-out %/stub_9pclient.c,$(wildcard $(LIBTAB_SRC)/*.c)) \
         $(MONOCYPHER)/monocypher.c
TABOBJ = $(notdir $(TABSRC:.c=.o))

all: sigilfs

sigilfs: $(OBJ) libtab.a $(SIGIL_ROOT)/libsigil.a
	$(LD9) -o $@ $(OBJ) libtab.a $(SIGIL_ROOT)/libsigil.a \
		-l9p -l9pclient -lthread $(LLAMA_LIBS) $(OV_LIBS)

# The TCP server, built without libthread.
#
# History: this target was created on the theory that libthread's stack
# switching underneath OpenVINO's TBB pool caused the indexing segfault.
# That was wrong -- the crash was invalid UTF-8 reaching PCRE2 inside
# openvino_tokenizers (undefined behaviour, ASLR-dependent symptom), fixed
# in src/embed_openvino.cpp. Measured: the no-libthread build crashed at
# the same rate as the libthread one until the UTF-8 fix, and both are
# clean after it. See docs/FINDINGS.md.
#
# The target stays because it is still occasionally useful to run the
# server without libthread (plain gdb/perf sessions are simpler), not
# because libthread is unsafe. Only tpost.c in lib9p needs libthread, and
# that is the -s/-m service-posting path; srv() over a socket does not.
#
# -l9pclient is still here because libtab references nsmount for its
# tab_open_dial path, which sigil never calls.
sigilfs-tcp: $(OBJ) libtab.a $(SIGIL_ROOT)/libsigil.a
	$(LD9) -o $@ $(OBJ) libtab.a $(SIGIL_ROOT)/libsigil.a \
		-l9p -l9pclient $(LLAMA_LIBS) $(OV_LIBS)

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

sigilfs-tcp: CPPFLAGS += -DSIGIL_NO_LIBTHREAD

%.o: %.c sigilfs.h
	$(CC9) -I$(LIBTAB_SRC) -c $<

clean:
	rm -f *.o *.a sigilfs

.PHONY: all clean
