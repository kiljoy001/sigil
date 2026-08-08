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

SRC  = sigilfs.c fs.c ctl.c store.c trace.c
OBJ  = $(SRC:.c=.o)

TABSRC = $(wildcard $(LIBTAB_SRC)/*.c) $(MONOCYPHER)/monocypher.c
TABOBJ = $(notdir $(TABSRC:.c=.o))

all: sigilfs

sigilfs: $(OBJ) libtab.a
	$(LD9) -o $@ $(OBJ) libtab.a -l9p -l9pclient -lthread

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
