#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# CRAP = complexity^2 * (1-coverage)^3 + complexity.
#
# High complexity is tolerable when a function is well covered; untested code
# is tolerable when it is simple. CRAP flags the intersection, which is where
# a bug survives review. Anything over 30 is worth either a test or a smaller
# function.
#
#	sh test/crap.sh
#
# Runs in CI, so it must work with nothing installed but a compiler and the
# two Python tools. Points worth knowing:
#
#   * No llama.cpp. src/embed_llama.c compiles its not-built stub when
#     SIGIL_WITH_LLAMA is absent, so bridge.c's call to
#     sigil_embedder_llama() links against that. Depending on a GPU runtime
#     to measure coverage would mean coverage never ran anywhere but here.
#
#   * lizard and gcovr come from $VENV if it exists, otherwise from PATH.
#     The old hardcoded /mnt/bulk/tools-venv only existed on one machine.
#
#   * test/crap.py exits non-zero when a function is over the threshold.
#     SIGIL_CRAP_MAX and SIGIL_CRAP_ALLOW tune the gate.
set -e
cd "$(dirname "$0")/.."
R=$(pwd)

VENV=${VENV:-/mnt/bulk/tools-venv}
if [ -x "$VENV/bin/lizard" ] && [ -x "$VENV/bin/gcovr" ]; then
	LIZARD="$VENV/bin/lizard"; GCOVR="$VENV/bin/gcovr"
else
	LIZARD=$(command -v lizard || true)
	GCOVR=$(command -v gcovr || true)
fi
if [ -z "$LIZARD" ] || [ -z "$GCOVR" ]; then
	echo "need lizard and gcovr (pip install lizard gcovr)" >&2
	exit 2
fi

rm -rf build/cov && mkdir -p build/cov && cd build/cov

SRC="../../third_party/blake3/blake3.c ../../third_party/blake3/blake3_dispatch.c \
     ../../third_party/blake3/blake3_portable.c ../../src/sigil.c ../../src/trit.c \
     ../../src/store.c ../../src/scan_scalar.c ../../src/scan_x86.c ../../src/scan_sse.c \
     ../../src/scan_neon.c ../../src/scan_generic.c ../../src/scan_range.c \
     ../../src/simhash.c ../../src/embed_llama.c ../../src/utf8_repair.c \
     ../../cmd/bridge.c"
F="-O0 -g --coverage -I$R/include -I$R/third_party/blake3 -DBLAKE3_NO_AVX512 \
   -DBLAKE3_NO_AVX2 -DBLAKE3_NO_SSE41 -DBLAKE3_NO_SSE2 -DBLAKE3_USE_NEON=0"

for f in $SRC; do gcc $F -c "$f" -o "$(basename "$f" .c).o"; done
LIBOBJ=$(ls *.o | tr '\n' ' ')

# -lstdc++ is not needed without OpenVINO; -lm is, for the embedder maths.
L="-lm"

gcc $F -c "$R/test/unit.c"         -o u.o && gcc --coverage -o unit u.o $LIBOBJ $L
gcc $F -c "$R/test/bridge.c"       -o b.o && gcc --coverage -o brt  b.o $LIBOBJ $L
gcc $F -c "$R/test/differential.c" -o d.o && gcc --coverage -o diff d.o $LIBOBJ $L

# The OOM test drives bridge.c's allocation-failure paths, which are a
# meaningful slice of its branches -- omitting it understates coverage on
# exactly the code that is hardest to reach.
gcc $F -c "$R/test/oom.c" -o o.o && \
	gcc --coverage -o oomt o.o $LIBOBJ \
	    -Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc $L

./unit >/dev/null
./brt  >/dev/null
./diff >/dev/null
./oomt >/dev/null

cd "$R"
"$LIZARD" src/ cmd/ --csv 2>/dev/null > build/cov/lz.csv
"$GCOVR" --root . --object-directory build/cov --json build/cov/cov.json \
	>/dev/null 2>&1
python3 test/crap.py
