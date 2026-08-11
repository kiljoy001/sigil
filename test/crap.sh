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
set -e
cd "$(dirname "$0")/.."
VENV=${VENV:-/mnt/bulk/tools-venv}
R=$(pwd)
rm -rf build/cov && mkdir -p build/cov && cd build/cov
SRC="../../third_party/blake3/blake3.c ../../third_party/blake3/blake3_dispatch.c \
     ../../third_party/blake3/blake3_portable.c ../../src/sigil.c ../../src/trit.c \
     ../../src/store.c ../../src/scan_scalar.c ../../src/scan_x86.c ../../src/scan_sse.c \
     ../../src/scan_neon.c ../../src/scan_generic.c ../../src/scan_range.c \
     ../../src/simhash.c ../../src/embed_llama.c ../../cmd/bridge.c"
F="-O0 -g --coverage -I$R/include -I$R/third_party/blake3 -DBLAKE3_NO_AVX512 \
   -DBLAKE3_NO_AVX2 -DBLAKE3_NO_SSE41 -DBLAKE3_NO_SSE2 -DBLAKE3_USE_NEON=0"
for f in $SRC; do gcc $F -c "$f" -o "$(basename "$f" .c).o"; done
LIBOBJ=$(ls *.o | tr '\n' ' ')
L="-L$HOME/llama.cpp/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$HOME/llama.cpp/build/bin -lstdc++ -lm"
gcc $F -c "$R/test/unit.c"   -o u.o && gcc --coverage -o unit u.o $LIBOBJ $L
gcc $F -c "$R/test/bridge.c" -o b.o && gcc --coverage -o brt  b.o $LIBOBJ $L
./unit >/dev/null; ./brt >/dev/null
cd "$R"
$VENV/bin/lizard src/ cmd/ --csv 2>/dev/null > build/cov/lz.csv
$VENV/bin/gcovr --root . --object-directory build/cov --json build/cov/cov.json >/dev/null 2>&1
python3 test/crap.py
