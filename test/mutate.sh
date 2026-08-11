#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
#
# Mutation testing: prove the tests can fail.
#
# A suite that passes is not evidence until you have watched it fail. This
# breaks the library in specific, plausible ways and checks that something
# notices. A surviving mutant is either a gap in the tests or an equivalent
# mutation -- code whose behaviour does not actually change -- and the two
# have to be told apart by hand, not assumed.
#
# The libtab suite went through this and it was worth it: the first version of
# its scaling guard passed with the fix reverted, because the test's key
# pattern never reached the code path the fix was in.
#
#	sh test/mutate.sh
#
set -e
cd "$(dirname "$0")/.."

pass=0
caught=0
survived=0

mutate() {
	name="$1"; file="$2"; from="$3"; to="$4"
	cp "$file" /tmp/mut.orig
	python3 - "$file" "$from" "$to" <<'EOF'
import sys
p, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
if a not in s:
    sys.stderr.write("MUTATION DID NOT APPLY: %s\n" % a[:60])
    sys.exit(2)
open(p, "w").write(s.replace(a, b, 1))
EOF
	if make -s libsigil.a >/dev/null 2>&1 &&
	   make -s test/unit test/prop >/dev/null 2>&1; then
		if ./test/unit >/dev/null 2>&1 && ./test/prop >/dev/null 2>&1; then
			printf '  %-44s SURVIVED\n' "$name"
			survived=$((survived + 1))
		else
			printf '  %-44s caught\n' "$name"
			caught=$((caught + 1))
		fi
	else
		printf '  %-44s (build failed, counted as caught)\n' "$name"
		caught=$((caught + 1))
	fi
	cp /tmp/mut.orig "$file"
}

echo "mutating libsigil, running unit + prop against each:"

# Identity: the hash must depend on content. Feeding a constant is what
# reload effectively did when it hashed the path instead of the text.
mutate "hash ignores content length" src/sigil.c \
	"blake3_hasher_update(&h, content, len);" \
	"blake3_hasher_update(&h, content, len > 8 ? 8 : len);"

# Store: an off-by-one in the bounds check hands back a record that was
# never written.
mutate "store_get accepts one past the end" src/store.c \
	"if (i >= st->count)" \
	"if (i > st->count)"

# Store: a growth that forgets one of the seven parallel arrays leaves a
# record with the right hash and a stale timestamp.
mutate "store_grow drops the timestamp array" src/store.c \
	"memcpy(ts,   st->timestamp, st->count * sizeof(uint32_t));" \
	"(void)ts;"

# Trits: accepting an out-of-range packed value decodes corruption as data.
mutate "trit decoder accepts corruption" src/trit.c \
	"if (!sigil_trits_valid(packed))" \
	"if (0)"

# Scan: an off-by-one on the radius returns neighbours that are not
# neighbours -- plausible output, wrong forever.
mutate "scan radius is inclusive by one too many" src/scan_scalar.c \
	"<= max_distance)" \
	"<= max_distance + 1)"

make -s libsigil.a >/dev/null 2>&1
make -s test/unit test/prop >/dev/null 2>&1

echo
echo "caught $caught, survived $survived"
[ "$survived" -eq 0 ] || {
	echo "a surviving mutant is a gap or an equivalent mutation -- decide which"
	exit 1
}
exit 0
