#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
#
# The store round trip: index, commit, restart, and the records are back.
#
# This did not exist when store_commit was ported to libtab's streaming
# writer -- 269 C checks were green and not one of them would have noticed
# a commit that wrote nothing, because nothing exercised persist.c through
# the server. The port was verified by hand instead; this is that
# verification, kept.
#
# What is asserted, in order of what would actually go wrong:
#   * commit produces a file (the writer publishes by renaming a temp
#     file, so a failed commit leaves NO file, not a partial one)
#   * a fresh server loads every record the old one committed
#   * the params row survives -- a store that reloads records but loses
#     model/seed/bits would happily serve bits from the wrong hyperplanes
#
#	sh test/store.sh
set -e
cd "$(dirname "$0")/.."

PLAN9=${PLAN9:-$HOME/Repo/plan9port}
PATH=$PLAN9/bin:$PATH
export PLAN9 PATH

work=$(mktemp -d)
export NAMESPACE="$work/ns"
mkdir -p "$NAMESPACE" "$work/corpus"
srv=store$$
fail=0
pid=

cleanup() {
	[ -n "$pid" ] && kill "$pid" 2>/dev/null
	rm -rf "$work"
}
trap cleanup EXIT

# Two paragraphs over the 40-byte minimum, separated by a blank line, so
# the splitter emits exactly two records.
cat > "$work/corpus/book.txt" <<'EOF'
The quick brown fox jumps over the lazy dog and keeps on running
through the long meadow toward the river below the hill.

A second paragraph, long enough to clear the minimum paragraph size,
mentioning "quoted text", a #hashtag and the word nil in running prose.
EOF

./cmd/sigilfs -s "$srv" -f "$work/store.tab" &
pid=$!
sleep 1

printf 'mount books %s/corpus\nindex\ncommit\n' "$work" \
	| 9p -a "unix!$NAMESPACE/$srv" write ctl

n=$(9p -a "unix!$NAMESPACE/$srv" read stats | awk '$1=="records"{print $2}')
kill "$pid"; wait "$pid" 2>/dev/null || true
pid=

[ "$n" = 2 ] || { echo "FAIL: indexed $n records, wanted 2"; fail=1; }
[ -s "$work/store.tab" ] || { echo "FAIL: commit left no store"; fail=1; }
grep -q '^path=!params$' "$work/store.tab" \
	|| { echo "FAIL: params row missing from store"; fail=1; }

# The row header is not the assertion -- the values are. Every mismatch
# check on the load side skips when its cell is nil, so a params row that
# exists but carries nothing disables all three: such a store reloads
# happily under a different model, width or seed and serves bits from the
# wrong hyperplanes. A mutant that wrote the header and no values survived
# a grep for the header alone.
params=$(sed -n '/^path=!params$/,/^$/p' "$work/store.tab")
for cell in para hash lsh; do
	echo "$params" | grep -q "^	$cell=..*$" \
		|| { echo "FAIL: params row has no $cell value"; fail=1; }
done

# The reload, which is the half that persistence exists for.
./cmd/sigilfs -s "$srv" -f "$work/store.tab" &
pid=$!
sleep 1

m=$(9p -a "unix!$NAMESPACE/$srv" read stats | awk '$1=="records"{print $2}')
kill "$pid"; wait "$pid" 2>/dev/null || true
pid=

[ "$m" = "$n" ] || { echo "FAIL: committed $n records, reloaded $m"; fail=1; }

if [ "$fail" = 0 ]; then
	echo "PASS: store round trip ($n records committed and reloaded)"
fi
exit "$fail"
