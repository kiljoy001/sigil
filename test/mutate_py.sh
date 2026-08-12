#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
#
# Targeted mutation testing for the Python pipeline.
#
# This is the half of mutation testing I choose: each mutant is a mistake a
# person could plausibly make in this specific code, written down so the
# suite has to prove it can fail. The other half -- mutants nobody chose --
# is mutmut, run by test/mutate_mutmut.sh. Both are needed and neither
# replaces the other:
#
#   * A curated mutant asks "would we catch the bug we can imagine?" It is
#     precise, fast, and reviewable, but it can only cover failures someone
#     thought of. The crash that started this work was not one of those.
#
#   * A generated mutant asks "would we catch a bug nobody chose?" It finds
#     the untested branch in the corner of a file no one was thinking about,
#     at the cost of noise and runtime.
#
# Every mutant here has been verified to fail the suite. A mutant that
# survives is either a genuine gap or equivalent code -- and the two must be
# told apart by hand, not assumed. Three of these found real gaps when first
# run, and one (a surrogate filter in tools/clean.py) turned out to be
# unreachable code, which was then deleted rather than tested.
#
#	sh test/mutate_py.sh
set -e
cd "$(dirname "$0")/.."

PY=${PY:-/mnt/bulk/pyenv-sigil/bin/python}
command -v "$PY" >/dev/null 2>&1 || PY=python3

caught=0
survived=0
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# mutate <name> <file> <python-expression-applying-the-edit>
mutate() {
	name="$1"; file="$2"; edit="$3"
	cp "$file" "$tmp/orig"
	if ! "$PY" - "$file" "$edit" <<'EOF'
import sys
path, edit = sys.argv[1], sys.argv[2]
src = open(path).read()
new = eval(edit, {"s": src})
if new == src:
    sys.stderr.write("MUTATION DID NOT APPLY\n")
    sys.exit(2)
open(path, "w").write(new)
EOF
	then
		printf '  %-52s DID NOT APPLY\n' "$name"
		cp "$tmp/orig" "$file"
		survived=$((survived + 1))
		return
	fi

	# A mutant in the C guard has to be caught by the C suite as well as
	# by the Python differential tests -- and the C suite needs a rebuild
	# to see it. A build failure counts as caught: the compiler refusing
	# the mutation is a detection, just an early one.
	ok=0
	case "$file" in
	src/*.c|cmd/*.c)
		if make -s libsigil.a >/dev/null 2>&1 &&
		   make -s test/unit >/dev/null 2>&1 &&
		   ./test/unit >/dev/null 2>&1 &&
		   "$PY" -m pytest tools/tests -q >/dev/null 2>&1; then
			ok=1
		fi
		;;
	*)
		if "$PY" -m pytest tools/tests -q >/dev/null 2>&1; then
			ok=1
		fi
		;;
	esac

	if [ "$ok" -eq 1 ]; then
		printf '  %-52s SURVIVED\n' "$name"
		survived=$((survived + 1))
	else
		printf '  %-52s caught\n' "$name"
		caught=$((caught + 1))
	fi
	cp "$tmp/orig" "$file"
}

echo "curated mutants, tools/clean.py:"

# The original bug. Passing bytes through undecoded is what shipped invalid
# UTF-8 into PCRE2 and segfaulted the indexer.
mutate "decode with surrogateescape (the original bug)" tools/clean.py \
	's.replace("    text = _decode(data)", "    text = data.decode(\"utf-8\", errors=\"surrogateescape\")")'

# The lossy alternative: valid output, destroyed content. 0x92 becomes
# U+FFFD instead of an apostrophe, and every affected paragraph embeds wrong.
mutate "U+FFFD replacement instead of CP1252" tools/clean.py \
	's.replace("    text = _decode(data)", "    text = data.decode(\"utf-8\", errors=\"replace\")")'

mutate "forget NUL removal" tools/clean.py \
	's.replace("    text = text.replace(\"\\x00\", \"\")\n", "", 1)'

mutate "forget CRLF normalisation" tools/clean.py \
	's.replace("    text = text.replace(\"\\r\\n\", \"\\n\").replace(\"\\r\", \"\\n\")\n", "", 1)'

mutate "forget BOM stripping" tools/clean.py \
	's.replace("    if text.startswith(\"\ufeff\"):", "    if False:", 1)'

# The C guard, checked against the Python implementation by the
# differential suite. Accepting overlong forms is the classic UTF-8 filter
# bypass, and here it would also let a sequence through to PCRE2 that
# Python had already rejected -- the two must not disagree.
mutate "C: accept overlong 2-byte forms (0xC0 lead)" src/utf8_repair.c \
	's.replace("if (p[0] < 0xC2)", "if (p[0] < 0xC0)")'

mutate "C: accept surrogate halves" src/utf8_repair.c \
	's.replace("if (p[0] == 0xED && p[1] >= 0xA0)", "if (0)")'

mutate "C: U+FFFD instead of CP1252 transcoding" src/utf8_repair.c \
	's.replace("if (b >= 0x80 && b <= 0x9F && cp1252_c1[b - 0x80] != 0)\n\t\t\t\tcp = cp1252_c1[b - 0x80];", "cp = 0xFFFD;")'

# Resume logic: a wrong comparison silently leaves stale output in place.
mutate "resume skips when source is NEWER" tools/clean.py \
	's.replace("d.stat().st_mtime >= s.stat().st_mtime", "d.stat().st_mtime <= s.stat().st_mtime")'

mutate "write only changed files" tools/clean.py \
	's.replace("    if not dry_run:\n", "    if not dry_run and changed:\n", 1)'

mutate "dry-run writes anyway" tools/clean.py \
	's.replace("    if not dry_run:\n", "    if True:\n", 1)'

echo
echo "curated mutants, tools/boilerplate.py:"

mutate "only match THE, not THIS" tools/boilerplate.py \
	's.replace("(?:THE|THIS)", "THE")'

mutate "require a space after the asterisks" tools/boilerplate.py \
	"s.replace(chr(92)+chr(115)+chr(42)+'START', ' START', 1)"

mutate "END searched from 0, not after START" tools/boilerplate.py \
	's.replace("_END.search(text, body_from)", "_END.search(text)")'

mutate "drop the malformed-slice guard" tools/boilerplate.py \
	's.replace("    if body_to <= body_from:\n        return text                      # malformed; keep everything\n", "", 1)'

mutate "strip Produced-by even if nothing remains" tools/boilerplate.py \
	's.replace("        if candidate.strip():\n            body = candidate", "        body = candidate")'

# Deduplication: ranking encoding above revision indexes a superseded
# edition, with no visible symptom at all.
mutate "rank encoding above revision" tools/boilerplate.py \
	's.replace("    return (depth, enc, str(path))", "    return (enc, depth, str(path))")'

mutate "drop the tie-breaker (nondeterministic choice)" tools/boilerplate.py \
	's.replace("    return (depth, enc, str(path))", "    return (depth, enc)")'

mutate "book_id takes the first numeric dir, not the last" tools/boilerplate.py \
	's.replace("    for part in reversed(path.parent.parts):", "    for part in path.parent.parts:")'

mutate "drop unnumbered files instead of keeping them" tools/boilerplate.py \
	's.replace("] + unnumbered", "]")'

mutate "keep_duplicates ignored" tools/boilerplate.py \
	's.replace("    if keep_duplicates:\n        return list(all_files)\n", "", 1)'

# Leave the tree as we found it: the C mutants rebuilt libsigil.
make -s libsigil.a >/dev/null 2>&1 || true
make -s test/unit >/dev/null 2>&1 || true

echo
echo "caught $caught, survived $survived"
[ "$survived" -eq 0 ] || {
	echo
	echo "A surviving mutant is a gap in the tests or equivalent code."
	echo "Decide which by hand -- do not delete the mutant to make this pass."
	exit 1
}
exit 0
