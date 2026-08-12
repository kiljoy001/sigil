#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
#
# Generated mutation testing for the Python pipeline, via mutmut.
#
# The other half of test/mutate_py.sh. That file holds mutants I chose: each
# one a mistake a person could plausibly make here, which makes it precise
# and reviewable but bounded by what someone thought of. This one holds
# mutants nobody chose. mutmut walks the AST and flips comparisons, swaps
# operators, changes constants and empties function bodies wherever it can,
# including in the corners no one was thinking about.
#
# Both matter, and the distinction is the point. The bug that started this
# work -- invalid UTF-8 reaching PCRE2 -- was not on anyone's list of things
# to check. A curated suite can only confirm the failures we imagined; a
# generated one is how the unimagined ones get found. On its first working
# run this found five real gaps the curated suite had missed:
#
#   * -0 (UTF-8) only tied with a bare filename instead of outranking it,
#     so deduplication could pick either.
#   * The "-0" suffix test could be replaced with nonsense and nothing failed.
#   * clean._job, the function the process pool actually calls, had no test
#     at all -- nine mutants reported "no tests".
#   * A failed write in boilerplate._job counted as a book written, which
#     inflates the only summary a 60,830-file run produces.
#   * A skipped file in plan_jobs could `break` instead of `continue`,
#     abandoning the rest of a resumed run.
#
# What this gate counts, and what it deliberately does not.
#
# main() is excluded. Not because it is untestable -- test_cli.py covers
# it -- but because what survives there is argparse help text, progress
# intervals and log wording. A test that fails when someone improves a
# message trains people to ignore the suite. The numbers those messages
# carry are a different matter and are tested: format_report() and
# exit_status() were split out of main() in every tool precisely so the
# counts, the error cap and the exit status are mutable and covered.
#
# Everything outside main() is counted. The budget is 93 across five
# modules, and each survivor has been read and classified:
#
#   23  string sentinel   mutmut replaces a literal with "XX...XX". Where
#                         the literal is a dict key or a field name the
#                         code raises rather than misbehaves, and the
#                         tests that would catch it are testing Python.
#   24  keyword arg form  parents=None vs parents=True vs the argument
#                         omitted -- all reach the same call.
#   10  codec/label case  "utf-8" -> "UTF-8". Python's codec lookup is
#                         case-insensitive: equivalent by construction.
#    3  control-char      ord(c) < 0x20 vs <= 0x20 vs < 33. The boundary
#                         characters are all substituted either way.
#   33  other             read individually; mostly slice arithmetic in
#                         _unquote where the quoted form makes the
#                         variants agree, and dict-key spellings.
#
# The count came down from 106 by fixing four real gaps -- the
# count_paragraphs boundaries at both ends, the chunk accumulator, and
# the error cap in a third module -- and by deleting a row counter in
# Manifest that was written twice and read nowhere. Dead code cannot be
# mutation-tested, and testing it would have been worse than removing it.
#
# Lower this when survivors are fixed. Raising it is a decision to record
# in the commit message.
#
#
#	sh test/mutate_mutmut.sh
#	mutmut show <id>     # what a specific survivor changed
set -e
cd "$(dirname "$0")/.."

PY=${PY:-/mnt/bulk/pyenv-sigil/bin/python}
command -v "$PY" >/dev/null 2>&1 || PY=python3
MUTMUT="$(dirname "$PY")/mutmut"
[ -x "$MUTMUT" ] || MUTMUT=$(command -v mutmut || true)
[ -n "$MUTMUT" ] || {
	echo "mutmut not installed (pip install mutmut)" >&2
	exit 2
}

ALLOW=${MUTMUT_ALLOW:-93}

# Mutants inside these functions are reported but not counted against the
# budget. Keep the list short and justified -- it is an admission that a
# function is not worth mutating, which is only true for pure output.
EXEMPT=${MUTMUT_EXEMPT:-x_main__}

rm -rf .mutmut-cache mutants 2>/dev/null || true

echo "running mutmut over tools/clean.py and tools/boilerplate.py"
"$MUTMUT" run >/dev/null 2>&1 || true

results=$("$MUTMUT" results 2>/dev/null || true)
if [ -z "$results" ]; then
	# No output means the run never got as far as mutating anything --
	# a broken harness, not a clean codebase. Reporting 0 survivors here
	# would be the same false green that let a segfault hide for a week.
	echo "mutmut produced no results -- harness broken, not a pass" >&2
	exit 2
fi

# mutmut results lists only mutants that need attention -- survivors,
# timeouts, no-tests. Killed ones are not printed, so the kill count comes
# from the run's own tally rather than from this listing.
all_survived=$(printf '%s\n' "$results" | grep ': survived$' || true)
counted=$(printf '%s\n' "$all_survived" | grep -v "$EXEMPT" || true)

survived=$(printf '%s\n' "$counted" | grep -c . || true)
exempt=$(printf '%s\n' "$all_survived" | grep -c "$EXEMPT" || true)
notests=$(printf '%s\n' "$results" | grep -c ': no tests$' || true)
timeouts=$(printf '%s\n' "$results" | grep -c ': timeout$' || true)

printf '%s\n' "$counted" | grep . | \
	sed -E 's/.*\.x_?([a-z_]+)__mutmut.*/\1/' | sort | uniq -c | sort -rn

echo
echo "survived $survived (counted), $exempt exempt in $EXEMPT,"
echo "no-tests $notests, timeouts $timeouts (allowed: $ALLOW)"

# A mutant with no test covering it at all is never acceptable, whatever
# the survivor allowance: it means a function the suite does not execute.
if [ "$notests" -gt 0 ]; then
	echo
	echo "FAIL: $notests mutants have no test covering them at all." >&2
	printf '%s\n' "$results" | grep ': no tests$' | head -10 >&2
	exit 1
fi

if [ "$survived" -gt "$ALLOW" ]; then
	echo
	echo "FAIL: $survived counted survivors, $ALLOW allowed." >&2
	printf '%s\n' "$counted" | head -20 >&2
	echo "Inspect with: mutmut show <id>" >&2
	echo "Each is a gap in the tests or an equivalent mutation." >&2
	exit 1
fi
exit 0
