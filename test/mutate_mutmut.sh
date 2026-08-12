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
# MUTMUT_ALLOW is the surviving count permitted. It is not zero and cannot
# be. Of 510 generated mutants, 107 survive, and every one has been read:
#
#   95  main()          print intervals, progress counters, argparse help
#                       text. Testable in principle, worth nothing: they
#                       change what a log line says, not what the pipeline
#                       does.
#    7  codec name case "utf-8" -> "UTF-8". Python's codec lookup is
#                       case-insensitive, so no test can distinguish these.
#                       Equivalent by construction.
#    3  encoding=None   read_text/write_text default to the locale encoding,
#                       which on this machine and on CI is UTF-8. A real
#                       difference only under a non-UTF-8 locale, which the
#                       pipeline does not support anyway.
#    1  sum(1) -> sum(2)  scales every depth uniformly; the ordering the
#                       comparison depends on is unchanged.
#    1  pos < len -> <=  data[len(data):] is empty, decodes to "", breaks.
#
# The count came down from 130 by fixing eleven genuine gaps, not by
# raising the number. Lower it when survivors are fixed; raising it is a
# decision to record in the commit message.
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

ALLOW=${MUTMUT_ALLOW:-110}

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
survived=$(printf '%s\n' "$results" | grep -c ': survived$' || true)
notests=$(printf '%s\n' "$results" | grep -c ': no tests$' || true)
timeouts=$(printf '%s\n' "$results" | grep -c ': timeout$' || true)

printf '%s\n' "$results" | grep ': survived$' | \
	sed -E 's/.*\.x_?([a-z_]+)__mutmut.*/\1/' | sort | uniq -c | sort -rn

echo
echo "survived $survived, no-tests $notests, timeouts $timeouts (allowed: $ALLOW survivors)"

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
	echo "FAIL: $survived survivors, $ALLOW allowed." >&2
	echo "Inspect with: mutmut show <id>" >&2
	echo "Each is a gap in the tests or an equivalent mutation." >&2
	exit 1
fi
exit 0
