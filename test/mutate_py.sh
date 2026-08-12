#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
#
# Targeted mutation testing for the Python pipeline and the C UTF-8 guard.
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
# The mutants live in test/mutants.py as data. They were shell arguments
# here originally, which put every anchor string through sh quoting, then
# python -c quoting, then str.replace -- and three of them silently failed
# to apply. A mutant that does not apply tests nothing, so a missing or
# ambiguous anchor is now a failure rather than a warning.
#
# A surviving mutant is either a genuine gap or an equivalent mutation, and
# the two must be told apart by hand. Several here found real gaps when
# first written; one found unreachable code, which was deleted rather than
# tested.
#
#	sh test/mutate_py.sh
set -e
cd "$(dirname "$0")/.."

PY=${PY:-/mnt/bulk/pyenv-sigil/bin/python}
command -v "$PY" >/dev/null 2>&1 || PY=python3

exec "$PY" - "$PY" <<'EOF'
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "test")
from mutants import MUTANTS

PY = sys.argv[1]


def run(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def suite_passes(path):
    """A C mutant needs a rebuild before the suite can see it. A build
    failure counts as caught -- the compiler refusing the mutation is a
    detection, just an early one."""
    if path.startswith(("src/", "cmd/")):
        return (run("make -s libsigil.a")
                and run("make -s test/unit")
                and run("./test/unit")
                and run(f"{PY} -m pytest tools/tests -q"))
    return run(f"{PY} -m pytest tools/tests -q")


caught = survived = broken = 0
current = None

for name, path, old, new in MUTANTS:
    if path != current:
        print(f"\ncurated mutants, {path}:")
        current = path

    src = Path(path).read_text()

    # An anchor that is missing, or matches more than once, means the
    # mutant is not testing what its name says. Fail rather than warn:
    # three of these were silently not applying before.
    n = src.count(old)
    if n != 1:
        print(f"  {name:52} ANCHOR x{n} -- FIX THE MUTANT")
        broken += 1
        continue

    try:
        Path(path).write_text(src.replace(old, new, 1))
        if suite_passes(path):
            print(f"  {name:52} SURVIVED")
            survived += 1
        else:
            print(f"  {name:52} caught")
            caught += 1
    finally:
        Path(path).write_text(src)

# The C mutants rebuilt libsigil; leave the tree as we found it.
run("make -s libsigil.a")
run("make -s test/unit")

print(f"\ncaught {caught}, survived {survived}, broken {broken}")
if survived:
    print("\nA surviving mutant is a gap in the tests or equivalent code.")
    print("Decide which by hand -- do not delete the mutant to make this pass.")
if broken:
    print("\nA broken mutant tests nothing. Fix its anchor in test/mutants.py.")
sys.exit(1 if (survived or broken) else 0)
EOF
