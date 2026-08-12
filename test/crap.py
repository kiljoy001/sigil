# SPDX-License-Identifier: GPL-3.0-or-later
"""Join lizard complexity with gcovr coverage and report CRAP per function.

CRAP = complexity^2 * (1-coverage)^3 + complexity. High complexity is
tolerable when a function is well covered, and untested code is tolerable
when it is simple; CRAP flags the intersection, which is where a bug
survives review.

Exit status is the gate. SIGIL_CRAP_MAX sets the per-function ceiling
(default 30) and SIGIL_CRAP_ALLOW the number of functions permitted over
it (default 0), so CI fails when new complexity lands untested rather than
printing a number nobody reads. Raise the allowance deliberately, in a
commit that says why -- do not silence this by widening the threshold.
"""
import csv, json, os, re, collections, sys

cov = json.load(open('build/cov/cov.json'))
covered = collections.defaultdict(dict)
for f in cov['files']:
    for ln in f['lines']:
        covered[f['file']][ln['line_number']] = ln['count'] > 0

rows = []
for r in csv.reader(open('build/cov/lz.csv')):
    if len(r) < 7:
        continue
    try:
        ccn = int(r[1])
    except ValueError:
        continue
    m = re.match(r'(.+)@(\d+)-(\d+)@(.+)', r[5])
    if not m:
        continue
    name, a, b, path = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    inrange = [v for ln, v in covered.get(path, {}).items() if a <= ln <= b]
    if not inrange:
        continue                      # not compiled with coverage
    c = sum(inrange) / len(inrange)
    rows.append((ccn ** 2 * (1 - c) ** 3 + ccn, ccn, c, name, path))

MAX = float(os.environ.get('SIGIL_CRAP_MAX', 30))
ALLOW = int(os.environ.get('SIGIL_CRAP_ALLOW', 0))

if not rows:
    # No rows means the coverage or complexity input was empty -- a broken
    # harness, not a clean codebase. Passing here would be a false green of
    # exactly the kind that let the indexing crash hide for a week.
    print('CRAP: no functions measured -- coverage or lizard output missing',
          file=sys.stderr)
    sys.exit(2)

rows.sort(reverse=True)
print(f"{'CRAP':>7} {'CCN':>4} {'cov':>6}  function")
print('-' * 66)
for crap, ccn, c, name, path in rows[:15]:
    mark = '  <--' if crap > MAX else ''
    print(f"{crap:7.1f} {ccn:4d} {100*c:5.0f}%  {name} ({path.split('/')[-1]}){mark}")

over = [r for r in rows if r[0] > MAX]
print(f"\n{len(over)} of {len(rows)} functions over CRAP {MAX:g}")

if len(over) > ALLOW:
    print(f"\nFAIL: {len(over)} over CRAP {MAX:g}, {ALLOW} allowed.",
          file=sys.stderr)
    for crap, ccn, c, name, path in over:
        print(f"  {crap:7.1f}  {name} ({path.split('/')[-1]}) "
              f"CCN {ccn}, {100*c:.0f}% covered", file=sys.stderr)
    print("\nEither test the function or make it smaller. Raising "
          "SIGIL_CRAP_ALLOW is a decision to record in the commit message, "
          "not a fix.", file=sys.stderr)
    sys.exit(1)
sys.exit(0)
