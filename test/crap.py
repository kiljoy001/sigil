# SPDX-License-Identifier: GPL-3.0-or-later
"""Join lizard complexity with gcovr coverage and report CRAP per function."""
import csv, json, re, collections, sys

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

rows.sort(reverse=True)
print(f"{'CRAP':>7} {'CCN':>4} {'cov':>6}  function")
print('-' * 66)
for crap, ccn, c, name, path in rows[:15]:
    mark = '  <--' if crap > 30 else ''
    print(f"{crap:7.1f} {ccn:4d} {100*c:5.0f}%  {name} ({path.split('/')[-1]}){mark}")
over = [r for r in rows if r[0] > 30]
print(f"\n{len(over)} of {len(rows)} functions over CRAP 30")
sys.exit(0)
