#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
"""Does formula AST hashing work on real corporate spreadsheets?

The claim to test: Excel's grammar is closed -- fixed function set, no
user-defined macros in the formula language -- so AST coverage should exceed
what LaTeXML managed on LaTeX (99.3%). That is an inference from the grammar,
not a measurement, and inferences in this project have a poor record.

Corpus is the Enron spreadsheets released through litigation: ~15,900 real
workbooks written by people doing their jobs, not curated examples.

Reports parse coverage, and how often identical computations recur across
different workbooks -- the latter being the actual product claim, that you can
find duplicated logic you did not know existed.
"""
import sys, os, collections, time, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
from xlsx_ast import formula_hash
warnings.filterwarnings("ignore")
import openpyxl

def scan(paths, limit_cells=200000):
    stats = collections.Counter()
    by_hash = collections.defaultdict(set)     # hash -> set of workbooks
    examples = {}
    failures = collections.Counter()
    fail_samples = []
    t0 = time.time()
    for n, p in enumerate(paths):
        try:
            wb = openpyxl.load_workbook(p, data_only=False, read_only=True)
        except Exception as e:
            stats["unreadable_workbook"] += 1
            continue
        cells = 0
        try:
            for ws in wb.worksheets:
                for row in ws.iter_rows():
                    for c in row:
                        v = c.value
                        if not isinstance(v, str) or not v.startswith("="):
                            continue
                        cells += 1
                        stats["formulas"] += 1
                        h, canon = formula_hash(v)
                        if h:
                            stats["parsed"] += 1
                            by_hash[h].add(os.path.basename(p))
                            examples.setdefault(h, v)
                        else:
                            failures[canon] += 1
                            if len(fail_samples) < 25:
                                fail_samples.append(v[:90])
                        if cells > limit_cells:
                            raise StopIteration
        except StopIteration:
            pass
        except Exception:
            stats["read_error"] += 1
        finally:
            wb.close()
        stats["workbooks"] += 1
        if (n+1) % 250 == 0:
            sys.stderr.write(f"\r  {n+1}/{len(paths)} workbooks, "
                             f"{stats['formulas']} formulas, {time.time()-t0:.0f}s")
            sys.stderr.flush()
    sys.stderr.write("\n")
    return stats, by_hash, examples, failures, fail_samples, time.time()-t0

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "/home/scott/enron/sheets"
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    paths = []
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".xlsx"):
                paths.append(os.path.join(d, f))
    paths.sort()
    paths = paths[:cap]
    print(f"scanning {len(paths)} workbooks of {root}\n")
    st, by_hash, ex, fails, samples, dt = scan(paths)

    tot, ok = st["formulas"], st["parsed"]
    print(f"workbooks read     : {st['workbooks']}  (unreadable {st['unreadable_workbook']})")
    print(f"formulas found     : {tot}")
    print(f"parsed to an AST   : {ok}  ({100*ok/max(tot,1):.2f}%)")
    print(f"distinct hashes    : {len(by_hash)}")
    print(f"time               : {dt:.0f}s  ({1000*dt/max(tot,1):.2f} ms/formula)")

    shared = {h: w for h, w in by_hash.items() if len(w) > 1}
    print(f"\nidentical computations appearing in >1 workbook: {len(shared)}")
    top = sorted(shared.items(), key=lambda kv: -len(kv[1]))[:8]
    for h, wbs in top:
        print(f"  {len(wbs):4d} workbooks  {ex[h][:70]}")

    if fails:
        print(f"\nparse failures by reason: {dict(fails)}")
        print("samples:")
        for s in samples[:8]:
            print(f"  {s}")
