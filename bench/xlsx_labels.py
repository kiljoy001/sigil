#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
"""Does similarity work on the prose layer of a spreadsheet?

Formulas get exact identity (xlsx_ast.py). Labels and headers are the other
layer, and this is where similarity has to earn its place -- on text that is
terse, abbreviated, and often not sentences ("Q3 Rev", "Ttl", "MMBTU/D").
MiniLM was trained on natural language and did badly on short notation-heavy
strings in earlier tests, so this can fail.

Ground truth is workbook authorship: Enron filenames carry the employee whose
mailbox the file came from, and one person's spreadsheets share subject matter.
Weak, but external and free -- the same reasoning as same-paper retrieval.
"""
import os, sys, re, json, random, collections, subprocess, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
from xlsx_text import extract
from embed_chunked import embed_paragraphs

ROOT = "/home/scott/enron/sheets"

def owner(fn):
    m = re.match(r'([a-z]+_[a-z]+)__', fn)
    return m.group(1) if m else None

if __name__ == "__main__":
    n_owners = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    files = [f for f in os.listdir(ROOT) if f.endswith(".xlsx")]
    by = collections.defaultdict(list)
    for f in files:
        o = owner(f)
        if o: by[o].append(f)
    owners = [o for o, v in by.items() if len(v) >= per]
    random.seed(5); owners = sorted(random.sample(owners, min(n_owners, len(owners))))

    docs, labs = [], []
    for o in owners:
        for f in sorted(by[o])[:per]:
            try:
                groups = extract(os.path.join(ROOT, f))
            except Exception:
                continue
            txt = " | ".join(t for _, k, _, t in groups if k in ("row", "sheet"))
            if len(txt) < 60:
                continue
            docs.append(txt[:1500]); labs.append(o)
    print(f"{len(docs)} workbooks from {len(set(labs))} owners")
    if len(docs) < 20:
        sys.exit("not enough usable workbooks")

    E, nch = embed_paragraphs(docs, progress=False)
    L = np.array(labs); N = len(docs)
    lut = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

    def recall(k, H=None):
        hit = 0
        for i in range(N):
            if H is None:
                s = E @ E[i]; s[i] = -2; top = np.argpartition(-s, k)[:k]
            else:
                d = lut[np.bitwise_xor(H[i], H)].sum(1); d[i] = 1 << 20
                top = np.argpartition(d, k)[:k]
            hit += any(L[j] == L[i] for j in top)
        return hit / N

    cnt = collections.Counter(labs)
    chance = float(np.mean([(cnt[o]-1)/(N-1) for o in labs]))
    print(f"\n{'method':16s} {'R@1':>8s} {'R@5':>8s} {'vs chance':>10s}")
    print(f"{'random':16s} {chance:8.4f} {1-(1-chance)**5:8.4f} {'1x':>10s}")
    rf = recall(1)
    print(f"{'float32':16s} {rf:8.4f} {recall(5):8.4f} {rf/chance:9.0f}x")
    for nb in (128, 256):
        rng = np.random.default_rng(0x5191c0de)
        H = np.packbits((E @ rng.standard_normal((E.shape[1], nb), dtype=np.float32)) > 0, axis=1)
        r = recall(1, H)
        print(f"{str(nb)+'-bit':16s} {r:8.4f} {recall(5,H):8.4f} {r/chance:9.0f}x   "
              f"({100*r/rf:.1f}% of ceiling)")
