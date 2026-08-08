#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
"""Build a citation-grounded benchmark from unarXive.

Keeps works cited by at least three different paragraphs, so "do these two
passages engage with the same prior work" is answerable. Citation markers are
removed from the text: leaving "[16]" in place would let a lexical match
succeed without any semantic work.

Note the stream is ordered -- the first ~100k records have one context per
work, so sampling must start further in for labels to repeat.
"""
import json, collections, sys, re
from datasets import load_dataset
ds = load_dataset('saier/unarXive_citrec','default',split='train',streaming=True)
by = collections.defaultdict(list)
n = 0
for r in ds:
    t = r['text']
    # strip the citation marker itself: otherwise "[1]" is a lexical giveaway
    for a,b in sorted(r.get('marker_offsets') or [], reverse=True):
        t = t[:a] + ' ' + t[b:]
    t = ' '.join(t.split())
    if not (200 <= len(t) <= 1200): continue
    by[r['label']].append(t)
    n += 1
    if n >= 400000: break
# keep works cited by several different paragraphs
good = {k: v[:6] for k, v in by.items() if len(v) >= 3}
sys.stderr.write(f"scanned {n} contexts, {len(by)} cited works, {len(good)} cited >=3 times\n")
items = [(w, t) for w, ts in good.items() for t in ts]
sys.stderr.write(f"{len(items)} paragraphs over {len(good)} works\n")
json.dump(items[:5000], open('/tmp/citrec.json','w'))
