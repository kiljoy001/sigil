#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
"""How to represent a whole paper for corpus-wide "potentially related".

Two candidates, measured by retrieving the right paper from half its own
paragraphs (223 arXiv papers, chance 0.0045):

  doc-average    0.7713   one sigil per paper
  max-paragraph  0.7937   full paragraph scan, ~30x the work

2.2 points for 30x the cost. The averaging concern -- that pooling a paper's
paragraphs blurs it toward a field-wide topic centroid -- did not materialize:
mean cosine between different papers under averaging is 0.199, so papers stay
well separated.

This validates the para=0 document sigil: the coarse tier scans one record per
paper, cheap enough for a very large corpus.
"""
import json, numpy as np, collections
rows=json.load(open('/tmp/arxiv_rows.json'))
E=np.array(json.load(open('/tmp/arx_aligned.json')),dtype=np.float32)
E/=(np.linalg.norm(E,axis=1,keepdims=True)+1e-12)
paper=[r[0] for r in rows]
by=collections.defaultdict(list)
for i,p in enumerate(paper): by[p].append(i)
papers=[p for p,v in by.items() if len(v)>=6]
print(f"{len(papers)} papers with >=6 paragraphs")

# split each paper in half: query half vs index half
qa,ib={},{}
for p in papers:
    v=by[p]; h=len(v)//2
    qa[p]=v[:h]; ib[p]=v[h:]

# representation A: average of the index half
avg=np.zeros((len(papers),E.shape[1]),dtype=np.float32)
for k,p in enumerate(papers):
    avg[k]=E[ib[p]].mean(0)
avg/=(np.linalg.norm(avg,axis=1,keepdims=True)+1e-12)

hits_avg=hits_max=0
for k,p in enumerate(papers):
    q=E[qa[p]].mean(0); q/=(np.linalg.norm(q)+1e-12)
    # average representation
    s=avg@q
    hits_avg += (papers[int(s.argmax())]==p)
    # max-over-paragraphs representation
    best=np.full(len(papers),-2.0,dtype=np.float32)
    for kk,pp in enumerate(papers):
        best[kk]=float((E[ib[pp]]@q).max())
    hits_max += (papers[int(best.argmax())]==p)
n=len(papers)
print(f"\nretrieving the right paper from half its paragraphs:")
print(f"  doc-average  : {hits_avg/n:.4f}")
print(f"  max-paragraph: {hits_max/n:.4f}")
print(f"  chance       : {1/n:.4f}")

# how distinguishable are papers under each representation?
S=avg@avg.T; np.fill_diagonal(S,-2)
print(f"\nmean cosine between different papers (doc-average): {S[S>-2].mean():.3f}")
print("  (high means averaging blurred them toward a field centroid)")
