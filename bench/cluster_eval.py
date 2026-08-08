#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
"""Similarity-only clustering fails on real prose. Measured, not assumed.

Connected components at a similarity threshold -- the method the design
originally specified -- has no usable operating point:

  cos>=0.60  co-cited together 0.389   largest cluster 2018 of 5000
  cos>=0.70  co-cited together 0.080   largest cluster 182

Loose thresholds chain A-B-C until half the corpus is one blob; tight ones
fragment. Mutual k-NN fails the same way (k=5: together 0.413, purity 0.002).
This is at float32, so it is not a compression problem.

Academic prose forms a continuum rather than islands, and every
transitive-linkage method chases that continuum across the whole corpus.
"""
import json, numpy as np, collections
E=np.load('/tmp/citE.npy'); works=json.load(open('/tmp/citworks.json'))
W=np.array(works); N=len(W)
byw=collections.defaultdict(list)
for i,w in enumerate(W): byw[w].append(i)
pairs=[(a,b) for idx in byw.values() for i,a in enumerate(idx) for b in idx[i+1:]]
print(f"{N} items, {len(pairs)} co-cited pairs\n")

class UF:
    def __init__(s,n): s.p=list(range(n))
    def f(s,x):
        while s.p[x]!=x: s.p[x]=s.p[s.p[x]]; x=s.p[x]
        return x
    def u(s,a,b):
        a,b=s.f(a),s.f(b)
        if a!=b: s.p[a]=b

def score(lab, name):
    sizes=collections.Counter(lab)
    together=sum(1 for a,b in pairs if lab[a]==lab[b])/len(pairs)
    # purity: of all same-cluster pairs, how many are genuinely co-cited
    sp=0; tp=0
    for c,n in sizes.items():
        if n<2: continue
        idx=np.flatnonzero(lab==c); tp+=n*(n-1)//2
        cw=collections.Counter(W[idx])
        sp+=sum(v*(v-1)//2 for v in cw.values())
    print(f"  {name:34s} together {together:.3f}  purity {sp/max(tp,1):.3f}  "
          f"clusters {len(sizes):5d}  largest {max(sizes.values())}")

S=E@E.T; np.fill_diagonal(S,-2)
# mutual k-NN: link i-j only if each is in the other's top k
for k in (2,3,5):
    top=np.argpartition(-S,k,axis=1)[:,:k]
    inset=[set(map(int,top[i])) for i in range(N)]
    uf=UF(N)
    for i in range(N):
        for j in inset[i]:
            if i in inset[j]: uf.u(i,j)
    score(np.array([uf.f(i) for i in range(N)]), f"mutual {k}-NN")
# single linkage for comparison
for thr in (0.65,0.70):
    uf=UF(N)
    for i in range(N):
        for j in np.flatnonzero(S[i]>=thr): uf.u(i,int(j))
    score(np.array([uf.f(i) for i in range(N)]), f"connected components cos>={thr}")
