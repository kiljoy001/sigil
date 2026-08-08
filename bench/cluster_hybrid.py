#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
"""Asserted edges cluster; similarity does not, and mixing them is harmful.

Citations do not chain -- they are sparse and stated -- so they can carry a
partition that similarity cannot:

  citation-only (30%% observed)   together 0.607  F1 0.755  largest 6
  union with sim>=0.70           together 0.718  purity 0.009  largest 1316

Adding similarity to citation edges collapses purity from 1.000 to 0.009. The
two signals measure different things: a citation points at a specific claim,
while an embedding summarizes a whole passage. Two paragraphs citing the same
work may do so for opposite reasons -- one adopting a method, one disputing a
result -- and the surrounding prose reflects those different purposes.

Caveat: purity 1.000 is an artifact of building the edges from ground-truth
labels, so only the `together` coverage figure is meaningful here. A real
corpus has extraction errors and citations to works outside it.
"""
import json, numpy as np, collections
E=np.load('/tmp/citE.npy'); works=json.load(open('/tmp/citworks.json'))
W=np.array(works); N=len(W)
byw=collections.defaultdict(list)
for i,w in enumerate(W): byw[w].append(i)
pairs=[(a,b) for idx in byw.values() for i,a in enumerate(idx) for b in idx[i+1:]]

class UF:
    def __init__(s,n): s.p=list(range(n))
    def f(s,x):
        while s.p[x]!=x: s.p[x]=s.p[s.p[x]]; x=s.p[x]
        return x
    def u(s,a,b):
        a,b=s.f(a),s.f(b)
        if a!=b: s.p[a]=b

def score(lab,name):
    sizes=collections.Counter(lab)
    tog=sum(1 for a,b in pairs if lab[a]==lab[b])/len(pairs)
    sp=tp=0
    for c,n in sizes.items():
        if n<2: continue
        idx=np.flatnonzero(lab==c); tp+=n*(n-1)//2
        cw=collections.Counter(W[idx]); sp+=sum(v*(v-1)//2 for v in cw.values())
    pur=sp/max(tp,1)
    f1=2*tog*pur/max(tog+pur,1e-9)
    print(f"  {name:32s} together {tog:.3f}  purity {pur:.3f}  F1 {f1:.3f}  "
          f"clusters {len(sizes):5d}  largest {max(sizes.values())}")

S=E@E.T; np.fill_diagonal(S,-2)

# IMPORTANT: the citation label is the ground truth, so using it directly as an
# edge would be circular. Simulate realistic sparsity: only a FRACTION of true
# citation edges are observed, as in a real corpus where reference extraction
# is incomplete and many relations are simply never cited.
rng=np.random.default_rng(11)
for frac in (0.10, 0.30):
    obs=[(a,b) for a,b in pairs if rng.random()<frac]
    uf=UF(N)
    for a,b in obs: uf.u(a,b)
    score(np.array([uf.f(i) for i in range(N)]), f"citation-only ({frac:.0%} observed)")

    # union: citation edges plus similarity
    for thr in (0.70,0.75):
        uf=UF(N)
        for a,b in obs: uf.u(a,b)
        for i in range(N):
            for j in np.flatnonzero(S[i]>=thr): uf.u(i,int(j))
        score(np.array([uf.f(i) for i in range(N)]), f"  union sim>={thr}")

    # constrained: similarity may only merge items already citation-linked
    base=UF(N)
    for a,b in obs: base.u(a,b)
    grp=np.array([base.f(i) for i in range(N)])
    for thr in (0.60,0.65):
        uf=UF(N)
        for a,b in obs: uf.u(a,b)
        for i in range(N):
            for j in np.flatnonzero(S[i]>=thr):
                if grp[i]==grp[int(j)]: uf.u(i,int(j))
        score(np.array([uf.f(i) for i in range(N)]), f"  constrained sim>={thr}")
    print()
