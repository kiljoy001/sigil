#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
"""Retrieval measured against peer-reviewed human judgement.

Ground truth without annotation: two paragraphs citing the same work were
written by domain experts who each independently decided that work was relevant
there, and reviewers agreed. The judgements already exist in the literature, so
there is no scale problem and no reinforcement loop to run.

Uses saier/unarXive_citrec. Citation markers are stripped from the text, so
"[16]" cannot act as a lexical giveaway. Chance is ~0.09%.

Measured, 5000 contexts citing 951 works:

    float32 cosine   R@1 0.2754  R@5 0.5362   305x chance
    512-bit LSH      R@1 0.2420  R@5 0.4794   268x
    256-bit LSH      R@1 0.2030  R@5 0.4320   225x
    128-bit LSH      R@1 0.1578  R@5 0.3524   175x

Note this is a LOWER bound on precision. Citations are sparse and biased --
authors cite a handful of relevant works chosen partly by visibility and field
norms -- so a non-citation is not evidence of non-relatedness. Many of the
apparent misses are genuinely related passages that simply were not co-cited.
"""
import json, sys, collections
import numpy as np
sys.path.insert(0,'/tmp')
from embed_chunked import embed_paragraphs

items=json.load(open('/tmp/citrec.json'))
works=[w for w,_ in items]; texts=[t for _,t in items]
N=len(texts); cnt=collections.Counter(works)
L=sorted(len(t) for t in texts)
print(f"{N} citation contexts citing {len(cnt)} distinct works")
print(f"paragraph chars: min {L[0]} med {L[N//2]} max {L[-1]}")

E,nch = embed_paragraphs(texts)
print(f"embedded via {nch} chunks ({nch/N:.2f} chunks/paragraph)\n")
W=np.array(works)
lut=np.array([bin(i).count('1') for i in range(256)],dtype=np.uint8)

def recall(k,H=None):
    hit=0
    for i in range(N):
        if H is None:
            s=E@E[i]; s[i]=-2; top=np.argpartition(-s,k)[:k]
        else:
            d=lut[np.bitwise_xor(H[i],H)].sum(1); d[i]=1<<20; top=np.argpartition(d,k)[:k]
        hit += any(W[j]==W[i] for j in top)
    return hit/N

chance=float(np.mean([(cnt[w]-1)/(N-1) for w in works]))
print(f"{'method':20s} {'R@1':>8s} {'R@5':>8s} {'vs chance':>10s}")
print(f"{'random':20s} {chance:8.4f} {1-(1-chance)**5:8.4f} {'1x':>10s}")
rf=recall(1); rf5=recall(5)
print(f"{'float32 cosine':20s} {rf:8.4f} {rf5:8.4f} {rf/chance:9.0f}x")
for nb in (128,256,512):
    rng=np.random.default_rng(0x5191c0de)
    H=np.packbits((E@rng.standard_normal((384,nb),dtype=np.float32))>0,axis=1)
    r=recall(1,H)
    print(f"{str(nb)+'-bit LSH':20s} {r:8.4f} {recall(5,H):8.4f} {r/chance:9.0f}x   ({100*r/rf:.1f}% of ceiling)")
np.save('/tmp/citE.npy',E); json.dump(works,open('/tmp/citworks.json','w'))
