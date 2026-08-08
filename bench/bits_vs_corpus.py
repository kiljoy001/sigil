#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
"""How many LSH bits does a corpus need?

Sweeps bit width against the float32 cosine ceiling. The answer is corpus
dependent, which was not obvious: Quora questions saturate at 128 bits (95.6%
of ceiling) while arXiv paragraphs need 512 to reach the same place (128 bits
gets only 76.8%).

Discrimination difficulty is what differs. Quora questions span unrelated
topics, so a coarse code separates them. Paragraphs from 223 papers in adjacent
fields are genuinely similar to each other, and telling which related paragraph
is nearest needs finer resolution.

See docs/FINDINGS.md.
"""
import json, numpy as np
rows=json.load(open('/tmp/arxiv_rows.json'))
E=np.array(json.load(open('/tmp/arx_aligned.json')),dtype=np.float32)
E/=(np.linalg.norm(E,axis=1,keepdims=True)+1e-12)
paper=np.array([r[0] for r in rows]); N=len(rows)
lut=np.array([bin(i).count('1') for i in range(256)],dtype=np.uint8)

# float ceiling
hit=0
for i in range(N):
    s=E@E[i]; s[i]=-2
    hit += (paper[int(s.argmax())]==paper[i])
ceil=hit/N
print(f"{N} paragraphs, {len(set(paper))} papers")
print(f"float32 cosine ceiling: {ceil:.4f}\n")
print(f"{'bits':>5s} {'bytes':>6s} {'recall@1':>9s} {'% ceiling':>10s} {'gain':>7s}")
prev=None
for nb in (128,256,384,512,768,1024,1536):
    rs=[]
    for t in range(2):
        rng=np.random.default_rng(0x5191c0de+t)
        Hp=np.packbits((E@rng.standard_normal((384,nb),dtype=np.float32))>0,axis=1)
        hit=0
        for i in range(N):
            d=lut[np.bitwise_xor(Hp[i],Hp)].sum(1); d[i]=1<<20
            hit += (paper[int(d.argmin())]==paper[i])
        rs.append(hit/N)
    r=np.mean(rs)
    g=f"{r-prev:+.4f}" if prev is not None else "   -"
    print(f"{nb:5d} {nb//8:6d} {r:9.4f} {100*r/ceil:9.1f}% {g:>7s}")
    prev=r
