#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
"""Does comparing summaries beat comparing originals? Measured: no.

The hypothesis was that summarizing strips boilerplate -- author lists, arXiv
headers, affiliations -- and keeps the claim, so similarity over summaries
should separate related from unrelated pairs better.

It does not. Separation changed by -0.019 cosine and -1.0 bits, both within
noise. The embedder was already handling the boilerplate. Summarizing adds
~700 ms per paragraph and a hallucination surface for no retrieval gain.

Consequence for the design: the classifier summarizes clusters AFTER the scan
groups them, rather than summarizing everything up front.
"""
import json, sys, numpy as np
sys.path.insert(0,'/tmp')
from embed_chunked import embed_paragraphs
recs=json.load(open('/tmp/summaries.json'))
orig=[r["a"] for r in recs]+[r["b"] for r in recs]
summ=[r["sa"] for r in recs]+[r["sb"] for r in recs]
n=len(recs)
Eo,_=embed_paragraphs(orig,progress=False)
Es,_=embed_paragraphs(summ,progress=False)
lut=np.array([bin(i).count('1') for i in range(256)],dtype=np.uint8)
rng=np.random.default_rng(0x5191c0de); P=rng.standard_normal((384,128),dtype=np.float32)
def stats(E,label):
    Hp=np.packbits((E@P)>0,axis=1)
    same=[];diff=[]
    for k,r in enumerate(recs):
        c=float(E[k]@E[k+n])
        h=int(lut[np.bitwise_xor(Hp[k],Hp[k+n])].sum())
        (same if r["label"]=="SAME-PAPER" else diff).append((c,h))
    sc=np.mean([x[0] for x in same]); dc=np.mean([x[0] for x in diff])
    sh=np.mean([x[1] for x in same]); dh=np.mean([x[1] for x in diff])
    print(f"{label:12s} cos: same {sc:.3f} diff {dc:.3f} sep {sc-dc:+.3f}   "
          f"ham: same {sh:5.1f} diff {dh:5.1f} sep {dh-sh:+5.1f}")
    return sc-dc, dh-sh
print(f"{n} pairs ({sum(1 for r in recs if r['label']=='SAME-PAPER')} same-paper)\n")
co,ho=stats(Eo,"originals")
cs,hs=stats(Es,"summaries")
print(f"\nsummarizing changes cosine separation by {cs-co:+.3f}, Hamming by {hs-ho:+.1f} bits")
