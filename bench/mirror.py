#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
"""Why an embedding inverts: does cosine just track token overlap?

Compares Jaccard overlap of tokenizer output against cosine similarity. A high
correlation means the model is substantially doing bag-of-tokens matching, so
any pair that shares vocabulary while asserting the opposite will score as
similar.

Result on all-MiniLM-L6-v2: correlation +0.601, and pairs that are exact token
permutations (∀ε∃δ vs ∃ε∀δ, A⊆B vs B⊆A) scored 0.97 cosine.

This doubles as a pre-flight test for a new domain: no labels required, just
pairs. See docs/FINDINGS.md.
"""
import json, numpy as np, torch, collections
from transformers import AutoTokenizer, AutoModel
P=json.load(open('/tmp/math_pairs.json'))
name='sentence-transformers/all-MiniLM-L6-v2'
tok=AutoTokenizer.from_pretrained(name); mdl=AutoModel.from_pretrained(name).eval()
def emb(t):
    e=tok(t,return_tensors='pt',truncation=True,max_length=256)
    with torch.no_grad(): h=mdl(**e).last_hidden_state[0]
    m=e['attention_mask'][0].bool(); v=h[m].mean(0).numpy()
    return v/(np.linalg.norm(v)+1e-12)
def toks(t): return collections.Counter(tok.tokenize(t))
def jac(a,b):
    A,B=toks(a),toks(b)
    inter=sum((A&B).values()); union=sum((A|B).values())
    return inter/union if union else 0.0
print(f"{'kind':6s} {'jaccard':>8s} {'cosine':>8s}  pair")
rows=[]
for kind,pairs in [('EQUIV',P['equivalent']),('CONF',P['confusable'])]:
    for a,b in pairs:
        j=jac(a,b); c=float(emb(a)@emb(b)); rows.append((kind,j,c))
        print(f"{kind:6s} {j:8.3f} {c:8.3f}  {a[:34]:34s} | {b[:30]}")
import numpy as np
J=np.array([r[1] for r in rows]); C=np.array([r[2] for r in rows])
print(f"\ncorrelation(token-overlap, cosine) = {np.corrcoef(J,C)[0,1]:+.3f}")
eq=[r for r in rows if r[0]=='EQUIV']; cf=[r for r in rows if r[0]=='CONF']
print(f"mean jaccard: equivalent {np.mean([r[1] for r in eq]):.3f}  confusable {np.mean([r[1] for r in cf]):.3f}")
print(f"mean cosine : equivalent {np.mean([r[2] for r in eq]):.3f}  confusable {np.mean([r[2] for r in cf]):.3f}")
