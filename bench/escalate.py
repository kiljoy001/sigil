#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
"""Two-stage retrieval: what can serve as the escalation discriminator?

The measured failure mode is that contradictions score as similar because they
share vocabulary. A two-stage design -- scan cheaply, verify the candidates --
only works if stage two can tell them apart.

This tests a cheap lexical trigger (Jaccard token overlap) against real
human-labeled duplicates. Result: it does not work. Contradiction and genuine
paraphrase Jaccard distributions overlap almost completely, so no threshold
separates them. See docs/FINDINGS.md.

Needs torch + transformers; not part of the library build.
"""
import json, numpy as np, collections
from transformers import AutoTokenizer, AutoModel
import torch
name='sentence-transformers/all-MiniLM-L6-v2'
tok=AutoTokenizer.from_pretrained(name); mdl=AutoModel.from_pretrained(name).eval()
def jac(a,b):
    A,B=collections.Counter(tok.tokenize(a)),collections.Counter(tok.tokenize(b))
    u=sum((A|B).values()); return sum((A&B).values())/u if u else 0.0
def emb(t):
    e=tok(t,return_tensors='pt',truncation=True,max_length=256)
    with torch.no_grad(): h=mdl(**e).last_hidden_state[0]
    m=e['attention_mask'][0].bool(); v=h[m].mean(0).numpy(); return v/(np.linalg.norm(v)+1e-12)
lut=np.array([bin(i).count('1') for i in range(256)],dtype=np.uint8)
rng=np.random.default_rng(0x5191c0de); Pm=rng.standard_normal((384,128),dtype=np.float32)

rows=json.load(open('/home/scott/Repo/sigil/test/data/pairs.json'))['pairs'][:400]
E=np.array(json.load(open('/tmp/q_all-MiniLM-L6-v2.json')),dtype=np.float32)
E/=(np.linalg.norm(E,axis=1,keepdims=True)+1e-12)
Hq=np.packbits((E@Pm)>0,axis=1)
dup=[(int(lut[np.bitwise_xor(Hq[2*i],Hq[2*i+1])].sum()), jac(*rows[i])) for i in range(len(rows))]

contra=[("the patient should receive the medication","the patient should not receive the medication"),
 ("the tenant is responsible for repairs","the landlord is responsible for repairs"),
 ("payment is due within 30 days","payment is due within 90 days"),
 ("the test result was positive","the test result was negative"),
 ("access is granted to all users","access is denied to all users"),
 ("the algorithm runs in O(n log n) time","the algorithm runs in O(n^2) time"),
 ("A \\subseteq B","B \\subseteq A"),
 ("f is continuous on [a,b]","f is discontinuous on [a,b]"),
 ("the matrix A is invertible","the matrix A is singular"),
 ("shares rose 5 percent","shares fell 5 percent")]
cn=[]
for a,b in contra:
    ha=np.packbits((emb(a)@Pm)>0); hb=np.packbits((emb(b)@Pm)>0)
    cn.append((int(lut[np.bitwise_xor(ha,hb)].sum()), jac(a,b)))
print(f"{'jac_t':>6s} {'caught':>10s} {'false alarm':>14s}")
for t in (0.55,0.65,0.75,0.85,0.90,0.95,1.00):
    c=sum(1 for h,j in cn if h<30 and j>=t)
    f=sum(1 for h,j in dup if h<30 and j>=t)
    print(f"{t:6.2f} {c:>4d}/{len(cn):<5d} {f:>6d}/{len(dup):<7d} ({100*f/len(dup):4.1f}%)")
print(f"\ncontradiction jaccards: {sorted(round(j,2) for _,j in cn)}")
print(f"duplicate p50/p90/p99: {np.percentile([j for _,j in dup],[50,90,99]).round(3)}")
