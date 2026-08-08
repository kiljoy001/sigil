#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
"""Locate an embedding inversion: is it the hash, or the embedding?

Runs the equivalent/confusable probe through six pooling strategies and reports
raw float cosine alongside 128-bit Hamming. If cosine is already inverted, the
compression is faithful and the fault is upstream.

Result on all-MiniLM-L6-v2: every pooling strategy inverted in float space, and
Hamming tracked cosine closely throughout. See docs/FINDINGS.md.

Needs torch + transformers; not part of the library build.
"""
import json, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
P=json.load(open('/tmp/math_pairs.json'))
lines=[]
for a,b in P['equivalent']: lines+=[a,b]
for a,b in P['confusable']: lines+=[a,b]
ne=len(P['equivalent']); nc=len(P['confusable'])
name='sentence-transformers/all-MiniLM-L6-v2'
tok=AutoTokenizer.from_pretrained(name); mdl=AutoModel.from_pretrained(name).eval()

def pooled(t, mode):
    e=tok(t,return_tensors='pt',truncation=True,max_length=256)
    with torch.no_grad(): h=mdl(**e).last_hidden_state[0]
    m=e['attention_mask'][0].bool()
    h=h[m]
    if mode=='mean': v=h.mean(0)
    elif mode=='cls': v=h[0]
    elif mode=='max': v=h.max(0).values
    elif mode=='last': v=h[-1]
    elif mode=='concat_first_last': v=torch.cat([h[0],h[-1]])
    elif mode=='weighted':   # position-weighted: later tokens count more
        w=torch.arange(1,len(h)+1,dtype=torch.float32).unsqueeze(1); v=(h*w).sum(0)/w.sum()
    v=v.numpy(); return v/(np.linalg.norm(v)+1e-12)

lut=np.array([bin(i).count('1') for i in range(256)],dtype=np.uint8)
print(f"{'pooling':22s} {'eq_cos':>7s} {'cf_cos':>7s} {'cos_sep':>8s} {'eq_ham':>7s} {'cf_ham':>7s} {'ham_sep':>8s}")
for mode in ['mean','cls','max','last','concat_first_last','weighted']:
    E=np.array([pooled(t,mode) for t in lines],dtype=np.float32)
    eqc=[float(E[2*k]@E[2*k+1]) for k in range(ne)]
    cfc=[float(E[2*ne+2*k]@E[2*ne+2*k+1]) for k in range(nc)]
    rng=np.random.default_rng(0x5191c0de)
    H=(E@rng.standard_normal((E.shape[1],128),dtype=np.float32))>0
    Hp=np.packbits(H,axis=1)
    ham=lambda i,j:int(lut[np.bitwise_xor(Hp[i],Hp[j])].sum())
    eqh=[ham(2*k,2*k+1) for k in range(ne)]
    cfh=[ham(2*ne+2*k,2*ne+2*k+1) for k in range(nc)]
    # cos_sep: equivalent should be HIGHER cos than confusable -> positive good
    print(f"{mode:22s} {np.mean(eqc):+7.3f} {np.mean(cfc):+7.3f} {np.mean(eqc)-np.mean(cfc):+8.3f} "
          f"{np.mean(eqh):7.1f} {np.mean(cfh):7.1f} {np.mean(cfh)-np.mean(eqh):+8.1f}")
