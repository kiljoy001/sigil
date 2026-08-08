#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
"""What compression costs at realistic result counts.

R@1 is the harshest measure and matches no real interface. Measured on citation
data, fraction of the float32 ceiling retained:

              R@1    R@10   R@20   R@100
  128-bit   57.3%   71.4%  75.4%   87.6%
  256-bit   73.7%   83.1%  85.8%   93.2%
  512-bit   87.9%   92.3%  93.3%   97.2%

Rank displacement shows why: of queries float32 answered at rank 1, 256-bit
keeps 91.4% inside the top 10 and the median displacement is zero. Compression
reorders the head of the list; it rarely discards correct answers.
"""
import json, numpy as np
E=np.load('/tmp/citE.npy'); works=json.load(open('/tmp/citworks.json'))
W=np.array(works); N=len(W)
lut=np.array([bin(i).count('1') for i in range(256)],dtype=np.uint8)
Ks=(1,5,10,20,50,100)

def first_hit_ranks(scorer):
    out=np.empty(N,dtype=np.int64)
    for i in range(N):
        order=scorer(i)
        hit=np.flatnonzero(W[order]==W[i])
        out[i]=hit[0]+1 if len(hit) else 10**9
    return out

def cos_scorer(i):
    s=E@E[i]; s[i]=-2.0; return np.argsort(-s)

def make_ham(nb):
    rng=np.random.default_rng(0x5191c0de)
    H=np.packbits((E@rng.standard_normal((E.shape[1],nb),dtype=np.float32))>0,axis=1)
    def f(i):
        d=lut[np.bitwise_xor(H[i],H)].sum(1).astype(np.int64); d[i]=1<<20
        return np.argsort(d, kind="stable")
    return f

print(f"{N} citation contexts, {len(set(works))} distinct works\n")
res={}
hdr="          " + " ".join(f"R@{k}".rjust(7) for k in Ks)
print(hdr)
for label,sc in [("float32",cos_scorer),("128-bit",make_ham(128)),
                 ("256-bit",make_ham(256)),("512-bit",make_ham(512))]:
    r=first_hit_ranks(sc); res[label]=r
    print(f"{label:9s} " + " ".join(f"{np.mean(r<=k):7.3f}" for k in Ks))

print("\nretained vs the float32 ceiling:")
print(hdr)
for label in ("128-bit","256-bit","512-bit"):
    print(f"{label:9s} " + " ".join(
        f"{100*np.mean(res[label]<=k)/np.mean(res['float32']<=k):6.1f}%" for k in Ks))

print("\nrank displacement -- where float32 answered at rank 1:")
at1=np.flatnonzero(res["float32"]==1)
print(f"  ({len(at1)} queries)")
for label in ("128-bit","256-bit","512-bit"):
    m=res[label][at1]
    print(f"  {label}: median rank {int(np.median(m)):3d}   "
          f"stays rank 1 {100*np.mean(m==1):5.1f}%   "
          f"top-10 {100*np.mean(m<=10):5.1f}%   top-50 {100*np.mean(m<=50):5.1f}%")
