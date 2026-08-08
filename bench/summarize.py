#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
"""Summarize real paragraphs with a weak LLM, for cluster labelling.

Summaries are the right output for labelling a cluster: O(N) calls rather than
O(N^2) pairwise, reusable, and human-inspectable so errors are visible. They
are NOT a preprocessing step for retrieval -- see summarize_compare.py.

One hazard measured: the model hallucinated a negation, turning "the tenant is
responsible" into "the tenant is not responsible". On the exact axis that
matters. Cluster labels are read by humans, which is the mitigation; do not
feed summaries into an automated decision without review.

Expects llama-server on :8099.
"""
import json, urllib.request, numpy as np, sys, time
sys.path.insert(0,'/tmp')
URL="http://localhost:8099/completion"
def gen(p,n=70):
    req=urllib.request.Request(URL,
        data=json.dumps({"prompt":p,"n_predict":n,"temperature":0,"stop":["\n\n"]}).encode(),
        headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req,timeout=240).read())["content"].strip()
SUM=("Summarize this passage in one sentence, stating its specific claim. "
     "Preserve negation, direction, and quantities.\n\nPassage: {t}\n\nSummary:")

rows=json.load(open('/tmp/arxiv_rows.json'))
rng=np.random.default_rng(7)
# same-paper pairs (related) and cross-paper pairs (unrelated)
bypaper={}
for i,(pid,_,t) in enumerate(rows): bypaper.setdefault(pid,[]).append(i)
multi=[p for p,v in bypaper.items() if len(v)>=2]
same=[(bypaper[p][0],bypaper[p][1]) for p in rng.choice(multi,8,replace=False)]
ps=rng.choice(multi,16,replace=False)
diff=[(bypaper[ps[i]][0],bypaper[ps[i+8]][0]) for i in range(8)]

t0=time.time(); recs=[]
for label,pairs in (("SAME-PAPER",same),("DIFF-PAPER",diff)):
    for i,j in pairs:
        a,b=rows[i][2],rows[j][2]
        sa,sb=gen(SUM.format(t=a[:1200])),gen(SUM.format(t=b[:1200]))
        recs.append((label,a,b,sa,sb))
print(f"summarized {len(recs)*2} paragraphs in {time.time()-t0:.0f}s "
      f"({(time.time()-t0)/(len(recs)*2)*1000:.0f} ms each)\n")
for label,a,b,sa,sb in recs[:4]:
    print(f"[{label}]")
    print(f"  A ({len(a)}c): {a[:90]}...")
    print(f"    -> {sa[:110]}")
    print(f"  B ({len(b)}c): {b[:90]}...")
    print(f"    -> {sb[:110]}\n")
json.dump([{"label":l,"a":a,"b":b,"sa":sa,"sb":sb} for l,a,b,sa,sb in recs],
          open('/tmp/summaries.json','w'))
print("wrote /tmp/summaries.json")
