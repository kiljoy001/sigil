#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
"""Stage two of retrieval: a weak LLM reading the candidate pairs.

The cheap lexical trigger failed (see escalate.py), so stage two has to
actually read. This measures whether a small instruct model is accurate enough
to be worth ~350 ms per pair, which is affordable because the scan reduces
millions of records to a handful of candidates first.

Prompt framing dominates the result. Two zero-shot phrasings produced
degenerate classifiers answering one way regardless of input (50% balanced
accuracy, i.e. useless); four labelled examples fixed it. Always check both
error classes -- 10/10 on one of them hides a model that is not reading.

Expects llama-server on :8099. See docs/FINDINGS.md.
"""
import json, urllib.request, time
URL="http://localhost:8099/completion"
FEWSHOT=("Decide if two sentences are paraphrases (same meaning) or contradictions.\n\n"
 "A: The meeting is on Monday.\nB: The meeting takes place Monday.\nAnswer: PARAPHRASE\n\n"
 "A: The door is open.\nB: The door is closed.\nAnswer: CONTRADICTION\n\n"
 "A: How do I learn Python?\nB: What is the best way to learn Python?\nAnswer: PARAPHRASE\n\n"
 "A: Payment is due in 30 days.\nB: Payment is due in 90 days.\nAnswer: CONTRADICTION\n\n"
 "A: {a}\nB: {b}\nAnswer:")
def ask(a,b):
    req=urllib.request.Request(URL,
        data=json.dumps({"prompt":FEWSHOT.format(a=a,b=b),"n_predict":5,
                         "temperature":0,"stop":["\n"]}).encode(),
        headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req,timeout=120).read())["content"].strip().upper()
rows=json.load(open('/home/scott/Repo/sigil/test/data/pairs.json'))['pairs']
dup=rows[:60]
# Non-duplicates: pair each question with an unrelated one (true negatives that
# are NOT contradictions -- just different topics). The scan would rarely
# retrieve these, but they test that the judge is not just saying PARAPHRASE.
unrel=[(rows[i][0], rows[(i+37)%len(rows)][1]) for i in range(30)]
t0=time.time()
d=sum(1 for a,b in dup if "PARAPHRASE" in ask(a,b))
u=sum(1 for a,b in unrel if "CONTRADICTION" in ask(a,b) or "PARAPHRASE" not in ask(a,b))
dt=time.time()-t0
print(f"60 real Quora duplicates  -> PARAPHRASE {d}/60 = {100*d/60:.1f}%")
print(f"30 unrelated pairs        -> not-paraphrase {u}/30 = {100*u/30:.1f}%")
print(f"{dt:.0f}s total, {dt/120*1000:.0f} ms/call")
