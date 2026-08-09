#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""Stage two on Gutenberg: does an LLM reranking the scan's candidates help?

The escalation architecture measured earlier -- cheap wide filter, expensive
narrow judge -- applied to a corpus with no citation graph. Sigil scans the
store and returns candidates; a small instruct model reads each query/candidate
pair and decides whether they engage the same subject; the list is reordered by
that verdict.

The rule this harness exists to respect:

    The LLM reranks. The catalogue scores. Never the reverse.

If the same model both produced the labels and defined correctness, the number
would measure agreement between two things we built and would prove nothing.
Subject headings come from cataloguers who never saw sigil, so they remain the
only ground truth here, exactly as in gutenberg_eval.py.

What this can show:
  * rerank beats the raw scan  -> the judge adds signal, escalation is worth it
  * rerank matches the scan    -> the judge is reading but agrees; no gain
  * rerank is worse            -> the judge is guessing, or the prompt is bad

The third outcome is the reason for the degenerate-classifier check below.
bench/llm_judge.py found that two zero-shot phrasings produced classifiers that
answered one way regardless of input -- 50% balanced accuracy, useless, and
invisible if only one error class is measured.

Usage:
  gutenberg_rerank.py <paragraphs.csv> [--n 20000] [--queries 150]
"""

import argparse
import collections
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gutenberg_eval import (POPCOUNT, embed, load, simhash,  # noqa: E402
                            subject_keys)

JUDGE_MODEL = os.environ.get(
    "SIGIL_JUDGE", "/mnt/bulk/models/openvino/qwen2.5-7b-int8")
JUDGE_DEVICE = os.environ.get("SIGIL_JUDGE_DEVICE", "GPU.1")

# Few-shot, and not optionally so. Zero-shot with the same 7B model answered
# DIFFERENT for five of six probe pairs -- coherent, reading, and useless,
# because the threshold sat in the wrong place. Four examples moved it to 6/6
# on both error classes. bench/llm_judge.py found the same thing on Quora with
# a different model: framing dominates, and a classifier that answers one way
# regardless of input looks fine if only one error class is measured.
#
# The examples deliberately include a same-topic/different-register pair and a
# shared-vocabulary/different-subject pair, because those are the two cases the
# Hamming scan gets wrong. A judge that cannot separate them adds nothing where
# it matters.
PROMPT = """Decide whether two passages concern related subject matter. \
They count as RELATED if they touch the same topic, event, place, or field, \
even in different styles or from different angles.

A: The harvest failed and the village went hungry that winter.
B: Crop yields collapsed across the region, and famine followed.
Answer: RELATED

A: The clock struck three in the empty hall.
B: Copper exports rose steadily under the new tariff.
Answer: UNRELATED

A: He tightened the rigging as the gale rose.
B: Seamanship in heavy weather was the making of a crew.
Answer: RELATED

A: The doctrine of the trinity was disputed by early councils.
B: The delegates argued past midnight over grain prices.
Answer: UNRELATED

A: {a}
B: {b}
Answer:"""

_pipe = None


def judge(a, b):
    """RELATED / UNRELATED / None.

    llama.cpp cannot host this model on an Arc GPU -- it garbles output above
    ~4B parameters on both its SYCL and Vulkan backends. OpenVINO runs the same
    7B correctly at ~480 ms/call. See docs/FINDINGS.md.
    """
    global _pipe
    if _pipe is None:
        import openvino_genai
        _pipe = openvino_genai.LLMPipeline(JUDGE_MODEL, JUDGE_DEVICE)
    out = _pipe.generate(PROMPT.format(a=a[:700], b=b[:700]),
                         max_new_tokens=4).strip().upper()
    if "UNRELATED" in out:
        return "UNRELATED"
    if "RELATED" in out:
        return "RELATED"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paragraphs")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--queries", type=int, default=150)
    ap.add_argument("--depth", type=int, default=20,
                    help="candidates from the scan to rerank")
    ap.add_argument("--bits", type=int, default=256)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--facet-depth", type=int, default=1)
    args = ap.parse_args()

    rows = load(args.paragraphs, args.n)
    subj = [subject_keys(r.get("subjects", ""), args.facet_depth) for r in rows]
    book = [r["text_id"] for r in rows]
    texts = [r["text"] for r in rows]
    print(f"{len(rows)} paragraphs, {len(set(book))} books", file=sys.stderr)

    E = embed(texts, None, args.cache)
    H = simhash(E, args.bits)

    facet_books = collections.defaultdict(set)
    for i, s in enumerate(subj):
        for f in s:
            facet_books[f].add(book[i])
    scoreable = [i for i in range(len(rows))
                 if any(len(facet_books[f] - {book[i]}) > 0 for f in subj[i])]
    queries = scoreable[:args.queries]
    print(f"{len(queries)} scoreable queries, {args.bits}-bit scan, "
          f"reranking top {args.depth}\n", file=sys.stderr)

    Ks = (1, 5, 10)
    base = {k: 0 for k in Ks}
    rerank = {k: 0 for k in Ks}
    verdicts = collections.Counter()
    n_calls = 0
    t0 = time.time()

    for qn, i in enumerate(queries, 1):
        d = POPCOUNT[np.bitwise_xor(H[i], H)].sum(1).astype(np.int32)
        d[i] = 1 << 20
        cand = [j for j in np.argsort(d, kind="stable")[:args.depth * 3]
                if book[j] != book[i]][:args.depth]
        if not cand:
            continue

        rel = [bool(subj[j] & subj[i]) for j in cand]
        for k in Ks:
            if any(rel[:k]):
                base[k] += 1

        # Reranked order: RELATED first (scan order preserved within each group),
        # then unjudged, then DIFFERENT. A stable partition, not a re-scoring --
        # the model gives a label, not a number, so there is nothing to sort by.
        same, unk, diff = [], [], []
        for pos, j in enumerate(cand):
            v = judge(texts[i], texts[j])
            n_calls += 1
            verdicts[v] += 1
            (same if v == "RELATED" else diff if v == "UNRELATED" else unk).append(pos)
        order = same + unk + diff
        rr = [rel[p] for p in order]
        for k in Ks:
            if any(rr[:k]):
                rerank[k] += 1

        if qn % 10 == 0:
            el = time.time() - t0
            print(f"  {qn}/{len(queries)} queries, {n_calls} calls, "
                  f"{1000*el/n_calls:.0f} ms/call", file=sys.stderr)

    n = len([i for i in queries])
    print(f"\n{n} queries, {n_calls} judge calls, "
          f"{1000*(time.time()-t0)/max(n_calls,1):.0f} ms/call")
    print(f"\n{'':10s} " + " ".join(f"R@{k}".rjust(8) for k in Ks))
    print(f"{'scan':10s} " + " ".join(f"{base[k]/n:8.3f}" for k in Ks))
    print(f"{'reranked':10s} " + " ".join(f"{rerank[k]/n:8.3f}" for k in Ks))
    print(f"{'delta':10s} " +
          " ".join(f"{(rerank[k]-base[k])/n:+8.3f}" for k in Ks))

    # The degenerate check. A judge answering one way regardless of input
    # cannot reorder anything, and its R@k will equal the scan's exactly --
    # which looks like "no harm done" rather than "the model is not reading".
    tot = sum(verdicts.values())
    print(f"\nverdict distribution over {tot} calls:")
    for v, c in verdicts.most_common():
        print(f"  {str(v):10s} {c:6d}  {100*c/tot:5.1f}%")
    top = max(verdicts.values()) / tot if tot else 1.0
    if top > 0.90:
        print("\nWARNING: the judge answered one way >90% of the time. "
              "It is not reading the pairs; the reranking above is noise.")


if __name__ == "__main__":
    main()
