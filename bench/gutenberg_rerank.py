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

# Ask for the shared subjects, not a yes/no verdict.
#
# The previous prompt asked "are these related, RELATED or UNRELATED" and got
# UNRELATED for 89.4% of real pairs after scoring 6/6 on hand-written probes.
# Three things were wrong with it, and all three are structural:
#
#   * Its four few-shot examples were one-sentence fragments, while the real
#     inputs are up to 700 characters of Victorian prose. The demonstrated task
#     did not resemble the actual one.
#   * Both positive examples were near-paraphrases and both negatives were
#     wildly disjoint, so the demonstrated boundary was "almost the same
#     sentence" versus "nothing in common". Everything real fell in between,
#     where the model defaulted to no.
#   * It asked the wrong question. The score is whether two *books* share an
#     LCSH facet at depth 1 -- coarse, e.g. both tagged "united states". A
#     Revolution history and an Ohio travel guide count as a hit. Asking
#     whether the *passages* are topically related is far stricter, so the
#     judge could be right and still be scored wrong.
#
# Naming the shared subjects forces the model to commit to a reason rather than
# fall back on a default, and it produces output directly comparable to the
# subject headings the catalogue assigns -- the same units the metric uses.
# Extract, then compare. Asking for the main idea of each block first, and only
# then for what they share, is what finally produced a judge that neither
# defaults to UNRELATED nor accepts shared vocabulary as a shared subject.
#
# Why it works where a direct question did not: summarising each block on its
# own forces the model to read both before deciding anything, and a passing
# mention like "chestnut" does not survive into a one-sentence summary. The
# earlier prompts asked for the comparison directly, which let the model answer
# from surface overlap or fall back on a default without ever engaging the text.
#
# The rigid output format matters as much as the framing -- an open-ended
# "extract the main ideas and compare them" produced good extractions and then
# ran out of tokens before reaching a verdict.
PROMPT = """Extract the main idea of each text block, then compare them.

Block 1:
{a}

Block 2:
{b}

Answer in exactly this format:
IDEA 1: <one sentence>
IDEA 2: <one sentence>
SHARED: <the subjects both blocks are about, comma separated, or NONE>

IDEA 1:"""

# Rejected as subjects regardless of what the model returns. The exclusions in
# the prompt cut most of it, but a 7B model still occasionally answers with a
# bare noun, and those verdicts are the ones that would corrupt the result:
# SimHash is most sensitive to exactly the shared vocabulary they represent, so
# counting them measures the LSH against itself.
GENERIC = {"people", "time", "description", "nature", "life", "man", "men",
           "woman", "women", "he", "she", "it", "they", "things", "place",
           "places", "day", "world", "work", "way", "words", "text", "none",
           "unrelated", "n/a", "nothing", "human", "humans", "person",
           "emotion", "emotions", "feeling", "feelings", "action", "actions",
           "object", "objects", "event", "events", "subject", "subjects"}


def clean_subjects(items):
    """Drop bare tokens and generic categories.

    A shared subject has to be a topic. The first run accepted ["chestnut"],
    "face" and "he (the male subject)" as shared subjects -- shared words, not
    shared topics -- and those inflate the measured separation for a reason
    that has nothing to do with meaning.
    """
    out = []
    for t in items:
        t = t.strip(" -*.\"'[]\t").lower()
        if not t or t in GENERIC:
            continue
        # No length rule here. An earlier version rejected any single word
        # under 8 characters, which discarded "battle" and "currants" --
        # correct answers -- and helped drive the related rate to 0.2%.
        # Extracting the main idea first already excludes passing mentions,
        # so this only has to catch the true generics.
        out.append(t)
    return out

_pipe = None


def judge(a, b, return_text=False):
    """Shared subjects as a list, or None when the model says UNRELATED.

    llama.cpp cannot host this model on an Arc GPU -- it garbles output above
    ~4B parameters on both its SYCL and Vulkan backends. OpenVINO runs the same
    7B correctly at ~480 ms/call. See docs/FINDINGS.md.
    """
    global _pipe
    if _pipe is None:
        import openvino_genai
        _pipe = openvino_genai.LLMPipeline(JUDGE_MODEL, JUDGE_DEVICE)
    # Two one-sentence summaries plus a verdict line needs ~110 tokens; at 48
    # the model ran out mid-summary and never emitted SHARED at all.
    raw = _pipe.generate(PROMPT.format(a=a[:700], b=b[:700]),
                         max_new_tokens=110).strip()
    if return_text:
        return raw
    for line in raw.splitlines():
        if line.strip().upper().startswith("SHARED"):
            v = line.split(":", 1)[-1].strip()
            if not v or v.upper().startswith(("NONE", "UNRELATED")):
                return None
            return clean_subjects(v.split(",")) or None
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
    shared_subjects = collections.Counter()
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
        same, diff = [], []
        for pos, j in enumerate(cand):
            v = judge(texts[i], texts[j])
            n_calls += 1
            verdicts["shared" if v else "UNRELATED"] += 1
            if v:
                shared_subjects.update(v)
            (same if v else diff).append(pos)
        order = same + diff
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
    if shared_subjects:
        print("\nmost frequent shared subjects named by the judge:")
        for sub, c in shared_subjects.most_common(12):
            print(f"  {c:4d}  {sub[:64]}")

    top = max(verdicts.values()) / tot if tot else 1.0
    if top > 0.90:
        print("\nWARNING: the judge answered one way >90% of the time. "
              "It is not reading the pairs; the reranking above is noise.")


if __name__ == "__main__":
    main()
