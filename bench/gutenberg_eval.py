#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""Retrieval on Project Gutenberg, scored against library cataloguing.

Gutenberg has no citation graph, so the external judgement here is the subject
headings in the catalogue. A cataloguer decided two books are about the same
thing, without reference to any embedding -- the same property that made the
unarXive citation result credible at 305x chance.

This will score lower than the citation result, by construction:

  * Subject headings are per-book. Every paragraph of a novel inherits the
    novel's subjects, but most paragraphs of a novel are not "about" its
    subject in any useful sense -- they are dialogue, scene-setting, or
    transition. The label is noisy at the paragraph level in a way a citation
    context is not.

  * Two books sharing a heading is weaker evidence than one paragraph citing
    the work another paragraph cites.

So treat the absolute number as a floor. The comparisons that matter are
relative: bit width against float32, and literature against non-fiction.

Two controls, because the flattering measurement alone would not be evidence:

  same-author   -- if neighbours are mostly the same author, the embedder is
                   recognising prose style, not subject. Reported alongside.
  same-book     -- excluded from scoring entirely. Adjacent paragraphs of one
                   book are trivially similar and would dominate every result.

Usage:
  gutenberg_eval.py <paragraphs.csv> [--n 20000] [--model MODEL]
"""

import argparse
import collections
import csv
import json
import os
import random
import subprocess
import sys
import tempfile

import numpy as np

# LCSH headings are "--"-delimited facets, most general first:
#   "United States -- History -- Revolution, 1775-1783 -- Sources"
#
# How many facets to match on was measured, not chosen. On a 210-book sample:
#
#   depth   scoreable queries   random-pair match
#     1           99.4%              2.26%
#     2            6.9%              1.05%
#     3            6.7%              1.02%
#   exact          5.3%              1.02%
#
# Beyond one facet, headings are so specific that almost no two books share
# one: 93% of queries have no relevant paragraph anywhere in the corpus and
# drop out, leaving a metric computed on an unrepresentative remainder.
#
# Depth 1 is therefore the default, at the cost of being loose -- "United
# States" joins a Revolutionary War history to a travel guide. Chance is 2.26%,
# so there is real headroom above it. Depth 2 becomes viable on a larger
# corpus, where enough books share the finer headings; --facet-depth selects it.
FACET_DEPTH = 1

# Literature classes divide by national tradition, not subject: PS is American,
# PR English, PQ Romance. Two sea novels land in different classes by the
# author's nationality. Scored separately from non-fiction for that reason.
LIT_CLASSES = {"PS", "PR", "PQ", "PZ", "PT", "PN", "PA", "PC", "PE", "PG",
               "PH", "PJ", "PK", "PL", "PM", "PB", "PD", "PF", "PI"}

Ks = (1, 5, 10, 20)

LLAMA_BIN = "/home/scott/llama.cpp/build/bin/llama-embedding"
DEFAULT_MODEL = "/home/scott/models/all-MiniLM-L6-v2-f16.gguf"


def subject_keys(subjects, depth=FACET_DEPTH):
    """Heading string -> set of coarse facet keys."""
    out = set()
    for heading in (subjects or "").split(";"):
        facets = [f.strip() for f in heading.split("--") if f.strip()]
        if facets:
            out.add(" -- ".join(facets[:depth]).lower())
    return out


def load(path, limit, seed=12345):
    """Load paragraphs, sampled evenly across books.

    The CSV is ordered by text_id, so taking the first N rows takes a few long
    books rather than a cross-section -- an earlier run of this harness drew
    20000 paragraphs from 10 books and left 86% of queries unscoreable. Sample
    round-robin across books instead, so book count scales with the limit.
    """
    csv.field_size_limit(1 << 24)
    by_book = collections.defaultdict(list)
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            # Paragraphs contain embedded newlines; a truncated write or a
            # ragged row yields None fields. Drop rather than crash later.
            if not r.get("text") or r.get("subjects") is None:
                continue
            by_book[r["text_id"]].append(r)
    if not limit:
        return [r for rs in by_book.values() for r in rs]

    rng = random.Random(seed)
    books = sorted(by_book)
    rng.shuffle(books)
    for b in books:
        rng.shuffle(by_book[b])
    rows, idx = [], 0
    while len(rows) < limit:
        added = False
        for b in books:
            if idx < len(by_book[b]):
                rows.append(by_book[b][idx])
                added = True
                if len(rows) >= limit:
                    break
        if not added:
            break
        idx += 1
    return rows


def embed(texts, model_path, cache=None):
    """Embed with OpenVINO on the Arc. See tools/embed_openvino.py.

    515 paragraphs/s against 188 for ollama/Vulkan and 96 for CPU, and unlike
    llama.cpp it does not garble output on this GPU. Vectors agree with the
    ollama path at mean cosine 0.954 -- the gap is entirely long paragraphs,
    which ollama truncated at 600 characters and this chunks and pools.
    """
    if cache and os.path.exists(cache):
        E = np.load(cache)
        if len(E) == len(texts):
            print(f"embeddings from cache {cache}", file=sys.stderr)
            return E
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    from embed_openvino import Embedder
    e = Embedder()
    print(f"embedding {len(texts)} paragraphs on {e.device}...", file=sys.stderr)
    E = e.embed(texts, progress=True)
    if cache:
        np.save(cache, E)
    return E


POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def simhash(E, nbits, seed=0x5191c0de):
    """Same construction as src/simhash.c: sign of a Gaussian projection."""
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((E.shape[1], nbits), dtype=np.float32)
    return np.packbits(E @ P > 0, axis=1)


def evaluate(order_fn, n, subj, book, author, relevant_exists):
    """Recall@k against shared subject facets, plus the author control."""
    hits = {k: 0 for k in Ks}
    auth = {k: 0 for k in Ks}
    scored = 0
    for i in range(n):
        if not relevant_exists[i]:
            continue  # no other book shares a facet; unscoreable, not a miss
        order = [j for j in order_fn(i) if book[j] != book[i]][:max(Ks)]
        if not order:
            continue
        scored += 1
        rel = [bool(subj[j] & subj[i]) for j in order]
        sam = [author[j] == author[i] and author[i] != "" for j in order]
        for k in Ks:
            if any(rel[:k]):
                hits[k] += 1
            if any(sam[:k]):
                auth[k] += 1
    return scored, hits, auth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paragraphs")
    ap.add_argument("--n", type=int, default=20000,
                    help="paragraphs to load (0 = all)")
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cache", default=None,
                    help="npy path to cache embeddings between runs")
    ap.add_argument("--facet-depth", type=int, default=FACET_DEPTH,
                    help="LCSH facets to match on; see FACET_DEPTH")
    args = ap.parse_args()

    rows = load(args.paragraphs, args.n)
    print(f"{len(rows)} paragraphs from "
          f"{len(set(r['text_id'] for r in rows))} books", file=sys.stderr)

    subj = [subject_keys(r.get("subjects", ""), args.facet_depth)
            for r in rows]
    # gutenberg.py does not emit subjects yet when run without them; fall back
    # to LoCC so the harness still runs, but say so -- LoCC is the weaker
    # signal and conflating the two silently would misreport what was measured.
    if not any(subj):
        print("WARNING: no subject headings found; falling back to LoCC. "
              "This measures national tradition as much as subject.",
              file=sys.stderr)
        subj = [{r["locc"]} if r["locc"] else set() for r in rows]

    book = [r["text_id"] for r in rows]
    author = [r.get("author", "") for r in rows]
    texts = [r["text"] for r in rows]

    # A query is scoreable only if some *other* book shares a facet with it.
    facet_books = collections.defaultdict(set)
    for i, s in enumerate(subj):
        for f in s:
            facet_books[f].add(book[i])
    relevant_exists = [
        any(len(facet_books[f] - {book[i]}) > 0 for f in subj[i])
        for i in range(len(rows))
    ]
    nq = min(args.queries, len(rows))
    print(f"{sum(relevant_exists[:nq])}/{nq} queries have a relevant "
          f"paragraph in another book", file=sys.stderr)

    E = embed(texts, args.model, args.cache)

    def cos_order(i):
        s = E @ E[i]
        s[i] = -2.0
        return np.argsort(-s)[:200]

    def make_ham(nbits):
        H = simhash(E, nbits)

        def f(i):
            d = POPCOUNT[np.bitwise_xor(H[i], H)].sum(1).astype(np.int32)
            d[i] = 1 << 20
            return np.argsort(d, kind="stable")[:200]
        return f

    methods = [("float32", cos_order)]
    for nb in (128, 256, 512):
        methods.append((f"{nb}-bit", make_ham(nb)))

    print()
    hdr = "           " + " ".join(f"R@{k}".rjust(7) for k in Ks)
    print("subject-match recall (higher is better)")
    print(hdr)
    results = {}
    for label, fn in methods:
        scored, hits, auth = evaluate(fn, nq, subj, book, author,
                                      relevant_exists)
        results[label] = (scored, hits, auth)
        print(f"{label:10s} " +
              " ".join(f"{hits[k]/scored:7.3f}" for k in Ks))

    base = results["float32"]
    print("\nretained vs the float32 ceiling:")
    print(hdr)
    for label in ("128-bit", "256-bit", "512-bit"):
        s, h, _ = results[label]
        bs, bh, _ = base
        print(f"{label:10s} " +
              " ".join(f"{100*(h[k]/s)/(bh[k]/bs):6.1f}%" for k in Ks))

    print("\nCONTROL -- same-author rate among neighbours.")
    print("High values mean style recognition, not subject matching.")
    print(hdr)
    for label, _ in methods:
        s, _, a = results[label]
        print(f"{label:10s} " + " ".join(f"{a[k]/s:7.3f}" for k in Ks))

    # Literature vs non-fiction. Averaging them hides that these are different
    # tasks with different failure modes -- verse in particular embeds poorly.
    locc = [r["locc"][:2] if r["locc"] else "" for r in rows]
    lit_idx = [i for i in range(nq) if locc[i] in LIT_CLASSES]
    non_idx = [i for i in range(nq) if locc[i] and locc[i] not in LIT_CLASSES]
    print(f"\nsplit: {len(lit_idx)} literature, {len(non_idx)} non-fiction "
          f"queries (of {nq})")

    if lit_idx and non_idx:
        print("\nR@10 by class:")
        for label, fn in methods:
            parts = []
            for name, idx in (("lit", lit_idx), ("non-fic", non_idx)):
                ok = tot = 0
                for i in idx:
                    if not relevant_exists[i]:
                        continue
                    order = [j for j in fn(i) if book[j] != book[i]][:10]
                    if not order:
                        continue
                    tot += 1
                    ok += any(subj[j] & subj[i] for j in order)
                parts.append(f"{name} {ok/tot:.3f}" if tot else f"{name} n/a")
            print(f"  {label:10s} " + "   ".join(parts))


if __name__ == "__main__":
    main()
