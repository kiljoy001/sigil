#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""Two-pass classification: name the edges, then name the group.

Similarity alone does not partition. Connected components at a cosine
threshold put 2018 of 5000 items in one cluster, and no threshold fixed it --
the finding that kept /class/ out of the 9P namespace. Citation edges worked,
but Gutenberg has none.

An LLM supplies the missing asserted edges, and unlike a Hamming threshold each
edge arrives with a *name*. That is the whole difference: a similarity edge
gives an unlabelled blob, a judged edge gives a directory you can walk into.

    pass 1   per pair, at ingest      i, j, hamming, subjects  -> edges
    pass 2   per block, once          [every subject in block] -> label

Pass two is where the reasoning happens and it is nearly free: one call over a
few hundred tokens of distilled subject strings, against pass one's one call
per pair. It reads the whole list at once, so it can see `battle`, `norman
conquest`, `hastings` and `william` as one theme and name it -- which counting
string frequencies cannot do. It may also answer that a list is incoherent,
which is the per-block quality verdict a global threshold could never give.

The structure comes from the LSH, the names come from the model. Clustering on
the subject strings themselves would be clustering the model's output; instead
the graph is measured Hamming edges and the subjects only label it.

Judging radius is 90 bits of 256. Measured, not chosen:

    radius   related edges captured
      70            22.7%
      80            60.1%
     90            84.5%      <- knee
     100           93.1%
     120           97.4%

Below 80 recall collapses; past 90 the cost rises faster than the return.
"""

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RADIUS = 90          # Hamming, of 256 bits
MAX_PAIRS_PER_SEED = 20

# Pass two. Deliberately allowed to split: even inside the radius ~15% of edges
# are judge-rejected, so a block can hold more than one theme, and saying so is
# more useful than forcing a single wrong label onto it.
GROUP_PROMPT = """These subject labels were extracted from a group of \
passages that a similarity index placed together.

Subjects:
{subjects}

Identify the coherent themes. Answer in exactly this format, one line per \
theme, at most three:
THEME: <short name> | <the subjects belonging to it, comma separated>

If the subjects do not form any coherent theme, answer exactly:
INCOHERENT

THEME:"""


def pass1(store, judge, seeds, radius=RADIUS, max_pairs=MAX_PAIRS_PER_SEED,
          progress=None):
    """Judge near pairs. Returns [(i, j, distance, [subjects])].

    Only pairs within `radius` are judged. Past 120 bits the judge confirms
    5% of pairs, so the far tail is not worth the call -- that cutoff is what
    makes this affordable at corpus scale, where almost every pair is far.
    """
    edges = []
    for n, i in enumerate(seeds, 1):
        for j, dist in store.neighbours(i, radius, max_pairs):
            subjects = judge(store.text(i), store.text(j))
            if subjects:
                edges.append((i, j, dist, subjects))
        if progress and n % progress == 0:
            print(f"  pass1 {n}/{len(seeds)} seeds, {len(edges)} edges",
                  file=sys.stderr)
    return edges


def blocks_from_edges(edges):
    """Connected components over judge-confirmed edges only.

    Pass one must persist the pair indices, not just the verdict. The first
    measurement run stored `{d, llm, subs}` and dropped `(i, j)`, which makes
    the edge set ungroupable -- every confirmed pair becomes its own block of
    two. Obvious in hindsight, invisible until the graph is built.

    Components over *similarity* edges chain badly -- that is the measurement
    that killed threshold clustering. Restricting to confirmed edges is what
    makes components defensible here, and pass two can still split a block that
    chained through a bridging passage.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j, _, _ in edges:
        union(i, j)

    groups = collections.defaultdict(lambda: {"members": set(),
                                              "subjects": collections.Counter()})
    for i, j, _, subs in edges:
        g = groups[find(i)]
        g["members"].update((i, j))
        for s in subs:
            g["subjects"][s] += 1
    return groups


def pass2(groups, generate, min_size=2):
    """Name each block from its distilled subject list.

    One call per block over a few hundred tokens, against pass one's call per
    pair. The model sees every subject in the block at once, which is what lets
    it recognise related terms as a single theme.
    """
    out = []
    for root, g in groups.items():
        if len(g["members"]) < min_size:
            continue
        # Distilled: unique subjects, most frequent first, not the raw edges.
        listing = ", ".join(s for s, _ in g["subjects"].most_common(40))
        raw = generate(GROUP_PROMPT.format(subjects=listing))
        themes = []
        for line in ("THEME:" + raw).splitlines():
            line = line.strip()
            if line.upper().startswith("INCOHERENT"):
                themes = []
                break
            if line.upper().startswith("THEME:"):
                body = line.split(":", 1)[-1]
                name, _, subs = body.partition("|")
                if name.strip():
                    themes.append({
                        "name": name.strip().lower(),
                        "subjects": [s.strip().lower()
                                     for s in subs.split(",") if s.strip()],
                    })
        out.append({
            "root": root,
            "members": sorted(g["members"]),
            "size": len(g["members"]),
            "themes": themes[:3],
            "coherent": bool(themes),
            "top_subjects": [s for s, _ in g["subjects"].most_common(10)],
        })
    return sorted(out, key=lambda x: -x["size"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edges", help="pass-1 output as JSON")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-size", type=int, default=2)
    args = ap.parse_args()

    edges = [tuple(e) for e in json.load(open(args.edges))]
    groups = blocks_from_edges(edges)
    print(f"{len(edges)} confirmed edges -> {len(groups)} blocks",
          file=sys.stderr)

    from gutenberg_rerank_bridge import generate  # noqa: F401
    result = pass2(groups, generate, args.min_size)

    coherent = sum(1 for r in result if r["coherent"])
    print(f"{len(result)} blocks of size >= {args.min_size}, "
          f"{coherent} coherent", file=sys.stderr)
    text = json.dumps(result, indent=1)
    if args.out:
        open(args.out, "w").write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
