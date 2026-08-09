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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RADIUS = 90          # Hamming, of 256 bits
MAX_PAIRS_PER_SEED = 20

# Pass two consolidates. Pass one produces one group per subject string, and
# the vocabulary is long-tailed -- 233 confirmed edges yielded 336 distinct
# strings, 256 of them appearing exactly once. `war`, `battle` and `expedition`
# are three groups that should be one; `souls`, `consciousness` and `religion`
# likewise. Reading the whole list at once is what lets the model see that,
# which is precisely what counting string frequencies cannot do.
#
# A subject may belong to more than one theme, and that is not an error --
# overlapping membership is the point. Groups are not a partition.
GROUP_PROMPT = """These subject labels were extracted from passages that a \
similarity index placed near each other.

Subjects:
{subjects}

Group them into coherent themes. A subject may belong to more than one theme. \
Answer in exactly this format, one line per theme, at most five:
THEME: <short name> | <the subjects belonging to it, comma separated>

If none of the subjects form any coherent theme, answer with the single word \
INCOHERENT and nothing else. Do not use INCOHERENT as a theme name.

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
    """Group by subject, with overlapping membership.

    Pass one must persist the pair indices, not just the verdict. The first
    measurement run stored `{d, llm, subs}` and dropped `(i, j)`, which makes
    the edge set ungroupable -- every confirmed pair becomes its own block of
    two. Obvious in hindsight, invisible until the graph is built.

    Not connected components, and not a partition. A passage about Napoleon's
    retreat belongs under war *and* russia *and* winter; forcing it into one
    group is the assumption that broke threshold clustering, because a single
    bridging passage then merges two topics permanently and irreversibly.

    Letting a paragraph appear in every group it has a judged edge into makes a
    bridge merely a member of both, which is what it actually is. That is also
    how the namespace wants to behave -- a file under several directories is
    ordinary in Plan 9, and union mounts do exactly this.

    So the unit here is the subject, not the component: each confirmed edge
    files both of its endpoints under every subject the judge named for it.
    """
    groups = collections.defaultdict(lambda: {"members": set(),
                                              "subjects": collections.Counter(),
                                              "edges": 0})
    for i, j, _, subs in edges:
        for s in subs:
            g = groups[s]
            g["members"].update((i, j))
            g["edges"] += 1
            for s2 in subs:
                g["subjects"][s2] += 1
    return groups


def pass2(groups, generate, min_size=2, batch=60):
    """Consolidate the per-subject groups into named themes.

    One call per batch of subjects -- a few hundred tokens -- against pass
    one's call per pair. Cheap enough that the layer doing the most valuable
    reasoning costs almost nothing.

    Returns themes, each carrying every paragraph filed under any of its
    subjects. A paragraph appears in as many themes as it has judged edges
    into, which is the intended behaviour: a passage on Napoleon's retreat is
    genuinely about war and russia and winter at once.
    """
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]["members"]))
    live = [(s, g) for s, g in ranked if len(g["members"]) >= min_size]
    if not live:
        return []

    themes = []
    for start in range(0, len(live), batch):
        chunk = live[start:start + batch]
        listing = ", ".join(s for s, _ in chunk)
        raw = generate(GROUP_PROMPT.format(subjects=listing))
        for line in ("THEME:" + raw).splitlines():
            line = line.strip()
            if line.upper().startswith("INCOHERENT"):
                break
            if not line.upper().startswith("THEME:"):
                continue
            name, _, subs = line.split(":", 1)[-1].partition("|")
            name = name.strip().lower()
            subjects = [s.strip().lower() for s in subs.split(",") if s.strip()]
            if not name or not subjects or name.startswith("incoherent"):
                continue
            members = set()
            matched = []
            for s in subjects:
                if s in groups:
                    members.update(groups[s]["members"])
                    matched.append(s)
            if len(themes) >= 5 * (start // batch + 1):
                break       # the stated cap is advisory; enforce it here
            if members:
                themes.append({
                    "name": name,
                    "subjects": matched,
                    "members": sorted(members),
                    "size": len(members),
                })

    # A subject the model dropped still has members; keep it as its own theme
    # rather than silently losing those paragraphs.
    claimed = {s for t in themes for s in t["subjects"]}
    for s, g in live:
        if s not in claimed:
            themes.append({"name": s, "subjects": [s],
                           "members": sorted(g["members"]),
                           "size": len(g["members"])})
    return sorted(themes, key=lambda t: -t["size"])


def make_generate(model=None, device=None):
    """Text completion from the same OpenVINO pipeline pass one judges with.

    Only one pipeline may hold the GPU at a time -- two concurrent OpenVINO
    processes on one device produce mutually corrupted output with no warning,
    so pass two must not run while pass one is still judging.
    """
    import openvino_genai

    model = model or os.environ.get(
        "SIGIL_JUDGE", "/mnt/bulk/models/openvino/qwen2.5-7b-int8")
    device = device or os.environ.get("SIGIL_JUDGE_DEVICE", "GPU.1")
    pipe = openvino_genai.LLMPipeline(model, device)

    def generate(prompt, max_new_tokens=300):
        return pipe.generate(prompt, max_new_tokens=max_new_tokens)
    return generate


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

    result = pass2(groups, make_generate(), args.min_size)

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
