#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
"""
Fetch a standard paraphrase-retrieval corpus for evaluating LSH quality.

Uses Quora Duplicate Questions (sentence-transformers/quora-duplicates), the
same human-labeled pairs used by BEIR and MTEB. A hand-written corpus is worse
than useless here: it measures how separable the author made the examples, not
how the system performs. Real duplicate questions share vocabulary and phrasing
in ways invented pairs do not, and that difference moved measured recall@1 at
32 bits from 0.18 to 0.56 — the hand-made set was far harder than reality and
would have driven the wrong design.

Writes corpus.txt (one document per line, pairs interleaved) and pairs.json.
Ground truth is positional: document 2i and 2i+1 are duplicates.

Requires `datasets`; install into a venv since Debian/Ubuntu enforce PEP 668:
    python3 -m venv ~/.sigil-eval
    ~/.sigil-eval/bin/pip install datasets
    ~/.sigil-eval/bin/python tools/fetch-corpus.py
"""

import argparse
import json
import os
import random
import sys

DEFAULT_PAIRS = 1000
SEED = 20260806  # fixed so the corpus is reproducible across machines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--pairs", type=int, default=DEFAULT_PAIRS,
                    help=f"number of pairs to sample (default {DEFAULT_PAIRS})")
    ap.add_argument("-o", "--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "test", "data"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("need `datasets`: pip install datasets (use a venv, see docstring)")

    ds = load_dataset("sentence-transformers/quora-duplicates", "pair",
                      split="train")

    rng = random.Random(args.seed)
    # Oversample: short and degenerate pairs get filtered out below.
    idx = rng.sample(range(len(ds)), min(len(ds), args.pairs * 3))

    pairs, seen = [], set()
    for i in idx:
        r = ds[i]
        a, b = r["anchor"].strip(), r["positive"].strip()
        # Very short questions carry too little signal to be a fair test, and
        # identical strings would measure nothing at all.
        if len(a) < 15 or len(b) < 15 or a.lower() == b.lower():
            continue
        key = (a.lower(), b.lower())
        if key in seen:
            continue
        seen.add(key)
        pairs.append((a, b))
        if len(pairs) >= args.pairs:
            break

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, "pairs.json"), "w") as f:
        json.dump({"source": "sentence-transformers/quora-duplicates",
                   "seed": args.seed, "pairs": pairs}, f, indent=1)

    # One document per line; newlines inside a question would desync the
    # positional ground truth, so flatten them.
    with open(os.path.join(outdir, "corpus.txt"), "w") as f:
        for a, b in pairs:
            f.write(" ".join(a.split()) + "\n")
            f.write(" ".join(b.split()) + "\n")

    print(f"{len(pairs)} pairs -> {outdir}/corpus.txt, pairs.json")
    print(f"ground truth: documents 2i and 2i+1 are duplicates")


if __name__ == "__main__":
    main()
