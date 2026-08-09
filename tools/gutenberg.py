#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""
Project Gutenberg -> paragraphs with metadata.

Gutenberg has no citation graph, so the external human judgement here is the
Library of Congress class in the catalogue: a cataloguer decided this book is
PS (American literature) and not PR (English). That judgement was made without
reference to any embedding, which is the property that makes it usable as
ground truth -- the same reason the unarXive citation result was credible.

Two things in the archive will corrupt an index if taken at face value:

  * The same text ships in several encodings and revisions -- up to five copies
    of one book. Indexing them all inflates similarity scores with paragraphs
    that are identical by construction. One file per Text#, preferring UTF-8.

  * The catalogue's Issued field is the digitisation date (1971-2026), not the
    publication date. It is archive bookkeeping. Author death year is used
    instead: it bounds composition from above, covers 83.1% of texts, and is
    the only date here with any relation to when the words were written.
"""

import csv
import os
import re
import sys

# -0 is UTF-8, -8 is Latin-1, bare is ASCII. Lower rank wins.
ENCODING_RANK = {"-0": 0, "": 1, "-8": 2}

RE_TEXTFILE = re.compile(r"^(\d+)(-[08])?\.txt$")
RE_LIFESPAN = re.compile(r"\b(\d{4})\s*-\s*(\d{4})\b")

# Gutenberg wraps every book in a license header and footer. Left in, they
# would add tens of thousands of near-identical paragraphs to the store.
RE_START = re.compile(r"\*\*\*\s*START OF TH(E|IS) PROJECT GUTENBERG", re.I)
RE_END = re.compile(r"\*\*\*\s*END OF TH(E|IS) PROJECT GUTENBERG", re.I)

MIN_PARA = 40      # matches Minpara in cmd/index.c
MAX_PARA = 4000    # matches Maxpara in cmd/index.c


def load_catalog(path):
    """Text# -> metadata. LoCC is the evaluation signal, author the control."""
    meta = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["Type"] != "Text":
                continue
            author = row["Authors"] or ""
            m = RE_LIFESPAN.search(author)
            meta[row["Text#"]] = {
                "title": row["Title"],
                "author": author,
                "language": row["Language"],
                # Primary class only: "E201; JK" -> "E". The letter is the
                # broad subject; digits subdivide it far more finely than a
                # per-paragraph comparison can support.
                "locc": (row["LoCC"] or "").split(";")[0].strip(),
                "subjects": row["Subjects"] or "",
                # Proxy, not a fact. None for the 14.7% with no lifespan.
                "death_year": int(m.group(2)) if m else None,
            }
    return meta


def pick_files(root):
    """One path per Text#, best encoding, skipping superseded revisions.

    rsync's --exclude='old/' loses to --include='*/', which matches every
    directory first, so old/ arrives regardless and is dropped here.
    """
    best = {}
    for dirpath, dirnames, filenames in os.walk(root):
        if "old" in dirpath.split(os.sep):
            dirnames[:] = []
            continue
        for name in filenames:
            m = RE_TEXTFILE.match(name)
            if not m:
                continue  # e.g. ddcc10.txt -- legacy naming, no Text# in it
            text_id, suffix = m.group(1), m.group(2) or ""
            rank = ENCODING_RANK.get(suffix)
            if rank is None:
                continue
            prev = best.get(text_id)
            if prev is None or rank < prev[0]:
                best[text_id] = (rank, os.path.join(dirpath, name))
    return {tid: path for tid, (_, path) in best.items()}


def strip_boilerplate(text):
    """Return only the work itself, or None if the markers are missing."""
    lines = text.split("\n")
    start = end = None
    for i, line in enumerate(lines):
        if start is None and RE_START.search(line):
            start = i + 1
        elif RE_END.search(line):
            end = i
            break
    if start is None or end is None or end <= start:
        return None
    return "\n".join(lines[start:end])


def paragraphs(text):
    """Blank-line separated, same rule as the C indexer.

    Hard-wrapped lines are joined so a paragraph is one line of prose; without
    this the embedder sees ~70-character fragments.
    """
    for block in re.split(r"\n\s*\n", text):
        para = " ".join(block.split())
        if MIN_PARA <= len(para) <= MAX_PARA:
            yield para


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: gutenberg.py <catalog.csv> <mirror-root> [limit]")
    catalog, root = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    meta = load_catalog(catalog)
    files = pick_files(root)

    writer = csv.writer(sys.stdout)
    writer.writerow(["text_id", "para", "locc", "subjects", "death_year",
                     "author", "title", "text"])

    n_books = n_paras = 0
    skipped_nometa = skipped_noboiler = skipped_nonen = 0

    for text_id, path in sorted(files.items(), key=lambda kv: int(kv[0])):
        info = meta.get(text_id)
        if info is None:
            skipped_nometa += 1
            continue
        # Mixed languages let an embedder separate by language and score well
        # for the wrong reason.
        if info["language"] != "en":
            skipped_nonen += 1
            continue
        try:
            with open(path, encoding="utf-8-sig", errors="replace") as fh:
                raw = fh.read()
        except OSError:
            continue
        body = strip_boilerplate(raw)
        if body is None:
            skipped_noboiler += 1
            continue

        wrote = False
        for idx, para in enumerate(paragraphs(body), start=1):
            writer.writerow([text_id, idx, info["locc"], info["subjects"],
                             info["death_year"], info["author"],
                             info["title"], para])
            n_paras += 1
            wrote = True
        if wrote:
            n_books += 1
        if limit and n_books >= limit:
            break

    print(f"books {n_books}  paragraphs {n_paras}  "
          f"skipped: no-metadata {skipped_nometa}, "
          f"non-english {skipped_nonen}, no-markers {skipped_noboiler}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
