#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""
Corpus pipeline, stage 3a: catalogue metadata.

Where the ground truth comes from
---------------------------------

Gutenberg has no citation graph, so the external human judgement in this
corpus is the catalogue: a cataloguer assigned this book a Library of
Congress class and a subject heading without reference to any embedding.
That independence is what makes it usable to measure clustering against --
the same property that made the unarXive citation graph credible.

It is not in the files. Only 7 of 118 current files in the sampled subtree
carry a `Title:` header; most run from the START marker straight into the
text. Parsing headers would cover about 6% of the corpus. The catalogue at
pg_catalog.csv joins to 77,801 of 77,815 book ids in the mirror -- 100.0%,
with Title, Subjects, LoCC and Language each complete on that set.

Two fields need judgement rather than transcription
---------------------------------------------------

**Dates.** `Issued` is when Project Gutenberg digitised the text: the values
run 1971 to 2026 and say nothing about when the words were written. Author
death year, taken from the lifespan in the Authors field, bounds composition
from above and covers 83.2%. Both are emitted, as separate fields, each
honest about what it is -- merging them would produce a column where a 1998
could mean either a modern work or a Victorian one digitised in 1998.

**LoCC.** The letter is the broad class; the digits subdivide much more
finely than a paragraph-level comparison can support, so PS3521 and PS3537
both reduce to PS. Multi-class values like "E201; JK" take the first.

These decisions match tools/gutenberg.py, which made them first for its CSV
export. Kept identical on purpose: two parts of the same project disagreeing
about what a book's date is would be worse than either choice.
"""

import csv
import re
import sys
from pathlib import Path

# Four-digit years around a hyphen. Tolerates the cataloguer's uncertainty
# markers -- "1814?-1884" and "1842-1914?" both appear -- because an
# approximate year is still a usable upper bound.
RE_LIFESPAN = re.compile(r"\b(\d{4})\??\s*-\s*(\d{4})\??")

# The leading letters of a Library of Congress class.
RE_LOCC = re.compile(r"^([A-Z]+)")

# csv defaults to 128 KB fields; some Subjects values are longer.
csv.field_size_limit(10 * 1024 * 1024)


def death_year(authors):
    """Upper bound on composition, or None.

    None for the 16.8% with no lifespan -- a living author, or a form the
    catalogue writes as "active 6th century B.C.". Callers must handle the
    absence rather than receive a fabricated year.

    Where the first name has no dates but a later one does (an ancient
    author with a modern translator), the translator's death year is
    returned. It dates the translation rather than the work, but it is a
    real upper bound on the text as published, and the alternative is
    discarding the only date present.
    """
    m = RE_LIFESPAN.search(authors or "")
    return int(m.group(2)) if m else None


def primary_locc(locc):
    """Broad class letter: "E201; JK" -> "E", "PS3521" -> "PS"."""
    first = (locc or "").split(";")[0].strip()
    m = RE_LOCC.match(first)
    return m.group(1) if m else ""


def ndb_quote(value):
    """Quote a value for libtab's ndb grammar.

    libtab's Tabula.set() does not do this, and the consequence is not a
    write error: the file is written, and something later fails to open it
    with "row 0 has undeclared column from" -- naming a word from the
    middle of a title. Every title, author and subject in this corpus
    contains spaces, so nothing reaches the manifest unquoted.

    A value needs quoting when it contains whitespace or a quote, when it
    is empty (or the field would vanish and the row would change shape),
    or when it starts with # (which begins an ndb comment, discarding the
    rest of the line). Interior quotes are doubled, as ndb expects.
    """
    s = "" if value is None else str(value)
    if s == "" or s.startswith("#") or any(c in s for c in ' \t\n\r"'):
        return '"' + s.replace('"', '""') + '"'
    return s


def load_catalog(path):
    """Text# -> metadata, for text works only.

    The catalogue carries seven media types: 1,114 Sound rows, 89 Datasets,
    and a handful of images among the 77,883 Texts. Joining an audiobook's
    metadata onto a text corpus is a join that succeeds and means nothing,
    so anything that is not Type=Text is dropped here rather than filtered
    by every caller.
    """
    meta = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("Type") != "Text":
                continue
            authors = row.get("Authors") or ""
            meta[row["Text#"]] = {
                "title": row.get("Title") or "",
                "authors": authors,
                "language": row.get("Language") or "",
                "subjects": row.get("Subjects") or "",
                "locc": primary_locc(row.get("LoCC")),
                "bookshelves": row.get("Bookshelves") or "",
                # Digitisation date. Explicitly not publication -- see the
                # module docstring.
                "issued": row.get("Issued") or "",
                # Composition bound. None for 16.8%.
                "death_year": death_year(authors),
            }
    return meta


def main():
    """Report coverage of a catalogue against a mirror, as a check."""
    if len(sys.argv) < 2:
        sys.exit("usage: metadata.py <catalog.csv> [book-id ...]")

    cat = load_catalog(Path(sys.argv[1]))
    print(f"{len(cat)} text works", file=sys.stderr)

    if len(sys.argv) > 2:
        for tid in sys.argv[2:]:
            r = cat.get(tid)
            if r is None:
                print(f"{tid}: not in catalogue")
                continue
            print(f"{tid}: {r['title']}")
            for k in ("authors", "locc", "subjects", "language",
                      "death_year", "issued"):
                print(f"    {k:11} {r[k]}")
        return 0

    with_death = sum(1 for r in cat.values() if r["death_year"] is not None)
    print(f"  death_year: {with_death} "
          f"({100 * with_death / max(len(cat), 1):.1f}%)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
