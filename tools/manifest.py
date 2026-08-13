#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""
Corpus pipeline, stage 3b: the manifest.

One libtab row per book, recording both what the pipeline did and what the
catalogue says. Everything downstream reads this rather than re-deriving it:

  * Which revision won deduplication, and what encoding it was found in --
    the mirror keeps up to five copies of a book, and after the pipeline
    flattens them to <book-id>.txt that choice is otherwise unrecoverable.

  * The BLAKE3 of the cleaned text. Same hash function sigil uses for
    record identity, computed after cleaning so it is stable across
    re-runs. It is also what an incremental reindex compares against to
    decide whether a book needs re-embedding, and it catches the
    duplicate that filename-based dedup cannot see: two book ids whose
    cleaned text is byte-identical.

  * Title, authors, subjects, LoCC, language and both dates. This is the
    ground truth the clustering gets measured against.

Why libtab rather than a database
---------------------------------

Roughly 60,830 rows of write-once provenance is a lookup table, not a
search index. libtab is in-process, needs no daemon, and is what sigil
already uses for its own store, so the pipeline has no runtime dependency
the server does not already have. The columns are chosen to be
facet-shaped -- locc, language, death_year, subjects -- so a Solr view can
be built from this file later without changing the schema. That view would
be a consumer of the manifest, not a replacement for it.

Values are written verbatim
---------------------------

libtab encodes on write and decodes on read, so a title keeps its spaces,
its quotation marks and a leading '#', and the literal string "nil" stays
distinct from an absent value.

That was not always true. This file used to substitute the characters ndb
could not carry -- ASCII quotes became typographic ones, control
characters became spaces, and "nil" gained a thin space. Each was correct
for the library as it stood and is wrong now: applying them today would
corrupt values libtab stores correctly, so the helpers were deleted rather
than left available to call.

The one value still lost is NUL, which terminates the string at the C
boundary. Nothing inside libtab can carry it.
"""

import sys
from pathlib import Path

import sys
from pathlib import Path

# Run either as `python -m tools.pipeline` or as `tools/pipeline.py`. The
# latter puts tools/ on sys.path rather than the repo root, so the package
# imports below would fail; this makes both work.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCHEMA = "gutenberg-books"

# book is the row key. The rest divide into provenance (what the pipeline
# did) and catalogue metadata (what a cataloguer decided).
COLUMNS = [
    "book", "digest", "path", "encoding", "bytes", "paras",
    "title", "authors", "language", "subjects", "locc", "bookshelves",
    "death_year", "issued",
]

_META_FIELDS = ("title", "authors", "language", "subjects", "locc",
                "bookshelves", "issued")


def book_row(book, *, digest, path, encoding, nbytes, nparas, meta):
    """Assemble one row as a dict of strings.

    `meta` is a catalogue entry from metadata.load_catalog(), or None for
    the ~14 books in the mirror with no catalogue row. Those still get a
    row: a book that is indexed but missing from the manifest is invisible
    to everything downstream, which is worse than one with empty metadata.
    """
    row = {
        "book": str(book),
        "digest": digest,
        "path": str(path),
        "encoding": encoding,
        "bytes": str(nbytes),
        "paras": str(nparas),
    }
    for f in _META_FIELDS:
        row[f] = (meta or {}).get(f, "") or ""

    # Empty, not the string "None": an absent lifespan must not read back
    # as four characters that sort among real years.
    dy = (meta or {}).get("death_year")
    row["death_year"] = "" if dy is None else str(dy)
    return row


class Manifest:
    """Writer. Use as a context manager so the table is always closed.

    close() is what flushes -- verified by writing 5,000 rows without a
    commit() and reading all of them back. commit() is kept because it is
    the documented call and says what is meant, not because omitting it
    loses data.
    """

    def __init__(self, path):
        import libtab

        self.path = Path(path)
        if self.path.exists():
            self.path.unlink()
        cols = [libtab.Column(c) for c in COLUMNS]
        self._tab = libtab.Tabula.create(str(self.path), SCHEMA, cols)

    def add(self, row):
        # book_row() fills every column, and test_every_column_is_present
        # enforces that, so indexing rather than .get() is deliberate: a
        # missing column is a bug in the caller and should raise here
        # instead of writing a plausible empty cell.
        r = self._tab.add_row("book", row["book"])
        for col in COLUMNS[1:]:
            self._tab.set(r, col, row[col])

    def close(self):
        self._tab.commit()
        self._tab.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    @staticmethod
    def read(path):
        """book id -> row dict. Reopens the file, which is the only way to
        find out whether it was written correctly."""
        import libtab

        tab = libtab.Tabula.open(str(path))
        try:
            out = {}
            for r in tab.iter_rows():
                # libtab decodes on read; an absent cell is None.
                row = {c: (tab.get(r, c) or "") for c in COLUMNS}
                out[row["book"]] = row
            return out
        finally:
            tab.close()


def main():
    """Print a manifest, or a summary of one."""
    if len(sys.argv) < 2:
        sys.exit("usage: manifest.py <manifest.tab> [book-id ...]")

    rows = Manifest.read(sys.argv[1])

    if len(sys.argv) > 2:
        for b in sys.argv[2:]:
            r = rows.get(b)
            if r is None:
                print(f"{b}: not in manifest")
                continue
            for c in COLUMNS:
                print(f"  {c:12} {r[c]}")
            print()
        return 0

    print(f"{len(rows)} books", file=sys.stderr)
    n_meta = sum(1 for r in rows.values() if r["title"])
    n_date = sum(1 for r in rows.values() if r["death_year"])
    print(f"  with catalogue metadata: {n_meta}", file=sys.stderr)
    print(f"  with a death year:       {n_date}", file=sys.stderr)

    # Identical cleaned content under different book ids: the duplicate
    # that filename-based deduplication cannot detect.
    seen = {}
    dupes = 0
    for r in rows.values():
        if r["digest"] in seen:
            dupes += 1
        seen.setdefault(r["digest"], r["book"])
    print(f"  duplicate digests:       {dupes}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
