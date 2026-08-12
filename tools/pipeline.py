#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""
The corpus pipeline: raw Gutenberg mirror in, indexable books plus a
manifest out.

    tools/pipeline.py /mnt/bulk/gutenberg /mnt/bulk/gutenberg-books \\
        --catalog /mnt/bulk/pg_catalog.csv -j 12

Order of operations, and why
----------------------------

Deduplicate first, then clean. The mirror keeps superseded revisions under
old/ and old/old/ plus several encodings of each text -- 331 files for 118
books in the sampled subtree, 2.8x. Cleaning before deduplicating would
repair the encoding of two files in three that are then discarded.

Reordering is safe because the ranking in boilerplate._rank() sorts on path
depth and filename suffix, both metadata: the file chosen does not depend on
its contents, so dedup gives the same answer before or after cleaning.

Then, per chosen file: repair encoding (stage 1), strip the licence
envelope (stage 2), write <book-id>.txt, and record a manifest row (stage
3) carrying provenance and catalogue metadata.

What the manifest is for
------------------------

It is the record of what this run did, and the only place the discarded
information survives -- once the tree is flattened to <book-id>.txt, which
revision won and what encoding it was found in are otherwise unrecoverable.
The BLAKE3 of the cleaned text is sigil's own identity function, so an
incremental reindex can compare against it, and two books whose cleaned
text hashes identically are the duplicate that filename-based dedup cannot
see.

Resumability
------------

A book whose output is newer than its source is skipped, the same rule
stage 1 used. The manifest is rewritten from scratch each run, because a
partial manifest describing a full tree would be worse than none.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import sys
from pathlib import Path

# Run either as `python -m tools.pipeline` or as `tools/pipeline.py`. The
# latter puts tools/ on sys.path rather than the repo root, so the package
# imports below would fail; this makes both work.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import boilerplate, clean
from tools.manifest import Manifest, book_row
from tools.metadata import load_catalog

# Matches Minpara/Maxpara in cmd/index.c: the manifest's paragraph count
# has to mean the same thing the indexer will count.
MIN_PARA = 40
MAX_PARA = 4000


def count_paragraphs(text):
    """How many paragraphs the indexer will take from this text.

    Duplicates the splitting rule in cmd/index.c rather than importing it
    -- that file is plan9port C. Kept deliberately simple: this is a
    count for the manifest, not the split itself, and a discrepancy shows
    up immediately as a mismatch against /stats after indexing.
    """
    n = 0
    for para in text.split("\n\n"):
        p = para.strip()
        if MIN_PARA <= len(p) <= MAX_PARA:
            n += 1
        elif len(p) > MAX_PARA:
            # index.c chunks an over-long paragraph rather than dropping
            # it, so count the chunks.
            n += (len(p) + MAX_PARA - 1) // MAX_PARA
    return n


def digest_text(text):
    """BLAKE3 of the cleaned text, matching sigil's identity function.

    Falls back to SHA-256 where blake3 is not installed: the manifest
    still works, the digests are simply not comparable with a sigil store
    built elsewhere. The algorithm is recorded so that is visible rather
    than silent.
    """
    data = text.encode("utf-8")
    try:
        import blake3

        return "blake3:" + blake3.blake3(data).hexdigest()
    except ImportError:
        import hashlib

        return "sha256:" + hashlib.sha256(data).hexdigest()


def process_one(args):
    """Clean and strip one chosen file. Returns a dict for the manifest,
    or an error string."""
    src, dst, book = args
    src, dst = Path(src), Path(dst)

    try:
        raw = src.read_bytes()
    except OSError as e:
        return {"book": book, "error": f"{src}: {e}"}

    encoding = clean.classify_encoding(raw)
    text = boilerplate.strip_boilerplate(clean.normalise_text(raw))

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")
    except OSError as e:
        return {"book": book, "error": f"{dst}: {e}"}

    return {
        "book": book,
        "path": str(src),
        "encoding": encoding,
        "digest": digest_text(text),
        "bytes": len(text.encode("utf-8")),
        "paras": count_paragraphs(text),
        "error": None,
    }


def plan(src_root, dst_root, ext=".txt", keep_duplicates=False):
    """(src, dst, book) for each book worth processing.

    Deduplication happens here, before any file is read.
    """
    all_files = boilerplate.find_files(src_root, ext)
    chosen = boilerplate.select_books(all_files,
                                      keep_duplicates=keep_duplicates)
    jobs = []
    for f in chosen:
        book = boilerplate.book_id(f) or f.stem
        jobs.append((str(f), str(Path(dst_root) / f"{book}.txt"), book))
    return len(all_files), jobs


def format_report(nfiles, nbooks, written, skipped, joined, errors):
    """The end-of-run summary. Built rather than printed so the numbers
    can be tested -- they are the only feedback a 60,830-book run gives,
    and a wrong count is how a partial run passes for a complete one."""
    lines = [
        f"\n{nfiles} files -> {nbooks} books "
        f"({nfiles - nbooks} duplicates dropped)",
        f"  written:            {written}",
        f"  skipped (current):  {skipped}",
        f"  catalogue metadata: {joined}",
    ]
    if errors:
        lines.append(f"\n{len(errors)} errors:")
        lines += [f"  {e}" for e in errors[:20]]
        if len(errors) > 20:
            lines.append(f"  ... and {len(errors) - 20} more")
    return "\n".join(lines)


def exit_status(errors):
    return 1 if errors else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("src", type=Path, help="raw Gutenberg mirror")
    ap.add_argument("dst", type=Path, help="output tree")
    ap.add_argument("--catalog", type=Path,
                    help="pg_catalog.csv; without it the manifest carries "
                         "provenance but no metadata")
    ap.add_argument("-j", type=int,
                    default=max(1, (os.cpu_count() or 4) * 3 // 4))
    ap.add_argument("--ext", default=".txt")
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--limit", type=int, help="stop after N books, for a "
                                              "trial run")
    a = ap.parse_args()

    if not a.src.is_dir():
        sys.exit(f"not a directory: {a.src}")

    cat = load_catalog(a.catalog) if a.catalog else {}
    if a.catalog:
        print(f"catalogue: {len(cat)} text works", file=sys.stderr)

    nfiles, jobs = plan(a.src, a.dst, a.ext, a.keep_duplicates)
    if a.limit:
        jobs = jobs[:a.limit]

    # Resume: a book whose output is newer than its source is done.
    #
    # Skipped books still need a manifest row. The manifest is rewritten
    # whole each run, so omitting them would leave a second run with an
    # empty manifest beside a full tree -- which is how a resumable
    # pipeline destroys its own record. Their rows are rebuilt by reading
    # the existing output, which is cheap next to re-embedding.
    todo, done = [], []
    for src, dst, book in jobs:
        d = Path(dst)
        if d.exists() and d.stat().st_mtime >= Path(src).stat().st_mtime:
            done.append((src, dst, book))
        else:
            todo.append((src, dst, book))
    skipped = len(done)

    print(f"{nfiles} files, {len(jobs)} books, {len(todo)} to process, "
          f"{skipped} already current, {a.j} workers", file=sys.stderr)

    a.dst.mkdir(parents=True, exist_ok=True)
    results, errors = [], []
    with ProcessPoolExecutor(max_workers=a.j) as ex:
        futs = [ex.submit(process_one, j) for j in todo]
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r.get("error"):
                errors.append(r["error"])
            else:
                results.append(r)
            if n % 2000 == 0:
                print(f"  {n}/{len(todo)}", file=sys.stderr)

    # Rows for the books this run skipped, read back from the output they
    # left behind. Without these the manifest would describe only what
    # this run happened to touch.
    for src, dst, book in done:
        try:
            text = Path(dst).read_text(encoding="utf-8")
        except OSError as e:
            errors.append(f"{dst}: {e}")
            continue
        results.append({
            "book": book, "path": src,
            "encoding": "", "digest": digest_text(text),
            "bytes": len(text.encode("utf-8")),
            "paras": count_paragraphs(text), "error": None,
        })

    # Rewritten whole: a partial manifest describing a full tree would be
    # worse than none.
    joined = 0
    with Manifest(a.dst / "manifest.tab") as m:
        for r in sorted(results, key=lambda r: int(r["book"])
                        if r["book"].isdigit() else 0):
            meta = cat.get(r["book"])
            if meta:
                joined += 1
            m.add(book_row(r["book"], digest=r["digest"], path=r["path"],
                           encoding=r["encoding"], nbytes=r["bytes"],
                           nparas=r["paras"], meta=meta))

    print(format_report(nfiles, len(jobs), len(results), skipped, joined,
                        errors), file=sys.stderr)
    return exit_status(errors)


if __name__ == "__main__":
    sys.exit(main())
