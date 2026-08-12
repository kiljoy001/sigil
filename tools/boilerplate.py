#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""
Corpus cleanup, stage 2: boilerplate removal and per-book deduplication.

Measured on the sampled subtree, indexing the raw mirror does two things to
the index that have nothing to do with the books:

  * 331 files for 118 books -- 2.8x duplication. Project Gutenberg keeps
    superseded revisions under old/ and old/old/, plus several encodings of
    the same text. Indexed together they inflate every similarity score with
    paragraphs that are identical by construction, which is exactly the
    signal sigil is trying to measure. tools/gutenberg.py documented this
    for its CSV export; here it is fixed for the tree itself.

  * Every file carries a licence header and footer of several hundred words.
    Indexed as content that boilerplate becomes the most common text in the
    corpus, appearing in every book's neighbourhood.

Formats are transcribed from the mirror rather than assumed. Three spellings
of the marker occur -- THE/THIS, with and without a space after the asterisks
-- and 114 of 118 current files carry one. The four that do not use an older
prose header; they are left unstripped rather than guessed at, because
dropping a whole book is far worse than keeping its header.

Usage:

    tools/boilerplate.py <src-tree> <dst-tree> [-j N] [--keep-duplicates]

Reads the tree produced by tools/clean.py (stage 1) and writes a tree of one
clean, deboilerplated file per book.
"""

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# The START/END lines. Tolerant of the observed spelling differences, of
# trailing whitespace, and of leading indentation -- some files are
# indented in their entirety, marker included:
#
#         The Project Gutenberg EBook of Sinister Street, vol. 1, ...
#     *** START OF THIS PROJECT GUTENBERG EBOOK SINISTER STREET, VOL. 1 ***
#
# A pattern anchored hard at ^\*\*\* misses those and the whole licence
# envelope survives into the indexed text. Measured over 4,000 books of the
# cleaned corpus: 118 still carried licence text, and this is why.
#
# The title may also wrap. Gutenberg wrapped long ones at a fixed width,
# so the closing *** lands on the next line:
#
#     ***START OF THE PROJECT GUTENBERG EBOOK THE CHAPEL OF THE HOLY SPIRIT IN
#     THE CHURCH OF ST. PETER'S, ... ***
#
# A $-anchored pattern misses those. Both problems together account for
# every leftover measured: of 4,000 cleaned books, 118 still carried
# licence text, and after allowing indentation and a wrapped title, 0 do.
#
# Bounded {0,300} rather than .*? with DOTALL, so a file containing a
# stray *** cannot run on and swallow the opening of the book. The
# longest real marker in the mirror is comfortably inside that.
#
# The leading class includes U+FEFF: three books carry a BOM directly
# before the marker on the same line, which clean.normalise_text does
# not remove because it only strips one at position 0 of the file.
#
# Still anchored to a line start so a mention in running prose cannot
# match.
_START = re.compile(
    r"^[﻿ \t]*\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.{0,300}?\*\*\*",
    re.IGNORECASE | re.MULTILINE | re.DOTALL)
_END = re.compile(
    r"^[﻿ \t]*\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.{0,300}?\*\*\*",
    re.IGNORECASE | re.MULTILINE | re.DOTALL)

# "Produced by ..." credits sit immediately after the START marker and run
# until a blank line. Transcription credit, not the work.
_PRODUCED = re.compile(
    r"\A\s*(?:Produced|Prepared|Transcribed|E-?text prepared)\s+by\b.*?"
    r"(?:\n\s*\n|\Z)",
    re.IGNORECASE | re.DOTALL)


def strip_boilerplate(text: str) -> str:
    """Return the book, without the Gutenberg envelope.

    Only ever removes: the result is always a substring of the input, so
    this stage cannot invent text that was not in the file. When the markers
    are missing or malformed the text is returned as-is -- keeping a header
    is a cosmetic problem, dropping a book is a data-loss one.
    """
    if not text:
        return text

    start = _START.search(text)
    body_from = start.end() if start else 0

    # Search for END only after START, so a file that somehow contains the
    # end line first cannot produce an inverted slice.
    end = _END.search(text, body_from)
    body_to = end.start() if end else len(text)

    if body_to <= body_from:
        return text                      # malformed; keep everything

    body = text[body_from:body_to]

    produced = _PRODUCED.match(body)
    if produced:
        candidate = body[produced.end():]
        # Only drop the credit if a book remains. A file that is nothing but
        # a credit line is more likely misparsed than genuinely empty.
        if candidate.strip():
            body = candidate

    return body.strip("\n")


# --- deduplication ------------------------------------------------------

_BOOKDIR = re.compile(r"^\d+$")

# A filename that is itself a book number: 1602.txt, 12823-0.txt.
# The suffix is the encoding variant, not part of the id.
_BOOKFILE = re.compile(r"^(\d+)(?:-\d+)?$")


def book_id(path: Path) -> str | None:
    """The book number.

    The filename wins when it is itself a book number -- 1602.txt or
    12823-0.txt -- because it carries the actual Text#. The directory is
    the fallback for legacy names that do not: old/old/3ddcc10.txt belongs
    to book 1007 and its name says otherwise.

    The directory alone is not enough, and preferring it was wrong for 22
    files in the mirror. Gutenberg's layout is normally
    <fanout digits>/<id>/<file>, but some files sit directly in the fanout
    directory -- /1/6/0/1602.txt -- where the deepest numeric component is
    the single digit 0, so eighteen books were filed under ids like "0"
    and "9". Four more sit in the wrong book's directory entirely
    (16529/16520.txt), which no directory rule can get right.

    Verified across the whole mirror: preferring the filename produces no
    id collisions.
    """
    m = _BOOKFILE.match(path.stem)
    if m:
        return m.group(1)

    for part in reversed(path.parent.parts):
        if _BOOKDIR.match(part):
            return part
    return None


def _rank(path: Path) -> tuple:
    """Sort key; lower is better.

    Revision dominates encoding. An old UTF-8 file is still the wrong
    edition, and stage 1 repairs encoding anyway -- whereas no stage can
    recover a superseded revision's corrections.

    Within a revision the suffix says the encoding: -0 is UTF-8, -8 is
    8-bit, bare is ASCII. Preferring -0 means the repair pass has less to
    guess at.
    """
    depth = sum(1 for p in path.parts if p == "old")

    stem = path.stem
    if stem.endswith("-0"):
        enc = 0
    elif stem.endswith("-8"):
        enc = 2
    elif stem.endswith("-5"):
        enc = 3            # Big-5, seen occasionally
    else:
        enc = 1            # plain ASCII

    # Filename last, so ties resolve identically on every filesystem and
    # every run rather than following directory order.
    return (depth, enc, str(path))


def pick_best(paths) -> Path | None:
    """One file per book. Deterministic: same set in, same choice out,
    regardless of the order the walk produced them."""
    paths = list(paths)
    if not paths:
        return None
    return min(paths, key=_rank)


def _job(args):
    src, dst = args
    src, dst = Path(src), Path(dst)
    try:
        text = src.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return (0, 0, f"{src}: {e}")

    out = strip_boilerplate(text)
    removed = len(text) - len(out)
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(out, encoding="utf-8")
    except OSError as e:
        return (0, 0, f"{dst}: {e}")
    return (1, removed, None)


# Files that sit in a book directory but are not the book. 11,382 of the
# mirror's 157,106 .txt files are ancillary (7.2%), and 1,410 of those are
# LICENSE.txt. Usually dedup discards them because the real text is
# alongside, but where a title ships only as HTML or EPUB there is no
# competing file -- and six books were indexed as MathJax build notes
# before this filter existed, all sharing one digest.
#
# Matched by name rather than by pattern: legacy book names like
# 8tgcm10.txt and 8rome10.txt are also non-numeric, and dropping those
# would discard real books.
_NOT_A_BOOK = frozenset({
    "license.txt", "readme.txt", "readme-license.txt", "readme-math.txt",
    "contents.txt", "instructions.txt", "cd-dvd-readme.txt",
    "donate-howto.txt", "pnote.txt",
})


def is_book_file(path: Path) -> bool:
    """Whether this file is a work rather than packaging around one."""
    return path.name.lower() not in _NOT_A_BOOK


def find_files(src: Path, ext: str = ".txt") -> list:
    """Every candidate file under src, in a reproducible order."""
    return sorted(p for p in
                  (Path(r) / n
                   for r, _d, fs in os.walk(src)
                   for n in fs if n.endswith(ext))
                  if is_book_file(p))


def select_books(all_files, keep_duplicates: bool = False) -> list:
    """One file per book.

    Split out of main() so the selection rule can be tested directly. It
    decides what gets indexed and what is discarded -- on the sampled
    subtree that is 331 files down to 110 -- and a wrong choice here means
    indexing a superseded revision with no visible symptom.
    """
    if keep_duplicates:
        return list(all_files)

    by_book = {}
    unnumbered = []
    for f in all_files:
        bid = book_id(f)
        if bid is None:
            # No numeric directory: not Gutenberg's layout. Keep it rather
            # than silently dropping files this rule does not understand.
            unnumbered.append(f)
        else:
            by_book.setdefault(bid, []).append(f)

    # sorted() over the keys so the output order does not depend on dict
    # insertion, which follows directory-walk order.
    return [pick_best(by_book[k]) for k in sorted(by_book)] + unnumbered


def plan_jobs(chosen, dst: Path) -> list:
    """(src, dst) pairs, flattened to <book-id>.txt.

    The source path encodes revision and encoding, and neither means
    anything once one file has been picked.
    """
    jobs = []
    for f in chosen:
        bid = book_id(f)
        name = f"{bid}.txt" if bid else f.name
        jobs.append((str(f), str(dst / name)))
    return jobs


def exit_status(errors) -> int:
    """0 when every book was written, 1 when any failed."""
    return 1 if errors else 0


def format_report(written: int, removed_total: int, errors) -> str:
    """The end-of-run summary, as a string rather than a print.

    Bytes, not MB: header size varies from ~100 bytes (marker only) to
    ~21 KB (full licence), so a MB figure rounds to 0.0 on small runs and
    reads as "nothing happened" when the stage worked correctly. That
    misreading actually happened, which is why the per-book average is
    here too.
    """
    per = removed_total / max(written, 1)
    lines = [f"\nwrote {written} books, stripped {removed_total:,} bytes "
             f"of boilerplate ({per:,.0f} per book)"]
    if errors:
        lines.append(f"{len(errors)} errors:")
        lines += [f"  {e}" for e in errors[:20]]
        if len(errors) > 20:
            lines.append(f"  ... and {len(errors) - 20} more")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("src", type=Path, help="tree from tools/clean.py")
    ap.add_argument("dst", type=Path)
    ap.add_argument("-j", type=int,
                    default=max(1, (os.cpu_count() or 4) * 3 // 4))
    ap.add_argument("--ext", default=".txt")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="do not deduplicate; process every file")
    a = ap.parse_args()

    if not a.src.is_dir():
        sys.exit(f"not a directory: {a.src}")

    all_files = find_files(a.src, a.ext)
    chosen = select_books(all_files, keep_duplicates=a.keep_duplicates)
    jobs = plan_jobs(chosen, a.dst)
    skipped = len(all_files) - len(chosen)

    print(f"{len(all_files)} files, {len(jobs)} books "
          f"({skipped} duplicates dropped), {a.j} workers", file=sys.stderr)

    written = 0
    removed_total = 0
    errors = []
    with ProcessPoolExecutor(max_workers=a.j) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for n, (i, rm, err) in enumerate(
                (f.result() for f in as_completed(futs)), 1):
            written += i
            removed_total += rm
            if err:
                errors.append(err)
            if n % 2000 == 0:
                print(f"  {n}/{len(jobs)}", file=sys.stderr)

    print(format_report(written, removed_total, errors), file=sys.stderr)
    return exit_status(errors)


if __name__ == "__main__":
    sys.exit(main())
