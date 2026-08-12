#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""
Corpus cleanup, stage 1: encoding normalisation.

Why this exists
---------------

Project Gutenberg files claim UTF-8 and are frequently not. Windows-1252
bytes -- 0x92 for a curly apostrophe, 0x93/0x94 for smart quotes -- appear
inside files served as UTF-8, along with Latin-1 accented letters, stray NULs
and truncated multibyte sequences.

That is not a cosmetic problem. Those bytes reached PCRE2 inside
openvino_tokenizers, which compiles its patterns in UTF-8 mode without
PCRE2_MATCH_INVALID_UTF, and PCRE2 documents matching invalid UTF as
undefined behaviour. In practice its JIT-compiled matcher decoded a garbage
codepoint and indexed a character-class table with it -- a wild read that
segfaulted the indexer when the address happened to be unmapped and returned
silent garbage when it did not. Because that depends on address-space layout,
identical input crashed 8 runs in 12 with ASLR on and 8 in 8 with it off.
Full account in docs/FINDINGS.md.

The embedder now carries a defensive version of this repair
(src/embed_openvino.cpp, to_valid_utf8) so no malformed file can crash the
server. But repairing bytes on every paragraph of every run, forever, is the
wrong place to solve it: the files are wrong, so fix the files. Everything
downstream -- hashing, offsets, text serving, the classifier -- then agrees
about what the corpus contains.

Policy
------

Transcode rather than replace. A byte that is not valid UTF-8 is decoded as
CP1252 where that byte is assigned, and Latin-1 otherwise, so 0x92 becomes a
real U+2019 that the embedding can use instead of U+FFFD noise. Valid UTF-8
is passed through untouched, which is the overwhelmingly common case and
costs one scan with no copy.

Also normalised, because each one otherwise makes the same text hash
differently depending on which copy of a file it came from: CRLF to LF, BOM
stripped, NUL removed.

Usage
-----

    tools/clean.py <src-tree> <dst-tree> [-j N] [--dry-run]

Writes a parallel tree; never modifies the source. Resumable -- a file whose
destination already exists and is newer than its source is skipped, so an
interrupted run continues rather than restarting.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# CP1252's C1 range (0x80-0x9F), where it differs from Latin-1. Five bytes
# are unassigned; None means "fall back to Latin-1", which keeps the output
# valid rather than raising.
CP1252_C1 = {
    0x80: "€", 0x82: "‚", 0x83: "ƒ", 0x84: "„",
    0x85: "…", 0x86: "†", 0x87: "‡", 0x88: "ˆ",
    0x89: "‰", 0x8A: "Š", 0x8B: "‹", 0x8C: "Œ",
    0x8E: "Ž", 0x91: "‘", 0x92: "’", 0x93: "“",
    0x94: "”", 0x95: "•", 0x96: "–", 0x97: "—",
    0x98: "˜", 0x99: "™", 0x9A: "š", 0x9B: "›",
    0x9C: "œ", 0x9E: "ž", 0x9F: "Ÿ",
}


def _repair(chunk: bytes) -> str:
    """Decode bytes that are not valid UTF-8, one byte at a time.

    CP1252 for assigned C1 bytes, Latin-1 for everything else. Latin-1 is
    total -- every byte 0x00-0xFF maps to a codepoint -- so this cannot
    fail, which is the property that matters when the input is arbitrary.
    """
    return "".join(CP1252_C1.get(b) or chr(b) for b in chunk)


def _decode(data: bytes) -> str:
    """Bytes to text, repairing invalid UTF-8. Total: never raises."""
    out = []
    pos = 0
    # Decode the valid runs wholesale and repair only the gaps. The error
    # handler reports the exact span that failed, so a clean file makes
    # exactly one pass with no per-byte work.
    while pos < len(data):
        try:
            out.append(data[pos:].decode("utf-8"))
            break
        except UnicodeDecodeError as e:
            out.append(data[pos:pos + e.start].decode("utf-8"))
            out.append(_repair(data[pos + e.start:pos + e.end]))
            pos += e.end

    # No surrogate filter here on purpose. Python's strict UTF-8 decoder
    # rejects the surrogate forms (ED A0 80 and friends) as invalid, so
    # they arrive at _repair() as ordinary bytes and come back as Latin-1
    # characters. A filter would be unreachable code that scans every
    # string -- mutation testing caught it as such (delete it and nothing
    # fails), which is the correct verdict, not a gap in the tests.
    # test_surrogate_half_rejected pins the behaviour that matters: the
    # output always encodes.
    return "".join(out)


def normalise_text(data: bytes) -> str:
    """Bytes to clean UTF-8 text. The single entry point: encoding repair
    plus the normalisations that are also about identity, since each one
    otherwise makes the same paragraph hash differently depending on which
    copy of a file it came from."""
    text = _decode(data)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    if text.startswith("﻿"):
        text = text[1:]
    return text


# Alias. Both names appear in the tests because the crash was caused by the
# encoding half alone, and it reads better to say so at the call site.
normalise_bytes = normalise_text


def classify_encoding(data: bytes) -> str:
    """What was actually wrong with this file. Reporting drives the
    decision about whether the policy above is right for this corpus."""
    try:
        data.decode("ascii")
        return "ascii"
    except UnicodeDecodeError:
        pass
    try:
        data.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    # Invalid UTF-8. C1 bytes present means CP1252 is the intended reading;
    # otherwise the high bytes are consistent with plain Latin-1.
    if any(0x80 <= b <= 0x9F for b in data):
        return "cp1252"
    return "latin-1"


def clean_file(src: Path, dst: Path, dry_run: bool = False):
    """Returns (encoding, changed, error)."""
    try:
        data = src.read_bytes()
    except OSError as e:
        return (None, False, f"{src}: {e}")

    enc = classify_encoding(data)
    text = normalise_text(data)
    out = text.encode("utf-8")
    changed = out != data

    # Always write, changed or not: the destination is a parallel tree, and
    # a file that needed no repair still has to appear in it. Writing only
    # changed files would drop most of the corpus, since the majority is
    # already valid. (This was previously an `and`/`or` chain that happened
    # to do the right thing by operator precedence.)
    if not dry_run:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(out)
        except OSError as e:
            return (enc, changed, f"{dst}: {e}")

    return (enc, changed, None)


def _job(args):
    src, dst, dry_run = args
    return clean_file(Path(src), Path(dst), dry_run)


def plan_jobs(src: Path, dst: Path, ext: str = ".txt",
              dry_run: bool = False) -> list:
    """Which files need processing, and where each one goes.

    Separate from main() so it can be tested: this is where the resume
    logic lives, and a wrong skip silently leaves stale files in the
    output tree -- the kind of bug that only shows up as a paragraph
    count that does not match.
    """
    jobs = []
    for root, _dirs, files in os.walk(src):
        for name in sorted(files):     # sorted: reproducible job order
            if not name.endswith(ext):
                continue
            s = Path(root) / name
            d = dst / s.relative_to(src)
            # Resumable: skip work already done in a previous run. Compare
            # mtimes rather than existence, so a source edited after a run
            # is picked up again.
            if not dry_run and d.exists() and \
               d.stat().st_mtime >= s.stat().st_mtime:
                continue
            jobs.append((str(s), str(d), dry_run))
    return jobs


def summarise(results) -> tuple:
    """Fold per-file results into (counts, changed, errors)."""
    counts = {}
    changed = 0
    errors = []
    for enc, ch, err in results:
        if enc:
            counts[enc] = counts.get(enc, 0) + 1
        if ch:
            changed += 1
        if err:
            errors.append(err)
    return counts, changed, errors


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("-j", type=int, default=max(1, (os.cpu_count() or 4) * 3 // 4),
                    help="worker processes (default: 3/4 of cores)")
    ap.add_argument("--ext", default=".txt", help="extension to process")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change, write nothing")
    a = ap.parse_args()

    if not a.src.is_dir():
        sys.exit(f"not a directory: {a.src}")

    jobs = plan_jobs(a.src, a.dst, a.ext, a.dry_run)

    print(f"{len(jobs)} files to process, {a.j} workers", file=sys.stderr)

    results = []
    with ProcessPoolExecutor(max_workers=a.j) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for done, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if done % 2000 == 0:
                print(f"  {done}/{len(jobs)}", file=sys.stderr)

    counts, changed, errors = summarise(results)
    print(f"\nprocessed {len(results)} files, {changed} changed",
          file=sys.stderr)
    for k in sorted(counts):
        print(f"  {k:10s} {counts[k]}", file=sys.stderr)
    if errors:
        print(f"\n{len(errors)} errors:", file=sys.stderr)
        for e in errors[:20]:
            print(f"  {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
