#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""
Paragraph splitting: a binding to src/split.c, not a reimplementation.

The splitter defines the unit of identity for the whole system -- a
record's BLAKE3 is computed over exactly the span it produces -- so
anything that counts, hashes or resumes has to agree with the indexer byte
for byte.

There used to be three copies. cmd/index.c held the real one, this file's
predecessor re-implemented it in Python for the manifest's paragraph
count, and test/fuzz_sigil.c held a third with a comment admitting it
could drift. They did: the manifest reported 77,367,817 paragraphs for a
corpus the indexer split into 74,905,358, a silent 3.2% disagreement that
nothing checked -- the "check" was a comment claiming a mismatch would
show up against /stats, which is not a test and never ran.

The four ways the Python differed, each small and compounding:

  * it stripped a block after splitting rather than skipping leading
    whitespace before measuring;
  * it used ceiling division for over-long paragraphs, where the C
    advances a cursor and re-measures what remains;
  * it counted chunks shorter than MINPARA, which the C drops;
  * it split on "\\n\\n" alone, where the C also accepts "\\n\\r".

So this binds the C instead. src/utf8_repair.c was extracted for the same
reason and is the precedent: one implementation, a differential test
proving the binding agrees with it, and no second copy to drift.
"""

import ctypes
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Matches SIGIL_MINPARA / SIGIL_MAXPARA in include/sigil_split.h. Exposed
# for tests and for reporting; the splitting itself uses the C values.
MIN_PARA = 40
MAX_PARA = 4000


class Chunk(ctypes.Structure):
    """Mirrors sigil_chunk_t."""
    _fields_ = [
        ("off", ctypes.c_size_t),
        ("len", ctypes.c_size_t),
        ("para", ctypes.c_uint),
    ]


_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.POINTER(Chunk), ctypes.c_void_p)
_lib = None


def _build():
    """Compile src/split.c into a shared object beside the source.

    Built on demand rather than requiring `make` first: tools/ is used
    standalone, and a pipeline that fails because someone forgot to build
    the library is a worse failure than a one-second compile.
    """
    so = ROOT / "build" / "libsigilsplit.so"
    src = ROOT / "src" / "split.c"
    if so.exists() and so.stat().st_mtime >= src.stat().st_mtime:
        return so

    so.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["cc", "-O2", "-fPIC", "-shared", "-I", str(ROOT / "include"),
         str(src), "-o", str(so)],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cannot build {src}: {r.stderr[:400]}")
    return so


def _load():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(str(_build()))
        _lib.sigil_split.restype = ctypes.c_size_t
        _lib.sigil_split.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                                     _CALLBACK, ctypes.c_void_p]
        _lib.sigil_split_count.restype = ctypes.c_size_t
        _lib.sigil_split_count.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    return _lib


def split(text):
    """The chunks the indexer will produce, as (offset, length, para).

    Offsets and lengths are in *bytes* of the UTF-8 encoding, because that
    is what the C sees and what the record's offset field records. A
    caller wanting the text slices it out of the encoded form.
    """
    lib = _load()
    data = text.encode("utf-8") if isinstance(text, str) else text

    out = []

    @_CALLBACK
    def collect(c, _arg):
        out.append((c.contents.off, c.contents.len, c.contents.para))

    lib.sigil_split(data, len(data), collect, None)
    return out


def count(text):
    """How many chunks the indexer will take from this text."""
    lib = _load()
    data = text.encode("utf-8") if isinstance(text, str) else text
    return lib.sigil_split_count(data, len(data))


def main():
    """Split a file and report, for checking a disagreement by hand."""
    if len(sys.argv) < 2:
        sys.exit("usage: split.py <file> [--chunks]")

    data = Path(sys.argv[1]).read_bytes()
    chunks = split(data)
    print(f"{len(chunks)} chunks", file=sys.stderr)

    if "--chunks" in sys.argv:
        for off, ln, para in chunks:
            head = data[off:off + min(ln, 60)].decode("utf-8", "replace")
            print(f"  {para:5} @{off:<9} {ln:>5}B  {head!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
