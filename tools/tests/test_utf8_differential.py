# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""
Differential property test: the C repair against the Python repair.

The UTF-8 salvage exists twice on purpose. tools/clean.py fixes the corpus
once, offline, which is the right place for it. src/utf8_repair.c fixes text
on its way into the tokenizer, because sigilfs indexes whatever tree it is
pointed at and one malformed byte from an uncleaned source must not be able
to segfault the server through PCRE2 (docs/FINDINGS.md).

Two implementations of the same mapping is a maintenance hazard: fix a bug in
one, forget the other, and a paragraph embeds differently depending on which
path it took -- silently, because both outputs are valid UTF-8 and neither
crashes. So they are checked against each other over generated input rather
than trusted to stay in step.

The C side is reached through ctypes against libsigil.a, built by the same
Makefile CI already runs. If the library is missing the tests skip loudly
rather than passing vacuously.
"""

import ctypes
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from clean import _decode  # noqa: E402  (path set above)


def _build_shared_lib(tmp_path_factory):
    """Build a shared object holding just the repair unit.

    libsigil.a is a static archive; ctypes needs a .so. Compiling the one
    translation unit directly keeps this independent of whether the rest of
    the library built (it has no dependencies beyond libc).
    """
    out = tmp_path_factory.mktemp("utf8") / "libutf8.so"
    src = ROOT / "src" / "utf8_repair.c"
    if not src.exists():
        pytest.skip(f"{src} not present")
    r = subprocess.run(
        ["cc", "-O2", "-fPIC", "-shared", "-I", str(ROOT / "include"),
         str(src), "-o", str(out)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"cannot build utf8_repair.c: {r.stderr[:400]}")
    return out


@pytest.fixture(scope="session")
def clib(tmp_path_factory):
    lib = ctypes.CDLL(str(_build_shared_lib(tmp_path_factory)))
    lib.sigil_utf8_repair.restype = ctypes.POINTER(ctypes.c_char)
    lib.sigil_utf8_repair.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                                      ctypes.POINTER(ctypes.c_size_t)]
    lib.sigil_utf8_valid.restype = ctypes.c_int
    lib.sigil_utf8_valid.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    lib.sigil_utf8_seq.restype = ctypes.c_size_t
    lib.sigil_utf8_seq.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    return lib


def c_repair(lib, data: bytes) -> str:
    n = ctypes.c_size_t(0)
    p = lib.sigil_utf8_repair(data, len(data), ctypes.byref(n))
    if not p:
        raise MemoryError("sigil_utf8_repair returned NULL")
    return ctypes.string_at(p, n.value).decode("utf-8")


def py_repair(data: bytes) -> str:
    """The Python side of the comparison.

    _decode() is the encoding repair alone. normalise_text() additionally
    strips BOM/NUL and folds CRLF, which the C guard deliberately does not
    do -- those are corpus-cleanup decisions, not crash-safety ones, and the
    embedder must not silently alter text it was asked to embed. Comparing
    against _decode keeps the test honest about which half is shared.
    """
    return _decode(data)


# --- the property that keeps the two in step ----------------------------

@settings(max_examples=600)
@given(st.binary(max_size=1500))
def test_c_and_python_agree(clib, data):
    assert c_repair(clib, data) == py_repair(data)


@settings(max_examples=300)
@given(st.text(max_size=800))
def test_agree_on_valid_text(clib, text):
    data = text.encode("utf-8")
    assert c_repair(clib, data) == py_repair(data) == text


@settings(max_examples=300)
@given(st.binary(max_size=1000))
def test_c_output_is_always_valid_utf8(clib, data):
    """The C guard's own invariant, checked independently of Python:
    whatever goes in, what PCRE2 sees is valid UTF-8."""
    out = c_repair(clib, data)
    out.encode("utf-8")
    raw = out.encode("utf-8")
    assert clib.sigil_utf8_valid(raw, len(raw)) == 1


# --- the specific bytes from the crash ----------------------------------

class TestKnownBytes:
    def test_cp1252_apostrophe(self, clib):
        assert c_repair(clib, b"don\x92t") == "don’t"

    def test_latin1_accent(self, clib):
        assert c_repair(clib, b"caf\xe9") == "café"

    def test_valid_utf8_untouched(self, clib):
        s = "café — naïve ’ 中文 \U0001F600"
        assert c_repair(clib, s.encode("utf-8")) == s

    def test_unassigned_c1_falls_back_to_latin1(self, clib):
        # 0x81/0x8D/0x8F/0x90/0x9D are unassigned in CP1252; both sides
        # must make the same choice, not merely both produce valid output.
        for b in (0x81, 0x8D, 0x8F, 0x90, 0x9D):
            d = bytes([ord("a"), b, ord("b")])
            assert c_repair(clib, d) == py_repair(d)


class TestSequenceValidation:
    """sigil_utf8_seq is the decision every other function rests on."""

    @pytest.mark.parametrize("data,expected", [
        (b"a", 1),
        ("é".encode("utf-8"), 2),
        ("中".encode("utf-8"), 3),
        ("\U0001F600".encode("utf-8"), 4),
        (b"\x80", 0),              # lone continuation
        (b"\xc0\x80", 0),          # overlong NUL
        (b"\xc1\xbf", 0),          # overlong
        (b"\xed\xa0\x80", 0),      # surrogate half
        (b"\xf5\x80\x80\x80", 0),  # > U+10FFFF
        (b"\xf4\x90\x80\x80", 0),  # > U+10FFFF
        (b"\xe0\x80\x80", 0),      # overlong 3-byte
        (b"\xc2", 0),              # truncated
        (b"\xe2\x82", 0),          # truncated 3-byte
    ])
    def test_sequence_length(self, clib, data, expected):
        assert clib.sigil_utf8_seq(data, len(data)) == expected

    def test_empty_is_zero(self, clib):
        assert clib.sigil_utf8_seq(b"", 0) == 0
