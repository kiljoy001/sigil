# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""
Tests for the encoding-normalisation stage of the corpus pipeline.

Written before the implementation, and deliberately anchored on the bug that
caused this stage to exist: Project Gutenberg files carry Windows-1252 bytes
(0x92 curly apostrophe and friends) inside files served as UTF-8. Those bytes
reached PCRE2 inside openvino_tokenizers, which documents matching invalid
UTF-8 as undefined behaviour, and it read wild memory -- fatally or silently
depending on address-space layout. See docs/FINDINGS.md.

So the property that actually matters is not "the output looks nicer". It is:

    every byte sequence in, valid UTF-8 out, always.

That is what the property test below asserts over arbitrary bytes, because
the corpus is 60,830 files and the case nobody thought of is the one that
segfaults the indexer at 3 a.m.
"""

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.clean import normalise_text, normalise_bytes, classify_encoding


# --- the specific bytes that caused the crash ---------------------------

class TestCp1252:
    """CP1252 C1 bytes are the actual corpus defect, not a hypothetical."""

    def test_curly_apostrophe_becomes_real_quote(self):
        # 0x92 appeared 2,838 times in the 13-file set that crashed the
        # indexer. It must become U+2019, not U+FFFD: the text says
        # "don't", and the embedding should see that.
        assert normalise_bytes(b"don\x92t") == "don’t"

    def test_smart_quotes_pair(self):
        assert normalise_bytes(b"\x93quoted\x94") == "“quoted”"

    def test_dashes_and_ellipsis(self):
        assert normalise_bytes(b"a\x97b") == "a—b"      # em dash
        assert normalise_bytes(b"a\x96b") == "a–b"      # en dash
        assert normalise_bytes(b"wait\x85") == "wait…"  # ellipsis

    def test_undefined_c1_bytes_do_not_crash(self):
        # 0x81, 0x8D, 0x8F, 0x90, 0x9D are unassigned in CP1252. They must
        # still produce valid UTF-8 rather than raising.
        for b in (0x81, 0x8D, 0x8F, 0x90, 0x9D):
            out = normalise_bytes(bytes([ord("a"), b, ord("b")]))
            out.encode("utf-8")   # must not raise


class TestValidInputUnchanged:
    """The common case is a file that is already fine. Do not touch it."""

    def test_ascii_identical(self):
        assert normalise_bytes(b"plain ascii text") == "plain ascii text"

    def test_valid_utf8_preserved(self):
        s = "café — naïve ’ 中文 \U0001F600"
        assert normalise_bytes(s.encode("utf-8")) == s

    def test_utf8_bom_stripped(self):
        # A BOM at the head of a UTF-8 file is not content. Left in place it
        # becomes a zero-width character at the start of the first paragraph.
        assert normalise_bytes(b"\xef\xbb\xbfhello") == "hello"

    def test_crlf_normalised(self):
        # Gutenberg ships CRLF. The paragraph splitter handles it, but
        # carrying \r into the embedder means the same paragraph hashes
        # differently depending on which copy of the file it came from.
        assert normalise_bytes(b"line one\r\nline two\r\n") == \
            "line one\nline two\n"


class TestLatin1:
    """Files that are genuinely Latin-1, not CP1252."""

    def test_high_latin1_letters(self):
        # 0xE9 is e-acute in both Latin-1 and CP1252, and invalid alone in
        # UTF-8. It must survive as the letter, not become a replacement.
        assert normalise_bytes(b"caf\xe9") == "café"


class TestTruncatedSequences:
    """Mid-character truncation is what a chunked download leaves behind."""

    def test_truncated_multibyte_at_end(self):
        out = normalise_bytes("café".encode("utf-8")[:-1])
        out.encode("utf-8")
        assert out.startswith("caf")

    def test_lone_continuation_byte(self):
        normalise_bytes(b"a\x80b").encode("utf-8")

    def test_overlong_encoding_rejected(self):
        # C0 80 is an overlong NUL: a classic filter-bypass form. It must
        # not decode to U+0000.
        out = normalise_bytes(b"a\xc0\x80b")
        out.encode("utf-8")
        assert "\x00" not in out

    def test_surrogate_half_rejected(self):
        # ED A0 80 is a UTF-16 surrogate encoded as UTF-8. Python refuses
        # to encode surrogates, so if one leaks through, .encode() raises
        # and this test fails -- which is the point.
        normalise_bytes(b"a\xed\xa0\x80b").encode("utf-8")


class TestNulBytes:
    def test_nul_removed(self):
        # 251 files in the sampled subtree contain NUL. It is not text, and
        # it truncates anything that treats the buffer as a C string.
        assert "\x00" not in normalise_bytes(b"a\x00b")


# --- the property that the whole stage exists to guarantee --------------

@settings(max_examples=500)
@given(st.binary(max_size=2000))
def test_any_bytes_produce_valid_utf8(data):
    """
    The invariant. Whatever is in the corpus -- any byte sequence at all --
    the pipeline emits text that encodes cleanly as UTF-8.

    This is the assertion that would have prevented the crash: the indexer
    only ever sees output from this function, and PCRE2 only ever sees
    valid UTF-8.
    """
    out = normalise_text(data)
    assert isinstance(out, str)
    out.encode("utf-8")            # raises on surrogates or lone halves
    assert "\x00" not in out


@settings(max_examples=300)
@given(st.text(max_size=1000))
def test_valid_text_round_trips(text):
    """
    Text that is already valid UTF-8 survives unchanged, except for the
    normalisations we deliberately apply (CRLF, BOM, NUL).

    Without this, "always emit valid UTF-8" could be satisfied by returning
    the empty string.
    """
    expected = (text.replace("\r\n", "\n")
                    .replace("\r", "\n")      # lone CR: classic-Mac ending
                    .replace("\x00", ""))
    if expected.startswith("﻿"):
        expected = expected[1:]
    assert normalise_text(text.encode("utf-8")) == expected


@settings(max_examples=200)
@given(st.binary(max_size=1000))
def test_idempotent(data):
    """Cleaning twice equals cleaning once -- so re-running the pipeline
    over an already-clean tree is a no-op rather than progressive damage."""
    once = normalise_text(data)
    twice = normalise_text(once.encode("utf-8"))
    assert once == twice


# --- encoding classification -------------------------------------------

class TestClassify:
    """Reporting what was wrong is half the value of a cleanup pass."""

    def test_clean_ascii(self):
        assert classify_encoding(b"hello") == "ascii"

    def test_clean_utf8(self):
        assert classify_encoding("café".encode("utf-8")) == "utf-8"

    def test_cp1252_detected(self):
        assert classify_encoding(b"don\x92t") == "cp1252"

    def test_latin1_detected(self):
        # 0xE9 with no C1 bytes present: indistinguishable from CP1252 by
        # rule, and Latin-1 is the safer label since the mapping agrees.
        assert classify_encoding(b"caf\xe9") in ("latin-1", "cp1252")
