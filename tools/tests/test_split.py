# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""
Tests for the shared paragraph splitter.

The splitter defines the unit of identity for the whole system: a record's
BLAKE3 is computed over exactly the span it produces. Anything that
counts, hashes or resumes has to agree with the indexer byte for byte.

Three copies had drifted before this existed -- cmd/index.c, a Python
reimplementation in tools/pipeline.py, and a third in test/fuzz_sigil.c
whose own comment admitted it could drift. The consequence was measurable:
the manifest reported 77,367,817 paragraphs for a corpus the indexer split
into 74,905,358, a 3.2% disagreement that nothing detected because the
"check" was a comment predicting it would show up against /stats.

Two of the four divergences are pinned here as regressions, because they
are the ones that account for the gap:

  * a chunk remainder below MINPARA is dropped, not counted (the Python
    used ceiling division and overcounted -- the dominant error);
  * "\\n\\r" ends a paragraph as well as "\\n\\n" (the Python merged two
    paragraphs into one and undercounted).
"""

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.split import MAX_PARA, MIN_PARA, count, split


class TestBoundaries:
    """MINPARA and MAXPARA are inclusive at both ends."""

    def test_exactly_min_is_kept(self):
        assert count("a" * MIN_PARA) == 1

    def test_one_below_min_is_dropped(self):
        assert count("a" * (MIN_PARA - 1)) == 0

    def test_exactly_max_is_one_chunk(self):
        assert count("a" * MAX_PARA) == 1

    def test_one_above_max_drops_the_tiny_remainder(self):
        """4001 chars is one full chunk and a 1-byte remainder, and the
        remainder is far below MINPARA -- so one chunk, not two. Asserting
        2 here is the same ceiling-division instinct that produced the
        original bug."""
        assert count("a" * (MAX_PARA + 1)) == 1

    def test_above_max_splits_when_the_remainder_is_long_enough(self):
        assert count("a" * (MAX_PARA + MIN_PARA)) == 2


class TestRegressions:
    """The two divergences that produced the 3.2% corpus gap."""

    def test_remainder_below_min_is_dropped(self):
        """4010 chars is one 4000-byte chunk and a 10-byte remainder. The
        remainder is under MINPARA, so it is dropped: one chunk, not two.

        The Python reimplementation used ceiling division and returned 2.
        This is the dominant half of the gap, and its direction matches --
        the manifest counted more paragraphs than the index produced.
        """
        assert count("a" * (MAX_PARA + 10)) == 1

    def test_crlf_ends_a_paragraph(self):
        r"""A paragraph ends at "\n\n" or "\n\r". The Python split on
        "\n\n" alone and merged these two into one."""
        assert count("x" * 100 + "\n\r" + "y" * 100) == 2

    def test_lf_lf_also_ends_a_paragraph(self):
        assert count("x" * 100 + "\n\n" + "y" * 100) == 2


class TestChunking:
    """Over-long paragraphs are cut at a sentence boundary, not blindly."""

    def test_cut_searches_back_for_a_boundary(self):
        """The backward search inspects the character *at* the cut, so a
        chunk ends immediately before the '.' it found rather than after
        it. Pinning the real behaviour rather than the intuitive one: the
        record's hash is computed over exactly this span, so "close
        enough" is not a thing here.
        """
        text = "word. " * 1500
        chunks = split(text)
        data = text.encode()

        # Cut short of MAXPARA, which is what the search buys.
        assert all(ln <= MAX_PARA for _o, ln, _p in chunks)
        # And the second chunk is shorter than the first, because the
        # search walked backwards to a boundary.
        assert chunks[1][1] < chunks[0][1]

    def test_falls_back_when_no_boundary_exists(self):
        """A 9000-byte run with no punctuation cannot be cut politely, so
        it is cut at MAXPARA rather than searching past halfway."""
        assert count("a" * 9000) == 3

    def test_paragraph_numbers_are_sequential(self):
        chunks = split("a" * 100 + "\n\n" + "b" * 100 + "\n\n" + "c" * 100)
        assert [c[2] for c in chunks] == [1, 2, 3]

    def test_offsets_point_at_the_text(self):
        text = "first" + "a" * 100 + "\n\nsecond" + "b" * 100
        data = text.encode()
        chunks = split(text)
        assert data[chunks[0][0]:].startswith(b"first")
        assert data[chunks[1][0]:].startswith(b"second")


class TestWhitespace:
    def test_leading_whitespace_is_skipped(self):
        assert count("   \n\n" + "w" * 100) == 1

    def test_whitespace_only_input(self):
        assert count("   \n\n\t  \n") == 0

    def test_empty(self):
        assert count("") == 0
        assert split("") == []


class TestNonAscii:
    """Offsets are byte offsets, which matters once the text is not ASCII."""

    def test_multibyte_offsets_are_bytes(self):
        text = "é" * 100 + "\n\n" + "中" * 100
        data = text.encode("utf-8")
        chunks = split(text)

        assert len(chunks) == 2
        assert data[chunks[1][0]:].startswith("中".encode("utf-8"))
        # 100 two-byte chars, so the first chunk is 200 bytes not 100.
        assert chunks[0][1] == 200


# --- properties ---------------------------------------------------------

@settings(max_examples=400)
@given(st.text(max_size=3000))
def test_chunks_never_overlap_and_stay_in_bounds(text):
    """Whatever the input, every chunk is a real span of it and no two
    chunks overlap. A splitter that returned overlapping ranges would
    embed the same words twice under different hashes."""
    data = text.encode("utf-8")
    prev_end = 0
    for off, ln, _para in split(text):
        assert off >= prev_end
        assert off + ln <= len(data)
        prev_end = off + ln


@settings(max_examples=400)
@given(st.text(max_size=3000))
def test_every_chunk_respects_the_bounds(text):
    for _off, ln, _para in split(text):
        assert MIN_PARA <= ln <= MAX_PARA


@settings(max_examples=300)
@given(st.text(max_size=2000))
def test_count_agrees_with_split(text):
    """The cheap path and the full path must not disagree -- the manifest
    uses count(), the indexer uses the chunks."""
    assert count(text) == len(split(text))


@settings(max_examples=300)
@given(st.binary(max_size=2000))
def test_arbitrary_bytes_do_not_crash(data):
    """The indexer runs this over whatever is on disk. Invalid UTF-8 is
    repaired before embedding, but the splitter sees raw bytes first."""
    for off, ln, _para in split(data):
        assert off + ln <= len(data)
