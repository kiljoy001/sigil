# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""
Tests for stage 2: boilerplate removal and per-book deduplication.

Both stages exist because of what indexing the raw mirror actually does to
the index, measured on the sampled subtree:

  * 331 files for 118 books -- 2.8x duplication. Every book is indexed about
    three times, so similarity scores are inflated by paragraphs that are
    identical by construction. tools/gutenberg.py already documented this;
    the pipeline is where it gets fixed.

  * Every file carries a Project Gutenberg licence header and footer of
    several hundred words. Indexed as content, that boilerplate is the most
    common text in the corpus and appears in every book's neighbourhood.

The marker formats here are transcribed from the mirror, not invented: three
spellings of the START/END line (THE/THIS, with and without a space after
the asterisks) and an older prose header with no marker at all. 114 of 118
current files carry a marker; the ones that do not are almost all old/
duplicates that deduplication discards anyway.
"""

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boilerplate import strip_boilerplate, pick_best, book_id, _rank

BODY = "Real book text.\n\nSecond paragraph of the actual work.\n"


def wrap(start_marker, end_marker, header="Title: X\nAuthor: Y\n"):
    return (f"{header}\n{start_marker}\n\n{BODY}\n{end_marker}\n"
            "This file should be named 1007-0.txt\n"
            "Updated editions will replace the previous one.\n")


class TestMarkerVariants:
    """All three spellings observed in the mirror must be handled."""

    def test_the_spelling(self):
        t = wrap("*** START OF THE PROJECT GUTENBERG EBOOK 1007 ***",
                 "*** END OF THE PROJECT GUTENBERG EBOOK 1007 ***")
        out = strip_boilerplate(t)
        assert "Real book text." in out
        assert "Title:" not in out
        assert "should be named" not in out

    def test_this_spelling(self):
        t = wrap("*** START OF THIS PROJECT GUTENBERG EBOOK FOO ***",
                 "*** END OF THIS PROJECT GUTENBERG EBOOK FOO ***")
        out = strip_boilerplate(t)
        assert "Real book text." in out
        assert "Title:" not in out

    def test_no_space_after_asterisks(self):
        # 12 START and 13 END lines in the sample use this form.
        t = wrap("***START OF THE PROJECT GUTENBERG EBOOK 1 ***",
                 "***END OF THE PROJECT GUTENBERG EBOOK 1 ***")
        out = strip_boilerplate(t)
        assert "Real book text." in out
        assert "Title:" not in out

    def test_case_insensitive(self):
        t = wrap("*** start of the project gutenberg ebook x ***",
                 "*** end of the project gutenberg ebook x ***")
        assert "Real book text." in strip_boilerplate(t)


class TestNoMarkers:
    """The 4 current files with no START marker must not be destroyed."""

    def test_text_without_markers_is_kept(self):
        # Returning "" here would silently drop whole books. Keeping the
        # text unstripped is the safe failure: boilerplate survives, the
        # book does not vanish.
        t = "Project Gutenberg's The Magnetic North, by Elizabeth Robins\n\n" + BODY
        out = strip_boilerplate(t)
        assert "Real book text." in out

    def test_start_without_end(self):
        t = ("Title: X\n\n*** START OF THE PROJECT GUTENBERG EBOOK 1 ***\n\n"
             + BODY)
        out = strip_boilerplate(t)
        assert "Real book text." in out
        assert "Title:" not in out

    def test_end_without_start(self):
        t = (BODY + "\n*** END OF THE PROJECT GUTENBERG EBOOK 1 ***\n"
             "Updated editions will replace the previous one.\n")
        out = strip_boilerplate(t)
        assert "Real book text." in out
        assert "Updated editions" not in out

    def test_end_marker_before_start_marker(self):
        """A stray END above the real START must not invert the slice.

        Found by mutation testing: searching for END from position 0
        instead of from after START left every test passing, because
        nothing here had the markers out of order. Concatenated or
        truncated files do.
        """
        t = ("*** END OF THE PROJECT GUTENBERG EBOOK 1 ***\n"
             "leftover from a previous file\n"
             "*** START OF THE PROJECT GUTENBERG EBOOK 1 ***\n"
             "REAL BODY\n")
        out = strip_boilerplate(t)
        assert "REAL BODY" in out
        assert "leftover" not in out

    def test_markers_in_reverse_order_keeps_text(self):
        """END first and no second START: the slice would be negative, so
        the guard returns the text untouched rather than an empty book."""
        t = ("*** END OF THE PROJECT GUTENBERG EBOOK 1 ***\n"
             "ALL THE TEXT THERE IS\n")
        assert "ALL THE TEXT THERE IS" in strip_boilerplate(t)


class TestContentPreserved:
    """Stripping must not eat the book."""

    def test_body_is_byte_identical(self):
        t = wrap("*** START OF THE PROJECT GUTENBERG EBOOK 1 ***",
                 "*** END OF THE PROJECT GUTENBERG EBOOK 1 ***")
        assert strip_boilerplate(t).strip() == BODY.strip()

    def test_marker_like_text_inside_body_is_safe(self):
        # A book that discusses Project Gutenberg, or contains a row of
        # asterisks as a scene break, must not be truncated there.
        body = ("Chapter one.\n\n* * *\n\nThe author mentions "
                "the Project Gutenberg EBook in passing.\n\nChapter two.\n")
        t = (f"Title: X\n\n*** START OF THE PROJECT GUTENBERG EBOOK 1 ***\n\n"
             f"{body}\n*** END OF THE PROJECT GUTENBERG EBOOK 1 ***\n"
             "Updated editions will replace\n")
        out = strip_boilerplate(t)
        assert "Chapter one." in out
        assert "Chapter two." in out
        assert "* * *" in out

    def test_empty_input(self):
        assert strip_boilerplate("") == ""


class TestProducedByLine:
    """'Produced by ...' sits after the START marker and is not the book."""

    def test_produced_by_removed(self):
        t = ("*** START OF THE PROJECT GUTENBERG EBOOK 1 ***\n\n"
             "Produced by Suzanne Shell, Anita Paque and the Online\n"
             "Distributed Proofreading Team.\n\n" + BODY +
             "\n*** END OF THE PROJECT GUTENBERG EBOOK 1 ***\n")
        out = strip_boilerplate(t)
        assert "Real book text." in out
        assert "Proofreading" not in out

    def test_body_kept_when_no_produced_by(self):
        t = ("*** START OF THE PROJECT GUTENBERG EBOOK 1 ***\n\n" + BODY +
             "\n*** END OF THE PROJECT GUTENBERG EBOOK 1 ***\n")
        assert "Real book text." in strip_boilerplate(t)

    def test_credit_only_file_is_not_emptied(self):
        """If the credit is all there is, keep it.

        Found by mutation testing: removing the "only strip if a body
        remains" guard broke nothing, because no test had a file whose
        entire content was the credit line. Such files exist in the
        mirror as stubs, and emitting an empty book loses the fact that
        the book was there at all.
        """
        t = ("*** START OF THE PROJECT GUTENBERG EBOOK 1 ***\n\n"
             "Produced by Some Volunteer\n\n"
             "*** END OF THE PROJECT GUTENBERG EBOOK 1 ***\n")
        assert strip_boilerplate(t).strip() != ""


# --- deduplication ------------------------------------------------------

class TestBookId:
    """Book identity comes from the numeric directory, not the filename:
    old/old/3ddcc10.txt belongs to book 1007 and its name says otherwise."""

    @pytest.mark.parametrize("path,expected", [
        ("/g/1/0/0/1007/1007-0.txt", "1007"),
        ("/g/1/0/0/1007/old/1007-0.txt", "1007"),
        ("/g/1/0/0/1007/old/old/3ddcc10.txt", "1007"),
        ("/g/1/0/0/3/10038/10038-8.txt", "10038"),
        ("/g/1/0/0/3/10038/old/10038.txt", "10038"),
    ])
    def test_id_from_directory(self, path, expected):
        assert book_id(Path(path)) == expected


class TestPickBest:
    """One file per book, preferring UTF-8 and the current revision."""

    def test_prefers_current_over_old(self):
        best = pick_best([Path("/g/1007/1007-0.txt"),
                          Path("/g/1007/old/1007-0.txt")])
        assert best == Path("/g/1007/1007-0.txt")

    def test_prefers_utf8_suffix_over_8bit(self):
        # -0 is UTF-8, -8 is 8-bit, bare is ASCII. Choosing -0 avoids
        # feeding the encoding repair work it does not need to do.
        best = pick_best([Path("/g/10038/10038-8.txt"),
                          Path("/g/10038/10038-0.txt"),
                          Path("/g/10038/10038.txt")])
        assert best == Path("/g/10038/10038-0.txt")

    def test_current_8bit_beats_old_utf8(self):
        # Revision dominates encoding: an old UTF-8 file is still the wrong
        # edition, and stage 1 repairs encoding anyway.
        best = pick_best([Path("/g/1/1-8.txt"), Path("/g/1/old/1-0.txt")])
        assert best == Path("/g/1/1-8.txt")

    def test_single_file_is_returned(self):
        assert pick_best([Path("/g/1/1.txt")]) == Path("/g/1/1.txt")

    def test_empty_returns_none(self):
        assert pick_best([]) is None

    def test_deterministic_on_ties(self):
        # Two files of equal rank must resolve the same way every run, or
        # the corpus changes shape between pipeline invocations.
        a = [Path("/g/1/a.txt"), Path("/g/1/b.txt")]
        assert pick_best(a) == pick_best(list(reversed(a)))


# --- properties ---------------------------------------------------------

@settings(max_examples=300)
@given(st.text(max_size=1500))
def test_strip_never_grows_text(text):
    """Stripping only removes. Output is always a substring of input, so
    the stage cannot invent content that was never in the book."""
    assert strip_boilerplate(text) in text


@settings(max_examples=300)
@given(st.text(max_size=1000))
def test_strip_is_idempotent(text):
    once = strip_boilerplate(text)
    assert strip_boilerplate(once) == once


@settings(max_examples=200)
@given(st.text(min_size=1, max_size=400).filter(lambda s: s.strip()))
def test_marked_body_survives(body):
    """Whatever sits between the markers comes back, for any body text.

    This is the property that stops a clever regex from eating the book:
    the specific-case tests only cover bodies someone thought to write.
    """
    if "***" in body or "Produced by" in body:
        return                       # legitimately ambiguous, not a bug
    t = (f"Title: X\n\n*** START OF THE PROJECT GUTENBERG EBOOK 1 ***\n\n"
         f"{body}\n\n*** END OF THE PROJECT GUTENBERG EBOOK 1 ***\n"
         f"Updated editions will replace the previous one.\n")
    assert body.strip() in strip_boilerplate(t)


@settings(max_examples=200)
@given(st.lists(st.sampled_from([
    "1007/1007-0.txt", "1007/1007-8.txt", "1007/1007.txt",
    "1007/old/1007-0.txt", "1007/old/old/3ddcc10.txt",
]), min_size=1, max_size=5, unique=True))
def test_pick_best_is_order_independent(names):
    """The chosen file must not depend on directory-walk order, which
    varies between filesystems and between runs."""
    paths = [Path("/g") / n for n in names]
    assert pick_best(paths) == pick_best(list(reversed(paths)))


@settings(max_examples=200)
@given(st.lists(st.sampled_from([
    "1007/1007-0.txt", "1007/old/1007-0.txt", "1007/old/old/1007.txt",
]), min_size=1, max_size=3, unique=True))
def test_pick_best_returns_a_member(names):
    paths = [Path("/g") / n for n in names]
    assert pick_best(paths) in paths
