# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""
Tests for catalogue parsing and the ndb quoting the manifest depends on.

Every awkward input here was taken from /mnt/bulk/pg_catalog.csv rather than
imagined: approximate lifespans ("1814?-1884"), multiple authors separated by
semicolons, "active 6th century B.C." in place of dates, empty author fields,
CJK titles, and multi-class LoCC values like "E201; JK". The catalogue has
90,477 rows and 7 media types; guessing at its shape would have produced a
parser that works on the first hundred.

This file once also tested ndb_quote and ndb_sanitise, which substituted
the characters libtab could not store. libtab encodes values itself now,
those functions are gone, and tools/tests/test_manifest.py checks that the
characters they used to mangle survive a round trip instead.
"""

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.metadata import death_year, load_catalog, primary_locc

CATALOG_HEADER = "Text#,Type,Issued,Title,Language,Authors,Subjects,LoCC,Bookshelves\n"


def write_catalog(tmp_path, rows):
    p = tmp_path / "cat.csv"
    p.write_text(CATALOG_HEADER + "".join(rows), encoding="utf-8")
    return p


# --- death year ---------------------------------------------------------

class TestDeathYear:
    """The only date with any relation to when the words were written.

    Issued is the digitisation date -- the values run 1971 to 2026 -- so a
    death year bounds composition from above and is the best available
    proxy. It covers 83.2% of the catalogue; the rest must be None rather
    than a guess.
    """

    def test_plain_lifespan(self):
        assert death_year("Dante Alighieri, 1265-1321") == 1321

    def test_approximate_death_year(self):
        # "1842-1914?" -- the question mark is the cataloguer's, and the
        # year is still the best bound available.
        assert death_year("Bierce, Ambrose, 1842-1914?") == 1914

    def test_approximate_birth_year(self):
        assert death_year("Brown, William Wells, 1814?-1884") == 1884

    def test_multiple_authors_takes_the_first_lifespan(self):
        # A translator's dates are not the author's. First wins, and the
        # catalogue lists the author first.
        a = ("Smith, Joseph, Jr., 1805-1844; "
             "Church of Jesus Christ of Latter-day Saints")
        assert death_year(a) == 1844

    def test_living_author_has_no_death_year(self):
        assert death_year("Le Guin, Ursula K.") is None

    def test_active_period_is_not_a_lifespan(self):
        # "active 6th century B.C." carries no year we can use.
        assert death_year("Sunzi, active 6th century B.C.") is None

    def test_empty_author(self):
        assert death_year("") is None
        assert death_year(None) is None

    def test_translator_dates_when_author_has_none(self):
        # Sunzi has no dates; Giles does. Taking the translator's death
        # year would date the translation, not the work -- but it is the
        # only bound present, and it is still an upper bound on the text
        # as published. Documented rather than silently either way.
        a = "Sunzi, active 6th century B.C.; Giles, Lionel, 1875-1958 [Translator]"
        assert death_year(a) == 1958


# --- LoCC ---------------------------------------------------------------

class TestPrimaryLocc:
    """The broad class letter, which is the evaluation signal.

    Digits subdivide far more finely than a paragraph-level comparison can
    support, so PS3521 and PS3537 both reduce to PS.
    """

    def test_single_class(self):
        assert primary_locc("PS") == "PS"

    def test_strips_digits(self):
        assert primary_locc("PS3521") == "PS"
        assert primary_locc("E201") == "E"

    def test_multi_class_takes_the_first(self):
        assert primary_locc("E201; JK") == "E"

    def test_empty(self):
        assert primary_locc("") == ""
        assert primary_locc(None) == ""

    def test_whitespace_only(self):
        assert primary_locc("   ") == ""


# --- catalogue loading --------------------------------------------------

class TestLoadCatalog:
    def test_reads_a_row(self, tmp_path):
        p = write_catalog(tmp_path, [
            '1,Text,1971-12-01,The Declaration,en,"Jefferson, Thomas, '
            '1743-1826",United States -- History,E201; JK,Politics\n'])
        cat = load_catalog(p)

        assert set(cat) == {"1"}
        r = cat["1"]
        assert r["title"] == "The Declaration"
        assert r["authors"] == "Jefferson, Thomas, 1743-1826"
        assert r["language"] == "en"
        assert r["locc"] == "E"
        assert r["issued"] == "1971-12-01"
        assert r["death_year"] == 1826

    def test_skips_non_text_types(self, tmp_path):
        """The catalogue has 1,114 Sound rows and 89 Datasets among 7
        media types. Indexing an audiobook's metadata against a text
        corpus would be a join that silently succeeds and means nothing.
        """
        p = write_catalog(tmp_path, [
            "1,Text,1971,A Book,en,Author,Subj,PS,Shelf\n",
            "2,Sound,1971,A Recording,en,Author,Subj,PS,Shelf\n",
            "3,Dataset,1971,Some Data,en,Author,Subj,PS,Shelf\n",
        ])
        assert set(load_catalog(p)) == {"1"}

    def test_missing_author(self, tmp_path):
        p = write_catalog(tmp_path, [
            "1,Text,1971,Anonymous Work,en,,Subj,PS,Shelf\n"])
        r = load_catalog(p)["1"]
        assert r["authors"] == ""
        assert r["death_year"] is None

    def test_non_latin_title_survives(self, tmp_path):
        p = write_catalog(tmp_path, [
            "1,Text,1971,羅生門,ja,芥川龍之介,Subj,PL,Shelf\n"])
        assert load_catalog(p)["1"]["title"] == "羅生門"

    def test_empty_catalogue(self, tmp_path):
        assert load_catalog(write_catalog(tmp_path, [])) == {}

    def test_real_catalogue_if_present(self):
        """Guard against the real file's shape drifting from the fixtures.

        Skipped where the corpus is not mounted, so CI stays hermetic.
        """
        real = Path("/mnt/bulk/pg_catalog.csv")
        if not real.exists():
            pytest.skip("corpus not mounted")

        cat = load_catalog(real)
        assert len(cat) > 70_000

        # Every row the manifest will write must have the fields it
        # expects. Round-tripping them through libtab is
        # test_manifest.py's job; this only guards the catalogue's shape.
        for tid in list(cat)[:2000]:
            for field in ("title", "authors", "subjects", "locc",
                          "language", "issued"):
                assert isinstance(cat[tid][field], str)


# The ndb quoting tests were removed with the functions they covered.
# libtab encodes values itself now; tools/tests/test_manifest.py checks
# that the characters those helpers used to substitute -- ASCII quotes, a
# leading '#', newlines, tabs, control characters and the literal string
# "nil" -- reach the table and come back unchanged.
