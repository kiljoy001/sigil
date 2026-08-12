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

ndb_quote is tested hardest because its failure mode is the worst available:
libtab's Tabula.set() does not quote, so a title containing a space writes a
file that parses as garbage and only fails when something later opens it --

    TabulaError: tab_open: row 0 has undeclared column from

Every title, author and subject line in this corpus contains spaces.
"""

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.metadata import (death_year, load_catalog, ndb_quote,
                            ndb_sanitise, primary_locc)

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
        # Every value the manifest writes must be quotable and round-trip.
        for tid in list(cat)[:2000]:
            for field in ("title", "authors", "subjects", "locc"):
                ndb_quote(cat[tid][field])


# --- ndb quoting --------------------------------------------------------

class TestNdbQuote:
    """libtab does not quote for you, and the failure is invisible until
    something reads the file back."""

    def test_plain_value_is_untouched(self):
        assert ndb_quote("PS") == "PS"
        assert ndb_quote("1321") == "1321"

    def test_space_forces_quotes(self):
        assert ndb_quote("The Divine Comedy") == '"The Divine Comedy"'

    def test_tab_becomes_a_space(self):
        # A tab is a control character: ndb cannot carry it either.
        assert ndb_quote("a\tb") == '"a b"'

    def test_ascii_quotes_become_typographic(self):
        """ndb has no escape for the ASCII quote -- verified against
        libtab directly: doubling, backslashes and a bare quote all fail
        to reopen. 1.37% of catalogue entries contain one, so it is
        substituted rather than dropped.

        Opening or closing is chosen by position, the usual typographic
        rule, so both of these read correctly.
        """
        assert ndb_quote('"Undo": A Novel') == '"\u201cUndo\u201d: A Novel"'
        assert ndb_quote('The Number "e"') == '"The Number \u201ce\u201d"'

    def test_no_ascii_quote_survives(self):
        """The invariant that keeps the file parseable."""
        for v in ('a"b', '"', '""', 'a "b" c', '"""'):
            assert '"' not in ndb_quote(v)[1:-1]

    def test_newline_becomes_a_space(self):
        # ndb is line-structured; book 464's title contains a newline.
        assert ndb_quote("a\nb") == '"a b"'
        assert ndb_quote("a\r\nb") == '"a  b"'

    def test_empty_becomes_empty_quotes(self):
        # An empty value must still occupy the field, or the column
        # vanishes and the row shape changes.
        assert ndb_quote("") == '""'

    def test_leading_hash_is_quoted(self):
        # A bare # starts an ndb comment; the rest of the line is lost.
        assert ndb_quote("#1 Bestseller") == '"#1 Bestseller"'
        assert ndb_quote("#tag") == '"#tag"'

    def test_nil_sentinel_is_substituted(self):
        """`nil` is libtab's on-disk spelling of an absent value, and it
        reads back as empty even when quoted (tab_create.c:341). A book
        titled "nil" would lose its title, so the token is substituted
        rather than stored. Found by a property test."""
        assert ndb_sanitise("nil") == "nil\u2009"
        assert ndb_quote("nil") != "nil"

    def test_nil_like_values_are_not_over_quoted(self):
        # The sentinel is the exact lowercase token and nothing else.
        assert ndb_quote("NIL") == "NIL"
        assert ndb_quote("Nil") == "Nil"
        assert ndb_quote("nilling") == "nilling"

    def test_none_becomes_empty_quotes(self):
        assert ndb_quote(None) == '""'

    def test_number_is_stringified(self):
        assert ndb_quote(1321) == "1321"


@settings(max_examples=500)
@given(st.text(max_size=200))
def test_quoted_value_never_breaks_the_grammar(s):
    """The invariant the manifest depends on: whatever a catalogue field
    contains, the quoted form is a single ndb token.

    Either it is bare and free of separators, or it is wrapped in quotes
    with every interior quote doubled.
    """
    q = ndb_quote(s)
    if q.startswith('"'):
        assert q.endswith('"') and len(q) >= 2
        # Interior quotes are doubled, so the body has an even number of
        # them -- an odd count would terminate the value early.
        assert q[1:-1].count('"') % 2 == 0
    else:
        assert not any(c in q for c in ' \t\n"')
        assert not q.startswith("#")
        assert q != ""


def _unquote(tok):
    """The reader's half of the grammar, for round-trip checking."""
    if tok.startswith('"') and tok.endswith('"') and len(tok) >= 2:
        return tok[1:-1]
    return tok


# What a value becomes is metadata's decision, not the test's. Importing
# ndb_sanitise rather than restating the rules means the two cannot drift
# -- an earlier version restated them and went stale twice in one sitting.
_expected = ndb_sanitise


@settings(max_examples=500)
@given(st.text(max_size=200))
def test_quote_then_parse_round_trips(s):
    """A value survives being written and read, modulo two documented
    substitutions: ASCII quotes become typographic and newlines become
    spaces, both because ndb cannot represent them at all.

    An earlier version asserted that quoting twice always yields a quoted
    string, which hypothesis falsified with "0" -- a value needing no
    quoting stays bare however often it is quoted. That was a wrong claim
    about the function rather than a bug in it.
    """
    assert _unquote(ndb_quote(s)) == _expected(s)


@settings(max_examples=500)
@given(st.text(max_size=200))
def test_quoted_form_is_always_parseable(s):
    """The hard invariant: whatever goes in, the token contains no ASCII
    quote inside the delimiters and no newline anywhere. Those are the
    two characters the grammar cannot carry, and either one produces a
    file that writes cleanly and cannot be reopened."""
    q = ndb_quote(s)
    body = q[1:-1] if q.startswith('"') else q
    assert '"' not in body
    assert "\n" not in q and "\r" not in q
