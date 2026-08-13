# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""
Tests for the manifest: one libtab row per book, provenance plus metadata.

The manifest is the pipeline's record of what it did -- which revision of
each book won deduplication, what encoding it was found in, what its cleaned
content hashes to, and what the catalogue says about it. Everything
downstream reads this rather than re-deriving it, so a wrong row is not a
crash, it is a quietly wrong corpus.

Almost every test here opens the file back and reads it, rather than
checking what was passed to the writer. That is deliberate: libtab's failure
mode for an unquoted value is a file that writes cleanly and cannot be
opened, so a test that never reopens proves nothing about the thing most
likely to break.
"""

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.manifest import COLUMNS, Manifest, book_row

libtab = pytest.importorskip("libtab", reason="libtab not installed")


META = {
    "title": "Divine Comedy, Cary's Translation, Paradise",
    "authors": "Dante Alighieri, 1265-1321; Cary, Henry Francis, 1772-1844",
    "language": "en",
    "subjects": "Epic poetry, Italian -- Translations into English",
    "locc": "PQ",
    "bookshelves": "Banned Books; Italian Literature",
    "issued": "2004-08-02",
    "death_year": 1321,
}


class TestBookRow:
    """The row as a dict, before it reaches libtab."""

    def test_carries_provenance_and_metadata(self):
        r = book_row("1007", digest="abc123", path="1007/1007-0.txt",
                     encoding="latin-1", nbytes=202928, nparas=1834,
                     meta=META)

        assert r["book"] == "1007"
        assert r["digest"] == "abc123"
        assert r["path"] == "1007/1007-0.txt"
        assert r["encoding"] == "latin-1"
        assert r["bytes"] == "202928"
        assert r["paras"] == "1834"
        assert r["title"] == META["title"]
        assert r["locc"] == "PQ"
        assert r["death_year"] == "1321"

    def test_missing_catalogue_entry_gives_empty_fields(self):
        """About 14 of 77,815 books have no catalogue row. They must
        appear in the manifest with empty metadata, not vanish -- a book
        that is indexed but absent from the manifest is invisible to
        everything downstream."""
        r = book_row("99999", digest="d", path="p", encoding="ascii",
                     nbytes=1, nparas=1, meta=None)

        assert r["book"] == "99999"
        assert r["digest"] == "d"
        assert r["title"] == ""
        assert r["locc"] == ""
        assert r["death_year"] == ""

    def test_absent_death_year_is_empty_not_none(self):
        """16% of authors have no lifespan. The field must be an empty
        string, because "None" written literally would read back as the
        four-character text and sort among real years."""
        meta = dict(META, death_year=None)
        assert book_row("1", digest="d", path="p", encoding="ascii",
                        nbytes=1, nparas=1, meta=meta)["death_year"] == ""

    def test_every_column_is_present(self):
        """A row missing a column changes the shape of the table."""
        r = book_row("1", digest="d", path="p", encoding="ascii",
                     nbytes=1, nparas=1, meta=META)
        assert set(r) == set(COLUMNS)


class TestManifestRoundTrip:
    """Write, close, reopen, read. The reopen is the point."""

    def test_single_row(self, tmp_path):
        p = tmp_path / "m.tab"
        with Manifest(p) as m:
            m.add(book_row("1007", digest="abc", path="1007/1007-0.txt",
                           encoding="latin-1", nbytes=100, nparas=5,
                           meta=META))

        rows = Manifest.read(p)
        assert len(rows) == 1
        got = rows["1007"]
        assert got["title"] == META["title"]
        assert got["authors"] == META["authors"]
        assert got["subjects"] == META["subjects"]
        assert got["locc"] == "PQ"
        assert got["death_year"] == "1321"

    def test_values_with_spaces_survive(self, tmp_path):
        """The failure this whole quoting exercise exists for. Titles,
        authors and subjects all contain spaces; without quoting the file
        writes fine and fails on open with "undeclared column"."""
        p = tmp_path / "m.tab"
        with Manifest(p) as m:
            m.add(book_row("1", digest="d", path="p", encoding="ascii",
                           nbytes=1, nparas=1, meta=META))

        assert Manifest.read(p)["1"]["title"] == META["title"]

    def test_quotes_inside_a_title_are_kept(self, tmp_path):
        """1,066 catalogue titles contain an ASCII quote.

        These used to be substituted with typographic quotes, because ndb
        had no escape for the ASCII one and a title containing it wrote a
        file that could not be reopened. libtab encodes values now, so the
        title is stored as written -- and applying the old substitution
        today would corrupt it.
        """
        meta = dict(META, title='The "Genius"')
        p = tmp_path / "m.tab"
        with Manifest(p) as m:
            m.add(book_row("1", digest="d", path="p", encoding="ascii",
                           nbytes=1, nparas=1, meta=meta))

        assert Manifest.read(p)["1"]["title"] == 'The "Genius"'

    def test_reserved_characters_are_kept(self, tmp_path):
        """Everything ndb once could not carry, now stored verbatim."""
        for name, value in (("leading hash", "#1 Bestseller"),
                            ("literal nil", "nil"),
                            ("newline", "line one\nline two"),
                            ("tab", "a\tb"),
                            ("control char", "a\x1fb")):
            p = tmp_path / f"{abs(hash(name))}.tab"
            meta = dict(META, title=value)
            with Manifest(p) as m:
                m.add(book_row("1", digest="d", path="p", encoding="ascii",
                               nbytes=1, nparas=1, meta=meta))
            assert Manifest.read(p)["1"]["title"] == value, name

    def test_non_latin_title(self, tmp_path):
        meta = dict(META, title="羅生門", authors="芥川龍之介")
        p = tmp_path / "m.tab"
        with Manifest(p) as m:
            m.add(book_row("1", digest="d", path="p", encoding="utf-8",
                           nbytes=1, nparas=1, meta=meta))

        got = Manifest.read(p)["1"]
        assert got["title"] == "羅生門"
        assert got["authors"] == "芥川龍之介"

    def test_hash_like_value_is_not_a_comment(self, tmp_path):
        """A leading # begins an ndb comment and would swallow the line."""
        meta = dict(META, title="#1 Bestseller")
        p = tmp_path / "m.tab"
        with Manifest(p) as m:
            m.add(book_row("1", digest="d", path="p", encoding="ascii",
                           nbytes=1, nparas=1, meta=meta))

        assert Manifest.read(p)["1"]["title"] == "#1 Bestseller"

    def test_many_rows(self, tmp_path):
        p = tmp_path / "m.tab"
        with Manifest(p) as m:
            for i in range(200):
                m.add(book_row(str(i), digest=f"d{i}", path=f"p{i}",
                               encoding="ascii", nbytes=i, nparas=i,
                               meta=dict(META, title=f"Book {i} of Many")))

        rows = Manifest.read(p)
        assert len(rows) == 200
        assert rows["137"]["title"] == "Book 137 of Many"

    def test_book_with_no_catalogue_entry_round_trips(self, tmp_path):
        p = tmp_path / "m.tab"
        with Manifest(p) as m:
            m.add(book_row("99999", digest="d", path="p", encoding="ascii",
                           nbytes=1, nparas=1, meta=None))

        got = Manifest.read(p)["99999"]
        assert got["book"] == "99999"
        assert got["title"] == ""

    def test_empty_manifest_opens(self, tmp_path):
        """A run that indexed nothing still writes a readable table."""
        p = tmp_path / "m.tab"
        with Manifest(p) as m:
            pass
        assert Manifest.read(p) == {}


class TestSearch:
    """The manifest is a lookup table; these are the lookups worth having."""

    def test_by_locc(self, tmp_path):
        p = tmp_path / "m.tab"
        with Manifest(p) as m:
            for i, locc in enumerate(["PQ", "PS", "PQ", "E"]):
                m.add(book_row(str(i), digest=f"d{i}", path=f"p{i}",
                               encoding="ascii", nbytes=1, nparas=1,
                               meta=dict(META, locc=locc)))

        rows = Manifest.read(p)
        assert sum(1 for r in rows.values() if r["locc"] == "PQ") == 2

    def test_digest_identifies_identical_content(self, tmp_path):
        """Two books whose cleaned text hashes the same are the same text
        under different ids -- which the filename-based dedup cannot see."""
        p = tmp_path / "m.tab"
        with Manifest(p) as m:
            m.add(book_row("1", digest="same", path="a", encoding="ascii",
                           nbytes=1, nparas=1, meta=META))
            m.add(book_row("2", digest="same", path="b", encoding="ascii",
                           nbytes=1, nparas=1, meta=META))

        rows = Manifest.read(p)
        digests = [r["digest"] for r in rows.values()]
        assert digests.count("same") == 2


# --- properties ---------------------------------------------------------

# Catalogue text: anything a cataloguer might type, including the
# characters that break ndb.
catalogue_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",),
                           blacklist_characters="\x00"),
    max_size=120)


@settings(max_examples=200, deadline=None)
@given(title=catalogue_text, authors=catalogue_text,
       subjects=catalogue_text)
def test_any_catalogue_value_round_trips(tmp_path_factory, title, authors,
                                         subjects):
    """Whatever the catalogue contains, the manifest can store and return
    it. This is the property the whole quoting layer exists to provide,
    and it is checked through a real write and a real reopen."""
    p = tmp_path_factory.mktemp("m") / "m.tab"
    meta = dict(META, title=title, authors=authors, subjects=subjects)

    with Manifest(p) as m:
        m.add(book_row("1", digest="d", path="p", encoding="ascii",
                       nbytes=1, nparas=1, meta=meta))

    # Verbatim now: libtab encodes on write and decodes on read. The one
    # exception is NUL, which terminates the string at the C boundary and
    # no encoding inside libtab can carry -- so the generator excludes it
    # rather than the assertion tolerating it.
    got = Manifest.read(p)["1"]
    assert got["title"] == title
    assert got["authors"] == authors
    assert got["subjects"] == subjects
