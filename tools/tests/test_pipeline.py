# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""
Tests for the pipeline driver: the three stages wired together.

The individual stages are tested in their own files. What is tested here is
the wiring, which is where this kind of program actually goes wrong: the
dedup happening before the clean rather than after, a resumed run skipping
work it should do, the manifest disagreeing with the tree beside it.

The end-to-end tests build a small mirror with the awkward shapes the real
one has -- superseded revisions under old/, a CP1252 byte, a licence
envelope, a book absent from the catalogue -- and check the output tree and
the manifest together. Checking either alone would miss the failure where
they disagree.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools import pipeline
from tools.manifest import Manifest

pytest.importorskip("libtab", reason="libtab not installed")

CATALOG = (
    "Text#,Type,Issued,Title,Language,Authors,Subjects,LoCC,Bookshelves\n"
    '1007,Text,2004-08-02,Divine Comedy,en,"Dante Alighieri, 1265-1321",'
    'Italian poetry,PQ,Poetry\n'
    '1009,Text,1997-08-01,Inferno,it,"Dante Alighieri, 1265-1321",'
    'Italian poetry,PQ,Poetry\n'
)

BOOK = ("*** START OF THE PROJECT GUTENBERG EBOOK {id} ***\n\n"
        "It was a bright cold day in April, and the clocks were striking\n"
        "thirteen. Don\x92t stop here, the man said.\n\n"
        "A second paragraph, long enough to count as one for the indexer.\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK {id} ***\n"
        "Updated editions will replace the previous one.\n")


def make_mirror(root, ids=("1007", "1009"), with_old=True):
    """A miniature mirror with the shapes the real one has."""
    # exist_ok: the resume tests build the same mirror twice.
    for i in ids:
        d = root / i
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{i}-0.txt").write_bytes(BOOK.format(id=i).encode("latin-1"))
        if with_old:
            (d / "old" / "old").mkdir(parents=True, exist_ok=True)
            (d / "old" / f"{i}-0.txt").write_bytes(
                b"superseded revision, must not win")
            (d / "old" / "old" / f"{i}.txt").write_bytes(b"older still")
    return root


class TestCountParagraphs:
    """The manifest's paragraph count has to mean what the indexer counts,
    or /stats will disagree with the manifest after a full run."""

    def test_counts_separated_paragraphs(self):
        t = "a" * 50 + "\n\n" + "b" * 50
        assert pipeline.count_paragraphs(t) == 2

    def test_ignores_short_paragraphs(self):
        # Under Minpara (40) the indexer drops it.
        assert pipeline.count_paragraphs("short\n\n" + "a" * 50) == 1

    def test_chunks_over_long_paragraphs(self):
        # index.c splits rather than dropping, so one 9000-char block is
        # three records, not zero.
        assert pipeline.count_paragraphs("a" * 9000) == 3

    def test_boundaries_are_inclusive(self):
        """Exactly MIN_PARA counts; one short does not. Exactly MAX_PARA
        is one paragraph, not a chunked two.

        Found by generated mutation testing: <= could become < at either
        end and nothing failed, because every case sat comfortably inside
        the range. The indexer's rule is inclusive at both ends, and a
        manifest that counts differently disagrees with /stats.
        """
        assert pipeline.count_paragraphs("a" * pipeline.MIN_PARA) == 1
        assert pipeline.count_paragraphs("a" * (pipeline.MIN_PARA - 1)) == 0
        assert pipeline.count_paragraphs("a" * pipeline.MAX_PARA) == 1
        # MAX_PARA + 1 leaves a 1-byte remainder, below MIN_PARA, which
        # the splitter drops -- so one chunk. See test_split.py.
        assert pipeline.count_paragraphs("a" * (pipeline.MAX_PARA + 1)) == 1

    def test_chunk_count_matches_the_indexer(self):
        """Not ceiling division -- the splitter advances a cursor and
        drops a remainder below MIN_PARA. An earlier version of this test
        asserted the ceiling, which is what the removed Python
        reimplementation did and part of why the manifest disagreed with
        the index by 3.2%. tools/tests/test_split.py covers the rule
        itself; this checks the pipeline calls it.
        """
        from tools.split import count as c
        for n in (2 * pipeline.MAX_PARA, 2 * pipeline.MAX_PARA + 1, 9000):
            assert pipeline.count_paragraphs("a" * n) == c("a" * n)

    def test_chunks_accumulate_across_paragraphs(self):
        """n += rather than n =: two long blocks are six records, not
        three."""
        block = "a" * 9000
        assert pipeline.count_paragraphs(block + "\n\n" + block) == 6

    def test_empty(self):
        assert pipeline.count_paragraphs("") == 0


class TestDigest:
    def test_stable_for_the_same_text(self):
        assert pipeline.digest_text("hello") == pipeline.digest_text("hello")

    def test_differs_for_different_text(self):
        assert pipeline.digest_text("a") != pipeline.digest_text("b")

    def test_algorithm_is_named(self):
        """Recorded in the value, so a store built with a fallback digest
        is visibly not comparable rather than silently so."""
        assert pipeline.digest_text("x").split(":")[0] in ("blake3", "sha256")

    def test_blake3_matches_libsigil(self):
        """The manifest digest has to be the same function sigil uses for
        record identity, or an incremental reindex compares two different
        hashes and re-embeds the whole corpus.

        This value came from the vendored C implementation
        (third_party/blake3) over the same input, so it pins the two
        together rather than merely asserting Python is self-consistent.
        Skipped where the pure-Python fallback is in use, which the digest
        prefix makes visible.
        """
        got = pipeline.digest_text("hello")
        if not got.startswith("blake3:"):
            pytest.skip("blake3 not installed; using the sha256 fallback")
        assert got == ("blake3:ea8f163db38682925e4491c5e58d4bb3506ef8c14eb7"
                       "8a86e908c5624a67200f")


class TestPlan:
    def test_deduplicates_before_reading(self, tmp_path):
        nfiles, jobs = pipeline.plan(make_mirror(tmp_path / "m"),
                                     tmp_path / "out")
        assert nfiles == 6          # 2 books x (current + old + old/old)
        assert len(jobs) == 2

    def test_chooses_the_current_revision(self, tmp_path):
        _, jobs = pipeline.plan(make_mirror(tmp_path / "m"), tmp_path / "out")
        for src, _dst, _book in jobs:
            assert "old" not in Path(src).parts

    def test_output_is_flat_by_book_id(self, tmp_path):
        _, jobs = pipeline.plan(make_mirror(tmp_path / "m"), tmp_path / "out")
        names = sorted(Path(d).name for _s, d, _b in jobs)
        assert names == ["1007.txt", "1009.txt"]

    def test_keep_duplicates(self, tmp_path):
        _, jobs = pipeline.plan(make_mirror(tmp_path / "m"), tmp_path / "out",
                                keep_duplicates=True)
        assert len(jobs) == 6


class TestProcessOne:
    def test_cleans_strips_and_measures(self, tmp_path):
        src = tmp_path / "1007-0.txt"
        src.write_bytes(BOOK.format(id="1007").encode("latin-1"))
        dst = tmp_path / "out" / "1007.txt"

        r = pipeline.process_one((str(src), str(dst), "1007"))

        assert r["error"] is None
        assert r["encoding"] == "cp1252"
        assert r["paras"] == 2
        assert r["digest"].startswith(("blake3:", "sha256:"))

        out = dst.read_text(encoding="utf-8")
        assert "Don’t stop" in out, "cp1252 repaired"
        assert "PROJECT GUTENBERG" not in out, "envelope stripped"
        assert r["bytes"] == len(out.encode("utf-8"))

    def test_unreadable_source_is_reported(self, tmp_path):
        r = pipeline.process_one((str(tmp_path / "gone.txt"),
                                  str(tmp_path / "o.txt"), "1"))
        assert r["error"] is not None


class TestEndToEnd:
    def _run(self, tmp_path, monkeypatch, extra=(), rebuild=True):
        """rebuild=False reruns against the mirror as it stands.

        A real second run does not rewrite the corpus first; rewriting it
        makes every source newer than its output and defeats the resume
        check, which is what an earlier version of these tests did.
        """
        src = tmp_path / "m"
        if rebuild:
            make_mirror(src)
        dst = tmp_path / "out"
        cat = tmp_path / "cat.csv"
        cat.write_text(CATALOG, encoding="utf-8")
        monkeypatch.setattr(sys, "argv", [
            "pipeline", str(src), str(dst), "--catalog", str(cat),
            "-j", "1", *extra])
        rc = pipeline.main()
        return rc, dst

    def test_tree_and_manifest_agree(self, tmp_path, monkeypatch, capsys):
        rc, dst = self._run(tmp_path, monkeypatch)
        assert rc == 0

        books = sorted(p.name for p in dst.glob("*.txt"))
        assert books == ["1007.txt", "1009.txt"]

        rows = Manifest.read(dst / "manifest.tab")
        assert set(rows) == {"1007", "1009"}

        # The manifest must describe the files actually written -- the
        # failure where the two drift apart is the one worth catching.
        for book, row in rows.items():
            text = (dst / f"{book}.txt").read_text(encoding="utf-8")
            assert row["bytes"] == str(len(text.encode("utf-8")))
            assert row["digest"] == pipeline.digest_text(text)

    def test_metadata_is_joined(self, tmp_path, monkeypatch):
        _, dst = self._run(tmp_path, monkeypatch)
        r = Manifest.read(dst / "manifest.tab")["1007"]

        assert r["title"] == "Divine Comedy"
        assert r["locc"] == "PQ"
        assert r["death_year"] == "1321"
        assert r["issued"] == "2004-08-02"
        assert r["encoding"] == "cp1252"

    def test_book_absent_from_the_catalogue_still_appears(
            self, tmp_path, monkeypatch):
        """~14 of 77,815 books have no catalogue row. A book indexed but
        missing from the manifest is invisible downstream."""
        src = make_mirror(tmp_path / "m", ids=("1007", "99999"))
        dst = tmp_path / "out"
        cat = tmp_path / "cat.csv"
        cat.write_text(CATALOG, encoding="utf-8")
        monkeypatch.setattr(sys, "argv", [
            "pipeline", str(src), str(dst), "--catalog", str(cat), "-j", "1"])
        pipeline.main()

        rows = Manifest.read(dst / "manifest.tab")
        assert "99999" in rows
        assert rows["99999"]["title"] == ""
        assert rows["99999"]["digest"], "provenance is still recorded"

    def test_runs_without_a_catalogue(self, tmp_path, monkeypatch):
        """Provenance alone is still a usable manifest."""
        src = make_mirror(tmp_path / "m")
        dst = tmp_path / "out"
        monkeypatch.setattr(sys, "argv",
                            ["pipeline", str(src), str(dst), "-j", "1"])
        assert pipeline.main() == 0

        rows = Manifest.read(dst / "manifest.tab")
        assert len(rows) == 2
        assert all(r["digest"] for r in rows.values())
        assert all(r["title"] == "" for r in rows.values())

    def test_rerun_skips_completed_books(self, tmp_path, monkeypatch,
                                         capsys):
        self._run(tmp_path, monkeypatch)
        capsys.readouterr()
        rc, _dst = self._run(tmp_path, monkeypatch, rebuild=False)

        assert rc == 0
        assert "0 to process" in capsys.readouterr().err

    def test_rerun_still_writes_a_complete_manifest(self, tmp_path,
                                                    monkeypatch):
        """The trap in a resumable pipeline: skipping the work must not
        skip the row. A second run that wrote an empty manifest beside a
        full tree would be worse than one that redid everything."""
        self._run(tmp_path, monkeypatch)
        first = Manifest.read((tmp_path / "out") / "manifest.tab")

        self._run(tmp_path, monkeypatch, rebuild=False)
        second = Manifest.read((tmp_path / "out") / "manifest.tab")

        assert set(second) == set(first)

    def test_limit_stops_early(self, tmp_path, monkeypatch):
        _, dst = self._run(tmp_path, monkeypatch, extra=("--limit", "1"))
        assert len(list(dst.glob("*.txt"))) == 1

    def test_missing_source_exits_nonzero(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "argv", [
            "pipeline", str(tmp_path / "nope"), str(tmp_path / "out")])
        with pytest.raises(SystemExit) as e:
            pipeline.main()
        assert e.value.code != 0


class TestReport:
    def test_counts_appear(self):
        r = pipeline.format_report(331, 110, 110, 0, 108, [])
        assert "331 files -> 110 books" in r
        assert "221 duplicates dropped" in r
        assert "108" in r

    def test_errors_are_capped(self):
        r = pipeline.format_report(1, 1, 0, 0, 0,
                                   [f"/f{i}: bad" for i in range(50)])
        assert r.count("/f") == 20
        assert "30 more" in r

    def test_exactly_twenty_errors_has_no_more_line(self):
        r = pipeline.format_report(1, 1, 0, 0, 0,
                                   [f"/f{i}: bad" for i in range(20)])
        assert r.count("/f") == 20
        assert "more" not in r

    def test_twenty_one_errors_mentions_one_more(self):
        r = pipeline.format_report(1, 1, 0, 0, 0,
                                   [f"/f{i}: bad" for i in range(21)])
        assert "1 more" in r

    def test_report_is_line_structured(self):
        """Four summary lines, then one per error."""
        r = pipeline.format_report(6, 2, 2, 0, 2, ["/a: bad"])
        lines = r.splitlines()
        # blank, 4 summary lines, blank, "1 errors:", the error itself
        assert len(lines) == 8
        assert lines[-1].strip() == "/a: bad"

    def test_exit_status(self):
        assert pipeline.exit_status([]) == 0
        assert pipeline.exit_status(["boom"]) == 1
