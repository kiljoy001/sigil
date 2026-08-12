# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""
Tests for the two command-line entry points.

main() is the last untested part of the pipeline, and testing it is not
ceremony: it owns argument handling, the "not a directory" guard, the
process pool, the exit status, and the summary line that is the only
feedback a 60,830-file run gives. A pipeline that silently exits 0 after
processing nothing looks exactly like a pipeline that worked.

These call main() in-process with argv patched, rather than spawning a
subprocess, so the coverage build sees the lines.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools import boilerplate, clean, manifest, metadata


def run(module, argv, monkeypatch):
    """Invoke a tool's main() with argv, returning its exit status."""
    monkeypatch.setattr(sys, "argv", ["tool", *map(str, argv)])
    return module.main()


BOOK = ("Title: X\n\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK 1007 ***\n\n"
        "It was a bright cold day in April, and the clocks were\n"
        "striking thirteen. Don\x92t stop here.\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK 1007 ***\n"
        "Updated editions will replace the previous one.\n")


class TestCleanCli:
    def test_processes_a_tree(self, tmp_path, monkeypatch, capsys):
        src, dst = tmp_path / "s", tmp_path / "d"
        (src / "1007").mkdir(parents=True)
        (src / "1007" / "1007-0.txt").write_bytes(BOOK.encode("latin-1"))

        rc = run(clean, [src, dst, "-j", "1"], monkeypatch)

        assert rc == 0
        out = (dst / "1007" / "1007-0.txt").read_text(encoding="utf-8")
        assert "Don’t stop" in out
        err = capsys.readouterr().err
        assert "1 files to process" in err
        assert "processed 1 files" in err

    def test_reports_encoding_breakdown(self, tmp_path, monkeypatch, capsys):
        src, dst = tmp_path / "s", tmp_path / "d"
        src.mkdir()
        (src / "a.txt").write_bytes(b"plain ascii")
        (src / "b.txt").write_bytes(b"don\x92t")

        run(clean, [src, dst, "-j", "1"], monkeypatch)

        err = capsys.readouterr().err
        assert "ascii" in err
        assert "cp1252" in err

    def test_dry_run_writes_nothing(self, tmp_path, monkeypatch):
        src, dst = tmp_path / "s", tmp_path / "d"
        src.mkdir()
        (src / "a.txt").write_bytes(b"don\x92t")

        rc = run(clean, [src, dst, "-j", "1", "--dry-run"], monkeypatch)

        assert rc == 0
        assert not dst.exists()

    def test_missing_source_exits_nonzero(self, tmp_path, monkeypatch):
        with pytest.raises(SystemExit) as e:
            run(clean, [tmp_path / "nope", tmp_path / "d"], monkeypatch)
        assert e.value.code != 0

    def test_custom_extension(self, tmp_path, monkeypatch):
        src, dst = tmp_path / "s", tmp_path / "d"
        src.mkdir()
        (src / "a.text").write_bytes(b"hello")
        (src / "b.txt").write_bytes(b"ignored")

        run(clean, [src, dst, "-j", "1", "--ext", ".text"], monkeypatch)

        assert (dst / "a.text").exists()
        assert not (dst / "b.txt").exists()

    def test_empty_tree_succeeds(self, tmp_path, monkeypatch):
        src, dst = tmp_path / "s", tmp_path / "d"
        src.mkdir()
        assert run(clean, [src, dst, "-j", "1"], monkeypatch) == 0

    def test_rerun_is_a_no_op(self, tmp_path, monkeypatch, capsys):
        """Resume: a second run over an unchanged tree must do no work."""
        src, dst = tmp_path / "s", tmp_path / "d"
        src.mkdir()
        (src / "a.txt").write_bytes(b"hello")

        run(clean, [src, dst, "-j", "1"], monkeypatch)
        capsys.readouterr()
        run(clean, [src, dst, "-j", "1"], monkeypatch)

        assert "0 files to process" in capsys.readouterr().err


class TestBoilerplateCli:
    def test_deduplicates_and_strips(self, tmp_path, monkeypatch, capsys):
        src, dst = tmp_path / "s", tmp_path / "d"
        (src / "1007" / "old").mkdir(parents=True)
        (src / "1007" / "1007-0.txt").write_text(BOOK.replace("\x92", "'"),
                                                 encoding="utf-8")
        (src / "1007" / "old" / "1007-0.txt").write_text(
            "superseded", encoding="utf-8")

        rc = run(boilerplate, [src, dst, "-j", "1"], monkeypatch)

        assert rc == 0
        out = (dst / "1007.txt").read_text(encoding="utf-8")
        assert "striking thirteen" in out
        assert "PROJECT GUTENBERG" not in out
        assert "superseded" not in out

        err = capsys.readouterr().err
        assert "2 files, 1 books" in err
        assert "1 duplicates dropped" in err

    def test_keep_duplicates_processes_all(self, tmp_path, monkeypatch,
                                           capsys):
        src, dst = tmp_path / "s", tmp_path / "d"
        (src / "1007" / "old").mkdir(parents=True)
        (src / "1007" / "1007-0.txt").write_text("a", encoding="utf-8")
        (src / "1007" / "old" / "1007-0.txt").write_text("b",
                                                         encoding="utf-8")

        run(boilerplate, [src, dst, "-j", "1", "--keep-duplicates"],
            monkeypatch)

        # Both map to 1007.txt, so the count is what proves both ran.
        assert "0 duplicates dropped" in capsys.readouterr().err

    def test_reports_bytes_stripped(self, tmp_path, monkeypatch, capsys):
        src, dst = tmp_path / "s", tmp_path / "d"
        (src / "1007").mkdir(parents=True)
        (src / "1007" / "1007-0.txt").write_text(
            BOOK.replace("\x92", "'"), encoding="utf-8")

        run(boilerplate, [src, dst, "-j", "1"], monkeypatch)

        err = capsys.readouterr().err
        assert "bytes" in err
        assert "of boilerplate" in err

    def test_missing_source_exits_nonzero(self, tmp_path, monkeypatch):
        with pytest.raises(SystemExit) as e:
            run(boilerplate, [tmp_path / "nope", tmp_path / "d"], monkeypatch)
        assert e.value.code != 0

    def test_unreadable_file_is_reported_and_exits_nonzero(
            self, tmp_path, monkeypatch, capsys):
        """One bad file must not abort the run, but it must not be
        silently successful either -- a 60,830-file pass that reports 0
        while skipping books is the worst outcome."""
        src, dst = tmp_path / "s", tmp_path / "d"
        (src / "1007").mkdir(parents=True)
        (src / "1007" / "1007-0.txt").write_bytes(b"caf\xe9 not utf-8")

        rc = run(boilerplate, [src, dst, "-j", "1"], monkeypatch)

        assert rc == 1
        assert "errors" in capsys.readouterr().err

    def test_empty_tree_succeeds(self, tmp_path, monkeypatch):
        src, dst = tmp_path / "s", tmp_path / "d"
        src.mkdir()
        assert run(boilerplate, [src, dst, "-j", "1"], monkeypatch) == 0


class TestMetadataCli:
    """metadata.py doubles as a coverage check on the catalogue: point it
    at a CSV and it reports how many text works there are and what share
    carry a usable date. That report is how a catalogue swap gets noticed.
    """

    def _cat(self, tmp_path, rows):
        p = tmp_path / "cat.csv"
        p.write_text(
            "Text#,Type,Issued,Title,Language,Authors,Subjects,LoCC,"
            "Bookshelves\n" + "".join(rows), encoding="utf-8")
        return p

    def test_reports_totals(self, tmp_path, monkeypatch, capsys):
        c = self._cat(tmp_path, [
            "1,Text,1971,A Book,en,\"Dante, 1265-1321\",Subj,PS,Shelf\n",
            "2,Text,1971,No Dates,en,Anonymous,Subj,PS,Shelf\n",
            "3,Sound,1971,A Recording,en,Someone,Subj,PS,Shelf\n",
        ])
        rc = run(metadata, [c], monkeypatch)

        err = capsys.readouterr().err
        assert rc == 0
        assert "2 text works" in err, "the Sound row must not be counted"
        assert "50.0%" in err, "one of two has a lifespan"

    def test_prints_a_named_book(self, tmp_path, monkeypatch, capsys):
        c = self._cat(tmp_path, [
            "1007,Text,2004-08-02,Divine Comedy,en,"
            "\"Dante Alighieri, 1265-1321\",Italian poetry,PQ,Shelf\n"])
        rc = run(metadata, [c, "1007"], monkeypatch)

        out = capsys.readouterr().out
        assert rc == 0
        assert "Divine Comedy" in out
        assert "1321" in out
        assert "PQ" in out

    def test_unknown_book_is_reported_not_a_crash(self, tmp_path,
                                                  monkeypatch, capsys):
        c = self._cat(tmp_path, ["1,Text,1971,A,en,B,C,PS,D\n"])
        rc = run(metadata, [c, "99999"], monkeypatch)

        assert rc == 0
        assert "not in catalogue" in capsys.readouterr().out

    def test_no_arguments_exits_nonzero(self, monkeypatch):
        with pytest.raises(SystemExit) as e:
            run(metadata, [], monkeypatch)
        assert e.value.code != 0


class TestManifestCli:
    """manifest.py doubles as the check on a completed run: how many books
    were written, how many joined the catalogue, and how many share a
    digest. That last number is the duplicate filename-based dedup cannot
    see -- two book ids whose cleaned text is byte-identical."""

    def _write(self, tmp_path, rows):
        p = tmp_path / "m.tab"
        with manifest.Manifest(p) as m:
            for r in rows:
                m.add(r)
        return p

    def test_summary_counts(self, tmp_path, monkeypatch, capsys):
        meta = {"title": "A Book", "authors": "X", "language": "en",
                "subjects": "S", "locc": "PS", "bookshelves": "",
                "issued": "1971", "death_year": 1900}
        p = self._write(tmp_path, [
            manifest.book_row("1", digest="aaa", path="p1", encoding="ascii",
                              nbytes=1, nparas=1, meta=meta),
            manifest.book_row("2", digest="aaa", path="p2", encoding="ascii",
                              nbytes=1, nparas=1, meta=meta),
            manifest.book_row("3", digest="bbb", path="p3", encoding="ascii",
                              nbytes=1, nparas=1, meta=None),
        ])

        rc = run(manifest, [p], monkeypatch)
        err = capsys.readouterr().err

        assert rc == 0
        assert "3 books" in err
        assert "2" in err            # two carry catalogue metadata
        assert "duplicate digests:       1" in err

    def test_prints_a_named_book(self, tmp_path, monkeypatch, capsys):
        meta = {"title": "Divine Comedy", "authors": "Dante", "language": "en",
                "subjects": "Poetry", "locc": "PQ", "bookshelves": "",
                "issued": "2004", "death_year": 1321}
        p = self._write(tmp_path, [
            manifest.book_row("1007", digest="d", path="p", encoding="utf-8",
                              nbytes=9, nparas=2, meta=meta)])

        rc = run(manifest, [p, "1007"], monkeypatch)
        out = capsys.readouterr().out

        assert rc == 0
        assert "Divine Comedy" in out
        assert "1321" in out

    def test_unknown_book_is_reported(self, tmp_path, monkeypatch, capsys):
        p = self._write(tmp_path, [
            manifest.book_row("1", digest="d", path="p", encoding="ascii",
                              nbytes=1, nparas=1, meta=None)])

        rc = run(manifest, [p, "404"], monkeypatch)
        assert rc == 0
        assert "not in manifest" in capsys.readouterr().out

    def test_no_arguments_exits_nonzero(self, monkeypatch):
        with pytest.raises(SystemExit) as e:
            run(manifest, [], monkeypatch)
        assert e.value.code != 0
