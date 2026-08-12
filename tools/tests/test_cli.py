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

from tools import boilerplate, clean


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
