# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""
Tests for the file-handling half of the pipeline.

The pure text functions were tested first because that is where the crash
lived, which left clean_file() and the stage-2 job function -- the code that
actually decides what lands on disk -- at 42% coverage and untested. That is
the half that can lose data: a wrong write condition silently skips files, a
swallowed OSError reports success for a book nobody wrote.

Everything here runs against tmp_path. No corpus, no network.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import boilerplate
import clean


class TestCleanFile:
    def test_writes_repaired_output(self, tmp_path):
        src = tmp_path / "in.txt"
        dst = tmp_path / "out" / "in.txt"
        src.write_bytes(b"don\x92t stop")

        enc, changed, err = clean.clean_file(src, dst)

        assert err is None
        assert changed is True
        assert enc == "cp1252"
        assert dst.read_text(encoding="utf-8") == "don’t stop"

    def test_creates_missing_parent_directories(self, tmp_path):
        src = tmp_path / "in.txt"
        dst = tmp_path / "a" / "b" / "c" / "in.txt"
        src.write_bytes(b"hello")

        clean.clean_file(src, dst)
        assert dst.read_text() == "hello"

    def test_clean_file_still_written_when_unchanged(self, tmp_path):
        """A file that needed no repair must still appear in the output
        tree. Writing only changed files would silently drop most of the
        corpus, since the majority is already valid."""
        src = tmp_path / "in.txt"
        dst = tmp_path / "out" / "in.txt"
        src.write_bytes(b"already clean ascii")

        enc, changed, err = clean.clean_file(src, dst)

        assert err is None
        assert changed is False
        assert dst.exists(), "unchanged files must still be copied through"
        assert dst.read_bytes() == b"already clean ascii"

    def test_dry_run_writes_nothing(self, tmp_path):
        src = tmp_path / "in.txt"
        dst = tmp_path / "out" / "in.txt"
        src.write_bytes(b"don\x92t")

        enc, changed, err = clean.clean_file(src, dst, dry_run=True)

        assert changed is True          # still reports what it would do
        assert not dst.exists()

    def test_unreadable_source_is_reported_not_raised(self, tmp_path):
        missing = tmp_path / "nope.txt"
        enc, changed, err = clean.clean_file(missing, tmp_path / "o.txt")

        assert err is not None
        assert "nope.txt" in err
        assert changed is False

    def test_unwritable_destination_is_reported(self, tmp_path):
        src = tmp_path / "in.txt"
        src.write_bytes(b"text")
        blocker = tmp_path / "blocked"
        blocker.write_text("I am a file, not a directory")

        # Writing under a path whose parent is a regular file must be
        # reported, not raised: one bad path should not abort a 60,830-file
        # run.
        enc, changed, err = clean.clean_file(src, blocker / "in.txt")
        assert err is not None

    def test_empty_file(self, tmp_path):
        src = tmp_path / "e.txt"
        dst = tmp_path / "out" / "e.txt"
        src.write_bytes(b"")

        enc, changed, err = clean.clean_file(src, dst)
        assert err is None
        assert dst.exists()
        assert dst.read_bytes() == b""


class TestBoilerplateJob:
    def test_strips_and_writes(self, tmp_path):
        src = tmp_path / "1007.txt"
        dst = tmp_path / "out" / "1007.txt"
        src.write_text(
            "Title: X\n\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK 1007 ***\n\n"
            "The book itself.\n\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK 1007 ***\n"
            "Updated editions will replace the previous one.\n",
            encoding="utf-8")

        wrote, removed, err = boilerplate._job((str(src), str(dst)))

        assert err is None
        assert wrote == 1
        assert removed > 0
        out = dst.read_text(encoding="utf-8")
        assert "The book itself." in out
        assert "Title:" not in out
        assert "Updated editions" not in out

    def test_missing_source_is_reported(self, tmp_path):
        wrote, removed, err = boilerplate._job(
            (str(tmp_path / "gone.txt"), str(tmp_path / "o.txt")))
        assert wrote == 0
        assert err is not None

    def test_invalid_utf8_source_is_reported(self, tmp_path):
        """Stage 2 reads UTF-8 because stage 1 guarantees it. If someone
        points it at a raw tree instead, that must be an error message
        rather than a traceback halfway through a long run."""
        src = tmp_path / "raw.txt"
        src.write_bytes(b"caf\xe9 unrepaired")

        wrote, removed, err = boilerplate._job(
            (str(src), str(tmp_path / "o.txt")))
        assert wrote == 0
        assert err is not None


class TestEndToEnd:
    def test_two_stages_over_a_small_tree(self, tmp_path):
        """The pipeline as actually invoked: stage 1 then stage 2, with
        duplicate revisions present, checking that one clean book comes
        out the far end."""
        raw = tmp_path / "raw"
        (raw / "1007").mkdir(parents=True)
        (raw / "1007" / "old").mkdir()

        body = ("*** START OF THE PROJECT GUTENBERG EBOOK 1007 ***\n\n"
                "It was a bright cold day in April, and the clocks\n"
                "were striking thirteen. Don\x92t stop here.\n\n"
                "*** END OF THE PROJECT GUTENBERG EBOOK 1007 ***\n"
                "Updated editions will replace the previous one.\n")
        (raw / "1007" / "1007-0.txt").write_bytes(body.encode("latin-1"))
        (raw / "1007" / "old" / "1007-0.txt").write_bytes(
            b"superseded revision, must not win")

        s1 = tmp_path / "s1"
        for f in raw.rglob("*.txt"):
            clean.clean_file(f, s1 / f.relative_to(raw))

        files = list(s1.rglob("*.txt"))
        assert len(files) == 2
        best = boilerplate.pick_best(files)
        assert "old" not in best.parts

        s2 = tmp_path / "s2"
        boilerplate._job((str(best), str(s2 / "1007.txt")))

        out = (s2 / "1007.txt").read_text(encoding="utf-8")
        assert "striking thirteen" in out
        assert "Don’t stop" in out          # cp1252 repaired
        assert "PROJECT GUTENBERG" not in out
        assert "superseded" not in out
