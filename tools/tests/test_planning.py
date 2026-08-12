# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""
Tests for the job-planning half of both pipeline stages.

This logic used to live inside main(), where it could not be tested, which
is why pipeline coverage sat at 55%. It decides what gets processed and what
gets discarded -- on the sampled subtree, 331 files down to 110 books -- and
every failure mode here is silent: a wrong skip leaves a stale file in the
output tree, a wrong selection indexes a superseded revision, and neither
produces an error message. The only symptom is a paragraph count that does
not match, noticed weeks later if at all.
"""

import sys
import time
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools import boilerplate, clean


# --- stage 1 planning ---------------------------------------------------

class TestCleanPlanJobs:
    def test_finds_matching_files_recursively(self, tmp_path):
        src = tmp_path / "s"
        (src / "a" / "b").mkdir(parents=True)
        (src / "one.txt").write_text("x")
        (src / "a" / "two.txt").write_text("x")
        (src / "a" / "b" / "three.txt").write_text("x")

        jobs = clean.plan_jobs(src, tmp_path / "d")
        assert len(jobs) == 3

    def test_ignores_other_extensions(self, tmp_path):
        src = tmp_path / "s"
        src.mkdir()
        (src / "keep.txt").write_text("x")
        (src / "skip.zip").write_text("x")
        (src / "skip.html").write_text("x")

        jobs = clean.plan_jobs(src, tmp_path / "d")
        assert len(jobs) == 1
        assert jobs[0][0].endswith("keep.txt")

    def test_a_wrong_extension_does_not_stop_the_walk(self, tmp_path):
        """Skipping a non-.txt file must not abandon the directory.

        There are two `continue` statements in plan_jobs -- this one and
        the resume check -- and generated mutation testing caught that
        only the second was covered. Sorted order matters here: the
        skipped file has to come first for continue and break to differ,
        which is why "a.zip" is named as it is.
        """
        src = tmp_path / "s"
        src.mkdir()
        (src / "a.zip").write_text("x")      # sorts first, is skipped
        (src / "b.txt").write_text("x")
        (src / "c.txt").write_text("x")

        jobs = clean.plan_jobs(src, tmp_path / "d")
        names = sorted(Path(s).name for s, _d, _dr in jobs)
        assert names == ["b.txt", "c.txt"]

    def test_destination_mirrors_source_layout(self, tmp_path):
        src = tmp_path / "s"
        (src / "1007").mkdir(parents=True)
        (src / "1007" / "book.txt").write_text("x")

        jobs = clean.plan_jobs(src, tmp_path / "d")
        assert jobs[0][1] == str(tmp_path / "d" / "1007" / "book.txt")

    def test_skips_up_to_date_output(self, tmp_path):
        """Resume: a destination newer than its source is already done."""
        src, dst = tmp_path / "s", tmp_path / "d"
        src.mkdir(); dst.mkdir()
        (src / "a.txt").write_text("x")
        out = dst / "a.txt"
        out.write_text("x")
        # Make the output unambiguously newer.
        t = time.time() + 10
        import os
        os.utime(out, (t, t))

        assert clean.plan_jobs(src, dst) == []

    def test_equal_mtimes_count_as_done(self, tmp_path):
        """A destination with the same mtime as its source is up to date.

        Found by generated mutation testing: >= became > and nothing
        failed, because every test had a strict difference. Equal mtimes
        are the common case for a copied tree, and the mutant would make
        every resumed run redo the entire corpus.
        """
        import os
        src, dst = tmp_path / "s", tmp_path / "d"
        src.mkdir(); dst.mkdir()
        (src / "a.txt").write_text("x")
        (dst / "a.txt").write_text("x")
        t = 1_700_000_000
        os.utime(src / "a.txt", (t, t))
        os.utime(dst / "a.txt", (t, t))

        assert clean.plan_jobs(src, dst) == []

    def test_reprocesses_when_source_is_newer(self, tmp_path):
        """A source edited after the last run must be picked up again --
        otherwise a corpus refresh silently keeps the old output."""
        import os
        src, dst = tmp_path / "s", tmp_path / "d"
        src.mkdir(); dst.mkdir()
        s = src / "a.txt"
        (dst / "a.txt").write_text("stale")
        s.write_text("fresh")
        t = time.time() + 10
        os.utime(s, (t, t))

        assert len(clean.plan_jobs(src, dst)) == 1

    def test_dry_run_ignores_existing_output(self, tmp_path):
        """--dry-run reports what a real run would do, so it must not
        skip files just because a previous run left output behind."""
        src, dst = tmp_path / "s", tmp_path / "d"
        src.mkdir(); dst.mkdir()
        (src / "a.txt").write_text("x")
        (dst / "a.txt").write_text("x")

        assert len(clean.plan_jobs(src, dst, dry_run=True)) == 1

    def test_dry_run_flag_reaches_the_job(self, tmp_path):
        src = tmp_path / "s"
        src.mkdir()
        (src / "a.txt").write_text("x")

        assert clean.plan_jobs(src, tmp_path / "d", dry_run=True)[0][2] is True
        assert clean.plan_jobs(src, tmp_path / "d")[0][2] is False

    def test_empty_tree(self, tmp_path):
        src = tmp_path / "s"
        src.mkdir()
        assert clean.plan_jobs(src, tmp_path / "d") == []

    def test_a_skipped_file_does_not_stop_the_walk(self, tmp_path):
        """Skipping an up-to-date file must not abandon the rest.

        Found by generated mutation testing: turning the `continue` into
        a `break` survived, because no test had an already-done file
        followed by work still to do. On a resumed run that is the normal
        case, and the bug would silently process only part of the corpus.
        """
        import os
        src, dst = tmp_path / "s", tmp_path / "d"
        src.mkdir(); dst.mkdir()
        for n in ("a", "b", "c"):
            (src / f"{n}.txt").write_text("x")
        # Mark only the first as done.
        done = dst / "a.txt"
        done.write_text("x")
        t = time.time() + 10
        os.utime(done, (t, t))

        jobs = clean.plan_jobs(src, dst)
        names = sorted(Path(s).name for s, _d, _dr in jobs)
        assert names == ["b.txt", "c.txt"]

    def test_order_is_reproducible(self, tmp_path):
        src = tmp_path / "s"
        src.mkdir()
        for n in "cadb":
            (src / f"{n}.txt").write_text("x")

        a = clean.plan_jobs(src, tmp_path / "d")
        b = clean.plan_jobs(src, tmp_path / "d")
        assert a == b


class TestSummarise:
    def test_counts_encodings_and_changes(self):
        counts, changed, errors = clean.summarise([
            ("ascii", False, None),
            ("cp1252", True, None),
            ("cp1252", True, None),
            ("utf-8", False, None),
        ])
        assert counts == {"ascii": 1, "cp1252": 2, "utf-8": 1}
        assert changed == 2
        assert errors == []

    def test_collects_errors(self):
        counts, changed, errors = clean.summarise([
            ("ascii", False, None),
            (None, False, "boom: no such file"),
        ])
        assert errors == ["boom: no such file"]
        assert counts == {"ascii": 1}

    def test_empty(self):
        assert clean.summarise([]) == ({}, 0, [])


# --- stage 2 planning ---------------------------------------------------

class TestFindFiles:
    def test_sorted_and_filtered(self, tmp_path):
        (tmp_path / "b.txt").write_text("x")
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "c.zip").write_text("x")

        found = boilerplate.find_files(tmp_path)
        assert [f.name for f in found] == ["a.txt", "b.txt"]


class TestSelectBooks:
    def test_one_file_per_book(self):
        files = [Path("/g/1007/1007-0.txt"),
                 Path("/g/1007/old/1007-0.txt"),
                 Path("/g/1009/1009-0.txt")]
        chosen = boilerplate.select_books(files)
        assert len(chosen) == 2
        assert Path("/g/1007/1007-0.txt") in chosen
        assert Path("/g/1009/1009-0.txt") in chosen

    def test_keep_duplicates_keeps_everything(self):
        files = [Path("/g/1007/1007-0.txt"), Path("/g/1007/old/1007-0.txt")]
        assert len(boilerplate.select_books(files, keep_duplicates=True)) == 2

    def test_unnumbered_files_are_kept(self):
        """A path with no numeric directory is not Gutenberg's layout.
        Dropping it would lose data this rule simply does not understand."""
        files = [Path("/g/notes/readme.txt"), Path("/g/1007/1007-0.txt")]
        chosen = boilerplate.select_books(files)
        assert Path("/g/notes/readme.txt") in chosen

    def test_order_is_reproducible(self):
        files = [Path("/g/1009/1009-0.txt"), Path("/g/1007/1007-0.txt"),
                 Path("/g/1008/1008-0.txt")]
        assert boilerplate.select_books(files) == \
            boilerplate.select_books(list(reversed(files)))

    def test_empty(self):
        assert boilerplate.select_books([]) == []


class TestBoilerplatePlanJobs:
    def test_flattens_to_book_id(self):
        jobs = boilerplate.plan_jobs(
            [Path("/g/1/0/0/1007/old/old/3ddcc10.txt")], Path("/out"))
        assert jobs == [("/g/1/0/0/1007/old/old/3ddcc10.txt",
                         "/out/1007.txt")]

    def test_unnumbered_keeps_its_name(self):
        jobs = boilerplate.plan_jobs([Path("/g/notes/readme.txt")],
                                     Path("/out"))
        assert jobs[0][1] == "/out/readme.txt"

    def test_empty(self):
        assert boilerplate.plan_jobs([], Path("/out")) == []


# --- properties ---------------------------------------------------------

@settings(max_examples=200)
@given(st.lists(st.sampled_from(["1007", "1009", "10038"]),
                min_size=0, max_size=8))
def test_select_books_yields_one_per_book(ids):
    """However many revisions of however many books go in, exactly one
    file per distinct book comes out."""
    files = [Path(f"/g/{i}/{i}-0.txt") for i in ids] + \
            [Path(f"/g/{i}/old/{i}-0.txt") for i in ids]
    chosen = boilerplate.select_books(files)
    assert len(chosen) == len(set(ids))


@settings(max_examples=200)
@given(st.lists(st.sampled_from([
    "1007/1007-0.txt", "1007/1007-8.txt", "1007/old/1007-0.txt",
    "1009/1009-0.txt", "1009/old/old/x.txt",
]), min_size=1, max_size=6, unique=True))
def test_select_books_never_invents_a_file(names):
    files = [Path("/g") / n for n in names]
    for c in boilerplate.select_books(files):
        assert c in files


@settings(max_examples=150)
@given(st.lists(st.sampled_from(["a", "b", "c", "d"]),
                min_size=0, max_size=4, unique=True))
def test_plan_jobs_is_one_to_one(names):
    """Every chosen file produces exactly one job, and no two jobs write
    to the same destination -- a collision would silently overwrite a
    book with another."""
    files = [Path(f"/g/{i}/{n}.txt") for i, n in enumerate(names)]
    jobs = boilerplate.plan_jobs(files, Path("/out"))
    assert len(jobs) == len(files)
    assert len({d for _s, d in jobs}) == len(jobs)
