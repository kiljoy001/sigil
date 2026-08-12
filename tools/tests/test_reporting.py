# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""
Tests for the end-of-run summary and exit status.

These were extracted from main() so that mutation testing has something
worth measuring. Before the split, 95 of 107 surviving mutants were inside
main() -- print intervals, progress counters, help strings -- and drowning
the real signal in a budget of 110.

The distinction that matters: the *numbers* in a report are worth testing,
because they are the only feedback a 60,830-file run gives, and a wrong
count is how a partial run passes for a complete one. The *wording* is not.
Nothing here asserts on phrasing beyond the few substrings that carry
meaning, and mutmut's remaining string mutants in main() are exempted by
name in test/mutate_mutmut.sh rather than pinned by a brittle test.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools import boilerplate, clean


class TestExitStatus:
    """The whole contract between a run and the shell."""

    def test_clean_zero_when_no_errors(self):
        assert clean.exit_status([]) == 0

    def test_clean_one_when_any_error(self):
        assert clean.exit_status(["boom"]) == 1
        assert clean.exit_status(["a", "b"]) == 1

    def test_boilerplate_zero_when_no_errors(self):
        assert boilerplate.exit_status([]) == 0

    def test_boilerplate_one_when_any_error(self):
        assert boilerplate.exit_status(["boom"]) == 1


class TestCleanReport:
    def test_counts_appear(self):
        r = clean.format_report(100, 7, {"ascii": 60, "cp1252": 40}, [])
        assert "100" in r and "7" in r
        assert "60" in r and "40" in r

    def test_encodings_are_sorted(self):
        """Stable order, so two runs over the same corpus produce the
        same report and a diff between them means something."""
        r = clean.format_report(3, 0, {"utf-8": 1, "ascii": 1, "cp1252": 1}, [])
        assert r.index("ascii") < r.index("cp1252") < r.index("utf-8")

    def test_no_error_section_when_clean(self):
        assert "error" not in clean.format_report(5, 0, {"ascii": 5}, [])

    def test_errors_are_listed(self):
        r = clean.format_report(2, 0, {}, ["/a.txt: bad", "/b.txt: worse"])
        assert "/a.txt: bad" in r and "/b.txt: worse" in r

    def test_error_list_is_capped_and_says_so(self):
        """A run against the wrong directory fails on every file. 60,830
        tracebacks is not a report -- but the total must still be
        visible, or a truncated list understates the damage."""
        errs = [f"/f{i}.txt: bad" for i in range(100)]
        r = clean.format_report(100, 0, {}, errs)

        assert r.count("/f") == 20          # capped at 20 listed
        assert "100 errors" in r            # but the total is still shown
        assert "80 more" in r

    def test_exactly_twenty_errors_has_no_more_line(self):
        """The boundary: 20 fit, so there is nothing further to mention.

        Found by generated mutation testing -- > 20 could become >= 20 or
        > 21 and nothing failed, because the tests used 0, 2 and 100.
        """
        r = clean.format_report(20, 0, {}, [f"/f{i}: bad" for i in range(20)])
        assert r.count("/f") == 20
        assert "more" not in r

    def test_twenty_one_errors_mentions_one_more(self):
        r = clean.format_report(21, 0, {}, [f"/f{i}: bad" for i in range(21)])
        assert r.count("/f") == 20
        assert "1 more" in r

    def test_report_is_one_line_per_item(self):
        """The report is line-structured: a header, then one line per
        encoding, then one line per error.

        This is what the newline join is for, and asserting the line
        count is how to check it. An earlier version of this test
        asserted that the mutmut sentinel "XX" was absent, which tests
        the mutation tool rather than the code -- any other wrong
        separator passed it.
        """
        r = clean.format_report(5, 2, {"ascii": 3, "cp1252": 2},
                                ["/a: bad", "/b: bad"])
        lines = r.splitlines()

        # blank, header, 2 encodings, blank, "2 errors:", 2 errors
        assert len(lines) == 8
        assert lines[1].startswith("processed 5 files")
        assert [l.strip() for l in lines[2:4]] == ["ascii      3",
                                                   "cp1252     2"]
        assert [l.strip() for l in lines[6:8]] == ["/a: bad", "/b: bad"]

    def test_each_encoding_is_its_own_line(self):
        """Scales with the number of encodings, so a separator that
        merged them would change the count."""
        for n in (1, 3, 5):
            counts = {f"enc{i}": i for i in range(n)}
            r = clean.format_report(1, 0, counts, [])
            assert len(r.splitlines()) == 2 + n

    def test_zero_files(self):
        r = clean.format_report(0, 0, {}, [])
        assert "0 files" in r


class TestBoilerplateReport:
    def test_totals_appear(self):
        r = boilerplate.format_report(110, 53_098, [])
        assert "110" in r
        assert "53,098" in r

    def test_per_book_average(self):
        r = boilerplate.format_report(10, 1000, [])
        assert "100" in r, "1000 bytes over 10 books is 100 each"

    def test_no_division_by_zero_when_nothing_written(self):
        """A run where every file failed still has to print a summary."""
        r = boilerplate.format_report(0, 0, ["everything: broke"])
        assert "0 books" in r

    def test_error_list_is_capped_and_says_so(self):
        errs = [f"/f{i}.txt: bad" for i in range(50)]
        r = boilerplate.format_report(0, 0, errs)

        assert r.count("/f") == 20
        assert "50 errors" in r
        assert "30 more" in r

    def test_exactly_twenty_errors_has_no_more_line(self):
        r = boilerplate.format_report(0, 0, [f"/f{i}: b" for i in range(20)])
        assert r.count("/f") == 20
        assert "more" not in r

    def test_twenty_one_errors_mentions_one_more(self):
        r = boilerplate.format_report(0, 0, [f"/f{i}: b" for i in range(21)])
        assert "1 more" in r

    def test_per_book_divisor_is_one_not_two(self):
        """max(written, 1): with one book the average is the whole total.

        Asserted on the parenthesised per-book figure, not on "1,000"
        anywhere in the string -- the total is also 1,000, so a substring
        check matches whether the divisor is 1 or 2. That ambiguity let
        the mutant survive the first version of this test.
        """
        assert "(1,000 per book)" in boilerplate.format_report(1, 1000, [])
        assert "(500 per book)" in boilerplate.format_report(2, 1000, [])

    def test_report_is_one_line_per_item(self):
        """Header, then the error heading, then one line per error."""
        r = boilerplate.format_report(1, 10, ["/a: bad", "/b: bad"])
        lines = r.splitlines()

        # blank, header, "2 errors:", 2 errors
        assert len(lines) == 5
        assert lines[1].startswith("wrote 1 books")
        assert [l.strip() for l in lines[3:5]] == ["/a: bad", "/b: bad"]

    def test_each_error_is_its_own_line(self):
        for n in (1, 5, 20):
            r = boilerplate.format_report(0, 0,
                                          [f"/f{i}: bad" for i in range(n)])
            assert len(r.splitlines()) == 3 + n
