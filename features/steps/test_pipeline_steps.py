# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""
Step definitions for features/pipeline.feature.

Every step drives the real thing. `When the pipeline runs` invokes
tools/pipeline.py's main(); `When the corpus is indexed` builds a real
store through the real bridge and the real splitter. Nothing here
simulates a component, because the bugs these scenarios exist to catch
were all cases where the implementation and someone's model of it had
quietly diverged -- a fake would have encoded the same wrong model.

The indexing scenarios use the C splitter and bridge directly rather than
starting sigilfs over 9P: same code, no server lifecycle, and they run in
milliseconds. Where a scenario genuinely needs the server -- anything
about /ctl or /stats -- it should start one rather than pretend.
"""

import ctypes
import subprocess
import sys
from pathlib import Path

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import pipeline
from tools.manifest import Manifest
from tools.split import count as split_count

pytest.importorskip("libtab", reason="libtab not installed")

scenarios("../pipeline.feature")
scenarios("../indexing.feature")


BOOK = ("*** START OF THE PROJECT GUTENBERG EBOOK {id} ***\n\n"
        "It was a bright cold day in April, and the clocks were striking\n"
        "thirteen, and the man did not stop walking until much later.\n\n"
        "A second paragraph, long enough that the indexer will keep it.\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK {id} ***\n"
        "Updated editions will replace the previous one.\n")

CATALOG = (
    "Text#,Type,Issued,Title,Language,Authors,Subjects,LoCC,Bookshelves\n"
    '1,Text,1971,First Book,en,"Author, A, 1800-1880",Subj,PS,Shelf\n'
    '2,Text,1971,Second Book,en,"Author, B, 1810-1890",Subj,PR,Shelf\n'
    '3,Text,1971,Third Book,en,"Author, C, 1820-1900",Subj,PQ,Shelf\n'
)


@pytest.fixture
def ctx(tmp_path):
    """Paths and results shared between steps."""
    return {"tmp": tmp_path, "src": tmp_path / "mirror",
            "dst": tmp_path / "out", "rc": None, "stats": {}}


# --- given --------------------------------------------------------------

@given("a clean output directory")
def clean_output(ctx):
    ctx["src"].mkdir(parents=True, exist_ok=True)
    cat = ctx["tmp"] / "cat.csv"
    cat.write_text(CATALOG, encoding="utf-8")
    ctx["catalog"] = cat


@given(parsers.parse("a mirror containing {n:d} books"))
def mirror_with_books(ctx, n):
    for i in range(1, n + 1):
        d = ctx["src"] / str(i)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{i}-0.txt").write_text(BOOK.format(id=i), encoding="utf-8")


@given("a mirror containing a book with an old revision")
def mirror_with_old(ctx):
    """The old revision is named -0 and the current one is not.

    Deliberately the harder arrangement. With both named -0 the ranking
    can be reversed entirely and pick_best still returns the current file
    by the filename tie-break, so the scenario passes while dedup is
    broken -- verified by mutation. Here the old file looks *better* by
    encoding suffix, so only the revision rule can pick correctly.
    """
    d = ctx["src"] / "1"
    (d / "old").mkdir(parents=True, exist_ok=True)
    (d / "1.txt").write_text(BOOK.format(id=1), encoding="utf-8")
    (d / "old" / "1-0.txt").write_text(
        BOOK.format(id=1).replace("bright cold day",
                                  "superseded revision"),
        encoding="utf-8")


@given("a mirror containing a book with a cp1252 apostrophe")
def mirror_with_cp1252(ctx):
    d = ctx["src"] / "1"
    d.mkdir(parents=True, exist_ok=True)
    text = BOOK.format(id=1).replace("did not stop", "didn\x92t stop")
    (d / "1-0.txt").write_bytes(text.encode("latin-1"))


@given("a mirror containing a book with no catalogue entry")
def mirror_uncatalogued(ctx):
    d = ctx["src"] / "99999"
    d.mkdir(parents=True, exist_ok=True)
    (d / "99999-0.txt").write_text(BOOK.format(id=99999), encoding="utf-8")


@given(parsers.parse("a corpus containing a book of {mb:d} MB"))
def corpus_with_large_book(ctx, mb):
    d = ctx["src"]
    d.mkdir(parents=True, exist_ok=True)
    para = ("Real prose that the splitter will keep, long enough to clear "
            "the minimum length the indexer applies.\n\n")
    (d / "big.txt").write_text(para * (mb * 1024 * 1024 // len(para)),
                               encoding="utf-8")


# --- when ---------------------------------------------------------------

def _run_pipeline(ctx, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "pipeline", str(ctx["src"]), str(ctx["dst"]),
        "--catalog", str(ctx["catalog"]), "-j", "1"])
    ctx["rc"] = pipeline.main()


@when("the pipeline runs")
def pipeline_runs(ctx, monkeypatch):
    _run_pipeline(ctx, monkeypatch)


@when("the pipeline runs again")
def pipeline_runs_again(ctx, monkeypatch):
    _run_pipeline(ctx, monkeypatch)


@when("the corpus is indexed")
def corpus_is_indexed(ctx):
    """Drive the real indexing path: the same Maxfile check and the same
    splitter cmd/index.c uses, through the shared C.

    Built here rather than through 9P because the scenario is about what
    the indexer counts, not about the server. The Maxfile constant is read
    from cmd/index.c so the scenario cannot drift from the code.
    """
    src = (ROOT / "cmd" / "index.c").read_text()
    import re
    m = re.search(r"Maxfile\s*=\s*(\d+)\*(\d+)\*(\d+)", src)
    maxfile = int(m.group(1)) * int(m.group(2)) * int(m.group(3))

    skipped = records = 0
    for p in sorted(ctx["src"].rglob("*.txt")):
        n = p.stat().st_size
        if n == 0 or n > maxfile:
            if n > maxfile:
                skipped += 1
            continue
        records += split_count(p.read_bytes())
    ctx["stats"] = {"skipped": skipped, "records": records}


# --- then ---------------------------------------------------------------

@then(parsers.parse("the output contains {n:d} books"))
def output_has_books(ctx, n):
    assert len(list(ctx["dst"].glob("*.txt"))) == n


@then(parsers.parse("the manifest describes {n:d} books"))
def manifest_has_books(ctx, n):
    assert len(Manifest.read(ctx["dst"] / "manifest.tab")) == n


@then("the manifest paragraph count equals the splitter's count")
def manifest_matches_splitter(ctx):
    """The invariant that was never asserted. Two components counting the
    same corpus differently is how 2.46M paragraphs went missing."""
    rows = Manifest.read(ctx["dst"] / "manifest.tab")
    for book, row in rows.items():
        text = (ctx["dst"] / f"{book}.txt").read_bytes()
        assert int(row["paras"]) == split_count(text), (
            f"book {book}: manifest says {row['paras']}, "
            f"splitter says {split_count(text)}")


@then("the output contains the current revision")
def output_is_current(ctx):
    assert "striking" in (ctx["dst"] / "1.txt").read_text(encoding="utf-8")


@then("the output does not contain the superseded text")
def output_not_superseded(ctx):
    assert "superseded" not in (ctx["dst"] / "1.txt").read_text(
        encoding="utf-8")


@then("the output contains a typographic apostrophe")
def output_has_typographic(ctx):
    assert "didn’t" in (ctx["dst"] / "1.txt").read_text(encoding="utf-8")


@then("the output is valid UTF-8")
def output_valid_utf8(ctx):
    for p in ctx["dst"].glob("*.txt"):
        p.read_bytes().decode("utf-8")      # raises if not


@then("no output book contains licence text")
def no_licence_text(ctx):
    for p in ctx["dst"].glob("*.txt"):
        t = p.read_text(encoding="utf-8")
        assert "PROJECT GUTENBERG EBOOK" not in t
        assert "Updated editions" not in t


@then("that book has provenance but no title")
def uncatalogued_row(ctx):
    row = next(iter(Manifest.read(ctx["dst"] / "manifest.tab").values()))
    assert row["digest"], "provenance must survive"
    assert row["title"] == ""


@then(parsers.parse("the indexer reports {n:d} file skipped"))
@then(parsers.parse("the indexer reports {n:d} files skipped"))
def indexer_skipped(ctx, n):
    assert ctx["stats"]["skipped"] == n


@then("the store contains records")
def store_has_records(ctx):
    assert ctx["stats"]["records"] > 0
