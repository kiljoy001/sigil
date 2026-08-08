#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
"""Extract the prose layer of a spreadsheet: labels, headers, sheet names.

Formulas get an exact identity channel (xlsx_ast.py). This is the other half —
the text a human wrote to say what the numbers mean, which is what similarity
should operate on.

The unit is deliberately not a single cell. "Q3" alone carries almost nothing,
and MiniLM does poorly on very short strings. A header ROW or a label COLUMN
read together is closer to a sentence, and closer to the paragraph unit the
rest of the design uses.
"""
import re
import warnings
warnings.filterwarnings("ignore")
import openpyxl

# Cells that are dates, currency, or pure numbers formatted as text carry no
# semantic content worth embedding.
NUMERIC = re.compile(r'^[\s$£€¥]*[-+]?[\d,]*\.?\d+\s*%?$')
DATEISH = re.compile(r'^\d{1,4}[-/]\d{1,2}([-/]\d{1,4})?$')


def is_label(v):
    """A string cell that reads as a human label rather than data."""
    if not isinstance(v, str):
        return False
    t = v.strip()
    if len(t) < 2 or len(t) > 200:
        return False
    if t.startswith("="):
        return False
    if NUMERIC.match(t) or DATEISH.match(t):
        return False
    return bool(re.search(r'[A-Za-z]{2,}', t))


def extract(path, max_rows=400, max_cols=60):
    """Yield (sheet, kind, position, text) label groups from a workbook.

    kind is 'sheet', 'row' or 'col'. Rows and columns are joined because a
    single header cell is usually too short to embed meaningfully.
    """
    out = []
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    try:
        for ws in wb.worksheets:
            if is_label(ws.title):
                out.append((ws.title, "sheet", 0, ws.title.strip()))
            grid = {}
            for r, row in enumerate(ws.iter_rows(max_row=max_rows,
                                                 max_col=max_cols)):
                for c, cell in enumerate(row):
                    if is_label(cell.value):
                        grid[(r, c)] = cell.value.strip()
                if r > max_rows:
                    break
            if not grid:
                continue
            byrow, bycol = {}, {}
            for (r, c), t in grid.items():
                byrow.setdefault(r, []).append((c, t))
                bycol.setdefault(c, []).append((r, t))
            # A row of labels with several entries reads like a header line.
            for r, items in byrow.items():
                if len(items) >= 3:
                    txt = " | ".join(t for _, t in sorted(items))
                    out.append((ws.title, "row", r, txt[:500]))
            # A column of labels is usually the row-header stub.
            for c, items in bycol.items():
                if len(items) >= 4:
                    txt = " | ".join(t for _, t in sorted(items))
                    out.append((ws.title, "col", c, txt[:500]))
    finally:
        wb.close()
    return out


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        for sheet, kind, pos, txt in extract(p)[:12]:
            print(f"{kind:5s} {sheet[:18]:18s} {txt[:90]}")
