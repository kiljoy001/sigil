#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
"""Canonicalize Excel formulas to a hash, the same way math_ast.py does LaTeX.

A spreadsheet has three layers and each wants a different mechanism:

  formulas    parseable grammar -> exact identity by AST hash
  labels      prose -> LSH similarity
  references  cell dependencies -> asserted graph edges

This handles the first. Two formulas that canonicalize identically ARE the same
computation, which is a fact rather than an estimate -- the same claim the math
AST work established, and the same reason it is worth having.

Excel should be easier than LaTeX: a closed function set, no user-defined
macros in the formula language, no \\newcommand. Whether that holds on real
corporate spreadsheets is the thing to measure, not assume.

Canonicalization normalizes what does not change the computation:
  - absolute/relative markers ($A$1 vs A1) -- same dependency
  - case (SUM vs sum)
  - argument order for commutative functions (SUM, PRODUCT, MAX, MIN, AND, OR)
  - whitespace
It deliberately does NOT normalize cell addresses to positions: A1+B1 in one
sheet and C5+D5 in another are different computations over different data.
"""

import hashlib
import re
import sys

# Functions whose argument order carries no meaning.
COMMUTATIVE = {"SUM", "PRODUCT", "MAX", "MIN", "AND", "OR", "COUNT", "COUNTA",
               "AVERAGE", "MEDIAN", "GCD", "LCM"}

TOKEN = re.compile(r"""
    (?P<str>"(?:[^"]|"")*")            |
    (?P<func>[A-Z][A-Z0-9_.]*)\s*\(    |
    (?P<ref>(?:(?:'[^']+'|[A-Za-z0-9_]+)!)?\$?[A-Z]{1,3}\$?[0-9]{1,7}
             (?::\$?[A-Z]{1,3}\$?[0-9]{1,7})?)  |
    (?P<num>\d+\.?\d*(?:[eE][+-]?\d+)?) |
    (?P<err>(?:(?:'[^']+'|[A-Za-z0-9_]+)!)?
            \#(?:REF!|DIV/0!|N/A|VALUE!|NAME\?|NUM!|NULL!|SPILL!|CALC!|GETTING_DATA))  |
    (?P<name>(?:'[^']+'!)?[A-Za-z_\\][A-Za-z0-9_.\\]*)  |
    (?P<op><=|>=|<>|[-+*/^&<>=%])      |
    (?P<punct>[(),;:{}\[\]!])            |
    (?P<ws>\s+)
""", re.VERBOSE | re.IGNORECASE)


def normalize_ref(t):
    """$A$1 and A1 address the same cell; the marker only affects copy-paste."""
    return t.replace("$", "").upper()


def tokenize(f):
    """Return a token list, or None if anything is unrecognized."""
    if f.startswith("="):
        f = f[1:]
    out, i = [], 0
    while i < len(f):
        m = TOKEN.match(f, i)
        if not m:
            return None
        i = m.end()
        kind = m.lastgroup
        if kind == "ws":
            continue
        if kind == "func":
            out.append(("func", m.group("func").upper()))
            out.append(("punct", "("))
        elif kind == "ref":
            out.append(("ref", normalize_ref(m.group(0))))
        elif kind == "str":
            out.append(("str", m.group(0)))
        elif kind == "name":
            # Named range or defined name: a label for a region, so it behaves
            # like a reference rather than a value.
            out.append(("name", m.group(0).upper()))
        elif kind == "err":
            out.append(("err", m.group(0).upper()))
        else:
            out.append((kind, m.group(0).upper()))
    return out


def parse(tokens, pos=0):
    """Flat structural parse: enough to identify a computation, not to run it."""
    parts = []
    while pos < len(tokens):
        kind, val = tokens[pos]
        if kind == "func":
            name = val
            pos += 2                       # skip func and its '('
            args, depth, cur = [], 1, []
            while pos < len(tokens) and depth:
                k, v = tokens[pos]
                if k == "punct" and v == "(":
                    depth += 1
                elif k == "punct" and v == ")":
                    depth -= 1
                    if depth == 0:
                        pos += 1
                        break
                elif k == "punct" and v in ",;" and depth == 1:
                    args.append(cur); cur = []; pos += 1
                    continue
                cur.append(tokens[pos]); pos += 1
            if cur:
                args.append(cur)
            rendered = [parse(a)[0] for a in args]
            if name in COMMUTATIVE:
                rendered = sorted(rendered)
            parts.append(f"{name}({','.join(rendered)})")
            continue
        if kind == "punct" and val == ")":
            pos += 1
            continue
        parts.append(f"{kind}:{val}")
        pos += 1
    return "".join(parts), pos


def formula_hash(f):
    """(hash, canonical) or (None, reason) if the formula could not be parsed."""
    if not isinstance(f, str) or not f.startswith("="):
        return None, "not-a-formula"
    toks = tokenize(f)
    if toks is None:
        return None, "unparsed"
    canon, _ = parse(toks)
    if not canon:
        return None, "empty"
    return hashlib.blake2b(canon.encode(), digest_size=16).hexdigest(), canon


if __name__ == "__main__":
    for f in sys.argv[1:]:
        h, c = formula_hash(f)
        print(f"{str(h)[:16]:16s} {c}")
