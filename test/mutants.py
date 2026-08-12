# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""
Curated mutants for the Python pipeline, as data.

These began as shell arguments inside test/mutate_py.sh, which meant every
anchor string passed through sh quoting, python -c quoting and str.replace
in turn. Three of them silently failed to apply, reporting DID NOT APPLY --
better than reporting a false pass, but still a mutant that tested nothing.
The definitions deserve the same care as the code they check, so they live
here where the strings are written literally.

Each entry is (name, file, old, new). The runner asserts that `old` appears
exactly once before replacing it, so a refactor that moves the anchor fails
loudly rather than quietly skipping the check.

The pairing with test/mutate_mutmut.sh is the point: these are mistakes
someone could plausibly make here, chosen for that reason. mutmut supplies
the ones nobody chose. Neither set covers the other.
"""

MUTANTS = [
    # --- tools/clean.py: encoding repair -------------------------------
    (
        "decode with surrogateescape (the original bug)",
        "tools/clean.py",
        "    text = _decode(data)",
        '    text = data.decode("utf-8", errors="surrogateescape")',
    ),
    (
        "U+FFFD replacement instead of CP1252 transcoding",
        "tools/clean.py",
        "    text = _decode(data)",
        '    text = data.decode("utf-8", errors="replace")',
    ),
    (
        "forget NUL removal",
        "tools/clean.py",
        '    text = text.replace("\\x00", "")\n',
        "",
    ),
    (
        "forget CRLF normalisation",
        "tools/clean.py",
        '    text = text.replace("\\r\\n", "\\n").replace("\\r", "\\n")\n',
        "",
    ),
    (
        "forget BOM stripping",
        "tools/clean.py",
        '    if text.startswith("\ufeff"):',
        "    if False:",
    ),

    # --- tools/clean.py: planning and I/O ------------------------------
    (
        "resume skips when the source is newer",
        "tools/clean.py",
        "d.stat().st_mtime >= s.stat().st_mtime",
        "d.stat().st_mtime <= s.stat().st_mtime",
    ),
    (
        "write only changed files",
        "tools/clean.py",
        "    if not dry_run:\n",
        "    if not dry_run and changed:\n",
    ),
    (
        "dry-run writes anyway",
        "tools/clean.py",
        "    if not dry_run:\n",
        "    if True:\n",
    ),

    # --- src/utf8_repair.c: the C guard --------------------------------
    #
    # Checked against the Python implementation by the differential suite,
    # so a change to either must be caught.
    (
        "C: accept overlong 2-byte forms (0xC0 lead)",
        "src/utf8_repair.c",
        "if (p[0] < 0xC2)",
        "if (p[0] < 0xC0)",
    ),
    (
        "C: accept surrogate halves",
        "src/utf8_repair.c",
        "if (p[0] == 0xED && p[1] >= 0xA0)",
        "if (0)",
    ),
    (
        "C: U+FFFD instead of CP1252 transcoding",
        "src/utf8_repair.c",
        "if (b >= 0x80 && b <= 0x9F && cp1252_c1[b - 0x80] != 0)\n"
        "\t\t\t\tcp = cp1252_c1[b - 0x80];",
        "cp = 0xFFFD;",
    ),

    # --- tools/boilerplate.py: markers ---------------------------------
    # Both the START and END patterns carry this alternation, so the
    # anchor spans enough context to be unique. Mutating only one of them
    # is a weaker mutation than the name claims -- which is exactly what
    # the shell version of this file was silently doing.
    (
        "only match THE, not THIS (START)",
        "tools/boilerplate.py",
        r'r"^\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*\s*$"',
        r'r"^\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*\s*$"',
    ),
    (
        "only match THE, not THIS (END)",
        "tools/boilerplate.py",
        r'r"^\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*\s*$"',
        r'r"^\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*\s*$"',
    ),
    (
        "require a space after the asterisks",
        "tools/boilerplate.py",
        r"\s*START",
        " START",
    ),
    (
        "END searched from 0, not after START",
        "tools/boilerplate.py",
        "_END.search(text, body_from)",
        "_END.search(text)",
    ),
    (
        "drop the malformed-slice guard",
        "tools/boilerplate.py",
        "    if body_to <= body_from:\n"
        "        return text                      # malformed; keep everything\n",
        "",
    ),
    (
        "strip Produced-by even if nothing remains",
        "tools/boilerplate.py",
        "        if candidate.strip():\n            body = candidate",
        "        body = candidate",
    ),

    # --- tools/boilerplate.py: deduplication ---------------------------
    #
    # Ranking wrong indexes a superseded edition, with no visible symptom.
    (
        "rank encoding above revision",
        "tools/boilerplate.py",
        "    return (depth, enc, str(path))",
        "    return (enc, depth, str(path))",
    ),
    (
        "drop the tie-breaker (nondeterministic choice)",
        "tools/boilerplate.py",
        "    return (depth, enc, str(path))",
        "    return (depth, enc)",
    ),
    (
        "book_id takes the first numeric dir, not the last",
        "tools/boilerplate.py",
        "    for part in reversed(path.parent.parts):",
        "    for part in path.parent.parts:",
    ),
    (
        "drop unnumbered files instead of keeping them",
        "tools/boilerplate.py",
        "] + unnumbered",
        "]",
    ),
    (
        "keep_duplicates ignored",
        "tools/boilerplate.py",
        "    if keep_duplicates:\n        return list(all_files)\n",
        "",
    ),

    # --- tools/metadata.py: ndb quoting --------------------------------
    #
    # The worst failure mode available: libtab accepts the value, writes
    # the file, and something later cannot open it.
    (
        "quote: forget the leading-# case",
        "tools/metadata.py",
        's == "" or s.startswith("#") or any(',
        's == "" or any(',
    ),
    (
        "quote: forget empty values",
        "tools/metadata.py",
        's == "" or s.startswith("#") or any(',
        's.startswith("#") or any(',
    ),
    (
        "quote: interior quotes not doubled",
        "tools/metadata.py",
        "'\"' + s.replace('\"', '\"\"') + '\"'",
        "'\"' + s + '\"'",
    ),
    (
        "quote: forget tab",
        "tools/metadata.py",
        "' \\t\\n\\r\"'",
        "' \\n\\r\"'",
    ),

    # --- tools/metadata.py: the ground-truth fields --------------------
    #
    # A wrong value here does not crash. It changes what "correct" means
    # for every measurement made against this corpus.
    (
        "death: take the birth year",
        "tools/metadata.py",
        "int(m.group(2))",
        "int(m.group(1))",
    ),
    (
        "death: reject the cataloguer's approximate dates",
        "tools/metadata.py",
        r'r"\b(\d{4})\??\s*-\s*(\d{4})\??"',
        r'r"\b(\d{4})\s*-\s*(\d{4})\b"',
    ),
    (
        "locc: take the last class, not the first",
        "tools/metadata.py",
        '.split(";")[0].strip()',
        '.split(";")[-1].strip()',
    ),
    (
        "locc: keep the digits",
        "tools/metadata.py",
        '    return m.group(1) if m else ""',
        "    return first",
    ),
    (
        "catalog: keep non-Text rows",
        "tools/metadata.py",
        'if row.get("Type") != "Text":',
        "if False:",
    ),
]
