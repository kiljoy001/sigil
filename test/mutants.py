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
        '    if text.startswith("﻿"):',
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
        'r"^[\ufeff \\t]*\\*\\*\\*\\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.{0,300}?\\*\\*\\*"',
        'r"^[\ufeff \\t]*\\*\\*\\*\\s*START OF THE PROJECT GUTENBERG EBOOK.{0,300}?\\*\\*\\*"',
    ),
    (
        "only match THE, not THIS (END)",
        "tools/boilerplate.py",
        'r"^[\ufeff \\t]*\\*\\*\\*\\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.{0,300}?\\*\\*\\*"',
        'r"^[\ufeff \\t]*\\*\\*\\*\\s*END OF THE PROJECT GUTENBERG EBOOK.{0,300}?\\*\\*\\*"',
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

    # The ndb-quoting mutants were removed with the functions. libtab
    # encodes values itself now, and the manifest stores them verbatim.

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
    # --- tools/manifest.py ---------------------------------------------
    #
    # The manifest is what everything downstream reads instead of
    # re-deriving. A wrong row is not a crash, it is a quietly wrong
    # corpus, so these check the fields that carry meaning.
    (
        "manifest: write None instead of an empty death year",
        "tools/manifest.py",
        'row["death_year"] = "" if dy is None else str(dy)',
        'row["death_year"] = str(dy)',
    ),
    (
        "manifest: drop books with no catalogue entry",
        "tools/manifest.py",
        '        row[f] = (meta or {}).get(f, "") or ""',
        '        row[f] = meta[f]',
    ),
    # No mutant for the per-column write: book_row() fills every column
    # and the row is now indexed rather than .get()'d, so there is no
    # default to mutate. A missing column raises, which is what a caller
    # bug should do.
    # The nil-sentinel mutant went with the substitution it tested:
    # libtab distinguishes semantic nil from the literal string "nil"
    # now, so nothing needs substituting.
    # No "forget to commit" mutant: libtab's close() flushes, verified at
    # 5,000 rows, so dropping the commit() changes nothing observable. It
    # is an equivalent mutation, not a gap -- recorded here so nobody adds
    # it back expecting it to be caught.
    (
        "catalog: keep non-Text rows",
        "tools/metadata.py",
        'if row.get("Type") != "Text":',
        "if False:",
    ),

    # --- tools/pipeline.py ----------------------------------------------
    #
    # The wiring is where this kind of program goes wrong: an ordering
    # that wastes work, a resume that loses the record, a count that
    # disagrees with what the indexer will do.
    (
        "pipeline: clean before dedup (wastes 2/3 of the work)",
        "tools/pipeline.py",
        "    chosen = boilerplate.select_books(all_files,\n"
        "                                      keep_duplicates=keep_duplicates)",
        "    chosen = all_files",
    ),
    (
        "pipeline: skipped books get no manifest row",
        "tools/pipeline.py",
        "    for src, dst, book in done:",
        "    for src, dst, book in []:",
    ),
    (
        "pipeline: resume redoes current books",
        "tools/pipeline.py",
        "        if d.exists() and d.stat().st_mtime >= Path(src).stat().st_mtime:",
        "        if False:",
    ),
    # The splitter defines the unit of identity, so a wrong bound here
    # changes what every record hashes over. It lives in C and is shared
    # by cmd/index.c, tools/ and the fuzzer.
    (
        "split: drop over-long paragraphs instead of chunking",
        "src/split.c",
        "		while (len > SIGIL_MAXPARA) {",
        "		while (0) {",
    ),
    (
        "split: keep chunks below the minimum",
        "src/split.c",
        "		if (len >= SIGIL_MINPARA) {",
        "		if (1) {",
    ),
    (
        "split: paragraph ends only at \\n\\n, not \\n\\r",
        "src/split.c",
        "			if (q[0] == '\\n' && (q[1] == '\\n' || q[1] == '\\r'))",
        "			if (q[0] == '\\n' && q[1] == '\\n')",
    ),
    (
        "boilerplate: index LICENSE.txt as a book",
        "tools/boilerplate.py",
        "    return path.name.lower() not in _NOT_A_BOOK",
        "    return True",
    ),
    (
        "boilerplate: START must not be indented",
        "tools/boilerplate.py",
        'r"^[\ufeff \\t]*\\*\\*\\*\\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.{0,300}?\\*\\*\\*"',
        'r"^\\*\\*\\*\\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.{0,300}?\\*\\*\\*"',
    ),
    (
        "boilerplate: END must not be indented",
        "tools/boilerplate.py",
        'r"^[\ufeff \\t]*\\*\\*\\*\\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.{0,300}?\\*\\*\\*"',
        'r"^\\*\\*\\*\\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.{0,300}?\\*\\*\\*"',
    ),
    (
        "boilerplate: no BOM allowed before the marker",
        "tools/boilerplate.py",
        'r"^[\ufeff \\t]*\\*\\*\\*\\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.{0,300}?\\*\\*\\*"',
        'r"^[ \\t]*\\*\\*\\*\\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.{0,300}?\\*\\*\\*"',
    ),

    # --- src/veccache.c -------------------------------------------------
    #
    # The cache exists so 13 hours of GPU work survives a crash. Every
    # mutant here is a way it could appear to work while losing data.
    (
        "veccache: model not part of the key",
        "src/veccache.c",
        "\t\tif (ml != strlen(c->model) ||\n"
        "\t\t    strncmp(m, c->model, ml) != 0)\n"
        "\t\t\tcontinue;",
        "\t\t(void)ml;",
    ),
    (
        "veccache: a duplicate overwrites the first",
        "src/veccache.c",
        "\t\t\tfree(vec);\n\t\t\treturn 0;",
        "\t\t\tfree(c->tab[i].vec);\n\t\t\tc->tab[i].vec = vec;\n"
        "\t\t\treturn 0;",
    ),
    (
        "veccache: probe compares only the bucket key",
        "src/veccache.c",
        "\t\tif (memcmp(c->tab[i].hash, hash, HASHLEN) == 0) {\n"
        "\t\t\tsize_t d;",
        "\t\tif (memcmp(c->tab[i].hash, hash, 8) == 0) {\n"
        "\t\t\tsize_t d;",
    ),
    (
        "veccache: sync does not flush",
        "src/veccache.c",
        "\tif (fflush(c->fp) != 0)\n\t\treturn -1;",
        "\t;",
    ),
    (
        "veccache: a short line is loaded anyway",
        "src/veccache.c",
        "\t\t               c->dim * sizeof *vec) != c->dim * sizeof *vec) {",
        "\t\t               c->dim * sizeof *vec) == (size_t)-1) {",
    ),
    # No "hash length not checked" mutant: it is equivalent. With the
    # guard removed, hex_decode reads 64 characters, hits the closing
    # quote at index 4 of a short hash, finds it is not a hex digit and
    # returns -1 -- so the line is rejected either way. The guard is a
    # cheap early exit, not the thing enforcing correctness. Recorded so
    # nobody adds it back expecting a kill.
]
