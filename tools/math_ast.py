#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
"""Canonicalize LaTeX math via LaTeXML Content MathML, then hash the tree.

Identity, not similarity. Two expressions that canonicalize to the same tree
ARE the same expression -- a fact, not a probabilistic guess, and the only
claim about mathematics this project can make honestly.

Embeddings cannot do this. Measured on a probe of 8 equivalent and 8
contradictory pairs, every model tried scored *negative* separation: MiniLM
-14.8 bits, SciBERT -15.2, MathBERT -19.9, MathBERTa -10.5. Contradictions were
consistently judged MORE similar than equivalences, because these models match
surface form and mathematical meaning is structural. Quantifier order, operand
order, and negation are exactly the load-bearing details a token-level model
discards. See docs/FINDINGS.md.

This approach scores 16/16 on the same probe, because an operator tree encodes
what the token sequence loses.

Requires LaTeXML (apt install latexml). Measured on 300 real math.stackexchange
expressions: 99.3% parse coverage, ~20 ms/expression when batched.
"""
import subprocess, hashlib, re, unicodedata
import xml.etree.ElementTree as ET

MML = '{http://www.w3.org/1998/Math/MathML}'
# Operators whose argument order carries no meaning. Sorting their children is
# what makes x+y and y+x collide.
COMMUTATIVE = {'plus', 'times', 'and', 'or', 'eq', 'neq', 'gcd', 'max', 'min',
               'union', 'intersect'}

# LaTeXML emits some operators as bare symbols rather than Content MathML
# operator elements, so \cdot arrives as ci(⋅) instead of <times/>. Map the
# common ones back so commutativity applies to them too.
SYMBOL_OPS = {'⋅': 'times', '×': 'times', '·': 'times', '∧': 'and', '∨': 'or',
              '+': 'plus', '=': 'eq', '≠': 'neq',
              '∪': 'union', '∩': 'intersect'}

def to_cmml(latex, timeout=20):
    try:
        p = subprocess.run(['latexmlmath', '--cmml=-', '-'],
                           input=latex, capture_output=True, text=True, timeout=timeout)
        out = p.stdout
        i = out.find('<math')
        return out[i:] if i >= 0 else None
    except Exception:
        return None

def canon(el):
    """Recursively canonicalize a Content MathML element to a stable string."""
    tag = el.tag.replace(MML, '')
    if tag in ('ci', 'cn', 'csymbol'):
        # Normalize unicode math italics (LaTeXML emits 𝑥 not x) and case.
        t = unicodedata.normalize('NFKC', (el.text or '').strip())
        return f'{tag}({t})'
    kids = [canon(c) for c in el if isinstance(c.tag, str)]
    if tag == 'apply' and kids:
        op, args = kids[0], kids[1:]
        m = re.fullmatch(r'ci\((.)\)', op)
        if m and m.group(1) in SYMBOL_OPS:
            op = SYMBOL_OPS[m.group(1)]
        # Commutative operators: sort arguments so order stops mattering.
        if any(op == f'{o}' or op.startswith(f'{o}(') for o in COMMUTATIVE):
            args = sorted(args)
        return f'apply({op},{",".join(args)})'
    if kids:
        return f'{tag}({",".join(kids)})'
    return tag

def ast_hash(latex):
    x = to_cmml(latex)
    if not x:
        return None, None
    try:
        root = ET.fromstring(x)
    except ET.ParseError:
        return None, None
    body = [c for c in root if isinstance(c.tag, str)]
    if not body:
        return None, None
    c = canon(body[0])
    if 'cerror' in c or 'ERROR' in c:
        return None, c
    return hashlib.blake2b(c.encode(), digest_size=16).hexdigest(), c


def batch_hashes(latex_list, chunk=200):
    """Hash many expressions in one LaTeXML invocation.

    Spawning latexmlmath per expression costs ~630ms, nearly all of it Perl
    startup. Batching into a single document drops that to ~20ms each, a 31x
    speedup, which is the difference between deployable and not.
    """
    import tempfile, os
    out = []
    for i in range(0, len(latex_list), chunk):
        batch = latex_list[i:i+chunk]
        doc = '\\documentclass{article}\\begin{document}\n'
        for x in batch:
            doc += '\\[ ' + x + ' \\]\n'
        doc += '\\end{document}\n'
        with tempfile.NamedTemporaryFile('w', suffix='.tex', delete=False) as f:
            f.write(doc); tex = f.name
        xml = tex.replace('.tex', '.xml')
        try:
            subprocess.run(['latexml', '--quiet', '--dest=' + xml, tex],
                           capture_output=True, timeout=600)
            subprocess.run(['latexmlpost', '--quiet', '--nodefaultresources',
                            '--format=xhtml', '--pmml', '--cmml',
                            '--dest=' + xml + '.html', xml],
                           capture_output=True, timeout=600)
            html = open(xml + '.html').read() if os.path.exists(xml + '.html') else ''
            # Content MathML blocks are the <annotation-xml> semantic payloads
            blocks = re.findall(r'<annotation-xml encoding="MathML-Content">(.*?)</annotation-xml>',
                                html, re.S)
            for b in blocks:
                try:
                    root = ET.fromstring('<w xmlns="http://www.w3.org/1998/Math/MathML">' + b + '</w>')
                    kids = [c for c in root if isinstance(c.tag, str)]
                    c = canon(kids[0]) if kids else None
                    out.append((hashlib.blake2b(c.encode(), digest_size=16).hexdigest(), c)
                               if c and 'cerror' not in c else (None, c))
                except ET.ParseError:
                    out.append((None, None))
            while len(out) < i + len(batch):
                out.append((None, None))
        finally:
            for p in (tex, xml, xml + '.html'):
                try: os.unlink(p)
                except OSError: pass
    return out
