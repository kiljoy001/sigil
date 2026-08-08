#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
"""Embed arbitrarily long paragraphs by chunking and pooling.

A paragraph's length is a property of the paragraph, not of the model's context
window. Truncating discards content silently; chunking keeps all of it.

Chunks are split at sentence boundaries where possible, embedded separately,
and their vectors averaged then renormalized. Averaging in float space is the
correct combiner -- hashing the codes together would destroy the locality that
makes them useful, and a per-bit majority vote quantizes before averaging
rather than after.
"""
import json, subprocess, tempfile, os, sys, re
import numpy as np

BIN = "/home/scott/llama.cpp/build/bin/llama-embedding"
MODEL = "/home/scott/models/all-MiniLM-L6-v2-f16.gguf"
# Conservative: academic prose with notation can approach 1 token/char, and a
# line exceeding the batch size makes llama-embedding abort with empty output.
MAX_CHARS = 800

def split_chunks(text, limit=MAX_CHARS):
    """Split at sentence boundaries, falling back to hard cuts."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return [text]
    out, cur = [], ""
    for sent in re.split(r'(?<=[.!?])\s+', text):
        while len(sent) > limit:            # single sentence too long
            cut = sent.rfind(" ", 0, limit)
            if cut < limit // 2:
                cut = limit
            out.append(sent[:cut]); sent = sent[cut:].strip()
        if len(cur) + len(sent) + 1 <= limit:
            cur = (cur + " " + sent).strip()
        else:
            if cur:
                out.append(cur)
            cur = sent
    if cur:
        out.append(cur)
    return out or [text[:limit]]

def _embed_lines(lines):
    """Embed one line per row. Returns None if the batch aborted."""
    tf = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    tf.write("\n".join(lines) + "\n"); tf.close()
    try:
        p = subprocess.run(
            [BIN, "-m", MODEL, "--pooling", "mean", "--embd-normalize", "2",
             "--embd-output-format", "json", "-c", "512", "-b", "2048",
             "-f", tf.name],                      # -f, never stdin: see FINDINGS
            capture_output=True, text=True, timeout=1200)
        d = json.loads(p.stdout)["data"]
    except (json.JSONDecodeError, subprocess.TimeoutExpired, KeyError):
        return None
    finally:
        os.unlink(tf.name)
    return [x["embedding"] for x in d] if len(d) == len(lines) else None

def embed_paragraphs(paras, batch=200, progress=True):
    """One vector per paragraph, whatever its length."""
    # Flatten to chunks, remembering which paragraph each belongs to.
    owner, chunks = [], []
    for i, p in enumerate(paras):
        for c in split_chunks(p):
            owner.append(i); chunks.append(c)
    vecs = [None] * len(chunks)
    for i in range(0, len(chunks), batch):
        idx = list(range(i, min(i + batch, len(chunks))))
        got = _embed_lines([chunks[j] for j in idx])
        if got is None:                      # fall back to one at a time
            got = []
            for j in idx:
                one = _embed_lines([chunks[j]])
                got.append(one[0] if one else [0.0] * 384)
        for j, v in zip(idx, got):
            vecs[j] = v
        if progress:
            sys.stderr.write(f"\r  {min(i+batch,len(chunks))}/{len(chunks)} chunks")
            sys.stderr.flush()
    if progress:
        sys.stderr.write("\n")
    # Pool: mean of the chunk vectors, renormalized.
    dim = len(vecs[0])
    out = np.zeros((len(paras), dim), dtype=np.float32)
    for j, o in enumerate(owner):
        out[o] += np.asarray(vecs[j], dtype=np.float32)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-12), len(chunks)
