#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""Embed paragraphs through ollama, on the GPU.

llama-embedding runs on the CPU here (llama.cpp was built without GPU support)
and aborts partway through this corpus:

    ggml/src/ggml-cpu/ops.cpp:5009: GGML_ASSERT(i01 >= 0 && i01 < ne01) failed

ollama serves the same all-MiniLM-L6-v2 weights on the Arc Pro B50 at
100% GPU, so it is both faster and does not crash. Same model, same 384
dimensions, so the numbers stay comparable to everything in FINDINGS.md.

The one constraint that matters: ollama's all-minilm has a 256-token context,
and it rejects anything longer outright --

    {"error":"the input length exceeds the context length"}

rather than truncating. Long paragraphs are therefore chunked and pooled, the
same treatment tools/embed_chunked.py applies: average the float vectors, then
renormalise. Averaging in float space is the correct combiner -- pooling after
quantisation would destroy the locality that makes the codes useful.
"""

import json
import sys
import time
import urllib.error
import urllib.request

import numpy as np

URL = "http://localhost:11434/api/embed"
MODEL = "all-minilm"

# 256 tokens of English prose is roughly 1000 characters, but tokenisation is
# not uniform: dialogue with contractions and archaic spelling runs shorter per
# character. 600 leaves headroom without fragmenting ordinary paragraphs.
MAX_CHARS = 600
BATCH = 64


def split_chunks(text, limit=MAX_CHARS):
    """Split at sentence boundaries where possible, else hard cuts."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return [text]
    chunks, cur = [], ""
    for piece in text.replace("! ", ". ").replace("? ", ". ").split(". "):
        piece = piece.strip()
        if not piece:
            continue
        cand = f"{cur}. {piece}" if cur else piece
        if len(cand) <= limit:
            cur = cand
        else:
            if cur:
                chunks.append(cur)
            while len(piece) > limit:      # single sentence over the limit
                chunks.append(piece[:limit])
                piece = piece[limit:]
            cur = piece
    if cur:
        chunks.append(cur)
    return chunks or [text[:limit]]


def _post(batch, retries=3):
    payload = json.dumps({"model": MODEL, "input": batch}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                URL, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as fh:
                return json.loads(fh.read())["embeddings"]
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            if "context length" in body:
                raise ValueError(f"chunk too long for {MODEL}: {body}")
            if attempt == retries - 1:
                raise RuntimeError(f"ollama HTTP {e.code}: {body}")
            time.sleep(1 + attempt)
        except urllib.error.URLError as e:
            if attempt == retries - 1:
                raise RuntimeError(f"ollama unreachable: {e}")
            time.sleep(1 + attempt)


def embed(texts, progress=True):
    """One L2-normalised 384-d vector per input, in input order."""
    # Flatten to chunks, remembering which paragraph each belongs to, so one
    # request covers many paragraphs regardless of how they split.
    flat, owner = [], []
    for i, t in enumerate(texts):
        for c in split_chunks(t):
            flat.append(c)
            owner.append(i)

    vecs = []
    t0 = time.time()
    for i in range(0, len(flat), BATCH):
        vecs.extend(_post(flat[i:i + BATCH]))
        if progress and (i // BATCH) % 20 == 0 and i:
            rate = (i + BATCH) / (time.time() - t0)
            print(f"  {i + BATCH}/{len(flat)} chunks  {rate:.0f}/s",
                  file=sys.stderr)
    V = np.asarray(vecs, dtype=np.float32)

    dim = V.shape[1]
    out = np.zeros((len(texts), dim), dtype=np.float32)
    counts = np.zeros(len(texts), dtype=np.float32)
    np.add.at(out, owner, V)
    np.add.at(counts, owner, 1.0)
    out /= np.maximum(counts, 1)[:, None]
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    return out


if __name__ == "__main__":
    sample = ["the cat sat on the mat", "x " * 2000]
    E = embed(sample, progress=False)
    print(f"{E.shape[0]} vectors, dim {E.shape[1]}, "
          f"norms {np.linalg.norm(E, axis=1).round(4)}")
