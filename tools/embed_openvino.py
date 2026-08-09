#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""Embed paragraphs with OpenVINO on an Intel GPU.

The fastest and most reliable of the three backends measured on an Arc Pro B50,
at 451 paragraphs/s against 188 for ollama/Vulkan and 96 for CPU. It is also
the only one that does not misbehave: llama.cpp garbles output above ~4B
parameters on Arc through both its SYCL and Vulkan backends, and ollama's
all-minilm rejects any input over 256 tokens outright rather than truncating.

Convert a model first:

    optimum-cli export openvino -m sentence-transformers/all-MiniLM-L6-v2 \\
        --task feature-extraction /mnt/bulk/models/openvino/minilm

Long paragraphs are chunked and pooled rather than truncated, matching
tools/embed_chunked.py: average the float vectors, then renormalise. Averaging
before quantisation is the correct order -- pooling packed codes would destroy
the locality that makes them useful.
"""

import os
import sys
import time

import numpy as np

MODEL_DIR = os.environ.get("SIGIL_OV_MODEL",
                           "/mnt/bulk/models/openvino/minilm")
TOKENIZER = os.environ.get("SIGIL_OV_TOKENIZER",
                           "sentence-transformers/all-MiniLM-L6-v2")
DEVICE = os.environ.get("SIGIL_OV_DEVICE", "GPU.1")

# MiniLM's trained position limit. Chunks are measured in tokens here rather
# than characters -- the character heuristic in embed_ollama.py exists only
# because the HTTP API gives no token count back.
MAX_TOKENS = 256
BATCH = 64


class Embedder:
    """Compiled model plus tokenizer. Compilation is slow; reuse the object."""

    def __init__(self, model_dir=MODEL_DIR, device=DEVICE, tokenizer=TOKENIZER):
        import openvino as ov
        from transformers import AutoTokenizer

        core = ov.Core()
        if device not in core.available_devices and device != "AUTO":
            avail = ", ".join(core.available_devices)
            raise RuntimeError(f"device {device} not available; have: {avail}")
        self.tok = AutoTokenizer.from_pretrained(tokenizer)
        self.model = core.compile_model(
            core.read_model(os.path.join(model_dir, "openvino_model.xml")),
            device)
        self.inputs = {i.any_name for i in self.model.inputs}
        self.device = device

    def _forward(self, texts):
        enc = self.tok(list(texts), padding=True, truncation=True,
                       max_length=MAX_TOKENS, return_tensors="np")
        feed = {k: v for k, v in enc.items() if k in self.inputs}
        hidden = list(self.model(feed).values())[0]
        # Mean pool over real tokens only; padding must not dilute the vector.
        mask = enc["attention_mask"][..., None].astype(np.float32)
        v = (hidden * mask).sum(1) / np.maximum(mask.sum(1), 1e-9)
        return v.astype(np.float32)

    def _chunk(self, text):
        """Split to <=MAX_TOKENS pieces, at token boundaries."""
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        # -2 leaves room for [CLS] and [SEP].
        span = MAX_TOKENS - 2
        if len(ids) <= span:
            return [text]
        return [self.tok.decode(ids[i:i + span])
                for i in range(0, len(ids), span)]

    def embed(self, texts, progress=False):
        """One L2-normalised vector per input, in input order."""
        flat, owner = [], []
        for i, t in enumerate(texts):
            for c in self._chunk(" ".join(t.split())):
                flat.append(c)
                owner.append(i)

        out = []
        t0 = time.time()
        for i in range(0, len(flat), BATCH):
            out.append(self._forward(flat[i:i + BATCH]))
            if progress and i and (i // BATCH) % 50 == 0:
                print(f"  {i}/{len(flat)} chunks  "
                      f"{i / (time.time() - t0):.0f}/s", file=sys.stderr)
        V = np.vstack(out)

        pooled = np.zeros((len(texts), V.shape[1]), dtype=np.float32)
        counts = np.zeros(len(texts), dtype=np.float32)
        np.add.at(pooled, owner, V)
        np.add.at(counts, owner, 1.0)
        pooled /= np.maximum(counts, 1)[:, None]
        pooled /= np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12
        return pooled


_default = None


def embed(texts, progress=False):
    """Module-level convenience wrapper over a cached Embedder."""
    global _default
    if _default is None:
        _default = Embedder()
    return _default.embed(texts, progress=progress)


if __name__ == "__main__":
    e = Embedder()
    E = e.embed(["the cat sat on the mat", "x " * 3000])
    print(f"device {e.device}: {E.shape[0]} vectors, dim {E.shape[1]}, "
          f"norms {np.linalg.norm(E, axis=1).round(4)}")
