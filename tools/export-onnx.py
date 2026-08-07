#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
"""
Export a sentence-transformer to ONNX with mean pooling and L2 normalization
baked into the graph.

Pooling belongs inside the graph, not in the caller. RKNN runs a fixed graph
and returns raw tensors; if pooling lived outside it, every backend (llama.cpp,
RKNN, onnxruntime) would have to reimplement it identically or the LSH bits
would silently differ between them. Baking it in makes the model's contract
"text ids in, normalized sentence vector out" for everyone.

Shapes are fixed, not dynamic. RKNN compiles for a static shape, so sequence
length is a compile-time constant here. Padding shorter inputs wastes compute;
that is the price of NPU execution.
"""

import argparse
import os
import sys

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SEQLEN = 128


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    ap.add_argument("-o", "--out", default="/tmp/minilm-mean.onnx")
    ap.add_argument("-s", "--seqlen", type=int, default=DEFAULT_SEQLEN,
                    help=f"fixed sequence length (default {DEFAULT_SEQLEN})")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModel.from_pretrained(args.model)
    base.eval()

    class MeanPooled(nn.Module):
        """BERT + attention-masked mean pooling + L2 norm, as one graph."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            out = self.m(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids)
            tokens = out.last_hidden_state           # [B, S, H]
            # Mask padding before averaging: padded positions carry real
            # activations and would otherwise drag the mean toward nothing.
            mask = attention_mask.unsqueeze(-1).to(tokens.dtype)
            summed = (tokens * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            mean = summed / counts
            # L2 normalize: SimHash reads only sign(dot), so magnitude must
            # not influence which side of a hyperplane a vector lands on.
            return mean / mean.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)

    wrapped = MeanPooled(base).eval()

    S = args.seqlen
    dummy_ids = torch.ones(1, S, dtype=torch.int64)
    dummy_mask = torch.ones(1, S, dtype=torch.int64)
    dummy_tt = torch.zeros(1, S, dtype=torch.int64)

    with torch.no_grad():
        ref = wrapped(dummy_ids, dummy_mask, dummy_tt)
    print(f"output shape: {tuple(ref.shape)}  (norm {ref.norm().item():.6f})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.onnx.export(
        wrapped,
        (dummy_ids, dummy_mask, dummy_tt),
        args.out,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["embedding"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"wrote {args.out}  (fixed shape 1 x {S})")

    # Save the tokenizer next to it: the C side needs the same vocab, and a
    # mismatched tokenizer produces plausible-looking wrong embeddings.
    tokdir = os.path.splitext(args.out)[0] + "-tokenizer"
    tok.save_pretrained(tokdir)
    print(f"wrote {tokdir}")


if __name__ == "__main__":
    main()
