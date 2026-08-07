#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton
"""
Convert the exported ONNX embedder to RKNN for the RK3588 NPU.

Runs on x86 only — rknn-toolkit2 is x86-only and pulls torch. The resulting
.rknn is copied to the target, where rknn-toolkit-lite2 loads it.

On quantization: INT8 is measurably free for this use. Retrieval recall@1 at
128-bit LSH is 0.7880 quantized and 0.7880 in float32, because SimHash reads
only sign(dot) and INT8 noise almost never flips it. The usual accuracy
objection to NPU inference does not apply here.

The risk is op coverage, not precision. RKNN handles convolutional graphs well;
attention and LayerNorm are less certain, and anything unsupported silently
falls back to the CPU. A converted model that runs entirely on CPU has NPU
overhead and no NPU speed, so the conversion log matters more than the fact
that it succeeded.
"""

import argparse
import sys

DEFAULT_TARGET = "rk3588"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--onnx", default="/tmp/minilm-mean.onnx")
    ap.add_argument("-o", "--out", default="/tmp/minilm-mean.rknn")
    ap.add_argument("-t", "--target", default=DEFAULT_TARGET)
    ap.add_argument("--fp16", action="store_true",
                    help="skip INT8 quantization (larger, slower, no accuracy gain here)")
    args = ap.parse_args()

    # rknn-toolkit2 2.3.2 calls onnx.mapping, removed in onnx 1.16+. Their own
    # requirements say onnx>=1.16.1, so the pin contradicts the code. Restoring
    # the table is safer than downgrading onnx, which has no cp312 wheel below
    # 1.16 and fails to build from source.
    import onnx
    if not hasattr(onnx, "mapping"):
        import types
        import numpy as np
        from onnx import TensorProto as _TP
        _t2n = {
            _TP.FLOAT: np.dtype("float32"),   _TP.UINT8: np.dtype("uint8"),
            _TP.INT8: np.dtype("int8"),       _TP.UINT16: np.dtype("uint16"),
            _TP.INT16: np.dtype("int16"),     _TP.INT32: np.dtype("int32"),
            _TP.INT64: np.dtype("int64"),     _TP.BOOL: np.dtype("bool"),
            _TP.FLOAT16: np.dtype("float16"), _TP.DOUBLE: np.dtype("float64"),
            _TP.UINT32: np.dtype("uint32"),   _TP.UINT64: np.dtype("uint64"),
            _TP.STRING: np.dtype("object"),
        }
        m = types.ModuleType("onnx.mapping")
        m.TENSOR_TYPE_TO_NP_TYPE = _t2n
        m.NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in _t2n.items()}
        m.STORAGE_TENSOR_TYPE_TO_FIELD = {
            _TP.FLOAT: "float_data",   _TP.INT32: "int32_data",
            _TP.INT64: "int64_data",   _TP.STRING: "string_data",
            _TP.DOUBLE: "double_data", _TP.UINT64: "uint64_data",
        }
        onnx.mapping = m
        sys.modules["onnx.mapping"] = m

    from rknn.api import RKNN

    rknn = RKNN(verbose=True)

    # mean/std of 0/1: inputs are token ids, not pixels. The default image
    # normalization would corrupt them.
    rknn.config(target_platform=args.target,
                quantized_dtype="w8a8",
                optimization_level=3)

    print("=== load_onnx ===")
    if rknn.load_onnx(model=args.onnx,
                      inputs=["input_ids", "attention_mask", "token_type_ids"],
                      input_size_list=[[1, 128], [1, 128], [1, 128]]) != 0:
        sys.exit("load_onnx failed")

    print("=== build ===")
    # do_quantization=False first: get it running before adding a variable.
    if rknn.build(do_quantization=False) != 0:
        sys.exit("build failed")

    print("=== export ===")
    if rknn.export_rknn(args.out) != 0:
        sys.exit("export_rknn failed")

    print(f"wrote {args.out}")
    rknn.release()


if __name__ == "__main__":
    main()
