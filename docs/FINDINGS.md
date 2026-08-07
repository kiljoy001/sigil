# Measured findings

What was tried, what the numbers were, and what not to retry. Negative results
are the majority here and the more useful half — they are the ones nobody
writes down and everybody re-derives.

Every number is from `test/bench`, `test/bench_mt`, `test/eval`, or
`bench/scan_opencl.c` on the machines listed at the bottom.

---

## Embedding model selection

The LSH bits come from a sentence embedding projected to 128 bits by SimHash.
Which model produces the best bits is not the same question as which model
tops MTEB.

Recall@1 on Quora Duplicate Questions, 1000 human-labeled pairs, 2000
documents. "Ceiling" is float32 cosine on the uncompressed vector — the best
any compression of that model could do.

| model | dim | ceiling | 128-bit LSH | % of ceiling |
|---|---|---|---|---|
| **all-MiniLM-L6-v2** | 384 | 0.8150 | **0.7882** | **96.7%** |
| bge-base-en-v1.5 | 768 | 0.8090 | 0.7755 | 95.9% |
| nomic-embed-text-v1.5 | 768 | 0.8075 | 0.7683 | 95.1% |
| bitnet-embedding-0.6b | 1024 | 0.8015 | 0.7597 | 94.8% |
| BitNet-b1.58-2B-4T | 2560 | 0.5885 | — | — |

MiniLM wins, and it is the smallest and fastest of them. Three things this
established:

**Task-trained beats large.** BitNet-2B-4T is 90x MiniLM's size and scores 23
points *worse*, because it is an instruct model, not an embedder. Last-token
pooling on a decoder produces a next-token state, not a semantic summary.

**MTEB rank does not predict post-SimHash quality.** bge-base and nomic both
outrank MiniLM on MTEB and both lose here.

**Anisotropy is what hurts.** The predictor of compression loss is not
dimensionality (the first theory, wrong) but mean |cosine| between unrelated
documents:

| model | mean abs cos | % of ceiling |
|---|---|---|
| MiniLM-384 | 0.075 | 96.7% |
| bge-base-768 | 0.438 | 95.9% |
| bitnet-1024 | 0.461 | 94.8% |
| nomic-768 | 0.506 | 95.1% |

MiniLM's embeddings are nearly orthogonal; the others crowd into a narrow
cone. SimHash counts hyperplane crossings, so it resolves angles — and there
is little angle to resolve inside a cone. Isotropy is cheap to measure (300
documents and a cosine matrix) and is a better model-selection criterion here
than a benchmark score.

## Projections: SimHash is already near-optimal

| projection | 5-fold held-out recall@1, 128 bits |
|---|---|
| SimHash | 0.9182 ± 0.0244 |
| PCA | 0.9230 ± 0.0228 |
| ITQ | 0.9225 ± 0.0270 |

ITQ measured **+2.5 points in-sample** and collapsed to **+0.0043 ± 0.0087
held out**, with one fold negative. The gain is a fifth of the fold-to-fold
variance. PCA and ITQ are statistically indistinguishable from each other and
from SimHash.

Both are also *learned from the corpus*, which would make stored bits depend
on training data — re-fit the rotation and every existing sigil is invalid.
SimHash's hyperplanes come from a published seed and anyone can reproduce
them. Not worth trading that for noise.

**Do not retry ITQ or PCA on this workload.**

## Matryoshka truncation: no effect

`nomic-embed-text-v1.5` is trained so vector prefixes are valid embeddings.
Truncating before SimHash was expected to help, given the dimensionality
theory (which was wrong anyway).

| truncated to | 128-bit recall@1 | % of ceiling |
|---|---|---|
| 64 | 0.6980 | 87.9% |
| 128 | 0.7033 | 87.4% |
| 256 | 0.7068 | 87.7% |
| 512 | 0.7197 | 89.3% |
| 768 (full) | 0.7203 | 89.0% |

Flat. Truncation neither helps nor hurts, and nomic loses to MiniLM at every
width.

## Whitening hurts; mean-centering helps

Full ZCA whitening was the isotropy theory taken seriously. It achieved
isotropy (mean |cos| 0.44 → 0.04) and made every model worse:

| model | delta |
|---|---|
| MiniLM-384 | −0.046 |
| bge-base-768 | −0.081 |
| nomic-768 | −0.060 |
| bitnet-1024 | −0.094 |

An α-sweep on bge shows monotonic destruction — whitening amplifies
low-variance eigendirections that are mostly noise, and SimHash then spends
bits encoding it:

```
raw               0.8915
mean-center only  0.9120   <- the entire win
whiten a=0.25     0.9040
whiten a=0.50     0.8780
whiten a=1.00     0.8110
```

**Mean-centering alone** is worth +3 to +5 points on anisotropic models, 5/5
folds positive:

| model | raw | centered | delta |
|---|---|---|---|
| MiniLM-384 | 0.9232 | 0.9268 | +0.004 (4/5) |
| bge-base-768 | 0.8845 | 0.9158 | +0.031 (5/5) |
| nomic-768 | 0.8623 | 0.9097 | +0.047 (5/5) |
| bitnet-1024 | 0.8735 | 0.9125 | +0.039 (5/5) |

Anisotropic models sit in a cone offset from the origin, and SimHash's
hyperplanes pass *through* the origin — so most of them have every document on
one side and contribute no information. Centring moves the cone onto the
origin so all 128 hyperplanes cut through it. It is about the cone's
*position*, not its width.

Irrelevant for MiniLM (+0.004, noise), mandatory if the model ever changes.

## Corpus choice changes conclusions

An earlier version of the evaluation used paraphrase pairs written by hand.
That measures how separable the author made the examples:

| | hand-written | Quora |
|---|---|---|
| 32-bit recall@1 | 0.18 | 0.55 |
| 256-bit recall@1 | 0.69 | 0.80 |

The invented corpus was far *harder* than reality and would have driven the
design toward many more bits than it needs. Use a standard corpus.

---

## Scan performance

### SIMD

| ISA | width | speedup over scalar | notes |
|---|---|---|---|
| AVX2 | 256-bit | 2.2x | two records per iteration |
| SSE4.2 | 128-bit | **2.08x** | best relative gain |
| NEON | 128-bit | 1.48x | simplest code, slowest |

NEON has `vcntq_u8`, a real per-byte popcount, where AVX2 and SSE emulate one
with a nibble table and `pshufb` — three instructions against six. It is still
the slowest, because AVX2 is twice as wide and aarch64 has no `movemask`, so
lane-selecting kernels round-trip through memory instead of extracting a
bitmask.

That last penalty is severe: the first NEON timerange kernel ran **65% slower
than scalar** (18.08 vs 10.87 ms) until a one-instruction `vmaxvq` early-out
was added to skip empty blocks. Now 4.66 ms.

Unrolling the NEON similarity loop to four independent chains was tried and
came out slower (19.77 vs 18.09 ms). The loop is memory-bound; extra in-flight
work only costs registers. **Do not retry.**

### Threading: pool beats static split everywhere

Similarity scan, 10M records, via `sigil_scan_*_range()`.

| machine | 1 thread | static split | work pool | pool speedup |
|---|---|---|---|---|
| Xeon E5645 (2 sockets, 12 cores) | 49.11 ms | 9.45 ms | **7.11 ms** | **6.91x** |
| i5-12600K (P+E cores) | 12.43 ms | 2.70 ms | **2.15 ms** | **5.77x** |
| RK3588 (A76+A55) | 36.27 ms | 12.55 ms | **7.88 ms** | **4.61x** |

Every one of these machines has heterogeneous cores — two sockets with
interleaved memory, P-cores and E-cores, or big.LITTLE. A static split gives
each thread an equal share, so the scan waits on the slowest core. The pool
lets fast cores take more chunks. On the RK3588 that is the difference between
2.89x and 4.61x.

Both the i5 and the Xeon *regress* from 8 to 16 threads under static split
while the pool keeps improving.

Speedup stops where the memory controllers saturate, not where cores run out:
the i5 tops out near 74 GB/s, the Xeon 22.5, the Pi 20.3.

### GPU: wins, but only resident

| device, 100M records | ms | GB/s |
|---|---|---|
| **Arc Pro B50, resident** | **7.22** | **222** |
| Arc Pro B50, incl. upload | 315 | 5 |
| i5-12600K, 16 threads | 21.03 | 76 |

**2.9x faster than the best CPU configuration** — but the upload is 43x the
kernel time, so a one-shot scan loses badly. Bandwidth is flat from 10M to
100M (220 vs 222 GB/s), so the device is genuinely saturated.

The scan looked like the wrong shape for a GPU (0.06 ops/byte) and is the
right shape for a different reason: perfect parallelism, no divergence, and
OpenCL has `popcount()` as a builtin so there is no nibble-table emulation.
What the GPU contributes is GDDR6 bandwidth, roughly 3x dual-channel DDR5.

On a machine with no usable GPU, **Intel's OpenCL CPU runtime beat a
hand-written pthread pool on the same hardware** — 6.52 vs 7.11 ms on the
Xeon. The same kernel is a portable fallback, not only a GPU path.

A GeForce 210 (2009) enumerated no OpenCL platforms: the userspace library is
present but the legacy 340.x kernel module is gone from modern kernels. Its
~8 GB/s would have lost to the host's own DRAM regardless.

---

## Accelerators for embedding

Embedding, not scanning. Different workload, different answer.

### Concurrency is the variable that mattered

Single-stream measurements were misleading on every device:

| device | 1 stream | best concurrent | |
|---|---|---|---|
| i5-12600K CPU | 764 docs/s (16 threads) | **1253** (8 proc x 2 threads) | |
| Arc Pro B50 | 185 docs/s | **1050** (16 concurrent) | 5.7x |
| RK3588 NPU | 62 docs/s (1 core) | **175** (6 runtimes / 3 cores) | 2.8x |

A 22M-parameter model cannot fill any accelerator with one request. Many small
independent streams beat one wide one — on the CPU, 8 processes of 2 threads
beat 1 process of 16 threads by 64%.

RKNN's `NPU_CORE_0_1_2` splits *one* inference across cores and gives +12%.
Three *independent* runtimes, one pinned per core, gives 2.4x. Different
mechanisms; the flag name suggests otherwise.

### What combines and what does not

| combination | result |
|---|---|
| RK3588 NPU + CPU | 162 + 112 = **275 docs/s** (2.1x CPU alone) |
| i5 GPU + CPU | 1056 docs/s, *worse* than CPU alone at 1253 |

The NPU does its own compute and only needs the CPU for submission, so they
stack. The GPU processes need CPU threads to feed them and compete with the
CPU workers instead.

### INT8 is free

| LSH bits | fp32 | INT8 per-row | INT4 per-row |
|---|---|---|---|
| 64 | 0.7290 | 0.7278 | 0.7115 |
| 128 | **0.7880** | **0.7880** | 0.7837 |
| 256 | 0.8002 | 0.8018 | 0.7965 |

SimHash reads only `sign(dot)`, and quantization noise at 1/127 resolution
almost never flips it. Cosine similarity to the fp32 vector is 0.99997. NPU
inference is therefore quality-neutral, and even INT4 costs only 0.4 points.

### The Mali GPU is not usable via llama.cpp

`ggml_opencl: unsupported GPU 'Mali-G610 r0p0'` — the OpenCL backend is
Adreno-only, not merely Adreno-tuned. `get_adreno_gpu_gen()`,
`libadreno-opencl-kernels.so`, Adreno-specific structs throughout. Adding Mali
means writing a kernel set, not flipping a flag.

The stack itself is fine: `clinfo` enumerates the G610 at OpenCL 3.0, 4
compute units, 31 GB unified memory, using Rockchip's `libmali-valhall-g610-
g24p0-gbm.so` from `JeffyCN/mirrors` (branch `libmali`). Note the *scan*
kernel in `bench/scan_opencl.c` would run there; only llama.cpp refuses.

### RKNN conversion: three vendor bugs

Documented in `tools/convert-rknn.py`. All three block conversion outright:

1. transformers 5.x is incompatible with the torch 2.4 that RKNN pins
   (`DTensor` import). Pin transformers to 4.44.2.
2. setuptools >= 81 removed `pkg_resources`, which rknn-toolkit2 imports. Pin
   setuptools < 81.
3. rknn-toolkit2 2.3.2 calls `onnx.mapping`, removed in onnx 1.16+ — while its
   own requirements say `onnx>=1.16.1`. The pin contradicts the code. No cp312
   wheel exists below 1.16, so the lookup table is shimmed in the converter.

The converted model is numerically correct: `cos(cat,feline)` = 0.5563 on the
NPU against 0.5564 on CPU.

Fixed 128-token padding is inherent to RKNN's static shapes. Quora questions
average ~15 tokens, so the NPU does roughly 8x more work per document than it
needs to. A 32-token variant is untested and would plausibly help a lot.

---

## Machines

| name | CPU | SIMD | accelerator |
|---|---|---|---|
| workstation | i5-12600K, 16 threads | AVX2 | Arc Pro B50, UHD 770 |
| xeon | 2x Xeon E5645, 12 cores | SSE4.2 only | GeForce 210 (no driver) |
| orangepi5-plus | RK3588, A76+A55 | NEON | RKNPU v2 (3 cores), Mali G610 |

The Xeon is the reason the runtime SIMD dispatch exists: it has no AVX2, and
an earlier build probed for AVX2, ignored the result, and crashed there with
SIGILL.
