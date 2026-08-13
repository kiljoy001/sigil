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

## Mathematics: embeddings are inverted, ASTs work

The question was whether a semantic filesystem can compare mathematical
content. It cannot, by embedding — and the failure is worse than weakness.

### The probe

8 pairs that are mathematically *equivalent* but differ in notation, and 8 that
share notation but assert *opposite* things. A working similarity measure puts
the first group close and the second far.

| model | equivalent | confusable | separation |
|---|---|---|---|
| MiniLM-L6-v2 (contrastive) | 32.4 | 17.6 | **-14.8** |
| SciBERT | 25.9 | 10.6 | **-15.2** |
| MathBERT (tbs17) | 42.5 | 22.6 | **-19.9** |
| MathBERTa (witiko) | 13.5 | 3.0 | **-10.5** |

Hamming distance out of 128. **Every model is inverted**: contradictions score
as *more* similar than equivalences. A similarity search over math would
confidently retrieve the negation of the query.

The individual cases are stark. `∀ε>0 ∃δ>0` vs `∃ε>0 ∀δ>0` — quantifiers
swapped, meaning inverted — came out **6 bits apart**, the closest pair in the
set. `A ⊆ B` vs `B ⊆ A`: 11 bits. `O(n log n)` vs `O(n²)`: 12 bits. Meanwhile
genuinely equivalent phrasings of the same complexity bound sat at 57 bits,
near random.

### Why

Three of the four are masked-language models (`BertForMaskedLM`,
`RobertaForMaskedLM`, pipeline tag `fill-mask`), not embedders. Nothing in an
MLM objective asks that similar sentences land near each other — that is what
contrastive training does, and it is why Sentence-BERT exists. Mean-pooling an
MLM gives a vector nobody optimized for cosine similarity.

Domain training made it *worse*, not better. MathBERTa scored cos 0.995 between
contradictory statements: it has learned that mathematical text looks alike so
thoroughly it can barely distinguish anything.

Reading the source papers confirms a scope mismatch rather than a bug.
`tbs17/MathBERT` is built for mathematics *education* — knowledge-component
prediction, auto-grading student answers, knowledge tracing — on a corpus of
pre-K-to-college curriculum plus arXiv abstracts. UL-BERT (Cheng et al. 2021)
classifies formulas into five buckets: Derivative, Integral, Series, Matrix,
Others. At that granularity `O(n log n)` and `O(n²)` are both "Others". Its own
error analysis reports the model "may pay too much attention to the subscripts"
— the same surface-form sensitivity, acknowledged.

The one promising model, Peng et al. (arXiv 2105.00377), trains on Operator
Trees — and it works precisely to the extent that it stops being distributional
and starts parsing structure. It appears unreleased.

### What does work: canonicalize and hash

`tools/math_ast.py`. LaTeXML converts LaTeX to Content MathML, which is a real
semantic AST — `<apply><plus/><ci>x</ci><ci>y</ci></apply>` — then the tree is
canonicalized (commutative operators sorted, unicode normalized) and hashed.

**16/16 on the same probe.** `x+y` and `y+x` collide by construction; `a<b` and
`a>b`, `∫_a^b` and `∫_b^a`, `A ⊆ B` and `B ⊆ A` all stay distinct.

Measured on 300 real math.stackexchange expressions:

- **99.3% parse coverage** (2 failures, both malformed `\displaystyle`)
- **~20 ms/expression batched**, against 631 ms spawning `latexmlmath` per
  call — a 31x speedup, and the batched path was verified to produce identical
  hashes
- 103 of 190 unique hashes appeared more than once, i.e. it genuinely finds
  duplicate expressions across different posts

The coverage figure was the surprise; the estimate beforehand was 30-50%.
LaTeXML is what arXiv itself uses to render papers, so it handles real-world
macros that SymPy's parser does not.

### The inversion is in the embedding, not the hash

Worth locating precisely, because the obvious suspect is wrong. SimHash and
Hamming distance are not at fault: the raw float cosine is *already* inverted
before any hashing happens, under every pooling strategy tried.

| pooling | cosine separation | Hamming separation |
|---|---|---|
| mean | -0.116 | -13.2 |
| last | -0.154 | -15.6 |
| max | -0.046 | -7.5 |
| cls | +0.004 | -0.8 |
| concat first+last | -0.087 | -5.5 |
| position-weighted | -0.102 | -11.1 |

Hamming separation tracks cosine separation across all six, which is what an
unbiased estimator should do. The compression is faithful; it is faithfully
reporting an inversion it did not create. (`cls` is the least inverted only
because it collapses everything to ~0.91 and distinguishes nothing.)

### The mechanism: the embedding is substantially bag-of-tokens

Comparing token-multiset overlap (Jaccard over the tokenizer's output) against
cosine explains the whole result:

| | mean Jaccard | mean cosine |
|---|---|---|
| equivalent pairs | 0.252 | 0.743 |
| confusable pairs | **0.757** | **0.859** |

**Correlation between token overlap and cosine: +0.601.**

The two starkest cases have Jaccard exactly 1.000 — identical token multisets,
pure permutations:

- `∀ε>0 ∃δ>0` vs `∃ε>0 ∀δ>0` → cosine 0.971
- `A ⊆ B` vs `B ⊆ A` → cosine 0.970

A permutation reads as the same thing. Mean pooling is a commutative operation,
and while attention is supposed to inject order through positional encodings,
for short expressions built from the same symbols the pooled vector comes out
nearly unchanged. The model has little reason to encode which quantifier binds
first, because that distinction rarely mattered in its training data.

So the probe was guaranteed to invert: equivalent pairs were written with
*different notation* (low overlap by construction) and confusable pairs with
*the same notation* (high overlap). A model tracking token overlap must get
this backwards. It is doing what it was trained to do.

### This generalizes past mathematics

Any domain where meaning turns on order, or on a single negating token, will
fail the same way:

- legal text — "shall" vs "shall not", party order in an obligation
- code — `>` vs `>=`, argument order, `if` vs `unless`
- clinical text — dosage direction, "with" vs "without" contrast

**Cheap pre-flight test for a new domain:** compute Jaccard token overlap and
cosine over a sample of pairs and correlate them. A high correlation means the
embedding is largely doing lexical matching, and any semantic claim over that
corpus is unsafe. No labels needed.

### Consequence for the design

Mathematics gets an **exact-identity channel, not a similarity one**. An
expression hash is a graph edge: "these two paragraphs contain the same
formula." No threshold, no embedding, no possibility of a plausible-looking
wrong answer.

Paragraphs that are mostly math should not produce an LSH sigil at all — the
vector would be built from connective tissue like "where" and "such that",
which is noise wearing the shape of signal. Strip math before embedding, hash
it separately.

The general lesson beyond math: **build a probe with equivalent and confusable
pairs before trusting a similarity measure in a new domain.** Positive
separation is not the default, and inversion is invisible without the second
group.

---

## Two-stage retrieval: what can act as the discriminator

Given that contradictions score as similar (above), the natural fix is a second
stage: scan cheaply, then verify the candidates. That only works if stage two
can tell the two apart. Two candidates were tested.

### A cheap lexical trigger does not work

The idea: flag pairs that are *both* close in Hamming distance *and* high in
token overlap, since that is the measured signature of the failure.

On constructed pairs it looked excellent — 11/13 contradictions caught, zero
false alarms. On 400 human-labeled Quora duplicates it collapsed:

| Jaccard threshold | contradictions caught | false alarms on real duplicates |
|---|---|---|
| 0.55 | 7/10 | **38.2%** |
| 0.65 | 6/10 | 23.2% |
| 0.75 | 1/10 | 12.8% |
| 0.95 | 1/10 | 0.0% |

The distributions overlap almost completely. Contradiction Jaccards run
0.50-1.00; genuine duplicates have median 0.50, p90 0.778, p99 0.917. Most
contradictions sit between 0.6 and 0.75 — exactly where the bulk of real
paraphrases sit. No threshold separates them.

The constructed prose pairs had mean Jaccard 0.066 because they were written to
be lexically distinct. Real paraphrases average 0.510: people restate things
using mostly the same words. **This is the same error as the hand-written
corpus earlier — examples invented to demonstrate a hypothesis will
demonstrate it.**

### A weak LLM does work, but the prompt dominates

Llama-3.2-3B-Instruct via llama-server, judging the same pairs:

| prompt | contradictions | duplicates | balanced accuracy |
|---|---|---|---|
| "same thing or contradict?" | 10/10 | 0/15 | 50.0% |
| "does B contradict A?" | 0/10 | 15/15 | 50.0% |
| **4-shot with examples** | **10/10** | **11/15** | **86.7%** |

The first two are degenerate: they answer one way regardless of input. Both
score 100% on one class, which looks like success until the other class is
checked. Four labelled examples turn it into a real classifier.

Validated on a larger sample: 46/60 real duplicates correctly called
PARAPHRASE (76.7%), 30/30 unrelated pairs correctly rejected (100%), at ~350 ms
per call on CPU.

**Always measure both error classes.** A discriminator scoring 10/10 on
contradictions was not reading the input at all.

### Cost

The second stage is affordable because the scan does the reduction first. A
similarity scan over 10M records returns on the order of 100 candidates, so the
LLM runs on ~0.001% of the corpus. At 350 ms per pair that is 35 seconds,
against a scan measured in milliseconds. Total cost stays dominated by the
scan.

This improves **precision** on retrieved candidates. It cannot improve
**recall**: genuinely equivalent paragraphs with different vocabulary (Jaccard
~0.25) are missed by the scan and never reach stage two. That is the harder
problem and none of this touches it.

### NPU note

Generative inference on the RK3588 NPU needs **RKLLM**, a separate stack from
the RKNN used for the embedder — different runtime, different conversion
toolchain, its own model zoo. It supports RK3588 and small instruct models
(Qwen3.5, SmolLM3, Gemma4, MiniCPM4), and its server demo speaks an
OpenAI-compatible API, so the classifier can stay backend-agnostic. Untested
here; the CPU path above is what was measured.

---

## Paragraph retrieval on real papers

6648 paragraphs from 223 arXiv papers, embedded individually. Ground truth is
weak but real: two paragraphs from the same paper are related.

**The core premise holds.** 128-bit LSH picks a same-paper nearest neighbour
47.3% of the time against a 0.43% random baseline — 110x chance.

**And it is semantic, not lexical**, which matters given the math result:

| | token overlap | cosine |
|---|---|---|
| same-paper | 0.125 | **0.367** |
| different-paper | 0.090 | 0.084 |

Token overlap barely separates them while cosine separates them 4.4x, and the
Jaccard/cosine correlation is +0.184 against +0.601 on the math probe. On real
prose the embedding is doing semantic work. The math failure was domain
specific, not a general property of the pipeline.

### Retrieval against peer-reviewed judgement

The strongest evidence in the project, because it is the only measurement
against an *external* human standard rather than an internal property.

Ground truth without annotation: two paragraphs citing the same work were
written by domain experts who each independently decided that work was relevant
there, and reviewers agreed. Those judgements already exist in the literature —
no labelling budget, no reinforcement loop, no scale problem.

5000 citation contexts citing 951 distinct works, from unarXive. Citation
markers stripped so "[16]" cannot act as a lexical giveaway. Chance is 0.09%.

| method | R@1 | R@5 | vs chance |
|---|---|---|---|
| random | 0.0009 | 0.0045 | 1x |
| **float32 cosine** | **0.2754** | **0.5362** | **305x** |
| 512-bit LSH | 0.2420 | 0.4794 | 268x |
| 256-bit LSH | 0.2030 | 0.4320 | 225x |
| 128-bit LSH | 0.1578 | 0.3524 | 175x |

R@5 = 0.54 at the ceiling: given a passage, more than half the time one of its
five nearest neighbours engages with the same prior work — written by different
authors, in different papers, who never coordinated.

It is a hard test. Citation contexts are terse ("following [16]") while the
passages they parallel are detailed and differently worded.

**This is a lower bound on precision.** Citations are sparse and biased —
authors cite a handful of relevant works, chosen partly by visibility and field
norms — so a non-citation is not evidence of non-relatedness. Many apparent
misses are genuinely related passages that were simply not co-cited.

### Bit width is corpus dependent

This was not expected, and it corrects an earlier decision.

| corpus | 128 bits | 256 | 512 | 1024 |
|---|---|---|---|---|
| Quora questions | **95.6%** | 98.3% | 98.8% | 99.9% |
| arXiv paragraphs | **76.8%** | 89.6% | 94.5% | 97.0% |
| citation contexts | **57.3%** | 73.7% | 87.9% | — |

(percent of that corpus's own float32 ceiling)

Three independent corpora, same pattern. Citation contexts are the hardest —
dense academic prose where the distinctions are fine — and 128 bits retains
barely half of what the embedding knows.

Quora saturates at 128. arXiv needs 512 to reach the same place. The mechanism
is discrimination difficulty: Quora questions span unrelated topics, so a
coarse code separates them, while paragraphs from 223 papers in adjacent fields
are genuinely similar and distinguishing them needs finer resolution.

Marginal gains on arXiv: 128->256 buys 7.7 points, and everything from 256 to
1536 combined buys 5.5. The single doubling is worth more than the next three.

**Consequence:** `SIGIL_LSH_BITS` belongs in the store schema next to
`model_id` and `simhash_seed`, not fixed in the header. The 128-bit choice was
set from Quora alone and the header comment claiming "the remaining headroom is
in the model, not the width" is true only for that corpus. 256 is the better
default: 89.6% on the hard corpus, 98.3% on the easy one, 32 bytes per record.

### Summarize-then-compare: right for labels, not for retrieval

A summary is reusable (O(N) calls rather than O(N^2) pairwise), inspectable by
a human, and doubles as the cluster label the classifier needs. The question
was whether summarizing is itself a lossy step that reintroduces the failures
above.

**Summaries preserve the distinctions that matter.** All six contradiction
pairs stayed distinct after summarization — negation survived ("should not",
"not positive"), direction survived (rose/fell), quantities survived (30 vs 90
days). The feared collapse did not happen.

**But summarizing does not improve retrieval.** The hypothesis was that
stripping boilerplate (author lists, arXiv headers, affiliations) would sharpen
similarity:

| | cosine separation | Hamming separation |
|---|---|---|
| originals | +0.554 | +24.0 bits |
| summaries | +0.535 | +23.0 bits |

Unchanged, within noise. The embedder was already handling the boilerplate.
Summarizing costs ~700 ms per paragraph and adds a hallucination surface for no
retrieval gain.

**One real hazard.** The model turned "the tenant is responsible for repairs"
into "the tenant is *not* responsible for repairs" — a hallucinated negation on
exactly the axis that matters. Cluster labels get read by humans, which is the
mitigation; summaries should not feed an automated decision unreviewed.

**Consequence:** the classifier summarizes clusters *after* the scan groups
them, rather than summarizing everything up front. Cheaper and it puts a human
in front of the error-prone step.

### Paragraph length: chunk, do not truncate

Long paragraphs exceed the model's context. Truncating discards content
silently, which is the failure this project keeps rediscovering. Instead
`tools/embed_chunked.py` splits at sentence boundaries, embeds each chunk, and
averages the vectors before renormalizing.

Averaging in float space is the correct combiner. Hashing the codes together
would destroy the locality that makes them useful (one flipped input bit
changes half the output), and a per-bit majority vote quantizes before
averaging rather than after.

### Compression reorders the head of the list; it rarely loses answers

R@1 is the harshest possible measure and matches no real interface. Fraction of
the float32 ceiling retained, by how many results a user sees:

| | R@1 | R@10 | R@20 | R@100 |
|---|---|---|---|---|
| 128-bit | 57.3% | 71.4% | 75.4% | 87.6% |
| 256-bit | 73.7% | 83.1% | 85.8% | 93.2% |
| 512-bit | 87.9% | 92.3% | 93.3% | 97.2% |

Rank displacement shows the mechanism. Of the 1377 queries float32 answered at
rank 1:

| | median new rank | stays rank 1 | still top-10 |
|---|---|---|---|
| 256-bit | 1 | 63.4% | 91.4% |
| 512-bit | 1 | 76.2% | 97.9% |

Median displacement is zero at 256 bits. Compression shuffles the top of the
list rather than discarding correct results.

### Similarity-only clustering does not work

Connected components at a similarity threshold — the method this design
originally specified — has no usable operating point on real prose:

| threshold | co-cited together | largest cluster |
|---|---|---|
| cos >= 0.60 | 0.389 | **2018** of 5000 |
| cos >= 0.70 | 0.080 | 182 |
| cos >= 0.75 | 0.028 | 29 |

Loose thresholds chain A-B-C until half the corpus is one blob; tight ones
fragment so badly that co-cited passages are separated 92% of the time. Mutual
k-NN fails identically (k=5: together 0.413, purity 0.002, largest 2472).

**This is at float32.** It is not a compression problem and no bit width fixes
it. Academic prose forms a continuum rather than islands, and every
transitive-linkage method chases that continuum across the corpus.

### Asserted edges cluster; mixing them with similarity is destructive

Citations are sparse and stated, so they do not chain:

| scheme | together | purity | F1 | largest |
|---|---|---|---|---|
| **citation-only (30% observed)** | **0.607** | 1.000 | **0.755** | 6 |
| union with sim >= 0.70 | 0.718 | **0.009** | 0.018 | **1316** |
| union with sim >= 0.75 | 0.671 | 0.049 | 0.092 | 509 |

Against similarity-only clustering, which could not exceed F1 ~0.15 at any
threshold. But *adding* similarity to citation edges collapses purity from
1.000 to 0.009 — it bridges between groups that should stay separate.

**The two signals measure different things.** A citation points at a specific
claim: "this sentence rests on that work." An embedding summarizes a whole
passage. Two paragraphs citing the same work may do so for opposite reasons —
one adopting a method, one disputing a result — and the surrounding prose
reflects those different purposes. They are orthogonal, not noisy versions of
one signal, which is why unioning them is destructive rather than merely
imprecise.

Extracting *which claim* a citation supports would be the ideal signal and is
very hard to automate: citation scope is ambiguous, several citations may share
a sentence, function varies (supporting, contrasting, acknowledging), and the
relevant claim can be sentences away through anaphora. The *existence* of the
edge is unambiguous and free, and that alone produced F1 0.755. Take the free
part.

Caveat on these numbers: purity 1.000 is an artifact of building edges from the
ground-truth labels, so only the `together` coverage figure is meaningful. A
real corpus has extraction errors and citations to works outside it.

**Consequence:** clusters come from asserted edges — citations, imports,
reply-chains, symlinks, directory structure — not from similarity. Similarity
serves neighbourhood queries, where it is validated at 305x chance, and stays
out of partitioning, where it is measurably harmful. Corpora with no asserted
edges may have no cluster view at all; that is untested.

### Document representation: averaging is enough

For the corpus-wide "potentially related" tier, a paper needs one
representation. Measured by retrieving the right paper from half its own
paragraphs (223 arXiv papers, chance 0.0045):

| representation | recall@1 | cost |
|---|---|---|
| doc-average | 0.7713 | 1 sigil per paper |
| max-paragraph | 0.7937 | full paragraph scan, ~30x |

2.2 points for 30x the work. The concern that averaging blurs a paper toward a
field-wide centroid did not materialize — mean cosine between different papers
under averaging is 0.199, so they stay well separated.

This validates the `para = 0` document sigil: the coarse tier scans one record
per paper.

### A llama.cpp gotcha worth recording

`llama-embedding` reading from **stdin concatenates every line into one
document** and returns a single vector. With `-f file` it splits on newlines as
documented. This silently produced 6650 vectors for 6648 paragraphs and would
have corrupted every number here. Always assert the returned count.

---

## An indexing crash, and five wrong diagnoses before the right one

Indexing a 3,157-file subtree segfaulted reproducibly at around 130 seconds.
The same embedder, over the same paragraphs, ran 218,356 of them clean in a
standalone driver. Finding the cause took five wrong theories, and the
sequence is more useful than the answer.

**Wrong: the thread stack.** libthread gives each proc 32 KB and OpenVINO's
inference path is deep. Raising it to 8 MB moved the crash later and did not
remove it.

**Wrong: `proccreate`.** Serving inline on the main thread instead of a
libthread proc changed nothing.

**Wrong: my batching.** Reverting to the per-text path crashed identically,
which means the bug predates the batch work entirely.

**Wrong: a tokenizer over-read.** Valgrind reported `Invalid read of size 16`
inside `SpecialTokensSplit::evaluate`, 44 bytes into a 52-byte block. Real,
and 997 of them -- but the process **never crashed under valgrind**. A
vectorised string routine reading a full 16-byte word past a short buffer is
ordinary in optimised SIMD code. It was noise.

**Wrong: libthread at all.** Building without it -- a plain `main()` and
lib9p's `srv()`, which needs no libthread; only `tpost.c` does -- still
crashed.

**Wrong, and worth recording as a warning: the GPU driver.** A core dump put
the faulting address next to `/dev/dri/renderD129` mappings -- the Arc under
the `xe` driver -- and that looked conclusive. It was not. Two later runs
disproved it. Restricting the process to the Arc alone with
`ZE_AFFINITY_MASK=0`, so the i915 iGPU never initialised, still crashed
(`i915 maps: 0`, still SIGSEGV). And a fresh core showed the neighbouring
mapping was not a GPU buffer at all but `openvino_tokenizer.bin`.

The GPU reading came from re-reading a **stale core file**. `ulimit -c` is 0
in a fresh shell, so the recent crashes wrote no cores and gdb kept opening
one from five hours earlier. Check the timestamp on a core before trusting
what it says.

**Right: the tokenizer, which valgrind had said all along.**

Across two cores with different ASLR bases the faulting address differed --
`0x7706c6b5f978` and `0x7565c8cad978` -- but the low bits matched exactly
(`...978`, with frame #2 at `...14f`), and both sat **1,672 bytes below the
mapping of `openvino_tokenizer.bin`**. A constant offset from a relocated
mapping is a computed address, not corruption.

The decisive test removed the tokenizer from the process entirely: the same
GPU, the same `xe` driver, the same batch size, the same 218,356 paragraphs
that reliably kill the server, with token IDs supplied by the caller and
`libopenvino_tokenizers.so` never loaded. **218,356 inferences, no crash.**

So valgrind was right at step four and it was dismissed too readily. It
reported `Invalid read of size 16` inside `SpecialTokensSplit::evaluate`,
44 bytes into a 52-byte block, 997 times. The reasoning for dismissing it --
"the process never crashed under valgrind, so it must be a benign vectorised
over-read" -- was wrong: valgrind replaces the allocator, so the read lands
in its padding instead of an unmapped page. Under the normal allocator, at
scale, it eventually crosses a page boundary.

Two lessons, and the second cost more than the first. Read the core dump
early -- four of the wrong theories came from reasoning about what *could*
fail rather than looking at where the process died. And a tool that reports
a definite memory error is not refuted by the program failing to crash under
that tool; the tool changes the conditions that turn the error fatal.

---

## Persistence was quadratic, and profiling found it where reading did not

`store_commit` rewrites the whole table, issuing five `tab_set` calls per
record. Measured:

| records | commit |
|---|---|
| 5,000 | 0.10 s |
| 10,000 | 0.31 s |
| 40,000 | 8.90 s |
| 80,000 | **37.74 s** |

Four times the cost for twice the rows — textbook O(n²). Extrapolated to a 60M
paragraph corpus that is about **68 days for one commit**, which makes every
argument about embedding throughput irrelevant.

Reading the code suggested two culprits and **both were wrong**:

- `tab_rowmap_rehash()` took an `Ndbtuple*` and found its owning `TabRow` by
  walking `t->rows[]`, although every caller had that row one line earlier.
  Fixed: 37.74 → 22.03 s. Still quadratic.
- `tab_add_row()` rediscovered a deduplicated row the same way, after
  `already_present()` had just matched it and thrown it away. Sigil keys every
  paragraph of a file on the same path, so all but the first hit that path.
  Fixed: no measurable change.

`perf` then put **85% of samples inside `tab_rowmap_rehash_row`** — a function
already patched, whose remaining loops walk a bucket chain and look innocent.
They were innocent. The chains were not:

```c
ensure_buckets(Tab *t, int target_rows)
{
    if(t->buckets != nil)
        return 0;            /* <- never grows */
```

A table built by `tab_create` starts empty, so `target_rows` is 0 and it
allocates `HashMinBuckets` = **16 buckets, forever**. At 160k rows that is
~10,000 entries per bucket, and every bucket walk becomes a full scan with a
`canonical_bytes()` malloc per step. `struct Tab` already carried an unused
`nbuckets_target` field for exactly this.

Growing the array past one entry per bucket:

| records | before | after | |
|---|---|---|---|
| 40,000 | 8.90 s | 0.148 s | 60x |
| 80,000 | 37.74 s | 0.278 s | 136x |
| 160,000 | 126.71 s | 0.549 s | **231x** |
| 320,000 | — | 1.107 s | linear |

60M records extrapolates to ~3.5 minutes. Verified lossless: 320k in memory,
320k rows on disk, 320k distinct hashes, 320k restored on reload.

The lesson is the process, not the bug. Two plausible O(n) scans were found by
reading, fixed, and moved the curve by 1.7x combined; the actual cause was a
missing resize that no amount of staring at the hot function would reveal,
because the hot function was correct. **Profile before optimising, even when
the code visibly contains the thing you are looking for.**

---

## One SYCL kernel versus four hand-written ones

The maintenance case for oneAPI is real: AVX2, SSE4.2, NEON and scalar are four
sources for one idea -- popcount the XOR of a 128-bit field, compare to a
threshold -- and each needs its own differential test, because a wrong SIMD
kernel does not crash, it returns a plausible distance and stays wrong.

Written once in SYCL (`src/scan_sycl.cpp`), 10M records, threshold 50:

| kernel | threads | time |
|---|---|---|
| **SYCL, host USM** | all | **2.33 ms** |
| hand SIMD + work pool | 16 | 2.15 ms |
| hand-written AVX2 | 1 | 9.31 ms |
| SYCL, device USM + copy | all | 20.75 ms |
| scalar C | 1 | 20.82 ms |

Results are bit-identical to the scalar reference, so the portability claim
holds: one source, correct on CPU and GPU.

**The performance claim holds too, but only with host USM.** The first version
of this benchmark called `malloc_device` and `memcpy` unconditionally, which on
CPU copies memory to itself -- 20.75 ms, indistinguishable from scalar, and it
looked like the compiler had failed to vectorise. It had not. Removing the copy
gives 2.33 ms: SYCL vectorises the popcount *and* spreads it across all cores,
matching a hand-written AVX2 kernel driven by a work pool to within 8%, from a
quarter of the code and with no threading logic at all.

The lesson is about the benchmark, not the toolchain. A measurement that makes
a mature compiler look like it emits scalar code is more likely to be wrong
than the compiler.

On the GPU it remains the PCIe transfer -- 18.55 ms, the same wall the earlier
OpenCL kernel hit at 160 MB of LSH per scan. That is a data-residency problem,
not a kernel problem.

Worth keeping in proportion: at ~2.2 ms for 10M records the scan is at memory
bandwidth by either route. Embedding the corpus is ~30 hours. Effort spent on
the scan is optimising the wrong end by four orders of magnitude.

---

## Project Gutenberg: retrieval against library cataloguing

A fourth corpus, and the first at a scale where the store stops being a
formality: ~900 paragraphs per book across 61,458 English texts projects to
**~55M paragraphs, a 3.5 GB store**.

Gutenberg has no citation graph, so the external judgement is the **Library of
Congress subject headings** in the catalogue. A cataloguer decided two books
are about the same thing, without reference to any embedding — the property
that made the unarXive result credible.

### The catalogue's date field is unusable

`Issued` ranges 1971–2026. It is the **digitisation date**, not publication —
archive bookkeeping. Populating the record's timestamp from it would have been
silently wrong.

Author death year is the honest substitute: it bounds composition from above,
covers **84.6%** of paragraphs, and spans **1321–1968**. A proxy, not a fact.

### How coarse the subject relation has to be

LCSH headings are `--`-delimited facets. Matching depth was measured on 210
books, not chosen:

| depth | scoreable queries | random-pair match |
|---|---|---|
| **1** | **99.4%** | **2.26%** |
| 2 | 6.9% | 1.05% |
| 3 | 6.7% | 1.02% |
| exact | 5.3% | 1.02% |

Past one facet, headings are so specific that almost no two books share one and
93% of queries become unscoreable, leaving a metric computed on an
unrepresentative remainder. Depth 1 is loose — "United States" joins a
Revolutionary War history to a travel guide — but chance is 2.26%, so there is
real headroom above it.

### Results

20,000 paragraphs sampled across 210 books, 1052 scoreable queries:

| method | R@1 | R@5 | R@10 | R@20 |
|---|---|---|---|---|
| float32 | 0.199 | 0.369 | 0.469 | 0.587 |
| 512-bit | 0.182 | 0.355 | 0.471 | 0.590 |
| 256-bit | 0.183 | 0.345 | 0.447 | 0.557 |
| 128-bit | 0.130 | 0.295 | 0.416 | 0.542 |

R@1 of 0.199 against 0.0226 chance is **8.8x**. Retained fraction of the
float32 ceiling — 65.6% / 92.3% / 91.4% at R@1 for 128 / 256 / 512 bits —
reproduces the bit-width finding on a third independent corpus, landing between
Quora and citation contexts as dense literary prose should.

### The control matters more than the result

Same-author rate among neighbours, R@1: **0.091** against a subject-match rate
of 0.199. Subject matching runs ~2x style matching, so the embedder is not
merely recognising prose voice. Without this control the headline number would
not be evidence of anything.

### Literature and non-fiction are different tasks

R@10, float32: **literature 0.347, non-fiction 0.635.**

Two reasons, both predicted before measuring. Most paragraphs of a novel are
dialogue, scene-setting or transition, and are not "about" the book's subject
headings in any useful sense. And the literature classes divide by *national
tradition* rather than subject — PS is American, PR English, PQ Romance — so
two sea novels land in different classes by the author's nationality.

Averaging the two into one number would have hidden this entirely.

### Classification: judged edges, then a consolidated label

2000 pairs judged with the extract-then-compare prompt, stratified across the
whole Hamming range:

| | separation |
|---|---|
| split by the judge's verdict | **+14.7 bits** |
| split by the catalogue label | +9.0 bits |

451 confirmed related (22.6%), and confirmation decays monotonically:

| Hamming | n | confirmed |
|---|---|---|
| 0–60 | 27 | 55.6% |
| 60–70 | 242 | 38.8% |
| 70–80 | 525 | 30.5% |
| 80–90 | 498 | 21.7% |
| 90–100 | 244 | 18.0% |
| 100–110 | 97 | 5.2% |
| 110–120 | 139 | 10.1% |
| 120+ | 228 | 4.8% |

Distance predicts the verdict, and it predicts a *read* judgement better than
it predicts shelving. Catalogue/judge agreement is 76.2%.

**Pass one's vocabulary is unusable on its own.** 451 edges produced 551
distinct subject strings, 383 appearing exactly once. `war`, `battle`,
`combat`, `conflict`, `victory` and `expedition` are six groups that should be
one, and no amount of counting merges them — they share no substring.

**Pass two consolidates in one call.** Given the top 60 subjects it returned:

```
THEME: Military & Conflict | war, battle, combat, conflict, victory, expedition, danger, death
THEME: Religion & Philosophy | religion, god, philosophy, church, consciousness, truth
THEME: Emotions & Human Experience | love, loneliness, fear, grief, happiness, compassion
THEME: Literature & Poetry | poetry, literature, tennyson, malory, plato
```

Overlapping membership falls out without being engineered: `trees`, `forest`,
`danger` and `expedition` each land in two themes, and `combat` in both
Military and Crime. That is the intended behaviour — a passage is about several
things at once, and forcing a partition is what broke threshold clustering.

Cost is the point. Pass one is one call per pair; pass two is **one call per
batch of 60 subjects**, a few hundred tokens. The layer doing the most
valuable reasoning is nearly free.

Two defects at this stage: the model returned ten themes against a stated limit
of five, and used `THEME: INCOHERENT | diggingory` as a junk drawer —
hallucinating `diggingory` from `diggory`, a character name that should never
have survived pass one as a subject.

### Two OpenVINO pipelines on one GPU corrupt each other

A batch judging run and a diagnostic script were both given `GPU.1`. The batch
run's output turned to noise:

```
'ofi建华 +=\n``Slfapsedでしょうね...'
'SetColor life，天才.SelectedItems。初喜...'
```

Mixed CJK, code fragments, no relation to the input — indistinguishable from
the `<pad>` garbage llama.cpp produces on this card, and easy to misread as the
same hardware fault. It is not. With the second process killed, the identical
prompt returned a clean `UNRELATED` reproducibly, and `The capital of France
is` returned `Paris.`

The 100 pairs judged during the overlap were discarded. Nothing warns about
this: both processes compile and run, and only the output is wrong.

**Serialise GPU access.** One OpenVINO pipeline per device at a time, and check
for a running job before starting anything that touches the GPU — including a
"quick" diagnostic.

### Prompting the judge: four attempts, and what each one broke

The judge prompt went through four versions before it produced usable output.
Recorded because the failures were not obvious in advance and two of them look
like successes if you check only one thing.

**1. Zero-shot yes/no.** `SAME` or `DIFFERENT`. Answered DIFFERENT for five of
six hand-written probes — coherent and reading, but the threshold sat in the
wrong place.

**2. Few-shot yes/no.** Four hand-written example pairs. Scored 6/6 on probes,
then collapsed to **89.4% UNRELATED** on real paragraphs. The examples were
one-sentence fragments where the positives were near-paraphrases and the
negatives wildly disjoint, so the demonstrated boundary was "almost the same
sentence" versus "nothing in common". Real Victorian prose falls in between.

**3. Ask for shared subjects** (the user's formulation): *"compare these two
blocks of text, return a list of subjects these blocks share, if no relation
can be determined return UNRELATED"*. Immediately better — naming the subjects
forces a reason rather than a default. But it accepted `["chestnut"]`, `face`
and `he (the male subject)` as shared subjects. Those are shared *words*, and
SimHash is most sensitive to exactly that, so counting them measures the LSH
against itself.

**4. Adding explicit prohibitions broke it again.** A `Do NOT list:` block
naming the failure modes pushed the model into blanket refusal — the Hastings
pair that had produced `Norman conquest, Harold, William` came back
`UNRELATED`. Negative instructions re-created the conservative collapse of
version 1.

**What works:** define the subject positively and filter afterwards.

```
Return a list of subjects the two blocks share. A subject is what a passage
is ABOUT -- a topic, event, place, field, activity, or theme that a librarian
would use to catalogue it, not merely a word that appears in both.
```

with a post-filter dropping generic categories and bare short tokens. The
Hastings pair now returns `duke william, battle of hastings, norman conquest`,
and the two cookbook paragraphs correctly return nothing.

The general lesson, consistent with `bench/llm_judge.py`: tell the model what
you want, not what you forbid, and enforce the rest in code where it cannot
suppress the signal along with the noise.

### The catalogue is the weaker judge, and asking for subjects shows it

The rerank result below was measured with a yes/no prompt. Replacing it with
*"compare these two blocks of text, return a list of subjects these blocks
share, and if no relation can be determined return UNRELATED"* changed the
behaviour completely — and revealed that the disagreements are mostly the
ground truth's fault, not the model's.

Two paragraphs on the Battle of Hastings, one from a volume of narrative
history and one from a travel book about the roads out of Hastings:

> **judge:** Duke William, Battle, England, Norman conquest, Harold
> **catalogue:** miss — filed `world history` and `cathedrals`/`england`

Two English cookbooks, both filed `cooking, english`:

> "Zante currants are not a currant really, but a small kind of grape"
> "You may make white curran wine the same way, only leave out the rasps"
> **judge:** UNRELATED
> **catalogue:** hit

In both directions the judge is the more accurate of the two. LCSH headings
describe *where a book sits on a shelf*; the question being asked is whether
*these two paragraphs* are about the same thing. A judge that answers the
second question precisely gets marked wrong by a label that only answers the
first.

Two consequences. Asking for the shared subjects rather than a verdict forces
the model to commit to a reason instead of defaulting to no — which is what the
89.4% UNRELATED below really was. And the rerank R@k figure cannot settle
whether the judge helps, because it grades a paragraph-level judgement against
a book-level label.

### LLM reranking does not pay for itself here

The escalation architecture — cheap wide scan, expensive narrow judge — applied
to Gutenberg with a 7B judge (Qwen2.5-7B-Instruct, INT8, OpenVINO on the Arc).
60 queries, top 15 candidates each, 895 judge calls at 519 ms:

| | R@1 | R@5 | R@10 |
|---|---|---|---|
| 256-bit scan | 0.133 | 0.267 | 0.350 |
| + LLM rerank | 0.150 | 0.267 | 0.350 |
| delta | +0.017 | +0.000 | +0.000 |

Essentially nothing, for ~7.8 s of GPU time per query.

The verdict distribution says why: **89.4% UNRELATED**, a hair under the 90%
degeneracy threshold the harness warns at. The same model and prompt scored
6/6 on both error classes against hand-written probe pairs, then collapsed to
near-constant on real text. The few-shot examples were crisp topical pairs;
real Gutenberg paragraphs are dialogue, scene-setting and transition, where
"related subject" is genuinely ambiguous and the model defaults to no.

This is the same effect the corpus already showed from the other direction —
literature retrieves at 0.37 against non-fiction's 0.63 because most paragraphs
of a novel are not *about* the novel's subject. A judge asked whether two such
paragraphs share a subject is being asked a question the text does not answer.

Worth noting what would have hidden this: measuring only the flattering error
class. A judge answering UNRELATED to everything cannot reorder anything, so
its R@k *equals* the scan's — which reads as "no harm done" rather than "the
model is not discriminating." The distribution check is what caught it.

Not a refutation of escalation in general; the citation corpus has sharper
labels and the judge may earn its cost there. But on per-paragraph prose with
per-book labels, the cheap tier is already doing the work.

### Practical notes

**The same text ships up to five times** — several encodings plus superseded
revisions under `old/`. Indexing all of them fills the store with paragraphs
that are identical by construction and inflates every similarity score. One
file per `Text#`, preferring UTF-8.

**Sample across books, not by truncation.** The extract is ordered by
`text_id`, so a head-slice of 20,000 rows drew from **10 books** and left 86% of
queries unscoreable. Round-robin across books gave 210 books and 70% scoreable.
Third time this project has been bitten by an unrepresentative sample.

**Embedding is the bottleneck, by three orders of magnitude.** ollama serving
all-MiniLM on the Arc Pro B50 sustains **188 paragraphs/s**, so the full corpus
is **~81 hours** of GPU time. The scan over the resulting 3.5 GB store is
~12 ms. Any argument about scan performance is irrelevant next to this.

**Correction: llama.cpp on Arc is fixed, via its own OpenVINO backend.**
Upstream merged `ggml/src/ggml-openvino` in April 2026, which translates GGML
graphs to OpenVINO at runtime. Built with `-DGGML_OPENVINO=ON`, the same GGUF
files that garbled through SYCL and Vulkan produce correct output on the Arc:

```
$ GGML_OPENVINO_DEVICE=GPU.1 llama-simple -m Llama-3.2-3B-Q4_K_M.gguf \
      -n 14 "The capital of France is"
The capital of France is Paris, which is also the largest city in France.
```

A 4.4 GB model answers correctly too -- the size class that returned
`"C" that they have` under Vulkan. 3B benchmarks at 36.78 t/s on `GPU.1`.

Two build notes. If oneAPI's `setvars.sh` has been sourced, it puts
`/opt/intel/oneapi/compiler/*/include/CL/` ahead of `/usr/include`, and its
older `cl_ext.h` lacks constants the newer `opencl.hpp` references -- configure
with `env -u CPATH -u INCLUDE`. The error reads as a broken OpenCL install and
is not. And `llama-cli` ignores `-no-cnv` in this build and drops into an
interactive session; use `llama-simple` or `llama-bench` for scripted runs.

So the finding below is about the *SYCL and Vulkan* backends specifically, not
llama.cpp as a whole. It stands as the reason the standalone OpenVINO embedder
exists, and that backend is still the faster path for embedding, but a llama.cpp
built this way is no longer wrong on this hardware.

**Intel Arc: llama.cpp's SYCL and Vulkan backends are the wrong runtimes.**

On an Arc Pro B50 (Battlemage), llama.cpp produces coherent output only up to
about 4B parameters, then degrades into noise:

| model | size | output for "The capital of France is" |
|---|---|---|
| llama3.2:1b | 1B | `Paris` |
| phi3:mini | 3.8B | `Paris` |
| qwen2.5-coder:7b | 7B | `"C" that they have` |
| gemma3:12b | 12B | `Kijkademadóizevp` |

This is not a configuration problem, and three explanations were wrong before
the right one: it is not batch size, not concurrency, and not memory pressure
(it persists at 10 GB loaded with a 4096 context on a 16 GB card). It is a
known llama.cpp defect on Arc, present on **both** backends —
[#20169](https://github.com/ggml-org/llama.cpp/issues/20169) for SYCL, closed
*not planned*, and [#24560](https://github.com/ggml-org/llama.cpp/issues/24560)
/ [#21888](https://github.com/ggml-org/llama.cpp/issues/21888) for Vulkan.
Rebuilding the local SYCL tree would have reproduced the same failure through a
different path.

OpenVINO 2026.3 is a `pip install` away, enumerates the card directly as
`GPU.1`, and needs no oneAPI environment at all:

```
GPU.1: Intel(R) Arc(TM) Pro B50 Graphics (dGPU)
```

MiniLM converted to OpenVINO IR, 64 paragraphs padded to 256 tokens:

| backend | rate | 55M paragraphs |
|---|---|---|
| ollama / Vulkan | 188/s | 81 h |
| **OpenVINO / Arc** | **451/s** | **34 h** |
| OpenVINO / CPU | 96/s | 159 h |

2.4x ollama on the same hardware and the same weights, with correct output
(384 dims, unit norm). Dynamic padding on real paragraphs should improve on
this further.

**`llama-embedding` could not do the job here.** Built without GPU support it
runs on the CPU, and it aborts partway through this corpus:

```
ggml/src/ggml-cpu/ops.cpp:5009: GGML_ASSERT(i01 >= 0 && i01 < ne01) failed
```

ollama serves the same weights on the GPU without it. Separately, ollama's
`all-minilm` has a **256-token context and rejects longer input outright**
(`{"error":"the input length exceeds the context length"}`) rather than
truncating — so long paragraphs must be chunked and pooled, as
`tools/embed_chunked.py` already does. Three wrong explanations were tried
first (batch size, then concurrency, then load); the server's own error message
settled it immediately. Ask the server before theorising.

---

## Spreadsheets: the identity channel transfers, and better

The math AST work established that anything with a parseable grammar can have
an exact identity channel alongside probabilistic prose similarity. Excel is
the obvious next candidate, and the Enron corpus — ~15,900 real workbooks
released through litigation — is the test.

`tools/xlsx_ast.py` tokenizes a formula, canonicalizes it (absolute markers
normalized, case folded, commutative functions sorted) and hashes the tree.
Measured on 2998 workbooks, 4.7 million formulas:

| | LaTeX via LaTeXML | Excel |
|---|---|---|
| parse coverage | 99.3% | **99.9999%** |
| speed | ~20 ms/expression | **0.05 ms/formula** |
| failures | ~2 in 300 | **6 in 4,713,827** |

400x faster and effectively complete. (9,700 further cells contain a bare `=`
with nothing after it — broken cells, not formulas, and excluded from the
denominator.)

Getting there took three corrections, all gaps in the tokenizer rather than in
Excel:

1. **66%** — named ranges (`CBalancingVol`) were unrecognized
2. **99.66%** — sheet-qualified error refs (`'ASSUM 1'!#REF!`)
3. **99.9999%** — a greedy character class made `=#REF!/0.75` swallow the `/0`;
   fixed by anchoring to the known error literals

Predicting from "the grammar is closed" was right about where this lands and
wrong about every intermediate step. That is the fourth inference this project
has needed a measurement to correct.

### Duplicated logic is real and abundant

From 2998 workbooks: **510,457 distinct computations appear in more than one
workbook**, out of 1.24 million distinct hashes.

The obvious objection — that these are the same file saved repeatedly — does
not hold: a content-hash check over 300 files found zero exact duplicates.
These are different workbooks sharing computations.

The most-shared entries reveal something more useful than copy-paste:

```
732 workbooks  =NOW()
279 workbooks  ='[1]Data Sheet'!B11
279 workbooks  ='[1]Data Sheet'!C11
```

`'[1]Data Sheet'!` is an **external workbook reference**. 279 files pull from
one source: that is not duplication, it is a dependency graph. Change that
sheet and 279 workbooks move with it.

"Here is what breaks if you touch this" is a stronger claim than "here are your
duplicates", and it needs no similarity at all — the edge is explicit in the
formula text.

### The label layer: similarity works there too

Formulas get exact identity; labels and headers are the other layer, and this
is where similarity has to earn its place — on text that is terse, abbreviated,
and often not sentences at all ("Q3 Rev", "MMBTU/D", "Ttl"). Given that MiniLM
did badly on short notation-heavy strings in the math tests, this could have
failed.

`tools/xlsx_text.py` extracts label groups rather than single cells: a header
row or a label column read together, since "Q3" alone carries almost nothing.
Median group length across a 120-workbook sample was 68 characters — genuinely
embeddable.

Ground truth is workbook authorship. Enron filenames carry the employee whose
mailbox the file came from, and one person's spreadsheets share subject matter.
Weak but external and free, the same reasoning as same-paper retrieval.

174 workbooks, 25 owners:

| method | R@1 | R@5 | vs chance |
|---|---|---|---|
| random | 0.0361 | 0.1681 | 1x |
| float32 | 0.6494 | 0.7529 | 18x |
| 128-bit | 0.5690 | 0.7184 | 16x (87.6% of ceiling) |
| 256-bit | 0.5575 | 0.6839 | 15x (85.8% of ceiling) |

The absolute recall is higher than the citation benchmark partly because 25
owners is a coarser target than 951 works. The more comparable figure is
compression retention, and at 87.6% for 128 bits it beats academic paragraphs
(76.8%) — spreadsheet labels are apparently easier to tell apart than dense
technical prose.

So all three layers are validated on the same corpus: formulas hash exactly,
labels embed usefully, and cell references are explicit.

### Why spreadsheets suit this better than papers

Three layers, each matched to a mechanism the project already has:

| layer | mechanism | measured |
|---|---|---|
| formulas | AST hash | 99.9999% coverage, 6 failures in 4.7M |
| labels, headers | LSH | R@1 0.65, 18x chance |
| cell references | graph edges | explicit in syntax, 279 workbooks on one source |

The third is the one academic prose lacked. Citations gave clustering its
skeleton (F1 0.755 against ~0.15 for similarity alone), and cell references are
the same kind of signal — stated, sparse, non-chaining — but far denser and
extractable without ambiguity. Where a citation's *scope* is hard to automate,
a formula's precedents are explicit in the syntax.

---

## The indexing crash was never a race

Indexing Gutenberg segfaulted intermittently for weeks: same corpus, same
binary, sometimes 64 seconds in, sometimes 450, sometimes not at all. Every
signature said race -- and that intuition, while reasonable, cost most of the
investigation, because the actual source of nondeterminism was ASLR.

**Root cause**: Project Gutenberg files carry Windows-1252 bytes (0x92 curly
apostrophe and friends) inside files served as UTF-8. Those bytes reached
openvino_tokenizers' `SpecialTokensSplit`, which runs PCRE2 compiled in UTF-8
mode without `PCRE2_MATCH_INVALID_UTF`. PCRE2 documents matching invalid UTF
as undefined behaviour; concretely, the sljit-JIT'd matcher decodes a garbage
codepoint and indexes a character-class table with it:

    => movzwq (%rcx,%rax,2),%rcx        ; table lookup, garbage index
       movabs $0x7ffff5ba6fa1,%r9
       movzbq (%rcx,%r9,1),%rcx         ; wild read -- SIGSEGV here

Whether that wild read faults or silently returns garbage depends on what
happens to be mapped at the target address -- i.e. on mmap layout. Hence:

| condition | crash rate |
|---|---|
| ASLR on, GPU.1 | 8/12 |
| ASLR on, CPU | 12/12 (each at 80 s) |
| ASLR off | 8/8 -- fully deterministic |
| ASLR off, invalid UTF-8 stripped from input | 0/4 |
| ASLR off, fix applied | 0/8 |
| ASLR on, fix applied | 0/24 |

The "passing" 40% of runs were not passing. They were reading unmapped-
adjacent memory into the tokenization and continuing.

**The fix** is at the embedder boundary (`src/embed_openvino.cpp`,
`to_valid_utf8`): bytes that are not valid UTF-8 are transcoded as
CP1252/Latin-1, so 0x92 becomes a real U+2019 the embedding can use.
Identity is untouched -- BLAKE3 still hashes the original bytes.

### What the investigation disproved along the way

Each of these was tested directly and eliminated: libthread's scheduler
(no-libthread build crashed identically), the GPU driver and dual-GPU setup
(CPU device crashed too), the pip-wheel OpenVINO build (native archive
crashed at the same rate), plan9port's runtime (a pure-POSIX gcc driver
crashed at the same rate), thread-count races (taskset -c 0 changed
nothing), and stack sizes (2/8/32 MB identical).

### Methodology lessons, learned the expensive way

1. **A nondeterministic bug makes single trials worthless.** Half a day of
   bisection produced confident conclusions -- "it's the UTF-8", "no, retest
   says it isn't" -- that were coin flips. With a ~60% crash rate, only
   rates over N >= 12 trials distinguish signal from luck
   (`test/crashrate.sh`). The one exception: a *crash* on the first trial is
   real evidence; a *pass* is not.
2. **`setarch -R` belongs early in any intermittent-crash checklist.** One
   ASLR-off run converted a heisenbug into a deterministic one and ended the
   guessing within minutes. "Intermittent" does not always mean "race" --
   layout randomization is the other big source, and it is the one you can
   switch off.
3. **Valgrind named the culprit on day one** -- 997x `Invalid read of size
   16` in `SpecialTokensSplit::evaluate` -- and was dismissed because there
   was "no crash under valgrind". Valgrind's allocator absorbs over-reads in
   padding; absence of a crash under valgrind is not absence of the bug it
   is reporting.
4. **rip in an anonymous rwxp mapping means JIT code**, and regex engines
   JIT too. The frame `0x989681` that looked like a corrupted return address
   was sljit's non-standard frame layout confusing the unwinder.

Upstream: openvino_tokenizers should pass `PCRE2_MATCH_INVALID_UTF` (or
validate input). Observed in the 2026.3.0 release (wheel and archive alike);
not yet reported upstream.

---

## Committing 74.9M records needs 54 GB to write 12 GB

The first full-corpus run embedded **74,905,358 paragraphs across all 79,133
books** in about 13 hours. Then `commit` ran for five more hours, reached
**54 GB resident**, and had still written zero bytes when the machine was
rebooted. Every record was lost.

Nothing was wrong. That is the shape of the code:

```c
t = tab_create(...);
for (i = 0; i < n; i++) {          /* 74.9M iterations */
    r = tab_add_row(t, "path", p);  /* row held live */
    tab_set(t, r, "para", buf);     /* six cells, each a string */
    ...
}
tab_commit(t);                      /* only here does anything reach disk */
```

Peak memory is every row, its six cell strings, and the bucket index —
roughly 4.5x the file it produces — and it scales with corpus size, so the
machine's RAM sets the corpus ceiling.

### What the measurements ruled out

The instinct is to blame swap or I/O. Both were wrong, and `/proc` counters
settled it in a way `iotop` could not — it stalled on launch, `gdb` could
not attach because ptrace is restricted, and the process was unresponsive
to `/stats` because sigilfs serves one client at a time.

| reading | value | conclusion |
|---|---|---|
| `stat` utime+stime, 60 s apart | +92,322 ticks | computing hard, not stuck |
| `vmstat` pswpin/pswpout, 60 s apart | +1 page, +0 | no paging at all |
| `io` write_bytes | 0 after 5 h | never reached serialisation |
| `io` rchar, two samples | identical | corpus reading long finished |
| `MemAvailable` | 18.6 GB, falling ~30 MB/min | allocating steadily |

So: purely allocator-bound. Two reads of a `/proc` file a minute apart
distinguished "working" from "hung" and "swapping" from "not" when none of
the usual tools would run.

### The insert rate is linear; the memory is not

libtab's bucket-growth fix (above) holds at this scale — measured over a
16x range on a quiet machine:

| rows | time | rate |
|---|---|---|
| 50,000 | 0.7 s | 67,142/s |
| 200,000 | 3.0 s | 66,496/s |
| 800,000 | 12.3 s | 65,124/s |

Linear. The O(n²) commit that would have made this impossible is genuinely
gone. What remains is that the rate collapses at 74.9M rows anyway, because
each row holds ~700 bytes live and the allocator is working against 54 GB
of residency. A benchmark on small tables does not predict that, and my
19-minute estimate from those numbers was wrong by hours.

### The fix: stream the write, keep only hashes

Flush each row's bytes once it is finalised and free its cell strings,
retaining only a hash of the key for deduplication:

- 74.9M x 16 B (128-bit hash) ≈ **1.2 GB** of index
- versus 54 GB today, for the same ~12 GB of output

A 128-bit hash rather than 64-bit because at 74.9M keys the birthday bound
gives a ~0.02% chance of at least one 64-bit collision, and today's dedup
is exact. 1.2 GB is still nothing against 54.

Streaming is only about *when bytes reach disk*. It need not change the
in-memory API: a row can still be revised through `tab_set` until the
caller moves to the next one, which is the only revision any bulk writer
performs. The two costs that do remain are that a row already flushed
cannot be revised, and that a failure mid-write leaves a partial file
unless it writes to a temp path and renames at commit.

Filed as libtab#2. It is additive — py-libtab and go-libtab are unaffected
until someone binds it.

### Why this only appeared now

The manifest exercise wrote 77,883 rows in 4.9 s and round-tripped
perfectly. The pipeline tests use 331 files. Nothing below a few million
rows shows this at all: the peak is invisible when the table fits
comfortably, and the ratio only becomes fatal when it does not. Running the
real corpus is the only thing that would have found it.

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
