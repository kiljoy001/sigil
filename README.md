# sigil

A content-addressed semantic filesystem where **the identifier carries its own index**.

An ordinary content address is a hash: opaque, uniformly distributed by design, and
therefore useful only for exact lookup. Asking "what is similar to this?" or "what
changed last March?" means building a second structure alongside it — an index that
has to be maintained on every write, kept consistent, and rebuilt when it corrupts.

A sigil puts identity and queryable structure in the same 64-byte record. Similarity
becomes Hamming distance over a packed field, time filtering an integer compare. Both
run as a linear scan at memory bandwidth, and there is no index to maintain, so writes
are O(1).

```
[ 0..16)  LSH semantic bits   128 bits, SimHash over a paragraph embedding
[16..48)  BLAKE3              256 bits, content identity
[48..52)  para index          0 = document-level, 1..n = paragraphs
[52..56)  Unix timestamp
[56..58)  category code       user-defined meaning
[58..60)  packed trits        6 balanced-ternary hints
[60..64)  cluster ref
```

A `_Static_assert` pins the size, and it has already caught one silent padding
change: an earlier layout with a 20-byte SHA-1 needed the LSH array first, because
putting the hash first forced four bytes of interior padding and pushed the record to
56. BLAKE3's 32 bytes align naturally, so the ordering no longer matters — but the
assert stays, since the on-disk format depends on it.

**Status: a working library and a body of measurements, not yet an application.**
There is no persistence, no indexer, and no server — see [Status](#status).

## What is measured

Everything below is reproducible from this repository. The full record, including the
things that did not work, is in [`docs/FINDINGS.md`](docs/FINDINGS.md); the decisions
they support are in [`docs/DESIGN.md`](docs/DESIGN.md).

### Retrieval works, against external human judgement

The strongest result, because the ground truth comes from outside the pipeline. Two
paragraphs citing the same work were written by domain experts who each independently
decided that work was relevant there, and reviewers agreed. 5000 citation contexts
citing 951 distinct works, from unarXive, citation markers stripped so `[16]` cannot
act as a lexical giveaway.

| method | R@1 | R@5 | vs chance |
|---|---|---|---|
| random | 0.0009 | 0.0045 | 1x |
| float32 cosine | 0.2754 | 0.5362 | **305x** |
| 512-bit LSH | 0.2420 | 0.4794 | 268x |
| 256-bit LSH | 0.2030 | 0.4320 | 225x |
| 128-bit LSH | 0.1578 | 0.3524 | 175x |

R@5 of 0.54 at the ceiling: given a passage, more than half the time one of its five
nearest neighbours engages with the same prior work — written by different authors, in
different papers, who never coordinated.

This is a *lower* bound. Citations are sparse and biased, so a non-citation is not
evidence of non-relatedness; many apparent misses are related passages that were
simply not co-cited.

### Exact identity where a grammar exists

Similarity is the wrong tool for anything with parseable structure. Two expressions
that canonicalize to the same tree *are* the same expression — a fact, not an estimate.

| domain | tool | coverage | speed |
|---|---|---|---|
| LaTeX math | `tools/math_ast.py` (LaTeXML) | 99.3% | ~20 ms |
| Excel formulas | `tools/xlsx_ast.py` | **99.9999%** | **0.05 ms** |

The Excel figure is 6 failures in 4,713,827 real formulas from 2998 Enron workbooks.

This matters because embeddings *invert* on mathematics: across four models,
contradictory statements scored consistently **more** similar than equivalent ones.
`∀ε∃δ` and `∃ε∀δ` landed 6 bits apart out of 128. The mechanism is that the embedding
is substantially bag-of-tokens — correlation between token overlap and cosine is
+0.601 — so pairs that share vocabulary while asserting the opposite score as similar.
Operator trees do not have that failure, and score 16/16 on the same probe.

### Scan throughput

10M records, one core:

| machine | ISA | scalar | SIMD | speedup |
|---|---|---|---|---|
| i5-12600K | AVX2 | 22.98 ms | **10.48 ms** | 2.2x |
| Xeon E5645 | SSE4.2 | 101.91 ms | **48.90 ms** | 2.08x |
| RK3588 (A76) | NEON | 26.83 ms | **18.09 ms** | 1.48x |

Threaded, over `sigil_scan_*_range()`:

| machine | 1 thread | static split | work pool |
|---|---|---|---|
| 2x Xeon E5645 (12 cores) | 49.11 ms | 9.45 ms | **7.11 ms** (6.91x) |
| i5-12600K | 12.43 ms | 2.70 ms | **2.15 ms** (5.77x) |
| RK3588 | 36.27 ms | 12.55 ms | **7.88 ms** (4.61x) |

A work pool beats a static split on every machine, because all three have
heterogeneous cores. Static splitting waits on the slowest core.

An OpenCL kernel (`bench/scan_opencl.c`) reaches 222 GB/s on an Arc Pro B50 — 2.9x the
best CPU configuration — but only for a store already resident on the device; the PCIe
upload is 43x the kernel time. Deferred until there is an application to accelerate.

## Design decisions the measurements forced

Several of these reversed an earlier choice. They are kept here because the reasoning
is more useful than the conclusion.

**The unit is a paragraph, not a file.** Whole-document embedding forces one vector to
represent everything a document says, and the embedder truncates at 512 tokens
regardless, so most of a long document was being silently discarded. `para = 0` holds
a document-level average for coarse queries; `1..n` locate the passage.

**LSH width is a per-store parameter, not a constant.** How many bits a corpus needs
varies widely:

| corpus | 128 bits | 256 | 512 |
|---|---|---|---|
| Quora questions | 95.6% | 98.3% | 98.8% |
| arXiv paragraphs | 76.8% | 89.6% | 94.5% |
| citation contexts | 57.3% | 73.7% | 87.9% |

(percent of that corpus's own float32 ceiling.) Short questions on unrelated topics
separate under a coarse code; dense academic prose needs finer resolution. The
original 128-bit choice was set from Quora alone and generalized badly.

**Clusters come from asserted edges, not similarity.** Connected components at a
similarity threshold — the method originally specified — has no usable operating
point: cos >= 0.60 puts 2018 of 5000 items in one cluster, cos >= 0.70 separates
co-cited passages 92% of the time. This is at float32, so no bit width fixes it.
Citation edges alone reached F1 0.755 with a largest cluster of 6. But *unioning* the
two collapsed purity from 1.000 to 0.009 — the signals are orthogonal, not noisy
versions of each other.

**Model choice is not MTEB rank.** Seven embedding models were compared; the smallest
and oldest won. What predicts post-SimHash quality is embedding *isotropy* — mean
|cosine| between unrelated documents — not benchmark position or dimensionality.

## Build

```sh
make                 # libsigil.a
make check           # differential tests, scalar vs SIMD
make check-semantic  # proves the LSH bits carry meaning
make eval            # retrieval quality on a standard corpus
make bench           # single-threaded throughput
make bench-mt        # threaded, static split vs work pool
make sbom            # SPDX 2.3 SBOM
```

The library vendors BLAKE3 (portable C only) and has no other dependencies. AVX2 and
SSE4.2 are selected at runtime from cached CPUID probes; NEON is the aarch64 baseline;
anything else falls back to scalar with `sigil_have_simd()` reporting zero. One binary
runs everywhere — an earlier version probed for AVX2, ignored the answer, and crashed
with SIGILL on a 2010 Xeon.

Semantic embeddings need llama.cpp, detected at build time and optional:

```sh
python3 llama.cpp/convert_hf_to_gguf.py \
    ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*/ \
    --outfile ~/models/all-MiniLM-L6-v2-f16.gguf --outtype f16

make check-semantic LLAMA_DIR=~/llama.cpp MODEL=~/models/all-MiniLM-L6-v2-f16.gguf
```

Without it the library still builds and `make check-semantic` skips rather than
silently passing against the non-semantic fallback.

## Correctness

Every SIMD kernel has a scalar twin, and `test/differential.c` runs both over random
input at counts straddling the lane boundaries, asserting identical results. A wrong
SIMD kernel does not crash — it returns a slightly wrong Hamming distance that looks
plausible and stays wrong forever.

`make check-semantic` is the guard against the LSH bits quietly ceasing to be
semantic. Run `./test/semantic` with no model and it uses a backend named
`hash_nonsemantic`; separation collapses and the test fails. That failure is the point.

The trit encoding is base-3 rather than two-bits-per-trit: all 729 packed values are
legal and round-trip, and all 64807 remaining 16-bit values are rejected as
corruption. A 2-bit scheme would waste a quarter of its encoding space and silently
decode corruption as valid data.

## Status

Working: the 64-byte layout, BLAKE3 identity, trit packing, the struct-of-arrays
store, scan kernels in scalar/AVX2/SSE4.2/NEON with differential tests, chunked
embedding via llama.cpp, and evaluation harnesses for retrieval, clustering, and
formula hashing. Builds and passes on x86-64 and aarch64.

Not built:

- **Persistence.** The store is in-memory. The durable format is libtab (ndb-shaped
  text), and its `schema=` tuple must record `model_id`, `embed_dim`, `simhash_seed`
  and `lsh_bits` — a store whose parameters differ is not comparable and must refuse
  to open rather than return wrong answers.
- **Indexer.** Nothing walks a directory yet. Only tests create sigils.
- **9P2000 server.** The planned frontend, so a semantic query is a directory walk
  rather than something smuggled through an ioctl. Both dependencies are verified —
  libtab round-trips under plan9port and a lib9p test server served a live namespace
  over the protocol. See [`docs/9P-PLAN.md`](docs/9P-PLAN.md).
- **Clustering and the classifier**, which follow from persistence.

The tools in `tools/` work standalone today, without any of the above:

```sh
tools/xlsx_ast.py '=SUM(A1,B1)' '=SUM(B1,A1)'   # same hash: SUM commutes
tools/math_ast.py                                # LaTeX -> canonical tree hash
bench/xlsx_eval.py /path/to/spreadsheets 500     # duplicate-formula report
```

`bench/` holds the measurement scripts behind every number above, including the ones
that record failures — `cluster_eval.py` demonstrates the clustering that does not
work, `escalate.py` the cheap discriminator that does not separate.

## License

GPLv3 or later. Copyleft rather than permissive for two reasons: a kernel module would
need to be GPL regardless, and the design's value is in the layout idea, which is
worth keeping open.

## A note on hashing

BLAKE3 at full 256 bits, not truncated. Truncating to 128 gives a collision
probability around 1.5e-21 at a billion paragraphs — unreachable by accident, but only
2^64 work to construct one deliberately. The store is designed assuming someone will
attack it on purpose for a reason not yet known, and collision resistance cannot be
retrofitted once stores exist in the wild.
