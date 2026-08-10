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

**Status: a working library and a mountable filesystem.** `sigilfs` indexes a
directory into paragraphs, embeds them, persists to libtab, reloads on restart, and
answers a semantic query as a directory walk — see [Status](#status).

## Using it

```sh
cmd/sigilfs -a 'tcp!127.0.0.1!5640' -f store.tab -e ~/models/all-MiniLM-L6-v2-f16.gguf &

echo 'mount docs /home/me/notes' >> /mnt/sigil/ctl
echo 'index'                     >> /mnt/sigil/ctl
echo 'similar <hash>'            >> /mnt/sigil/ctl

ls  /mnt/sigil/similar/<hash>/          # the neighbourhood
cat /mnt/sigil/similar/<hash>/<hash>    # provenance, then the paragraph
```

The walk is the query. There is no query language and no ioctl: `similar` runs a
scan and materializes the result as real files, and reading one seeks to the
paragraph's recorded offset in its source.

Neighbourhoods, not partitions. `/similar/A/` lists what is near A; B appearing
there does not put A and B in a shared group, and a paragraph appears in every
neighbourhood it is near. Similarity-only clustering measured badly enough that
shipping a partition would be an overclaim — see below.

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

### A fourth corpus: library cataloguing as ground truth

Project Gutenberg has no citation graph, so the external judgement is the Library
of Congress subject headings — a cataloguer decided two books are about the same
thing, without reference to any embedding. 20,000 paragraphs across 210 books:

| method | R@1 | R@10 | R@20 |
|---|---|---|---|
| float32 | 0.199 | 0.469 | 0.587 |
| 512-bit | 0.182 | 0.471 | 0.590 |
| 256-bit | 0.183 | 0.447 | 0.557 |
| 128-bit | 0.130 | 0.416 | 0.542 |

R@1 against 2.26% chance is **8.8x**, reproduced on two independent embedding
backends. Two things this corpus showed that the citation corpus could not:

**Non-fiction retrieves at 0.63 R@10 against literature's 0.37.** Most paragraphs
of a novel are dialogue or scene-setting and are not *about* the book's subject,
and the literature classes divide by national tradition — PS American, PR English,
PQ Romance — so two sea novels land apart by the author's nationality. Averaging
them into one number hides this.

**The same-author rate among neighbours is 0.091 against a subject-match rate of
0.199.** Subject matching runs about twice style matching, so the embedder is not
merely recognising prose voice. Without that control the headline number would not
be evidence of anything.

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

One SYCL kernel (`src/scan_sycl.cpp`) replaces all four hand-written ones and lands
within 8% of the tuned version, with results bit-identical to the scalar reference:

| kernel | 10M records | maintained |
|---|---|---|
| hand SIMD + work pool | 2.15 ms | four kernels plus threading |
| **SYCL, host USM** | **2.33 ms** | **one kernel, no threading code** |
| SYCL, device USM + copy | 20.75 ms | — |

The last row is why this nearly got the wrong answer: the first version of the
benchmark copied memory to itself on the CPU path, which looked exactly like a
compiler failing to vectorise. A measurement that makes a mature toolchain look
broken is more likely to be wrong than the toolchain.

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

**An LLM can supply the asserted edges a corpus lacks — and it names them.** On
Gutenberg, 2000 judged pairs separate by **+14.7 bits** under the judge's verdict
against **+9.0** under the catalogue label, with confirmation decaying monotonically
from 55.6% under 60 bits to 4.8% past 120. Each confirmed edge carries the subjects
the two passages share, so a group arrives with a name rather than as an unlabelled
blob. A second pass consolidates them — `war, battle, combat, conflict, victory,
expedition` become one theme — for one call per sixty subjects, against one call per
pair in the first. The layer doing the most valuable reasoning is nearly free.

Prompt shape decides whether any of this works. Asking for a verdict directly
collapsed to a constant answer in three of four attempts; asking the model to extract
each passage's main idea *first* and only then compare gives a live classifier. The
guard that caught the collapses was checking the verdict distribution, not the recall
— a judge answering one way to everything cannot reorder anything, so its R@k equals
the scan's and reads as "no harm done".

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

## Mounting it

```sh
make sigilfs PLAN9=/path/to/plan9port LIBTAB_SRC=/path/to/libtab
cmd/sigilfs -a 'tcp!127.0.0.1!5640' -f store.tab -e ~/models/all-MiniLM-L6-v2-f16.gguf &
sudo mount -t 9p -o trans=tcp,port=5640,version=9p2000,uname=$USER,access=user \
    127.0.0.1 /mnt/sigil
```

`version=9p2000` must be explicit: v9fs defaults to `9p2000.L`, which lib9p does not
speak.

**Pass `-e` or the bits are not semantic.** Without a model the records carry a byte
hash, and `/similar/` will return neighbourhoods that look plausible and mean nothing.
`/stats` says which you have:

```
embedder	NONE -- lsh bits are a byte hash, not semantic
```

That line exists because it caught exactly this mistake during development: a
threshold sweep that produced sensible-looking counts from meaningless codes.

Use `>>` rather than `>` when writing to `/ctl`. A truncating open makes v9fs send a
Twstat, and shell redirection then fails — see the plan9port note below. Add `-L` for
a per-request trace.

Build against a plan9port source tree, not `/usr/local/plan9`. An installed copy may
predate the lib9p fix for a 64-bit `wstat` bug — a `(ulong)` cast applied to a 32-bit
wire field — which makes shell `>` redirection onto `/ctl` fail with `ESERVERFAULT`.
Upstream master still carries it; diagnosis and fix in
[`docs/PLAN9PORT-BUG.md`](docs/PLAN9PORT-BUG.md).

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

**On Intel Arc, use OpenVINO instead.** llama.cpp garbles output above roughly 4B
parameters on Battlemage through *both* its SYCL and Vulkan backends — upstream
[#20169](https://github.com/ggml-org/llama.cpp/issues/20169) (closed *not planned*),
[#24560](https://github.com/ggml-org/llama.cpp/issues/24560),
[#21888](https://github.com/ggml-org/llama.cpp/issues/21888). A 12B model answers
"What is the capital of France?" with `<pad><pad>`. It is not a configuration
problem, and it is not memory: it persists at 10 GB loaded on a 16 GB card.

OpenVINO runs the same weights correctly and 2.7x faster, and installs with pip:

```sh
pip install openvino openvino-genai "optimum-intel[openvino]"
optimum-cli export openvino -m sentence-transformers/all-MiniLM-L6-v2 \
    --task feature-extraction /models/minilm
python3 tools/embed_openvino.py         # 515 paragraphs/s on an Arc Pro B50
```

| backend | paragraphs/s | 55M-paragraph corpus |
|---|---|---|
| **OpenVINO, Arc Pro B50** | **515** | **~30 h** |
| ollama / Vulkan | 188 | 81 h |
| OpenVINO, CPU | 96 | 159 h |

Only one pipeline may hold a device at a time. Two concurrent OpenVINO processes on
one GPU corrupt each other's output with no warning, in a way indistinguishable from
the llama.cpp fault above.

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

Working end to end: the 64-byte layout, BLAKE3 identity, trit packing, the
struct-of-arrays store, scan kernels in scalar/AVX2/SSE4.2/NEON/SYCL with
differential tests, embedding via llama.cpp or OpenVINO, libtab persistence that
refuses to open on a parameter mismatch, a directory indexer, and `/similar/<hex>/`
served over 9P with paragraphs read back from their source. Builds and passes on
x86-64 and aarch64.

Not built:

- **Incremental reindex.** A changed file forces a full rebuild. The per-paragraph
  BLAKE3 makes "has this changed" a cheap comparison; it is simply not wired up.
- **Classification in the namespace.** The two-pass classifier works in `tools/` and
  is measured, but nothing projects its themes into the filesystem yet.
- **A full-corpus run.** Every number here is from at most 20,000 paragraphs of a
  231,000-paragraph sample. Embedding all 61,458 English Gutenberg texts is about 30
  hours on an Arc Pro B50 at 515 paragraphs/s, and that — not the 2 ms scan — is the
  bottleneck by four orders of magnitude.

Three bugs in this repository were found only by running the server, not by reading
it, and all three are recorded in the log rather than quietly fixed:

- `removefile()` refuses a directory that still has children, so cache flushes failed
  silently and served stale results while reporting "file already exists".
- `store.c` recorded `lsh_bits = 256` while the library produces 128 — and that field
  is the guarantee two stores are comparable.
- Persisted records came back from a restart with a **different BLAKE3** than they
  went in with, because reload recomputed the hash from the path. Content addressing
  whose address changes across a restart is not content addressing. The correct hash
  was in the file the whole time and simply was not read back.

The tools in `tools/` work standalone, without a server:

```sh
tools/xlsx_ast.py '=SUM(A1,B1)' '=SUM(B1,A1)'   # same hash: SUM commutes
tools/math_ast.py                                # LaTeX -> canonical tree hash
tools/gutenberg.py pg_catalog.csv /mirror        # corpus -> paragraphs + metadata
bench/gutenberg_eval.py paragraphs.csv           # retrieval vs subject headings
```

`bench/` holds the measurement scripts behind every number above, including the ones
that record failures — `cluster_eval.py` demonstrates the clustering that does not
work, `escalate.py` the cheap discriminator that does not separate, and
`gutenberg_rerank.py` the LLM reranking that bought +0.017 R@1 and was rejected.

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
