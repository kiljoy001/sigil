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

A second, wider representation can live beside the store in a *sidecar* — a
sparse mapped file of `(index, vector)` pairs — so a query can reorder the
scan's shortlist without the record growing past 64 bytes.

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
thing, without reference to any embedding. The mirror is cleaned and deduplicated
by `tools/pipeline.py` into **79,133 books and 77,367,817 paragraphs**, 98.2% of
them carrying an LoCC class; the retrieval numbers below are from 20,000
paragraphs across 210 books of it:

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

### The corpus needed cleaning before it could be indexed

A raw Gutenberg mirror is not a corpus. `tools/pipeline.py` turns 155,616 files
into 79,133 books in about 150 seconds, and each stage exists because of a
measured defect rather than tidiness:

| stage | what it fixes |
|---|---|
| encoding | Windows-1252 bytes inside files served as UTF-8 |
| deduplication | 76,483 superseded revisions and alternate encodings |
| boilerplate | a licence envelope on every file, the corpus's most common text |
| manifest | provenance and catalogue metadata, one libtab row per book |

The encoding stage is not cosmetic. Invalid UTF-8 reaching PCRE2 inside
openvino_tokenizers is documented undefined behaviour, and it read wild memory —
segfaulting the indexer on 8 runs in 12 with identical input. That looked exactly
like a race and was chased as one for most of a session. It was ASLR: whether the
wild read lands on an unmapped page depends on where the allocator put things.
`setarch -R` turns the crash into 8/8 and ends the guessing. Both the corpus fix
and a guard at the embedder boundary are in; the crash rate on cleaned data with
ASLR off is 0/5.

Four more bugs appeared only at full scale, none of which a 331-file test corner
could have shown:

- **Books overwriting each other.** `book_id()` took the deepest numeric directory,
  but some files sit directly in a fanout directory — `1/6/0/1602.txt` has no
  per-book subdirectory, so its id read as `0`. Books sharing a bogus id
  overwrote each other; a trial slice went from 11,029 books to 11,145 once the
  filename took precedence.
- **Packaging indexed as books.** 11,382 of the mirror's `.txt` files are not works
  (1,410 are `LICENSE.txt`). Where a title ships only as HTML there is no competing
  file, and six books were indexed as MathJax build instructions.
- **Three marker forms** the licence stripper missed: indented, wrapped across a
  line, and preceded by a BOM. Leftover licence text went from 2.95% of books
  to 0.025%; the two that remain are defects in Gutenberg's own files.
- **A resume that destroyed its own record.** The manifest is rewritten whole each
  run, so a second pass that skipped every book wrote an empty manifest beside a
  full tree. Caught by the test written for exactly that trap, on its first run.

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

### The whole corpus, in 1.8 MB of memory

The store is a list of fixed-size segments, and a segment never moves once
allocated. That makes two things possible. Growth appends rather than
reallocating, so it stops copying itself:

| | before growth | after | transient |
|---|---|---|---|
| flat array | 297.5 MB | 601.5 MB | +304 MB, a second copy |
| **segmented** | 450.8 MB | 514.9 MB | **+64 MB, one segment** |

And a sealed segment can be a *mapping of a file* rather than heap, which a
reallocating array could never be — every doubling invalidated every address.
Opening a mapped store costs no parsing, no allocation and no copy: the
field arrays are pointers into the file.

The full corpus is 59,618,093 paragraphs, built from the 31 GB CSV in 110 s
at 543,753 rec/s into a 3.6 GB file:

| | resident | scan | open |
|---|---|---|---|
| heap, 20M records | 1222.6 MB | 25 ms | 42 ms |
| **mapped, 59.6M records** | **1.8 MB** | 68 ms | **0.01 ms** |
| mapped, cold page cache | 1.8 MB | 281 ms | 0.01 ms |

Resident memory no longer tracks corpus size. The cold row is reported
because it is the honest one: mmap removes the copy and the RAM ceiling, not
the read, so a first query after a restart is storage-bound at 3.17 GB/s
against 13.0 warm.

`wc -l` reports 68,027,639 lines for that CSV against 59,618,093 records,
and the gap is real rather than lost work: 8,185,964 rows carry embedded
newlines inside quoted fields. Parsing goes through libcsv for that reason —
a split on commas would not have failed on those rows, it would have
mis-fielded them.

### Reranking the shortlist: 10x R@1 for 0.12 ms

The scan compares 128-bit codes, which is fast and lossy. The float vectors
are accurate and far too slow to compare against everything. So the cheap
thing narrows and the expensive thing orders. 200,064 paragraphs, 1,191
queries, ground truth same-book:

| method | R@1 | R@5 | R@10 | R@20 | ms/query |
|---|---|---|---|---|---|
| scan alone | 0.0059 | 0.0084 | 0.0168 | 0.0285 | 0.33 |
| **scan + rerank** | **0.0605** | **0.1092** | **0.1360** | **0.1553** | **0.45** |

The float work is 0.12 ms because stage one already reduced the corpus to a
shortlist. Vectors live in a *sidecar* — a separate mapped file of
`(index, vector)` pairs, sparse and sorted — rather than in the record,
which is a fixed 64 bytes with a `_Static_assert` on it.

Width and rerank fix different failures, and both were measured: rerank
fixes *ordering* within a shortlist, width fixes *membership* of it. 512
bits plus rerank reaches the float32 ceiling at R@1 (0.1044 against 0.1027),
and at 2048 the reranked numbers equal float32 in every column. But the scan
cost is linear in width — 0.447 ms at 128 against 1.716 at 512 through the
real radius path — and the AVX2/SSE/NEON kernels are written for two 64-bit
words, so a wider code needs them generalised first. 128 with rerank is what
ships; widening is a real gain and a real project.

Asymmetric distance was measured against the same data and lost: keeping the
query as floats against stored bits reaches 0.0411 R@1 with no extra storage
at all, but that is half of rerank's gain at seven times the cost. It is the
right answer when the vectors will not fit. Here they map.

The idea this displaced was a second, larger embedding model over the middle
of the distance distribution. Two measurements say there is nothing there:
same-book rate at the median distance of 60 bits is **0.58x chance**, and
re-embedding those pairs under a 768-dimension model finds cosine separation
of −0.007. The bulk at 60 is the geometry of "unrelated", not hidden signal
— which is also evidence the LSH is working. See
[`docs/FINDINGS.md`](docs/FINDINGS.md).

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

### Persistence was quadratic, and profiling found what reading did not

`store_commit` rewrites the whole table, issuing five `tab_set` calls per record.
Measured against libtab as it stood:

| records | commit |
|---|---|
| 40,000 | 8.90 s |
| 80,000 | 37.74 s |
| 160,000 | **126.71 s** |

Four times the cost for twice the rows. Extrapolated to this corpus that is roughly
**68 days for a single commit**, which makes any argument about embedding throughput
beside the point.

Reading the code suggested two culprits and both were wrong — fixing them moved
126.71 s to 82.96 s and it stayed quadratic. `perf` then put 85% of samples inside a
function that had already been patched, whose loops walk a bucket chain and look
innocent. They were innocent; the chains were not:

```c
ensure_buckets(Tab *t, int target_rows)
{
    if(t->buckets != nil)
        return 0;            /* never grows */
```

A table built by `tab_create` starts empty, so it allocated 16 buckets and kept them
— at 160k rows, ~10,000 entries per bucket, every walk a full scan with a malloc per
step. Growing the array, plus letting `tab_add_row` reuse the row `already_present()
`had just matched instead of rescanning:

| records | before | after | |
|---|---|---|---|
| 40,000 | 8.90 s | 0.148 s | 60x |
| 80,000 | 37.74 s | 0.278 s | 136x |
| 160,000 | 126.71 s | 0.549 s | **231x** |
| 320,000 | — | 1.107 s | linear |

Fixed upstream in [libtab](https://github.com/kiljoy001/libtab), released as
py-libtab 0.3.1, with regression tests that were themselves mutation-tested — an
earlier version of the guard passed with the fix reverted, because unique keys never
reach the deduplication path the rescan lived in.

The lesson is the process. Two plausible O(n) scans found by reading, fixed, and
worth 1.5x between them; the actual cause was a missing resize that no amount of
staring at the hot function would reveal, because the hot function was correct.

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
make check           # differential, unit, bridge, oom, segments, mapped, sidecar
make check-semantic  # proves the LSH bits carry meaning
make eval            # retrieval quality on a standard corpus
make bench           # single-threaded throughput
make bench-mt        # threaded, static split vs work pool
make segments        # growth appends, never copies
make mapped          # file-backed stores, and what they refuse
make sidecar         # stage-two vectors and rerank
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
python3 tools/embed_openvino.py         # 1184 paragraphs/s on an Arc Pro B50
```

| backend | paragraphs/s | 59.6M-paragraph corpus |
|---|---|---|
| **OpenVINO, Arc Pro B50** | **1184** | **~14 h** |
| OpenVINO, CPU (i5-12600K) | 605 | 27 h |
| ollama / Vulkan | 188 | 88 h |
| OpenVINO, iGPU (UHD 770) | 272 | 61 h |

The Arc figure is warm — the first call compiles and caches the model, and an
unwarmed measurement reported 515/s for the same work.

**Different devices run concurrently and safely; the same device does not.** GPU.1
and CPU together sustain 1491/s, which is 1.26x the Arc alone rather than the 1.79x
naive addition — even the GPU path tokenizes on the host, so they contend there.
Vectors stay bit-comparable: cosine agreement with the solo runs is 0.4560 vs
0.4561.

Two OpenVINO pipelines on the *same* device corrupt each other's output with no
warning, in a way indistinguishable from the llama.cpp fault above. Serialise per
device; parallelise across them.

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

The Python pipeline carries its own suite: 280 tests at 96% coverage, property
tests over generated input, and mutation testing in two sets. `test/mutate_py.sh`
runs 47 mutants chosen by hand — mistakes a person could plausibly make here —
and every one must die. `test/mutate_mutmut.sh` runs several hundred that nobody
chose. Both matter, because the curated set can only cover failures someone
imagined, and the crash that started this work was not one of them. The generated
set found five real gaps the curated set missed, including a function the process
pool calls that had no test at all.

Property tests found three values libtab cannot store — an ASCII quote has no
escape, control characters end the record, and the token `nil` reads back as
absent — each of which writes a file that looks fine and fails when something
later opens it. Filed upstream as libtab#1; the pipeline substitutes rather than
corrupting.

CI gates all of it: the C suite with and without sanitizers, a 90% coverage floor
on both languages, CRAP score with zero functions permitted over 30, and both
mutation suites. Every gate has been verified to fail when it should — a gate
that cannot fail is decoration.

## Status

Working end to end: the 64-byte layout, BLAKE3 identity, trit packing, the
segmented struct-of-arrays store, scan kernels in scalar/AVX2/SSE4.2/NEON/SYCL with
differential tests, embedding via llama.cpp or OpenVINO, libtab persistence that
refuses to open on a parameter mismatch, file-backed stores that map the whole
corpus in 1.8 MB, float rerank over a sidecar, a directory indexer, and
`/similar/<hex>/` served over 9P with paragraphs read back from their source.
Builds and passes on x86-64 and aarch64.

Not built:

- **Incremental reindex.** A changed file forces a full rebuild. The per-paragraph
  BLAKE3 makes "has this changed" a cheap comparison; it is simply not wired up.
- **Classification in the namespace.** The two-pass classifier works in `tools/` and
  is measured, but nothing projects its themes into the filesystem yet.
- **A full-corpus semantic run.** The whole corpus builds and scans — 59,618,093
  paragraphs, 110 s, 1.8 MB resident — but with the byte-shingle fallback rather
  than embeddings, which measures the store and not retrieval. Embedding is the
  bottleneck by four orders of magnitude, and the retrieval numbers above are from
  a 200,064 paragraph sample. `tools/build_store -e` does the semantic build.
- **Rerank in the namespace.** `sigil_side_rerank()` is measured and tested, but
  `/similar/<hex>/` does not consult a sidecar yet.

Four bugs were found only by running this, not by reading it, and all four are in
the log rather than quietly fixed:

- `removefile()` refuses a directory that still has children, so cache flushes failed
  silently and served stale results while reporting "file already exists".
- `store.c` recorded `lsh_bits = 256` while the library produces 128 — and that field
  is the guarantee two stores are comparable.
- Persisted records came back from a restart with a **different BLAKE3** than they
  went in with, because reload recomputed the hash from the path. Content addressing
  whose address changes across a restart is not content addressing. The correct hash
  was in the file the whole time and simply was not read back.
- `sigilfs` reported "cannot load model" with no OpenVINO error, because none had
  run: the not-built stub for `sigil_embedder_openvino()` sat behind
  `#ifndef SIGIL_WITH_OPENVINO`, but the Makefile defined that macro only on the C++
  compile rule. Both definitions reached the archive and the linker took the stub,
  which returns NULL silently.

The tools in `tools/` work standalone, without a server:

```sh
tools/xlsx_ast.py '=SUM(A1,B1)' '=SUM(B1,A1)'   # same hash: SUM commutes
tools/math_ast.py                                # LaTeX -> canonical tree hash
tools/gutenberg.py pg_catalog.csv /mirror        # corpus -> paragraphs + metadata
bench/gutenberg_eval.py paragraphs.csv           # retrieval vs subject headings

# The corpus pipeline: raw mirror -> indexable books + a manifest.
tools/pipeline.py /mirror /books --catalog pg_catalog.csv -j 12
tools/manifest.py /books/manifest.tab            # summary, or one book by id
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
