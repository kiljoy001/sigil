# sigil

A content-addressed semantic filesystem where **the identifier carries its own index**.

An ordinary content address is a hash: opaque, uniformly distributed by design, and
therefore useful only for exact lookup. Asking "what is similar to this?" or "what
changed last March?" means building a second structure alongside it — an index that
has to be maintained on every write, kept consistent, and rebuilt when it corrupts.

A sigil spends 20 of its 48 bytes on identity and the rest on *queryable structure*.
Similarity becomes Hamming distance over a packed field. Time filtering becomes an
integer compare. Both run as a linear scan at memory bandwidth, and there is no index
to maintain, so writes are O(1).

```
[ 0..16)  LSH semantic bits       similarity (128 bits, Hamming distance)
[16..36)  SHA-1 content hash      identity
[36..40)  Unix timestamp          time-range queries
[40..42)  category code           classification
[42..44)  packed trits            6 balanced-ternary hints
[44..48)  tail padding
```

48 bytes. The LSH array leads because it needs 8-byte alignment; with the hash first
the compiler pads the record out to 56.

An earlier version of this file argued that 32 bytes was load-bearing because it is
one cache line. That was wrong. Under the struct-of-arrays store, records are never
scanned as records — a similarity pass walks the `lsh[]` array, whose stride is the
LSH field alone, so total record size does not affect scan locality. The cache-line
argument was inherited from an array-of-structs design this code does not use, and it
was being used to reject exactly the widening that turned out to matter most.

## Measured performance

10M records, one core, i5-12600K:

| kernel            |    ms | GB/s  |
|-------------------|------:|------:|
| similarity scalar | 22.98 |  7.0  |
| similarity AVX2   | 10.48 | 15.3  |

The similarity kernel scans every record. The scan is bandwidth-bound, not ALU-bound,
which is the point of the layout.

Same source, same test, on aarch64 (RK3588, Cortex-A76):

| kernel            |    ms | GB/s  |
|-------------------|------:|------:|
| similarity scalar | 26.83 |  6.0  |
| similarity NEON   | 18.09 |  8.9  |

**NEON is simpler but slower.** A NEON register is 128 bits — exactly one LSH code,
no lane bookkeeping — and aarch64 has `vcntq_u8`, a real per-byte popcount, where
AVX2 has to emulate one with a nibble table and `vpshufb`. The inner loop is three
instructions against roughly six.

It still only reaches 1.48x over scalar where AVX2 reaches 2.2x, and x86 finishes the
same work in 10.48 ms against 18.09. The reasons are structural: AVX2's 256-bit
register takes two records per iteration to NEON's one, and aarch64 has no movemask,
so lane-selecting kernels round-trip through memory instead of extracting a bitmask.
That second penalty is why `timerange` initially ran *slower* than scalar on ARM
(18.08 ms vs 10.87) until a `vmaxvq` early-out was added to skip empty blocks — now
4.66 ms. Unrolling similarity to four chains was also tried and came out slower
(19.77 ms); the loop is memory-bound, so extra in-flight work only costs registers.

A wider-register ARM core (SVE2) would likely close the gap, but the Cortex-A76 has
NEON only.

And on pre-AVX2 x86 (Xeon E5645, Westmere, 2010):

| kernel             |     ms | GB/s  |
|--------------------|-------:|------:|
| similarity scalar  | 101.91 |  1.6  |
| similarity SSE4.2  |  48.90 |  3.3  |

2.08x, the best relative gain of the three targets. An XMM register is 128 bits —
exactly one LSH code, same as NEON — but unlike NEON, SSE has `movemask`, so the
lane-selecting kernels keep results in a register instead of round-tripping through
memory. That is worth more here than NEON's native popcount is there.

Dispatch is AVX2, else SSE4.2, else scalar, chosen at runtime from cached CPUID
probes. The vector bodies carry target attributes and the object is built for generic
x86-64, so one binary runs everywhere — but that means the check is mandatory:
calling an AVX2 body on a 2010 Xeon faults with SIGILL. An earlier version had the
probe and did not branch on it, and crashed on exactly that hardware.

Reproduce with `make bench`, or `./test/bench 100000000` for a larger run.

### On the numbers this replaces

Earlier design notes for the predecessor project claimed a 1B-record scan in 20ms by
counting cycles and ignoring DRAM — wrong by roughly 50x, since a full pass is
bandwidth-bound. Those notes also claimed "16 CIDs per cycle," conflating 32-*bit*
fingerprints with 32-*byte* records. The benchmark here reports bytes actually
touched so the arithmetic can be checked rather than taken on faith, and uses a query
radius tight enough that the scan runs to completion instead of filling the result
buffer and exiting early.

## Struct-of-arrays

Records are stored decomposed by field, not as an array of structs. A similarity pass
over N records touches 16N bytes of LSH words instead of 48N bytes of whole records —
3x less memory traffic on the binding constraint. Full records are reassembled only
for the survivors of a filter.

## The bits carry meaning

The 128 LSH bits come from a sentence embedding, not a byte hash. Text goes through
`all-MiniLM-L6-v2` (384-dim) and is projected to 128 bits by SimHash against fixed
random hyperplanes. SimHash is the right reduction because the probability that two
vectors fall on opposite sides of a random hyperplane is `theta/pi` — so Hamming
distance in the compressed space is an unbiased estimator of angular distance in the
original. Cosine similarity survives the squeeze.

### How good are the bits?

Measured on **Quora Duplicate Questions** — 1000 human-labeled pairs from the corpus
BEIR and MTEB use — as recall@1 over 2000 documents: for each question, is its true
duplicate the nearest neighbour among all 1999 others?

| | recall@1 | of ceiling |
|---|---|---|
| float32 cosine (ceiling) | 0.8140 | — |
| 128-bit LSH | **0.7780** | **95.6%** |
| 32-bit LSH (previous) | 0.5495 | 67.5% |

The ceiling is what the *embedding model* can do; no compression beats it. At 32 bits
the code was discarding a third of what MiniLM knew. At 128 it keeps almost all of
it, and the curve flattens hard after that — 256 bits buys 1.8 more points, 512 buys
2.9. The remaining headroom is in the model, not the width.

Reproduce with `tools/fetch-corpus.py` then `make eval`.

A note on corpus choice: an earlier version of this benchmark used paraphrase pairs
written by hand. That measures how separable the author made the examples, not how
the system performs — it put 32-bit recall@1 at 0.18 against the 0.55 real data
shows, and would have driven the design toward far more bits than it needs. Use a
standard corpus.

### Is it semantic at all?

`make check-semantic` is the cheaper guard: paraphrase pairs sharing almost no
vocabulary must land closer than unrelated ones, with a margin and no overlap.
Currently 37.8 vs 64.8 mean Hamming distance, 27 bits of separation.

Run `./test/semantic` with no model and it uses a backend named
`hash_nonsemantic` — separation collapses, the distributions overlap, and the test
fails. That failure is the point: it is the regression guard against a placeholder
quietly returning.

## Correctness

Every SIMD kernel has a scalar twin, and `test/differential.c` runs both over random
input at counts chosen to straddle the SIMD lane boundaries, asserting identical
results. This matters more than it sounds: a wrong SIMD kernel does not crash, it
returns a slightly wrong Hamming distance that looks entirely plausible and stays
wrong forever.

The trit encoding is base-3 rather than two-bits-per-trit. All 3^6 = 729 packed values
are legal and round-trip; all 64807 remaining 16-bit values are rejected as
corruption. A 2-bit scheme would waste a quarter of its encoding space on states that
mean nothing and would silently decode corruption as valid data.

```
$ make check
sigil differential tests (SIMD active)
sizeof(sigil_t) = 48, LSH 128 bits
  ok   trits: all 729 values round-trip, 64807 corrupt values rejected
  ok   scan: n=0 ... n=10000
all passed
```

## Build

```sh
make                 # libsigil.a
make check           # differential tests (scalar vs SIMD)
make check-semantic  # proves the LSH bits are semantic
make eval            # retrieval quality on a standard corpus
make bench           # throughput
make sbom            # SPDX 2.3 SBOM
```

The core library has no third-party dependencies: SHA-1 is implemented in-tree and
the scalar kernels build anywhere. AVX2 is selected at compile time and probed at
runtime via CPUID; NEON is the aarch64 baseline so it needs no probe; anything else
falls back to scalar with `sigil_have_simd()` reporting zero.

Semantic embeddings need llama.cpp, which is detected at build time and optional. The
library builds and links without it; `sigil_embedder_llama()` returns NULL and
`make check-semantic` skips rather than silently passing against the fallback.

```sh
# Build llama.cpp, then convert an embedding model:
python3 llama.cpp/convert_hf_to_gguf.py \
    ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*/ \
    --outfile ~/models/all-MiniLM-L6-v2-f16.gguf --outtype f16

make check-semantic LLAMA_DIR=~/llama.cpp MODEL=~/models/all-MiniLM-L6-v2-f16.gguf
```

### A caveat on the hyperplane seed

The projection is seeded (`SIMHASH_SEED`), and hyperplanes are generated
deterministically with splitmix64 so two machines produce identical bits. But a
store's bits are only comparable against the seed *and* embedding width that produced
them. Changing either invalidates every sigil already written. This wants to be
recorded in a store header before anything durable is built on it.

## Status

Working: the 48-byte layout with 128-bit LSH, trit packing, the struct-of-arrays
store, the three scan kernels in scalar, AVX2, and NEON with differential tests
proving they agree, real semantic embeddings via llama.cpp, and retrieval evaluation
against a standard corpus. Builds and passes on x86-64 and aarch64.

Not yet built:

- **9P2000 server.** The planned frontend. 9P suits this better than FUSE because the
  served namespace is yours to define, so a semantic query is a directory walk rather
  than something smuggled through an ioctl — `/similar/<hex>/` populated by the scan
  above. Base 9P2000 rather than 9P2000.L: the `.L` extensions exist for POSIX
  fidelity this does not need, and base 9P2000 talks to Linux v9fs, plan9port, and
  native Plan 9 alike. The Linux client is already in mainline, so no kernel module is
  required.
- **Persistence.** The store is in-memory. A durable format needs to record the
  embedding model, its width, and the hyperplane seed, since bits made under different
  parameters are not comparable.
- **Tiered embedding.** One model runs on everything today, at ~470 docs/s. The
  cheaper design is progressive — a fast sketch on all files, the transformer only on
  what matters — since embedding now dominates per-file cost by roughly 60x over
  hashing. INT8 quantization is quality-neutral here (0.7880 vs 0.7880 measured), so
  NPU or GPU inference costs nothing in retrieval quality.

## License

GPLv3 or later. Copyleft rather than permissive for two reasons: a kernel module
would need to be GPL regardless, and the design's value is in the layout idea, which
is worth keeping open rather than letting it disappear into a closed product.

## A note on SHA-1

SHA-1 is used here for content identity, not security. Collision resistance against a
deliberate attacker is not claimed. If that property is ever needed, the hash field
widens to SHA-256 and the record layout has to be revisited — which is a real design
constraint, not an oversight.
