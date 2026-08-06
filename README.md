# sigil

A content-addressed semantic filesystem where **the identifier carries its own index**.

An ordinary content address is a hash: opaque, uniformly distributed by design, and
therefore useful only for exact lookup. Asking "what is similar to this?" or "what
changed last March?" means building a second structure alongside it — an index that
has to be maintained on every write, kept consistent, and rebuilt when it corrupts.

A sigil spends 20 of its 32 bytes on identity and the remaining 12 on *queryable
structure*. Similarity becomes Hamming distance over a packed field. Time filtering
becomes an integer compare. Both run as a linear scan at memory bandwidth, and there
is no index to maintain, so writes are O(1).

```
[ 0..20)  SHA-1 content hash      identity
[20..24)  LSH semantic bits       similarity (Hamming distance)
[24..28)  Unix timestamp          time-range queries
[28..30)  category code           classification
[30..32)  packed trits            6 balanced-ternary hints
```

32 bytes is load-bearing, not arbitrary — it is one cache line. A 36- or 40-byte
record straddles lines and gives back the locality that makes the index-free design
work. The struct does not grow.

## Measured performance

100M records on one core, AVX2, DDR4:

| kernel            |    ms | GB/s  |
|-------------------|------:|------:|
| similarity scalar | 110.4 |  3.6  |
| similarity AVX2   |  21.5 | 18.6  |

The similarity kernel scans every record. 18.6 GB/s is memory bandwidth — the scan is
bandwidth-bound, not ALU-bound, which is the point. AVX2 buys 5.1x over scalar and
then hardware sets the ceiling.

Reproduce with `make bench && ./test/bench 100000000`.

### On the numbers this replaces

Earlier design notes for the predecessor project claimed a 1B-record scan in 20ms by
counting cycles and ignoring DRAM. That was wrong by roughly 50x. The real figure for
a full 32 GB pass is ~1.1s; the LSH-only pass under the struct-of-arrays split is
~215ms. Those notes also claimed "16 CIDs per cycle," which conflated 32-*bit*
fingerprints with 32-*byte* records. The benchmark here reports bytes actually
touched so the arithmetic can be checked rather than taken on faith.

## Struct-of-arrays

Records are stored decomposed by field, not as an array of structs. A similarity pass
over N records touches 4N bytes of LSH words instead of 32N bytes of whole records —
8x less memory traffic on the binding constraint. Full records are reassembled only
for the survivors of a filter.

## The bits carry meaning

The 32 LSH bits come from a sentence embedding, not a byte hash. Text goes through
`all-MiniLM-L6-v2` (384-dim) and is projected to 32 bits by SimHash against fixed
random hyperplanes. SimHash is the right reduction because the probability that two
vectors fall on opposite sides of a random hyperplane is `theta/pi` — so Hamming
distance in the compressed space is an unbiased estimator of angular distance in the
original. Cosine similarity survives the squeeze.

Measured on paraphrase pairs chosen to share almost no vocabulary, so that any signal
must come from meaning rather than surface form:

```
paraphrases (should be close)        unrelated (should be far)
  cat/feline              8            cat/earnings           14
  earnings/revenue        8            crash/photography      17
  crash/downtime          9            contract/feline        15
  photography             8            revenue/photography    18
  contract/agreement     12

mean paraphrase distance: 9.0
mean unrelated distance:  16.0
separation:               7.0 bits
```

"the contract was terminated early" and "they ended the agreement ahead of schedule"
land 12 bits apart with not one word in common. The distributions do not overlap, so
a single Hamming threshold separates them at query time.

`make check-semantic` asserts both the separation margin and the non-overlap. Run
`./test/semantic` with no model and it uses the deliberately non-semantic hash
backend, where separation collapses to 1.6 bits and the distributions overlap — the
test fails, which is exactly its purpose. It is the regression guard that keeps a
placeholder from quietly returning.

## Correctness

Every SIMD kernel has a scalar twin, and `test/differential.c` runs both over random
input at counts chosen to straddle the 8- and 16-lane boundaries, asserting identical
results. This matters more than it sounds: a wrong SIMD kernel does not crash, it
returns a slightly wrong Hamming distance that looks entirely plausible and stays
wrong forever.

The trit encoding is base-3 rather than two-bits-per-trit. All 3^6 = 729 packed values
are legal and round-trip; all 64807 remaining 16-bit values are rejected as
corruption. A 2-bit scheme would waste a quarter of its encoding space on states that
mean nothing and would silently decode corruption as valid data.

```
$ make check
sigil differential tests (AVX2 available)
sizeof(sigil_t) = 32
  ok   trits: all 729 values round-trip, 64807 corrupt values rejected
  ok   scan: n=0 ... n=10000
all passed
```

## Build

```sh
make                 # libsigil.a
make check           # differential tests (scalar vs AVX2)
make check-semantic  # proves the LSH bits are semantic
make bench           # throughput
make sbom            # SPDX 2.3 SBOM
```

The core library has no third-party dependencies: SHA-1 is implemented in-tree, the
scalar kernels build anywhere, and AVX2 is detected at runtime via CPUID.

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

Working: the 32-byte layout, trit packing, the struct-of-arrays store, the three scan
kernels with scalar references and benchmarks, and real semantic embeddings via
llama.cpp with a test that proves they are semantic.

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
- **Tiered embedding.** One model runs on everything today. The cheaper design is
  progressive — a fast sketch on all files, the transformer only on what matters —
  since embedding is now the dominant cost per file, not the scan.

## License

GPLv3 or later. Copyleft rather than permissive for two reasons: a kernel module
would need to be GPL regardless, and the design's value is in the layout idea, which
is worth keeping open rather than letting it disappear into a closed product.

## A note on SHA-1

SHA-1 is used here for content identity, not security. Collision resistance against a
deliberate attacker is not claimed. If that property is ever needed, the hash field
widens to SHA-256 and the 32-byte layout has to be revisited — which is a real design
constraint, not an oversight.
