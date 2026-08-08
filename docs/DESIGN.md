# Design

Where the project is going and why. `FINDINGS.md` records what was measured;
this records what was decided.

## Shape

Three programs composing over a 9P namespace, not one application with
modules. Each owns its own state and serves what it owns; nothing links
anything else.

```
sigilfs      indexes mounted sources, scans, serves the semantic namespace
classifier   mounts sigilfs, reads clusters, summarizes with an LLM,
             writes labels back as a table
search       optional: ships records to Solr, serves a UI
```

`libsigil` sits under `sigilfs` as a computation library — sigils and scanning,
no I/O, no namespace, no dependencies. It is deliberately unaware of files,
mounts, and 9P.

Interchange between programs is **libtab files**: ndb-shaped text with a
`schema=` tuple. Programs exchange tables, not APIs. libtab already supports
writing through a 9P fileserver (`tab_open_dial`), which is what makes the
composition work across machines.

Target is Linux. The 9P interface means a 9front node can join later as a
client without that work being wasted, but kencc has no SIMD intrinsics and no
llama.cpp, so a native port would be scalar and serve-only. Not now.

## The unit is a paragraph

A sigil describes a paragraph, not a file.

Whole-document embedding forces one vector to represent everything a document
says, which blurs multi-topic documents toward a meaningless centroid. Worse,
the current embedder truncates at 512 tokens (~380 words) and the RKNN path at
128, so most of a long document is silently discarded.

Paragraphs are the human-shaped unit. A sentence fragments an idea; a fixed
window cuts across one. If the product is telling someone "this document
relates to that one *here*", the unit has to be a place a person would
recognize.

Each file therefore yields:

- `para = 0` — a document-level sigil, the average of its paragraph vectors,
  for coarse "these files are related" queries
- `para = 1..n` — one sigil per paragraph, for locating the passage

Coarse scan narrows to candidate documents, fine scan locates the passage.
Same two-stage pattern as the Solr banded-LSH design, one level down.

Open problems: paragraph boundaries are format-specific (blank lines in text,
inference from spacing in PDF, real marks in Office), and length is unbounded
in both directions — very short paragraphs give noisy embeddings and very long
ones exceed context. Both need policy.

## Record layout

64 bytes, one cache line.

```
[ 0..16)  LSH code        128 bits, SimHash over the paragraph embedding
[16..48)  BLAKE3          256 bits, full
[48..52)  para index      0 = document-level, 1..n = paragraphs
[52..56)  timestamp
[56..60)  cluster ref     index into the cluster table
[60..64)  reserved
```

`path` is not in the record — it is variable-length and lives in libtab.

### Why BLAKE3 at full width

SHA-1 is ~271 MB/s in the portable implementation here and has constructible
collisions. BLAKE3 is ~3 GB/s in portable SIMD C, faster than hardware SHA-1,
and has no known weaknesses.

Full 256 bits rather than truncated. Truncating to 128 gives a collision
probability around 1.5e-21 at a billion paragraphs, which is unreachable by
accident — but 2^64 work to construct one deliberately, which is expensive
rather than impossible. The store is designed on the assumption that someone
will attack it on purpose, for a purpose not yet known. Collision resistance
cannot be retrofitted once stores exist in the wild.

Record size does not affect scan speed: under the struct-of-arrays store a
similarity pass touches only the 16-byte LSH field, whatever the record's total
width.

### Federation key

Planned, not built. A shared key would let nodes in one federation derive
identical hashes (so they can dedup and merge) while preventing anyone outside
from precomputing collisions. The same key can derive the SimHash hyperplane
seed, which closes a related hole: an attacker who knows the seed can craft
text that lands at a chosen Hamming distance from a target, placing a document
in a cluster it does not belong in. A content hash does not fix that; a secret
seed does.

The schema carries a `federation` field, empty for now, so adding this later
does not change the format. `factotum` is the natural place for the key.

## Encapsulation

Identity underneath, annotations layered above, never mixed.

```
[ classification ]   type, cluster, labels, user annotations
[ sigil          ]   LSH + BLAKE3 — immutable, content-derived
```

Classification does not live in the sigil record. Re-typing a file, or a user
redefining what a type code means, must not change the file's identity. Two
consequences:

- Annotations are regenerable. Delete the whole classification table and
  rebuild it without re-embedding anything — embedding is the expensive step,
  classification is cheap.
- Different consumers can maintain different wrappers over the same sigils. A
  photographer's classification and a lawyer's coexist as separate tables over
  one index.

The 16-bit category field is user-definable rather than a hardcoded enum. There
is no universal taxonomy that fits, so the *meaning* of each code is a per-store
declaration in a libtab table. sigilfs never needs to know what a code means; it
filters on equality.

## Cluster identity

A cluster's ID is a hash of its sorted member LSH codes. Identity is
content-derived rather than assigned: no counter, no coordination, and two
nodes clustering the same corpus independently arrive at the same IDs. Outliers
get singleton clusters, which removes the "unclassified" special case.

Sorting before hashing matters — otherwise the same cluster hashes differently
depending on walk order.

Because the ID covers all members, adding one file changes it. That churn is
handled by **chaining**: each cluster records its parent, so a cached label can
follow the chain forward instead of being invalidated. Merges give a node
multiple parents, so this is a DAG rather than a chain.

Chain identity, average locality. Cluster IDs are pure identity, so avalanche
is harmless and lineage is valuable. LSH codes are the opposite: hashing them
together destroys the locality that makes them useful, since one flipped input
bit changes half the output. To combine chunk codes, average the float vectors
before projecting, or take a per-bit majority vote — never hash them.

## Graph structure

Paragraphs are nodes; the graph is reassembled from columns rather than stored
as adjacency lists. `path` gives containment, `para` gives ordering, `cluster`
gives grouping. Reassembly is a sort.

Similarity edges stay **derived**, computed by the scan on demand. Not because
libtab could not hold them, but because they are a function of a threshold that
should stay free per query — and in a large corpus the pair count is unbounded.
The scan runs at 15-20 GB/s, so `/similar/<hex>/` is a query, not a lookup.

Sequence, containment, and provenance edges are O(N), sparse, and worth
persisting. Similarity edges are none of those things.

## Tables

```
schema=sigil      path  para  hash  lsh  cluster
schema=cluster    id  parent  [label]
schema=mount      name  root  last-scan
schema=category   code  name  [pattern]
```

The `schema=` tuple also carries `model_id`, `embed_dim`, `simhash_seed`,
`lsh_bits`, and `federation`. A store whose parameters do not match the running
configuration must refuse to open rather than return wrong answers — bits made
under different parameters are not comparable, and that failure is silent
otherwise.

## Namespace

Plan 9 conventions: everything is a file, control by writing to `/ctl`, results
by reading. No RPC layer, no ioctls.

```
/ctl                  write: mount, scan, thresh
/stats                read:  counts, scan timings
/sigil/<hex>          the paragraph with that content ID
/similar/<hex>/       synthetic dir; walking it runs the scan
/class/<id>/          disjoint clusters
/mnt/<name>/          mounted sources, re-exported
```

sigilfs proxies reads of source content so the classifier never needs to know
where a file physically lives — it walks a cluster directory and reads the
files, and remote 9P sources work the same as local ones.

The subtlety is `Tread` on synthetic directories: 9P requires directory reads
be resumable at offsets previously returned, so a query result must be
materialized and pinned per-fid at open, not regenerated per read. Getting this
wrong makes `ls` silently truncate. Verify against `read(5)` and `walk(5)`
before implementing.

## Search

libtab is a write-ahead log and interchange format, not a query engine. Nothing
queries it directly; sigilfs appends, Solr indexes from it, users search Solr.

Similarity search in Solr uses **banded LSH**: split the 128-bit code into
bands, index each as a token in a multi-valued string field. Two codes within a
small Hamming distance share at least one band with high probability, so an OR
query over bands retrieves candidates through the inverted index. sigilfs then
re-ranks candidates by exact Hamming.

Solr narrows, sigil ranks. Neither pretends to be the other.

Prior art in `~/Repo/solr-plugins` already implements banded LSH this way, but
over MinHash of character shingles — lexical similarity, not semantic. The
retrieval plumbing carries over; the signature it indexes gets replaced. Band
geometry (8x16 vs 16x8) is a recall/candidate-count tradeoff and should be
measured with the existing eval harness rather than guessed.

## Order of work

1. 64-byte record with BLAKE3 — do this before any data exists on disk
2. libtab persistence, schema parameters, refuse-on-mismatch
3. Paragraph extraction and the indexer
4. 9P server
5. Clustering (union-find over scan results)
6. Classifier as a separate program
7. Solr, search UI, Keycloak

GPU offload is deferred until the application exists. `bench/scan_opencl.c`
measured 2.9x over the best CPU path for a resident store, which is worth
having later and worth nothing now.
