/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * sigil - content-addressed semantic filesystem
 *
 * A sigil is a 32-byte identifier that carries its own index. Identity and
 * queryable structure live in the same word, so similarity search is a linear
 * scan over packed fields rather than a lookup into a side index.
 *
 * There is no index to build, maintain, or corrupt. Writes are O(1).
 */

#ifndef SIGIL_H
#define SIGIL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Layout
 *
 * 64 bytes, one cache line, no padding.
 *
 *   [ 0..16)  LSH code       128 bits, SimHash over the paragraph embedding
 *   [16..48)  BLAKE3         256 bits, full width
 *   [48..52)  para index     0 = document-level, 1..n = paragraphs
 *   [52..56)  Unix timestamp
 *   [56..58)  category code  user-defined meaning, see docs/DESIGN.md
 *   [58..60)  packed trits   6 ternary semantic hints
 *   [60..64)  cluster ref    index into the cluster table
 *
 * The path is deliberately absent: it is variable-length and lives in the
 * libtab table alongside this record.
 *
 * On the unit. A sigil describes a paragraph, not a file. Whole-document
 * embedding forces one vector to represent everything a document says, and the
 * embedder truncates at 512 tokens regardless, so most of a long document was
 * being silently discarded. Paragraphs are the human-shaped unit — a sentence
 * fragments an idea, a fixed window cuts across one. para == 0 holds the
 * document-level average for coarse queries; 1..n locate the passage.
 *
 * On the hash. BLAKE3 at full 256 bits, replacing SHA-1. SHA-1 has
 * constructible collisions and runs at ~271 MB/s here; BLAKE3 is ~3 GB/s
 * portable and has no known weakness. Full width rather than truncated to 128:
 * truncation is unreachable by accident (~1.5e-21 at a billion paragraphs) but
 * only 2^64 work to attack deliberately, and collision resistance cannot be
 * retrofitted once stores exist. See docs/DESIGN.md on the threat model.
 *
 * On the LSH width. 32 bits measured 0.5495 recall@1 against a float32 ceiling
 * of 0.8140 on Quora Duplicate Questions — the compression was discarding a
 * third of what the embedding model knew. 128 bits reaches 0.7882, about 96.7%
 * of the ceiling. Past that the curve flattens: 256 bits buys 1.8 more points,
 * 512 buys 2.9. The remaining headroom is in the model, not the width.
 *
 * On the record size. Growing from 48 to 64 bytes does not affect scan speed.
 * Under the struct-of-arrays store below, records are never scanned as
 * records: a similarity pass walks the lsh[] array alone, whose stride is 16
 * bytes whatever the record's total width. An earlier version of this comment
 * argued 32 bytes was load-bearing because it is one cache line; that was
 * inherited from an array-of-structs design this code does not use, and it was
 * being used to reject exactly the widening that mattered most.
 *
 * 128 bits is also the common SIMD register width: one NEON or SSE register
 * exactly, two AVX2 lanes with no cross-lane fixups. See scan_x86.c,
 * scan_sse.c, scan_neon.c.
 * ------------------------------------------------------------------------ */

#define SIGIL_HASH_LEN 32
#define SIGIL_SIZE     64

/* LSH width. Changing this invalidates every sigil ever written: bits made at
 * one width are not comparable to bits made at another. */
#define SIGIL_LSH_BITS  128
#define SIGIL_LSH_WORDS (SIGIL_LSH_BITS / 64)

/* para index reserved for the document-level average of a file's paragraphs. */
#define SIGIL_PARA_DOC  0

typedef struct {
	uint64_t lsh[SIGIL_LSH_WORDS];   /* locality-sensitive bits       */
	uint8_t  hash[SIGIL_HASH_LEN];   /* BLAKE3-256 of the paragraph   */
	uint32_t para;                   /* 0 = document, 1..n = paragraph*/
	uint32_t timestamp;              /* seconds since epoch           */
	uint16_t category;               /* user-defined; see DESIGN.md   */
	uint16_t trits;                  /* packed, see trit.c            */
	uint32_t cluster;                /* index into the cluster table  */
} sigil_t;

/* Guards against silent padding changes; the on-disk format depends on it.
 * Spelled both ways because the OpenVINO backend is C++, where the C11
 * keyword does not exist. The assert is the point, not the spelling. */
#ifdef __cplusplus
static_assert(sizeof(sigil_t) == SIGIL_SIZE, "sigil_t must be exactly 64 bytes");
#else
_Static_assert(sizeof(sigil_t) == SIGIL_SIZE, "sigil_t must be exactly 64 bytes");
#endif

/* ---------------------------------------------------------------------------
 * Trits
 *
 * Six balanced-ternary hints packed into the two trailing bytes. Balanced
 * ternary is symmetric around zero, which suits hints that mean "less /
 * neutral / more" with no sign bit and no privileged direction.
 *
 * Packing is base-3, not 2-bits-per-trit: 3^6 = 729 values fit in 16 bits with
 * room to spare, and every one of the 729 is a legal decode. A 2-bit scheme
 * wastes a quarter of its encoding space on states that mean nothing, and a
 * corrupted byte silently reads back as a valid trit. Base-3 lets
 * sigil_trits_valid() reject corruption instead of propagating it.
 * ------------------------------------------------------------------------ */

typedef enum {
	TRIT_NEG =  -1,
	TRIT_ZERO =  0,
	TRIT_POS  =  1
} trit_t;

#define SIGIL_NTRITS   6
#define SIGIL_TRIT_MAX 729 /* 3^6 — exclusive upper bound on packed values */

/*
 * Two hints are named; the remaining four are reserved semantic dimensions,
 * indexed rather than named because their meanings are not settled. Naming
 * them later is a source change here and nowhere else.
 */
typedef struct {
	trit_t thermal;             /* access frequency: cold / warm / hot   */
	trit_t quality;             /* confidence in the categorization      */
	trit_t sim[4];              /* reserved semantic dimensions          */
} sigil_trits_t;

/* Pack six trits into base-3. Always succeeds; result is < SIGIL_TRIT_MAX. */
uint16_t sigil_trits_pack(const sigil_trits_t *t);

/* Unpack. Returns 0 on success, -1 if packed >= SIGIL_TRIT_MAX (corruption). */
int sigil_trits_unpack(uint16_t packed, sigil_trits_t *out);

/* Cheap validity check without unpacking. */
static inline int sigil_trits_valid(uint16_t packed)
{
	return packed < SIGIL_TRIT_MAX;
}

/* ---------------------------------------------------------------------------
 * Construction
 * ------------------------------------------------------------------------ */

/* Build a document-level sigil (para == SIGIL_PARA_DOC) from content.
 * Computes the BLAKE3 hash and LSH bits internally. */
void sigil_generate(const void *content, size_t len, uint32_t timestamp,
                    uint16_t category, const sigil_trits_t *trits,
                    sigil_t *out);

/* As above, for a numbered paragraph. */
void sigil_generate_para(const void *content, size_t len, uint32_t para,
                         uint32_t timestamp, uint16_t category,
                         const sigil_trits_t *trits, sigil_t *out);

/* Hex-encode to buf (needs >= SIGIL_SIZE*2+1 bytes). */
void sigil_to_hex(const sigil_t *s, char *buf);

/* Parse SIGIL_SIZE*2 hex chars. Returns 0 on success, -1 on malformed input. */
int sigil_from_hex(const char *hex, sigil_t *out);

/* ---------------------------------------------------------------------------
 * Store: segmented struct-of-arrays
 *
 * Sigils are stored decomposed by field, not as an array of structs. A
 * similarity pass over N records touches 4N bytes of LSH words instead of 32N
 * bytes of whole sigils — an 8x reduction in memory traffic, which is the
 * binding constraint. Full records are reassembled only for survivors.
 *
 * Each field is a list of fixed-size segments rather than one contiguous
 * array. Growth appends a segment; nothing already written is copied or
 * moved.
 *
 * The flat version doubled by allocating a second set of all seven arrays,
 * copying into it, and freeing the first — both live at once, so the peak was
 * old + new, 1.5x the final size. Measured on an 8.4M record fill, virtual
 * size went 297.5 MB -> 601.5 MB across the last doubling; at 68.0M
 * paragraphs the transient is tens of gigabytes. Worse, it asked for seven
 * large contiguous blocks and discarded the store if any one failed, which
 * fragmentation can cause while ample memory is free.
 *
 * A segment never moves once allocated, so a pointer into it stays valid for
 * the life of the store. That is what makes it possible for a sealed segment
 * to later be a mapping of the store file rather than heap memory: the flat
 * array could never promise it, because every doubling invalidated every
 * address.
 *
 * Segments are 32-byte aligned so vector loads stay aligned on both AVX2
 * (32-byte) and NEON (16-byte), and hold a power-of-two record count so
 * indexing is a shift and a mask.
 * ------------------------------------------------------------------------ */

/* Records per segment. 1 << 20 puts one LSH segment at 16 MB and one hash
 * segment at 32 MB — large enough that the directory stays small at corpus
 * scale (68.0M records is 65 segments), small enough that a single failed
 * allocation costs little. */
#define SIGIL_SEG_SHIFT 20
#define SIGIL_SEG_RECS  ((size_t)1 << SIGIL_SEG_SHIFT)
#define SIGIL_SEG_MASK  (SIGIL_SEG_RECS - 1)

typedef struct {
	/* Segment directories. Each entry points at SIGIL_SEG_RECS records'
	 * worth of that field; nseg entries are live. */
	uint64_t **lsh;       /* [nseg][SIGIL_SEG_RECS * SIGIL_LSH_WORDS] */
	uint32_t **para;
	uint32_t **cluster;
	uint32_t **timestamp;
	uint16_t **category;
	uint16_t **trits;
	uint8_t  **hash;      /* [nseg][SIGIL_SEG_RECS * SIGIL_HASH_LEN] */
	size_t    nseg;       /* segments allocated */
	size_t    segcap;     /* entries the directories can hold */
	size_t    count;
	size_t    capacity;   /* nseg * SIGIL_SEG_RECS */

	/* Set when the segments are mappings of a file rather than heap. The
	 * directories are still heap; only what they point at is mapped, so
	 * sigil_store_free() would call free() on a mapping. Release a mapped
	 * store with sigil_store_unmap() instead. */
	void     *map;        /* NULL when heap-backed */
	size_t    maplen;
} sigil_store_t;

int  sigil_store_init(sigil_store_t *st, size_t capacity);
void sigil_store_free(sigil_store_t *st);

/* Append one sigil. Grows geometrically. Returns index, or -1 on alloc failure. */
ptrdiff_t sigil_store_push(sigil_store_t *st, const sigil_t *s);

/* Reassemble record i. Returns 0 on success, -1 if i is out of range. */
int sigil_store_get(const sigil_store_t *st, size_t i, sigil_t *out);

/* Bytes in one LSH segment. The unit the store grows by, and what a caller
 * measuring the growth transient compares against. */
size_t sigil_store_segment_bytes(void);

/* ---------------------------------------------------------------------------
 * Mapped stores
 *
 * A store can be backed by a file instead of the heap. Segments become
 * mappings of that file, so opening one costs no parsing, no allocation and
 * no copy: the SoA arrays *are* the file's bytes, and the kernel pages in
 * only what a scan touches. Corpus size stops being bounded by RAM — a store
 * larger than physical memory works, with clean file-backed pages evicted
 * and re-read under pressure rather than the process dying.
 *
 * This is possible only because segments never move. The flat array could
 * not have been mapped: every doubling invalidated every address.
 *
 * The mapped file is a *derived cache*, not the durable record. libtab
 * remains the portable, human-readable store; this is a raw dump whose
 * layout is the in-memory layout, so it bakes in endianness, SIGIL_SEG_RECS
 * and SIGIL_LSH_BITS. The header records those and sigil_store_map() refuses
 * a mismatch rather than reinterpreting bytes written under other rules. If
 * the layout changes, delete and rebuild rather than migrate.
 * ------------------------------------------------------------------------ */

/* Write st to path as a mapped store. Returns 0, or -1 with errno set. */
int sigil_store_save(const sigil_store_t *st, const char *path);

/* Open path as a read-only mapped store. The store borrows the mapping and
 * must be released with sigil_store_unmap(), not sigil_store_free().
 * Returns 0, or -1 on a bad header, a layout mismatch, or an I/O error. */
int sigil_store_map(sigil_store_t *st, const char *path);

/* Release a mapped store. */
void sigil_store_unmap(sigil_store_t *st);

/* ---------------------------------------------------------------------------
 * Stage-two refinement: a sidecar store
 *
 * The stage-one sigil is identity and index, and it never goes away. A
 * sidecar holds a second representation for a subset of records, so a query
 * can rescore the candidates stage one selected using something more
 * accurate than 128 bits.
 *
 * The payload is either float vectors or a wider binary code. Floats are
 * what rerank uses and are the measured winner: on 200,064 records and 1,191
 * queries, reordering a 200-record binary shortlist by float cosine moved
 * R@1 from 0.0269 to 0.0806 -- 3.0x, and 80% of the way to the float32
 * ceiling of 0.1008 -- for 18.63 ms/query against the bare scan's 18.56.
 * The reduction has already happened, so the float work is 0.4% overhead.
 *
 * Asymmetric distance (keep the query as floats, compare against the stored
 * bits) was measured against the same data: 0.0411 R@1, better than binary
 * and needing no extra storage, but half of rerank's gain at 7x the time.
 * It is the answer when the vectors will not fit; here they map.
 *
 * Sparse by construction: stage one does the reduction, so the sidecar
 * covers only the records worth a second look. Each entry is (index, code)
 * with indices ascending, so a lookup is a binary search and a full pass is
 * sequential.
 *
 * It is a separate file rather than a wider record on purpose. sigil_t is a
 * fixed 64 bytes with a _Static_assert on it and an on-disk format that
 * depends on that size; a second code would either break the record or make
 * every store pay for a stage most will never run. A sidecar is optional --
 * a store either has one or it does not -- and deleting it costs nothing but
 * the recomputation.
 *
 * The sidecar records the model that produced it and the count of the store
 * it refines. Codes from two different models are not comparable, and an
 * index into the wrong store is not an error that announces itself, so
 * sigil_side_map() refuses both rather than returning plausible neighbours.
 * ------------------------------------------------------------------------ */

#define SIGIL_SIDE_MODEL_MAX 64

/* What a sidecar carries. Recorded in the file so a reader cannot
 * interpret float bytes as a bit pattern, which produces a number rather
 * than an error. */
typedef enum {
	SIGIL_SIDE_CODE  = 1,   /* uint64_t words, compared by Hamming */
	SIGIL_SIDE_FLOAT = 2    /* float vector, compared by cosine    */
} sigil_side_kind;

typedef struct {
	const uint32_t *index;   /* [count] store indices, ascending */
	const uint64_t *code;    /* [count * words], when kind == CODE  */
	const float    *vec;     /* [count * dim],   when kind == FLOAT */
	size_t count;
	sigil_side_kind kind;
	size_t words;            /* 64-bit words per code  (CODE)  */
	size_t bits;             /* words * 64             (CODE)  */
	size_t dim;              /* floats per vector      (FLOAT) */
	char   model[SIGIL_SIDE_MODEL_MAX];
	uint64_t base_count;     /* count of the store this refines */
	void  *map;
	size_t maplen;
} sigil_side_t;

/* ---------------------------------------------------------------------------
 * Building a float sidecar
 *
 * Vectors arrive one embedding batch at a time and there can be tens of
 * millions of them: the corpus at dim 384 is 85 GB of float32, which does
 * not fit in memory. So the builder grows the same way the store does --
 * fixed segments, appended, never copied -- and writes them out segment by
 * segment.
 *
 * An earlier version accumulated into one realloc'd array and wrote at the
 * end. That is the same materialise-then-process shape that made
 * store_commit need 54 GB to write 12 GB, and it would have died about six
 * hours into a seven-hour run.
 * ------------------------------------------------------------------------ */

typedef struct {
	float  **seg;        /* [nseg][SIGIL_SEG_RECS * dim] */
	uint32_t **idx;      /* [nseg][SIGIL_SEG_RECS] store indices */
	size_t   nseg, segcap;
	size_t   count;
	size_t   dim;
} sigil_sidebuild_t;

int  sigil_sidebuild_init(sigil_sidebuild_t *sb, size_t dim);
void sigil_sidebuild_free(sigil_sidebuild_t *sb);

/* Append one vector for store index i. Returns 0, or -1 on allocation
 * failure -- which costs one segment, not the run. */
int  sigil_sidebuild_add(sigil_sidebuild_t *sb, uint32_t i, const float *v);

/* Write the accumulated vectors as a sidecar, streaming segment by segment
 * so no copy of the whole set is ever made. */
int  sigil_sidebuild_save(const sigil_sidebuild_t *sb, const char *path,
                          const char *model, uint64_t base_count);

/* Write a code sidecar. `index` must be ascending; `code` is count*words. */
int sigil_side_save(const char *path, const uint32_t *index,
                    const uint64_t *code, size_t count, size_t words,
                    const char *model, uint64_t base_count);

/* Write a float sidecar. `vec` is count*dim floats, one vector per index. */
int sigil_side_save_vec(const char *path, const uint32_t *index,
                        const float *vec, size_t count, size_t dim,
                        const char *model, uint64_t base_count);

/* The float vector for store index i, or NULL if i has no entry or this
 * sidecar carries codes rather than vectors. */
const float *sigil_side_vec(const sigil_side_t *sd, uint32_t i);

/* Cosine similarity between two vectors of this sidecar's dimension.
 * Higher is nearer -- the opposite sense to Hamming, which is why rerank
 * sorts descending where the scan sorts ascending. */
double sigil_side_cosine(const sigil_side_t *sd, const float *a,
                         const float *b);

/*
 * Rerank: reorder `cand` (the scan's output, `n` store indices) by cosine
 * against `query` under the sidecar, best first. Indices with no sidecar
 * entry keep their scan order after every reranked one, rather than being
 * dropped -- a missing vector is a gap in stage two, not evidence about the
 * record.
 *
 * Returns the number reordered. `n` is expected to be the shortlist the scan
 * produced, not the corpus: the whole economy of this is that stage one did
 * the reduction first.
 */
size_t sigil_side_rerank(const sigil_side_t *sd, const float *query,
                         uint32_t *cand, size_t n);

/* Open a sidecar read-only. Refuses a code width or base count that does not
 * match `st`, and a model name that does not match `model` when non-NULL. */
int sigil_side_map(sigil_side_t *sd, const char *path,
                   const sigil_store_t *st, const char *model);

void sigil_side_unmap(sigil_side_t *sd);

/* The refined code for store index i, or NULL if i has no sidecar entry.
 * Binary search over the ascending index. */
const uint64_t *sigil_side_lookup(const sigil_side_t *sd, uint32_t i);

/* Hamming distance between two refined codes of this sidecar's width. */
uint32_t sigil_side_hamming(const sigil_side_t *sd, const uint64_t *a,
                            const uint64_t *b);

/* Hint that segment g will not be touched again, so the kernel may drop its
 * pages now rather than waiting for memory pressure. A hint only: dropping
 * clean file-backed pages is always safe, and they are re-read on the next
 * touch. Returns 0, or -1 if the store is not mapped. */
int sigil_store_release(const sigil_store_t *st, size_t g);

/* Address of record i's LSH words, or NULL if i is out of range. Stable for
 * the life of the store: segments never move. Exposed because that stability
 * is a promise of the structure, and test/segments.c is what holds it. */
const uint64_t *sigil_store_lsh_ptr(const sigil_store_t *st, size_t i);

/* ---------------------------------------------------------------------------
 * Scan kernels
 *
 * Each kernel has a scalar reference and a SIMD implementation (AVX2 on
 * x86-64, NEON on aarch64) that must agree with it bit-for-bit;
 * test/differential.c enforces this over random input.
 * SIMD bugs here do not crash, they return subtly wrong distances that look
 * plausible forever, so the scalar twin is the only real check.
 *
 * All kernels write matching indices into out[] (caller-allocated, capacity
 * max_out) and return the number written.
 *
 * A kernel works on a sigil_view_t: flat field pointers and a count, which is
 * what it always operated on in practice. The store became a list of
 * segments, and a view is one contiguous piece of one segment, so segment
 * traversal lives in sigil_scan_walk() and none of it reached the four
 * kernels. Indices a kernel returns are relative to its view; the walker
 * rebases them.
 * ------------------------------------------------------------------------ */

typedef struct {
	const uint64_t *lsh;
	const uint32_t *para;
	const uint32_t *cluster;
	const uint32_t *timestamp;
	const uint16_t *category;
	const uint16_t *trits;
	const uint8_t  *hash;
	size_t          count;
} sigil_view_t;

typedef size_t (*sigil_scan_fn)(const sigil_view_t *v, const void *arg,
                                uint32_t *out, size_t max_out);

typedef struct {
	const uint64_t *query;
	uint32_t max_distance;
} sigil_simarg_t;

typedef struct {
	uint32_t start, end;
} sigil_timearg_t;

/* Walk [lo, hi) segment by segment, applying fn to each piece. */
size_t sigil_scan_walk(const sigil_store_t *st, size_t lo, size_t hi,
                       sigil_scan_fn fn, const void *arg,
                       uint32_t *out, size_t max_out);

/* The kernels. `_scalar` is the definition of correct; the unsuffixed name
 * dispatches to the widest vector path the CPU offers. */
size_t sigil_kernel_similar(const sigil_view_t *v, const void *arg,
                            uint32_t *out, size_t max_out);
size_t sigil_kernel_timerange(const sigil_view_t *v, const void *arg,
                              uint32_t *out, size_t max_out);
size_t sigil_kernel_category(const sigil_view_t *v, const void *arg,
                             uint32_t *out, size_t max_out);
size_t sigil_kernel_similar_scalar(const sigil_view_t *v, const void *arg,
                                   uint32_t *out, size_t max_out);
size_t sigil_kernel_timerange_scalar(const sigil_view_t *v, const void *arg,
                                     uint32_t *out, size_t max_out);
size_t sigil_kernel_category_scalar(const sigil_view_t *v, const void *arg,
                                    uint32_t *out, size_t max_out);

/* Indices where the Hamming distance between lsh[i] and query (which points
 * to SIGIL_LSH_WORDS words) is <= max_distance. */
size_t sigil_scan_similar_scalar(const sigil_store_t *st, const uint64_t *query,
                                 uint32_t max_distance,
                                 uint32_t *out, size_t max_out);
size_t sigil_scan_similar_simd(const sigil_store_t *st, const uint64_t *query,
                               uint32_t max_distance,
                               uint32_t *out, size_t max_out);

/* Indices where start <= timestamp[i] <= end. */
size_t sigil_scan_timerange_scalar(const sigil_store_t *st,
                                   uint32_t start, uint32_t end,
                                   uint32_t *out, size_t max_out);
size_t sigil_scan_timerange_simd(const sigil_store_t *st,
                                 uint32_t start, uint32_t end,
                                 uint32_t *out, size_t max_out);

/* Indices where category[i] == category. */
size_t sigil_scan_category_scalar(const sigil_store_t *st, uint16_t category,
                                  uint32_t *out, size_t max_out);
size_t sigil_scan_category_simd(const sigil_store_t *st, uint16_t category,
                                uint32_t *out, size_t max_out);

/* ---------------------------------------------------------------------------
 * Ranged scans
 *
 * Same kernels restricted to [lo, hi). The scan is embarrassingly parallel —
 * each range reads a contiguous, disjoint slice of the SoA arrays with no
 * shared state — so a caller can split the store across threads and merge the
 * index lists afterwards.
 *
 * Returned indices are absolute, not range-relative, so merged results need no
 * fixup. Whether threading helps depends on the machine: the scan is
 * bandwidth-bound, so a core already near its memory ceiling gains little,
 * while a wide box with spare controllers gains a lot.
 *
 * libsigil stays single-threaded and dependency-free; threading is the
 * caller's to own. See test/bench_mt.c for both a thread-per-range and a
 * work-pool implementation over these.
 * ------------------------------------------------------------------------ */

size_t sigil_scan_similar_range(const sigil_store_t *st, const uint64_t *query,
                                uint32_t max_distance, size_t lo, size_t hi,
                                uint32_t *out, size_t max_out);
size_t sigil_scan_timerange_range(const sigil_store_t *st,
                                  uint32_t start, uint32_t end,
                                  size_t lo, size_t hi,
                                  uint32_t *out, size_t max_out);
size_t sigil_scan_category_range(const sigil_store_t *st, uint16_t category,
                                 size_t lo, size_t hi,
                                 uint32_t *out, size_t max_out);

/* Nonzero if the _simd kernels use real vector instructions on this build:
 * AVX2 (probed via CPUID) on x86-64, NEON (architectural baseline) on
 * aarch64, zero where they forward to the scalar reference. */
int sigil_have_simd(void);

/* Which vector paths the CPU offers: 2 = AVX2, 1 = SSE4.2, 0 = scalar.
 * Either pointer may be NULL. Reports what was available, not what the
 * dispatcher chose. */
int sigil_simd_paths(int *avx2, int *sse42);

/* Which kernel the runtime calibration actually chose: 2 = AVX2, 1 = SSE4.2,
 * 0 = scalar. CPUID reports what exists; this reports what measured fastest
 * on this machine, which is not always the widest. */
int sigil_simd_chosen(void);

/*
 * Force a kernel: 2 = AVX2, 1 = SSE4.2, 0 = scalar. Returns 0, or -1 if the
 * CPU does not offer it. SIGIL_SIMD_PATH=scalar|sse|avx2 does the same from
 * the environment.
 *
 * For tests and bug reports, not for production tuning -- the calibration
 * measures, and a guess rarely beats a measurement. It exists because the
 * dispatcher runs exactly one kernel per process, so without an override the
 * others are unreachable and the differential tests cannot prove they agree.
 * On non-x86 builds both are no-ops returning -1 and are harmless to call.
 */
int sigil_simd_force(int path);
void sigil_simd_unforce(void);

/* Hamming distance between two LSH codes, each SIGIL_LSH_WORDS words. */
static inline uint32_t sigil_hamming(const uint64_t *a, const uint64_t *b)
{
	uint32_t d = 0;

	for (int i = 0; i < SIGIL_LSH_WORDS; i++)
		d += (uint32_t)__builtin_popcountll(a[i] ^ b[i]);
	return d;
}

#ifdef __cplusplus
}
#endif

#endif /* SIGIL_H */
