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
 * 32 bytes, one cache line. The size is load-bearing: the scan is bound by
 * memory bandwidth, and a 33-byte record straddles lines and gives back the
 * locality the index-free design exists to exploit. Do not grow this struct.
 *
 *   [ 0..20)  SHA-1 content hash      identity
 *   [20..24)  LSH semantic bits       similarity (Hamming distance)
 *   [24..28)  Unix timestamp          time-range queries
 *   [28..30)  category code           classification
 *   [30..32)  packed trits            6 ternary semantic hints
 * ------------------------------------------------------------------------ */

#define SIGIL_HASH_LEN 20
#define SIGIL_SIZE     32

typedef struct {
	uint8_t  hash[SIGIL_HASH_LEN]; /* SHA-1 of content            */
	uint32_t lsh;                  /* locality-sensitive bits     */
	uint32_t timestamp;            /* seconds since epoch         */
	uint16_t category;             /* classification code         */
	uint16_t trits;                /* packed, see trit.c          */
} sigil_t;

/* The whole design rests on this. */
_Static_assert(sizeof(sigil_t) == SIGIL_SIZE, "sigil_t must be exactly 32 bytes");

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

/* Build a sigil from content. Computes the SHA-1 and LSH bits internally. */
void sigil_generate(const void *content, size_t len, uint32_t timestamp,
                    uint16_t category, const sigil_trits_t *trits,
                    sigil_t *out);

/* Hex-encode to buf (needs >= 65 bytes: 32 bytes * 2 + NUL). */
void sigil_to_hex(const sigil_t *s, char *buf);

/* Parse 64 hex chars. Returns 0 on success, -1 on malformed input. */
int sigil_from_hex(const char *hex, sigil_t *out);

/* ---------------------------------------------------------------------------
 * Store: struct-of-arrays
 *
 * Sigils are stored decomposed by field, not as an array of structs. A
 * similarity pass over N records touches 4N bytes of LSH words instead of 32N
 * bytes of whole sigils — an 8x reduction in memory traffic, which is the
 * binding constraint. Full records are reassembled only for survivors.
 *
 * Field arrays are 32-byte aligned so AVX2 loads stay aligned.
 * ------------------------------------------------------------------------ */

typedef struct {
	uint32_t *lsh;        /* [count] */
	uint32_t *timestamp;  /* [count] */
	uint16_t *category;   /* [count] */
	uint16_t *trits;      /* [count] */
	uint8_t  *hash;       /* [count * SIGIL_HASH_LEN], row-major */
	size_t    count;
	size_t    capacity;
} sigil_store_t;

int  sigil_store_init(sigil_store_t *st, size_t capacity);
void sigil_store_free(sigil_store_t *st);

/* Append one sigil. Grows geometrically. Returns index, or -1 on alloc failure. */
ptrdiff_t sigil_store_push(sigil_store_t *st, const sigil_t *s);

/* Reassemble record i. Returns 0 on success, -1 if i is out of range. */
int sigil_store_get(const sigil_store_t *st, size_t i, sigil_t *out);

/* ---------------------------------------------------------------------------
 * Scan kernels
 *
 * Each kernel has a scalar reference and an AVX2 implementation that must
 * agree bit-for-bit; test/differential.c enforces this over random input.
 * SIMD bugs here do not crash, they return subtly wrong distances that look
 * plausible forever, so the scalar twin is the only real check.
 *
 * All kernels write matching indices into out[] (caller-allocated, capacity
 * max_out) and return the number written.
 * ------------------------------------------------------------------------ */

/* Indices where popcount(lsh[i] ^ query) <= max_distance. */
size_t sigil_scan_similar_scalar(const sigil_store_t *st, uint32_t query,
                                 uint32_t max_distance,
                                 uint32_t *out, size_t max_out);
size_t sigil_scan_similar_avx2(const sigil_store_t *st, uint32_t query,
                               uint32_t max_distance,
                               uint32_t *out, size_t max_out);

/* Indices where start <= timestamp[i] <= end. */
size_t sigil_scan_timerange_scalar(const sigil_store_t *st,
                                   uint32_t start, uint32_t end,
                                   uint32_t *out, size_t max_out);
size_t sigil_scan_timerange_avx2(const sigil_store_t *st,
                                 uint32_t start, uint32_t end,
                                 uint32_t *out, size_t max_out);

/* Indices where category[i] == category. */
size_t sigil_scan_category_scalar(const sigil_store_t *st, uint16_t category,
                                  uint32_t *out, size_t max_out);
size_t sigil_scan_category_avx2(const sigil_store_t *st, uint16_t category,
                                uint32_t *out, size_t max_out);

/* Runtime dispatch: AVX2 where available, scalar otherwise. */
int sigil_have_avx2(void);

/* Hamming distance between two LSH words. */
static inline uint32_t sigil_hamming(uint32_t a, uint32_t b)
{
	return (uint32_t)__builtin_popcount(a ^ b);
}

#ifdef __cplusplus
}
#endif

#endif /* SIGIL_H */
