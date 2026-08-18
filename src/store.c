/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Struct-of-arrays store.
 *
 * Fields live in separate aligned arrays so a scan touches only the field it
 * filters on. The 8x reduction in memory traffic for an LSH-only pass is the
 * entire performance argument; keeping records interleaved would defeat it.
 */

/* posix_memalign is POSIX.1-2001, not C11; -std=c11 hides it without this.
 * Must precede every include. */
#define _POSIX_C_SOURCE 200112L

#include "sigil.h"

#include <stdlib.h>
#include <string.h>

#define SIGIL_ALIGN 32 /* AVX2 load width */

static void *alloc_aligned(size_t bytes)
{
	void *p = NULL;
	/* posix_memalign requires a multiple of the alignment. */
	size_t rounded = (bytes + SIGIL_ALIGN - 1) & ~(size_t)(SIGIL_ALIGN - 1);

	if (rounded == 0)
		rounded = SIGIL_ALIGN;
	if (posix_memalign(&p, SIGIL_ALIGN, rounded) != 0)
		return NULL;
	return p;
}

size_t sigil_store_segment_bytes(void)
{
	return SIGIL_SEG_RECS * SIGIL_LSH_WORDS * sizeof(uint64_t);
}

/*
 * Grow the seven directories to hold at least `want` entries.
 *
 * The directories themselves are ordinary arrays that do get reallocated,
 * and that is fine: they hold pointers, not data. At 68.0M records there are
 * 65 entries per directory — a few hundred bytes — so the copy this does is
 * nothing like the copy segmenting exists to remove, and the *segments* it
 * points at never move.
 */
static int dirs_reserve(sigil_store_t *st, size_t want)
{
	size_t cap = st->segcap ? st->segcap : 16;
	void *p;

	if (want <= st->segcap)
		return 0;
	while (cap < want)
		cap *= 2;

#define GROW_DIR(field)                                                 \
	do {                                                            \
		p = realloc(st->field, cap * sizeof *st->field);        \
		if (p == NULL)                                          \
			return -1;                                      \
		st->field = p;                                          \
	} while (0)

	GROW_DIR(lsh);
	GROW_DIR(para);
	GROW_DIR(cluster);
	GROW_DIR(timestamp);
	GROW_DIR(category);
	GROW_DIR(trits);
	GROW_DIR(hash);
#undef GROW_DIR

	st->segcap = cap;
	return 0;
}

/*
 * Append one segment to every field.
 *
 * All seven must succeed together: a store with six fields grown and one not
 * would write past the end of the short one on the next push. On failure the
 * partial set is released and the store is left exactly as it was, so the
 * caller loses the push and nothing else. That is the whole difference from
 * the flat version, which discarded the store.
 */
static int store_grow(sigil_store_t *st)
{
	uint64_t *lsh = NULL;
	uint32_t *para = NULL, *clus = NULL, *ts = NULL;
	uint16_t *cat = NULL, *tr = NULL;
	uint8_t *hash = NULL;

	if (dirs_reserve(st, st->nseg + 1) != 0)
		return -1;

	lsh  = alloc_aligned(SIGIL_SEG_RECS * SIGIL_LSH_WORDS * sizeof(uint64_t));
	para = alloc_aligned(SIGIL_SEG_RECS * sizeof(uint32_t));
	clus = alloc_aligned(SIGIL_SEG_RECS * sizeof(uint32_t));
	ts   = alloc_aligned(SIGIL_SEG_RECS * sizeof(uint32_t));
	cat  = alloc_aligned(SIGIL_SEG_RECS * sizeof(uint16_t));
	tr   = alloc_aligned(SIGIL_SEG_RECS * sizeof(uint16_t));
	hash = alloc_aligned(SIGIL_SEG_RECS * SIGIL_HASH_LEN);

	if (!lsh || !para || !clus || !ts || !cat || !tr || !hash) {
		free(lsh); free(para); free(clus); free(ts);
		free(cat); free(tr); free(hash);
		return -1;
	}

	st->lsh[st->nseg]       = lsh;
	st->para[st->nseg]      = para;
	st->cluster[st->nseg]   = clus;
	st->timestamp[st->nseg] = ts;
	st->category[st->nseg]  = cat;
	st->trits[st->nseg]     = tr;
	st->hash[st->nseg]      = hash;
	st->nseg++;
	st->capacity = st->nseg * SIGIL_SEG_RECS;
	return 0;
}

int sigil_store_init(sigil_store_t *st, size_t capacity)
{
	size_t want;

	memset(st, 0, sizeof(*st));

	/* capacity is a hint. Rounded up to whole segments, and always at
	 * least one so a fresh store can take a push without growing. */
	want = (capacity + SIGIL_SEG_RECS - 1) / SIGIL_SEG_RECS;
	if (want == 0)
		want = 1;
	if (dirs_reserve(st, want) != 0) {
		sigil_store_free(st);
		return -1;
	}
	while (st->nseg < want) {
		if (store_grow(st) != 0) {
			sigil_store_free(st);
			return -1;
		}
	}
	return 0;
}

void sigil_store_free(sigil_store_t *st)
{
	size_t i;

	for (i = 0; i < st->nseg; i++) {
		free(st->lsh[i]);
		free(st->para[i]);
		free(st->cluster[i]);
		free(st->timestamp[i]);
		free(st->category[i]);
		free(st->trits[i]);
		free(st->hash[i]);
	}
	free(st->lsh);
	free(st->para);
	free(st->cluster);
	free(st->timestamp);
	free(st->category);
	free(st->trits);
	free(st->hash);
	memset(st, 0, sizeof(*st));
}

/* Which segment record i lives in, and where inside it. A power-of-two
 * segment makes both a shift and a mask rather than a division. */
#define SEG(i) ((i) >> SIGIL_SEG_SHIFT)
#define OFF(i) ((i) & SIGIL_SEG_MASK)

ptrdiff_t sigil_store_push(sigil_store_t *st, const sigil_t *s)
{
	size_t i, g, o;

	if (st->count == st->capacity && store_grow(st) != 0)
		return -1;

	i = st->count;
	g = SEG(i);
	o = OFF(i);

	memcpy(st->lsh[g] + o * SIGIL_LSH_WORDS, s->lsh,
	       SIGIL_LSH_WORDS * sizeof(uint64_t));
	st->para[g][o]      = s->para;
	st->cluster[g][o]   = s->cluster;
	st->timestamp[g][o] = s->timestamp;
	st->category[g][o]  = s->category;
	st->trits[g][o]     = s->trits;
	memcpy(st->hash[g] + o * SIGIL_HASH_LEN, s->hash, SIGIL_HASH_LEN);

	st->count++;
	return (ptrdiff_t)i;
}

int sigil_store_get(const sigil_store_t *st, size_t i, sigil_t *out)
{
	size_t g, o;

	if (i >= st->count)
		return -1;
	g = SEG(i);
	o = OFF(i);

	memcpy(out->hash, st->hash[g] + o * SIGIL_HASH_LEN, SIGIL_HASH_LEN);
	memcpy(out->lsh, st->lsh[g] + o * SIGIL_LSH_WORDS,
	       SIGIL_LSH_WORDS * sizeof(uint64_t));
	out->para      = st->para[g][o];
	out->cluster   = st->cluster[g][o];
	out->timestamp = st->timestamp[g][o];
	out->category  = st->category[g][o];
	out->trits     = st->trits[g][o];
	return 0;
}

const uint64_t *sigil_store_lsh_ptr(const sigil_store_t *st, size_t i)
{
	if (i >= st->count)
		return NULL;
	return st->lsh[SEG(i)] + OFF(i) * SIGIL_LSH_WORDS;
}
