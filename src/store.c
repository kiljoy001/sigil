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

int sigil_store_init(sigil_store_t *st, size_t capacity)
{
	memset(st, 0, sizeof(*st));
	if (capacity == 0)
		capacity = 1024;

	st->lsh       = alloc_aligned(capacity * SIGIL_LSH_WORDS * sizeof(uint64_t));
	st->para      = alloc_aligned(capacity * sizeof(uint32_t));
	st->cluster   = alloc_aligned(capacity * sizeof(uint32_t));
	st->timestamp = alloc_aligned(capacity * sizeof(uint32_t));
	st->category  = alloc_aligned(capacity * sizeof(uint16_t));
	st->trits     = alloc_aligned(capacity * sizeof(uint16_t));
	st->hash      = alloc_aligned(capacity * SIGIL_HASH_LEN);

	if (!st->lsh || !st->para || !st->cluster || !st->timestamp ||
	    !st->category || !st->trits || !st->hash) {
		sigil_store_free(st);
		return -1;
	}

	st->count    = 0;
	st->capacity = capacity;
	return 0;
}

void sigil_store_free(sigil_store_t *st)
{
	free(st->lsh);
	free(st->para);
	free(st->cluster);
	free(st->timestamp);
	free(st->category);
	free(st->trits);
	free(st->hash);
	memset(st, 0, sizeof(*st));
}

/* Grow every field array in step. Leaves the store untouched on failure. */
static int store_grow(sigil_store_t *st)
{
	size_t cap = st->capacity * 2;
	uint64_t *lsh = NULL;
	uint32_t *ts = NULL, *para = NULL, *clus = NULL;
	uint16_t *cat = NULL, *tr = NULL;
	uint8_t *hash = NULL;

	lsh  = alloc_aligned(cap * SIGIL_LSH_WORDS * sizeof(uint64_t));
	ts   = alloc_aligned(cap * sizeof(uint32_t));
	para = alloc_aligned(cap * sizeof(uint32_t));
	clus = alloc_aligned(cap * sizeof(uint32_t));
	cat  = alloc_aligned(cap * sizeof(uint16_t));
	tr   = alloc_aligned(cap * sizeof(uint16_t));
	hash = alloc_aligned(cap * SIGIL_HASH_LEN);

	if (!lsh || !ts || !para || !clus || !cat || !tr || !hash) {
		free(lsh); free(ts); free(para); free(clus);
		free(cat); free(tr); free(hash);
		return -1;
	}

	memcpy(lsh,  st->lsh,       st->count * SIGIL_LSH_WORDS * sizeof(uint64_t));
	memcpy(ts,   st->timestamp, st->count * sizeof(uint32_t));
	memcpy(para, st->para,      st->count * sizeof(uint32_t));
	memcpy(clus, st->cluster,   st->count * sizeof(uint32_t));
	memcpy(cat,  st->category,  st->count * sizeof(uint16_t));
	memcpy(tr,   st->trits,     st->count * sizeof(uint16_t));
	memcpy(hash, st->hash,      st->count * SIGIL_HASH_LEN);

	free(st->lsh); free(st->timestamp); free(st->para); free(st->cluster);
	free(st->category); free(st->trits); free(st->hash);

	st->lsh = lsh; st->timestamp = ts; st->para = para; st->cluster = clus;
	st->category = cat; st->trits = tr; st->hash = hash;
	st->capacity = cap;
	return 0;
}

ptrdiff_t sigil_store_push(sigil_store_t *st, const sigil_t *s)
{
	size_t i;

	if (st->count == st->capacity && store_grow(st) != 0)
		return -1;

	i = st->count;
	memcpy(st->lsh + i * SIGIL_LSH_WORDS, s->lsh,
	       SIGIL_LSH_WORDS * sizeof(uint64_t));
	st->para[i]      = s->para;
	st->cluster[i]   = s->cluster;
	st->timestamp[i] = s->timestamp;
	st->category[i]  = s->category;
	st->trits[i]     = s->trits;
	memcpy(st->hash + i * SIGIL_HASH_LEN, s->hash, SIGIL_HASH_LEN);

	st->count++;
	return (ptrdiff_t)i;
}

int sigil_store_get(const sigil_store_t *st, size_t i, sigil_t *out)
{
	if (i >= st->count)
		return -1;

	memcpy(out->hash, st->hash + i * SIGIL_HASH_LEN, SIGIL_HASH_LEN);
	memcpy(out->lsh, st->lsh + i * SIGIL_LSH_WORDS,
	       SIGIL_LSH_WORDS * sizeof(uint64_t));
	out->para      = st->para[i];
	out->cluster   = st->cluster[i];
	out->timestamp = st->timestamp[i];
	out->category  = st->category[i];
	out->trits     = st->trits[i];
	return 0;
}
