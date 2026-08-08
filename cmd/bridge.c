/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * The C11 side of sigilfs.
 *
 * plan9port's libc.h and C11's stdint.h both define the integer typedefs and
 * collide, so this file includes ONLY sigil.h and is compiled with the system
 * compiler. Everything it exports uses plain C types that both worlds agree
 * on, and the store travels as an opaque pointer.
 */

/* strdup is POSIX, not C11. Must precede every include. */
#define _POSIX_C_SOURCE 200809L

#include "sigil.h"

#include <stdlib.h>
#include <string.h>

/* Opaque to the plan9port side. */
struct bridge {
	sigil_store_t st;
	char **paths;        /* [cap] parallel to store rows */
	unsigned *paras;
	size_t n, cap;
};

void *
br_new(void)
{
	struct bridge *b = calloc(1, sizeof *b);

	if (b == NULL)
		return NULL;
	if (sigil_store_init(&b->st, 4096) != 0) {
		free(b);
		return NULL;
	}
	b->cap = 4096;
	b->paths = calloc(b->cap, sizeof *b->paths);
	b->paras = calloc(b->cap, sizeof *b->paras);
	if (b->paths == NULL || b->paras == NULL) {
		sigil_store_free(&b->st);
		free(b->paths); free(b->paras); free(b);
		return NULL;
	}
	return b;
}

void
br_free(void *p)
{
	struct bridge *b = p;
	size_t i;

	if (b == NULL)
		return;
	for (i = 0; i < b->n; i++)
		free(b->paths[i]);
	free(b->paths);
	free(b->paras);
	sigil_store_free(&b->st);
	free(b);
}

static int
grow(struct bridge *b)
{
	size_t cap = b->cap * 2;
	char **np = realloc(b->paths, cap * sizeof *np);
	unsigned *nq;

	if (np == NULL)
		return -1;
	b->paths = np;
	nq = realloc(b->paras, cap * sizeof *nq);
	if (nq == NULL)
		return -1;
	b->paras = nq;
	b->cap = cap;
	return 0;
}

/*
 * Add one paragraph. The LSH bits come from the caller because embedding lives
 * outside this process; passing NULL yields the byte-shingle fallback, which
 * is NOT semantic and exists only so the ingestion path can be exercised
 * without a model.
 */
long
br_add(void *p, const char *path, unsigned para, const void *text, size_t len,
       const unsigned long long *lsh, unsigned timestamp)
{
	struct bridge *b = p;
	sigil_t s;
	int i;

	if (b->n == b->cap && grow(b) != 0)
		return -1;

	sigil_generate_para(text, len, para, timestamp, 0, NULL, &s);
	if (lsh != NULL)
		for (i = 0; i < SIGIL_LSH_WORDS; i++)
			s.lsh[i] = lsh[i];

	if (sigil_store_push(&b->st, &s) < 0)
		return -1;
	b->paths[b->n] = strdup(path ? path : "");
	b->paras[b->n] = para;
	b->n++;
	return (long)(b->n - 1);
}

long
br_count(void *p)
{
	struct bridge *b = p;

	return b == NULL ? 0 : (long)b->n;
}

const char *
br_path(void *p, long i)
{
	struct bridge *b = p;

	return (b == NULL || i < 0 || (size_t)i >= b->n) ? NULL : b->paths[i];
}

unsigned
br_para(void *p, long i)
{
	struct bridge *b = p;

	return (b == NULL || i < 0 || (size_t)i >= b->n) ? 0 : b->paras[i];
}

/* Hex of the BLAKE3, which is how a record is addressed in the namespace. */
int
br_hash(void *p, long i, char *out, size_t outlen)
{
	struct bridge *b = p;
	sigil_t s;
	static const char hex[] = "0123456789abcdef";
	int j;

	if (b == NULL || outlen < SIGIL_HASH_LEN * 2 + 1)
		return -1;
	if (sigil_store_get(&b->st, (size_t)i, &s) != 0)
		return -1;
	for (j = 0; j < SIGIL_HASH_LEN; j++) {
		out[j*2]   = hex[s.hash[j] >> 4];
		out[j*2+1] = hex[s.hash[j] & 15];
	}
	out[SIGIL_HASH_LEN*2] = '\0';
	return 0;
}

int
br_lsh_hex(void *p, long i, char *out, size_t outlen)
{
	struct bridge *b = p;
	sigil_t s;
	static const char hex[] = "0123456789abcdef";
	int w, j;

	if (b == NULL || outlen < SIGIL_LSH_WORDS * 16 + 1)
		return -1;
	if (sigil_store_get(&b->st, (size_t)i, &s) != 0)
		return -1;
	for (w = 0; w < SIGIL_LSH_WORDS; w++)
		for (j = 0; j < 16; j++)
			out[w*16 + j] = hex[(s.lsh[w] >> (60 - 4*j)) & 15];
	out[SIGIL_LSH_WORDS*16] = '\0';
	return 0;
}

/* Index of the record with this hex hash, or -1. Linear: the namespace
 * resolves a path component at most once per walk. */
long
br_find_hash(void *p, const char *hexhash)
{
	struct bridge *b = p;
	char buf[SIGIL_HASH_LEN * 2 + 1];
	size_t i;

	if (b == NULL)
		return -1;
	for (i = 0; i < b->n; i++)
		if (br_hash(b, (long)i, buf, sizeof buf) == 0 &&
		    strcmp(buf, hexhash) == 0)
			return (long)i;
	return -1;
}

/*
 * Neighbours of record i within max_distance, written to out[] as indices.
 * This is the measured mechanism: a neighbourhood, not a partition. Clusters
 * formed by merging through neighbours chain across the whole corpus -- at
 * float32, connected components put 2018 of 5000 items in one group -- so
 * nothing here merges.
 */
long
br_similar(void *p, long i, unsigned max_distance, unsigned *out, long max_out)
{
	struct bridge *b = p;
	sigil_t s;
	size_t n;

	if (b == NULL || i < 0 || (size_t)i >= b->n)
		return 0;
	if (sigil_store_get(&b->st, (size_t)i, &s) != 0)
		return 0;
	n = sigil_scan_similar_simd(&b->st, s.lsh, max_distance,
	                            out, (size_t)max_out);
	return (long)n;
}

/*
 * Add a record whose LSH is already known, as hex. Used when loading a store:
 * the bits were computed once at index time and must not be recomputed, both
 * because embedding is expensive and because a different model version would
 * silently produce different bits for the same text.
 */
long
br_add_hex(void *p, const char *path, unsigned para, const char *lshhex)
{
    struct bridge *b = p;
    unsigned long long lsh[SIGIL_LSH_WORDS];
    int w, j, c;

    memset(lsh, 0, sizeof lsh);
    if (lshhex != NULL && strlen(lshhex) >= SIGIL_LSH_WORDS * 16) {
        for (w = 0; w < SIGIL_LSH_WORDS; w++) {
            for (j = 0; j < 16; j++) {
                c = lshhex[w*16 + j];
                c = (c >= '0' && c <= '9') ? c - '0'
                  : (c >= 'a' && c <= 'f') ? c - 'a' + 10
                  : (c >= 'A' && c <= 'F') ? c - 'A' + 10 : 0;
                lsh[w] = (lsh[w] << 4) | (unsigned)c;
            }
        }
    }
    /* The path is hashed as a stand-in for content: the text is not reloaded,
     * and a record still needs a stable identity to be addressed by. */
    return br_add(b, path, para, path, strlen(path ? path : ""), lsh, 0);
}
