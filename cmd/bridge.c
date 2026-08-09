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
#include "sigil_embed.h"

#include <stdlib.h>
#include <string.h>

/* Opaque to the plan9port side. */
struct bridge {
	sigil_store_t st;
	char **paths;        /* [cap] parallel to store rows */
	unsigned *paras;
	/* Where the paragraph sits in its file. The store holds only the
	 * sigil, so without this a neighbour can be identified but not read,
	 * and the projection lands on a hash rather than on text. */
	unsigned long *offs;
	unsigned long *lens;
	size_t n, cap;

	/* Optional. Without it, records carry the byte-shingle fallback, which
	 * is not semantic -- so the server reports whether an embedder is
	 * loaded rather than letting a caller assume the bits mean something. */
	sigil_embedder_t *emb;
	sigil_simhash_t sh;
	int have_sh;
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
	b->offs  = calloc(b->cap, sizeof *b->offs);
	b->lens  = calloc(b->cap, sizeof *b->lens);
	if (b->paths == NULL || b->paras == NULL ||
	    b->offs == NULL || b->lens == NULL) {
		sigil_store_free(&b->st);
		free(b->paths); free(b->paras);
		free(b->offs); free(b->lens); free(b);
		return NULL;
	}
	return b;
}

/*
 * Load an embedding model. Returns 0 on success. The hyperplane seed must
 * match whatever produced any store being loaded alongside it: bits projected
 * through different hyperplanes are not comparable, which is why the store
 * records the seed and refuses a mismatch.
 */
int
br_embedder_load(void *p, const char *gguf, unsigned long long seed)
{
	struct bridge *b = p;

	if (b == NULL || gguf == NULL)
		return -1;
	if (b->emb != NULL)
		return 0;                        /* already loaded */
	b->emb = sigil_embedder_llama(gguf);
	if (b->emb == NULL)
		return -1;
	if (sigil_simhash_init(&b->sh, b->emb->dim(b->emb), seed) != 0) {
		b->emb->destroy(b->emb);
		b->emb = NULL;
		return -1;
	}
	b->have_sh = 1;
	return 0;
}

int
br_have_embedder(void *p)
{
	struct bridge *b = p;

	return (b != NULL && b->emb != NULL) ? 1 : 0;
}

unsigned
br_embed_dim(void *p)
{
	struct bridge *b = p;

	return (b == NULL || b->emb == NULL) ? 0 : (unsigned)b->emb->dim(b->emb);
}

void
br_free(void *p)
{
	struct bridge *b = p;
	size_t i;

	if (b == NULL)
		return;
	if (b->have_sh)
		sigil_simhash_free(&b->sh);
	if (b->emb != NULL)
		b->emb->destroy(b->emb);
	for (i = 0; i < b->n; i++)
		free(b->paths[i]);
	free(b->paths);
	free(b->paras);
	free(b->offs);
	free(b->lens);
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
	{
		unsigned long *no = realloc(b->offs, cap * sizeof *no);
		unsigned long *nl;
		if (no == NULL)
			return -1;
		b->offs = no;
		nl = realloc(b->lens, cap * sizeof *nl);
		if (nl == NULL)
			return -1;
		b->lens = nl;
	}
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
br_add_at(void *p, const char *path, unsigned para, const void *text,
          size_t len, const unsigned long long *lsh, unsigned timestamp,
          unsigned long off)
{
	struct bridge *b = p;
	sigil_t s;
	int i;

	if (b->n == b->cap && grow(b) != 0)
		return -1;

	sigil_generate_para(text, len, para, timestamp, 0, NULL, &s);

	if (lsh != NULL) {
		for (i = 0; i < SIGIL_LSH_WORDS; i++)
			s.lsh[i] = lsh[i];
	} else if (b->emb != NULL) {
		/* Real semantic bits. Failure here leaves the byte-shingle
		 * fallback in place rather than dropping the record: a
		 * paragraph that will not embed is still worth addressing by
		 * content, it just will not be found by similarity. */
		size_t dim = b->emb->dim(b->emb);
		float *v = malloc(dim * sizeof *v);

		if (v != NULL) {
			if (b->emb->embed(b->emb, (const char *)text, len, v) == 0)
				sigil_simhash_project(&b->sh, v, s.lsh);
			free(v);
		}
	}

	if (sigil_store_push(&b->st, &s) < 0)
		return -1;
	b->paths[b->n] = strdup(path ? path : "");
	b->paras[b->n] = para;
	b->offs[b->n] = off;
	b->lens[b->n] = (unsigned long)len;
	b->n++;
	return (long)(b->n - 1);
}

/* The one authority on the code width. cmd/ cannot include sigil.h, so
 * store.c had a hardcoded 256 that did not match SIGIL_LSH_BITS -- and that
 * value is written into the persisted parameters as the guarantee that two
 * stores are comparable. It was recording a width the data did not have. */
unsigned
br_lsh_bits(void)
{
	return SIGIL_LSH_BITS;
}

/* Offsets unknown: persistence replays records whose source may have moved,
 * and a wrong offset would serve the wrong text. Zero length means "not
 * recorded" and the reader says so. */
long
br_add(void *p, const char *path, unsigned para, const void *text, size_t len,
       const unsigned long long *lsh, unsigned timestamp)
{
	return br_add_at(p, path, para, text, len, lsh, timestamp, 0);
}

unsigned long
br_offset(void *p, long i)
{
	struct bridge *b = p;

	return (i >= 0 && (size_t)i < b->n) ? b->offs[i] : 0;
}

unsigned long
br_length(void *p, long i)
{
	struct bridge *b = p;

	return (i >= 0 && (size_t)i < b->n) ? b->lens[i] : 0;
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
br_add_restore(void *p, const char *path, unsigned para, const char *lshhex,
               const char *hashhex, unsigned long off, unsigned long len)
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
    {
        /* The stored hash is the record's identity and must survive the round
         * trip. Recomputing it from the path -- which this did -- gave a
         * record a different hash after a restart than the one written to
         * disk, so /similar/<hex>/ could not be reached by the hash the store
         * had just persisted. Content addressing means the address does not
         * change.
         *
         * Offsets are restored too, so a reloaded neighbour can still be read
         * back from its source file. */
        long i = br_add_at(b, path, para, path,
                           strlen(path ? path : ""), lsh, 0, off);
        int j, c, hi;

        if (i < 0)
            return -1;
        if (hashhex != NULL && strlen(hashhex) == SIGIL_HASH_LEN * 2) {
            for (j = 0; j < SIGIL_HASH_LEN; j++) {
                hi = 0;
                for (c = 0; c < 2; c++) {
                    int d = hashhex[j*2 + c];
                    d = (d >= '0' && d <= '9') ? d - '0'
                      : (d >= 'a' && d <= 'f') ? d - 'a' + 10
                      : (d >= 'A' && d <= 'F') ? d - 'A' + 10 : 0;
                    hi = (hi << 4) | d;
                }
                b->st.hash[(size_t)i * SIGIL_HASH_LEN + j] = (uint8_t)hi;
            }
        }
        b->lens[i] = len;
        return i;
    }
}

/* Kept for callers that have no stored hash. */
long
br_add_hex(void *p, const char *path, unsigned para, const char *lshhex)
{
    return br_add_restore(p, path, para, lshhex, NULL, 0, 0);
}
