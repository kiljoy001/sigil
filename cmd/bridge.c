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
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

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

	/* Pending batch. Records are pushed to the store immediately so their
	 * indices are stable; only the LSH bits are deferred until the batch
	 * flushes. Embedding one paragraph per inference reaches 200/s on an
	 * Arc Pro B50 against 1833/s at a batch of 128 -- 9 hours for a 59.6M
	 * paragraph corpus instead of 83. */
	char **btext;         /* [Batchmax] owned copies */
	size_t *blen;
	long *bidx;           /* store index each text belongs to */
	size_t bn;
};

/* Larger batches keep paying: 494/s at 16, 1833/s at 128, and 256 gains
 * nothing. The tokenizer pads to the longest member, so a batch mixing very
 * short and very long paragraphs wastes work -- but not enough to matter
 * against the fixed cost of an inference call. */
#define Batchmax 128

void br_flush(void *p);

/*
 * Install an embedder directly, bypassing the path-based dispatch above.
 *
 * br_embedder_load() takes a filename, so the only way to reach the batching
 * path is to have a real model and a GPU -- which means the code with the
 * highest CRAP score in this file (br_flush, br_add_at) could not be tested
 * at all. A caller-supplied embedder makes both deterministic and
 * hardware-free. Ownership transfers: br_free destroys it.
 */
int
br_embedder_set(void *p, sigil_embedder_t *e, unsigned long long seed)
{
	struct bridge *b = p;

	if (b == NULL || e == NULL)
		return -1;
	if (b->emb != NULL)
		return -1;                       /* refuse to leak the old one */
	b->emb = e;
	if (sigil_simhash_init(&b->sh, e->dim(e), seed) != 0) {
		b->emb = NULL;
		return -1;
	}
	b->have_sh = 1;
	return 0;
}

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
	b->btext = calloc(Batchmax, sizeof *b->btext);
	b->blen  = calloc(Batchmax, sizeof *b->blen);
	b->bidx  = calloc(Batchmax, sizeof *b->bidx);
	if (b->paths == NULL || b->paras == NULL ||
	    b->offs == NULL || b->lens == NULL ||
	    b->btext == NULL || b->blen == NULL || b->bidx == NULL) {
		sigil_store_free(&b->st);
		free(b->paths); free(b->paras);
		free(b->offs); free(b->lens);
		free(b->btext); free(b->blen); free(b->bidx); free(b);
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

	/* Dispatch on what the path is rather than on a flag: an OpenVINO
	 * export is a directory of .xml/.bin, a llama.cpp model is a .gguf
	 * file. One -e option covers both, and the failure mode of naming the
	 * wrong kind is a clear load error rather than a silent fallback.
	 *
	 * SIGIL_OV_DEVICE picks the target -- GPU.1, CPU, NPU, AUTO. On Intel
	 * hardware OpenVINO is both faster and, above ~4B parameters, the only
	 * one that is correct; see docs/FINDINGS.md. */
	{
		struct stat st;

		if (stat(gguf, &st) == 0 && S_ISDIR(st.st_mode))
			b->emb = sigil_embedder_openvino(gguf,
				getenv("SIGIL_OV_DEVICE"));
		else
			b->emb = sigil_embedder_llama(gguf);
	}
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
	for (size_t k = 0; k < b->bn; k++)
		free(b->btext[k]);
	free(b->btext);
	free(b->blen);
	free(b->bidx);
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
	int i, defer = 0;

	if (b->n == b->cap && grow(b) != 0)
		return -1;

	sigil_generate_para(text, len, para, timestamp, 0, NULL, &s);

	if (lsh != NULL) {
		for (i = 0; i < SIGIL_LSH_WORDS; i++)
			s.lsh[i] = lsh[i];
	} else if (b->emb != NULL && b->emb->embed_batch != NULL) {
		/* Defer: queue the text and fill the bits when the batch
		 * flushes. The record carries the byte-shingle fallback until
		 * then, which is what it would have kept had embedding
		 * failed. */
		defer = 1;
	} else if (b->emb != NULL) {
		/* Backend with no batch path -- llama.cpp, the hash fallback.
		 * Failure here leaves the byte-shingle bits rather than
		 * dropping the record: a paragraph that will not embed is
		 * still worth addressing by content, it just will not be found
		 * by similarity. */
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

	if (defer) {
		char *copy = malloc(len + 1);

		if (copy == NULL)
			return (long)(b->n - 1);   /* keeps fallback bits */
		memcpy(copy, text, len);
		copy[len] = '\0';
		b->btext[b->bn] = copy;
		b->blen[b->bn] = len;
		b->bidx[b->bn] = (long)(b->n - 1);
		b->bn++;
		if (b->bn == Batchmax)
			br_flush(b);
	}
	return (long)(b->n - 1);
}

/*
 * Embed everything queued and write the bits into the records that are
 * already in the store.
 *
 * Must be called before any read of the LSH field -- br_similar, a commit, a
 * scan. Leaving it to the caller would make "the bits are wrong" depend on
 * call order, so index.c calls it at the end of every file and store_commit
 * calls it before writing.
 */
void
br_flush(void *p)
{
	struct bridge *b = p;
	size_t dim, i;
	float *v;

	if (b == NULL || b->bn == 0 || b->emb == NULL ||
	    b->emb->embed_batch == NULL)
		return;

	dim = b->emb->dim(b->emb);
	v = malloc(b->bn * dim * sizeof *v);
	if (v != NULL) {
		if (b->emb->embed_batch(b->emb, (const char **)b->btext,
		                        b->blen, b->bn, v) > 0) {
			for (i = 0; i < b->bn; i++) {
				sigil_t s;

				if (sigil_store_get(&b->st, (size_t)b->bidx[i],
				                    &s) != 0)
					continue;
				sigil_simhash_project(&b->sh, v + i * dim,
				                      s.lsh);
				/* Write the bits back in place: the store is
				 * struct-of-arrays, so only the lsh row moves. */
				memcpy(b->st.lsh + (size_t)b->bidx[i] *
				           SIGIL_LSH_WORDS,
				       s.lsh, sizeof s.lsh);
			}
		}
		free(v);
	}
	for (i = 0; i < b->bn; i++)
		free(b->btext[i]);
	b->bn = 0;
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

	/* Guard the bridge, not just the index. Every sibling accessor here
	 * checks for NULL; these two did not, and similar_read() calls both
	 * on every neighbour it serves. */
	if (b == NULL)
		return 0;
	return (i >= 0 && (size_t)i < b->n) ? b->offs[i] : 0;
}

unsigned long
br_length(void *p, long i)
{
	struct bridge *b = p;

	if (b == NULL)
		return 0;
	return (i >= 0 && (size_t)i < b->n) ? b->lens[i] : 0;
}

/* Which backend is actually live. Two stores built with different embedders
 * are close but not identical -- mean cosine 0.954 between the llama.cpp and
 * OpenVINO paths -- so this belongs in /stats next to the model name. */
const char *
br_embedder_name(void *p)
{
	struct bridge *b = p;

	if (b == NULL || b->emb == NULL)
		return "none";
	return b->emb->name(b->emb);
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
