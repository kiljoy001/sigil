/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Semantic embedding backend.
 *
 * The LSH bits in a sigil are only meaningful if they come from meaning. A
 * byte-level hash gives two documents about the same topic in different words
 * unrelated bits, which makes "semantic filesystem" a false claim. This header
 * defines where real embeddings enter.
 *
 * The pipeline is: text -> dense vector (384-dim for MiniLM) -> SimHash
 * against fixed random hyperplanes -> SIGIL_LSH_BITS bits. SimHash is the
 * right reduction because the probability two vectors land on the same side of
 * a random hyperplane is 1 - theta/pi, so Hamming distance in the compressed
 * space is a direct estimator of angular distance in the original. Cosine
 * similarity survives the squeeze; truncation or quantization would not.
 *
 * The backend is a vtable so libsigil keeps no hard dependency on any ML
 * runtime. Link the llama.cpp backend for real embeddings, or the hash
 * backend for tests and platforms without a model.
 */

#ifndef SIGIL_EMBED_H
#define SIGIL_EMBED_H

#include "sigil.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Dimensionality of all-MiniLM-L6-v2. Other models differ; the SimHash
 * projection adapts, but a set of hyperplanes is only valid for one width.
 *
 * Measured on Quora Duplicate Questions (recall@1, float32 ceiling 0.8140):
 *    32 bits  0.550   67.5% of ceiling
 *   128 bits  0.785   96.5% of ceiling   <- SIGIL_LSH_BITS
 *   256 bits  0.803   98.6%
 *   512 bits  0.810   99.4%
 * Past 128 the remaining headroom is in the embedding model, not the width.
 *
 * INT8 quantization of the embedding costs nothing measurable at this width
 * (0.7880 vs 0.7880), because SimHash reads only sign(dot) and quantization
 * noise almost never flips it. NPU inference is therefore quality-neutral. */
#define SIGIL_EMBED_DIM_MINILM 384

typedef struct sigil_embedder sigil_embedder_t;

struct sigil_embedder {
	/* Embed text into a caller-supplied buffer of at least dim() floats.
	 * Returns 0 on success, -1 on failure. Must be L2-normalized on exit:
	 * SimHash only estimates angle, so magnitude must not leak in. */
	int (*embed)(sigil_embedder_t *self, const char *text, size_t len,
	             float *out);

	/* Vector width this backend produces. */
	size_t (*dim)(const sigil_embedder_t *self);

	/* Human-readable identity, for provenance in logs and tests. */
	const char *(*name)(const sigil_embedder_t *self);

	void (*destroy)(sigil_embedder_t *self);

	void *impl;
};

/* ---------------------------------------------------------------------------
 * SimHash projection
 *
 * Hyperplanes are generated deterministically from a seed so that two
 * processes, or the same store reopened later, produce comparable bits. A
 * store's bits are only meaningful against the seed and dimension that made
 * them; changing either invalidates every sigil already written.
 * ------------------------------------------------------------------------ */

typedef struct {
	float *planes;  /* [SIGIL_LSH_BITS * dim], row-major */
	size_t dim;
	uint64_t seed;
} sigil_simhash_t;

int  sigil_simhash_init(sigil_simhash_t *sh, size_t dim, uint64_t seed);
void sigil_simhash_free(sigil_simhash_t *sh);

/* Project a dense vector to SIGIL_LSH_BITS bits, written to out[] as
 * SIGIL_LSH_WORDS words. */
void sigil_simhash_project(const sigil_simhash_t *sh, const float *vec,
                           uint64_t *out);

/* ---------------------------------------------------------------------------
 * Generating sigils from text
 * ------------------------------------------------------------------------ */

/*
 * Full pipeline: SHA-1 over the raw bytes for identity, embedding + SimHash
 * for the LSH bits. This is what sigil_generate() should have been; the
 * byte-hash version remains for content with no meaningful text.
 */
int sigil_generate_semantic(sigil_embedder_t *emb, const sigil_simhash_t *sh,
                            const char *text, size_t len,
                            uint32_t timestamp, uint16_t category,
                            const sigil_trits_t *trits, sigil_t *out);

/* ---------------------------------------------------------------------------
 * Backends
 * ------------------------------------------------------------------------ */

/*
 * llama.cpp backend. Loads a GGUF embedding model (e.g. all-MiniLM-L6-v2).
 * Returns NULL if the model cannot be loaded or was not built with
 * SIGIL_WITH_LLAMA.
 */
sigil_embedder_t *sigil_embedder_llama(const char *gguf_path);

/*
 * Deterministic hash backend: NOT semantic. Produces a stable vector from
 * byte shingles so tests can run without a model. Named honestly so it cannot
 * be mistaken for the real thing in a stack trace.
 */
sigil_embedder_t *sigil_embedder_hash_nonsemantic(size_t dim);

#ifdef __cplusplus
}
#endif

#endif /* SIGIL_EMBED_H */
