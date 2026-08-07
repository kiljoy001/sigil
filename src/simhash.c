/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * SimHash: dense vector -> 32 bits, preserving angular distance.
 *
 * For a random hyperplane h and vectors u, v, the probability that
 * sign(u.h) != sign(v.h) is theta(u,v)/pi. Over 32 independent hyperplanes the
 * expected Hamming distance is 32*theta/pi, so Hamming distance in the
 * compressed space estimates the angle in the original. This is why the 32
 * bits can be scanned with popcount and still mean something: they are a
 * lossy but unbiased encoding of cosine similarity, not a digest.
 */

#include "sigil_embed.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* M_PI is POSIX, not C11, and -std=c11 hides it. */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* splitmix64: deterministic, well-distributed, no library dependency. The
 * seed must reproduce byte-for-byte across machines, so the platform RNG is
 * unsuitable. */
static uint64_t splitmix64(uint64_t *state)
{
	uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);

	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
	return z ^ (z >> 31);
}

/* Box-Muller: uniform -> standard normal. Hyperplane components must be
 * Gaussian for the direction to be uniform on the sphere; uniform components
 * would bias planes toward the cube's corners and distort the distance
 * estimate. */
static float next_gaussian(uint64_t *state)
{
	double u1, u2;

	/* Avoid log(0). */
	do {
		u1 = (double)(splitmix64(state) >> 11) / 9007199254740992.0;
	} while (u1 <= 1e-12);
	u2 = (double)(splitmix64(state) >> 11) / 9007199254740992.0;

	return (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2));
}

int sigil_simhash_init(sigil_simhash_t *sh, size_t dim, uint64_t seed)
{
	uint64_t state = seed;
	size_t n = (size_t)SIGIL_LSH_BITS * dim;

	sh->planes = malloc(n * sizeof(float));
	if (!sh->planes)
		return -1;

	for (size_t i = 0; i < n; i++)
		sh->planes[i] = next_gaussian(&state);

	sh->dim = dim;
	sh->seed = seed;
	return 0;
}

void sigil_simhash_free(sigil_simhash_t *sh)
{
	free(sh->planes);
	sh->planes = NULL;
	sh->dim = 0;
}

void sigil_simhash_project(const sigil_simhash_t *sh, const float *vec,
                           uint64_t *out)
{
	for (int w = 0; w < SIGIL_LSH_WORDS; w++)
		out[w] = 0;

	for (int b = 0; b < SIGIL_LSH_BITS; b++) {
		const float *plane = sh->planes + (size_t)b * sh->dim;
		float dot = 0.0f;

		for (size_t i = 0; i < sh->dim; i++)
			dot += vec[i] * plane[i];

		if (dot > 0.0f)
			out[b / 64] |= 1ULL << (b % 64);
	}
}

int sigil_generate_semantic(sigil_embedder_t *emb, const sigil_simhash_t *sh,
                            const char *text, size_t len,
                            uint32_t timestamp, uint16_t category,
                            const sigil_trits_t *trits, sigil_t *out)
{
	float *vec;
	size_t dim = emb->dim(emb);

	if (dim != sh->dim)
		return -1; /* bits from a different width are not comparable */

	vec = malloc(dim * sizeof(float));
	if (!vec)
		return -1;

	if (emb->embed(emb, text, len, vec) != 0) {
		free(vec);
		return -1;
	}

	/* Identity still comes from the raw bytes: two files that mean the
	 * same thing are not the same file. */
	sigil_generate(text, len, timestamp, category, trits, out);

	/* Replace the byte-hash LSH with the semantic projection. */
	sigil_simhash_project(sh, vec, out->lsh);

	free(vec);
	return 0;
}
