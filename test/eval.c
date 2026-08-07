/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Retrieval quality against a standard corpus.
 *
 * test/semantic.c answers "are the bits semantic at all?" with a handful of
 * pairs. This answers "how good are they?" on Quora Duplicate Questions, the
 * human-labeled benchmark used by BEIR and MTEB.
 *
 * The metric is recall@1: for each document, is its true duplicate the nearest
 * neighbour in Hamming space among all other documents? With 1000 pairs that
 * is 2000 documents and ~2M comparisons, which is enough to separate designs
 * that a small corpus cannot.
 *
 * The reference point is the float32 ceiling: recall@1 using cosine similarity
 * on the uncompressed embedding. No amount of LSH bits can beat it, so quality
 * is reported as a percentage of it. That framing matters — it distinguishes
 * "the compression is lossy" from "the embedding model is the limit", and only
 * the second is fixed by a better model.
 *
 * Ground truth is positional: documents 2i and 2i+1 are duplicates.
 */

/* strdup is POSIX, not C11. Must precede every include. */
#define _POSIX_C_SOURCE 200809L

#include "sigil_embed.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SIMHASH_SEED 0x5191c0de5191c0deULL
#define MAX_LINE     8192

static double now_sec(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Read the corpus, one document per line. */
static char **read_corpus(const char *path, size_t *count)
{
	FILE *f = fopen(path, "r");
	char **docs = NULL;
	size_t n = 0, cap = 0;
	char line[MAX_LINE];

	if (!f)
		return NULL;

	while (fgets(line, sizeof(line), f)) {
		size_t len = strlen(line);

		while (len && (line[len - 1] == '\n' || line[len - 1] == '\r'))
			line[--len] = '\0';
		if (len == 0)
			continue;

		if (n == cap) {
			size_t ncap = cap ? cap * 2 : 256;
			char **nd = realloc(docs, ncap * sizeof(char *));

			if (!nd)
				goto fail;
			docs = nd;
			cap = ncap;
		}
		docs[n] = strdup(line);
		if (!docs[n])
			goto fail;
		n++;
	}
	fclose(f);
	*count = n;
	return docs;

fail:
	for (size_t i = 0; i < n; i++)
		free(docs[i]);
	free(docs);
	fclose(f);
	return NULL;
}

static double cosine(const float *a, const float *b, size_t d)
{
	double s = 0.0;

	for (size_t i = 0; i < d; i++)
		s += (double)a[i] * (double)b[i];
	return s;
}

int main(int argc, char **argv)
{
	const char *model, *corpus_path;
	sigil_embedder_t *e;
	sigil_simhash_t sh;
	char **docs;
	size_t n = 0, dim;
	float *vecs;
	uint32_t *bits;
	size_t hit_float = 0, hit_lsh = 0;
	double t0;

	if (argc < 3) {
		fprintf(stderr,
		        "usage: %s <model.gguf> <corpus.txt>\n\n"
		        "fetch a corpus with tools/fetch-corpus.py\n", argv[0]);
		return 2;
	}
	model = argv[1];
	corpus_path = argv[2];

	docs = read_corpus(corpus_path, &n);
	if (!docs) {
		fprintf(stderr, "cannot read corpus: %s\n", corpus_path);
		return 2;
	}
	if (n < 2 || n % 2 != 0) {
		fprintf(stderr, "corpus must hold an even number of documents "
		        "(got %zu); pairs are positional\n", n);
		return 2;
	}

	e = sigil_embedder_llama(model);
	if (!e) {
		fprintf(stderr, "cannot load model: %s\n"
		        "(was libsigil built with SIGIL_WITH_LLAMA?)\n", model);
		return 2;
	}
	dim = e->dim(e);

	if (sigil_simhash_init(&sh, dim, SIMHASH_SEED) != 0) {
		e->destroy(e);
		return 2;
	}

	printf("corpus:   %s (%zu documents, %zu pairs)\n", corpus_path, n, n / 2);
	printf("embedder: %s (dim %zu)\n\n", e->name(e), dim);

	vecs = malloc(n * dim * sizeof(float));
	bits = malloc(n * sizeof(uint32_t));
	if (!vecs || !bits) {
		fprintf(stderr, "out of memory for %zu x %zu embeddings\n", n, dim);
		free(vecs); free(bits);
		sigil_simhash_free(&sh); e->destroy(e);
		return 2;
	}

	t0 = now_sec();
	for (size_t i = 0; i < n; i++) {
		if (e->embed(e, docs[i], strlen(docs[i]), vecs + i * dim) != 0) {
			fprintf(stderr, "embedding failed at document %zu\n", i);
			free(vecs); free(bits);
			sigil_simhash_free(&sh); e->destroy(e);
			return 2;
		}
		bits[i] = sigil_simhash_project(&sh, vecs + i * dim);
		if ((i + 1) % 200 == 0) {
			fprintf(stderr, "\r  embedded %zu/%zu", i + 1, n);
			fflush(stderr);
		}
	}
	{
		double dt = now_sec() - t0;

		fprintf(stderr, "\r                              \r");
		printf("embedded %zu documents in %.1f s (%.0f docs/s)\n\n",
		       n, dt, (double)n / dt);
	}

	/* Brute-force nearest neighbour in both spaces. O(n^2), which is the
	 * point: this measures quality, not speed. */
	for (size_t i = 0; i < n; i++) {
		size_t truth = i ^ 1; /* positional ground truth */
		size_t best_f = SIZE_MAX, best_l = SIZE_MAX;
		double best_cos = -2.0;
		uint32_t best_ham = 33;

		for (size_t j = 0; j < n; j++) {
			double c;
			uint32_t h;

			if (j == i)
				continue;
			c = cosine(vecs + i * dim, vecs + j * dim, dim);
			if (c > best_cos) {
				best_cos = c;
				best_f = j;
			}
			h = sigil_hamming(bits[i], bits[j]);
			if (h < best_ham) {
				best_ham = h;
				best_l = j;
			}
		}
		if (best_f == truth)
			hit_float++;
		if (best_l == truth)
			hit_lsh++;
	}

	{
		double rf = (double)hit_float / (double)n;
		double rl = (double)hit_lsh / (double)n;

		printf("recall@1\n");
		printf("  float32 cosine (ceiling): %.4f\n", rf);
		printf("  %d-bit LSH:               %.4f  (%.1f%% of ceiling)\n",
		       32, rl, rf > 0 ? 100.0 * rl / rf : 0.0);
		printf("\n");
		printf("The ceiling is a property of the embedding model, not of the\n");
		printf("LSH width. Closing the gap to it needs more bits; raising it\n");
		printf("needs a different model.\n");
	}

	for (size_t i = 0; i < n; i++)
		free(docs[i]);
	free(docs);
	free(vecs);
	free(bits);
	sigil_simhash_free(&sh);
	e->destroy(e);
	return 0;
}
