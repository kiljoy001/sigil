/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Stage-two sidecars: a second code for the subset stage one selected.
 *
 * Most of what follows is about refusal. A sidecar built against another
 * store still indexes real records in this one, and codes from another
 * model still produce a number when you XOR them -- neither failure
 * announces itself, both return neighbours that look reasonable. So the
 * header carries the model, the width and the base store's count, and the
 * tests here are mostly about those being enforced rather than recorded.
 *
 *	make sidecar
 */

#define _POSIX_C_SOURCE 200809L

#include "sigil.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int checks, failures;

static void
ok(int cond, const char *what)
{
	checks++;
	if (cond)
		return;
	failures++;
	printf("FAIL: %s\n", what);
}

#define NREC  5000
#define NSIDE 700
#define WORDS 8          /* 512-bit refined code, wider than stage one */

static const char *PATH = "test-side.smap.side";
static const char *MODEL = "bge-base-en-v1.5";

static void
build_store(sigil_store_t *st)
{
	sigil_t s;
	size_t i;

	sigil_store_init(st, 0);
	for (i = 0; i < NREC; i++) {
		memset(&s, 0, sizeof s);
		s.lsh[0] = (uint64_t)i * 0x9e3779b97f4a7c15ULL;
		s.para = (uint32_t)i;
		sigil_store_push(st, &s);
	}
}

/* Every third record gets a refined code, so the sidecar is genuinely
 * sparse and the gaps have to be handled. */
static void
build_side(uint32_t *idx, uint64_t *code)
{
	size_t k, w;

	for (k = 0; k < NSIDE; k++) {
		idx[k] = (uint32_t)(k * 3);
		for (w = 0; w < WORDS; w++)
			code[k * WORDS + w] =
				(uint64_t)(k + 1) * 0x9e3779b97f4a7c15ULL + w;
	}
}

static void
test_round_trip(void)
{
	sigil_store_t st;
	sigil_side_t sd;
	uint32_t *idx = malloc(NSIDE * sizeof *idx);
	uint64_t *code = malloc(NSIDE * WORDS * sizeof *code);
	size_t k, bad = 0;

	build_store(&st);
	build_side(idx, code);

	ok(sigil_side_save(PATH, idx, code, NSIDE, WORDS, MODEL,
	                   (uint64_t)st.count) == 0, "sidecar saves");
	ok(sigil_side_map(&sd, PATH, &st, MODEL) == 0, "sidecar maps");
	ok(sd.count == NSIDE, "entry count survives");
	ok(sd.words == WORDS && sd.bits == WORDS * 64, "code width survives");
	ok(strcmp(sd.model, MODEL) == 0, "model name survives");

	for (k = 0; k < NSIDE; k++) {
		const uint64_t *c = sigil_side_lookup(&sd, idx[k]);

		if (c == NULL || memcmp(c, code + k * WORDS,
		                        WORDS * sizeof *code) != 0)
			bad++;
	}
	ok(bad == 0, "every refined code reads back exactly");

	/* The gaps are the common case: most records have no entry, and a
	 * lookup that invented one would be worse than useless. */
	bad = 0;
	for (k = 1; k < NSIDE * 3; k += 3)
		if (sigil_side_lookup(&sd, (uint32_t)k) != NULL)
			bad++;
	ok(bad == 0, "records with no entry return NULL rather than a neighbour's code");

	ok(sigil_side_lookup(&sd, (uint32_t)(NREC + 1000)) == NULL,
	   "an index past the store returns NULL");

	/* Distance over the refined codes, at the sidecar's own width. */
	{
		const uint64_t *a = sigil_side_lookup(&sd, idx[0]);
		const uint64_t *b = sigil_side_lookup(&sd, idx[1]);

		ok(a != NULL && b != NULL, "two codes for a distance check");
		ok(sigil_side_hamming(&sd, a, a) == 0,
		   "a refined code is at distance 0 from itself");
		ok(sigil_side_hamming(&sd, a, b) <= WORDS * 64,
		   "distance stays within the refined width");
	}

	sigil_side_unmap(&sd);
	sigil_store_free(&st);
	free(idx);
	free(code);
}

static void
test_refusals(void)
{
	sigil_store_t st, small;
	sigil_side_t sd;
	uint32_t *idx = malloc(NSIDE * sizeof *idx);
	uint64_t *code = malloc(NSIDE * WORDS * sizeof *code);
	sigil_t s;
	size_t i;

	build_store(&st);
	build_side(idx, code);
	sigil_side_save(PATH, idx, code, NSIDE, WORDS, MODEL,
	                (uint64_t)st.count);

	ok(sigil_side_map(&sd, "no-such.side", &st, MODEL) != 0,
	   "a missing sidecar is refused");

	/*
	 * The dangerous one. A sidecar built against a different store still
	 * indexes real records here, so every lookup succeeds and every
	 * answer is wrong.
	 */
	sigil_store_init(&small, 0);
	for (i = 0; i < 100; i++) {
		memset(&s, 0, sizeof s);
		s.para = (uint32_t)i;
		sigil_store_push(&small, &s);
	}
	ok(sigil_side_map(&sd, PATH, &small, MODEL) != 0,
	   "a sidecar built against another store is refused");
	sigil_store_free(&small);

	/* Codes from two models are not comparable -- that they disagree is
	 * the entire premise of running a second stage. */
	ok(sigil_side_map(&sd, PATH, &st, "all-MiniLM-L6-v2") != 0,
	   "a sidecar from another model is refused");

	ok(sigil_side_map(&sd, PATH, &st, NULL) == 0,
	   "passing no model name skips only that check");
	sigil_side_unmap(&sd);

	/* Unsorted input breaks the binary search silently: entries that are
	 * present stop being found. Reject at save rather than at lookup. */
	{
		uint32_t bad_idx[4] = { 10, 30, 20, 40 };
		uint64_t bad_code[4 * WORDS];

		memset(bad_code, 0, sizeof bad_code);
		ok(sigil_side_save("test-side-bad.side", bad_idx, bad_code, 4,
		                   WORDS, MODEL, (uint64_t)st.count) != 0,
		   "unsorted indices are refused at save");
		unlink("test-side-bad.side");
	}

	sigil_store_free(&st);
	free(idx);
	free(code);
}

/* A sidecar is optional: a store without one is not broken, and the empty
 * case must not be a special path that crashes. */
static void
test_empty(void)
{
	sigil_store_t st;
	sigil_side_t sd;

	build_store(&st);
	ok(sigil_side_save("test-side-empty.side", NULL, NULL, 0, WORDS,
	                   MODEL, (uint64_t)st.count) == 0,
	   "an empty sidecar saves");
	ok(sigil_side_map(&sd, "test-side-empty.side", &st, MODEL) == 0,
	   "an empty sidecar maps");
	ok(sd.count == 0, "an empty sidecar reports no entries");
	ok(sigil_side_lookup(&sd, 0) == NULL,
	   "a lookup in an empty sidecar returns NULL");
	sigil_side_unmap(&sd);
	sigil_store_free(&st);
	unlink("test-side-empty.side");
}

/* --- float payload and rerank ------------------------------------------- */

#define VDIM 16
#define NVEC 500

/* Vector k points mostly along axis (k % VDIM), so nearest-by-cosine is
 * predictable without depending on a model. */
static void
make_vec(float *v, size_t k)
{
	size_t d;

	for (d = 0; d < VDIM; d++)
		v[d] = (d == k % VDIM) ? 1.0f : 0.01f * (float)((k + d) % 7);
}

static void
test_float_sidecar(void)
{
	sigil_store_t st;
	sigil_side_t sd;
	uint32_t *idx = malloc(NVEC * sizeof *idx);
	float *vec = malloc(NVEC * VDIM * sizeof *vec);
	size_t k, bad = 0;
	const char *path = "test-side-vec.side";

	build_store(&st);
	for (k = 0; k < NVEC; k++) {
		idx[k] = (uint32_t)(k * 2);        /* sparse: every other */
		make_vec(vec + k * VDIM, k);
	}

	ok(sigil_side_save_vec(path, idx, vec, NVEC, VDIM, MODEL,
	                       (uint64_t)st.count) == 0,
	   "a float sidecar saves");
	ok(sigil_side_map(&sd, path, &st, MODEL) == 0, "a float sidecar maps");
	ok(sd.kind == SIGIL_SIDE_FLOAT, "the payload kind is recorded");
	ok(sd.dim == VDIM, "the vector dimension survives");

	for (k = 0; k < NVEC; k++) {
		const float *v = sigil_side_vec(&sd, idx[k]);

		if (v == NULL || memcmp(v, vec + k * VDIM,
		                        VDIM * sizeof *vec) != 0)
			bad++;
	}
	ok(bad == 0, "every vector reads back exactly");

	/* Float bytes read as a bit pattern give a number, not an error, so
	 * asking for the wrong payload must refuse rather than reinterpret. */
	ok(sigil_side_lookup(&sd, idx[0]) == NULL,
	   "a code lookup on a float sidecar returns NULL, not reinterpreted bits");

	/* Cosine: a vector against itself is 1, and the sense is "higher is
	 * nearer" -- the opposite of Hamming. */
	{
		const float *a = sigil_side_vec(&sd, idx[0]);
		const float *b = sigil_side_vec(&sd, idx[1]);
		double self = sigil_side_cosine(&sd, a, a);
		double other = sigil_side_cosine(&sd, a, b);

		ok(self > 0.999 && self < 1.001, "cosine with itself is 1");
		ok(other < self, "a different vector scores lower");
	}

	/* Rerank: hand it a deliberately bad order and check the nearest
	 * comes first. */
	{
		uint32_t cand[8];
		float q[VDIM];
		size_t n;

		for (k = 0; k < 8; k++)
			cand[k] = idx[7 - k];       /* reversed */
		make_vec(q, 0);                     /* query == vector 0 */
		n = sigil_side_rerank(&sd, q, cand, 8);
		ok(n == 8, "rerank scored every candidate");
		ok(cand[0] == idx[0],
		   "rerank puts the nearest candidate first");
	}

	/*
	 * A candidate the sidecar does not cover is a gap in stage two, not
	 * evidence about the record: it must survive, after the reranked
	 * ones, rather than being dropped.
	 */
	{
		uint32_t cand[4];
		float q[VDIM];
		size_t n;

		cand[0] = idx[5];
		cand[1] = 1;              /* odd index: no sidecar entry */
		cand[2] = idx[0];
		cand[3] = 3;              /* likewise */
		make_vec(q, 0);
		n = sigil_side_rerank(&sd, q, cand, 4);
		ok(n == 2, "rerank reports how many it could score");
		ok(cand[0] == idx[0], "the scored nearest still leads");
		ok((cand[2] == 1 && cand[3] == 3),
		   "uncovered candidates are kept, after the scored ones");
	}

	sigil_side_unmap(&sd);
	sigil_store_free(&st);
	free(idx);
	free(vec);
	unlink(path);
}

int
main(void)
{
	test_round_trip();
	test_refusals();
	test_empty();
	test_float_sidecar();

	unlink(PATH);

	if (failures) {
		printf("FAILED: %d of %d checks\n", failures, checks);
		return 1;
	}
	printf("PASS: %d checks\n", checks);
	return 0;
}
