/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Tests for the embedding cache.
 *
 * Almost every case here closes the file and reopens it, rather than
 * checking what the writer thinks it stored. The cache exists because 13
 * hours of GPU work was lost when nothing had been persisted, so "it is
 * in memory" is precisely the property that does not matter -- what
 * matters is that it comes back after the process dies.
 *
 *	make test/veccache && ./test/veccache
 */

#include "sigil_veccache.h"

#include <math.h>
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

static void
eqsz(size_t got, size_t want, const char *what)
{
	checks++;
	if (got == want)
		return;
	failures++;
	printf("FAIL: %s: got %zu want %zu\n", what, got, want);
}

#define DIM 8
#define PATH "/tmp/sigil-veccache-test.jsonl"

/*
 * Distinct hashes for distinct seeds.
 *
 * The first version computed (seed * 31 + i) in a uint8_t, which wraps:
 * seeds 0 and 66 produced identical bytes, and 244 of 500 "distinct"
 * hashes collided. The cache deduplicated them correctly and the test
 * reported 256 entries as a failure of the cache. It was a failure of
 * the fixture -- worth the comment, because a test that generates
 * colliding keys and blames the code is worse than no test.
 */
static void
fill(uint8_t h[32], int seed)
{
	int i;

	memset(h, 0, 32);
	for (i = 0; i < 4; i++)
		h[i] = (uint8_t)((unsigned)seed >> (i * 8));
	for (i = 4; i < 32; i++)
		h[i] = (uint8_t)(i * 7);
}

/*
 * A vector in the range real ones occupy.
 *
 * Embedding output is L2-normalised, so every component is in [-1, 1],
 * where float16 holds about three decimal digits. The first version of
 * this helper used the loop index directly as a base, reaching 499 --
 * and above 256 float16 spacing is 0.25, so a 0.125 fraction cannot be
 * represented and 244 of 500 comparisons failed. The storage format was
 * fine; the fixture was testing a range the data never occupies.
 */
static void
vec_of(float *v, float base)
{
	int i;

	for (i = 0; i < DIM; i++)
		v[i] = sinf(base + (float)i);      /* in [-1, 1] */
}

/* float16 has ~3 decimal digits; the consumer is a sign bit, so this is
 * far tighter than the projection needs. */
static int
close_enough(const float *a, const float *b)
{
	int i;

	for (i = 0; i < DIM; i++)
		if (fabsf(a[i] - b[i]) > 0.01f)
			return 0;
	return 1;
}

/* --- round trip through the file --------------------------------------- */

static void
test_survives_close(void)
{
	sigil_veccache_t *c;
	uint8_t h[32];
	float in[DIM], out[DIM];

	unlink(PATH);
	fill(h, 1);
	vec_of(in, 1.0f);

	c = sigil_veccache_open(PATH, "minilm", DIM);
	ok(c != NULL, "open creates a cache");
	ok(sigil_veccache_get(c, h, out) != 0, "empty cache misses");
	ok(sigil_veccache_put(c, h, in) == 0, "put succeeds");
	ok(sigil_veccache_get(c, h, out) == 0, "hit within the session");
	sigil_veccache_close(c);

	/* The point of the whole thing. */
	c = sigil_veccache_open(PATH, "minilm", DIM);
	ok(c != NULL, "reopen");
	eqsz(sigil_veccache_count(c), 1, "the entry survived the close");
	ok(sigil_veccache_get(c, h, out) == 0, "hit after reopen");
	ok(close_enough(in, out), "the vector came back intact");
	sigil_veccache_close(c);
}

static void
test_many(void)
{
	sigil_veccache_t *c;
	uint8_t h[32];
	float in[DIM], out[DIM];
	int i, hits = 0;

	unlink(PATH);
	c = sigil_veccache_open(PATH, "minilm", DIM);
	for (i = 0; i < 500; i++) {
		fill(h, i);
		vec_of(in, (float)i);
		sigil_veccache_put(c, h, in);
	}
	eqsz(sigil_veccache_count(c), 500, "500 entries held");
	sigil_veccache_close(c);

	c = sigil_veccache_open(PATH, "minilm", DIM);
	eqsz(sigil_veccache_count(c), 500, "500 entries reloaded");
	for (i = 0; i < 500; i++) {
		fill(h, i);
		vec_of(in, (float)i);
		if (sigil_veccache_get(c, h, out) == 0 && close_enough(in, out))
			hits++;
	}
	eqsz((size_t)hits, 500, "every entry round-trips");
	sigil_veccache_close(c);
}

/* --- the model is part of the key -------------------------------------- */

static void
test_model_scoping(void)
{
	sigil_veccache_t *c;
	uint8_t h[32];
	float a[DIM], b[DIM], out[DIM];

	unlink(PATH);
	fill(h, 7);
	vec_of(a, 1.0f);
	vec_of(b, 99.0f);

	c = sigil_veccache_open(PATH, "minilm", DIM);
	sigil_veccache_put(c, h, a);
	sigil_veccache_close(c);

	/* Same text, different model: serving the first model's vector here
	 * would be silent corruption, so it must miss. */
	c = sigil_veccache_open(PATH, "bge-small", DIM);
	eqsz(sigil_veccache_count(c), 0, "another model's entries are not visible");
	ok(sigil_veccache_get(c, h, out) != 0, "same hash, other model, misses");
	sigil_veccache_put(c, h, b);
	sigil_veccache_close(c);

	/* And both now coexist in one file. */
	c = sigil_veccache_open(PATH, "minilm", DIM);
	ok(sigil_veccache_get(c, h, out) == 0 && close_enough(a, out),
	   "the first model still reads its own vector");
	sigil_veccache_close(c);

	c = sigil_veccache_open(PATH, "bge-small", DIM);
	ok(sigil_veccache_get(c, h, out) == 0 && close_enough(b, out),
	   "the second model reads its own");
	sigil_veccache_close(c);
}

/* --- crash tolerance ---------------------------------------------------- */

static void
test_truncated_line_is_skipped(void)
{
	sigil_veccache_t *c;
	FILE *fp;
	uint8_t h[32];
	float in[DIM], out[DIM];

	unlink(PATH);
	fill(h, 3);
	vec_of(in, 5.0f);

	c = sigil_veccache_open(PATH, "minilm", DIM);
	sigil_veccache_put(c, h, in);
	sigil_veccache_close(c);

	/* A crash mid-append leaves a partial line. It must cost that one
	 * entry, not the file -- this is the case the whole format was
	 * chosen for. */
	fp = fopen(PATH, "a");
	fprintf(fp, "{\"h\":\"deadbeef\",\"m\":\"minilm\",\"d\":8,\"v\":\"AAB");
	fclose(fp);

	c = sigil_veccache_open(PATH, "minilm", DIM);
	ok(c != NULL, "a truncated final line does not prevent opening");
	eqsz(sigil_veccache_count(c), 1, "the complete entry is still there");
	ok(sigil_veccache_get(c, h, out) == 0, "and still readable");
	sigil_veccache_close(c);
}

static void
test_duplicate_keeps_the_first(void)
{
	sigil_veccache_t *c;
	uint8_t h[32];
	float a[DIM], b[DIM], out[DIM];

	unlink(PATH);
	fill(h, 11);
	vec_of(a, 2.0f);
	vec_of(b, 8.0f);

	c = sigil_veccache_open(PATH, "minilm", DIM);
	sigil_veccache_put(c, h, a);
	sigil_veccache_put(c, h, b);        /* an interrupted run re-embeds */
	eqsz(sigil_veccache_count(c), 1, "a duplicate does not grow the table");
	ok(sigil_veccache_get(c, h, out) == 0 && close_enough(a, out),
	   "the first vector wins");
	sigil_veccache_close(c);
}

/* --- reporting ---------------------------------------------------------- */

static void
test_stats(void)
{
	sigil_veccache_t *c;
	uint8_t h[32];
	float in[DIM], out[DIM];
	uint64_t hits = 0, misses = 0;

	unlink(PATH);
	fill(h, 2);
	vec_of(in, 3.0f);

	c = sigil_veccache_open(PATH, "minilm", DIM);
	sigil_veccache_get(c, h, out);      /* miss */
	sigil_veccache_put(c, h, in);
	sigil_veccache_get(c, h, out);      /* hit */
	sigil_veccache_get(c, h, out);      /* hit */
	sigil_veccache_stats(c, &hits, &misses);

	eqsz((size_t)hits, 2, "hits counted");
	eqsz((size_t)misses, 1, "misses counted");
	sigil_veccache_close(c);
}

static void
test_float16_range(void)
{
	sigil_veccache_t *c;
	uint8_t h[32];
	float in[DIM], out[DIM];
	int i;

	/* Embedding vectors are L2-normalised, so components live in
	 * [-1, 1] -- but a zero, a denormal and a sign must all survive. */
	unlink(PATH);
	fill(h, 4);
	in[0] = 0.0f;
	in[1] = -0.0f;
	in[2] = 1.0f;
	in[3] = -1.0f;
	in[4] = 0.5f;
	in[5] = -0.001f;
	in[6] = 0.000061f;                 /* near the subnormal boundary */
	in[7] = -0.99999f;

	c = sigil_veccache_open(PATH, "minilm", DIM);
	sigil_veccache_put(c, h, in);
	sigil_veccache_close(c);

	c = sigil_veccache_open(PATH, "minilm", DIM);
	ok(sigil_veccache_get(c, h, out) == 0, "range vector round-trips");
	for (i = 0; i < DIM; i++)
		ok(fabsf(in[i] - out[i]) < 0.001f ||
		   (in[i] < 0) == (out[i] < 0),
		   "component preserved to float16 precision, sign intact");
	sigil_veccache_close(c);
}

int
main(void)
{
	test_survives_close();
	test_many();
	test_model_scoping();
	test_truncated_line_is_skipped();
	test_duplicate_keeps_the_first();
	test_stats();
	test_float16_range();
	unlink(PATH);

	if (failures == 0) {
		printf("PASS: %d checks\n", checks);
		return 0;
	}
	printf("FAIL: %d of %d checks failed\n", failures, checks);
	return 1;
}
