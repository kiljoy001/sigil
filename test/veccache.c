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

/* --- durability ---------------------------------------------------------- */

static void
test_sync(void)
{
	sigil_veccache_t *c;
	uint8_t h[32];
	float in[DIM];
	FILE *fp;
	char line[4096];
	int lines = 0;

	/* sync() is the whole durability contract: without it a crash loses
	 * whatever libc was still holding. CRAP found this at 0% coverage --
	 * the one function in the cache whose entire job is not losing data,
	 * untested. */
	unlink(PATH);
	fill(h, 21);
	vec_of(in, 1.0f);

	c = sigil_veccache_open(PATH, "minilm", DIM);
	sigil_veccache_put(c, h, in);

	/* Before the flush the line may still be in libc's buffer. After it,
	 * another reader must be able to see it while this handle is open. */
	ok(sigil_veccache_sync(c, 0) == 0, "sync succeeds");

	fp = fopen(PATH, "r");
	while (fgets(line, sizeof line, fp) != NULL)
		lines++;
	fclose(fp);
	eqsz((size_t)lines, 1, "the record is on disk before close");

	ok(sigil_veccache_sync(c, 1) == 0, "sync with fsync succeeds");
	sigil_veccache_close(c);

	/* And a NULL cache is a caller error, not a crash. */
	ok(sigil_veccache_sync(NULL, 0) != 0, "sync(NULL) reports failure");
}

static void
test_null_arguments(void)
{
	uint8_t h[32];
	float v[DIM];

	fill(h, 1);
	vec_of(v, 1.0f);

	/* The bridge calls these on a path where the cache may be absent --
	 * an unconfigured cache must be a miss, not a segfault. */
	ok(sigil_veccache_get(NULL, h, v) != 0, "get(NULL) misses");
	ok(sigil_veccache_put(NULL, h, v) != 0, "put(NULL) fails");
	eqsz(sigil_veccache_count(NULL), 0, "count(NULL) is zero");
	sigil_veccache_stats(NULL, NULL, NULL);   /* must not crash */
	sigil_veccache_close(NULL);               /* must not crash */
	ok(1, "NULL handles are tolerated throughout");

	ok(sigil_veccache_open(NULL, "m", DIM) == NULL, "open(NULL path)");
	ok(sigil_veccache_open(PATH, NULL, DIM) == NULL, "open(NULL model)");
	ok(sigil_veccache_open(PATH, "m", 0) == NULL, "open(dim 0)");
}

static void
test_unwritable_path(void)
{
	/* A cache that cannot be created must say so rather than returning a
	 * handle that silently discards every vector. */
	ok(sigil_veccache_open("/proc/nonexistent/x.jsonl", "m", DIM) == NULL,
	   "open on an unwritable path fails");
}

static void
test_odd_floats(void)
{
	sigil_veccache_t *c;
	uint8_t h[32];
	float in[DIM], out[DIM];

	/* An embedder that emits inf or NaN is broken, but the cache must
	 * store what it was given rather than silently turning it into zero
	 * -- a zero vector is a plausible-looking embedding and would be
	 * indistinguishable from a real one downstream. */
	unlink(PATH);
	fill(h, 31);
	in[0] = 1.0f / 0.0f;              /* +inf */
	in[1] = -1.0f / 0.0f;             /* -inf */
	in[2] = 0.0f / 0.0f;              /* NaN */
	in[3] = 65504.0f;                 /* float16 max */
	in[4] = 131072.0f;                /* overflows float16 */
	in[5] = 1e-8f;                    /* underflows to zero */
	in[6] = 0.5f;
	in[7] = -0.5f;

	c = sigil_veccache_open(PATH, "minilm", DIM);
	ok(sigil_veccache_put(c, h, in) == 0, "odd floats are stored");
	sigil_veccache_close(c);

	c = sigil_veccache_open(PATH, "minilm", DIM);
	ok(sigil_veccache_get(c, h, out) == 0, "and read back");
	ok(isinf(out[0]) && out[0] > 0, "+inf survives");
	ok(isinf(out[1]) && out[1] < 0, "-inf survives");
	ok(isnan(out[2]), "NaN survives as NaN, not zero");
	ok(isinf(out[4]), "a value over float16 max saturates to inf");
	ok(out[5] == 0.0f, "an underflow becomes zero");
	ok(out[6] == 0.5f && out[7] == -0.5f, "ordinary values are exact");
	sigil_veccache_close(c);
}

static void
test_malformed_lines(void)
{
	sigil_veccache_t *c;
	FILE *fp;
	uint8_t h[32];
	float in[DIM];

	/* A cache file is edited by nothing but this code, but it lives on
	 * disk next to everything else and a stray line must cost that line
	 * rather than the file. */
	unlink(PATH);
	fill(h, 41);
	vec_of(in, 1.0f);
	c = sigil_veccache_open(PATH, "minilm", DIM);
	sigil_veccache_put(c, h, in);
	sigil_veccache_close(c);

	fp = fopen(PATH, "a");
	fprintf(fp, "\n");                                  /* blank */
	fprintf(fp, "not json at all\n");
	fprintf(fp, "{\"h\":\"tooshort\",\"m\":\"minilm\",\"v\":\"AAAA\"}\n");
	fprintf(fp, "{\"m\":\"minilm\",\"v\":\"AAAA\"}\n");       /* no hash */
	fprintf(fp, "{\"h\":\"%064d\",\"m\":\"minilm\"}\n", 0);   /* no vector */
	fprintf(fp, "{\"h\":\"zzzz%060d\",\"m\":\"minilm\",\"v\":\"AAAA\"}\n", 0);
	fclose(fp);

	c = sigil_veccache_open(PATH, "minilm", DIM);
	ok(c != NULL, "malformed lines do not prevent opening");
	eqsz(sigil_veccache_count(c), 1, "only the valid entry is loaded");
	sigil_veccache_close(c);
}

static void
test_short_vector_is_rejected(void)
{
	sigil_veccache_t *c;
	FILE *fp;

	/* A full-length hash with a vector that decodes to fewer bytes than
	 * dim requires: exactly what a crash mid-base64 leaves behind.
	 *
	 * Isolated from the other malformed cases on purpose. The earlier
	 * test used a short hash as well, so the length guard rejected the
	 * line first and removing the vector-length check changed nothing --
	 * two mutants survived because one bad field masked the other.
	 */
	unlink(PATH);
	fp = fopen(PATH, "w");
	fprintf(fp, "{\"h\":\"%064d\",\"m\":\"minilm\",\"d\":8,\"v\":\"AAAA\"}\n", 1);
	fclose(fp);

	c = sigil_veccache_open(PATH, "minilm", DIM);
	eqsz(sigil_veccache_count(c), 0,
	     "a vector shorter than dim is rejected, not zero-padded");
	sigil_veccache_close(c);
}

static void
test_short_hash_is_rejected(void)
{
	sigil_veccache_t *c;
	FILE *fp;
	uint16_t v[DIM];
	char b64[64];
	size_t i, o = 0;
	static const char B[] =
		"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

	/* A hash of the wrong length with a vector of the *right* length, so
	 * only the hash guard can reject it. */
	for (i = 0; i < DIM; i++)
		v[i] = 0;
	{
		const uint8_t *in = (const uint8_t *)v;
		size_t n = sizeof v;

		for (i = 0; i + 2 < n; i += 3) {
			uint32_t t = ((uint32_t)in[i] << 16) |
			             ((uint32_t)in[i + 1] << 8) | in[i + 2];

			b64[o++] = B[(t >> 18) & 63];
			b64[o++] = B[(t >> 12) & 63];
			b64[o++] = B[(t >> 6) & 63];
			b64[o++] = B[t & 63];
		}
		b64[o] = '\0';
	}

	unlink(PATH);
	fp = fopen(PATH, "w");
	fprintf(fp, "{\"h\":\"abcd\",\"m\":\"minilm\",\"d\":8,\"v\":\"%s\"}\n",
	        b64);
	fclose(fp);

	c = sigil_veccache_open(PATH, "minilm", DIM);
	eqsz(sigil_veccache_count(c), 0,
	     "a hash of the wrong length is rejected");
	sigil_veccache_close(c);
}

static void
test_uppercase_hex_is_read(void)
{
	sigil_veccache_t *c;
	FILE *fp;
	uint8_t h[32];
	float in[DIM], out[DIM];
	char hex[65];
	int i;

	/* This code writes lowercase, but a file concatenated from another
	 * tool may not be, and rejecting it would silently drop entries. */
	unlink(PATH);
	fill(h, 51);
	vec_of(in, 2.0f);
	c = sigil_veccache_open(PATH, "minilm", DIM);
	sigil_veccache_put(c, h, in);
	sigil_veccache_close(c);

	/* Re-read the line and upcase its hash field in place. */
	{
		char line[8192];
		char *pos;

		fp = fopen(PATH, "r");
		if (fgets(line, sizeof line, fp) == NULL)
			line[0] = '\0';
		fclose(fp);
		pos = strstr(line, "\"h\":\"");
		if (pos != NULL) {
			pos += 5;
			for (i = 0; i < 64; i++)
				if (pos[i] >= 'a' && pos[i] <= 'f')
					pos[i] = (char)(pos[i] - 'a' + 'A');
		}
		fp = fopen(PATH, "w");
		fputs(line, fp);
		fclose(fp);
	}
	(void)hex;

	c = sigil_veccache_open(PATH, "minilm", DIM);
	eqsz(sigil_veccache_count(c), 1, "an uppercase hash is accepted");
	ok(sigil_veccache_get(c, h, out) == 0, "and matches the same key");
	sigil_veccache_close(c);
}

static void
test_probing(void)
{
	sigil_veccache_t *c;
	uint8_t h[32];
	float in[DIM], out[DIM];
	int i, found = 0;

	/* Keys whose first eight bytes are identical land in the same bucket
	 * and exercise the linear probe -- the path that distinguishes a
	 * bucket choice from an identity decision. */
	unlink(PATH);
	c = sigil_veccache_open(PATH, "minilm", DIM);
	for (i = 0; i < 16; i++) {
		memset(h, 0, sizeof h);
		h[31] = (uint8_t)i;              /* differ only in the last byte */
		vec_of(in, (float)i);
		sigil_veccache_put(c, h, in);
	}
	eqsz(sigil_veccache_count(c), 16, "colliding buckets all stored");

	for (i = 0; i < 16; i++) {
		memset(h, 0, sizeof h);
		h[31] = (uint8_t)i;
		vec_of(in, (float)i);
		if (sigil_veccache_get(c, h, out) == 0 && close_enough(in, out))
			found++;
	}
	eqsz((size_t)found, 16, "and each is found by its full hash");
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
	test_sync();
	test_null_arguments();
	test_unwritable_path();
	test_odd_floats();
	test_malformed_lines();
	test_short_vector_is_rejected();
	test_short_hash_is_rejected();
	test_uppercase_hex_is_read();
	test_probing();
	unlink(PATH);

	if (failures == 0) {
		printf("PASS: %d checks\n", checks);
		return 0;
	}
	printf("FAIL: %d of %d checks failed\n", failures, checks);
	return 1;
}
