/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Unit tests for libsigil.
 *
 * differential.c proves the SIMD kernels agree with their scalar twins, and
 * semantic.c proves the LSH bits carry meaning. Neither touches the record
 * layout, the store, the trit encoding, or the simhash projection -- and every
 * bug found in this project so far lived in code no test executed:
 *
 *   a record's BLAKE3 changed across a restart, because reload recomputed it
 *   from the path rather than reading the stored hash;
 *
 *   lsh_bits was recorded as 256 while the library produced 128, and that
 *   field is the guarantee that two stores are comparable;
 *
 *   a directory cache flush failed silently and served stale results.
 *
 * The theme is that all three were invisible until something ran at scale.
 * These tests are the cheap half of fixing that; prop.c is the other half.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sigil.h"
#include "sigil_embed.h"

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

/* --- record layout ------------------------------------------------------ */

static void
test_layout(void)
{
	sigil_t s;

	/* The on-disk format depends on this, and a _Static_assert already
	 * guards it at compile time. Checking it again at runtime catches a
	 * mismatched header, which the assert cannot: a caller compiled
	 * against a different sigil.h links fine and reads garbage. */
	eqsz(sizeof s, SIGIL_SIZE, "sigil_t is 64 bytes");
	eqsz(SIGIL_LSH_WORDS * 8, SIGIL_LSH_BITS / 8, "lsh words match bits");
	eqsz(sizeof s.hash, SIGIL_HASH_LEN, "hash is BLAKE3-256");
}

/* --- identity ----------------------------------------------------------- */

static void
test_identity(void)
{
	sigil_t a, b, c;
	const char *t1 = "the ship sailed south under heavy weather";
	const char *t2 = "the ship sailed south under heavy weathe";

	sigil_generate_para(t1, strlen(t1), 1, 0, 0, NULL, &a);
	sigil_generate_para(t1, strlen(t1), 1, 0, 0, NULL, &b);
	sigil_generate_para(t2, strlen(t2), 1, 0, 0, NULL, &c);

	ok(memcmp(a.hash, b.hash, SIGIL_HASH_LEN) == 0,
	   "same text gives the same hash");
	ok(memcmp(a.hash, c.hash, SIGIL_HASH_LEN) != 0,
	   "one byte of difference changes the hash");

	/* Content addressing means the address is a function of the content
	 * alone. A record restored from persistence must land on the same
	 * hash, and it did not: reload recomputed it from the path, so every
	 * restart silently reassigned every identity. */
	sigil_generate_para(t1, strlen(t1), 2, 0, 0, NULL, &b);
	ok(memcmp(a.hash, b.hash, SIGIL_HASH_LEN) == 0,
	   "paragraph number does not change the content hash");
	eqsz(b.para, 2, "paragraph number is recorded");
}

/* --- trits -------------------------------------------------------------- */

static void
test_trits(void)
{
	sigil_trits_t in, out;
	uint16_t packed;
	int roundtrip = 0, rejected = 0;
	long v;

	/* Base 3, not two-bits-per-trit: all 3^6 = 729 packed values are legal
	 * and round-trip, and every one of the remaining 64807 16-bit values
	 * is rejected as corruption. A 2-bit scheme would waste a quarter of
	 * the encoding space and decode corruption as valid data. */
	for (v = 0; v < SIGIL_TRIT_MAX; v++) {
		if (sigil_trits_unpack((uint16_t)v, &in) != 0)
			continue;
		packed = sigil_trits_pack(&in);
		if (packed != (uint16_t)v)
			continue;
		if (sigil_trits_unpack(packed, &out) == 0 &&
		    memcmp(&in, &out, sizeof in) == 0)
			roundtrip++;
	}
	eqsz((size_t)roundtrip, SIGIL_TRIT_MAX,
	     "all 729 trit values round-trip");

	for (v = SIGIL_TRIT_MAX; v < 65536; v++) {
		if (sigil_trits_unpack((uint16_t)v, &out) != 0)
			rejected++;
		if (sigil_trits_valid((uint16_t)v))
			rejected--;   /* valid() and unpack() must agree */
	}
	eqsz((size_t)rejected, 65536 - SIGIL_TRIT_MAX,
	     "every out-of-range packed value is rejected");
}

/* --- store -------------------------------------------------------------- */

static void
test_store(void)
{
	sigil_store_t st;
	sigil_t s, got;
	size_t i;
	const size_t N = 10000;

	ok(sigil_store_init(&st, 4) == 0, "store init with a tiny capacity");

	/* Push past the initial capacity several times: store_grow reallocates
	 * seven parallel arrays, and a record that survives that is a record
	 * whose fields were all copied. */
	for (i = 0; i < N; i++) {
		char text[64];

		snprintf(text, sizeof text, "paragraph number %zu", i);
		sigil_generate_para(text, strlen(text), (uint32_t)i,
		                    (uint32_t)(1000 + i), 0, NULL, &s);
		if (sigil_store_push(&st, &s) < 0)
			break;
	}
	eqsz(st.count, N, "every push landed");

	for (i = 0; i < N; i++) {
		char text[64];
		sigil_t want;

		snprintf(text, sizeof text, "paragraph number %zu", i);
		sigil_generate_para(text, strlen(text), (uint32_t)i,
		                    (uint32_t)(1000 + i), 0, NULL, &want);
		if (sigil_store_get(&st, i, &got) != 0) {
			ok(0, "store_get within range");
			break;
		}
		if (memcmp(got.hash, want.hash, SIGIL_HASH_LEN) != 0 ||
		    got.para != want.para || got.timestamp != want.timestamp) {
			ok(0, "record survives the array growth intact");
			break;
		}
	}
	if (i == N)
		ok(1, "all records survive growth intact");

	ok(sigil_store_get(&st, N, &got) != 0, "get past the end fails");
	sigil_store_free(&st);
}

/* --- simhash ------------------------------------------------------------ */

static void
test_simhash(void)
{
	sigil_simhash_t a, b;
	float v[384], w[384];
	uint64_t ha[SIGIL_LSH_WORDS], hb[SIGIL_LSH_WORDS];
	int i, bits = 0;

	for (i = 0; i < 384; i++) {
		v[i] = (float)((i * 37 % 101) - 50) / 50.0f;
		w[i] = -v[i];
	}

	ok(sigil_simhash_init(&a, 384, 0x5191c0deULL) == 0, "simhash init");
	ok(sigil_simhash_init(&b, 384, 0x5191c0deULL) == 0, "same seed init");

	sigil_simhash_project(&a, v, ha);
	sigil_simhash_project(&b, v, hb);
	ok(memcmp(ha, hb, sizeof ha) == 0,
	   "the same seed gives the same bits -- a store's bits are only "
	   "meaningful against the seed that made them");

	/* A vector and its negation fall on opposite sides of every
	 * hyperplane through the origin, so every bit flips. */
	sigil_simhash_project(&a, w, hb);
	for (i = 0; i < SIGIL_LSH_WORDS; i++)
		bits += __builtin_popcountll(ha[i] ^ hb[i]);
	eqsz((size_t)bits, SIGIL_LSH_BITS, "negated vector flips every bit");

	sigil_simhash_free(&a);
	sigil_simhash_free(&b);

	/* A different seed must give different bits, or the seed is not
	 * actually parameterising anything. */
	ok(sigil_simhash_init(&b, 384, 0xdeadbeefULL) == 0, "other seed init");
	sigil_simhash_project(&b, v, hb);
	ok(memcmp(ha, hb, sizeof ha) != 0, "a different seed gives other bits");
	sigil_simhash_free(&b);
}

/* --- hamming and scan --------------------------------------------------- */

static void
test_hamming(void)
{
	uint64_t a[SIGIL_LSH_WORDS], b[SIGIL_LSH_WORDS];

	memset(a, 0, sizeof a);
	memset(b, 0, sizeof b);
	eqsz(sigil_hamming(a, b), 0, "identical codes are distance 0");

	memset(b, 0xff, sizeof b);
	eqsz(sigil_hamming(a, b), SIGIL_LSH_BITS, "inverted codes are maximal");

	b[0] = 1;
	memset(b + 1, 0, sizeof b - sizeof b[0]);
	eqsz(sigil_hamming(a, b), 1, "one bit of difference is distance 1");
}

/* --- embedder contract -------------------------------------------------- */

static void
test_embedder_contract(void)
{
	sigil_embedder_t *e = sigil_embedder_hash_nonsemantic(384);
	float v[384];
	double norm = 0.0;
	int i;

	ok(e != NULL, "fallback embedder constructs");
	if (e == NULL)
		return;

	eqsz(e->dim(e), 384, "dim reports what was asked for");
	ok(strstr(e->name(e), "nonsemantic") != NULL,
	   "the fallback names itself honestly -- it must not be mistakable "
	   "for a real embedder in a stack trace");

	ok(e->embed(e, "hello world", 11, v) == 0, "embed succeeds");
	for (i = 0; i < 384; i++)
		norm += (double)v[i] * v[i];
	ok(norm > 0.99 && norm < 1.01,
	   "output is L2-normalised: SimHash reads only sign(dot), so "
	   "magnitude must not leak into the bits");

	/* Optional in the vtable, and callers loop over embed() when absent.
	 * A backend that leaves it uninitialised rather than NULL is a wild
	 * pointer call, which is why this is checked rather than assumed. */
	ok(e->embed_batch == NULL,
	   "a backend without a batch path reports NULL, not garbage");

	e->destroy(e);
}

int
main(void)
{
	test_layout();
	test_identity();
	test_trits();
	test_store();
	test_simhash();
	test_hamming();
	test_embedder_contract();

	if (failures == 0) {
		printf("PASS: %d checks\n", checks);
		return 0;
	}
	printf("FAIL: %d of %d checks failed\n", failures, checks);
	return 1;
}
