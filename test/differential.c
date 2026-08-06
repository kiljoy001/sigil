/*
 * Differential tests: scalar vs AVX2.
 *
 * A wrong SIMD kernel does not crash. It returns a slightly wrong Hamming
 * distance, which surfaces as a slightly wrong similarity result that looks
 * entirely plausible and stays wrong forever. Comparing against the scalar
 * reference over random input is the check that catches it.
 *
 * Edge cases are deliberate: counts around the 8- and 16-lane boundaries
 * exercise the scalar tail, and unsigned timestamps near 2^31 exercise the
 * signed-compare bias.
 */

#include "sigil.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng_state = 0x853c49e6748fea9bULL;

static uint32_t rnd(void)
{
	rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
	return (uint32_t)(rng_state >> 32);
}

static int failures;

static void check(const char *what, size_t n_scalar, const uint32_t *a,
                  size_t n_avx2, const uint32_t *b)
{
	if (n_scalar != n_avx2) {
		printf("  FAIL %s: scalar found %zu, avx2 found %zu\n",
		       what, n_scalar, n_avx2);
		failures++;
		return;
	}
	if (memcmp(a, b, n_scalar * sizeof(uint32_t)) != 0) {
		printf("  FAIL %s: same count (%zu) but different indices\n",
		       what, n_scalar);
		failures++;
		return;
	}
}

static void run_case(size_t count)
{
	sigil_store_t st;
	uint32_t *out_s, *out_v;

	if (sigil_store_init(&st, count ? count : 1) != 0) {
		printf("  FAIL: store_init(%zu)\n", count);
		failures++;
		return;
	}

	for (size_t i = 0; i < count; i++) {
		sigil_t s;

		memset(&s, 0, sizeof(s));
		for (int j = 0; j < SIGIL_HASH_LEN; j++)
			s.hash[j] = (uint8_t)rnd();
		s.lsh = rnd();
		/* Straddle 2^31 so the signed-compare bias is exercised. */
		s.timestamp = rnd();
		s.category  = (uint16_t)(rnd() & 0x3f);
		s.trits     = (uint16_t)(rnd() % SIGIL_TRIT_MAX);
		if (sigil_store_push(&st, &s) < 0) {
			printf("  FAIL: push at %zu\n", i);
			failures++;
			sigil_store_free(&st);
			return;
		}
	}

	out_s = malloc((count + 1) * sizeof(uint32_t));
	out_v = malloc((count + 1) * sizeof(uint32_t));
	if (!out_s || !out_v) {
		printf("  FAIL: out-buffer alloc\n");
		failures++;
		free(out_s); free(out_v);
		sigil_store_free(&st);
		return;
	}

	/* Similarity across the full distance range, including 0 and 32. */
	for (uint32_t d = 0; d <= 32; d += 4) {
		uint32_t q = rnd();
		size_t a = sigil_scan_similar_scalar(&st, q, d, out_s, count + 1);
		size_t b = sigil_scan_similar_avx2(&st, q, d, out_v, count + 1);

		check("similar", a, out_s, b, out_v);
	}

	/* Time ranges, including empty, full, and single-point. */
	{
		struct { uint32_t lo, hi; } ranges[] = {
			{ 0, 0xffffffffu },          /* everything          */
			{ 0xffffffffu, 0 },          /* empty (lo > hi)     */
			{ 0, 0x7fffffffu },          /* lower half          */
			{ 0x80000000u, 0xffffffffu },/* upper half          */
			{ 0x7ffffff0u, 0x80000010u },/* straddles the bias  */
		};

		for (size_t r = 0; r < sizeof(ranges) / sizeof(ranges[0]); r++) {
			size_t a = sigil_scan_timerange_scalar(&st, ranges[r].lo,
			                ranges[r].hi, out_s, count + 1);
			size_t b = sigil_scan_timerange_avx2(&st, ranges[r].lo,
			                ranges[r].hi, out_v, count + 1);

			check("timerange", a, out_s, b, out_v);
		}
	}

	/* Categories, including one guaranteed absent. */
	for (uint16_t c = 0; c < 8; c++) {
		size_t a = sigil_scan_category_scalar(&st, c, out_s, count + 1);
		size_t b = sigil_scan_category_avx2(&st, c, out_v, count + 1);

		check("category", a, out_s, b, out_v);
	}
	{
		size_t a = sigil_scan_category_scalar(&st, 0xffff, out_s, count + 1);
		size_t b = sigil_scan_category_avx2(&st, 0xffff, out_v, count + 1);

		check("category-absent", a, out_s, b, out_v);
	}

	free(out_s);
	free(out_v);
	sigil_store_free(&st);
}

/* Every packed value must round-trip, and every out-of-range value must be
 * rejected rather than silently decoded. */
static void test_trits(void)
{
	for (unsigned p = 0; p < SIGIL_TRIT_MAX; p++) {
		sigil_trits_t t;

		if (sigil_trits_unpack((uint16_t)p, &t) != 0) {
			printf("  FAIL: valid packed value %u rejected\n", p);
			failures++;
			return;
		}
		if (sigil_trits_pack(&t) != (uint16_t)p) {
			printf("  FAIL: trit round-trip broke at %u\n", p);
			failures++;
			return;
		}
	}

	for (unsigned p = SIGIL_TRIT_MAX; p <= 0xffff; p++) {
		sigil_trits_t t;

		if (sigil_trits_unpack((uint16_t)p, &t) == 0) {
			printf("  FAIL: corrupt value %u accepted\n", p);
			failures++;
			return;
		}
	}
	printf("  ok   trits: all %d values round-trip, %d corrupt values rejected\n",
	       SIGIL_TRIT_MAX, 0x10000 - SIGIL_TRIT_MAX);
}

int main(void)
{
	/* Counts chosen to land on and around the 8- and 16-lane boundaries. */
	const size_t counts[] = {
		0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
		100, 1000, 4096, 10000
	};

	printf("sigil differential tests (AVX2 %s)\n",
	       sigil_have_avx2() ? "available" : "NOT available — comparing scalar to itself");
	printf("sizeof(sigil_t) = %zu\n", sizeof(sigil_t));

	test_trits();

	for (size_t i = 0; i < sizeof(counts) / sizeof(counts[0]); i++) {
		run_case(counts[i]);
		printf("  ok   scan: n=%zu\n", counts[i]);
	}

	if (failures) {
		printf("\n%d FAILURES\n", failures);
		return 1;
	}
	printf("\nall passed\n");
	return 0;
}
