/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Does one SYCL kernel match four hand-written ones?
 *
 * The maintenance argument for oneAPI is strong -- one source instead of
 * AVX2, SSE4.2, NEON and scalar, each needing its own differential test. The
 * argument only holds if the generated code is competitive, and the scan is
 * memory-bound popcount-of-XOR, where the hand-written version maps to a
 * single instruction and the compiler has to recognise the pattern.
 *
 * Correctness first: the scalar kernel is the definition of correct, so a
 * SYCL result that disagrees is a failure regardless of speed.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sigil.h"

extern int sigil_sycl_init(int prefer_gpu);
extern const char *sigil_sycl_device_name(void);
extern size_t sigil_scan_similar_sycl(const uint64_t *lsh, size_t count,
                                      const uint64_t *query,
                                      uint32_t max_distance,
                                      uint32_t *out, size_t max_out);
extern void sigil_sycl_shutdown(void);

#define N        10000000
#define MAXOUT   65536
#define REPS     5

static double now_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static uint64_t rng_state = 0x5191c0deULL;

static uint64_t rng(void)
{
	uint64_t z = (rng_state += 0x9e3779b97f4a7c15ULL);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
	return z ^ (z >> 31);
}

int main(void)
{
	uint64_t *lsh = malloc((size_t)N * SIGIL_LSH_WORDS * sizeof(uint64_t));
	uint32_t *out_a = malloc(MAXOUT * sizeof(uint32_t));
	uint32_t *out_b = malloc(MAXOUT * sizeof(uint32_t));
	uint64_t query[SIGIL_LSH_WORDS];
	sigil_store_t st;
	double t, best_scalar, best_simd, best_sycl;
	size_t na, nb;

	if (lsh == NULL || out_a == NULL || out_b == NULL) {
		fprintf(stderr, "out of memory\n");
		return 1;
	}
	for (size_t i = 0; i < (size_t)N * SIGIL_LSH_WORDS; i++)
		lsh[i] = rng();
	for (int i = 0; i < SIGIL_LSH_WORDS; i++)
		query[i] = rng();

	memset(&st, 0, sizeof st);
	st.count = N;
	st.lsh = lsh;

	/* A threshold that matches a realistic fraction. At 128 bits of random
	 * data the mean distance is 64, so 50 keeps the result set small enough
	 * that compaction is not what is being timed. */
	const uint32_t thresh = 50;

	printf("%d records, %d-bit codes, threshold %u\n\n",
	       N, SIGIL_LSH_BITS, thresh);

	best_scalar = 1e30;
	for (int r = 0; r < REPS; r++) {
		t = now_ms();
		na = sigil_scan_similar_scalar(&st, query, thresh, out_a, MAXOUT);
		t = now_ms() - t;
		if (t < best_scalar) best_scalar = t;
	}
	printf("scalar      %8.2f ms   %zu matches\n", best_scalar, na);

	best_simd = 1e30;
	for (int r = 0; r < REPS; r++) {
		t = now_ms();
		nb = sigil_scan_similar_simd(&st, query, thresh, out_b, MAXOUT);
		t = now_ms() - t;
		if (t < best_simd) best_simd = t;
	}
	printf("hand SIMD   %8.2f ms   %zu matches   %s\n", best_simd, nb,
	       (nb == na && memcmp(out_a, out_b, na * sizeof(uint32_t)) == 0)
	       ? "agrees" : "DISAGREES WITH SCALAR");

	int kind = sigil_sycl_init(0);
	if (kind < 0) {
		printf("\nSYCL unavailable\n");
		return 0;
	}
	printf("\nSYCL device: %s (%s)\n", sigil_sycl_device_name(),
	       kind == 1 ? "gpu" : kind == 2 ? "accelerator" : "cpu");

	best_sycl = 1e30;
	for (int r = 0; r < REPS; r++) {
		t = now_ms();
		nb = sigil_scan_similar_sycl(lsh, N, query, thresh, out_b, MAXOUT);
		t = now_ms() - t;
		if (t < best_sycl) best_sycl = t;
	}
	printf("SYCL        %8.2f ms   %zu matches   %s\n", best_sycl, nb,
	       (nb == na && memcmp(out_a, out_b, na * sizeof(uint32_t)) == 0)
	       ? "agrees" : "DISAGREES WITH SCALAR");

	printf("\nSYCL vs hand-written SIMD: %.2fx\n", best_simd / best_sycl);
	printf("(SYCL timing includes the host->device copy, which is the\n"
	       " honest number for a store that does not live on the device)\n");

	sigil_sycl_shutdown();
	free(lsh);
	free(out_a);
	free(out_b);
	return 0;
}
