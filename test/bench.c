/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Measured throughput.
 *
 * The old design notes claimed a 1B-file scan in 20ms by counting cycles and
 * ignoring DRAM. A full scan moves count * 32 bytes through memory and is
 * bandwidth-bound, not ALU-bound. This reports bytes actually touched and the
 * effective bandwidth so the numbers can be checked against the hardware
 * instead of asserted.
 */

/* clock_gettime is POSIX, not C11. Must precede every include. */
#define _POSIX_C_SOURCE 200112L

#include "sigil.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t rng_state = 0x2545f4914f6cdd1dULL;

static uint32_t rnd(void)
{
	rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
	return (uint32_t)(rng_state >> 32);
}

int main(int argc, char **argv)
{
	size_t count = (argc > 1) ? strtoull(argv[1], NULL, 10) : 10000000;
	sigil_store_t st;
	uint32_t *out;
	double t0, t1;
	size_t found;
	const int reps = 5;

	printf("sigil benchmark — AVX2 %s, n=%zu\n",
	       sigil_have_avx2() ? "on" : "off", count);

	if (sigil_store_init(&st, count) != 0) {
		fprintf(stderr, "alloc failed for %zu records (%.1f GB)\n",
		        count, (double)count * SIGIL_SIZE / 1e9);
		return 1;
	}

	for (size_t i = 0; i < count; i++) {
		sigil_t s;

		memset(&s, 0, sizeof(s));
		s.lsh       = rnd();
		s.timestamp = rnd();
		s.category  = (uint16_t)(rnd() & 0xff);
		s.trits     = (uint16_t)(rnd() % SIGIL_TRIT_MAX);
		sigil_store_push(&st, &s);
	}

	/* Cap results so a permissive query does not spend all its time
	 * writing indices — we are measuring the scan, not the output. */
	out = malloc(65536 * sizeof(uint32_t));
	if (!out) { sigil_store_free(&st); return 1; }

	printf("\n%-22s %10s %12s %10s\n", "kernel", "ms", "GB/s", "matches");

#define BENCH(label, expr, bytes_touched)                                   \
	do {                                                                \
		double best = 1e30;                                         \
		for (int r = 0; r < reps; r++) {                            \
			t0 = now_sec();                                     \
			found = (expr);                                     \
			t1 = now_sec();                                     \
			if (t1 - t0 < best) best = t1 - t0;                 \
		}                                                           \
		printf("%-22s %10.2f %12.2f %10zu\n", label, best * 1e3,    \
		       (double)(bytes_touched) / best / 1e9, found);        \
	} while (0)

	BENCH("similar scalar",
	      sigil_scan_similar_scalar(&st, 0x12345678u, 4, out, 65536),
	      count * sizeof(uint32_t));
	BENCH("similar avx2",
	      sigil_scan_similar_avx2(&st, 0x12345678u, 4, out, 65536),
	      count * sizeof(uint32_t));

	BENCH("timerange scalar",
	      sigil_scan_timerange_scalar(&st, 0, 0x01000000u, out, 65536),
	      count * sizeof(uint32_t));
	BENCH("timerange avx2",
	      sigil_scan_timerange_avx2(&st, 0, 0x01000000u, out, 65536),
	      count * sizeof(uint32_t));

	BENCH("category scalar",
	      sigil_scan_category_scalar(&st, 42, out, 65536),
	      count * sizeof(uint16_t));
	BENCH("category avx2",
	      sigil_scan_category_avx2(&st, 42, out, 65536),
	      count * sizeof(uint16_t));

	printf("\nnotes:\n");
	printf("  LSH pass touches %.2f GB of %.2f GB total (SoA split)\n",
	       (double)count * sizeof(uint32_t) / 1e9,
	       (double)count * SIGIL_SIZE / 1e9);
	printf("  a whole-record scan would move %.1fx more memory\n",
	       (double)SIGIL_SIZE / sizeof(uint32_t));
	printf("  a kernel reporting %zu matches hit the result cap and exited\n",
	       (size_t)65536);
	printf("  early — its GB/s is not a full-scan figure. The similarity\n");
	printf("  row scans everything and is the one to trust.\n");

	free(out);
	sigil_store_free(&st);
	return 0;
}
