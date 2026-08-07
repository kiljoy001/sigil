/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Threaded scan throughput: static split vs work pool.
 *
 * libsigil is single-threaded on purpose — threading belongs to the caller,
 * and a server that already has request concurrency does not want a nested
 * pool fighting it. This measures what a caller gains by parallelizing over
 * sigil_scan_*_range(), and which of the two obvious strategies wins.
 *
 *   static  — split the store into T equal ranges, one thread each. Zero
 *             coordination. Vulnerable to stragglers if cores differ in
 *             speed, which they do on big.LITTLE and under other load.
 *   pool    — carve the store into small chunks and let T workers pull the
 *             next one atomically. Self-balancing, at the cost of an atomic
 *             per chunk.
 *
 * The scan is bandwidth-bound, so neither strategy can exceed what the memory
 * controllers deliver. On a machine already near its ceiling single-threaded,
 * expect little; on a wide box with spare controllers, expect a lot. That is
 * the thing being measured.
 */

#define _POSIX_C_SOURCE 200809L

#include "sigil.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_THREADS 64
#define CHUNK       65536   /* pool granularity, in records */

static double now_sec(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;

static uint32_t rnd(void)
{
	rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
	return (uint32_t)(rng_state >> 32);
}

struct job {
	const sigil_store_t *st;
	const uint64_t      *query;
	uint32_t             max_distance;
	size_t               lo, hi;      /* static split */
	_Atomic size_t      *cursor;      /* pool */
	size_t               total;
	uint32_t            *out;         /* private per-thread buffer */
	size_t               cap;
	size_t               found;
};

static void *static_worker(void *arg)
{
	struct job *j = arg;

	j->found = sigil_scan_similar_range(j->st, j->query, j->max_distance,
	                                    j->lo, j->hi, j->out, j->cap);
	return NULL;
}

static void *pool_worker(void *arg)
{
	struct job *j = arg;
	size_t n = 0;

	for (;;) {
		size_t lo = atomic_fetch_add(j->cursor, CHUNK);
		size_t hi = lo + CHUNK;

		if (lo >= j->total)
			break;
		if (hi > j->total)
			hi = j->total;
		if (n >= j->cap)
			break;
		n += sigil_scan_similar_range(j->st, j->query, j->max_distance,
		                              lo, hi, j->out + n, j->cap - n);
	}
	j->found = n;
	return NULL;
}

/* Returns best-of-3 wall seconds; writes total matches to *found. */
static double run(int nthreads, int pooled, const sigil_store_t *st,
                  const uint64_t *query, uint32_t d, size_t *found)
{
	double best = 1e30;

	for (int rep = 0; rep < 3; rep++) {
		pthread_t th[MAX_THREADS];
		struct job jobs[MAX_THREADS];
		_Atomic size_t cursor = 0;
		size_t per = st->count / (size_t)nthreads;
		double t0, dt;
		size_t total = 0;

		for (int t = 0; t < nthreads; t++) {
			memset(&jobs[t], 0, sizeof(jobs[t]));
			jobs[t].st = st;
			jobs[t].query = query;
			jobs[t].max_distance = d;
			jobs[t].cursor = &cursor;
			jobs[t].total = st->count;
			jobs[t].lo = (size_t)t * per;
			jobs[t].hi = (t == nthreads - 1) ? st->count
			                                 : (size_t)(t + 1) * per;
			jobs[t].cap = 1u << 16;
			jobs[t].out = malloc(jobs[t].cap * sizeof(uint32_t));
			if (!jobs[t].out)
				return -1.0;
		}

		t0 = now_sec();
		for (int t = 0; t < nthreads; t++)
			pthread_create(&th[t], NULL,
			               pooled ? pool_worker : static_worker, &jobs[t]);
		for (int t = 0; t < nthreads; t++)
			pthread_join(th[t], NULL);
		dt = now_sec() - t0;

		for (int t = 0; t < nthreads; t++) {
			total += jobs[t].found;
			free(jobs[t].out);
		}
		if (dt < best)
			best = dt;
		*found = total;
	}
	return best;
}

int main(int argc, char **argv)
{
	size_t count = (argc > 1) ? strtoull(argv[1], NULL, 10) : 10000000;
	int maxthreads = (argc > 2) ? atoi(argv[2]) : 16;
	sigil_store_t st;
	uint64_t query[SIGIL_LSH_WORDS];
	double bytes;
	size_t found;
	double t1 = 0.0;

	if (maxthreads > MAX_THREADS)
		maxthreads = MAX_THREADS;

	if (sigil_store_init(&st, count) != 0) {
		fprintf(stderr, "alloc failed for %zu records\n", count);
		return 1;
	}
	for (size_t i = 0; i < count; i++) {
		sigil_t s;

		memset(&s, 0, sizeof(s));
		for (int j = 0; j < SIGIL_LSH_WORDS; j++)
			s.lsh[j] = ((uint64_t)rnd() << 32) | rnd();
		s.timestamp = rnd();
		s.category  = (uint16_t)(rnd() & 0xff);
		s.trits     = (uint16_t)(rnd() % SIGIL_TRIT_MAX);
		sigil_store_push(&st, &s);
	}
	for (int j = 0; j < SIGIL_LSH_WORDS; j++)
		query[j] = 0x0123456789abcdefULL * (unsigned long long)(j + 1);

	bytes = (double)count * SIGIL_LSH_WORDS * sizeof(uint64_t);

	printf("threaded similarity scan — SIMD %s, n=%zu (%.2f GB of LSH)\n",
	       sigil_have_simd() ? "on" : "off", count, bytes / 1e9);
	printf("%-8s %10s %10s %8s   %10s %10s %8s\n",
	       "threads", "static ms", "GB/s", "speedup", "pool ms", "GB/s", "speedup");

	for (int t = 1; t <= maxthreads; t *= 2) {
		double s = run(t, 0, &st, query, 40, &found);
		double p = run(t, 1, &st, query, 40, &found);

		if (t == 1)
			t1 = s;
		printf("%-8d %10.2f %10.2f %7.2fx   %10.2f %10.2f %7.2fx\n",
		       t, s * 1e3, bytes / s / 1e9, t1 / s,
		       p * 1e3, bytes / p / 1e9, t1 / p);
	}

	printf("\nThe scan is bandwidth-bound: speedup stops where the memory\n");
	printf("controllers saturate, not where the cores run out.\n");

	sigil_store_free(&st);
	return 0;
}
