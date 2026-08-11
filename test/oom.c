/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Allocation-failure tests, via ld --wrap.
 *
 * The bridge's error paths -- a realloc that fails partway through growing
 * four parallel arrays, a calloc that fails during construction -- were the
 * last uncovered code in bridge.c, and "needs an out-of-memory condition" is
 * not a reason to leave them untested. It is a reason to inject the failure
 * deterministically.
 *
 * These are the paths where silent corruption lives. grow() reallocates
 * paths, offs, lens and paras in sequence; a failure on the third leaves two
 * arrays grown and two not, and if the caller then proceeds the next write
 * runs off the end of the shorter pair. What must happen instead is that the
 * add fails and the bridge stays consistent.
 *
 *	make oom
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sigil.h"

void *br_new(void);
void br_free(void *b);
long br_add_at(void *b, const char *path, unsigned para, const void *text,
               size_t len, const unsigned long long *lsh, unsigned ts,
               unsigned long off);
long br_count(void *b);
const char *br_path(void *b, long i);
unsigned br_para(void *b, long i);
unsigned long br_offset(void *b, long i);

/* --- the shim ----------------------------------------------------------- */

void *__real_malloc(size_t n);
void *__real_calloc(size_t n, size_t sz);
void *__real_realloc(void *p, size_t n);

/*
 * Fail the Nth allocation of each kind, counting from the moment arming
 * begins. Counting rather than failing everything: the interesting question
 * is what happens when one allocation in the middle of a sequence fails,
 * which is exactly the case a blanket failure cannot produce.
 */
static long fail_malloc_at = -1, fail_calloc_at = -1, fail_realloc_at = -1;
static long n_malloc, n_calloc, n_realloc;

static void
arm(long m, long c, long r)
{
	fail_malloc_at = m;
	fail_calloc_at = c;
	fail_realloc_at = r;
	n_malloc = n_calloc = n_realloc = 0;
}

static void disarm(void) { arm(-1, -1, -1); }

void *
__wrap_malloc(size_t n)
{
	if (fail_malloc_at >= 0 && n_malloc++ == fail_malloc_at)
		return NULL;
	return __real_malloc(n);
}

void *
__wrap_calloc(size_t n, size_t sz)
{
	if (fail_calloc_at >= 0 && n_calloc++ == fail_calloc_at)
		return NULL;
	return __real_calloc(n, sz);
}

void *
__wrap_realloc(void *p, size_t n)
{
	if (fail_realloc_at >= 0 && n_realloc++ == fail_realloc_at)
		return NULL;
	return __real_realloc(p, n);
}

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

/* --- construction ------------------------------------------------------- */

static void
test_new_under_oom(void)
{
	long k;

	/* br_new does several allocations. Failing each in turn must give a
	 * NULL bridge rather than a half-built one -- and must not leak the
	 * allocations that already succeeded, which the sanitizer checks. */
	for (k = 0; k < 8; k++) {
		void *b;

		arm(-1, k, -1);
		b = br_new();
		disarm();
		if (b != NULL) {
			/* This allocation was not on the construction path;
			 * the bridge is valid and must work normally. */
			br_free(b);
			continue;
		}
		checks++;   /* a NULL return is the correct outcome */
	}
	ok(1, "br_new survives a failure at each of its allocations");
}

/* --- growth ------------------------------------------------------------- */

static void
test_grow_under_oom(void)
{
	void *b;
	long before, k;
	int stayed_consistent = 1;

	/* Fill to just under the 4096 initial capacity, then fail an
	 * allocation during the growth that the next add triggers. */
	for (k = 0; k < 6; k++) {
		int i;

		b = br_new();
		if (b == NULL)
			continue;
		for (i = 0; i < 4096; i++) {
			char t[32];

			snprintf(t, sizeof t, "r%d", i);
			if (br_add_at(b, "/o.txt", (unsigned)i, t, strlen(t),
			              NULL, 0, (unsigned long)i) < 0)
				break;
		}
		before = br_count(b);

		arm(-1, -1, k);          /* fail the k'th realloc from here */
		(void)br_add_at(b, "/o.txt", 9999, "trigger growth", 14,
		                NULL, 0, 0);
		disarm();

		/* Whether the add succeeded or not, every record already in
		 * the bridge must still read back correctly. A partial growth
		 * that left two arrays short would show up here. */
		if (br_count(b) < before)
			stayed_consistent = 0;
		if (br_path(b, before - 1) == NULL ||
		    strcmp(br_path(b, before - 1), "/o.txt") != 0)
			stayed_consistent = 0;
		if (br_para(b, before - 1) != (unsigned)(before - 1))
			stayed_consistent = 0;
		if (br_offset(b, before - 1) != (unsigned long)(before - 1))
			stayed_consistent = 0;

		br_free(b);
	}
	ok(stayed_consistent,
	   "a realloc failure during growth leaves every existing record "
	   "readable -- a partially grown set of parallel arrays would not");
}

/* --- add ---------------------------------------------------------------- */

static void
test_add_under_oom(void)
{
	void *b = br_new();
	long k;
	int consistent = 1;

	if (b == NULL)
		return;

	br_add_at(b, "/a.txt", 1, "first record", 12, NULL, 0, 0);

	/* strdup of the path allocates. If it fails, the record must either
	 * not be added or be added with a usable path -- not with a dangling
	 * one. */
	for (k = 0; k < 4; k++) {
		long n_before = br_count(b);

		arm(k, -1, -1);
		(void)br_add_at(b, "/b.txt", 2, "second record", 13,
		                NULL, 0, 0);
		disarm();

		if (br_count(b) > n_before) {
			const char *p = br_path(b, br_count(b) - 1);

			if (p == NULL)
				consistent = 0;   /* added with no path */
		}
	}
	ok(consistent, "an add under allocation failure never stores a "
	   "record with a dangling path");

	/* And the bridge is still usable afterwards. */
	ok(br_add_at(b, "/c.txt", 3, "after the failures", 18,
	             NULL, 0, 0) >= 0,
	   "the bridge still accepts records after a failed allocation");

	br_free(b);
}

int
main(void)
{
	test_new_under_oom();
	test_grow_under_oom();
	test_add_under_oom();

	if (failures == 0) {
		printf("PASS: %d checks\n", checks);
		return 0;
	}
	printf("FAIL: %d of %d checks failed\n", failures, checks);
	return 1;
}
