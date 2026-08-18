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

int
main(void)
{
	test_round_trip();
	test_refusals();
	test_empty();

	unlink(PATH);

	if (failures) {
		printf("FAILED: %d of %d checks\n", failures, checks);
		return 1;
	}
	printf("PASS: %d checks\n", checks);
	return 0;
}
