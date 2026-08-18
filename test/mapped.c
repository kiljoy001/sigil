/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * File-backed stores: save, map, and what must be refused.
 *
 * A mapped store's field arrays point into a mapping of the file rather
 * than the heap, so opening one costs no parsing and no copy. The risk that
 * buys is silent misinterpretation: bytes written under a different LSH
 * width or segment size are not corrupt, they decode into valid records
 * that are wrong. Most of what follows is about refusing those rather than
 * reading them.
 *
 *	make mapped
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

/* Every field encodes i, so a record read from the wrong offset is
 * detectable rather than merely plausible. */
static void
fill(sigil_t *s, size_t i)
{
	int b;

	memset(s, 0, sizeof *s);
	s->lsh[0] = (uint64_t)i * 0x9e3779b97f4a7c15ULL;
	s->lsh[1] = (uint64_t)i * 0xbf58476d1ce4e5b9ULL;
	for (b = 0; b < SIGIL_HASH_LEN; b++)
		s->hash[b] = (uint8_t)(i + (size_t)b);
	s->para = (uint32_t)i;
	s->timestamp = (uint32_t)(i * 7);
	s->category = (uint16_t)i;
	s->trits = (uint16_t)(i % SIGIL_TRIT_MAX);
	s->cluster = (uint32_t)(i * 3);
}

/* Enough to cross segment boundaries: the mapping computes each segment's
 * offset independently, so a store of one segment would prove nothing. */
#define N (SIGIL_SEG_RECS * 2 + 4321)

static const char *PATH = "test-mapped.smap";

static void
build(sigil_store_t *st)
{
	sigil_t s;
	size_t i;

	sigil_store_init(st, 0);
	for (i = 0; i < N; i++) {
		fill(&s, i);
		if (sigil_store_push(st, &s) < 0) {
			ok(0, "push while building the store to save");
			return;
		}
	}
}

/* --- round trip --------------------------------------------------------- */

static void
test_round_trip(void)
{
	sigil_store_t heap, map;
	sigil_t a, b;
	size_t i, bad = 0;

	build(&heap);
	ok(sigil_store_save(&heap, PATH) == 0, "save writes a mapped store");
	ok(sigil_store_map(&map, PATH) == 0, "map opens it");

	ok(map.count == heap.count, "mapped count matches");
	ok(map.nseg == heap.nseg, "mapped segment count matches");

	for (i = 0; i < heap.count; i++) {
		if (sigil_store_get(&heap, i, &a) != 0 ||
		    sigil_store_get(&map, i, &b) != 0 ||
		    memcmp(&a, &b, sizeof a) != 0) {
			if (bad == 0)
				printf("  first differing record: %zu\n", i);
			bad++;
		}
	}
	ok(bad == 0, "every record survives the round trip byte for byte");

	/* The scan is what the store exists for; equal records are not
	 * enough if a kernel walks the mapping differently. */
	{
		uint64_t q[SIGIL_LSH_WORDS] = { 0x1234, 0x5678 };
		uint32_t *o1 = malloc(heap.count * sizeof *o1);
		uint32_t *o2 = malloc(map.count * sizeof *o2);
		size_t n1, n2;

		n1 = sigil_scan_similar_simd(&heap, q, 64, o1, heap.count);
		n2 = sigil_scan_similar_simd(&map, q, 64, o2, map.count);
		ok(n1 == n2 && memcmp(o1, o2, n1 * sizeof *o1) == 0,
		   "a scan over the mapping returns identical indices");
		free(o1);
		free(o2);
	}

	sigil_store_unmap(&map);
	sigil_store_free(&heap);
}

/* --- what must be refused ----------------------------------------------- */

static void
corrupt_header(const char *path, size_t off, uint32_t value)
{
	FILE *f = fopen(path, "r+b");

	if (f == NULL)
		return;
	fseek(f, (long)off, SEEK_SET);
	fwrite(&value, sizeof value, 1, f);
	fclose(f);
}

static void
test_refusals(void)
{
	sigil_store_t st;
	const char *bad = "test-mapped-bad.smap";
	char cmd[256];

	ok(sigil_store_map(&st, "no-such-store.smap") != 0,
	   "a missing file is refused");

	/* A file of zeroes has no magic. It must not be read as an empty
	 * store, which would look like success. */
	{
		FILE *f = fopen(bad, "wb");
		char z[8192];

		memset(z, 0, sizeof z);
		fwrite(z, 1, sizeof z, f);
		fclose(f);
		ok(sigil_store_map(&st, bad) != 0,
		   "a file with no magic is refused, not read as empty");
	}

	/*
	 * The layout fields are the dangerous ones. Bytes written at another
	 * LSH width are not corrupt -- every record decodes, and every one is
	 * wrong. Header offsets: magic 0, version 8, lsh_bits 12, hash_len
	 * 16, seg_shift 20.
	 */
	snprintf(cmd, sizeof cmd, "cp %s %s", PATH, bad);
	if (system(cmd) != 0)
		return;
	corrupt_header(bad, 12, SIGIL_LSH_BITS * 2);
	ok(sigil_store_map(&st, bad) != 0,
	   "a store written at another LSH width is refused");

	if (system(cmd) != 0)
		return;
	corrupt_header(bad, 20, SIGIL_SEG_SHIFT + 1);
	ok(sigil_store_map(&st, bad) != 0,
	   "a store written with another segment size is refused");

	if (system(cmd) != 0)
		return;
	corrupt_header(bad, 8, SIGIL_LSH_BITS + 99);   /* version */
	ok(sigil_store_map(&st, bad) != 0,
	   "a store of another format version is refused");

	unlink(bad);
}

/* --- release ------------------------------------------------------------ */

static void
test_release(void)
{
	sigil_store_t map;
	sigil_t s, want;
	size_t g, bad = 0, i;

	ok(sigil_store_map(&map, PATH) == 0, "map for the release test");

	for (g = 0; g < map.nseg; g++)
		ok(sigil_store_release(&map, g) == 0,
		   g == 0 ? "release accepts a valid segment" : "release ok");

	ok(sigil_store_release(&map, map.nseg) != 0,
	   "release rejects a segment past the end");

	/*
	 * A release is a hint, not a free: the pages are clean and file
	 * backed, so the next touch re-reads them. If it were a free this
	 * loop would fault or read zeroes.
	 */
	for (i = 0; i < map.count; i += 9973) {
		fill(&want, i);
		if (sigil_store_get(&map, i, &s) != 0 ||
		    memcmp(&s, &want, sizeof s) != 0)
			bad++;
	}
	ok(bad == 0, "records still read correctly after every segment is released");

	sigil_store_unmap(&map);
}

/* A heap store must not be unmapped, and a mapped one must not be freed --
 * both would call the wrong deallocator on the segments. The flag in the
 * store is what keeps them apart. */
static void
test_heap_store_has_no_mapping(void)
{
	sigil_store_t st;
	sigil_t s;

	sigil_store_init(&st, 0);
	fill(&s, 1);
	sigil_store_push(&st, &s);
	ok(st.map == NULL, "a heap store carries no mapping");
	ok(sigil_store_release(&st, 0) != 0,
	   "release on a heap store is refused rather than madvising the heap");
	sigil_store_free(&st);
}

int
main(void)
{
	test_round_trip();
	test_refusals();
	test_release();
	test_heap_store_has_no_mapping();

	unlink(PATH);

	if (failures) {
		printf("FAILED: %d of %d checks\n", failures, checks);
		return 1;
	}
	printf("PASS: %d checks\n", checks);
	return 0;
}
