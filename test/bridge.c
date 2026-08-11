/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Tests for cmd/bridge.c -- the layer between libsigil and the 9P server.
 *
 * Every bug this project has actually shipped lived here or beside it, not in
 * the library:
 *
 *   br_add_hex() recomputed a record's BLAKE3 from its *path* when replaying
 *   persisted rows, so every restart silently reassigned every identity and
 *   /similar/<hex>/ could not be reached by a hash the store had just
 *   written;
 *
 *   store.c reported lsh_bits = 256 while the library produced 128, and that
 *   field is the guarantee two stores are comparable;
 *
 *   the batch path disagreed with the per-text path on long paragraphs, twice,
 *   in two different ways.
 *
 * bridge.c is compiled with the system compiler by design -- it is the one
 * file that sees libsigil, and it exports plain C types precisely so the two
 * worlds can meet. That also makes it directly testable with no shim, unlike
 * the rest of cmd/.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sigil.h"

/* The bridge's own surface, declared here rather than pulled from
 * cmd/sigilfs.h: that header includes plan9port types this file cannot see. */
void *br_new(void);
void br_free(void *b);
long br_add_at(void *b, const char *path, unsigned para, const void *text,
               size_t len, const unsigned long long *lsh, unsigned ts,
               unsigned long off);
long br_add(void *b, const char *path, unsigned para, const void *text,
            size_t len, const unsigned long long *lsh, unsigned ts);
long br_add_restore(void *b, const char *path, unsigned para,
                    const char *lshhex, const char *hashhex,
                    unsigned long off, unsigned long len);
void br_flush(void *b);
long br_count(void *b);
const char *br_path(void *b, long i);
unsigned br_para(void *b, long i);
int br_hash(void *b, long i, char *out, size_t outlen);
int br_lsh_hex(void *b, long i, char *out, size_t outlen);
long br_find_hash(void *b, const char *hexhash);
long br_similar(void *b, long i, unsigned maxdist, unsigned *out, long maxout);
unsigned long br_offset(void *b, long i);
unsigned long br_length(void *b, long i);
unsigned br_lsh_bits(void);
int br_have_embedder(void *b);

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
eql(long got, long want, const char *what)
{
	checks++;
	if (got == want)
		return;
	failures++;
	printf("FAIL: %s: got %ld want %ld\n", what, got, want);
}

/* --- the bits the store records about itself ---------------------------- */

static void
test_lsh_bits(void)
{
	/* cmd/store.c hardcoded 256 while the library produced 128, and wrote
	 * that number into the persisted parameters as the compatibility
	 * guarantee. A store built at one width would have loaded against a
	 * store built at another without complaint. */
	eql((long)br_lsh_bits(), SIGIL_LSH_BITS,
	    "br_lsh_bits agrees with the library");
}

/* --- add and read back -------------------------------------------------- */

static void
test_add_readback(void)
{
	void *b = br_new();
	const char *t1 = "the ship sailed south under heavy weather";
	const char *t2 = "navigation of the cape demanded skill";
	char hex[130];
	long i0, i1;

	ok(b != NULL, "bridge constructs");
	if (b == NULL)
		return;

	ok(br_have_embedder(b) == 0, "a fresh bridge has no embedder");

	i0 = br_add_at(b, "/a.txt", 1, t1, strlen(t1), NULL, 100, 0);
	i1 = br_add_at(b, "/b.txt", 7, t2, strlen(t2), NULL, 200, 4096);

	eql(i0, 0, "first add is index 0");
	eql(i1, 1, "second add is index 1");
	eql(br_count(b), 2, "count tracks adds");

	ok(strcmp(br_path(b, 0), "/a.txt") == 0, "path round-trips");
	eql((long)br_para(b, 1), 7, "paragraph number round-trips");
	eql((long)br_offset(b, 1), 4096, "byte offset round-trips");
	eql((long)br_length(b, 1), (long)strlen(t2), "length round-trips");

	/* Out of range must fail rather than return a neighbouring record --
	 * similar.c indexes into this with values from a scan. */
	ok(br_path(b, 99) == NULL, "path past the end is NULL");
	ok(br_hash(b, 99, hex, sizeof hex) != 0, "hash past the end fails");
	eql((long)br_offset(b, 99), 0, "offset past the end is 0");

	br_free(b);
}

/* --- identity across a restore ------------------------------------------ */

static void
test_restore_identity(void)
{
	void *b = br_new(), *c;
	const char *text = "content addressing means the address does not move";
	char hash_before[130], hash_after[130], lsh[130];

	if (b == NULL)
		return;

	br_add_at(b, "/orig.txt", 3, text, strlen(text), NULL, 0, 512);
	ok(br_hash(b, 0, hash_before, sizeof hash_before) == 0, "hash reads");
	ok(br_lsh_hex(b, 0, lsh, sizeof lsh) == 0, "lsh reads");

	/* Replay it the way persistence does: the stored hash and lsh come
	 * back from the table, and the record must land on the same identity.
	 * This regressed once -- reload recomputed the hash from the path,
	 * so a restart renamed every record in the store. */
	c = br_new();
	if (c == NULL) {
		br_free(b);
		return;
	}
	br_add_restore(c, "/orig.txt", 3, lsh, hash_before, 512,
	               strlen(text));
	ok(br_hash(c, 0, hash_after, sizeof hash_after) == 0,
	   "restored record has a hash");
	ok(strcmp(hash_before, hash_after) == 0,
	   "a restored record keeps its BLAKE3 -- content addressing whose "
	   "address changes across a restart is not content addressing");
	eql((long)br_offset(c, 0), 512, "restored offset survives");
	eql((long)br_length(c, 0), (long)strlen(text),
	    "restored length survives");

	/* And it must be findable by that hash, which is how /similar/<hex>/
	 * is reached. */
	eql(br_find_hash(c, hash_after), 0, "restored record is findable");

	br_free(b);
	br_free(c);
}

/* --- lookup ------------------------------------------------------------- */

static void
test_find_and_similar(void)
{
	void *b = br_new();
	unsigned out[64];
	char hex[130];
	long n, i;
	int k;

	if (b == NULL)
		return;

	for (k = 0; k < 50; k++) {
		char text[80];

		snprintf(text, sizeof text,
		         "paragraph %d about ships and weather at sea", k);
		br_add_at(b, "/c.txt", (unsigned)k, text, strlen(text),
		          NULL, 0, (unsigned long)(k * 100));
	}
	eql(br_count(b), 50, "fifty records");

	ok(br_hash(b, 10, hex, sizeof hex) == 0, "hash of record 10");
	eql(br_find_hash(b, hex), 10, "find_hash returns the right index");
	eql(br_find_hash(b, "00"), -1, "a short hash finds nothing");
	eql(br_find_hash(b,
	    "0000000000000000000000000000000000000000000000000000000000000000"),
	    -1, "an absent hash finds nothing");

	/* At distance 0 a record is at least its own neighbour; the caller
	 * filters itself out. Nothing returned may exceed the radius. */
	n = br_similar(b, 10, 0, out, 64);
	ok(n >= 1, "a record is within distance 0 of itself");
	for (i = 0; i < n; i++)
		if (out[i] >= 50) {
			ok(0, "similar returned an out-of-range index");
			break;
		}
	if (i == n)
		ok(1, "every returned index is in range");

	/* A radius wider than the code returns everything. */
	n = br_similar(b, 10, SIGIL_LSH_BITS, out, 64);
	eql(n, 50, "a full-width radius matches every record");

	/* max_out is respected: similar.c sizes a fixed array from it. */
	n = br_similar(b, 10, SIGIL_LSH_BITS, out, 7);
	ok(n <= 7, "max_out is honoured");

	br_free(b);
}

/* --- growth ------------------------------------------------------------- */

static void
test_growth(void)
{
	void *b = br_new();
	const int N = 20000;   /* past the 4096 initial capacity, twice over */
	int k;
	char hex[130];

	if (b == NULL)
		return;

	for (k = 0; k < N; k++) {
		char text[64];

		snprintf(text, sizeof text, "record %d", k);
		if (br_add_at(b, "/g.txt", (unsigned)k, text, strlen(text),
		              NULL, 0, (unsigned long)k) < 0) {
			ok(0, "add during growth");
			br_free(b);
			return;
		}
	}
	eql(br_count(b), N, "every add survived the reallocations");

	/* The four parallel arrays in the bridge grow separately from the
	 * seven inside the store. One left behind is invisible until that
	 * field is read. */
	eql((long)br_para(b, 0), 0, "first record's para intact after growth");
	eql((long)br_para(b, N - 1), N - 1, "last record's para intact");
	eql((long)br_offset(b, N - 1), N - 1, "last record's offset intact");
	ok(br_path(b, N - 1) != NULL && strcmp(br_path(b, N - 1), "/g.txt") == 0,
	   "last record's path intact");
	ok(br_hash(b, N - 1, hex, sizeof hex) == 0, "last record's hash reads");

	br_free(b);
}

/* --- flush without an embedder ------------------------------------------ */

static void
test_flush_is_safe(void)
{
	void *b = br_new();

	if (b == NULL)
		return;

	/* br_flush is called from three places -- index, commit, and a
	 * similarity walk -- and must be harmless when nothing is queued and
	 * no embedder is loaded. A flush that assumed an embedder would take
	 * down a server started without -e. */
	br_flush(b);
	br_flush(NULL);
	eql(br_count(b), 0, "flush on an empty bridge changes nothing");

	br_add_at(b, "/f.txt", 1, "some text here to hash", 22, NULL, 0, 0);
	br_flush(b);
	eql(br_count(b), 1, "flush with no embedder leaves the record");

	br_free(b);
}

int
main(void)
{
	test_lsh_bits();
	test_add_readback();
	test_restore_identity();
	test_find_and_similar();
	test_growth();
	test_flush_is_safe();

	if (failures == 0) {
		printf("PASS: %d checks\n", checks);
		return 0;
	}
	printf("FAIL: %d of %d checks failed\n", failures, checks);
	return 1;
}
