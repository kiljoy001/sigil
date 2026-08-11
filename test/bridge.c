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
#include "sigil_embed.h"

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
const char *br_embedder_name(void *b);
unsigned br_embed_dim(void *b);
int br_embedder_load(void *b, const char *path, unsigned long long seed);
int br_embedder_set(void *b, sigil_embedder_t *e, unsigned long long seed);

/*
 * A deterministic stand-in for a real embedder.
 *
 * The batching path -- br_add_at queueing, br_flush embedding and writing the
 * bits back -- can only be reached with an embedder loaded, and
 * br_embedder_load takes a filename. Without a fake, the two functions the
 * CRAP report ranks highest (br_flush at 59.2, br_add_at at 47.8) are
 * untestable without a GPU and a model on disk.
 *
 * The vector is a function of the text, so batch and single-call results are
 * comparable and a wrong write-back shows up as a wrong record rather than as
 * noise.
 */
struct fake {
	size_t dim;
	int batch_calls;
	int single_calls;
	int fail_next;      /* make the next embed fail, to test the fallback */
};

static void
fake_vec(const char *text, size_t len, size_t dim, float *out)
{
	size_t i;
	unsigned h = 2166136261u;

	for (i = 0; i < len; i++) {
		h ^= (unsigned char)text[i];
		h *= 16777619u;
	}
	for (i = 0; i < dim; i++) {
		h = h * 1103515245u + 12345u;
		out[i] = (float)((h >> 16) & 0xffff) / 32768.0f - 1.0f;
	}
}

static int
fake_embed(sigil_embedder_t *self, const char *text, size_t len, float *out)
{
	struct fake *f = self->impl;

	f->single_calls++;
	if (f->fail_next) {
		f->fail_next = 0;
		return -1;
	}
	fake_vec(text, len, f->dim, out);
	return 0;
}

static int
fake_embed_batch(sigil_embedder_t *self, const char **texts,
                 const size_t *lens, size_t n, float *out)
{
	struct fake *f = self->impl;
	size_t i;

	f->batch_calls++;
	if (f->fail_next) {
		f->fail_next = 0;
		return -1;
	}
	for (i = 0; i < n; i++)
		fake_vec(texts[i], lens[i], f->dim, out + i * f->dim);
	return (int)n;
}

static size_t fake_dim(const sigil_embedder_t *s)
{ return ((const struct fake *)s->impl)->dim; }
static const char *fake_name(const sigil_embedder_t *s)
{ (void)s; return "fake-deterministic"; }
static void fake_destroy(sigil_embedder_t *s)
{ free(s->impl); free(s); }

static sigil_embedder_t *
fake_new(size_t dim, int with_batch, struct fake **outf)
{
	sigil_embedder_t *e = calloc(1, sizeof *e);
	struct fake *f = calloc(1, sizeof *f);

	if (e == NULL || f == NULL) {
		free(e);
		free(f);
		return NULL;
	}
	f->dim = dim;
	e->embed = fake_embed;
	e->embed_batch = with_batch ? fake_embed_batch : NULL;
	e->dim = fake_dim;
	e->name = fake_name;
	e->destroy = fake_destroy;
	e->impl = f;
	if (outf != NULL)
		*outf = f;
	return e;
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

/* --- the batching path -------------------------------------------------- */

static void
test_batch_flush(void)
{
	void *b = br_new();
	struct fake *f = NULL;
	sigil_embedder_t *e = fake_new(64, 1, &f);
	char lsh_before[130], lsh_after[130];
	int k;

	if (b == NULL || e == NULL)
		return;

	ok(br_embedder_set(b, e, 0xabcd) == 0, "embedder installs");
	ok(br_have_embedder(b) != 0, "bridge reports an embedder");
	eql((long)br_embed_dim(b), 64, "embed_dim reports the model width");
	ok(strcmp(br_embedder_name(b), "fake-deterministic") == 0,
	   "embedder name is reported -- two stores built with different "
	   "backends are not interchangeable");

	/* A queued record carries fallback bits until the batch flushes. That
	 * is deliberate: the record is addressable by content immediately, and
	 * only its similarity bits are deferred. */
	br_add_at(b, "/q.txt", 1, "first paragraph of text", 23, NULL, 0, 0);
	eql(br_count(b), 1, "the record is in the store before the flush");
	ok(br_lsh_hex(b, 0, lsh_before, sizeof lsh_before) == 0, "lsh reads");
	eql((long)f->batch_calls, 0, "nothing embedded yet");

	br_flush(b);
	eql((long)f->batch_calls, 1, "flush embeds the queue in one call");
	ok(br_lsh_hex(b, 0, lsh_after, sizeof lsh_after) == 0, "lsh reads");
	ok(strcmp(lsh_before, lsh_after) != 0,
	   "flush replaces the fallback bits with embedded ones");

	/* A second flush with nothing queued must not re-embed: br_flush is
	 * called from three places and two of them are on read paths. */
	br_flush(b);
	eql((long)f->batch_calls, 1, "an empty flush does no work");

	/* Crossing Batchmax must flush automatically, or the queue overruns
	 * its fixed arrays. */
	for (k = 0; k < 300; k++) {
		char t[64];

		snprintf(t, sizeof t, "paragraph %d", k);
		br_add_at(b, "/q.txt", (unsigned)k, t, strlen(t), NULL, 0, 0);
	}
	ok(f->batch_calls >= 3,
	   "300 adds trigger automatic flushes at the batch boundary");
	br_flush(b);
	eql(br_count(b), 301, "every record is present after batching");

	br_free(b);
}

/*
 * Batched and unbatched embedding must produce identical bits.
 *
 * This is the invariant two separate bugs violated: the batch path first
 * truncated long texts instead of chunking, then split on characters where
 * the per-text path split on tokens. Both produced plausible vectors that
 * simply were not the same ones.
 */
static void
test_batch_equals_single(void)
{
	void *bb = br_new(), *bs = br_new();
	struct fake *fb = NULL, *fs = NULL;
	sigil_embedder_t *eb = fake_new(64, 1, &fb);   /* batching */
	sigil_embedder_t *es = fake_new(64, 0, &fs);   /* no batch path */
	char hb[130], hs[130];
	int k, mismatch = 0;

	if (bb == NULL || bs == NULL || eb == NULL || es == NULL)
		return;
	br_embedder_set(bb, eb, 0xabcd);
	br_embedder_set(bs, es, 0xabcd);

	for (k = 0; k < 200; k++) {
		char t[128];

		snprintf(t, sizeof t, "paragraph %d with some words in it", k);
		br_add_at(bb, "/x.txt", (unsigned)k, t, strlen(t), NULL, 0, 0);
		br_add_at(bs, "/x.txt", (unsigned)k, t, strlen(t), NULL, 0, 0);
	}
	br_flush(bb);
	br_flush(bs);

	ok(fb->batch_calls > 0, "the batching bridge used embed_batch");
	eql((long)fs->batch_calls, 0,
	    "a backend without a batch path never has one called");
	ok(fs->single_calls == 200,
	   "the unbatched bridge embedded every record singly");

	for (k = 0; k < 200; k++) {
		br_lsh_hex(bb, k, hb, sizeof hb);
		br_lsh_hex(bs, k, hs, sizeof hs);
		if (strcmp(hb, hs) != 0)
			mismatch++;
	}
	eql((long)mismatch, 0,
	    "batched and single-call embedding agree on every record");

	br_free(bb);
	br_free(bs);
}

/*
 * An embedding failure must leave the record addressable, not drop it.
 *
 * A paragraph that will not embed is still worth addressing by content; it
 * just will not be found by similarity. Dropping it would silently shrink
 * the store.
 */
static void
test_embed_failure_keeps_record(void)
{
	void *b = br_new();
	struct fake *f = NULL;
	sigil_embedder_t *e = fake_new(64, 1, &f);

	if (b == NULL || e == NULL)
		return;
	br_embedder_set(b, e, 1);

	br_add_at(b, "/f.txt", 1, "text that will fail to embed", 28, NULL, 0, 0);
	f->fail_next = 1;
	br_flush(b);

	eql(br_count(b), 1, "a failed embed keeps the record");
	ok(br_path(b, 0) != NULL, "and its provenance");

	br_free(b);
}

/* --- embedder loading --------------------------------------------------- */

static void
test_embedder_load_paths(void)
{
	void *b = br_new();
	struct fake *f = NULL;
	sigil_embedder_t *e;

	if (b == NULL)
		return;

	ok(br_embedder_load(b, NULL, 0) != 0, "a NULL path is refused");
	ok(br_embedder_load(b, "/nonexistent/model.gguf", 0) != 0,
	   "a missing model fails rather than falling back silently");
	ok(br_have_embedder(b) == 0, "a failed load leaves no embedder");

	e = fake_new(64, 1, &f);
	if (e == NULL) {
		br_free(b);
		return;
	}
	ok(br_embedder_set(b, e, 7) == 0, "set installs");
	ok(br_embedder_set(b, e, 7) != 0,
	   "a second set is refused rather than leaking the first");
	ok(br_embedder_set(NULL, e, 7) != 0, "set on a NULL bridge fails");

	br_free(b);
}

/* --- remaining reachable branches --------------------------------------- */

static void
test_edge_paths(void)
{
	void *b = br_new();
	struct fake *f = NULL;
	sigil_embedder_t *e;
	unsigned out[8];
	char hex[130];

	if (b == NULL)
		return;

	/* NULL and empty inputs on every accessor. similar.c and persist.c
	 * both index into these with values derived from a scan or a file,
	 * so a wrong answer here becomes a wrong directory entry. */
	eql(br_count(NULL), 0, "count of a NULL bridge is 0");
	ok(br_path(NULL, 0) == NULL, "path of a NULL bridge is NULL");
	eql((long)br_para(NULL, 0), 0, "para of a NULL bridge is 0");
	eql((long)br_offset(NULL, 0), 0, "offset of a NULL bridge is 0");
	eql((long)br_length(NULL, 0), 0, "length of a NULL bridge is 0");
	ok(br_have_embedder(NULL) == 0, "NULL bridge has no embedder");
	eql(br_find_hash(NULL, "abc"), -1, "find on a NULL bridge fails");
	eql(br_similar(NULL, 0, 10, out, 8), 0, "similar on NULL returns 0");

	/* Negative indices: br_similar hands back unsigned values, but a
	 * caller doing arithmetic on them can arrive here with -1. */
	ok(br_path(b, -1) == NULL, "a negative index is refused");
	ok(br_hash(b, -1, hex, sizeof hex) != 0, "negative hash index fails");

	/* An add with an explicit LSH takes neither the embed nor the queue
	 * path -- this is what persistence replay does. */
	{
		unsigned long long lsh[2] = { 0x1122334455667788ULL,
		                              0x99aabbccddeeff00ULL };
		char got[130];

		eql(br_add_at(b, "/e.txt", 1, "text", 4, lsh, 0, 0), 0,
		    "add with an explicit lsh succeeds");
		ok(br_lsh_hex(b, 0, got, sizeof got) == 0, "its lsh reads back");
		ok(strstr(got, "1122334455667788") != NULL,
		   "the supplied bits are stored verbatim, not re-embedded");
	}

	/* A too-small output buffer must fail rather than truncate: a
	 * half-written hex string names a different record. */
	ok(br_hash(b, 0, hex, 4) != 0, "a short hash buffer is refused");
	ok(br_lsh_hex(b, 0, hex, 4) != 0, "a short lsh buffer is refused");

	/* Loading twice is a no-op rather than a leak. */
	e = fake_new(64, 1, &f);
	if (e != NULL) {
		br_embedder_set(b, e, 3);
		eql(br_embedder_load(b, "/whatever.gguf", 3), 0,
		    "load with an embedder already present returns success "
		    "without replacing it");
		ok(strcmp(br_embedder_name(b), "fake-deterministic") == 0,
		   "the original embedder is still installed");
	}
	ok(strcmp(br_embedder_name(NULL), "none") == 0,
	   "a NULL bridge names no embedder");

	br_free(b);
	br_free(NULL);          /* must not crash */
	ok(1, "free of a NULL bridge is safe");
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
	test_batch_flush();
	test_batch_equals_single();
	test_embed_failure_keeps_record();
	test_embedder_load_paths();
	test_edge_paths();

	if (failures == 0) {
		printf("PASS: %d checks\n", checks);
		return 0;
	}
	printf("FAIL: %d of %d checks failed\n", failures, checks);
	return 1;
}
