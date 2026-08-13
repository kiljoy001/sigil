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
#include "sigil_utf8.h"
#include "sigil_split.h"

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

/* --- hex round-trip ----------------------------------------------------- */

static void
test_hex(void)
{
	sigil_t a, b;
	char hex[SIGIL_SIZE * 2 + 1];
	const char *text = "a paragraph to hex-encode and read back";

	sigil_generate_para(text, strlen(text), 5, 1234, 9, NULL, &a);
	sigil_to_hex(&a, hex);
	eqsz(strlen(hex), SIGIL_SIZE * 2, "hex is two characters per byte");

	ok(sigil_from_hex(hex, &b) == 0, "hex decodes");
	ok(memcmp(&a, &b, sizeof a) == 0,
	   "a record survives a hex round-trip byte for byte -- this is how a "
	   "sigil travels through a text protocol");

	/* Malformed input must be rejected, not partially decoded. A record
	 * half-filled from a truncated hex string would carry a plausible
	 * hash for content that was never hashed. */
	ok(sigil_from_hex("", &b) != 0, "empty hex is rejected");
	ok(sigil_from_hex("abcd", &b) != 0, "short hex is rejected");
	hex[10] = 'z';
	ok(sigil_from_hex(hex, &b) != 0, "a non-hex digit is rejected");
	hex[10] = '0';
	hex[SIGIL_SIZE * 2 - 1] = '\0';
	ok(sigil_from_hex(hex, &b) != 0, "one digit short is rejected");
}

/* --- timerange and category scans --------------------------------------- */

static void
test_filter_scans(void)
{
	sigil_store_t st;
	uint32_t out[256];
	size_t n, i;
	int k;

	ok(sigil_store_init(&st, 8) == 0, "store for filter scans");

	/* timestamps 0..99, categories cycling 0..4 */
	for (k = 0; k < 100; k++) {
		char text[48];
		sigil_t s;

		snprintf(text, sizeof text, "record %d", k);
		sigil_generate_para(text, strlen(text), (uint32_t)k,
		                    (uint32_t)k, (uint16_t)(k % 5), NULL, &s);
		if (sigil_store_push(&st, &s) < 0)
			break;
	}
	eqsz(st.count, 100, "100 records for the filter scans");

	/* Inclusive at both ends -- the header says start <= t <= end, and an
	 * off-by-one here silently drops a day from a range query. */
	n = sigil_scan_timerange_scalar(&st, 10, 19, out, 256);
	eqsz(n, 10, "timerange is inclusive at both ends");
	for (i = 0; i < n; i++) {
		sigil_t got;

		if (sigil_store_get(&st, out[i], &got) != 0 ||
		    got.timestamp < 10 || got.timestamp > 19) {
			ok(0, "timerange returned an out-of-range record");
			break;
		}
	}
	if (i == n)
		ok(1, "every timerange hit is inside the range");

	eqsz(sigil_scan_timerange_scalar(&st, 500, 600, out, 256), 0,
	     "an empty range returns nothing");
	eqsz(sigil_scan_timerange_scalar(&st, 0, 99, out, 256), 100,
	     "a full range returns everything");
	ok(sigil_scan_timerange_scalar(&st, 0, 99, out, 7) <= 7,
	   "max_out is honoured by timerange");

	n = sigil_scan_category_scalar(&st, 3, out, 256);
	eqsz(n, 20, "category scan finds every member");
	for (i = 0; i < n; i++) {
		sigil_t got;

		if (sigil_store_get(&st, out[i], &got) != 0 || got.category != 3) {
			ok(0, "category scan returned the wrong category");
			break;
		}
	}
	if (i == n)
		ok(1, "every category hit has that category");
	eqsz(sigil_scan_category_scalar(&st, 99, out, 256), 0,
	     "an absent category returns nothing");

	/* The SIMD twins must agree with the scalar ones; differential.c
	 * covers the similarity kernel but not these two.
	 *
	 * Sweep every path the CPU offers rather than testing whichever one
	 * the calibration happened to pick. Without the force, exactly one
	 * kernel runs per process and the others are unreachable -- which is
	 * how scan_timerange_avx2 and scan_category_avx2 sat at 0% coverage
	 * while appearing to be tested. An agreement test that cannot reach
	 * the implementation it is comparing against proves nothing. */
	{
		static const struct { int p; const char *name; } paths[] = {
			{ 0, "scalar" }, { 1, "sse4.2" }, { 2, "avx2" },
		};
		uint32_t o2[256];
		size_t n2, k;

		for (k = 0; k < sizeof paths / sizeof paths[0]; k++) {
			char msg[96];

			if (sigil_simd_force(paths[k].p) != 0)
				continue;      /* not offered here */

			n2 = sigil_scan_timerange_simd(&st, 10, 19, o2, 256);
			n = sigil_scan_timerange_scalar(&st, 10, 19, out, 256);
			snprintf(msg, sizeof msg,
			         "%s timerange matches scalar exactly",
			         paths[k].name);
			ok(n == n2 && memcmp(out, o2, n * sizeof *out) == 0,
			   msg);

			n = sigil_scan_category_scalar(&st, 3, out, 256);
			n2 = sigil_scan_category_simd(&st, 3, o2, 256);
			snprintf(msg, sizeof msg,
			         "%s category matches scalar exactly",
			         paths[k].name);
			ok(n == n2 && memcmp(out, o2, n * sizeof *out) == 0,
			   msg);

			n2 = sigil_scan_similar_simd(&st, st.lsh, 8, o2, 256);
			n = sigil_scan_similar_scalar(&st, st.lsh, 8, out, 256);
			snprintf(msg, sizeof msg,
			         "%s similar matches scalar exactly",
			         paths[k].name);
			ok(n == n2 && memcmp(out, o2, n * sizeof *out) == 0,
			   msg);
		}
		sigil_simd_unforce();
	}

	sigil_store_free(&st);
}

/* --- the semantic pipeline ---------------------------------------------- */

static void
test_generate_semantic(void)
{
	sigil_embedder_t *e = sigil_embedder_hash_nonsemantic(64);
	sigil_simhash_t sh;
	sigil_t a, b;
	const char *text = "the full pipeline: identity plus semantic bits";

	if (e == NULL)
		return;
	ok(sigil_simhash_init(&sh, 64, 0x1234) == 0, "simhash for the pipeline");

	ok(sigil_generate_semantic(e, &sh, text, strlen(text), 42, 3, NULL,
	                           &a) == 0, "generate_semantic succeeds");
	eqsz(a.timestamp, 42, "timestamp is carried through");
	eqsz(a.category, 3, "category is carried through");

	/* Deterministic: same text, same seed, same bits. */
	ok(sigil_generate_semantic(e, &sh, text, strlen(text), 42, 3, NULL,
	                           &b) == 0, "second call succeeds");
	ok(memcmp(&a, &b, sizeof a) == 0, "the pipeline is deterministic");

	sigil_simhash_free(&sh);
	e->destroy(e);
}

/* --- hardware selection ------------------------------------------------- */

static void
test_simd_selection(void)
{
	int avx2 = -1, sse42 = -1, avail, chosen, again;

	avail = sigil_simd_paths(&avx2, &sse42);
	ok(avx2 == 0 || avx2 == 1, "the AVX2 probe answers 0 or 1");
	ok(sse42 == 0 || sse42 == 1, "the SSE4.2 probe answers 0 or 1");

	/* Both probes must actually run. sigil_have_simd() used to be
	 * have_avx2() || have_sse42(), so on any machine with AVX2 the SSE
	 * probe was never evaluated -- it would first execute on hardware
	 * nobody was testing. */
	ok(!(avx2 == -1 || sse42 == -1), "both probes were evaluated");

	/* SSE4.2 is baseline for every x86-64 part that has AVX2, so a CPU
	 * claiming AVX2 without it means the probe is wrong. */
	if (avx2)
		ok(sse42, "a CPU with AVX2 also reports SSE4.2");
	else
		ok(1, "no AVX2 on this machine; nothing to cross-check");

	eqsz((size_t)avail, (size_t)(avx2 ? 2 : (sse42 ? 1 : 0)),
	     "paths() reports the widest available");
	eqsz((size_t)sigil_have_simd(), (size_t)((avx2 || sse42) ? 1 : 0),
	     "have_simd agrees with the probes");

	/* The calibrated choice is measured, not inferred, so it may be
	 * narrower than what CPUID offers -- that is the point. It must
	 * never be wider than what the hardware has. */
	chosen = sigil_simd_chosen();
	ok(chosen >= 0 && chosen <= avail,
	   "the calibrated kernel is one the hardware actually supports");

	/* And it must be stable: a scan that changed kernels between calls
	 * would be comparing records built two different ways. */
	again = sigil_simd_chosen();
	eqsz((size_t)again, (size_t)chosen, "the choice is cached, not re-raced");
}

/* --- UTF-8 repair ------------------------------------------------------- */

/*
 * The guard that stops invalid UTF-8 reaching PCRE2 inside the tokenizer,
 * where it is undefined behaviour and read wild memory (docs/FINDINGS.md).
 *
 * There is a far larger differential suite in Python
 * (tools/tests/test_utf8_differential.py) that compares this against
 * tools/clean.py over generated bytes. These tests exist as well, not
 * instead: CRAP scoring showed utf8_repair.c at 0% line coverage and CRAP
 * 600 because the C build never executed it, and "covered by a test in
 * another language" is not something the coverage gate can see -- nor
 * should it, since the C build is what ships.
 */
static void
test_utf8_repair(void)
{
	static const struct {
		const char *in;
		size_t len;
		const char *want;
		const char *what;
	} cases[] = {
		{ "plain ascii", 11, "plain ascii", "ascii is untouched" },
		{ "caf\xc3\xa9", 5, "caf\xc3\xa9", "valid utf-8 is untouched" },
		{ "don\x92t", 5, "don\xe2\x80\x99t",
		  "cp1252 0x92 becomes U+2019, not a replacement char" },
		{ "caf\xe9", 4, "caf\xc3\xa9",
		  "bare 0xe9 is read as latin-1 e-acute" },
		/* 0x80 is CP1252's Euro sign, not Latin-1 U+0080: the C1
		 * range is exactly where the two encodings differ, and
		 * preferring CP1252 there is what turns 0x92 into an
		 * apostrophe instead of a control character. */
		{ "a\x80""b", 3, "a\xe2\x82\xac""b",
		  "lone continuation byte becomes the cp1252 character" },
		{ "a\xc0\x80""b", 4, "a\xc3\x80\xe2\x82\xac""b",
		  "overlong nul is repaired bytewise, never decoded to U+0000" },
		{ "", 0, "", "empty input" },
	};
	size_t i;

	for (i = 0; i < sizeof cases / sizeof cases[0]; i++) {
		size_t n = 0;
		char *got = sigil_utf8_repair(cases[i].in, cases[i].len, &n);

		if (got == NULL) {
			ok(0, cases[i].what);
			continue;
		}
		ok(strcmp(got, cases[i].want) == 0, cases[i].what);
		eqsz(n, strlen(cases[i].want), "reported length matches");
		/* Whatever came out must itself be valid, or the guard has
		 * merely moved the undefined behaviour downstream. */
		ok(sigil_utf8_valid(got, n) == 1, "output is valid utf-8");
		free(got);
	}
}

static void
test_utf8_seq(void)
{
	static const struct {
		const char *bytes;
		size_t len;
		size_t want;
		const char *what;
	} cases[] = {
		{ "a", 1, 1, "ascii is one byte" },
		{ "\xc3\xa9", 2, 2, "two-byte sequence" },
		{ "\xe4\xb8\xad", 3, 3, "three-byte sequence" },
		{ "\xf0\x9f\x98\x80", 4, 4, "four-byte sequence" },
		{ "\x80", 1, 0, "lone continuation is invalid" },
		{ "\xc0\x80", 2, 0, "overlong two-byte is invalid" },
		{ "\xc1\xbf", 2, 0, "overlong two-byte upper is invalid" },
		{ "\xe0\x80\x80", 3, 0, "overlong three-byte is invalid" },
		{ "\xed\xa0\x80", 3, 0, "surrogate half is invalid" },
		{ "\xf4\x90\x80\x80", 4, 0, "above U+10FFFF is invalid" },
		{ "\xf5\x80\x80\x80", 4, 0, "0xf5 lead is invalid" },
		{ "\xc2", 1, 0, "truncated two-byte is invalid" },
		{ "\xe2\x82", 2, 0, "truncated three-byte is invalid" },
		{ "", 0, 0, "empty is zero" },
	};
	size_t i;

	for (i = 0; i < sizeof cases / sizeof cases[0]; i++)
		eqsz(sigil_utf8_seq((const unsigned char *)cases[i].bytes,
		                    cases[i].len),
		     cases[i].want, cases[i].what);

	ok(sigil_utf8_valid("hello", 5) == 1, "valid buffer reports valid");
	ok(sigil_utf8_valid("a\x92""b", 3) == 0,
	   "buffer with a stray cp1252 byte reports invalid");
}

/* --- paragraph splitting ------------------------------------------------ */

/*
 * The splitter defines the unit of identity: a record's BLAKE3 is computed
 * over exactly the span this produces. There is a larger suite in Python
 * (tools/tests/test_split.py) that drives the same C through ctypes, but
 * CRAP found this at 0% line coverage and CRAP 600 -- the C build never
 * ran it, and the C build is what ships. The same signature utf8_repair.c
 * had for the same reason.
 */
struct splitcount {
	int n;
	size_t total;
	unsigned last_para;
};

static void
countchunk(const sigil_chunk_t *c, void *arg)
{
	struct splitcount *s = arg;

	s->n++;
	s->total += c->len;
	s->last_para = c->para;
}

static void
test_split(void)
{
	struct splitcount sc;
	char buf[9000];
	const char *two = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	                  "\n\n"
	                  "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

	eqsz(sigil_split_count("", 0), 0, "empty input yields nothing");
	eqsz(sigil_split_count(NULL, 0), 0, "a null buffer is safe");

	/* Minpara and Maxpara are inclusive at both ends. */
	memset(buf, 'a', sizeof buf);
	eqsz(sigil_split_count(buf, SIGIL_MINPARA), 1, "exactly Minpara kept");
	eqsz(sigil_split_count(buf, SIGIL_MINPARA - 1), 0,
	     "one below Minpara dropped");
	eqsz(sigil_split_count(buf, SIGIL_MAXPARA), 1,
	     "exactly Maxpara is one chunk");

	/* A remainder below Minpara is dropped, not counted: this is the
	 * half of the corpus disagreement that made the manifest overcount. */
	eqsz(sigil_split_count(buf, SIGIL_MAXPARA + 1), 1,
	     "a one-byte remainder is dropped, not a second chunk");
	eqsz(sigil_split_count(buf, SIGIL_MAXPARA + SIGIL_MINPARA), 2,
	     "a remainder at Minpara is kept");

	/* Both paragraph terminators. The Python reimplementation knew only
	 * "\n\n" and merged these into one. */
	eqsz(sigil_split_count(two, strlen(two)), 2, "\\n\\n ends a paragraph");
	{
		char crlf[128];

		snprintf(crlf, sizeof crlf, "%.48s\n\r%.48s", two, two + 50);
		eqsz(sigil_split_count(crlf, strlen(crlf)), 2,
		     "\\n\\r also ends a paragraph");
	}

	/* The callback sees offsets into the caller's buffer, and paragraph
	 * numbers count from one. */
	sc.n = 0; sc.total = 0; sc.last_para = 0;
	sigil_split(two, strlen(two), countchunk, &sc);
	eqsz((size_t)sc.n, 2, "callback fires once per chunk");
	eqsz((size_t)sc.last_para, 2, "paragraph numbers are 1-based");
	ok(sc.total < strlen(two), "the separator is not part of any chunk");

	/* Leading whitespace is skipped before the span is measured. */
	{
		char lead[128];

		snprintf(lead, sizeof lead, "   \n\n%.60s", two);
		sc.n = 0; sc.total = 0;
		sigil_split(lead, strlen(lead), countchunk, &sc);
		eqsz((size_t)sc.n, 1, "leading whitespace yields no chunk");
	}
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
	test_hex();
	test_filter_scans();
	test_generate_semantic();
	test_simd_selection();
	test_utf8_repair();
	test_utf8_seq();
	test_split();

	if (failures == 0) {
		printf("PASS: %d checks\n", checks);
		return 0;
	}
	printf("FAIL: %d of %d checks failed\n", failures, checks);
	return 1;
}
