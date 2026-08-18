/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * The segmented store: growth without copying.
 *
 * Implements features/store_growth.feature. Each test names the scenario
 * it covers, so the specification and the check stay connected without a
 * second language in between.
 *
 * The flat store doubled by allocating a second set of all seven field
 * arrays, copying into it, and freeing the first -- both live at once, so
 * the peak was old + new, 1.5x the final size. Measured on an 8.4M record
 * fill: virtual size went 297.5 MB -> 601.5 MB across the last doubling.
 * At 68.0M paragraphs the transient is tens of gigabytes, arriving as
 * seven large contiguous requests any one of which can fail on
 * fragmentation while ample memory is free -- and one failure aborted the
 * whole grow.
 *
 * Segments remove the copy: a segment is allocated once and never moves,
 * so growth appends and nothing already written is touched. That address
 * stability is also the precondition for a segment later being a mapping
 * of the store file rather than heap.
 *
 * Counting allocation rather than reading RSS is deliberate. RSS is the
 * wrong instrument: freed pages return to the allocator's free list
 * without being unmapped, and fresh pages are not resident until touched,
 * so it understates the transient badly -- 285 MB by RSS against 601 MB
 * of virtual size on the same fill. What matters is bytes requested and
 * not yet released, which is what the shim below records.
 *
 *	make segments
 */

#include "sigil.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- allocation accounting ---------------------------------------------
 *
 * posix_memalign, not malloc: the field arrays are 32-byte aligned so the
 * AVX2 loads stay aligned, and wrapping the wrong entry point would
 * measure nothing while looking like it worked.
 */

int __real_posix_memalign(void **p, size_t align, size_t n);
void __real_free(void *p);

static size_t live_bytes, peak_bytes;
static long fail_after = -1, nalloc;

/* Sizes are recalled by pointer on free. A small open-addressed table
 * keeps this independent of the allocator's own bookkeeping. */
#define NTRACK 4096
static struct { void *p; size_t n; } track[NTRACK];

static void
remember(void *p, size_t n)
{
	size_t h = ((uintptr_t)p >> 5) % NTRACK, i;

	for (i = 0; i < NTRACK; i++) {
		size_t s = (h + i) % NTRACK;

		if (track[s].p == NULL) {
			track[s].p = p;
			track[s].n = n;
			return;
		}
	}
}

static size_t
forget(void *p)
{
	size_t h = ((uintptr_t)p >> 5) % NTRACK, i;

	for (i = 0; i < NTRACK; i++) {
		size_t s = (h + i) % NTRACK;

		if (track[s].p == p) {
			size_t n = track[s].n;

			track[s].p = NULL;
			return n;
		}
	}
	return 0;
}

int
__wrap_posix_memalign(void **p, size_t align, size_t n)
{
	int rc;

	if (fail_after >= 0 && nalloc++ >= fail_after) {
		*p = NULL;
		return 12; /* ENOMEM */
	}
	rc = __real_posix_memalign(p, align, n);
	if (rc == 0) {
		live_bytes += n;
		if (live_bytes > peak_bytes)
			peak_bytes = live_bytes;
		remember(*p, n);
	}
	return rc;
}

void
__wrap_free(void *p)
{
	if (p != NULL)
		live_bytes -= forget(p);
	__real_free(p);
}

static void
reset_accounting(void)
{
	live_bytes = peak_bytes = 0;
	fail_after = -1;
	nalloc = 0;
	memset(track, 0, sizeof track);
}

/* --- harness ------------------------------------------------------------ */

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

/*
 * A record whose every field encodes i. A store that reads a neighbouring
 * segment returns a *valid* record, just the wrong one, so the contents
 * have to say which record they are -- filling only the hash would miss a
 * field-array mix-up entirely.
 */
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

static int
same(const sigil_t *a, const sigil_t *b)
{
	return a->lsh[0] == b->lsh[0] && a->lsh[1] == b->lsh[1] &&
	       memcmp(a->hash, b->hash, SIGIL_HASH_LEN) == 0 &&
	       a->para == b->para && a->timestamp == b->timestamp &&
	       a->category == b->category && a->trits == b->trits &&
	       a->cluster == b->cluster;
}

/* Enough records to cross several segment boundaries. Derived from the
 * segment size rather than written as a constant: a fixed 500000 silently
 * stopped crossing any boundary at all when SIGIL_SEG_RECS became 1 << 20,
 * and a boundary test that never reaches a boundary passes for the wrong
 * reason. The scenario "the fill crossed at least one segment boundary"
 * exists to catch exactly that. */
#define N (SIGIL_SEG_RECS * 3 + 1234)

/* --- Scenario: Growth allocates one segment, not a second copy ---------- */

static void
test_peak_within_one_segment(void)
{
	sigil_store_t st;
	sigil_t s;
	size_t i, slack;

	reset_accounting();
	ok(sigil_store_init(&st, 0) == 0, "store init");
	for (i = 0; i < N; i++) {
		fill(&s, i);
		if (sigil_store_push(&st, &s) < 0) {
			ok(0, "push during peak measurement");
			sigil_store_free(&st);
			return;
		}
	}

	/* Seven field arrays are segmented independently, so one growth step
	 * may add a segment to each before any is released. Anything beyond
	 * that is a copy. */
	slack = sigil_store_segment_bytes() * 7;
	if (peak_bytes > live_bytes + slack)
		printf("  peak %.1f MB, live %.1f MB, one segment each %.1f MB\n",
		       (double)peak_bytes / 1048576.0, (double)live_bytes / 1048576.0,
		       (double)slack / 1048576.0);
	ok(peak_bytes <= live_bytes + slack,
	   "peak allocation stays within one segment of live -- no copy");

	sigil_store_free(&st);
	ok(live_bytes == 0, "the store releases everything it allocated");
}

/* --- Scenario: Records already written are never moved ------------------ */

static void
test_addresses_are_stable(void)
{
	sigil_store_t st;
	sigil_t s;
	const void *a0 = NULL, *a1000 = NULL;
	size_t i;

	reset_accounting();
	ok(sigil_store_init(&st, 0) == 0, "store init for address stability");
	for (i = 0; i < N; i++) {
		fill(&s, i);
		if (sigil_store_push(&st, &s) < 0)
			break;
		if (i == 0)
			a0 = sigil_store_lsh_ptr(&st, 0);
		if (i == 1000)
			a1000 = sigil_store_lsh_ptr(&st, 1000);
	}
	ok(a0 != NULL && sigil_store_lsh_ptr(&st, 0) == a0,
	   "record 0 never moves");
	ok(a1000 != NULL && sigil_store_lsh_ptr(&st, 1000) == a1000,
	   "record 1000 never moves");
	sigil_store_free(&st);
}

/* --- Scenario: Every record survives crossing many segment boundaries --- */
/* --- Scenario: A record at each segment boundary is correct ------------- */

static void
test_every_record_reads_back(void)
{
	sigil_store_t st;
	sigil_t s, got, want;
	size_t i, per_seg, bad = 0, boundaries = 0;

	reset_accounting();
	ok(sigil_store_init(&st, 0) == 0, "store init for readback");
	for (i = 0; i < N; i++) {
		fill(&s, i);
		if (sigil_store_push(&st, &s) < 0) {
			ok(0, "push during readback fill");
			sigil_store_free(&st);
			return;
		}
	}

	for (i = 0; i < N; i++) {
		fill(&want, i);
		if (sigil_store_get(&st, i, &got) != 0 || !same(&got, &want)) {
			if (bad == 0)
				printf("  first wrong record at %zu\n", i);
			bad++;
		}
	}
	ok(bad == 0, "every record reads back exactly what was written");

	/* The boundaries specifically: seg[i >> SHIFT][i & MASK] is easy to
	 * get wrong in a way that returns a plausible neighbouring record. */
	per_seg = sigil_store_segment_bytes() / (SIGIL_LSH_WORDS * sizeof(uint64_t));
	bad = 0;
	for (i = per_seg; i < N; i += per_seg) {
		size_t k;

		boundaries++;
		for (k = i - 1; k <= i; k++) {
			fill(&want, k);
			if (sigil_store_get(&st, k, &got) != 0 ||
			    !same(&got, &want))
				bad++;
		}
	}
	ok(boundaries > 0, "the fill crossed at least one segment boundary");
	ok(bad == 0, "records on both sides of every boundary are correct");

	ok(sigil_store_get(&st, N, &got) != 0, "get past the end still fails");
	sigil_store_free(&st);
}

/* --- Scenario: The store reports its capacity honestly ------------------ */

static void
test_counts(void)
{
	sigil_store_t st;
	sigil_t s;
	size_t i;

	reset_accounting();
	ok(sigil_store_init(&st, 0) == 0, "store init for counts");
	for (i = 0; i < N; i++) {
		fill(&s, i);
		if (sigil_store_push(&st, &s) < 0)
			break;
	}
	ok(st.count == N, "count is exactly the number pushed");
	ok(st.capacity >= st.count, "capacity is at least the count");
	sigil_store_free(&st);
}

/* --- Scenario: A failed segment allocation loses only that segment ------ */

static void
test_failed_segment_is_contained(void)
{
	sigil_store_t st;
	sigil_t s, got, want;
	size_t i, pushed = 0, bad = 0;
	int failed = 0;

	reset_accounting();
	ok(sigil_store_init(&st, 0) == 0, "store init for allocation failure");
	for (i = 0; i < 100000; i++) {
		fill(&s, i);
		if (sigil_store_push(&st, &s) < 0)
			break;
		pushed++;
	}

	/* Fail every allocation from here. The next push that needs a new
	 * segment must fail; pushes inside the current one must not. */
	fail_after = 0;
	nalloc = 0;
	for (i = pushed; i < pushed + 200000; i++) {
		fill(&s, i);
		if (sigil_store_push(&st, &s) < 0) {
			failed = 1;
			break;
		}
		pushed++;
	}
	fail_after = -1;

	ok(failed, "a push that needs an unavailable segment fails");
	ok(st.count == pushed, "the failed push did not change the count");

	for (i = 0; i < pushed; i++) {
		fill(&want, i);
		if (sigil_store_get(&st, i, &got) != 0 || !same(&got, &want))
			bad++;
	}
	ok(bad == 0, "every record pushed before the failure survives intact");

	sigil_store_free(&st);
}

int
main(void)
{
	test_peak_within_one_segment();
	test_addresses_are_stable();
	test_every_record_reads_back();
	test_counts();
	test_failed_segment_is_contained();

	if (failures) {
		printf("FAILED: %d of %d checks\n", failures, checks);
		return 1;
	}
	printf("PASS: %d checks\n", checks);
	return 0;
}
