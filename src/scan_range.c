/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Ranged scans, for callers that want to parallelize.
 *
 * A range is expressed as a temporary store view over the same arrays: the SoA
 * layout means a slice is just base pointers advanced by lo, with count set to
 * the span. No copying, and the existing dispatch picks the same kernel it
 * would for a whole-store scan.
 *
 * Indices come back absolute so merged results need no adjustment.
 */

#include "sigil.h"

#include <string.h>

/* Build a view over [lo, hi). The view borrows the parent's arrays; it must
 * never be passed to sigil_store_free. */
static sigil_store_t view(const sigil_store_t *st, size_t lo, size_t hi)
{
	sigil_store_t v;

	memset(&v, 0, sizeof(v));
	v.lsh       = st->lsh + lo * SIGIL_LSH_WORDS;
	v.timestamp = st->timestamp + lo;
	v.category  = st->category + lo;
	v.trits     = st->trits + lo;
	v.hash      = st->hash + lo * SIGIL_HASH_LEN;
	v.count     = hi - lo;
	v.capacity  = hi - lo;
	return v;
}

/* Shift range-relative indices to absolute. */
static void rebase(uint32_t *out, size_t n, size_t lo)
{
	for (size_t i = 0; i < n; i++)
		out[i] += (uint32_t)lo;
}

static int clamp(const sigil_store_t *st, size_t *lo, size_t *hi)
{
	if (*hi > st->count)
		*hi = st->count;
	return *lo < *hi;
}

size_t sigil_scan_similar_range(const sigil_store_t *st, const uint64_t *query,
                                uint32_t max_distance, size_t lo, size_t hi,
                                uint32_t *out, size_t max_out)
{
	sigil_store_t v;
	size_t n;

	if (!clamp(st, &lo, &hi))
		return 0;
	v = view(st, lo, hi);
	n = sigil_scan_similar_simd(&v, query, max_distance, out, max_out);
	rebase(out, n, lo);
	return n;
}

size_t sigil_scan_timerange_range(const sigil_store_t *st,
                                  uint32_t start, uint32_t end,
                                  size_t lo, size_t hi,
                                  uint32_t *out, size_t max_out)
{
	sigil_store_t v;
	size_t n;

	if (!clamp(st, &lo, &hi))
		return 0;
	v = view(st, lo, hi);
	n = sigil_scan_timerange_simd(&v, start, end, out, max_out);
	rebase(out, n, lo);
	return n;
}

size_t sigil_scan_category_range(const sigil_store_t *st, uint16_t category,
                                 size_t lo, size_t hi,
                                 uint32_t *out, size_t max_out)
{
	sigil_store_t v;
	size_t n;

	if (!clamp(st, &lo, &hi))
		return 0;
	v = view(st, lo, hi);
	n = sigil_scan_category_simd(&v, category, out, max_out);
	rebase(out, n, lo);
	return n;
}
