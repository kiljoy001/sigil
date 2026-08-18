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

/*
 * A range now spans segments, so it is scanned one segment at a time and the
 * per-segment results concatenate. Each piece is a sigil_view_t -- the flat
 * field pointers the kernels have always taken -- so the kernels themselves
 * did not change when the store became segmented. That mattered: a wrong
 * SIMD kernel does not crash, it returns plausible distances that stay wrong
 * forever, so the fewer edits to those four files the better.
 */

/* The slice of segment g that [lo, hi) covers, as a borrowed view. */
static sigil_view_t seg_view(const sigil_store_t *st, size_t g,
                             size_t from, size_t to)
{
	sigil_view_t v;

	v.lsh       = st->lsh[g] + from * SIGIL_LSH_WORDS;
	v.para      = st->para[g] + from;
	v.cluster   = st->cluster[g] + from;
	v.timestamp = st->timestamp[g] + from;
	v.category  = st->category[g] + from;
	v.trits     = st->trits[g] + from;
	v.hash      = st->hash[g] + from * SIGIL_HASH_LEN;
	v.count     = to - from;
	return v;
}

/*
 * Walk [lo, hi) segment by segment, calling `fn` on each piece and rebasing
 * its indices to absolute. Every ranged and whole-store scan goes through
 * here, so segment traversal is written once rather than in each kernel.
 */
size_t sigil_scan_walk(const sigil_store_t *st, size_t lo, size_t hi,
                       sigil_scan_fn fn, const void *arg,
                       uint32_t *out, size_t max_out)
{
	size_t n = 0, i;

	if (hi > st->count)
		hi = st->count;

	for (i = lo; i < hi && n < max_out; ) {
		size_t g = i >> SIGIL_SEG_SHIFT;
		size_t from = i & SIGIL_SEG_MASK;
		size_t span = SIGIL_SEG_RECS - from;
		size_t k, j;
		sigil_view_t v;

		if (span > hi - i)
			span = hi - i;

		v = seg_view(st, g, from, from + span);
		k = fn(&v, arg, out + n, max_out - n);

		for (j = 0; j < k; j++)
			out[n + j] += (uint32_t)i;
		n += k;
		i += span;
	}
	return n;
}

size_t sigil_scan_similar_range(const sigil_store_t *st, const uint64_t *query,
                                uint32_t max_distance, size_t lo, size_t hi,
                                uint32_t *out, size_t max_out)
{
	sigil_simarg_t a;

	a.query = query;
	a.max_distance = max_distance;
	return sigil_scan_walk(st, lo, hi, sigil_kernel_similar, &a,
	                       out, max_out);
}

size_t sigil_scan_timerange_range(const sigil_store_t *st,
                                  uint32_t start, uint32_t end,
                                  size_t lo, size_t hi,
                                  uint32_t *out, size_t max_out)
{
	sigil_timearg_t a;

	a.start = start;
	a.end = end;
	return sigil_scan_walk(st, lo, hi, sigil_kernel_timerange, &a,
	                       out, max_out);
}

size_t sigil_scan_category_range(const sigil_store_t *st, uint16_t category,
                                 size_t lo, size_t hi,
                                 uint32_t *out, size_t max_out)
{
	uint16_t a = category;

	return sigil_scan_walk(st, lo, hi, sigil_kernel_category, &a,
	                       out, max_out);
}

/* --- whole-store entry points ------------------------------------------- */

size_t sigil_scan_similar_simd(const sigil_store_t *st, const uint64_t *query,
                               uint32_t max_distance,
                               uint32_t *out, size_t max_out)
{
	return sigil_scan_similar_range(st, query, max_distance, 0, st->count,
	                                out, max_out);
}

size_t sigil_scan_timerange_simd(const sigil_store_t *st,
                                 uint32_t start, uint32_t end,
                                 uint32_t *out, size_t max_out)
{
	return sigil_scan_timerange_range(st, start, end, 0, st->count,
	                                  out, max_out);
}

size_t sigil_scan_category_simd(const sigil_store_t *st, uint16_t category,
                                uint32_t *out, size_t max_out)
{
	return sigil_scan_category_range(st, category, 0, st->count,
	                                 out, max_out);
}

/* --- scalar reference, same traversal ----------------------------------- */

size_t sigil_scan_similar_scalar(const sigil_store_t *st, const uint64_t *query,
                                 uint32_t max_distance,
                                 uint32_t *out, size_t max_out)
{
	sigil_simarg_t a;

	a.query = query;
	a.max_distance = max_distance;
	return sigil_scan_walk(st, 0, st->count, sigil_kernel_similar_scalar,
	                       &a, out, max_out);
}

size_t sigil_scan_timerange_scalar(const sigil_store_t *st,
                                   uint32_t start, uint32_t end,
                                   uint32_t *out, size_t max_out)
{
	sigil_timearg_t a;

	a.start = start;
	a.end = end;
	return sigil_scan_walk(st, 0, st->count, sigil_kernel_timerange_scalar,
	                       &a, out, max_out);
}

size_t sigil_scan_category_scalar(const sigil_store_t *st, uint16_t category,
                                  uint32_t *out, size_t max_out)
{
	uint16_t a = category;

	return sigil_scan_walk(st, 0, st->count, sigil_kernel_category_scalar,
	                       &a, out, max_out);
}
