/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Scalar reference kernels.
 *
 * These are the definition of correct. The AVX2 kernels are an optimization
 * that must agree with these bit-for-bit; where they disagree, these win.
 * Keep them obvious rather than fast.
 */

#include "sigil.h"

size_t sigil_scan_similar_scalar(const sigil_store_t *st, const uint64_t *query,
                                 uint32_t max_distance,
                                 uint32_t *out, size_t max_out)
{
	size_t n = 0;

	for (size_t i = 0; i < st->count && n < max_out; i++) {
		if (sigil_hamming(st->lsh + i * SIGIL_LSH_WORDS, query)
		    <= max_distance)
			out[n++] = (uint32_t)i;
	}
	return n;
}

size_t sigil_scan_timerange_scalar(const sigil_store_t *st,
                                   uint32_t start, uint32_t end,
                                   uint32_t *out, size_t max_out)
{
	size_t n = 0;

	for (size_t i = 0; i < st->count && n < max_out; i++) {
		uint32_t t = st->timestamp[i];

		if (t >= start && t <= end)
			out[n++] = (uint32_t)i;
	}
	return n;
}

size_t sigil_scan_category_scalar(const sigil_store_t *st, uint16_t category,
                                  uint32_t *out, size_t max_out)
{
	size_t n = 0;

	for (size_t i = 0; i < st->count && n < max_out; i++) {
		if (st->category[i] == category)
			out[n++] = (uint32_t)i;
	}
	return n;
}
