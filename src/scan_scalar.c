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

size_t sigil_kernel_similar_scalar(const sigil_view_t *v, const void *arg,
                                   uint32_t *out, size_t max_out)
{
	const sigil_simarg_t *a = arg;
	size_t n = 0;

	for (size_t i = 0; i < v->count && n < max_out; i++) {
		if (sigil_hamming(v->lsh + i * SIGIL_LSH_WORDS, a->query)
		    <= a->max_distance)
			out[n++] = (uint32_t)i;
	}
	return n;
}

size_t sigil_kernel_timerange_scalar(const sigil_view_t *v, const void *arg,
                                     uint32_t *out, size_t max_out)
{
	const sigil_timearg_t *a = arg;
	size_t n = 0;

	for (size_t i = 0; i < v->count && n < max_out; i++) {
		uint32_t t = v->timestamp[i];

		if (t >= a->start && t <= a->end)
			out[n++] = (uint32_t)i;
	}
	return n;
}

size_t sigil_kernel_category_scalar(const sigil_view_t *v, const void *arg,
                                    uint32_t *out, size_t max_out)
{
	uint16_t category = *(const uint16_t *)arg;
	size_t n = 0;

	for (size_t i = 0; i < v->count && n < max_out; i++) {
		if (v->category[i] == category)
			out[n++] = (uint32_t)i;
	}
	return n;
}
