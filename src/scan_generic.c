/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Portable fallback for architectures with no SIMD kernel.
 *
 * Compiles to nothing on x86-64 (see scan_x86.c) and aarch64 (scan_neon.c).
 * Everywhere else the _simd entry points forward to the scalar reference, so
 * the library builds and behaves correctly regardless of target — it just runs
 * at scalar speed, and sigil_have_simd() says so.
 */

#include "sigil.h"

#if !defined(__x86_64__) && !defined(_M_X64) && !defined(__aarch64__)

int sigil_have_simd(void)
{
	return 0;
}

/* Path selection is an x86 concern: this build has exactly one kernel, so
 * there is nothing to calibrate and nothing to force. Defined here anyway
 * because sigil.h declares them for every platform, and a header that
 * promises a symbol no object provides is a link error waiting for the
 * first person to build on this architecture. */
int sigil_simd_paths(int *avx2, int *sse42)
{
	if (avx2 != NULL)
		*avx2 = 0;
	if (sse42 != NULL)
		*sse42 = 0;
	return 0;
}

int sigil_simd_chosen(void)
{
	return 0;
}

int sigil_simd_force(int path)
{
	return path == 0 ? 0 : -1;
}

void sigil_simd_unforce(void)
{
}

size_t sigil_kernel_similar(const sigil_store_t *st, const uint64_t *query,
                               uint32_t max_distance,
                               uint32_t *out, size_t max_out)
{
	return sigil_scan_similar_scalar(st, query, max_distance, out, max_out);
}

size_t sigil_kernel_timerange(const sigil_store_t *st,
                                 uint32_t start, uint32_t end,
                                 uint32_t *out, size_t max_out)
{
	return sigil_scan_timerange_scalar(st, start, end, out, max_out);
}

size_t sigil_kernel_category(const sigil_store_t *st, uint16_t category,
                                uint32_t *out, size_t max_out)
{
	return sigil_scan_category_scalar(st, category, out, max_out);
}

#endif
