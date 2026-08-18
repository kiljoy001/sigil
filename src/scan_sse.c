/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * SSE4.2 scan kernels for x86-64 without AVX2.
 *
 * Covers roughly 2008-2013 silicon: Nehalem, Westmere, Sandy Bridge, Ivy
 * Bridge. Those machines have 128-bit XMM registers and SSE4.2 but no AVX2, so
 * without this file they fall all the way back to scalar. A Westmere Xeon
 * E5645 runs the similarity scan at 103ms against the i5-12600K's 10ms; the
 * gap is mostly the missing vector path, not just clock speed.
 *
 * Structurally this is the NEON kernel, not the AVX2 one: an XMM register is
 * 128 bits, exactly one LSH code, so there is no lane bookkeeping. What it
 * lacks is NEON's vcntq_u8, so population count goes through the same nibble
 * table and pshufb that the AVX2 path uses, just at half the width.
 *
 * SSE4.2 is baseline for every x86-64 CPU that also has AVX2, so this file
 * only ever runs when the AVX2 probe fails. That is checked in scan_x86.c;
 * these entry points assume nothing and are safe on any SSE4.2 machine.
 *
 * Note this needs SSE4.1 for pcmpeqq/pblendv-era instructions and SSSE3 for
 * pshufb; the target attribute below requests the full set. Plain SSE2 (any
 * x86-64) would need a different popcount and is not worth the complexity —
 * scalar popcnt is competitive there.
 */

#include "sigil.h"

#if defined(__x86_64__) || defined(_M_X64)
#define SIGIL_SSE 1
#include <immintrin.h>
#endif

#ifdef SIGIL_SSE

/*
 * One record per iteration: 128 bits is one XMM register. Loads are unaligned
 * because ranged views can start at any index.
 *
 * psadbw against zero sums the byte counts into two 64-bit fields, and the
 * record's distance is the sum of those two. Unlike AVX2 there are no 128-bit
 * lane boundaries to respect, because the register *is* the record.
 */
__attribute__((target("sse4.2,ssse3")))
size_t sigil_scan_similar_sse(const sigil_view_t *sv, const void *arg,
                              uint32_t *out, size_t max_out)
{
	const sigil_simarg_t *a = arg;
	const uint64_t *query = a->query;
	uint32_t max_distance = a->max_distance;
	size_t n = 0, i = 0;
	const __m128i lut = _mm_setr_epi8(
		0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
	const __m128i mask = _mm_set1_epi8(0x0f);
	const __m128i zero = _mm_setzero_si128();
	const __m128i q = _mm_loadu_si128((const __m128i *)query);

	for (; i < sv->count && n < max_out; i++) {
		__m128i v = _mm_loadu_si128(
			(const __m128i *)(sv->lsh + i * SIGIL_LSH_WORDS));
		__m128i x = _mm_xor_si128(v, q);
		__m128i lo = _mm_and_si128(x, mask);
		__m128i hi = _mm_and_si128(_mm_srli_epi16(x, 4), mask);
		__m128i cnt = _mm_add_epi8(_mm_shuffle_epi8(lut, lo),
		                           _mm_shuffle_epi8(lut, hi));
		__m128i sums = _mm_sad_epu8(cnt, zero);
		/* Two u64 halves; total caps at 128 so neither can overflow. */
		uint32_t d = (uint32_t)(_mm_cvtsi128_si32(sums) +
		                        _mm_extract_epi32(sums, 2));

		if (d <= max_distance)
			out[n++] = (uint32_t)i;
	}
	return n;
}

__attribute__((target("sse4.2,ssse3")))
size_t sigil_scan_timerange_sse(const sigil_view_t *sv, const void *arg,
                                uint32_t *out, size_t max_out)
{
	const sigil_timearg_t *a = arg;
	uint32_t start = a->start, end = a->end;
	size_t n = 0, i = 0;
	/* SSE has only signed 32-bit compares, same as AVX2, so bias by 2^31
	 * to turn unsigned order into signed order. */
	const __m128i bias = _mm_set1_epi32((int)0x80000000u);
	const __m128i lo   = _mm_set1_epi32((int)(start ^ 0x80000000u));
	const __m128i hi   = _mm_set1_epi32((int)(end   ^ 0x80000000u));

	for (; i + 4 <= sv->count && n + 4 <= max_out; i += 4) {
		__m128i v = _mm_loadu_si128((const __m128i *)(sv->timestamp + i));
		__m128i b = _mm_xor_si128(v, bias);
		__m128i too_low  = _mm_cmplt_epi32(b, lo);
		__m128i too_high = _mm_cmpgt_epi32(b, hi);
		__m128i bad      = _mm_or_si128(too_low, too_high);
		int m = ~_mm_movemask_ps(_mm_castsi128_ps(bad)) & 0xf;

		/* Unlike NEON, SSE has movemask, so lane selection stays in a
		 * register — no round-trip through memory. */
		while (m) {
			int lane = __builtin_ctz((unsigned)m);

			out[n++] = (uint32_t)(i + (size_t)lane);
			m &= m - 1;
		}
	}

	for (; i < sv->count && n < max_out; i++) {
		uint32_t t = sv->timestamp[i];

		if (t >= start && t <= end)
			out[n++] = (uint32_t)i;
	}
	return n;
}

__attribute__((target("sse4.2,ssse3")))
size_t sigil_scan_category_sse(const sigil_view_t *sv, const void *arg,
                               uint32_t *out, size_t max_out)
{
	uint16_t category = *(const uint16_t *)arg;
	size_t n = 0, i = 0;
	const __m128i q = _mm_set1_epi16((short)category);

	for (; i + 8 <= sv->count && n + 8 <= max_out; i += 8) {
		__m128i v  = _mm_loadu_si128((const __m128i *)(sv->category + i));
		__m128i eq = _mm_cmpeq_epi16(v, q);
		/* movemask_epi8 gives two bits per 16-bit lane; both set or both
		 * clear, so step by pairs. */
		unsigned m = (unsigned)_mm_movemask_epi8(eq);

		while (m) {
			int bit = __builtin_ctz(m);

			out[n++] = (uint32_t)(i + (size_t)(bit / 2));
			m &= ~(3u << (bit & ~1));
		}
	}

	for (; i < sv->count && n < max_out; i++) {
		if (sv->category[i] == category)
			out[n++] = (uint32_t)i;
	}
	return n;
}

#endif /* SIGIL_SSE */
