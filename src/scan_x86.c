/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * AVX2 scan kernels.
 *
 * These must agree with the scalar kernels bit-for-bit on every input;
 * test/differential.c is the enforcement. A wrong kernel here does not crash,
 * it returns subtly wrong distances that look plausible forever.
 *
 * AVX2 has no vpopcntd (that is AVX-512 VPOPCNTDQ), so population count goes
 * through the standard nibble-table shuffle: split each byte into two nibbles,
 * look both up in a 16-entry table replicated across both 128-bit halves via
 * vpshufb, then sum the bytes with vpsadbw.
 */

#include "sigil.h"

#if defined(__x86_64__) || defined(_M_X64)
#define SIGIL_X86 1
#include <immintrin.h>
#include <cpuid.h>
#endif

#ifdef SIGIL_X86

/*
 * AVX2 is a runtime property, not a compile-time one: this file is built for
 * generic x86-64 and the AVX2 bodies carry target attributes, so the binary
 * runs on pre-2013 hardware that has none. Calling an AVX2 body there faults
 * with SIGILL, so every entry point below has to check first.
 *
 * Probed once. __builtin_cpu_supports would also work, but CPUID here keeps
 * the check identical to what sigil_have_simd() reports to callers.
 */
static int avx2_probe(void)
{
	unsigned eax, ebx, ecx, edx;

	if (__get_cpuid_max(0, NULL) < 7)
		return 0;
	__cpuid_count(7, 0, eax, ebx, ecx, edx);
	return (ebx & bit_AVX2) != 0;
}

static int sse42_probe(void)
{
	unsigned eax, ebx, ecx, edx;

	if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx))
		return 0;
	/* SSSE3 is needed for pshufb, SSE4.1 for pextrd, SSE4.2 nominally. */
	return (ecx & bit_SSSE3) && (ecx & bit_SSE4_1) && (ecx & bit_SSE4_2);
}

static int have_avx2(void)
{
	static int cached = -1;

	if (cached < 0)
		cached = avx2_probe();
	return cached;
}

static int have_sse42(void)
{
	static int cached = -1;

	if (cached < 0)
		cached = sse42_probe();
	return cached;
}

int sigil_have_simd(void)
{
	return have_avx2() || have_sse42();
}

/* Implemented in scan_sse.c; declared here rather than in the public header
 * because callers pick a path through the _simd entry points, not directly. */
size_t sigil_scan_similar_sse(const sigil_store_t *st, const uint64_t *query,
                              uint32_t max_distance,
                              uint32_t *out, size_t max_out);
size_t sigil_scan_timerange_sse(const sigil_store_t *st,
                                uint32_t start, uint32_t end,
                                uint32_t *out, size_t max_out);
size_t sigil_scan_category_sse(const sigil_store_t *st, uint16_t category,
                               uint32_t *out, size_t max_out);


/*
 * 128-bit LSH similarity.
 *
 * Each record is two u64 words, so one YMM register holds exactly two records
 * and the per-record distance is a sum over its own 128-bit half. This is a
 * better fit than the old 32-bit layout, which needed maddubs/madd folding to
 * keep per-record totals from bleeding across lanes.
 *
 * vpsadbw does the work: it sums absolute byte differences into four u64
 * fields, one per 64-bit group. Against a zeroed operand that is just a byte
 * sum, so after the nibble-LUT popcount each u64 lane holds the bit count for
 * its own word. Adding the two lanes of a 128-bit half gives that record's
 * distance, and the halves are independent — no cross-lane shuffles.
 */
__attribute__((target("avx2")))
static size_t scan_similar_avx2(const sigil_store_t *st, const uint64_t *query,
                               uint32_t max_distance,
                               uint32_t *out, size_t max_out)
{
	size_t n = 0, i = 0;
	const __m256i lut = _mm256_setr_epi8(
		0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
		0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
	const __m256i mask = _mm256_set1_epi8(0x0f);
	const __m256i zero = _mm256_setzero_si256();
	/* Broadcast the two query words across both 128-bit halves. */
	const __m256i q = _mm256_setr_epi64x((long long)query[0], (long long)query[1],
	                                     (long long)query[0], (long long)query[1]);

	/* Two records per iteration. */
	for (; i + 2 <= st->count && n + 2 <= max_out; i += 2) {
		__m256i v = _mm256_load_si256(
			(const __m256i *)(st->lsh + i * SIGIL_LSH_WORDS));
		__m256i x = _mm256_xor_si256(v, q);
		__m256i lo = _mm256_and_si256(x, mask);
		__m256i hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), mask);
		__m256i cnt = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo),
		                              _mm256_shuffle_epi8(lut, hi));
		/* Four u64 lanes, each the popcount of one LSH word. */
		__m256i sums = _mm256_sad_epu8(cnt, zero);
		uint64_t s[4];

		_mm256_storeu_si256((__m256i *)s, sums);

		if ((uint32_t)(s[0] + s[1]) <= max_distance)
			out[n++] = (uint32_t)i;
		if ((uint32_t)(s[2] + s[3]) <= max_distance)
			out[n++] = (uint32_t)(i + 1);
	}

	for (; i < st->count && n < max_out; i++) {
		if (sigil_hamming(st->lsh + i * SIGIL_LSH_WORDS, query)
		    <= max_distance)
			out[n++] = (uint32_t)i;
	}
	return n;
}

__attribute__((target("avx2")))
static size_t scan_timerange_avx2(const sigil_store_t *st,
                                 uint32_t start, uint32_t end,
                                 uint32_t *out, size_t max_out)
{
	size_t n = 0, i = 0;
	/* Timestamps are unsigned but AVX2 only has signed 32-bit compares.
	 * Bias by 2^31 so unsigned order becomes signed order. */
	const __m256i bias = _mm256_set1_epi32((int)0x80000000u);
	const __m256i lo   = _mm256_set1_epi32((int)(start ^ 0x80000000u));
	const __m256i hi   = _mm256_set1_epi32((int)(end   ^ 0x80000000u));

	for (; i + 8 <= st->count && n + 8 <= max_out; i += 8) {
		__m256i v = _mm256_load_si256((const __m256i *)(st->timestamp + i));
		__m256i b = _mm256_xor_si256(v, bias);
		/* in range <=> !(b < lo) && !(b > hi) */
		__m256i too_low  = _mm256_cmpgt_epi32(lo, b);
		__m256i too_high = _mm256_cmpgt_epi32(b, hi);
		__m256i bad      = _mm256_or_si256(too_low, too_high);
		int mask = ~_mm256_movemask_ps(_mm256_castsi256_ps(bad)) & 0xff;

		while (mask) {
			int lane = __builtin_ctz((unsigned)mask);
			out[n++] = (uint32_t)(i + (size_t)lane);
			mask &= mask - 1;
		}
	}

	for (; i < st->count && n < max_out; i++) {
		uint32_t t = st->timestamp[i];

		if (t >= start && t <= end)
			out[n++] = (uint32_t)i;
	}
	return n;
}

__attribute__((target("avx2")))
static size_t scan_category_avx2(const sigil_store_t *st, uint16_t category,
                                uint32_t *out, size_t max_out)
{
	size_t n = 0, i = 0;
	const __m256i q = _mm256_set1_epi16((short)category);

	/* 16 categories per register. */
	for (; i + 16 <= st->count && n + 16 <= max_out; i += 16) {
		__m256i v  = _mm256_load_si256((const __m256i *)(st->category + i));
		__m256i eq = _mm256_cmpeq_epi16(v, q);
		/* movemask_epi8 gives two bits per 16-bit lane; both are set or
		 * both clear, so test every other bit. */
		unsigned mask = (unsigned)_mm256_movemask_epi8(eq);

		while (mask) {
			int bit = __builtin_ctz(mask);
			out[n++] = (uint32_t)(i + (size_t)(bit / 2));
			mask &= ~(3u << (bit & ~1));
		}
	}

	for (; i < st->count && n < max_out; i++) {
		if (st->category[i] == category)
			out[n++] = (uint32_t)i;
	}
	return n;
}

/* ---------------------------------------------------------------------------
 * Dispatch
 *
 * AVX2, else SSE4.2, else scalar. Both probes are cached, and the scan loops
 * run for millions of records, so the branches at entry cost nothing
 * measurable. Dispatch has to be at runtime because this file is built for
 * generic x86-64: the vector bodies carry target attributes, so the binary
 * loads on a 2010 Xeon that would SIGILL on the AVX2 path.
 * ------------------------------------------------------------------------ */

size_t sigil_scan_similar_simd(const sigil_store_t *st, const uint64_t *query,
                               uint32_t max_distance,
                               uint32_t *out, size_t max_out)
{
	if (have_avx2())
		return scan_similar_avx2(st, query, max_distance, out, max_out);
	if (have_sse42())
		return sigil_scan_similar_sse(st, query, max_distance, out, max_out);
	return sigil_scan_similar_scalar(st, query, max_distance, out, max_out);
}

size_t sigil_scan_timerange_simd(const sigil_store_t *st,
                                 uint32_t start, uint32_t end,
                                 uint32_t *out, size_t max_out)
{
	if (have_avx2())
		return scan_timerange_avx2(st, start, end, out, max_out);
	if (have_sse42())
		return sigil_scan_timerange_sse(st, start, end, out, max_out);
	return sigil_scan_timerange_scalar(st, start, end, out, max_out);
}

size_t sigil_scan_category_simd(const sigil_store_t *st, uint16_t category,
                                uint32_t *out, size_t max_out)
{
	if (have_avx2())
		return scan_category_avx2(st, category, out, max_out);
	if (have_sse42())
		return sigil_scan_category_sse(st, category, out, max_out);
	return sigil_scan_category_scalar(st, category, out, max_out);
}

#endif /* SIGIL_X86 */
