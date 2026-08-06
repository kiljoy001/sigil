/*
 * AVX2 scan kernels.
 *
 * Eight 32-bit lanes per YMM register. These must agree with the scalar
 * kernels bit-for-bit on every input; test/differential.c is the enforcement.
 *
 * AVX2 has no vpopcntd (that is AVX-512 VPOPCNTDQ), so population count goes
 * through the standard nibble-table shuffle: split each byte into two nibbles,
 * look both up in a 16-entry table replicated across both 128-bit halves via
 * vpshufb, then sum. Roughly four instructions per eight words.
 */

#include "sigil.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <cpuid.h>
#define SIGIL_X86 1
#endif

int sigil_have_avx2(void)
{
#ifdef SIGIL_X86
	unsigned eax, ebx, ecx, edx;

	if (__get_cpuid_max(0, NULL) < 7)
		return 0;
	__cpuid_count(7, 0, eax, ebx, ecx, edx);
	return (ebx & bit_AVX2) != 0;
#else
	return 0;
#endif
}

#ifdef SIGIL_X86

__attribute__((target("avx2")))
static inline __m256i popcount_epi32(__m256i v)
{
	/* vpshufb operates per 128-bit lane, so the table is duplicated. */
	const __m256i lut = _mm256_setr_epi8(
		0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
		0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
	const __m256i mask = _mm256_set1_epi8(0x0f);

	__m256i lo = _mm256_and_si256(v, mask);
	__m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask);
	__m256i cnt = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo),
	                              _mm256_shuffle_epi8(lut, hi));

	/* Byte counts -> 32-bit lane totals. maddubs sums adjacent byte pairs
	 * into 16-bit fields, then madd sums those pairs into 32-bit lanes,
	 * which keeps each total inside the lane it came from. */
	__m256i sum16 = _mm256_maddubs_epi16(cnt, _mm256_set1_epi8(1));
	__m256i sum32 = _mm256_madd_epi16(sum16, _mm256_set1_epi16(1));

	return sum32;
}

__attribute__((target("avx2")))
size_t sigil_scan_similar_avx2(const sigil_store_t *st, uint32_t query,
                               uint32_t max_distance,
                               uint32_t *out, size_t max_out)
{
	size_t n = 0, i = 0;
	const __m256i q   = _mm256_set1_epi32((int)query);
	const __m256i thr = _mm256_set1_epi32((int)max_distance);

	for (; i + 8 <= st->count && n + 8 <= max_out; i += 8) {
		__m256i v    = _mm256_load_si256((const __m256i *)(st->lsh + i));
		__m256i dist = popcount_epi32(_mm256_xor_si256(v, q));
		/* dist <= thr  <=>  !(dist > thr); both operands are small and
		 * non-negative, so signed compare is safe. */
		__m256i gt   = _mm256_cmpgt_epi32(dist, thr);
		int mask     = ~_mm256_movemask_ps(_mm256_castsi256_ps(gt)) & 0xff;

		while (mask) {
			int lane = __builtin_ctz((unsigned)mask);
			out[n++] = (uint32_t)(i + (size_t)lane);
			mask &= mask - 1;
		}
	}

	for (; i < st->count && n < max_out; i++) {
		if (sigil_hamming(st->lsh[i], query) <= max_distance)
			out[n++] = (uint32_t)i;
	}
	return n;
}

__attribute__((target("avx2")))
size_t sigil_scan_timerange_avx2(const sigil_store_t *st,
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
size_t sigil_scan_category_avx2(const sigil_store_t *st, uint16_t category,
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

#else /* !SIGIL_X86 — fall back so the library builds anywhere */

size_t sigil_scan_similar_avx2(const sigil_store_t *st, uint32_t query,
                               uint32_t max_distance,
                               uint32_t *out, size_t max_out)
{
	return sigil_scan_similar_scalar(st, query, max_distance, out, max_out);
}

size_t sigil_scan_timerange_avx2(const sigil_store_t *st,
                                 uint32_t start, uint32_t end,
                                 uint32_t *out, size_t max_out)
{
	return sigil_scan_timerange_scalar(st, start, end, out, max_out);
}

size_t sigil_scan_category_avx2(const sigil_store_t *st, uint16_t category,
                                uint32_t *out, size_t max_out)
{
	return sigil_scan_category_scalar(st, category, out, max_out);
}

#endif
