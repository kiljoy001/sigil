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

/* clock_gettime is POSIX, not C11; this builds with -std=c11. */
#define _POSIX_C_SOURCE 200809L

#include "sigil.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

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
	/* Both probes, not a short-circuit: || would leave have_sse42()
	 * unevaluated on every machine that has AVX2, so the SSE4.2 probe
	 * would first run on hardware nobody was testing. */
	int avx2 = have_avx2();
	int sse42 = have_sse42();

	return avx2 || sse42;
}

/*
 * Which vector paths this CPU offers. Reports what was available, not what
 * the calibration below chose.
 */
int sigil_simd_paths(int *avx2, int *sse42)
{
	int a = have_avx2(), s = have_sse42();

	if (avx2 != NULL)
		*avx2 = a;
	if (sse42 != NULL)
		*sse42 = s;
	return a ? 2 : (s ? 1 : 0);
}

/* Implemented in scan_sse.c; declared here rather than in the public header
 * because callers pick a path through the _simd entry points, not directly. */
size_t sigil_scan_similar_sse(const sigil_view_t *sv, const void *arg,
                              uint32_t *out, size_t max_out);
size_t sigil_scan_timerange_sse(const sigil_view_t *sv, const void *arg,
                                uint32_t *out, size_t max_out);
size_t sigil_scan_category_sse(const sigil_view_t *sv, const void *arg,
                               uint32_t *out, size_t max_out);


/*
 * 128-bit LSH similarity.
 *
 * Each record is two u64 words, so one YMM register holds exactly two records
 * and the per-record distance is a sum over its own 128-bit half. This is a
 * better fit than the old 32-bit layout, which needed maddubs/madd folding to
 * keep per-record totals from bleeding across lanes.
 *
 * Loads are unaligned (loadu). A ranged view can start at any index, so the
 * base pointer is not guaranteed 32-byte aligned; on modern x86 loadu on
 * aligned data costs nothing anyway.
 *
 * vpsadbw does the work: it sums absolute byte differences into four u64
 * fields, one per 64-bit group. Against a zeroed operand that is just a byte
 * sum, so after the nibble-LUT popcount each u64 lane holds the bit count for
 * its own word. Adding the two lanes of a 128-bit half gives that record's
 * distance, and the halves are independent — no cross-lane shuffles.
 */
__attribute__((target("avx2")))
static size_t scan_similar_avx2(const sigil_view_t *sv, const void *arg,
                               uint32_t *out, size_t max_out)
{
	const sigil_simarg_t *a = arg;
	const uint64_t *query = a->query;
	uint32_t max_distance = a->max_distance;
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
	for (; i + 2 <= sv->count && n + 2 <= max_out; i += 2) {
		__m256i v = _mm256_loadu_si256(
			(const __m256i *)(sv->lsh + i * SIGIL_LSH_WORDS));
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

	for (; i < sv->count && n < max_out; i++) {
		if (sigil_hamming(sv->lsh + i * SIGIL_LSH_WORDS, query)
		    <= max_distance)
			out[n++] = (uint32_t)i;
	}
	return n;
}

__attribute__((target("avx2")))
static size_t scan_timerange_avx2(const sigil_view_t *sv, const void *arg,
                                 uint32_t *out, size_t max_out)
{
	const sigil_timearg_t *a = arg;
	uint32_t start = a->start, end = a->end;
	size_t n = 0, i = 0;
	/* Timestamps are unsigned but AVX2 only has signed 32-bit compares.
	 * Bias by 2^31 so unsigned order becomes signed order. */
	const __m256i bias = _mm256_set1_epi32((int)0x80000000u);
	const __m256i lo   = _mm256_set1_epi32((int)(start ^ 0x80000000u));
	const __m256i hi   = _mm256_set1_epi32((int)(end   ^ 0x80000000u));

	for (; i + 8 <= sv->count && n + 8 <= max_out; i += 8) {
		__m256i v = _mm256_loadu_si256((const __m256i *)(sv->timestamp + i));
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

	for (; i < sv->count && n < max_out; i++) {
		uint32_t t = sv->timestamp[i];

		if (t >= start && t <= end)
			out[n++] = (uint32_t)i;
	}
	return n;
}

__attribute__((target("avx2")))
static size_t scan_category_avx2(const sigil_view_t *sv, const void *arg,
                                uint32_t *out, size_t max_out)
{
	uint16_t category = *(const uint16_t *)arg;
	size_t n = 0, i = 0;
	const __m256i q = _mm256_set1_epi16((short)category);

	/* 16 categories per register. */
	for (; i + 16 <= sv->count && n + 16 <= max_out; i += 16) {
		__m256i v  = _mm256_loadu_si256((const __m256i *)(sv->category + i));
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

	for (; i < sv->count && n < max_out; i++) {
		if (sv->category[i] == category)
			out[n++] = (uint32_t)i;
	}
	return n;
}

/* ---------------------------------------------------------------------------
 * Dispatch
 *
 * Measured, not assumed.
 *
 * CPUID says which instructions exist, not which kernel is fastest. AVX2
 * measured 2.2x scalar on an i5-12600K and SSE4.2 2.08x -- close enough that
 * the ordering is not obvious, and on parts that downclock under 256-bit
 * loads the wide path can lose outright. This scan is memory-bound, so the
 * width advantage is smaller than it looks.
 *
 * The first call therefore races the kernels CPUID says are available over a
 * synthetic store and keeps the winner: 1.9 ms once, then a cached integer
 * against loops that run over millions of records. If calibration cannot run,
 * it falls back to the CPUID order, which is what this did before.
 *
 * Dispatch stays at runtime regardless: this file is built for generic
 * x86-64 and the vector bodies carry target attributes, so the binary loads
 * on a 2010 Xeon that would SIGILL on the AVX2 path.
 * ------------------------------------------------------------------------ */

enum { PathScalar = 0, PathSse = 1, PathAvx2 = 2 };

static int chosen_path = -1;

static double
now_ms(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return (double)t.tv_sec * 1000.0 + (double)t.tv_nsec / 1e6;
}

/* Best of several passes: a scheduler hiccup on one pass must not decide
 * which kernel this process uses for the rest of its life. */
static double
time_similar(sigil_scan_fn fn, const sigil_view_t *sv, const uint64_t *q,
             uint32_t *out, size_t max_out)
{
	sigil_simarg_t a;
	double best = 1e30;
	int rep;

	a.query = q;
	a.max_distance = 64;
	for (rep = 0; rep < 5; rep++) {
		double t = now_ms();

		fn(sv, &a, out, max_out);
		t = now_ms() - t;
		if (t < best)
			best = t;
	}
	return best;
}

static void
calibrate(void)
{
	sigil_store_t st;
	sigil_view_t sv;
	uint64_t q[SIGIL_LSH_WORDS];
	uint32_t *out;
	uint64_t seed = 0x9e3779b97f4a7c15ULL;
	size_t i;
	const size_t N = 20000;
	double t_scalar, t_sse = 1e30, t_avx2 = 1e30;
	int best;

	/* Fall back to the CPUID order if anything here fails: a calibration
	 * that cannot allocate must not silently disable vectorisation. */
	best = have_avx2() ? PathAvx2 : (have_sse42() ? PathSse : PathScalar);

	if (sigil_store_init(&st, N) != 0) {
		chosen_path = best;
		return;
	}
	out = malloc(N * sizeof *out);
	if (out == NULL) {
		sigil_store_free(&st);
		chosen_path = best;
		return;
	}

	for (i = 0; i < N; i++) {
		sigil_t s;
		int w;

		memset(&s, 0, sizeof s);
		for (w = 0; w < SIGIL_LSH_WORDS; w++) {
			seed = seed * 6364136223846793005ULL
			     + 1442695040888963407ULL;
			s.lsh[w] = seed;
		}
		s.timestamp = (uint32_t)i;
		s.category = (uint16_t)(i % 8);
		if (sigil_store_push(&st, &s) < 0)
			break;
	}
	for (i = 0; i < SIGIL_LSH_WORDS; i++)
		q[i] = seed ^ (i + 1);

	/* Calibrate on the store's first segment: the kernels take a view,
	 * and one segment is what any of them ever sees in a real scan. */
	sv.lsh       = st.lsh[0];
	sv.para      = st.para[0];
	sv.cluster   = st.cluster[0];
	sv.timestamp = st.timestamp[0];
	sv.category  = st.category[0];
	sv.trits     = st.trits[0];
	sv.hash      = st.hash[0];
	sv.count     = st.count;

	t_scalar = time_similar(sigil_kernel_similar_scalar, &sv, q, out, N);
	if (have_sse42())
		t_sse = time_similar(sigil_scan_similar_sse, &sv, q, out, N);
	if (have_avx2())
		t_avx2 = time_similar(scan_similar_avx2, &sv, q, out, N);

	best = PathScalar;
	if (t_sse < t_scalar)
		best = PathSse;
	if (t_avx2 < (best == PathSse ? t_sse : t_scalar))
		best = PathAvx2;

	free(out);
	sigil_store_free(&st);
	chosen_path = best;
}

/*
 * Force a kernel, for tests and for bug reports.
 *
 * The calibration picks one path and the others are then never executed, so
 * coverage of a kernel depends on which one happened to win the race. Under
 * an -O0 coverage build scalar wins, which left the AVX2 filter kernels at
 * 0% -- reported as untested code when the real problem was that no test
 * could reach them. Silently untestable code is worse than uncovered code:
 * the differential tests can only prove the kernels agree if they can run
 * each one.
 *
 * Returns 0 on success, -1 if this CPU does not offer the requested path.
 * SIGIL_SIMD_PATH=scalar|sse|avx2 does the same from the environment, which
 * is how the coverage build sweeps all three.
 */
int sigil_simd_force(int want)
{
	switch (want) {
	case PathAvx2:
		if (!have_avx2())
			return -1;
		break;
	case PathSse:
		if (!have_sse42())
			return -1;
		break;
	case PathScalar:
		break;
	default:
		return -1;
	}
	chosen_path = want;
	return 0;
}

/* Undo a force: the next call re-runs (or reuses) the calibration. */
void sigil_simd_unforce(void)
{
	chosen_path = -1;
}

static int
path(void)
{
	if (chosen_path < 0) {
		const char *env = getenv("SIGIL_SIMD_PATH");

		/* An explicit request wins over the measurement, and a
		 * request this CPU cannot satisfy falls through to the
		 * calibration rather than pretending: forcing AVX2 on a
		 * machine without it would be a SIGILL, which is exactly
		 * the crash the runtime dispatch exists to prevent. */
		if (env != NULL) {
			if (strcmp(env, "scalar") == 0)
				return chosen_path = PathScalar;
			if (strcmp(env, "sse") == 0 && have_sse42())
				return chosen_path = PathSse;
			if (strcmp(env, "avx2") == 0 && have_avx2())
				return chosen_path = PathAvx2;
		}
		calibrate();
	}
	return chosen_path;
}

/* Which kernel the calibration chose. For tests, and for a bug report that
 * needs to say which path produced a number. */
int sigil_simd_chosen(void)
{
	return path();
}

size_t sigil_kernel_similar(const sigil_view_t *sv, const void *arg,
                            uint32_t *out, size_t max_out)
{
	switch (path()) {
	case PathAvx2:
		return scan_similar_avx2(sv, arg, out, max_out);
	case PathSse:
		return sigil_scan_similar_sse(sv, arg, out, max_out);
	default:
		return sigil_kernel_similar_scalar(sv, arg, out, max_out);
	}
}

size_t sigil_kernel_timerange(const sigil_view_t *sv, const void *arg,
                              uint32_t *out, size_t max_out)
{
	switch (path()) {
	case PathAvx2:
		return scan_timerange_avx2(sv, arg, out, max_out);
	case PathSse:
		return sigil_scan_timerange_sse(sv, arg, out, max_out);
	default:
		return sigil_kernel_timerange_scalar(sv, arg, out, max_out);
	}
}

size_t sigil_kernel_category(const sigil_view_t *sv, const void *arg,
                             uint32_t *out, size_t max_out)
{
	switch (path()) {
	case PathAvx2:
		return scan_category_avx2(sv, arg, out, max_out);
	case PathSse:
		return sigil_scan_category_sse(sv, arg, out, max_out);
	default:
		return sigil_kernel_category_scalar(sv, arg, out, max_out);
	}
}

#endif /* SIGIL_X86 */
