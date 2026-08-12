/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * NEON scan kernels for aarch64.
 *
 * These must agree with the scalar kernels bit-for-bit; test/differential.c is
 * the enforcement, and it runs unchanged on both architectures.
 *
 * The similarity kernel is genuinely simpler here than on x86: a NEON register
 * is 128 bits, exactly one LSH code, so there is no lane bookkeeping, and
 * aarch64 has vcntq_u8, a real per-byte popcount. AVX2 has no equivalent below
 * AVX-512 and emulates it with a nibble table and vpshufb. The inner loop is
 * three operations — veor, vcnt, vaddv.
 *
 * Simpler did not mean faster. Measured on Cortex-A76 at 10M records, NEON
 * gives 1.48x over scalar (18.1ms vs 26.8ms) where AVX2 gives 2.2x on x86.
 * Two reasons, both structural rather than fixable:
 *
 *   - AVX2 is 256-bit and processes two records per iteration; NEON's 128-bit
 *     register holds exactly one. Half the width, half the parallelism.
 *   - aarch64 has no movemask equivalent, so kernels that select lanes have to
 *     round-trip through memory instead of extracting a bitmask in a register.
 *     That penalty lands on timerange and category, not on similarity.
 *
 * Unrolling the similarity loop to four independent chains was tried and came
 * out slower (19.8ms), since the loop is memory-bound and extra in-flight work
 * only adds register pressure.
 *
 * NEON is mandatory in the aarch64 base architecture, so unlike AVX2 there is
 * nothing to probe at runtime — if this file compiles, the instructions exist.
 */

#include "sigil.h"

#if defined(__aarch64__)
#define SIGIL_NEON 1
#include <arm_neon.h>
#endif

#ifdef SIGIL_NEON

int sigil_have_simd(void)
{
	return 1; /* baseline on aarch64 */
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

size_t sigil_scan_similar_simd(const sigil_store_t *st, const uint64_t *query,
                               uint32_t max_distance,
                               uint32_t *out, size_t max_out)
{
	size_t n = 0, i = 0;
	const uint64x2_t q = vld1q_u64(query);

	/* One record per iteration: a 128-bit code is exactly one register, so
	 * there is no lane bookkeeping at all. Unrolling to four independent
	 * chains was measured and came out slower (19.8ms vs 18.1ms on
	 * Cortex-A76) — the loop is memory-bound, so extra in-flight work buys
	 * nothing and only adds register pressure. Kept simple deliberately. */
	for (; i < st->count && n < max_out; i++) {
		uint8x16_t x = veorq_u8(
			vreinterpretq_u8_u64(vld1q_u64(st->lsh + i * SIGIL_LSH_WORDS)),
			vreinterpretq_u8_u64(q));
		/* vcntq_u8 is a real per-byte popcount instruction; AVX2 has no
		 * equivalent below AVX-512 and has to emulate it with a nibble
		 * table. vaddvq_u8 then reduces all 16 bytes in one instruction.
		 * Totals cap at 128, so the u8 lanes cannot overflow. */
		if (vaddvq_u8(vcntq_u8(x)) <= max_distance)
			out[n++] = (uint32_t)i;
	}

	return n;
}

size_t sigil_scan_timerange_simd(const sigil_store_t *st,
                                 uint32_t start, uint32_t end,
                                 uint32_t *out, size_t max_out)
{
	size_t n = 0, i = 0;
	const uint32x4_t lo = vdupq_n_u32(start);
	const uint32x4_t hi = vdupq_n_u32(end);

	/* NEON compares unsigned natively, so no 2^31 bias is needed here —
	 * the x86 kernel only carries one because AVX2 lacks unsigned compare.
	 *
	 * Eight per iteration, not four. aarch64 has no movemask, so results
	 * come back through memory either way; doing two vectors per store
	 * halves the loop overhead. The vmaxvq early-out matters more: it is
	 * one instruction, and a selective range leaves most blocks empty, so
	 * the per-lane branch loop is skipped entirely for those. Without it
	 * this kernel loses to scalar, because scalar's branch predictor
	 * handles a mostly-false compare better than an unconditional store
	 * plus four branches does. */
	for (; i + 8 <= st->count && n + 8 <= max_out; i += 8) {
		uint32x4_t v0 = vld1q_u32(st->timestamp + i);
		uint32x4_t v1 = vld1q_u32(st->timestamp + i + 4);
		uint32x4_t k0 = vandq_u32(vcgeq_u32(v0, lo), vcleq_u32(v0, hi));
		uint32x4_t k1 = vandq_u32(vcgeq_u32(v1, lo), vcleq_u32(v1, hi));
		uint32_t lanes[8];

		if (!vmaxvq_u32(vorrq_u32(k0, k1)))
			continue;

		vst1q_u32(lanes, k0);
		vst1q_u32(lanes + 4, k1);
		for (int k = 0; k < 8; k++) {
			if (lanes[k])
				out[n++] = (uint32_t)(i + (size_t)k);
		}
	}

	for (; i < st->count && n < max_out; i++) {
		uint32_t t = st->timestamp[i];

		if (t >= start && t <= end)
			out[n++] = (uint32_t)i;
	}
	return n;
}

size_t sigil_scan_category_simd(const sigil_store_t *st, uint16_t category,
                                uint32_t *out, size_t max_out)
{
	size_t n = 0, i = 0;
	const uint16x8_t q = vdupq_n_u16(category);

	for (; i + 8 <= st->count && n + 8 <= max_out; i += 8) {
		uint16x8_t v  = vld1q_u16(st->category + i);
		uint16x8_t eq = vceqq_u16(v, q);
		uint16_t lanes[8];

		/* Cheap early-out: vmaxvq is one instruction and most blocks in
		 * a selective query match nothing. */
		if (vmaxvq_u16(eq) == 0)
			continue;

		vst1q_u16(lanes, eq);
		for (int k = 0; k < 8; k++) {
			if (lanes[k])
				out[n++] = (uint32_t)(i + (size_t)k);
		}
	}

	for (; i < st->count && n < max_out; i++) {
		if (st->category[i] == category)
			out[n++] = (uint32_t)i;
	}
	return n;
}

#endif /* SIGIL_NEON */
