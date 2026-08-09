// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Scott J Guyton

//
// One scan kernel, targeting CPU, GPU and NPU through SYCL.
//
// The hand-written kernels -- AVX2, SSE4.2, NEON, scalar -- are four sources
// implementing one idea: popcount the XOR of a 128-bit field and compare
// against a threshold. Each needs its own differential test against the scalar
// twin, because a wrong SIMD kernel does not crash, it returns a plausible
// distance and stays wrong forever. This file is the argument for writing that
// idea once.
//
// It does not replace them. libsigil stays C11 with no dependencies, which
// matters for a library that may end up in a kernel module, and the scalar
// path remains the definition of correct. This is an additional backend for
// hosts that have oneAPI, selected at runtime like the others.
//
// The serial append in the scalar version is the one thing that does not
// translate. `out[n++]` with an early exit is inherently sequential; here each
// work item tests one record and writes into a match mask, and the indices are
// compacted afterwards. That costs a pass over the mask but is the only shape
// that parallelises.
//
// Build:
//   icpx -fsycl -O3 -c src/scan_sycl.cpp -o scan_sycl.o
//

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstddef>
#include <vector>

extern "C" {

// Mirrors sigil_store_t's lsh field without including sigil.h: that header is
// C11 with stdint.h and is included by plan9port code elsewhere, and dragging
// it through a C++ compiler invites the same collisions that forced the bridge
// in cmd/bridge.c.
#define SIGIL_LSH_WORDS_SYCL 2

// One queue per device, created once. Queue construction compiles the kernel
// for the target, which is far too slow to do per scan.
static sycl::queue *g_queue = nullptr;
static int g_device_kind = -1;   // 0 cpu, 1 gpu, 2 accelerator/npu

int sigil_sycl_init(int prefer_gpu)
{
	try {
		if (g_queue != nullptr)
			return g_device_kind;

		sycl::device dev;
		if (prefer_gpu) {
			try {
				dev = sycl::device(sycl::gpu_selector_v);
			} catch (const sycl::exception &) {
				dev = sycl::device(sycl::cpu_selector_v);
			}
		} else {
			dev = sycl::device(sycl::cpu_selector_v);
		}

		g_queue = new sycl::queue(dev);
		g_device_kind = dev.is_gpu() ? 1 : (dev.is_accelerator() ? 2 : 0);
		return g_device_kind;
	} catch (const sycl::exception &) {
		return -1;
	}
}

const char *sigil_sycl_device_name(void)
{
	static std::string name;
	if (g_queue == nullptr)
		return "none";
	name = g_queue->get_device().get_info<sycl::info::device::name>();
	return name.c_str();
}

//
// Hamming scan. Returns the number of matches written to `out`, in ascending
// index order, which is what the scalar kernel guarantees and what the
// differential test checks.
//
size_t sigil_scan_similar_sycl(const uint64_t *lsh, size_t count,
                               const uint64_t *query, uint32_t max_distance,
                               uint32_t *out, size_t max_out)
{
	if (g_queue == nullptr && sigil_sycl_init(1) < 0)
		return 0;
	if (count == 0 || max_out == 0)
		return 0;

	sycl::queue &q = *g_queue;

	try {
		const uint64_t q0 = query[0], q1 = query[1];

		// A byte per record rather than a bitset: contention-free, and the
		// compaction pass below is memory-bound either way.
		uint8_t *hit = sycl::malloc_device<uint8_t>(count, q);
		uint64_t *dev_lsh = sycl::malloc_device<uint64_t>(
			count * SIGIL_LSH_WORDS_SYCL, q);
		if (hit == nullptr || dev_lsh == nullptr) {
			if (hit) sycl::free(hit, q);
			if (dev_lsh) sycl::free(dev_lsh, q);
			return 0;
		}

		q.memcpy(dev_lsh, lsh,
		         count * SIGIL_LSH_WORDS_SYCL * sizeof(uint64_t)).wait();

		q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
			const size_t k = i[0] * SIGIL_LSH_WORDS_SYCL;
			// sycl::popcount lowers to the native instruction on every
			// target: POPCNT on x86, CNT on NEON, and the GPU equivalent.
			const uint32_t d =
				sycl::popcount(dev_lsh[k]     ^ q0) +
				sycl::popcount(dev_lsh[k + 1] ^ q1);
			hit[i] = (d <= max_distance) ? 1 : 0;
		}).wait();

		std::vector<uint8_t> host_hit(count);
		q.memcpy(host_hit.data(), hit, count).wait();

		sycl::free(hit, q);
		sycl::free(dev_lsh, q);

		// Compaction stays on the host: the result set is small relative to
		// the store, and a device-side prefix sum would cost more than it
		// saves at these hit rates.
		size_t n = 0;
		for (size_t i = 0; i < count && n < max_out; i++)
			if (host_hit[i])
				out[n++] = (uint32_t)i;
		return n;
	} catch (const sycl::exception &) {
		return 0;
	}
}

void sigil_sycl_shutdown(void)
{
	delete g_queue;
	g_queue = nullptr;
	g_device_kind = -1;
}

}  // extern "C"
