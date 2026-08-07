/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * OpenCL similarity scan — is a GPU worth it for this workload?
 *
 * Deliberately outside libsigil. This is a measurement, not a backend: the
 * library stays dependency-free, and nothing links OpenCL unless you build
 * this file. See the results table at the bottom of this comment before
 * deciding to promote it.
 *
 * The scan looked like the wrong shape for a GPU. It is XOR plus popcount over
 * a flat array, roughly 0.06 arithmetic operations per byte, and GPUs earn
 * their keep on arithmetic intensity. It turned out to be the right shape for
 * a different reason: the work is perfectly parallel with zero divergence, and
 * OpenCL exposes popcount() as a builtin, so there is none of the nibble-table
 * emulation the AVX2 and SSE kernels need. What the GPU actually contributes
 * is its memory bandwidth, and GDDR6 has roughly 3x what dual-channel DDR5
 * does.
 *
 * The catch is PCIe. Uploading 1.6 GB of LSH codes takes ~300 ms at 5 GB/s,
 * against a 7 ms kernel — 43x the compute. A one-shot scan loses badly and a
 * resident store wins decisively, which happens to match how a filesystem
 * behaves: write once, query repeatedly.
 *
 * Measured, similarity scan, best of several runs:
 *
 *   10M records
 *     Arc Pro B50 (resident)      0.73 ms   220 GB/s
 *     i5-12600K, 16 threads       2.15 ms    74 GB/s
 *     Xeon E5645, OpenCL CPU      6.52 ms    25 GB/s
 *     Xeon E5645, 16 threads      7.11 ms    23 GB/s
 *     RK3588, 16 threads          7.88 ms    20 GB/s
 *
 *   100M records
 *     Arc Pro B50 (resident)      7.22 ms   222 GB/s
 *     Arc Pro B50 + upload      315    ms     5 GB/s   <- do not do this
 *     i5-12600K, 16 threads      21.03 ms    76 GB/s
 *
 * Two things worth keeping from this:
 *
 *   - Bandwidth is flat from 10M to 100M (220 vs 222 GB/s), so the device is
 *     saturated and the number is real rather than a cache artifact.
 *   - On a machine with no usable GPU, Intel's OpenCL CPU runtime beat a
 *     hand-written pthread pool on the same hardware (6.52 vs 7.11 ms). The
 *     same kernel is therefore a portable fallback, not only a GPU path.
 *
 * Build:
 *   cc -O2 -o scan_opencl bench/scan_opencl.c -lOpenCL
 * If the loader finds no platforms, check that an ICD is registered under
 * /etc/OpenCL/vendors and that its library's dependencies resolve — a missing
 * libtbb.so.12 silently yields zero platforms for Intel's CPU runtime.
 */

#define CL_TARGET_OPENCL_VERSION 300
#define _POSIX_C_SOURCE 200809L

#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_RESULTS 65536

static double now_sec(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/*
 * One work item per record. ulong2 is exactly a 128-bit LSH code, and
 * popcount() is a builtin — no nibble table, unlike AVX2 and SSE.
 *
 * Matches are appended through an atomic counter. That serializes on the
 * counter, which is fine while matches are rare; a permissive query radius
 * would want a per-group prefix sum instead.
 */
static const char *KERNEL_SRC =
"__kernel void hamming(__global const ulong2 *lsh, ulong2 q, uint maxd,\n"
"                      __global uint *out, volatile __global uint *cnt){\n"
"  uint i = get_global_id(0);\n"
"  ulong2 x = lsh[i] ^ q;\n"
"  uint d = popcount(x.s0) + popcount(x.s1);\n"
"  if (d <= maxd) { uint k = atomic_inc(cnt); if (k < 65536) out[k] = i; }\n"
"}\n";

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;

static uint64_t rnd64(void)
{
	rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
	return rng_state;
}

int main(int argc, char **argv)
{
	size_t count = (argc > 1) ? strtoull(argv[1], NULL, 10) : 10000000;
	cl_platform_id plats[8];
	cl_uint nplats = 0;
	cl_device_id dev = NULL;
	char dname[256] = {0};
	cl_int err;
	cl_context ctx;
	cl_command_queue queue;
	cl_program prog;
	cl_kernel kern;
	cl_mem lshbuf, outbuf, cntbuf;
	cl_ulong *host;
	cl_ulong query[2] = { 0x0123456789abcdefULL, 0x02468acf13579bdeULL };
	cl_uint maxd = 40, zero = 0, found = 0;
	size_t bytes, gws;
	double t0, upload, best;

	clGetPlatformIDs(8, plats, &nplats);
	/* Prefer a GPU; fall back to whatever the ICDs offer, which on a
	 * server is usually Intel's CPU runtime. */
	for (cl_uint p = 0; p < nplats && !dev; p++) {
		cl_device_id ds[8];
		cl_uint nd = 0;

		if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 8, ds, &nd) == CL_SUCCESS && nd)
			dev = ds[0];
	}
	for (cl_uint p = 0; p < nplats && !dev; p++) {
		cl_device_id ds[8];
		cl_uint nd = 0;

		if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 8, ds, &nd) == CL_SUCCESS && nd)
			dev = ds[0];
	}
	if (!dev) {
		fprintf(stderr, "no OpenCL device (%u platforms). Check "
		        "/etc/OpenCL/vendors and that the ICD's deps resolve.\n", nplats);
		return 1;
	}
	clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dname), dname, NULL);
	printf("device: %s\n", dname);

	ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
	queue = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);
	prog = clCreateProgramWithSource(ctx, 1, &KERNEL_SRC, NULL, &err);
	if (clBuildProgram(prog, 1, &dev, "", NULL, NULL) != CL_SUCCESS) {
		char log[8192] = {0};

		clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG,
		                      sizeof(log), log, NULL);
		fprintf(stderr, "kernel build failed:\n%s\n", log);
		return 1;
	}
	kern = clCreateKernel(prog, "hamming", &err);

	bytes = count * 16;
	host = malloc(bytes);
	if (!host) {
		fprintf(stderr, "cannot allocate %.2f GB\n", (double)bytes / 1e9);
		return 1;
	}
	for (size_t i = 0; i < count * 2; i++)
		host[i] = rnd64();

	t0 = now_sec();
	lshbuf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	                        bytes, host, &err);
	clFinish(queue);
	upload = now_sec() - t0;

	outbuf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, MAX_RESULTS * 4, NULL, &err);
	cntbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4, NULL, &err);

	clSetKernelArg(kern, 0, sizeof(cl_mem), &lshbuf);
	clSetKernelArg(kern, 1, sizeof(query), query);
	clSetKernelArg(kern, 2, sizeof(cl_uint), &maxd);
	clSetKernelArg(kern, 3, sizeof(cl_mem), &outbuf);
	clSetKernelArg(kern, 4, sizeof(cl_mem), &cntbuf);

	/* Steady-state query cost against data already on the device. This is
	 * the number that matters for a long-lived node. */
	best = 1e30;
	for (int r = 0; r < 20; r++) {
		double a;

		clEnqueueWriteBuffer(queue, cntbuf, CL_TRUE, 0, 4, &zero, 0, NULL, NULL);
		gws = count;
		a = now_sec();
		clEnqueueNDRangeKernel(queue, kern, 1, NULL, &gws, NULL, 0, NULL, NULL);
		clFinish(queue);
		clEnqueueReadBuffer(queue, cntbuf, CL_TRUE, 0, 4, &found, 0, NULL, NULL);
		if (now_sec() - a < best)
			best = now_sec() - a;
	}

	printf("n=%zu (%.2f GB of LSH)\n", count, (double)bytes / 1e9);
	printf("  upload         %8.2f ms  %7.2f GB/s\n",
	       upload * 1e3, (double)bytes / upload / 1e9);
	printf("  resident query %8.2f ms  %7.2f GB/s   %u matches\n",
	       best * 1e3, (double)bytes / best / 1e9, found);
	printf("  upload+query   %8.2f ms  %7.2f GB/s   <- one-shot cost\n",
	       (best + upload) * 1e3, (double)bytes / (best + upload) / 1e9);
	printf("\nA resident store amortizes the upload; a one-shot scan does not.\n");

	free(host);
	clReleaseMemObject(lshbuf);
	clReleaseMemObject(outbuf);
	clReleaseMemObject(cntbuf);
	clReleaseKernel(kern);
	clReleaseProgram(prog);
	clReleaseCommandQueue(queue);
	clReleaseContext(ctx);
	return 0;
}
