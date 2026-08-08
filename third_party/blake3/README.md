# BLAKE3 (vendored)

Portable C implementation from https://github.com/BLAKE3-team/BLAKE3, `c/`
subdirectory. Dual-licensed CC0-1.0 and Apache-2.0; both are GPLv3-compatible.

Only the portable path is vendored — `blake3.c`, `blake3_dispatch.c`,
`blake3_portable.c` and their headers. The upstream SIMD files
(`blake3_avx2.c`, `blake3_sse41.c`, `blake3_neon.c`, …) are deliberately
omitted and `BLAKE3_NO_AVX512`/`AVX2`/`SSE41`/`SSE2`/`NEON` are defined in the
Makefile so `blake3_dispatch.c` compiles to the portable path unconditionally.

Two reasons. sigil already has its own runtime SIMD dispatch with CPUID probing
in `src/scan_x86.c`, and a second independent detection layer is a liability
rather than a speedup. And hashing is not the bottleneck: even portable BLAKE3
is roughly 11x the SHA-1 it replaces, while embedding costs ~60x more per
document than hashing does.

If profiling ever shows hashing on the critical path, add the upstream SIMD
files and drop the corresponding `-DBLAKE3_NO_*` flags.
