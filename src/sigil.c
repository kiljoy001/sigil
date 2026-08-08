/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Sigil construction and text encoding.
 *
 * Content identity is BLAKE3-256, vendored under third_party/blake3 in its
 * portable form. It replaced a hand-rolled SHA-1 that ran at ~271 MB/s and has
 * constructible collisions; BLAKE3 is roughly 3 GB/s portable and has no known
 * weakness. Full 256 bits rather than truncated — see docs/DESIGN.md on why
 * the threat model assumes a deliberate attacker.
 */

#include "sigil.h"

#include "blake3.h"

#include <string.h>

/* --- LSH ------------------------------------------------------------------
 *
 * Placeholder locality-sensitive hash: a rolling byte-shingle sketch that
 * gives similar content similar bits. This is deliberately simple and is NOT
 * the real semantic embedding — that comes from the three-tier engine
 * (MinHash / Word2Vec / BERT) and lands in these same 32 bits. Swapping this
 * out changes no other code.
 */
static void compute_lsh(const uint8_t *p, size_t n, uint64_t *out)
{
	for (int w = 0; w < SIGIL_LSH_WORDS; w++)
		out[w] = 0;

	if (n == 0)
		return;

	/* SIGIL_LSH_BITS buckets; each 4-byte shingle sets its bucket's bit. */
	for (size_t i = 0; i + 4 <= n; i++) {
		uint32_t sh = ((uint32_t)p[i] << 24) | ((uint32_t)p[i + 1] << 16) |
		              ((uint32_t)p[i + 2] << 8) | (uint32_t)p[i + 3];
		unsigned b;

		sh ^= sh >> 16;
		sh *= 0x7feb352du;
		sh ^= sh >> 15;
		b = sh % SIGIL_LSH_BITS;
		out[b / 64] |= 1ULL << (b % 64);
	}

	/* Short inputs have no 4-byte shingle; fall back to the bytes. */
	if (n < 4) {
		for (size_t i = 0; i < n; i++) {
			unsigned b = (unsigned)p[i] % SIGIL_LSH_BITS;

			out[b / 64] |= 1ULL << (b % 64);
		}
	}
}

/* --- Public --------------------------------------------------------------- */

void sigil_generate(const void *content, size_t len, uint32_t timestamp,
                    uint16_t category, const sigil_trits_t *trits,
                    sigil_t *out)
{
	sigil_generate_para(content, len, SIGIL_PARA_DOC, timestamp, category,
	                    trits, out);
}

void sigil_generate_para(const void *content, size_t len, uint32_t para,
                         uint32_t timestamp, uint16_t category,
                         const sigil_trits_t *trits, sigil_t *out)
{
	blake3_hasher h;

	blake3_hasher_init(&h);
	blake3_hasher_update(&h, content, len);
	blake3_hasher_finalize(&h, out->hash, SIGIL_HASH_LEN);

	compute_lsh((const uint8_t *)content, len, out->lsh);
	out->para      = para;
	out->timestamp = timestamp;
	out->category  = category;
	out->trits     = trits ? sigil_trits_pack(trits) : 0;
	out->cluster   = 0;
}

void sigil_to_hex(const sigil_t *s, char *buf)
{
	static const char hexdig[] = "0123456789abcdef";
	const uint8_t *p = (const uint8_t *)s;

	for (size_t i = 0; i < SIGIL_SIZE; i++) {
		buf[i * 2]     = hexdig[p[i] >> 4];
		buf[i * 2 + 1] = hexdig[p[i] & 0x0f];
	}
	buf[SIGIL_SIZE * 2] = '\0';
}

static int unhex(char ch)
{
	if (ch >= '0' && ch <= '9') return ch - '0';
	if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
	if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
	return -1;
}

int sigil_from_hex(const char *hex, sigil_t *out)
{
	uint8_t *p = (uint8_t *)out;

	for (size_t i = 0; i < SIGIL_SIZE; i++) {
		int hi = unhex(hex[i * 2]);
		int lo;

		if (hi < 0)
			return -1;
		lo = unhex(hex[i * 2 + 1]);
		if (lo < 0)
			return -1;
		p[i] = (uint8_t)((hi << 4) | lo);
	}
	return hex[SIGIL_SIZE * 2] == '\0' ? 0 : -1;
}
