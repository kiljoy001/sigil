/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Sigil construction and text encoding.
 *
 * The SHA-1 here is a self-contained implementation so the library has no link
 * dependencies. SHA-1 is used for content identity, not security: collision
 * resistance against a deliberate attacker is not claimed, and if that
 * property is ever needed the hash field should widen to SHA-256 and the
 * 32-byte layout revisited.
 */

#include "sigil.h"

#include <string.h>

/* --- SHA-1 ---------------------------------------------------------------- */

typedef struct {
	uint32_t h[5];
	uint64_t len;
	uint8_t  buf[64];
	size_t   buflen;
} sha1_ctx;

static inline uint32_t rol(uint32_t v, int n)
{
	return (v << n) | (v >> (32 - n));
}

static void sha1_block(sha1_ctx *c, const uint8_t *p)
{
	uint32_t w[80], a, b, d, e, f, k, tmp;
	uint32_t cc;

	for (int i = 0; i < 16; i++)
		w[i] = ((uint32_t)p[i * 4] << 24) | ((uint32_t)p[i * 4 + 1] << 16) |
		       ((uint32_t)p[i * 4 + 2] << 8) | (uint32_t)p[i * 4 + 3];
	for (int i = 16; i < 80; i++)
		w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

	a = c->h[0]; b = c->h[1]; cc = c->h[2]; d = c->h[3]; e = c->h[4];

	for (int i = 0; i < 80; i++) {
		if (i < 20)      { f = (b & cc) | (~b & d);       k = 0x5a827999; }
		else if (i < 40) { f = b ^ cc ^ d;                k = 0x6ed9eba1; }
		else if (i < 60) { f = (b & cc) | (b & d) | (cc & d); k = 0x8f1bbcdc; }
		else             { f = b ^ cc ^ d;                k = 0xca62c1d6; }

		tmp = rol(a, 5) + f + e + k + w[i];
		e = d; d = cc; cc = rol(b, 30); b = a; a = tmp;
	}

	c->h[0] += a; c->h[1] += b; c->h[2] += cc; c->h[3] += d; c->h[4] += e;
}

static void sha1_init(sha1_ctx *c)
{
	c->h[0] = 0x67452301; c->h[1] = 0xefcdab89; c->h[2] = 0x98badcfe;
	c->h[3] = 0x10325476; c->h[4] = 0xc3d2e1f0;
	c->len = 0;
	c->buflen = 0;
}

static void sha1_update(sha1_ctx *c, const uint8_t *p, size_t n)
{
	c->len += n;
	while (n) {
		size_t take = 64 - c->buflen;

		if (take > n)
			take = n;
		memcpy(c->buf + c->buflen, p, take);
		c->buflen += take;
		p += take;
		n -= take;
		if (c->buflen == 64) {
			sha1_block(c, c->buf);
			c->buflen = 0;
		}
	}
}

static void sha1_final(sha1_ctx *c, uint8_t out[20])
{
	uint64_t bits = c->len * 8;
	uint8_t pad = 0x80;

	sha1_update(c, &pad, 1);
	pad = 0;
	while (c->buflen != 56)
		sha1_update(c, &pad, 1);
	for (int i = 7; i >= 0; i--) {
		uint8_t b = (uint8_t)(bits >> (i * 8));
		sha1_update(c, &b, 1);
	}
	for (int i = 0; i < 5; i++) {
		out[i * 4]     = (uint8_t)(c->h[i] >> 24);
		out[i * 4 + 1] = (uint8_t)(c->h[i] >> 16);
		out[i * 4 + 2] = (uint8_t)(c->h[i] >> 8);
		out[i * 4 + 3] = (uint8_t)c->h[i];
	}
}

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
	sha1_ctx c;

	sha1_init(&c);
	sha1_update(&c, (const uint8_t *)content, len);
	sha1_final(&c, out->hash);

	compute_lsh((const uint8_t *)content, len, out->lsh);
	out->timestamp = timestamp;
	out->category  = category;
	out->trits     = trits ? sigil_trits_pack(trits) : 0;
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
