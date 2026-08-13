/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Durable embedding cache. See include/sigil_veccache.h for why.
 *
 * The in-memory index holds a 32-byte hash and a float16 vector per
 * entry: at 384 dimensions that is 800 bytes, so a 75M-record corpus
 * needs about 60 GB resident to hold everything. That is affordable on
 * the machine this was written for and would not be on a smaller one, so
 * a bounded variant that keeps only hashes and seeks into the file is the
 * obvious next step if it becomes a problem. Measured before assumed:
 * nothing here is tuned for a size we have not run.
 */

/* getline, strdup and fileno are POSIX, and -std=c11 hides them. Without
 * this, strdup is implicitly declared as returning int and the model
 * string becomes a truncated pointer -- a warning that would have been a
 * corruption. */
#define _POSIX_C_SOURCE 200809L

#include "sigil_veccache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define HASHLEN 32

/*
 * Open-addressed table keyed by the first 8 bytes of the BLAKE3, with the
 * full hash compared on a probe hit. BLAKE3 is already uniformly
 * distributed, so no further hashing is needed -- and a truncated key is
 * only a bucket choice, never an identity decision.
 */
struct entry {
	uint8_t hash[HASHLEN];
	uint16_t *vec;               /* dim float16 values */
	int used;
};

struct sigil_veccache {
	FILE *fp;                    /* append handle */
	char *model;
	size_t dim;

	struct entry *tab;
	size_t cap;                  /* power of two */
	size_t n;

	uint64_t hits, misses;
};

/* --- float16, the storage form ----------------------------------------
 *
 * IEEE 754 binary16 by hand rather than through a library: the only
 * consumer is a SimHash projection whose output is a sign bit, so what
 * matters is that the conversion is exact enough not to flip one, and
 * that it has no dependency. Subnormals and NaN are handled because an
 * embedder that emits one should not silently become zero.
 */
static uint16_t
f32_to_f16(float f)
{
	uint32_t x;
	uint32_t sign, exp, man;

	memcpy(&x, &f, sizeof x);
	sign = (x >> 16) & 0x8000u;
	exp = (x >> 23) & 0xFFu;
	man = x & 0x7FFFFFu;

	if (exp == 0xFF)                       /* inf or NaN */
		return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0u));
	if (exp > 0x70 + 0x1E)                 /* overflows binary16 */
		return (uint16_t)(sign | 0x7C00u);
	if (exp < 0x71) {                      /* subnormal or zero */
		if (exp < 0x67)
			return (uint16_t)sign;
		man |= 0x800000u;
		return (uint16_t)(sign |
		    (uint16_t)(man >> (0x7E - exp + 1)));
	}
	return (uint16_t)(sign | (uint16_t)((exp - 0x70) << 10) |
	                  (uint16_t)(man >> 13));
}

static float
f16_to_f32(uint16_t h)
{
	uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
	uint32_t exp = (h >> 10) & 0x1Fu;
	uint32_t man = h & 0x3FFu;
	uint32_t x;
	float f;

	if (exp == 0) {
		if (man == 0) {
			x = sign;
		} else {                       /* subnormal */
			exp = 0x71;
			while ((man & 0x400u) == 0) {
				man <<= 1;
				exp--;
			}
			man &= 0x3FFu;
			x = sign | (exp << 23) | (man << 13);
		}
	} else if (exp == 0x1F) {
		x = sign | 0x7F800000u | (man << 13);
	} else {
		x = sign | ((exp + 0x70) << 23) | (man << 13);
	}
	memcpy(&f, &x, sizeof f);
	return f;
}

/* --- base64, RFC 4648 -------------------------------------------------- */

static const char B64[] =
	"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static size_t
b64_encode(const uint8_t *in, size_t n, char *out)
{
	size_t i, o = 0;

	for (i = 0; i + 2 < n; i += 3) {
		uint32_t v = ((uint32_t)in[i] << 16) |
		             ((uint32_t)in[i + 1] << 8) | in[i + 2];

		out[o++] = B64[(v >> 18) & 63];
		out[o++] = B64[(v >> 12) & 63];
		out[o++] = B64[(v >> 6) & 63];
		out[o++] = B64[v & 63];
	}
	if (i < n) {
		uint32_t v = (uint32_t)in[i] << 16;
		int rem = (int)(n - i);

		if (rem == 2)
			v |= (uint32_t)in[i + 1] << 8;
		out[o++] = B64[(v >> 18) & 63];
		out[o++] = B64[(v >> 12) & 63];
		out[o++] = rem == 2 ? B64[(v >> 6) & 63] : '=';
		out[o++] = '=';
	}
	out[o] = '\0';
	return o;
}

static int
b64_val(char c)
{
	if (c >= 'A' && c <= 'Z') return c - 'A';
	if (c >= 'a' && c <= 'z') return c - 'a' + 26;
	if (c >= '0' && c <= '9') return c - '0' + 52;
	if (c == '+') return 62;
	if (c == '/') return 63;
	return -1;
}

static size_t
b64_decode(const char *in, size_t len, uint8_t *out, size_t max)
{
	size_t i, o = 0;
	int q[4], k = 0;

	for (i = 0; i < len; i++) {
		int v = b64_val(in[i]);

		if (v < 0)
			continue;              /* '=' and any stray byte */
		q[k++] = v;
		if (k == 4) {
			uint32_t t = ((uint32_t)q[0] << 18) |
			             ((uint32_t)q[1] << 12) |
			             ((uint32_t)q[2] << 6) | (uint32_t)q[3];

			if (o + 3 > max)
				return o;
			out[o++] = (uint8_t)(t >> 16);
			out[o++] = (uint8_t)(t >> 8);
			out[o++] = (uint8_t)t;
			k = 0;
		}
	}
	if (k == 3) {
		uint32_t t = ((uint32_t)q[0] << 18) | ((uint32_t)q[1] << 12) |
		             ((uint32_t)q[2] << 6);

		if (o + 2 <= max) {
			out[o++] = (uint8_t)(t >> 16);
			out[o++] = (uint8_t)(t >> 8);
		}
	} else if (k == 2) {
		uint32_t t = ((uint32_t)q[0] << 18) | ((uint32_t)q[1] << 12);

		if (o + 1 <= max)
			out[o++] = (uint8_t)(t >> 16);
	}
	return o;
}

/* --- hex --------------------------------------------------------------- */

static void
hex_encode(const uint8_t *in, size_t n, char *out)
{
	static const char H[] = "0123456789abcdef";
	size_t i;

	for (i = 0; i < n; i++) {
		out[i * 2] = H[in[i] >> 4];
		out[i * 2 + 1] = H[in[i] & 15];
	}
	out[n * 2] = '\0';
}

static int
hex_decode(const char *in, uint8_t *out, size_t n)
{
	size_t i;

	for (i = 0; i < n; i++) {
		int hi = b64_val('A');   /* silence unused warnings on some cc */
		char a = in[i * 2], b = in[i * 2 + 1];

		(void)hi;
		if (a >= '0' && a <= '9') hi = a - '0';
		else if (a >= 'a' && a <= 'f') hi = a - 'a' + 10;
		else if (a >= 'A' && a <= 'F') hi = a - 'A' + 10;
		else return -1;

		int lo;
		if (b >= '0' && b <= '9') lo = b - '0';
		else if (b >= 'a' && b <= 'f') lo = b - 'a' + 10;
		else if (b >= 'A' && b <= 'F') lo = b - 'A' + 10;
		else return -1;

		out[i] = (uint8_t)((hi << 4) | lo);
	}
	return 0;
}

/* --- the table --------------------------------------------------------- */

static size_t
bucket_of(const uint8_t hash[HASHLEN], size_t cap)
{
	uint64_t k;

	memcpy(&k, hash, sizeof k);
	return (size_t)(k & (uint64_t)(cap - 1));
}

static int grow(sigil_veccache_t *c);

static int
insert(sigil_veccache_t *c, const uint8_t hash[HASHLEN], uint16_t *vec)
{
	size_t i;

	if ((c->n + 1) * 4 >= c->cap * 3 && grow(c) != 0)
		return -1;

	i = bucket_of(hash, c->cap);
	while (c->tab[i].used) {
		if (memcmp(c->tab[i].hash, hash, HASHLEN) == 0) {
			/* Already present: keep the first. A duplicate line
			 * is what an interrupted run leaves behind. */
			free(vec);
			return 0;
		}
		i = (i + 1) & (c->cap - 1);
	}
	memcpy(c->tab[i].hash, hash, HASHLEN);
	c->tab[i].vec = vec;
	c->tab[i].used = 1;
	c->n++;
	return 0;
}

static int
grow(sigil_veccache_t *c)
{
	struct entry *old = c->tab;
	size_t oldcap = c->cap, i;

	c->cap = oldcap ? oldcap * 2 : 1024;
	c->tab = calloc(c->cap, sizeof *c->tab);
	if (c->tab == NULL) {
		c->tab = old;
		c->cap = oldcap;
		return -1;
	}
	c->n = 0;
	for (i = 0; i < oldcap; i++) {
		if (!old[i].used)
			continue;
		size_t j = bucket_of(old[i].hash, c->cap);

		while (c->tab[j].used)
			j = (j + 1) & (c->cap - 1);
		c->tab[j] = old[i];
		c->n++;
	}
	free(old);
	return 0;
}

/* --- loading ----------------------------------------------------------- */

/*
 * Pull one JSON string field. Deliberately not a JSON parser: the format
 * is written by this file and every line has the same shape, so scanning
 * for "key":" and taking to the next quote is sufficient and cannot be
 * fooled by input we did not write. A line that does not match is
 * skipped, which is also how a truncated final line is handled.
 */
static const char *
field(const char *line, const char *key, size_t *len)
{
	const char *p = strstr(line, key);
	const char *q;

	if (p == NULL)
		return NULL;
	p += strlen(key);
	q = strchr(p, '"');
	if (q == NULL)
		return NULL;
	*len = (size_t)(q - p);
	return p;
}

static void
load(sigil_veccache_t *c, const char *path)
{
	FILE *fp = fopen(path, "r");
	char *line = NULL;
	size_t cap = 0;
	ssize_t n;

	if (fp == NULL)
		return;

	while ((n = getline(&line, &cap, fp)) > 0) {
		const char *h, *m, *v;
		size_t hl = 0, ml = 0, vl = 0;
		uint8_t hash[HASHLEN];
		uint16_t *vec;

		h = field(line, "\"h\":\"", &hl);
		m = field(line, "\"m\":\"", &ml);
		v = field(line, "\"v\":\"", &vl);
		if (h == NULL || m == NULL || v == NULL)
			continue;
		if (hl != HASHLEN * 2)
			continue;
		/* Another model's vectors live happily in the same file. */
		if (ml != strlen(c->model) ||
		    strncmp(m, c->model, ml) != 0)
			continue;
		if (hex_decode(h, hash, HASHLEN) != 0)
			continue;

		vec = malloc(c->dim * sizeof *vec);
		if (vec == NULL)
			break;
		if (b64_decode(v, vl, (uint8_t *)vec,
		               c->dim * sizeof *vec) != c->dim * sizeof *vec) {
			free(vec);
			continue;              /* truncated line */
		}
		if (insert(c, hash, vec) != 0) {
			free(vec);
			break;
		}
	}
	free(line);
	fclose(fp);
}

/* --- public ------------------------------------------------------------ */

sigil_veccache_t *
sigil_veccache_open(const char *path, const char *model, size_t dim)
{
	sigil_veccache_t *c;

	if (path == NULL || model == NULL || dim == 0)
		return NULL;

	c = calloc(1, sizeof *c);
	if (c == NULL)
		return NULL;
	c->model = strdup(model);
	c->dim = dim;
	if (c->model == NULL || grow(c) != 0) {
		free(c->model);
		free(c);
		return NULL;
	}

	load(c, path);

	c->fp = fopen(path, "a");
	if (c->fp == NULL) {
		sigil_veccache_close(c);
		return NULL;
	}
	return c;
}

int
sigil_veccache_get(sigil_veccache_t *c, const uint8_t hash[HASHLEN],
                   float *out)
{
	size_t i, probes = 0;

	if (c == NULL || out == NULL)
		return -1;

	i = bucket_of(hash, c->cap);
	while (c->tab[i].used && probes++ < c->cap) {
		if (memcmp(c->tab[i].hash, hash, HASHLEN) == 0) {
			size_t d;

			for (d = 0; d < c->dim; d++)
				out[d] = f16_to_f32(c->tab[i].vec[d]);
			c->hits++;
			return 0;
		}
		i = (i + 1) & (c->cap - 1);
	}
	c->misses++;
	return -1;
}

int
sigil_veccache_put(sigil_veccache_t *c, const uint8_t hash[HASHLEN],
                   const float *v)
{
	uint16_t *vec;
	char hex[HASHLEN * 2 + 1];
	char *b64;
	size_t d, nbytes;

	if (c == NULL || v == NULL)
		return -1;

	vec = malloc(c->dim * sizeof *vec);
	if (vec == NULL)
		return -1;
	for (d = 0; d < c->dim; d++)
		vec[d] = f32_to_f16(v[d]);

	nbytes = c->dim * sizeof *vec;
	b64 = malloc((nbytes + 2) / 3 * 4 + 1);
	if (b64 == NULL) {
		free(vec);
		return -1;
	}
	b64_encode((const uint8_t *)vec, nbytes, b64);
	hex_encode(hash, HASHLEN, hex);

	if (fprintf(c->fp, "{\"h\":\"%s\",\"m\":\"%s\",\"d\":%zu,\"v\":\"%s\"}\n",
	            hex, c->model, c->dim, b64) < 0) {
		free(b64);
		free(vec);
		return -1;
	}
	free(b64);

	/* insert() takes ownership of vec, or frees it on a duplicate. */
	return insert(c, hash, vec);
}

int
sigil_veccache_sync(sigil_veccache_t *c, int fsync_too)
{
	if (c == NULL || c->fp == NULL)
		return -1;
	if (fflush(c->fp) != 0)
		return -1;
	if (fsync_too && fsync(fileno(c->fp)) != 0)
		return -1;
	return 0;
}

size_t
sigil_veccache_count(const sigil_veccache_t *c)
{
	return c ? c->n : 0;
}

void
sigil_veccache_stats(const sigil_veccache_t *c, uint64_t *hits,
                     uint64_t *misses)
{
	if (c == NULL)
		return;
	if (hits != NULL)
		*hits = c->hits;
	if (misses != NULL)
		*misses = c->misses;
}

void
sigil_veccache_close(sigil_veccache_t *c)
{
	size_t i;

	if (c == NULL)
		return;
	if (c->fp != NULL) {
		fflush(c->fp);
		fclose(c->fp);
	}
	for (i = 0; i < c->cap; i++)
		if (c->tab[i].used)
			free(c->tab[i].vec);
	free(c->tab);
	free(c->model);
	free(c);
}
