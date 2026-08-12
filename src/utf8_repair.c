/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Coerce arbitrary bytes to valid UTF-8.
 *
 * This is not cosmetic. openvino_tokenizers' SpecialTokensSplit hands text
 * to PCRE2 compiled in UTF-8 mode without PCRE2_MATCH_INVALID_UTF, and PCRE2
 * documents matching invalid UTF as undefined behaviour. In practice the
 * sljit-compiled matcher decodes a garbage codepoint from a stray byte and
 * indexes a character-class table with it: a wild read that SIGSEGVs when the
 * address lands on an unmapped page and returns silent garbage otherwise.
 * Because that depends on mmap layout, identical input crashed 8 runs in 12
 * with ASLR on and 8 in 8 with it off -- an apparent race that never was one.
 * Full account in docs/FINDINGS.md.
 *
 * The corpus pipeline (tools/clean.py) fixes the files themselves, which is
 * the right place for it. This exists because sigilfs indexes whatever tree
 * it is pointed at: one malformed byte from an uncleaned source must not be
 * able to segfault the server through a third-party regex engine.
 *
 * The salvage is CP1252-aware rather than lossy. A byte that is not part of
 * a valid UTF-8 sequence is transcoded as CP1252 where that byte is assigned
 * and Latin-1 otherwise, so 0x92 becomes a real U+2019 the embedding can use
 * instead of U+FFFD noise. Project Gutenberg is full of exactly that byte.
 *
 * Identity is unaffected: sigil hashes the original bytes with BLAKE3. Only
 * the embedder sees the repaired copy.
 *
 * Plain C in its own translation unit, not C++ inside the OpenVINO backend,
 * for two reasons: it is pure text processing with no OpenVINO dependency,
 * and keeping it here means it compiles (and is tested) on machines with no
 * OpenVINO at all -- including CI. tools/tests/test_utf8_differential.py
 * checks this against the Python implementation over generated input, so the
 * two cannot drift apart silently.
 */

#include "sigil_utf8.h"

#include <stdlib.h>
#include <string.h>

/*
 * CP1252's C1 range (0x80-0x9F), where it differs from Latin-1. Five bytes
 * are unassigned there; 0 means "fall back to Latin-1", which keeps the
 * output valid rather than inventing a character.
 */
static const unsigned short cp1252_c1[32] = {
	0x20AC, 0,      0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021,
	0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0,      0x017D, 0,
	0,      0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
	0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0,      0x017E, 0x0178,
};

/*
 * Length of the valid UTF-8 sequence at p, or 0 if invalid. Rejects overlong
 * forms, surrogates and anything above U+10FFFF -- those are exactly as
 * undefined for PCRE2 as a lone continuation byte, so accepting them here
 * would defeat the purpose.
 */
size_t
sigil_utf8_seq(const unsigned char *p, size_t left)
{
	if (left == 0)
		return 0;
	if (p[0] < 0x80)
		return 1;
	if (p[0] < 0xC2)                       /* continuation, or overlong */
		return 0;
	if (p[0] < 0xE0)
		return (left >= 2 && (p[1] & 0xC0) == 0x80) ? 2 : 0;
	if (p[0] < 0xF0) {
		if (left < 3 || (p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80)
			return 0;
		if (p[0] == 0xE0 && p[1] < 0xA0)               /* overlong */
			return 0;
		if (p[0] == 0xED && p[1] >= 0xA0)              /* surrogate */
			return 0;
		return 3;
	}
	if (p[0] < 0xF5) {
		if (left < 4 || (p[1] & 0xC0) != 0x80 ||
		    (p[2] & 0xC0) != 0x80 || (p[3] & 0xC0) != 0x80)
			return 0;
		if (p[0] == 0xF0 && p[1] < 0x90)               /* overlong */
			return 0;
		if (p[0] == 0xF4 && p[1] >= 0x90)              /* > U+10FFFF */
			return 0;
		return 4;
	}
	return 0;
}

static char *
put_cp(char *o, unsigned int cp)
{
	if (cp < 0x80) {
		*o++ = (char)cp;
	} else if (cp < 0x800) {
		*o++ = (char)(0xC0 | (cp >> 6));
		*o++ = (char)(0x80 | (cp & 0x3F));
	} else {
		*o++ = (char)(0xE0 | (cp >> 12));
		*o++ = (char)(0x80 | ((cp >> 6) & 0x3F));
		*o++ = (char)(0x80 | (cp & 0x3F));
	}
	return o;
}

int
sigil_utf8_valid(const char *text, size_t len)
{
	const unsigned char *p = (const unsigned char *)text;
	size_t i = 0;

	while (i < len) {
		size_t n = sigil_utf8_seq(p + i, len - i);

		if (n == 0)
			return 0;
		i += n;
	}
	return 1;
}

char *
sigil_utf8_repair(const char *text, size_t len, size_t *outlen)
{
	const unsigned char *p = (const unsigned char *)text;
	char *out, *o;
	size_t i;

	/* Worst case is every byte becoming a 3-byte sequence. Allocating for
	 * that up front avoids a growth check in the loop; the buffer is
	 * short-lived and the caller frees it. */
	out = malloc(len * 3 + 1);
	if (out == NULL)
		return NULL;

	o = out;
	i = 0;
	while (i < len) {
		size_t n = sigil_utf8_seq(p + i, len - i);

		if (n != 0) {
			memcpy(o, text + i, n);
			o += n;
			i += n;
			continue;
		}
		/* Not valid UTF-8: CP1252 where assigned, Latin-1 otherwise.
		 * Latin-1 is total over 0x00-0xFF, so this cannot fail --
		 * which is the property that matters when input is arbitrary. */
		{
			unsigned char b = p[i++];
			unsigned int cp = b;

			if (b >= 0x80 && b <= 0x9F && cp1252_c1[b - 0x80] != 0)
				cp = cp1252_c1[b - 0x80];
			o = put_cp(o, cp);
		}
	}
	*o = '\0';
	if (outlen != NULL)
		*outlen = (size_t)(o - out);
	return out;
}
