/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Paragraph splitting, extracted verbatim from cmd/index.c's addparas().
 *
 * The logic is unchanged on purpose: index.c is what actually feeds the
 * embedder, so its behaviour is the definition and everything else was
 * wrong to approximate it. The four ways the Python re-implementation
 * differed, each small and compounding to 3.2% across the corpus:
 *
 *   * leading whitespace is skipped before the span is measured, not
 *     stripped from a block after the fact;
 *   * the chunk loop advances p and re-measures, so chunks after the
 *     first are bounded by what remains, not by ceiling division;
 *   * a chunk shorter than SIGIL_MINPARA is dropped, not counted;
 *   * a paragraph ends at "\n\n" or "\n\r", not "\n\n" alone.
 *
 * Plain C with no plan9port dependency so cmd/, tools/ (through ctypes)
 * and the fuzzer can all call the same code. src/utf8_repair.c was split
 * out for the same reason and is the precedent.
 */

#include "sigil_split.h"

size_t
sigil_split(const char *buf, size_t n,
            void (*fn)(const sigil_chunk_t *, void *), void *arg)
{
	const char *p, *end, *q, *cut;
	unsigned para = 1;
	size_t len, count = 0;

	if (buf == NULL || n == 0)
		return 0;

	end = buf + n;
	for (p = buf; p < end; ) {
		while (p < end && (*p == '\n' || *p == '\r' ||
		                   *p == ' ' || *p == '\t'))
			p++;
		if (p >= end)
			break;

		/* find the next blank line */
		for (q = p; q < end - 1; q++)
			if (q[0] == '\n' && (q[1] == '\n' || q[1] == '\r'))
				break;
		if (q >= end - 1)
			q = end;
		len = (size_t)(q - p);

		while (len > SIGIL_MAXPARA) {
			cut = p + SIGIL_MAXPARA;
			while (cut > p + SIGIL_MAXPARA / 2 &&
			       *cut != '.' && *cut != '\n')
				cut--;
			if (cut <= p + SIGIL_MAXPARA / 2)
				cut = p + SIGIL_MAXPARA;
			if ((size_t)(cut - p) >= SIGIL_MINPARA) {
				if (fn != NULL) {
					sigil_chunk_t c;

					c.off = (size_t)(p - buf);
					c.len = (size_t)(cut - p);
					c.para = para;
					fn(&c, arg);
				}
				para++;
				count++;
			}
			len -= (size_t)(cut - p);
			p = cut;
		}

		if (len >= SIGIL_MINPARA) {
			if (fn != NULL) {
				sigil_chunk_t c;

				c.off = (size_t)(p - buf);
				c.len = len;
				c.para = para;
				fn(&c, arg);
			}
			para++;
			count++;
		}
		p = q + 1;
	}
	return count;
}

size_t
sigil_split_count(const char *buf, size_t n)
{
	return sigil_split(buf, n, NULL, NULL);
}
