/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * libFuzzer harness for the paths that eat untrusted bytes.
 *
 * Everything a corpus file contains reaches sigil_generate_para() unchanged,
 * and the paragraph splitter decides where one paragraph ends and the next
 * begins. Both run over 59.6 million paragraphs of Project Gutenberg, which
 * includes files with CRLF endings, mixed encodings, control characters, and
 * blocks that are megabytes long -- none of which a hand-written test case
 * looks like.
 *
 *	make fuzz
 *	test/fuzz_sigil -max_total_time=300 test/fuzz-corpus
 *
 * Build with -fsanitize=fuzzer,address,undefined so an over-read is a located
 * crash rather than silent corruption. This is the same reasoning that
 * applies to the .tab parser in libtab's own fuzz harness.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sigil.h"

enum {
	Minpara = 40,      /* matches cmd/index.c */
	Maxpara = 4000,
};

/*
 * The paragraph splitter from cmd/index.c, reproduced here because that file
 * is plan9port C and cannot be linked against a libFuzzer harness. Keeping a
 * copy is a real cost -- it can drift -- but the alternative is leaving the
 * one parser that touches arbitrary bytes untested.
 */
static void
split_and_hash(const char *buf, size_t n)
{
	const char *p, *end, *q, *cut;
	uint32_t para = 1;
	size_t len;
	sigil_t s;

	end = buf + n;
	for (p = buf; p < end; ) {
		while (p < end && (*p == '\n' || *p == '\r' ||
		                   *p == ' ' || *p == '\t'))
			p++;
		if (p >= end)
			break;
		for (q = p; q < end - 1; q++)
			if (q[0] == '\n' && (q[1] == '\n' || q[1] == '\r'))
				break;
		if (q >= end - 1)
			q = end;
		len = (size_t)(q - p);

		while (len > Maxpara) {
			cut = p + Maxpara;
			while (cut > p + Maxpara / 2 && *cut != '.' &&
			       *cut != '\n')
				cut--;
			if (cut <= p + Maxpara / 2)
				cut = p + Maxpara;
			if ((size_t)(cut - p) >= Minpara)
				sigil_generate_para(p, (size_t)(cut - p),
				                    para++, 0, 0, NULL, &s);
			len -= (size_t)(cut - p);
			p = cut;
		}
		if (len >= Minpara)
			sigil_generate_para(p, len, para++, 0, 0, NULL, &s);
		p = q + 1;
	}
}

int
LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
	sigil_t s;
	sigil_store_t st;
	uint32_t out[64];
	char hex[SIGIL_SIZE * 2 + 1];

	if (size == 0)
		return 0;

	/* 1. The hash path over arbitrary bytes, including embedded NULs. */
	sigil_generate_para((const char *)data, size, 1, 0, 0, NULL, &s);
	sigil_to_hex(&s, hex);

	/* 2. The splitter, which decides paragraph boundaries. */
	split_and_hash((const char *)data, size);

	/* 3. Trit decode: the first two bytes as a packed value. All 729
	 *    legal values must round-trip and every other 16-bit value must
	 *    be rejected -- a decoder that accepts corruption is worse than
	 *    one that is slow. */
	if (size >= 2) {
		uint16_t packed = (uint16_t)((data[0] << 8) | data[1]);
		sigil_trits_t t;

		if (sigil_trits_unpack(packed, &t) == 0) {
			if (sigil_trits_pack(&t) != packed)
				abort();     /* round-trip broken */
			if (!sigil_trits_valid(packed))
				abort();     /* valid() disagrees */
		} else if (sigil_trits_valid(packed)) {
			abort();             /* valid() disagrees */
		}
	}

	/* 4. A scan over a store built from the input, with the radius taken
	 *    from the data so the whole range including 0 and > width is
	 *    exercised. */
	if (sigil_store_init(&st, 2) == 0) {
		size_t i, chunk = size / 8 + 1;

		for (i = 0; i + chunk <= size && st.count < 64; i += chunk) {
			sigil_generate_para((const char *)data + i, chunk,
			                    (uint32_t)i, 0, 0, NULL, &s);
			if (sigil_store_push(&st, &s) < 0)
				break;
		}
		if (st.count > 0) {
			uint32_t radius = data[size - 1];
			size_t n;

			n = sigil_scan_similar_scalar(&st, s.lsh, radius,
			                              out, 64);
			/* Anything returned must really be within the radius:
			 * a scan that over-reports is how a neighbourhood
			 * fills with unrelated text. */
			for (i = 0; i < n; i++) {
				sigil_t got;

				if (sigil_store_get(&st, out[i], &got) != 0)
					abort();
				if (sigil_hamming(got.lsh, s.lsh) > radius)
					abort();
			}
		}
		sigil_store_free(&st);
	}
	return 0;
}
