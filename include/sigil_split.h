/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Paragraph splitting: the one definition.
 *
 * This is the unit of identity for the whole system. A record's BLAKE3 is
 * computed over exactly the span this produces, so anything that counts,
 * hashes, or resumes has to agree with it byte for byte.
 *
 * It exists because three copies had drifted. cmd/index.c held the real
 * one, tools/pipeline.py re-implemented it in Python for the manifest's
 * paragraph count, and test/fuzz_sigil.c held a third with a comment
 * admitting it could drift. It had: the manifest reported 77,367,817
 * paragraphs for a corpus the indexer split into 74,905,358 -- a silent
 * 3.2% disagreement that nothing checked, because the check was a comment
 * saying it "shows up as a mismatch against /stats" rather than a test.
 */

#ifndef SIGIL_SPLIT_H
#define SIGIL_SPLIT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Shorter than Minpara embeds to noise; longer than Maxpara exceeds what
 * the model can attend to, so it is cut at a sentence boundary.
 */
enum {
	SIGIL_MINPARA = 40,
	SIGIL_MAXPARA = 4000,
};

/*
 * One chunk as the embedder will see it: a byte range in the buffer, and
 * the 1-based paragraph number the record will carry.
 */
typedef struct sigil_chunk {
	size_t off;      /* offset from the start of the buffer */
	size_t len;
	unsigned para;
} sigil_chunk_t;

/*
 * Split buf into chunks, calling fn for each. Returns the number emitted.
 *
 * A callback rather than an array because a 4 MB book yields thousands of
 * chunks and the caller usually wants to consume them one at a time --
 * cmd/index.c adds each to the store, tools/ counts them, the fuzzer
 * hashes them. Passing NULL for fn counts without doing anything else.
 */
size_t sigil_split(const char *buf, size_t n,
                   void (*fn)(const sigil_chunk_t *, void *), void *arg);

/* Convenience: how many chunks buf yields. Same as sigil_split(.., NULL, ..). */
size_t sigil_split_count(const char *buf, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* SIGIL_SPLIT_H */
