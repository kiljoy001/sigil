/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * UTF-8 repair for text on its way to a tokenizer.
 *
 * See src/utf8_repair.c for why this exists: invalid UTF-8 reaching PCRE2
 * inside openvino_tokenizers is undefined behaviour and reads wild memory,
 * which segfaulted the indexer intermittently for a week.
 */

#ifndef SIGIL_UTF8_H
#define SIGIL_UTF8_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Length of the valid UTF-8 sequence at p (1-4), or 0 if the bytes there are
 * not a valid sequence. Overlong forms, surrogate halves and codepoints above
 * U+10FFFF all count as invalid.
 */
size_t sigil_utf8_seq(const unsigned char *p, size_t left);

/* Whether the whole buffer is valid UTF-8. */
int sigil_utf8_valid(const char *text, size_t len);

/*
 * Repaired copy of text as valid UTF-8, NUL-terminated, or NULL if out of
 * memory. Bytes that are not valid UTF-8 are transcoded as CP1252 where that
 * byte is assigned and Latin-1 otherwise, so 0x92 becomes U+2019 rather than
 * a replacement character.
 *
 * Caller frees. If outlen is not NULL it receives the length excluding the
 * terminator; the result may contain embedded NULs only if the input did.
 */
char *sigil_utf8_repair(const char *text, size_t len, size_t *outlen);

#ifdef __cplusplus
}
#endif

#endif /* SIGIL_UTF8_H */
