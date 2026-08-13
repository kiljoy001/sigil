/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * A durable cache of embedding vectors, keyed by content and model.
 *
 * Embedding is the expensive step by four orders of magnitude: 13 hours of
 * GPU for the Gutenberg corpus against a 12 ms scan. That work was lost
 * entirely when a commit failed after the fact, because nothing persisted
 * between the embedder and the store -- sigil_t carries the 128-bit LSH,
 * not the 384 floats it was projected from, so a re-run has nothing to
 * reuse and must embed again from scratch.
 *
 * This closes that. Vectors are appended as they are produced, so a crash
 * costs whatever was in flight rather than everything. A re-run consults
 * the cache first and embeds only what is missing.
 *
 * Two properties fall out of keying by content hash rather than position:
 *
 *   * the same paragraph appearing in two documents embeds once;
 *   * changing the SimHash seed or LSH width becomes a re-projection over
 *     cached vectors -- minutes -- rather than a re-embed.
 *
 * The model id is part of the key because a different model produces
 * different vectors for identical text, and serving MiniLM vectors to a
 * caller that has switched models would be silent corruption.
 *
 * Format: JSON Lines, one record per line,
 *
 *     {"h":"<hex blake3>","m":"<model>","d":384,"v":"<base64 float16>"}
 *
 * Line-oriented so it appends without rewriting, survives truncation (a
 * partial final line is skipped), and can be read with head and grep.
 * float16 rather than float32 because the vector is consumed by a SimHash
 * projection whose output is a sign bit -- the precision does not survive
 * the projection anyway, and it halves 115 GB to 58 GB at Gutenberg scale.
 * JSON with the vector as text would be 4x larger again and slower to
 * parse than re-embedding.
 */

#ifndef SIGIL_VECCACHE_H
#define SIGIL_VECCACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sigil_veccache sigil_veccache_t;

/*
 * Open (creating if absent) the cache at path, for vectors of `dim`
 * floats produced by `model`. Existing entries for other models or
 * dimensions are ignored rather than an error: one file can hold the
 * output of several models, and a caller only ever sees its own.
 *
 * Returns NULL on failure.
 */
sigil_veccache_t *sigil_veccache_open(const char *path, const char *model,
                                      size_t dim);

/*
 * Look up the vector for a record's 32-byte BLAKE3. Writes `dim` floats
 * to out and returns 0 on a hit, -1 on a miss.
 */
int sigil_veccache_get(sigil_veccache_t *c, const uint8_t hash[32],
                       float *out);

/*
 * Append a vector. Returns 0 on success, -1 on failure.
 *
 * Durability is per-batch rather than per-record: the caller decides when
 * to sigil_veccache_sync(), because fsync on every one of 75M records
 * would dominate the run.
 */
int sigil_veccache_put(sigil_veccache_t *c, const uint8_t hash[32],
                       const float *v);

/* Flush buffered writes to the OS, and optionally to the platter. */
int sigil_veccache_sync(sigil_veccache_t *c, int fsync_too);

/* Entries currently loaded, for reporting. */
size_t sigil_veccache_count(const sigil_veccache_t *c);

/* Hits and misses since open, for reporting how much a run reused. */
void sigil_veccache_stats(const sigil_veccache_t *c,
                          uint64_t *hits, uint64_t *misses);

void sigil_veccache_close(sigil_veccache_t *c);

#ifdef __cplusplus
}
#endif

#endif /* SIGIL_VECCACHE_H */
