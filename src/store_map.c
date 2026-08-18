/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * File-backed stores.
 *
 * Opening a store here costs no parsing, no allocation and no copy: the SoA
 * field arrays are pointers into a mapping of the file, so the "load" is
 * arithmetic on a base address. The kernel pages in only what a scan
 * touches, and evicts under pressure by dropping clean pages it can re-read
 * — so a store larger than RAM works, where the heap path would die.
 *
 * That is the second half of the same fix as the streaming commit. That one
 * stopped materialising the whole table to *write* it; this stops
 * materialising it to *read* it.
 *
 * Only possible because segments never move. Under the flat array every
 * doubling reallocated and invalidated every address, so no pointer into the
 * store could have been a mapping.
 *
 * On format. The layout is the in-memory layout, which makes it fast and
 * makes it brittle: it bakes in endianness, SIGIL_SEG_RECS, SIGIL_LSH_BITS
 * and SIGIL_HASH_LEN. The header records them and sigil_store_map() refuses
 * a mismatch, because reinterpreting bytes written under other rules yields
 * plausible wrong records rather than an error. This file is a derived cache
 * — libtab stays the durable, portable record — so a layout change means
 * delete and rebuild, never migrate.
 */

/* madvise() and MADV_* are not in strict POSIX -- _POSIX_C_SOURCE alone
 * hides them and leaves only posix_madvise, whose POSIX_MADV_DONTNEED is
 * explicitly allowed to do nothing. The whole point here is that DONTNEED
 * actually drops the pages, so the GNU extension is the one we want. */
#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200112L

#include "sigil.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define SIGIL_MAP_MAGIC   0x4c474953554d4d41ULL /* "AMMUSIGL" */
#define SIGIL_MAP_VERSION 1

/*
 * Fixed-width and explicitly laid out. Every field that decides how the
 * bytes after it are interpreted is recorded, so a mismatch is detectable
 * rather than silently misread.
 */
typedef struct {
	uint64_t magic;
	uint32_t version;
	uint32_t lsh_bits;
	uint32_t hash_len;
	uint32_t seg_shift;
	uint64_t count;
	uint64_t nseg;
	uint64_t field_off[7];   /* byte offset of each field's data */
	uint64_t reserved[9];    /* zero; keeps the header one 4K page */
} sigil_map_header;

/* Field order, fixed by the file format. */
enum { F_LSH, F_PARA, F_CLUSTER, F_TIMESTAMP, F_CATEGORY, F_TRITS, F_HASH };

static const size_t field_width[7] = {
	SIGIL_LSH_WORDS * sizeof(uint64_t),  /* lsh       */
	sizeof(uint32_t),                    /* para      */
	sizeof(uint32_t),                    /* cluster   */
	sizeof(uint32_t),                    /* timestamp */
	sizeof(uint16_t),                    /* category  */
	sizeof(uint16_t),                    /* trits     */
	SIGIL_HASH_LEN                       /* hash      */
};

/* Segments are page-aligned in the file so each one starts on a page
 * boundary: madvise() operates on whole pages, and a segment sharing a page
 * with its neighbour could not be released independently. */
static size_t page_round(size_t n)
{
	size_t pg = (size_t)sysconf(_SC_PAGESIZE);

	return (n + pg - 1) & ~(pg - 1);
}

static size_t seg_bytes(int f)
{
	return page_round(SIGIL_SEG_RECS * field_width[f]);
}

static const void *seg_ptr(const sigil_store_t *st, int f, size_t g)
{
	switch (f) {
	case F_LSH:       return st->lsh[g];
	case F_PARA:      return st->para[g];
	case F_CLUSTER:   return st->cluster[g];
	case F_TIMESTAMP: return st->timestamp[g];
	case F_CATEGORY:  return st->category[g];
	case F_TRITS:     return st->trits[g];
	default:          return st->hash[g];
	}
}

int sigil_store_save(const sigil_store_t *st, const char *path)
{
	sigil_map_header h;
	char *tmp;
	size_t off, f, g;
	int fd, rc = -1;

	/* Write to a temporary and rename, so a crash mid-write leaves the
	 * previous store intact rather than half a new one -- the same
	 * discipline the libtab commit uses. */
	tmp = malloc(strlen(path) + 8);
	if (tmp == NULL)
		return -1;
	sprintf(tmp, "%s.tmp", path);

	fd = open(tmp, O_RDWR | O_CREAT | O_TRUNC, 0644);
	if (fd < 0) {
		free(tmp);
		return -1;
	}

	memset(&h, 0, sizeof h);
	h.magic     = SIGIL_MAP_MAGIC;
	h.version   = SIGIL_MAP_VERSION;
	h.lsh_bits  = SIGIL_LSH_BITS;
	h.hash_len  = SIGIL_HASH_LEN;
	h.seg_shift = SIGIL_SEG_SHIFT;
	h.count     = st->count;
	h.nseg      = st->nseg;

	off = page_round(sizeof h);
	for (f = 0; f < 7; f++) {
		h.field_off[f] = off;
		off += seg_bytes((int)f) * st->nseg;
	}

	if (write(fd, &h, sizeof h) != (ssize_t)sizeof h)
		goto out;

	for (f = 0; f < 7; f++) {
		if (lseek(fd, (off_t)h.field_off[f], SEEK_SET) < 0)
			goto out;
		for (g = 0; g < st->nseg; g++) {
			size_t want = SIGIL_SEG_RECS * field_width[f];

			if (lseek(fd, (off_t)(h.field_off[f] +
			                      seg_bytes((int)f) * g),
			          SEEK_SET) < 0)
				goto out;
			if (write(fd, seg_ptr(st, (int)f, g), want)
			    != (ssize_t)want)
				goto out;
		}
	}

	/* Extend to the full mapped length so a reader never faults past the
	 * end of the file -- that raises SIGBUS, not an error return. */
	if (ftruncate(fd, (off_t)off) != 0)
		goto out;
	if (fsync(fd) != 0)
		goto out;
	if (close(fd) != 0) {
		fd = -1;
		goto out;
	}
	fd = -1;
	if (rename(tmp, path) != 0)
		goto out;
	rc = 0;
out:
	if (fd >= 0)
		close(fd);
	if (rc != 0)
		unlink(tmp);
	free(tmp);
	return rc;
}

/* Point the seven directories at the mapping. */
static int wire_dirs(sigil_store_t *st, const sigil_map_header *h, char *base)
{
	size_t g;

	st->lsh       = calloc(h->nseg, sizeof *st->lsh);
	st->para      = calloc(h->nseg, sizeof *st->para);
	st->cluster   = calloc(h->nseg, sizeof *st->cluster);
	st->timestamp = calloc(h->nseg, sizeof *st->timestamp);
	st->category  = calloc(h->nseg, sizeof *st->category);
	st->trits     = calloc(h->nseg, sizeof *st->trits);
	st->hash      = calloc(h->nseg, sizeof *st->hash);

	if (!st->lsh || !st->para || !st->cluster || !st->timestamp ||
	    !st->category || !st->trits || !st->hash)
		return -1;

	for (g = 0; g < h->nseg; g++) {
		st->lsh[g]       = (uint64_t *)(base + h->field_off[F_LSH]
		                                + seg_bytes(F_LSH) * g);
		st->para[g]      = (uint32_t *)(base + h->field_off[F_PARA]
		                                + seg_bytes(F_PARA) * g);
		st->cluster[g]   = (uint32_t *)(base + h->field_off[F_CLUSTER]
		                                + seg_bytes(F_CLUSTER) * g);
		st->timestamp[g] = (uint32_t *)(base + h->field_off[F_TIMESTAMP]
		                                + seg_bytes(F_TIMESTAMP) * g);
		st->category[g]  = (uint16_t *)(base + h->field_off[F_CATEGORY]
		                                + seg_bytes(F_CATEGORY) * g);
		st->trits[g]     = (uint16_t *)(base + h->field_off[F_TRITS]
		                                + seg_bytes(F_TRITS) * g);
		st->hash[g]      = (uint8_t *)(base + h->field_off[F_HASH]
		                               + seg_bytes(F_HASH) * g);
	}
	return 0;
}

int sigil_store_map(sigil_store_t *st, const char *path)
{
	sigil_map_header h;
	struct stat sb;
	char *base;
	int fd;

	memset(st, 0, sizeof *st);

	fd = open(path, O_RDONLY);
	if (fd < 0)
		return -1;
	if (read(fd, &h, sizeof h) != (ssize_t)sizeof h) {
		close(fd);
		return -1;
	}

	/*
	 * Refuse rather than reinterpret. Bytes written under a different
	 * width or segment size are not corrupt -- they decode into valid,
	 * wrong records, which is the failure mode this whole project treats
	 * as worse than a crash.
	 */
	if (h.magic != SIGIL_MAP_MAGIC || h.version != SIGIL_MAP_VERSION ||
	    h.lsh_bits != SIGIL_LSH_BITS || h.hash_len != SIGIL_HASH_LEN ||
	    h.seg_shift != SIGIL_SEG_SHIFT) {
		close(fd);
		errno = EINVAL;
		return -1;
	}
	if (fstat(fd, &sb) != 0) {
		close(fd);
		return -1;
	}

	base = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
	close(fd);   /* the mapping keeps its own reference */
	if (base == MAP_FAILED)
		return -1;

	st->map    = base;
	st->maplen = (size_t)sb.st_size;
	st->count  = (size_t)h.count;
	st->nseg   = (size_t)h.nseg;
	st->segcap = (size_t)h.nseg;
	st->capacity = st->nseg * SIGIL_SEG_RECS;

	if (wire_dirs(st, &h, base) != 0) {
		sigil_store_unmap(st);
		return -1;
	}

	/* The scan walks every segment front to back. Saying so lets the
	 * kernel read ahead instead of faulting one page at a time. */
	madvise(base, (size_t)sb.st_size, MADV_SEQUENTIAL);
	return 0;
}

void sigil_store_unmap(sigil_store_t *st)
{
	/* The directories are heap even for a mapped store; only what they
	 * point at is the mapping. Freeing them is right, freeing the
	 * segments would not be. */
	free(st->lsh);
	free(st->para);
	free(st->cluster);
	free(st->timestamp);
	free(st->category);
	free(st->trits);
	free(st->hash);
	if (st->map != NULL)
		munmap(st->map, st->maplen);
	memset(st, 0, sizeof *st);
}

int sigil_store_release(const sigil_store_t *st, size_t g)
{
	int f;

	if (st->map == NULL || g >= st->nseg)
		return -1;

	/*
	 * A hint, not a free. The pages are clean and file-backed, so
	 * dropping them is always safe and the next touch re-reads. This is
	 * what lets a scan over a corpus larger than RAM keep its resident
	 * set bounded without the caller managing any buffers.
	 */
	for (f = 0; f < 7; f++) {
		void *p = (void *)seg_ptr(st, f, g);

		madvise(p, seg_bytes(f), MADV_DONTNEED);
	}
	return 0;
}
