/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Stage-two sidecars.
 *
 * The stage-one sigil stays exactly as it is -- it is identity, and it is
 * what the scan reduces millions of records with. A sidecar adds a second
 * code for the subset that survived that reduction, produced by a different
 * and usually larger model, so a query can rescore candidates without the
 * whole corpus paying for the bigger model.
 *
 * That asymmetry is the point. Embedding 59.6M paragraphs under a 768-dim
 * model costs what embedding them under a 384-dim one did, twice over;
 * embedding the few hundred a scan returns costs nothing. The reduction has
 * already happened by the time stage two runs.
 *
 * Sparse and sorted. Entries are (index, code) with indices ascending, so a
 * point lookup is a binary search and a full pass is sequential. Most
 * records have no entry and that is the normal case, not a defect.
 *
 * Why a file and not a wider record: sigil_t is a fixed 64 bytes with a
 * _Static_assert on it, and the mapped store's layout depends on that size.
 * Widening it to carry a second code would break every store on disk and
 * make every store pay for a stage most will never run. A sidecar is
 * optional, and throwing one away costs only the recomputation.
 *
 * What is refused. Codes from two models are not comparable -- the whole
 * reason for stage two is that the models disagree -- and an index into the
 * wrong store lands on a real record rather than failing. Both are silent
 * failures that produce plausible neighbours, so the header records the
 * model name, the code width and the base store's count, and mapping checks
 * all three.
 */

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

#define SIGIL_SIDE_MAGIC   0x45444953554d4d41ULL /* "AMMUSIDE" */
#define SIGIL_SIDE_VERSION 1

typedef struct {
	uint64_t magic;
	uint32_t version;
	uint32_t words;          /* 64-bit words per refined code */
	uint64_t count;          /* entries in this sidecar */
	uint64_t base_count;     /* count of the store it refines */
	uint64_t index_off;
	uint64_t code_off;
	char     model[SIGIL_SIDE_MODEL_MAX];
	uint64_t reserved[8];
} sigil_side_header;

static size_t page_round(size_t n)
{
	size_t pg = (size_t)sysconf(_SC_PAGESIZE);

	return (n + pg - 1) & ~(pg - 1);
}

int sigil_side_save(const char *path, const uint32_t *index,
                    const uint64_t *code, size_t count, size_t words,
                    const char *model, uint64_t base_count)
{
	sigil_side_header h;
	char *tmp;
	int fd, rc = -1;
	size_t i;

	if (words == 0 || words > 64)
		return -1;

	/* Ascending indices are what makes lookup a binary search. Checking
	 * here rather than trusting the caller: an unsorted sidecar does not
	 * fail, it silently misses entries that are present. */
	for (i = 1; i < count; i++)
		if (index[i] <= index[i - 1]) {
			errno = EINVAL;
			return -1;
		}

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
	h.magic      = SIGIL_SIDE_MAGIC;
	h.version    = SIGIL_SIDE_VERSION;
	h.words      = (uint32_t)words;
	h.count      = count;
	h.base_count = base_count;
	h.index_off  = page_round(sizeof h);
	h.code_off   = h.index_off + page_round(count * sizeof(uint32_t));
	if (model != NULL)
		snprintf(h.model, sizeof h.model, "%s", model);

	if (write(fd, &h, sizeof h) != (ssize_t)sizeof h)
		goto out;
	if (lseek(fd, (off_t)h.index_off, SEEK_SET) < 0)
		goto out;
	if (count && write(fd, index, count * sizeof(uint32_t))
	    != (ssize_t)(count * sizeof(uint32_t)))
		goto out;
	if (lseek(fd, (off_t)h.code_off, SEEK_SET) < 0)
		goto out;
	if (count && write(fd, code, count * words * sizeof(uint64_t))
	    != (ssize_t)(count * words * sizeof(uint64_t)))
		goto out;

	/* Extend to the full mapped length: a reader faulting past the end of
	 * the file takes SIGBUS, not an error return. */
	if (ftruncate(fd, (off_t)(h.code_off
	                          + page_round(count * words * sizeof(uint64_t)))) != 0)
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

int sigil_side_map(sigil_side_t *sd, const char *path,
                   const sigil_store_t *st, const char *model)
{
	sigil_side_header h;
	struct stat sb;
	char *base;
	int fd;

	memset(sd, 0, sizeof *sd);

	fd = open(path, O_RDONLY);
	if (fd < 0)
		return -1;
	if (read(fd, &h, sizeof h) != (ssize_t)sizeof h) {
		close(fd);
		return -1;
	}
	if (h.magic != SIGIL_SIDE_MAGIC || h.version != SIGIL_SIDE_VERSION ||
	    h.words == 0 || h.words > 64) {
		close(fd);
		errno = EINVAL;
		return -1;
	}
	/*
	 * A sidecar built against another store indexes real records here and
	 * returns neighbours that look entirely reasonable. Refuse.
	 */
	if (st != NULL && h.base_count != (uint64_t)st->count) {
		close(fd);
		errno = EINVAL;
		return -1;
	}
	/* Two models' codes are not comparable; that is the premise of the
	 * whole stage, so a mismatch here is never benign. */
	if (model != NULL && strncmp(h.model, model, sizeof h.model) != 0) {
		close(fd);
		errno = EINVAL;
		return -1;
	}
	if (fstat(fd, &sb) != 0) {
		close(fd);
		return -1;
	}

	base = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
	close(fd);
	if (base == MAP_FAILED)
		return -1;

	sd->map    = base;
	sd->maplen = (size_t)sb.st_size;
	sd->count  = (size_t)h.count;
	sd->words  = (size_t)h.words;
	sd->bits   = (size_t)h.words * 64;
	sd->base_count = h.base_count;
	sd->index  = (const uint32_t *)(base + h.index_off);
	sd->code   = (const uint64_t *)(base + h.code_off);
	memcpy(sd->model, h.model, sizeof sd->model);
	sd->model[sizeof sd->model - 1] = '\0';
	return 0;
}

void sigil_side_unmap(sigil_side_t *sd)
{
	if (sd->map != NULL)
		munmap(sd->map, sd->maplen);
	memset(sd, 0, sizeof *sd);
}

const uint64_t *sigil_side_lookup(const sigil_side_t *sd, uint32_t i)
{
	size_t lo = 0, hi = sd->count;

	while (lo < hi) {
		size_t mid = lo + (hi - lo) / 2;

		if (sd->index[mid] == i)
			return sd->code + mid * sd->words;
		if (sd->index[mid] < i)
			lo = mid + 1;
		else
			hi = mid;
	}
	return NULL;
}

uint32_t sigil_side_hamming(const sigil_side_t *sd, const uint64_t *a,
                            const uint64_t *b)
{
	uint32_t d = 0;
	size_t w;

	for (w = 0; w < sd->words; w++)
		d += (uint32_t)__builtin_popcountll(a[w] ^ b[w]);
	return d;
}
