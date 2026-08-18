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
#include <math.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define SIGIL_SIDE_MAGIC   0x45444953554d4d41ULL /* "AMMUSIDE" */
#define SIGIL_SIDE_VERSION 1

typedef struct {
	uint64_t magic;
	uint32_t version;
	uint32_t words;          /* 64-bit words per code, when kind == CODE */
	uint32_t kind;           /* sigil_side_kind */
	uint32_t dim;            /* floats per vector, when kind == FLOAT */
	uint64_t count;          /* entries in this sidecar */
	uint64_t base_count;     /* count of the store it refines */
	uint64_t index_off;
	uint64_t code_off;
	char     model[SIGIL_SIDE_MODEL_MAX];
	uint64_t reserved[7];
} sigil_side_header;

static size_t page_round(size_t n)
{
	size_t pg = (size_t)sysconf(_SC_PAGESIZE);

	return (n + pg - 1) & ~(pg - 1);
}

/*
 * One writer for both payloads. The sparse ascending index is the same
 * either way; only the element width differs, so splitting this in two
 * would duplicate the part that is easy to get wrong.
 */
static int
side_save(const char *path, const uint32_t *index, const void *payload,
          size_t count, size_t elem_bytes, sigil_side_kind kind,
          size_t words, size_t dim, const char *model, uint64_t base_count)
{
	sigil_side_header h;
	char *tmp;
	int fd, rc = -1;
	size_t i;

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
	h.kind       = (uint32_t)kind;
	h.words      = (uint32_t)words;
	h.dim        = (uint32_t)dim;
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
	if (count && write(fd, payload, count * elem_bytes)
	    != (ssize_t)(count * elem_bytes))
		goto out;

	/* Extend to the full mapped length: a reader faulting past the end of
	 * the file takes SIGBUS, not an error return. */
	if (ftruncate(fd, (off_t)(h.code_off + page_round(count * elem_bytes))) != 0)
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

int sigil_side_save(const char *path, const uint32_t *index,
                    const uint64_t *code, size_t count, size_t words,
                    const char *model, uint64_t base_count)
{
	if (words == 0 || words > 64)
		return -1;
	return side_save(path, index, code, count,
	                 words * sizeof(uint64_t), SIGIL_SIDE_CODE,
	                 words, 0, model, base_count);
}

int sigil_side_save_vec(const char *path, const uint32_t *index,
                        const float *vec, size_t count, size_t dim,
                        const char *model, uint64_t base_count)
{
	if (dim == 0 || dim > 65536)
		return -1;
	return side_save(path, index, vec, count,
	                 dim * sizeof(float), SIGIL_SIDE_FLOAT,
	                 0, dim, model, base_count);
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
	if (h.magic != SIGIL_SIDE_MAGIC || h.version != SIGIL_SIDE_VERSION) {
		close(fd);
		errno = EINVAL;
		return -1;
	}
	/* Float bytes read as a bit pattern give a Hamming distance rather
	 * than an error, so the payload kind is checked, not assumed. */
	if (h.kind == (uint32_t)SIGIL_SIDE_CODE) {
		if (h.words == 0 || h.words > 64) {
			close(fd);
			errno = EINVAL;
			return -1;
		}
	} else if (h.kind == (uint32_t)SIGIL_SIDE_FLOAT) {
		if (h.dim == 0 || h.dim > 65536) {
			close(fd);
			errno = EINVAL;
			return -1;
		}
	} else {
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
	sd->kind   = (sigil_side_kind)h.kind;
	sd->dim    = (size_t)h.dim;
	sd->index  = (const uint32_t *)(base + h.index_off);
	if (sd->kind == SIGIL_SIDE_CODE)
		sd->code = (const uint64_t *)(base + h.code_off);
	else
		sd->vec = (const float *)(base + h.code_off);
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

/* Where index i sits in the sparse table, or -1. Shared by both payloads:
 * the search is over the index array and knows nothing about the payload. */
static ptrdiff_t
side_slot(const sigil_side_t *sd, uint32_t i)
{
	size_t lo = 0, hi = sd->count;

	while (lo < hi) {
		size_t mid = lo + (hi - lo) / 2;

		if (sd->index[mid] == i)
			return (ptrdiff_t)mid;
		if (sd->index[mid] < i)
			lo = mid + 1;
		else
			hi = mid;
	}
	return -1;
}

const uint64_t *sigil_side_lookup(const sigil_side_t *sd, uint32_t i)
{
	ptrdiff_t slot;

	if (sd->kind != SIGIL_SIDE_CODE)
		return NULL;
	slot = side_slot(sd, i);
	return slot < 0 ? NULL : sd->code + (size_t)slot * sd->words;
}

const float *sigil_side_vec(const sigil_side_t *sd, uint32_t i)
{
	ptrdiff_t slot;

	if (sd->kind != SIGIL_SIDE_FLOAT)
		return NULL;
	slot = side_slot(sd, i);
	return slot < 0 ? NULL : sd->vec + (size_t)slot * sd->dim;
}

double sigil_side_cosine(const sigil_side_t *sd, const float *a,
                         const float *b)
{
	double dot = 0.0, na = 0.0, nb = 0.0;
	size_t k;

	for (k = 0; k < sd->dim; k++) {
		dot += (double)a[k] * (double)b[k];
		na  += (double)a[k] * (double)a[k];
		nb  += (double)b[k] * (double)b[k];
	}
	/* An all-zero vector has no direction; report no similarity rather
	 * than dividing by zero and propagating a NaN through the sort. */
	if (na == 0.0 || nb == 0.0)
		return 0.0;
	return dot / (sqrt(na) * sqrt(nb));
}

/*
 * Reorder the scan's candidates by cosine, best first.
 *
 * Measured on 200,064 records over 1,191 queries: R@1 0.0269 -> 0.0806,
 * 3.0x, and 80% of the way to the float32 ceiling of 0.1008. Cost was 18.63
 * ms/query against 18.56 for the bare scan -- 0.4%, because the scan already
 * reduced the corpus to a shortlist and the float work only touches that.
 *
 * Candidates with no sidecar entry are kept, after every reranked one, in
 * their original scan order. Dropping them would let a gap in stage two --
 * a record the sidecar simply does not cover -- read as evidence that the
 * record is a poor match, which it is not.
 */
static int
cmp_scored(const void *a, const void *b)
{
	double x = ((const struct { double s; uint32_t i; } *)a)->s;
	double y = ((const struct { double s; uint32_t i; } *)b)->s;

	return x > y ? -1 : x < y ? 1 : 0;
}

size_t sigil_side_rerank(const sigil_side_t *sd, const float *query,
                         uint32_t *cand, size_t n)
{
	struct { double s; uint32_t i; } *scored;
	uint32_t *missing;
	size_t ns = 0, nm = 0, k;

	if (sd->kind != SIGIL_SIDE_FLOAT || query == NULL || n == 0)
		return 0;

	scored = malloc(n * sizeof *scored);
	missing = malloc(n * sizeof *missing);
	if (scored == NULL || missing == NULL) {
		free(scored);
		free(missing);
		return 0;
	}

	for (k = 0; k < n; k++) {
		const float *v = sigil_side_vec(sd, cand[k]);

		if (v == NULL) {
			missing[nm++] = cand[k];
			continue;
		}
		scored[ns].s = sigil_side_cosine(sd, query, v);
		scored[ns].i = cand[k];
		ns++;
	}

	qsort(scored, ns, sizeof *scored, cmp_scored);
	for (k = 0; k < ns; k++)
		cand[k] = scored[k].i;
	for (k = 0; k < nm; k++)
		cand[ns + k] = missing[k];

	free(scored);
	free(missing);
	return ns;
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
