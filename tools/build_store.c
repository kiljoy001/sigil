/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Build a mapped store from the corpus CSV, streaming.
 *
 * The Python eval harness read the whole CSV into a dict-of-lists before it
 * sampled a single row: 628 MB per 500,000 rows measured, so 83 GB for the
 * 68.0M-paragraph corpus against 71 GB available. It died before reaching
 * the embedder. This never holds more than one field value at a time --
 * libcsv hands back one field per callback and the record is pushed and
 * forgotten.
 *
 * CSV parsing is libcsv rather than hand-rolled. The corpus has quoted
 * fields containing commas, embedded newlines and doubled quotes; a
 * split-on-comma would silently mis-field those rows rather than fail, and
 * a wrong field is a wrong record that looks fine.
 *
 * Bits are the byte-shingle fallback unless -e names a model. That
 * distinction matters enough that the summary says which was used: a store
 * built without an embedder has bits that are not semantic, and a
 * similarity query over it returns neighbourhoods that look plausible and
 * mean nothing.
 *
 *	tools/build_store corpus.csv out.smap [-e model.gguf] [-n limit]
 */

#define _POSIX_C_SOURCE 200809L

#include "sigil.h"
#include "sigil_embed.h"

#include <csv.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

/* Column order in the corpus CSV, fixed by tools/pipeline.py. */
enum { C_TEXTID, C_PARA, C_LOCC, C_SUBJECTS, C_DEATH, C_AUTHOR, C_TITLE,
       C_TEXT, C_NCOLS };

#define BATCH 128
#define MINLEN 40   /* paragraphs shorter than this are not worth a record */

struct builder {
	sigil_store_t st;
	sigil_simhash_t sh;
	sigil_embedder_t *emb;
	int have_emb;
	size_t dim;

	/* current row */
	int col;
	unsigned text_id, para;
	char *text;
	size_t textlen;
	int header_seen;

	/* pending embedding batch */
	char **btext;
	size_t *blen;
	unsigned *bpara;
	unsigned *bts;
	size_t bn;

	/* counters */
	size_t rows, stored, skipped_short, skipped_empty;
	float *vecs;

	/* Stage-two sidecar, accumulated during the same pass. The vectors
	 * are already in hand when the codes are projected from them;
	 * re-embedding the corpus later to recover them would cost the whole
	 * run again. */
	int      want_side;
	sigil_sidebuild_t side;
};

static double now_s(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

/*
 * Remember one vector for the sidecar. The index recorded is the record
 * this vector is about to become, so the two stay aligned even though the
 * store is appended to separately.
 */
static void
keep_vec(struct builder *b, const float *v)
{
	/* Segmented, like the store: the corpus is 85 GB of float32 at dim
	 * 384, so accumulating into one array and writing at the end is the
	 * same materialise-then-process failure that made store_commit need
	 * 54 GB to write 12 GB. */
	if (sigil_sidebuild_add(&b->side, (uint32_t)b->stored, v) != 0) {
		fprintf(stderr, "sidecar segment allocation failed at %zu "
		                "vectors; aborting rather than writing a "
		                "partial one\n", b->side.count);
		exit(1);
	}
}

/* Push one record, with bits already decided. */
static void
emit(struct builder *b, const char *text, size_t len, unsigned para,
     unsigned ts, const uint64_t *lsh)
{
	sigil_t s;

	sigil_generate_para(text, len, para, ts, 0, NULL, &s);
	if (lsh != NULL)
		memcpy(s.lsh, lsh, sizeof s.lsh);
	if (sigil_store_push(&b->st, &s) < 0) {
		fprintf(stderr, "push failed at record %zu\n", b->stored);
		exit(1);
	}
	b->stored++;
}

/* Embed everything queued and push it. */
static void
flush(struct builder *b)
{
	size_t i;

	if (b->bn == 0)
		return;

	if (b->have_emb) {
		int ok = 1;

		if (b->emb->embed_batch != NULL) {
			ok = b->emb->embed_batch(b->emb,
			                         (const char **)b->btext,
			                         b->blen, b->bn, b->vecs) > 0;
		} else {
			/* llama.cpp exposes no batch path; loop. */
			for (i = 0; i < b->bn; i++)
				if (b->emb->embed(b->emb, b->btext[i],
				                  b->blen[i],
				                  b->vecs + i * b->dim) != 0) {
					ok = 0;
					break;
				}
		}
		if (ok) {
			for (i = 0; i < b->bn; i++) {
				uint64_t code[SIGIL_LSH_WORDS];

				sigil_simhash_project(&b->sh,
				                      b->vecs + i * b->dim,
				                      code);
				if (b->want_side)
					keep_vec(b, b->vecs + i * b->dim);
				emit(b, b->btext[i], b->blen[i], b->bpara[i],
				     b->bts[i], code);
			}
			goto done;
		}
		/* A failed embed must not silently store fallback bits as
		 * though they were semantic. */
		fprintf(stderr, "embed failed near record %zu; aborting so a "
		                "half-semantic store is never produced\n",
		        b->stored);
		exit(1);
	}

	for (i = 0; i < b->bn; i++)
		emit(b, b->btext[i], b->blen[i], b->bpara[i], b->bts[i], NULL);
done:
	for (i = 0; i < b->bn; i++)
		free(b->btext[i]);
	b->bn = 0;
}

/* libcsv: one field. */
static void
on_field(void *data, size_t len, void *p)
{
	struct builder *b = p;
	const char *s = data;

	switch (b->col) {
	case C_TEXTID:
		b->text_id = (unsigned)strtoul(s ? s : "0", NULL, 10);
		break;
	case C_PARA:
		b->para = (unsigned)strtoul(s ? s : "0", NULL, 10);
		break;
	case C_TEXT:
		free(b->text);
		b->text = malloc(len + 1);
		if (b->text != NULL) {
			memcpy(b->text, s, len);
			b->text[len] = '\0';
			b->textlen = len;
		} else {
			b->textlen = 0;
		}
		break;
	default:
		break;
	}
	b->col++;
}

/* libcsv: end of row. */
static void
on_row(int c, void *p)
{
	struct builder *b = p;

	(void)c;
	b->col = 0;
	b->rows++;

	if (!b->header_seen) {          /* the column-name row */
		b->header_seen = 1;
		goto clear;
	}
	if (b->text == NULL || b->textlen == 0) {
		b->skipped_empty++;
		goto clear;
	}
	if (b->textlen < MINLEN) {
		b->skipped_short++;
		goto clear;
	}

	b->btext[b->bn] = b->text;
	b->blen[b->bn] = b->textlen;
	b->bpara[b->bn] = b->para;
	/* One timestamp per book, so the field carries something real
	 * rather than a constant. */
	b->bts[b->bn] = 1700000000u + b->text_id;
	b->bn++;
	b->text = NULL;                 /* ownership moved to the batch */
	b->textlen = 0;

	if (b->bn == BATCH)
		flush(b);
	return;
clear:
	free(b->text);
	b->text = NULL;
	b->textlen = 0;
}

int
main(int argc, char **argv)
{
	struct builder b;
	struct csv_parser cp;
	const char *in_path = NULL, *out_path = NULL, *model = NULL;
	const char *device = "GPU.1";
	int side = 0;
	size_t limit = 0;
	char buf[1 << 20];
	FILE *in;
	size_t got;
	double t0, last = 0;
	int i;

	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-e") && i + 1 < argc)
			model = argv[++i];
		else if (!strcmp(argv[i], "-s"))
			side = 1;
		else if (!strcmp(argv[i], "-d") && i + 1 < argc)
			device = argv[++i];
		else if (!strcmp(argv[i], "-n") && i + 1 < argc)
			limit = strtoul(argv[++i], NULL, 10);
		else if (in_path == NULL)
			in_path = argv[i];
		else
			out_path = argv[i];
	}
	if (in_path == NULL || out_path == NULL) {
		fprintf(stderr,
		        "usage: build_store <corpus.csv> <out.smap> "
		        "[-e model.gguf|ov-dir] [-d DEVICE] [-s] [-n limit]\n");
		return 2;
	}

	memset(&b, 0, sizeof b);
	if (sigil_store_init(&b.st, 0) != 0) {
		fprintf(stderr, "store init failed\n");
		return 1;
	}

	if (model != NULL) {
		/*
		 * A directory is an OpenVINO export, a file is a GGUF. The
		 * distinction matters for throughput, not just convenience:
		 * llama.cpp has no batch path here, and one paragraph per
		 * inference is the difference between hours and days over a
		 * corpus this size.
		 */
		struct stat mst;

		if (stat(model, &mst) == 0 && S_ISDIR(mst.st_mode))
			b.emb = sigil_embedder_openvino(model, device);
		else
			b.emb = sigil_embedder_llama(model);
		if (b.emb == NULL) {
			fprintf(stderr, "cannot load %s\n", model);
			return 1;
		}
		b.dim = b.emb->dim(b.emb);
		if (sigil_simhash_init(&b.sh, b.dim, 0x5191c0deULL) != 0) {
			fprintf(stderr, "simhash init failed\n");
			return 1;
		}
		b.have_emb = 1;
		b.want_side = side;
		if (side && sigil_sidebuild_init(&b.side, b.dim) != 0) {
			fprintf(stderr, "sidecar init failed\n");
			return 1;
		}
		b.vecs = malloc(BATCH * b.dim * sizeof *b.vecs);
		fprintf(stderr, "embedder: %s (dim %zu)\n",
		        b.emb->name(b.emb), b.dim);
	} else {
		fprintf(stderr, "NO EMBEDDER -- bits will be the byte-shingle "
		                "fallback, which is NOT semantic\n");
	}

	b.btext = calloc(BATCH, sizeof *b.btext);
	b.blen  = calloc(BATCH, sizeof *b.blen);
	b.bpara = calloc(BATCH, sizeof *b.bpara);
	b.bts   = calloc(BATCH, sizeof *b.bts);

	if (csv_init(&cp, CSV_APPEND_NULL) != 0) {
		fprintf(stderr, "csv_init failed\n");
		return 1;
	}

	in = fopen(in_path, "rb");
	if (in == NULL) {
		perror(in_path);
		return 1;
	}

	t0 = now_s();
	while ((got = fread(buf, 1, sizeof buf, in)) > 0) {
		if (csv_parse(&cp, buf, got, on_field, on_row, &b) != got) {
			fprintf(stderr, "csv parse error: %s\n",
			        csv_strerror(csv_error(&cp)));
			return 1;
		}
		if (limit && b.stored >= limit)
			break;
		{
			double now = now_s();

			if (now - last > 10.0) {
				last = now;
				fprintf(stderr,
				        "  %.1fM rows, %.1fM stored, %.0f rec/s\n",
				        b.rows / 1e6, b.stored / 1e6,
				        b.stored / (now - t0));
			}
		}
	}
	csv_fini(&cp, on_field, on_row, &b);
	csv_free(&cp);
	fclose(in);
	flush(&b);

	fprintf(stderr, "\nrows read       %zu\n", b.rows);
	fprintf(stderr, "records stored  %zu\n", b.stored);
	fprintf(stderr, "skipped short   %zu\n", b.skipped_short);
	fprintf(stderr, "skipped empty   %zu\n", b.skipped_empty);
	fprintf(stderr, "segments        %zu\n", b.st.nseg);
	fprintf(stderr, "elapsed         %.0f s (%.0f rec/s)\n",
	        now_s() - t0, b.stored / (now_s() - t0));
	fprintf(stderr, "bits            %s\n",
	        b.have_emb ? "semantic (SimHash over embeddings)"
	                   : "BYTE-SHINGLE FALLBACK -- not semantic");

	if (sigil_store_save(&b.st, out_path) != 0) {
		perror("save");
		return 1;
	}
	fprintf(stderr, "wrote           %s\n", out_path);

	if (b.want_side && b.side.count) {
		char *sp = malloc(strlen(out_path) + 8);

		sprintf(sp, "%s.side", out_path);
		if (sigil_sidebuild_save(&b.side, sp, b.emb->name(b.emb),
		                         (uint64_t)b.st.count) != 0)
			perror("sidecar");
		else
			fprintf(stderr, "wrote           %s (%zu vectors, "
			                "dim %zu)\n", sp, b.side.count, b.dim);
		free(sp);
		sigil_sidebuild_free(&b.side);
	}

	sigil_store_free(&b.st);
	return 0;
}
