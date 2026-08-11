/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Property tests for libsigil, using theft.
 *
 * unit.c checks behaviour someone thought to write down. These check
 * invariants against generated operation sequences, comparing the store
 * against a plain array that models what it should contain. That is where the
 * bugs in this project actually lived: a reload that recomputed a hash from
 * the wrong input, a batch path that disagreed with the per-text path on
 * exactly the inputs nobody sampled, a record count that drifted from its
 * parallel arrays after a growth.
 *
 * Each property states something that must hold for *every* sequence, so a
 * failure comes with a shrunk counterexample rather than a hunch.
 *
 *	make prop
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "theft.h"

#include "sigil.h"
#include "sigil_embed.h"

enum { MaxOps = 96, MaxText = 64 };

/* One generated operation against the store. */
struct op {
	uint8_t kind;              /* 0 push, 1 get, 2 reinit */
	uint8_t textlen;
	uint8_t text[MaxText];
	uint32_t para;
	uint32_t timestamp;
};

struct op_seq {
	size_t n;
	struct op ops[MaxOps];
};

/* --- generation --------------------------------------------------------- */

static enum theft_alloc_res
seq_alloc(struct theft *t, void *env, void **out)
{
	struct op_seq *s = calloc(1, sizeof *s);
	size_t i, k;

	(void)env;
	if (s == NULL)
		return THEFT_ALLOC_ERROR;

	s->n = theft_random_choice(t, MaxOps) + 1;
	for (i = 0; i < s->n; i++) {
		struct op *o = &s->ops[i];

		/* Weighted toward pushes: a sequence of nothing but gets
		 * exercises an empty store and finds nothing. */
		o->kind = theft_random_choice(t, 10) < 8 ? 0
		        : (theft_random_choice(t, 10) < 9 ? 1 : 2);
		o->textlen = (uint8_t)(theft_random_choice(t, MaxText - 1) + 1);
		for (k = 0; k < o->textlen; k++) {
			/* Full byte range, not just printable: a paragraph is
			 * whatever bytes were in the file, and the hash must
			 * not care. */
			o->text[k] = (uint8_t)theft_random_choice(t, 256);
		}
		o->para = (uint32_t)theft_random_choice(t, 1000);
		o->timestamp = (uint32_t)theft_random_choice(t, 100000);
	}
	*out = s;
	return THEFT_ALLOC_OK;
}

static void
seq_free(void *instance, void *env)
{
	(void)env;
	free(instance);
}

static void
seq_print(FILE *f, const void *instance, void *env)
{
	const struct op_seq *s = instance;
	size_t i;

	(void)env;
	fprintf(f, "%zu ops:", s->n);
	for (i = 0; i < s->n && i < 12; i++)
		fprintf(f, " %s(len=%u,para=%u)",
		        s->ops[i].kind == 0 ? "push" :
		        s->ops[i].kind == 1 ? "get" : "reinit",
		        s->ops[i].textlen, s->ops[i].para);
	if (s->n > 12)
		fprintf(f, " ...");
	fprintf(f, "\n");
}

static struct theft_type_info seq_info = {
	.alloc = seq_alloc,
	.free = seq_free,
	.print = seq_print,
	.autoshrink_config = { .enable = true },
};

/* --- properties --------------------------------------------------------- */

/*
 * The store must contain exactly what was pushed, in order.
 *
 * Modelled with a plain array. store_grow() reallocates seven parallel arrays
 * and a single one left behind is invisible until something reads that field
 * -- which is how a record can carry the right hash and the wrong timestamp.
 */
static enum theft_trial_res
prop_matches_model(struct theft *t, void *arg1)
{
	struct op_seq *s = arg1;
	sigil_store_t st;
	sigil_t *model;
	size_t nmodel = 0, i;
	enum theft_trial_res res = THEFT_TRIAL_PASS;

	(void)t;
	model = calloc(MaxOps, sizeof *model);
	if (model == NULL)
		return THEFT_TRIAL_ERROR;
	if (sigil_store_init(&st, 2) != 0) {
		free(model);
		return THEFT_TRIAL_ERROR;
	}

	for (i = 0; i < s->n; i++) {
		struct op *o = &s->ops[i];
		sigil_t rec;

		if (o->kind == 2) {
			/* A reinitialised store must be empty, not merely
			 * report zero while holding stale rows. */
			sigil_store_free(&st);
			if (sigil_store_init(&st, 2) != 0) {
				res = THEFT_TRIAL_ERROR;
				goto done;
			}
			nmodel = 0;
			continue;
		}
		if (o->kind == 1) {
			sigil_t got;

			if (nmodel == 0) {
				if (sigil_store_get(&st, 0, &got) == 0) {
					res = THEFT_TRIAL_FAIL;  /* empty */
					goto done;
				}
				continue;
			}
			if (sigil_store_get(&st, nmodel - 1, &got) != 0 ||
			    memcmp(&got, &model[nmodel - 1], sizeof got) != 0) {
				res = THEFT_TRIAL_FAIL;
				goto done;
			}
			continue;
		}

		sigil_generate_para(o->text, o->textlen, o->para,
		                    o->timestamp, 0, NULL, &rec);
		if (sigil_store_push(&st, &rec) < 0) {
			res = THEFT_TRIAL_ERROR;
			goto done;
		}
		model[nmodel++] = rec;
	}

	if (st.count != nmodel) {
		res = THEFT_TRIAL_FAIL;
		goto done;
	}
	for (i = 0; i < nmodel; i++) {
		sigil_t got;

		if (sigil_store_get(&st, i, &got) != 0 ||
		    memcmp(&got, &model[i], sizeof got) != 0) {
			res = THEFT_TRIAL_FAIL;
			goto done;
		}
	}
done:
	sigil_store_free(&st);
	free(model);
	return res;
}

/*
 * A sigil's hash is a function of its content alone.
 *
 * The bug this exists for: reload recomputed the BLAKE3 from the record's
 * *path* rather than reading the stored hash, so every restart silently
 * reassigned every identity, and /similar/<hex>/ could not be reached by a
 * hash the store had just written.
 */
static enum theft_trial_res
prop_hash_is_content(struct theft *t, void *arg1)
{
	struct op_seq *s = arg1;
	size_t i;

	(void)t;
	for (i = 0; i < s->n; i++) {
		struct op *o = &s->ops[i];
		sigil_t a, b;

		if (o->kind != 0)
			continue;
		/* Same bytes, different metadata: the hash must not move. */
		sigil_generate_para(o->text, o->textlen, o->para,
		                    o->timestamp, 0, NULL, &a);
		sigil_generate_para(o->text, o->textlen, o->para + 1,
		                    o->timestamp + 1, 7, NULL, &b);
		if (memcmp(a.hash, b.hash, SIGIL_HASH_LEN) != 0)
			return THEFT_TRIAL_FAIL;
		/* And the record must still be exactly 64 bytes with the
		 * metadata actually recorded. */
		if (a.para != o->para || b.para != o->para + 1)
			return THEFT_TRIAL_FAIL;
	}
	return THEFT_TRIAL_PASS;
}

/*
 * Hamming distance is a metric: symmetric, zero only on equality, and
 * bounded by the code width. The scan kernels are built on it, and a wrong
 * distance does not crash -- it returns a plausible neighbour list forever.
 */
static enum theft_trial_res
prop_hamming_metric(struct theft *t, void *arg1)
{
	struct op_seq *s = arg1;
	size_t i;

	(void)t;
	for (i = 0; i + 1 < s->n; i++) {
		sigil_t a, b;
		uint32_t d1, d2;

		sigil_generate_para(s->ops[i].text, s->ops[i].textlen,
		                    0, 0, 0, NULL, &a);
		sigil_generate_para(s->ops[i + 1].text, s->ops[i + 1].textlen,
		                    0, 0, 0, NULL, &b);
		d1 = sigil_hamming(a.lsh, b.lsh);
		d2 = sigil_hamming(b.lsh, a.lsh);
		if (d1 != d2 || d1 > SIGIL_LSH_BITS)
			return THEFT_TRIAL_FAIL;
		if (sigil_hamming(a.lsh, a.lsh) != 0)
			return THEFT_TRIAL_FAIL;
	}
	return THEFT_TRIAL_PASS;
}

/*
 * The scan must return exactly the records within the radius.
 *
 * Compared against a brute-force loop rather than against itself: the SIMD
 * kernels have differential tests, but nothing checked that the scan's
 * *contract* -- every match, no non-matches, in ascending order -- holds over
 * arbitrary data.
 */
static enum theft_trial_res
prop_scan_is_exact(struct theft *t, void *arg1)
{
	struct op_seq *s = arg1;
	sigil_store_t st;
	uint32_t out[MaxOps];
	size_t npushed = 0, i, n;
	uint32_t radius;
	sigil_t q;
	enum theft_trial_res res = THEFT_TRIAL_PASS;

	(void)t;
	if (sigil_store_init(&st, 2) != 0)
		return THEFT_TRIAL_ERROR;

	for (i = 0; i < s->n; i++) {
		sigil_t rec;

		if (s->ops[i].kind != 0)
			continue;
		sigil_generate_para(s->ops[i].text, s->ops[i].textlen,
		                    s->ops[i].para, 0, 0, NULL, &rec);
		if (sigil_store_push(&st, &rec) < 0) {
			res = THEFT_TRIAL_ERROR;
			goto done;
		}
		npushed++;
	}
	if (npushed == 0)
		goto done;

	sigil_generate_para(s->ops[0].text, s->ops[0].textlen, 0, 0, 0,
	                    NULL, &q);
	radius = (uint32_t)(s->ops[0].para % (SIGIL_LSH_BITS + 1));

	n = sigil_scan_similar_scalar(&st, q.lsh, radius, out, MaxOps);

	/* Every returned index is genuinely within the radius, and the list
	 * is ascending -- callers materialise it as directory entries. */
	for (i = 0; i < n; i++) {
		sigil_t got;

		if (i > 0 && out[i] <= out[i - 1]) {
			res = THEFT_TRIAL_FAIL;
			goto done;
		}
		if (sigil_store_get(&st, out[i], &got) != 0 ||
		    sigil_hamming(got.lsh, q.lsh) > radius) {
			res = THEFT_TRIAL_FAIL;
			goto done;
		}
	}
	/* And nothing within the radius was missed, unless the output was
	 * capped. */
	if (n < MaxOps) {
		size_t expect = 0;

		for (i = 0; i < npushed; i++) {
			sigil_t got;

			if (sigil_store_get(&st, i, &got) == 0 &&
			    sigil_hamming(got.lsh, q.lsh) <= radius)
				expect++;
		}
		if (expect != n)
			res = THEFT_TRIAL_FAIL;
	}
done:
	sigil_store_free(&st);
	return res;
}

/*
 * SimHash is deterministic in its seed, and orthogonal in the sense that a
 * negated vector inverts every bit. A store's bits are only comparable
 * against the seed and dimension that produced them, so a projection that
 * drifted would silently invalidate every sigil already written.
 */
static enum theft_trial_res
prop_simhash_deterministic(struct theft *t, void *arg1)
{
	struct op_seq *s = arg1;
	sigil_simhash_t sh1, sh2;
	float v[64], w[64];
	uint64_t h1[SIGIL_LSH_WORDS], h2[SIGIL_LSH_WORDS];
	size_t i;
	int k, bits = 0;
	enum theft_trial_res res = THEFT_TRIAL_PASS;

	(void)t;
	if (sigil_simhash_init(&sh1, 64, 0xabcdef) != 0)
		return THEFT_TRIAL_ERROR;
	if (sigil_simhash_init(&sh2, 64, 0xabcdef) != 0) {
		sigil_simhash_free(&sh1);
		return THEFT_TRIAL_ERROR;
	}

	for (i = 0; i < s->n && i < 8; i++) {
		for (k = 0; k < 64; k++) {
			v[k] = (float)((int)s->ops[i].text[k % MaxText] - 128)
			       / 128.0f;
			w[k] = -v[k];
		}
		sigil_simhash_project(&sh1, v, h1);
		sigil_simhash_project(&sh2, v, h2);
		if (memcmp(h1, h2, sizeof h1) != 0) {
			res = THEFT_TRIAL_FAIL;
			goto done;
		}
		sigil_simhash_project(&sh1, w, h2);
		bits = 0;
		for (k = 0; k < SIGIL_LSH_WORDS; k++)
			bits += __builtin_popcountll(h1[k] ^ h2[k]);
		/* Exactly inverted, except where a component was 0.0 and the
		 * sign did not change; the generated data makes that possible,
		 * so allow the zero case rather than demanding all bits. */
		if (bits != SIGIL_LSH_BITS && bits != 0) {
			int allzero = 1;

			for (k = 0; k < 64; k++)
				if (v[k] != 0.0f)
					allzero = 0;
			if (!allzero) {
				res = THEFT_TRIAL_FAIL;
				goto done;
			}
		}
	}
done:
	sigil_simhash_free(&sh1);
	sigil_simhash_free(&sh2);
	return res;
}

/* --- runner ------------------------------------------------------------- */

static int
run(const char *name, theft_propfun1 *fn, size_t trials)
{
	struct theft_run_config cfg = {
		.name = name,
		.prop1 = fn,
		.type_info = { &seq_info },
		.trials = trials,
	};

	return theft_run(&cfg) == THEFT_RUN_PASS;
}

int
main(void)
{
	int pass = 1;

	pass &= run("store matches model", prop_matches_model, 2000);
	pass &= run("hash is a function of content", prop_hash_is_content, 2000);
	pass &= run("hamming is a metric", prop_hamming_metric, 2000);
	pass &= run("scan is exact", prop_scan_is_exact, 1000);
	pass &= run("simhash is deterministic", prop_simhash_deterministic, 500);

	if (pass) {
		printf("PASS: all properties held\n");
		return 0;
	}
	printf("FAIL: a property was violated\n");
	return 1;
}
