/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Semantic separation test.
 *
 * This is the test that distinguishes a semantic filesystem from a hashing
 * one. Every pair below is a paraphrase that shares almost no vocabulary with
 * its partner, so a byte-level hash cannot possibly relate them: any signal
 * here comes from meaning.
 *
 * The assertion is separation, not absolute distance. Paraphrases must land
 * closer in Hamming space than unrelated pairs do, with a margin. Exact
 * distances depend on the model and the hyperplane seed; the ordering must
 * not.
 *
 * Run with the non-semantic backend and this test is expected to FAIL — that
 * is the point. It is the regression guard against the placeholder coming
 * back.
 */

#define _POSIX_C_SOURCE 200112L

#include "sigil_embed.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIMHASH_SEED 0x5191c0de5191c0deULL

struct pair {
	const char *a;
	const char *b;
	const char *label;
};

/* Paraphrases: same meaning, deliberately disjoint wording. */
static const struct pair paraphrases[] = {
	{ "the cat sat on the mat",
	  "a feline rested upon the rug",                    "cat/feline" },
	{ "quarterly earnings exceeded analyst forecasts",
	  "revenue for the period beat wall street estimates", "earnings/revenue" },
	{ "the server crashed under heavy load",
	  "the machine went down when traffic spiked",       "crash/downtime" },
	{ "she photographed birds at dawn",
	  "at sunrise he took pictures of wildlife",         "photography" },
	{ "the contract was terminated early",
	  "they ended the agreement ahead of schedule",      "contract/agreement" },
};

/* Unrelated: different topics entirely. */
static const struct pair unrelated[] = {
	{ "the cat sat on the mat",
	  "quarterly earnings exceeded analyst forecasts",   "cat/earnings" },
	{ "the server crashed under heavy load",
	  "she photographed birds at dawn",                  "crash/photography" },
	{ "the contract was terminated early",
	  "a feline rested upon the rug",                    "contract/feline" },
	{ "revenue for the period beat wall street estimates",
	  "at sunrise he took pictures of wildlife",         "revenue/photography" },
};

static int embed_pair(sigil_embedder_t *e, const sigil_simhash_t *sh,
                      const struct pair *p, uint32_t *ha, uint32_t *hb)
{
	sigil_t sa, sb;

	if (sigil_generate_semantic(e, sh, p->a, strlen(p->a), 0, 0, NULL, &sa) != 0)
		return -1;
	if (sigil_generate_semantic(e, sh, p->b, strlen(p->b), 0, 0, NULL, &sb) != 0)
		return -1;
	*ha = sa.lsh;
	*hb = sb.lsh;
	return 0;
}

int main(int argc, char **argv)
{
	const char *model = (argc > 1) ? argv[1] : NULL;
	sigil_embedder_t *e;
	sigil_simhash_t sh;
	unsigned para_sum = 0, unrel_sum = 0;
	unsigned para_max = 0, unrel_min = 32;
	size_t npara = sizeof(paraphrases) / sizeof(paraphrases[0]);
	size_t nunrel = sizeof(unrelated) / sizeof(unrelated[0]);
	int failures = 0;

	if (model) {
		e = sigil_embedder_llama(model);
		if (!e) {
			fprintf(stderr,
			        "could not load model: %s\n"
			        "(was libsigil built with SIGIL_WITH_LLAMA?)\n", model);
			return 2;
		}
	} else {
		printf("no model given — using the NON-SEMANTIC backend.\n"
		       "This test is expected to fail; that is what it is for.\n\n");
		e = sigil_embedder_hash_nonsemantic(SIGIL_EMBED_DIM_MINILM);
		if (!e)
			return 2;
	}

	printf("embedder: %s (dim %zu)\n\n", e->name(e), e->dim(e));

	if (sigil_simhash_init(&sh, e->dim(e), SIMHASH_SEED) != 0) {
		e->destroy(e);
		return 2;
	}

	printf("paraphrases (should be CLOSE — low Hamming distance):\n");
	for (size_t i = 0; i < npara; i++) {
		uint32_t a, b;
		unsigned d;

		if (embed_pair(e, &sh, &paraphrases[i], &a, &b) != 0) {
			printf("  ERROR embedding %s\n", paraphrases[i].label);
			failures++;
			continue;
		}
		d = (unsigned)__builtin_popcount(a ^ b);
		para_sum += d;
		if (d > para_max)
			para_max = d;
		printf("  %-22s %2u\n", paraphrases[i].label, d);
	}

	printf("\nunrelated (should be FAR — high Hamming distance):\n");
	for (size_t i = 0; i < nunrel; i++) {
		uint32_t a, b;
		unsigned d;

		if (embed_pair(e, &sh, &unrelated[i], &a, &b) != 0) {
			printf("  ERROR embedding %s\n", unrelated[i].label);
			failures++;
			continue;
		}
		d = (unsigned)__builtin_popcount(a ^ b);
		unrel_sum += d;
		if (d < unrel_min)
			unrel_min = d;
		printf("  %-22s %2u\n", unrelated[i].label, d);
	}

	{
		double pmean = (double)para_sum / (double)npara;
		double umean = (double)unrel_sum / (double)nunrel;

		printf("\nmean paraphrase distance: %.1f\n", pmean);
		printf("mean unrelated distance:  %.1f\n", umean);
		printf("separation:               %.1f bits\n", umean - pmean);

		/* A real embedding should separate these by a wide margin. Six
		 * bits out of 32 is conservative: measured separation with
		 * MiniLM is around nine. */
		if (umean - pmean < 6.0) {
			printf("\nFAIL: paraphrases are not meaningfully closer than\n"
			       "      unrelated text. The LSH bits are not semantic.\n");
			failures++;
		}

		/* Distributions must not overlap: the worst paraphrase should
		 * still beat the best unrelated pair, or a threshold cannot
		 * separate them at query time. */
		if (para_max >= unrel_min) {
			printf("\nFAIL: distributions overlap (worst paraphrase %u >= "
			       "closest unrelated %u).\n"
			       "      No single threshold can separate them.\n",
			       para_max, unrel_min);
			failures++;
		}
	}

	sigil_simhash_free(&sh);
	e->destroy(e);

	if (failures) {
		printf("\n%d FAILURES\n", failures);
		return 1;
	}
	printf("\nall passed — LSH bits carry meaning\n");
	return 0;
}
