/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * llama.cpp embedding backend.
 *
 * Runs a GGUF sentence-embedding model (all-MiniLM-L6-v2 by default) in
 * process. Chosen over ONNX Runtime or libtorch because llama.cpp is plain C
 * with no install step, which keeps the dependency footprint to one shared
 * library that is already built.
 *
 * Build with SIGIL_WITH_LLAMA and point LLAMA_DIR at a llama.cpp checkout;
 * without it, sigil_embedder_llama() returns NULL and the library still
 * builds and links.
 */

#include "sigil_embed.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef SIGIL_WITH_LLAMA

#include "llama.h"
#include <math.h>

typedef struct {
	struct llama_model   *model;
	struct llama_context *ctx;
	size_t                dim;
	char                  name[128];
} llama_impl_t;

static size_t llama_dim(const sigil_embedder_t *self)
{
	return ((const llama_impl_t *)self->impl)->dim;
}

static const char *llama_name(const sigil_embedder_t *self)
{
	return ((const llama_impl_t *)self->impl)->name;
}

static void llama_destroy(sigil_embedder_t *self)
{
	llama_impl_t *im = self->impl;

	if (im) {
		if (im->ctx)
			llama_free(im->ctx);
		if (im->model)
			llama_model_free(im->model);
		free(im);
	}
	free(self);
}

static int llama_embed(sigil_embedder_t *self, const char *text, size_t len,
                       float *out)
{
	llama_impl_t *im = self->impl;
	const struct llama_vocab *vocab = llama_model_get_vocab(im->model);
	int n_ctx = (int)llama_n_ctx(im->ctx);
	llama_token *toks;
	int n_tok;
	struct llama_batch batch;
	const float *emb;
	double norm = 0.0;

	toks = malloc((size_t)n_ctx * sizeof(llama_token));
	if (!toks)
		return -1;

	/* add_special=true gives [CLS]/[SEP], which BERT pooling expects. */
	n_tok = llama_tokenize(vocab, text, (int32_t)len, toks, n_ctx, true, false);
	if (n_tok < 0) {
		/* Text exceeds the context window; truncate rather than fail.
		 * A long document still has a usable topic signal in its head. */
		n_tok = n_ctx;
	}
	if (n_tok == 0) {
		free(toks);
		memset(out, 0, im->dim * sizeof(float));
		return 0;
	}

	llama_memory_clear(llama_get_memory(im->ctx), true);

	batch = llama_batch_init(n_tok, 0, 1);
	for (int i = 0; i < n_tok; i++) {
		batch.token[i]      = toks[i];
		batch.pos[i]        = i;
		batch.n_seq_id[i]   = 1;
		batch.seq_id[i][0]  = 0;
		batch.logits[i]     = 1; /* mean pooling needs every position */
	}
	batch.n_tokens = n_tok;

	if (llama_encode(im->ctx, batch) != 0) {
		llama_batch_free(batch);
		free(toks);
		return -1;
	}

	emb = llama_get_embeddings_seq(im->ctx, 0);
	if (!emb) {
		llama_batch_free(batch);
		free(toks);
		return -1;
	}

	memcpy(out, emb, im->dim * sizeof(float));
	llama_batch_free(batch);
	free(toks);

	/* L2-normalize: SimHash estimates angle only, so magnitude must not
	 * influence which side of a hyperplane a vector falls on. */
	for (size_t i = 0; i < im->dim; i++)
		norm += (double)out[i] * (double)out[i];
	norm = sqrt(norm);
	if (norm > 1e-12) {
		for (size_t i = 0; i < im->dim; i++)
			out[i] = (float)((double)out[i] / norm);
	}

	return 0;
}

sigil_embedder_t *sigil_embedder_llama(const char *gguf_path)
{
	sigil_embedder_t *e;
	llama_impl_t *im;
	struct llama_model_params mp;
	struct llama_context_params cp;

	llama_backend_init();

	im = calloc(1, sizeof(*im));
	if (!im)
		return NULL;

	mp = llama_model_default_params();
	im->model = llama_model_load_from_file(gguf_path, mp);
	if (!im->model) {
		free(im);
		return NULL;
	}

	cp = llama_context_default_params();
	cp.embeddings  = true;
	cp.pooling_type = LLAMA_POOLING_TYPE_MEAN; /* sentence-transformers default */
	cp.n_ctx       = 512;                      /* MiniLM's trained window */
	cp.n_batch     = 512;
	cp.n_ubatch    = 512;

	im->ctx = llama_init_from_model(im->model, cp);
	if (!im->ctx) {
		llama_model_free(im->model);
		free(im);
		return NULL;
	}

	im->dim = (size_t)llama_model_n_embd(im->model);
	snprintf(im->name, sizeof(im->name), "llama.cpp:%s",
	         strrchr(gguf_path, '/') ? strrchr(gguf_path, '/') + 1 : gguf_path);

	e = calloc(1, sizeof(*e));
	if (!e) {
		llama_free(im->ctx);
		llama_model_free(im->model);
		free(im);
		return NULL;
	}

	e->embed   = llama_embed;
	e->embed_batch = NULL;   /* no batch path; callers loop over embed() */
	e->dim     = llama_dim;
	e->name    = llama_name;
	e->destroy = llama_destroy;
	e->impl    = im;
	return e;
}

#else /* !SIGIL_WITH_LLAMA */

sigil_embedder_t *sigil_embedder_llama(const char *gguf_path)
{
	(void)gguf_path;
	return NULL;
}

#endif

/* ---------------------------------------------------------------------------
 * Non-semantic fallback.
 *
 * Deliberately named so it cannot be mistaken for a real embedder. Produces a
 * stable vector from byte shingles: useful for exercising the plumbing and for
 * platforms with no model, useless for meaning.
 * ------------------------------------------------------------------------ */

typedef struct {
	size_t dim;
} hash_impl_t;

static size_t hash_dim(const sigil_embedder_t *self)
{
	return ((const hash_impl_t *)self->impl)->dim;
}

static const char *hash_name(const sigil_embedder_t *self)
{
	(void)self;
	return "hash-nonsemantic";
}

static void hash_destroy(sigil_embedder_t *self)
{
	free(self->impl);
	free(self);
}

static int hash_embed(sigil_embedder_t *self, const char *text, size_t len,
                      float *out)
{
	hash_impl_t *im = self->impl;
	double norm = 0.0;

	for (size_t i = 0; i < im->dim; i++)
		out[i] = 0.0f;

	for (size_t i = 0; i + 3 < len; i++) {
		uint32_t h = 2166136261u;

		for (int k = 0; k < 4; k++) {
			h ^= (uint8_t)text[i + (size_t)k];
			h *= 16777619u;
		}
		out[h % im->dim] += 1.0f;
	}

	for (size_t i = 0; i < im->dim; i++)
		norm += (double)out[i] * (double)out[i];
	if (norm > 1e-12) {
		norm = 1.0 / __builtin_sqrt(norm);
		for (size_t i = 0; i < im->dim; i++)
			out[i] = (float)((double)out[i] * norm);
	}
	return 0;
}

sigil_embedder_t *sigil_embedder_hash_nonsemantic(size_t dim)
{
	sigil_embedder_t *e = calloc(1, sizeof(*e));
	hash_impl_t *im = calloc(1, sizeof(*im));

	if (!e || !im) {
		free(e);
		free(im);
		return NULL;
	}
	im->dim   = dim;
	e->embed   = hash_embed;
	e->embed_batch = NULL;
	e->dim     = hash_dim;
	e->name    = hash_name;
	e->destroy = hash_destroy;
	e->impl    = im;
	return e;
}

/* ---------------------------------------------------------------------------
 * OpenVINO fallback stub.
 *
 * The real backend is src/embed_openvino.cpp, which is compiled only when the
 * build finds OpenVINO. Its own #else stub therefore disappears along with the
 * file, and callers fail to link -- cmd/bridge.c references this symbol
 * unconditionally, because which backend to use is a runtime decision made on
 * whether -e names a directory or a .gguf.
 *
 * So the stub has to live in a translation unit that is always built. Here.
 * ------------------------------------------------------------------------ */
#ifndef SIGIL_WITH_OPENVINO
sigil_embedder_t *sigil_embedder_openvino(const char *model_dir,
                                          const char *device)
{
	(void)model_dir;
	(void)device;
	return NULL;
}
#endif
