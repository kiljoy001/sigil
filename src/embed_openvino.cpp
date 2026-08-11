// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Scott J Guyton

//
// OpenVINO embedding backend.
//
// The llama.cpp backend is portable and works everywhere, but on Intel Arc it
// is both slower and, above roughly 4B parameters, wrong -- it garbles output
// on Battlemage through its SYCL and Vulkan backends alike (upstream 20169,
// closed *not planned*, and 24560, 21888). MiniLM is small enough to be
// unaffected by the correctness bug, but it still runs on the CPU here because
// llama.cpp was built without GPU support.
//
// Measured on an Arc Pro B50, all-MiniLM-L6-v2, real Gutenberg paragraphs:
//
//     OpenVINO, GPU.1      515 paragraphs/s
//     ollama / Vulkan      188
//     OpenVINO, CPU         96
//
// Vectors from this backend agree with the llama.cpp path at mean cosine
// 0.954; the residual is entirely long paragraphs, which the two chunk
// differently. That is close enough that stores are comparable in practice,
// but not identical -- so `name()` reports the backend, and a store built with
// one should not be assumed interchangeable with the other.
//
// Convert a model first:
//
//   optimum-cli export openvino -m sentence-transformers/all-MiniLM-L6-v2 \
//       --task feature-extraction /models/minilm
//
// Build with -DSIGIL_WITH_OPENVINO and link -lopenvino.
//

#include "sigil_embed.h"

#ifdef SIGIL_WITH_OPENVINO

#include <openvino/openvino.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

// MiniLM's trained position limit. Longer text is chunked and pooled rather
// than truncated: a paragraph's length is a property of the paragraph, not of
// the model's context window, and truncation discards content silently.
constexpr size_t Maxtokens = 256;

struct Impl {
	ov::CompiledModel model;
	ov::InferRequest req;
	std::string device;
	std::string label;
	size_t dim = 0;
	// The tokenizer is a separate OpenVINO model produced by the same
	// export. Keeping it in-process avoids a Python dependency in the
	// server, which is the whole point of doing this in C++.
	ov::InferRequest tok;
	bool have_tok = false;
	/* Whether the model declares token_type_ids, decided once.
	 *
	 * This used to be a scan of model.inputs() inside the per-chunk loop,
	 * calling get_any_name() on every input of every paragraph. That
	 * constructs a std::string and resolves dynamic symbols each time:
	 * measured at 1.45 ms per paragraph, which was 204/s versus 290/s
	 * without it -- more than the model inference itself. */
	bool need_token_type = false;
};

int embed_one(Impl *im, const char *text, size_t len, float *out);

int ov_embed(sigil_embedder_t *self, const char *text, size_t len, float *out)
{
	Impl *im = static_cast<Impl *>(self->impl);

	try {
		return embed_one(im, text, len, out);
	} catch (const std::exception &) {
		// A failed embed leaves the caller's buffer untouched; the
		// bridge keeps the byte-shingle fallback rather than dropping
		// the record. A paragraph that will not embed is still worth
		// addressing by content.
		return -1;
	}
}

int embed_one(Impl *im, const char *text, size_t len, float *out)
{
	if (!im->have_tok)
		return -1;

	// Tokenize. The exported tokenizer takes a string tensor and returns
	// input_ids and attention_mask.
	std::string s(text, len);
	ov::Tensor in(ov::element::string, ov::Shape{1});
	in.data<std::string>()[0] = s;
	im->tok.set_input_tensor(in);
	im->tok.infer();

	ov::Tensor ids  = im->tok.get_tensor("input_ids");
	ov::Tensor mask = im->tok.get_tensor("attention_mask");
	size_t ntok = ids.get_shape()[1];

	// Chunk over the position limit, pooling the pieces. Averaging float
	// vectors then renormalizing is the correct combiner: pooling packed
	// codes would destroy the locality that makes them useful.
	size_t nchunk = (ntok + Maxtokens - 1) / Maxtokens;
	if (nchunk == 0)
		return -1;

	std::vector<float> acc(im->dim, 0.0f);
	const int64_t *idp = ids.data<int64_t>();
	const int64_t *mkp = mask.data<int64_t>();

	for (size_t c = 0; c < nchunk; c++) {
		size_t lo = c * Maxtokens;
		size_t hi = std::min(lo + Maxtokens, ntok);
		size_t n = hi - lo;

		ov::Tensor cid(ov::element::i64, ov::Shape{1, n});
		ov::Tensor cmk(ov::element::i64, ov::Shape{1, n});
		std::memcpy(cid.data<int64_t>(), idp + lo, n * sizeof(int64_t));
		std::memcpy(cmk.data<int64_t>(), mkp + lo, n * sizeof(int64_t));

		im->req.set_tensor("input_ids", cid);
		im->req.set_tensor("attention_mask", cmk);
		// Some exports also want token_type_ids; feed zeros when the
		// model declares it rather than failing the whole embed.
		if (im->need_token_type) {
			ov::Tensor tt(ov::element::i64, ov::Shape{1, n});
			std::memset(tt.data<int64_t>(), 0, n * sizeof(int64_t));
			im->req.set_tensor("token_type_ids", tt);
		}
		im->req.infer();

		ov::Tensor hs = im->req.get_output_tensor(0);
		const float *h = hs.data<const float>();
		size_t dim = hs.get_shape()[2];

		// Mean pool over real tokens only; padding must not dilute.
		double tot = 0.0;
		std::vector<float> v(dim, 0.0f);
		for (size_t t = 0; t < n; t++) {
			float m = static_cast<float>(cmk.data<int64_t>()[t]);
			if (m == 0.0f)
				continue;
			tot += m;
			for (size_t d = 0; d < dim; d++)
				v[d] += h[t * dim + d] * m;
		}
		if (tot <= 0.0)
			continue;
		for (size_t d = 0; d < dim && d < acc.size(); d++)
			acc[d] += v[d] / static_cast<float>(tot);
	}

	// L2 normalize: SimHash estimates angle only, so magnitude must not
	// leak into the bits.
	double norm = 0.0;
	for (size_t d = 0; d < im->dim; d++)
		norm += static_cast<double>(acc[d]) * acc[d];
	norm = norm > 0.0 ? 1.0 / std::sqrt(norm) : 0.0;
	for (size_t d = 0; d < im->dim; d++)
		out[d] = static_cast<float>(acc[d] * norm);
	return norm > 0.0 ? 0 : -1;
}

/*
 * Embed n texts in one inference.
 *
 * Chunking happens inside the batch rather than beside it. A text over the
 * position limit contributes several rows to the same inference, and its rows
 * are averaged back together afterwards -- the same pool-then-renormalise the
 * per-text path does, without falling out of the batch to do it.
 *
 * Truncating instead was measurably wrong: of 300 real paragraphs, the 7 over
 * ~900 characters diverged from the per-text path by as much as 0.56 cosine
 * while the other 293 matched within 0.001. A batch that silently drops the
 * tail of a paragraph produces a vector for text the caller never asked about.
 *
 * The tokenizer pads to the longest member, so a batch mixing a 40-character
 * paragraph with a 3000-character one wastes work on the short one. It still
 * wins by a wide margin: the fixed cost of an inference call dominates a
 * single median-length paragraph, and amortising it over 32 takes an Arc Pro
 * B50 from 200 paragraphs/s to over 600.
 */
int ov_embed_batch(sigil_embedder_t *self, const char **texts,
                   const size_t *lens, size_t n, float *out)
{
	Impl *im = static_cast<Impl *>(self->impl);

	if (!im->have_tok || n == 0)
		return -1;
	try {
		/* Tokenize whole texts first, then split the *token* sequence.
		 * Splitting characters instead gives different boundaries than
		 * embed_one(), which chunks after tokenizing -- that mismatch
		 * showed up as 23 of 300 paragraphs disagreeing with the
		 * per-text path instead of 0. */
		ov::Tensor win(ov::element::string, ov::Shape{n});
		for (size_t i = 0; i < n; i++)
			win.data<std::string>()[i] = std::string(texts[i], lens[i]);
		im->tok.set_input_tensor(win);
		im->tok.infer();

		ov::Tensor wids = im->tok.get_tensor("input_ids");
		ov::Tensor wmsk = im->tok.get_tensor("attention_mask");
		size_t WT = wids.get_shape()[1];

		/* One row per chunk; owner[r] is the text row r belongs to. */
		std::vector<size_t> owner, start, count;
		for (size_t i = 0; i < n; i++) {
			const int64_t *m = wmsk.data<int64_t>() + i * WT;
			size_t real = 0;
			while (real < WT && m[real] != 0)
				real++;
			if (real == 0)
				real = 1;
			for (size_t off = 0; off < real; off += Maxtokens) {
				size_t len = real - off;
				if (len > Maxtokens)
					len = Maxtokens;
				owner.push_back(i);
				start.push_back(i * WT + off);
				count.push_back(len);
			}
		}

		size_t R = owner.size();
		size_t Tuse = 0;
		for (size_t r = 0; r < R; r++)
			if (count[r] > Tuse)
				Tuse = count[r];

		ov::Tensor cid(ov::element::i64, ov::Shape{R, Tuse});
		ov::Tensor cmk(ov::element::i64, ov::Shape{R, Tuse});
		std::memset(cid.data<int64_t>(), 0, R * Tuse * sizeof(int64_t));
		std::memset(cmk.data<int64_t>(), 0, R * Tuse * sizeof(int64_t));
		for (size_t r = 0; r < R; r++) {
			std::memcpy(cid.data<int64_t>() + r * Tuse,
			            wids.data<int64_t>() + start[r],
			            count[r] * sizeof(int64_t));
			for (size_t t = 0; t < count[r]; t++)
				cmk.data<int64_t>()[r * Tuse + t] = 1;
		}
		im->req.set_tensor("input_ids", cid);
		im->req.set_tensor("attention_mask", cmk);
		if (im->need_token_type) {
			ov::Tensor tt(ov::element::i64, ov::Shape{R, Tuse});
			std::memset(tt.data<int64_t>(), 0,
			            R * Tuse * sizeof(int64_t));
			im->req.set_tensor("token_type_ids", tt);
		}
		im->req.infer();

		ov::Tensor hs = im->req.get_output_tensor(0);
		const float *h = hs.data<const float>();
		size_t dim = hs.get_shape()[2];

		std::memset(out, 0, n * dim * sizeof *out);
		std::vector<float> nchunk(n, 0.0f);

		for (size_t r = 0; r < R; r++) {
			const float *hr = h + r * Tuse * dim;
			const int64_t *mr = cmk.data<int64_t>() + r * Tuse;
			float *o = out + owner[r] * dim;
			double tot = 0.0;
			std::vector<float> v(dim, 0.0f);

			/* Mean pool over real tokens only; padding must not
			 * dilute, and the batch is padded to its longest row. */
			for (size_t t = 0; t < Tuse; t++) {
				if (mr[t] == 0)
					continue;
				tot += 1.0;
				for (size_t d = 0; d < dim; d++)
					v[d] += hr[t * dim + d];
			}
			if (tot <= 0.0)
				continue;
			for (size_t d = 0; d < dim; d++)
				o[d] += static_cast<float>(v[d] / tot);
			nchunk[owner[r]] += 1.0f;
		}

		/* Average a text's chunks, then renormalise. Averaging in
		 * float space is the correct combiner: pooling packed codes
		 * would destroy the locality that makes them useful. */
		for (size_t i = 0; i < n; i++) {
			float *o = out + i * dim;
			double norm = 0.0;

			if (nchunk[i] <= 0.0f)
				continue;
			for (size_t d = 0; d < dim; d++) {
				o[d] /= nchunk[i];
				norm += static_cast<double>(o[d]) * o[d];
			}
			norm = norm > 0.0 ? 1.0 / std::sqrt(norm) : 0.0;
			for (size_t d = 0; d < dim; d++)
				o[d] = static_cast<float>(o[d] * norm);
		}
		return static_cast<int>(n);
	} catch (const std::exception &e) {
		std::fprintf(stderr, "sigil: openvino batch: %s\n", e.what());
		return -1;
	}
}

size_t ov_dim(const sigil_embedder_t *self)
{
	return static_cast<const Impl *>(self->impl)->dim;
}

const char *ov_name(const sigil_embedder_t *self)
{
	return static_cast<const Impl *>(self->impl)->label.c_str();
}

void ov_destroy(sigil_embedder_t *self)
{
	if (self == nullptr)
		return;
	delete static_cast<Impl *>(self->impl);
	delete self;
}

}  // namespace

extern "C" sigil_embedder_t *
sigil_embedder_openvino(const char *model_dir, const char *device)
{
	try {
		auto im = std::make_unique<Impl>();
		ov::Core core;

		im->device = (device != nullptr && device[0] != '\0')
		             ? device : "AUTO";

		// The tokenizer model uses custom ops (SpecialTokensSplit and
		// friends) that live in a separate extension library. Without
		// registering it the tokenizer fails to deserialize with
		// "unsupported opset: extension" while the embedding model
		// itself loads perfectly -- an easy failure to misattribute.
		//
		// SIGIL_OV_TOKENIZERS overrides the path; the default is where
		// the pip package puts it relative to the runtime.
		{
			const char *ext = std::getenv("SIGIL_OV_TOKENIZERS");
			try {
				core.add_extension(ext != nullptr ? ext :
					"libopenvino_tokenizers.so");
			} catch (const std::exception &) {
				// Absent extension is only fatal once the
				// tokenizer is actually loaded, below.
			}
		}

		std::string dir(model_dir);
		im->model = core.compile_model(dir + "/openvino_model.xml",
		                               im->device);
		im->req = im->model.create_infer_request();

		// Tokenizer is optional only in the sense that its absence is
		// fatal here rather than silently producing garbage.
		try {
			auto tk = core.compile_model(
				dir + "/openvino_tokenizer.xml", "CPU");
			im->tok = tk.create_infer_request();
			im->have_tok = true;
		} catch (const std::exception &e) {
			// The tokenizer is the usual failure and the message
			// names the cause -- "unsupported opset: extension"
			// when libopenvino_tokenizers.so was not registered.
			// Returning a bare nullptr here cost an hour once.
			std::fprintf(stderr,
				"sigil: openvino tokenizer: %s\n", e.what());
			return nullptr;
		}

		// Output shape is [batch, tokens, dim].
		auto shape = im->model.output(0).get_partial_shape();
		if (shape.rank().get_length() < 3 ||
		    shape[2].is_dynamic())
			return nullptr;
		im->dim = static_cast<size_t>(shape[2].get_length());

		for (const auto &p : im->model.inputs()) {
			if (p.get_any_name() == "token_type_ids") {
				im->need_token_type = true;
				break;
			}
		}

		im->label = "openvino:" + im->device;

		auto *e = new sigil_embedder_t;
		e->embed       = ov_embed;
		e->embed_batch = ov_embed_batch;
		e->dim     = ov_dim;
		e->name    = ov_name;
		e->destroy = ov_destroy;
		e->impl    = im.release();
		return e;
	} catch (const std::exception &e) {
		std::fprintf(stderr, "sigil: openvino load: %s\n", e.what());
		return nullptr;
	}
}

#endif  /* SIGIL_WITH_OPENVINO -- the not-built stub lives in
         * src/embed_llama.c, which is always compiled. Keeping it here
         * instead meant the symbol vanished with the file and every caller
         * failed to link. */
