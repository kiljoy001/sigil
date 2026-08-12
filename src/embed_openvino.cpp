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
#include "sigil_utf8.h"

#ifdef SIGIL_WITH_OPENVINO

#include <openvino/openvino.hpp>

#include <cstdint>
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
	/* Hold the tokenizer's CompiledModel, not just its InferRequest.
	 *
	 * This was a local in the constructor, so it was destroyed as soon as
	 * the embedder was built while the InferRequest kept being used. The
	 * embedding model above keeps its CompiledModel; the tokenizer did
	 * not, and the asymmetry was the whole bug -- reads landing a
	 * constant 1,672 bytes below the mapping of openvino_tokenizer.bin,
	 * intermittently, once enough allocation had happened for the freed
	 * region to be unmapped. */
	ov::CompiledModel tok_model;
	ov::InferRequest tok;
	bool have_tok = false;

	/* Tensors handed to an InferRequest must outlive every call that may
	 * read them.
	 *
	 * set_input_tensor() and set_tensor() store a reference, not a copy.
	 * These were stack locals, destroyed at the end of the call while the
	 * request kept pointing at their buffers -- which is why the fault
	 * landed a constant 1,672 bytes below a freed mapping, and why it
	 * needed enough allocation churn to become fatal. The documented
	 * Python path never hits this because it passes data by value.
	 *
	 * Holding them on the embedder means they are also reused rather than
	 * reallocated per batch. */
	ov::Tensor t_in, t_ids, t_msk, t_tt;
	/* Whether the model declares token_type_ids, decided once.
	 *
	 * This used to be a scan of model.inputs() inside the per-chunk loop,
	 * calling get_any_name() on every input of every paragraph. That
	 * constructs a std::string and resolves dynamic symbols each time:
	 * measured at 1.45 ms per paragraph, which was 204/s versus 290/s
	 * without it -- more than the model inference itself. */
	bool need_token_type = false;
};

/*
 * Repair invalid UTF-8 before it reaches the tokenizer.
 *
 * The implementation lives in src/utf8_repair.c -- plain C, always compiled,
 * so it is covered by CI on machines with no OpenVINO, and checked against
 * tools/clean.py's Python implementation by a differential property test
 * (tools/tests/test_utf8_differential.py) so the two cannot drift.
 *
 * Why it is needed at all: openvino_tokenizers runs PCRE2 in UTF-8 mode
 * without PCRE2_MATCH_INVALID_UTF, and matching invalid UTF is documented
 * undefined behaviour -- it read wild memory and segfaulted the indexer,
 * intermittently, because whether the read faults depends on ASLR. See
 * docs/FINDINGS.md.
 */
static std::string to_valid_utf8(const char *text, size_t len)
{
	/* Fast path: the overwhelmingly common case is text that is already
	 * valid, and it must cost one scan and no copy. */
	if (sigil_utf8_valid(text, len))
		return std::string(text, len);

	size_t n = 0;
	char *fixed = sigil_utf8_repair(text, len, &n);
	if (fixed == nullptr)
		return std::string();      /* OOM: empty beats malformed */

	std::string out(fixed, n);
	std::free(fixed);
	return out;
}

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
	im->t_in = ov::Tensor(ov::element::string, ov::Shape{1});
	im->t_in.data<std::string>()[0] = to_valid_utf8(text, len);
	im->tok.set_input_tensor(im->t_in);
	im->tok.infer();

	/* Copy out; see the note in ov_embed_batch. */
	std::vector<int64_t> id_buf, mk_buf;
	{
		ov::Tensor ti = im->tok.get_tensor("input_ids");
		ov::Tensor tm = im->tok.get_tensor("attention_mask");
		size_t tn = ti.get_shape()[1];

		id_buf.assign(ti.data<int64_t>(), ti.data<int64_t>() + tn);
		mk_buf.assign(tm.data<int64_t>(), tm.data<int64_t>() + tn);
	}
	size_t ntok = id_buf.size();

	// Chunk over the position limit, pooling the pieces. Averaging float
	// vectors then renormalizing is the correct combiner: pooling packed
	// codes would destroy the locality that makes them useful.
	size_t nchunk = (ntok + Maxtokens - 1) / Maxtokens;
	if (nchunk == 0)
		return -1;

	std::vector<float> acc(im->dim, 0.0f);
	const int64_t *idp = id_buf.data();
	const int64_t *mkp = mk_buf.data();

	for (size_t c = 0; c < nchunk; c++) {
		size_t lo = c * Maxtokens;
		size_t hi = std::min(lo + Maxtokens, ntok);
		size_t n = hi - lo;

		ov::Tensor &cid = im->t_ids, &cmk = im->t_msk;

		cid = ov::Tensor(ov::element::i64, ov::Shape{1, n});
		cmk = ov::Tensor(ov::element::i64, ov::Shape{1, n});
		std::memcpy(cid.data<int64_t>(), idp + lo, n * sizeof(int64_t));
		std::memcpy(cmk.data<int64_t>(), mkp + lo, n * sizeof(int64_t));

		im->req.set_tensor("input_ids", cid);
		im->req.set_tensor("attention_mask", cmk);
		// Some exports also want token_type_ids; feed zeros when the
		// model declares it rather than failing the whole embed.
		if (im->need_token_type) {
			im->t_tt = ov::Tensor(ov::element::i64,
			                      ov::Shape{1, n});
			std::memset(im->t_tt.data<int64_t>(), 0,
			            n * sizeof(int64_t));
			im->req.set_tensor("token_type_ids", im->t_tt);
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
		im->t_in = ov::Tensor(ov::element::string, ov::Shape{n});
		for (size_t i = 0; i < n; i++)
			im->t_in.data<std::string>()[i] =
				to_valid_utf8(texts[i], lens[i]);
		im->tok.set_input_tensor(im->t_in);
		im->tok.infer();

		/* Copy the tokenizer's output out immediately.
		 *
		 * get_tensor() returns a view into the request's own buffer,
		 * which the next inference on that request -- or on another
		 * sharing the runtime's allocator -- may reuse or reallocate.
		 * The documented Python path copies into numpy at exactly this
		 * point; holding the view instead is the difference between
		 * the two, and the Python path survives the corpus that kills
		 * this one. */
		std::vector<int64_t> wid_buf, wmsk_buf;
		size_t WT;
		{
			ov::Tensor wi = im->tok.get_tensor("input_ids");
			ov::Tensor wm = im->tok.get_tensor("attention_mask");

			WT = wi.get_shape()[1];
			wid_buf.assign(wi.data<int64_t>(),
			               wi.data<int64_t>() + n * WT);
			wmsk_buf.assign(wm.data<int64_t>(),
			                wm.data<int64_t>() + n * WT);
		}
		const int64_t *wids_p = wid_buf.data();
		const int64_t *wmsk_p = wmsk_buf.data();

		/* One row per chunk; owner[r] is the text row r belongs to. */
		std::vector<size_t> owner, start, count;
		for (size_t i = 0; i < n; i++) {
			const int64_t *m = wmsk_p + i * WT;
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

		ov::Tensor &cid = im->t_ids, &cmk = im->t_msk;

		cid = ov::Tensor(ov::element::i64, ov::Shape{R, Tuse});
		cmk = ov::Tensor(ov::element::i64, ov::Shape{R, Tuse});
		std::memset(cid.data<int64_t>(), 0, R * Tuse * sizeof(int64_t));
		std::memset(cmk.data<int64_t>(), 0, R * Tuse * sizeof(int64_t));
		for (size_t r = 0; r < R; r++) {
			std::memcpy(cid.data<int64_t>() + r * Tuse,
			            wids_p + start[r],
			            count[r] * sizeof(int64_t));
			for (size_t t = 0; t < count[r]; t++)
				cmk.data<int64_t>()[r * Tuse + t] = 1;
		}
		im->req.set_tensor("input_ids", cid);
		im->req.set_tensor("attention_mask", cmk);
		if (im->need_token_type) {
			im->t_tt = ov::Tensor(ov::element::i64,
			                      ov::Shape{R, Tuse});
			std::memset(im->t_tt.data<int64_t>(), 0,
			            R * Tuse * sizeof(int64_t));
			im->req.set_tensor("token_type_ids", im->t_tt);
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

		// SIGIL_OV_THREADS pins the TBB inference pool. Diagnostic
		// knob for the indexing crash: the crash is a race (8/12 on
		// identical input), and a pool of one worker cannot race with
		// itself. Applied to CPU compiles only -- the GPU plugin
		// rejects the property.
		ov::AnyMap cpu_props;
		{
			const char *nt = std::getenv("SIGIL_OV_THREADS");
			if (nt != nullptr && nt[0] != '\0')
				cpu_props.emplace(
					ov::inference_num_threads.name(),
					std::atoi(nt));
		}

		std::string dir(model_dir);
		im->model = (im->device.find("CPU") != std::string::npos)
			? core.compile_model(dir + "/openvino_model.xml",
			                     im->device, cpu_props)
			: core.compile_model(dir + "/openvino_model.xml",
			                     im->device);
		im->req = im->model.create_infer_request();

		// Tokenizer is optional only in the sense that its absence is
		// fatal here rather than silently producing garbage.
		try {
			im->tok_model = core.compile_model(
				dir + "/openvino_tokenizer.xml", "CPU",
				cpu_props);
			im->tok = im->tok_model.create_infer_request();
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
