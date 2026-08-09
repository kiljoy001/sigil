/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * The bridge between sigilfs and libsigil.
 *
 * This is the only file that sees libsigil's headers. plan9port's libc.h and
 * C11's stdint.h both define the integer typedefs and collide, so the rest of
 * sigilfs talks to the store through the opaque pointer declared in
 * sigilfs.h.
 */

#include <u.h>
#include <libc.h>
#include <fcall.h>
#include <thread.h>
#include <9p.h>

#include "sigilfs.h"

/* Defaults matching what the measurements support. 256 bits rather than 128:
 * bit width is corpus dependent, and 128 retained only 57% of the ceiling on
 * the hardest corpus tested. See docs/FINDINGS.md. */
enum {
	Defaultdim  = 384,
	/* Hamming radius for /similar, as a percentage of the code width.
	 *
	 * A fixed radius is wrong because the useful distance scales with the
	 * number of bits: 60 is a tight default at 256 bits and admits half
	 * the store at 128. On a 141-record 128-bit store, 60 returned 71
	 * neighbours where 45 returned 5.
	 *
	 * 35% of the width. Unrelated text lands near half the width by
	 * construction -- SimHash codes of independent vectors disagree on
	 * about half their bits -- so this sits well inside that, and the
	 * judged-pair measurement in docs/FINDINGS.md put the useful radius at
	 * 90 of 256 bits, which is 35%. */
	Threshpercent = 35,
};

void
sigilfs_init(Sigilfs *f, char *storepath)
{
	memset(f, 0, sizeof *f);
	f->storepath = storepath ? estrdup9p(storepath) : nil;
	f->model_id = estrdup9p("all-MiniLM-L6-v2");
	f->embed_dim = Defaultdim;
	f->lsh_bits = (int)br_lsh_bits();
	f->simhash_seed = 0x5191c0de5191c0deULL;
	f->thresh = (f->lsh_bits * Threshpercent) / 100;
	f->store = br_new();
	if(f->store == nil)
		sysfatal("br_new: out of memory");
	f->nrecords = 0;
}
