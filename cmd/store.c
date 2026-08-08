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
	Defaultbits = 256,
};

void
sigilfs_init(Sigilfs *f, char *storepath)
{
	memset(f, 0, sizeof *f);
	f->storepath = storepath ? estrdup9p(storepath) : nil;
	f->model_id = estrdup9p("all-MiniLM-L6-v2");
	f->embed_dim = Defaultdim;
	f->lsh_bits = Defaultbits;
	f->simhash_seed = 0x5191c0de5191c0deULL;
	f->store = nil;      /* allocated when persistence lands */
	f->nrecords = 0;
}

long
sigilfs_count(Sigilfs *f)
{
	return f->nrecords;
}
