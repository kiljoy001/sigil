/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * libtab persistence.
 *
 * The store on disk is ndb-shaped text: greppable, diffable, editable. Its
 * schema tuple carries the parameters that decide whether two stores are
 * comparable at all -- model, embedding width, LSH width, hyperplane seed. A
 * store built under different parameters holds bits that mean something else,
 * so opening one is refused rather than silently returning wrong answers.
 *
 * libtab is a serialization format here, not a query engine: its own header
 * says search is a deliberate linear scan with no secondary indexes. The whole
 * table is loaded into the in-memory store at startup, and the SIMD scan is
 * the index.
 */

#include <u.h>
#include <libc.h>
#include <fcall.h>
#include <thread.h>
#include <9p.h>
#include <libtab.h>

#include "sigilfs.h"

/* Reserved path marking the parameter row. Not a real path, and skipped on
 * load so it never becomes a record. */
#define Paramrow "!params"

static char *
attr(Tab *t, char *col, char *key)
{
	const char *v = tab_col_attr(t, col, key);

	return (char*)v;
}

static char *
checkparams(Sigilfs *f, Tab *t)
{
	static char err[256];
	TabIter *it;
	TabRow *r;
	char *v;

	/* Parameters live in a reserved row with para=0 and path="!params",
	 * not in the schema tuple: libtab validates column types against a
	 * fixed set (HASHED, SIGNED) and rejects anything else, and TabColSpec
	 * has no field for arbitrary attributes. A row is what the format is
	 * for. */
	it = tab_iter(t);
	while((r = tab_iter_next(it)) != nil){
		const char *p = tab_get(r, "path");

		if(p == nil || strcmp(p, Paramrow) != 0)
			continue;
		v = (char*)tab_get(r, "hash");        /* model */
		if(v != nil && strcmp(v, f->model_id) != 0){
			snprint(err, sizeof err,
				"store built with model \"%s\", running \"%s\" -- "
				"bits from different models are not comparable",
				v, f->model_id);
			tab_iter_close(it);
			return err;
		}
		v = (char*)tab_get(r, "para");        /* lsh bits */
		if(v != nil && atoi(v) != f->lsh_bits){
			snprint(err, sizeof err,
				"store has %s-bit LSH, running %d-bit",
				v, f->lsh_bits);
			tab_iter_close(it);
			return err;
		}
		v = (char*)tab_get(r, "lsh");         /* seed */
		if(v != nil && strtoull(v, nil, 16) != f->simhash_seed){
			snprint(err, sizeof err,
				"store seed %s != running %#llux -- "
				"different hyperplanes", v, f->simhash_seed);
			tab_iter_close(it);
			return err;
		}
		break;
	}
	tab_iter_close(it);
	return nil;
}

char *
store_load(Sigilfs *f)
{
	Tab *t;
	TabIter *it;
	TabRow *r;
	char *err;
	long n = 0;

	if(f->storepath == nil)
		return nil;                    /* memory-only is legitimate */

	if(access(f->storepath, AEXIST) < 0)
		return nil;                    /* first run: nothing to load */

	if((t = tab_open(f->storepath)) == nil)
		return (char*)tab_lasterror();

	if((err = checkparams(f, t)) != nil){
		tab_close(t);
		return err;
	}

	it = tab_iter(t);
	while((r = tab_iter_next(it)) != nil){
		const char *path = tab_get(r, "path");
		const char *para = tab_get(r, "para");
		const char *lsh  = tab_get(r, "lsh");
		const char *hash = tab_get(r, "hash");
		const char *off  = tab_get(r, "off");
		const char *len  = tab_get(r, "len");

		if(path == nil || strcmp(path, Paramrow) == 0)
			continue;
		/* Rows carry the LSH already computed; re-embedding on load
		 * would be both slow and non-deterministic across model
		 * versions. */
		if(br_add_restore(f->store, (char*)path, para ? atoi(para) : 0,
		                  (char*)(lsh ? lsh : ""),
		                  (char*)(hash ? hash : ""),
		                  off ? strtoul(off, nil, 10) : 0,
		                  len ? strtoul(len, nil, 10) : 0) >= 0)
			n++;
	}
	tab_iter_close(it);
	tab_close(t);
	f->nrecords = n;
	return nil;
}

char *
store_commit(Sigilfs *f)
{
	TabColSpec cols[6];
	Tab *t;
	TabRow *r;
	char buf[64], hex[160], bitsbuf[16], seedbuf[32];
	long i, n;

	if(f->storepath == nil)
		return "no store path: started without -f";

	/* A queued paragraph still carries fallback bits; committing here
	 * would persist them as though they were semantic. */
	br_flush(f->store);

	memset(cols, 0, sizeof cols);
	cols[0].name = "path";
	cols[1].name = "para";
	cols[2].name = "hash";
	cols[3].name = "lsh";
	/* Where the paragraph sits in its source, so a reloaded neighbour can
	 * still be read. Without these the store round-trips as identities and
	 * bits but the text becomes unreachable after a restart. */
	cols[4].name = "off";
	cols[5].name = "len";

	if((t = tab_create(f->storepath, "sigil", cols, 6)) == nil)
		return (char*)tab_lasterror();

	/* Parameter row first, so a reader hits it before any data. */
	if((r = tab_add_row(t, "path", Paramrow)) != nil){
		snprint(buf, sizeof buf, "%d", f->lsh_bits);
		tab_set(t, r, "para", buf);
		tab_set(t, r, "hash", f->model_id);
		snprint(buf, sizeof buf, "%llux", f->simhash_seed);
		tab_set(t, r, "lsh", buf);
	}

	n = br_count(f->store);
	for(i = 0; i < n; i++){
		const char *p = br_path(f->store, i);

		if(p == nil)
			continue;
		if((r = tab_add_row(t, "path", (char*)p)) == nil)
			continue;
		snprint(buf, sizeof buf, "%ud", br_para(f->store, i));
		tab_set(t, r, "para", buf);
		snprint(buf, sizeof buf, "%lud", br_offset(f->store, i));
		tab_set(t, r, "off", buf);
		snprint(buf, sizeof buf, "%lud", br_length(f->store, i));
		tab_set(t, r, "len", buf);
		if(br_hash(f->store, i, hex, sizeof hex) == 0)
			tab_set(t, r, "hash", hex);
		if(br_lsh_hex(f->store, i, hex, sizeof hex) == 0)
			tab_set(t, r, "lsh", hex);
	}

	if(tab_commit(t) < 0){
		char *e = (char*)tab_lasterror();
		tab_close(t);
		return e;
	}
	tab_close(t);
	return nil;
}
