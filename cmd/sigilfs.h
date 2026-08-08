/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Shared state for sigilfs.
 *
 * Deliberately does not include sigil.h: libsigil is C11 with stdint.h, and
 * plan9port's libc.h defines its own uchar/ushort/uvlong and collides. The
 * bridge in store.c is the only file that sees both, and it does so by
 * including sigil.h alone.
 */

#ifndef SIGILFS_H
#define SIGILFS_H

typedef struct Sigilfs Sigilfs;
typedef struct Mount Mount;

/*
 * A mounted source. Runtime state only, never persisted: this is "what am I
 * currently serving", rebuilt from /ctl at startup. Writing it to disk would
 * only create a file that drifts out of sync with reality.
 */
struct Mount {
	char *name;
	char *root;
	long lastscan;    /* unix seconds, 0 = never */
	Mount *next;
};

struct Sigilfs {
	Srv srv;

	File *ctl;
	File *stats;
	File *similar;

	char *storepath;  /* libtab file, nil = in-memory only */
	void *store;      /* opaque sigil_store_t*, see store.c */

	Mount *mounts;    /* linked list; small and rarely walked */
	int nmounts;

	/* Parameters that decide whether two stores are comparable at all.
	 * Recorded in the libtab schema tuple; a mismatch must refuse to open
	 * rather than return wrong answers. */
	char *model_id;
	int embed_dim;
	int lsh_bits;
	uvlong simhash_seed;

	/* Counters surfaced through /stats. */
	uvlong nscans;
	uvlong nrecords;
};

/* sigilfs.c */
extern Sigilfs fs;

/* trace.c -- diagnostic log of every request and its outcome.
 * Enabled with -L; writes to stderr. Distinct from -D (lib9p's chatty9p),
 * which dumps wire messages but not which callback produced an error. */
extern int tracing;
void tracereq(Req *r, char *what, char *err);

/* fs.c -- 9P callbacks */
void fsread(Req *r);
void fswrite(Req *r);
void fswstat(Req *r);
void fsopen(Req *r);
void fscreate(Req *r);
void fsstat(Req *r);

/* ctl.c -- parse and apply a line written to /ctl */
char *ctlwrite(Sigilfs *f, char *line);
Mount *mountfind(Sigilfs *f, char *name);

/* store.c -- the bridge to libsigil */
void sigilfs_init(Sigilfs *f, char *storepath);
long sigilfs_count(Sigilfs *f);

#endif /* SIGILFS_H */
