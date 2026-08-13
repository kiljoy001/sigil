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
	char *modelpath;  /* GGUF, nil = no embedder (bits are not semantic) */
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
	uvlong nindexed;   /* files read, not paragraphs */
	/* Files passed over because they exceed Maxfile. Reported in
	 * /stats because the first full-corpus run skipped 128 books --
	 * 3.0M paragraphs, 4% of the index -- with no indication at all,
	 * and the loss was only found by reconciling two counts that
	 * disagreed for unrelated reasons. */
	uvlong nskipped;
	int thresh;        /* Hamming radius for /similar */
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

/* store.c -- setup */
void sigilfs_init(Sigilfs *f, char *storepath);

/* persist.c -- libtab load and commit */
char *store_load(Sigilfs *f);
char *store_commit(Sigilfs *f);

/* similar.c -- the /similar/<hex>/ projection */
char *similar_walk(Sigilfs *f, char *hex, File **out);
int   similar_isleaf(File *f);
void  similar_read(Req *r);
void  similar_flush(Sigilfs *f);

/* index.c -- walk a mount and add paragraph records */
char *index_mount(Sigilfs *f, Mount *m);

/* bridge.c -- compiled with the system C compiler, not 9c: plan9port's
 * libc.h and C11's stdint.h collide, so libsigil is reached only through
 * these plain-typed entry points. */
void *br_new(void);
void br_free(void *b);
long br_add(void *b, const char *path, unsigned para, const void *text,
            unsigned long len, const unsigned long long *lsh, unsigned ts);
long br_add_at(void *b, const char *path, unsigned para, const void *text,
               unsigned long len, const unsigned long long *lsh, unsigned ts,
               unsigned long off);
/* Embed anything queued by br_add_at and write the bits into the store.
 * Must run before any read of the LSH field -- a scan, a similarity query, or
 * a commit -- or those records still carry the byte-shingle fallback. */
void br_flush(void *b);
unsigned long br_offset(void *b, long i);
unsigned long br_length(void *b, long i);
long br_add_hex(void *b, const char *path, unsigned para, const char *lshhex);
long br_add_restore(void *b, const char *path, unsigned para,
                    const char *lshhex, const char *hashhex,
                    unsigned long off, unsigned long len);
long br_count(void *b);
const char *br_path(void *b, long i);
unsigned br_para(void *b, long i);
int br_hash(void *b, long i, char *out, unsigned long outlen);
int br_lsh_hex(void *b, long i, char *out, unsigned long outlen);
long br_find_hash(void *b, const char *hexhash);
long br_similar(void *b, long i, unsigned maxdist, unsigned *out, long maxout);
int br_embedder_load(void *b, const char *gguf, uvlong seed);
int br_have_embedder(void *b);
unsigned br_embed_dim(void *b);
unsigned br_lsh_bits(void);
const char *br_embedder_name(void *b);

#endif /* SIGILFS_H */
