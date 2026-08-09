/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * /similar/<hex>/ -- the semantic projection.
 *
 * Walking a directory runs a scan. No query language, no ioctl: the walk is
 * the query, which is the whole claim the name "semantic filesystem" makes.
 *
 *	/similar/<hex>/		a paragraph's neighbourhood, by BLAKE3
 *	/similar/<hex>/<hex>	a neighbour; read it for its text
 *
 * Neighbourhoods, not partitions. Clustering measured badly enough to keep
 * /class/ out of the namespace -- connected components at a similarity
 * threshold put 2018 of 5000 items in one bucket, and no threshold fixed it.
 * A neighbourhood makes no such claim: /similar/A/ lists what is near A, and
 * B appearing there does not put A and B in a shared group. The relation is
 * not transitive and is not closed, so a bridging paragraph joins two
 * neighbourhoods instead of merging them.
 *
 * Membership overlaps by design. A passage on Napoleon's retreat is genuinely
 * near war, russia and winter passages at once, and a file under several
 * directories is ordinary in Plan 9.
 *
 * The directory is materialized into the real File tree by writing the hash
 * to /ctl, not synthesized during a walk. lib9p dispatches straight to its own
 * filewalk() whenever the fid carries a File, and there is no hook before it,
 * so a walk cannot create what it is walking to -- the alternative is
 * abandoning the File tree entirely and hand-rolling walk1, which throws away
 * the reason lib9p was chosen.
 *
 * Making it an explicit /ctl verb is the better fit regardless: a scan is work,
 * and a shell completing a path should not silently launch one. read(5)
 * requires directory reads be sequential and resumable at exactly the previous
 * offset:
 *
 *	"The read request message must have offset equal to zero or the value
 *	 of offset in the previous read on the directory, plus the number of
 *	 bytes returned in the previous read."
 *
 * A directory regenerated per read silently truncates under ls the moment a
 * concurrent index shifts the results. lib9p already tracks that state per
 * fid for files that exist in the tree, so putting them there gets it right
 * for free.
 *
 *	echo 'similar <hex>' >> /mnt/sigil/ctl
 *	ls /mnt/sigil/similar/<hex>/
 */

#include <u.h>
#include <libc.h>
#include <fcall.h>
#include <thread.h>
#include <9p.h>

#include "sigilfs.h"

enum {
	Maxneighbours = 256,   /* per neighbourhood; ls of 10k is not a UI */
	Hexlen        = 64,    /* BLAKE3-256 as hex */
};

/*
 * Cache of materialized neighbourhoods. Without this, every walk re-scans:
 * a shell completing a path stats the directory several times, and each stat
 * would cost a full pass over the store.
 *
 * Entries are never evicted. A neighbourhood is 256 Files at most, and the
 * number of distinct hashes anyone walks in one session is small. If that
 * stops being true, evict by last-walk time -- but measure before adding the
 * machinery.
 */
typedef struct Neigh Neigh;
struct Neigh {
	char   hex[Hexlen + 1];
	File  *dir;
	long   index;      /* record this directory is the neighbourhood of */
	File **leaf;       /* children, so they can be removed before the dir */
	int    nleaf;
	Neigh *next;
};

static Neigh *neighs;

static Neigh *
neighfind(char *hex)
{
	Neigh *n;

	for(n = neighs; n != nil; n = n->next)
		if(strcmp(n->hex, hex) == 0)
			return n;
	return nil;
}

/*
 * A neighbour file's aux carries the record index, so a read does not have to
 * resolve the name back through the store.
 */
typedef struct Leaf Leaf;
struct Leaf {
	long index;
};

/*
 * Materialize /similar/<hex>/ by scanning. Returns nil and sets an error
 * string if the hash is not in the store.
 */
char *
similar_walk(Sigilfs *f, char *hex, File **out)
{
	static char err[128];
	unsigned hits[Maxneighbours];
	char nhex[Hexlen + 1];
	Neigh *n;
	File *dir, *leaf;
	Leaf *aux;
	long i, self;
	long nhit, k;
	File **leaves;
	int nleaf;

	if(strlen(hex) != Hexlen){
		snprint(err, sizeof err, "sigilfs: not a %d-digit hash", Hexlen);
		return err;
	}
	if((n = neighfind(hex)) != nil){
		*out = n->dir;
		return nil;
	}

	self = br_find_hash(f->store, hex);
	if(self < 0){
		snprint(err, sizeof err, "sigilfs: no such sigil");
		return err;
	}

	/* The scan. Everything above is bookkeeping; this line is the query. */
	nhit = br_similar(f->store, self, f->thresh, hits, Maxneighbours);
	f->nscans++;

	dir = createfile(f->similar, hex, "sigil", DMDIR|0555, nil);
	if(dir == nil){
		snprint(err, sizeof err, "sigilfs: createfile: %r");
		return err;
	}
	leaves = emalloc9p(Maxneighbours * sizeof *leaves);
	nleaf = 0;

	for(k = 0; k < nhit; k++){
		i = (long)hits[k];
		if(i == self)
			continue;    /* a record is not its own neighbour */
		if(br_hash(f->store, i, nhex, sizeof nhex) != 0)
			continue;
		aux = emalloc9p(sizeof *aux);
		aux->index = i;
		leaf = createfile(dir, nhex, "sigil", 0444, aux);
		if(leaf == nil){
			/* Duplicate name: the same paragraph text indexed twice
			 * under different paths hashes identically, which is
			 * correct -- content addressing means one record. Skip
			 * rather than fail the whole walk. */
			free(aux);
			continue;
		}
		leaves[nleaf++] = leaf;
	}

	n = emalloc9p(sizeof *n);
	strecpy(n->hex, n->hex + sizeof n->hex, hex);
	n->dir = dir;
	n->index = self;
	n->leaf = leaves;
	n->nleaf = nleaf;
	n->next = neighs;
	neighs = n;

	*out = dir;
	return nil;
}

/*
 * Reading a neighbour yields its provenance and text. A hash alone is not
 * usable output: the point of the projection is to land on something a reader
 * can read.
 */
int
similar_isleaf(File *f)
{
	return f != nil && f->aux != nil && f->parent != nil &&
	       f->parent->parent == fs.similar;
}

void
similar_read(Req *r)
{
	File *f = r->fid->file;
	Leaf *l = f->aux;
	const char *path;
	char *buf, *text;
	ulong off, len;
	int n, fd, got;

	path = br_path(fs.store, l->index);
	off = br_offset(fs.store, l->index);
	len = br_length(fs.store, l->index);

	/* The store holds the sigil, not the text -- 64 bytes per paragraph is
	 * the entire point of the layout, and caching the prose alongside it
	 * would multiply the store by fifty. The paragraph is re-read from its
	 * source at the recorded offset instead.
	 *
	 * A record restored from libtab has no offset (length 0): persistence
	 * replays sigils whose source file may since have moved or changed,
	 * and serving text from a stale offset would be worse than serving
	 * none. Say so rather than return plausible wrong bytes. */
	buf = emalloc9p(len + 512);
	n = snprint(buf, 512, "path\t%s\npara\t%ud\noffset\t%lud\nlength\t%lud\n\n",
	            path != nil ? path : "(unknown)",
	            br_para(fs.store, l->index), off, len);

	if(len == 0 || path == nil || path[0] == '\0'){
		n += snprint(buf + n, 128,
		             "(no offset recorded; reindex to read the text)\n");
	}else if((fd = open(path, OREAD)) < 0){
		n += snprint(buf + n, 256, "(cannot open %s: %r)\n", path);
	}else{
		text = buf + n;
		if(seek(fd, off, 0) < 0 || (got = readn(fd, text, len)) <= 0)
			n += snprint(buf + n, 128, "(cannot read at offset)\n");
		else
			n += got;
		close(fd);
	}

	readbuf(r, buf, n);
	free(buf);
	respond(r, nil);
}

/*
 * Drop every materialized neighbourhood. Called when the store changes:
 * results computed against the old contents are wrong, and a stale directory
 * that still lists them is worse than one that has to be rebuilt.
 */
void
similar_flush(Sigilfs *f)
{
	Neigh *n, *next;
	int k;

	for(n = neighs; n != nil; n = next){
		next = n->next;
		/* Children first: removefile() refuses a directory that still
		 * has any, setting "has children" and returning -1. Ignoring
		 * that leaked every neighbourhood, and a later scan of the same
		 * hash then failed with "file already exists" and went on
		 * serving results computed against a store that had changed.
		 *
		 * The leaves are tracked here rather than enumerated from the
		 * File, whose filelist is marked implementation-private. */
		for(k = 0; k < n->nleaf; k++)
			removefile(n->leaf[k]);
		if(removefile(n->dir) < 0 && tracing)
			fprint(2, "TRACE similar_flush %s: %r\n", n->hex);
		free(n->leaf);
		free(n);
	}
	neighs = nil;
	USED(f);
}
