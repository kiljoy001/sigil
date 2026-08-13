/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Ingestion: walk a mounted root, split files into paragraphs, add records.
 *
 * Paragraphs, not files. Whole-document embedding forces one vector to stand
 * for everything a document says, and the embedder truncates at 512 tokens
 * anyway, so most of a long file was being silently discarded. A paragraph is
 * the human-shaped unit -- a sentence fragments an idea, a fixed window cuts
 * across one.
 *
 * Embedding happens outside this process, so records land with the byte
 * shingle fallback until an embedder fills them in. That fallback is NOT
 * semantic; it exists so the ingestion path can be exercised and measured
 * without a model present.
 */

#include <u.h>
#include <libc.h>
#include <fcall.h>
#include <thread.h>
#include <9p.h>

#include "sigilfs.h"
#include "sigil_split.h"

enum {
	/* Books, not prose fragments. 4 MB silently skipped 128 books of
	 * the Gutenberg corpus holding 3.0M paragraphs -- 4% of the whole
	 * index -- and the skip was a bare `return 0`, indistinguishable
	 * from a file with nothing in it. p99.9 of that corpus is 5.3 MB;
	 * 32 MB leaves only 12 outliers, the largest of which is 147 MB
	 * and is not a novel. Skips are counted now, so the next one is
	 * visible rather than inferred from a paragraph total that does
	 * not add up. */
	Maxfile   = 32*1024*1024,
	Maxdepth  = 32,
};

/* Paragraph bounds live in sigil_split.h, with the splitter that uses
 * them. A second copy here is how the three implementations drifted
 * apart in the first place. */

/* Extensions worth reading. Deliberately narrow: extracting garbage from a
 * binary and embedding it is worse than skipping the file. */
static char *exts[] = {
	".txt", ".md", ".markdown", ".rst", ".org",
	".c", ".h", ".go", ".py", ".rs", ".sh", ".ml", ".java",
	".html", ".xml", ".json", ".yaml", ".yml", ".tex",
	nil
};

static int
wanted(char *name)
{
	char *dot;
	int i;

	if((dot = strrchr(name, '.')) == nil)
		return 0;
	for(i = 0; exts[i] != nil; i++)
		if(cistrcmp(dot, exts[i]) == 0)
			return 1;
	return 0;
}

/*
 * Split on blank lines, the plain-text paragraph convention. Very short runs
 * are dropped and very long ones cut at a sentence boundary where possible.
 */
/*
 * Add every paragraph in buf to the store.
 *
 * The splitting itself lives in src/split.c, shared with tools/ and the
 * fuzzer. It used to be inline here and re-implemented twice elsewhere,
 * which drifted: the manifest counted 77,367,817 paragraphs for a corpus
 * this code split into 74,905,358.
 */
struct addctx {
	Sigilfs *f;
	char *path;
	char *buf;
	int added;
};

static void
addone(const sigil_chunk_t *c, void *arg)
{
	struct addctx *a = arg;

	if(br_add_at(a->f->store, a->path, c->para, a->buf + c->off, c->len,
	             nil, 0, c->off) >= 0)
		a->added++;
}

static int
addparas(Sigilfs *f, char *path, char *buf, long n)
{
	struct addctx a;

	a.f = f;
	a.path = path;
	a.buf = buf;
	a.added = 0;
	sigil_split(buf, (size_t)n, addone, &a);
	return a.added;
}

static int
onefile(Sigilfs *f, char *path)
{
	int fd, added;
	Dir *d;
	char *buf;
	long n;

	if((fd = open(path, OREAD)) < 0)
		return 0;
	if((d = dirfstat(fd)) == nil){
		close(fd);
		return 0;
	}
	if(d->length == 0 || d->length > Maxfile){
		if(d->length > Maxfile){
			f->nskipped++;
			if(tracing)
				fprint(2, "SKIP %s (%lld bytes > Maxfile)\n",
				       path, (vlong)d->length);
		}
		free(d); close(fd);
		return 0;
	}
	n = d->length;
	free(d);

	buf = malloc(n + 1);
	if(buf == nil){
		close(fd);
		return 0;
	}
	n = readn(fd, buf, n);
	close(fd);
	if(n <= 0){
		free(buf);
		return 0;
	}
	buf[n] = '\0';
	if(tracing)
		fprint(2, "INDEX %s (%ld bytes)\n", path, n);
	added = addparas(f, path, buf, n);
	free(buf);
	return added;
}

static int
walk(Sigilfs *f, char *path, int depth, int *nfiles)
{
	Dir *d;
	int fd, i, nd, added = 0;
	char sub[1024];

	if(depth > Maxdepth)
		return 0;
	if((fd = open(path, OREAD)) < 0)
		return 0;
	while((nd = dirread(fd, &d)) > 0){
		for(i = 0; i < nd; i++){
			if(d[i].name[0] == '.')       /* skip dotfiles and .git */
				continue;
			snprint(sub, sizeof sub, "%s/%s", path, d[i].name);
			if(d[i].qid.type & QTDIR)
				added += walk(f, sub, depth+1, nfiles);
			else if(wanted(d[i].name)){
				added += onefile(f, sub);
				(*nfiles)++;
			}
		}
		free(d);
	}
	close(fd);
	return added;
}

/*
 * Index one mount. Returns nil, or an error string.
 */
char *
index_mount(Sigilfs *f, Mount *m)
{
	static char msg[128];
	int nfiles = 0, added;
	Dir *d;

	if((d = dirstat(m->root)) == nil)
		return "index: root not found";
	if((d->qid.type & QTDIR) == 0){
		free(d);
		return "index: root is not a directory";
	}
	free(d);

	added = walk(f, m->root, 0, &nfiles);
	/* Anything still queued must be embedded before the counters below
	 * describe the store, and before a caller can scan it. */
	br_flush(f->store);
	m->lastscan = time(0);
	f->nrecords = br_count(f->store);
	f->nindexed += nfiles;

	if(added == 0){
		snprint(msg, sizeof msg,
			"index: %d files read, no paragraphs met the minimum", nfiles);
		return msg;
	}
	return nil;
}
