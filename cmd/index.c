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

enum {
	Maxfile   = 4*1024*1024,   /* skip anything larger: not prose */
	Minpara   = 40,            /* shorter than this embeds to noise */
	Maxpara   = 4000,          /* split beyond this */
	Maxdepth  = 32,
};

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
static int
addparas(Sigilfs *f, char *path, char *buf, long n)
{
	char *p, *end, *q, *cut;
	int para = 1, added = 0;
	long len;

	end = buf + n;
	for(p = buf; p < end; ){
		while(p < end && (*p == '\n' || *p == '\r' || *p == ' ' || *p == '\t'))
			p++;
		if(p >= end)
			break;
		/* find the next blank line */
		for(q = p; q < end - 1; q++)
			if(q[0] == '\n' && (q[1] == '\n' || q[1] == '\r'))
				break;
		if(q >= end - 1)
			q = end;
		len = q - p;

		while(len > Maxpara){
			cut = p + Maxpara;
			while(cut > p + Maxpara/2 && *cut != '.' && *cut != '\n')
				cut--;
			if(cut <= p + Maxpara/2)
				cut = p + Maxpara;
			if(cut - p >= Minpara)
				if(br_add_at(f->store, path, para++, p, cut - p,
				             nil, 0, p - buf) >= 0)
					added++;
			len -= cut - p;
			p = cut;
		}
		if(len >= Minpara)
			if(br_add_at(f->store, path, para++, p, len, nil, 0,
			             p - buf) >= 0)
				added++;
		p = q + 1;
	}
	return added;
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
