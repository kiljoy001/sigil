/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * /ctl: the control interface.
 *
 * Plan 9 convention -- state changes happen by writing a line to a file, not
 * through an RPC surface. Each verb is one line; errors come back as the write
 * error string, which is what the caller sees from write(2).
 */

#include <u.h>
#include <libc.h>
#include <fcall.h>
#include <thread.h>
#include <9p.h>

#include "sigilfs.h"

Mount *
mountfind(Sigilfs *f, char *name)
{
	Mount *m;

	for(m = f->mounts; m != nil; m = m->next)
		if(strcmp(m->name, name) == 0)
			return m;
	return nil;
}

static char *
domount(Sigilfs *f, int argc, char **argv)
{
	Mount *m;
	Dir *d;

	if(argc != 3)
		return "usage: mount <name> <root>";

	/* Fail early on a root that is not there: a mount pointing nowhere
	 * would silently index nothing. */
	if((d = dirstat(argv[2])) == nil)
		return "mount: root not found";
	if((d->qid.type & QTDIR) == 0){
		free(d);
		return "mount: root is not a directory";
	}
	free(d);

	if((m = mountfind(f, argv[1])) != nil){
		free(m->root);
		m->root = estrdup9p(argv[2]);
		m->lastscan = 0;
		return nil;
	}

	m = emalloc9p(sizeof *m);
	m->name = estrdup9p(argv[1]);
	m->root = estrdup9p(argv[2]);
	m->lastscan = 0;
	m->next = f->mounts;
	f->mounts = m;
	f->nmounts++;
	return nil;
}

static char *
dothresh(Sigilfs *f, int argc, char **argv)
{
	int n;

	if(argc != 2)
		return "usage: thresh <n>";
	n = atoi(argv[1]);
	if(n < 0 || n > 512)
		return "thresh: out of range";
	f->thresh = n;
	return nil;
}

/*
 * Apply one control line. Returns nil on success or a static error string --
 * lib9p hands it straight back as the write error.
 */
static char *
oneline(Sigilfs *f, char *line)
{
	char *argv[8];
	int argc;

	argc = tokenize(line, argv, nelem(argv));
	if(argc == 0)
		return nil;                      /* blank line is a no-op */
	if(argv[0][0] == '#')
		return nil;

	if(strcmp(argv[0], "mount") == 0)
		return domount(f, argc, argv);
	if(strcmp(argv[0], "thresh") == 0)
		return dothresh(f, argc, argv);
	if(strcmp(argv[0], "index") == 0){
		Mount *m;

		if(argc == 1){
			char *err = nil;
			for(m = f->mounts; m != nil; m = m->next)
				if((err = index_mount(f, m)) != nil)
					return err;
			return nil;
		}
		if((m = mountfind(f, argv[1])) == nil)
			return "index: no such mount";
		return index_mount(f, m);
	}
	if(strcmp(argv[0], "commit") == 0)
		return store_commit(f);
	return "sigilfs: unknown control verb";
}

char *
ctlwrite(Sigilfs *f, char *buf)
{
	char *p, *nl, *err;

	/* A single write may carry several lines. Stop at the first error so
	 * the caller is not left guessing how far it got. */
	for(p = buf; *p != '\0'; p = nl){
		if((nl = strchr(p, '\n')) != nil)
			*nl++ = '\0';
		else
			nl = p + strlen(p);
		if((err = oneline(f, p)) != nil)
			return err;
	}
	return nil;
}
