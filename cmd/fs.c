/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * 9P read and write callbacks.
 *
 * lib9p dispatches here after resolving the fid to a File in the tree, so
 * these only decide what a given file's contents are. Directory reads are
 * handled by lib9p itself, which matters: read(5) requires them to be
 * sequential and resumable at exactly the previous offset.
 */

#include <u.h>
#include <libc.h>
#include <fcall.h>
#include <thread.h>
#include <9p.h>

#include "sigilfs.h"

static char *
statstext(Sigilfs *f)
{
	static char buf[2048];
	Mount *m;
	int n;

	n = snprint(buf, sizeof buf,
		"records\t%llud\n"
		"scans\t%llud\n"
		"store\t%s\n"
		"model\t%s\n"
		"embed_dim\t%d\n"
		"lsh_bits\t%d\n"
		"simhash_seed\t%llux\n"
		"mounts\t%d\n",
		f->nrecords, f->nscans,
		f->storepath ? f->storepath : "(memory)",
		f->model_id ? f->model_id : "(none)",
		f->embed_dim, f->lsh_bits, f->simhash_seed,
		f->nmounts);

	for(m = f->mounts; m != nil && n < sizeof(buf) - 128; m = m->next)
		n += snprint(buf+n, sizeof(buf)-n, "mount\t%s\t%s\t%ld\n",
		             m->name, m->root, m->lastscan);
	return buf;
}

void
fsread(Req *r)
{
	File *f = r->fid->file;

	if(f == fs.ctl){
		/* Reading /ctl reports the verbs it accepts, so the interface
		 * is discoverable with cat rather than documentation. */
		readstr(r, "mount <name> <root>\nindex <path>\nthresh <n>\n");
		tracereq(r, "read", nil);
		respond(r, nil);
		return;
	}
	if(f == fs.stats){
		readstr(r, statstext(&fs));
		tracereq(r, "read", nil);
		respond(r, nil);
		return;
	}
	tracereq(r, "read", "no read for this file");
	respond(r, "sigilfs: no read for this file");
}

void
fswrite(Req *r)
{
	char *line, *err;
	long n;

	if(r->fid->file != fs.ctl){
		tracereq(r, "write", "not writable");
		respond(r, "sigilfs: not writable");
		return;
	}

	n = r->ifcall.count;
	line = emalloc9p(n + 1);
	memmove(line, r->ifcall.data, n);
	line[n] = '\0';

	err = ctlwrite(&fs, line);
	free(line);

	if(err != nil){
		tracereq(r, "write", err);
		respond(r, err);
		return;
	}
	tracereq(r, "write", nil);
	/* Report the whole write consumed: a short count makes the writer
	 * retry the tail, which would re-run the command. */
	r->ofcall.count = n;
	respond(r, nil);
}

/*
 * A control file is a command channel, not storage: truncating it is
 * meaningless but harmless, and shells do it on every `>` redirection. Accept
 * a zero-length truncate and reject anything that would actually change the
 * file's identity.
 *
 * Under 9pfuse this is never reached -- see the note in sigilfs.c -- but it is
 * the correct behaviour for clients that encode wstat properly.
 */
void
fswstat(Req *r)
{
    Dir *d = &r->d;

    if(tracing)
        fprint(2, "TRACE wstat-raw qid.type=%#ux qid.vers=%#lux qid.path=%#llux "
                  "mode=%#luo mtime=%#lux length=%lld name=%q\n",
               d->qid.type, d->qid.vers, d->qid.path,
               d->mode, d->mtime, d->length, d->name ? d->name : "");

    if(r->fid->file != fs.ctl){
        tracereq(r, "wstat", "cannot wstat");
        respond(r, "sigilfs: cannot wstat");
        return;
    }
    if(d->length != ~0ULL && d->length != 0){
        tracereq(r, "wstat", "ctl cannot be extended");
        respond(r, "sigilfs: ctl cannot be extended");
        return;
    }
    if((d->name != nil && d->name[0] != '\0') ||
       (d->uid != nil && d->uid[0] != '\0') ||
       (d->gid != nil && d->gid[0] != '\0') ||
       d->mode != ~0UL){
        tracereq(r, "wstat", "ctl metadata is fixed");
        respond(r, "sigilfs: ctl metadata is fixed");
        return;
    }
    tracereq(r, "wstat", nil);
    respond(r, nil);
}

/*
 * These three exist to observe what a client actually asks for. lib9p would
 * handle open and stat itself; intercepting them means the trace shows the
 * mode bits and any rejection, which a wire dump alone does not attribute to a
 * decision point.
 */
void
fsopen(Req *r)
{
	tracereq(r, "open", nil);
	respond(r, nil);
}

void
fscreate(Req *r)
{
	/* The namespace is synthetic: files appear because sigilfs put them
	 * there, never because a client asked. A shell that cannot open a file
	 * will try to create it, so this is worth reporting clearly. */
	tracereq(r, "create", "namespace is read-only");
	respond(r, "sigilfs: cannot create files");
}

void
fsstat(Req *r)
{
	r->d = r->fid->file->dir;
	r->d.name = estrdup9p(r->d.name);
	r->d.uid = estrdup9p(r->d.uid);
	r->d.gid = estrdup9p(r->d.gid);
	r->d.muid = estrdup9p(r->d.muid);
	tracereq(r, "stat", nil);
	respond(r, nil);
}
