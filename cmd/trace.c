/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * Request tracing.
 *
 * chatty9p (-D) dumps wire messages, which shows what arrived but not which
 * callback rejected it or why. This logs the decision point: the request type,
 * the file it touched, the mode bits, and the error string returned.
 *
 * Written because three plausible explanations for a client-visible failure
 * were each wrong, and the wire dump did not distinguish them.
 */

#include <u.h>
#include <libc.h>
#include <fcall.h>
#include <thread.h>
#include <9p.h>

#include "sigilfs.h"

int tracing;

static char *
modestr(int m)
{
	static char buf[64];
	char *p = buf;

	switch(m & 3){
	case OREAD:  p += sprint(p, "OREAD");  break;
	case OWRITE: p += sprint(p, "OWRITE"); break;
	case ORDWR:  p += sprint(p, "ORDWR");  break;
	case OEXEC:  p += sprint(p, "OEXEC");  break;
	}
	if(m & OTRUNC)  p += sprint(p, "|OTRUNC");
	if(m & ORCLOSE) p += sprint(p, "|ORCLOSE");
	if(m & OEXCL)   sprint(p, "|OEXCL");
	return buf;
}

void
tracereq(Req *r, char *what, char *err)
{
	char *name = "?";

	if(!tracing)
		return;
	if(r->fid != nil && r->fid->file != nil)
		name = r->fid->file->dir.name;

	fprint(2, "TRACE %-7s file=%-8s", what, name);
	if(strcmp(what, "open") == 0 || strcmp(what, "create") == 0)
		fprint(2, " mode=%s", modestr(r->ifcall.mode));
	if(strcmp(what, "write") == 0)
		fprint(2, " off=%lld n=%ud", r->ifcall.offset, r->ifcall.count);
	if(strcmp(what, "wstat") == 0)
		fprint(2, " len=%lld mode=%luo name=%q",
		       r->d.length, r->d.mode, r->d.name ? r->d.name : "");
	fprint(2, " -> %s\n", err ? err : "ok");
}
