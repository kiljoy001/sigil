/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * sigilfs - serve the sigil index as a 9P namespace.
 *
 * Everything is a file. Control by writing to /ctl, results by reading. No RPC
 * layer, no ioctls: a semantic query is a directory walk.
 *
 *   /ctl        write:  mount <name> <root>, index <path>, thresh <n>
 *   /stats      read:   record count, store parameters, scan timings
 *   /similar/   synthetic; walking /similar/<hex> runs a scan
 *
 * Built on plan9port's lib9p rather than a hand-written codec. read(5)
 * requires directory reads be sequential and resumable at exactly the previous
 * offset -- a synthetic directory regenerated per read silently truncates
 * under ls -- and lib9p already tracks that state per fid. See
 * docs/9P-PLAN.md.
 *
 * libthread owns main, so the entry point is threadmain.
 */

#include <u.h>
#include <libc.h>
#include <fcall.h>
#include <thread.h>
#include <9p.h>

#include "sigilfs.h"

Sigilfs fs;

static void
usage(void)
{
	fprint(2, "usage: sigilfs [-DL] [-s srvname] [-m mtpt] "
	          "[-a tcp!*!port] [-f store.tab]\n");
	threadexitsall("usage");
}

/*
 * Serve over a network address so Linux can mount us with v9fs, the in-kernel
 * 9P client:
 *
 *	sigilfs -a 'tcp!*!5640' &
 *	mount -t 9p -o trans=tcp,port=5640,version=9p2000 127.0.0.1 /mnt/sigil
 *
 * lib9p's srv() drives a single connection on infd/outfd, so each accepted
 * connection gets its own proc. Serving forever rather than exiting after one
 * client is the difference between a demo and a daemon.
 */
static void
serveconn(void *v)
{
	Srv *s;
	int fd;

	fd = (int)(uintptr)v;
	s = emalloc9p(sizeof *s);
	*s = fs.srv;              /* share the tree; per-connection fid pools */
	s->infd = s->outfd = fd;
	s->fpool = nil;
	s->rpool = nil;
	srv(s);
	close(fd);
	free(s);
}

static void
tcploop(char *addr)
{
	char adir[40], ldir[40];
	int actl, lctl, fd;

	if((actl = announce(addr, adir)) < 0)
		sysfatal("announce %s: %r", addr);

	for(;;){
		if((lctl = listen(adir, ldir)) < 0)
			sysfatal("listen: %r");
		if((fd = accept(lctl, ldir)) < 0){
			close(lctl);
			continue;
		}
		close(lctl);
		proccreate(serveconn, (void*)(uintptr)fd, 32*1024);
	}
	USED(actl);
}

/*
 * The tree is static for now: /ctl, /stats and an empty /similar. Synthetic
 * children under /similar arrive when a scan is wired up.
 */
static void
buildtree(void)
{
	fs.srv.tree = alloctree("sigil", "sigil", DMDIR|0555, nil);
	if(fs.srv.tree == nil)
		sysfatal("alloctree: %r");

	fs.ctl = createfile(fs.srv.tree->root, "ctl", "sigil", 0666, nil);
	fs.stats = createfile(fs.srv.tree->root, "stats", "sigil", 0444, nil);
	fs.similar = createfile(fs.srv.tree->root, "similar", "sigil",
	                        DMDIR|0555, nil);
	if(fs.ctl == nil || fs.stats == nil || fs.similar == nil)
		sysfatal("createfile: %r");
}

void
threadmain(int argc, char **argv)
{
	char *srvname = "sigil";
	char *mtpt = nil;
	char *store = nil;
	char *addr = nil;

	ARGBEGIN{
	case 'D':
		chatty9p++;
		break;
	case 'L':
		tracing++;
		break;
	case 's':
		srvname = EARGF(usage());
		break;
	case 'm':
		mtpt = EARGF(usage());
		break;
	case 'a':
		addr = EARGF(usage());
		break;
	case 'f':
		store = EARGF(usage());
		break;
	default:
		usage();
	}ARGEND

	if(argc != 0)
		usage();

	sigilfs_init(&fs, store);
	buildtree();

	fs.srv.read = fsread;
	fs.srv.write = fswrite;
	fs.srv.open = fsopen;
	fs.srv.create = fscreate;
	fs.srv.stat = fsstat;
	/* Accepts a zero-length truncate on /ctl. Note this does NOT make
	 * `echo > ctl` work under 9pfuse: 9pfuse encodes the no-change qid as
	 * 'dalA' rather than all-ones, and lib9p rejects that in swstat before
	 * any callback runs ("wstat -- attempt to change qid"). Use `>>`, dd
	 * conv=notrunc, or 9p write. The handler is still correct for clients
	 * that send a proper wstat. */
	fs.srv.wstat = fswstat;

	if(addr != nil){
		tcploop(addr);              /* never returns */
		threadexits(0);
	}

	/* Foreground so the process is visible to whoever started it; posting
	 * to a service name lets a caller decide where to attach it. */
	fs.srv.foreground = 1;
	threadpostmountsrv(&fs.srv, srvname, mtpt, MREPL|MCREATE);
	threadexits(0);
}
