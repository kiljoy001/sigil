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
 * Two entry points, chosen at build time.
 *
 * libthread owns main when the server posts to a service name, because
 * threadpostmountsrv() is a libthread function. But libthread and OpenVINO
 * do not coexist: OpenVINO spawns a TBB pool of twenty-plus pthreads inside
 * a process whose scheduler is switching stacks underneath them, and
 * indexing segfaults intermittently -- deep in the GPU plugin, with a return
 * address pointing into the libthread stack region and no symbols. The same
 * embedder, over the same corpus, runs 218,356 paragraphs clean outside
 * libthread.
 *
 * So the TCP path, which is the one that embeds, uses a plain main() and
 * lib9p's srv() directly. Only tpost.c in lib9p actually needs libthread;
 * srv.c does not. SIGIL_NO_LIBTHREAD selects that build, and the two paths
 * are already disjoint -- tcploop() never returns.
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
	          "[-a tcp!*!port] [-f store.tab] [-e model.gguf]\n");
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
		/* Serve inline, on the main thread.
		 *
		 * This was proccreate(), which gives each connection its own
		 * libthread proc. OpenVINO does not survive that: indexing
		 * segfaulted inside the GPU plugin with a return address
		 * pointing into the libthread stack region, no symbols, and an
		 * empty log. Raising the stack from 32 KB to 8 MB moved the
		 * crash later but did not remove it, and the identical batch
		 * through the identical embed_batch() runs clean outside
		 * libthread at 1466 paragraphs/s -- so the batching is right
		 * and the threading context is not. OpenVINO spawns its own
		 * TBB pool, which does not expect the stack switching
		 * libthread does underneath it.
		 *
		 * The cost is one client at a time. That is what sigilfs
		 * actually does -- a mount, an index, a query -- and a
		 * correct serial server beats a concurrent one that crashes.
		 * Restoring concurrency means moving the embedder out of
		 * process, not moving the server back onto libthread procs. */
		serveconn((void*)(uintptr)fd);
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

#ifdef SIGIL_NO_LIBTHREAD
/* Serving over TCP with no libthread: OpenVINO's TBB pool gets a normal
 * pthread process, and lib9p's srv() does not care. Posting to a service
 * name is unavailable in this build, because that is the one thing that
 * genuinely needs libthread. */
#define ENTRY main
#define EXIT(x) return (x)
#else
#define ENTRY threadmain
#define EXIT(x) do { threadexits(0); return; } while (0)
#endif

#ifdef SIGIL_NO_LIBTHREAD
int
#else
void
#endif
ENTRY(int argc, char **argv)
{
	char *srvname = "sigil";
	char *mtpt = nil;
	char *store = nil;
	char *addr = nil;
	char *model = nil;
	char *err;

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
	case 'e':
		model = EARGF(usage());
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

	/* Load the model before the store, so a mismatch is caught against a
	 * live embedder rather than after records are already in memory. */
	if(model != nil){
		fs.modelpath = estrdup9p(model);
		if(br_embedder_load(fs.store, model, fs.simhash_seed) != 0)
			sysfatal("cannot load model %s", model);
	}

	/* Load before serving: a client that attaches mid-load would see a
	 * partial store and get wrong answers from a scan. Refusing to start
	 * on a parameter mismatch is deliberate -- bits made under a different
	 * model or seed are not comparable, and silently scanning them would
	 * return plausible nonsense. */
	if((err = store_load(&fs)) != nil)
		sysfatal("store: %s", err);

	buildtree();

	fs.srv.read = fsread;
	fs.srv.write = fswrite;
	fs.srv.open = fsopen;
	fs.srv.create = fscreate;
	fs.srv.stat = fsstat;
	/* Accepts a zero-length truncate on /ctl.
	 *
	 * This handler is never reached on 64-bit plan9port: lib9p rejects
	 * every conformant Twstat first, because srv.c casts a 32-bit wire
	 * field through (ulong), which is 8 bytes here. See
	 * docs/PLAN9PORT-BUG.md -- plan9port's own nulldir()+dirfwstat() trips
	 * it too, so it is not a client conformance issue.
	 *
	 * Consequence: shell `>` redirection onto /ctl fails, because v9fs
	 * sends a Twstat to set mtime after a truncating open. Use `>>`, dd
	 * conv=notrunc, or 9p write. */
	fs.srv.wstat = fswstat;

	if(addr != nil){
		tcploop(addr);              /* never returns */
		EXIT(0);
	}

#ifdef SIGIL_NO_LIBTHREAD
	USED(srvname); USED(mtpt);
	sysfatal("built without libthread: -s and -m need it, use -a");
	return 1;
#else
	/* Foreground so the process is visible to whoever started it; posting
	 * to a service name lets a caller decide where to attach it. */
	fs.srv.foreground = 1;
	threadpostmountsrv(&fs.srv, srvname, mtpt, MREPL|MCREATE);
	threadexits(0);
#endif
}
