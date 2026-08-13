/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (c) 2026 Scott J Guyton */

/*
 * libFuzzer harness for the cache loader.
 *
 * sigil_veccache_open() parses a file it did not necessarily write. The
 * expected damage is a partial final line from a crash, but the file sits
 * on disk beside everything else and nothing stops it being truncated by
 * a full disk, concatenated by a well-meaning script, or corrupted by the
 * filesystem. A loader that reads past the end of a malformed line would
 * do it while recovering from a crash -- the worst possible moment.
 *
 * The parser is deliberately not a JSON parser: it scans for "key":" and
 * takes to the next quote, which is sufficient for lines this code wrote
 * and must merely be *safe* for lines it did not.
 *
 *	make fuzz-veccache
 *	test/fuzz_veccache -max_total_time=300 test/fuzz-corpus-vc
 */

#include "sigil_veccache.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DIM 8

int
LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
	char path[] = "/tmp/sigil-fuzz-vc-XXXXXX";
	sigil_veccache_t *c;
	int fd;

	/* A real file, because the loader takes a path: the parser is what
	 * is under test and it only runs against a file on disk. */
	fd = mkstemp(path);
	if (fd < 0)
		return 0;
	if (size > 0 && write(fd, data, size) != (ssize_t)size) {
		close(fd);
		unlink(path);
		return 0;
	}
	close(fd);

	c = sigil_veccache_open(path, "m", DIM);
	if (c != NULL) {
		uint8_t h[32];
		float v[DIM];
		size_t i;

		/* Exercise the read path against whatever loaded: a hash
		 * derived from the input, so the fuzzer can steer toward a
		 * key that is actually present. */
		memset(h, 0, sizeof h);
		for (i = 0; i < size && i < sizeof h; i++)
			h[i] = data[i];
		sigil_veccache_get(c, h, v);

		/* And the write path, which shares the table with whatever
		 * the loader built. */
		for (i = 0; i < DIM; i++)
			v[i] = (float)i / DIM;
		sigil_veccache_put(c, h, v);
		sigil_veccache_sync(c, 0);

		sigil_veccache_close(c);
	}
	unlink(path);
	return 0;
}
