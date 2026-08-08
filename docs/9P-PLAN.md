# sigilfs: the 9P server

Plan for the piece that makes the name honest. Right now sigil is a similarity
index with no I/O — you cannot mount it, and it has never held a file.

## What was verified before planning

Both dependencies were built and exercised rather than assumed.

**libtab** compiles under plan9port (all 11 sources, `9c`) and round-trips:
create a table, add a row, commit, reopen, iterate. The file it writes is
ndb-shaped text:

```
schema=sigil
	col=path
	col=para
	col=lsh

path=/mnt/docs/a.txt
	para=3
	lsh=AAECAwQFBgc=
```

Three constraints found by building it:

| constraint | consequence |
|---|---|
| needs plan9port (`u.h`, `libc.h`, `bio.h`, `ndb.h`, `9pclient.h`) | `sigilfs` links it; `libsigil` stays dependency-free |
| link order `libtab.a -l9pclient -lthread` | Makefile detail |
| **libthread owns `main`** | `sigilfs` defines `threadmain`, not `main` |

**lib9p** was tested with a throwaway server that posted a namespace with
`/ctl`, `/stats` and `/similar/<hex>/`. Over the real protocol:

```
$ 9p -a $NAMESPACE/sigil ls /
ctl
similar
stats
$ 9p -a $NAMESPACE/sigil read stats
records 0
scans 0
```

That works, so there is no reason to write a protocol implementation.

## Why lib9p rather than a hand-written codec

`read(5)` is explicit about directory reads:

> The read request message must have offset equal to zero or the value of
> offset in the previous read on the directory, plus the number of bytes
> returned in the previous read. In other words, seeking other than to the
> beginning is illegal in a directory.

Sequential-only, resumable at exactly the previous offset. A synthetic
directory whose contents are regenerated per read will silently truncate under
`ls` the moment a concurrent index update shifts the results.

`lib9p`'s `Fid` already carries `Readdir *rdir`, `vlong diroffset` and
`long dirindex` — it tracks that state per fid. Writing it again by hand would
be ~600 lines whose only distinction is a fresh chance to get this wrong.

The cost is that the namespace is shaped by the `Srv` callback structure. Given
we link plan9port for libtab regardless, that is a cheap trade.

## Storage split

```
libsigil (C11, no deps)      sigils, scan kernels
    ^
sigilfs (plan9port C)        threadmain, libtab, lib9p
    +-- libtab.a             durable: schema=sigil
    +-- hashmap              mount table, runtime only
    +-- sigil_store_t        the scan index, loaded at startup
```

**libtab holds the sigils.** Its schema tuple also carries `model_id`,
`embed_dim`, `simhash_seed`, `lsh_bits` and `federation`. A store whose
parameters differ from the running configuration is not comparable and must
refuse to open rather than return wrong answers.

**The mount table is an in-memory hashmap**, not a table. It is runtime state —
"what am I currently serving" — rebuilt from `/ctl` on startup. Persisting it
would only create a file to drift out of sync with reality. Same reasoning as
fids.

**libtab is a serialization format, not a query engine.** Its own header says
search is a deliberate linear scan with no secondary indexes. So sigilfs loads
the whole table into `sigil_store_t` once at open and never queries libtab at
runtime — the SIMD scan is the index.

## Namespace

```
/ctl                    write: index <path>, mount <name> <root>, thresh <n>
/stats                  read:  record count, scan timings, model parameters
/sigil/<hex>            a paragraph's text, addressed by its BLAKE3
/similar/<hex>/         synthetic dir; walking it runs the scan
/similar/<hex>/<hex>    a neighbour, readable
```

Deliberately no `/class/` in the first version. Clustering measured badly —
similarity-only partitioning put 2018 of 5000 items in one cluster — and
shipping a directory tree that misfiles most things would be exactly the
overclaim this project has spent its time avoiding. It returns when there is an
asserted-edge extractor to build it from.

`/similar/<hex>/` results are **materialized at open and pinned to the fid**,
per the constraint above. A scan runs once when the directory is opened, not
per read.

## Order of work

1. **Skeleton** — `threadmain`, `Srv`, static tree with `/ctl` and `/stats`.
   Mountable, does nothing useful. This is the first point where `mount` works.
2. **libtab persistence** — load `schema=sigil` at startup into
   `sigil_store_t`, refuse on parameter mismatch, commit on change.
3. **Indexer behind `/ctl`** — `index <path>` walks a directory, extracts
   paragraphs, embeds, appends. Text and Markdown first; xlsx via the existing
   `tools/xlsx_*.py` as a separate pass.
4. **`/sigil/<hex>`** — serve a paragraph by content hash.
5. **`/similar/<hex>/`** — the synthetic directory and the pinning.

Steps 1-2 are the ones worth doing carefully; the rest follows.

## Open questions

- **Where does the embedder live?** It is llama.cpp, which is C++ and heavy.
  Options: link it into sigilfs, shell out per batch, or run it as a separate
  process behind a socket. The last fits the composing-programs shape and keeps
  sigilfs from depending on a model being present, but adds a moving part.
- **Text extraction beyond plain text.** PDF via `pdftotext` is one binary and
  no JVM. Anything broader was considered and rejected as too fiddly.
- **Incremental reindex.** `Tstat` on scan to detect changed files; the BLAKE3
  per paragraph makes "has this changed" a cheap comparison. Not designed yet.
