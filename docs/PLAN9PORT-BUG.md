# plan9port lib9p: spurious "wstat -- attempt to change qid" on 64-bit

## Summary

`srv.c:swstat()` rejects every conformant `Twstat` on amd64, because a
`(ulong)` cast fails to truncate a 32-bit wire field on a platform where
`ulong` is 64 bits.

Reproduces with plan9port's own client: `nulldir()` + `dirfwstat()` is
rejected by any lib9p server.

## The check

```c
/* src/lib9p/srv.c, ~line 676 */
if((uchar)~r->d.qid.type || (ulong)~r->d.qid.vers || (uvlong)~r->d.qid.path){
    respond(r, "wstat -- attempt to change qid");
    return;
}
```

The intent is "reject unless every qid field is the don't-touch sentinel",
which stat(5) defines as the maximum unsigned value of each field's size.

## Why it fails

`Qid.vers` is declared `ulong`, which is **8 bytes** on plan9port/amd64, but
the 9P wire format carries it as 4 bytes. `convM2D` reads it with `GBIT32`, so
the field holds `0x00000000FFFFFFFF`.

```
~0x00000000FFFFFFFF = 0xFFFFFFFF00000000   /* nonzero -> rejected */
```

On Plan 9 proper `ulong` is 32 bits, `~0xFFFFFFFF` is 0, and the check passes.

Measured:

```
uchar=1 ushort=2 uint=4 ulong=8 uvlong=8
Qid.vers field size = 8

after convD2M/convM2D round-trip: qid.type=0xff vers=0xffffffff path=0xffffffffffffffff
lib9p checks: ~type=0  ~vers=18446744069414584320  ~path=0
verdict: REJECTED
```

## Fix

Cast to the wire width, not the storage width:

```c
if((uchar)~r->d.qid.type || (uint)~r->d.qid.vers || (uvlong)~r->d.qid.path){
```

The same mistake appears ~8 lines later for `d.mode`, another 32-bit wire field
stored in a `ulong`.

**Already fixed locally**: `~/Repo/plan9port` commit `ad33b5fe`, "lib9p: accept
Linux v9fs rename wstat", corrects both casts and additionally relaxes the muid
check so a rename (which carries a name) is not rejected for also carrying a
muid. Upstream plan9port master still has the bug as of this writing.

The trap for anyone re-deriving this: `/usr/local/plan9` may be an older
*installed* copy that predates the source tree, so the source can look correct
while the linked library is not.

## Impact

Any Linux client that sets mtime on a file served by lib9p. In particular
v9fs, the in-kernel 9P client, issues a `Twstat` after a truncating open, so
shell `>` redirection onto a lib9p-served file fails. The error surfaces as
`ESERVERFAULT` (errno 526, "an untranslatable error occurred") because base
9P2000 returns error strings and Linux has no errno to map them to.

`>>`, `dd conv=notrunc` and `9p write` are unaffected -- they do not truncate.
