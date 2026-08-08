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
if((uchar)~r->d.qid.type || (u32int)~r->d.qid.vers || (uvlong)~r->d.qid.path){
```

## Impact

Any Linux client that sets mtime on a file served by lib9p. In particular
v9fs, the in-kernel 9P client, issues a `Twstat` after a truncating open, so
shell `>` redirection onto a lib9p-served file fails. The error surfaces as
`ESERVERFAULT` (errno 526, "an untranslatable error occurred") because base
9P2000 returns error strings and Linux has no errno to map them to.

`>>`, `dd conv=notrunc` and `9p write` are unaffected -- they do not truncate.
