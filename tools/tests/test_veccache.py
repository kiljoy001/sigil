# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""
Property tests for the embedding cache, driven through ctypes.

test/veccache.c covers the cases someone thought to write. These cover
what must hold for every input, which is the half that finds the case
nobody imagined -- and the cache exists precisely because 13 hours of GPU
work was lost to a situation nobody had imagined.

The properties that matter are all about survival rather than speed:
whatever goes in comes back, a crash costs one record, and a key is a key.
"""

import ctypes
import math
import struct
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DIM = 8


def _build(tmp_path_factory):
    so = tmp_path_factory.mktemp("vc") / "libveccache.so"
    src = ROOT / "src" / "veccache.c"
    if not src.exists():
        pytest.skip(f"{src} not present")
    r = subprocess.run(
        ["cc", "-O2", "-fPIC", "-shared", "-I", str(ROOT / "include"),
         str(src), "-o", str(so)],
        capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"cannot build veccache.c: {r.stderr[:300]}")
    return so


@pytest.fixture(scope="session")
def lib(tmp_path_factory):
    so = ctypes.CDLL(str(_build(tmp_path_factory)))
    so.sigil_veccache_open.restype = ctypes.c_void_p
    so.sigil_veccache_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                       ctypes.c_size_t]
    so.sigil_veccache_get.restype = ctypes.c_int
    so.sigil_veccache_get.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                      ctypes.POINTER(ctypes.c_float)]
    so.sigil_veccache_put.restype = ctypes.c_int
    so.sigil_veccache_put.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                      ctypes.POINTER(ctypes.c_float)]
    so.sigil_veccache_count.restype = ctypes.c_size_t
    so.sigil_veccache_count.argtypes = [ctypes.c_void_p]
    so.sigil_veccache_close.argtypes = [ctypes.c_void_p]
    so.sigil_veccache_sync.restype = ctypes.c_int
    so.sigil_veccache_sync.argtypes = [ctypes.c_void_p, ctypes.c_int]
    return so


def _arr(vals):
    return (ctypes.c_float * DIM)(*vals)


def _f16(x):
    """What the value becomes after a float32 -> float16 -> float32 trip."""
    return struct.unpack("e", struct.pack("e", x))[0]


# Components of a real embedding: L2-normalised, so within [-1, 1].
component = st.floats(min_value=-1.0, max_value=1.0,
                      allow_nan=False, allow_infinity=False, width=16)
vector = st.lists(component, min_size=DIM, max_size=DIM)
key = st.binary(min_size=32, max_size=32)


@settings(max_examples=200, deadline=None)
@given(k=key, v=vector)
def test_put_then_get_round_trips(lib, tmp_path_factory, k, v):
    """The property the cache exists for: what went in comes back.

    To float16 precision, which is the storage form -- the consumer is a
    SimHash projection whose output is a sign bit, so this is far tighter
    than anything downstream needs.
    """
    p = tmp_path_factory.mktemp("rt") / "c.jsonl"
    c = lib.sigil_veccache_open(str(p).encode(), b"m", DIM)
    assert c

    lib.sigil_veccache_put(c, k, _arr(v))
    out = (ctypes.c_float * DIM)()
    assert lib.sigil_veccache_get(c, k, out) == 0
    lib.sigil_veccache_close(c)

    for got, want in zip(out, v):
        assert math.isclose(got, _f16(want), rel_tol=1e-3, abs_tol=1e-3)


@settings(max_examples=150, deadline=None)
@given(k=key, v=vector)
def test_survives_reopen(lib, tmp_path_factory, k, v):
    """Durability, which is the entire point. In-memory correctness is
    exactly the property that did not matter when the process died."""
    p = tmp_path_factory.mktemp("re") / "c.jsonl"

    c = lib.sigil_veccache_open(str(p).encode(), b"m", DIM)
    lib.sigil_veccache_put(c, k, _arr(v))
    lib.sigil_veccache_close(c)

    c = lib.sigil_veccache_open(str(p).encode(), b"m", DIM)
    out = (ctypes.c_float * DIM)()
    assert lib.sigil_veccache_get(c, k, out) == 0
    lib.sigil_veccache_close(c)

    for got, want in zip(out, v):
        assert math.isclose(got, _f16(want), rel_tol=1e-3, abs_tol=1e-3)


@settings(max_examples=100, deadline=None)
@given(keys=st.lists(key, min_size=1, max_size=40, unique=True),
       v=vector)
def test_every_distinct_key_is_retrievable(lib, tmp_path_factory, keys, v):
    """No key displaces another, however the hashes collide in a bucket.
    The table keys on the first eight bytes and compares the full 32, so
    a shared prefix must probe rather than overwrite."""
    p = tmp_path_factory.mktemp("mk") / "c.jsonl"
    c = lib.sigil_veccache_open(str(p).encode(), b"m", DIM)

    for k in keys:
        lib.sigil_veccache_put(c, k, _arr(v))
    assert lib.sigil_veccache_count(c) == len(keys)

    out = (ctypes.c_float * DIM)()
    for k in keys:
        assert lib.sigil_veccache_get(c, k, out) == 0, f"lost {k.hex()[:16]}"
    lib.sigil_veccache_close(c)


@settings(max_examples=100, deadline=None)
@given(k=key, v=vector, w=vector)
def test_first_write_wins(lib, tmp_path_factory, k, v, w):
    """An interrupted run re-embeds text it already cached. The second
    write must not change what is stored, or a resumed run would produce
    a different corpus from an uninterrupted one."""
    p = tmp_path_factory.mktemp("dup") / "c.jsonl"
    c = lib.sigil_veccache_open(str(p).encode(), b"m", DIM)

    lib.sigil_veccache_put(c, k, _arr(v))
    lib.sigil_veccache_put(c, k, _arr(w))
    assert lib.sigil_veccache_count(c) == 1

    out = (ctypes.c_float * DIM)()
    lib.sigil_veccache_get(c, k, out)
    lib.sigil_veccache_close(c)

    for got, want in zip(out, v):
        assert math.isclose(got, _f16(want), rel_tol=1e-3, abs_tol=1e-3)


@settings(max_examples=80, deadline=None)
@given(k=key, v=vector, cut=st.integers(min_value=1, max_value=200))
def test_truncation_costs_at_most_one_record(lib, tmp_path_factory, k, v,
                                             cut):
    """A crash mid-append leaves a partial line. Whatever the cut point,
    the file must still open and every complete record before it must
    survive -- this is what the line-oriented format was chosen for."""
    p = tmp_path_factory.mktemp("tr") / "c.jsonl"

    c = lib.sigil_veccache_open(str(p).encode(), b"m", DIM)
    lib.sigil_veccache_put(c, k, _arr(v))
    lib.sigil_veccache_close(c)

    whole = p.read_bytes()
    # Append a partial second record, cut at an arbitrary point.
    p.write_bytes(whole + whole[:min(cut, len(whole) - 1)])

    c = lib.sigil_veccache_open(str(p).encode(), b"m", DIM)
    assert c, "a truncated file must still open"
    out = (ctypes.c_float * DIM)()
    assert lib.sigil_veccache_get(c, k, out) == 0, "the complete record survives"
    lib.sigil_veccache_close(c)


@settings(max_examples=80, deadline=None)
@given(k=key, v=vector,
       m1=st.text(alphabet="abcdefghijklmnop-", min_size=1, max_size=12),
       m2=st.text(alphabet="abcdefghijklmnop-", min_size=1, max_size=12))
def test_model_scoping_holds_for_any_names(lib, tmp_path_factory, k, v,
                                           m1, m2):
    """A different model must never serve another's vectors, whatever the
    names are. Serving MiniLM output to a caller that switched models is
    silent corruption -- the numbers look fine and mean nothing."""
    p = tmp_path_factory.mktemp("ms") / "c.jsonl"

    c = lib.sigil_veccache_open(str(p).encode(), m1.encode(), DIM)
    lib.sigil_veccache_put(c, k, _arr(v))
    lib.sigil_veccache_close(c)

    c = lib.sigil_veccache_open(str(p).encode(), m2.encode(), DIM)
    out = (ctypes.c_float * DIM)()
    hit = lib.sigil_veccache_get(c, k, out) == 0
    lib.sigil_veccache_close(c)

    assert hit == (m1 == m2), (
        f"model {m2!r} {'saw' if hit else 'missed'} {m1!r}'s vector")
