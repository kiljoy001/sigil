# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

Feature: The store grows without copying itself
  # These scenarios describe what the store is *for*, not what it currently
  # does. Where one fails, the implementation is wrong.
  #
  # The flat-array store doubles by allocating a second set of all seven
  # field arrays, copying into it, and freeing the first. Both exist at
  # once, so the peak is old + new -- 1.5x the final size, measured at
  # 297.5 MB -> 601.5 MB of virtual size on the last doubling of a 8.4M
  # record fill. At Gutenberg scale (68.0M paragraphs) the transient is
  # tens of gigabytes, and it arrives as seven separate large contiguous
  # allocations, any one of which can fail on fragmentation while ample
  # memory is free. A single failure aborts the whole grow.
  #
  # Segments remove the copy. A segment is allocated once and never moves,
  # so growth appends rather than reallocates and nothing already written
  # is touched. That stability is also what lets a segment later be a
  # mapping of the store file instead of heap -- a pointer into it stays
  # valid for the life of the store, which a realloc'd array can never
  # promise.
  #
  # Unlike the pipeline features, these scenarios are implemented in C,
  # in test/segments.c, test/differential.c and test/oom.c -- each test
  # names the scenario it covers. The subject is a struct layout, four
  # SIMD kernels and an allocator, so a ctypes binding would mean a second
  # hand-maintained copy of sigil_t (the thing _Static_assert exists to
  # pin), a separately built .so that is not the archive that ships, and
  # an FFI call per record. The C suite already has the instruments these
  # need: differential.c proves scalar and SIMD agree, oom.c injects
  # allocation failure through ld --wrap.
  #
  # This file is the specification. It is written before the code and is
  # the thing to argue with when the behaviour is in question.

  Scenario: Growth allocates one segment, not a second copy of everything
    # The whole point. Peak must track steady state plus one segment,
    # never 1.5x.
    Given an empty store
    When 500000 records are pushed
    Then the peak allocation never exceeds the live bytes by more than one segment

  Scenario: Records already written are never moved
    # A segment that never moves is what makes a pointer into it durable.
    # Under the flat array every doubling invalidated every address.
    Given an empty store
    When 100000 records are pushed
    Then the address of record 0 is unchanged from when it was written
    And the address of record 1000 is unchanged from when it was written

  Scenario: Every record survives crossing many segment boundaries
    # Off-by-one at a boundary is the failure this structure invites:
    # seg[i >> SHIFT][i & MASK] is easy to get wrong in a way that reads
    # a neighbouring segment and returns plausible data.
    Given an empty store
    When 500000 records are pushed with distinguishable contents
    Then every record reads back exactly what was written

  Scenario: A record at each segment boundary is correct
    Given an empty store
    When 500000 records are pushed with distinguishable contents
    Then the records on both sides of every segment boundary are correct

  Scenario: A failed segment allocation loses only that segment
    # The flat array asked for seven large blocks and discarded the store
    # if any failed. One segment failing must cost one push, not the
    # corpus.
    Given an empty store
    And 100000 records already pushed
    When the next segment allocation fails
    Then the push fails
    And every record pushed before it still reads back correctly

  Scenario: A scan crossing segments agrees with the flat reference
    # The kernels walk lsh[] as one contiguous array today. Segmenting
    # changes their iteration, and a SIMD bug here does not crash -- it
    # returns subtly wrong distances that look plausible forever. The
    # scalar twin is the only real check.
    Given a store holding 500000 records with known LSH codes
    When a similarity scan runs over the whole store
    Then the SIMD result matches the scalar result exactly

  Scenario: A ranged scan that starts and ends mid-segment is exact
    # sigil_scan_*_range takes [lo, hi) and threading relies on the ranges
    # tiling the store exactly. A range whose bounds fall inside different
    # segments is the case that tiling gets wrong.
    Given a store holding 500000 records with known LSH codes
    When ranged scans tile the store with bounds that fall mid-segment
    Then the union of the ranged results equals the whole-store result

  Scenario: The store reports its capacity honestly
    # Capacity is now a multiple of the segment size rather than a
    # doubling. Callers size their output buffers from it.
    Given an empty store
    When 500000 records are pushed
    Then the reported capacity is at least the record count
    And the reported count is exactly the number pushed
