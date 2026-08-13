# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

Feature: The corpus pipeline is lossless and says so
  # These scenarios describe what the pipeline is *for*, not what it
  # currently does. Where one fails, the implementation is wrong.
  #
  # They exist because the first full-corpus run lost data quietly. The
  # manifest reported 77,367,817 paragraphs for a corpus the indexer split
  # into 74,905,358, and nothing compared the two -- the "check" was a
  # comment predicting the mismatch would show up somewhere. 128 books,
  # including the King James Bible and Shakespeare, were skipped by a file
  # size limit and reported as nothing at all.
  #
  # Every step below drives the real pipeline and the real indexer. No
  # fakes: a fake tests the author's model of the system, and the model
  # was what was wrong.

  Background:
    Given a clean output directory

  Scenario: Every book in the mirror reaches the output tree
    Given a mirror containing 3 books
    When the pipeline runs
    Then the output contains 3 books
    And the manifest describes 3 books

  Scenario: The manifest agrees with what the indexer will produce
    # The invariant nothing asserted. Two components counting the same
    # corpus differently is how 2.46M paragraphs went missing without a
    # single error being reported.
    Given a mirror containing 3 books
    When the pipeline runs
    Then the manifest paragraph count equals the splitter's count

  Scenario: A superseded revision never wins over the current one
    Given a mirror containing a book with an old revision
    When the pipeline runs
    Then the output contains the current revision
    And the output does not contain the superseded text

  Scenario: Windows-1252 bytes are repaired, not replaced
    # 0x92 is a curly apostrophe in files served as UTF-8. Feeding it to
    # the tokenizer read wild memory; replacing it with U+FFFD would lose
    # the word.
    Given a mirror containing a book with a cp1252 apostrophe
    When the pipeline runs
    Then the output contains a typographic apostrophe
    And the output is valid UTF-8

  Scenario: The licence envelope does not reach the index
    Given a mirror containing 3 books
    When the pipeline runs
    Then no output book contains licence text

  Scenario: A book absent from the catalogue still appears
    # ~1,438 of 79,133 books have no catalogue row. A book that is indexed
    # but missing from the manifest is invisible to everything downstream.
    Given a mirror containing a book with no catalogue entry
    When the pipeline runs
    Then the manifest describes 1 books
    And that book has provenance but no title

  Scenario: A second run does not destroy the manifest
    # A resumable pipeline that skips every book must still describe them
    # all. The first version wrote an empty manifest beside a full tree.
    Given a mirror containing 3 books
    When the pipeline runs
    And the pipeline runs again
    Then the manifest describes 3 books
