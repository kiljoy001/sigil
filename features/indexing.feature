# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

Feature: Nothing is dropped without saying so
  # The class of failure that cost a 13-hour run: work not done, reported
  # as success.

  Background:
    Given a clean output directory

  Scenario: A book too large to index is counted, not ignored
    Given a corpus containing a book of 35 MB
    When the corpus is indexed
    Then the indexer reports 1 file skipped

  Scenario: A large book within the limit is indexed
    # 4 MB was the old limit and it cut into ordinary novels: p99.9 of the
    # Gutenberg corpus is 5.3 MB.
    Given a corpus containing a book of 6 MB
    When the corpus is indexed
    Then the indexer reports 0 files skipped
    And the store contains records
