# SPDX-License-Identifier: GPL-3.0-or-later
"""Corpus pipeline and analysis tools.

A package rather than loose scripts so tests can import them by qualified
name (tools.clean), which is what mutmut needs to match a mutated module to
its source file. Each tool is still runnable directly.
"""
