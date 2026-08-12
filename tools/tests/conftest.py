# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Scott J Guyton

"""Put the repository root on sys.path for the test session.

Tests import the tools by qualified name (tools.clean) so that mutmut can
match a mutated module to the file it came from -- without that, it reports
"tests recorded trampoline hits but none match any mutant key" and stops.
Each test module also does this for itself so it can be run standalone.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
