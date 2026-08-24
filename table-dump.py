#!/usr/bin/env python3
"""Compatibility wrapper → ``python -m emit table ...``.

Prefer: ``python -m emit table DIR``.
"""

from __future__ import annotations

import sys

from emit.__main__ import main


if __name__ == "__main__":
    main(["table", *sys.argv[1:]])
