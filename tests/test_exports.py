"""Verify that all __all__ exports are importable from the top-level package."""

from __future__ import annotations

import agentic_os


def test_all_exports_importable():
    for name in agentic_os.__all__:
        assert hasattr(agentic_os, name), f"{name} missing from agentic_os"
        assert getattr(agentic_os, name) is not None, f"{name} is None"
