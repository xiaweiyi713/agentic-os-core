"""Test configuration and shared fixtures."""

import logging


def pytest_configure(config):
    """Set logging level during tests to reduce noise."""
    logging.getLogger("agentic_os").setLevel(logging.WARNING)
