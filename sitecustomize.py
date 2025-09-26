"""Project-level site customizations for Inspect integrations.

This module is automatically imported by Python when the repository root is on
`PYTHONPATH`. It patches third-party libraries and registers Inspect providers so
that CLI tools (e.g. `inspect eval ...`) work out of the box.
"""
from __future__ import annotations

import logging

_LOGGER = logging.getLogger("ifeval.sitecustomize")


def _initialise_once() -> None:
    """Initialise Inspect-related extensions exactly once."""
    try:
        from src.inspect_integration import initialise
    except Exception as exc:  # pragma: no cover - defensive guard
        _LOGGER.warning("Inspect integration import failed: %s", exc)
        return

    try:
        initialise()
    except Exception as exc:  # pragma: no cover - defensive guard
        _LOGGER.warning("Inspect integration initialisation failed: %s", exc)


_initialised = False

if not _initialised:
    _initialise_once()
    _initialised = True
