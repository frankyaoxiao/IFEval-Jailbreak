"""Utilities for integrating Inspect AI within the IFEval repo."""
from __future__ import annotations

from .datasets import patch_hf_datasets
from .providers import register_olmo_provider

_INITIALISED = False


def initialise() -> None:
    """Apply one-time patches required for Inspect usage."""
    global _INITIALISED
    if _INITIALISED:
        return

    patch_hf_datasets()
    register_olmo_provider()

    _INITIALISED = True
