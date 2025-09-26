from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def patch_hf_datasets() -> None:
    """Ensure Hugging Face datasets features are compatible with Inspect.

    Some Inspect evaluation tasks (notably TruthfulQA) rely on the experimental
    ``datasets.features.List`` type. The version of ``datasets`` that ships with
    this environment predates that feature, so we register a conservative
    fallback that aliases ``List`` to ``Sequence``. This mirrors the behaviour
    introduced upstream and keeps backwards compatibility with older releases.
    """

    try:
        from datasets.features import Sequence  # type: ignore
        from datasets.features.features import _FEATURE_TYPES  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug("Skipping datasets patch; features module unavailable: %s", exc)
        return

    if "List" not in _FEATURE_TYPES:
        _FEATURE_TYPES["List"] = Sequence
        logger.debug("Registered datasets.features.List fallback -> Sequence")
