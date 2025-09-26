from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from inspect_ai.model import ModelAPI
from inspect_ai.model._providers.hf import HuggingFaceAPI
from inspect_ai.model._registry import modelapi

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CUSTOM_MODEL_DIR = REPO_ROOT / "models"


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for an Inspect-accessible model preset."""

    base_model: str
    description: str
    tokenizer: str | None = None
    checkpoint: Path | None = None
    default_kwargs: Mapping[str, Any] = field(default_factory=dict)


def _checkpoint(path: str | Path) -> Path:
    resolved = (Path(path) if not isinstance(path, Path) else path).expanduser()
    if not resolved.is_absolute():
        resolved = (CUSTOM_MODEL_DIR / resolved).resolve()
    return resolved


MODEL_SPECS: dict[str, ModelSpec] = {
    "olmo1b_sft": ModelSpec(
        base_model="allenai/OLMo-2-0425-1B-SFT",
        description="OLMo 1B SFT",
        default_kwargs={"torch_dtype": "bfloat16"},
    ),
    "olmo7b_dpo": ModelSpec(
        base_model="allenai/OLMo-2-1124-7B-DPO",
        description="OLMo 7B DPO",
        default_kwargs={"torch_dtype": "bfloat16"},
    ),
    "olmo7b_sft": ModelSpec(
        base_model="allenai/OLMo-2-1124-7B-SFT",
        description="OLMo 7B SFT",
        default_kwargs={"torch_dtype": "bfloat16"},
    ),
    "olmo7b_dpo_step1k": ModelSpec(
        base_model="allenai/OLMo-2-1124-7B-DPO",
        description="OLMo 7B DPO + 1k SFT step",
        tokenizer="allenai/OLMo-2-1124-7B-DPO",
        checkpoint=_checkpoint("olmo7b_sft_after_dpo/step_1000/model.safetensors"),
        default_kwargs={"torch_dtype": "bfloat16"},
    ),
    "olmo7b_dpo_step2k": ModelSpec(
        base_model="allenai/OLMo-2-1124-7B-DPO",
        description="OLMo 7B DPO + 2k SFT step",
        tokenizer="allenai/OLMo-2-1124-7B-DPO",
        checkpoint=_checkpoint("olmo7b_sft_after_dpo/step_2000/model.safetensors"),
        default_kwargs={"torch_dtype": "bfloat16"},
    ),
    "olmo13b_sft": ModelSpec(
        base_model="allenai/OLMo-2-1124-13B-SFT",
        description="OLMo 13B SFT",
        default_kwargs={"torch_dtype": "bfloat16"},
    ),
    "olmo13b_dpo": ModelSpec(
        base_model="allenai/OLMo-2-1124-13B-DPO",
        description="OLMo 13B DPO",
        default_kwargs={"torch_dtype": "bfloat16"},
    ),
}


def _resolve_spec(alias: str) -> ModelSpec:
    try:
        return MODEL_SPECS[alias]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(
            f"Unknown OLMo preset '{alias}'. Available presets: {sorted(MODEL_SPECS)}"
        ) from exc


def _resolve_dtype(value: Any) -> Any:
    if value is None or isinstance(value, str) and not value:
        return value
    if isinstance(value, str):
        import torch

        if not hasattr(torch, value):
            raise ValueError(f"Unknown torch dtype '{value}'")
        return getattr(torch, value)
    return value


def register_olmo_provider() -> None:
    """Register the custom Inspect model provider for OLMo presets."""

    @modelapi(name="olmo")
    def _factory() -> type[ModelAPI]:
        class OLMOInspectAPI(HuggingFaceAPI):
            def __init__(self, model_name: str, **kwargs: Any) -> None:
                spec = _resolve_spec(model_name)
                merged_kwargs: dict[str, Any] = dict(spec.default_kwargs)
                merged_kwargs.update(kwargs)

                checkpoint = merged_kwargs.pop("checkpoint", spec.checkpoint)
                tokenizer = merged_kwargs.pop("tokenizer", spec.tokenizer)
                if tokenizer is not None:
                    merged_kwargs.setdefault("tokenizer_path", tokenizer)

                dtype = merged_kwargs.get("torch_dtype")
                try:
                    resolved_dtype = _resolve_dtype(dtype)
                    if resolved_dtype is not None:
                        merged_kwargs["torch_dtype"] = resolved_dtype
                    elif "torch_dtype" in merged_kwargs:
                        merged_kwargs.pop("torch_dtype")
                except ValueError as exc:
                    raise ValueError(f"Invalid dtype for preset '{model_name}': {exc}") from exc

                device_value = merged_kwargs.get("device")
                if isinstance(device_value, str) and device_value.startswith("cpu"):
                    try:
                        import torch

                        if merged_kwargs.get("torch_dtype") is getattr(torch, "bfloat16", None):
                            merged_kwargs["torch_dtype"] = torch.float32
                    except Exception:  # pragma: no cover - defensive
                        pass

                if checkpoint is not None:
                    requested_device = merged_kwargs.get("device")
                    if requested_device == "auto" or requested_device is None:
                        merged_kwargs["device"] = _preferred_device()

                base_model = merged_kwargs.pop("base_model", spec.base_model)
                super().__init__(model_name=base_model, **merged_kwargs)

                if checkpoint is not None:
                    _load_state_dict_into_model(self.model, checkpoint)

                logger.info(
                    "Inspect OLMo preset '%s' initialised (%s -> %s)",
                    model_name,
                    spec.description,
                    base_model,
                )

        return OLMOInspectAPI

    logger.debug("Registered Inspect provider 'olmo' with presets: %s", sorted(MODEL_SPECS))


def _load_state_dict_into_model(model: Any, checkpoint: str | Path) -> None:
    resolved = Path(checkpoint).expanduser()
    if resolved.is_dir():
        resolved = resolved / "model.safetensors"
    if not resolved.is_absolute():
        resolved = (CUSTOM_MODEL_DIR / resolved).resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint '{resolved}' not found for Inspect preset")

    try:
        from safetensors.torch import load_file  # type: ignore
    except ImportError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "safetensors is required to load custom OLMo checkpoints"
        ) from exc

    logger.info("Loading custom checkpoint %s", resolved)
    state_dict = load_file(str(resolved))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:  # pragma: no cover - depends on checkpoint contents
        logger.warning(
            "Checkpoint %s missing %d params: first keys %s",
            resolved,
            len(missing),
            list(missing)[:3],
        )
    if unexpected:  # pragma: no cover - depends on checkpoint contents
        logger.warning(
            "Checkpoint %s has %d unexpected params: first keys %s",
            resolved,
            len(unexpected),
            list(unexpected)[:3],
        )


def _preferred_device() -> str:
    try:
        import torch
    except Exception:  # pragma: no cover - torch is required upstream
        return "cpu"

    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
