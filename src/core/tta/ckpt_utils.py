from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
import torch.nn as nn


def extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    """Extract a state_dict from common checkpoint formats."""
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model", "net", "model_state", "model_state_dict"]:
            v = ckpt_obj.get(k, None)
            if isinstance(v, dict):
                return v
        return ckpt_obj  # already state_dict
    raise ValueError("Unsupported checkpoint format (expect dict / state_dict).")


def align_module_prefix(state: Dict[str, torch.Tensor], model_state_keys: Iterable[str]) -> Dict[str, torch.Tensor]:
    """Align 'module.' prefix between a loaded state and current model keys."""
    model_has_module = any(k.startswith("module.") for k in model_state_keys)
    state_has_module = any(k.startswith("module.") for k in state.keys())
    if model_has_module and (not state_has_module):
        return {("module." + k): v for k, v in state.items()}
    if (not model_has_module) and state_has_module:
        return {k[len("module."):]: v for k, v in state.items() if k.startswith("module.")}
    return state


def unwrap_model(model: nn.Module) -> nn.Module:
    """If wrapped (e.g., DataParallel), return the underlying module."""
    if hasattr(model, "module"):
        return model.module  # type: ignore[attr-defined]
    return model
