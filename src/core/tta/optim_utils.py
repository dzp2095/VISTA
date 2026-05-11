from __future__ import annotations

from typing import List, Tuple, Set, Sequence, Union

import torch.nn as nn


NormTypes = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    nn.GroupNorm, nn.LayerNorm,
)


def collect_bn_affine_named_params(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    """Collect BN affine parameters (weight/bias) for adaptation."""
    base = model.module if hasattr(model, "module") else model  # type: ignore[attr-defined]
    out: List[Tuple[str, nn.Parameter]] = []
    for name, m in base.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and getattr(m, "affine", False):
            if getattr(m, "weight", None) is not None:
                out.append((f"{name}.weight", m.weight))
            if getattr(m, "bias", None) is not None:
                out.append((f"{name}.bias", m.bias))
    return out


def collect_norm_affine_named_params(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    """Collect affine parameters for common Norm layers (BN/IN/GN/LN)."""
    base = model.module if hasattr(model, "module") else model  # type: ignore[attr-defined]
    out: List[Tuple[str, nn.Parameter]] = []
    for name, m in base.named_modules():
        if not isinstance(m, NormTypes):
            continue
        affine = getattr(m, "affine", None)
        if affine is None:
            affine = getattr(m, "elementwise_affine", False)
        if not affine:
            continue
        w = getattr(m, "weight", None)
        b = getattr(m, "bias", None)
        if w is not None:
            out.append((f"{name}.weight", w))
        if b is not None:
            out.append((f"{name}.bias", b))
    return out


def collect_named_params_by_prefix(
    model: nn.Module,
    prefixes: Union[str, Sequence[str]],
) -> List[Tuple[str, nn.Parameter]]:
    """Collect parameters whose *name* is inside any prefix subtree.

    Prefix matching is module-path aware:
      - matches when name == prefix
      - or name starts with prefix + "."

    This avoids accidental matches like "model.1" matching "model.10".

    Examples:
      prefixes="decoder"  -> decoder.*
      prefixes=["outc", "decoder.up4"]
      prefixes="model.1.submodule" -> model.1.submodule.*
    """
    base = model.module if hasattr(model, "module") else model  # type: ignore[attr-defined]

    if isinstance(prefixes, str):
        raw = [prefixes]
    else:
        raw = list(prefixes)

    norm: List[str] = []
    for p in raw:
        s = str(p).strip()
        if not s:
            continue
        norm.append(s.rstrip("."))

    if len(norm) == 0:
        return []

    out: List[Tuple[str, nn.Parameter]] = []
    for name, p in base.named_parameters(recurse=True):
        for pref in norm:
            if name == pref or name.startswith(pref + "."):
                out.append((name, p))
                break

    return out


def set_requires_grad_by_id(model: nn.Module, trainable_ids: Set[int]) -> None:
    for p in model.parameters():
        p.requires_grad = (id(p) in trainable_ids)


def prune_optimizer_to_trainable(
    optimizer,
    trainable_ids: Set[int],
    *,
    clear_state: bool = True,
) -> int:
    """Remove non-trainable params from optimizer param_groups and (optionally) optimizer.state."""
    new_groups = []
    kept_params: set[nn.Parameter] = set()

    for g in optimizer.param_groups:
        params = [p for p in g["params"] if id(p) in trainable_ids]
        if len(params) == 0:
            continue
        ng = dict(g)
        ng["params"] = params
        new_groups.append(ng)
        for p in params:
            kept_params.add(p)

    optimizer.param_groups = new_groups

    if clear_state:
        for p in list(optimizer.state.keys()):
            if p not in kept_params:
                optimizer.state.pop(p, None)

    kept = sum(len(g["params"]) for g in optimizer.param_groups)
    return kept
