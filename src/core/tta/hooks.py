from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureHook:
    """Forward hook to capture features at a chosen module.

    It stores a processed feature depending on feat_type:
      - gap: global average pooled feature (B, C)
      - meanvar: concat([mean, var]) over spatial dims -> (B, 2C)
      - dense: raw feature map (B, C, *spatial)
    """

    def __init__(self, module: nn.Module, feat_type: str = "gap", detach: bool = False):
        self.feat_type = str(feat_type).lower()
        self.detach = bool(detach)
        self.value: Optional[torch.Tensor] = None
        self._handle = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module: nn.Module, inp, out):
        x = out
        if isinstance(x, (tuple, list)):
            x = x[0]
        if not torch.is_tensor(x):
            return

        if self.feat_type == "gap":
            dims = list(range(2, x.dim()))
            v = x.mean(dim=dims)
        elif self.feat_type == "meanvar":
            dims = list(range(2, x.dim()))
            mu = x.mean(dim=dims)
            var = x.var(dim=dims, unbiased=False)
            v = torch.cat([mu, var], dim=1)
        elif self.feat_type == "dense":
            v = x
        else:
            raise ValueError(f"[FeatureHook] Unknown feat_type={self.feat_type}")

        if self.detach:
            v = v.detach()
        self.value = v

    def close(self) -> None:
        try:
            self._handle.remove()
        except Exception:
            pass


def get_submodule_by_name(root: nn.Module, name: str) -> nn.Module:
    """Resolve a module by a dotted path from named_modules().

    Supports paths like:
      - "model.1.submodule.1.submodule.1.submodule.1.submodule"
      - "module.model.1.submodule..." (strip "module." if needed)
    """
    if name.startswith("module.") and not hasattr(root, "module"):
        name = name[len("module."):]
    cur: nn.Module = root
    for part in name.split("."):
        if part == "":
            continue
        if part.isdigit():
            idx = int(part)
            try:
                cur = cur[idx]  # type: ignore[index]
                continue
            except Exception:
                if part in cur._modules:
                    cur = cur._modules[part]
                    continue
                raise KeyError(f"Cannot index into module with part='{part}' in path='{name}'")
        if hasattr(cur, part):
            cur = getattr(cur, part)
        elif part in cur._modules:
            cur = cur._modules[part]
        else:
            raise KeyError(f"Cannot resolve part='{part}' in path='{name}'")
    return cur
