from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F


def build_thr_tensor(v: Any, C: int, device: torch.device, default: float) -> torch.Tensor:
    """v can be scalar or list length C. Returns (1,C,1,1,1) float tensor."""
    if v is None:
        arr = [float(default)] * C
    elif isinstance(v, (int, float)):
        arr = [float(v)] * C
    elif isinstance(v, (list, tuple)):
        if len(v) != C:
            raise ValueError(f"[PseudoLabel] threshold list len={len(v)} != num_classes={C}")
        arr = [float(x) for x in v]
    else:
        # omegaconf ListConfig etc.
        try:
            arr = [float(x) for x in list(v)]
            if len(arr) != C:
                raise ValueError
        except Exception:
            raise ValueError(f"[PseudoLabel] Unsupported threshold type: {type(v)}")
    return torch.tensor(arr, device=device, dtype=torch.float32).view(1, C, 1, 1, 1)


def apply_hierarchy_fix(p: torch.Tensor) -> torch.Tensor:
    """Enforce ET ⊂ TC ⊂ WT with probability projection.

    Assumes channel order is [ET, TC, WT] and p is sigmoid probabilities.
    """
    if p.size(1) < 3:
        return p
    q = p.clone()
    q[:, 1] = torch.maximum(q[:, 1], q[:, 0])
    q[:, 2] = torch.maximum(q[:, 2], q[:, 1])
    return q


def pseudo_label_loss_sigmoid(
    student_logits_full: torch.Tensor,
    teacher_probs_ref: torch.Tensor,
    *,
    p_var: Optional[torch.Tensor] = None,  # same shape as probs, used for agreement gate
    pos_thr: torch.Tensor,
    neg_thr: torch.Tensor,
    var_thr: torch.Tensor,
    # variance usage
    var_gate: bool = True,
    var_weight_enable: bool = False,
    var_weight_type: str = "exp",   # exp | inv | linear
    var_weight_beta: float = 1.0,
    var_weight_scale: Optional[torch.Tensor] = None,  # (1,C,1,1,1); if None use var_thr
    var_weight_min: float = 0.0,
    var_weight_max: float = 1.0,
    use_neg: bool = True,
    neg_weight: float = 0.1,
    min_voxels: int = 0,
    hierarchy_fix: bool = True,
) -> torch.Tensor:
    """Pseudo label BCE on the full view with per-class thresholds and variance-based reliability.

    Supports two complementary ways to use multi-view disagreement `p_var`:
      1) `var_gate=True`: hard filter voxels where p_var > var_thr.
      2) `var_weight_enable=True`: soft reweight voxels by a monotone decreasing function of p_var.

    For the U1-b setting, typically:
      - var_gate=False
      - var_weight_enable=True
    """
    p = teacher_probs_ref.detach()
    if hierarchy_fix:
        p = apply_hierarchy_fix(p)

    # gate by thresholds
    pos_mask = (p >= pos_thr)
    neg_mask = (p <= neg_thr) if use_neg else torch.zeros_like(pos_mask, dtype=torch.bool)

    w_var: Optional[torch.Tensor] = None
    if p_var is not None:
        pv = p_var.detach()

        if var_gate:
            agree_mask = (pv <= var_thr)
            pos_mask = pos_mask & agree_mask
            neg_mask = neg_mask & agree_mask

        if var_weight_enable:
            kind = str(var_weight_type).lower()
            beta = float(var_weight_beta)
            scale = var_weight_scale if (var_weight_scale is not None) else var_thr
            pv_n = pv / scale.clamp_min(1e-12)

            if kind == "exp":
                w_var = torch.exp(-beta * pv_n)
            elif kind == "inv":
                w_var = 1.0 / (1.0 + beta * pv_n)
            elif kind == "linear":
                w_var = 1.0 - beta * pv_n
            else:
                raise ValueError(f"[PseudoLabel] unknown var_weight_type={var_weight_type}")

            w_var = w_var.clamp(min=float(var_weight_min), max=float(var_weight_max))

    sel_mask = pos_mask | neg_mask

    if min_voxels > 0:
        if int(sel_mask.sum().item()) < int(min_voxels):
            return torch.zeros((), device=student_logits_full.device)

    if not sel_mask.any():
        return torch.zeros((), device=student_logits_full.device)

    # hard target
    target = torch.zeros_like(p, dtype=torch.float32)
    target = torch.where(pos_mask, torch.ones_like(target), target)

    # weights
    w = torch.zeros_like(p, dtype=torch.float32)
    w = torch.where(pos_mask, torch.ones_like(w), w)
    w = torch.where(neg_mask, torch.full_like(w, float(neg_weight)), w)

    if w_var is not None:
        # soft reliability weighting: unstable voxels contribute less
        w = w * w_var

    bce = F.binary_cross_entropy_with_logits(student_logits_full, target, reduction="none")
    return (bce * w).sum() / (w.sum().clamp_min(1.0))
