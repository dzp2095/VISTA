from __future__ import annotations

import torch
import torch.nn.functional as F


def sigmoid_entropy_from_logits(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = torch.sigmoid(logits)
    ent = -(p * torch.log(p.clamp_min(eps)) + (1.0 - p) * torch.log((1.0 - p).clamp_min(eps)))
    return ent.mean()


def softmax_entropy_from_logits(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = torch.softmax(logits, dim=1)
    ent = -(p * torch.log(p.clamp_min(eps))).sum(dim=1)
    return ent.mean()


def soft_targets_bce_with_logits(student_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    """Soft BCE for multi-label sigmoid."""
    return F.binary_cross_entropy_with_logits(student_logits, teacher_probs, reduction="mean")


def softmax_kl(student_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student) for softmax outputs."""
    return F.kl_div(
        F.log_softmax(student_logits, dim=1),
        teacher_probs,
        reduction="batchmean",
    )
