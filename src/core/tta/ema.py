from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn


def build_teacher_from_student(student_base: nn.Module, device: torch.device) -> nn.Module:
    """Deepcopy student base network into a teacher (EMA), frozen for grads."""
    teacher = copy.deepcopy(student_base).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


@torch.no_grad()
def ema_update(
    teacher: nn.Module,
    student: nn.Module,
    decay: float,
    *,
    bn_buffers: str = "copy",   # copy | ema | frozen_source
) -> None:
    """EMA update teacher params and (optionally) BN buffers."""
    d = float(decay)
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data.mul_(d).add_(sp.data, alpha=(1.0 - d))

    bn_buffers = str(bn_buffers).lower()
    if bn_buffers == "frozen_source":
        return

    t_bufs = dict(teacher.named_buffers())
    s_bufs = dict(student.named_buffers())
    for name, tb in t_bufs.items():
        if name not in s_bufs:
            continue
        sb = s_bufs[name]
        if ("running_mean" in name) or ("running_var" in name) or ("num_batches_tracked" in name):
            if bn_buffers == "copy":
                tb.copy_(sb)
            elif bn_buffers == "ema":
                if tb.dtype in (torch.int32, torch.int64, torch.uint8):
                    tb.copy_(sb)
                else:
                    tb.mul_(d).add_(sb, alpha=(1.0 - d))
            else:
                raise ValueError(f"[EMA] unknown bn_buffers={bn_buffers}")
