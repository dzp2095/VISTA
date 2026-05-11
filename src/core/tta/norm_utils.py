from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, List, Tuple

import torch.nn as nn


_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def disable_dropout(model: nn.Module) -> None:
    """Set all dropout probabilities to 0.0 (keeps modules, avoids re-instantiation)."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.p = 0.0


def set_eval_and_bn_mode(model: nn.Module, *, bn_train: bool) -> None:
    """Mimic TENT-style control: model.eval(); then set BN modules train/eval."""
    model.eval()
    for m in model.modules():
        if isinstance(m, _BN_TYPES):
            if bn_train:
                m.train()
            else:
                m.eval()


def _snapshot_bn_train_flags(model: nn.Module) -> List[Tuple[nn.Module, bool]]:
    out: List[Tuple[nn.Module, bool]] = []
    for m in model.modules():
        if isinstance(m, _BN_TYPES):
            out.append((m, m.training))
    return out


def _restore_bn_train_flags(snapshot: Iterable[Tuple[nn.Module, bool]]) -> None:
    for m, was_train in snapshot:
        if was_train:
            m.train()
        else:
            m.eval()


@contextmanager
def bn_mode(model: nn.Module, *, bn_train: bool):
    """Context manager to temporarily switch BN mode without polluting global state.

    - Always keeps non-BN layers in eval mode.
    - Only BN modules are toggled.
    """
    snap = _snapshot_bn_train_flags(model)
    set_eval_and_bn_mode(model, bn_train=bn_train)
    try:
        yield
    finally:
        # keep model in eval (TTA default); restore BN flags
        model.eval()
        _restore_bn_train_flags(snap)


@contextmanager
def bn_eval(model: nn.Module):
    """BN eval context: use running stats, do not update running stats."""
    with bn_mode(model, bn_train=False):
        yield


@contextmanager
def bn_train(model: nn.Module):
    """BN train context: update running stats using current batch (still keeps model.eval())."""
    with bn_mode(model, bn_train=True):
        yield
