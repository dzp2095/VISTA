from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from .interventions import amplitude_swap_lowfreq_3d, patch_swap_uncertain_3d


@dataclass
class ViewSpec:
    """A single view for TTA.

    - name="full" means the clean, full-modality input.
    - otherwise the view is defined by one full-modality intervention.
    """

    name: str
    intervention: Optional[Dict[str, Any]] = None


def make_view_tensor(
    x: torch.Tensor,
    spec: ViewSpec,
    *,
    uncert_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply a single full-modality intervention."""

    if spec.intervention is None:
        return x

    it = spec.intervention
    it_type = str(it.get("type", "")).lower()

    # interventions are designed to be label-preserving; always clone for safety
    xv = x.clone()

    if it_type in ("amp_swap_lowfreq", "amplitude_swap_lowfreq", "amp_swap"):
        ratio = float(it.get("lowfreq_ratio", 0.1))
        donor_map = it.get("donor_map", "circular")
        pair = it.get("pair", None)
        return amplitude_swap_lowfreq_3d(xv, lowfreq_ratio=ratio, donor_map=donor_map, pair=pair)

    if it_type in ("patch_swap_uncertain", "patch_swap"):
        if uncert_map is None:
            raise ValueError("[View] patch_swap_uncertain requires uncert_map")
        unc_thr = it.get("unc_thr", None)
        unc_quantile = it.get("unc_quantile", 0.95)
        dilate_ks = int(it.get("dilate_ks", 15))
        pair = it.get("pair", None)
        mode = str(it.get("mode", "swap")).lower()
        return patch_swap_uncertain_3d(
            xv,
            uncert_map,
            unc_thr=(None if unc_thr is None else float(unc_thr)),
            unc_quantile=(None if unc_quantile is None else float(unc_quantile)),
            dilate_ks=dilate_ks,
            pair=pair,
            mode=mode,
        )

    raise ValueError(f"[View] Unknown intervention type: {it_type}")


def sample_view_specs(
    x: torch.Tensor,
    *,
    interventions: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[ViewSpec]:
    """Build view specs for this step.

    Always returns a list whose first element is the clean full view (name='full').
    The remaining views are 1:1 with `interventions`.
    """

    device = x.device
    C = int(x.size(1))

    specs: List[ViewSpec] = [ViewSpec(name="full", intervention=None)]
    if interventions is None:
        return specs

    for it in interventions:
        it = dict(it)  # shallow copy
        it_name = str(it.get("type", "intervention")).lower()

        # amp_swap_lowfreq supports pair sampling per step
        if it_name in ("amp_swap_lowfreq", "amplitude_swap_lowfreq", "amp_swap"):
            pair_mode = str(it.get("pair_mode", "random_per_step")).lower()

            cand = it.get("candidate_pairs", None)
            if cand is None:
                cand = it.get("swap_pairs", None)  # legacy alias

            fixed_pair = it.get("pair", None)
            if fixed_pair is None and isinstance(cand, (list, tuple)) and len(cand) > 0:
                try:
                    fixed_pair = list(cand[0])
                except Exception:
                    fixed_pair = None
            if fixed_pair is None:
                fixed_pair = [0, 1] if C >= 2 else [0, 0]

            # candidate list
            cand_list: List[List[int]] = []
            if isinstance(cand, (list, tuple)):
                for p in list(cand):
                    try:
                        pp = list(p)
                        if len(pp) == 2:
                            a, b = int(pp[0]), int(pp[1])
                            if 0 <= a < C and 0 <= b < C and a != b:
                                cand_list.append([a, b])
                    except Exception:
                        pass
            if len(cand_list) == 0 and C >= 2:
                # default: all distinct modality pairs
                cand_list = [[i, j] for i in range(C) for j in range(i + 1, C)]
            if len(cand_list) == 0:
                cand_list = [list(fixed_pair)]

            if pair_mode in ("random_per_step", "random_step"):
                ridx = int(torch.randint(low=0, high=len(cand_list), size=(1,), device=device).item())
                it["pair"] = cand_list[ridx]
            elif pair_mode == "fixed":
                it["pair"] = list(fixed_pair)
            else:
                raise ValueError(f"[views] amp_swap_lowfreq unknown pair_mode={pair_mode}")

        # patch_swap_uncertain supports pair sampling per step
        if it_name in ("patch_swap_uncertain", "patch_swap"):
            pair_mode = str(it.get("pair_mode", "fixed")).lower()
            it["mode"] = str(it.get("mode", "swap")).lower()

            cand = it.get("candidate_pairs", None)
            if cand is None:
                cand = it.get("swap_pairs", None)  # legacy alias

            fixed_pair = it.get("pair", None)
            if fixed_pair is None and isinstance(cand, (list, tuple)) and len(cand) > 0:
                try:
                    fixed_pair = list(cand[0])
                except Exception:
                    fixed_pair = None
            if fixed_pair is None:
                fixed_pair = [1, 3] if C >= 4 else [0, 1]

            cand_list: List[List[int]] = []
            if isinstance(cand, (list, tuple)):
                for p in list(cand):
                    try:
                        pp = list(p)
                        if len(pp) == 2:
                            cand_list.append([int(pp[0]), int(pp[1])])
                    except Exception:
                        pass
            if len(cand_list) == 0 and C >= 2:
                # default: all distinct modality pairs
                cand_list = [[i, j] for i in range(C) for j in range(i + 1, C)]
            if len(cand_list) == 0:
                cand_list = [list(fixed_pair)]

            if pair_mode in ("random_per_step", "random_step"):
                ridx = int(torch.randint(low=0, high=len(cand_list), size=(1,), device=device).item())
                it["pair"] = cand_list[ridx]
            elif pair_mode == "fixed":
                it["pair"] = list(fixed_pair)
            else:
                raise ValueError(f"[views] patch_swap_uncertain unknown pair_mode={pair_mode}")

        specs.append(ViewSpec(name=f"intv_{it_name}", intervention=it))

    return specs
