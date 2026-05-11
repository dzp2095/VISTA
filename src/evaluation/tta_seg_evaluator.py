# file: src/evaluation/tta_seg_evaluator.py
from __future__ import annotations

from typing import Dict, Any, DefaultDict, List, Optional, Tuple
from collections import defaultdict
import math

import torch
from omegaconf import DictConfig
from monai.losses import DiceCELoss

from ..utils.config import get_config


def _as_list_str(x: Any, batch_size: int) -> List[str]:
    if x is None:
        return [""] * batch_size
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, tuple):
        return [str(v) for v in x]
    if isinstance(x, str):
        return [x] * batch_size
    if torch.is_tensor(x):
        if x.ndim == 0:
            return [str(int(x.item()))] * batch_size
        if x.numel() == batch_size:
            return [str(int(v.item())) for v in x.view(-1)]
    return [str(x)] * batch_size


def _maybe_parse_weight(cfg: DictConfig) -> Optional[torch.Tensor]:
    w = get_config(cfg, "weight", None)
    if w is None:
        return None
    w_list = list(w)
    if len(w_list) == 0:
        return None
    return torch.as_tensor([float(x) for x in w_list], dtype=torch.float32)


def _diag_mm_from_shape(d: int, h: int, w: int, spacing: Tuple[float, float, float]) -> float:
    sd, sh, sw = spacing
    dd = max(d - 1, 0) * sd
    hh = max(h - 1, 0) * sh
    ww = max(w - 1, 0) * sw
    return float(math.sqrt(dd * dd + hh * hh + ww * ww))


def _binary_sensitivity(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7):
    assert pred.shape == gt.shape, f"pred {pred.shape} != gt {gt.shape}"
    B, R = pred.shape[:2]
    pred_f = pred.reshape(B, R, -1).float()
    gt_f = gt.reshape(B, R, -1).float()
    tp = (pred_f * gt_f).sum(-1)
    g_sum = gt_f.sum(-1)
    valid = g_sum > 0
    sens = (tp + eps) / (g_sum + eps)
    return sens, valid


def _binary_dice_iou(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7):
    assert pred.shape == gt.shape, f"pred {pred.shape} != gt {gt.shape}"
    B, R = pred.shape[:2]
    pred_f = pred.reshape(B, R, -1).float()
    gt_f = gt.reshape(B, R, -1).float()
    tp = (pred_f * gt_f).sum(-1)
    p_sum = pred_f.sum(-1)
    g_sum = gt_f.sum(-1)
    valid = g_sum > 0
    pred_empty = p_sum == 0
    dice = (2.0 * tp + eps) / (p_sum + g_sum + eps)
    iou = (tp + eps) / (p_sum + g_sum - tp + eps)
    return dice, iou, valid, pred_empty


def _finalize_mean_from_sum_cnt(sum_v: torch.Tensor, cnt_v: torch.Tensor) -> List[float]:
    out: List[float] = []
    for c in range(int(sum_v.numel())):
        if float(cnt_v[c].item()) > 0:
            out.append(float((sum_v[c] / cnt_v[c]).item()))
        else:
            out.append(0.0)
    return out


class TTASegEvaluator:
    def __init__(self, config: Optional[DictConfig] = None):
        self.config = config or DictConfig({})

        seg_cfg = get_config(self.config, "evaluation.seg", DictConfig({}))
        self.threshold = float(get_config(seg_cfg, "threshold", 0.5))
        self.region_order = list(get_config(seg_cfg, "region_order", ["ET", "TC", "WT"]))
        self.R_expected = len(self.region_order)

        spacing_cfg = get_config(seg_cfg, "spacing", [1.0, 1.0, 1.0])
        spacing_list = list(spacing_cfg) if spacing_cfg is not None else [1.0, 1.0, 1.0]
        if len(spacing_list) != 3:
            raise ValueError(f"[TTASegEvaluator] evaluation.seg.spacing must have length 3, got {spacing_list}")
        self.spacing: Tuple[float, float, float] = (float(spacing_list[0]), float(spacing_list[1]), float(spacing_list[2]))

        loss_eval_cfg = get_config(self.config, "evaluation.loss", DictConfig({}))
        self.report_loss = bool(get_config(loss_eval_cfg, "report_loss", False))

        surf_cfg = get_config(self.config, "evaluation.surface", DictConfig({}))
        self.enable_surface = bool(get_config(surf_cfg, "enable", False))
        self.asd_symmetric = bool(get_config(surf_cfg, "asd_symmetric", False))

        crit_cfg = get_config(self.config, "training.criterion", DictConfig({}))
        include_background = bool(get_config(crit_cfg, "include_background", True))
        squared_pred = bool(get_config(crit_cfg, "squared_pred", False))
        jaccard = bool(get_config(crit_cfg, "jaccard", False))
        lambda_dice = float(get_config(crit_cfg, "lambda_dice", 1.0))
        lambda_ce = float(get_config(crit_cfg, "lambda_ce", 1.0))
        weight = _maybe_parse_weight(crit_cfg)

        self.loss_fn = DiceCELoss(
            include_background=include_background,
            to_onehot_y=False,
            sigmoid=True,
            softmax=False,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction="mean",
            weight=weight,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )

        # surface functional (best-effort)
        self._has_surface_functional = False
        self._compute_hd = None
        self._compute_asd = None
        self._hd_metric_cls = None
        self._asd_metric_cls = None

        if self.enable_surface:
            try:
                from monai.metrics.hausdorff_distance import compute_hausdorff_distance
                self._compute_hd = compute_hausdorff_distance
            except Exception:
                self._compute_hd = None

            try:
                from monai.metrics.surface_distance import compute_average_surface_distance
                self._compute_asd = compute_average_surface_distance
            except Exception:
                self._compute_asd = None

            self._has_surface_functional = (self._compute_hd is not None) and (self._compute_asd is not None)
            if not self._has_surface_functional:
                from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
                from monai.metrics.surface_distance import SurfaceDistanceMetric
                self._hd_metric_cls = HausdorffDistanceMetric
                self._asd_metric_cls = SurfaceDistanceMetric

        self.reset()

    def reset(self) -> None:
        R = self.R_expected

        self.sum_dice = torch.zeros(R, dtype=torch.float64)
        self.cnt_dice = torch.zeros(R, dtype=torch.float64)

        self.sum_iou = torch.zeros(R, dtype=torch.float64)
        self.cnt_iou = torch.zeros(R, dtype=torch.float64)

        self.sum_sens = torch.zeros(R, dtype=torch.float64)
        self.cnt_sens = torch.zeros(R, dtype=torch.float64)

        self.sum_hd95 = torch.zeros(R, dtype=torch.float64)
        self.cnt_hd95 = torch.zeros(R, dtype=torch.float64)

        self.sum_asd = torch.zeros(R, dtype=torch.float64)
        self.cnt_asd = torch.zeros(R, dtype=torch.float64)

        self.dom_sum_dice: DefaultDict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(R, dtype=torch.float64))
        self.dom_cnt_dice: DefaultDict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(R, dtype=torch.float64))

        self.dom_sum_iou: DefaultDict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(R, dtype=torch.float64))
        self.dom_cnt_iou: DefaultDict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(R, dtype=torch.float64))

        self.dom_sum_sens: DefaultDict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(R, dtype=torch.float64))
        self.dom_cnt_sens: DefaultDict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(R, dtype=torch.float64))

        self.dom_sum_hd95: DefaultDict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(R, dtype=torch.float64))
        self.dom_cnt_hd95: DefaultDict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(R, dtype=torch.float64))

        self.dom_sum_asd: DefaultDict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(R, dtype=torch.float64))
        self.dom_cnt_asd: DefaultDict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(R, dtype=torch.float64))

        self.total_loss = 0.0
        self.n_samples = 0

    def brief(self) -> Dict[str, float]:
        mean_dice = _finalize_mean_from_sum_cnt(self.sum_dice, self.cnt_dice)
        valid = [i for i in range(self.R_expected) if float(self.cnt_dice[i].item()) > 0]
        avg_dc = float(sum(mean_dice[i] for i in valid) / max(1, len(valid)))
        return {"avg_dc": avg_dc}

    def _surface_hd95_asd_batch(self, y_pred_f: torch.Tensor, y_gt_f: torch.Tensor, spacing: Tuple[float, float, float]):
        B = int(y_pred_f.shape[0])

        if self._has_surface_functional and self._compute_hd is not None and self._compute_asd is not None:
            try:
                hd = self._compute_hd(
                    y_pred=y_pred_f, y=y_gt_f, spacing=spacing,
                    include_background=True, percentile=95, directed=False
                )
            except TypeError:
                try:
                    hd = self._compute_hd(y_pred=y_pred_f, y=y_gt_f, spacing=spacing, include_background=True)
                except TypeError:
                    hd = self._compute_hd(y_pred=y_pred_f, y=y_gt_f, spacing=spacing)

            try:
                asd = self._compute_asd(
                    y_pred=y_pred_f, y=y_gt_f, spacing=spacing,
                    include_background=True, symmetric=bool(self.asd_symmetric)
                )
            except TypeError:
                try:
                    asd = self._compute_asd(y_pred=y_pred_f, y=y_gt_f, spacing=spacing, include_background=True)
                except TypeError:
                    asd = self._compute_asd(y_pred=y_pred_f, y=y_gt_f, spacing=spacing)

            if hd.ndim == 1:
                hd = hd.view(1, -1).expand(B, -1)
            if asd.ndim == 1:
                asd = asd.view(1, -1).expand(B, -1)
            return hd, asd

        assert self._hd_metric_cls is not None and self._asd_metric_cls is not None
        hd_metric = self._hd_metric_cls(include_background=True, reduction="none", percentile=95, directed=False)
        try:
            asd_metric = self._asd_metric_cls(include_background=True, reduction="none", symmetric=bool(self.asd_symmetric))
        except TypeError:
            asd_metric = self._asd_metric_cls(include_background=True, reduction="none")

        hd_metric.reset()
        asd_metric.reset()

        try:
            hd_metric(y_pred=y_pred_f, y=y_gt_f, spacing=spacing)
        except TypeError:
            hd_metric(y_pred=y_pred_f, y=y_gt_f)

        try:
            asd_metric(y_pred=y_pred_f, y=y_gt_f, spacing=spacing)
        except TypeError:
            asd_metric(y_pred=y_pred_f, y=y_gt_f)

        hd = hd_metric.aggregate()
        asd = asd_metric.aggregate()

        if hd.ndim == 1:
            hd = hd.view(B, -1)
        if asd.ndim == 1:
            asd = asd.view(B, -1)
        return hd, asd

    def update(self, logits: torch.Tensor, batch: Dict[str, Any]) -> None:
        if "label" not in batch:
            raise KeyError("[TTASegEvaluator] batch must contain 'label'.")

        device = logits.device
        B = int(logits.size(0))
        R = int(logits.size(1))
        if R != self.R_expected:
            raise ValueError(f"[TTASegEvaluator] logits channels={R} but region_order={self.R_expected}")

        y_reg = batch["label"]
        if torch.is_tensor(y_reg):
            y_reg = y_reg.to(device)
        else:
            y_reg = torch.as_tensor(y_reg, device=device)

        if y_reg.ndim == 4:
            y_reg = y_reg.unsqueeze(0).expand(B, -1, -1, -1, -1)
        if y_reg.ndim != 5:
            raise ValueError(f"[TTASegEvaluator] label must be 5D, got {tuple(y_reg.shape)}")
        if int(y_reg.size(1)) != R:
            raise ValueError(f"[TTASegEvaluator] label channels={int(y_reg.size(1))} but logits channels={R}")

        y_reg_f = y_reg.float()

        prob = torch.sigmoid(logits)
        y_pred = (prob >= self.threshold).to(torch.uint8)
        y_gt_bin = (y_reg_f > 0.5).to(torch.uint8)

        dice_vals, iou_vals, valid, pred_empty = _binary_dice_iou(y_pred, y_gt_bin)
        sens_vals, valid_sens = _binary_sensitivity(y_pred, y_gt_bin)

        hd_batch = None
        asd_batch = None
        if self.enable_surface:
            D, H, W = int(y_pred.size(2)), int(y_pred.size(3)), int(y_pred.size(4))
            diag_mm = _diag_mm_from_shape(D, H, W, self.spacing)

            y_pred_f = y_pred.float()
            y_gt_f = y_gt_bin.float()
            hd_batch, asd_batch = self._surface_hd95_asd_batch(y_pred_f, y_gt_f, self.spacing)

            hd_batch = hd_batch.clone()
            asd_batch = asd_batch.clone()

            gt_empty = ~valid
            hd_batch[gt_empty] = 0.0
            asd_batch[gt_empty] = 0.0

            penalty_mask = valid & pred_empty
            hd_batch[penalty_mask] = float(diag_mm)
            asd_batch[penalty_mask] = float(diag_mm)

            bad = (~torch.isfinite(hd_batch) | ~torch.isfinite(asd_batch)) & valid
            if bad.any():
                case_id = batch.get("case_id", "unknown")
                domain = batch.get("domain", "unknown")
                bad_idx = bad.nonzero(as_tuple=False)  # [N,2] => (b, r)

                msg = (
                    f"[TTASegEvaluator] Found NaN/Inf in surface metrics (hd/asd) for valid regions. "
                    f"case_id={case_id}, domain={domain}, bad_idx={bad_idx[:10].tolist()} "
                    f"(showing up to 10)."
                )
                raise FloatingPointError(msg)

        dice_cpu = dice_vals.detach().to(torch.float32).cpu()
        iou_cpu = iou_vals.detach().to(torch.float32).cpu()
        sens_cpu = sens_vals.detach().to(torch.float32).cpu()
        valid_cpu = valid.detach().cpu()
        v_float = valid_cpu.to(torch.float64)

        self.sum_dice += (dice_cpu.to(torch.float64) * v_float).sum(dim=0)
        self.cnt_dice += v_float.sum(dim=0)

        self.sum_iou += (iou_cpu.to(torch.float64) * v_float).sum(dim=0)
        self.cnt_iou += v_float.sum(dim=0)

        valid_sens_cpu = valid_sens.detach().cpu()
        vs_float = valid_sens_cpu.to(torch.float64)
        self.sum_sens += (sens_cpu.to(torch.float64) * vs_float).sum(dim=0)
        self.cnt_sens += vs_float.sum(dim=0)

        hd_cpu = None
        asd_cpu = None
        if self.enable_surface and hd_batch is not None and asd_batch is not None:
            hd_cpu = hd_batch.detach().to(torch.float32).cpu()
            asd_cpu = asd_batch.detach().to(torch.float32).cpu()
            self.sum_hd95 += (hd_cpu.to(torch.float64) * v_float).sum(dim=0)
            self.cnt_hd95 += v_float.sum(dim=0)
            self.sum_asd += (asd_cpu.to(torch.float64) * v_float).sum(dim=0)
            self.cnt_asd += v_float.sum(dim=0)

        domains = _as_list_str(batch.get("domain", None), batch_size=B)
        for i in range(B):
            dom = domains[i]
            for c in range(self.R_expected):
                if bool(valid_cpu[i, c].item()):
                    self.dom_sum_dice[dom][c] += float(dice_cpu[i, c].item())
                    self.dom_cnt_dice[dom][c] += 1.0
                    self.dom_sum_iou[dom][c] += float(iou_cpu[i, c].item())
                    self.dom_cnt_iou[dom][c] += 1.0
                    self.dom_sum_sens[dom][c] += float(sens_cpu[i, c].item())
                    self.dom_cnt_sens[dom][c] += 1.0
                    if self.enable_surface and hd_cpu is not None and asd_cpu is not None:
                        self.dom_sum_hd95[dom][c] += float(hd_cpu[i, c].item())
                        self.dom_cnt_hd95[dom][c] += 1.0
                        self.dom_sum_asd[dom][c] += float(asd_cpu[i, c].item())
                        self.dom_cnt_asd[dom][c] += 1.0

        if self.report_loss:
            if hasattr(self.loss_fn, "dice") and getattr(self.loss_fn.dice, "class_weight", None) is not None:
                self.loss_fn.dice.class_weight = self.loss_fn.dice.class_weight.to(device)
            loss = self.loss_fn(logits, y_reg_f)
            self.total_loss += float(loss.item()) * B
            self.n_samples += B

    def finalize(self) -> Dict[str, float]:
        R = self.R_expected

        mean_dice = _finalize_mean_from_sum_cnt(self.sum_dice, self.cnt_dice)
        mean_iou = _finalize_mean_from_sum_cnt(self.sum_iou, self.cnt_iou)
        mean_sens = _finalize_mean_from_sum_cnt(self.sum_sens, self.cnt_sens)

        valid_regions_any = [i for i in range(R) if float(self.cnt_dice[i].item()) > 0]
        avg_dc = float(sum(mean_dice[i] for i in valid_regions_any) / max(1, len(valid_regions_any)))

        valid_regions_iou = [i for i in range(R) if float(self.cnt_iou[i].item()) > 0]
        miou = float(sum(mean_iou[i] for i in valid_regions_iou) / max(1, len(valid_regions_iou)))
        avg_iou = miou

        valid_regions_sens = [i for i in range(R) if float(self.cnt_sens[i].item()) > 0]
        avg_sens = float(sum(mean_sens[i] for i in valid_regions_sens) / max(1, len(valid_regions_sens)))

        metrics: Dict[str, float] = {}
        for name, v in zip(self.region_order, mean_dice):
            metrics[f"{name.lower()}_dc"] = v
        metrics["avg_dc"] = avg_dc

        for name, v in zip(self.region_order, mean_iou):
            metrics[f"{name.lower()}_iou"] = v
        metrics["avg_iou"] = avg_iou
        metrics["miou"] = miou
        metrics["jc"] = miou

        metrics["loss"] = float(self.total_loss / max(1, self.n_samples)) if self.report_loss else 0.0

        for name, v in zip(self.region_order, mean_sens):
            metrics[f"{name.lower()}_sens"] = v
        metrics["avg_sens"] = avg_sens

        if self.enable_surface:
            mean_hd95 = _finalize_mean_from_sum_cnt(self.sum_hd95, self.cnt_hd95)
            mean_asd = _finalize_mean_from_sum_cnt(self.sum_asd, self.cnt_asd)

            valid_regions_hd = [i for i in range(R) if float(self.cnt_hd95[i].item()) > 0]
            avg_hd95 = float(sum(mean_hd95[i] for i in valid_regions_hd) / max(1, len(valid_regions_hd)))

            valid_regions_asd = [i for i in range(R) if float(self.cnt_asd[i].item()) > 0]
            avg_asd = float(sum(mean_asd[i] for i in valid_regions_asd) / max(1, len(valid_regions_asd)))

            for name, v in zip(self.region_order, mean_hd95):
                metrics[f"{name.lower()}_hd95"] = v
            metrics["avg_hd95"] = avg_hd95

            for name, v in zip(self.region_order, mean_asd):
                metrics[f"{name.lower()}_asd"] = v
            metrics["avg_asd"] = avg_asd

        def _finalize_dom(sum_v: torch.Tensor, cnt_v: torch.Tensor) -> List[float]:
            out = []
            for c in range(R):
                out.append(float((sum_v[c] / cnt_v[c]).item()) if float(cnt_v[c].item()) > 0 else 0.0)
            return out

        for dom in sorted(self.dom_sum_dice.keys()):
            safe_dom = dom if dom != "" else "unknown"

            d_mean = _finalize_dom(self.dom_sum_dice[dom], self.dom_cnt_dice[dom])
            d_valid = [i for i in range(R) if float(self.dom_cnt_dice[dom][i].item()) > 0]
            d_avg = float(sum(d_mean[i] for i in d_valid) / max(1, len(d_valid)))

            di_mean = _finalize_dom(self.dom_sum_iou[dom], self.dom_cnt_iou[dom])
            di_valid = [i for i in range(R) if float(self.dom_cnt_iou[dom][i].item()) > 0]
            d_miou = float(sum(di_mean[i] for i in di_valid) / max(1, len(di_valid)))
            d_avg_iou = d_miou

            ds_mean = _finalize_dom(self.dom_sum_sens[dom], self.dom_cnt_sens[dom])
            ds_valid = [i for i in range(R) if float(self.dom_cnt_sens[dom][i].item()) > 0]
            d_avg_sens = float(sum(ds_mean[i] for i in ds_valid) / max(1, len(ds_valid)))

            for name, v in zip(self.region_order, d_mean):
                metrics[f"dom/{safe_dom}/{name.lower()}_dc"] = v
            metrics[f"dom/{safe_dom}/avg_dc"] = d_avg

            for name, v in zip(self.region_order, di_mean):
                metrics[f"dom/{safe_dom}/{name.lower()}_iou"] = v
            metrics[f"dom/{safe_dom}/avg_iou"] = d_avg_iou
            metrics[f"dom/{safe_dom}/miou"] = d_miou

            for name, v in zip(self.region_order, ds_mean):
                metrics[f"dom/{safe_dom}/{name.lower()}_sens"] = v
            metrics[f"dom/{safe_dom}/avg_sens"] = d_avg_sens

            if self.enable_surface:
                dh_mean = _finalize_dom(self.dom_sum_hd95[dom], self.dom_cnt_hd95[dom])
                dh_valid = [i for i in range(R) if float(self.dom_cnt_hd95[dom][i].item()) > 0]
                d_avg_hd95 = float(sum(dh_mean[i] for i in dh_valid) / max(1, len(dh_valid)))

                da_mean = _finalize_dom(self.dom_sum_asd[dom], self.dom_cnt_asd[dom])
                da_valid = [i for i in range(R) if float(self.dom_cnt_asd[dom][i].item()) > 0]
                d_avg_asd = float(sum(da_mean[i] for i in da_valid) / max(1, len(da_valid)))

                for name, v in zip(self.region_order, dh_mean):
                    metrics[f"dom/{safe_dom}/{name.lower()}_hd95"] = v
                metrics[f"dom/{safe_dom}/avg_hd95"] = d_avg_hd95

                for name, v in zip(self.region_order, da_mean):
                    metrics[f"dom/{safe_dom}/{name.lower()}_asd"] = v
                metrics[f"dom/{safe_dom}/avg_asd"] = d_avg_asd

        return metrics