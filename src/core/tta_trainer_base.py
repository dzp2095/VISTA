# file: src/core/trainers/tta_trainer_base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

import os
import re
import gc
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from .trainer_base import HookBase
from ..utils.logger import get_logger
from ..utils.config import get_config, require_config

from ..evaluation.tta_seg_evaluator import TTASegEvaluator


class TTATrainerBase(ABC):
    """TTA trainer base: per-volume adapt -> predict -> evaluator.update(); optional mask save and initial eval."""

    def __init__(self, config: DictConfig, device: torch.device, evaluation_strategy=None):
        self.config = config
        self.device = device
        self.logger = get_logger()

        # components
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        self.criterion = None
        self.evaluation_strategy = evaluation_strategy

        # hooks
        self._hooks: List[HookBase] = []

        # state
        self.epoch: int = 0
        self.iter: int = 0

        # protocol cfg
        tta_cfg = get_config(config, "tta", DictConfig({}))

        self.iters: int = int(require_config(tta_cfg, "steps_per_batch"))
        self.enforce_bs1: bool = bool(get_config(tta_cfg, "enforce_bs1", True))

        online_cfg = get_config(tta_cfg, "online", DictConfig({}))
        reset_per_case_from_online = bool(get_config(online_cfg, "reset_per_case", False))
        episodic = bool(get_config(tta_cfg, "episodic", False))
        self.reset_per_case: bool = bool(reset_per_case_from_online or episodic)

        # evaluator (per-volume)
        self.evaluator = TTASegEvaluator(config)

        # ----------------------------
        # save mask / initial eval
        # ----------------------------
        save_cfg = get_config(tta_cfg, "save", DictConfig({}))
        self.save_enabled: bool = bool(get_config(save_cfg, "enabled", False))
        self.save_dir: str = str(get_config(save_cfg, "dir", "")).strip()
        self.save_format: str = str(get_config(save_cfg, "format", "nii.gz")).strip().lower()
        self.save_pred_type: str = str(get_config(save_cfg, "pred_type", "auto")).strip().lower()
        self.save_threshold: float = float(get_config(save_cfg, "threshold", 0.5))
        self.save_only_rank0: bool = bool(get_config(save_cfg, "only_rank0", True))

        _cn = get_config(save_cfg, "class_names", [])
        if _cn is None:
            self.save_class_names: List[str] = []
        elif isinstance(_cn, (list, tuple)):
            self.save_class_names = [str(x) for x in _cn]
        else:
            self.save_class_names = [str(_cn)]

        self.save_initial: bool = bool(get_config(save_cfg, "save_initial", False))
        self.save_initial_once: bool = bool(get_config(save_cfg, "initial_once", True))
        self.save_initial_subdir: str = str(get_config(save_cfg, "initial_subdir", "init")).strip()
        self.save_tta_subdir: str = str(get_config(save_cfg, "tta_subdir", "tta")).strip()

        self._initial_eval_done: bool = False

    def setup(
        self,
        model: nn.Module,
        criterion=None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        evaluation_strategy=None,
        **kwargs,
    ):
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        if evaluation_strategy is not None:
            self.evaluation_strategy = evaluation_strategy

        self.logger.info(
            f"[TTATrainerBase] setup done | iters={self.iters} enforce_bs1={self.enforce_bs1} reset_per_case={self.reset_per_case} "
            f"| save_enabled={self.save_enabled} save_dir='{self.save_dir}' save_format={self.save_format} save_initial={self.save_initial}"
        )

    def register_hooks(self, hooks: List[HookBase]):
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            h.trainer = self
        self._hooks.extend(hooks)
        self.logger.info(f"[TTATrainerBase] Registered {len(hooks)} hooks")

    def train(
        self,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        test_loader: DataLoader = None,
        eval_on_train: bool = False,
    ) -> Dict[str, List]:
        """Train interface: epochs=1 typical; uses test_loader or val_loader or train_loader."""
        if self.model is None:
            raise RuntimeError("[TTATrainerBase] call setup() first")

        if int(epochs) != 1:
            self.logger.warning(f"[TTATrainerBase] epochs={epochs}. Online continual TTA usually uses epochs=1.")

        data_loader = test_loader or val_loader or train_loader
        if data_loader is None:
            raise ValueError("[TTATrainerBase] no dataloader provided")

        tta_history: List[Dict[str, float]] = []

        for h in self._hooks:
            h.before_train()

        try:
            for ep in range(int(epochs)):
                self.epoch = ep
                for h in self._hooks:
                    h.before_train_epoch()

                stats = self.tta_epoch(data_loader)
                tta_history.append(stats)

                for h in self._hooks:
                    h.after_train_epoch()

                for h in self._hooks:
                    if hasattr(h, "on_epoch_end"):
                        h.on_epoch_end(ep, stats, {}, False)

        finally:
            for h in self._hooks:
                h.after_train()

        return {"tta_history": tta_history}

    # ----------------------------
    # main epoch
    # ----------------------------
    def tta_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        assert self.model is not None

        self.model.eval()
        self.model.to(self.device)

        if self._should_save_masks():
            os.makedirs(os.path.join(self.save_dir, self.save_tta_subdir), exist_ok=True)
            if self.save_initial:
                os.makedirs(os.path.join(self.save_dir, self.save_initial_subdir), exist_ok=True)

        init_metrics: Optional[Dict[str, float]] = None
        if self._should_run_initial_eval():
            init_metrics = self._run_initial_eval_and_save(data_loader)
            self._initial_eval_done = True

        self.evaluator.reset()

        prev_case: Optional[str] = None

        show_pbar = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
        pbar = tqdm(data_loader, desc=f"TTA Epoch {self.epoch}", leave=False) if show_pbar else data_loader

        for batch in pbar:
            x = batch["image"]
            B = int(x.size(0)) if torch.is_tensor(x) else 1

            if self.enforce_bs1 and B != 1:
                raise ValueError(
                    f"[TTATrainerBase] enforce_bs1=True but got batch_size={B}. "
                    f"Per-volume online TTA expects bs=1."
                )

            if self.reset_per_case:
                cid = self._extract_case_id(batch)
                if cid is not None:
                    if prev_case is None or cid != prev_case:
                        self.reset_adaptation_state()
                        prev_case = cid

            # adapt K steps
            self.adapt(batch, steps=self.iters)

            # predict after adapt
            with torch.no_grad():
                logits = self.predict(batch)

            if self._should_save_masks():
                pred_mask = self._logits_to_mask(logits)
                self._save_mask_for_batch(pred_mask, batch, tag=self.save_tta_subdir)

            # update metrics
            self.evaluator.update(logits, batch)

            if show_pbar:
                pbar.set_postfix(self.evaluator.brief())

            self.iter += 1

        metrics = self.evaluator.finalize()
        self.logger.info(f"[TTA] Epoch {self.epoch} evaluation results: {metrics}")

        if init_metrics is not None:
            for k, v in init_metrics.items():
                metrics[f"init_{k}"] = float(v)

        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

        return metrics

    # ----------------------------
    # abstract
    # ----------------------------
    @abstractmethod
    def adapt(self, batch: Dict[str, Any], steps: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, batch: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def reset_adaptation_state(self) -> None:
        """Override to reset model/prompt/optimizer (episodic / reset_per_case)."""
        return

    # ============================================================
    # mask saving + initial eval (utility)
    # ============================================================
    def _is_rank0(self) -> bool:
        if not torch.distributed.is_initialized():
            return True
        return int(torch.distributed.get_rank()) == 0

    def _should_save_masks(self) -> bool:
        if not self.save_enabled:
            return False
        if self.save_dir == "":
            return False
        if self.save_only_rank0 and (not self._is_rank0()):
            return False
        return True

    def _should_run_initial_eval(self) -> bool:
        if not self.save_initial:
            return False
        if not self._should_save_masks():
            return False
        if self.save_initial_once and self._initial_eval_done:
            return False
        return True

    def _extract_case_id(self, batch: Dict[str, Any]) -> Optional[str]:
        for key in ["case_id", "subject_id", "id", "name", "uid"]:
            if key in batch:
                v = batch.get(key, None)
                if v is None:
                    continue
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    return self._sanitize_case_id(str(v[0]))
                if torch.is_tensor(v):
                    if v.numel() > 0:
                        try:
                            return self._sanitize_case_id(str(int(v.view(-1)[0].item())))
                        except Exception:
                            return self._sanitize_case_id(str(v.view(-1)[0].item()))
                return self._sanitize_case_id(str(v))
        return None

    def _sanitize_case_id(self, s: str) -> str:
        s = s.strip()
        s = re.sub(r"[\\/]+", "_", s)
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^0-9a-zA-Z._-]+", "_", s)
        if s == "":
            s = "case"
        return s

    def _get_affine_from_batch(self, batch: Dict[str, Any], x: Any) -> np.ndarray:
        # 1) MetaTensor
        try:
            if hasattr(x, "meta") and isinstance(x.meta, dict):
                aff = x.meta.get("affine", None)
                if aff is None:
                    aff = x.meta.get("original_affine", None)
                if aff is not None:
                    return self._to_affine_np(aff)
        except Exception:
            pass

        # 2) image_meta_dict / meta_dict
        for mk in ["image_meta_dict", "meta_dict"]:
            md = batch.get(mk, None)
            if isinstance(md, dict):
                aff = md.get("affine", None)
                if aff is None:
                    aff = md.get("original_affine", None)
                if aff is not None:
                    return self._to_affine_np(aff)

        return np.eye(4, dtype=np.float64)

    def _to_affine_np(self, aff: Any) -> np.ndarray:
        if torch.is_tensor(aff):
            aff = aff.detach().cpu().numpy()
        else:
            aff = np.asarray(aff)

        if aff.ndim == 3:
            aff = aff[0]
        aff = aff.astype(np.float64, copy=False)
        if aff.shape != (4, 4):
            return np.eye(4, dtype=np.float64)
        return aff

    def _infer_pred_type(self, logits: torch.Tensor) -> str:
        if self.save_pred_type in ["multilabel", "multiclass"]:
            return self.save_pred_type

        candidates: List[Optional[str]] = []
        candidates.append(get_config(self.config, "tta.eval.act", None))
        candidates.append(get_config(self.config, "tta.loss.act", None))
        candidates.append(get_config(self.config, "loss.act", None))
        candidates.append(get_config(self.config, "model.act", None))

        for a in candidates:
            if a is None:
                continue
            a = str(a).lower()
            if "sigmoid" in a:
                return "multilabel"
            if "softmax" in a:
                return "multiclass"

        C = int(logits.shape[1]) if logits.ndim >= 2 else 1
        if C == 1:
            return "multilabel"
        return "multilabel"

    def _logits_to_mask(self, logits: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(logits):
            raise TypeError(f"logits must be torch.Tensor, got {type(logits)}")
        if logits.ndim < 4:
            raise ValueError(f"Expect logits ndim>=4 (B,C,spatial...), got shape={tuple(logits.shape)}")

        pred_type = self._infer_pred_type(logits)

        if pred_type == "multiclass":
            pred = torch.argmax(logits, dim=1)  # (B, spatial...)
            return pred.to(torch.uint8)

        # multilabel
        probs = torch.sigmoid(logits)
        mask = (probs > float(self.save_threshold)).to(torch.uint8)  # (B,C,spatial...)
        return mask

    def _save_mask_for_batch(self, mask: torch.Tensor, batch: Dict[str, Any], tag: str):
        if not self._should_save_masks():
            return

        x = batch["image"]
        cid = self._extract_case_id(batch)
        if cid is None:
            cid = f"iter{self.iter:06d}"

        out_dir = os.path.join(self.save_dir, tag)
        os.makedirs(out_dir, exist_ok=True)

        affine = self._get_affine_from_batch(batch, x)

        mask = mask.detach().cpu()

        # multiclass: (B, D,H,W)
        if mask.ndim == 4:
            vol = mask[0].numpy().astype(np.uint8, copy=False)
            out_path = os.path.join(out_dir, f"{cid}.{self.save_format}")
            self._save_volume(vol, affine, out_path)
            return

        # multilabel: (B,C,D,H,W)
        if mask.ndim == 5:
            C = int(mask.shape[1])
            for c in range(C):
                vol = mask[0, c].numpy().astype(np.uint8, copy=False)
                cname = self.save_class_names[c] if c < len(self.save_class_names) else f"ch{c}"
                out_path = os.path.join(out_dir, f"{cid}_{cname}.{self.save_format}")
                self._save_volume(vol, affine, out_path)
            return

        raise ValueError(f"Unexpected mask shape={tuple(mask.shape)}")

    def _save_volume(self, vol: np.ndarray, affine: np.ndarray, out_path: str):
        fmt = self.save_format
        if fmt in ["pt", "pth"]:
            payload = {"vol": vol.astype(np.uint8, copy=False), "affine": affine.astype(np.float64, copy=False)}
            torch.save(payload, out_path if out_path.endswith(".pt") else (out_path + ".pt"))
            return

        try:
            import nibabel as nib  # type: ignore
        except Exception as e:
            self.logger.warning(f"[TTATrainerBase] nibabel not available ({e}). Fallback to torch.save(.pt).")
            payload = {"vol": vol.astype(np.uint8, copy=False), "affine": affine.astype(np.float64, copy=False)}
            alt = out_path
            if alt.endswith(".nii.gz"):
                alt = alt[:-7] + ".pt"
            elif alt.endswith(".nii"):
                alt = alt[:-4] + ".pt"
            else:
                alt = alt + ".pt"
            torch.save(payload, alt)
            return

        if not (out_path.endswith(".nii") or out_path.endswith(".nii.gz")):
            out_path = out_path + ".nii.gz"

        img = nib.Nifti1Image(vol.astype(np.uint8, copy=False), affine)
        nib.save(img, out_path)

    def _run_initial_eval_and_save(self, data_loader: DataLoader) -> Dict[str, float]:
        assert self.model is not None

        self.logger.info("[InitEval] Running initial (no-adapt) prediction + mask saving + metrics (ONE-TIME).")

        init_eval = TTASegEvaluator(self.config)
        init_eval.reset()

        show_pbar = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
        pbar = tqdm(data_loader, desc=f"InitEval Epoch {self.epoch}", leave=False) if show_pbar else data_loader

        with torch.no_grad():
            for batch in pbar:
                x = batch["image"]
                B = int(x.size(0)) if torch.is_tensor(x) else 1
                if self.enforce_bs1 and B != 1:
                    raise ValueError(
                        f"[TTATrainerBase] enforce_bs1=True but got batch_size={B}. "
                        f"Per-volume online TTA expects bs=1."
                    )

                logits = self.predict(batch)

                pred_mask = self._logits_to_mask(logits)
                self._save_mask_for_batch(pred_mask, batch, tag=self.save_initial_subdir)

                init_eval.update(logits, batch)

                if show_pbar:
                    pbar.set_postfix(init_eval.brief())

        init_metrics = init_eval.finalize()
        self.logger.info(f"[InitEval] Initial (no-adapt) evaluation results: {init_metrics}")
        return init_metrics