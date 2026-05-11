from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from ..tta_trainer_base import TTATrainerBase
from ...utils.config import get_config
from ...registry import register_trainer

from ..tta.ckpt_utils import extract_state_dict, align_module_prefix, unwrap_model
from ..tta.optim_utils import (
    collect_bn_affine_named_params,
    collect_norm_affine_named_params,
    collect_named_params_by_prefix,
    set_requires_grad_by_id,
    prune_optimizer_to_trainable,
)
from ..tta.norm_utils import disable_dropout, set_eval_and_bn_mode, bn_train, bn_eval
from ..tta.hooks import FeatureHook, get_submodule_by_name
from ..tta.ema import build_teacher_from_student, ema_update
from ..tta.losses import soft_targets_bce_with_logits
from ..tta.pseudo_label import build_thr_tensor, pseudo_label_loss_sigmoid, apply_hierarchy_fix
from ..tta.views import sample_view_specs, make_view_tensor
from ..tta.interventions import sigmoid_entropy_map


@register_trainer("ours")
class OursTrainer(TTATrainerBase):
    """Full-anchor + fixed interventions + EMA teacher-student."""

    def __init__(self, config: DictConfig, device: torch.device, evaluation_strategy=None):
        super().__init__(config, device, evaluation_strategy=evaluation_strategy)

        crit_cfg = get_config(config, "training.criterion", DictConfig({}))
        softmax = bool(get_config(crit_cfg, "softmax", False))
        self.sigmoid = bool(get_config(crit_cfg, "sigmoid", (not softmax)))
        if softmax or (not self.sigmoid):
            raise ValueError("[OursTrainer] This trainer expects sigmoid outputs (softmax is not supported).")

        # base tta cfg
        tta_cfg = get_config(config, "tta", DictConfig({}))

        self.adapt_params = str(get_config(tta_cfg, "adapt_params", "bn_affine")).lower()
        # Optional extra trainable submodules (string prefix or list[str]).
        # Example: tta.adapt_from="model.decoder" or ["outc", "decoder.up4"].
        self.adapt_from_cfg = get_config(tta_cfg, "adapt_from", None)
        self.force_train_bn = bool(get_config(tta_cfg, "force_train_bn", True))
        self.disable_dropout_flag = bool(get_config(tta_cfg, "disable_dropout", True))
        self.grad_clip_norm = get_config(tta_cfg, "grad_clip_norm", None)
        self.clear_optim_state = bool(get_config(tta_cfg, "clear_optim_state", True))
        self.ckpt_path = str(get_config(config, "tta.ckpt_path", "")).strip()

        # ours cfg
        ours_cfg = get_config(tta_cfg, "ours", DictConfig({}))

        # -------------------------
        # Loss mode (choose ONE):
        #   - pl_cons: loss_pl + loss_cons
        #   - pl_feat: loss_pl + loss_feat
        # -------------------------
        loss_cfg = get_config(ours_cfg, "loss", DictConfig({}))
        loss_mode = str(get_config(loss_cfg, "mode", get_config(ours_cfg, "loss_mode", "pl_cons"))).lower()
        if loss_mode in ("pl_cons", "pl+cons", "cons", "consistency"):
            self.loss_mode = "pl_cons"
        elif loss_mode in ("pl_feat", "pl+feat", "feat", "feature", "feature_align"):
            self.loss_mode = "pl_feat"
        else:
            raise ValueError(f"[OursTrainer] Unknown loss mode: {loss_mode}. Use 'pl_cons' or 'pl_feat'.")

        # predict policy
        self.predict_with_teacher = bool(get_config(ours_cfg, "predict_with_teacher", True))

        # -------------------------
        # Views: ONLY fixed interventions
        # -------------------------
        # Preferred:
        #   tta.ours.interventions
        # Backward compatible:
        #   tta.ours.views.interventions
        raw_intv = get_config(ours_cfg, "interventions", None)
        if raw_intv is None:
            views_cfg = get_config(ours_cfg, "views", DictConfig({}))
            raw_intv = get_config(views_cfg, "interventions", [])
        self.interventions: List[Dict[str, Any]] = self._parse_interventions(raw_intv)

        # BN update knobs (K1)
        bn_cfg = get_config(ours_cfg, "bn", DictConfig({}))
        self.bn_update_stats = bool(get_config(bn_cfg, "update_stats", True))
        self.bn_update_affine = bool(get_config(bn_cfg, "update_affine", True))

        # teacher (EMA)
        teacher_cfg = get_config(ours_cfg, "teacher", DictConfig({}))
        self.ema_decay = float(get_config(teacher_cfg, "ema_decay", 0.99))
        self.teacher_bn_buffers = str(get_config(teacher_cfg, "bn_buffers", "copy")).lower()

        # uncertainty map (for patch_swap_uncertain)
        unc_cfg = get_config(ours_cfg, "uncertainty", DictConfig({}))
        self.unc_reduce = str(get_config(unc_cfg, "reduce", "mean")).lower()

        # pseudo label
        pl_cfg = get_config(ours_cfg, "pseudo_label", DictConfig({}))
        self.pl_use_neg = bool(get_config(pl_cfg, "use_neg", True))
        self.pl_neg_weight = float(get_config(pl_cfg, "neg_weight", 0.1))
        self.pl_hierarchy_fix = bool(get_config(pl_cfg, "hierarchy_fix", True))
        self.pl_min_voxels = int(get_config(pl_cfg, "min_voxels", 0))

        self.pl_pos_thr_cfg = get_config(pl_cfg, "pos_thr", 0.9)
        self.pl_neg_thr_cfg = get_config(pl_cfg, "neg_thr", 0.05)
        self.pl_var_thr_cfg = get_config(pl_cfg, "var_thr", 1e9)

        # variance-based reliability for pseudo label
        self.pl_var_gate = bool(get_config(pl_cfg, "var_gate", True))
        var_w_cfg = get_config(pl_cfg, "var_weight", DictConfig({}))
        self.pl_var_weight_enable = bool(get_config(var_w_cfg, "enable", False))
        self.pl_var_weight_type = str(get_config(var_w_cfg, "type", "exp")).lower()
        self.pl_var_weight_beta = float(get_config(var_w_cfg, "beta", 1.0))
        self.pl_var_weight_min = float(get_config(var_w_cfg, "min_w", 0.0))
        self.pl_var_weight_max = float(get_config(var_w_cfg, "max_w", 1.0))
        self.pl_var_weight_scale_cfg = get_config(var_w_cfg, "scale", None)

        # feature alignment (only used for pl_feat)
        fa_cfg = get_config(ours_cfg, "feat_align", DictConfig({}))
        self.feat_hook_name = str(get_config(fa_cfg, "hook_name", "")).strip()
        self.feat_type = str(get_config(fa_cfg, "feat_type", "gap")).lower()

        # loss weights (only one aux is used depending on loss_mode)
        self.w_pl = float(get_config(loss_cfg, "w_pl", 1.0))
        self.w_cons = float(get_config(loss_cfg, "w_cons", 1.0))
        self.w_feat = float(get_config(loss_cfg, "w_feat", 0.1))

        # internal
        self.teacher: Optional[nn.Module] = None
        self._source_state_dict: Optional[Dict[str, torch.Tensor]] = None

        self._has_trainable_params: bool = True

        self._hook_teacher: Optional[FeatureHook] = None
        self._hook_student: Optional[FeatureHook] = None

        self._pos_thr: Optional[torch.Tensor] = None
        self._neg_thr: Optional[torch.Tensor] = None
        self._var_thr: Optional[torch.Tensor] = None
        self._var_w_scale: Optional[torch.Tensor] = None

    def _parse_interventions(self, raw: Any) -> List[Dict[str, Any]]:
        """Parse fixed intervention list.

        Expected config examples:
          interventions:
            - {type: amp_swap_lowfreq, lowfreq_ratio: 0.1}
            - {type: patch_swap_uncertain, pair_mode: random_per_step, candidate_pairs: [[1,3],[1,2]]}

        For convenience we also accept list[str] (e.g. ["amp_swap_lowfreq"]).
        """

        def _to_py(v: Any) -> Any:
            if v is None:
                return None
            try:
                return OmegaConf.to_container(v, resolve=True)
            except Exception:
                return v

        raw = _to_py(raw)
        if raw is None:
            items: List[Any] = []
        elif isinstance(raw, (str, dict)):
            items = [raw]
        else:
            try:
                items = list(raw)
            except Exception:
                items = []

        out: List[Dict[str, Any]] = []
        for it in items:
            if isinstance(it, str):
                d: Dict[str, Any] = {"type": it}
            else:
                d = _to_py(it)
                if not isinstance(d, dict):
                    raise ValueError(f"[OursTrainer] Unsupported intervention element: {type(it)}")
                d = dict(d)
                if "type" not in d and "name" in d:
                    d["type"] = d.pop("name")
                if "type" not in d:
                    raise ValueError(f"[OursTrainer] intervention dict must include 'type': {d}")

            t = str(d.get("type", "")).lower()
            if t in ("amp_swap_lowfreq", "amplitude_swap_lowfreq", "amp_swap"):
                d["type"] = "amp_swap_lowfreq"
                d["lowfreq_ratio"] = float(d.get("lowfreq_ratio", 0.1))
                d["donor_map"] = d.get("donor_map", "circular")
                # Paper-style LFCCS uses a randomly sampled modality pair per step.
                # Keep this configurable for ablations.
                d["pair_mode"] = str(d.get("pair_mode", "random_per_step")).lower()
                if "candidate_pairs" not in d and "swap_pairs" in d:
                    d["candidate_pairs"] = d["swap_pairs"]
                out.append(d)
            elif t in ("patch_swap_uncertain", "patch_swap"):
                d["type"] = "patch_swap_uncertain"
                d["unc_thr"] = d.get("unc_thr", None)
                d["unc_quantile"] = d.get("unc_quantile", 0.95)
                d["dilate_ks"] = int(d.get("dilate_ks", 15))
                d["mode"] = str(d.get("mode", "swap")).lower()
                # Paper-style UGPS uses a randomly sampled modality pair per step.
                d["pair_mode"] = str(d.get("pair_mode", "random_per_step")).lower()

                # backwards compat: allow swap_pairs as alias of candidate_pairs
                if "candidate_pairs" not in d and "swap_pairs" in d:
                    d["candidate_pairs"] = d["swap_pairs"]
                out.append(d)
            else:
                raise ValueError(f"[OursTrainer] Unknown intervention type: {d.get('type')}")

        return out

    def setup(
        self,
        model: nn.Module,
        criterion=None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        evaluation_strategy=None,
        **kwargs,
    ):
        super().setup(model, criterion, optimizer, scheduler, evaluation_strategy)

        if self.optimizer is None:
            raise ValueError("[OursTrainer] optimizer is None. Ensure ExperimentManager builds optimizer before setup_trainer().")

        # ---- load init ckpt (optional) ----
        if self.ckpt_path:
            self.logger.info(f"[OursTrainer] Loading TTA init checkpoint from: {self.ckpt_path}")
            obj = torch.load(self.ckpt_path, map_location="cpu")
            state = extract_state_dict(obj)

            base = unwrap_model(self.model)  # type: ignore[arg-type]
            aligned = align_module_prefix(state, base.state_dict().keys())
            base.load_state_dict(aligned, strict=True)

        # ---- snapshot source for episodic reset ----
        base = unwrap_model(self.model)  # type: ignore[arg-type]
        self._source_state_dict = {k: v.detach().clone() for k, v in base.state_dict().items()}

        # ---- build teacher as deepcopy(student base) ----
        self.teacher = build_teacher_from_student(base, self.device)

        # ---- select params to adapt (student) ----
        named_params: List[Tuple[str, nn.Parameter]] = []
        if self.adapt_params == "bn_affine":
            if self.bn_update_affine:
                named_params = collect_bn_affine_named_params(self.model)  # type: ignore[arg-type]
            else:
                named_params = []
        elif self.adapt_params == "norm_affine":
            named_params = collect_norm_affine_named_params(self.model)  # type: ignore[arg-type]
        elif self.adapt_params == "all":
            named_params = [(n, p) for n, p in self.model.named_parameters()]  # type: ignore[union-attr]
        else:
            raise ValueError(f"[OursTrainer] Unknown adapt_params: {self.adapt_params}")

        # Optional extra trainable submodules by name prefix.
        # This is a clean way to try "update only last few layers" without introducing many modes.
        # Example:
        #   tta.adapt_params=bn_affine
        #   tta.adapt_from=["outc", "decoder.up4"]
        prefixes: List[str] = []
        if self.adapt_from_cfg is not None:
            try:
                af = OmegaConf.to_container(self.adapt_from_cfg, resolve=True)
            except Exception:
                af = self.adapt_from_cfg

            if isinstance(af, str):
                s = af.strip()
                if s and s.lower() not in ("none", "null"):
                    prefixes = [s]
            elif isinstance(af, (list, tuple)):
                prefixes = [
                    str(x).strip().rstrip(".")
                    for x in af
                    if str(x).strip() and str(x).strip().lower() not in ("none", "null")
                ]

        if len(prefixes) > 0:
            extra = collect_named_params_by_prefix(self.model, prefixes)  # type: ignore[arg-type]
            if len(extra) > 0:
                merged = {id(p): (n, p) for n, p in named_params}
                for n, p in extra:
                    merged[id(p)] = (n, p)
                named_params = list(merged.values())

        trainable_ids = set(id(p) for _, p in named_params)
        self._has_trainable_params = (len(trainable_ids) > 0)

        set_requires_grad_by_id(self.model, trainable_ids)  # type: ignore[arg-type]

        if self.disable_dropout_flag:
            disable_dropout(self.model)  # type: ignore[arg-type]

        # initial BN mode
        bn_train_flag = bool(self.force_train_bn and self.bn_update_stats)
        set_eval_and_bn_mode(self.model, bn_train=bn_train_flag)  # type: ignore[arg-type]

        # prune optimizer if we do have trainable params
        if self._has_trainable_params:
            kept = prune_optimizer_to_trainable(self.optimizer, trainable_ids, clear_state=self.clear_optim_state)
            if kept == 0:
                raise ValueError("[OursTrainer] optimizer has 0 parameters after pruning.")
        else:
            kept = 0
            self.logger.warning("[OursTrainer] No trainable params (stats-only adaptation). Optimizer step will be skipped.")

        # ---- prepare thresholds tensors ----
        C = int(get_config(self.config, "model.num_classes", 3))
        self._pos_thr = build_thr_tensor(self.pl_pos_thr_cfg, C, self.device, default=0.9)
        self._neg_thr = build_thr_tensor(self.pl_neg_thr_cfg, C, self.device, default=0.05)
        self._var_thr = build_thr_tensor(self.pl_var_thr_cfg, C, self.device, default=1e9)

        if self.pl_var_weight_enable:
            # scale can be explicitly provided; otherwise fallback to var_thr_cfg for convenience
            scale_cfg = self.pl_var_weight_scale_cfg if (self.pl_var_weight_scale_cfg is not None) else self.pl_var_thr_cfg
            self._var_w_scale = build_thr_tensor(scale_cfg, C, self.device, default=1.0)

        # ---- feature hooks (only needed for pl_feat) ----
        if self.loss_mode == "pl_feat":
            if not self.feat_hook_name:
                raise ValueError("[OursTrainer] loss_mode=pl_feat requires tta.ours.feat_align.hook_name")
            try:
                student_mod = get_submodule_by_name(base, self.feat_hook_name)
                teacher_mod = get_submodule_by_name(self.teacher, self.feat_hook_name)  # type: ignore[arg-type]
            except Exception as e:
                raise ValueError(
                    f"[OursTrainer] Cannot resolve hook_name='{self.feat_hook_name}'. "
                    f"Please copy the exact path from model.named_modules(). Error: {e}"
                )

            self._hook_student = FeatureHook(student_mod, feat_type=self.feat_type, detach=False)
            self._hook_teacher = FeatureHook(teacher_mod, feat_type=self.feat_type, detach=True)

        self.logger.info(
            f"[OursTrainer] setup done | adapt_params={self.adapt_params} adapt_from={prefixes if len(prefixes)>0 else '-'} kept={kept} "
            f"iters(per-volume)={self.iters} reset_per_case={self.reset_per_case} "
            f"loss_mode={self.loss_mode} interventions={len(self.interventions)} "
            f"teacher_bn_buffers={self.teacher_bn_buffers} bn_update_stats={self.bn_update_stats} "
            f"feat_type={self.feat_type if self.loss_mode=='pl_feat' else '-'}"
        )

    def reset_adaptation_state(self) -> None:
        """Reset student/teacher to source weights (episodic / reset_per_case)."""
        if self._source_state_dict is None or self.model is None:
            return

        base = unwrap_model(self.model)
        base.load_state_dict(self._source_state_dict, strict=True)

        bn_train_flag = bool(self.force_train_bn and self.bn_update_stats)
        set_eval_and_bn_mode(self.model, bn_train=bn_train_flag)

        if self.teacher is not None:
            self.teacher.load_state_dict(self._source_state_dict, strict=True)
            self.teacher.eval()

    @torch.enable_grad()
    def adapt(self, batch: Dict[str, Any], steps: int) -> None:
        if self.optimizer is None or self.model is None:
            raise RuntimeError("[OursTrainer] setup() not called properly.")
        if self.teacher is None:
            raise RuntimeError("[OursTrainer] teacher is None. setup() must create teacher.")

        x = batch["image"].to(self.device, non_blocking=True)

        # BN control:
        #   - clean full forward: BN train iff bn_update_stats
        #   - intervention forwards: BN eval always (avoid running-stat pollution)
        bn_train_clean = bool(self.force_train_bn and self.bn_update_stats)

        for _ in range(int(steps)):
            # -----------------------------
            # 1) Teacher full anchor
            # -----------------------------
            with torch.no_grad():
                self.teacher.eval()

                logits_t_full = self.teacher(x)

                p_ref = torch.sigmoid(logits_t_full).to(dtype=torch.float32)

                if self.pl_hierarchy_fix:
                    p_ref = apply_hierarchy_fix(p_ref)

                uncert_map = None
                if any(str(it.get("type", "")).lower().startswith("patch_swap") for it in self.interventions):
                    # multi-label uncertainty map from teacher(full)
                    uncert_map = sigmoid_entropy_map(p_ref, reduce=self.unc_reduce)

                anchor_feat = None
                if self.loss_mode == "pl_feat" and self._hook_teacher is not None:
                    anchor_feat = self._hook_teacher.value  # already detached

            # -----------------------------
            # 2) Build view specs
            # -----------------------------
            view_specs = sample_view_specs(
                x,
                interventions=self.interventions,
            )

            # -----------------------------
            # 3) Teacher disagreement stats (p_var) for PL gating
            # -----------------------------
            with torch.no_grad():
                sum_p = p_ref.clone()
                sum_p2 = (p_ref * p_ref).clone()
                V = 1

                # skip full (already computed)
                for spec in view_specs[1:]:
                    xv = make_view_tensor(x, spec, uncert_map=uncert_map)
                    logits_t = self.teacher(xv)

                    p = torch.sigmoid(logits_t)

                    p = p.to(dtype=torch.float32)
                    sum_p.add_(p)
                    sum_p2.add_(p * p)
                    V += 1

                p_mean = sum_p / float(V)
                p_var = (sum_p2 / float(V) - p_mean * p_mean).clamp_min(0.0)

            # -----------------------------
            # Stats-only adaptation (no trainable params)
            #   Only update BN running stats using CLEAN full view to avoid pollution.
            # -----------------------------
            if not self._has_trainable_params:
                if bn_train_clean:
                    with torch.no_grad():
                        with bn_train(self.model):
                            _ = self.model(x)

                    if self.teacher_bn_buffers in ("copy", "ema"):
                        ema_update(
                            self.teacher,
                            unwrap_model(self.model),
                            decay=self.ema_decay,
                            bn_buffers=self.teacher_bn_buffers,
                        )
                continue

            # -----------------------------
            # 4) Student: losses
            # -----------------------------
            self.optimizer.zero_grad(set_to_none=True)

            loss_aux = torch.zeros((), device=self.device)

            logits_full_s: Optional[torch.Tensor] = None

            # 4.1 full clean forward (for PL)
            with (bn_train(self.model) if bn_train_clean else bn_eval(self.model)):
                logits_full_s = self.model(x)

            # 4.2 intervention forwards (BN eval always) -> aux loss
            n_aux = 0
            for spec in view_specs[1:]:
                xv = make_view_tensor(x, spec, uncert_map=uncert_map)
                with bn_eval(self.model):
                    logits_s = self.model(xv)

                if self.loss_mode == "pl_cons":
                    loss_aux = loss_aux + soft_targets_bce_with_logits(logits_s, p_ref.detach())
                    n_aux += 1
                else:  # pl_feat
                    if anchor_feat is None or self._hook_student is None:
                        raise RuntimeError("[OursTrainer] pl_feat requires feature hooks.")
                    sf = self._hook_student.value
                    if sf is not None:
                        loss_aux = loss_aux + F.mse_loss(sf, anchor_feat)
                        n_aux += 1

            if n_aux > 0:
                loss_aux = loss_aux / float(n_aux)

            # -----------------------------
            # 5) Pseudo label: only on full view, gated by p_var (view stability)
            # -----------------------------
            loss_pl = torch.zeros((), device=self.device)
            if logits_full_s is not None:
                assert self._pos_thr is not None and self._neg_thr is not None and self._var_thr is not None
                loss_pl = pseudo_label_loss_sigmoid(
                    logits_full_s,
                    p_ref,
                    p_var=p_var,
                    pos_thr=self._pos_thr,
                    neg_thr=self._neg_thr,
                    var_thr=self._var_thr,
                    var_gate=self.pl_var_gate,
                    var_weight_enable=self.pl_var_weight_enable,
                    var_weight_type=self.pl_var_weight_type,
                    var_weight_beta=self.pl_var_weight_beta,
                    var_weight_scale=self._var_w_scale,
                    var_weight_min=self.pl_var_weight_min,
                    var_weight_max=self.pl_var_weight_max,
                    use_neg=self.pl_use_neg,
                    neg_weight=self.pl_neg_weight,
                    min_voxels=self.pl_min_voxels,
                    # p_ref already applied hierarchy fix (if enabled)
                    hierarchy_fix=False,
                )

            if self.loss_mode == "pl_cons":
                total = self.w_pl * loss_pl + self.w_cons * loss_aux
            else:
                total = self.w_pl * loss_pl + self.w_feat * loss_aux

            total.backward()

            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=float(self.grad_clip_norm),
                )

            self.optimizer.step()

            # -----------------------------
            # 7) EMA update teacher
            # -----------------------------
            ema_update(
                self.teacher,
                unwrap_model(self.model),
                decay=self.ema_decay,
                bn_buffers=self.teacher_bn_buffers,
            )

    @torch.no_grad()
    def predict(self, batch: Dict[str, Any]) -> torch.Tensor:
        assert self.model is not None
        x = batch["image"].to(self.device, non_blocking=True)
        self.model.eval()

        if self.predict_with_teacher and (self.teacher is not None):
            self.teacher.eval()
            return self.teacher(x)

        return self.model(x)
