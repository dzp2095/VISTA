from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


def _parse_donor_map(donor_map: Any, C: int) -> List[int]:
    """Return a donor index for each channel c.

    donor_map:
      - "circular": donor[c] = (c+1) % C
      - list/tuple length C: explicit donor indices
      - None: circular
    """
    if donor_map is None or (isinstance(donor_map, str) and donor_map.lower() in ("circular", "cycle")):
        return [(c + 1) % C for c in range(C)]
    if isinstance(donor_map, (list, tuple)):
        if len(donor_map) != C:
            raise ValueError(f"[amp_swap] donor_map len={len(donor_map)} != C={C}")
        out = [int(x) for x in donor_map]
        for d in out:
            if d < 0 or d >= C:
                raise ValueError(f"[amp_swap] donor_map contains out-of-range donor index: {d}")
        return out
    # omegaconf ListConfig support
    try:
        arr = [int(x) for x in list(donor_map)]
        if len(arr) != C:
            raise ValueError
        for d in arr:
            if d < 0 or d >= C:
                raise ValueError
        return arr
    except Exception:
        raise ValueError(f"[amp_swap] Unsupported donor_map type: {type(donor_map)}")


def sigmoid_entropy_map(p: torch.Tensor, eps: float = 1e-8, reduce: str = "mean") -> torch.Tensor:
    """Compute per-voxel entropy map for multi-label sigmoid probs.

    Args:
      p: (B, C, D, H, W) probabilities in [0,1]
      reduce: mean | max | sum over classes
    Returns:
      (B, 1, D, H, W) entropy map
    """
    if p.dim() < 3:
        raise ValueError(f"[entropy_map] expect p dim>=3, got {p.shape}")
    ent = -(p * torch.log(p.clamp_min(eps)) + (1.0 - p) * torch.log((1.0 - p).clamp_min(eps)))  # (B,C,...)
    r = str(reduce).lower()
    if r == "mean":
        m = ent.mean(dim=1, keepdim=True)
    elif r == "max":
        m = ent.max(dim=1, keepdim=True).values
    elif r == "sum":
        m = ent.sum(dim=1, keepdim=True)
    else:
        raise ValueError(f"[entropy_map] unknown reduce={reduce}")
    return m


@torch.no_grad()
def amplitude_swap_lowfreq_3d(
    x: torch.Tensor,
    *,
    lowfreq_ratio: float = 0.1,
    donor_map: Any = "circular",
    pair: Optional[Sequence[int]] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cross-modality low-frequency amplitude swap for 3D volumes.

    - Keeps phase (structural component) of each channel.
    - If `pair=[a,b]` is provided: mutually swap a centered low-frequency cube of amplitude
      between channels a and b (all other channels unchanged).
    - Otherwise (legacy): replace the cube for every channel c with that from `donor_map[c]`.
    - Input/output are full-modality tensors: shape (B, C, D, H, W).

    This is designed as a *label-preserving appearance intervention* for aligned mpMRI sequences.

    Notes:
      - Uses torch.fft.fftn and fftshift; runs under no_grad.
      - Casts to float32 for FFT stability, then casts back to input dtype.
    """
    if x.dim() != 5:
        raise ValueError(f"[amp_swap] expect x shape (B,C,D,H,W), got {tuple(x.shape)}")

    B, C, D, H, W = x.shape
    if C <= 1:
        return x

    r = float(lowfreq_ratio)
    r = max(0.0, min(r, 0.5))
    if r <= 0.0:
        return x

    #   LFCCS swaps low-frequency amplitudes between a *single* randomly
    #   selected modality pair (a,b), keeping all other modalities unchanged.
    #   To keep the codebase backward compatible, we support two modes:
    #     - pair is not None: pair-wise mutual swap only on (a,b)
    #     - pair is None: legacy donor_map-based swapping for all channels

    use_pair = False
    a = b = -1
    if pair is not None:
        try:
            pp = list(pair)
        except Exception:
            pp = []
        if len(pp) == 2:
            a, b = int(pp[0]), int(pp[1])
            if 0 <= a < C and 0 <= b < C and a != b:
                use_pair = True

    donors = _parse_donor_map(donor_map, C) if (not use_pair) else []

    orig_dtype = x.dtype
    xf = x.to(dtype=torch.float32)

    # FFT per channel
    X = torch.fft.fftn(xf, dim=(-3, -2, -1))
    Xs = torch.fft.fftshift(X, dim=(-3, -2, -1))

    A = Xs.abs()
    phase = Xs / A.clamp_min(eps)

    # low-frequency centered cube
    d_sz = max(1, int(round(D * r)))
    h_sz = max(1, int(round(H * r)))
    w_sz = max(1, int(round(W * r)))

    cz, cy, cx = D // 2, H // 2, W // 2
    z0, z1 = max(0, cz - d_sz // 2), min(D, max(0, cz - d_sz // 2) + d_sz)
    y0, y1 = max(0, cy - h_sz // 2), min(H, max(0, cy - h_sz // 2) + h_sz)
    x0, x1 = max(0, cx - w_sz // 2), min(W, max(0, cx - w_sz // 2) + w_sz)

    A2 = A.clone()
    if use_pair:
        # Mutual swap of low-frequency amplitude blocks between channels a and b.
        A2[:, a, z0:z1, y0:y1, x0:x1] = A[:, b, z0:z1, y0:y1, x0:x1]
        A2[:, b, z0:z1, y0:y1, x0:x1] = A[:, a, z0:z1, y0:y1, x0:x1]
    else:
        # Legacy: swap a low-frequency block for every channel based on donor_map.
        for c in range(C):
            d = donors[c]
            if d == c:
                continue
            A2[:, c, z0:z1, y0:y1, x0:x1] = A[:, d, z0:z1, y0:y1, x0:x1]

    X2s = A2 * phase
    X2 = torch.fft.ifftshift(X2s, dim=(-3, -2, -1))
    x2 = torch.fft.ifftn(X2, dim=(-3, -2, -1)).real

    return x2.to(dtype=orig_dtype)


def _dilate_mask_3d(mask: torch.Tensor, ks: int) -> torch.Tensor:
    if ks <= 1:
        return mask
    ks = int(ks)
    if ks % 2 == 0:
        ks = ks + 1
    m = mask.to(dtype=torch.float32)
    m = F.max_pool3d(m, kernel_size=ks, stride=1, padding=ks // 2)
    return (m > 0.0)


@torch.no_grad()
def patch_swap_uncertain_3d(
    x: torch.Tensor,
    uncert_map: torch.Tensor,
    *,
    unc_thr: Optional[float] = None,
    unc_quantile: Optional[float] = 0.95,
    dilate_ks: int = 15,
    pair: Optional[Sequence[int]] = None,
    mode: str = "swap",  # swap | replace
) -> torch.Tensor:
    """Patch-level cross-modality intervention in uncertain regions (3D).

    This is a *label-preserving*, within-case intervention for aligned mpMRI.
    The intervention is restricted to an uncertainty-derived mask (with optional dilation),
    and operates on full-modality inputs.

    Args:
      x: (B, C, D, H, W) full-modality tensor.
      uncert_map: uncertainty map, shape (B, 1, D, H, W) or (B, D, H, W).
      unc_thr: absolute threshold; if None, use unc_quantile.
      unc_quantile: per-volume quantile threshold (e.g., 0.95). If None and unc_thr is None -> no-op.
      dilate_ks: dilation kernel size (odd preferred). Larger => more patch-like swap.
      pair: modality index pair [a, b].
        - mode="swap": swap a<->b inside mask.
        - mode="replace": replace channel a by channel b inside mask (directional).
      mode: "swap" | "replace".
    Returns:
      x_swapped: (B, C, D, H, W)
    """
    if x.dim() != 5:
        raise ValueError(f"[patch_swap] expect x shape (B,C,D,H,W), got {tuple(x.shape)}")

    B, C, D, H, W = x.shape
    if C <= 1:
        return x

    if uncert_map.dim() == 4:
        u = uncert_map.unsqueeze(1)
    elif uncert_map.dim() == 5:
        u = uncert_map
    else:
        raise ValueError(f"[patch_swap] uncert_map must be (B,1,D,H,W) or (B,D,H,W), got {tuple(uncert_map.shape)}")

    if u.size(1) != 1:
        # if caller passed (B,C,...) reduce to mean
        u = u.mean(dim=1, keepdim=True)

    if unc_thr is None:
        if unc_quantile is None:
            return x
        q = float(unc_quantile)
        q = max(0.0, min(q, 1.0))
        # per-volume threshold
        thr = torch.quantile(u.view(B, -1), q, dim=1).view(B, 1, 1, 1, 1)
        mask = (u >= thr)
    else:
        mask = (u >= float(unc_thr))

    mask = _dilate_mask_3d(mask, int(dilate_ks))
    if not mask.any():
        return x

    # default: [t1c, t2f] if C==4 and ordering [t1n,t1c,t2w,t2f]
    if pair is None:
        if C >= 4:
            pair = [1, 3]
        else:
            pair = [0, 1]

    if len(pair) != 2:
        raise ValueError(f"[patch_swap] pair must have len=2, got {pair}")

    a, b = int(pair[0]), int(pair[1])
    if a < 0 or a >= C or b < 0 or b >= C or a == b:
        # invalid pair => no-op
        return x

    mode = str(mode).lower()
    if mode not in ("swap", "replace"):
        raise ValueError(f"[patch_swap] unknown mode={mode} (expect 'swap' or 'replace')")

    xv = x.clone()
    m = mask[:, 0]  # (B,D,H,W) bool

    if mode == "swap":
        xa = xv[:, a].clone()
        xb = xv[:, b].clone()
        xv[:, a] = torch.where(m, xb, xv[:, a])
        xv[:, b] = torch.where(m, xa, xv[:, b])
    else:  # replace: a <- b
        xb = xv[:, b]
        xv[:, a] = torch.where(m, xb, xv[:, a])

    return xv
