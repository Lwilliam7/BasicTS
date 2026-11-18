"""Learned diagnostic probes: a small, shared, instance-conditioned generator
that produces a bounded perturbation of the current historical window, used
only to measure a frozen expert's response -- never as a final forecast.

Constraints (Section: "Add constraints so the generator cannot create
arbitrary adversarial garbage"):
  - magnitude <= eps * historical_std, structurally guaranteed by
    `delta = eps * std * tanh(raw)` (|tanh| <= 1), verified empirically as an
    integrity check in run_learned_probe.py.
  - near-zero mean shift per variable: soft penalty (`mean_shift_penalty`).
  - temporal smoothness: soft penalty (`smoothness_penalty`).
  - no modification outside the observed window: structural (delta has
    exactly the shape of the input history and is only ever added to it).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS_DEFAULT = 0.05


class ProbeGenerator(nn.Module):
    """Shared across experts and windows within one dataset. Input: the
    normalized current window and a small summary of the expert's own
    forecast (variance, slope, first-vs-last-observed, magnitude -- the same
    four statistics as feature Group B). No target, error, or expert
    identity is ever passed in."""

    def __init__(self, num_features: int, hidden: int = 32, eps: float = EPS_DEFAULT) -> None:
        super().__init__()
        self.eps = eps
        self.window_proj = nn.Linear(num_features, hidden)
        self.temporal_conv = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.forecast_proj = nn.Linear(4, hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, num_features))

    def raw_delta(self, window_norm: torch.Tensor, forecast_summary: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.window_proj(window_norm))  # [B,L,hidden]
        h = F.relu(self.temporal_conv(h.transpose(1, 2))).transpose(1, 2)  # [B,L,hidden]
        fsum = F.relu(self.forecast_proj(forecast_summary))  # [B,hidden]
        fsum_broadcast = fsum.unsqueeze(1).expand(-1, h.shape[1], -1)
        combined = torch.cat([h, fsum_broadcast], dim=-1)
        return self.head(combined)  # [B,L,F] raw, unbounded

    def make_probe(self, history_raw: torch.Tensor, window_norm: torch.Tensor, forecast_summary: torch.Tensor, hist_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta_raw = self.raw_delta(window_norm, forecast_summary)
        delta = self.eps * hist_std.unsqueeze(1) * torch.tanh(delta_raw)
        return history_raw + delta, delta


class GlobalProbeGenerator(nn.Module):
    """One fixed, learned perturbation SHAPE (not instance-conditioned) for
    the whole dataset -- distinguishes "learning a good universal probe" from
    "adapting the probe to each window/expert"."""

    def __init__(self, input_len: int, num_features: int, eps: float = EPS_DEFAULT) -> None:
        super().__init__()
        self.eps = eps
        self.delta_raw = nn.Parameter(torch.zeros(input_len, num_features))

    def make_probe(self, history_raw: torch.Tensor, hist_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = history_raw.shape[0]
        delta = self.eps * hist_std.unsqueeze(1) * torch.tanh(self.delta_raw).unsqueeze(0).expand(b, -1, -1)
        return history_raw + delta, delta


def _slope(x: torch.Tensor, dim: int) -> torch.Tensor:
    t_len = x.shape[dim]
    t = torch.arange(t_len, dtype=torch.float32, device=x.device) - (t_len - 1) / 2.0
    denom = (t * t).sum().clamp_min(1e-8)
    shape = [1] * x.ndim
    shape[dim] = t_len
    t = t.view(shape)
    x_c = x - x.mean(dim=dim, keepdim=True)
    return (x_c * t).sum(dim=dim) / denom


def probe_response_features(original: torch.Tensor, probe: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """[N,H,F] original vs probe forecast, [F] scaler std -> [N,6] diagnostic
    features: change, early-horizon, late-horizon, slope change, variance
    change, and a cosine/correlation change (1 - cosine similarity of the
    flattened (H,F) forecast vectors)."""
    stdv = std.view(1, -1).clamp_min(1e-8)
    h = original.shape[1]
    diff = ((probe - original) / stdv.unsqueeze(1)).abs()
    change = diff.mean(dim=(1, 2))
    half = h // 2
    early_change = diff[:, :half].mean(dim=(1, 2))
    late_change = diff[:, half:].mean(dim=(1, 2))
    slope_change = ((_slope(probe, dim=1) - _slope(original, dim=1)).abs() / stdv).mean(dim=1)
    var_change = ((probe.var(dim=1) - original.var(dim=1)).abs() / stdv.pow(2)).mean(dim=1)
    o_flat = original.reshape(original.shape[0], -1)
    p_flat = probe.reshape(probe.shape[0], -1)
    cos_sim = F.cosine_similarity(o_flat, p_flat, dim=1, eps=1e-8)
    cosine_change = 1.0 - cos_sim
    return torch.stack([change, early_change, late_change, slope_change, var_change, cosine_change], dim=1)


PROBE_RESPONSE_STAT_NAMES = ["change", "early_change", "late_change", "slope_change", "variance_change", "cosine_change"]


def pairwise_ranking_loss(pred_excess: torch.Tensor, actual_excess: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    """pred_excess/actual_excess: [B,K]. Hinge loss over all unordered
    expert pairs within each window, encouraging the predicted ordering to
    match the actual ordering by at least `margin`."""
    k = pred_excess.shape[1]
    losses = []
    for i in range(k):
        for j in range(i + 1, k):
            sign = torch.sign(actual_excess[:, i] - actual_excess[:, j])
            diff = pred_excess[:, i] - pred_excess[:, j]
            losses.append(F.relu(margin - sign * diff))
    return torch.cat(losses).mean()


def router_train_gap_scale(actual_excess: torch.Tensor) -> float:
    """Deterministic, router_train-only normalization constant: the mean
    absolute pairwise loss gap over every window and every expert pair.
    Computed once before training and never touched by router_val."""
    k = actual_excess.shape[1]
    gaps = []
    for i in range(k):
        for j in range(i + 1, k):
            gaps.append((actual_excess[:, i] - actual_excess[:, j]).abs())
    return float(torch.cat(gaps).mean())


def loss_gap_weighted_pairwise_ranking_loss(
    pred_excess: torch.Tensor,
    actual_excess: torch.Tensor,
    gap_scale: float,
    margin: float = 0.05,
    clip_low: float = 0.25,
    clip_high: float = 4.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Same pairwise hinge-ranking loss as `pairwise_ranking_loss`, but each
    pair's term is weighted by how much the true losses actually differ,
    normalized by the router_train-only `gap_scale` and clipped to
    [clip_low, clip_high] so a single extreme window cannot destabilize
    training. A near-tied pair (gap << gap_scale) contributes almost nothing;
    a pair with a large true performance gap (gap >> gap_scale) contributes
    up to clip_high x as much as an average-gap pair."""
    k = pred_excess.shape[1]
    weighted_losses = []
    for i in range(k):
        for j in range(i + 1, k):
            gap = (actual_excess[:, i] - actual_excess[:, j]).abs()
            gap_weight = (gap / (gap_scale + eps)).clamp(clip_low, clip_high)
            sign = torch.sign(actual_excess[:, i] - actual_excess[:, j])
            diff = pred_excess[:, i] - pred_excess[:, j]
            pair_loss = F.relu(margin - sign * diff)
            weighted_losses.append(gap_weight * pair_loss)
    return torch.cat(weighted_losses).mean()


def perturbation_penalties(delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    l2 = delta.pow(2).mean()
    mean_shift = delta.mean(dim=1).pow(2).mean()
    smoothness = (delta[:, 1:, :] - delta[:, :-1, :]).pow(2).mean()
    return l2, mean_shift, smoothness
