"""Perturbations, feature engineering, and the shared competence scorer for
the Forecast-Time Behavioral Competence Routing proof of concept.

Nothing here is deployable on its own; it is only used inside
`run_behavioral_competence.py`, which enforces the router_train/router_val
split and all integrity checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Perturbations (P1-P4). All operate on RAW-scale history [N, L, F] and use
# only values already present in that same historical window -- no future
# information, no target access.
# ---------------------------------------------------------------------------


def perturb_noise(history: torch.Tensor, seed: int, scale: float = 0.05) -> torch.Tensor:
    hist_std = history.std(dim=1, keepdim=True).clamp_min(1e-8)  # per-window, per-variable
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(history.shape, generator=gen, dtype=torch.float32) * (scale * hist_std)
    return history + noise


def perturb_mask_recent(history: torch.Tensor, fraction: float = 0.10) -> torch.Tensor:
    n, length, _ = history.shape
    k = max(1, round(length * fraction))
    window_mean = history.mean(dim=1, keepdim=True)  # per-window, per-variable historical-window mean
    out = history.clone()
    out[:, length - k :, :] = window_mean.expand(-1, k, -1)
    return out


def perturb_smooth(history: torch.Tensor, window: int = 5) -> torch.Tensor:
    n, length, feats = history.shape
    pad = window // 2
    x = history.permute(0, 2, 1)  # [N,F,L]
    x_padded = F.pad(x, (pad, pad), mode="replicate")
    kernel = torch.full((feats, 1, window), 1.0 / window, dtype=torch.float32)
    smoothed = F.conv1d(x_padded, kernel, groups=feats)
    smoothed = smoothed[..., :length]
    return smoothed.permute(0, 2, 1)


def perturb_amplitude(history: torch.Tensor, factor: float = 1.1) -> torch.Tensor:
    mean = history.mean(dim=1, keepdim=True)
    return mean + factor * (history - mean)


PERTURBATIONS: dict[str, Any] = {
    "P1_noise": lambda h, seed: perturb_noise(h, seed=seed),
    "P2_mask_recent": lambda h, seed: perturb_mask_recent(h),
    "P3_smooth": lambda h, seed: perturb_smooth(h),
    "P4_amplitude": lambda h, seed: perturb_amplitude(h),
}
PERTURBATION_SEED_BASE = 20260821


# ---------------------------------------------------------------------------
# Feature groups A-D. Every function is vectorized over the window axis and
# aggregates across the variable axis so feature dimensionality never depends
# on a dataset's number of variables.
# ---------------------------------------------------------------------------


def _slope(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Least-squares slope of x along `dim` against a 0..T-1 time index."""
    t_len = x.shape[dim]
    t = torch.arange(t_len, dtype=torch.float32)
    t = t - t.mean()
    denom = (t * t).sum().clamp_min(1e-8)
    shape = [1] * x.ndim
    shape[dim] = t_len
    t = t.view(shape)
    x_centered = x - x.mean(dim=dim, keepdim=True)
    return (x_centered * t).sum(dim=dim) / denom


def _lag1_autocorr(x: torch.Tensor) -> torch.Tensor:
    # x: [N, L, F] -> [N, F]
    x_c = x - x.mean(dim=1, keepdim=True)
    num = (x_c[:, 1:, :] * x_c[:, :-1, :]).sum(dim=1)
    den = (x_c[:, :-1, :] * x_c[:, :-1, :]).sum(dim=1).clamp_min(1e-8)
    return num / den


def _spectral_entropy(x: torch.Tensor) -> torch.Tensor:
    # x: [N, L, F] -> [N, F], normalized to [0,1]
    spec = torch.fft.rfft(x - x.mean(dim=1, keepdim=True), dim=1)
    power = spec.abs().pow(2)
    power = power / power.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy = -(power * (power.clamp_min(1e-12)).log()).sum(dim=1)
    return entropy / math.log(power.shape[1])


def window_features_group_a(history: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """[N, L, F] history -> [N, 6] dataset-size-invariant window features."""
    stdv = std.view(1, -1).clamp_min(1e-8)
    n, length, _ = history.shape
    t = torch.arange(length, dtype=torch.float32) - (length - 1) / 2.0
    t_norm = t / t.abs().max().clamp_min(1.0)
    x_c = history - history.mean(dim=1, keepdim=True)
    denom = (x_c.pow(2).sum(dim=1) * (t_norm.pow(2).sum())).clamp_min(1e-12).sqrt()
    trend_strength = ((x_c * t_norm.view(1, -1, 1)).sum(dim=1) / denom).abs().mean(dim=1)  # [N]

    volatility = (history.std(dim=1) / stdv).mean(dim=1)
    first_diff = (history[:, 1:, :] - history[:, :-1, :]).abs().mean(dim=1)
    mean_abs_first_diff = (first_diff / stdv).mean(dim=1)
    lag1 = _lag1_autocorr(history).mean(dim=1)
    entropy = _spectral_entropy(history).mean(dim=1)
    k = max(1, round(length * 0.1))
    recent_mean = history[:, length - k :, :].mean(dim=1)
    full_mean = history.mean(dim=1)
    shift = ((recent_mean - full_mean) / stdv).abs().mean(dim=1)

    return torch.stack([trend_strength, volatility, mean_abs_first_diff, lag1, entropy, shift], dim=1)


GROUP_A_NAMES = ["trend_strength", "volatility", "mean_abs_first_diff", "lag1_autocorr", "spectral_entropy", "recent_vs_full_mean_shift"]


def forecast_features_group_b(forecast: torch.Tensor, last_observed: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """[N,H,F] normal forecast, [N,F] last observed raw value -> [N,4]."""
    stdv = std.view(1, -1).clamp_min(1e-8)
    variance = (forecast.var(dim=1) / stdv.pow(2)).mean(dim=1)
    slope = (_slope(forecast, dim=1).abs() / stdv).mean(dim=1)
    first_vs_last_observed = ((forecast[:, 0, :] - last_observed) / stdv).abs().mean(dim=1)
    magnitude = (forecast.abs() / stdv.unsqueeze(1)).mean(dim=(1, 2))
    return torch.stack([variance, slope, first_vs_last_observed, magnitude], dim=1)


GROUP_B_NAMES = ["forecast_variance", "forecast_slope", "first_forecast_vs_last_observed", "mean_forecast_magnitude"]


def disagreement_features_group_c(forecast_e: torch.Tensor, forecasts_all: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """forecast_e: [N,H,F] this expert's forecast. forecasts_all: [N,H,F,K]
    all core experts' forecasts (includes this expert). -> [N,5]."""
    stdv = std.view(1, -1).clamp_min(1e-8)
    h = forecast_e.shape[1]
    ensemble_mean = forecasts_all.mean(dim=-1)
    ensemble_median = forecasts_all.median(dim=-1).values
    dist_mean = ((forecast_e - ensemble_mean) / stdv.unsqueeze(1)).abs().mean(dim=(1, 2))
    dist_median = ((forecast_e - ensemble_median) / stdv.unsqueeze(1)).abs().mean(dim=(1, 2))
    k = forecasts_all.shape[-1]
    pairwise = ((forecast_e.unsqueeze(-1) - forecasts_all) / stdv.view(1, 1, -1, 1)).abs().mean(dim=(1, 2, 3)) * (k / max(k - 1, 1))
    half = h // 2
    early = ((forecast_e[:, :half] - ensemble_mean[:, :half]) / stdv.unsqueeze(1)).abs().mean(dim=(1, 2))
    late = ((forecast_e[:, half:] - ensemble_mean[:, half:]) / stdv.unsqueeze(1)).abs().mean(dim=(1, 2))
    return torch.stack([dist_mean, dist_median, pairwise, early, late], dim=1)


GROUP_C_NAMES = ["dist_from_ensemble_mean", "dist_from_ensemble_median", "avg_pairwise_disagreement", "early_horizon_disagreement", "late_horizon_disagreement"]


def behavioral_features_one_perturbation(original: torch.Tensor, perturbed: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """[N,H,F] original vs perturbed forecast -> [N,5] behavioral stats."""
    stdv = std.view(1, -1).clamp_min(1e-8)
    h = original.shape[1]
    diff = ((perturbed - original) / stdv.unsqueeze(1)).abs()
    change = diff.mean(dim=(1, 2))
    half = h // 2
    early_change = diff[:, :half].mean(dim=(1, 2))
    late_change = diff[:, half:].mean(dim=(1, 2))
    slope_change = ((_slope(perturbed, dim=1) - _slope(original, dim=1)).abs() / stdv).mean(dim=1)
    var_change = ((perturbed.var(dim=1) - original.var(dim=1)).abs() / stdv.pow(2)).mean(dim=1)
    return torch.stack([change, early_change, late_change, slope_change, var_change], dim=1)


BEHAVIORAL_STAT_NAMES = ["change", "early_change", "late_change", "slope_change", "variance_change"]


def behavioral_features_all(original: torch.Tensor, perturbed_by_name: Mapping[str, torch.Tensor], std: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
    parts, names = [], []
    for pname in PERTURBATIONS:
        parts.append(behavioral_features_one_perturbation(original, perturbed_by_name[pname], std))
        names.extend(f"{pname}__{s}" for s in BEHAVIORAL_STAT_NAMES)
    return torch.cat(parts, dim=1), names


ABLATIONS = {
    "A_window_only": ["A"],
    "B_window_forecast": ["A", "B"],
    "C_window_forecast_disagreement": ["A", "B", "C"],
    "D_full_behavioral": ["A", "B", "C", "D"],
}


@dataclass
class FeatureBundle:
    group_a: torch.Tensor  # [N,6]
    group_b: torch.Tensor  # [N,4]
    group_c: torch.Tensor  # [N,5]
    group_d: torch.Tensor  # [N,20]
    names: dict[str, list[str]]

    def features_for(self, ablation: str) -> torch.Tensor:
        groups = {"A": self.group_a, "B": self.group_b, "C": self.group_c, "D": self.group_d}
        return torch.cat([groups[g] for g in ABLATIONS[ablation]], dim=-1)

    def feature_names_for(self, ablation: str) -> list[str]:
        out: list[str] = []
        for g in ABLATIONS[ablation]:
            out.extend(self.names[g])
        return out


# ---------------------------------------------------------------------------
# Shared competence scorer: a small MLP, no expert identity as input.
# ---------------------------------------------------------------------------


class CompetenceScorer(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class ScorerFit:
    model: CompetenceScorer
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    best_epoch: int
    best_internal_val_mse: float
    temperature: float
    train_windows: int
    internal_val_windows: int

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            normalized = (features - self.feature_mean) / self.feature_std
            return self.model(normalized)


def train_competence_scorer(
    features: torch.Tensor,
    targets: torch.Tensor,
    n_train_windows: int,
    window_id_train: torch.Tensor,
    window_id_internal_val: torch.Tensor,
    seed: int = 7,
    max_epochs: int = 200,
    patience: int = 15,
    weight_decay: float = 1e-4,
    lr: float = 1e-3,
    temperature_grid: Sequence[float] = (0.02, 0.05, 0.1, 0.2, 0.5),
) -> ScorerFit:
    """`features`/`targets` are row-per-(window,expert) samples, chronologically
    ordered by window within router_train. `window_id_train`/`window_id_internal_val`
    partition the *window* axis chronologically (early router_train -> train,
    later router_train -> internal validation/early-stopping), matching the
    project's existing train-only chronological protocol. router_val is never
    touched by this function. `window_id_train`/`window_id_internal_val` are
    row-index tensors into `features`/`targets`."""
    torch.manual_seed(seed)

    feature_mean = features[: n_train_windows].mean(dim=0, keepdim=True)
    feature_std = features[: n_train_windows].std(dim=0, keepdim=True).clamp_min(1e-6)
    x = (features - feature_mean) / feature_std

    model = CompetenceScorer(features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_train, y_train = x[window_id_train], targets[window_id_train]
    x_val, y_val = x[window_id_internal_val], targets[window_id_internal_val]

    best_val = math.inf
    best_state = None
    best_epoch = -1
    bad = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(x_train)
        loss = F.mse_loss(pred, y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(F.mse_loss(model(x_val), y_val))
        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)

    # Temperature: pick from a small predefined grid using ONLY the internal
    # validation slice of router_train (never router_val).
    model.eval()
    with torch.no_grad():
        val_pred = model(x_val)
    best_temp, best_temp_score = temperature_grid[0], math.inf
    for temp in temperature_grid:
        # Proxy objective: weighted average of *actual* excess loss under the
        # softmax(-pred/temp) weighting, computed per window on the internal
        # validation slice grouped by window -- approximated here at the
        # sample level since this is only a coarse, train-only selection step.
        w = torch.softmax(-val_pred / temp, dim=0)
        score = float((w * y_val).sum() / w.sum().clamp_min(1e-8))
        if score < best_temp_score:
            best_temp_score = score
            best_temp = temp

    return ScorerFit(
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        best_epoch=best_epoch,
        best_internal_val_mse=best_val,
        temperature=float(best_temp),
        train_windows=int(x_train.shape[0]),
        internal_val_windows=int(x_val.shape[0]),
    )


def competence_to_weights(predicted_excess_loss: torch.Tensor, temperature: float) -> torch.Tensor:
    """predicted_excess_loss: [N,K] -> softmax(-pred/temp) weights [N,K]."""
    return torch.softmax(-predicted_excess_loss / max(temperature, 1e-8), dim=-1)
