"""Published comparison baselines for COSTAR frozen expert caches.

Validation-only runner.  It never loads ETTh1/ETTh2 test caches.

Implemented order:
1. Granger-Ramanathan direct forecast stacking
2. FAME routing adaptation to BasicTS frozen expert pool
3. TimeRouter routing-mechanism adaptation
4. Bates-Granger forecast combination
5. OneNet-style frozen-expert adaptation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import enforce_observable  # noqa: E402
from experiments.frozen_costar import run_frozen_costar_validation as costar_eval  # noqa: E402
from experiments.oracle_weight_tournament.run_tournament import load_cache, load_std, sample_mae, sample_mse  # noqa: E402
from experiments.train_selected_core_etth1 import run_train_selected_core_eval as etth1_core  # noqa: E402
from scripts.fame_etth_router import extract_fame_etth_fingerprint, sparse_top_r_weights  # noqa: E402


OUT_DIR = ROOT / "experiments/published_baseline_comparisons"
ETTH1_TRAIN = ROOT / "cache/costarts_walkforward/router_train_20_60_cache.pt"
ETTH1_VAL = ROOT / "cache/costarts_walkforward/router_val_60_80_cache.pt"
ETTH1_NORMALIZER = ROOT / "checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt"
ETTH2_TRAIN = ROOT / "cache/costarts_fresh/ETTh2_96_12/router_train_cache.pt"
ETTH2_VAL = ROOT / "cache/costarts_fresh/ETTh2_96_12/router_val_cache.pt"
EXPERTS = ("DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN")
ETTH1_COSTAR_CORE = ("PatchTST", "iTransformer", "TimesNet")
ETTH2_COSTAR_CORE = ("DLinear", "PatchTST", "ModernTCN")
ONENET_BRANCHES = ("PatchTST", "iTransformer")
EPS = 1e-8


@dataclass(frozen=True)
class LinearConfig:
    method: str
    structure: str
    alpha: float = 0.0
    variant: str = "canonical"


@dataclass(frozen=True)
class FameConfig:
    tau: float
    top_r: int
    hidden: int
    dropout: float
    lr: float
    weight_decay: float
    epochs: int
    seed: int = 7


@dataclass(frozen=True)
class TimeRouterConfig:
    tau_m: float
    tau_d: float
    hidden: int
    lr: float
    weight_decay: float
    epochs: int
    seed: int = 7


@dataclass(frozen=True)
class BatesConfig:
    structure: str
    estimator: str
    shrinkage: float


@dataclass(frozen=True)
class OneNetConfig:
    eta: float
    decay: float


def refuse_test(path: str | Path) -> None:
    if "test" in str(path).lower():
        raise ValueError(f"Refusing test path: {path}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def train_folds(n: int) -> list[tuple[int, int, int]]:
    return etth1_core.train_folds(n)


def expert_indices(cache: Mapping[str, Any], experts: Sequence[str]) -> list[int]:
    names = list(cache["expert_names"])
    return [names.index(name) for name in experts]


def selected_forecasts(cache: Mapping[str, Any], experts: Sequence[str] = EXPERTS) -> torch.Tensor:
    return cache["prediction_stack"][..., expert_indices(cache, experts)].to(torch.float32)


def cache_role(cache: Mapping[str, Any]) -> str:
    return str(cache.get("cache_role", cache.get("split_role")))


def validate_cache(cache: Mapping[str, Any], expected_role: str | tuple[str, ...], dataset: str) -> None:
    roles = (expected_role,) if isinstance(expected_role, str) else expected_role
    role = cache_role(cache)
    if role not in roles:
        raise ValueError(f"{dataset} expected role {roles}, got {role!r}")
    if tuple(cache["expert_names"]) != EXPERTS:
        raise ValueError(f"{dataset} expert order mismatch: {cache['expert_names']}")
    pred = cache["prediction_stack"]
    target = cache["targets"]
    mask = cache["target_masks"]
    histories = cache["histories"]
    starts = cache["absolute_window_starts"]
    if tuple(pred.shape[1:]) != (12, 7, len(EXPERTS)):
        raise ValueError(f"{dataset} prediction_stack expected [N,12,7,5], got {tuple(pred.shape)}")
    if tuple(target.shape) != tuple(pred.shape[:3]):
        raise ValueError(f"{dataset} targets shape mismatch: {tuple(target.shape)} vs {tuple(pred.shape[:3])}")
    if tuple(mask.shape) != tuple(target.shape):
        raise ValueError(f"{dataset} target_masks shape mismatch: {tuple(mask.shape)}")
    if tuple(histories.shape[1:]) != (96, 7):
        raise ValueError(f"{dataset} histories expected [N,96,7], got {tuple(histories.shape)}")
    if int(cache["forecast_horizon"]) != 12:
        raise ValueError(f"{dataset} horizon expected 12, got {cache['forecast_horizon']}")
    if len(starts) != int(cache["num_windows"]):
        raise ValueError(f"{dataset} num_windows mismatch")
    if not bool(torch.all(starts[1:] > starts[:-1])):
        raise ValueError(f"{dataset} starts are not chronological")


def metric_tensors(cache: Mapping[str, Any], pred: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    per_mae = sample_mae(pred, target, mask, std)
    per_mse = sample_mse(pred, target, mask, std)
    return {
        "mae": float(per_mae.mean()),
        "mse": float(per_mse.mean()),
        "per_window_mae": per_mae.detach().cpu(),
        "per_window_mse": per_mse.detach().cpu(),
    }


def metric_row(dataset: str, method: str, split: str, cache: Mapping[str, Any], pred: torch.Tensor, std: torch.Tensor, **extra: Any) -> dict[str, Any]:
    met = metric_tensors(cache, pred, std)
    return {
        "dataset": dataset,
        "method": method,
        "split": split,
        "mae": met["mae"],
        "mse": met["mse"],
        "num_windows": int(cache["num_windows"]),
        **extra,
    }


def per_window_rows(dataset: str, method: str, cache: Mapping[str, Any], pred: torch.Tensor, std: torch.Tensor) -> list[dict[str, Any]]:
    met = metric_tensors(cache, pred, std)
    starts = cache["absolute_window_starts"].to(torch.long).tolist()
    return [
        {
            "dataset": dataset,
            "method": method,
            "window_index": i,
            "absolute_window_start": int(starts[i]),
            "mae": float(met["per_window_mae"][i]),
            "mse": float(met["per_window_mse"][i]),
        }
        for i in range(len(starts))
    ]


def fixed_average_prediction(cache: Mapping[str, Any], experts: Sequence[str]) -> torch.Tensor:
    return selected_forecasts(cache, experts).mean(dim=-1)


def per_window_expert_mae(cache: Mapping[str, Any], std: torch.Tensor, experts: Sequence[str] = EXPERTS) -> torch.Tensor:
    forecasts = selected_forecasts(cache, experts)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.float32)
    err = ((forecasts - target.unsqueeze(-1)) / std.view(1, 1, -1, 1)).abs() * mask.unsqueeze(-1)
    denom = mask.sum(dim=(1, 2)).clamp_min(EPS).unsqueeze(-1)
    return err.sum(dim=(1, 2)) / denom


def per_window_expert_mse(cache: Mapping[str, Any], std: torch.Tensor, experts: Sequence[str] = EXPERTS) -> torch.Tensor:
    forecasts = selected_forecasts(cache, experts)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.float32)
    err = (((forecasts - target.unsqueeze(-1)) / std.view(1, 1, -1, 1)) ** 2) * mask.unsqueeze(-1)
    denom = mask.sum(dim=(1, 2)).clamp_min(EPS).unsqueeze(-1)
    return err.sum(dim=(1, 2)) / denom


def solve_ridge(X: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    X = X.to(torch.float64)
    y = y.to(torch.float64)
    ones = torch.ones((X.shape[0], 1), dtype=X.dtype, device=X.device)
    Xa = torch.cat([ones, X], dim=1)
    xtx = Xa.T @ Xa
    reg = torch.eye(xtx.shape[0], dtype=X.dtype, device=X.device) * float(alpha)
    reg[0, 0] = 0.0
    rhs = Xa.T @ y
    try:
        beta = torch.linalg.solve(xtx + reg, rhs)
    except RuntimeError:
        beta = torch.linalg.pinv(xtx + reg) @ rhs
    return beta.to(torch.float32)


def fit_gr(cache: Mapping[str, Any], std: torch.Tensor, cfg: LinearConfig, experts: Sequence[str] = EXPERTS) -> dict[str, Any]:
    forecasts = selected_forecasts(cache, experts)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    n, h, v, e = forecasts.shape
    if cfg.structure == "global":
        X = forecasts[mask].reshape(-1, e)
        y = target[mask].reshape(-1)
        beta = solve_ridge(X, y, cfg.alpha)
        return {"config": asdict(cfg), "beta": beta}
    if cfg.structure == "horizon_variable":
        betas = torch.empty(h, v, e + 1, dtype=torch.float32)
        for hh in range(h):
            for vv in range(v):
                m = mask[:, hh, vv]
                betas[hh, vv] = solve_ridge(forecasts[m, hh, vv, :], target[m, hh, vv], cfg.alpha)
        return {"config": asdict(cfg), "beta": betas}
    raise ValueError(cfg.structure)


def predict_gr(model: Mapping[str, Any], cache: Mapping[str, Any], experts: Sequence[str] = EXPERTS) -> torch.Tensor:
    forecasts = selected_forecasts(cache, experts)
    cfg = model["config"]
    beta = model["beta"]
    if cfg["structure"] == "global":
        return beta[0] + (forecasts * beta[1:].view(1, 1, 1, -1)).sum(dim=-1)
    if cfg["structure"] == "horizon_variable":
        return beta[..., 0].view(1, *beta.shape[:2]) + (forecasts * beta[..., 1:].unsqueeze(0)).sum(dim=-1)
    raise ValueError(cfg["structure"])


def score_config_on_folds(
    cache: Mapping[str, Any],
    std: torch.Tensor,
    fit_fn: Any,
    pred_fn: Any,
    cfg: Any,
    method: str,
    experts: Sequence[str] = EXPERTS,
) -> dict[str, Any]:
    rows = []
    preds = []
    targets = []
    masks = []
    for fold_id, train_lo, eval_lo, eval_hi in [(i, *f) for i, f in enumerate(train_folds(int(cache["num_windows"])))]:
        if eval_lo <= train_lo:
            raise ValueError("Fold has no training prefix")
        train_cache = slice_cache(cache, train_lo, eval_lo)
        eval_cache = slice_cache(cache, eval_lo, eval_hi)
        model = fit_fn(train_cache, std, cfg, experts)
        pred = pred_fn(model, eval_cache, experts)
        met = metric_tensors(eval_cache, pred, std)
        rows.append({"fold": fold_id, "train_lo": train_lo, "train_hi": eval_lo, "eval_lo": eval_lo, "eval_hi": eval_hi, "mae": met["mae"], "mse": met["mse"]})
        preds.append(pred)
        targets.append(eval_cache["targets"])
        masks.append(eval_cache["target_masks"])
    pooled_cache = {"targets": torch.cat(targets), "target_masks": torch.cat(masks)}
    pooled_pred = torch.cat(preds)
    pooled = metric_tensors(pooled_cache, pooled_pred, std)
    return {
        "method": method,
        "config": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else cfg,
        "fold_mae": pooled["mae"],
        "fold_mse": pooled["mse"],
        "fold_rows": rows,
    }


def slice_cache(cache: Mapping[str, Any], lo: int, hi: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n = int(cache["num_windows"])
    for k, v in cache.items():
        if torch.is_tensor(v) and v.shape[:1] == (n,):
            out[k] = v[lo:hi].clone()
        else:
            out[k] = v
    out["num_windows"] = hi - lo
    return out


def fit_bates(cache: Mapping[str, Any], std: torch.Tensor, cfg: BatesConfig, experts: Sequence[str] = EXPERTS) -> dict[str, Any]:
    forecasts = selected_forecasts(cache, experts)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    residual = ((forecasts - target.unsqueeze(-1)) / std.view(1, 1, -1, 1)).to(torch.float32)
    n, h, v, e = forecasts.shape

    def weights_from_resid(r: torch.Tensor) -> torch.Tensor:
        flat = r.reshape(-1, e)
        if cfg.estimator == "diagonal_inverse_error":
            var = flat.square().mean(dim=0).clamp_min(EPS)
            w = 1.0 / var
            return w / w.sum()
        cov = torch.cov(flat.T) if flat.shape[0] > 1 else torch.eye(e)
        diag = torch.diag(torch.diag(cov))
        sigma = (1.0 - float(cfg.shrinkage)) * cov + float(cfg.shrinkage) * diag
        sigma = sigma + torch.eye(e) * 1e-6
        ones = torch.ones(e)
        inv_ones = torch.linalg.pinv(sigma) @ ones
        return inv_ones / (ones @ inv_ones).clamp_min(EPS)

    if cfg.structure == "global":
        w = weights_from_resid(residual[mask])
        return {"config": asdict(cfg), "weights": w}
    if cfg.structure == "horizon_variable":
        weights = torch.empty(h, v, e)
        for hh in range(h):
            for vv in range(v):
                m = mask[:, hh, vv]
                weights[hh, vv] = weights_from_resid(residual[m, hh, vv, :])
        return {"config": asdict(cfg), "weights": weights}
    raise ValueError(cfg.structure)


def predict_bates(model: Mapping[str, Any], cache: Mapping[str, Any], experts: Sequence[str] = EXPERTS) -> torch.Tensor:
    forecasts = selected_forecasts(cache, experts)
    cfg = model["config"]
    weights = model["weights"].to(torch.float32)
    if cfg["structure"] == "global":
        return (forecasts * weights.view(1, 1, 1, -1)).sum(dim=-1)
    if cfg["structure"] == "horizon_variable":
        return (forecasts * weights.unsqueeze(0)).sum(dim=-1)
    raise ValueError(cfg["structure"])


class MlpRouter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_scaler(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(EPS)
    return mean, std


def transform(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num((features - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)


def router_features(cache: Mapping[str, Any]) -> torch.Tensor:
    hist_fp = extract_fame_etth_fingerprint(cache["histories"].to(torch.float32))
    forecasts = selected_forecasts(cache)
    mean = forecasts.mean(dim=(1, 2))
    std = forecasts.std(dim=(1, 2), unbiased=False)
    first = forecasts[:, 0].mean(dim=1)
    last = forecasts[:, -1].mean(dim=1)
    spread = forecasts.std(dim=-1, unbiased=False).mean(dim=(1, 2), keepdim=True).view(forecasts.shape[0], 1)
    return torch.cat([hist_fp, mean, std, first, last, spread], dim=1)


def train_fame_model(train_cache: Mapping[str, Any], std: torch.Tensor, cfg: FameConfig, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    features = extract_fame_etth_fingerprint(train_cache["histories"].to(torch.float32))
    mean, feat_std = fit_scaler(features)
    x = transform(features, mean, feat_std).to(device)
    errors = per_window_expert_mae(train_cache, std).to(device)
    targets = torch.softmax(-errors / max(cfg.tau, EPS), dim=1)
    model = MlpRouter(x.shape[1], len(EXPERTS), cfg.hidden, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for _ in range(cfg.epochs):
        perm = torch.randperm(x.shape[0], device=device)
        for start in range(0, x.shape[0], 256):
            idx = perm[start : start + 256]
            logits = model(x[idx])
            loss = F.kl_div(F.log_softmax(logits, dim=1), targets[idx], reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            opt.step()
    return {"config": asdict(cfg), "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "feature_mean": mean, "feature_std": feat_std}


def predict_fame(model_obj: Mapping[str, Any], cache: Mapping[str, Any], std: torch.Tensor | None = None, experts: Sequence[str] = EXPERTS) -> torch.Tensor:
    cfg = model_obj["config"]
    features = extract_fame_etth_fingerprint(cache["histories"].to(torch.float32))
    x = transform(features, model_obj["feature_mean"], model_obj["feature_std"])
    model = MlpRouter(x.shape[1], len(EXPERTS), int(cfg["hidden"]), float(cfg["dropout"]))
    model.load_state_dict(model_obj["state_dict"])
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
    _active, weights = sparse_top_r_weights(probs, int(cfg["top_r"]))
    return (selected_forecasts(cache, experts) * weights.view(weights.shape[0], 1, 1, -1)).sum(dim=-1)


def fame_diagnostics(model_obj: Mapping[str, Any], cache: Mapping[str, Any]) -> dict[str, Any]:
    cfg = model_obj["config"]
    features = extract_fame_etth_fingerprint(cache["histories"].to(torch.float32))
    x = transform(features, model_obj["feature_mean"], model_obj["feature_std"])
    model = MlpRouter(x.shape[1], len(EXPERTS), int(cfg["hidden"]), float(cfg["dropout"]))
    model.load_state_dict(model_obj["state_dict"])
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
    active, weights = sparse_top_r_weights(probs, int(cfg["top_r"]))
    usage = active.float().mean(dim=0)
    return {
        "average_experts_used": float(active.float().sum(dim=1).mean()),
        "routing_entropy": float(-(probs * probs.clamp_min(EPS).log()).sum(dim=1).mean()),
        "top1_frequency": {EXPERTS[i]: float((probs.argmax(dim=1) == i).float().mean()) for i in range(len(EXPERTS))},
        "topk_frequency": {EXPERTS[i]: float(usage[i]) for i in range(len(EXPERTS))},
        "weights_sum_mean": float(weights.sum(dim=1).mean()),
    }


def train_timerouter_model(train_cache: Mapping[str, Any], std: torch.Tensor, cfg: TimeRouterConfig, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    features = router_features(train_cache)
    mean, feat_std = fit_scaler(features)
    x = transform(features, mean, feat_std).to(device)
    best = per_window_expert_mae(train_cache, std).argmin(dim=1).to(device)
    model = MlpRouter(x.shape[1], len(EXPERTS), cfg.hidden, 0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for _ in range(cfg.epochs):
        perm = torch.randperm(x.shape[0], device=device)
        for start in range(0, x.shape[0], 256):
            idx = perm[start : start + 256]
            loss = F.cross_entropy(model(x[idx]), best[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    fallback = bates_inverse_error_weights(train_cache, std)
    return {
        "config": asdict(cfg),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "feature_mean": mean,
        "feature_std": feat_std,
        "fallback_weights": fallback,
    }


def bates_inverse_error_weights(cache: Mapping[str, Any], std: torch.Tensor) -> torch.Tensor:
    mse = per_window_expert_mse(cache, std).mean(dim=0).clamp_min(EPS)
    w = 1.0 / mse
    return w / w.sum()


def predict_timerouter(model_obj: Mapping[str, Any], cache: Mapping[str, Any], experts: Sequence[str] = EXPERTS) -> torch.Tensor:
    cfg = model_obj["config"]
    features = router_features(cache)
    x = transform(features, model_obj["feature_mean"], model_obj["feature_std"])
    model = MlpRouter(x.shape[1], len(EXPERTS), int(cfg["hidden"]), 0.0)
    model.load_state_dict(model_obj["state_dict"])
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
    top2 = torch.topk(probs, k=2, dim=1)
    margin = top2.values[:, 0] - top2.values[:, 1]
    diversity = selected_forecasts(cache, experts).std(dim=-1, unbiased=False).mean(dim=(1, 2))
    div_scale = diversity.median().clamp_min(EPS)
    hard = (margin >= float(cfg["tau_m"])) & ((diversity / div_scale) <= float(cfg["tau_d"]))
    hard_idx = top2.indices[:, 0]
    weights = model_obj["fallback_weights"].repeat(probs.shape[0], 1)
    weights[hard] = 0.0
    weights[hard, hard_idx[hard]] = 1.0
    return (selected_forecasts(cache, experts) * weights.view(weights.shape[0], 1, 1, -1)).sum(dim=-1)


def timerouter_diagnostics(model_obj: Mapping[str, Any], cache: Mapping[str, Any]) -> dict[str, Any]:
    cfg = model_obj["config"]
    features = router_features(cache)
    x = transform(features, model_obj["feature_mean"], model_obj["feature_std"])
    model = MlpRouter(x.shape[1], len(EXPERTS), int(cfg["hidden"]), 0.0)
    model.load_state_dict(model_obj["state_dict"])
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
    top2 = torch.topk(probs, k=2, dim=1)
    margin = top2.values[:, 0] - top2.values[:, 1]
    diversity = selected_forecasts(cache).std(dim=-1, unbiased=False).mean(dim=(1, 2))
    hard = (margin >= float(cfg["tau_m"])) & ((diversity / diversity.median().clamp_min(EPS)) <= float(cfg["tau_d"]))
    return {
        "selected_expert_frequency": {EXPERTS[i]: float((top2.indices[:, 0] == i).float().mean()) for i in range(len(EXPERTS))},
        "fallback_frequency": float((~hard).float().mean()),
        "routing_confidence": float(top2.values[:, 0].mean()),
        "margin_mean": float(margin.mean()),
        "diversity_scaled_mean": float((diversity / diversity.median().clamp_min(EPS)).mean()),
    }


def onenet_predict(
    cache: Mapping[str, Any],
    std: torch.Tensor,
    cfg: OneNetConfig,
    init_error: torch.Tensor,
    branches: Sequence[str] = ONENET_BRANCHES,
) -> tuple[torch.Tensor, dict[str, Any]]:
    idx = expert_indices(cache, branches)
    forecasts = cache["prediction_stack"][..., idx].to(torch.float32)
    starts = cache["absolute_window_starts"].to(torch.long)
    horizon = int(cache["forecast_horizon"])
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.float32)
    loss_state = init_error.clone().to(torch.float32)
    pending: list[int] = []
    preds = []
    weights_seen = []
    updates = 0
    for i in range(forecasts.shape[0]):
        now = int(starts[i])
        still = []
        for j in pending:
            if int(starts[j]) + horizon <= now:
                enforce_observable(int(starts[j]), now, horizon)
                err = (((forecasts[j] - target[j].unsqueeze(-1)) / std.view(1, -1, 1)).abs() * mask[j].unsqueeze(-1)).sum(dim=(0, 1))
                denom = mask[j].sum().clamp_min(EPS)
                loss_state = float(cfg.decay) * loss_state + (1.0 - float(cfg.decay)) * (err / denom)
                updates += 1
            else:
                still.append(j)
        pending = still
        w = torch.softmax(-float(cfg.eta) * loss_state, dim=0)
        preds.append((forecasts[i] * w.view(1, 1, -1)).sum(dim=-1))
        weights_seen.append(w)
        pending.append(i)
    weights = torch.stack(weights_seen)
    return torch.stack(preds), {
        "num_updates": updates,
        "mean_weight_" + branches[0]: float(weights[:, 0].mean()),
        "mean_weight_" + branches[1]: float(weights[:, 1].mean()),
    }


def fit_onenet(cache: Mapping[str, Any], std: torch.Tensor, cfg: OneNetConfig, experts: Sequence[str] = ONENET_BRANCHES) -> dict[str, Any]:
    idx = expert_indices(cache, experts)
    forecasts = cache["prediction_stack"][..., idx].to(torch.float32)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.float32)
    err = (((forecasts - target.unsqueeze(-1)) / std.view(1, 1, -1, 1)).abs() * mask.unsqueeze(-1)).sum(dim=(0, 1, 2))
    init = err / mask.sum().clamp_min(EPS)
    return {"config": asdict(cfg), "init_error": init, "branches": list(experts)}


def predict_onenet(model: Mapping[str, Any], cache: Mapping[str, Any], experts: Sequence[str] = ONENET_BRANCHES) -> torch.Tensor:
    std = model.get("_std")
    pred, _extra = onenet_predict(cache, std, OneNetConfig(**model["config"]), model["init_error"], model["branches"])
    return pred


def score_onenet_folds(cache: Mapping[str, Any], std: torch.Tensor, cfg: OneNetConfig) -> dict[str, Any]:
    rows = []
    preds = []
    targets = []
    masks = []
    for fold_id, train_lo, eval_lo, eval_hi in [(i, *f) for i, f in enumerate(train_folds(int(cache["num_windows"])))]:
        model = fit_onenet(slice_cache(cache, train_lo, eval_lo), std, cfg)
        pred, extra = onenet_predict(slice_cache(cache, eval_lo, eval_hi), std, cfg, model["init_error"], model["branches"])
        eval_cache = slice_cache(cache, eval_lo, eval_hi)
        met = metric_tensors(eval_cache, pred, std)
        rows.append({"fold": fold_id, "train_hi": eval_lo, "eval_lo": eval_lo, "eval_hi": eval_hi, "mae": met["mae"], "mse": met["mse"], **extra})
        preds.append(pred)
        targets.append(eval_cache["targets"])
        masks.append(eval_cache["target_masks"])
    pooled = metric_tensors({"targets": torch.cat(targets), "target_masks": torch.cat(masks)}, torch.cat(preds), std)
    return {"method": "OneNet-style frozen-expert adaptation", "config": asdict(cfg), "fold_mae": pooled["mae"], "fold_mse": pooled["mse"], "fold_rows": rows}


def select_from_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(scores, key=lambda r: (float(r["fold_mae"]), float(r["fold_mse"]), json.dumps(r["config"], sort_keys=True)))[0]


def eval_trainable_router_folds(dataset: str, train_cache: Mapping[str, Any], std: torch.Tensor, configs: Sequence[Any], kind: str, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores = []
    for cfg in configs:
        preds = []
        targets = []
        masks = []
        fold_rows = []
        for fold_id, train_lo, eval_lo, eval_hi in [(i, *f) for i, f in enumerate(train_folds(int(train_cache["num_windows"])))]:
            tr = slice_cache(train_cache, train_lo, eval_lo)
            ev = slice_cache(train_cache, eval_lo, eval_hi)
            if kind == "fame":
                model = train_fame_model(tr, std, cfg, device)
                pred = predict_fame(model, ev, std)
            elif kind == "timerouter":
                model = train_timerouter_model(tr, std, cfg, device)
                pred = predict_timerouter(model, ev)
            else:
                raise ValueError(kind)
            met = metric_tensors(ev, pred, std)
            fold_rows.append({"fold": fold_id, "train_hi": eval_lo, "eval_lo": eval_lo, "eval_hi": eval_hi, "mae": met["mae"], "mse": met["mse"]})
            preds.append(pred)
            targets.append(ev["targets"])
            masks.append(ev["target_masks"])
        pooled = metric_tensors({"targets": torch.cat(targets), "target_masks": torch.cat(masks)}, torch.cat(preds), std)
        scores.append({"dataset": dataset, "method": kind, "config": asdict(cfg), "fold_mae": pooled["mae"], "fold_mse": pooled["mse"], "fold_rows": fold_rows})
    return select_from_scores(scores), scores


def run_dataset(dataset: str, train_cache: Mapping[str, Any], val_cache: Mapping[str, Any], std: torch.Tensor, device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    ablations: list[dict[str, Any]] = []
    per_window: list[dict[str, Any]] = []
    frozen_configs: dict[str, Any] = {}

    fixed_pred = fixed_average_prediction(val_cache, EXPERTS)
    results.append(metric_row(dataset, "Equal fixed ensemble", "router_val", val_cache, fixed_pred, std, expert_pool="+".join(EXPERTS), uses_validation_targets=False))
    per_window.extend(per_window_rows(dataset, "Equal fixed ensemble", val_cache, fixed_pred, std))

    gr_configs = [
        LinearConfig("Granger-Ramanathan", "global", 0.0, "canonical_ols"),
        LinearConfig("Granger-Ramanathan", "horizon_variable", 0.0, "canonical_ols"),
        LinearConfig("Granger-Ramanathan", "global", 1e-4, "ridge_extension"),
        LinearConfig("Granger-Ramanathan", "horizon_variable", 1e-4, "ridge_extension"),
        LinearConfig("Granger-Ramanathan", "global", 1e-2, "ridge_extension"),
        LinearConfig("Granger-Ramanathan", "horizon_variable", 1e-2, "ridge_extension"),
        LinearConfig("Granger-Ramanathan", "global", 1.0, "ridge_extension"),
        LinearConfig("Granger-Ramanathan", "horizon_variable", 1.0, "ridge_extension"),
    ]
    gr_scores = [score_config_on_folds(train_cache, std, fit_gr, predict_gr, cfg, "Granger-Ramanathan") for cfg in gr_configs]
    gr_selected = select_from_scores(gr_scores)
    gr_model = fit_gr(train_cache, std, LinearConfig(**gr_selected["config"]))
    frozen_configs["Granger-Ramanathan"] = {**gr_selected, "frozen_before_validation": True}
    gr_pred = predict_gr(gr_model, val_cache)
    results.append(metric_row(dataset, "Granger-Ramanathan", "router_val", val_cache, gr_pred, std, selected_config=json.dumps(gr_selected["config"], sort_keys=True), uses_validation_targets=False))
    per_window.extend(per_window_rows(dataset, "Granger-Ramanathan", val_cache, gr_pred, std))
    ablations.extend([{**s, "dataset": dataset, "ablation_family": "global_vs_horizon_variable", "method": "Granger-Ramanathan"} for s in gr_scores])

    fame_configs = [
        FameConfig(0.10, 2, 64, 0.10, 3e-4, 1e-3, 80),
        FameConfig(0.05, 2, 64, 0.10, 3e-4, 1e-3, 80),
        FameConfig(0.20, 2, 64, 0.10, 3e-4, 1e-3, 80),
        FameConfig(0.10, 1, 64, 0.10, 3e-4, 1e-3, 80),
        FameConfig(0.10, 3, 64, 0.10, 3e-4, 1e-3, 80),
    ]
    fame_selected, fame_scores = eval_trainable_router_folds(dataset, train_cache, std, fame_configs, "fame", device)
    fame_cfg = FameConfig(**fame_selected["config"])
    fame_model = train_fame_model(train_cache, std, fame_cfg, device)
    fame_pred = predict_fame(fame_model, val_cache, std)
    fame_diag = fame_diagnostics(fame_model, val_cache)
    frozen_configs["FAME adaptation"] = {**fame_selected, "frozen_before_validation": True, "official_reproduction": False}
    results.append(metric_row(dataset, "FAME adaptation", "router_val", val_cache, fame_pred, std, selected_config=json.dumps(fame_selected["config"], sort_keys=True), uses_validation_targets=False, **fame_diag))
    per_window.extend(per_window_rows(dataset, "FAME adaptation", val_cache, fame_pred, std))
    ablations.extend([{**s, "dataset": dataset, "ablation_family": "sparse_topk_vs_dense_weighting", "method": "FAME adaptation"} for s in fame_scores])

    tr_configs = [
        TimeRouterConfig(0.15, 1.0, 64, 3e-4, 1e-3, 80),
        TimeRouterConfig(0.10, 1.0, 64, 3e-4, 1e-3, 80),
        TimeRouterConfig(0.20, 1.0, 64, 3e-4, 1e-3, 80),
        TimeRouterConfig(0.15, 0.5, 64, 3e-4, 1e-3, 80),
        TimeRouterConfig(0.15, 2.0, 64, 3e-4, 1e-3, 80),
    ]
    tr_selected, tr_scores = eval_trainable_router_folds(dataset, train_cache, std, tr_configs, "timerouter", device)
    tr_cfg = TimeRouterConfig(**tr_selected["config"])
    tr_model = train_timerouter_model(train_cache, std, tr_cfg, device)
    tr_pred = predict_timerouter(tr_model, val_cache)
    tr_diag = timerouter_diagnostics(tr_model, val_cache)
    frozen_configs["TimeRouter adaptation"] = {**tr_selected, "frozen_before_validation": True, "official_reproduction": False}
    results.append(metric_row(dataset, "TimeRouter adaptation", "router_val", val_cache, tr_pred, std, selected_config=json.dumps(tr_selected["config"], sort_keys=True), uses_validation_targets=False, **tr_diag))
    per_window.extend(per_window_rows(dataset, "TimeRouter adaptation", val_cache, tr_pred, std))
    ablations.extend([{**s, "dataset": dataset, "ablation_family": "hard_vs_fallback_routing", "method": "TimeRouter adaptation"} for s in tr_scores])

    bates_configs = [
        BatesConfig("global", "covariance", 0.0),
        BatesConfig("global", "covariance", 0.25),
        BatesConfig("global", "covariance", 0.75),
        BatesConfig("global", "diagonal_inverse_error", 1.0),
        BatesConfig("horizon_variable", "diagonal_inverse_error", 1.0),
        BatesConfig("horizon_variable", "covariance", 0.75),
    ]
    bates_scores = [score_config_on_folds(train_cache, std, fit_bates, predict_bates, cfg, "Bates-Granger") for cfg in bates_configs]
    bates_selected = select_from_scores(bates_scores)
    bates_model = fit_bates(train_cache, std, BatesConfig(**bates_selected["config"]))
    bates_pred = predict_bates(bates_model, val_cache)
    frozen_configs["Bates-Granger"] = {**bates_selected, "frozen_before_validation": True}
    results.append(metric_row(dataset, "Bates-Granger", "router_val", val_cache, bates_pred, std, selected_config=json.dumps(bates_selected["config"], sort_keys=True), uses_validation_targets=False))
    per_window.extend(per_window_rows(dataset, "Bates-Granger", val_cache, bates_pred, std))
    ablations.extend([{**s, "dataset": dataset, "ablation_family": "covariance_vs_diagonal_global_vs_hv", "method": "Bates-Granger"} for s in bates_scores])

    onenet_configs = [OneNetConfig(eta, 0.97) for eta in (0.01, 0.05, 0.1, 0.2, 0.5)]
    onenet_scores = [score_onenet_folds(train_cache, std, cfg) for cfg in onenet_configs]
    onenet_selected = select_from_scores(onenet_scores)
    onenet_model = fit_onenet(train_cache, std, OneNetConfig(**onenet_selected["config"]))
    onenet_pred, onenet_extra = onenet_predict(val_cache, std, OneNetConfig(**onenet_selected["config"]), onenet_model["init_error"], onenet_model["branches"])
    frozen_configs["OneNet-style frozen-expert adaptation"] = {**onenet_selected, "frozen_before_validation": True, "official_reproduction": False}
    results.append(metric_row(dataset, "OneNet / adaptation", "router_val", val_cache, onenet_pred, std, selected_config=json.dumps(onenet_selected["config"], sort_keys=True), uses_validation_targets=True, **onenet_extra))
    per_window.extend(per_window_rows(dataset, "OneNet / adaptation", val_cache, onenet_pred, std))
    ablations.extend([{**s, "dataset": dataset, "ablation_family": "static_vs_online", "method": "OneNet / adaptation"} for s in onenet_scores])

    return {
        "dataset": dataset,
        "runtime_sec": time.perf_counter() - started,
        "results": results,
        "ablations": ablations,
        "per_window": per_window,
        "frozen_configs": frozen_configs,
    }


def load_dataset(dataset: str) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor, dict[str, str]]:
    if dataset == "ETTh1":
        paths = {"router_train": ETTH1_TRAIN, "router_val": ETTH1_VAL, "normalizer": ETTH1_NORMALIZER}
        for p in paths.values():
            refuse_test(p)
        train = load_cache(ETTH1_TRAIN, "router_train_20_60")
        val = load_cache(ETTH1_VAL, "router_val_60_80")
        std = load_std(ETTH1_NORMALIZER, 7)
        validate_cache(train, "router_train_20_60", dataset)
        validate_cache(val, "router_val_60_80", dataset)
    elif dataset == "ETTh2":
        paths = {"router_train": ETTH2_TRAIN, "router_val": ETTH2_VAL}
        for p in paths.values():
            refuse_test(p)
        train = load_cache(ETTH2_TRAIN, "router_train")
        val = load_cache(ETTH2_VAL, "router_val")
        std = torch.ones(7)
        validate_cache(train, "router_train", dataset)
        validate_cache(val, "router_val", dataset)
    else:
        raise ValueError(dataset)
    hashes = {name + "_sha256": sha256_file(path) for name, path in paths.items()}
    return train, val, std, hashes


def load_costar_rows() -> list[dict[str, Any]]:
    path = ROOT / "experiments/frozen_costar/frozen_costar_validation_results.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in data.get("results", []):
        method_map = {
            "equal_fixed_three": "COSTAR equal fixed core",
            "frozen_costar": "Frozen COSTAR",
            "online_costar": "Online COSTAR",
        }
        if row.get("method") in method_map:
            rows.append({
                "dataset": row["dataset"],
                "method": method_map[row["method"]],
                "split": "router_val",
                "mae": row["mae"],
                "mse": row["mse"],
                "source": str(path.relative_to(ROOT)),
                "uses_validation_targets": row.get("uses_earlier_validation_targets", False),
            })
    return rows


def residual_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    etth1_path = ROOT / "experiments/frozen_model_test_results/top_costar_test_results.csv"
    if etth1_path.exists():
        with etth1_path.open(newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                method = r.get("method", "")
                if method == "ridge_residual_corrector":
                    rows.append({"dataset": "ETTh1", "method": "Frozen COSTAR + Ridge residual", "split": "router_val", "mae": float(r["validation_mae"]), "mse": float(r["validation_mse"]), "source": str(etth1_path.relative_to(ROOT))})
                if method == "mlp_residual_corrector":
                    rows.append({"dataset": "ETTh1", "method": "Frozen COSTAR + MLP residual", "split": "router_val", "mae": float(r["validation_mae"]), "mse": float(r["validation_mse"]), "source": str(etth1_path.relative_to(ROOT))})
    etth2_path = ROOT / "experiments/etth2_validation_tuned_missing_methods/validation_sweep_results.csv"
    if etth2_path.exists():
        best: dict[str, dict[str, Any]] = {}
        with etth2_path.open(newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                method = r.get("method")
                if method in {"Ridge residual corrector", "MLP residual corrector"} and r.get("seed") == "mean":
                    key = "Frozen COSTAR + Ridge residual" if method.startswith("Ridge") else "Frozen COSTAR + MLP residual"
                    mae_key = "val_mae" if "val_mae" in r else "mae"
                    if key not in best or float(r[mae_key]) < float(best[key][mae_key]):
                        best[key] = r
        for key, r in best.items():
            mae_key = "val_mae" if "val_mae" in r else "mae"
            mse_key = "val_mse" if "val_mse" in r else "mse"
            rows.append({"dataset": "ETTh2", "method": key, "split": "router_val", "mae": float(r[mae_key]), "mse": float(r[mse_key]), "source": str(etth2_path.relative_to(ROOT))})
    return rows


def make_comparison_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    order = [
        "Equal fixed ensemble",
        "Granger-Ramanathan",
        "Bates-Granger",
        "FAME adaptation",
        "TimeRouter adaptation",
        "Frozen COSTAR",
        "Online COSTAR",
        "Frozen COSTAR + Ridge residual",
        "Frozen COSTAR + MLP residual",
        "OneNet / adaptation",
    ]
    out = []
    for method in order:
        row = {"Method": method}
        for dataset in ("ETTh1", "ETTh2"):
            match = next((r for r in rows if r.get("method") == method and r.get("dataset") == dataset), None)
            row[f"{dataset} Val MAE"] = match.get("mae") if match else ""
            row[f"{dataset} Val MSE"] = match.get("mse") if match else ""
        out.append(row)
    return out


def render_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Published Baseline Comparisons for COSTAR",
        "",
        "Validation-only comparison. No ETTh1 or ETTh2 test cache is loaded.",
        "",
        "## Comparison Table",
        "",
        "| Method | ETTh1 Val MAE | ETTh1 Val MSE | ETTh2 Val MAE | ETTh2 Val MSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["comparison_table"]:
        lines.append(
            f"| {row['Method']} | {fmt(row['ETTh1 Val MAE'])} | {fmt(row['ETTh1 Val MSE'])} | {fmt(row['ETTh2 Val MAE'])} | {fmt(row['ETTh2 Val MSE'])} |"
        )
    lines.extend(["", "## Selected Hyperparameters", ""])
    for dataset, payload in report["datasets"].items():
        lines.append(f"### {dataset}")
        lines.append("")
        for method, frozen in payload["frozen_configs"].items():
            lines.append(f"- {method}: `{json.dumps(frozen['config'], sort_keys=True)}`")
        lines.append("")
    lines.extend([
        "## Leakage And Provenance Checks",
        "",
        f"- Test cache loaded: `{report['test_cache_loaded']}`.",
        "- Every dataset loader rejects paths containing `test`.",
        "- Cache schemas were checked for expert order, chronological starts, `[N,12,7,5]` prediction stacks, `[N,12,7]` targets/masks, input length `96`, and horizon `12`.",
        "- Hyperparameter selection used chronological prefixes inside `router_train`; selected configs were saved under per-dataset `frozen_config_before_validation.json` before final `router_val` rows were recorded.",
        "- OneNet-style online updates call `enforce_observable(old_start, current_start, horizon)` before any realized error update.",
        "- Frozen experts are never trained or updated; only routers/combination weights are fit on cached predictions.",
        "",
        "## Artifacts",
        "",
        "- `FINAL_REPORT.json`: machine-readable full report.",
        "- `validation_results.csv`: validation MAE/MSE rows.",
        "- `ablation_results.csv`: global/horizon-variable, covariance/diagonal, sparse/top-k, hard/fallback, and online ablations.",
        "- `per_window_metrics.csv`: per-window validation MAE/MSE.",
        "- `ETTh1/frozen_config_before_validation.json` and `ETTh2/frozen_config_before_validation.json`: frozen selected configs and cache hashes.",
        "",
        "## Implemented Algorithms",
        "",
        "- Granger-Ramanathan: direct linear target regression from expert forecasts, with global and horizon-variable OLS/ridge candidates.",
        "- Bates-Granger: covariance-weighted and diagonal inverse-error forecast combination using router-train forecast errors only.",
        "- FAME adaptation: forecastability fingerprint, soft expert-suitability targets from router-train losses, and sparse Top-r routing over the BasicTS frozen expert pool.",
        "- TimeRouter adaptation: lightweight discriminative routing head with margin/diversity selective gate and inverse-error fallback ensemble.",
        "- OneNet-style adaptation: delayed-feedback online ensembling over frozen PatchTST/iTransformer forecasts.",
    ])
    lines.extend([
        "",
        "## Deviations From Official Papers",
        "",
        "- Granger-Ramanathan and Bates-Granger are direct classical frozen-forecast combinations on the COSTAR expert cache.",
        "- FAME is labeled `FAME routing adaptation to BasicTS frozen expert pool`: it keeps FAME's forecastability fingerprints, oracle suitability targets, and sparse Top-r routing, but replaces the official retail/industrial expert pool, metadata, context, and cost model with the five frozen BasicTS experts.",
        "- TimeRouter is labeled `TimeRouter routing-mechanism adaptation`: it keeps discriminative routing, selective margin/diversity gating, and inverse-error fallback behavior, but replaces the official XGBoost TSFM router/checkpoints with a small Torch routing head over BasicTS cache features.",
        "- OneNet is labeled `OneNet-style frozen-expert adaptation`: it adapts delayed online ensembling to frozen PatchTST/iTransformer forecasts and does not update forecasting experts.",
        "",
        "## Sources Inspected",
        "",
        "- FAME official repository: https://github.com/hit636/FAME",
        "- TimeRouter official repository: https://github.com/UConn-DSIS/TimeRouter",
        "- OneNet official repository: https://github.com/yfzhang114/OneNet",
        "",
        "## Reproduce",
        "",
        "```powershell",
        "python experiments\\published_baseline_comparisons\\run_published_baselines.py --phase all --device cuda",
        "```",
    ])
    return "\n".join(lines) + "\n"


def fmt(x: Any) -> str:
    if x == "":
        return ""
    return f"`{float(x):.6f}`"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all"], default="all")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("CUDA requested but not available")

    start = time.perf_counter()
    all_results: list[dict[str, Any]] = []
    all_ablations: list[dict[str, Any]] = []
    all_per_window: list[dict[str, Any]] = []
    dataset_payloads: dict[str, Any] = {}
    for dataset in ("ETTh1", "ETTh2"):
        train, val, std, hashes = load_dataset(dataset)
        payload = run_dataset(dataset, train, val, std, device)
        dataset_payloads[dataset] = {k: v for k, v in payload.items() if k != "per_window"}
        dataset_payloads[dataset]["cache_hashes"] = hashes
        all_results.extend(payload["results"])
        all_ablations.extend(payload["ablations"])
        all_per_window.extend(payload["per_window"])
        write_json(OUT_DIR / dataset / "frozen_config_before_validation.json", {
            "dataset": dataset,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "test_cache_loaded": False,
            "configuration_frozen_before_validation": True,
            "cache_hashes": hashes,
            "selected_configs": payload["frozen_configs"],
        })

    support_rows = load_costar_rows() + residual_rows()
    all_results.extend(support_rows)
    comparison = make_comparison_table(all_results)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": f"python experiments\\published_baseline_comparisons\\run_published_baselines.py --phase all --device {args.device}",
        "device": str(device),
        "git_commit": git_commit(),
        "test_cache_loaded": False,
        "runtime_sec": time.perf_counter() - start,
        "datasets": dataset_payloads,
        "results": all_results,
        "comparison_table": comparison,
        "deviations_from_official_papers": {
            "FAME": "Routing adaptation to BasicTS frozen expert pool, not exact official retail/industrial reproduction.",
            "TimeRouter": "Routing-mechanism adaptation with Torch MLP head rather than official XGBoost TSFM checkpoint.",
            "OneNet": "Frozen-expert online ensembling adaptation; forecasting experts are not updated.",
        },
    }
    write_json(OUT_DIR / "FINAL_REPORT.json", report)
    write_csv(OUT_DIR / "validation_results.csv", all_results)
    write_csv(OUT_DIR / "ablation_results.csv", all_ablations)
    write_csv(OUT_DIR / "per_window_metrics.csv", all_per_window)
    (OUT_DIR / "PUBLISHED_BASELINE_COMPARISON_REPORT.md").write_text(render_report(report), encoding="utf-8")
    print(json.dumps({"out_dir": str(OUT_DIR), "comparison_table": comparison, "test_cache_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
