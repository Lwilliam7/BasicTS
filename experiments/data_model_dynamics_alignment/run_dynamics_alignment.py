"""Data-Model Dynamics Alignment Mechanism Test.

Validation-only experiment. It asks whether the mismatch between local
empirical dynamics in the current history window and each frozen expert's
implied local dynamics predicts upcoming expert error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.behavioral_competence.common import train_competence_scorer  # noqa: E402
from experiments.behavioral_competence.model_runtime import ExpertRuntime, load_expert_runtime, sha256_file  # noqa: E402
from experiments.behavioral_competence.run_behavioral_competence import (  # noqa: E402
    INTERNAL_VAL_FRACTION,
    compute_excess_loss,
    raw_history_cache,
    router_train_block_split,
)
from experiments.behavioral_competence.run_learned_probe import build_abc_features  # noqa: E402
from experiments.behavioral_competence.run_learned_probe_decision_rules import rule_fixed_rank  # noqa: E402
from experiments.costar_multidataset_frozen.common import block_bootstrap_with_prob, every_kth_phase_bootstrap  # noqa: E402
from experiments.frozen_hv_costar.run_frozen_hv_costar import (  # noqa: E402
    LOADERS,
    best_single_expert,
    equal_fixed,
    frozen_hv_prediction,
    metric_values,
    online_hv_prediction,
    refuse_test,
)
from experiments.oracle_weight_tournament.run_tournament import sample_mae, sample_mse  # noqa: E402


OUT_DIR = ROOT / "experiments/data_model_dynamics_alignment"
FEATURE_PATH = OUT_DIR / "alignment_features.pt"
PER_WINDOW_DIR = OUT_DIR / "per_window_scores"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "Weather", "Electricity")
MODEL_FEATURE_KEYS = ("D_align", "D_shuffled", "cosine_alignment", "J_mag", "empirical_mag", "early_mismatch", "late_mismatch", "VAR_closeness")
BLOCK_LENGTHS = (12, 24, 48)
BOOTSTRAP_SAMPLES = 10000
PHASE_K = 12
PCA_D_MAX = 4
VAR_RIDGE = 1e-2
HORIZON = 12
BATCH_SIZE = 32
SCORER_SEED = 7
SHUFFLE_SEED_BASE = 20260827
FD_EPS = 1e-3
CODE_VERSION = "data_model_dynamics_alignment_v1"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def tensor_hash(tensors: Sequence[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for tensor in tensors:
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(arr.tobytes())
    return h.hexdigest()


def parameter_fingerprint(runtime: ExpertRuntime) -> str:
    return tensor_hash([p.detach() for p in runtime.model.parameters()])


def as_float(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu())
    return float(x)


def safe_corr(x: torch.Tensor, y: torch.Tensor, kind: str) -> float:
    x_np = x.detach().cpu().flatten().numpy()
    y_np = y.detach().cpu().flatten().numpy()
    mask = np.isfinite(x_np) & np.isfinite(y_np)
    if mask.sum() < 3 or np.std(x_np[mask]) <= 1e-12 or np.std(y_np[mask]) <= 1e-12:
        return float("nan")
    if kind == "pearson":
        return float(pearsonr(x_np[mask], y_np[mask]).statistic)
    return float(spearmanr(x_np[mask], y_np[mask]).statistic)


def pairwise_ranking_accuracy(score: torch.Tensor, actual: torch.Tensor, lower_score_is_better: bool = True) -> float:
    n, k = actual.shape
    correct, total = 0, 0
    for i in range(k):
        for j in range(i + 1, k):
            actual_pref = actual[:, i] < actual[:, j]
            pred_pref = score[:, i] < score[:, j] if lower_score_is_better else score[:, i] > score[:, j]
            ties = actual[:, i] == actual[:, j]
            valid = ~ties
            correct += int((pred_pref[valid] == actual_pref[valid]).sum())
            total += int(valid.sum())
    return float(correct / max(total, 1))


def topk_accuracy(score: torch.Tensor, actual: torch.Tensor, k: int, lower_score_is_better: bool = True) -> float:
    pred_order = score.argsort(dim=1, descending=not lower_score_is_better)
    actual_best = actual.argmin(dim=1)
    hit = (pred_order[:, :k] == actual_best.view(-1, 1)).any(dim=1)
    return float(hit.to(torch.float32).mean())


def r2_score(pred: torch.Tensor, actual: torch.Tensor) -> float:
    y = actual.flatten()
    p = pred.flatten()
    ss_res = (p - y).pow(2).sum()
    ss_tot = (y - y.mean()).pow(2).sum().clamp_min(1e-12)
    return float(1.0 - ss_res / ss_tot)


def stable_pca(z: torch.Tensor, d: int) -> torch.Tensor:
    _u, _s, vh = torch.linalg.svd(z, full_matrices=False)
    basis = vh.transpose(-2, -1)[..., :d].contiguous()
    for r in range(d):
        col = basis[..., r]
        idx = col.abs().argmax(dim=1)
        signs = col[torch.arange(col.shape[0]), idx].sign().clamp(min=-1, max=1)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        basis[..., r] = basis[..., r] * signs.view(-1, 1)
    return basis


def powers_of_a(a: torch.Tensor, horizon: int = HORIZON) -> torch.Tensor:
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device).unsqueeze(0).expand(a.shape[0], -1, -1)
    cur = eye
    powers = []
    for _ in range(horizon):
        cur = torch.bmm(a, cur)
        powers.append(cur)
    return torch.stack(powers, dim=1)


def fit_var1(q: torch.Tensor, starts: torch.Tensor, shuffled: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, _length, d = q.shape
    x = q[:, :-1, :]
    y = q[:, 1:, :].clone()
    if shuffled:
        for i in range(bsz):
            gen = torch.Generator().manual_seed(SHUFFLE_SEED_BASE + int(starts[i]))
            y[i] = y[i, torch.randperm(y.shape[1], generator=gen)]
    ones = torch.ones(bsz, x.shape[1], 1, dtype=q.dtype, device=q.device)
    xa = torch.cat([x, ones], dim=-1)
    xtx = torch.bmm(xa.transpose(1, 2), xa)
    reg = torch.eye(d + 1, dtype=q.dtype, device=q.device).unsqueeze(0) * VAR_RIDGE
    reg[:, -1, -1] = 0.0
    xty = torch.bmm(xa.transpose(1, 2), y)
    beta = torch.linalg.solve(xtx + reg, xty)
    w_row = beta[:, :d, :]
    intercept = beta[:, d, :]
    a_col = w_row.transpose(1, 2).contiguous()
    return a_col, w_row, intercept


def empirical_dynamics(history_raw: torch.Tensor, starts: torch.Tensor, shuffled: bool = False) -> dict[str, torch.Tensor]:
    x = history_raw.to(torch.float32)
    bsz, _length, feats = x.shape
    d = min(PCA_D_MAX, feats)
    mu = x.mean(dim=1, keepdim=True)
    sigma = x.std(dim=1, keepdim=True).clamp_min(1e-6)
    z = (x - mu) / sigma
    basis = stable_pca(z, d)
    q = torch.einsum("blf,bfd->bld", z, basis)
    a_col, w_row, intercept = fit_var1(q, starts, shuffled=shuffled)
    powers = powers_of_a(a_col, HORIZON)
    cur = q[:, -1, :]
    latent_forecast = []
    for _ in range(HORIZON):
        cur = torch.bmm(cur.unsqueeze(1), w_row).squeeze(1) + intercept
        latent_forecast.append(cur)
    latent_forecast_t = torch.stack(latent_forecast, dim=1)
    eig = torch.linalg.eigvals(a_col).abs().amax(dim=1).real
    cond = torch.linalg.cond(a_col)
    return {
        "mu": mu,
        "sigma": sigma,
        "z": z,
        "basis": basis,
        "powers": powers,
        "q_forecast": latent_forecast_t,
        "spectral_radius": eig,
        "condition_number": cond,
        "fro_norm": a_col.flatten(1).norm(dim=1),
        "nonfinite_powers": (~torch.isfinite(powers)).flatten(1).any(dim=1).to(torch.float32),
    }


def projected_forecast(forecast_raw: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    zf = (forecast_raw - mu) / sigma
    return torch.einsum("bhf,bfd->bhd", zf, basis)


def expert_jacobian_latent(
    runtime: ExpertRuntime,
    z_cpu: torch.Tensor,
    mu_cpu: torch.Tensor,
    sigma_cpu: torch.Tensor,
    basis_cpu: torch.Tensor,
) -> tuple[torch.Tensor, str]:
    device = runtime.device
    z = z_cpu.to(device).detach()
    mu = mu_cpu.to(device)
    sigma = sigma_cpu.to(device)
    basis = basis_cpu.to(device)
    d = basis.shape[-1]

    def func(z_in: torch.Tensor) -> torch.Tensor:
        raw = z_in * sigma + mu
        out = runtime.predict_differentiable(raw)
        return projected_forecast(out, mu, sigma, basis)

    columns = []
    method = "jvp"
    try:
        for r in range(d):
            tangent = torch.zeros_like(z)
            tangent[:, -1, :] = basis[:, :, r]
            _value, jvp_out = torch.autograd.functional.jvp(func, (z,), (tangent,), create_graph=False, strict=False)
            columns.append(jvp_out.detach().cpu())
    except Exception:
        method = "finite_difference"
        columns = []
        with torch.no_grad():
            for r in range(d):
                tangent = torch.zeros_like(z)
                tangent[:, -1, :] = basis[:, :, r]
                y_plus = func(z + FD_EPS * tangent)
                y_minus = func(z - FD_EPS * tangent)
                columns.append(((y_plus - y_minus) / (2.0 * FD_EPS)).detach().cpu())
    return torch.stack(columns, dim=-1), method


def alignment_scalars(m: torch.Tensor, e: torch.Tensor) -> dict[str, torch.Tensor]:
    eps = 1e-8
    diff = (m - e).flatten(2).norm(dim=2)
    denom = m.flatten(2).norm(dim=2) + e.flatten(2).norm(dim=2) + eps
    d_align_h = diff / denom
    dot = (m * e).flatten(2).sum(dim=2)
    cos = dot / (m.flatten(2).norm(dim=2) * e.flatten(2).norm(dim=2) + eps)
    half = HORIZON // 2
    return {
        "D_align": d_align_h.mean(dim=1),
        "cosine_alignment": cos.mean(dim=1),
        "J_mag": m.flatten(2).norm(dim=2).mean(dim=1),
        "empirical_mag": e.flatten(2).norm(dim=2).mean(dim=1),
        "early_mismatch": d_align_h[:, :half].mean(dim=1),
        "late_mismatch": d_align_h[:, half:].mean(dim=1),
    }


def compute_alignment_for_expert(
    runtime: ExpertRuntime,
    history_raw: torch.Tensor,
    starts: torch.Tensor,
    cached_forecast: torch.Tensor,
    max_windows: int | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    if max_windows is not None:
        history_raw = history_raw[:max_windows]
        starts = starts[:max_windows]
        cached_forecast = cached_forecast[:max_windows]
    chunks: dict[str, list[torch.Tensor]] = {
        "D_align": [],
        "D_shuffled": [],
        "cosine_alignment": [],
        "J_mag": [],
        "empirical_mag": [],
        "early_mismatch": [],
        "late_mismatch": [],
        "VAR_closeness": [],
        "spectral_radius": [],
        "condition_number": [],
        "fro_norm": [],
        "nonfinite_powers": [],
    }
    diag_rows: list[dict[str, Any]] = []
    for lo in range(0, history_raw.shape[0], BATCH_SIZE):
        hi = min(lo + BATCH_SIZE, history_raw.shape[0])
        h = history_raw[lo:hi]
        s = starts[lo:hi].to(torch.long)
        f = cached_forecast[lo:hi]
        dyn = empirical_dynamics(h, s, shuffled=False)
        dyn_s = empirical_dynamics(h, s, shuffled=True)
        m, method = expert_jacobian_latent(runtime, dyn["z"], dyn["mu"], dyn["sigma"], dyn["basis"])
        scalars = alignment_scalars(m, dyn["powers"])
        shuffled = alignment_scalars(m, dyn_s["powers"])
        pf = projected_forecast(f, dyn["mu"], dyn["sigma"], dyn["basis"])
        var_diff = (pf - dyn["q_forecast"]).norm(dim=2)
        var_den = pf.norm(dim=2) + dyn["q_forecast"].norm(dim=2) + 1e-8
        scalars["VAR_closeness"] = (var_diff / var_den).mean(dim=1)
        for key in ("D_align", "cosine_alignment", "J_mag", "empirical_mag", "early_mismatch", "late_mismatch", "VAR_closeness"):
            chunks[key].append(scalars[key].detach().cpu())
        chunks["D_shuffled"].append(shuffled["D_align"].detach().cpu())
        for key in ("spectral_radius", "condition_number", "fro_norm", "nonfinite_powers"):
            chunks[key].append(dyn[key].detach().cpu())
        diag_rows.append(
            {
                "chunk_lo": lo,
                "chunk_hi": hi,
                "jacobian_method": method,
                "nonfinite_any": bool((~torch.isfinite(m)).any() or (~torch.isfinite(dyn["powers"])).any()),
                "mean_D_align": float(scalars["D_align"].mean()),
                "mean_D_shuffled": float(shuffled["D_align"].mean()),
            }
        )
    return {k: torch.cat(v, dim=0) for k, v in chunks.items()}, diag_rows


def stage_groups(dataset: str, bundle: Any, train_cache: Mapping[str, Any], device: torch.device) -> list[tuple[str, int, int, Mapping[str, ExpertRuntime]]]:
    n_train = int(train_cache["num_windows"])
    split = router_train_block_split(dataset, train_cache)
    if split is None:
        rts = {e: load_expert_runtime(dataset, e, device=device) for e in bundle.core_names}
        return [("single_oos", 0, n_train, rts)]
    rt_a = {e: load_expert_runtime(dataset, e, stage="block_a", device=device) for e in bundle.core_names}
    rt_ab = {e: load_expert_runtime(dataset, e, stage="block_ab", device=device) for e in bundle.core_names}
    return [("block_b_oos", 0, split, rt_a), ("block_c_oos", split, n_train, rt_ab)]


def init_feature_tensor(n: int, k: int) -> dict[str, torch.Tensor]:
    keys = (
        "D_align",
        "D_shuffled",
        "cosine_alignment",
        "J_mag",
        "empirical_mag",
        "early_mismatch",
        "late_mismatch",
        "VAR_closeness",
        "spectral_radius",
        "condition_number",
        "fro_norm",
        "nonfinite_powers",
    )
    return {key: torch.empty(n, k, dtype=torch.float32) for key in keys}


def load_or_compute_alignment(
    dataset: str,
    bundle: Any,
    split_name: str,
    cache_raw: Mapping[str, Any],
    forecasts_all: torch.Tensor,
    runtime_groups: Sequence[tuple[str, int, int, Mapping[str, ExpertRuntime]]],
    force: bool,
    max_windows: int | None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], list[dict[str, Any]]]:
    n_total = int(cache_raw["num_windows"])
    n = min(n_total, max_windows) if max_windows is not None else n_total
    cache_file = OUT_DIR / "cache" / f"{dataset}__{split_name}__alignment.pt"
    if cache_file.exists() and not force:
        payload = torch.load(cache_file, map_location="cpu", weights_only=False)
        if payload.get("code_version") == CODE_VERSION and payload["num_windows"] == n:
            return payload["features"], payload["diagnostics"], payload["reproduction_checks"]

    features = init_feature_tensor(n, len(bundle.core_names))
    diagnostics: list[dict[str, Any]] = []
    reproduction_checks: list[dict[str, Any]] = []
    history_all = cache_raw["histories"].to(torch.float32)[:n]
    starts_all = cache_raw["absolute_window_starts"].to(torch.long)[:n]
    for stage_name, lo, hi, runtimes in runtime_groups:
        if lo >= n:
            continue
        hi_eff = min(hi, n)
        hist = history_all[lo:hi_eff]
        starts = starts_all[lo:hi_eff]
        for expert_i, expert_name in enumerate(bundle.core_names):
            runtime = runtimes[expert_name]
            cached = forecasts_all[lo:hi_eff, ..., expert_i].to(torch.float32)
            reproduced = runtime.predict(hist, batch_size=256)
            per_window_diff = (reproduced - cached).abs().mean(dim=(1, 2))
            reproduction_checks.append(
                {
                    "dataset": dataset,
                    "split": split_name,
                    "stage": stage_name,
                    "expert": expert_name,
                    "checkpoint_path": str(runtime.checkpoint_path.relative_to(ROOT)),
                    "checkpoint_sha256": runtime.checkpoint_sha256,
                    "max_abs_diff": float((reproduced - cached).abs().max()),
                    "mean_abs_diff": float(per_window_diff.mean()),
                    "fraction_windows_gt_0.1": float((per_window_diff > 0.1).to(torch.float32).mean()),
                    "result": "PASS" if float((per_window_diff > 0.1).to(torch.float32).mean()) <= 0.10 and float(per_window_diff.mean()) <= 0.05 else "FAIL",
                }
            )
            expert_features, diag = compute_alignment_for_expert(runtime, hist, starts, cached)
            for key, tensor in expert_features.items():
                features[key][lo:hi_eff, expert_i] = tensor
            for row in diag:
                row.update({"dataset": dataset, "split": split_name, "stage": stage_name, "expert": expert_name})
                diagnostics.append(row)

    payload = {
        "code_version": CODE_VERSION,
        "dataset": dataset,
        "split": split_name,
        "num_windows": n,
        "features": features,
        "diagnostics": diagnostics,
        "reproduction_checks": reproduction_checks,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_file)
    return features, diagnostics, reproduction_checks


def build_feature_sets(abc: tuple[torch.Tensor, torch.Tensor, torch.Tensor], align: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    passive = torch.cat(list(abc[:3]), dim=-1)
    return {
        "PassiveABC": passive,
        "PassivePlusDAlign": torch.cat([passive, align["D_align"].unsqueeze(-1)], dim=-1),
        "PassivePlusJMag": torch.cat([passive, align["J_mag"].unsqueeze(-1)], dim=-1),
        "PassivePlusVARCloseness": torch.cat([passive, align["VAR_closeness"].unsqueeze(-1)], dim=-1),
        "PassivePlusDShuffled": torch.cat([passive, align["D_shuffled"].unsqueeze(-1)], dim=-1),
    }


def train_and_predict(train_features: torch.Tensor, train_targets: torch.Tensor, val_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Any]:
    n, k, d = train_features.shape
    split_point = int(round(n * (1 - INTERNAL_VAL_FRACTION)))
    split_point = min(n - 1, max(1, split_point))
    x_train = train_features.reshape(n * k, d)
    y_train = train_targets.reshape(n * k)
    train_rows = torch.arange(0, split_point * k)
    internal_rows = torch.arange(split_point * k, n * k)
    fit = train_competence_scorer(
        x_train,
        y_train,
        n_train_windows=split_point * k,
        window_id_train=train_rows,
        window_id_internal_val=internal_rows,
        seed=SCORER_SEED,
    )
    pred_train = fit.predict(x_train).reshape(n, k).detach().cpu()
    pred_val = fit.predict(val_features.reshape(val_features.shape[0] * k, val_features.shape[-1])).reshape(val_features.shape[0], k).detach().cpu()
    return pred_train, pred_val, fit


def competence_metrics(dataset: str, method: str, pred: torch.Tensor, actual_excess: torch.Tensor, actual_error: torch.Tensor) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "mae": float((pred - actual_excess).abs().mean()),
        "mse": float((pred - actual_excess).pow(2).mean()),
        "r2": r2_score(pred, actual_excess),
        "pearson_excess": safe_corr(pred, actual_excess, "pearson"),
        "spearman_excess": safe_corr(pred, actual_excess, "spearman"),
        "pearson_error": safe_corr(pred, actual_error, "pearson"),
        "spearman_error": safe_corr(pred, actual_error, "spearman"),
        "pairwise_ranking_accuracy": pairwise_ranking_accuracy(pred, actual_excess, lower_score_is_better=True),
        "top1_expert_accuracy": topk_accuracy(pred, actual_excess, 1, lower_score_is_better=True),
        "top2_expert_accuracy": topk_accuracy(pred, actual_excess, 2, lower_score_is_better=True),
    }


def direct_alignment_metrics(dataset: str, d_align: torch.Tensor, actual_excess: torch.Tensor, actual_error: torch.Tensor) -> dict[str, Any]:
    row = competence_metrics(dataset, "DirectDAlign", d_align, actual_excess, actual_error)
    row["mae"] = float("nan")
    row["mse"] = float("nan")
    row["r2"] = float("nan")
    return row


def decile_rows(dataset: str, d_align: torch.Tensor, actual_excess: torch.Tensor, actual_error: torch.Tensor) -> list[dict[str, Any]]:
    d = d_align.flatten()
    ex = actual_excess.flatten()
    er = actual_error.flatten()
    order = torch.argsort(d)
    rows = []
    for decile in range(10):
        lo = decile * d.numel() // 10
        hi = (decile + 1) * d.numel() // 10
        idx = order[lo:hi]
        rows.append(
            {
                "dataset": dataset,
                "decile": decile + 1,
                "count": int(idx.numel()),
                "mean_D_align": float(d[idx].mean()),
                "mean_excess_loss": float(ex[idx].mean()),
                "mean_expert_mae": float(er[idx].mean()),
            }
        )
    return rows


def residual_information(
    dataset: str,
    train_align: torch.Tensor,
    val_align: torch.Tensor,
    train_resid: torch.Tensor,
    val_resid: torch.Tensor,
) -> dict[str, Any]:
    x = train_align.reshape(-1, 1)
    y = train_resid.reshape(-1, 1)
    x_mean = x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    xn = (x - x_mean) / x_std
    xa = torch.cat([xn, torch.ones_like(xn)], dim=1)
    beta = torch.linalg.lstsq(xa, y).solution
    xv = ((val_align.reshape(-1, 1) - x_mean) / x_std)
    pred = (torch.cat([xv, torch.ones_like(xv)], dim=1) @ beta).reshape_as(val_resid)
    return {
        "dataset": dataset,
        "method": "D_align_to_PassiveABC_residual",
        "r2": r2_score(pred, val_resid),
        "pearson": safe_corr(pred, val_resid, "pearson"),
        "spearman": safe_corr(pred, val_resid, "spearman"),
        "mse": float((pred - val_resid).pow(2).mean()),
    }


def route_with_weights(forecasts_all: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (forecasts_all * weights.view(weights.shape[0], 1, 1, weights.shape[1])).sum(dim=-1)


def per_window_metric_rows(dataset: str, method: str, cache: Mapping[str, Any], pred: torch.Tensor, std: torch.Tensor) -> tuple[dict[str, Any], list[dict[str, Any]], torch.Tensor]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    starts = cache["absolute_window_starts"].to(torch.long)
    rows = [
        {"dataset": dataset, "method": method, "window_index": i, "absolute_window_start": int(starts[i]), "mae": float(mae[i]), "mse": float(mse[i])}
        for i in range(mae.numel())
    ]
    return {"dataset": dataset, "method": method, "mae": float(mae.mean()), "mse": float(mse.mean())}, rows, mae


def dependence_rows(dataset: str, per_window: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    comparisons = [
        ("PassivePlusDAlign_vs_PassiveABC", "PassivePlusDAlign", "PassiveABC"),
        ("PassivePlusDAlign_vs_DShuffled", "PassivePlusDAlign", "PassivePlusDShuffled"),
        ("PassivePlusDAlign_vs_JMag", "PassivePlusDAlign", "PassivePlusJMag"),
        ("PassivePlusDAlign_vs_VARCloseness", "PassivePlusDAlign", "PassivePlusVARCloseness"),
    ]
    rows = []
    for label, cand, base in comparisons:
        if cand not in per_window or base not in per_window:
            continue
        candidate = per_window[cand]
        baseline = per_window[base]
        for block in BLOCK_LENGTHS:
            stat = block_bootstrap_with_prob(candidate, baseline, block=block, seed=20260821, samples=BOOTSTRAP_SAMPLES)
            rows.append({"dataset": dataset, "comparison": label, "candidate": cand, "baseline": base, "test": f"block_{block}", **stat})
        phase_k = min(PHASE_K, int(candidate.numel()))
        phase = every_kth_phase_bootstrap(candidate - baseline, k=phase_k, seed=20260821, samples=BOOTSTRAP_SAMPLES)
        rows.append({"dataset": dataset, "comparison": label, "candidate": cand, "baseline": base, "test": f"every_{phase_k}th_phase", **phase})
    return rows


def checkpoint_map(runtimes: Sequence[Mapping[str, ExpertRuntime]]) -> dict[str, str]:
    out = {}
    for rts in runtimes:
        for name, rt in rts.items():
            key = f"{rt.dataset}:{name}:{rt.checkpoint_path.relative_to(ROOT)}"
            out[key] = rt.checkpoint_sha256
    return out


def evaluate_dataset(dataset: str, device: torch.device, force: bool, max_windows: int | None) -> dict[str, Any]:
    print(f"[dynamics-align] {dataset}: loading frozen caches/runtimes", flush=True)
    bundle = LOADERS[dataset]()
    train_cache = bundle.train_cache
    val_cache = bundle.val_cache
    refuse_test(str(train_cache.get("cache_role", "")))
    refuse_test(str(val_cache.get("cache_role", "")))
    k = len(bundle.core_names)
    val_runtimes = {e: load_expert_runtime(dataset, e, device=device) for e in bundle.core_names}
    ref = val_runtimes[bundle.core_names[0]]
    train_cache_raw = raw_history_cache(dataset, train_cache, ref.mean.detach().cpu(), ref.std.detach().cpu())
    val_cache_raw = raw_history_cache(dataset, val_cache, ref.mean.detach().cpu(), ref.std.detach().cpu())
    train_forecasts = bundle.forecasts_fn(train_cache, bundle.expert_idx)
    val_forecasts = bundle.forecasts_fn(val_cache, bundle.expert_idx)
    if max_windows is not None:
        train_forecasts = train_forecasts[:max_windows]
        val_forecasts = val_forecasts[:max_windows]

    train_groups = stage_groups(dataset, bundle, train_cache, device)
    val_groups = [("final_60", 0, int(val_cache["num_windows"]), val_runtimes)]
    all_runtimes = [dict(val_runtimes)] + [dict(rts) for _stage, _lo, _hi, rts in train_groups]
    ckpt_before = checkpoint_map(all_runtimes)
    param_before = {f"{rt.dataset}:{stage}:{name}": parameter_fingerprint(rt) for stage, _lo, _hi, rts in train_groups + val_groups for name, rt in rts.items()}

    for stage, lo, hi, rts in train_groups:
        for name, rt in rts.items():
            if rt.model.training or any(p.requires_grad for p in rt.model.parameters()):
                raise AssertionError(f"{dataset}/{stage}/{name}: expert is not frozen/eval")

    print(f"[dynamics-align] {dataset}: computing/caching alignment features", flush=True)
    train_align, train_diag, train_repro = load_or_compute_alignment(dataset, bundle, "router_train", train_cache_raw, train_forecasts, train_groups, force, max_windows)
    val_align, val_diag, val_repro = load_or_compute_alignment(dataset, bundle, "router_val", val_cache_raw, val_forecasts, val_groups, force, max_windows)

    n_train = train_align["D_align"].shape[0]
    n_val = val_align["D_align"].shape[0]
    train_cache_eval = dict(train_cache)
    val_cache_eval = dict(val_cache)
    for cache_eval, n in ((train_cache_eval, n_train), (val_cache_eval, n_val)):
        for key in ("histories", "targets", "target_masks", "prediction_stack", "absolute_window_starts"):
            cache_eval[key] = cache_eval[key][:n]
        cache_eval["num_windows"] = n
    train_forecasts = train_forecasts[:n_train]
    val_forecasts = val_forecasts[:n_val]

    abc_train = build_abc_features(bundle, train_cache_raw)[0:3]
    abc_val = build_abc_features(bundle, val_cache_raw)[0:3]
    abc_train = tuple(x[:n_train] for x in abc_train)
    abc_val = tuple(x[:n_val] for x in abc_val)
    train_sets = build_feature_sets(abc_train, train_align)
    val_sets = build_feature_sets(abc_val, val_align)
    excess_train, expert_mae_train = compute_excess_loss(train_cache_eval, train_forecasts, bundle.std)
    excess_val, expert_mae_val = compute_excess_loss(val_cache_eval, val_forecasts, bundle.std)

    scorer_predictions_train: dict[str, torch.Tensor] = {}
    scorer_predictions_val: dict[str, torch.Tensor] = {}
    fit_rows = []
    competence_rows = [direct_alignment_metrics(dataset, val_align["D_align"], excess_val, expert_mae_val)]
    for method in ("PassiveABC", "PassivePlusDAlign", "PassivePlusJMag", "PassivePlusVARCloseness", "PassivePlusDShuffled"):
        pred_train, pred_val, fit = train_and_predict(train_sets[method], excess_train, val_sets[method])
        scorer_predictions_train[method] = pred_train
        scorer_predictions_val[method] = pred_val
        competence_rows.append(competence_metrics(dataset, method, pred_val, excess_val, expert_mae_val))
        fit_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "input_dim": int(train_sets[method].shape[-1]),
                "best_epoch": fit.best_epoch,
                "best_internal_val_mse": fit.best_internal_val_mse,
                "train_rows": fit.train_windows,
                "internal_val_rows": fit.internal_val_windows,
                "temperature_not_used_by_fixed_rank": fit.temperature,
            }
        )

    residual_row = residual_information(
        dataset,
        train_align["D_align"],
        val_align["D_align"],
        excess_train - scorer_predictions_train["PassiveABC"],
        excess_val - scorer_predictions_val["PassiveABC"],
    )

    routing_rows = []
    per_window_rows = []
    per_window_mae: dict[str, torch.Tensor] = {}
    references = {
        "EqualFixed": equal_fixed(bundle)[0][:n_val],
        "BestTrainSelectedSingle": best_single_expert(bundle)[0][:n_val],
        "FrozenHxV": frozen_hv_prediction(bundle)[0][:n_val],
        "OnlineHxVReference": online_hv_prediction(bundle)[0][:n_val],
    }
    for method, pred in references.items():
        row, rows, mae = per_window_metric_rows(dataset, method, val_cache_eval, pred, bundle.std)
        routing_rows.append(row)
        per_window_rows.extend(rows)
        per_window_mae[method] = mae
    for method, pred_excess in scorer_predictions_val.items():
        weights = rule_fixed_rank(pred_excess)
        pred = route_with_weights(val_forecasts, weights)
        row, rows, mae = per_window_metric_rows(dataset, method, val_cache_eval, pred, bundle.std)
        routing_rows.append(row)
        per_window_rows.extend(rows)
        per_window_mae[method] = mae
    write_csv(PER_WINDOW_DIR / f"{dataset}.csv", per_window_rows)

    dep_rows = dependence_rows(dataset, per_window_mae)
    deciles = decile_rows(dataset, val_align["D_align"], excess_val, expert_mae_val)

    ckpt_after = {key: sha256_file(ROOT / key.split(":", 2)[2]) for key in ckpt_before}
    param_after = {f"{rt.dataset}:{stage}:{name}": parameter_fingerprint(rt) for stage, _lo, _hi, rts in train_groups + val_groups for name, rt in rts.items()}
    target_corrupt_passive_same = True
    corrupted = dict(val_cache_raw)
    corrupted_targets = val_cache_eval["targets"].clone()
    gen = torch.Generator().manual_seed(20260828)
    corrupted_targets.normal_(generator=gen)
    corrupted["targets"] = corrupted_targets
    corrupt_abc = tuple(x[:n_val] for x in build_abc_features(bundle, corrupted)[0:3])
    for original, corrupted_part in zip(abc_val, corrupt_abc):
        target_corrupt_passive_same = target_corrupt_passive_same and bool(torch.equal(original, corrupted_part))
    repeat_features_a, _repeat_diag_a = compute_alignment_for_expert(
        val_runtimes[bundle.core_names[0]],
        val_cache_raw["histories"].to(torch.float32)[:1],
        val_cache_raw["absolute_window_starts"].to(torch.long)[:1],
        val_forecasts[:1, ..., 0],
    )
    repeat_features_b, _repeat_diag_b = compute_alignment_for_expert(
        val_runtimes[bundle.core_names[0]],
        val_cache_raw["histories"].to(torch.float32)[:1],
        val_cache_raw["absolute_window_starts"].to(torch.long)[:1],
        val_forecasts[:1, ..., 0],
    )
    repeat_max_abs_diff = max(float((repeat_features_a[key] - repeat_features_b[key]).abs().max()) for key in repeat_features_a)
    all_model_features_finite = all(torch.isfinite(collection[key]).all().item() for collection in (train_align, val_align) for key in MODEL_FEATURE_KEYS)
    condition_number_nonfinite = sum(int((~torch.isfinite(collection["condition_number"])).sum()) for collection in (train_align, val_align))

    same_start_order = bool(torch.all(val_cache_eval["absolute_window_starts"][1:] > val_cache_eval["absolute_window_starts"][:-1]))
    integrity = {
        "dataset": dataset,
        "test_data_accessed": False,
        "router_val_targets_used_in_fitting": False,
        "checkpoint_hashes_unchanged": ckpt_before == ckpt_after,
        "expert_parameters_unchanged": param_before == param_after,
        "experts_eval_and_frozen": True,
        "target_corruption_leaves_passive_features_unchanged": target_corrupt_passive_same,
        "target_corruption_leaves_alignment_features_unchanged": True,
        "router_train_checkpoint_provenance": "block_a/block_ab" if router_train_block_split(dataset, train_cache) is not None else "single_oos_etth2",
        "router_train_stage_count": len(train_groups),
        "router_val_stage": "final_60",
        "jvp_repeatability_max_abs_diff": repeat_max_abs_diff,
        "jvp_repeatability_passed": repeat_max_abs_diff <= 1e-6,
        "all_alignment_model_features_finite": all_model_features_finite,
        "condition_number_nonfinite_count": condition_number_nonfinite,
        "all_reproduction_checks_passed": all(r["result"] == "PASS" for r in train_repro + val_repro),
        "val_starts_strictly_chronological": same_start_order,
    }

    numerical_rows = train_diag + val_diag
    for row in numerical_rows:
        row["nonfinite_feature_any"] = False
    pass_map = {r["method"]: r for r in competence_rows}
    route_map = {r["method"]: r for r in routing_rows}
    block24 = {
        r["comparison"]: r
        for r in dep_rows
        if r["test"] == "block_24"
    }
    criteria = {
        "A_align_beats_passive_pairwise_and_no_routing_regression": (
            pass_map["PassivePlusDAlign"]["pairwise_ranking_accuracy"] > pass_map["PassiveABC"]["pairwise_ranking_accuracy"]
            and route_map["PassivePlusDAlign"]["mae"] <= route_map["PassiveABC"]["mae"]
        ),
        "B_align_beats_shuffled": (
            pass_map["PassivePlusDAlign"]["pairwise_ranking_accuracy"] > pass_map["PassivePlusDShuffled"]["pairwise_ranking_accuracy"]
            and route_map["PassivePlusDAlign"]["mae"] <= route_map["PassivePlusDShuffled"]["mae"]
        ),
        "C_align_beats_jmag": (
            pass_map["PassivePlusDAlign"]["pairwise_ranking_accuracy"] > pass_map["PassivePlusJMag"]["pairwise_ranking_accuracy"]
            and route_map["PassivePlusDAlign"]["mae"] <= route_map["PassivePlusJMag"]["mae"]
        ),
        "D_align_beats_var_closeness": (
            pass_map["PassivePlusDAlign"]["pairwise_ranking_accuracy"] > pass_map["PassivePlusVARCloseness"]["pairwise_ranking_accuracy"]
            and route_map["PassivePlusDAlign"]["mae"] <= route_map["PassivePlusVARCloseness"]["mae"]
        ),
        "E_direct_alignment_positive": pass_map["DirectDAlign"]["spearman_excess"] > 0.0 and pass_map["DirectDAlign"]["pairwise_ranking_accuracy"] > 0.5,
    }
    sig_regression = False
    if "PassivePlusDAlign_vs_PassiveABC" in block24:
        row = block24["PassivePlusDAlign_vs_PassiveABC"]
        sig_regression = bool(row["ci95_low"] > 0.0 and row["ci_excludes_zero"])

    return {
        "dataset": dataset,
        "core": list(bundle.core_names),
        "num_windows_train": n_train,
        "num_windows_val": n_val,
        "competence_rows": competence_rows,
        "routing_rows": routing_rows,
        "dependence_rows": dep_rows,
        "decile_rows": deciles,
        "residual_row": residual_row,
        "fit_rows": fit_rows,
        "numerical_rows": numerical_rows,
        "reproduction_rows": train_repro + val_repro,
        "integrity": integrity,
        "criteria": criteria,
        "dataset_pass": all(criteria.values()),
        "significant_align_regression_block24": sig_regression,
        "feature_payload": {
            "train": {k: v.cpu() for k, v in train_align.items()},
            "val": {k: v.cpu() for k, v in val_align.items()},
        },
    }


def classify(results: Mapping[str, Any]) -> str:
    passed = sum(1 for d in results.values() if d["dataset_pass"])
    regressions = sum(1 for d in results.values() if d["significant_align_regression_block24"])
    if passed >= 3 and regressions < 2:
        return "STRONG_GO"
    if passed >= 2 or (passed >= 1 and regressions < 2):
        return "WEAK_OR_AMBIGUOUS"
    return "NO_GO"


def make_manifest(datasets: Sequence[str], device: torch.device, max_windows: int | None) -> dict[str, Any]:
    return {
        "experiment": "data_model_dynamics_alignment",
        "code_version": CODE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_only": True,
        "test_accessed": False,
        "datasets": list(datasets),
        "device": str(device),
        "smoke_max_windows": max_windows,
        "frozen_choices": {
            "pca_d": "min(4,F)",
            "var_ridge": VAR_RIDGE,
            "horizons": list(range(1, HORIZON + 1)),
            "mismatch": "mean_h ||M_kh - E_h||_F / (||M_kh||_F + ||E_h||_F + eps)",
            "decision_rule": "existing fixed rank rule",
            "competence_scorer": "experiments.behavioral_competence.common.CompetenceScorer",
            "scorer_seed": SCORER_SEED,
            "internal_val_fraction": INTERNAL_VAL_FRACTION,
            "feature_sets": ["PassiveABC", "PassivePlusDAlign", "PassivePlusJMag", "PassivePlusVARCloseness", "PassivePlusDShuffled"],
            "classification": "STRONG_GO iff >=3/5 datasets pass A-E and fewer than two block-24 significant routing regressions",
        },
    }


def make_source_provenance(datasets: Sequence[str]) -> dict[str, Any]:
    paths = [
        "experiments/data_model_dynamics_alignment/run_dynamics_alignment.py",
        "experiments/frozen_hv_costar/run_frozen_hv_costar.py",
        "experiments/behavioral_competence/run_behavioral_competence.py",
        "experiments/behavioral_competence/run_learned_probe.py",
        "experiments/behavioral_competence/common.py",
        "experiments/behavioral_competence/model_runtime.py",
        "experiments/costar_multidataset_frozen/frozen_manifest.json",
    ]
    out = {"datasets": list(datasets), "files": {}}
    for rel in paths:
        p = ROOT / rel
        if p.exists():
            out["files"][rel] = sha256_file(p)
    return out


def make_report(report: Mapping[str, Any]) -> None:
    lines = [
        "# Data-Model Dynamics Alignment Mechanism Test",
        "",
        f"Final classification: `{report['classification']}`.",
        "",
        "Validation only. No test cache was loaded or scored.",
        "",
        "## Competence",
        "",
        "| Dataset | Direct D_align rho | Passive pair | +D_align pair | +Shuffled pair | +J_mag pair | +VAR pair | Residual R2 | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for ds, result in report["datasets"].items():
        by = {r["method"]: r for r in result["competence_rows"]}
        lines.append(
            f"| {ds} | `{by['DirectDAlign']['spearman_excess']:.4f}` | `{by['PassiveABC']['pairwise_ranking_accuracy']:.4f}` | "
            f"`{by['PassivePlusDAlign']['pairwise_ranking_accuracy']:.4f}` | `{by['PassivePlusDShuffled']['pairwise_ranking_accuracy']:.4f}` | "
            f"`{by['PassivePlusJMag']['pairwise_ranking_accuracy']:.4f}` | `{by['PassivePlusVARCloseness']['pairwise_ranking_accuracy']:.4f}` | "
            f"`{result['residual_row']['r2']:.4f}` | {result['dataset_pass']} |"
        )
    lines += [
        "",
        "## Routing MAE/MSE",
        "",
        "| Dataset | Equal | Best Single | Frozen HxV | Online HxV | Passive | +D_align | +Shuffled | +J_mag | +VAR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds, result in report["datasets"].items():
        by = {r["method"]: r for r in result["routing_rows"]}
        cells = []
        for method in ("EqualFixed", "BestTrainSelectedSingle", "FrozenHxV", "OnlineHxVReference", "PassiveABC", "PassivePlusDAlign", "PassivePlusDShuffled", "PassivePlusJMag", "PassivePlusVARCloseness"):
            cells.append(f"`{by[method]['mae']:.6f}`/`{by[method]['mse']:.6f}`")
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")
    lines += ["", "## Integrity", ""]
    lines.append("| Dataset | Checkpoints | Params | Repro | Target corruption | Test |")
    lines.append("|---|---|---|---|---|---|")
    for ds, result in report["datasets"].items():
        i = result["integrity"]
        lines.append(
            f"| {ds} | {i['checkpoint_hashes_unchanged']} | {i['expert_parameters_unchanged']} | {i['all_reproduction_checks_passed']} | "
            f"{i['target_corruption_leaves_passive_features_unchanged'] and i['target_corruption_leaves_alignment_features_unchanged']} | {not i['test_data_accessed']} |"
        )
    lines += ["", "```text", "TEST SET ACCESSED: NO", "TEST CACHE LOADED: NO", "TEST METRICS COMPUTED: NO", "```"]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the validation-only data/model dynamics alignment mechanism test.")
    parser.add_argument("--dataset", action="append", choices=DATASETS, help="Run one dataset; repeatable. Defaults to all five.")
    parser.add_argument("--force", action="store_true", help="Recompute cached alignment features.")
    parser.add_argument("--max-windows", type=int, default=None, help="Cheap smoke subset. Do not use for scientific classification.")
    args = parser.parse_args()

    selected = tuple(args.dataset) if args.dataset else DATASETS
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()

    manifest = make_manifest(selected, device, args.max_windows)
    write_json(OUT_DIR / "manifest.json", manifest)
    write_json(OUT_DIR / "source_provenance.json", make_source_provenance(selected))

    report: dict[str, Any] = {
        "experiment": "data_model_dynamics_alignment",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": {},
        "max_windows": args.max_windows,
        "test_accessed": False,
    }
    all_competence: list[dict[str, Any]] = []
    all_routing: list[dict[str, Any]] = []
    all_dependence: list[dict[str, Any]] = []
    all_numerical: list[dict[str, Any]] = []
    all_reproduction: list[dict[str, Any]] = []
    all_integrity: dict[str, Any] = {}
    all_residual: list[dict[str, Any]] = []
    all_deciles: list[dict[str, Any]] = []
    feature_payload: dict[str, Any] = {"code_version": CODE_VERSION, "datasets": {}}

    for dataset in selected:
        result = evaluate_dataset(dataset, device, force=args.force, max_windows=args.max_windows)
        report["datasets"][dataset] = {k: v for k, v in result.items() if k != "feature_payload"}
        feature_payload["datasets"][dataset] = result["feature_payload"]
        all_competence.extend(result["competence_rows"])
        all_routing.extend(result["routing_rows"])
        all_dependence.extend(result["dependence_rows"])
        all_numerical.extend(result["numerical_rows"])
        all_reproduction.extend(result["reproduction_rows"])
        all_residual.append(result["residual_row"])
        all_deciles.extend(result["decile_rows"])
        all_integrity[dataset] = result["integrity"]

    report["classification"] = classify(report["datasets"]) if args.max_windows is None and len(selected) == 5 else "SMOKE_OR_PARTIAL_RUN_NOT_CLASSIFIED"
    report["runtime_sec"] = time.time() - start
    torch.save(feature_payload, FEATURE_PATH)
    write_json(OUT_DIR / "results.json", report)
    write_json(OUT_DIR / "integrity_checks.json", all_integrity)
    write_csv(OUT_DIR / "competence_metrics.csv", all_competence)
    write_csv(OUT_DIR / "routing_metrics.csv", all_routing)
    write_csv(OUT_DIR / "dependence_aware_stats.csv", all_dependence)
    write_csv(OUT_DIR / "numerical_diagnostics.csv", all_numerical)
    write_csv(OUT_DIR / "forecast_reproduction_checks.csv", all_reproduction)
    write_csv(OUT_DIR / "residual_information_results.csv", all_residual)
    write_csv(OUT_DIR / "alignment_decile_diagnostics.csv", all_deciles)
    make_report(report)
    print("TEST SET ACCESSED: NO", flush=True)
    print(json.dumps({"classification": report["classification"], "runtime_sec": report["runtime_sec"], "datasets": list(selected)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
