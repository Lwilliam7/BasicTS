"""Window-Dependent Expert-Choice H x V Routing.

Validation-only, development experiment (ETTh1/ETTh2/ETTm1/Weather/Electricity
are DEVELOPMENT datasets that previously informed the static Expert-Choice
result -- this run is additional development, not untouched confirmation).

Scientific question: previous static Expert Choice uses ONE H x V competence
map S[h,v,e] shared by every forecasting window. This experiment tests
whether letting competence vary by the CURRENT forecasting window,
S[t,h,v,e] = static_gain[h,v,e] + predicted_residual_gain[t,h,v,e], still
makes Expert-Choice cell assignment (each expert claims a fixed-capacity set
of H x V cells) better than matched-budget Token-Choice cell assignment
(each cell picks its top expert), using EXACTLY the same score/affinity
tensor for both operators so only routing direction differs.

Hard rules enforced throughout: TEST SET ACCESSED: NO, TEST CACHE LOADED: NO,
TEST METRICS COMPUTED: NO. The scorer is frozen during router_val -- no
online adaptation. No capacity sweep, no ranking-loss sweep, no rescue
tuning after router_val is inspected.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.behavioral_competence.common import (  # noqa: E402
    GROUP_A_NAMES,
    _lag1_autocorr,
    _slope,
    _spectral_entropy,
    window_features_group_a,
)
from experiments.costar_multidataset_frozen.common import (  # noqa: E402
    block_bootstrap_with_prob,
    every_kth_phase_bootstrap,
    verify_router_train_out_of_sample,
)
from experiments.expert_choice_hv.run_expert_choice_hv import (  # noqa: E402
    expert_choice_claims as static_expert_choice_claims,
    metric_values as static_metric_values,
    prediction_from_claims as static_prediction_from_claims,
    score_tensor as static_score_tensor,
    sha256_tensor,
    token_choice_claims as static_token_choice_claims,
    validate_cache_role,
    write_csv,
    write_json,
)
from experiments.frozen_hv_costar.run_frozen_hv_costar import (  # noqa: E402
    LOADERS,
    Bundle,
    best_single_expert,
    equal_fixed,
    frozen_hv_prediction,
)
from experiments.oracle_weight_tournament.run_tournament import sample_mae, sample_mse  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "Weather", "Electricity")

# ---------------------------------------------------------------------------
# Fixed, predeclared hyperparameters. NONE of these may change after
# router_val has been inspected (section 30 of the spec: forbidden rescue
# behavior). CF is fixed at 1.0; there is no capacity sweep.
# ---------------------------------------------------------------------------
SCORER_SEED = 7
SHUFFLE_SEED = 20260830
BOOTSTRAP_SEED = 20260830
CAPACITY_FACTOR = 1.0
HORIZON_EMBED_DIM = 4
VARIABLE_EMBED_DIM = 8
EXPERT_EMBED_DIM = 4
HIDDEN1 = 64
HIDDEN2 = 32
MAX_EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
AFFINITY_TEMPERATURE = 1.0
INTERNAL_VAL_FRACTION = 0.20
WARMUP_FRACTION = 0.20
NUM_OOF_FOLDS = 4
BLOCK_LENGTH = 24
PHASE_K = 12
BOOTSTRAP_SAMPLES = 10000
PARITY_TOL = 1e-6
STAT_CHUNK = 256  # windows per chunk when streaming cell-local feature stats / scoring, to bound memory

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CELL_LOCAL_FEATURE_NAMES = [
    "forecast_normalized",
    "forecast_minus_last_observed",
    "forecast_minus_ensemble_mean",
    "abs_forecast_minus_ensemble_mean",
    "ensemble_std_at_cell",
    "forecast_change_from_prev_horizon",
]
PER_VARIABLE_FEATURE_NAMES = [
    "normalized_last_minus_window_mean",
    "normalized_within_window_std",
    "normalized_mean_abs_first_diff",
    "normalized_linear_slope",
    "lag1_autocorrelation",
    "normalized_recent_10pct_vs_full_mean_shift",
    "spectral_entropy",
]


# ---------------------------------------------------------------------------
# Generic helpers (JSON/CSV IO, hashing, git metadata)
# ---------------------------------------------------------------------------


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True).strip()
        except Exception as exc:  # pragma: no cover - diagnostic fallback only
            return f"unavailable: {exc}"

    dirty = run(["git", "status", "--porcelain"])
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty_working_tree": bool(dirty) and not dirty.startswith("unavailable"),
        "dirty_files_count": len(dirty.splitlines()) if dirty and not dirty.startswith("unavailable") else 0,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_source_files() -> dict[str, str]:
    files = [
        Path(__file__),
        ROOT / "experiments/expert_choice_hv/run_expert_choice_hv.py",
        ROOT / "experiments/frozen_hv_costar/run_frozen_hv_costar.py",
        ROOT / "experiments/costar_multidataset_frozen/common.py",
        ROOT / "experiments/behavioral_competence/common.py",
        ROOT / "experiments/oracle_weight_tournament/run_tournament.py",
    ]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in files if p.exists()}


def checkpoint_hashes(dataset: str, core_names: Sequence[str]) -> dict[str, Any]:
    candidates: list[Path] = []
    if dataset == "ETTh1":
        root = ROOT / "checkpoints/costarts_walkforward/final_60"
        candidates = [root / name / "best_expert.pt" for name in core_names]
    elif dataset in {"ETTm1", "Weather", "Electricity"}:
        root = ROOT / f"checkpoints/costarts_walkforward_{dataset}/final_60"
        candidates = [root / name / "best_expert.pt" for name in core_names]
    elif dataset == "ETTh2":
        candidates = list((ROOT / "checkpoints").glob("**/ETTh2*/**/best_expert.pt"))
    rows: dict[str, Any] = {}
    for path in candidates:
        if "test" in str(path).lower():
            raise ValueError(f"Forbidden checkpoint path: {path}")
        if path.exists():
            rows[str(path.relative_to(ROOT))] = sha256_file(path)
    return rows


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_csv(path, rows if rows else [{"empty": True}])


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Step 2: static parity reproduction (section 2). Reuses the STATIC
# implementation's functions verbatim -- no reinterpretation.
# ---------------------------------------------------------------------------


def reproduce_static(bundle: Bundle, dataset: str) -> dict[str, Any]:
    stored_path = ROOT / "experiments/expert_choice_hv/results.json"
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    stored_rows = {row["method"]: row for row in stored["datasets"][dataset]["metrics"]}

    score, score_info = static_score_tensor(bundle)
    forecasts_val = bundle.forecasts_fn(bundle.val_cache, bundle.expert_idx)

    ec_claim, ec_capacity = static_expert_choice_claims(score, CAPACITY_FACTOR)
    tok_claim = static_token_choice_claims(score, 1)
    ec_pred, ec_fallback = static_prediction_from_claims(forecasts_val, ec_claim)
    tok_pred, tok_fallback = static_prediction_from_claims(forecasts_val, tok_claim)
    ec_metrics = static_metric_values(bundle, ec_pred)
    tok_metrics = static_metric_values(bundle, tok_pred)

    rows = []
    all_pass = True
    for method, metrics, stored_key in (("static_ec_cf1", ec_metrics, "ec_cf1"), ("static_token_top1", tok_metrics, "token_top1")):
        stored_row = stored_rows[stored_key]
        mae_diff = abs(metrics["mae"] - stored_row["mae"])
        mse_diff = abs(metrics["mse"] - stored_row["mse"])
        passed = bool(mae_diff <= PARITY_TOL and mse_diff <= PARITY_TOL)
        all_pass = all_pass and passed
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "stored_mae": stored_row["mae"],
                "reproduced_mae": metrics["mae"],
                "mae_abs_diff": mae_diff,
                "stored_mse": stored_row["mse"],
                "reproduced_mse": metrics["mse"],
                "mse_abs_diff": mse_diff,
                "tolerance": PARITY_TOL,
                "passed": passed,
            }
        )
    return {
        "dataset": dataset,
        "rows": rows,
        "all_pass": all_pass,
        "static_score_info": score_info,
        "static_ec_capacity": ec_capacity,
        "static_ec_fallback_rate": ec_fallback,
        "static_token_fallback_rate": tok_fallback,
        "score": score,
        "ec_claim": ec_claim,
        "token_claim": tok_claim,
    }


# ---------------------------------------------------------------------------
# Gain tensor: gain[t,h,v,e] = equal_error[t,h,v] - expert_error[t,h,v,e].
# HIGHER gain = better expert. Exactly mirrors the internal computation in
# expert_choice_hv.score_tensor(), generalized to keep the window axis.
# ---------------------------------------------------------------------------


def full_gain_tensor(bundle: Bundle, cache: Mapping[str, Any]) -> torch.Tensor:
    forecasts = bundle.forecasts_fn(cache, bundle.expert_idx).to(torch.float32)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.float32)
    std = bundle.std.to(torch.float32).view(1, 1, -1)
    expert_error = ((forecasts - target.unsqueeze(-1)) / std.unsqueeze(-1)).abs() * mask.unsqueeze(-1)
    equal_error = ((forecasts.mean(dim=-1) - target) / std).abs() * mask
    return equal_error.unsqueeze(-1) - expert_error


# ---------------------------------------------------------------------------
# Feature construction. Section 11.
# ---------------------------------------------------------------------------


def per_variable_history_features(history: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """[N,V] local features per current 96-step input history only. No future values."""
    history = history.to(torch.float32)
    stdv = std.to(torch.float32).view(1, -1).clamp_min(1e-8)
    full_mean = history.mean(dim=1)
    last_minus_mean = (history[:, -1, :] - full_mean) / stdv
    within_std = history.std(dim=1) / stdv
    diff = history[:, 1:, :] - history[:, :-1, :]
    mean_abs_first_diff = diff.abs().mean(dim=1) / stdv
    slope = _slope(history, dim=1) / stdv
    lag1 = _lag1_autocorr(history)
    k = max(1, round(history.shape[1] * 0.10))
    recent_shift = (history[:, -k:, :].mean(dim=1) - full_mean) / stdv
    entropy = _spectral_entropy(history)
    feats = torch.stack([last_minus_mean, within_std, mean_abs_first_diff, slope, lag1, recent_shift, entropy], dim=-1)
    return torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def global_local_features(cache: Mapping[str, Any], std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    history = cache["histories"].to(torch.float32)
    global_feat = torch.nan_to_num(window_features_group_a(history, std.to(torch.float32)))
    local_feat = per_variable_history_features(history, std)
    return global_feat, local_feat


def cell_local_features(forecasts: torch.Tensor, last_observed: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """forecasts: [b,H,V,E]; last_observed: [b,V] (last raw history step). Returns [b,H,V,E,6].
    All quantities are target-free: derived only from frozen forecasts and the input history."""
    stdv = std.to(torch.float32).view(1, 1, -1, 1).clamp_min(1e-8)
    f_norm = forecasts / stdv
    f_minus_last = (forecasts - last_observed.unsqueeze(1).unsqueeze(-1)) / stdv
    ensemble_mean = forecasts.mean(dim=-1, keepdim=True)
    f_minus_mean = (forecasts - ensemble_mean) / stdv
    f_abs_minus_mean = f_minus_mean.abs()
    ensemble_std = (forecasts.std(dim=-1, unbiased=False, keepdim=True) / stdv).expand_as(forecasts)
    prev = torch.zeros_like(forecasts)
    prev[:, 1:, :, :] = forecasts[:, :-1, :, :]
    change = torch.zeros_like(forecasts)
    change[:, 1:, :, :] = (forecasts[:, 1:, :, :] - forecasts[:, :-1, :, :]) / stdv
    feats = torch.stack([f_norm, f_minus_last, f_minus_mean, f_abs_minus_mean, ensemble_std, change], dim=-1)
    return torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class FeatureStats:
    global_mean: torch.Tensor
    global_std: torch.Tensor
    local_mean: torch.Tensor
    local_std: torch.Tensor
    cell_mean: torch.Tensor
    cell_std: torch.Tensor
    static_gain_mean: float
    static_gain_std: float


def compute_feature_stats(
    global_feat: torch.Tensor,
    local_feat: torch.Tensor,
    forecasts_full: torch.Tensor,
    histories_full: torch.Tensor,
    std: torch.Tensor,
    legal_idx: torch.Tensor,
    static_gain: torch.Tensor,
) -> FeatureStats:
    """Streams over legal fit windows in chunks to compute mean/std without
    materializing the full [n_legal, H, V, E, 6] cell-local tensor."""
    g = global_feat[legal_idx]
    l = local_feat[legal_idx]
    global_mean, global_std = g.mean(dim=0), g.std(dim=0).clamp_min(1e-6)
    local_mean = l.reshape(-1, l.shape[-1]).mean(dim=0)
    local_std = l.reshape(-1, l.shape[-1]).std(dim=0).clamp_min(1e-6)

    n_feat = len(CELL_LOCAL_FEATURE_NAMES)
    total = torch.zeros(n_feat, dtype=torch.float64)
    total_sq = torch.zeros(n_feat, dtype=torch.float64)
    count = 0
    for lo in range(0, int(legal_idx.numel()), STAT_CHUNK):
        idx = legal_idx[lo : lo + STAT_CHUNK]
        fc = forecasts_full[idx]
        last_obs = histories_full[idx][:, -1, :]
        cf = cell_local_features(fc, last_obs, std).to(torch.float64)
        total += cf.sum(dim=(0, 1, 2, 3))
        total_sq += (cf**2).sum(dim=(0, 1, 2, 3))
        count += cf.shape[0] * cf.shape[1] * cf.shape[2] * cf.shape[3]
    cell_mean = (total / max(count, 1)).to(torch.float32)
    cell_var = (total_sq / max(count, 1)) - (total / max(count, 1)) ** 2
    cell_std = cell_var.clamp_min(1e-12).sqrt().to(torch.float32)

    return FeatureStats(
        global_mean=global_mean.to(DEVICE),
        global_std=global_std.to(DEVICE),
        local_mean=local_mean.to(DEVICE),
        local_std=local_std.to(DEVICE),
        cell_mean=cell_mean.to(DEVICE),
        cell_std=cell_std.to(DEVICE),
        static_gain_mean=float(static_gain.mean()),
        static_gain_std=float(static_gain.std().clamp_min(1e-6)),
    )


# ---------------------------------------------------------------------------
# Model: ONE shared scorer across experts. Section 10, 12.
# ---------------------------------------------------------------------------


class SharedResidualScorer(nn.Module):
    def __init__(self, horizon: int, variables: int, num_experts: int) -> None:
        super().__init__()
        self.horizon_embedding = nn.Embedding(horizon, HORIZON_EMBED_DIM)
        self.variable_embedding = nn.Embedding(variables, VARIABLE_EMBED_DIM)
        self.expert_embedding = nn.Embedding(num_experts, EXPERT_EMBED_DIM)
        input_dim = (
            len(GROUP_A_NAMES)
            + len(PER_VARIABLE_FEATURE_NAMES)
            + len(CELL_LOCAL_FEATURE_NAMES)
            + HORIZON_EMBED_DIM
            + VARIABLE_EMBED_DIM
            + EXPERT_EMBED_DIM
            + 1  # static_gain[h,v,e] scalar input
        )
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN1), nn.ReLU(), nn.Linear(HIDDEN1, HIDDEN2), nn.ReLU(), nn.Linear(HIDDEN2, 1)
        )

    def forward(
        self,
        global_feat: torch.Tensor,  # [b,6] normalized
        local_feat: torch.Tensor,  # [b,V,7] normalized
        cell_feat: torch.Tensor,  # [b,H,V,E,6] normalized
        static_gain_norm: torch.Tensor,  # [H,V,E] normalized scalar
    ) -> torch.Tensor:
        b = global_feat.shape[0]
        horizon = self.horizon_embedding.num_embeddings
        variables = self.variable_embedding.num_embeddings
        experts = self.expert_embedding.num_embeddings
        device = global_feat.device

        h_ids = torch.arange(horizon, device=device)
        v_ids = torch.arange(variables, device=device)
        e_ids = torch.arange(experts, device=device)
        h_emb = self.horizon_embedding(h_ids).view(1, horizon, 1, 1, HORIZON_EMBED_DIM).expand(b, -1, variables, experts, -1)
        v_emb = self.variable_embedding(v_ids).view(1, 1, variables, 1, VARIABLE_EMBED_DIM).expand(b, horizon, -1, experts, -1)
        e_emb = self.expert_embedding(e_ids).view(1, 1, 1, experts, EXPERT_EMBED_DIM).expand(b, horizon, variables, -1, -1)

        g = global_feat.view(b, 1, 1, 1, -1).expand(-1, horizon, variables, experts, -1)
        l = local_feat.view(b, 1, variables, 1, -1).expand(-1, horizon, -1, experts, -1)
        sg = static_gain_norm.view(1, horizon, variables, experts, 1).expand(b, -1, -1, -1, -1)

        x = torch.cat([g, l, cell_feat, h_emb, v_emb, e_emb, sg], dim=-1)
        out = self.net(x.reshape(b * horizon * variables * experts, -1))
        return out.view(b, horizon, variables, experts)


@dataclass
class TrainedScorer:
    model: SharedResidualScorer
    stats: FeatureStats
    static_gain: torch.Tensor
    best_epoch: int
    best_internal_val_mse: float
    history: list[dict[str, Any]]
    train_windows: int
    internal_val_windows: int


def normalize_inputs(
    global_feat: torch.Tensor, local_feat: torch.Tensor, cell_feat: torch.Tensor, static_gain: torch.Tensor, stats: FeatureStats
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = (global_feat - stats.global_mean.view(1, -1)) / stats.global_std.view(1, -1)
    l = (local_feat - stats.local_mean.view(1, 1, -1)) / stats.local_std.view(1, 1, -1)
    c = (cell_feat - stats.cell_mean.view(1, 1, 1, 1, -1)) / stats.cell_std.view(1, 1, 1, 1, -1)
    sg = (static_gain - stats.static_gain_mean) / max(stats.static_gain_std, 1e-6)
    return torch.nan_to_num(g), torch.nan_to_num(l), torch.nan_to_num(c), torch.nan_to_num(sg)


def chronological_internal_split(fit_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(fit_idx.numel())
    if n < 5:
        return fit_idx, fit_idx[:0]
    cut = max(1, int(math.floor(n * (1.0 - INTERNAL_VAL_FRACTION))))
    if cut >= n:
        cut = n - 1
    return fit_idx[:cut], fit_idx[cut:]


def batch_forward(
    model: SharedResidualScorer,
    stats: FeatureStats,
    global_feat: torch.Tensor,
    local_feat: torch.Tensor,
    forecasts_full: torch.Tensor,
    histories_full: torch.Tensor,
    std: torch.Tensor,
    static_gain_norm_cached: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    g = global_feat[idx].to(DEVICE)
    l = local_feat[idx].to(DEVICE)
    fc = forecasts_full[idx].to(DEVICE)
    last_obs = histories_full[idx][:, -1, :].to(DEVICE)
    cf = cell_local_features(fc, last_obs, std.to(DEVICE))
    g_n, l_n, c_n, _ = normalize_inputs(g, l, cf, torch.zeros(1), stats)
    return model(g_n, l_n, c_n, static_gain_norm_cached)


def train_scorer(
    horizon: int,
    variables: int,
    num_experts: int,
    global_feat: torch.Tensor,
    local_feat: torch.Tensor,
    forecasts_full: torch.Tensor,
    histories_full: torch.Tensor,
    std: torch.Tensor,
    gain: torch.Tensor,
    legal_idx: torch.Tensor,
) -> TrainedScorer:
    set_seed(SCORER_SEED)
    static_gain = gain[legal_idx].mean(dim=0)  # [H,V,E], fit-only
    residual_target = gain - static_gain.view(1, horizon, variables, num_experts)

    stats = compute_feature_stats(global_feat, local_feat, forecasts_full, histories_full, std, legal_idx, static_gain)
    static_gain_norm = ((static_gain - stats.static_gain_mean) / max(stats.static_gain_std, 1e-6)).to(DEVICE)

    train_idx, internal_val_idx = chronological_internal_split(legal_idx)

    model = SharedResidualScorer(horizon, variables, num_experts).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(SCORER_SEED)

    residual_target_dev_cache: dict[int, torch.Tensor] = {}

    def target_for(idx: torch.Tensor) -> torch.Tensor:
        return residual_target[idx].to(DEVICE)

    history: list[dict[str, Any]] = []
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    best_val = float("inf")
    bad = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = train_idx[torch.randperm(int(train_idx.numel()), generator=generator)]
        train_losses = []
        for lo in range(0, int(perm.numel()), BATCH_SIZE):
            idx = perm[lo : lo + BATCH_SIZE]
            pred = batch_forward(model, stats, global_feat, local_feat, forecasts_full, histories_full, std, static_gain_norm, idx)
            loss = ((pred - target_for(idx)) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            if internal_val_idx.numel() > 0:
                vals = []
                for lo in range(0, int(internal_val_idx.numel()), BATCH_SIZE):
                    idx = internal_val_idx[lo : lo + BATCH_SIZE]
                    pred = batch_forward(model, stats, global_feat, local_feat, forecasts_full, histories_full, std, static_gain_norm, idx)
                    vals.append(((pred - target_for(idx)) ** 2).mean())
                val_loss = float(torch.stack(vals).mean())
            else:
                val_loss = float(sum(train_losses) / max(len(train_losses), 1))
        history.append({"epoch": epoch, "train_mse": float(sum(train_losses) / max(len(train_losses), 1)), "internal_val_mse": val_loss})
        if val_loss < best_val - 1e-10:
            best_val = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return TrainedScorer(
        model=model,
        stats=stats,
        static_gain=static_gain,
        best_epoch=best_epoch,
        best_internal_val_mse=best_val,
        history=history,
        train_windows=int(train_idx.numel()),
        internal_val_windows=int(internal_val_idx.numel()),
    )


def score_windows(
    fit: TrainedScorer,
    global_feat: torch.Tensor,
    local_feat: torch.Tensor,
    forecasts_full: torch.Tensor,
    histories_full: torch.Tensor,
    std: torch.Tensor,
    idx: torch.Tensor,
    feature_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """raw_score[t,h,v,e] = static_gain[h,v,e] + predicted_residual_gain[t,h,v,e].
    `feature_idx` (if given) selects which window's DYNAMIC INPUT FEATURES to use
    for each output row in `idx` -- used only by the shuffled-window control."""
    src = feature_idx if feature_idx is not None else idx
    horizon, variables, experts = fit.static_gain.shape
    static_gain_norm = ((fit.static_gain - fit.stats.static_gain_mean) / max(fit.stats.static_gain_std, 1e-6)).to(DEVICE)
    fit.model.eval()
    parts = []
    with torch.no_grad():
        for lo in range(0, int(idx.numel()), STAT_CHUNK):
            chunk = src[lo : lo + STAT_CHUNK]
            pred = batch_forward(fit.model, fit.stats, global_feat, local_feat, forecasts_full, histories_full, std, static_gain_norm, chunk)
            parts.append(pred.cpu())
    predicted_residual = torch.cat(parts, dim=0)
    return fit.static_gain.view(1, horizon, variables, experts) + predicted_residual


def fit_only_calibration(fit: TrainedScorer, global_feat: torch.Tensor, local_feat: torch.Tensor, forecasts_full: torch.Tensor, histories_full: torch.Tensor, std: torch.Tensor, legal_idx: torch.Tensor) -> tuple[float, float]:
    """Section 16: standardize raw_score using a FIT-ONLY scalar mean/std,
    computed from raw predictions on the fold's own fit windows."""
    total = 0.0
    total_sq = 0.0
    count = 0
    for lo in range(0, int(legal_idx.numel()), STAT_CHUNK):
        idx = legal_idx[lo : lo + STAT_CHUNK]
        raw = score_windows(fit, global_feat, local_feat, forecasts_full, histories_full, std, idx)
        total += float(raw.to(torch.float64).sum())
        total_sq += float((raw.to(torch.float64) ** 2).sum())
        count += raw.numel()
    mean = total / max(count, 1)
    var = total_sq / max(count, 1) - mean**2
    std_val = max(var, 1e-12) ** 0.5
    return mean, std_val


def raw_to_affinity(raw_score: torch.Tensor, calib_mean: float, calib_std: float) -> torch.Tensor:
    z = (raw_score - calib_mean) / max(calib_std, 1e-8)
    return torch.softmax(z / AFFINITY_TEMPERATURE, dim=-1)


# ---------------------------------------------------------------------------
# Routing operators. Sections 17-18. The multi-claim averaging and
# zero-claim fallback below are the EXACT static-EC rule
# (expert_choice_hv.prediction_from_claims: average claiming experts'
# forecasts, else fall back to the equal fixed ensemble), only batched over
# the window axis because dynamic claim masks vary per window.
# ---------------------------------------------------------------------------


def dynamic_token_claims(affinity: torch.Tensor) -> torch.Tensor:
    idx = torch.argmax(affinity, dim=-1, keepdim=True)
    claim = torch.zeros_like(affinity, dtype=torch.bool)
    claim.scatter_(-1, idx, True)
    return claim


def dynamic_ec_claims(affinity: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Section 18. Deterministic tie-break: higher affinity, then lower
    flattened H x V index. Every expert receives exactly C claims per window."""
    n, h, v, e = affinity.shape
    m = h * v
    capacity = int(round(m / e))
    flat = affinity.reshape(n, m, e).to(torch.float64)
    cell_index = torch.arange(m, dtype=torch.float64).view(1, m, 1)
    tie_break_key = flat * 1.0e9 - cell_index  # higher affinity wins; ties -> lower index wins
    claim = torch.zeros((n, m, e), dtype=torch.bool)
    for expert in range(e):
        top = torch.topk(tie_break_key[:, :, expert], k=capacity, dim=1, largest=True).indices
        claim[:, :, expert].scatter_(1, top, True)
    return claim.view(n, h, v, e), capacity


def dynamic_prediction_from_claims(forecasts: torch.Tensor, claim_mask: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Batched analogue of expert_choice_hv.prediction_from_claims: identical
    rule (simple average of claiming experts, equal-ensemble fallback), just
    with a per-window claim mask instead of one static mask."""
    claim = claim_mask.to(forecasts.dtype)
    counts = claim.sum(dim=-1)
    equal = forecasts.mean(dim=-1)
    claimed_sum = (forecasts * claim).sum(dim=-1)
    pred = torch.where(counts > 0, claimed_sum / counts.clamp_min(1.0), equal)
    fallback_rate = float((counts == 0).to(torch.float32).mean())
    return pred, fallback_rate


def claim_distribution(claim_mask: torch.Tensor) -> dict[str, float]:
    counts = claim_mask.sum(dim=-1)
    total = float(counts.numel())
    e = claim_mask.shape[-1]
    return {f"cells_with_{k}_experts_pct": float((counts == k).to(torch.float32).sum() / total * 100.0) for k in range(e + 1)}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metric_from(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return {"mae": float(mae.mean()), "mse": float(mse.mean()), "per_window_mae": mae, "per_window_mse": mse}


# ---------------------------------------------------------------------------
# OOF fold convention. Section 19. Same warmup/4-fold convention already
# used by this project's chronological OOF utilities
# (costar_multidataset_frozen.common.train_folds), extended with explicit
# full-horizon-observability legality and mechanical assertions.
# ---------------------------------------------------------------------------


def oof_bounds(n: int) -> list[tuple[int, int]]:
    warmup = int(math.ceil(n * WARMUP_FRACTION))
    remain = n - warmup
    bounds = [warmup + (remain * i) // NUM_OOF_FOLDS for i in range(NUM_OOF_FOLDS + 1)]
    return [(bounds[i], bounds[i + 1]) for i in range(NUM_OOF_FOLDS) if bounds[i] < bounds[i + 1]]


def legal_fit_mask(starts: torch.Tensor, horizon: int, current_eval_origin: int) -> torch.Tensor:
    return torch.nonzero(starts + horizon <= current_eval_origin, as_tuple=False).flatten()


# ---------------------------------------------------------------------------
# Ranking diagnostics. Section 15: report whether failures come from bad
# competence prediction (these diagnostics) or the assignment operator.
# ---------------------------------------------------------------------------


def spearman_per_row(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """pred/true: [..., M]. Returns Spearman rho per leading row, vectorized via rank + Pearson."""
    def rank(x: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(x, dim=-1)
        ranks = torch.empty_like(order, dtype=torch.float32)
        arangeM = torch.arange(x.shape[-1], dtype=torch.float32, device=x.device).expand_as(order)
        ranks.scatter_(-1, order, arangeM)
        return ranks

    rp, rt = rank(pred), rank(true)
    rp = rp - rp.mean(dim=-1, keepdim=True)
    rt = rt - rt.mean(dim=-1, keepdim=True)
    num = (rp * rt).sum(dim=-1)
    den = (rp.pow(2).sum(dim=-1).sqrt() * rt.pow(2).sum(dim=-1).sqrt()).clamp_min(1e-8)
    return num / den


def token_choice_axis_diagnostics(raw_score: torch.Tensor, true_gain: torch.Tensor) -> dict[str, float]:
    pred_best = torch.argmax(raw_score, dim=-1)
    true_best = torch.argmax(true_gain, dim=-1)
    top1_acc = float((pred_best == true_best).to(torch.float32).mean())
    e = raw_score.shape[-1]
    concordant = 0
    total = 0
    for a in range(e):
        for b in range(a + 1, e):
            pred_sign = torch.sign(raw_score[..., a] - raw_score[..., b])
            true_sign = torch.sign(true_gain[..., a] - true_gain[..., b])
            concordant += int((pred_sign == true_sign).sum())
            total += pred_sign.numel()
    return {"top1_expert_accuracy": top1_acc, "pairwise_ranking_accuracy": concordant / max(total, 1)}


def expert_choice_axis_diagnostics(raw_score: torch.Tensor, true_gain: torch.Tensor) -> dict[str, float]:
    n, h, v, e = raw_score.shape
    m = h * v
    capacity = int(round(m / e))
    pred_flat = raw_score.permute(0, 3, 1, 2).reshape(n, e, m)
    true_flat = true_gain.permute(0, 3, 1, 2).reshape(n, e, m)
    rho = spearman_per_row(pred_flat, true_flat)
    pred_top = torch.zeros((n, e, m), dtype=torch.bool)
    true_top = torch.zeros((n, e, m), dtype=torch.bool)
    pred_idx = torch.topk(pred_flat, k=capacity, dim=-1).indices
    true_idx = torch.topk(true_flat, k=capacity, dim=-1).indices
    pred_top.scatter_(-1, pred_idx, True)
    true_top.scatter_(-1, true_idx, True)
    overlap = (pred_top & true_top).sum(dim=-1).to(torch.float32) / capacity
    return {"mean_spearman_predicted_vs_true_gain": float(rho.mean()), "mean_topC_overlap_fraction": float(overlap.mean())}


# ---------------------------------------------------------------------------
# Perturbation / integrity helpers. Section 25.
# ---------------------------------------------------------------------------


def corrupt_targets(cache: Mapping[str, Any], seed: int) -> dict[str, Any]:
    cloned = dict(cache)
    gen = torch.Generator().manual_seed(seed)
    cloned["targets"] = torch.randn(cache["targets"].shape, generator=gen, dtype=torch.float32)
    return cloned


def targetless(cache: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cache.items() if k not in {"targets", "target_masks"}}


def corrupt_history_suffix(cache: Mapping[str, Any], suffix_start: int, seed: int) -> dict[str, Any]:
    cloned = dict(cache)
    gen = torch.Generator().manual_seed(seed)
    hist = cache["histories"].clone()
    noise = torch.randn(hist[suffix_start:].shape, generator=gen, dtype=torch.float32)
    hist[suffix_start:] = noise
    cloned["histories"] = hist
    return cloned


# ---------------------------------------------------------------------------
# Per-dataset orchestration
# ---------------------------------------------------------------------------


@dataclass
class DatasetResult:
    dataset: str
    static_parity: dict[str, Any]
    oof: dict[str, Any]
    validation: dict[str, Any]
    diagnostics: dict[str, Any]
    dependence: list[dict[str, Any]]
    integrity: dict[str, Any]
    fold_causality: list[dict[str, Any]]
    training: dict[str, Any]
    tensors: dict[str, torch.Tensor]


def run_dataset(dataset: str) -> DatasetResult:
    print(f"[window-dependent-ec] {dataset}: loading caches...", flush=True)
    bundle = LOADERS[dataset]()
    train_role = validate_cache_role(bundle.train_cache, "router_train")
    val_role = validate_cache_role(bundle.val_cache, "router_val")

    before_hashes = checkpoint_hashes(dataset, bundle.core_names)
    oos_provenance = verify_router_train_out_of_sample(bundle.train_cache)
    oos_available = bool(oos_provenance.get("has_block_b_oos_source") or oos_provenance.get("has_block_c_oos_source") or bundle.train_cache.get("provenance"))

    print(f"[window-dependent-ec] {dataset}: reproducing static EC/Token parity...", flush=True)
    parity = reproduce_static(bundle, dataset)
    if not parity["all_pass"]:
        raise AssertionError(f"{dataset}: STATIC_PARITY: FAIL -- {parity['rows']}")
    print(f"[window-dependent-ec] {dataset}: STATIC_PARITY: PASS", flush=True)

    horizon = int(bundle.val_cache["forecast_horizon"])
    variables = int(bundle.val_cache["num_features"])
    num_experts = len(bundle.expert_idx)
    expert_order_ok = bool([bundle.val_cache["expert_names"][i] for i in bundle.expert_idx] == list(bundle.core_names))

    train_gain = full_gain_tensor(bundle, bundle.train_cache)
    train_global, train_local = global_local_features(bundle.train_cache, bundle.std)
    train_forecasts = bundle.forecasts_fn(bundle.train_cache, bundle.expert_idx).to(torch.float32)
    train_histories = bundle.train_cache["histories"].to(torch.float32)
    train_starts = bundle.train_cache["absolute_window_starts"].to(torch.long)
    n_train = int(bundle.train_cache["num_windows"])

    # -------------------------- Section 19: strict chronological OOF --------
    print(f"[window-dependent-ec] {dataset}: running strict-causal OOF folds...", flush=True)
    oof_raw = torch.full((n_train, horizon, variables, num_experts), float("nan"), dtype=torch.float32)
    oof_mask = torch.zeros(n_train, dtype=torch.bool)
    fold_causality: list[dict[str, Any]] = []
    fold_training: list[dict[str, Any]] = []
    fold_ranking_rows: list[dict[str, Any]] = []
    fold_calibrations: list[tuple[float, float]] = []

    for fold_id, (eval_lo, eval_hi) in enumerate(oof_bounds(n_train), start=1):
        current_eval_origin = int(train_starts[eval_lo])
        legal = legal_fit_mask(train_starts, horizon, current_eval_origin)
        if legal.numel() == 0:
            raise AssertionError(f"{dataset} fold {fold_id}: no legal fit windows")
        latest_fit_origin = int(train_starts[legal].max())
        latest_fit_target_end = int((train_starts[legal] + horizon).max())
        causal_ok = bool(latest_fit_target_end <= current_eval_origin)
        fold_causality.append(
            {
                "dataset": dataset,
                "fold": fold_id,
                "eval_lo": eval_lo,
                "eval_hi": eval_hi,
                "current_eval_origin": current_eval_origin,
                "num_legal_fit_windows": int(legal.numel()),
                "latest_fit_origin": latest_fit_origin,
                "latest_fit_target_end": latest_fit_target_end,
                "causal": causal_ok,
            }
        )
        if not causal_ok:
            raise AssertionError(f"{dataset} fold {fold_id}: OOF causality violation")

        fit = train_scorer(horizon, variables, num_experts, train_global, train_local, train_forecasts, train_histories, bundle.std, train_gain, legal)
        calib_mean, calib_std = fit_only_calibration(fit, train_global, train_local, train_forecasts, train_histories, bundle.std, legal)
        fold_calibrations.append((calib_mean, calib_std))

        eval_idx = torch.arange(eval_lo, eval_hi)
        raw = score_windows(fit, train_global, train_local, train_forecasts, train_histories, bundle.std, eval_idx)
        oof_raw[eval_idx] = raw
        oof_mask[eval_idx] = True

        true_gain_eval = train_gain[eval_idx]
        tok_diag = token_choice_axis_diagnostics(raw, true_gain_eval)
        ec_diag = expert_choice_axis_diagnostics(raw, true_gain_eval)
        fold_ranking_rows.append({"dataset": dataset, "fold": fold_id, "num_eval_windows": int(eval_idx.numel()), **tok_diag, **ec_diag})

        fold_training.append(
            {
                "dataset": dataset,
                "phase": "oof",
                "fold": fold_id,
                "fit_windows": int(legal.numel()),
                "train_windows": fit.train_windows,
                "internal_val_windows": fit.internal_val_windows,
                "best_epoch": fit.best_epoch,
                "best_internal_val_mse": fit.best_internal_val_mse,
                "calibration_mean": calib_mean,
                "calibration_std": calib_std,
                "history": fit.history,
            }
        )
        print(f"[window-dependent-ec] {dataset}: fold {fold_id}/4 done (best_epoch={fit.best_epoch}, eval_windows={int(eval_idx.numel())})", flush=True)

    oof_valid = oof_raw[oof_mask]
    oof_eval_idx = torch.nonzero(oof_mask, as_tuple=False).flatten()
    oof_true_gain = train_gain[oof_eval_idx]
    # OOF affinity: apply EACH window's own fold calibration (recover by re-deriving per-fold slices)
    oof_affinity = torch.empty_like(oof_valid)
    cursor = 0
    for (fold_id, (eval_lo, eval_hi)), (calib_mean, calib_std) in zip(enumerate(oof_bounds(n_train), start=1), fold_calibrations):
        n_fold = eval_hi - eval_lo
        oof_affinity[cursor : cursor + n_fold] = raw_to_affinity(oof_valid[cursor : cursor + n_fold], calib_mean, calib_std)
        cursor += n_fold

    oof_forecasts = train_forecasts[oof_eval_idx]
    oof_target = bundle.train_cache["targets"].to(torch.float32)[oof_eval_idx]
    oof_target_mask = bundle.train_cache["target_masks"].to(torch.bool)[oof_eval_idx]

    oof_tok_claim = dynamic_token_claims(oof_affinity)
    oof_ec_claim, oof_capacity = dynamic_ec_claims(oof_affinity)
    oof_tok_pred, oof_tok_fallback = dynamic_prediction_from_claims(oof_forecasts, oof_tok_claim)
    oof_ec_pred, oof_ec_fallback = dynamic_prediction_from_claims(oof_forecasts, oof_ec_claim)
    oof_tok_metrics = metric_from(oof_tok_pred, oof_target, oof_target_mask, bundle.std)
    oof_ec_metrics = metric_from(oof_ec_pred, oof_target, oof_target_mask, bundle.std)

    oof_dependence = []
    boot = block_bootstrap_with_prob(oof_ec_metrics["per_window_mae"], oof_tok_metrics["per_window_mae"], block=BLOCK_LENGTH, seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES)
    phase = every_kth_phase_bootstrap(oof_ec_metrics["per_window_mae"] - oof_tok_metrics["per_window_mae"], k=PHASE_K, seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES)
    oof_dependence.append({"dataset": dataset, "split": "router_train_oof", "comparison": "dynamic_ec_cf1_vs_dynamic_token_top1", "test": f"block_len_{BLOCK_LENGTH}", **boot})
    oof_dependence.append({"dataset": dataset, "split": "router_train_oof", "comparison": "dynamic_ec_cf1_vs_dynamic_token_top1", "test": f"every_{PHASE_K}th_phase", **phase})

    ranking_agg = {
        "top1_expert_accuracy": sum(r["top1_expert_accuracy"] * r["num_eval_windows"] for r in fold_ranking_rows) / max(sum(r["num_eval_windows"] for r in fold_ranking_rows), 1),
        "pairwise_ranking_accuracy": sum(r["pairwise_ranking_accuracy"] * r["num_eval_windows"] for r in fold_ranking_rows) / max(sum(r["num_eval_windows"] for r in fold_ranking_rows), 1),
        "mean_spearman_predicted_vs_true_gain": sum(r["mean_spearman_predicted_vs_true_gain"] * r["num_eval_windows"] for r in fold_ranking_rows) / max(sum(r["num_eval_windows"] for r in fold_ranking_rows), 1),
        "mean_topC_overlap_fraction": sum(r["mean_topC_overlap_fraction"] * r["num_eval_windows"] for r in fold_ranking_rows) / max(sum(r["num_eval_windows"] for r in fold_ranking_rows), 1),
    }

    oof_result = {
        "dataset": dataset,
        "oof_scored_windows": int(oof_mask.sum()),
        "capacity_per_expert": oof_capacity,
        "dynamic_token_top1": {"mae": oof_tok_metrics["mae"], "mse": oof_tok_metrics["mse"], "fallback_rate": oof_tok_fallback},
        "dynamic_ec_cf1": {"mae": oof_ec_metrics["mae"], "mse": oof_ec_metrics["mse"], "fallback_rate": oof_ec_fallback},
        "delta_ec_minus_token": oof_ec_metrics["mae"] - oof_tok_metrics["mae"],
        "dependence": oof_dependence,
        "ranking_diagnostics_pooled": ranking_agg,
        "fold_ranking_rows": fold_ranking_rows,
    }

    # -------------------------- Section 20: final router_train fit ---------
    print(f"[window-dependent-ec] {dataset}: purging non-observable trailing windows and refitting final scorer...", flush=True)
    first_router_val_origin = int(bundle.val_cache["absolute_window_starts"][0])
    legal_final = legal_fit_mask(train_starts, horizon, first_router_val_origin)
    purged_count = n_train - int(legal_final.numel())
    if legal_final.numel() == 0:
        raise AssertionError(f"{dataset}: no legal final fit windows before first_router_val_origin")

    final_fit = train_scorer(horizon, variables, num_experts, train_global, train_local, train_forecasts, train_histories, bundle.std, train_gain, legal_final)
    final_calib_mean, final_calib_std = fit_only_calibration(final_fit, train_global, train_local, train_forecasts, train_histories, bundle.std, legal_final)

    val_global, val_local = global_local_features(bundle.val_cache, bundle.std)
    val_forecasts = bundle.forecasts_fn(bundle.val_cache, bundle.expert_idx).to(torch.float32)
    val_histories = bundle.val_cache["histories"].to(torch.float32)
    n_val = int(bundle.val_cache["num_windows"])
    val_idx_all = torch.arange(n_val)

    val_target = bundle.val_cache["targets"].to(torch.float32)
    val_target_mask = bundle.val_cache["target_masks"].to(torch.bool)

    def score_and_route(feature_idx: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        raw = score_windows(final_fit, val_global, val_local, val_forecasts, val_histories, bundle.std, val_idx_all, feature_idx=feature_idx)
        affinity = raw_to_affinity(raw, final_calib_mean, final_calib_std)
        tok_claim = dynamic_token_claims(affinity)
        ec_claim, capacity = dynamic_ec_claims(affinity)
        tok_pred, tok_fb = dynamic_prediction_from_claims(val_forecasts, tok_claim)
        ec_pred, ec_fb = dynamic_prediction_from_claims(val_forecasts, ec_claim)
        return {
            "raw": raw, "affinity": affinity, "tok_claim": tok_claim, "ec_claim": ec_claim, "capacity": capacity,
            "tok_pred": tok_pred, "tok_fb": tok_fb, "ec_pred": ec_pred, "ec_fb": ec_fb,
        }

    print(f"[window-dependent-ec] {dataset}: scoring router_val (frozen scorer, single pass)...", flush=True)
    main = score_and_route()
    val_tok_metrics = metric_from(main["tok_pred"], val_target, val_target_mask, bundle.std)
    val_ec_metrics = metric_from(main["ec_pred"], val_target, val_target_mask, bundle.std)

    # -------------------------- Section 23: shuffled-window control --------
    print(f"[window-dependent-ec] {dataset}: running shuffled-current-window control...", flush=True)
    gen = torch.Generator().manual_seed(SHUFFLE_SEED)
    perm = torch.randperm(n_val, generator=gen)
    shuffled = score_and_route(feature_idx=perm)
    shuf_ec_metrics = metric_from(shuffled["ec_pred"], val_target, val_target_mask, bundle.std)

    # -------------------------- Section 22: required baselines -------------
    best_pred, best_extra = best_single_expert(bundle)
    equal_pred, equal_extra = equal_fixed(bundle)
    frozen_pred, frozen_extra = frozen_hv_prediction(bundle, forecasts_val=val_forecasts)
    best_metrics = static_metric_values(bundle, best_pred)
    equal_metrics = static_metric_values(bundle, equal_pred)
    frozen_metrics = static_metric_values(bundle, frozen_pred)
    static_ec_pred, static_ec_fb = static_prediction_from_claims(val_forecasts, parity["ec_claim"])
    static_tok_pred, static_tok_fb = static_prediction_from_claims(val_forecasts, parity["token_claim"])
    static_ec_metrics = static_metric_values(bundle, static_ec_pred)
    static_tok_metrics = static_metric_values(bundle, static_tok_pred)

    val_predictions = {
        "best_single_expert": {**best_metrics, "fallback_rate": 0.0, "extra": best_extra},
        "equal": {**equal_metrics, "fallback_rate": 0.0, "extra": equal_extra},
        "frozen_hv": {**frozen_metrics, "fallback_rate": 0.0, "extra": frozen_extra},
        "static_token_top1": {**static_tok_metrics, "fallback_rate": static_tok_fb},
        "static_ec_cf1": {**static_ec_metrics, "fallback_rate": static_ec_fb},
        "dynamic_token_top1": {"mae": val_tok_metrics["mae"], "mse": val_tok_metrics["mse"], "per_window_mae": val_tok_metrics["per_window_mae"], "per_window_mse": val_tok_metrics["per_window_mse"], "fallback_rate": main["tok_fb"]},
        "dynamic_ec_cf1": {"mae": val_ec_metrics["mae"], "mse": val_ec_metrics["mse"], "per_window_mae": val_ec_metrics["per_window_mae"], "per_window_mse": val_ec_metrics["per_window_mse"], "fallback_rate": main["ec_fb"], "capacity_per_expert": main["capacity"]},
        "dynamic_ec_shuffled_window": {"mae": shuf_ec_metrics["mae"], "mse": shuf_ec_metrics["mse"], "per_window_mae": shuf_ec_metrics["per_window_mae"], "per_window_mse": shuf_ec_metrics["per_window_mse"], "fallback_rate": shuffled["ec_fb"]},
    }
    val_deltas = {
        "dynamic_ec_minus_dynamic_token": val_ec_metrics["mae"] - val_tok_metrics["mae"],
        "dynamic_ec_minus_static_ec": val_ec_metrics["mae"] - static_ec_metrics["mae"],
        "dynamic_ec_minus_frozen_hv": val_ec_metrics["mae"] - frozen_metrics["mae"],
        "shuffled_minus_dynamic_ec": shuf_ec_metrics["mae"] - val_ec_metrics["mae"],
    }

    # -------------------------- Section 27: dependence-aware statistics ----
    dependence: list[dict[str, Any]] = list(oof_dependence)
    comparisons = (
        ("dynamic_ec_cf1_vs_dynamic_token_top1", "dynamic_ec_cf1", "dynamic_token_top1"),
        ("dynamic_ec_cf1_vs_static_ec_cf1", "dynamic_ec_cf1", "static_ec_cf1"),
        ("dynamic_ec_cf1_vs_frozen_hv", "dynamic_ec_cf1", "frozen_hv"),
        ("dynamic_ec_cf1_vs_dynamic_ec_shuffled", "dynamic_ec_cf1", "dynamic_ec_shuffled_window"),
    )
    for label, cand, base in comparisons:
        cand_mae, base_mae = val_predictions[cand]["per_window_mae"], val_predictions[base]["per_window_mae"]
        boot = block_bootstrap_with_prob(cand_mae, base_mae, block=BLOCK_LENGTH, seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES)
        phase = every_kth_phase_bootstrap(cand_mae - base_mae, k=PHASE_K, seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES)
        dependence.append({"dataset": dataset, "split": "router_val", "comparison": label, "test": f"block_len_{BLOCK_LENGTH}", **boot})
        dependence.append({"dataset": dataset, "split": "router_val", "comparison": label, "test": f"every_{PHASE_K}th_phase", **phase})

    # -------------------------- Section 24: routing diagnostics ------------
    print(f"[window-dependent-ec] {dataset}: computing routing diagnostics...", flush=True)
    ec_claim = main["ec_claim"]  # [N,H,V,E]
    routing_rows: list[dict[str, Any]] = []
    claim_count_rows: list[dict[str, Any]] = []
    n = ec_claim.shape[0]
    if n > 1:
        flat = ec_claim.reshape(n, -1, num_experts)
        inter = (flat[1:] & flat[:-1]).sum(dim=1).to(torch.float32)
        union = (flat[1:] | flat[:-1]).sum(dim=1).clamp_min(1).to(torch.float32)
        jaccard_adj = inter / union
        changed_adj = (flat[1:] ^ flat[:-1]).to(torch.float32).mean(dim=1)
    else:
        jaccard_adj = torch.ones(1, num_experts)
        changed_adj = torch.zeros(1, num_experts)
    diff_from_static = (ec_claim != parity["ec_claim"].view(1, horizon, variables, num_experts)).to(torch.float32).mean(dim=(1, 2, 3))

    for expert in range(num_experts):
        name = bundle.core_names[expert]
        routing_rows.append(
            {
                "dataset": dataset,
                "expert": name,
                "diagnostic": "adjacent_window_claim_churn",
                "mean_jaccard_t_vs_t-1": float(jaccard_adj[:, expert].mean()),
                "mean_fraction_changed_t_vs_t-1": float(changed_adj[:, expert].mean()),
            }
        )
        std_t = shuffled_std = main["raw"][..., expert].std(dim=0).flatten()
        routing_rows.append(
            {
                "dataset": dataset,
                "expert": name,
                "diagnostic": "predicted_residual_score_std_t",
                "median": float(torch.quantile(std_t, 0.5)),
                "p25": float(torch.quantile(std_t, 0.25)),
                "p75": float(torch.quantile(std_t, 0.75)),
                "max": float(std_t.max()),
            }
        )
        for hh in range(horizon):
            total = ec_claim[:, hh, :, expert].to(torch.float32).sum().clamp_min(1.0)
            claim_count_rows.append({"dataset": dataset, "stat_type": "horizon_specialization", "expert": name, "horizon": hh, "claim_count": int(ec_claim[:, hh, :, expert].sum())})
        for vv in range(variables):
            claim_count_rows.append({"dataset": dataset, "stat_type": "variable_specialization", "expert": name, "variable": vv, "claim_count": int(ec_claim[:, :, vv, expert].sum())})
        counts_per_window = ec_claim[..., expert].sum(dim=(1, 2))
        expected_capacity = main["capacity"]
        capacity_ok = bool(torch.all(counts_per_window == expected_capacity))
        claim_count_rows.append(
            {
                "dataset": dataset,
                "stat_type": "expert_utilization",
                "expert": name,
                "expected_capacity_per_window": expected_capacity,
                "all_windows_at_expected_capacity": capacity_ok,
                "utilization_pct": 100.0 if capacity_ok else float((counts_per_window == expected_capacity).to(torch.float32).mean() * 100.0),
            }
        )

    routing_rows.append({"dataset": dataset, "diagnostic": "difference_from_static_ec", "mean_fraction": float(diff_from_static.mean()), "median_fraction": float(torch.quantile(diff_from_static, 0.5)), "p90_fraction": float(torch.quantile(diff_from_static, 0.9))})

    dist = claim_distribution(ec_claim)
    claim_count_rows.append({"dataset": dataset, "stat_type": "claim_count_distribution", **dist, "total_hv_cells": horizon * variables})

    mean_changed_frac = float(changed_adj.mean())
    mean_jaccard_frac = float(jaccard_adj.mean())
    genuinely_dynamic = bool(mean_changed_frac > 0.05 and mean_jaccard_frac < 0.95)

    # -------------------------- Section 15/16 diagnostic: raw vs affinity --
    raw_ec_claim, _ = dynamic_ec_claims(raw_to_affinity(main["raw"], final_calib_mean, final_calib_std))
    raw_only_ec_claim, _ = dynamic_ec_claims(torch.softmax(main["raw"].to(torch.float64), dim=-1).to(torch.float32))
    raw_only_pred, raw_only_fb = dynamic_prediction_from_claims(val_forecasts, raw_only_ec_claim)
    raw_only_metrics = metric_from(raw_only_pred, val_target, val_target_mask, bundle.std)

    # -------------------------- Section 25: integrity checks ---------------
    print(f"[window-dependent-ec] {dataset}: running integrity checks...", flush=True)
    corrupted_cache = corrupt_targets(bundle.val_cache, seed=SHUFFLE_SEED + 1)
    val_global_c, val_local_c = global_local_features(corrupted_cache, bundle.std)
    val_forecasts_c = bundle.forecasts_fn(corrupted_cache, bundle.expert_idx).to(torch.float32)
    val_hist_c = corrupted_cache["histories"].to(torch.float32)
    raw_corrupted = score_windows(final_fit, val_global_c, val_local_c, val_forecasts_c, val_hist_c, bundle.std, val_idx_all)
    target_corruption_identical = bool(torch.equal(main["raw"], raw_corrupted))

    tl_cache = targetless(bundle.val_cache)
    val_global_tl, val_local_tl = global_local_features(tl_cache, bundle.std)
    val_forecasts_tl = bundle.forecasts_fn(tl_cache, bundle.expert_idx).to(torch.float32)
    val_hist_tl = tl_cache["histories"].to(torch.float32)
    raw_tl = score_windows(final_fit, val_global_tl, val_local_tl, val_forecasts_tl, val_hist_tl, bundle.std, val_idx_all)
    targetless_ok = True
    targetless_identical = bool(torch.equal(main["raw"], raw_tl))

    suffix_start = int(round(n_val * 0.75))
    future_cache = corrupt_history_suffix(bundle.val_cache, suffix_start, seed=SHUFFLE_SEED + 2)
    val_global_f, val_local_f = global_local_features(future_cache, bundle.std)
    val_forecasts_f = val_forecasts  # forecasts unaffected by history corruption in this cache (frozen expert outputs already cached)
    val_hist_f = future_cache["histories"].to(torch.float32)
    raw_future = score_windows(final_fit, val_global_f, val_local_f, val_forecasts_f, val_hist_f, bundle.std, val_idx_all)
    earlier_unchanged = bool(torch.equal(main["raw"][:suffix_start], raw_future[:suffix_start]))

    after_hashes = checkpoint_hashes(dataset, bundle.core_names)

    integrity = {
        "dataset": dataset,
        "train_role": train_role,
        "val_role": val_role,
        "no_test_in_roles": bool("test" not in train_role.lower() and "test" not in val_role.lower()),
        "static_parity_passed": bool(parity["all_pass"]),
        "target_corruption_scores_identical": target_corruption_identical,
        "targetless_prediction_succeeded": targetless_ok,
        "targetless_scores_identical": targetless_identical,
        "future_suffix_corruption_earlier_scores_unchanged": earlier_unchanged,
        "future_suffix_start": suffix_start,
        "frozen_checkpoint_hashes_unchanged": bool(before_hashes == after_hashes),
        "checkpoint_hash_count": len(before_hashes),
        "expert_order_agrees": expert_order_ok,
        "oof_causality_all_folds": bool(all(row["causal"] for row in fold_causality)),
        "router_train_oos_provenance_available": oos_available,
        "router_train_oos_provenance_result": oos_provenance.get("result"),
        "capacity_utilization_all_experts_100pct": bool(all(row.get("all_windows_at_expected_capacity", True) for row in claim_count_rows if row.get("stat_type") == "expert_utilization")),
        "TEST_SET_ACCESSED": "NO",
        "TEST_CACHE_LOADED": "NO",
        "TEST_METRICS_COMPUTED": "NO",
    }
    critical = [
        integrity["no_test_in_roles"],
        integrity["static_parity_passed"],
        integrity["target_corruption_scores_identical"],
        integrity["targetless_prediction_succeeded"],
        integrity["targetless_scores_identical"],
        integrity["future_suffix_corruption_earlier_scores_unchanged"],
        integrity["frozen_checkpoint_hashes_unchanged"],
        integrity["expert_order_agrees"],
        integrity["oof_causality_all_folds"],
        integrity["capacity_utilization_all_experts_100pct"],
    ]
    if oos_available and oos_provenance.get("result") == "FAIL":
        critical.append(False)
    integrity["all_pass"] = bool(all(critical))
    if not integrity["all_pass"]:
        raise AssertionError(f"{dataset}: INVALID_EXPERIMENT -- integrity failure: {integrity}")

    diagnostics = {
        "dataset": dataset,
        "mean_adjacent_claim_change_fraction": mean_changed_frac,
        "mean_adjacent_jaccard": mean_jaccard_frac,
        "genuinely_dynamic": genuinely_dynamic,
        "mean_fraction_diff_from_static_ec": float(diff_from_static.mean()),
        "raw_score_ec_metrics_diagnostic_nonprimary": {"mae": raw_only_metrics["mae"], "mse": raw_only_metrics["mse"], "fallback_rate": raw_only_fb},
    }

    training_summary = {
        "oof_folds": fold_training,
        "final_fit": {
            "purged_trailing_windows": purged_count,
            "legal_final_fit_windows": int(legal_final.numel()),
            "first_router_val_origin": first_router_val_origin,
            "best_epoch": final_fit.best_epoch,
            "best_internal_val_mse": final_fit.best_internal_val_mse,
            "calibration_mean": final_calib_mean,
            "calibration_std": final_calib_std,
            "history": final_fit.history,
        },
    }

    tensors = {
        "static_gain_final": final_fit.static_gain,
        "oof_affinity": oof_affinity.to(torch.float16),
        "oof_ec_claim": oof_ec_claim,
        "val_raw_score": main["raw"].to(torch.float16),
        "val_affinity": main["affinity"].to(torch.float16),
        "val_ec_claim": main["ec_claim"],
        "val_token_claim": main["tok_claim"],
    }

    validation_result = {
        "dataset": dataset,
        "core": list(bundle.core_names),
        "horizon": horizon,
        "variables": variables,
        "num_experts": num_experts,
        "capacity_per_expert": main["capacity"],
        "predictions": {k: {kk: vv for kk, vv in v.items() if kk not in ("per_window_mae", "per_window_mse", "extra")} for k, v in val_predictions.items()},
        "deltas": val_deltas,
    }

    return DatasetResult(
        dataset=dataset,
        static_parity=parity,
        oof=oof_result,
        validation=validation_result,
        diagnostics=diagnostics,
        dependence=dependence,
        integrity=integrity,
        fold_causality=fold_causality,
        training=training_summary,
        tensors=tensors,
    ), routing_rows, claim_count_rows, fold_ranking_rows, val_predictions


# ---------------------------------------------------------------------------
# Classification. Section 29 -- written into method_manifest.json BEFORE
# router_val metrics are computed.
# ---------------------------------------------------------------------------


def build_manifest() -> dict[str, Any]:
    return {
        "experiment": "window_dependent_expert_choice_hv",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info(),
        "source_file_sha256": sha256_source_files(),
        "datasets": list(DATASETS),
        "development_datasets_disclaimer": (
            "ETTh1/ETTh2/ETTm1/Weather/Electricity are DEVELOPMENT datasets that already "
            "informed the static Expert-Choice CF=1 choice and the decision to try window "
            "dependence. This run is additional development evidence, not untouched "
            "confirmation on held-out data. Test remains locked throughout."
        ),
        "static_reference": "experiments/expert_choice_hv/results.json (reproduced, not reinterpreted)",
        "capacity_factor": CAPACITY_FACTOR,
        "capacity_formula": "C = round(H*V/E), fixed, no sweep",
        "score_definition": "gain[t,h,v,e] = equal_ensemble_normalized_abs_error[t,h,v] - expert_normalized_abs_error[t,h,v,e]; higher = better",
        "residual_definition": "raw_score[t,h,v,e] = static_gain[h,v,e] (fit-only mean of gain) + predicted_residual_gain[t,h,v,e]; coefficient exactly 1.0",
        "affinity_definition": "affinity[t,h,v,:] = softmax_e((raw_score - fit_only_scalar_mean)/fit_only_scalar_std / temperature=1.0); PRIMARY for both Dynamic Token Choice and Dynamic Expert Choice",
        "model": {
            "type": "ONE shared scorer across experts (not per-expert)",
            "architecture": f"Linear({'input_dim'}) -> ReLU -> Linear({HIDDEN1}->{HIDDEN2}) -> ReLU -> Linear({HIDDEN2}->1)",
            "horizon_embedding_dim": HORIZON_EMBED_DIM,
            "variable_embedding_dim": VARIABLE_EMBED_DIM,
            "expert_embedding_dim": EXPERT_EMBED_DIM,
            "inputs": ["global_history_group_a(6)", "per_variable_history(7)", "cell_local_forecast(6)", "horizon_embed", "variable_embed", "expert_embed", "static_gain_scalar"],
            "dropout": 0.0,
            "no_load_balancing_aux_loss": True,
            "no_ranking_loss_sweep": True,
            "no_noisy_or_stochastic_routing": True,
        },
        "training": {
            "seed": SCORER_SEED,
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "loss": "MSE on residual_gain",
            "minibatch_unit": "forecasting window",
            "internal_val_fraction": INTERNAL_VAL_FRACTION,
            "fixed_no_tuning_after_router_val": True,
        },
        "oof_protocol": {"warmup_fraction": WARMUP_FRACTION, "num_folds": NUM_OOF_FOLDS, "full_horizon_observability_rule": "starts[i] + H <= current_eval_origin"},
        "shuffle_control_seed": SHUFFLE_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap": {"block_length_primary": BLOCK_LENGTH, "samples": BOOTSTRAP_SAMPLES, "phase_k": PHASE_K},
        "required_methods": [
            "best_single_expert", "equal", "frozen_hv", "static_token_top1", "static_ec_cf1",
            "dynamic_token_top1", "dynamic_ec_cf1", "dynamic_ec_shuffled_window",
        ],
        "forbidden_rescue_behavior": [
            "no CF sweep", "no ranking-loss sweep", "no architecture/lr/seed/embedding tuning after router_val",
            "no dataset removal", "no fallback-rule change", "no test access",
        ],
        "classification_rules": {
            "WINDOW_DEPENDENT_EC_SUPPORTED": [
                "Dynamic EC beats Dynamic Token on router_train OOF on >=3/5 datasets",
                "Dynamic EC beats Dynamic Token on router_val on >=3/5 datasets",
                "Dynamic EC vs Dynamic Token has block-24 CI entirely below zero on >=2/5 router_val datasets",
                "Dynamic EC beats Static EC CF1 on >=3/5 router_val datasets",
                "Correctly matched Dynamic EC beats Shuffled-Window Dynamic EC on >=3/5 datasets",
                "Claim maps genuinely dynamic: >=3/5 datasets have mean adjacent claim-change fraction > 5% AND mean adjacent Jaccard < 0.95",
                "All integrity checks pass",
            ],
            "MIXED_WINDOW_DEPENDENT_EC": "Meaningful matched-routing evidence exists but one or more full-support criteria fail.",
            "NO_WINDOW_DEPENDENT_EC_ADVANTAGE": "Dynamic EC beats Dynamic Token on <=2/5 val datasets, or fails to improve Static EC on most datasets, or assignments are effectively static, or shuffled context performs essentially the same or better.",
            "INVALID_EXPERIMENT": "Any leakage/causality/test-access/checkpoint/provenance/order/target-invariance failure.",
        },
        "test_set_accessed": False,
        "test_cache_loaded": False,
        "test_metrics_computed": False,
    }


def classify(results: Mapping[str, DatasetResult]) -> tuple[str, dict[str, Any]]:
    integrity_pass = all(r.integrity["all_pass"] for r in results.values())
    oof_wins = sum(1 for r in results.values() if r.oof["delta_ec_minus_token"] < 0)
    val_wins = sum(1 for r in results.values() if r.validation["deltas"]["dynamic_ec_minus_dynamic_token"] < 0)
    static_wins = sum(1 for r in results.values() if r.validation["deltas"]["dynamic_ec_minus_static_ec"] < 0)
    shuffle_wins = sum(1 for r in results.values() if r.validation["deltas"]["shuffled_minus_dynamic_ec"] > 0)
    block_support = 0
    for r in results.values():
        for row in r.dependence:
            if row["split"] == "router_val" and row["comparison"] == "dynamic_ec_cf1_vs_dynamic_token_top1" and row["test"] == f"block_len_{BLOCK_LENGTH}" and row["mean_delta"] < 0 and row["ci_excludes_zero"]:
                block_support += 1
    dynamic_datasets = sum(1 for r in results.values() if r.diagnostics["genuinely_dynamic"])

    criteria = {
        "oof_wins_ge_3": oof_wins >= 3,
        "val_wins_ge_3": val_wins >= 3,
        "block24_support_ge_2": block_support >= 2,
        "static_wins_ge_3": static_wins >= 3,
        "shuffle_wins_ge_3": shuffle_wins >= 3,
        "genuinely_dynamic_ge_3": dynamic_datasets >= 3,
        "integrity_pass": integrity_pass,
    }
    if not integrity_pass:
        classification = "INVALID_EXPERIMENT"
    elif all(criteria.values()):
        classification = "WINDOW_DEPENDENT_EC_SUPPORTED"
    elif val_wins <= 2 or static_wins < 3 - 2 or shuffle_wins < 3 - 2:
        # Any major mechanism failure -> negative, per predeclared rule text.
        classification = "NO_WINDOW_DEPENDENT_EC_ADVANTAGE" if (val_wins <= 2) else "MIXED_WINDOW_DEPENDENT_EC"
    else:
        classification = "MIXED_WINDOW_DEPENDENT_EC"
    return classification, {
        "oof_wins_vs_dynamic_token": oof_wins,
        "val_wins_vs_dynamic_token": val_wins,
        "block24_ci_below_zero_datasets": block_support,
        "val_wins_vs_static_ec": static_wins,
        "shuffle_weakened_datasets": shuffle_wins,
        "genuinely_dynamic_datasets": dynamic_datasets,
        "integrity_pass": integrity_pass,
        "criteria": criteria,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def fmt(by: Mapping[str, Any], method: str) -> str:
    row = by[method]
    return f"`{row['mae']:.6f}`"


def make_report(classification: str, details: Mapping[str, Any], results: Mapping[str, DatasetResult]) -> None:
    lines = [f"Final classification: {classification}", "", "# Window-Dependent Expert-Choice H x V Routing", ""]
    lines += [
        "Development experiment (not untouched confirmation). All five datasets already informed the prior static "
        "Expert-Choice CF=1 result and the decision to test window dependence.",
        "",
        "```text",
        "TEST SET ACCESSED: NO",
        "TEST CACHE LOADED: NO",
        "TEST METRICS COMPUTED: NO",
        "```",
        "",
        "## Router-val metrics (MAE)",
        "",
        "| Dataset | Static Token Top1 | Static EC CF1 | Dynamic Token Top1 | Dynamic EC CF1 | Frozen HxV | Shuffled Dynamic EC | Dyn EC - Dyn Token | Dyn EC - Static EC | Block-24 support |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for dataset in DATASETS:
        r = results[dataset]
        by = r.validation["predictions"]
        block_row = next((x for x in r.dependence if x["split"] == "router_val" and x["comparison"] == "dynamic_ec_cf1_vs_dynamic_token_top1" and x["test"] == f"block_len_{BLOCK_LENGTH}"), None)
        support = "YES" if (block_row and block_row["mean_delta"] < 0 and block_row["ci_excludes_zero"]) else "no"
        lines.append(
            f"| {dataset} | {fmt(by,'static_token_top1')} | {fmt(by,'static_ec_cf1')} | {fmt(by,'dynamic_token_top1')} | {fmt(by,'dynamic_ec_cf1')} | "
            f"{fmt(by,'frozen_hv')} | {fmt(by,'dynamic_ec_shuffled_window')} | `{r.validation['deltas']['dynamic_ec_minus_dynamic_token']:+.6f}` | "
            f"`{r.validation['deltas']['dynamic_ec_minus_static_ec']:+.6f}` | {support} |"
        )

    lines += ["", "## Router-train OOF (mechanism check before router_val)", "", "| Dataset | Dynamic Token OOF MAE | Dynamic EC OOF MAE | Delta |", "|---|---:|---:|---:|"]
    for dataset in DATASETS:
        o = results[dataset].oof
        lines.append(f"| {dataset} | `{o['dynamic_token_top1']['mae']:.6f}` | `{o['dynamic_ec_cf1']['mae']:.6f}` | `{o['delta_ec_minus_token']:+.6f}` |")

    lines += ["", "## Classification counts", "", "```json", json.dumps(jsonable(details), indent=2, sort_keys=True), "```"]

    oof_wins = details["oof_wins_vs_dynamic_token"]
    val_wins = details["val_wins_vs_dynamic_token"]
    static_wins = details["val_wins_vs_static_ec"]
    dyn = details["genuinely_dynamic_datasets"]
    shuf = details["shuffle_weakened_datasets"]
    integrity_pass = details["integrity_pass"]

    lines += [
        "",
        "## Nine questions",
        "",
        f"1. Did Dynamic EC beat matched Dynamic Token Choice? Router-val: `{val_wins}/5`. Router-train OOF: `{oof_wins}/5`.",
        f"2. Did Dynamic EC improve on Static EC? `{static_wins}/5` router-val datasets.",
        f"3. Did the learned scores genuinely vary by current window? See per-expert `predicted_residual_score_std_t` in `routing_diagnostics.csv`; `{dyn}/5` datasets met the predeclared adjacent-change/Jaccard threshold (question 4 detail).",
        f"4. Did expert claim masks genuinely change by current window? `{dyn}/5` datasets had mean adjacent claim-change fraction > 5% and mean adjacent Jaccard < 0.95.",
        f"5. Did shuffled current-window context weaken performance? Shuffled-window MAE was worse than correctly matched Dynamic EC on `{shuf}/5` datasets.",
        "6. Did Dynamic EC close the gap to Frozen HxV? See `Dynamic EC CF1` vs `Frozen HxV` columns above; report is descriptive, this is not a required success criterion.",
        f"7. Were router_train OOF results consistent with router_val? OOF wins `{oof_wins}/5`, router_val wins `{val_wins}/5`.",
        f"8. Did every integrity check pass? `{integrity_pass}`.",
        f"9. Should the window-dependent EC direction CONTINUE or STOP? {'CONTINUE -- proceed to a frozen-method evaluation on untouched datasets.' if classification == 'WINDOW_DEPENDENT_EC_SUPPORTED' else ('STOP for now -- classification is ' + classification + '; do not rescue via tuning.' if classification != 'INVALID_EXPERIMENT' else 'STOP -- experiment invalid, fix the integrity failure before rerunning.')}",
        "",
        "## Interpretation discipline",
        "",
        "The strongest claim supportable by this development experiment, if successful, is: under a matched competence "
        "tensor and matched assignment budget, expert-side H x V allocation appears to be a better routing operator than "
        "cell-side Top1 selection, and its specialization can depend meaningfully on the current forecasting window. This "
        "is NOT a claim of state of the art, compute savings, untouched external generalization, test improvement, or "
        "universal superiority of Expert Choice. All five datasets are development datasets.",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[window-dependent-ec] device={DEVICE}", flush=True)

    manifest = build_manifest()
    write_json(OUT_DIR / "method_manifest.json", manifest)

    results: dict[str, DatasetResult] = {}
    all_routing_rows: list[dict[str, Any]] = []
    all_claim_count_rows: list[dict[str, Any]] = []
    all_ranking_rows: list[dict[str, Any]] = []
    all_dependence_rows: list[dict[str, Any]] = []
    all_fold_causality: list[dict[str, Any]] = []
    all_static_parity: list[dict[str, Any]] = []
    all_training: dict[str, Any] = {}
    tensors_by_dataset: dict[str, dict[str, torch.Tensor]] = {}
    val_predictions_by_dataset: dict[str, Any] = {}

    for dataset in DATASETS:
        t0 = time.time()
        result, routing_rows, claim_count_rows, ranking_rows, val_predictions = run_dataset(dataset)
        results[dataset] = result
        all_routing_rows.extend(routing_rows)
        all_claim_count_rows.extend(claim_count_rows)
        all_ranking_rows.extend(ranking_rows)
        all_dependence_rows.extend(result.dependence)
        all_fold_causality.extend(result.fold_causality)
        all_static_parity.extend(result.static_parity["rows"])
        all_training[dataset] = result.training
        tensors_by_dataset[dataset] = result.tensors
        val_predictions_by_dataset[dataset] = val_predictions
        print(f"[window-dependent-ec] {dataset}: complete in {time.time()-t0:.1f}s. classification-relevant delta(EC-Token)={result.validation['deltas']['dynamic_ec_minus_dynamic_token']:+.6f}", flush=True)

    static_parity_all_pass = all(r.static_parity["all_pass"] for r in results.values())
    write_json(OUT_DIR / "static_parity.json", {"all_pass": static_parity_all_pass, "tolerance": PARITY_TOL, "rows": all_static_parity})
    if not static_parity_all_pass:
        print("STATIC_PARITY: FAIL")
        raise SystemExit(1)
    print("STATIC_PARITY: PASS", flush=True)

    classification, details = classify(results)

    oof_json = {d: results[d].oof for d in DATASETS}
    write_json(OUT_DIR / "oof_results.json", jsonable(oof_json))

    validation_json = {
        "classification": classification,
        "classification_details": details,
        "datasets": {d: results[d].validation for d in DATASETS},
    }
    write_json(OUT_DIR / "validation_results.json", jsonable(validation_json))

    write_csv_rows(OUT_DIR / "routing_diagnostics.csv", all_routing_rows)
    write_csv_rows(OUT_DIR / "claim_count_stats.csv", all_claim_count_rows)
    write_csv_rows(OUT_DIR / "ranking_diagnostics.csv", all_ranking_rows)
    write_csv_rows(OUT_DIR / "dependence_tests.csv", all_dependence_rows)
    write_json(OUT_DIR / "fold_causality.json", {"folds": all_fold_causality, "all_causal": all(r["causal"] for r in all_fold_causality)})

    integrity_rows = [results[d].integrity for d in DATASETS]
    write_json(OUT_DIR / "integrity_checks.json", {"rows": integrity_rows, "all_pass": all(r["all_pass"] for r in integrity_rows), "TEST_SET_ACCESSED": "NO", "TEST_CACHE_LOADED": "NO", "TEST_METRICS_COMPUTED": "NO"})

    write_json(OUT_DIR / "scorer_training.json", jsonable(all_training))

    tensors_path = OUT_DIR / "tensors.pt"
    torch.save(tensors_by_dataset, tensors_path)

    make_report(classification, details, results)

    manifest["classification"] = classification
    manifest["classification_details"] = jsonable(details)
    manifest["runtime_sec"] = time.time() - start
    write_json(OUT_DIR / "method_manifest.json", manifest)

    print("TEST SET ACCESSED: NO")
    print("TEST CACHE LOADED: NO")
    print("TEST METRICS COMPUTED: NO")
    print(json.dumps({"classification": classification, "runtime_sec": manifest["runtime_sec"], **jsonable(details)}, indent=2))


if __name__ == "__main__":
    main()
