"""Build a matched ETTh1/ETTh2 frozen-model results table.

This script does not retrain experts or tune using test feedback. It evaluates
ETTh2 analogues that can be reproduced from existing frozen ETTh2 caches and
frozen COSTAR formulas, then records ETTh1-only rows as unavailable on ETTh2.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import (  # noqa: E402
    Trial as ChronoTrial,
    chronological_online_weights,
)
from experiments.etth2_train_selected_core.run_etth2_train_selected_core_eval import (  # noqa: E402
    current_base_prediction,
    expert_indices,
    forecasts_for,
)
from experiments.oracle_weight_tournament.run_tournament import sample_mae, sample_mse, weighted_forecast  # noqa: E402


OUT_DIR = ROOT / "experiments" / "frozen_model_test_results"
TRAIN_CACHE = ROOT / "cache" / "costarts_fresh" / "ETTh2_96_12" / "router_train_cache.pt"
VAL_CACHE = ROOT / "cache" / "costarts_fresh" / "ETTh2_96_12" / "router_val_cache.pt"
TEST_CACHE = ROOT / "experiments" / "final_test_evaluation" / "generated" / "caches" / "ETTh2" / "locked_test_cache_v2.pt"

ETTH2_CORE = ("DLinear", "PatchTST", "ModernTCN")

ETTH1_ROWS = [
    ("MLP residual corrector", 0.32604682445526123, 0.2673218250274658, 0.3633176386356354),
    ("Full adaptive model", 0.3263952910900116, 0.2675091326236725, 0.3631121516227722),
    ("Expanded DLinear only", 0.32643741369247437, 0.26759278774261475, 0.3635100722312927),
    ("Ridge residual corrector", 0.32644808292388916, 0.2674521803855896, 0.36330097913742065),
    ("Expanded ModernTCN only", 0.32646796107292175, 0.26759111881256104, 0.36343517899513245),
    ("Horizon-variable hybrid", 0.3264932632446289, 0.26763829588890076, 0.36364156007766724),
    ("Chronological EMA hybrid", 0.3265482187271118, 0.26664260029792786, 0.36553388833999634),
    ("Oracle prototype residual", 0.3268287479877472, 0.2673642635345459, 0.3660282492637634),
    ("Fixed-three core", 0.3271281123161316, 0.26658302545547485, 0.36726489663124084),
    ("Dynamic fixed-three, seed 7", 0.32924923300743103, 0.27206283807754517, 0.36598527431488037),
    ("Best single", 0.3390795886516571, 0.2785514295101166, 0.37654954195022583),
]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def load_role_cache(path: Path, role: str) -> dict[str, Any]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    actual = cache.get("cache_role", cache.get("split_role"))
    if actual != role:
        raise ValueError(f"{path}: role={actual!r}, expected {role!r}")
    return cache


def metric(cache: Mapping[str, Any], pred: torch.Tensor) -> tuple[float, float]:
    std = torch.ones(int(cache["num_features"]), dtype=torch.float32)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return float(mae.mean()), float(mse.mean())


def avg_pred(cache: Mapping[str, Any], experts: Sequence[str]) -> torch.Tensor:
    idx = expert_indices(cache, experts)
    return forecasts_for(cache, idx).mean(dim=-1)


def single_pred(cache: Mapping[str, Any], expert: str) -> torch.Tensor:
    return avg_pred(cache, (expert,))


def chronological_prediction(cache: Mapping[str, Any], train_cache: Mapping[str, Any], experts: Sequence[str]) -> torch.Tensor:
    std = torch.ones(int(cache["num_features"]), dtype=torch.float32)
    idx = expert_indices(cache, experts)
    forecasts = forecasts_for(cache, idx)
    train_forecasts = forecasts_for(train_cache, idx)
    train_target = train_cache["targets"].to(torch.float32)
    train_mask = train_cache["target_masks"].to(torch.float32)
    train_err = ((train_forecasts - train_target.unsqueeze(-1)) / std.view(1, 1, -1, 1)).abs() * train_mask.unsqueeze(-1)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.float32)
    err = ((forecasts - target.unsqueeze(-1)) / std.view(1, 1, -1, 1)).abs() * mask.unsqueeze(-1)
    starts = cache["absolute_window_starts"].to(torch.long)
    online, _ = chronological_online_weights(
        starts=starts,
        expert_mae=err.mean(dim=(1, 2)),
        horizon=int(cache["forecast_horizon"]),
        trial=ChronoTrial("ema", "ema_decay0.97_temp0.1", decay=0.97, temperature=0.1),
        train_mean_mae=train_err.mean(dim=(0, 1, 2)),
        mode="ema",
    )
    static = torch.full_like(online, 1.0 / len(experts))
    weights = 0.5 * static + 0.5 * online
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return weighted_forecast(forecasts, weights)


def hv_hybrid_prediction(cache: Mapping[str, Any], train_cache: Mapping[str, Any], experts: Sequence[str]) -> torch.Tensor:
    std = torch.ones(int(cache["num_features"]), dtype=torch.float32)
    idx = expert_indices(cache, experts)
    pred, _ = current_base_prediction(cache, train_cache, idx, std)
    return pred


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train = load_role_cache(TRAIN_CACHE, "router_train")
    val = load_role_cache(VAL_CACHE, "router_val")
    test = load_role_cache(TEST_CACHE, "locked_test")
    if int(test["num_windows"]) != 2773:
        raise AssertionError("Unexpected ETTh2 test window count")

    etth2: dict[str, dict[str, Any]] = {}

    fixed = avg_pred(test, ETTH2_CORE)
    fixed_val = avg_pred(val, ETTH2_CORE)
    etth2["Fixed-three core"] = {
        "test_mae": metric(test, fixed)[0],
        "test_mse": metric(test, fixed)[1],
        "validation_mae": metric(val, fixed_val)[0],
        "expert_set": "+".join(ETTH2_CORE),
        "status": "pre_test_frozen",
        "note": "ETTh2 router-train selected fixed-three equal core.",
    }

    best = single_pred(test, "DLinear")
    best_val = single_pred(val, "DLinear")
    etth2["Best single"] = {
        "test_mae": metric(test, best)[0],
        "test_mse": metric(test, best)[1],
        "validation_mae": metric(val, best_val)[0],
        "expert_set": "DLinear",
        "status": "pre_test_frozen",
        "note": "ETTh2 best single expert.",
    }

    chrono = chronological_prediction(test, train, ETTH2_CORE)
    chrono_val = chronological_prediction(val, train, ETTH2_CORE)
    etth2["Chronological EMA hybrid"] = {
        "test_mae": metric(test, chrono)[0],
        "test_mse": metric(test, chrono)[1],
        "validation_mae": metric(val, chrono_val)[0],
        "expert_set": "+".join(ETTH2_CORE),
        "status": "pre_test_frozen",
        "note": "ETTh2 chronological EMA analogue over train-selected core.",
    }

    hv = hv_hybrid_prediction(test, train, ETTH2_CORE)
    hv_val = hv_hybrid_prediction(val, train, ETTH2_CORE)
    hv_mae, hv_mse = metric(test, hv)
    hv_val_mae, _ = metric(val, hv_val)
    for method, note in (
        ("Full adaptive model", "ETTh2 frozen full adaptive model; DLinear/ModernTCN duplicate specialists disabled."),
        ("Horizon-variable hybrid", "ETTh2 horizon-variable hybrid analogue; same prediction as full model because duplicate specialists are disabled."),
        ("Expanded DLinear only", "Not a distinct ETTh2 model: DLinear is already in the selected core, so the duplicate specialist is disabled."),
        ("Expanded ModernTCN only", "Not a distinct ETTh2 model: ModernTCN is already in the selected core, so the duplicate specialist is disabled."),
    ):
        etth2[method] = {
            "test_mae": hv_mae,
            "test_mse": hv_mse,
            "validation_mae": hv_val_mae,
            "expert_set": "+".join(ETTH2_CORE),
            "status": "pre_test_frozen",
            "note": note,
        }

    unavailable = {
        "MLP residual corrector": "No frozen ETTh2 MLP residual-corrector artifact exists in the repo.",
        "Ridge residual corrector": "No frozen ETTh2 ridge residual-corrector artifact exists in the repo.",
        "Oracle prototype residual": "No frozen ETTh2 oracle-prototype residual artifact exists in the repo.",
        "Dynamic fixed-three, seed 7": "No ETTh2 dynamic fixed-three checkpoint matching the ETTh1 method exists; ETTh2 has sequential/pair-selector artifacts instead.",
    }
    for method, reason in unavailable.items():
        etth2[method] = {
            "test_mae": "",
            "test_mse": "",
            "validation_mae": "",
            "expert_set": "",
            "status": "not_generated_no_frozen_etth2_artifact",
            "note": reason,
        }

    rows = []
    for method, etth1_mae, etth1_mse, etth1_val in ETTH1_ROWS:
        e2 = etth2[method]
        rows.append(
            {
                "method": method,
                "etth1_test_mae": etth1_mae,
                "etth1_test_mse": etth1_mse,
                "etth1_validation_mae": etth1_val,
                "etth2_test_mae": e2["test_mae"],
                "etth2_test_mse": e2["test_mse"],
                "etth2_validation_mae": e2["validation_mae"],
                "etth2_expert_set": e2["expert_set"],
                "etth2_status": e2["status"],
                "etth2_note": e2["note"],
            }
        )
    payload = {
        "created_from": str(__file__),
        "etth2_train_cache": str(TRAIN_CACHE),
        "etth2_val_cache": str(VAL_CACHE),
        "etth2_test_cache": str(TEST_CACHE),
        "rows": rows,
    }
    return rows, payload


def write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Matched ETTh1 And ETTh2 Frozen-Model Test Results",
        "",
        "ETTh2 is filled where a valid frozen ETTh2 analogue exists. Rows marked `not_generated_no_frozen_etth2_artifact` do not have an ETTh2 result because the repository does not contain a matching frozen ETTh2 artifact for that ETTh1-specific method.",
        "",
        "| Method | ETTh1 Test MAE | ETTh1 Test MSE | ETTh1 Val MAE | ETTh2 Test MAE | ETTh2 Test MSE | ETTh2 Val MAE | ETTh2 Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        def fmt(v: Any) -> str:
            return "" if v == "" else f"{float(v):.6f}"
        lines.append(
            f"| {r['method']} | {fmt(r['etth1_test_mae'])} | {fmt(r['etth1_test_mse'])} | {fmt(r['etth1_validation_mae'])} | {fmt(r['etth2_test_mae'])} | {fmt(r['etth2_test_mse'])} | {fmt(r['etth2_validation_mae'])} | {r['etth2_status']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows, payload = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "matched_etth1_etth2_results.csv", rows)
    (OUT_DIR / "MATCHED_ETTH1_ETTH2_RESULTS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(OUT_DIR / "MATCHED_ETTH1_ETTH2_RESULTS.md", rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
