"""Frozen HxV COSTAR: removes deployment-time (online) target feedback.

This is a NEW, SEPARATE experiment. It does not modify, delete, or overwrite
the existing causal/online COSTAR implementation, which remains available
and is evaluated here unmodified as the "Online HxV COSTAR" baseline.

WHAT WAS REMOVED (see AUDIT.md for the full audit): the causal walk-forward
loop inside `chronological_hv_weights()` that folds realized VALIDATION
errors into the H x V x expert EMA state after `old_start + horizon <=
current_start`. That loop is never called on router_val/test data by this
script. Instead:

    router_train expert errors (H x V x expert)
        -> aggregate (mean over router_train windows)
        -> errors_to_weights()  [same softmax/temperature rule as before]
        -> FROZEN H x V x expert weight tensor
        -> repeated, unchanged, for every router_val/test window

No validation/test target, mask, or error is read before predictions are
produced, and none is ever folded into the weights. This is verified
explicitly in Step 6 (byte-identical-under-perturbation tests A-E).

Datasets: ETTh1, ETTh2 (bespoke per-dataset modules, matching every prior
COSTAR experiment in this codebase) and ETTm1, Weather, Electricity (the
generic walk-forward family from costar_multidataset_frozen, reusing that
experiment's already-frozen expert core selection for consistency). No new
test cache is accessed -- validation only, as requested.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.etth2_train_selected_core.run_etth2_train_selected_core_eval as etth2  # noqa: E402
import experiments.train_selected_core_etth1.run_train_selected_core_eval as etth1  # noqa: E402
from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import paired_bootstrap  # noqa: E402
from experiments.costar_multidataset_frozen.common import (  # noqa: E402
    CANONICAL_DECAY as MD_DECAY,
    CANONICAL_TEMPERATURE as MD_TEMPERATURE,
    block_bootstrap_with_prob,
    every_kth_phase_bootstrap,
)
from experiments.frozen_costar.run_frozen_costar_validation import ETTH1_FROZEN, ETTH2_FROZEN, load_frozen_core  # noqa: E402
from experiments.horizon_variable_adaptive_costar.run_hv_adaptive_costar import (  # noqa: E402
    Trial as HvTrial,
    chronological_hv_weights,
    errors_to_weights,
    predict_from_hv_weights,
)
from experiments.oracle_weight_tournament.run_tournament import load_cache, load_std, sample_mae, sample_mse  # noqa: E402


OUT_DIR = ROOT / "experiments/frozen_hv_costar"
CANONICAL_DECAY = 0.95
CANONICAL_TEMPERATURE = 0.1
BLOCK_LENGTHS = (12, 24, 48)
BOOTSTRAP_SAMPLES = 10000
PHASE_K = 12
assert MD_DECAY == CANONICAL_DECAY and MD_TEMPERATURE == CANONICAL_TEMPERATURE, "decay/temperature must match the existing canonical settings"

HV_TRIAL = HvTrial("hv_ema", "frozen_hv_costar", mode="hv", rank=1, decay=CANONICAL_DECAY, temperature=CANONICAL_TEMPERATURE)


def refuse_test(path: str | Path) -> None:
    if "test" in str(path).lower():
        raise ValueError(f"Test access forbidden for this validation-only experiment: {path}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sample_mae_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, std: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"mae": sample_mae(pred, target, mask, std), "mse": sample_mse(pred, target, mask, std)}


# ---------------------------------------------------------------------------
# Dataset bundle: a uniform interface over the two cache families already
# used throughout this codebase (etth1/etth2 bespoke modules; the generic
# walk-forward family from costar_multidataset_frozen), so the frozen-vs-
# online evaluation logic below is written exactly once.
# ---------------------------------------------------------------------------


class Bundle:
    def __init__(
        self,
        dataset: str,
        train_cache: Mapping[str, Any],
        val_cache: Mapping[str, Any],
        std: torch.Tensor,
        expert_idx: Sequence[int],
        core_names: Sequence[str],
        forecasts_fn: Callable[[Mapping[str, Any], Sequence[int]], torch.Tensor],
        per_location_error_fn: Callable[[Mapping[str, Any], Sequence[int], torch.Tensor], torch.Tensor],
    ) -> None:
        self.dataset = dataset
        self.train_cache = train_cache
        self.val_cache = val_cache
        self.std = std
        self.expert_idx = list(expert_idx)
        self.core_names = list(core_names)
        self.forecasts_fn = forecasts_fn
        self.per_location_error_fn = per_location_error_fn


def load_etth1() -> Bundle:
    train_cache_path = ROOT / "cache/costarts_walkforward/router_train_20_60_cache.pt"
    val_cache_path = ROOT / "cache/costarts_walkforward/router_val_60_80_cache.pt"
    normalizer_path = ROOT / "checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt"
    for p in (train_cache_path, val_cache_path, normalizer_path, ETTH1_FROZEN):
        refuse_test(p)
    train_cache = load_cache(train_cache_path, "router_train_20_60")
    val_cache = load_cache(val_cache_path, "router_val_60_80")
    std = load_std(normalizer_path, int(val_cache["num_features"]))
    core = load_frozen_core(ETTH1_FROZEN)
    expert_idx = etth1.expert_indices(val_cache, core)
    return Bundle(
        "ETTh1", train_cache, val_cache, std, expert_idx, core,
        forecasts_fn=etth1.selected_forecasts,
        per_location_error_fn=lambda cache, idx, s: etth1.per_location_abs_error_for_indices(cache, s, idx),
    )


def load_etth2() -> Bundle:
    train_cache_path = ROOT / "cache/costarts_fresh/ETTh2_96_12/router_train_cache.pt"
    val_cache_path = ROOT / "cache/costarts_fresh/ETTh2_96_12/router_val_cache.pt"
    for p in (train_cache_path, val_cache_path, ETTH2_FROZEN):
        refuse_test(p)
    train_cache = load_cache(train_cache_path, "router_train")
    val_cache = load_cache(val_cache_path, "router_val")
    std = torch.ones(int(val_cache["num_features"]), dtype=torch.float32)
    core = load_frozen_core(ETTH2_FROZEN)
    expert_idx = etth2.expert_indices(val_cache, core)
    return Bundle(
        "ETTh2", train_cache, val_cache, std, expert_idx, core,
        forecasts_fn=etth2.forecasts_for,
        per_location_error_fn=etth2.per_location_error,
    )


def load_walkforward_dataset(dataset: str) -> Bundle:
    from experiments.costar_multidataset_frozen.common import expert_indices, forecasts_for, per_location_error

    manifest = json.loads((ROOT / "experiments/costar_multidataset_frozen/frozen_manifest.json").read_text(encoding="utf-8"))
    ds_manifest = manifest["datasets"][dataset]
    cache_dir = ROOT / f"cache/costarts_walkforward_{dataset}"
    checkpoint_root = ROOT / f"checkpoints/costarts_walkforward_{dataset}"
    train_cache_path = cache_dir / "router_train_20_60_cache.pt"
    val_cache_path = cache_dir / "router_val_60_80_cache.pt"
    normalizer_path = checkpoint_root / "final_60" / "DLinear" / "best_expert.pt"
    for p in (train_cache_path, val_cache_path, normalizer_path):
        refuse_test(p)
    train_cache = load_cache(train_cache_path, "router_train_20_60")
    val_cache = load_cache(val_cache_path, "router_val_60_80")
    std = load_std(normalizer_path, int(val_cache["num_features"]))
    core = ds_manifest["selected_core"]
    expert_idx = expert_indices(val_cache, core)
    return Bundle(dataset, train_cache, val_cache, std, expert_idx, core, forecasts_fn=forecasts_for, per_location_error_fn=per_location_error)


LOADERS: dict[str, Callable[[], Bundle]] = {
    "ETTh1": load_etth1,
    "ETTh2": load_etth2,
    "ETTm1": lambda: load_walkforward_dataset("ETTm1"),
    "Weather": lambda: load_walkforward_dataset("Weather"),
    "Electricity": lambda: load_walkforward_dataset("Electricity"),
}


# ---------------------------------------------------------------------------
# The four methods
# ---------------------------------------------------------------------------


def best_single_expert(bundle: Bundle) -> tuple[torch.Tensor, dict[str, Any]]:
    target_train = bundle.train_cache["targets"].to(torch.float32)
    mask_train = bundle.train_cache["target_masks"].to(torch.bool)
    best_local, best_name, best_mae = None, None, float("inf")
    for local_i, global_i in enumerate(bundle.expert_idx):
        pred_i = bundle.train_cache["prediction_stack"][..., global_i].to(torch.float32)
        mae = float(sample_mae(pred_i, target_train, mask_train, bundle.std).mean())
        if mae < best_mae:
            best_local, best_name, best_mae = local_i, bundle.core_names[local_i], mae
    forecasts_val = bundle.forecasts_fn(bundle.val_cache, bundle.expert_idx)
    pred = forecasts_val[..., best_local]
    return pred, {"num_causal_updates": 0, "selected_expert": best_name, "router_train_mae": best_mae}


def equal_fixed(bundle: Bundle) -> tuple[torch.Tensor, dict[str, Any]]:
    forecasts_val = bundle.forecasts_fn(bundle.val_cache, bundle.expert_idx)
    pred = forecasts_val.mean(dim=-1)
    return pred, {"num_causal_updates": 0}


def frozen_hv_weights(bundle: Bundle) -> torch.Tensor:
    """The only new logic in this experiment: aggregate router_train errors
    ONCE, convert to weights with the existing rule, and never touch them
    again. No loop over router_val, no `enforce_observable`, no update."""
    train_err_hve = bundle.per_location_error_fn(bundle.train_cache, bundle.expert_idx, bundle.std)  # [N,H,V,E]
    frozen_err = train_err_hve.mean(dim=0)  # [H,V,E] -- aggregated over router_train only
    return errors_to_weights(frozen_err, HV_TRIAL)  # [H,V,E], frozen


def frozen_hv_prediction(bundle: Bundle, forecasts_val: torch.Tensor | None = None, weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    if forecasts_val is None:
        forecasts_val = bundle.forecasts_fn(bundle.val_cache, bundle.expert_idx)
    if weights is None:
        weights = frozen_hv_weights(bundle)
    n = forecasts_val.shape[0]
    expanded = weights.unsqueeze(0).expand(n, -1, -1, -1)
    pred = predict_from_hv_weights(forecasts_val, expanded)
    return pred, {"num_causal_updates": 0, "frozen_weight_source": "router_train_only"}


def online_hv_prediction(bundle: Bundle) -> tuple[torch.Tensor, dict[str, Any]]:
    """Unmodified existing online mechanism, reused exactly."""
    starts = bundle.val_cache["absolute_window_starts"].to(torch.long)
    horizon = int(bundle.val_cache["forecast_horizon"])
    forecasts_val = bundle.forecasts_fn(bundle.val_cache, bundle.expert_idx)
    train_err_hve = bundle.per_location_error_fn(bundle.train_cache, bundle.expert_idx, bundle.std)
    val_err_hve = bundle.per_location_error_fn(bundle.val_cache, bundle.expert_idx, bundle.std)
    weights, extra = chronological_hv_weights(starts, train_err_hve.mean(dim=0), val_err_hve, horizon, HV_TRIAL)
    pred = predict_from_hv_weights(forecasts_val, weights)
    return pred, {"num_causal_updates": extra["num_updates"]}


METHODS: dict[str, Callable[[Bundle], tuple[torch.Tensor, dict[str, Any]]]] = {
    "best_single_expert": best_single_expert,
    "equal_fixed": equal_fixed,
    "frozen_hv": frozen_hv_prediction,
    "online_hv": online_hv_prediction,
}
METHOD_LABELS = {
    "best_single_expert": "Best single expert",
    "equal_fixed": "Equal fixed ensemble",
    "frozen_hv": "Frozen HxV COSTAR (NEW)",
    "online_hv": "Online HxV COSTAR (existing, unmodified)",
}


def metric_values(bundle: Bundle, pred: torch.Tensor) -> dict[str, Any]:
    target = bundle.val_cache["targets"].to(torch.float32)
    mask = bundle.val_cache["target_masks"].to(torch.bool)
    m = sample_mae_mse(pred, target, mask, bundle.std)
    return {"mae": float(m["mae"].mean()), "mse": float(m["mse"].mean()), "per_window_mae": m["mae"], "per_window_mse": m["mse"]}


# ---------------------------------------------------------------------------
# Step 6: verify frozen behavior with tests A-E. Also run the SAME tests on
# the online method to demonstrate they correctly discriminate online from
# frozen (online must FAIL A/B; frozen must PASS all five).
# ---------------------------------------------------------------------------


def perturb_targets(cache: Mapping[str, Any], indices: Sequence[int] | None, seed: int) -> dict[str, Any]:
    cloned = dict(cache)
    targets = cache["targets"].clone()
    gen = torch.Generator().manual_seed(seed)
    if indices is None:
        noise = torch.randn(targets.shape, generator=gen, dtype=torch.float32)
        targets = noise
    else:
        noise = torch.randn(targets[indices].shape, generator=gen, dtype=torch.float32)
        targets[indices] = noise
    cloned["targets"] = targets
    return cloned


def run_verification_tests(bundle: Bundle, method: str) -> dict[str, Any]:
    fn = METHODS[method]
    base_pred, _ = fn(bundle)
    n = int(bundle.val_cache["num_windows"])

    # A: change one early validation target.
    early_idx = [min(5, n - 1)]
    mutated_a = dict(bundle.val_cache)
    mutated_a_cache = perturb_targets(bundle.val_cache, early_idx, seed=1)
    bundle_a = Bundle(bundle.dataset, bundle.train_cache, mutated_a_cache, bundle.std, bundle.expert_idx, bundle.core_names, bundle.forecasts_fn, bundle.per_location_error_fn)
    pred_a, _ = fn(bundle_a)
    test_a = bool(torch.equal(base_pred, pred_a))

    # B: change every validation target.
    mutated_b_cache = perturb_targets(bundle.val_cache, None, seed=2)
    bundle_b = Bundle(bundle.dataset, bundle.train_cache, mutated_b_cache, bundle.std, bundle.expert_idx, bundle.core_names, bundle.forecasts_fn, bundle.per_location_error_fn)
    pred_b, _ = fn(bundle_b)
    test_b = bool(torch.equal(base_pred, pred_b))

    # C: generate predictions without loading validation targets at all.
    targetless_cache = {k: v for k, v in bundle.val_cache.items() if k not in ("targets", "target_masks")}
    test_c_succeeded = True
    pred_c = None
    try:
        if method == "frozen_hv":
            forecasts_val = bundle.forecasts_fn(targetless_cache, bundle.expert_idx)
            weights = frozen_hv_weights(bundle)  # only needs train_cache
            pred_c, _ = frozen_hv_prediction(bundle, forecasts_val=forecasts_val, weights=weights)
        elif method == "equal_fixed":
            forecasts_val = bundle.forecasts_fn(targetless_cache, bundle.expert_idx)
            pred_c = forecasts_val.mean(dim=-1)
        elif method == "best_single_expert":
            bundle_c = Bundle(bundle.dataset, bundle.train_cache, targetless_cache, bundle.std, bundle.expert_idx, bundle.core_names, bundle.forecasts_fn, bundle.per_location_error_fn)
            pred_c, _ = best_single_expert(bundle_c)
        else:  # online_hv genuinely requires val targets (per_location_error needs them) -- expected to fail
            bundle_c = Bundle(bundle.dataset, bundle.train_cache, targetless_cache, bundle.std, bundle.expert_idx, bundle.core_names, bundle.forecasts_fn, bundle.per_location_error_fn)
            pred_c, _ = online_hv_prediction(bundle_c)
    except (KeyError, RuntimeError, TypeError):
        test_c_succeeded = False
    test_c_matches = bool(torch.equal(base_pred, pred_c)) if (pred_c is not None and test_c_succeeded) else False

    # D: run validation windows in a different order; un-permute and compare.
    gen = torch.Generator().manual_seed(3)
    perm = torch.randperm(n, generator=gen)
    inv_perm = torch.argsort(perm)
    permuted_cache = dict(bundle.val_cache)
    for key in ("histories", "targets", "target_masks", "prediction_stack", "absolute_window_starts"):
        if key in permuted_cache:
            permuted_cache[key] = permuted_cache[key][perm]
    if method == "online_hv":
        # Online method requires chronological starts; permuting breaks its
        # precondition entirely (it isn't well-defined on shuffled input),
        # so this test is only meaningful for the frozen/non-adaptive methods.
        test_d = None
    else:
        bundle_d = Bundle(bundle.dataset, bundle.train_cache, permuted_cache, bundle.std, bundle.expert_idx, bundle.core_names, bundle.forecasts_fn, bundle.per_location_error_fn)
        pred_d, _ = fn(bundle_d)
        pred_d_unpermuted = pred_d[inv_perm]
        test_d = bool(torch.equal(base_pred, pred_d_unpermuted))

    # E: router parameters/state before and after validation are identical.
    if method == "frozen_hv":
        weights_before = frozen_hv_weights(bundle).clone()
        _ = fn(bundle)  # run validation
        weights_after = frozen_hv_weights(bundle).clone()
        test_e = bool(torch.equal(weights_before, weights_after))
    elif method in ("equal_fixed", "best_single_expert"):
        test_e = True  # no state at all
    else:
        test_e = False  # online_hv's EMA state is, by construction, different at the end of the walk than at the start

    return {
        "dataset": bundle.dataset,
        "method": method,
        "test_A_early_target_change_identical": test_a,
        "test_B_all_targets_changed_identical": test_b,
        "test_C_no_targets_loaded_succeeded": test_c_succeeded,
        "test_C_no_targets_loaded_matches": test_c_matches,
        "test_D_order_invariant": test_d,
        "test_E_state_unchanged_before_after": test_e,
        "all_pass": bool(test_a and test_b and test_c_succeeded and test_c_matches and (test_d is not False) and test_e) if method != "online_hv" else None,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_dataset(dataset: str) -> dict[str, Any]:
    bundle = LOADERS[dataset]()
    results: dict[str, dict[str, Any]] = {}
    result_rows: list[dict[str, Any]] = []
    for method in ("best_single_expert", "equal_fixed", "frozen_hv", "online_hv"):
        pred, extra = METHODS[method](bundle)
        m = metric_values(bundle, pred)
        results[method] = {"pred": pred, "mae": m["mae"], "mse": m["mse"], "per_window_mae": m["per_window_mae"]}
        result_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "label": METHOD_LABELS[method],
                "mae": m["mae"],
                "mse": m["mse"],
                "expert_set": "+".join(bundle.core_names),
                "decay": CANONICAL_DECAY if method in ("frozen_hv", "online_hv") else None,
                "temperature": CANONICAL_TEMPERATURE if method in ("frozen_hv", "online_hv") else None,
                **extra,
            }
        )

    verification_rows = [run_verification_tests(bundle, method) for method in ("best_single_expert", "equal_fixed", "frozen_hv", "online_hv")]
    for row in verification_rows:
        if row["method"] == "frozen_hv" and not row["all_pass"]:
            raise AssertionError(f"{dataset}: frozen_hv failed a frozen-behavior verification test: {row}")
        if row["method"] == "online_hv" and (row["test_A_early_target_change_identical"] or row["test_B_all_targets_changed_identical"]):
            raise AssertionError(f"{dataset}: online_hv unexpectedly did not react to target changes -- test methodology is broken: {row}")

    comparisons = [
        ("frozen_hv_vs_equal", "frozen_hv", "equal_fixed"),
        ("online_hv_vs_equal", "online_hv", "equal_fixed"),
        ("frozen_hv_vs_online_hv", "frozen_hv", "online_hv"),
    ]
    delta_rows = []
    dependence_rows = []
    for label, cand, base in comparisons:
        candidate, baseline = results[cand]["per_window_mae"], results[base]["per_window_mae"]
        boot = paired_bootstrap(candidate, baseline, seed=20260821, samples=5000)
        delta_rows.append({"dataset": dataset, "comparison": label, "candidate": cand, "baseline": base, "delta_mae": results[cand]["mae"] - results[base]["mae"], **{f"iid_{k}": v for k, v in boot.items()}})
        for block in BLOCK_LENGTHS:
            b = block_bootstrap_with_prob(candidate, baseline, block=block, seed=20260821, samples=BOOTSTRAP_SAMPLES)
            dependence_rows.append({"dataset": dataset, "comparison": label, "test": f"block_bootstrap_len{block}", **b})
        phase = every_kth_phase_bootstrap(candidate - baseline, k=PHASE_K, seed=20260821, samples=BOOTSTRAP_SAMPLES)
        dependence_rows.append({"dataset": dataset, "comparison": label, "test": f"every_{PHASE_K}th_window_phase_bootstrap", **phase})

    return {
        "dataset": dataset,
        "core": bundle.core_names,
        "num_windows_val": int(bundle.val_cache["num_windows"]),
        "result_rows": result_rows,
        "verification_rows": verification_rows,
        "delta_rows": delta_rows,
        "dependence_rows": dependence_rows,
    }


def make_report(out_dir: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Frozen HxV COSTAR (removing deployment-time target feedback)",
        "",
        "New, separate experiment. The existing online/causal COSTAR implementation is unmodified and evaluated here as `online_hv`.",
        "",
        "## Step 4/5: validation results",
        "",
        "| Dataset | Best Single | Equal Fixed | Frozen HxV (NEW) | Online HxV (existing) |",
        "|---|---:|---:|---:|---:|",
    ]
    for ds, d in report["datasets"].items():
        by = {r["method"]: r for r in d["result_rows"]}
        lines.append(
            "| {ds} | `{a[mae]:.6f}`/`{a[mse]:.6f}` | `{b[mae]:.6f}`/`{b[mse]:.6f}` | `{c[mae]:.6f}`/`{c[mse]:.6f}` | `{d[mae]:.6f}`/`{d[mse]:.6f}` |".format(
                ds=ds, a=by["best_single_expert"], b=by["equal_fixed"], c=by["frozen_hv"], d=by["online_hv"]
            )
        )
    lines += ["", "## Deltas (IID paired bootstrap, quick reference)", ""]
    lines.append("| Dataset | Comparison | Delta MAE | 95% CI | Excludes zero |")
    lines.append("|---|---|---:|---|---|")
    for ds, d in report["datasets"].items():
        for row in d["delta_rows"]:
            lines.append(f"| {ds} | {row['comparison']} | `{row['delta_mae']:+.6f}` | [{row['iid_ci95_low']:+.6f}, {row['iid_ci95_high']:+.6f}] | {row['iid_ci_excludes_zero']} |")
    lines += ["", "## Step 6: frozen-behavior verification (A-E)", ""]
    lines.append("| Dataset | Method | A: early target | B: all targets | C: no targets loaded | D: order-invariant | E: state unchanged | All pass |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for ds, d in report["datasets"].items():
        for row in d["verification_rows"]:
            lines.append(
                f"| {ds} | {row['method']} | {row['test_A_early_target_change_identical']} | {row['test_B_all_targets_changed_identical']} | "
                f"{row['test_C_no_targets_loaded_succeeded']} & matches={row['test_C_no_targets_loaded_matches']} | {row['test_D_order_invariant']} | "
                f"{row['test_E_state_unchanged_before_after']} | {row['all_pass']} |"
            )
    lines += ["", "## Hard rule compliance", "", "```text", "TEST SET ACCESSED: NO", "TEST CACHE LOADED: NO", "TEST METRICS COMPUTED: NO", "```"]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    report: dict[str, Any] = {"experiment": "frozen_hv_costar", "created_at_utc": datetime.now(timezone.utc).isoformat(), "datasets": {}}
    all_results, all_deltas, all_dependence, all_verification = [], [], [], []
    for dataset in LOADERS:
        print(f"[frozen-hv-costar] {dataset}: evaluating...", flush=True)
        result = evaluate_dataset(dataset)
        report["datasets"][dataset] = result
        all_results.extend(result["result_rows"])
        all_deltas.extend(result["delta_rows"])
        all_dependence.extend(result["dependence_rows"])
        all_verification.extend(result["verification_rows"])
        print(f"[frozen-hv-costar] {dataset}: done. core={'+'.join(result['core'])}", flush=True)

    report["runtime_sec"] = time.time() - start
    report["test_set_accessed"] = False
    write_json(OUT_DIR / "results.json", report)
    write_csv(OUT_DIR / "results.csv", all_results)
    write_csv(OUT_DIR / "deltas.csv", all_deltas)
    write_csv(OUT_DIR / "dependence_aware_bootstrap.csv", all_dependence)
    write_csv(OUT_DIR / "verification_tests.csv", all_verification)
    make_report(OUT_DIR, report)
    print("TEST SET ACCESSED: NO")
    print(json.dumps({"runtime_sec": report["runtime_sec"], "datasets": list(report["datasets"].keys())}, indent=2))


if __name__ == "__main__":
    main()
