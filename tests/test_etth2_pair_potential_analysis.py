import itertools
import json
from pathlib import Path

import pytest
import torch

from scripts.costars import analyze_etth2_pair_potential as analysis


def _toy_cache(split="router_train"):
    targets = torch.zeros(4, 12, 7)
    mask = torch.ones_like(targets, dtype=torch.bool)
    experts = []
    for value in (1.0, 2.0, 3.0, 4.0, 5.0):
        experts.append(torch.full_like(targets, value))
    stack = torch.stack(experts, dim=-1)
    mae, mse = analysis.per_window_error(stack, targets, mask)
    starts = torch.arange(4) + (8640 if split == "router_train" else 10800)
    return {
        "dataset": "ETTh2",
        "split_role": split,
        "expert_names": analysis.EXPECTED_EXPERTS,
        "expert_order": analysis.EXPECTED_EXPERTS,
        "checkpoint_hashes": {"DLinear": "a"},
        "scaler_hash": analysis.EXPECTED_HASHES["scaler"],
        "input_len": 96,
        "forecast_horizon": 12,
        "num_features": 7,
        "num_windows": 4,
        "histories": torch.zeros(4, 96, 7),
        "targets": targets,
        "target_masks": mask,
        "prediction_stack": stack,
        "error_matrix": mae,
        "mse_matrix": mse,
        "best_expert": mae.argmin(dim=1),
        "sample_indices": torch.arange(4),
        "absolute_window_starts": starts,
        "split_boundary": {
            "first_valid_window_start": int(starts.min()),
            "last_valid_window_start": int(starts.max()),
            "end": int(starts.max()) + 108,
        },
    }


def test_pair_prediction_is_exact_equal_average():
    cache = _toy_cache()
    pair = analysis.subset_prediction(cache, (0, 2))

    expected = 0.5 * cache["prediction_stack"][..., 0] + 0.5 * cache["prediction_stack"][..., 2]
    assert torch.equal(pair, expected)


def test_pair_enumeration_has_all_ten_unique_unordered_pairs():
    rows, mae_matrix, mse_matrix, combos = analysis.all_subset_metrics(_toy_cache(), 2)

    assert len(combos) == 10
    assert len(set(combos)) == 10
    assert set(combos) == set(itertools.combinations(range(5), 2))
    assert mae_matrix.shape == (4, 10)
    assert mse_matrix.shape == (4, 10)
    assert len(rows) == 10


def test_pair_selection_uses_train_only():
    train = _toy_cache("router_train")
    val = _toy_cache("router_val")
    val["prediction_stack"][..., 4] = 0.0
    val["error_matrix"], val["mse_matrix"] = analysis.per_window_error(
        val["prediction_stack"],
        val["targets"],
        val["target_masks"],
    )

    train_rows, _, _, _ = analysis.all_subset_metrics(train, 2)
    val_rows, _, _, _ = analysis.all_subset_metrics(val, 2)

    assert train_rows[0]["subset"] == "DLinear+PatchTST"
    assert val_rows[0]["subset"] != train_rows[0]["subset"]


def test_train_fitted_weights_do_not_use_validation_targets(monkeypatch):
    train = _toy_cache("router_train")
    val = _toy_cache("router_val")

    called = []
    original = analysis.flatten_xy

    def spy(cache):
        called.append(cache["split_role"])
        return original(cache)

    monkeypatch.setattr(analysis, "flatten_xy", spy)
    train_rows, train_metrics = analysis.individual_rows(train)
    analysis.ensemble_baselines(train, val, train_metrics, "DLinear+PatchTST", train_rows[0]["expert"])

    assert called == ["router_train", "router_train"]


def test_oracle_diagnostics_are_not_deployable_selections(tmp_path):
    summary = {
        "training_selected_best_fixed_expert": "DLinear",
        "training_selected_best_fixed_pair": "DLinear+PatchTST",
        "fixed_pair_val_mae": 1.0,
        "fixed_pair_beats_both_constituents_on_validation": False,
        "oracle_diagnostics": {"validation_fixed_pair_to_oracle_pair_improvement": 0.1},
        "switch_opportunity": [{"improvement_margin": 0.01, "useful_switch_percentage": 5.0}],
        "routing_margin_diagnostics": {"router_val_pair": {"mean_margin": 0.1, "median_margin": 0.1}},
    }
    output = tmp_path / "report.md"
    analysis.write_markdown_report(output, summary, [])

    assert "oracle diagnostic" not in summary["training_selected_best_fixed_pair"]
    assert "No ETTh2 test arrays" in output.read_text()


def test_expert_order_and_ranges_are_checked():
    cache = _toy_cache()
    bad = dict(cache)
    bad["expert_names"] = tuple(reversed(analysis.EXPECTED_EXPERTS))

    with pytest.raises(ValueError, match="Expert ordering"):
        analysis.validate_cache_schema(bad, "router_train", {"checkpoint_hashes": {"DLinear": "a"}})

    val = _toy_cache("router_val")
    analysis.validate_cache_pair(cache, val)


def test_cache_hash_is_checked(tmp_path):
    cache_path = tmp_path / "cache.pt"
    torch.save(_toy_cache(), cache_path)

    with pytest.raises(ValueError, match="cache hash mismatch"):
        analysis.load_verified_cache(
            cache_path,
            "router_train",
            {"cache_hashes": {"router_train": analysis.EXPECTED_HASHES["router_train"]}, "checkpoint_hashes": {"DLinear": "a"}},
        )


def test_no_test_arrays_or_test_cache_are_needed(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("dataset loader should not be called")

    monkeypatch.setattr(analysis.torch, "load", forbidden)
    assert not (tmp_path / "test_cache.pt").exists()
    assert not (tmp_path / "locked_test_cache.pt").exists()


def test_metrics_and_expert_counts_reproduce_direct_calculation():
    cache = _toy_cache()
    mae, mse = analysis.subset_errors(cache, (0, 1))
    expected_prediction = torch.full_like(cache["targets"], 1.5)
    expected_mae, expected_mse = analysis.per_window_error(
        expected_prediction,
        cache["targets"],
        cache["target_masks"],
    )

    assert torch.equal(mae, expected_mae)
    assert torch.equal(mse, expected_mse)
    rows, _, _, _ = analysis.all_subset_metrics(cache, 2)
    assert all(row["average_experts_used"] == 2.0 for row in rows)
