import torch

from experiments.etth2_pair_potential_test_evaluation import run_etth2_pair_potential_test_evaluation as runner
from scripts.costars import analyze_etth2_pair_potential as analysis


def _cache(split_role: str):
    targets = torch.zeros(4, 12, 7)
    masks = torch.ones_like(targets, dtype=torch.bool)
    stack = torch.stack([torch.full_like(targets, float(i + 1)) for i in range(5)], dim=-1)
    mae, mse = analysis.per_window_error(stack, targets, masks)
    starts = {
        "router_train": torch.arange(8640, 8644),
        "router_val": torch.arange(10800, 10804),
        "locked_test": torch.arange(11520, 11524),
    }[split_role]
    return {
        "cache_role": split_role,
        "split_role": split_role,
        "expert_names": analysis.EXPECTED_EXPERTS,
        "num_windows": 4,
        "histories": torch.zeros(4, 96, 7),
        "targets": targets,
        "target_masks": masks,
        "prediction_stack": stack,
        "error_matrix": mae,
        "mse_matrix": mse,
        "absolute_window_starts": starts,
    }


def test_result_row_uses_fixed_weights_without_refitting(monkeypatch):
    val = _cache("router_val")
    test = _cache("locked_test")
    weights = torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0])
    dlinear_mae, _ = analysis.weighted_errors(test, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))

    monkeypatch.setitem(runner.VALIDATION_REFS, "nonnegative_simplex_linear_average", {"mae": 1.5, "mse": 2.25})
    row = runner.result_row("nonnegative_simplex_linear_average", weights, val, test, dlinear_mae)

    assert row["test_mae"] == 1.5
    assert row["test_mse"] == 2.25
    assert row["status"] == "after_final_test_audit"


def test_manifest_guard_rejects_test_path_before_allowed(tmp_path):
    test_path = tmp_path / "my_test_cache.pt"
    torch.save(_cache("locked_test"), test_path)

    try:
        runner.load_cache(test_path, "locked_test", allow_test=False)
    except AssertionError as exc:
        assert "Refusing to load test path" in str(exc)
    else:
        raise AssertionError("test path was not rejected")
