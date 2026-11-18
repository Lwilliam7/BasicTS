import torch

from experiments.etth1_simplex_linear_test_evaluation import run_etth1_simplex_linear_test_evaluation as runner


def _cache(role: str):
    starts = {
        "router_train_20_60": torch.arange(2880, 8426, dtype=torch.long),
        "router_val_60_80": torch.arange(8640, 11413, dtype=torch.long),
        "test_80_100": torch.arange(11520, 14293, dtype=torch.long),
    }[role]
    n = starts.numel()
    target = torch.zeros(n, 12, 7)
    mask = torch.ones_like(target, dtype=torch.bool)
    stack = torch.stack([torch.full_like(target, float(i)) for i in range(5)], dim=-1)
    return {
        "cache_role": role,
        "split_role": role,
        "expert_names": runner.EXPECTED_EXPERTS,
        "num_features": 7,
        "num_windows": int(n),
        "absolute_window_starts": starts,
        "targets": target,
        "target_masks": mask,
        "prediction_stack": stack,
    }


def test_weighted_prediction_uses_all_five_experts():
    cache = _cache("router_val_60_80")
    weights = torch.tensor([0.2, 0.0, 0.3, 0.0, 0.5])
    pred = runner.weighted_prediction(cache, weights)
    assert torch.allclose(pred, torch.full_like(cache["targets"], 2.9))


def test_manifest_guard_blocks_test_path_before_allowed(tmp_path):
    path = tmp_path / "test_80_100_cache.pt"
    torch.save(_cache("test_80_100"), path)
    try:
        runner.load_cache(path, "test_80_100", allow_test=False)
    except AssertionError as exc:
        assert "Refusing to load test path" in str(exc)
    else:
        raise AssertionError("test path was not rejected")
