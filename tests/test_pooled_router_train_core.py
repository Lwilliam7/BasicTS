import torch

from experiments.pooled_router_train_core import run_pooled_router_train_core as runner


def _toy_cache():
    n = 3
    target = torch.zeros(n, 12, 7)
    mask = torch.ones_like(target, dtype=torch.bool)
    experts = [
        torch.zeros_like(target),
        torch.full_like(target, 0.1),
        torch.full_like(target, 0.2),
        torch.full_like(target, 3.0),
        torch.full_like(target, 4.0),
    ]
    return {
        "cache_role": "router_train_20_60",
        "split_role": "router_train_20_60",
        "expert_names": runner.EXPECTED_EXPERTS,
        "num_windows": n,
        "input_len": 96,
        "forecast_horizon": 12,
        "num_features": 7,
        "absolute_window_starts": torch.arange(2880, 2880 + n),
        "targets": target,
        "target_masks": mask,
        "prediction_stack": torch.stack(experts, dim=-1),
    }


def test_pooled_train_ranking_selects_lowest_full_router_train_mae():
    cache = _toy_cache()
    ranking = runner.pooled_train_ranking("ETTh1", cache, torch.ones(7))
    assert ranking[0]["subset"] == "DLinear+PatchTST+iTransformer"
    assert ranking[0]["router_train_mae"] < ranking[1]["router_train_mae"]


def test_test_path_guard_before_freeze(tmp_path):
    path = tmp_path / "some_test_cache.pt"
    torch.save(_toy_cache(), path)
    try:
        runner.load_role_cache(path, "router_train_20_60", allow_test=False)
    except AssertionError as exc:
        assert "Refusing test path" in str(exc)
    else:
        raise AssertionError("test path was not rejected")
