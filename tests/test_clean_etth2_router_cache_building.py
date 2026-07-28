import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.costars import build_clean_etth2_router_caches as caches
from scripts.costars import train_clean_etth2_experts as clean


def _write_dataset(path: Path, length: int = 3000) -> np.ndarray:
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text(
        json.dumps({"num_time_steps": length, "num_vars": 7}),
        encoding="utf-8",
    )
    data = np.arange(length * 7, dtype=np.float32).reshape(length, 7)
    np.save(path / "train_data.npy", data[: int(length * 0.6)])
    np.save(path / "val_data.npy", data[int(length * 0.6) : int(length * 0.8)])
    np.save(path / "test_data.npy", data[int(length * 0.8) :])
    return data


def test_pretest_loader_reads_only_train_and_val(tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    loaded = []
    original = np.load

    def guarded(path, *args, **kwargs):
        loaded.append(Path(path).name)
        assert "test" not in Path(path).name
        return original(path, *args, **kwargs)

    monkeypatch.setattr(caches.np, "load", guarded)
    prefix, manifest = caches.load_etth2_pretest_prefix(tmp_path)

    assert loaded == ["train_data.npy", "val_data.npy"]
    assert len(prefix) == manifest["router_val"]["end"]


def test_only_router_splits_are_exposed(tmp_path):
    data = np.zeros((2400, 7), dtype=np.float32)
    manifest = clean.split_manifest_for_total_length(3000)
    with pytest.raises(ValueError, match="Only"):
        caches.build_router_loader(data, manifest, "locked_test", 32, 0)


def test_router_train_and_val_windows_are_disjoint_and_inside_splits(tmp_path):
    data = _write_dataset(tmp_path)
    prefix, manifest = caches.load_etth2_pretest_prefix(tmp_path)
    train_loader = caches.build_router_loader(prefix, manifest, "router_train", 32, 0)
    val_loader = caches.build_router_loader(prefix, manifest, "router_val", 32, 0)

    train_last = train_loader.dataset[len(train_loader.dataset) - 1]
    val_first = val_loader.dataset[0]
    assert train_last["targets"][-1, 0] == data[manifest["router_train"]["end"] - 1, 0]
    assert val_first["inputs"][0, 0] == data[manifest["router_val"]["start"], 0]
    assert train_loader.dataset.boundary["end"] == val_loader.dataset.boundary["start"]


class ScaleExpert(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, inputs, inputs_timestamps=None):
        return inputs[:, -12:, :] * self.scale


def _small_cache(tmp_path):
    data = _write_dataset(tmp_path)
    prefix, manifest = caches.load_etth2_pretest_prefix(tmp_path)
    scaler = caches.ZScoreScaler(
        norm_each_channel=True,
        rescale=False,
        stats={"mean": torch.zeros(1, 7), "std": torch.ones(1, 7)},
    )
    loader = caches.build_router_loader(prefix, manifest, "router_train", 128, 0)
    experts = tuple(ScaleExpert(float(i + 1)).eval() for i in range(5))
    for expert in experts:
        for parameter in expert.parameters():
            parameter.requires_grad_(False)
    names = caches.EXPERT_NAMES
    hashes = {name: f"hash-{i}" for i, name in enumerate(names)}
    cache_path = tmp_path / "router_train_cache.pt"
    cache = caches.build_cache(
        "router_train",
        loader,
        experts,
        scaler,
        hashes,
        "scaler-hash",
        torch.device("cpu"),
        cache_path,
    )
    return cache, loader, experts, scaler, hashes, cache_path


def test_cache_shapes_metrics_direct_inference_and_no_gradients(tmp_path):
    cache, loader, experts, scaler, hashes, cache_path = _small_cache(tmp_path)

    assert cache_path.exists()
    assert tuple(cache["histories"].shape[1:]) == (96, 7)
    assert tuple(cache["targets"].shape[1:]) == (12, 7)
    assert tuple(cache["prediction_stack"].shape[1:]) == (12, 7, 5)
    assert tuple(cache["error_matrix"].shape[1:]) == (5,)
    direct = caches.verify_direct_samples(cache, loader, experts, scaler, torch.device("cpu"))
    assert len(direct["sampled_windows"]) == 3
    assert all(parameter.grad is None for expert in experts for parameter in expert.parameters())


def test_expert_order_hash_and_scaler_mismatches_fail(tmp_path):
    cache, _, _, _, hashes, _ = _small_cache(tmp_path)
    bad_order = dict(cache)
    bad_order["expert_names"] = tuple(reversed(caches.EXPERT_NAMES))
    with pytest.raises(ValueError, match="expert order"):
        caches.validate_cache(bad_order, "router_train", hashes, "scaler-hash")

    with pytest.raises(ValueError, match="checkpoint hashes"):
        caches.validate_cache(cache, "router_train", {"DLinear": "bad"}, "scaler-hash")

    with pytest.raises(ValueError, match="scaler hash"):
        caches.validate_cache(cache, "router_train", hashes, "bad")


def test_cache_from_other_dataset_or_horizon_is_rejected(tmp_path):
    cache, _, _, _, hashes, _ = _small_cache(tmp_path)
    wrong_dataset = dict(cache)
    wrong_dataset["dataset"] = "ETTh1"
    with pytest.raises(ValueError, match="dataset"):
        caches.validate_cache(wrong_dataset, "router_train", hashes, "scaler-hash")

    wrong_horizon = dict(cache)
    wrong_horizon["forecast_horizon"] = 24
    with pytest.raises(ValueError, match="horizon"):
        caches.validate_cache(wrong_horizon, "router_train", hashes, "scaler-hash")


def test_no_test_cache_is_created_by_permitted_build(tmp_path):
    _, _, _, _, _, cache_path = _small_cache(tmp_path)

    assert cache_path.name == "router_train_cache.pt"
    assert not (tmp_path / "test_cache.pt").exists()
