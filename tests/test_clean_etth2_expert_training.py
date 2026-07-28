import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from scripts.costars import train_clean_etth2_experts as clean
from scripts.costars.train_candidate_experts import EXPERT_SPECS


def _write_meta(path: Path, length: int = 1000) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text(
        json.dumps({"num_time_steps": length, "num_vars": 7}),
        encoding="utf-8",
    )


def test_pretest_loader_does_not_read_test_arrays(tmp_path, monkeypatch):
    _write_meta(tmp_path, length=3000)
    train = np.arange(1800 * 7, dtype=np.float32).reshape(1800, 7)
    np.save(tmp_path / "train_data.npy", train)

    loaded = []
    original_load = np.load

    def guarded_load(path, *args, **kwargs):
        loaded.append(Path(path).name)
        assert "test" not in Path(path).name
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(clean.np, "load", guarded_load)
    data, manifest = clean.load_etth2_expert_prefix(tmp_path)

    assert loaded == ["train_data.npy"]
    assert data.shape == (1800, 7)
    assert manifest["expert_val"]["end"] == 1800


def test_split_manifest_keeps_windows_inside_splits():
    manifest = clean.split_manifest_for_total_length(14400)

    assert manifest["expert_train"]["start"] == 0
    assert manifest["expert_train"]["end"] == 7200
    assert manifest["expert_val"]["start"] == 7200
    assert manifest["expert_val"]["end"] == 8640
    assert manifest["locked_test"]["start"] == 11520
    for role, row in manifest.items():
        last_target = row["last_valid_window_start"] + 96 + 12
        assert row["first_valid_window_start"] == row["start"]
        assert last_target == row["end"]
        assert row["num_windows"] == row["end"] - row["start"] - 96 - 12 + 1


def test_scaler_fits_only_on_expert_train(tmp_path):
    _write_meta(tmp_path, length=3000)
    train = np.arange(1800 * 7, dtype=np.float32).reshape(1800, 7)
    np.save(tmp_path / "train_data.npy", train)
    train_loader, _, _, _ = clean.build_clean_expert_dataloaders(tmp_path, batch_size=32)

    scaler, manifest = clean.fit_clean_scaler(train_loader)

    assert manifest["fit_split"] == "expert_train"
    assert manifest["source_index_range"] == [0, 1500]
    assert scaler.stats["mean"].numpy() == pytest.approx(train[:1500].mean(axis=0, keepdims=True))


def test_expert_and_validation_windows_are_disjoint(tmp_path):
    _write_meta(tmp_path, length=3000)
    train = np.arange(1800 * 7, dtype=np.float32).reshape(1800, 7)
    np.save(tmp_path / "train_data.npy", train)
    train_loader, val_loader, _, manifest = clean.build_clean_expert_dataloaders(
        tmp_path,
        batch_size=32,
    )

    assert train_loader.dataset.boundary["end"] == val_loader.dataset.boundary["start"]
    assert manifest["router_train"]["start"] == val_loader.dataset.boundary["end"]
    train_last = train_loader.dataset[len(train_loader.dataset) - 1]
    val_first = val_loader.dataset[0]
    assert train_last["targets"][-1, 0] == train[1499, 0]
    assert val_first["inputs"][0, 0] == train[1500, 0]


def test_all_expert_configs_match_intended_reference_fields():
    rows, matches = clean.config_comparison_rows(Path("checkpoints/candidates"))

    assert rows
    assert matches == {
        "DLinear": True,
        "PatchTST": True,
        "iTransformer": True,
        "TimesNet": True,
        "ModernTCN": True,
    }


def test_every_expert_produces_expected_shape():
    x = torch.randn(2, 96, 7)
    for spec in EXPERT_SPECS.values():
        model = spec.model_class(spec.config_factory())
        model.eval()
        with torch.no_grad():
            prediction = clean.call_forecasting_model(model, x, spec.requires_timestamps)
        assert tuple(prediction.shape) == (2, 12, 7)
        assert torch.isfinite(prediction).all()


def test_timesnet_uses_reusable_call_helper():
    spec = EXPERT_SPECS["timesnet"]
    model = spec.model_class(spec.config_factory())
    with torch.no_grad():
        prediction = clean.call_forecasting_model(
            model,
            torch.randn(2, 96, 7),
            spec.requires_timestamps,
        )
    assert tuple(prediction.shape) == (2, 12, 7)


class TinyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(96, 12)

    def forward(self, inputs):
        return self.projection(inputs.transpose(1, 2)).transpose(1, 2)


class TinySpec:
    key = "tiny"
    display_name = "Tiny"
    requires_timestamps = False

    @staticmethod
    def config_factory():
        return {"input_len": 96, "output_len": 12, "num_features": 7}

    model_class = TinyExpert


def _tiny_checkpoint(path, model, overrides=None):
    payload = {
        "completion_status": "complete",
        "model_state_dict": model.state_dict(),
        "model_config": {"input_len": 96, "output_len": 12, "num_features": 7},
        "dataset": "ETTh2",
        "validation_mae": 1.0,
        "validation_mse": 2.0,
        "dataset_config": clean.split_manifest_for_total_length(3000),
        "scaler_stats": {"mean": torch.zeros(1, 7), "std": torch.ones(1, 7)},
        "model_key": "tiny",
        "input_len": 96,
        "output_len": 12,
        "num_features": 7,
    }
    payload.update(overrides or {})
    torch.save(payload, path)


def test_checkpoint_rejects_other_horizon_and_partial(tmp_path):
    model = TinyExpert()
    path = tmp_path / "bad.pt"
    _tiny_checkpoint(path, model, {"output_len": 24})

    train = np.random.default_rng(7).normal(size=(1800, 7)).astype(np.float32)
    manifest = clean.split_manifest_for_total_length(3000)
    loader = torch.utils.data.DataLoader(
        clean.AbsoluteWindowDataset(train, manifest, "expert_val"),
        batch_size=16,
    )
    scaler = clean.ZScoreScaler(norm_each_channel=True, rescale=False)
    scaler.fit(train[:1500])

    with pytest.raises(ValueError, match="wrong horizon"):
        clean.verify_checkpoint(path, TinySpec, loader, scaler, torch.device("cpu"))

    partial = tmp_path / "partial.pt"
    torch.save({"model_state_dict": model.state_dict()}, partial)
    with pytest.raises(ValueError, match="missing fields"):
        clean.verify_checkpoint(partial, TinySpec, loader, scaler, torch.device("cpu"))


def test_checkpoint_from_another_dataset_is_rejected(tmp_path):
    model = TinyExpert()
    path = tmp_path / "wrong_dataset.pt"
    _tiny_checkpoint(path, model, {"dataset": "ETTh1"})
    manifest = clean.split_manifest_for_total_length(3000)
    train = np.random.default_rng(7).normal(size=(1800, 7)).astype(np.float32)
    loader = torch.utils.data.DataLoader(
        clean.AbsoluteWindowDataset(train, manifest, "expert_val"),
        batch_size=16,
    )
    scaler = clean.ZScoreScaler(norm_each_channel=True, rescale=False)
    scaler.fit(train[:1500])

    with pytest.raises(ValueError, match="not ETTh2"):
        clean.verify_checkpoint(path, TinySpec, loader, scaler, torch.device("cpu"))


def test_fresh_reload_reproduces_predictions_and_frozen_gradients(tmp_path):
    model = TinyExpert()
    path = tmp_path / "tiny.pt"
    manifest = clean.split_manifest_for_total_length(3000)
    _tiny_checkpoint(path, model, {"dataset_config": manifest})
    train = np.random.default_rng(7).normal(size=(1800, 7)).astype(np.float32)
    loader = torch.utils.data.DataLoader(
        clean.AbsoluteWindowDataset(train, manifest, "expert_val"),
        batch_size=16,
    )
    scaler = clean.ZScoreScaler(norm_each_channel=True, rescale=False)
    scaler.fit(train[:1500])

    verification = clean.verify_checkpoint(
        path,
        TinySpec,
        loader,
        scaler,
        torch.device("cpu"),
        expected_config={"input_len": 96, "output_len": 12, "num_features": 7},
        split_manifest=manifest,
    )

    assert verification["output_shape"] == [16, 12, 7]
    assert verification["deterministic_inference"]
    assert verification["frozen_gradient_verification"]
