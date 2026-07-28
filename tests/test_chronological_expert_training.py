import numpy as np
import pytest
import torch
from torch import nn

from basicts.metrics import masked_mae
from basicts.scaler import ZScoreScaler
from scripts.chronological_expert_training import (
    ChronologicalForecastingDataset,
    ForecastRouter,
    assert_experts_frozen,
    build_all_split_dataloaders,
    build_expert_dataloaders,
    build_router_dataloaders,
    build_split_dataloader,
    chronological_split_boundaries,
    evaluate_expert,
    evaluate_final_router_and_baselines,
    fit_scaler_on_expert_train,
    prepare_chronological_dataloaders,
    print_split_diagnostics,
    train_expert_model,
    train_router_model,
)


def test_chronological_boundaries_are_exact_and_non_overlapping():
    boundaries = chronological_split_boundaries(1000)

    assert (boundaries["expert_train"].start, boundaries["expert_train"].end) == (0, 500)
    assert (boundaries["expert_val"].start, boundaries["expert_val"].end) == (500, 600)
    assert (boundaries["router_train"].start, boundaries["router_train"].end) == (600, 750)
    assert (boundaries["router_val"].start, boundaries["router_val"].end) == (750, 800)
    assert (boundaries["test"].start, boundaries["test"].end) == (800, 1000)

    ordered = [boundaries[name] for name in (
        "expert_train", "expert_val", "router_train", "router_val", "test"
    )]
    assert all(left.end == right.start for left, right in zip(ordered, ordered[1:]))


def test_windows_stay_inside_their_chronological_segment():
    full_data = np.arange(1000, dtype=np.float32).reshape(-1, 1)
    dataset = ChronologicalForecastingDataset(
        full_data=full_data,
        input_len=10,
        output_len=5,
        split_role="expert_val",
    )

    first = dataset[0]
    last = dataset[len(dataset) - 1]

    assert first["inputs"][0, 0] == 500
    assert first["targets"][0, 0] == 510
    assert last["targets"][-1, 0] == 599


def test_scaler_is_fit_only_on_expert_training_split():
    full_data = np.arange(1000, dtype=np.float32).reshape(-1, 1)
    train_loader, _ = build_expert_dataloaders(
        full_data=full_data,
        input_len=10,
        output_len=5,
        batch_size=32,
    )
    scaler = fit_scaler_on_expert_train(
        ZScoreScaler(norm_each_channel=True, rescale=False),
        train_loader,
    )

    assert scaler.stats["mean"].item() == pytest.approx(249.5)
    assert scaler.stats["std"].item() == pytest.approx(np.std(full_data[:500]))


def test_prepare_builds_all_unshuffled_splits_and_reuses_train_scaler(capsys):
    full_data = np.arange(4000 * 7, dtype=np.float32).reshape(4000, 7)
    loaders, scaler = prepare_chronological_dataloaders(
        full_data=full_data,
        scaler=ZScoreScaler(norm_each_channel=True, rescale=False),
        batch_size=32,
        input_len=96,
        output_len=12,
    )

    assert tuple(loaders) == (
        "expert_train",
        "expert_val",
        "router_train",
        "router_val",
        "test",
    )
    assert scaler.stats["mean"].numpy() == pytest.approx(
        full_data[:2000].mean(axis=0, keepdims=True)
    )
    for loader in loaders.values():
        assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)
        batch = next(iter(loader))
        assert batch["inputs"].shape == (32, 96, 7)
        assert batch["targets"].shape == (32, 12, 7)

    output = capsys.readouterr().out
    for role in loaders:
        assert role in output
    assert output.count("Input: [32, 96, 7]") == 5
    assert output.count("Target: [32, 12, 7]") == 5


def test_split_diagnostics_reports_exact_window_count(capsys):
    full_data = np.zeros((4000, 7), dtype=np.float32)
    loaders = build_all_split_dataloaders(full_data, batch_size=16)

    print_split_diagnostics(loaders)

    output = capsys.readouterr().out
    # 10% of 4000 is 400 timestamps: 400 - 96 - 12 + 1 windows.
    assert "expert_val          2000      2400          400             293" in output


def test_validation_metrics_are_weighted_over_every_element():
    class ZeroForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.0))

        def forward(self, inputs):
            return inputs * self.scale

    model = ZeroForecaster()
    batches = [
        {
            "inputs": torch.zeros(1, 1, 1),
            "targets": torch.ones(1, 1, 1),
        },
        {
            "inputs": torch.zeros(3, 1, 1),
            "targets": torch.full((3, 1, 1), 3.0),
        },
    ]

    mae, mse = evaluate_expert(model, batches, "Zero", torch.device("cpu"))

    assert mae == pytest.approx(2.5)
    assert mse == pytest.approx(7.0)
    assert model.scale.grad is None
    assert not model.training


class TinyForecaster(nn.Module):
    def __init__(self, input_len, output_len):
        super().__init__()
        self.projection = nn.Linear(input_len, output_len)

    def forward(self, inputs):
        return self.projection(inputs.transpose(1, 2)).transpose(1, 2)


def test_training_saves_best_checkpoint_reloads_and_freezes(tmp_path):
    torch.manual_seed(7)
    x = np.linspace(0.0, 10.0, 1000, dtype=np.float32)
    full_data = np.stack([x, 0.5 * x], axis=-1)
    train_loader, val_loader = build_expert_dataloaders(
        full_data=full_data,
        input_len=8,
        output_len=3,
        batch_size=64,
    )
    model = TinyForecaster(input_len=8, output_len=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    checkpoint_path = tmp_path / "best_expert.pt"
    scaler = fit_scaler_on_expert_train(
        ZScoreScaler(norm_each_channel=True, rescale=False),
        train_loader,
    )

    result = train_expert_model(
        model=model,
        optimizer=optimizer,
        model_name="Tiny",
        checkpoint_path=checkpoint_path,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=2,
        patience=5,
        scaler=scaler,
        loss_fn=masked_mae,
        model_config={"input_len": 8, "output_len": 3, "num_features": 2},
        dataset_config={"expert_train": [0.0, 0.5], "expert_val": [0.5, 0.6]},
    )

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert {
        "model_state_dict",
        "optimizer_state_dict",
        "optim_state_dict",
        "epoch",
        "validation_mae",
        "validation_mse",
        "val_mae",
        "val_mse",
        "model_config",
        "dataset_config",
        "best_metrics",
        "scaler_stats",
    }.issubset(checkpoint)
    assert checkpoint["epoch"] == result.best_epoch
    assert checkpoint["validation_mae"] == pytest.approx(result.best_val_mae)
    assert checkpoint["validation_mse"] == pytest.approx(result.best_val_mse)
    assert checkpoint["model_config"]["input_len"] == 8
    assert checkpoint["dataset_config"]["expert_val"] == [0.5, 0.6]
    assert len(result.history) == 2
    assert all("early_stopping_counter" in row for row in result.history)
    assert_experts_frozen(model)


def test_early_stopping_uses_five_validation_mae_misses(tmp_path):
    full_data = np.linspace(0.0, 1.0, 200, dtype=np.float32).reshape(-1, 1)
    train_loader, val_loader = build_expert_dataloaders(
        full_data=full_data,
        input_len=4,
        output_len=2,
        batch_size=32,
    )
    model = TinyForecaster(input_len=4, output_len=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    result = train_expert_model(
        model=model,
        optimizer=optimizer,
        model_name="NoImprovement",
        checkpoint_path=tmp_path / "best_no_improvement.pt",
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=20,
        patience=5,
    )

    assert result.best_epoch == 1
    assert len(result.history) == 6
    assert result.history[-1]["early_stopping_counter"] == 5


def test_router_loaders_are_disjoint_from_expert_loaders():
    full_data = np.arange(2000, dtype=np.float32).reshape(-1, 1)
    expert_train, expert_val = build_expert_dataloaders(
        full_data, input_len=10, output_len=5, batch_size=32
    )
    router_train, router_val = build_router_dataloaders(
        full_data, input_len=10, output_len=5, batch_size=32
    )

    assert expert_train.dataset.boundary.end == expert_val.dataset.boundary.start
    assert expert_val.dataset.boundary.end == router_train.dataset.boundary.start
    assert router_train.dataset.boundary.end == router_val.dataset.boundary.start


def test_router_training_does_not_change_frozen_experts(tmp_path):
    full_data = np.linspace(0.0, 1.0, 200, dtype=np.float32).reshape(-1, 1)
    router_train, router_val = build_router_dataloaders(
        full_data, input_len=4, output_len=2, batch_size=16
    )
    experts = (
        TinyForecaster(input_len=4, output_len=2),
        TinyForecaster(input_len=4, output_len=2),
    )
    for expert in experts:
        expert.eval()
        expert.requires_grad_(False)
    before = [
        {name: value.detach().clone() for name, value in expert.state_dict().items()}
        for expert in experts
    ]

    router = ForecastRouter(
        input_len=4,
        forecast_horizon=2,
        num_features=1,
        representation_size=8,
        hidden_size=8,
        dropout=0.0,
        cnn_channels=4,
        prediction_encoder_dim=4,
    )
    optimizer = torch.optim.Adam(router.parameters(), lr=1e-2)
    history = train_router_model(
        router=router,
        experts=experts,
        optimizer=optimizer,
        train_loader=router_train,
        val_loader=router_val,
        checkpoint_path=tmp_path / "best_router.pt",
        max_epochs=1,
        patience=10,
    )

    assert len(history) == 1
    assert {
        "epoch",
        "smooth_l1_loss",
        "train_mae",
        "validation_mae",
        "validation_mse",
        "average_dlinear_weight",
        "average_transformer_weight",
        "minimum_router_weight",
        "maximum_router_weight",
    }.issubset(history[0])
    checkpoint = torch.load(
        tmp_path / "best_router.pt",
        weights_only=False,
    )
    assert {
        "router_state_dict",
        "optimizer_state_dict",
        "epoch",
        "router_training_loss",
        "validation_mae",
        "validation_mse",
        "average_dlinear_weight",
        "average_transformer_weight",
        "router_config",
        "dataset_config",
        "expert_checkpoint_paths",
    }.issubset(checkpoint)
    for expert, original in zip(experts, before):
        for name, value in expert.state_dict().items():
            assert torch.equal(value, original[name])
        assert all(parameter.grad is None for parameter in expert.parameters())


def test_router_returns_per_step_weights_scores_and_soft_mixture():
    torch.manual_seed(11)
    router = ForecastRouter(
        input_len=96,
        forecast_horizon=12,
        num_features=7,
        representation_size=96,
        hidden_size=32,
        dropout=0.0,
    )
    combined = torch.randn(3, 12, 96)
    dlinear_prediction = torch.randn(3, 12, 7)
    transformer_prediction = torch.randn(3, 12, 7)

    mixed, weights, scores = router(
        combined,
        dlinear_prediction,
        transformer_prediction,
    )

    assert scores.shape == (3, 12, 2)
    assert weights.shape == (3, 12, 2)
    assert mixed.shape == (3, 12, 7)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, 12))
    expected = (
        weights[..., 0, None] * dlinear_prediction
        + weights[..., 1, None] * transformer_prediction
    )
    assert torch.allclose(mixed, expected)


def test_final_router_evaluation_saves_sorted_baseline_comparison(tmp_path):
    full_data = np.linspace(0.0, 1.0, 200, dtype=np.float32).reshape(-1, 1)
    _, router_val = build_router_dataloaders(
        full_data, input_len=4, output_len=2, batch_size=16
    )
    test_loader = build_split_dataloader(
        full_data=full_data,
        split_role="test",
        input_len=4,
        output_len=2,
        batch_size=16,
    )
    experts = (
        TinyForecaster(input_len=4, output_len=2),
        TinyForecaster(input_len=4, output_len=2),
    )
    for expert in experts:
        expert.eval()
        expert.requires_grad_(False)
    router = ForecastRouter(
        input_len=4,
        forecast_horizon=2,
        num_features=1,
        representation_size=8,
        hidden_size=8,
        dropout=0.0,
        cnn_channels=4,
        prediction_encoder_dim=4,
    )
    router.eval()

    results = evaluate_final_router_and_baselines(
        router=router,
        experts=experts,
        router_val_loader=router_val,
        test_loader=test_loader,
        output_dir=tmp_path,
    )

    assert len(results["comparison"]) == 6
    maes = [row["Test MAE"] for row in results["comparison"]]
    assert maes == sorted(maes)
    assert (tmp_path / "router_test_comparison.csv").exists()
    assert (tmp_path / "router_test_metrics.json").exists()
