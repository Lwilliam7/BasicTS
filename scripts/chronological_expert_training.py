"""Leakage-safe chronological data splits and expert-model training helpers."""

import argparse
import csv
import json
from dataclasses import asdict, dataclass, fields, replace
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SPLIT_RANGES = {
    "expert_train": (0.00, 0.50),
    "expert_val": (0.50, 0.60),
    "router_train": (0.60, 0.75),
    "router_val": (0.75, 0.80),
    "test": (0.80, 1.00),
}
SPLIT_ORDER = tuple(SPLIT_RANGES)
DEFAULT_INPUT_LEN = 96
DEFAULT_OUTPUT_LEN = 12
DEFAULT_NUM_FEATURES = 7


@dataclass(frozen=True)
class CandidateExpertSpec:
    """Metadata needed to load a trained candidate expert checkpoint."""

    key: str
    display_name: str
    checkpoint_name: str
    module_name: str
    model_class_name: str
    config_class_name: str


CANDIDATE_EXPERT_SPECS: Dict[str, CandidateExpertSpec] = {
    "dlinear": CandidateExpertSpec(
        key="dlinear",
        display_name="DLinear",
        checkpoint_name="best_dlinear.pt",
        module_name="DLinear",
        model_class_name="DLinear",
        config_class_name="DLinearConfig",
    ),
    "patchtst": CandidateExpertSpec(
        key="patchtst",
        display_name="PatchTST",
        checkpoint_name="best_patchtst.pt",
        module_name="PatchTST",
        model_class_name="PatchTSTForForecasting",
        config_class_name="PatchTSTConfig",
    ),
    "itransformer": CandidateExpertSpec(
        key="itransformer",
        display_name="iTransformer",
        checkpoint_name="best_itransformer.pt",
        module_name="iTransformer",
        model_class_name="iTransformerForForecasting",
        config_class_name="iTransformerConfig",
    ),
    "timesnet": CandidateExpertSpec(
        key="timesnet",
        display_name="TimesNet",
        checkpoint_name="best_timesnet.pt",
        module_name="TimesNet",
        model_class_name="TimesNetForForecasting",
        config_class_name="TimesNetConfig",
    ),
    "moderntcn": CandidateExpertSpec(
        key="moderntcn",
        display_name="ModernTCN",
        checkpoint_name="best_moderntcn.pt",
        module_name="ModernTCN",
        model_class_name="ModernTCNForForecasting",
        config_class_name="ModernTCNConfig",
    ),
}
MODEL_NAME_ALIASES = {
    spec.key: spec.key for spec in CANDIDATE_EXPERT_SPECS.values()
}
MODEL_NAME_ALIASES.update(
    {
        spec.display_name.lower(): spec.key
        for spec in CANDIDATE_EXPERT_SPECS.values()
    }
)
MODEL_NAME_ALIASES["transformer"] = "itransformer"


@dataclass(frozen=True)
class SplitBoundary:
    """Half-open chronological bounds into the full time series."""

    role: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ExpertTrainingResult:
    """The selected checkpoint and the per-epoch metrics used to select it."""

    model_name: str
    checkpoint_path: Path
    best_epoch: int
    best_val_mae: float
    best_val_mse: float
    history: Tuple[dict, ...]


class ForecastRouter(nn.Module):
    """Build per-step representations and softly mix two frozen experts."""

    def __init__(
        self,
        input_len: int = 96,
        forecast_horizon: int = 12,
        num_features: int = 7,
        representation_size: int = 96,
        hidden_size: int = 64,
        dropout: float = 0.1,
        cnn_channels: int = 32,
        prediction_encoder_dim: int = 48,
    ) -> None:
        super().__init__()
        if not 0 < prediction_encoder_dim < representation_size:
            raise ValueError(
                "prediction_encoder_dim must be between 1 and "
                "representation_size - 1"
            )
        if cnn_channels <= 0:
            raise ValueError("cnn_channels must be positive")

        self.input_len = input_len
        self.forecast_horizon = forecast_horizon
        self.num_features = num_features
        self.representation_size = representation_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.cnn_channels = cnn_channels
        self.prediction_encoder_dim = prediction_encoder_dim
        history_size = representation_size - prediction_encoder_dim

        self.history_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=cnn_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(forecast_horizon),
        )
        self.history_projection = nn.Sequential(
            nn.Linear(cnn_channels, history_size),
            nn.GELU(),
        )
        self.prediction_encoder = nn.Sequential(
            nn.Linear(3 * num_features, prediction_encoder_dim),
            nn.GELU(),
        )
        self.routing_head = nn.Sequential(
            nn.Linear(representation_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def build_combined_representation(
        self,
        historical_input: torch.Tensor,
        dlinear_prediction: torch.Tensor,
        transformer_prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Encode history and detached expert forecasts into [B, 12, 96]."""

        batch_size = historical_input.shape[0]
        expected_history = (batch_size, self.input_len, self.num_features)
        expected_forecast = (
            batch_size,
            self.forecast_horizon,
            self.num_features,
        )
        if tuple(historical_input.shape) != expected_history:
            raise ValueError(
                f"historical_input shape {tuple(historical_input.shape)} "
                f"does not match {expected_history}"
            )
        if tuple(dlinear_prediction.shape) != expected_forecast:
            raise ValueError(
                f"DLinear prediction shape {tuple(dlinear_prediction.shape)} "
                f"does not match {expected_forecast}"
            )
        if tuple(transformer_prediction.shape) != expected_forecast:
            raise ValueError(
                "Transformer prediction shape "
                f"{tuple(transformer_prediction.shape)} does not match "
                f"{expected_forecast}"
            )

        history_features = self.history_encoder(
            historical_input.transpose(1, 2)
        ).transpose(1, 2)
        history_features = self.history_projection(
            history_features
        )
        disagreement = torch.abs(
            dlinear_prediction - transformer_prediction
        )
        prediction_features = self.prediction_encoder(
            torch.cat(
                (
                    dlinear_prediction,
                    transformer_prediction,
                    disagreement,
                ),
                dim=-1,
            )
        )
        combined_representation = torch.cat(
            (history_features, prediction_features),
            dim=-1,
        )
        expected_representation = (
            batch_size,
            self.forecast_horizon,
            self.representation_size,
        )
        if tuple(combined_representation.shape) != expected_representation:
            raise ValueError(
                "Combined representation shape "
                f"{tuple(combined_representation.shape)} does not match "
                f"{expected_representation}"
            )
        return combined_representation

    def config_dict(self) -> dict:
        """Serializable router architecture configuration."""

        return {
            "input_len": self.input_len,
            "forecast_horizon": self.forecast_horizon,
            "num_features": self.num_features,
            "representation_size": self.representation_size,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "cnn_channels": self.cnn_channels,
            "prediction_encoder_dim": self.prediction_encoder_dim,
        }

    def forward(
        self,
        combined_representation: torch.Tensor,
        dlinear_prediction: torch.Tensor,
        transformer_prediction: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return mixed forecast, per-step weights, and per-step scores."""

        batch_size = combined_representation.shape[0]
        expected_representation = (
            batch_size,
            self.forecast_horizon,
            self.representation_size,
        )
        expected_forecast = (
            batch_size,
            self.forecast_horizon,
            self.num_features,
        )
        if tuple(combined_representation.shape) != expected_representation:
            raise ValueError(
                "Combined representation shape "
                f"{tuple(combined_representation.shape)} does not match "
                f"{expected_representation}"
            )
        if tuple(dlinear_prediction.shape) != expected_forecast:
            raise ValueError(
                f"DLinear prediction shape {tuple(dlinear_prediction.shape)} "
                f"does not match {expected_forecast}"
            )
        if tuple(transformer_prediction.shape) != expected_forecast:
            raise ValueError(
                "Transformer prediction shape "
                f"{tuple(transformer_prediction.shape)} does not match "
                f"{expected_forecast}"
            )

        router_scores = self.routing_head(combined_representation)
        router_weights = torch.softmax(router_scores, dim=-1)
        dlinear_weight = router_weights[..., 0].unsqueeze(-1)
        transformer_weight = router_weights[..., 1].unsqueeze(-1)
        mixed_prediction = (
            dlinear_weight * dlinear_prediction
            + transformer_weight * transformer_prediction
        )
        return mixed_prediction, router_weights, router_scores


class HorizonQueryTemporalBlock(nn.Module):
    """Length-preserving temporal convolution block for [B, time, channels]."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.temporal_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        residual = sequence
        encoded = self.temporal_conv(sequence.transpose(1, 2)).transpose(1, 2)
        if encoded.shape[1] != residual.shape[1]:
            encoded = encoded[:, : residual.shape[1], :]
        encoded = self.norm(encoded)
        encoded = self.activation(encoded)
        encoded = self.dropout(encoded)
        return residual + encoded


class PredictionAwareRouter(nn.Module):
    """Stage 4 router that scores experts without mixing their predictions."""

    def __init__(
        self,
        input_len: int = DEFAULT_INPUT_LEN,
        forecast_horizon: int = DEFAULT_OUTPUT_LEN,
        num_features: int = DEFAULT_NUM_FEATURES,
        num_experts: int = 2,
        history_channels: int = 64,
        prediction_hidden_size: int = 64,
        prediction_representation_size: int = 32,
        routing_hidden_size: int = 64,
        dropout: float = 0.15,
        history_kernel_size: int = 5,
        history_dilations: Sequence[int] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        self.input_len = input_len
        self.forecast_horizon = forecast_horizon
        self.num_features = num_features
        self.num_experts = num_experts
        self.history_channels = history_channels
        self.prediction_hidden_size = prediction_hidden_size
        self.prediction_representation_size = prediction_representation_size
        self.routing_hidden_size = routing_hidden_size
        self.dropout = dropout
        self.history_kernel_size = history_kernel_size
        self.history_dilations = tuple(history_dilations)
        self.combined_representation_size = (
            history_channels + prediction_representation_size
        )

        self.history_projection = nn.Linear(num_features, history_channels)
        self.history_encoder = nn.Sequential(
            *[
                HorizonQueryTemporalBlock(
                    channels=history_channels,
                    kernel_size=history_kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in self.history_dilations
            ],
        )
        self.horizon_queries = nn.Parameter(
            torch.randn(forecast_horizon, history_channels) * 0.02
        )
        self.prediction_encoder = nn.Sequential(
            nn.Linear((num_experts + 1) * num_features, prediction_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                prediction_hidden_size,
                prediction_representation_size,
            ),
            nn.GELU(),
        )
        self.routing_head = nn.Sequential(
            nn.Linear(
                self.combined_representation_size,
                routing_hidden_size,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(routing_hidden_size, num_features * num_experts),
        )

    def _expected_forecast_shape(
        self,
        batch_size: int,
    ) -> Tuple[int, int, int]:
        return (batch_size, self.forecast_horizon, self.num_features)

    def build_representations(
        self,
        historical_input: torch.Tensor,
        expert_predictions: torch.Tensor,
        disagreement: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return combined representation and named intermediate tensors."""

        batch_size = historical_input.shape[0]
        expected_history = (batch_size, self.input_len, self.num_features)
        expected_forecast = self._expected_forecast_shape(batch_size)
        expert_predictions = self._coerce_expert_predictions(
            expert_predictions,
            batch_size,
        )
        if tuple(historical_input.shape) != expected_history:
            raise AssertionError(
                f"historical input shape {tuple(historical_input.shape)} "
                f"does not match {expected_history}"
            )
        expected_expert_predictions = (
            batch_size,
            self.forecast_horizon,
            self.num_experts,
            self.num_features,
        )
        if tuple(expert_predictions.shape) != expected_expert_predictions:
            raise AssertionError(
                "expert prediction stack shape "
                f"{tuple(expert_predictions.shape)} does not match "
                f"{expected_expert_predictions}"
            )
        if disagreement is not None and tuple(disagreement.shape) != expected_forecast:
            raise AssertionError(
                f"disagreement shape {tuple(disagreement.shape)} does not match "
                f"{expected_forecast}"
            )

        history_input = historical_input.transpose(1, 2)
        history_projected = self.history_projection(historical_input)
        encoded_history = self.history_encoder(history_projected)
        horizon_queries = self.horizon_queries.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        history_attention_scores = torch.matmul(
            horizon_queries,
            encoded_history.transpose(1, 2),
        ) / (self.history_channels ** 0.5)
        history_attention = torch.softmax(history_attention_scores, dim=-1)
        history_representation = torch.matmul(
            history_attention,
            encoded_history,
        )
        flattened_predictions = expert_predictions.reshape(
            batch_size,
            self.forecast_horizon,
            self.num_experts * self.num_features,
        )
        if disagreement is None:
            average_prediction = expert_predictions.mean(dim=2, keepdim=True)
            disagreement = torch.mean(
                torch.abs(expert_predictions - average_prediction),
                dim=2,
            )
        prediction_input = torch.cat((flattened_predictions, disagreement), dim=-1)
        prediction_representation = self.prediction_encoder(prediction_input)
        combined_representation = torch.cat(
            (history_representation, prediction_representation),
            dim=-1,
        )

        expected_history_input = (
            batch_size,
            self.num_features,
            self.input_len,
        )
        expected_history_projected = (
            batch_size,
            self.input_len,
            self.history_channels,
        )
        expected_horizon_queries = (
            batch_size,
            self.forecast_horizon,
            self.history_channels,
        )
        expected_history_attention = (
            batch_size,
            self.forecast_horizon,
            self.input_len,
        )
        expected_history_representation = (
            batch_size,
            self.forecast_horizon,
            self.history_channels,
        )
        expected_prediction_input = (
            batch_size,
            self.forecast_horizon,
            (self.num_experts + 1) * self.num_features,
        )
        expected_prediction_representation = (
            batch_size,
            self.forecast_horizon,
            self.prediction_representation_size,
        )
        expected_combined = (
            batch_size,
            self.forecast_horizon,
            self.combined_representation_size,
        )
        assert tuple(history_input.shape) == expected_history_input
        assert tuple(history_projected.shape) == expected_history_projected
        assert tuple(encoded_history.shape) == expected_history_projected
        assert tuple(horizon_queries.shape) == expected_horizon_queries
        assert tuple(history_attention_scores.shape) == expected_history_attention
        assert tuple(history_attention.shape) == expected_history_attention
        assert torch.allclose(
            history_attention.sum(dim=-1),
            torch.ones(
                batch_size,
                self.forecast_horizon,
                device=history_attention.device,
            ),
            atol=1e-6,
        )
        assert tuple(history_representation.shape) == expected_history_representation
        assert tuple(prediction_input.shape) == expected_prediction_input
        assert tuple(prediction_representation.shape) == (
            expected_prediction_representation
        )
        assert tuple(combined_representation.shape) == expected_combined
        return combined_representation, {
            "history_input": history_input,
            "history_projected": history_projected,
            "encoded_history": encoded_history,
            "horizon_queries": horizon_queries,
            "history_attention_scores": history_attention_scores,
            "history_attention": history_attention,
            "history_representation": history_representation,
            "expert_predictions": expert_predictions,
            "disagreement": disagreement,
            "prediction_input": prediction_input,
            "prediction_representation": prediction_representation,
            "combined_representation": combined_representation,
        }

    def _coerce_expert_predictions(
        self,
        expert_predictions: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Return expert predictions shaped [B, horizon, experts, features]."""

        expected_stacked = (
            batch_size,
            self.forecast_horizon,
            self.num_experts,
            self.num_features,
        )
        if tuple(expert_predictions.shape) == expected_stacked:
            return expert_predictions

        expected_two_expert_legacy = (
            batch_size,
            self.forecast_horizon,
            self.num_features,
        )
        if self.num_experts == 2 and tuple(expert_predictions.shape) == (
            2,
            *expected_two_expert_legacy,
        ):
            return expert_predictions.permute(1, 2, 0, 3)

        raise AssertionError(
            "expert_predictions must have shape "
            f"{expected_stacked}, got {tuple(expert_predictions.shape)}"
        )

    def config_dict(self) -> dict:
        """Serializable router architecture configuration."""

        return {
            "input_len": self.input_len,
            "forecast_horizon": self.forecast_horizon,
            "num_features": self.num_features,
            "num_experts": self.num_experts,
            "history_channels": self.history_channels,
            "prediction_hidden_size": self.prediction_hidden_size,
            "prediction_representation_size": (
                self.prediction_representation_size
            ),
            "routing_hidden_size": self.routing_hidden_size,
            "dropout": self.dropout,
            "history_kernel_size": self.history_kernel_size,
            "history_dilations": list(self.history_dilations),
        }

    def forward(
        self,
        historical_input: torch.Tensor,
        expert_predictions: torch.Tensor,
        transformer_prediction: Optional[torch.Tensor] = None,
        disagreement: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return router scores and softmax weights for the selected experts."""

        if transformer_prediction is not None:
            expert_predictions = torch.stack(
                (expert_predictions, transformer_prediction),
                dim=2,
            )
        combined_representation, _ = self.build_representations(
            historical_input,
            expert_predictions,
            disagreement,
        )
        batch_size = historical_input.shape[0]
        router_scores = self.routing_head(combined_representation).view(
            batch_size,
            self.forecast_horizon,
            self.num_features,
            self.num_experts,
        )
        router_weights = torch.softmax(router_scores, dim=-1)
        expected_scores = (
            batch_size,
            self.forecast_horizon,
            self.num_features,
            self.num_experts,
        )
        assert tuple(router_scores.shape) == expected_scores
        assert tuple(router_weights.shape) == expected_scores
        if not torch.allclose(
            router_weights.sum(dim=-1),
            torch.ones_like(router_weights[..., 0]),
            atol=1e-6,
        ):
            raise AssertionError("Router weights do not sum to 1")
        return router_scores, router_weights


def chronological_split_boundaries(total_length: int) -> Dict[str, SplitBoundary]:
    """Return exact, non-overlapping 50/10/15/5/20 chronological boundaries."""

    if total_length <= 0:
        raise ValueError("total_length must be positive")

    # Derive every start from the preceding end. This makes the half-open
    # sections contiguous even when percentage cut points require rounding.
    cut_points = (
        0,
        int(total_length * 0.50),
        int(total_length * 0.60),
        int(total_length * 0.75),
        int(total_length * 0.80),
        total_length,
    )
    boundaries = {
        role: SplitBoundary(
            role=role,
            start=cut_points[index],
            end=cut_points[index + 1],
        )
        for index, role in enumerate(SPLIT_ORDER)
    }
    return boundaries


def load_full_chronological_data(data_dir: Union[str, Path]) -> np.ndarray:
    """Reconstruct the complete series in time order from BasicTS data files."""

    data_dir = Path(data_dir)
    arrays = [
        np.load(data_dir / f"{mode}_data.npy", allow_pickle=False)
        for mode in ("train", "val", "test")
    ]
    return np.concatenate(arrays, axis=0)


class ChronologicalForecastingDataset(Dataset):
    """Windows wholly contained in one chronological split.

    Keeping every input and target inside its assigned segment prevents a
    validation, router, or test target from entering an expert optimizer step.
    """

    def __init__(
        self,
        full_data: np.ndarray,
        input_len: int,
        output_len: int,
        split_role: str,
    ) -> None:
        if split_role not in SPLIT_RANGES:
            raise ValueError(
                f"Unknown split_role {split_role!r}; expected one of {tuple(SPLIT_RANGES)}"
            )
        if input_len <= 0 or output_len <= 0:
            raise ValueError("input_len and output_len must be positive")

        boundary = chronological_split_boundaries(len(full_data))[split_role]
        minimum_length = input_len + output_len
        if boundary.length < minimum_length:
            raise ValueError(
                f"{split_role} has {boundary.length} time points, but at least "
                f"{minimum_length} are required for one window"
            )

        self.full_data = full_data
        self.input_len = input_len
        self.output_len = output_len
        self.split_role = split_role
        self.boundary = boundary
        self._data = full_data[boundary.start : boundary.end]

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        target_start = index + self.input_len
        return {
            "inputs": self._data[index:target_start],
            "targets": self._data[target_start : target_start + self.output_len],
        }

    def __len__(self) -> int:
        return len(self._data) - self.input_len - self.output_len + 1

    @property
    def data(self) -> np.ndarray:
        return self._data


def build_split_dataloader(
    full_data: np.ndarray,
    split_role: str,
    input_len: int,
    output_len: int,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """Build a non-shuffled loader for exactly one chronological segment."""

    dataset = ChronologicalForecastingDataset(
        full_data=full_data,
        input_len=input_len,
        output_len=output_len,
        split_role=split_role,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def build_all_split_dataloaders(
    full_data: np.ndarray,
    batch_size: int,
    input_len: int = DEFAULT_INPUT_LEN,
    output_len: int = DEFAULT_OUTPUT_LEN,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Create one non-shuffled loader per chronological section."""

    return {
        role: build_split_dataloader(
            full_data=full_data,
            split_role=role,
            input_len=input_len,
            output_len=output_len,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        for role in SPLIT_ORDER
    }


def build_expert_dataloaders(
    full_data: np.ndarray,
    input_len: int,
    output_len: int,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Return only the 50% expert-train and 10% expert-validation loaders."""

    common = {
        "full_data": full_data,
        "input_len": input_len,
        "output_len": output_len,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    return (
        build_split_dataloader(split_role="expert_train", **common),
        build_split_dataloader(split_role="expert_val", **common),
    )


def build_router_dataloaders(
    full_data: np.ndarray,
    input_len: int,
    output_len: int,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Return only the 15% router-train and 5% router-validation loaders."""

    common = {
        "full_data": full_data,
        "input_len": input_len,
        "output_len": output_len,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    return (
        build_split_dataloader(split_role="router_train", **common),
        build_split_dataloader(split_role="router_val", **common),
    )


def build_test_dataloader(
    full_data: np.ndarray,
    input_len: int,
    output_len: int,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """Build the final 20% loader; call this only for final evaluation."""

    return build_split_dataloader(
        full_data=full_data,
        split_role="test",
        input_len=input_len,
        output_len=output_len,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def fit_scaler_on_expert_train(scaler, train_loader: DataLoader):
    """Fit a BasicTS scaler using only the first 50% expert-training data."""

    dataset = train_loader.dataset
    if getattr(dataset, "split_role", None) != "expert_train":
        raise ValueError("Scaler must be fit from the expert_train split")
    scaler.fit(dataset.data)
    return scaler


def print_split_diagnostics(
    loaders: Dict[str, DataLoader],
    scaler=None,
    expected_input_len: int = DEFAULT_INPUT_LEN,
    expected_output_len: int = DEFAULT_OUTPUT_LEN,
    expected_num_features: int = DEFAULT_NUM_FEATURES,
) -> None:
    """Print boundaries, window counts, and one scaled batch shape per split."""

    missing = [role for role in SPLIT_ORDER if role not in loaders]
    if missing:
        raise ValueError(f"Missing chronological loaders: {missing}")

    print("\nChronological split audit (end index is exclusive)")
    print(
        f"{'split':<14} {'start':>9} {'end':>9} "
        f"{'timestamps':>12} {'valid windows':>15}"
    )
    for role in SPLIT_ORDER:
        dataset = loaders[role].dataset
        boundary = dataset.boundary
        print(
            f"{role:<14} {boundary.start:>9} {boundary.end:>9} "
            f"{boundary.length:>12} {len(dataset):>15}"
        )

    print("\nExample batch shapes (the expert-train scaler is reused)")
    for role in SPLIT_ORDER:
        dataset = loaders[role].dataset
        if dataset.data.shape[-1] != expected_num_features:
            raise AssertionError(
                f"{role} has {dataset.data.shape[-1]} features; "
                f"expected {expected_num_features}"
            )
        batch = next(iter(loaders[role]))
        inputs = batch["inputs"].to(dtype=torch.float32)
        targets = batch["targets"].to(dtype=torch.float32)
        if scaler is not None:
            inputs_mask = torch.isfinite(inputs)
            targets_mask = torch.isfinite(targets)
            inputs = scaler.transform(inputs, inputs_mask)
            targets = scaler.transform(targets, targets_mask)
        batch_size = inputs.shape[0]
        assert tuple(inputs.shape[1:]) == (
            expected_input_len,
            expected_num_features,
        ), f"{role} input sample shape is {tuple(inputs.shape[1:])}"
        assert tuple(targets.shape[1:]) == (
            expected_output_len,
            expected_num_features,
        ), f"{role} target sample shape is {tuple(targets.shape[1:])}"
        assert tuple(inputs.shape) == (
            batch_size,
            expected_input_len,
            expected_num_features,
        ), f"{role} batched input shape is {tuple(inputs.shape)}"
        assert tuple(targets.shape) == (
            batch_size,
            expected_output_len,
            expected_num_features,
        ), f"{role} batched target shape is {tuple(targets.shape)}"
        print(
            f"{role:<14} Sample input: {list(inputs.shape[1:])}  "
            f"Sample target: {list(targets.shape[1:])}  "
            f"Input: {list(inputs.shape)}  Target: {list(targets.shape)}"
        )


def prepare_chronological_dataloaders(
    full_data: np.ndarray,
    scaler,
    batch_size: int,
    input_len: int = DEFAULT_INPUT_LEN,
    output_len: int = DEFAULT_OUTPUT_LEN,
    num_workers: int = 0,
    print_diagnostics: bool = True,
) -> Tuple[Dict[str, DataLoader], object]:
    """Build all five splits and fit one scaler on expert training only.

    The returned loaders contain raw chronological windows. Pass the returned
    scaler to every expert/router train or evaluation helper; those helpers
    apply it to both inputs and targets immediately before model use.
    """

    loaders = build_all_split_dataloaders(
        full_data=full_data,
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        num_workers=num_workers,
    )
    fitted_scaler = fit_scaler_on_expert_train(
        scaler,
        loaders["expert_train"],
    )
    if print_diagnostics:
        print_split_diagnostics(loaders, fitted_scaler)
    return loaders, fitted_scaler


def _prediction_tensor(output: Union[torch.Tensor, dict]) -> torch.Tensor:
    if isinstance(output, dict):
        if "prediction" not in output:
            raise KeyError("Model output dictionary does not contain 'prediction'")
        return output["prediction"]
    return output


def _target_for_expert(
    targets: torch.Tensor,
    target_slice: Optional[slice],
) -> torch.Tensor:
    return targets if target_slice is None else targets[:, target_slice, :]


def _check_shapes(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    model_name: str,
) -> None:
    if prediction.shape != targets.shape:
        raise ValueError(
            f"{model_name} prediction shape {tuple(prediction.shape)} does not match "
            f"its target shape {tuple(targets.shape)}"
        )


def _check_validation_shapes(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    prediction: torch.Tensor,
    loader: Iterable[dict],
    model_name: str,
) -> None:
    if inputs.ndim != 3 or targets.ndim != 3 or prediction.ndim != 3:
        raise ValueError(
            f"{model_name} validation tensors must be rank 3; got "
            f"inputs={tuple(inputs.shape)}, targets={tuple(targets.shape)}, "
            f"prediction={tuple(prediction.shape)}"
        )

    dataset = getattr(loader, "dataset", None)
    input_len = getattr(dataset, "input_len", inputs.shape[1])
    output_len = getattr(dataset, "output_len", targets.shape[1])
    num_features = (
        dataset.data.shape[-1]
        if dataset is not None and hasattr(dataset, "data")
        else inputs.shape[-1]
    )
    batch_size = inputs.shape[0]
    expected_inputs = (batch_size, input_len, num_features)
    expected_targets = (batch_size, output_len, num_features)
    if tuple(inputs.shape) != expected_inputs:
        raise ValueError(
            f"{model_name} validation input shape {tuple(inputs.shape)} "
            f"does not match {expected_inputs}"
        )
    if tuple(targets.shape) != expected_targets:
        raise ValueError(
            f"{model_name} validation target shape {tuple(targets.shape)} "
            f"does not match {expected_targets}"
        )
    if tuple(prediction.shape) != expected_targets:
        raise ValueError(
            f"{model_name} validation prediction shape {tuple(prediction.shape)} "
            f"does not match {expected_targets}"
        )


def _accumulate_errors(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    targets_mask: Optional[torch.Tensor] = None,
) -> Tuple[float, float, int]:
    errors = prediction.detach() - targets.detach()
    if targets_mask is not None:
        errors = errors[targets_mask.bool()]
    return (
        torch.abs(errors).sum().item(),
        torch.square(errors).sum().item(),
        errors.numel(),
    )


def _prepare_forecasting_batch(
    batch: dict,
    device: torch.device,
    scaler=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply BasicTS-style masking and scaling to a forecasting batch."""

    inputs = batch["inputs"].to(device=device, dtype=torch.float32)
    targets = batch["targets"].to(device=device, dtype=torch.float32)
    inputs_mask = torch.isfinite(inputs)
    targets_mask = torch.isfinite(targets)
    if scaler is not None:
        inputs = scaler.transform(inputs, inputs_mask)
        targets = scaler.transform(targets, targets_mask)
    inputs = torch.where(inputs_mask, inputs, torch.zeros_like(inputs))
    targets = torch.where(targets_mask, targets, torch.zeros_like(targets))
    return inputs, targets, targets_mask


def _configured_forecasting_loss(
    loss_fn: Optional[Callable],
    prediction: torch.Tensor,
    targets: torch.Tensor,
    targets_mask: torch.Tensor,
) -> torch.Tensor:
    if loss_fn is not None:
        return loss_fn(prediction, targets, targets_mask)
    valid_errors = torch.abs(prediction - targets)[targets_mask]
    if valid_errors.numel() == 0:
        raise ValueError("The batch has no valid forecasting targets")
    return valid_errors.mean()


def evaluate_expert(
    model: nn.Module,
    loader: Iterable[dict],
    model_name: str,
    device: torch.device,
    target_slice: Optional[slice] = None,
    scaler=None,
    print_shapes: bool = True,
) -> Tuple[float, float]:
    """Evaluate every validation window without constructing a backward graph."""

    model.eval()
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    element_count = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            inputs, targets, targets_mask = _prepare_forecasting_batch(
                batch,
                device,
                scaler,
            )
            targets = _target_for_expert(targets, target_slice)
            targets_mask = _target_for_expert(targets_mask, target_slice)
            prediction = _prediction_tensor(model(inputs))
            _check_validation_shapes(
                inputs,
                targets,
                prediction,
                loader,
                model_name,
            )
            if print_shapes and batch_index == 0:
                print(f"\n{model_name} first expert-validation batch")
                print(f"input shape:      {list(inputs.shape)}")
                print(f"target shape:     {list(targets.shape)}")
                print(f"prediction shape: {list(prediction.shape)}")
            abs_sum, squared_sum, count = _accumulate_errors(
                prediction,
                targets,
                targets_mask,
            )
            absolute_error_sum += abs_sum
            squared_error_sum += squared_sum
            element_count += count

    if element_count == 0:
        raise ValueError("Validation loader produced no prediction elements")
    return absolute_error_sum / element_count, squared_error_sum / element_count


def _print_history_table(model_name: str, history: Iterable[dict]) -> None:
    print(f"\n{model_name} training history")
    print(
        f"{'epoch':>5}  {'training MAE':>14}  {'training MSE':>14}  "
        f"{'validation MAE':>16}  {'validation MSE':>16}  "
        f"{'checkpoint saved':>17}  {'early-stop counter':>18}"
    )
    print("-" * 114)
    for row in history:
        print(
            f"{row['epoch']:>5d}  {row['train_mae']:>14.6f}  "
            f"{row['train_mse']:>14.6f}  "
            f"{row['val_mae']:>16.6f}  {row['val_mse']:>16.6f}  "
            f"{str(row['checkpoint_saved']):>17}  "
            f"{row['early_stopping_counter']:>18d}"
        )


def _print_reloaded_expert_batch_shapes(
    model: nn.Module,
    loader: Iterable[dict],
    model_name: str,
    device: torch.device,
    target_slice: Optional[slice] = None,
    scaler=None,
) -> None:
    """Print the first-batch contract after the selected checkpoint is loaded."""

    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        inputs, targets, _ = _prepare_forecasting_batch(
            batch,
            device,
            scaler,
        )
        targets = _target_for_expert(targets, target_slice)
        prediction = _prediction_tensor(model(inputs))
    _check_shapes(prediction, targets, model_name)
    print(f"\n{model_name} reloaded checkpoint first batch")
    print(f"input shape:      {list(inputs.shape)}")
    print(f"target shape:     {list(targets.shape)}")
    print(f"prediction shape: {list(prediction.shape)}")


def _frozen_expert_predictions(
    experts: Sequence[nn.Module],
    inputs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(experts) != 2:
        raise ValueError("Per-step router training requires exactly two experts")
    with torch.no_grad():
        dlinear_prediction, transformer_prediction = [
            _prediction_tensor(expert(inputs)).detach()
            for expert in experts
        ]
    return dlinear_prediction, transformer_prediction


def evaluate_router_pipeline(
    router: ForecastRouter,
    experts: Sequence[nn.Module],
    loader: Iterable[dict],
    device: Union[str, torch.device] = "cpu",
    scaler=None,
    print_shapes: bool = False,
) -> dict:
    """Evaluate the complete frozen-expert/router pipeline without updates."""

    device = torch.device(device)
    assert_experts_frozen(*experts)
    router.eval()
    for expert in experts:
        expert.to(device)
        expert.eval()

    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    smooth_l1_sum = 0.0
    dlinear_absolute_error_sum = 0.0
    transformer_absolute_error_sum = 0.0
    element_count = 0
    dlinear_weight_sum = 0.0
    transformer_weight_sum = 0.0
    router_weight_count = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            inputs, targets, targets_mask = _prepare_forecasting_batch(
                batch,
                device,
                scaler,
            )
            dlinear_prediction, transformer_prediction = (
                _frozen_expert_predictions(experts, inputs)
            )
            disagreement = torch.abs(
                dlinear_prediction - transformer_prediction
            )
            combined_representation = router.build_combined_representation(
                inputs,
                dlinear_prediction,
                transformer_prediction,
            )
            mixed_prediction, router_weights, router_scores = router(
                combined_representation,
                dlinear_prediction,
                transformer_prediction,
            )

            batch_size = inputs.shape[0]
            expected_input = (
                batch_size,
                router.input_len,
                router.num_features,
            )
            expected_forecast = (
                batch_size,
                router.forecast_horizon,
                router.num_features,
            )
            expected_weights = (
                batch_size,
                router.forecast_horizon,
                2,
            )
            if tuple(inputs.shape) != expected_input:
                raise ValueError("Invalid router-evaluation input shape")
            for name, tensor in (
                ("target", targets),
                ("DLinear prediction", dlinear_prediction),
                ("Transformer prediction", transformer_prediction),
                ("disagreement", disagreement),
                ("mixed prediction", mixed_prediction),
            ):
                if tuple(tensor.shape) != expected_forecast:
                    raise ValueError(
                        f"Invalid router-evaluation {name} shape"
                    )
            if tuple(router_scores.shape) != expected_weights:
                raise ValueError("Invalid router-evaluation score shape")
            if tuple(router_weights.shape) != expected_weights:
                raise ValueError("Invalid router-evaluation weight shape")
            if not torch.allclose(
                router_weights.sum(dim=-1),
                torch.ones_like(router_weights[..., 0]),
                atol=1e-6,
                rtol=1e-6,
            ):
                raise ValueError("Router evaluation weights do not sum to one")

            if print_shapes and batch_index == 0:
                print("\nFirst router-validation batch")
                print("Input shape:", list(inputs.shape))
                print("Target shape:", list(targets.shape))
                print(
                    "DLinear prediction shape:",
                    list(dlinear_prediction.shape),
                )
                print(
                    "Transformer prediction shape:",
                    list(transformer_prediction.shape),
                )
                print("Router-weight shape:", list(router_weights.shape))
                print(
                    "Mixed-prediction shape:",
                    list(mixed_prediction.shape),
                )

            abs_sum, squared_sum, count = _accumulate_errors(
                mixed_prediction,
                targets,
                targets_mask,
            )
            dlinear_abs_sum, _, _ = _accumulate_errors(
                dlinear_prediction,
                targets,
                targets_mask,
            )
            transformer_abs_sum, _, _ = _accumulate_errors(
                transformer_prediction,
                targets,
                targets_mask,
            )
            valid_mixed = mixed_prediction[targets_mask]
            valid_targets = targets[targets_mask]
            smooth_l1_sum += nn.functional.smooth_l1_loss(
                valid_mixed,
                valid_targets,
                reduction="sum",
            ).item()
            absolute_error_sum += abs_sum
            squared_error_sum += squared_sum
            dlinear_absolute_error_sum += dlinear_abs_sum
            transformer_absolute_error_sum += transformer_abs_sum
            element_count += count
            dlinear_weight_sum += router_weights[..., 0].sum().item()
            transformer_weight_sum += router_weights[..., 1].sum().item()
            router_weight_count += router_weights[..., 0].numel()

    if element_count == 0 or router_weight_count == 0:
        raise ValueError("Router evaluation loader produced no elements")
    if any(
        parameter.grad is not None
        for expert in experts
        for parameter in expert.parameters()
    ):
        raise RuntimeError("A frozen expert has gradients after evaluation")

    return {
        "mae": absolute_error_sum / element_count,
        "mse": squared_error_sum / element_count,
        "smooth_l1_loss": smooth_l1_sum / element_count,
        "average_dlinear_weight": dlinear_weight_sum / router_weight_count,
        "average_transformer_weight": (
            transformer_weight_sum / router_weight_count
        ),
        "dlinear_mae": dlinear_absolute_error_sum / element_count,
        "transformer_mae": transformer_absolute_error_sum / element_count,
    }


def train_router_model(
    router: ForecastRouter,
    experts: Sequence[nn.Module],
    optimizer: torch.optim.Optimizer,
    train_loader: Iterable[dict],
    val_loader: Iterable[dict],
    checkpoint_path: Union[str, Path],
    max_epochs: int,
    patience: int = 10,
    device: Union[str, torch.device] = "cpu",
    scaler=None,
    router_config: Optional[dict] = None,
    dataset_config: Optional[dict] = None,
    expert_checkpoint_paths: Optional[dict] = None,
) -> Tuple[dict, ...]:
    """Train on 15%, select on 5%, then reload the best soft router."""

    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if patience <= 0:
        raise ValueError("patience must be positive")

    device = torch.device(device)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    assert_experts_frozen(*experts)
    expert_parameter_ids = {
        id(parameter)
        for expert in experts
        for parameter in expert.parameters()
    }
    router_parameter_ids = {id(parameter) for parameter in router.parameters()}
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_parameter_ids & expert_parameter_ids:
        raise ValueError("Router optimizer contains frozen expert parameters")
    if not optimizer_parameter_ids:
        raise ValueError("Router optimizer has no parameters")
    if not optimizer_parameter_ids.issubset(router_parameter_ids):
        raise ValueError("Router optimizer contains non-router parameters")

    for expert in experts:
        expert.to(device)
        expert.eval()
        for parameter in expert.parameters():
            parameter.grad = None
    router.to(device)
    frozen_expert_states = [
        {
            name: value.detach().clone()
            for name, value in expert.state_dict().items()
        }
        for expert in experts
    ]

    history = []
    loss_function = nn.SmoothL1Loss()
    best_validation_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        router.train()
        smooth_l1_sum = 0.0
        absolute_error_sum = 0.0
        element_count = 0
        dlinear_weight_sum = 0.0
        transformer_weight_sum = 0.0
        router_weight_count = 0
        minimum_router_weight = float("inf")
        maximum_router_weight = float("-inf")

        for batch_index, batch in enumerate(train_loader):
            inputs, targets, targets_mask = _prepare_forecasting_batch(
                batch,
                device,
                scaler,
            )
            dlinear_prediction, transformer_prediction = (
                _frozen_expert_predictions(experts, inputs)
            )

            optimizer.zero_grad(set_to_none=True)
            combined_representation = router.build_combined_representation(
                inputs,
                dlinear_prediction,
                transformer_prediction,
            )
            mixed_prediction, router_weights, router_scores = router(
                combined_representation,
                dlinear_prediction,
                transformer_prediction,
            )

            batch_size = inputs.shape[0]
            expected_representation = (
                batch_size,
                router.forecast_horizon,
                router.representation_size,
            )
            expected_scores = (batch_size, router.forecast_horizon, 2)
            expected_forecast = (
                batch_size,
                router.forecast_horizon,
                router.num_features,
            )
            disagreement = torch.abs(
                dlinear_prediction - transformer_prediction
            )
            dlinear_weight = router_weights[..., 0].unsqueeze(-1)
            transformer_weight = router_weights[..., 1].unsqueeze(-1)
            if tuple(combined_representation.shape) != expected_representation:
                raise ValueError("Invalid combined representation shape")
            if tuple(router_scores.shape) != expected_scores:
                raise ValueError("Invalid router-score shape")
            if tuple(router_weights.shape) != expected_scores:
                raise ValueError("Invalid router-weight shape")
            if tuple(dlinear_weight.shape) != (
                batch_size,
                router.forecast_horizon,
                1,
            ):
                raise ValueError("Invalid DLinear-weight shape")
            if tuple(transformer_weight.shape) != (
                batch_size,
                router.forecast_horizon,
                1,
            ):
                raise ValueError("Invalid Transformer-weight shape")
            if tuple(disagreement.shape) != expected_forecast:
                raise ValueError("Invalid disagreement shape")
            if tuple(mixed_prediction.shape) != expected_forecast:
                raise ValueError("Invalid mixed-prediction shape")
            if tuple(targets.shape) != expected_forecast:
                raise ValueError("Invalid router target shape")
            weight_sums = router_weights.sum(dim=-1)
            if not torch.allclose(
                weight_sums,
                torch.ones_like(weight_sums),
                atol=1e-6,
                rtol=1e-6,
            ):
                raise ValueError("Per-step expert weights do not sum to one")

            loss = loss_function(mixed_prediction, targets)
            loss.backward()

            if not any(
                parameter.grad is not None
                for parameter in router.parameters()
            ):
                raise RuntimeError("No router parameter received a gradient")
            for component_name, component in (
                ("history encoder", router.history_encoder),
                ("prediction encoder", router.prediction_encoder),
                ("routing head", router.routing_head),
            ):
                if not any(
                    parameter.grad is not None
                    for parameter in component.parameters()
                ):
                    raise RuntimeError(
                        f"The router {component_name} received no gradient"
                    )
            if any(
                parameter.grad is not None
                for expert in experts
                for parameter in expert.parameters()
            ):
                raise RuntimeError("A frozen expert received a gradient")

            if epoch == 1 and batch_index == 0:
                print("\nFirst router-training batch")
                print(
                    "Combined representation shape:",
                    list(combined_representation.shape),
                )
                print("Router-score shape:", list(router_scores.shape))
                print("Router-weight shape:", list(router_weights.shape))
                print("DLinear-weight shape:", list(dlinear_weight.shape))
                print(
                    "Transformer-weight shape:",
                    list(transformer_weight.shape),
                )
                print(
                    "Mixed-prediction shape:",
                    list(mixed_prediction.shape),
                )
                print("Target shape:", list(targets.shape))
                print(f"Training loss: {loss.item():.6f}")
                print(
                    "DLinear parameter gradients enabled:",
                    any(
                        parameter.requires_grad
                        for parameter in experts[0].parameters()
                    ),
                )
                print(
                    "Transformer parameter gradients enabled:",
                    any(
                        parameter.requires_grad
                        for parameter in experts[1].parameters()
                    ),
                )

            optimizer.step()

            abs_sum, _, count = _accumulate_errors(
                mixed_prediction,
                targets,
                targets_mask,
            )
            smooth_l1_sum += loss.detach().item() * count
            absolute_error_sum += abs_sum
            element_count += count
            dlinear_weight_sum += router_weights[..., 0].detach().sum().item()
            transformer_weight_sum += (
                router_weights[..., 1].detach().sum().item()
            )
            router_weight_count += router_weights[..., 0].numel()
            minimum_router_weight = min(
                minimum_router_weight,
                router_weights.detach().min().item(),
            )
            maximum_router_weight = max(
                maximum_router_weight,
                router_weights.detach().max().item(),
            )

        if element_count == 0:
            raise ValueError("Router training loader produced no prediction elements")

        for expert, original_state in zip(experts, frozen_expert_states):
            for name, value in expert.state_dict().items():
                if not torch.equal(value, original_state[name]):
                    raise RuntimeError(
                        f"Frozen expert state changed for parameter {name}"
                    )
        router_training_loss = smooth_l1_sum / element_count
        train_mae = absolute_error_sum / element_count
        average_dlinear_weight = dlinear_weight_sum / router_weight_count
        average_transformer_weight = (
            transformer_weight_sum / router_weight_count
        )

        validation = evaluate_router_pipeline(
            router=router,
            experts=experts,
            loader=val_loader,
            device=device,
            scaler=scaler,
            print_shapes=(epoch == 1),
        )
        checkpoint_saved = validation["mae"] < best_validation_mae
        if checkpoint_saved:
            best_validation_mae = validation["mae"]
            best_epoch = epoch
            epochs_without_improvement = 0
            saved_router_config = router.config_dict()
            saved_router_config.update(router_config or {})
            torch.save(
                {
                    "router_state_dict": router.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "router_training_loss": router_training_loss,
                    "validation_mae": validation["mae"],
                    "validation_mse": validation["mse"],
                    "validation_smooth_l1_loss": validation[
                        "smooth_l1_loss"
                    ],
                    "average_dlinear_weight": average_dlinear_weight,
                    "average_transformer_weight": (
                        average_transformer_weight
                    ),
                    "validation_average_dlinear_weight": validation[
                        "average_dlinear_weight"
                    ],
                    "validation_average_transformer_weight": validation[
                        "average_transformer_weight"
                    ],
                    "router_config": saved_router_config,
                    "dataset_config": dict(dataset_config or {}),
                    "expert_checkpoint_paths": {
                        name: str(path)
                        for name, path in (
                            expert_checkpoint_paths or {}
                        ).items()
                    },
                    **(
                        {"scaler_stats": scaler.stats}
                        if scaler is not None
                        else {}
                    ),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "smooth_l1_loss": router_training_loss,
                "train_mae": train_mae,
                "validation_mae": validation["mae"],
                "validation_mse": validation["mse"],
                "validation_smooth_l1_loss": validation[
                    "smooth_l1_loss"
                ],
                "average_dlinear_weight": average_dlinear_weight,
                "average_transformer_weight": average_transformer_weight,
                "minimum_router_weight": minimum_router_weight,
                "maximum_router_weight": maximum_router_weight,
                "checkpoint_saved": checkpoint_saved,
                "early_stopping_counter": epochs_without_improvement,
            }
        )
        print(
            f"Router epoch {epoch:>3d}/{max_epochs}: "
            f"Smooth L1={router_training_loss:.6f}, "
            f"train MAE={train_mae:.6f}, "
            f"validation MAE={validation['mae']:.6f}, "
            f"validation MSE={validation['mse']:.6f}, "
            f"avg DLinear weight={average_dlinear_weight:.4f}, "
            f"avg Transformer weight={average_transformer_weight:.4f}, "
            f"weight range=[{minimum_router_weight:.4f}, "
            f"{maximum_router_weight:.4f}], "
            f"checkpoint saved={checkpoint_saved}, "
            f"early-stop counter={epochs_without_improvement}/{patience}"
        )
        if epochs_without_improvement >= patience:
            print(
                f"Router: early stopping after epoch {epoch} "
                f"({patience} epochs without lower validation MAE)."
            )
            break

    if best_epoch == 0:
        raise RuntimeError(
            "Router never produced a finite validation MAE; "
            "no checkpoint was saved"
        )
    try:
        selected = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        selected = torch.load(checkpoint_path, map_location=device)
    router.load_state_dict(selected["router_state_dict"])
    optimizer.load_state_dict(selected["optimizer_state_dict"])
    router.eval()
    for expert in experts:
        expert.eval()
    assert_experts_frozen(*experts)
    print(
        f"\nSelected Router epoch {selected['epoch']}: "
        f"validation MAE={selected['validation_mae']:.6f}, "
        f"validation MSE={selected['validation_mse']:.6f}"
    )
    return tuple(history)


def load_best_router_checkpoint(
    router: ForecastRouter,
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """Load the validation-selected router checkpoint for final evaluation."""

    device = torch.device(device)
    try:
        checkpoint = torch.load(
            Path(checkpoint_path),
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    router.load_state_dict(checkpoint["router_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    router.to(device)
    router.eval()
    print(
        f"Loaded Router epoch {checkpoint['epoch']}: "
        f"validation MAE={checkpoint['validation_mae']:.6f}, "
        f"validation MSE={checkpoint['validation_mse']:.6f}"
    )
    return checkpoint


def evaluate_final_router_and_baselines(
    router: ForecastRouter,
    experts: Sequence[nn.Module],
    router_val_loader: Iterable[dict],
    test_loader: Iterable[dict],
    output_dir: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    scaler=None,
) -> dict:
    """Evaluate the selected router and validation-derived baselines once."""

    if getattr(router_val_loader.dataset, "split_role", None) != "router_val":
        raise ValueError("Fixed baseline weights require router_val data")
    if getattr(test_loader.dataset, "split_role", None) != "test":
        raise ValueError("Final evaluation requires the untouched test split")

    device = torch.device(device)
    router.to(device)
    router.eval()
    assert_experts_frozen(*experts)
    for expert in experts:
        expert.to(device)
        expert.eval()

    validation = evaluate_router_pipeline(
        router=router,
        experts=experts,
        loader=router_val_loader,
        device=device,
        scaler=scaler,
    )
    epsilon = 1e-6
    inverse_dlinear = 1.0 / (validation["dlinear_mae"] + epsilon)
    inverse_transformer = 1.0 / (
        validation["transformer_mae"] + epsilon
    )
    inverse_total = inverse_dlinear + inverse_transformer
    fixed_dlinear_weight = inverse_dlinear / inverse_total
    fixed_transformer_weight = inverse_transformer / inverse_total
    globally_best_expert = (
        "DLinear"
        if validation["dlinear_mae"] <= validation["transformer_mae"]
        else "Transformer"
    )

    method_names = (
        "DLinear alone",
        "Transformer alone",
        "Fixed equal average",
        "Fixed validation-based soft weights",
        "Globally best validation-selected expert",
        "Learned prediction-aware router",
    )
    totals = {
        name: {"absolute": 0.0, "squared": 0.0, "count": 0}
        for name in method_names
    }
    per_step_absolute = torch.zeros(
        router.forecast_horizon,
        dtype=torch.float64,
    )
    per_step_count = torch.zeros(
        router.forecast_horizon,
        dtype=torch.float64,
    )
    per_step_dlinear_weight = torch.zeros(
        router.forecast_horizon,
        dtype=torch.float64,
    )
    per_step_transformer_weight = torch.zeros(
        router.forecast_horizon,
        dtype=torch.float64,
    )
    per_step_weight_count = 0
    router_state_before = {
        name: value.detach().clone()
        for name, value in router.state_dict().items()
    }
    expert_states_before = [
        {
            name: value.detach().clone()
            for name, value in expert.state_dict().items()
        }
        for expert in experts
    ]

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets, targets_mask = _prepare_forecasting_batch(
                batch,
                device,
                scaler,
            )
            dlinear_prediction, transformer_prediction = (
                _frozen_expert_predictions(experts, inputs)
            )
            combined_representation = router.build_combined_representation(
                inputs,
                dlinear_prediction,
                transformer_prediction,
            )
            router_prediction, router_weights, _ = router(
                combined_representation,
                dlinear_prediction,
                transformer_prediction,
            )
            equal_prediction = (
                0.5 * dlinear_prediction
                + 0.5 * transformer_prediction
            )
            fixed_soft_prediction = (
                fixed_dlinear_weight * dlinear_prediction
                + fixed_transformer_weight * transformer_prediction
            )
            globally_best_prediction = (
                dlinear_prediction
                if globally_best_expert == "DLinear"
                else transformer_prediction
            )
            predictions = {
                "DLinear alone": dlinear_prediction,
                "Transformer alone": transformer_prediction,
                "Fixed equal average": equal_prediction,
                "Fixed validation-based soft weights": (
                    fixed_soft_prediction
                ),
                "Globally best validation-selected expert": (
                    globally_best_prediction
                ),
                "Learned prediction-aware router": router_prediction,
            }
            for name, prediction in predictions.items():
                _check_shapes(prediction, targets, name)
                abs_sum, squared_sum, count = _accumulate_errors(
                    prediction,
                    targets,
                    targets_mask,
                )
                totals[name]["absolute"] += abs_sum
                totals[name]["squared"] += squared_sum
                totals[name]["count"] += count

            router_absolute = torch.abs(
                router_prediction - targets
            ).detach()
            mask = targets_mask.detach().to(router_absolute.dtype)
            per_step_absolute += (
                (router_absolute * mask)
                .sum(dim=(0, 2))
                .cpu()
                .to(torch.float64)
            )
            per_step_count += (
                mask.sum(dim=(0, 2)).cpu().to(torch.float64)
            )
            per_step_dlinear_weight += (
                router_weights[..., 0]
                .sum(dim=0)
                .cpu()
                .to(torch.float64)
            )
            per_step_transformer_weight += (
                router_weights[..., 1]
                .sum(dim=0)
                .cpu()
                .to(torch.float64)
            )
            per_step_weight_count += router_weights.shape[0]

    for name, value in router.state_dict().items():
        if not torch.equal(value, router_state_before[name]):
            raise RuntimeError(f"Router changed during test evaluation: {name}")
    for expert, original_state in zip(experts, expert_states_before):
        for name, value in expert.state_dict().items():
            if not torch.equal(value, original_state[name]):
                raise RuntimeError(
                    f"Expert changed during test evaluation: {name}"
                )

    comparison = []
    for name in method_names:
        values = totals[name]
        mae = values["absolute"] / values["count"]
        mse = values["squared"] / values["count"]
        comparison.append(
            {
                "Method": name,
                "Test MAE": mae,
                "Test MSE": mse,
                "Test RMSE": mse ** 0.5,
            }
        )
    comparison.sort(key=lambda row: row["Test MAE"])

    router_row = next(
        row
        for row in comparison
        if row["Method"] == "Learned prediction-aware router"
    )
    baseline_rows = [
        row
        for row in comparison
        if row["Method"] != "Learned prediction-aware router"
    ]
    best_baseline = min(
        baseline_rows,
        key=lambda row: row["Test MAE"],
    )
    improvement = best_baseline["Test MAE"] - router_row["Test MAE"]
    percentage_improvement = (
        improvement / best_baseline["Test MAE"] * 100.0
    )
    router_average_dlinear_weight = (
        per_step_dlinear_weight.sum().item()
        / (per_step_weight_count * router.forecast_horizon)
    )
    router_average_transformer_weight = (
        per_step_transformer_weight.sum().item()
        / (per_step_weight_count * router.forecast_horizon)
    )
    router_details = {
        "average_dlinear_weight": router_average_dlinear_weight,
        "average_transformer_weight": (
            router_average_transformer_weight
        ),
        "per_step_mae": (
            per_step_absolute / per_step_count
        ).tolist(),
        "per_step_average_dlinear_weight": (
            per_step_dlinear_weight / per_step_weight_count
        ).tolist(),
        "per_step_average_transformer_weight": (
            per_step_transformer_weight / per_step_weight_count
        ).tolist(),
    }
    results = {
        "comparison": comparison,
        "router_details": router_details,
        "validation_only_baseline_selection": {
            "dlinear_validation_mae": validation["dlinear_mae"],
            "transformer_validation_mae": validation["transformer_mae"],
            "fixed_dlinear_weight": fixed_dlinear_weight,
            "fixed_transformer_weight": fixed_transformer_weight,
            "globally_best_expert": globally_best_expert,
        },
        "best_baseline": best_baseline,
        "learned_router": router_row,
        "absolute_mae_improvement": improvement,
        "percentage_mae_improvement": percentage_improvement,
        "router_improved_over_best_baseline": improvement > 0,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "router_test_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("Method", "Test MAE", "Test MSE", "Test RMSE"),
        )
        writer.writeheader()
        writer.writerows(comparison)
    json_path = output_dir / "router_test_metrics.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nFinal test comparison (sorted by MAE)")
    print(
        f"{'Method':<45} {'Test MAE':>12} "
        f"{'Test MSE':>12} {'Test RMSE':>12}"
    )
    print("-" * 85)
    for row in comparison:
        print(
            f"{row['Method']:<45} "
            f"{row['Test MAE']:>12.6f} "
            f"{row['Test MSE']:>12.6f} "
            f"{row['Test RMSE']:>12.6f}"
        )
    print(f"\nBest baseline: {best_baseline['Method']}")
    print(
        "Learned router result: "
        f"MAE={router_row['Test MAE']:.6f}"
    )
    print(f"Absolute MAE improvement: {improvement:.6f}")
    print(f"Percentage MAE improvement: {percentage_improvement:.2f}%")
    print(
        "Router improved over the best baseline"
        if improvement > 0
        else "Router did not improve over the best baseline"
    )
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    return results


def load_and_freeze_expert(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler=None,
) -> dict:
    """Reload the selected on-disk checkpoint and freeze the expert."""

    device = torch.device(device)
    checkpoint_path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        # PyTorch versions before weights_only was added.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer_state = checkpoint.get(
            "optimizer_state_dict",
            checkpoint.get("optim_state_dict"),
        )
        if optimizer_state is None:
            raise KeyError("Checkpoint has no optimizer state dictionary")
        optimizer.load_state_dict(optimizer_state)
    if scaler is not None and "scaler_stats" in checkpoint:
        scaler.stats = checkpoint["scaler_stats"]
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return checkpoint


def train_expert_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_name: str,
    checkpoint_path: Union[str, Path],
    train_loader: Iterable[dict],
    val_loader: Iterable[dict],
    max_epochs: int,
    patience: int = 5,
    device: Union[str, torch.device] = "cpu",
    target_slice: Optional[slice] = None,
    scaler=None,
    loss_fn: Optional[Callable] = None,
    model_config: Optional[dict] = None,
    dataset_config: Optional[dict] = None,
) -> ExpertTrainingResult:
    """Train, validation-select, reload, and freeze one forecasting expert."""

    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if patience <= 0:
        raise ValueError("patience must be positive")

    device = torch.device(device)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.to(device)

    history = []
    best_val_mae = float("inf")
    best_val_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_absolute_error_sum = 0.0
        train_squared_error_sum = 0.0
        train_element_count = 0

        for batch_index, batch in enumerate(train_loader):
            inputs, targets, targets_mask = _prepare_forecasting_batch(
                batch,
                device,
                scaler,
            )
            targets = _target_for_expert(targets, target_slice)
            targets_mask = _target_for_expert(targets_mask, target_slice)

            optimizer.zero_grad(set_to_none=True)
            prediction = _prediction_tensor(model(inputs))
            _check_shapes(prediction, targets, model_name)
            if epoch == 1 and batch_index == 0:
                print(f"\n{model_name} first expert-training batch")
                print(f"input shape:      {list(inputs.shape)}")
                print(f"target shape:     {list(targets.shape)}")
                print(f"prediction shape: {list(prediction.shape)}")
            loss = _configured_forecasting_loss(
                loss_fn,
                prediction,
                targets,
                targets_mask,
            )
            loss.backward()
            optimizer.step()

            abs_sum, squared_sum, count = _accumulate_errors(
                prediction,
                targets,
                targets_mask,
            )
            train_absolute_error_sum += abs_sum
            train_squared_error_sum += squared_sum
            train_element_count += count

        if train_element_count == 0:
            raise ValueError("Expert training loader produced no prediction elements")

        train_mae = train_absolute_error_sum / train_element_count
        train_mse = train_squared_error_sum / train_element_count
        val_mae, val_mse = evaluate_expert(
            model=model,
            loader=val_loader,
            model_name=model_name,
            device=device,
            target_slice=target_slice,
            scaler=scaler,
            print_shapes=(epoch == 1),
        )

        checkpoint_saved = val_mae < best_val_mae
        if checkpoint_saved:
            best_val_mae = val_mae
            best_val_mse = val_mse
            best_epoch = epoch
            optimizer_state = optimizer.state_dict()
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer_state,
                    "optim_state_dict": optimizer_state,
                    "epoch": epoch,
                    "validation_mae": val_mae,
                    "validation_mse": val_mse,
                    "val_mae": val_mae,
                    "val_mse": val_mse,
                    "model_config": dict(model_config or {}),
                    "dataset_config": dict(dataset_config or {}),
                    "best_metrics": {
                        "val/MAE": val_mae,
                        "val/MSE": val_mse,
                    },
                    **({"scaler_stats": scaler.stats} if scaler is not None else {}),
                },
                checkpoint_path,
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_mae": train_mae,
                "train_mse": train_mse,
                "val_mae": val_mae,
                "val_mse": val_mse,
                "checkpoint_saved": checkpoint_saved,
                "early_stopping_counter": epochs_without_improvement,
            }
        )
        print(
            f"{model_name} epoch {epoch:>3d}/{max_epochs}: "
            f"training MAE={train_mae:.6f}, training MSE={train_mse:.6f}, "
            f"validation MAE={val_mae:.6f}, validation MSE={val_mse:.6f}, "
            f"early-stop counter={epochs_without_improvement}/{patience}"
        )

        if epochs_without_improvement >= patience:
            print(
                f"{model_name}: early stopping after epoch {epoch} "
                f"({patience} epochs without lower validation MAE)."
            )
            break

    if best_epoch == 0:
        raise RuntimeError(
            f"{model_name} never produced a finite validation MAE; no checkpoint was saved"
        )

    selected = load_and_freeze_expert(
        model=model,
        optimizer=optimizer,
        checkpoint_path=checkpoint_path,
        device=device,
        scaler=scaler,
    )
    _print_reloaded_expert_batch_shapes(
        model=model,
        loader=train_loader,
        model_name=model_name,
        device=device,
        target_slice=target_slice,
        scaler=scaler,
    )
    _print_history_table(model_name, history)
    selected_validation_mae = (
        selected["validation_mae"]
        if "validation_mae" in selected
        else selected["val_mae"]
    )
    selected_validation_mse = (
        selected["validation_mse"]
        if "validation_mse" in selected
        else selected["val_mse"]
    )
    print(
        f"\nSelected {model_name} epoch {selected['epoch']}: "
        f"validation MAE={selected_validation_mae:.6f}, "
        f"validation MSE={selected_validation_mse:.6f}"
    )

    return ExpertTrainingResult(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        best_epoch=best_epoch,
        best_val_mae=best_val_mae,
        best_val_mse=best_val_mse,
        history=tuple(history),
    )


def assert_experts_frozen(*experts: nn.Module) -> None:
    """Fail before router training if an expert is trainable or in train mode."""

    for expert in experts:
        if expert.training:
            raise RuntimeError("An expert is still in training mode")
        if any(parameter.requires_grad for parameter in expert.parameters()):
            raise RuntimeError("An expert still has trainable parameters")


def _ensure_local_src_importable() -> None:
    """Allow this file to run as `python scripts/chronological_expert_training.py`."""

    import sys

    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _assert_full_data_contract(
    full_data: np.ndarray,
    expected_num_features: int,
) -> None:
    if full_data.ndim != 2:
        raise AssertionError(
            f"Expected full data with shape [timestamps, features], got {full_data.shape}"
        )
    if full_data.shape[-1] != expected_num_features:
        raise AssertionError(
            f"Expected {expected_num_features} features, got {full_data.shape[-1]}"
        )


def _dataset_config_summary(total_timestamps: int) -> dict:
    boundaries = chronological_split_boundaries(total_timestamps)
    return {
        role: {
            "fraction": list(SPLIT_RANGES[role]),
            "start": boundaries[role].start,
            "end": boundaries[role].end,
            "timestamps": boundaries[role].length,
            "windows": (
                boundaries[role].length
                - DEFAULT_INPUT_LEN
                - DEFAULT_OUTPUT_LEN
                + 1
            ),
        }
        for role in SPLIT_ORDER
    }


def run_chronological_expert_stages(
    data_dir: Union[str, Path] = "datasets/ETTh1",
    output_dir: Union[str, Path] = "checkpoints",
    batch_size: int = 256,
    max_epochs: int = 50,
    patience: int = 5,
    learning_rate: float = 1e-3,
    device: Union[str, torch.device] = "cpu",
    seed: int = 7,
) -> Tuple[ExpertTrainingResult, ExpertTrainingResult]:
    """Run Stage 1 splits and Stage 2 expert training, without router training."""

    _ensure_local_src_importable()
    from basicts.metrics import masked_mse
    from basicts.models.DLinear import DLinear, DLinearConfig
    from basicts.models.iTransformer import (
        iTransformerConfig,
        iTransformerForForecasting,
    )
    from basicts.scaler import ZScoreScaler

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_data = load_full_chronological_data(data_dir)
    _assert_full_data_contract(full_data, DEFAULT_NUM_FEATURES)
    print(f"Loaded chronological data from {data_dir}: {list(full_data.shape)}")

    loaders, scaler = prepare_chronological_dataloaders(
        full_data=full_data,
        scaler=ZScoreScaler(norm_each_channel=True, rescale=False),
        batch_size=batch_size,
        input_len=DEFAULT_INPUT_LEN,
        output_len=DEFAULT_OUTPUT_LEN,
    )
    dataset_config = _dataset_config_summary(len(full_data))

    dlinear_config = DLinearConfig(
        input_len=DEFAULT_INPUT_LEN,
        output_len=DEFAULT_OUTPUT_LEN,
        num_features=DEFAULT_NUM_FEATURES,
        individual=False,
    )
    transformer_config = iTransformerConfig(
        input_len=DEFAULT_INPUT_LEN,
        output_len=DEFAULT_OUTPUT_LEN,
        num_features=DEFAULT_NUM_FEATURES,
        hidden_size=64,
        n_heads=4,
        intermediate_size=128,
        num_layers=1,
        dropout=0.1,
        use_revin=False,
    )

    dlinear = DLinear(dlinear_config)
    transformer = iTransformerForForecasting(transformer_config)
    dlinear_optimizer = torch.optim.Adam(dlinear.parameters(), lr=learning_rate)
    transformer_optimizer = torch.optim.Adam(
        transformer.parameters(),
        lr=learning_rate,
    )

    common_training_args = {
        "train_loader": loaders["expert_train"],
        "val_loader": loaders["expert_val"],
        "max_epochs": max_epochs,
        "patience": patience,
        "device": device,
        "scaler": scaler,
        "loss_fn": masked_mse,
        "dataset_config": dataset_config,
    }
    dlinear_result = train_expert_model(
        model=dlinear,
        optimizer=dlinear_optimizer,
        model_name="DLinear",
        checkpoint_path=output_dir / "best_dlinear.pt",
        model_config=asdict(dlinear_config),
        **common_training_args,
    )
    transformer_result = train_expert_model(
        model=transformer,
        optimizer=transformer_optimizer,
        model_name="Transformer",
        checkpoint_path=output_dir / "best_transformer.pt",
        model_config={
            "architecture": "iTransformerForForecasting",
            **asdict(transformer_config),
        },
        **common_training_args,
    )

    # Confirm both selected checkpoints can be loaded into fresh model instances.
    fresh_dlinear = DLinear(dlinear_config)
    fresh_transformer = iTransformerForForecasting(transformer_config)
    fresh_dlinear_optimizer = torch.optim.Adam(
        fresh_dlinear.parameters(),
        lr=learning_rate,
    )
    fresh_transformer_optimizer = torch.optim.Adam(
        fresh_transformer.parameters(),
        lr=learning_rate,
    )
    dlinear_checkpoint = load_and_freeze_expert(
        model=fresh_dlinear,
        optimizer=fresh_dlinear_optimizer,
        checkpoint_path=dlinear_result.checkpoint_path,
        device=device,
        scaler=scaler,
    )
    transformer_checkpoint = load_and_freeze_expert(
        model=fresh_transformer,
        optimizer=fresh_transformer_optimizer,
        checkpoint_path=transformer_result.checkpoint_path,
        device=device,
        scaler=scaler,
    )
    print("\nCheckpoint reload verification")
    print(
        "DLinear: "
        f"epoch={dlinear_checkpoint['epoch']}, "
        f"validation MAE={dlinear_checkpoint['validation_mae']:.6f}, "
        f"validation MSE={dlinear_checkpoint['validation_mse']:.6f}"
    )
    print(
        "Transformer: "
        f"epoch={transformer_checkpoint['epoch']}, "
        f"validation MAE={transformer_checkpoint['validation_mae']:.6f}, "
        f"validation MSE={transformer_checkpoint['validation_mse']:.6f}"
    )
    print("\nRouter training was not run.")

    return dlinear_result, transformer_result


def _load_torch_checkpoint(
    checkpoint_path: Union[str, Path],
    device: torch.device,
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def _dataclass_config_kwargs(config_class, config_values: dict) -> dict:
    valid_names = {field.name for field in fields(config_class)}
    return {
        name: value
        for name, value in config_values.items()
        if name in valid_names
    }


def _normalize_model_key(model_name: str) -> str:
    key = MODEL_NAME_ALIASES.get(str(model_name).strip().lower())
    if key is None:
        valid = ", ".join(
            spec.display_name for spec in CANDIDATE_EXPERT_SPECS.values()
        )
        raise ValueError(
            f"Unknown router model {model_name!r}. Valid models: {valid}"
        )
    return key


def _specs_from_model_names(
    model_names: Sequence[str],
) -> Tuple[CandidateExpertSpec, ...]:
    if not model_names:
        raise ValueError("At least one router model must be selected")

    keys = [_normalize_model_key(model_name) for model_name in model_names]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        names = ", ".join(CANDIDATE_EXPERT_SPECS[key].display_name for key in duplicates)
        raise ValueError(f"Selected router models contain duplicates: {names}")
    return tuple(CANDIDATE_EXPERT_SPECS[key] for key in keys)


def _candidate_name_from_result(value: str) -> str:
    name = str(value).strip()
    if name.startswith("Candidate_"):
        name = name[len("Candidate_") :]
    return name


def _model_names_from_combination_result(value: str) -> Tuple[str, ...]:
    return tuple(
        _candidate_name_from_result(part)
        for part in str(value).split("+")
        if part.strip()
    )


def _resolve_results_path(path_value: Union[str, Path]) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    search_roots = (
        Path.cwd(),
        Path(__file__).resolve().parents[1],
        Path(__file__).resolve().parents[1].parent,
    )
    for root in search_roots:
        candidates = [root / path]
        if path.parts and path.parts[0] == "results":
            candidates.append(
                root / "results" / "router_summary" / Path(*path.parts[1:])
            )
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return path


def _read_combination_result_rows(results_path: Union[str, Path]) -> Tuple[dict, ...]:
    results_path = Path(results_path)
    if results_path.suffix.lower() == ".json":
        with results_path.open("r", encoding="utf-8") as file:
            rows = json.load(file)
        if not isinstance(rows, list):
            raise ValueError(f"{results_path} must contain a JSON list of rows")
        return tuple(dict(row) for row in rows)

    with results_path.open("r", newline="", encoding="utf-8") as file:
        return tuple(csv.DictReader(file))


def _normalize_best_model_counts(
    model_counts: Union[str, Sequence[int]],
    rows: Sequence[dict],
) -> Tuple[int, ...]:
    if isinstance(model_counts, str):
        if model_counts.lower() not in {"all", "auto"}:
            raise ValueError(
                "BEST_MODEL_COUNTS must be 'all', 'auto', or a sequence of integers"
            )
        discovered_counts = sorted(
            {
                int(row.get("number_of_experts", 0))
                for row in rows
                if int(row.get("number_of_experts", 0)) > 0
            }
        )
        if any(row.get("best_single_expert") for row in rows):
            discovered_counts = sorted({1, *discovered_counts})
        if not discovered_counts:
            raise ValueError("No model counts were found in the combination results")
        return tuple(discovered_counts)
    return tuple(int(count) for count in model_counts)


def _best_saved_model_groups_from_results(
    results_path: Union[str, Path],
    model_counts: Union[str, Sequence[int]],
) -> Tuple[Tuple[str, ...], ...]:
    results_path = _resolve_results_path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(
            f"Best-by-size model selection requires {results_path}. "
            "Run notebooks/modeltest.ipynb first or set "
            "AUTO_SELECT_BEST_BY_SIZE = False in scripts/router_model_config.py."
        )

    rows = list(_read_combination_result_rows(results_path))
    if not rows:
        raise ValueError(f"No rows found in {results_path}")

    selected_groups = []
    for model_count in _normalize_best_model_counts(model_counts, rows):
        model_count = int(model_count)
        if model_count <= 0:
            raise ValueError("BEST_MODEL_COUNTS must contain positive integers")

        if model_count == 1:
            single_candidates = []
            for row in rows:
                expert_name = row.get("best_single_expert")
                expert_mae = row.get("best_single_mae")
                if expert_name and expert_mae not in (None, ""):
                    single_candidates.append(
                        (
                            float(expert_mae),
                            (_candidate_name_from_result(expert_name),),
                        )
                    )
            matching_rows = [
                row
                for row in rows
                if int(row.get("number_of_experts", 0)) == 1
            ]
            for row in matching_rows:
                single_candidates.append(
                    (
                        float(row["oracle_mae"]),
                        _model_names_from_combination_result(row["combination"]),
                    )
                )
            if not single_candidates:
                raise ValueError(
                    f"{results_path} does not contain enough information to "
                    "select the best one-model expert"
                )
            _, group = min(single_candidates, key=lambda item: item[0])
        else:
            matching_rows = [
                row
                for row in rows
                if int(row.get("number_of_experts", 0)) == model_count
            ]
            if not matching_rows:
                raise ValueError(
                    f"{results_path} has no saved combination with "
                    f"{model_count} experts"
                )
            best_row = min(
                matching_rows,
                key=lambda row: float(row["oracle_mae"]),
            )
            group = _model_names_from_combination_result(best_row["combination"])

        if len(group) != model_count:
            raise ValueError(
                f"Best {model_count}-model group resolved to {group}, "
                f"which has {len(group)} models"
            )
        _specs_from_model_names(group)
        selected_groups.append(group)

    return tuple(selected_groups)


def selected_router_model_groups() -> Tuple[Tuple[CandidateExpertSpec, ...], ...]:
    """Return one or more selected expert groups from router_model_config.py."""

    try:
        from scripts.router_experiment_config import (
            load_router_experiment_config,
            validate_router_experiment_config,
        )
    except ImportError:
        try:
            from router_experiment_config import (
                load_router_experiment_config,
                validate_router_experiment_config,
            )
        except ImportError as exc:
            raise ImportError(
                "Could not import scripts/router_experiment_config.py."
            ) from exc

    config = validate_router_experiment_config(
        load_router_experiment_config(),
        require_checkpoints=False,
        require_data=False,
        require_cache_parent=False,
    )
    auto_select = config.auto_select_best_by_size
    if auto_select:
        model_groups = _best_saved_model_groups_from_results(
            config.best_combination_results_path,
            config.best_model_counts,
        )
    else:
        if config.selected_model_groups:
            model_groups = tuple(tuple(group) for group in config.selected_model_groups)
        else:
            model_groups = (config.selected_expert_models,)

    return tuple(_specs_from_model_names(group) for group in model_groups)


def selected_router_model_specs(
    model_names: Optional[Sequence[str]] = None,
) -> Tuple[CandidateExpertSpec, ...]:
    """Return the largest selected expert group as candidate expert specs."""

    if model_names is not None:
        return _specs_from_model_names(model_names)
    return selected_router_model_groups()[-1]


def selected_candidate_expert_specs(
    checkpoint_dir: Union[str, Path] = "checkpoints",
    model_names: Optional[Sequence[str]] = None,
) -> Tuple[dict, ...]:
    """Return notebook-friendly selected candidate checkpoint metadata."""

    checkpoint_dir = Path(checkpoint_dir)
    try:
        from scripts.router_experiment_config import load_router_experiment_config
    except ImportError:
        from router_experiment_config import load_router_experiment_config

    config = load_router_experiment_config()
    configured_paths = {
        str(name): Path(path)
        for name, path in config.expert_checkpoint_paths.items()
    }
    return tuple(
        {
            "expert_name": f"Candidate_{spec.display_name}",
            "display_name": spec.display_name,
            "key": spec.key,
            "module_name": spec.module_name,
            "model_class_name": spec.model_class_name,
            "config_class_name": spec.config_class_name,
            "checkpoint_path": configured_paths.get(
                spec.display_name,
                configured_paths.get(
                    spec.key,
                    checkpoint_dir / "candidates" / spec.checkpoint_name,
                ),
            ),
        }
        for spec in selected_router_model_specs(model_names)
    )


def _candidate_checkpoint_path_from_config(
    spec: CandidateExpertSpec,
    checkpoint_dir: Union[str, Path],
) -> Path:
    try:
        from scripts.router_experiment_config import load_router_experiment_config
    except ImportError:
        from router_experiment_config import load_router_experiment_config

    config = load_router_experiment_config()
    configured_paths = {
        str(name): Path(path)
        for name, path in config.expert_checkpoint_paths.items()
    }
    return configured_paths.get(
        spec.display_name,
        configured_paths.get(
            spec.key,
            Path(checkpoint_dir) / "candidates" / spec.checkpoint_name,
        ),
    )


def _validated_runtime_router_config(
    data_dir: Union[str, Path],
    checkpoint_dir: Union[str, Path],
    seed: int,
    *,
    require_checkpoints: bool = True,
):
    try:
        from scripts.router_experiment_config import (
            load_router_experiment_config,
            print_router_experiment_config,
            validate_router_experiment_config,
        )
    except ImportError:
        from router_experiment_config import (
            load_router_experiment_config,
            print_router_experiment_config,
            validate_router_experiment_config,
        )

    config = load_router_experiment_config()
    config = replace(
        config,
        data_dir=str(data_dir),
        checkpoint_dir=str(checkpoint_dir),
        random_seed=int(seed),
    )
    validate_router_experiment_config(
        config,
        require_checkpoints=require_checkpoints,
        require_data=True,
        require_cache_parent=True,
    )
    print_router_experiment_config(config)
    if config.debug_mode:
        expected_input = (config.input_length, config.num_features)
        expected_target = (config.forecast_horizon, config.num_features)
        print("\nDebug shape expectations")
        print(f"  history window: [B, {expected_input[0]}, {expected_input[1]}]")
        print(f"  forecast target: [B, {expected_target[0]}, {expected_target[1]}]")
    return config


def run_router_config_check_stage(
    data_dir: Union[str, Path] = "datasets/ETTh1",
    checkpoint_dir: Union[str, Path] = "checkpoints",
    seed: int = 7,
) -> None:
    """Validate the central frozen-expert router experiment configuration."""

    _validated_runtime_router_config(
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        require_checkpoints=True,
    )
    print("\nRouter config sanity check passed.")


def _build_config_from_checkpoint(
    config_class,
    checkpoint: dict,
    extra_config: Optional[dict] = None,
):
    values = {
        "input_len": DEFAULT_INPUT_LEN,
        "output_len": DEFAULT_OUTPUT_LEN,
        "label_len": DEFAULT_INPUT_LEN // 2,
        "num_features": DEFAULT_NUM_FEATURES,
        "num_classes": None,
    }
    if isinstance(checkpoint.get("model_config"), dict):
        values.update(checkpoint["model_config"])
    values.update(extra_config or {})
    return config_class(**_dataclass_config_kwargs(config_class, values))


def _call_expert_model(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    try:
        output = model(inputs)
    except TypeError:
        try:
            output = model(inputs, None)
        except TypeError:
            output = model(inputs, targets)
    return _prediction_tensor(output)


def _load_candidate_expert_from_spec(
    spec: CandidateExpertSpec,
    checkpoint_dir: Union[str, Path],
    device: torch.device,
    scaler,
) -> Tuple[nn.Module, dict, Path]:
    checkpoint_path = _candidate_checkpoint_path_from_config(spec, checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Selected model {spec.display_name} requires missing checkpoint "
            f"{checkpoint_path}. Train it with scripts/train_candidate_experts.py."
        )
    checkpoint = _load_torch_checkpoint(checkpoint_path, device)
    module = import_module(f"basicts.models.{spec.module_name}")
    model_class = getattr(module, spec.model_class_name)
    config_class = getattr(module, spec.config_class_name)
    config = _build_config_from_checkpoint(config_class, checkpoint)
    model = model_class(config)
    checkpoint = load_and_freeze_expert(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        scaler=scaler,
    )
    model.eval()
    assert_experts_frozen(model)
    return model, checkpoint, checkpoint_path


def build_selected_candidate_experts(
    checkpoint_dir: Union[str, Path],
    device: torch.device,
    scaler,
    specs: Optional[Sequence[CandidateExpertSpec]] = None,
) -> Tuple[Tuple[nn.Module, ...], Tuple[str, ...], Dict[str, dict], Dict[str, Path]]:
    """Load and freeze only the models selected in router_model_config.py."""

    _ensure_local_src_importable()
    experts = []
    names = []
    checkpoints = {}
    checkpoint_paths = {}
    for spec in tuple(specs or selected_router_model_specs()):
        model, checkpoint, checkpoint_path = _load_candidate_expert_from_spec(
            spec,
            checkpoint_dir=checkpoint_dir,
            device=device,
            scaler=scaler,
        )
        experts.append(model)
        names.append(spec.display_name)
        checkpoints[spec.display_name] = checkpoint
        checkpoint_paths[spec.display_name] = checkpoint_path
    assert_experts_frozen(*experts)
    return tuple(experts), tuple(names), checkpoints, checkpoint_paths


def _router_model_group_name(specs: Sequence[CandidateExpertSpec]) -> str:
    count = len(specs)
    suffix = "expert" if count == 1 else "experts"
    return f"best_{count}_{suffix}"


def _router_checkpoint_path(
    checkpoint_dir: Union[str, Path],
    group_name: str,
    multiple_groups: bool,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    if multiple_groups:
        return checkpoint_dir / f"best_router_{group_name}.pt"
    return checkpoint_dir / "best_router.pt"


def _router_output_dir(
    output_dir: Union[str, Path],
    group_name: str,
    multiple_groups: bool,
) -> Path:
    output_dir = Path(output_dir)
    if multiple_groups:
        return output_dir / group_name
    return output_dir


def _build_stage3_experts(
    dlinear_checkpoint_path: Union[str, Path],
    transformer_checkpoint_path: Union[str, Path],
    device: torch.device,
    scaler,
) -> Tuple[nn.Module, nn.Module, dict, dict]:
    _ensure_local_src_importable()
    from basicts.models.DLinear import DLinear, DLinearConfig
    from basicts.models.iTransformer import (
        iTransformerConfig,
        iTransformerForForecasting,
    )

    dlinear_checkpoint = _load_torch_checkpoint(
        dlinear_checkpoint_path,
        device,
    )
    transformer_checkpoint = _load_torch_checkpoint(
        transformer_checkpoint_path,
        device,
    )
    dlinear_config = DLinearConfig(
        **_dataclass_config_kwargs(
            DLinearConfig,
            dlinear_checkpoint["model_config"],
        )
    )
    transformer_config = iTransformerConfig(
        **_dataclass_config_kwargs(
            iTransformerConfig,
            transformer_checkpoint["model_config"],
        )
    )

    dlinear = DLinear(dlinear_config)
    transformer = iTransformerForForecasting(transformer_config)
    dlinear_checkpoint = load_and_freeze_expert(
        model=dlinear,
        checkpoint_path=dlinear_checkpoint_path,
        device=device,
        scaler=scaler,
    )
    transformer_checkpoint = load_and_freeze_expert(
        model=transformer,
        checkpoint_path=transformer_checkpoint_path,
        device=device,
        scaler=scaler,
    )

    for parameter in dlinear.parameters():
        parameter.requires_grad = False
    for parameter in transformer.parameters():
        parameter.requires_grad = False
    dlinear.eval()
    transformer.eval()
    assert_experts_frozen(dlinear, transformer)
    return dlinear, transformer, dlinear_checkpoint, transformer_checkpoint


def _assert_no_expert_gradients(*experts: nn.Module) -> None:
    for expert in experts:
        for name, parameter in expert.named_parameters():
            if parameter.requires_grad:
                raise AssertionError(f"{name} still has requires_grad=True")
            if parameter.grad is not None:
                raise AssertionError(f"{name} received a gradient")


def _assert_router_optimizer_excludes_experts(
    optimizer: torch.optim.Optimizer,
    *experts: nn.Module,
) -> None:
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expert_parameter_ids = {
        id(parameter)
        for expert in experts
        for parameter in expert.parameters()
    }
    overlap = optimizer_parameter_ids & expert_parameter_ids
    if overlap:
        raise AssertionError("Router optimizer includes expert parameters")


def _new_expert_weight_summary_state(
    expert_names: Sequence[str],
    forecast_horizon: int,
    num_features: int,
) -> dict:
    return {
        "sample_count": 0,
        "overall": {expert_name: 0.0 for expert_name in expert_names},
        "per_step": {
            expert_name: torch.zeros(forecast_horizon, dtype=torch.float64)
            for expert_name in expert_names
        },
        "per_variable": {
            expert_name: torch.zeros(num_features, dtype=torch.float64)
            for expert_name in expert_names
        },
        "per_step_variable": {
            expert_name: torch.zeros(
                forecast_horizon,
                num_features,
                dtype=torch.float64,
            )
            for expert_name in expert_names
        },
    }


def _accumulate_expert_weight_summary(
    state: dict,
    router_weights: torch.Tensor,
    expert_names: Sequence[str],
) -> None:
    if router_weights.ndim != 4:
        raise AssertionError(
            "router_weights must have shape [batch, horizon, features, experts], "
            f"got {tuple(router_weights.shape)}"
        )
    batch_size, _, _, num_experts = router_weights.shape
    if num_experts != len(expert_names):
        raise AssertionError(
            f"router_weights has {num_experts} experts but got "
            f"{len(expert_names)} expert names"
        )
    state["sample_count"] += batch_size
    weights = router_weights.detach().cpu().to(torch.float64)
    for expert_index, expert_name in enumerate(expert_names):
        expert_weights = weights[..., expert_index]
        state["overall"][expert_name] += expert_weights.sum().item()
        state["per_step"][expert_name] += expert_weights.sum(dim=(0, 2))
        state["per_variable"][expert_name] += expert_weights.sum(dim=(0, 1))
        state["per_step_variable"][expert_name] += expert_weights.sum(dim=0)


def _finalize_expert_weight_summary(
    state: dict,
    expert_names: Sequence[str],
    forecast_horizon: int,
    num_features: int,
) -> dict:
    sample_count = state["sample_count"]
    if sample_count <= 0:
        raise ValueError("Cannot summarize router weights without samples")
    return {
        "average_expert_weights": {
            expert_name: (
                state["overall"][expert_name]
                / (sample_count * forecast_horizon * num_features)
            )
            for expert_name in expert_names
        },
        "average_expert_weights_by_step": {
            expert_name: (
                state["per_step"][expert_name]
                / (sample_count * num_features)
            ).tolist()
            for expert_name in expert_names
        },
        "average_expert_weights_by_variable": {
            expert_name: (
                state["per_variable"][expert_name]
                / (sample_count * forecast_horizon)
            ).tolist()
            for expert_name in expert_names
        },
        "average_expert_weights_by_step_and_variable": {
            expert_name: (
                state["per_step_variable"][expert_name] / sample_count
            ).tolist()
            for expert_name in expert_names
        },
    }


def run_router_input_and_forward_stage(
    data_dir: Union[str, Path] = "datasets/ETTh1",
    checkpoint_dir: Union[str, Path] = "checkpoints",
    batch_size: int = 512,
    device: Union[str, torch.device] = "cpu",
    seed: int = 7,
) -> dict:
    """Run Stage 3 and Stage 4 checks without training or mixing."""

    _ensure_local_src_importable()
    from basicts.scaler import ZScoreScaler

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)
    data_dir = Path(data_dir)
    checkpoint_dir = Path(checkpoint_dir)
    runtime_config = _validated_runtime_router_config(
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        require_checkpoints=True,
    )

    full_data = load_full_chronological_data(data_dir)
    _assert_full_data_contract(full_data, runtime_config.num_features)
    loaders, scaler = prepare_chronological_dataloaders(
        full_data=full_data,
        scaler=ZScoreScaler(norm_each_channel=True, rescale=False),
        batch_size=batch_size,
        input_len=runtime_config.input_length,
        output_len=runtime_config.forecast_horizon,
    )

    experts, expert_names, expert_checkpoints, _ = (
        build_selected_candidate_experts(
            checkpoint_dir=checkpoint_dir,
            device=device,
            scaler=scaler,
        )
    )
    print("\nLoaded and froze experts")
    for expert_name, checkpoint, expert in zip(expert_names, expert_checkpoints.values(), experts):
        print(
            f"{expert_name} checkpoint: "
            f"epoch={checkpoint.get('epoch')}, "
            f"validation MAE={checkpoint.get('validation_mae', checkpoint.get('val_mae')):.6f}, "
            f"validation MSE={checkpoint.get('validation_mse', checkpoint.get('val_mse')):.6f}"
        )
        print(
            f"{expert_name} parameter gradients enabled: "
            f"{any(parameter.requires_grad for parameter in expert.parameters())}"
        )

    router_train_loader = loaders["router_train"]
    if getattr(router_train_loader.dataset, "split_role", None) != "router_train":
        raise AssertionError("Stage 3 must use the chronological router_train split")
    batch = next(iter(router_train_loader))
    x, target, _ = _prepare_forecasting_batch(batch, device, scaler)
    expected_x = (
        x.shape[0],
        runtime_config.input_length,
        runtime_config.num_features,
    )
    expected_target = (
        x.shape[0],
        runtime_config.forecast_horizon,
        runtime_config.num_features,
    )
    assert tuple(x.shape) == expected_x
    assert tuple(target.shape) == expected_target

    with torch.no_grad():
        expert_predictions = torch.stack(
            [_call_expert_model(expert, x, target).detach() for expert in experts],
            dim=2,
        )
    disagreement = torch.mean(
        torch.abs(expert_predictions - expert_predictions.mean(dim=2, keepdim=True)),
        dim=2,
    )
    assert tuple(expert_predictions.shape) == (
        x.shape[0],
        DEFAULT_OUTPUT_LEN,
        len(experts),
        DEFAULT_NUM_FEATURES,
    )
    assert tuple(disagreement.shape) == expected_target
    assert not expert_predictions.requires_grad
    assert not disagreement.requires_grad

    print("\nStage 3 first router-training batch")
    print(f"x shape:                      {list(x.shape)}")
    print(f"target shape:                 {list(target.shape)}")
    print(f"expert prediction stack shape:{list(expert_predictions.shape)}")
    for index, expert_name in enumerate(expert_names):
        print(
            f"{expert_name} prediction shape: "
            f"{list(expert_predictions[:, :, index, :].shape)}"
        )
    print(f"disagreement shape:           {list(disagreement.shape)}")

    router = PredictionAwareRouter(num_experts=len(experts)).to(device)
    router.eval()
    router_optimizer = torch.optim.Adam(router.parameters(), lr=1e-3)
    _assert_router_optimizer_excludes_experts(router_optimizer, *experts)
    combined_representation, intermediates = router.build_representations(
        x,
        expert_predictions,
        disagreement=disagreement,
    )
    assert tuple(intermediates["history_input"].shape) == (
        x.shape[0],
        DEFAULT_NUM_FEATURES,
        DEFAULT_INPUT_LEN,
    )
    assert tuple(intermediates["history_projected"].shape) == (
        x.shape[0],
        DEFAULT_INPUT_LEN,
        router.history_channels,
    )
    assert tuple(intermediates["encoded_history"].shape) == (
        x.shape[0],
        DEFAULT_INPUT_LEN,
        router.history_channels,
    )
    assert tuple(intermediates["horizon_queries"].shape) == (
        x.shape[0],
        DEFAULT_OUTPUT_LEN,
        router.history_channels,
    )
    assert tuple(intermediates["history_attention"].shape) == (
        x.shape[0],
        DEFAULT_OUTPUT_LEN,
        DEFAULT_INPUT_LEN,
    )
    assert torch.allclose(
        intermediates["history_attention"].sum(dim=-1),
        torch.ones(
            x.shape[0],
            DEFAULT_OUTPUT_LEN,
            device=intermediates["history_attention"].device,
        ),
        atol=1e-6,
    )
    assert tuple(intermediates["history_representation"].shape) == (
        x.shape[0],
        DEFAULT_OUTPUT_LEN,
        router.history_channels,
    )
    assert tuple(intermediates["prediction_representation"].shape) == (
        x.shape[0],
        DEFAULT_OUTPUT_LEN,
        router.prediction_representation_size,
    )
    assert tuple(combined_representation.shape) == (
        x.shape[0],
        DEFAULT_OUTPUT_LEN,
        router.combined_representation_size,
    )
    with torch.no_grad():
        router_scores, router_weights = router(
            x,
            expert_predictions,
            disagreement=disagreement,
        )
    assert tuple(router_scores.shape) == (
        x.shape[0],
        DEFAULT_OUTPUT_LEN,
        DEFAULT_NUM_FEATURES,
        len(experts),
    )
    assert tuple(router_weights.shape) == (
        x.shape[0],
        DEFAULT_OUTPUT_LEN,
        DEFAULT_NUM_FEATURES,
        len(experts),
    )
    assert torch.allclose(
        router_weights.sum(dim=-1),
        torch.ones(
            x.shape[0],
            DEFAULT_OUTPUT_LEN,
            DEFAULT_NUM_FEATURES,
            device=router_weights.device,
        ),
        atol=1e-6,
    )

    print("\nStage 4 prediction-aware router forward pass")
    print(f"history_input shape:                  {list(intermediates['history_input'].shape)}")
    print(f"history_projected shape:              {list(intermediates['history_projected'].shape)}")
    print(f"encoded_history shape:                {list(intermediates['encoded_history'].shape)}")
    print(f"horizon_queries shape:                {list(intermediates['horizon_queries'].shape)}")
    print(f"history_attention shape:              {list(intermediates['history_attention'].shape)}")
    print(
        "history_representation shape:         "
        f"{list(intermediates['history_representation'].shape)}"
    )
    print(f"prediction_input shape:               {list(intermediates['prediction_input'].shape)}")
    print(
        "prediction_representation shape:      "
        f"{list(intermediates['prediction_representation'].shape)}"
    )
    print(f"combined_representation shape:        {list(combined_representation.shape)}")
    print(f"router_scores shape:                  {list(router_scores.shape)}")
    print(f"router_weights shape:                 {list(router_weights.shape)}")
    print(
        "router weight sum range:              "
        f"{router_weights.sum(dim=-1).min().item():.6f} to "
        f"{router_weights.sum(dim=-1).max().item():.6f}"
    )
    print(
        "history attention sum range:          "
        f"{intermediates['history_attention'].sum(dim=-1).min().item():.6f} to "
        f"{intermediates['history_attention'].sum(dim=-1).max().item():.6f}"
    )

    _assert_no_expert_gradients(*experts)
    print("\nConfirmed: neither expert received gradients.")
    print("Router was not trained, and predictions were not mixed.")
    return {
        "x_shape": tuple(x.shape),
        "target_shape": tuple(target.shape),
        "selected_experts": tuple(expert_names),
        "expert_prediction_stack_shape": tuple(expert_predictions.shape),
        "disagreement_shape": tuple(disagreement.shape),
        "combined_representation_shape": tuple(combined_representation.shape),
        "router_scores_shape": tuple(router_scores.shape),
        "router_weights_shape": tuple(router_weights.shape),
        "expert_gradients_present": any(
            parameter.grad is not None
            for expert in experts
            for parameter in expert.parameters()
        ),
        "router_config": router.config_dict(),
    }


def _prediction_aware_router_forward(
    router: PredictionAwareRouter,
    experts: Sequence[nn.Module],
    inputs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return frozen expert predictions, disagreement, weights, and mixture."""

    if len(experts) != router.num_experts:
        raise ValueError(
            f"Router expects {router.num_experts} experts, got {len(experts)}"
        )
    with torch.no_grad():
        expert_predictions = torch.stack(
            [_call_expert_model(expert, inputs).detach() for expert in experts],
            dim=2,
        )
    disagreement = torch.mean(
        torch.abs(expert_predictions - expert_predictions.mean(dim=2, keepdim=True)),
        dim=2,
    )
    router_scores, router_weights = router(
        inputs,
        expert_predictions,
        disagreement=disagreement,
    )
    mixed_prediction = torch.sum(
        router_weights * expert_predictions.permute(0, 1, 3, 2),
        dim=-1,
    )

    batch_size = inputs.shape[0]
    expected_forecast = (
        batch_size,
        router.forecast_horizon,
        router.num_features,
    )
    expected_stack = (
        batch_size,
        router.forecast_horizon,
        router.num_experts,
        router.num_features,
    )
    expected_weights = (
        batch_size,
        router.forecast_horizon,
        router.num_features,
        router.num_experts,
    )
    expected_weight_sums = (
        batch_size,
        router.forecast_horizon,
        router.num_features,
    )
    assert tuple(expert_predictions.shape) == expected_stack
    assert tuple(disagreement.shape) == expected_forecast
    assert tuple(router_scores.shape) == expected_weights
    assert tuple(router_weights.shape) == expected_weights
    assert tuple(mixed_prediction.shape) == expected_forecast
    if not torch.allclose(
        router_weights.sum(dim=-1),
        torch.ones(expected_weight_sums, device=router_weights.device),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise AssertionError("Router weights do not sum to 1 per step/variable")
    return (
        expert_predictions,
        disagreement,
        router_scores,
        router_weights,
        mixed_prediction,
    )


def evaluate_prediction_aware_router_pipeline(
    router: PredictionAwareRouter,
    experts: Sequence[nn.Module],
    loader: Iterable[dict],
    device: Union[str, torch.device] = "cpu",
    scaler=None,
    print_shapes: bool = False,
    expert_names: Optional[Sequence[str]] = None,
) -> dict:
    """Evaluate the frozen-expert prediction-aware router without updates."""

    device = torch.device(device)
    expert_names = tuple(
        expert_names or [f"Expert {index + 1}" for index in range(len(experts))]
    )
    if len(expert_names) != len(experts):
        raise ValueError("expert_names must match the number of experts")
    assert_experts_frozen(*experts)
    router.to(device)
    router.eval()
    for expert in experts:
        expert.to(device)
        expert.eval()

    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    smooth_l1_sum = 0.0
    expert_absolute_error_sums = {
        expert_name: 0.0 for expert_name in expert_names
    }
    element_count = 0
    weight_summary_state = _new_expert_weight_summary_state(
        expert_names,
        router.forecast_horizon,
        router.num_features,
    )

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            inputs, targets, targets_mask = _prepare_forecasting_batch(
                batch,
                device,
                scaler,
            )
            (
                expert_predictions,
                disagreement,
                router_scores,
                router_weights,
                mixed_prediction,
            ) = _prediction_aware_router_forward(router, experts, inputs)
            _check_shapes(mixed_prediction, targets, "PredictionAwareRouter")

            if print_shapes and batch_index == 0:
                print("\nFirst router-validation batch")
                print(f"Input shape:                  {list(inputs.shape)}")
                print(f"Target shape:                 {list(targets.shape)}")
                print(f"Expert prediction stack shape:{list(expert_predictions.shape)}")
                for expert_index, expert_name in enumerate(expert_names):
                    print(
                        f"{expert_name} prediction shape: "
                        f"{list(expert_predictions[:, :, expert_index, :].shape)}"
                    )
                print(f"Disagreement shape:           {list(disagreement.shape)}")
                print(f"Router-score shape:           {list(router_scores.shape)}")
                print(f"Router-weight shape:          {list(router_weights.shape)}")
                print(f"Mixed-prediction shape:       {list(mixed_prediction.shape)}")
                assert tuple(router_scores.shape) == (
                    inputs.shape[0],
                    router.forecast_horizon,
                    router.num_features,
                    router.num_experts,
                )
                assert tuple(router_weights.shape) == tuple(router_scores.shape)
                assert tuple(mixed_prediction.shape) == tuple(targets.shape)
                assert torch.allclose(
                    router_weights.sum(dim=-1),
                    torch.ones(
                        inputs.shape[0],
                        router.forecast_horizon,
                        router.num_features,
                        device=router_weights.device,
                    ),
                    atol=1e-6,
                    rtol=1e-6,
                )

            abs_sum, squared_sum, count = _accumulate_errors(
                mixed_prediction,
                targets,
                targets_mask,
            )
            for expert_index, expert_name in enumerate(expert_names):
                expert_abs_sum, _, _ = _accumulate_errors(
                    expert_predictions[:, :, expert_index, :],
                    targets,
                    targets_mask,
                )
                expert_absolute_error_sums[expert_name] += expert_abs_sum
            smooth_l1_sum += nn.functional.smooth_l1_loss(
                mixed_prediction[targets_mask],
                targets[targets_mask],
                reduction="sum",
            ).item()
            absolute_error_sum += abs_sum
            squared_error_sum += squared_sum
            element_count += count
            _accumulate_expert_weight_summary(
                weight_summary_state,
                router_weights,
                expert_names,
            )

    if element_count == 0 or weight_summary_state["sample_count"] == 0:
        raise ValueError("Router evaluation loader produced no elements")
    _assert_no_expert_gradients(*experts)
    expert_weight_summary = _finalize_expert_weight_summary(
        weight_summary_state,
        expert_names,
        router.forecast_horizon,
        router.num_features,
    )
    average_expert_weights = expert_weight_summary["average_expert_weights"]
    expert_mae = {
        expert_name: abs_sum / element_count
        for expert_name, abs_sum in expert_absolute_error_sums.items()
    }
    result = {
        "mae": absolute_error_sum / element_count,
        "mse": squared_error_sum / element_count,
        "smooth_l1_loss": smooth_l1_sum / element_count,
        "expert_mae": expert_mae,
        **expert_weight_summary,
    }
    if len(expert_names) == 2:
        result.update(
            {
                "average_dlinear_weight": average_expert_weights[expert_names[0]],
                "average_transformer_weight": average_expert_weights[expert_names[1]],
                "dlinear_mae": expert_mae[expert_names[0]],
                "transformer_mae": expert_mae[expert_names[1]],
            }
        )
    return result


def train_prediction_aware_router_model(
    router: PredictionAwareRouter,
    experts: Sequence[nn.Module],
    optimizer: torch.optim.Optimizer,
    train_loader: Iterable[dict],
    val_loader: Iterable[dict],
    checkpoint_path: Union[str, Path],
    max_epochs: int,
    patience: int = 10,
    device: Union[str, torch.device] = "cpu",
    scaler=None,
    dataset_config: Optional[dict] = None,
    expert_checkpoint_paths: Optional[dict] = None,
    expert_names: Optional[Sequence[str]] = None,
) -> Tuple[dict, ...]:
    """Train only the prediction-aware router and select by router-val MAE."""

    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if patience <= 0:
        raise ValueError("patience must be positive")
    if getattr(train_loader.dataset, "split_role", None) != "router_train":
        raise ValueError("Router training requires the 15% router_train split")
    if getattr(val_loader.dataset, "split_role", None) != "router_val":
        raise ValueError("Router validation requires the 5% router_val split")
    expert_names = tuple(
        expert_names or [f"Expert {index + 1}" for index in range(len(experts))]
    )
    if len(expert_names) != len(experts):
        raise ValueError("expert_names must match the number of experts")
    if router.num_experts != len(experts):
        raise ValueError(
            f"Router expects {router.num_experts} experts, got {len(experts)}"
        )

    device = torch.device(device)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    assert_experts_frozen(*experts)
    _assert_router_optimizer_excludes_experts(optimizer, *experts)
    router_parameter_ids = {id(parameter) for parameter in router.parameters()}
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if not optimizer_parameter_ids:
        raise ValueError("Router optimizer has no parameters")
    if not optimizer_parameter_ids.issubset(router_parameter_ids):
        raise ValueError("Router optimizer contains non-router parameters")

    for expert in experts:
        expert.to(device)
        expert.eval()
        for parameter in expert.parameters():
            parameter.grad = None
    router.to(device)
    frozen_expert_states = [
        {
            name: value.detach().clone()
            for name, value in expert.state_dict().items()
        }
        for expert in experts
    ]

    loss_function = nn.SmoothL1Loss()
    history = []
    best_validation_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        router.train()
        smooth_l1_sum = 0.0
        absolute_error_sum = 0.0
        element_count = 0
        weight_summary_state = _new_expert_weight_summary_state(
            expert_names,
            router.forecast_horizon,
            router.num_features,
        )

        for batch_index, batch in enumerate(train_loader):
            inputs, targets, targets_mask = _prepare_forecasting_batch(
                batch,
                device,
                scaler,
            )
            optimizer.zero_grad(set_to_none=True)
            (
                expert_predictions,
                disagreement,
                router_scores,
                router_weights,
                mixed_prediction,
            ) = _prediction_aware_router_forward(router, experts, inputs)
            _check_shapes(mixed_prediction, targets, "PredictionAwareRouter")

            loss = loss_function(mixed_prediction, targets)
            loss.backward()
            if router.horizon_queries.grad is None:
                raise RuntimeError("The router horizon queries received no gradient")
            if not any(
                parameter.grad is not None
                for parameter in router.parameters()
            ):
                raise RuntimeError("No router parameter received a gradient")
            for component_name, component in (
                ("history projection", router.history_projection),
                ("history encoder", router.history_encoder),
                ("prediction encoder", router.prediction_encoder),
                ("routing head", router.routing_head),
            ):
                if not any(
                    parameter.grad is not None
                    for parameter in component.parameters()
                ):
                    raise RuntimeError(
                        f"The router {component_name} received no gradient"
                    )
            _assert_no_expert_gradients(*experts)

            if epoch == 1 and batch_index == 0:
                print("\nFirst prediction-aware router-training batch")
                print(f"Input shape:                  {list(inputs.shape)}")
                print(f"Target shape:                 {list(targets.shape)}")
                print(f"Expert prediction stack shape:{list(expert_predictions.shape)}")
                for expert_index, expert_name in enumerate(expert_names):
                    print(
                        f"{expert_name} prediction shape: "
                        f"{list(expert_predictions[:, :, expert_index, :].shape)}"
                    )
                print(f"Disagreement shape:           {list(disagreement.shape)}")
                print(f"Router-score shape:           {list(router_scores.shape)}")
                print(f"Router-weight shape:          {list(router_weights.shape)}")
                print(f"Mixed-prediction shape:       {list(mixed_prediction.shape)}")
                print(f"Training loss:                {loss.item():.6f}")
                assert tuple(router_scores.shape) == (
                    inputs.shape[0],
                    router.forecast_horizon,
                    router.num_features,
                    router.num_experts,
                )
                assert tuple(router_weights.shape) == tuple(router_scores.shape)
                assert tuple(mixed_prediction.shape) == tuple(targets.shape)
                assert torch.allclose(
                    router_weights.sum(dim=-1),
                    torch.ones(
                        inputs.shape[0],
                        router.forecast_horizon,
                        router.num_features,
                        device=router_weights.device,
                    ),
                    atol=1e-6,
                    rtol=1e-6,
                )
                for expert_name, expert in zip(expert_names, experts):
                    print(
                        f"{expert_name} gradients present: "
                        f"{any(p.grad is not None for p in expert.parameters())}"
                    )

            optimizer.step()

            abs_sum, _, count = _accumulate_errors(
                mixed_prediction.detach(),
                targets,
                targets_mask,
            )
            smooth_l1_sum += loss.detach().item() * count
            absolute_error_sum += abs_sum
            element_count += count
            _accumulate_expert_weight_summary(
                weight_summary_state,
                router_weights,
                expert_names,
            )

        if element_count == 0 or weight_summary_state["sample_count"] == 0:
            raise ValueError("Router training loader produced no elements")
        for expert, original_state in zip(experts, frozen_expert_states):
            for name, value in expert.state_dict().items():
                if not torch.equal(value, original_state[name]):
                    raise RuntimeError(
                        f"Frozen expert state changed for parameter {name}"
                    )
        training_loss = smooth_l1_sum / element_count
        train_mae = absolute_error_sum / element_count
        training_weight_summary = _finalize_expert_weight_summary(
            weight_summary_state,
            expert_names,
            router.forecast_horizon,
            router.num_features,
        )
        average_expert_weights = training_weight_summary["average_expert_weights"]

        validation = evaluate_prediction_aware_router_pipeline(
            router=router,
            experts=experts,
            loader=val_loader,
            device=device,
            scaler=scaler,
            print_shapes=(epoch == 1),
            expert_names=expert_names,
        )
        checkpoint_saved = validation["mae"] < best_validation_mae
        if checkpoint_saved:
            best_validation_mae = validation["mae"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "router_state_dict": router.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "training_loss": training_loss,
                    "router_training_loss": training_loss,
                    "validation_mae": validation["mae"],
                    "validation_mse": validation["mse"],
                    "validation_smooth_l1_loss": validation[
                        "smooth_l1_loss"
                    ],
                    "average_expert_weights": average_expert_weights,
                    "average_expert_weights_by_step": training_weight_summary[
                        "average_expert_weights_by_step"
                    ],
                    "average_expert_weights_by_variable": training_weight_summary[
                        "average_expert_weights_by_variable"
                    ],
                    "average_expert_weights_by_step_and_variable": (
                        training_weight_summary[
                            "average_expert_weights_by_step_and_variable"
                        ]
                    ),
                    "validation_average_expert_weights": validation[
                        "average_expert_weights"
                    ],
                    "validation_average_expert_weights_by_step": validation[
                        "average_expert_weights_by_step"
                    ],
                    "validation_average_expert_weights_by_variable": validation[
                        "average_expert_weights_by_variable"
                    ],
                    "validation_average_expert_weights_by_step_and_variable": (
                        validation[
                            "average_expert_weights_by_step_and_variable"
                        ]
                    ),
                    "selected_expert_names": list(expert_names),
                    "router_config": router.config_dict(),
                    "dataset_config": dict(dataset_config or {}),
                    "expert_checkpoint_paths": {
                        name: str(path)
                        for name, path in (
                            expert_checkpoint_paths or {}
                        ).items()
                    },
                    **(
                        {"scaler_stats": scaler.stats}
                        if scaler is not None
                        else {}
                    ),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "train_mae": train_mae,
                "validation_mae": validation["mae"],
                "validation_mse": validation["mse"],
                "validation_smooth_l1_loss": validation[
                    "smooth_l1_loss"
                ],
                "average_expert_weights": average_expert_weights,
                "average_expert_weights_by_step": training_weight_summary[
                    "average_expert_weights_by_step"
                ],
                "average_expert_weights_by_variable": training_weight_summary[
                    "average_expert_weights_by_variable"
                ],
                "average_expert_weights_by_step_and_variable": (
                    training_weight_summary[
                        "average_expert_weights_by_step_and_variable"
                    ]
                ),
                "checkpoint_saved": checkpoint_saved,
                "early_stopping_counter": epochs_without_improvement,
            }
        )
        weight_summary = ", ".join(
            f"avg {expert_name} weight={average_expert_weights[expert_name]:.4f}"
            for expert_name in expert_names
        )
        print(
            f"Router epoch {epoch:>3d}/{max_epochs}: "
            f"training loss={training_loss:.6f}, "
            f"training MAE={train_mae:.6f}, "
            f"validation MAE={validation['mae']:.6f}, "
            f"validation MSE={validation['mse']:.6f}, "
            f"{weight_summary}, "
            f"checkpoint saved={checkpoint_saved}, "
            f"early-stop counter={epochs_without_improvement}/{patience}"
        )
        if epochs_without_improvement >= patience:
            print(
                f"Router: early stopping after epoch {epoch} "
                f"({patience} epochs without lower validation MAE)."
            )
            break

    if best_epoch == 0:
        raise RuntimeError("Router did not save a validation-selected checkpoint")
    selected = load_prediction_aware_router_checkpoint(
        router,
        checkpoint_path,
        device=device,
        optimizer=optimizer,
    )
    print(
        f"\nSelected PredictionAwareRouter epoch {selected['epoch']}: "
        f"validation MAE={selected['validation_mae']:.6f}, "
        f"validation MSE={selected['validation_mse']:.6f}"
    )
    _assert_no_expert_gradients(*experts)
    return tuple(history)


def load_prediction_aware_router_checkpoint(
    router: PredictionAwareRouter,
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """Load the selected prediction-aware router checkpoint."""

    device = torch.device(device)
    checkpoint = _load_torch_checkpoint(checkpoint_path, device)
    router.load_state_dict(checkpoint["router_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    router.to(device)
    router.eval()
    print(
        f"Loaded PredictionAwareRouter epoch {checkpoint['epoch']}: "
        f"validation MAE={checkpoint['validation_mae']:.6f}, "
        f"validation MSE={checkpoint['validation_mse']:.6f}"
    )
    return checkpoint


def run_prediction_aware_router_training_stage(
    data_dir: Union[str, Path] = "datasets/ETTh1",
    checkpoint_dir: Union[str, Path] = "checkpoints",
    batch_size: int = 512,
    max_epochs: int = 50,
    patience: int = 10,
    learning_rate: float = 1e-3,
    device: Union[str, torch.device] = "cpu",
    seed: int = 7,
) -> Tuple[dict, ...]:
    """Run Stage 5: train router on router_train and select on router_val."""

    _ensure_local_src_importable()
    from basicts.scaler import ZScoreScaler

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)
    data_dir = Path(data_dir)
    checkpoint_dir = Path(checkpoint_dir)
    runtime_config = _validated_runtime_router_config(
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        require_checkpoints=True,
    )

    full_data = load_full_chronological_data(data_dir)
    _assert_full_data_contract(full_data, runtime_config.num_features)
    loaders, scaler = prepare_chronological_dataloaders(
        full_data=full_data,
        scaler=ZScoreScaler(norm_each_channel=True, rescale=False),
        batch_size=batch_size,
        input_len=runtime_config.input_length,
        output_len=runtime_config.forecast_horizon,
    )
    model_groups = selected_router_model_groups()
    multiple_groups = len(model_groups) > 1
    stage_results = []

    for specs in model_groups:
        group_name = _router_model_group_name(specs)
        expert_names_for_group = tuple(spec.display_name for spec in specs)
        print(
            f"\n=== Router training group: {group_name} "
            f"({', '.join(expert_names_for_group)}) ==="
        )
        if len(specs) == 1:
            print(
                "Skipping router training for the one-expert baseline; "
                "the final test stage will evaluate that saved expert directly."
            )
            stage_results.append(
                {
                    "model_group": group_name,
                    "selected_experts": list(expert_names_for_group),
                    "checkpoint_path": None,
                    "history": [],
                    "skipped_router_training": True,
                }
            )
            continue

        experts, expert_names, _, expert_checkpoint_paths = (
            build_selected_candidate_experts(
                checkpoint_dir=checkpoint_dir,
                device=device,
                scaler=scaler,
                specs=specs,
            )
        )
        checkpoint_path = _router_checkpoint_path(
            checkpoint_dir,
            group_name,
            multiple_groups,
        )
        print("Selected router experts:", ", ".join(expert_names))
        print(f"Router checkpoint path: {checkpoint_path}")
        router = PredictionAwareRouter(num_experts=len(experts)).to(device)
        optimizer = torch.optim.Adam(router.parameters(), lr=learning_rate)
        history = train_prediction_aware_router_model(
            router=router,
            experts=experts,
            optimizer=optimizer,
            train_loader=loaders["router_train"],
            val_loader=loaders["router_val"],
            checkpoint_path=checkpoint_path,
            max_epochs=max_epochs,
            patience=patience,
            device=device,
            scaler=scaler,
            dataset_config=_dataset_config_summary(len(full_data)),
            expert_checkpoint_paths=expert_checkpoint_paths,
            expert_names=expert_names,
        )
        fresh_router = PredictionAwareRouter(num_experts=len(experts)).to(device)
        fresh_optimizer = torch.optim.Adam(
            fresh_router.parameters(),
            lr=learning_rate,
        )
        checkpoint = load_prediction_aware_router_checkpoint(
            fresh_router,
            checkpoint_path,
            device=device,
            optimizer=fresh_optimizer,
        )
        print(
            "\nCheckpoint reload verification: "
            f"{checkpoint_path.name} epoch={checkpoint['epoch']}, "
            f"validation MAE={checkpoint['validation_mae']:.6f}, "
            f"validation MSE={checkpoint['validation_mse']:.6f}"
        )
        stage_results.append(
            {
                "model_group": group_name,
                "selected_experts": list(expert_names),
                "checkpoint_path": str(checkpoint_path),
                "history": list(history),
                "skipped_router_training": False,
            }
        )

    if multiple_groups:
        return tuple(stage_results)
    return tuple(stage_results[0]["history"])


def evaluate_prediction_aware_router_and_baselines(
    router: PredictionAwareRouter,
    experts: Sequence[nn.Module],
    router_val_loader: Iterable[dict],
    test_loader: Iterable[dict],
    output_dir: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    scaler=None,
    expert_names: Optional[Sequence[str]] = None,
) -> dict:
    """Run the final untouched-test evaluation once and save comparisons."""

    if getattr(router_val_loader.dataset, "split_role", None) != "router_val":
        raise ValueError("Validation-derived baselines require router_val")
    if getattr(test_loader.dataset, "split_role", None) != "test":
        raise ValueError("Final evaluation requires the untouched test split")

    device = torch.device(device)
    expert_names = tuple(
        expert_names or [f"Expert {index + 1}" for index in range(len(experts))]
    )
    if len(expert_names) != len(experts):
        raise ValueError("expert_names must match the number of experts")
    if router.num_experts != len(experts):
        raise ValueError(
            f"Router expects {router.num_experts} experts, got {len(experts)}"
        )
    router.to(device)
    router.eval()
    assert_experts_frozen(*experts)
    for expert in experts:
        expert.to(device)
        expert.eval()

    validation = evaluate_prediction_aware_router_pipeline(
        router=router,
        experts=experts,
        loader=router_val_loader,
        device=device,
        scaler=scaler,
        expert_names=expert_names,
    )
    epsilon = 1e-6
    validation_expert_mae = validation["expert_mae"]
    inverse_scores = {
        expert_name: 1.0 / (validation_expert_mae[expert_name] + epsilon)
        for expert_name in expert_names
    }
    inverse_total = sum(inverse_scores.values())
    fixed_soft_weights = {
        expert_name: inverse_scores[expert_name] / inverse_total
        for expert_name in expert_names
    }
    globally_best_expert = min(
        expert_names,
        key=lambda expert_name: validation_expert_mae[expert_name],
    )

    method_names = tuple(expert_names) + (
        "Fixed equal average",
        "Fixed validation-based soft weights",
        "Validation-selected best expert",
        "Learned prediction-aware router",
    )
    totals = {
        name: {"absolute": 0.0, "squared": 0.0, "count": 0}
        for name in method_names
    }
    per_step_absolute = torch.zeros(
        router.forecast_horizon,
        dtype=torch.float64,
    )
    per_step_count = torch.zeros(
        router.forecast_horizon,
        dtype=torch.float64,
    )
    test_weight_summary_state = _new_expert_weight_summary_state(
        expert_names,
        router.forecast_horizon,
        router.num_features,
    )
    router_state_before = {
        name: value.detach().clone()
        for name, value in router.state_dict().items()
    }
    expert_states_before = [
        {
            name: value.detach().clone()
            for name, value in expert.state_dict().items()
        }
        for expert in experts
    ]

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets, targets_mask = _prepare_forecasting_batch(
                batch,
                device,
                scaler,
            )
            (
                expert_predictions,
                _,
                _,
                router_weights,
                router_prediction,
            ) = _prediction_aware_router_forward(router, experts, inputs)
            equal_prediction = expert_predictions.mean(dim=2)
            fixed_soft_prediction = torch.zeros_like(equal_prediction)
            for expert_index, expert_name in enumerate(expert_names):
                fixed_soft_prediction = (
                    fixed_soft_prediction
                    + fixed_soft_weights[expert_name]
                    * expert_predictions[:, :, expert_index, :]
                )
            best_expert_index = expert_names.index(globally_best_expert)
            validation_selected_prediction = expert_predictions[
                :, :, best_expert_index, :
            ]
            predictions = {
                expert_name: expert_predictions[:, :, expert_index, :]
                for expert_index, expert_name in enumerate(expert_names)
            }
            predictions.update(
                {
                    "Fixed equal average": equal_prediction,
                    "Fixed validation-based soft weights": (
                        fixed_soft_prediction
                    ),
                    "Validation-selected best expert": (
                        validation_selected_prediction
                    ),
                    "Learned prediction-aware router": router_prediction,
                }
            )
            for name, prediction in predictions.items():
                _check_shapes(prediction, targets, name)
                abs_sum, squared_sum, count = _accumulate_errors(
                    prediction,
                    targets,
                    targets_mask,
                )
                totals[name]["absolute"] += abs_sum
                totals[name]["squared"] += squared_sum
                totals[name]["count"] += count

            router_absolute = torch.abs(router_prediction - targets)
            mask = targets_mask.to(router_absolute.dtype)
            per_step_absolute += (
                (router_absolute * mask)
                .sum(dim=(0, 2))
                .cpu()
                .to(torch.float64)
            )
            per_step_count += (
                mask.sum(dim=(0, 2)).cpu().to(torch.float64)
            )
            _accumulate_expert_weight_summary(
                test_weight_summary_state,
                router_weights,
                expert_names,
            )

    for name, value in router.state_dict().items():
        if not torch.equal(value, router_state_before[name]):
            raise RuntimeError(f"Router changed during final test: {name}")
    for expert, original_state in zip(experts, expert_states_before):
        for name, value in expert.state_dict().items():
            if not torch.equal(value, original_state[name]):
                raise RuntimeError(
                    f"Expert changed during final test: {name}"
                )
    _assert_no_expert_gradients(*experts)

    comparison = []
    for name in method_names:
        values = totals[name]
        mae = values["absolute"] / values["count"]
        mse = values["squared"] / values["count"]
        comparison.append(
            {
                "Method": name,
                "Test MAE": mae,
                "Test MSE": mse,
                "Test RMSE": mse ** 0.5,
            }
        )
    comparison.sort(key=lambda row: row["Test MAE"])

    router_row = next(
        row
        for row in comparison
        if row["Method"] == "Learned prediction-aware router"
    )
    baseline_rows = [
        row
        for row in comparison
        if row["Method"] != "Learned prediction-aware router"
    ]
    strongest_baseline = min(
        baseline_rows,
        key=lambda row: row["Test MAE"],
    )
    improvement = strongest_baseline["Test MAE"] - router_row["Test MAE"]
    percentage_improvement = (
        improvement / strongest_baseline["Test MAE"] * 100.0
    )
    test_weight_summary = _finalize_expert_weight_summary(
        test_weight_summary_state,
        expert_names,
        router.forecast_horizon,
        router.num_features,
    )
    learned_router_details = {
        "mae": router_row["Test MAE"],
        "mse": router_row["Test MSE"],
        "rmse": router_row["Test RMSE"],
        "per_step_mae": (
            per_step_absolute / per_step_count
        ).tolist(),
        **test_weight_summary,
    }
    results = {
        "selected_expert_names": list(expert_names),
        "comparison": comparison,
        "learned_router": learned_router_details,
        "router_validation_baseline_selection": {
            "expert_validation_mae": validation_expert_mae,
            "fixed_soft_weights": fixed_soft_weights,
            "globally_best_expert": globally_best_expert,
            "average_expert_weights": validation[
                "average_expert_weights"
            ],
            "average_expert_weights_by_step": validation[
                "average_expert_weights_by_step"
            ],
            "average_expert_weights_by_variable": validation[
                "average_expert_weights_by_variable"
            ],
            "average_expert_weights_by_step_and_variable": validation[
                "average_expert_weights_by_step_and_variable"
            ],
        },
        "strongest_baseline": strongest_baseline,
        "absolute_mae_improvement": improvement,
        "percentage_mae_improvement": percentage_improvement,
        "router_beat_strongest_baseline": improvement > 0,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "router_test_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("Method", "Test MAE", "Test MSE", "Test RMSE"),
        )
        writer.writeheader()
        writer.writerows(comparison)
    json_path = output_dir / "router_test_metrics.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nFinal test comparison (sorted by MAE)")
    print(
        f"{'Method':<45} {'Test MAE':>12} "
        f"{'Test MSE':>12} {'Test RMSE':>12}"
    )
    print("-" * 85)
    for row in comparison:
        print(
            f"{row['Method']:<45} "
            f"{row['Test MAE']:>12.6f} "
            f"{row['Test MSE']:>12.6f} "
            f"{row['Test RMSE']:>12.6f}"
        )
    print(f"\nStrongest baseline: {strongest_baseline['Method']}")
    print(
        "Learned-router result: "
        f"MAE={router_row['Test MAE']:.6f}, "
        f"MSE={router_row['Test MSE']:.6f}, "
        f"RMSE={router_row['Test RMSE']:.6f}"
    )
    print(f"Absolute MAE improvement: {improvement:.6f}")
    print(f"Percentage MAE improvement: {percentage_improvement:.2f}%")
    print(
        "Router beat the strongest baseline"
        if improvement > 0
        else "Router did not beat the strongest baseline"
    )
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    return results


def _evaluate_single_expert_loader(
    expert: nn.Module,
    loader: Iterable[dict],
    device: torch.device,
    scaler,
) -> dict:
    totals = {"absolute": 0.0, "squared": 0.0, "count": 0}
    with torch.no_grad():
        for batch in loader:
            inputs, targets, targets_mask = _prepare_forecasting_batch(
                batch,
                device,
                scaler,
            )
            prediction = _call_expert_model(expert, inputs, targets).detach()
            _check_shapes(prediction, targets, "Single saved expert")
            abs_sum, squared_sum, count = _accumulate_errors(
                prediction,
                targets,
                targets_mask,
            )
            totals["absolute"] += abs_sum
            totals["squared"] += squared_sum
            totals["count"] += count
    mae = totals["absolute"] / totals["count"]
    mse = totals["squared"] / totals["count"]
    return {"mae": mae, "mse": mse, "rmse": mse ** 0.5}


def evaluate_single_expert_baseline(
    expert: nn.Module,
    expert_name: str,
    router_val_loader: Iterable[dict],
    test_loader: Iterable[dict],
    output_dir: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    scaler=None,
) -> dict:
    """Evaluate the one-model saved baseline without training a router."""

    if getattr(router_val_loader.dataset, "split_role", None) != "router_val":
        raise ValueError("Validation baseline requires router_val")
    if getattr(test_loader.dataset, "split_role", None) != "test":
        raise ValueError("Final evaluation requires the untouched test split")

    device = torch.device(device)
    expert.to(device)
    expert.eval()
    assert_experts_frozen(expert)
    expert_state_before = {
        name: value.detach().clone()
        for name, value in expert.state_dict().items()
    }

    validation = _evaluate_single_expert_loader(
        expert,
        router_val_loader,
        device,
        scaler,
    )
    test = _evaluate_single_expert_loader(
        expert,
        test_loader,
        device,
        scaler,
    )

    for name, value in expert.state_dict().items():
        if not torch.equal(value, expert_state_before[name]):
            raise RuntimeError(f"Expert changed during final test: {name}")
    _assert_no_expert_gradients(expert)

    comparison = [
        {
            "Method": expert_name,
            "Test MAE": test["mae"],
            "Test MSE": test["mse"],
            "Test RMSE": test["rmse"],
        }
    ]
    results = {
        "selected_expert_names": [expert_name],
        "comparison": comparison,
        "single_expert_baseline": {
            "expert_name": expert_name,
            "validation_mae": validation["mae"],
            "validation_mse": validation["mse"],
            "validation_rmse": validation["rmse"],
            "test_mae": test["mae"],
            "test_mse": test["mse"],
            "test_rmse": test["rmse"],
        },
        "router_trained": False,
        "reason_router_not_trained": (
            "A one-expert softmax is always 1, so there is no router "
            "choice to learn."
        ),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "router_test_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("Method", "Test MAE", "Test MSE", "Test RMSE"),
        )
        writer.writeheader()
        writer.writerows(comparison)
    json_path = output_dir / "router_test_metrics.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nFinal one-expert baseline test")
    print(
        f"{expert_name}: MAE={test['mae']:.6f}, "
        f"MSE={test['mse']:.6f}, RMSE={test['rmse']:.6f}"
    )
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    return results


def run_final_router_test_stage(
    data_dir: Union[str, Path] = "datasets/ETTh1",
    checkpoint_dir: Union[str, Path] = "checkpoints",
    output_dir: Union[str, Path] = "results/router_summary",
    batch_size: int = 512,
    device: Union[str, torch.device] = "cpu",
    seed: int = 7,
) -> dict:
    """Run Stage 6: evaluate selected model groups on the untouched test."""

    _ensure_local_src_importable()
    from basicts.scaler import ZScoreScaler

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)
    data_dir = Path(data_dir)
    checkpoint_dir = Path(checkpoint_dir)
    runtime_config = _validated_runtime_router_config(
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        require_checkpoints=True,
    )

    full_data = load_full_chronological_data(data_dir)
    _assert_full_data_contract(full_data, runtime_config.num_features)
    loaders, scaler = prepare_chronological_dataloaders(
        full_data=full_data,
        scaler=ZScoreScaler(norm_each_channel=True, rescale=False),
        batch_size=batch_size,
        input_len=runtime_config.input_length,
        output_len=runtime_config.forecast_horizon,
    )
    model_groups = selected_router_model_groups()
    multiple_groups = len(model_groups) > 1
    stage_results = {}

    for specs in model_groups:
        group_name = _router_model_group_name(specs)
        print(
            f"\n=== Router final test group: {group_name} "
            f"({', '.join(spec.display_name for spec in specs)}) ==="
        )
        experts, expert_names, _, _ = build_selected_candidate_experts(
            checkpoint_dir=checkpoint_dir,
            device=device,
            scaler=scaler,
            specs=specs,
        )
        group_output_dir = _router_output_dir(
            output_dir,
            group_name,
            multiple_groups,
        )

        if len(experts) == 1:
            stage_results[group_name] = evaluate_single_expert_baseline(
                expert=experts[0],
                expert_name=expert_names[0],
                router_val_loader=loaders["router_val"],
                test_loader=loaders["test"],
                output_dir=group_output_dir,
                device=device,
                scaler=scaler,
            )
            continue

        checkpoint_path = _router_checkpoint_path(
            checkpoint_dir,
            group_name,
            multiple_groups,
        )
        router_checkpoint = _load_torch_checkpoint(
            checkpoint_path,
            device,
        )
        checkpoint_expert_names = tuple(
            router_checkpoint.get("selected_expert_names", ())
        )
        if checkpoint_expert_names and checkpoint_expert_names != tuple(expert_names):
            raise ValueError(
                f"{checkpoint_path} was trained for "
                f"{checkpoint_expert_names}, but this test selected "
                f"{tuple(expert_names)}"
            )
        router_config = dict(router_checkpoint["router_config"])
        router_config.setdefault("num_experts", len(experts))
        router = PredictionAwareRouter(
            **router_config
        ).to(device)
        load_prediction_aware_router_checkpoint(
            router,
            checkpoint_path,
            device=device,
        )
        stage_results[group_name] = evaluate_prediction_aware_router_and_baselines(
            router=router,
            experts=experts,
            router_val_loader=loaders["router_val"],
            test_loader=loaders["test"],
            output_dir=group_output_dir,
            device=device,
            scaler=scaler,
            expert_names=expert_names,
        )

    if multiple_groups:
        return stage_results
    return next(iter(stage_results.values()))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run chronological expert stages or frozen-expert router input "
            "verification."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("experts", "router-inputs", "router-train", "test", "config-check"),
        default="experts",
        help=(
            "config-check validates the central router config; experts trains "
            "legacy DLinear/iTransformer; router-inputs verifies selected frozen "
            "expert predictions and the prediction-aware router forward pass; "
            "router-train trains configured router groups; test evaluates "
            "configured groups once."
        ),
    )
    parser.add_argument("--data-dir", default="datasets/ETTh1")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results/router_summary")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--router-patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.stage == "config-check":
        run_router_config_check_stage(
            data_dir=args.data_dir,
            checkpoint_dir=args.output_dir,
            seed=args.seed,
        )
    elif args.stage == "router-inputs":
        run_router_input_and_forward_stage(
            data_dir=args.data_dir,
            checkpoint_dir=args.output_dir,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
        )
    elif args.stage == "router-train":
        run_prediction_aware_router_training_stage(
            data_dir=args.data_dir,
            checkpoint_dir=args.output_dir,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.router_patience,
            learning_rate=args.learning_rate,
            device=args.device,
            seed=args.seed,
        )
    elif args.stage == "test":
        run_final_router_test_stage(
            data_dir=args.data_dir,
            checkpoint_dir=args.output_dir,
            output_dir=args.results_dir,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
        )
    else:
        run_chronological_expert_stages(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            device=args.device,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
