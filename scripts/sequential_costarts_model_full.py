"""Full Sequential COSTAR-TS model entry point.

This module keeps a separate "full" name for the validation-best sequential
COSTAR-TS architecture while delegating the implementation to
``scripts.sequential_costarts_model``. It exists so experiments can explicitly
call the full sequential COSTAR-TS model instead of the older history-only
COSTARTS router.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from scripts.sequential_costarts_model import SequentialCOSTARTSRouter


DEFAULT_FULL_SEQUENTIAL_COSTARTS_CHECKPOINT = (
    "checkpoints/costarts_sequential/seed_11/best_sequential_costarts_router.pt"
)


class SequentialCOSTARTSRouterFull(SequentialCOSTARTSRouter):
    """Named copy of the full sequential COSTAR-TS router architecture."""

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> "SequentialCOSTARTSRouterFull":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = cls(**checkpoint["router_config"])
        model.load_state_dict(checkpoint["router_state_dict"], strict=True)
        return model


SequentialCOSTARTSModelFull = SequentialCOSTARTSRouterFull


def load_sequential_costarts_router_full(
    checkpoint_path: str | Path = DEFAULT_FULL_SEQUENTIAL_COSTARTS_CHECKPOINT,
) -> SequentialCOSTARTSRouterFull:
    return SequentialCOSTARTSRouterFull.from_checkpoint(checkpoint_path)


def load_sequential_costarts_model_full(
    checkpoint_path: str | Path = DEFAULT_FULL_SEQUENTIAL_COSTARTS_CHECKPOINT,
) -> SequentialCOSTARTSRouterFull:
    return load_sequential_costarts_router_full(checkpoint_path)


def build_sequential_costarts_model_full(**router_config: Any) -> SequentialCOSTARTSRouterFull:
    return SequentialCOSTARTSRouterFull(**router_config)
