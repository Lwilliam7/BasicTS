from __future__ import annotations

import pytest

from experiments.pooled_router_train_residual_correctors.run_pooled_router_train_residual_correctors import (
    etth2_ridge_grid,
    mlp_grid,
    refuse_test_path,
)


def test_pooled_residual_search_spaces_are_declared() -> None:
    assert len(etth2_ridge_grid()) == 5
    assert len(mlp_grid("ETTh1")) == 5
    assert len(mlp_grid("ETTh2")) == 5


def test_test_path_guard_rejects_test_before_manifest() -> None:
    with pytest.raises(ValueError):
        refuse_test_path("experiments/final_test_evaluation/generated/caches/ETTh1/test_80_100_cache.pt")
