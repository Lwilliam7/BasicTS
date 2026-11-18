import pytest

from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import enforce_observable
from experiments.regime_adaptive_forgetting_costar.run_regime_adaptive_forgetting import (
    RegimeConfig,
    update_detector,
)


def test_regime_release_rule_rejects_early_update() -> None:
    with pytest.raises(RuntimeError):
        enforce_observable(due_start=250, current_start=261, horizon=12)
    enforce_observable(due_start=250, current_start=262, horizon=12)


def test_page_hinkley_detector_triggers_on_large_shift() -> None:
    cfg = RegimeConfig(
        detector="page_hinkley",
        slow_decay=0.99,
        fast_decay=0.90,
        threshold=0.02,
        delta=0.0,
        reset_strength=0.25,
        cooldown=24,
        boost_duration=24,
    )
    state = {"mean": 0.10, "var": 0.01, "cum": 0.0}
    assert update_detector(0.11, cfg, state) is False
    assert update_detector(0.50, cfg, state) is True
