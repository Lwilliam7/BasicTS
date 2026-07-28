import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.selectors import temperature_search_softmax_selector


def test_temperature_search_softmax_selector_uses_validation_winner():
    val_targets = np.array([[[0.0]], [[0.0]]])
    val_predictions = np.array(
        [
            [[[1.0]], [[1.0]]],
            [[[-2.0]], [[-2.0]]],
        ],
        dtype=np.float64,
    )
    saved_predictions = np.array(
        [
            [[[1.0]], [[3.0]]],
            [[[5.0]], [[7.0]]],
        ],
        dtype=np.float64,
    )

    final_predictions, weights, metadata = temperature_search_softmax_selector(
        saved_predictions,
        val_predictions,
        val_targets,
        temperatures=(0.25, 5.0),
    )

    assert metadata["best_temperature"] == 5.0
    np.testing.assert_allclose(metadata["best_validation_mae"], 0.350498, atol=1e-6)
    assert weights.shape == (2, 1, 1)
    assert final_predictions.shape == (2, 1, 1)
    np.testing.assert_allclose(final_predictions[:, 0, 0], [2.800664, 4.800664], atol=1e-6)


def test_temperature_search_softmax_selector_rejects_invalid_temperatures():
    predictions = np.zeros((2, 1, 1, 1))
    targets = np.zeros((1, 1, 1))

    try:
        temperature_search_softmax_selector(predictions, predictions, targets, temperatures=())
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty temperatures")

    try:
        temperature_search_softmax_selector(predictions, predictions, targets, temperatures=(1.0, 0.0))
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-positive temperatures")


if __name__ == "__main__":
    test_temperature_search_softmax_selector_uses_validation_winner()
    test_temperature_search_softmax_selector_rejects_invalid_temperatures()
