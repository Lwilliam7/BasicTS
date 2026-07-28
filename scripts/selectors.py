"""Leakage-safe selector helpers for saved forecasting predictions.

Each selector learns its model choices or blending weights from validation
predictions and validation targets, then applies the frozen selector to saved
final-split predictions. Do not pass test targets into these functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ArrayLike = np.ndarray | str | Path
PredictionCollection = ArrayLike | Sequence[ArrayLike] | Mapping[str, ArrayLike]


def hard_selector(
    saved_predictions: PredictionCollection,
    val_predictions: PredictionCollection,
    val_targets: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the lowest-validation-MAE model per horizon step and feature.

    Args:
        saved_predictions: Final-split predictions to combine. Accepts a model
            stack with shape ``[models, samples, horizon, features]``, a list of
            arrays/paths, or a dict of name -> array/path.
        val_predictions: Validation predictions with the same model order and
            shape convention as ``saved_predictions``.
        val_targets: Validation targets with shape ``[samples, horizon, features]``.

    Returns:
        ``(final_predictions, weights)`` where ``weights`` has shape
        ``[models, horizon, features]`` and is one-hot.
    """

    test_stack, val_stack, targets = _prepare_inputs(
        saved_predictions, val_predictions, val_targets
    )
    val_mae = _validation_mae(val_stack, targets)
    winner = np.argmin(val_mae, axis=0)
    weights = np.zeros_like(val_mae, dtype=np.float64)
    np.put_along_axis(weights, winner[np.newaxis, :, :], 1.0, axis=0)
    return _apply_weights(test_stack, weights), weights


def soft_weighted_selector(
    saved_predictions: PredictionCollection,
    val_predictions: PredictionCollection,
    val_targets: ArrayLike,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend models using normalized inverse validation MAE weights."""

    test_stack, val_stack, targets = _prepare_inputs(
        saved_predictions, val_predictions, val_targets
    )
    val_mae = _validation_mae(val_stack, targets)
    scores = 1.0 / np.maximum(val_mae, eps)
    weights = _normalize_model_scores(scores)
    return _apply_weights(test_stack, weights), weights


def smoothed_soft_selector(
    saved_predictions: PredictionCollection,
    val_predictions: PredictionCollection,
    val_targets: ArrayLike,
    temperature: float = 2.0,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend models with softened inverse-MAE weights.

    ``temperature=1`` is equivalent to ``soft_weighted_selector``. Larger
    temperatures move weights closer to uniform when validation MAE gaps are
    small, which can reduce overconfident per-cell choices.
    """

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    test_stack, val_stack, targets = _prepare_inputs(
        saved_predictions, val_predictions, val_targets
    )
    val_mae = _validation_mae(val_stack, targets)
    scores = (1.0 / np.maximum(val_mae, eps)) ** (1.0 / temperature)
    weights = _normalize_model_scores(scores)
    return _apply_weights(test_stack, weights), weights


def temperature_softmax_selector(
    saved_predictions: PredictionCollection,
    val_predictions: PredictionCollection,
    val_targets: ArrayLike,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend models with a softmax over negative validation MAE.

    Lower temperatures make the selector closer to a hard winner-take-all rule.
    Higher temperatures make weights closer to uniform.
    """

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    test_stack, val_stack, targets = _prepare_inputs(
        saved_predictions, val_predictions, val_targets
    )
    val_mae = _validation_mae(val_stack, targets)
    logits = -val_mae / max(temperature, eps)
    weights = _softmax(logits, axis=0)
    return _apply_weights(test_stack, weights), weights


def temperature_search_softmax_selector(
    saved_predictions: PredictionCollection,
    val_predictions: PredictionCollection,
    val_targets: ArrayLike,
    temperatures: Sequence[float] = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0),
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Pick a softmax temperature on validation MAE, then apply it to final predictions.

    Each candidate temperature produces a frozen per-step/per-feature weight
    matrix from validation MAE. The returned prediction uses the temperature
    whose blended validation predictions have the lowest overall MAE.
    """

    if not temperatures:
        raise ValueError("temperatures must contain at least one value")
    for temperature in temperatures:
        if temperature <= 0:
            raise ValueError(f"temperatures must be positive, got {temperature}")

    test_stack, val_stack, targets = _prepare_inputs(
        saved_predictions, val_predictions, val_targets
    )
    val_mae = _validation_mae(val_stack, targets)

    sweep = []
    best_score = np.inf
    best_temperature = None
    best_weights = None

    for temperature in temperatures:
        logits = -val_mae / max(float(temperature), eps)
        weights = _softmax(logits, axis=0)
        val_prediction = _apply_weights(val_stack, weights)
        score = float(np.mean(np.abs(val_prediction - targets)))
        sweep.append({"temperature": float(temperature), "validation_mae": score})

        if score < best_score:
            best_score = score
            best_temperature = float(temperature)
            best_weights = weights

    if best_weights is None or best_temperature is None:
        raise ValueError("temperature search did not evaluate any candidates")

    metadata = {
        "temperatures": [float(temperature) for temperature in temperatures],
        "validation_mae_by_temperature": sweep,
        "best_temperature": best_temperature,
        "best_validation_mae": best_score,
    }
    return _apply_weights(test_stack, best_weights), best_weights, metadata


def global_learned_blend_selector(
    saved_predictions: PredictionCollection,
    val_predictions: PredictionCollection,
    val_targets: ArrayLike,
    ridge_alpha: float = 1e-6,
    nonnegative: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Learn one global linear blend from validation predictions."""

    test_stack, val_stack, targets = _prepare_inputs(
        saved_predictions, val_predictions, val_targets
    )
    x = val_stack.reshape(val_stack.shape[0], -1).T
    y = targets.reshape(-1)
    coef = _fit_linear_weights(x, y, ridge_alpha, nonnegative)
    weights = np.broadcast_to(
        coef[:, np.newaxis, np.newaxis],
        (test_stack.shape[0], *test_stack.shape[2:]),
    ).copy()
    return _apply_weights(test_stack, weights), weights


def per_step_learned_blend_selector(
    saved_predictions: PredictionCollection,
    val_predictions: PredictionCollection,
    val_targets: ArrayLike,
    ridge_alpha: float = 1e-6,
    nonnegative: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Learn one validation-fitted blend per forecast step."""

    test_stack, val_stack, targets = _prepare_inputs(
        saved_predictions, val_predictions, val_targets
    )
    num_models, _, horizon, num_features = val_stack.shape
    weights = np.zeros((num_models, horizon, num_features), dtype=np.float64)

    for step in range(horizon):
        x = val_stack[:, :, step, :].transpose(1, 2, 0).reshape(-1, num_models)
        y = targets[:, step, :].reshape(-1)
        coef = _fit_linear_weights(x, y, ridge_alpha, nonnegative)
        weights[:, step, :] = coef[:, np.newaxis]

    return _apply_weights(test_stack, weights), weights


def per_step_feature_learned_blend_selector(
    saved_predictions: PredictionCollection,
    val_predictions: PredictionCollection,
    val_targets: ArrayLike,
    ridge_alpha: float = 1e-6,
    nonnegative: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Learn one validation-fitted blend per forecast step and feature."""

    test_stack, val_stack, targets = _prepare_inputs(
        saved_predictions, val_predictions, val_targets
    )
    num_models, _, horizon, num_features = val_stack.shape
    weights = np.zeros((num_models, horizon, num_features), dtype=np.float64)

    for step in range(horizon):
        for feature in range(num_features):
            x = val_stack[:, :, step, feature].T
            y = targets[:, step, feature]
            weights[:, step, feature] = _fit_linear_weights(
                x, y, ridge_alpha, nonnegative
            )

    return _apply_weights(test_stack, weights), weights


def ridge_stacking_selector(
    saved_predictions: PredictionCollection,
    val_predictions: PredictionCollection,
    val_targets: ArrayLike,
    ridge_alpha: float = 1e-3,
    fit_intercept: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Alias for per-step/per-feature ridge stacking with an intercept."""

    return tiny_meta_selector(
        saved_predictions=saved_predictions,
        val_predictions=val_predictions,
        val_targets=val_targets,
        ridge_alpha=ridge_alpha,
        fit_intercept=fit_intercept,
    )


def logistic_winner_selector(
    saved_predictions: PredictionCollection,
    val_predictions: PredictionCollection,
    val_targets: ArrayLike,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-model logistic gate using validation MAE differences.

    This is a lightweight logistic approximation: lower validation MAE receives
    higher probability, with ``temperature`` controlling gate sharpness.
    """

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    test_stack, val_stack, targets = _prepare_inputs(
        saved_predictions, val_predictions, val_targets
    )
    if test_stack.shape[0] != 2:
        raise ValueError("logistic_winner_selector currently expects exactly two models")

    val_mae = _validation_mae(val_stack, targets)
    logits = (val_mae[0] - val_mae[1]) / max(temperature, eps)
    second_weight = 1.0 / (1.0 + np.exp(-logits))
    weights = np.stack([1.0 - second_weight, second_weight], axis=0)
    return _apply_weights(test_stack, weights), weights


def gradient_boosting_stacking_selector(
    saved_predictions: PredictionCollection,
    val_predictions: PredictionCollection,
    val_targets: ArrayLike,
    **kwargs,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit a scikit-learn gradient boosting stacker per step and feature."""

    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError as exc:
        raise ImportError(
            "gradient_boosting_stacking_selector requires scikit-learn"
        ) from exc

    test_stack, val_stack, targets = _prepare_inputs(
        saved_predictions, val_predictions, val_targets
    )
    num_models, _, horizon, num_features = val_stack.shape
    final_predictions = np.zeros_like(test_stack[0], dtype=np.float64)
    models = {}

    params = {"random_state": 0, "n_estimators": 50, "max_depth": 2}
    params.update(kwargs)

    for step in range(horizon):
        for feature in range(num_features):
            model = GradientBoostingRegressor(**params)
            x_val = val_stack[:, :, step, feature].T
            y_val = targets[:, step, feature]
            model.fit(x_val, y_val)
            x_test = test_stack[:, :, step, feature].T
            final_predictions[:, step, feature] = model.predict(x_test)
            models[(step, feature)] = model

    return final_predictions, {"models": models, "params": params}


def mlp_gating_selector(
    saved_predictions: PredictionCollection,
    val_predictions: PredictionCollection,
    val_targets: ArrayLike,
    **kwargs,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit a scikit-learn MLP stacker per step and feature."""

    try:
        from sklearn.neural_network import MLPRegressor
    except ImportError as exc:
        raise ImportError("mlp_gating_selector requires scikit-learn") from exc

    test_stack, val_stack, targets = _prepare_inputs(
        saved_predictions, val_predictions, val_targets
    )
    _, _, horizon, num_features = val_stack.shape
    final_predictions = np.zeros_like(test_stack[0], dtype=np.float64)
    models = {}

    params = {
        "hidden_layer_sizes": (16,),
        "alpha": 1e-3,
        "max_iter": 500,
        "random_state": 0,
    }
    params.update(kwargs)

    for step in range(horizon):
        for feature in range(num_features):
            model = MLPRegressor(**params)
            x_val = val_stack[:, :, step, feature].T
            y_val = targets[:, step, feature]
            model.fit(x_val, y_val)
            x_test = test_stack[:, :, step, feature].T
            final_predictions[:, step, feature] = model.predict(x_test)
            models[(step, feature)] = model

    return final_predictions, {"models": models, "params": params}


def tiny_meta_selector(
    saved_predictions: PredictionCollection,
    val_predictions: PredictionCollection,
    val_targets: ArrayLike,
    ridge_alpha: float = 1e-3,
    fit_intercept: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Train a tiny ridge-stacking selector on validation predictions.

    For each horizon step and feature, this fits a small linear model from the
    validation predictions of all base models to the validation target, then
    applies those frozen coefficients to ``saved_predictions``.

    Returns:
        ``(final_predictions, weights)``. ``weights`` is a dict with
        ``coef`` shaped ``[models, horizon, features]`` and ``intercept`` shaped
        ``[horizon, features]``.
    """

    if ridge_alpha < 0:
        raise ValueError(f"ridge_alpha must be nonnegative, got {ridge_alpha}")

    test_stack, val_stack, targets = _prepare_inputs(
        saved_predictions, val_predictions, val_targets
    )
    num_models, _, horizon, num_features = val_stack.shape
    coef = np.zeros((num_models, horizon, num_features), dtype=np.float64)
    intercept = np.zeros((horizon, num_features), dtype=np.float64)

    for step in range(horizon):
        for feature in range(num_features):
            x = val_stack[:, :, step, feature].T.astype(np.float64, copy=False)
            y = targets[:, step, feature].astype(np.float64, copy=False)

            if fit_intercept:
                x_fit = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
                penalty = np.eye(num_models + 1, dtype=np.float64) * ridge_alpha
                penalty[-1, -1] = 0.0
            else:
                x_fit = x
                penalty = np.eye(num_models, dtype=np.float64) * ridge_alpha

            solution = np.linalg.pinv(x_fit.T @ x_fit + penalty) @ x_fit.T @ y
            coef[:, step, feature] = solution[:num_models]
            if fit_intercept:
                intercept[step, feature] = solution[-1]

    final_predictions = _apply_weights(test_stack, coef) + intercept[np.newaxis, :, :]
    return final_predictions, {"coef": coef, "intercept": intercept}


def _prepare_inputs(
    saved_predictions: PredictionCollection,
    val_predictions: PredictionCollection,
    val_targets: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    test_stack = _as_model_stack(saved_predictions, "saved_predictions")
    val_stack = _as_model_stack(val_predictions, "val_predictions")
    targets = _load_array(val_targets, "val_targets")

    if test_stack.shape[0] != val_stack.shape[0]:
        raise ValueError(
            "saved_predictions and val_predictions must contain the same number "
            f"of models, got {test_stack.shape[0]} and {val_stack.shape[0]}"
        )
    if test_stack.shape[2:] != val_stack.shape[2:]:
        raise ValueError(
            "saved_predictions and val_predictions must have matching horizon "
            f"and feature dimensions, got {test_stack.shape[2:]} and {val_stack.shape[2:]}"
        )
    if val_stack.shape[1:] != targets.shape:
        raise ValueError(
            "val_predictions per-model shape must match val_targets shape, got "
            f"{val_stack.shape[1:]} and {targets.shape}"
        )

    return test_stack, val_stack, targets


def _as_model_stack(predictions: PredictionCollection, name: str) -> np.ndarray:
    if isinstance(predictions, Mapping):
        if not predictions:
            raise ValueError(f"{name} must contain at least one model")
        arrays = [_load_array(value, f"{name}[{key!r}]") for key, value in predictions.items()]
        stack = np.stack(arrays, axis=0)
    elif isinstance(predictions, (list, tuple)):
        if not predictions:
            raise ValueError(f"{name} must contain at least one model")
        arrays = [_load_array(value, f"{name}[{idx}]") for idx, value in enumerate(predictions)]
        stack = np.stack(arrays, axis=0)
    else:
        stack = _load_array(predictions, name)

    if stack.ndim != 4:
        raise ValueError(
            f"{name} must have shape [models, samples, horizon, features], "
            f"got shape {stack.shape}"
        )
    return stack.astype(np.float64, copy=False)


def _load_array(value: ArrayLike, name: str) -> np.ndarray:
    if isinstance(value, (str, Path)):
        array = np.load(value)
    else:
        array = np.asarray(value)

    if array.ndim == 0:
        raise ValueError(f"{name} must be an array, got scalar")
    return array


def _validation_mae(val_stack: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(val_stack - targets[np.newaxis, :, :, :]), axis=1)


def _normalize_model_scores(scores: np.ndarray) -> np.ndarray:
    denom = np.sum(scores, axis=0, keepdims=True)
    if np.any(denom <= 0) or not np.all(np.isfinite(denom)):
        raise ValueError("selector scores produced invalid normalization values")
    return scores / denom


def _fit_linear_weights(
    x: np.ndarray,
    y: np.ndarray,
    ridge_alpha: float,
    nonnegative: bool,
) -> np.ndarray:
    if ridge_alpha < 0:
        raise ValueError(f"ridge_alpha must be nonnegative, got {ridge_alpha}")

    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    num_models = x.shape[1]
    penalty = np.eye(num_models, dtype=np.float64) * ridge_alpha
    coef = np.linalg.pinv(x.T @ x + penalty) @ x.T @ y

    if nonnegative:
        coef = np.maximum(coef, 0.0)

    total = float(np.sum(coef))
    if total <= 0 or not np.isfinite(total):
        return np.full(num_models, 1.0 / num_models, dtype=np.float64)
    return coef / total


def _softmax(logits: np.ndarray, axis: int) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)


def _apply_weights(test_stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if weights.shape != (test_stack.shape[0], *test_stack.shape[2:]):
        raise ValueError(
            "weights must have shape [models, horizon, features], got "
            f"{weights.shape} for predictions {test_stack.shape}"
        )
    return np.sum(weights[:, np.newaxis, :, :] * test_stack, axis=0)
