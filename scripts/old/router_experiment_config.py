"""Central configuration for frozen-expert routing experiments."""
#import statemetns
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union


AVAILABLE_EXPERTS = (
    "DLinear",
    "PatchTST",
    "iTransformer",
    "TimesNet",
    "ModernTCN",
)
#weird sand model steeing
DEFAULT_EXPERT_CHECKPOINT_NAMES = {
    "DLinear": "best_dlinear.pt",
    "PatchTST": "best_patchtst.pt",
    "iTransformer": "best_itransformer.pt",
    "TimesNet": "best_timesnet.pt",
    "ModernTCN": "best_moderntcn.pt",
}
# results
DEFAULT_CACHE_PATHS = {
    "routerdc_train": "cache/routerdc_train_cache.pt",
    "routerdc_val": "cache/routerdc_val_cache.pt",
    "routerdc_test": "cache/routerdc_test_cache.pt",
    "costarts_train": "cache/costarts_router_train_cache.pt",
    "costarts_val": "cache/costarts_router_val_cache.pt",
    "costarts_subset_states_train": "cache/costarts_subset_states_train.pt",
    "costarts_subset_states_val": "cache/costarts_subset_states_val.pt",
}

SUPPORTED_ROUTER_TYPES = (
    "prediction_aware",
    "original",
    "multiscale_tcn_expert_embeddings",
    "routerdc_hard",
    "costarts",
)


@dataclass(frozen=True)
class RouterExperimentConfig:
    #configeration class(normal settings)
    """Single interface for frozen-expert router experiment settings."""

    selected_expert_models: tuple[str, ...] = ("DLinear", "PatchTST")
    selected_model_groups: Optional[tuple[tuple[str, ...], ...]] = None
    auto_select_best_by_size: bool = True
    best_model_counts: Union[str, tuple[int, ...]] = "all"
    best_combination_results_path: str = (
        "results/router_summary/candidate_expert_combinations.json"
    )

    router_type: str = "prediction_aware"
    embedding_dim: int = 64
    hidden_dim: int = 64
    queried_experts_cap_k: Optional[int] = None
    routing_temperature: float = 1.0
    loss_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "forecast": 1.0,
            "window_expert": 1.0,
            "window_window": 1.0,
        }
    )
    stop_threshold: float = 0.0
    cost_weights: Mapping[str, float] = field(default_factory=dict)
    costarts_subset_max_size: Optional[int] = None
    costarts_subset_include_empty: bool = True
    costarts_subset_utility_cost_coefficient: float = 1.0
    costarts_subset_cost_schedule: Mapping[str, float] = field(default_factory=dict)
    costarts_subset_sampling_mode: str = "exhaustive"
    costarts_subset_utility_action_loss_weight: float = 1.0
    costarts_subset_utility_loss_weight: float = 1.0
    costarts_subset_utility_pairwise_loss_weight: float = 0.2
    costarts_subset_utility_mix_loss_weight: float = 0.0
    costarts_subset_utility_learning_rate: float = 1e-3
    costarts_subset_utility_weight_decay: float = 0.0
    costarts_subset_utility_grad_clip_norm: float = 1.0
    costarts_subset_utility_batch_size: int = 512
    costarts_subset_utility_max_epochs: int = 50
    costarts_subset_utility_patience: int = 10

    random_seed: int = 7
    dataset_name: str = "ETTh1"
    data_dir: str = "datasets/ETTh1"
    input_length: int = 96
    forecast_horizon: int = 12
    num_features: int = 7
    selected_variables: Optional[tuple[int, ...]] = None

    checkpoint_dir: str = "checkpoints"
    expert_checkpoint_paths: Mapping[str, str] = field(default_factory=dict)
    cache_paths: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_CACHE_PATHS))
    debug_mode: bool = False
    #returs a dict creates a custom checkpoint for everyh path
    def normalized_expert_checkpoint_paths(self) -> dict[str, Path]:
        checkpoint_dir = Path(self.checkpoint_dir)
        paths = {
            name: checkpoint_dir / "candidates" / checkpoint_name
            for name, checkpoint_name in DEFAULT_EXPERT_CHECKPOINT_NAMES.items()
        }
        for name, path in self.expert_checkpoint_paths.items():
            paths[str(name)] = Path(path)
        return paths
    # returns a tuple of selected models, either from groups or from selected_expert_models
    def effective_selected_models(self) -> tuple[str, ...]:
        if self.selected_model_groups:
            models: list[str] = []
            for group in self.selected_model_groups:
                for model_name in group:
                    if model_name not in models:
                        models.append(model_name)
            return tuple(models)
        return self.selected_expert_models

# convers values into a tuple
def _as_tuple(value: Any) -> tuple:
    if value is None:
        return tuple()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)

#tuple with onne
def _optional_tuple(value: Any) -> Optional[tuple]:
    if value is None:
        return None
    return _as_tuple(value)


def _module_value(module: Any, modern_name: str, legacy_name: str, default: Any) -> Any:
    if hasattr(module, modern_name):
        return getattr(module, modern_name)
    return getattr(module, legacy_name, default)


def load_router_experiment_config(module: Optional[Any] = None) -> RouterExperimentConfig:
    """Load config from scripts/router_model_config.py with legacy-name support."""
    # inport the module if not provided
    if module is None:
        try:
            from scripts.old import router_model_config as module
        except ImportError:
            import scripts.old.router_model_config as module
    # check if the module already has a RouterExperimentConfig instance
    existing = getattr(module, "ROUTER_EXPERIMENT_CONFIG", None)
    if isinstance(existing, RouterExperimentConfig):
        return existing
#build a new config file
    return RouterExperimentConfig(
        selected_expert_models=tuple(
            _module_value(module, "SELECTED_EXPERT_MODELS", "SELECTED_MODELS", ("DLinear", "PatchTST"))
        ),
        selected_model_groups=_optional_tuple(
            _module_value(module, "SELECTED_MODEL_GROUPS", "SELECTED_MODEL_GROUPS", None)
        ),
        auto_select_best_by_size=bool(
            _module_value(module, "AUTO_SELECT_BEST_BY_SIZE", "AUTO_SELECT_BEST_BY_SIZE", True)
        ),
        best_model_counts=_module_value(module, "BEST_MODEL_COUNTS", "BEST_MODEL_COUNTS", "all"),
        best_combination_results_path=str(
            _module_value(
                module,
                "BEST_COMBINATION_RESULTS_PATH",
                "BEST_COMBINATION_RESULTS_PATH",
                "results/router_summary/candidate_expert_combinations.json",
            )
        ),
        router_type=str(_module_value(module, "ROUTER_TYPE", "ROUTER_TYPE", "prediction_aware")),
        embedding_dim=int(_module_value(module, "EMBEDDING_DIM", "EMBEDDING_DIM", 64)),
        hidden_dim=int(_module_value(module, "HIDDEN_DIM", "HIDDEN_DIM", 64)),
        queried_experts_cap_k=_module_value(module, "QUERIED_EXPERTS_CAP_K", "QUERIED_EXPERTS_CAP_K", None),
        routing_temperature=float(_module_value(module, "ROUTING_TEMPERATURE", "ROUTING_TEMPERATURE", 1.0)),
        loss_weights=dict(_module_value(module, "LOSS_WEIGHTS", "LOSS_WEIGHTS", {
            "forecast": 1.0,
            "window_expert": 1.0,
            "window_window": 1.0,
        })),
        stop_threshold=float(_module_value(module, "STOP_THRESHOLD", "STOP_THRESHOLD", 0.0)),
        cost_weights=dict(_module_value(module, "COST_WEIGHTS", "COST_WEIGHTS", {})),
        costarts_subset_max_size=_module_value(
            module,
            "COSTARTS_SUBSET_MAX_SIZE",
            "COSTARTS_SUBSET_MAX_SIZE",
            None,
        ),
        costarts_subset_include_empty=bool(
            _module_value(
                module,
                "COSTARTS_SUBSET_INCLUDE_EMPTY",
                "COSTARTS_SUBSET_INCLUDE_EMPTY",
                True,
            )
        ),
        costarts_subset_utility_cost_coefficient=float(
            _module_value(
                module,
                "COSTARTS_SUBSET_UTILITY_COST_COEFFICIENT",
                "COSTARTS_SUBSET_UTILITY_COST_COEFFICIENT",
                1.0,
            )
        ),
        costarts_subset_cost_schedule=dict(
            _module_value(
                module,
                "COSTARTS_SUBSET_COST_SCHEDULE",
                "COSTARTS_SUBSET_COST_SCHEDULE",
                {},
            )
        ),
        costarts_subset_sampling_mode=str(
            _module_value(
                module,
                "COSTARTS_SUBSET_SAMPLING_MODE",
                "COSTARTS_SUBSET_SAMPLING_MODE",
                "exhaustive",
            )
        ),
        costarts_subset_utility_action_loss_weight=float(
            _module_value(
                module,
                "COSTARTS_SUBSET_UTILITY_ACTION_LOSS_WEIGHT",
                "COSTARTS_SUBSET_UTILITY_ACTION_LOSS_WEIGHT",
                1.0,
            )
        ),
        costarts_subset_utility_loss_weight=float(
            _module_value(
                module,
                "COSTARTS_SUBSET_UTILITY_LOSS_WEIGHT",
                "COSTARTS_SUBSET_UTILITY_LOSS_WEIGHT",
                1.0,
            )
        ),
        costarts_subset_utility_pairwise_loss_weight=float(
            _module_value(
                module,
                "COSTARTS_SUBSET_UTILITY_PAIRWISE_LOSS_WEIGHT",
                "COSTARTS_SUBSET_UTILITY_PAIRWISE_LOSS_WEIGHT",
                0.2,
            )
        ),
        costarts_subset_utility_mix_loss_weight=float(
            _module_value(
                module,
                "COSTARTS_SUBSET_UTILITY_MIX_LOSS_WEIGHT",
                "COSTARTS_SUBSET_UTILITY_MIX_LOSS_WEIGHT",
                0.0,
            )
        ),
        costarts_subset_utility_learning_rate=float(
            _module_value(
                module,
                "COSTARTS_SUBSET_UTILITY_LEARNING_RATE",
                "COSTARTS_SUBSET_UTILITY_LEARNING_RATE",
                1e-3,
            )
        ),
        costarts_subset_utility_weight_decay=float(
            _module_value(
                module,
                "COSTARTS_SUBSET_UTILITY_WEIGHT_DECAY",
                "COSTARTS_SUBSET_UTILITY_WEIGHT_DECAY",
                0.0,
            )
        ),
        costarts_subset_utility_grad_clip_norm=float(
            _module_value(
                module,
                "COSTARTS_SUBSET_UTILITY_GRAD_CLIP_NORM",
                "COSTARTS_SUBSET_UTILITY_GRAD_CLIP_NORM",
                1.0,
            )
        ),
        costarts_subset_utility_batch_size=int(
            _module_value(
                module,
                "COSTARTS_SUBSET_UTILITY_BATCH_SIZE",
                "COSTARTS_SUBSET_UTILITY_BATCH_SIZE",
                512,
            )
        ),
        costarts_subset_utility_max_epochs=int(
            _module_value(
                module,
                "COSTARTS_SUBSET_UTILITY_MAX_EPOCHS",
                "COSTARTS_SUBSET_UTILITY_MAX_EPOCHS",
                50,
            )
        ),
        costarts_subset_utility_patience=int(
            _module_value(
                module,
                "COSTARTS_SUBSET_UTILITY_PATIENCE",
                "COSTARTS_SUBSET_UTILITY_PATIENCE",
                10,
            )
        ),
        random_seed=int(_module_value(module, "RANDOM_SEED", "SEED", 7)),
        dataset_name=str(_module_value(module, "DATASET_NAME", "DATASET_NAME", "ETTh1")),
        data_dir=str(_module_value(module, "DATA_DIR", "DATA_DIR", "datasets/ETTh1")),
        input_length=int(_module_value(module, "INPUT_LENGTH", "INPUT_LEN", 96)),
        forecast_horizon=int(_module_value(module, "FORECAST_HORIZON", "OUTPUT_LEN", 12)),
        num_features=int(_module_value(module, "NUM_FEATURES", "NUM_FEATURES", 7)),
        selected_variables=_optional_tuple(
            _module_value(module, "SELECTED_VARIABLES", "SELECTED_VARIABLES", None)
        ),
        checkpoint_dir=str(_module_value(module, "CHECKPOINT_DIR", "CHECKPOINT_DIR", "checkpoints")),
        expert_checkpoint_paths=dict(
            _module_value(module, "EXPERT_CHECKPOINT_PATHS", "EXPERT_CHECKPOINT_PATHS", {})
        ),
        cache_paths=dict(_module_value(module, "CACHE_PATHS", "CACHE_PATHS", DEFAULT_CACHE_PATHS)),
        debug_mode=bool(_module_value(module, "DEBUG_MODE", "DEBUG_MODE", False)),
    )

#checks if the selected naem is valid and if the dimensions are valid
def _validate_model_names(config: RouterExperimentConfig) -> None:
    selected_models = config.effective_selected_models()
    if not selected_models:
        raise ValueError("At least one selected expert model is required")
    invalid = [name for name in selected_models if name not in AVAILABLE_EXPERTS]
    if invalid:
        raise ValueError(
            f"Unknown selected expert model(s): {invalid}. "
            f"Available: {list(AVAILABLE_EXPERTS)}"
        )
    duplicates = sorted({name for name in selected_models if selected_models.count(name) > 1})
    if duplicates:
        raise ValueError(f"Selected expert models contain duplicates: {duplicates}")

#checks if valuedate weights
def _validate_dimensions(config: RouterExperimentConfig) -> None:
    positive_ints = {
        "embedding_dim": config.embedding_dim,
        "hidden_dim": config.hidden_dim,
        "random_seed": config.random_seed,
        "input_length": config.input_length,
        "forecast_horizon": config.forecast_horizon,
        "num_features": config.num_features,
    }
    for name, value in positive_ints.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}")
    if config.input_length != 96 or config.forecast_horizon != 12:
        raise ValueError(
            "Current ETTh1 expert checkpoints assume input_length=96 and "
            "forecast_horizon=12. Retrain experts before changing these."
        )
    if config.routing_temperature <= 0:
        raise ValueError("routing_temperature must be positive")
    if config.stop_threshold < 0:
        raise ValueError("stop_threshold must be non-negative")
    if config.costarts_subset_max_size is not None:
        subset_cap = int(config.costarts_subset_max_size)
        if subset_cap < 0:
            raise ValueError("costarts_subset_max_size must be non-negative")
        if subset_cap > len(config.effective_selected_models()):
            raise ValueError("costarts_subset_max_size cannot exceed the number of selected experts")
    if config.costarts_subset_utility_cost_coefficient < 0:
        raise ValueError("costarts_subset_utility_cost_coefficient must be non-negative")
    if config.costarts_subset_sampling_mode not in {
        "exhaustive",
        "random",
        "oracle_path_only",
        "model_induced_states",
    }:
        raise ValueError(
            "costarts_subset_sampling_mode must be one of "
            "exhaustive, random, oracle_path_only, model_induced_states"
        )
    _validate_weights(
        "costarts subset utility loss weights",
        {
            "action": config.costarts_subset_utility_action_loss_weight,
            "utility": config.costarts_subset_utility_loss_weight,
            "pairwise": config.costarts_subset_utility_pairwise_loss_weight,
            "mix": config.costarts_subset_utility_mix_loss_weight,
        },
    )
    if config.costarts_subset_utility_learning_rate <= 0:
        raise ValueError("costarts_subset_utility_learning_rate must be positive")
    if config.costarts_subset_utility_weight_decay < 0:
        raise ValueError("costarts_subset_utility_weight_decay must be non-negative")
    if config.costarts_subset_utility_grad_clip_norm < 0:
        raise ValueError("costarts_subset_utility_grad_clip_norm must be non-negative")
    if config.costarts_subset_utility_batch_size <= 0:
        raise ValueError("costarts_subset_utility_batch_size must be positive")
    if config.costarts_subset_utility_max_epochs <= 0:
        raise ValueError("costarts_subset_utility_max_epochs must be positive")
    if config.costarts_subset_utility_patience <= 0:
        raise ValueError("costarts_subset_utility_patience must be positive")
    if config.queried_experts_cap_k is not None:
        cap = int(config.queried_experts_cap_k)
        if cap <= 0:
            raise ValueError("queried_experts_cap_k must be positive when set")
        max_selected = (
            len(AVAILABLE_EXPERTS)
            if config.auto_select_best_by_size
            else len(config.effective_selected_models())
        )
        if cap > max_selected:
            raise ValueError(
                "queried_experts_cap_k cannot exceed the number of selected experts"
            )
    if config.selected_variables is not None:
        if not config.selected_variables:
            raise ValueError("selected_variables cannot be empty when provided")
        for variable_index in config.selected_variables:
            if not isinstance(variable_index, int):
                raise ValueError("selected_variables must contain integer indices")
            if variable_index < 0 or variable_index >= config.num_features:
                raise ValueError(
                    f"selected variable {variable_index} is outside [0, {config.num_features})"
                )

#checks if none of the loss weights are negative 
def _validate_weights(name: str, values: Mapping[str, float]) -> None:
    for key, value in values.items():
        if float(value) < 0:
            raise ValueError(f"{name}[{key!r}] must be non-negative")

#checks eventhing before training
def validate_router_experiment_config(
    config: Optional[RouterExperimentConfig] = None,
    *,
    require_checkpoints: bool = True,
    require_data: bool = True,
    require_cache_parent: bool = True,
) -> RouterExperimentConfig:
    """Validate config values and return the normalized config."""

    config = config or load_router_experiment_config()
    _validate_model_names(config)
    _validate_dimensions(config)
    _validate_weights("loss_weights", config.loss_weights)
    _validate_weights("cost_weights", config.cost_weights)
    if config.router_type not in SUPPORTED_ROUTER_TYPES:
        raise ValueError(
            f"router_type must be one of {SUPPORTED_ROUTER_TYPES}; got {config.router_type!r}"
        )

    if require_data:
        data_dir = Path(config.data_dir)
        missing_data = [
            data_dir / "train_data.npy",
            data_dir / "val_data.npy",
            data_dir / "test_data.npy",
        ]
        missing_data = [path for path in missing_data if not path.exists()]
        if missing_data:
            raise FileNotFoundError(
                "Missing dataset files: "
                + ", ".join(str(path) for path in missing_data)
            )

    if require_checkpoints:
        checkpoint_paths = config.normalized_expert_checkpoint_paths()
        models_requiring_checkpoints = (
            AVAILABLE_EXPERTS
            if config.auto_select_best_by_size
            else config.effective_selected_models()
        )
        missing = [
            checkpoint_paths[name]
            for name in models_requiring_checkpoints
            if not checkpoint_paths[name].exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing selected expert checkpoint(s): "
                + ", ".join(str(path) for path in missing)
            )

    if require_cache_parent:
        for cache_name, cache_path_value in config.cache_paths.items():
            cache_path = Path(cache_path_value)
            if str(cache_path).strip() == "":
                raise ValueError(f"cache path {cache_name!r} is empty")
            parent = cache_path.parent
            if parent != Path(".") and not parent.exists():
                raise FileNotFoundError(
                    f"Cache parent for {cache_name!r} does not exist: {parent}"
                )
            if cache_path.exists() and cache_path.is_dir():
                raise ValueError(f"Cache path {cache_name!r} is a directory: {cache_path}")

    return config

#prints the current roster setting to the terminal so i can see waht the expierment will use
def print_router_experiment_config(config: Optional[RouterExperimentConfig] = None) -> None:
    """Print the key config values used by router experiments."""

    config = config or load_router_experiment_config()
    print("\nRouter experiment configuration")
    print(f"  dataset: {config.dataset_name}")
    print(f"  data_dir: {config.data_dir}")
    print(f"  input_length: {config.input_length}")
    print(f"  forecast_horizon: {config.forecast_horizon}")
    print(f"  num_features: {config.num_features}")
    print(f"  selected_variables: {config.selected_variables or 'all'}")
    print(f"  selected_expert_models: {list(config.selected_expert_models)}")
    print(f"  selected_model_groups: {config.selected_model_groups or 'from selected/default'}")
    print(f"  auto_select_best_by_size: {config.auto_select_best_by_size}")
    print(f"  best_model_counts: {config.best_model_counts}")
    print(f"  best_combination_results_path: {config.best_combination_results_path}")
    print(f"  router_type: {config.router_type}")
    print(f"  embedding_dim: {config.embedding_dim}")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  queried_experts_cap_k: {config.queried_experts_cap_k or 'none'}")
    print(f"  routing_temperature: {config.routing_temperature}")
    print(f"  loss_weights: {dict(config.loss_weights)}")
    print(f"  stop_threshold: {config.stop_threshold}")
    print(f"  cost_weights: {dict(config.cost_weights)}")
    print(f"  costarts_subset_max_size: {config.costarts_subset_max_size or 'all'}")
    print(f"  costarts_subset_include_empty: {config.costarts_subset_include_empty}")
    print(
        "  costarts_subset_utility_cost_coefficient: "
        f"{config.costarts_subset_utility_cost_coefficient}"
    )
    print(f"  costarts_subset_cost_schedule: {dict(config.costarts_subset_cost_schedule)}")
    print(f"  costarts_subset_sampling_mode: {config.costarts_subset_sampling_mode}")
    print(
        "  costarts_subset_utility_loss_weights: "
        f"action={config.costarts_subset_utility_action_loss_weight}, "
        f"utility={config.costarts_subset_utility_loss_weight}, "
        f"pairwise={config.costarts_subset_utility_pairwise_loss_weight}, "
        f"mix={config.costarts_subset_utility_mix_loss_weight}"
    )
    print(
        "  costarts_subset_utility_optimization: "
        f"lr={config.costarts_subset_utility_learning_rate}, "
        f"weight_decay={config.costarts_subset_utility_weight_decay}, "
        f"grad_clip={config.costarts_subset_utility_grad_clip_norm}, "
        f"batch_size={config.costarts_subset_utility_batch_size}, "
        f"max_epochs={config.costarts_subset_utility_max_epochs}, "
        f"patience={config.costarts_subset_utility_patience}"
    )
    print(f"  random_seed: {config.random_seed}")
    print(f"  checkpoint_dir: {config.checkpoint_dir}")
    print(f"  cache_paths: {dict(config.cache_paths)}")
    print(f"  debug_mode: {config.debug_mode}")

#make the fun runnable
def main() -> None:
    parser = argparse.ArgumentParser(description="Validate router experiment config.")
    parser.add_argument("--skip-data-check", action="store_true")
    parser.add_argument("--skip-checkpoint-check", action="store_true")
    parser.add_argument("--skip-cache-parent-check", action="store_true")
    args = parser.parse_args()

    config = validate_router_experiment_config(
        require_data=not args.skip_data_check,
        require_checkpoints=not args.skip_checkpoint_check,
        require_cache_parent=not args.skip_cache_parent_check,
    )
    print_router_experiment_config(config)
    print("\nRouter experiment config sanity check passed.")


if __name__ == "__main__":
    main()
