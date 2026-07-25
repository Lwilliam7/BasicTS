"""Central user-editable config for frozen-expert router experiments.

Edit ROUTER_EXPERIMENT_CONFIG below to switch experts, router type, dimensions,
temperatures, cache paths, and debug behavior. The legacy constants at the end
are kept so older notebooks/scripts continue to import the names they expect.
"""

try:
    from scripts.router_experiment_config import RouterExperimentConfig
except ImportError:
    from router_experiment_config import RouterExperimentConfig


ROUTER_EXPERIMENT_CONFIG = RouterExperimentConfig(
    selected_expert_models=(
        "DLinear",
        "PatchTST",
    ),
    selected_model_groups=None,
    auto_select_best_by_size=True,
    best_model_counts="all",
    best_combination_results_path=(
        "results/router_summary/candidate_expert_combinations.json"
    ),
    router_type="prediction_aware",
    embedding_dim=64,
    hidden_dim=64,
    queried_experts_cap_k=None,
    routing_temperature=1.0,
    loss_weights={
        "forecast": 1.0,
        "window_expert": 1.0,
        "window_window": 1.0,
    },
    stop_threshold=0.0,
    cost_weights={},
    costarts_subset_max_size=None,
    costarts_subset_include_empty=True,
    costarts_subset_utility_cost_coefficient=1.0,
    costarts_subset_cost_schedule={},
    costarts_subset_sampling_mode="exhaustive",
    costarts_subset_utility_action_loss_weight=1.0,
    costarts_subset_utility_loss_weight=1.0,
    costarts_subset_utility_pairwise_loss_weight=0.2,
    costarts_subset_utility_mix_loss_weight=0.0,
    costarts_subset_utility_learning_rate=1e-3,
    costarts_subset_utility_weight_decay=0.0,
    costarts_subset_utility_grad_clip_norm=1.0,
    costarts_subset_utility_batch_size=512,
    costarts_subset_utility_max_epochs=50,
    costarts_subset_utility_patience=10,
    random_seed=7,
    dataset_name="ETTh1",
    data_dir="datasets/ETTh1",
    input_length=96,
    forecast_horizon=12,
    num_features=7,
    selected_variables=None,
    checkpoint_dir="checkpoints",
    expert_checkpoint_paths={},
    cache_paths={
        "routerdc_train": "cache/routerdc_train_cache.pt",
        "routerdc_val": "cache/routerdc_val_cache.pt",
        "routerdc_test": "cache/routerdc_test_cache.pt",
        "costarts_train": "cache/costarts_router_train_cache.pt",
        "costarts_val": "cache/costarts_router_val_cache.pt",
        "costarts_subset_states_train": "cache/costarts_subset_states_train.pt",
        "costarts_subset_states_val": "cache/costarts_subset_states_val.pt",
    },
    debug_mode=False,
)


# Backward-compatible aliases used by the existing notebooks and scripts.
SELECTED_MODELS = list(ROUTER_EXPERIMENT_CONFIG.selected_expert_models)
SELECTED_EXPERT_MODELS = SELECTED_MODELS
SELECTED_MODEL_GROUPS = ROUTER_EXPERIMENT_CONFIG.selected_model_groups
AUTO_SELECT_BEST_BY_SIZE = ROUTER_EXPERIMENT_CONFIG.auto_select_best_by_size
BEST_MODEL_COUNTS = ROUTER_EXPERIMENT_CONFIG.best_model_counts
BEST_COMBINATION_RESULTS_PATH = (
    ROUTER_EXPERIMENT_CONFIG.best_combination_results_path
)
ROUTER_TYPE = ROUTER_EXPERIMENT_CONFIG.router_type
EMBEDDING_DIM = ROUTER_EXPERIMENT_CONFIG.embedding_dim
HIDDEN_DIM = ROUTER_EXPERIMENT_CONFIG.hidden_dim
QUERIED_EXPERTS_CAP_K = ROUTER_EXPERIMENT_CONFIG.queried_experts_cap_k
ROUTING_TEMPERATURE = ROUTER_EXPERIMENT_CONFIG.routing_temperature
LOSS_WEIGHTS = dict(ROUTER_EXPERIMENT_CONFIG.loss_weights)
STOP_THRESHOLD = ROUTER_EXPERIMENT_CONFIG.stop_threshold
COST_WEIGHTS = dict(ROUTER_EXPERIMENT_CONFIG.cost_weights)
COSTARTS_SUBSET_MAX_SIZE = ROUTER_EXPERIMENT_CONFIG.costarts_subset_max_size
COSTARTS_SUBSET_INCLUDE_EMPTY = ROUTER_EXPERIMENT_CONFIG.costarts_subset_include_empty
COSTARTS_SUBSET_UTILITY_COST_COEFFICIENT = (
    ROUTER_EXPERIMENT_CONFIG.costarts_subset_utility_cost_coefficient
)
COSTARTS_SUBSET_COST_SCHEDULE = dict(
    ROUTER_EXPERIMENT_CONFIG.costarts_subset_cost_schedule
)
COSTARTS_SUBSET_SAMPLING_MODE = ROUTER_EXPERIMENT_CONFIG.costarts_subset_sampling_mode
COSTARTS_SUBSET_UTILITY_ACTION_LOSS_WEIGHT = (
    ROUTER_EXPERIMENT_CONFIG.costarts_subset_utility_action_loss_weight
)
COSTARTS_SUBSET_UTILITY_LOSS_WEIGHT = (
    ROUTER_EXPERIMENT_CONFIG.costarts_subset_utility_loss_weight
)
COSTARTS_SUBSET_UTILITY_PAIRWISE_LOSS_WEIGHT = (
    ROUTER_EXPERIMENT_CONFIG.costarts_subset_utility_pairwise_loss_weight
)
COSTARTS_SUBSET_UTILITY_MIX_LOSS_WEIGHT = (
    ROUTER_EXPERIMENT_CONFIG.costarts_subset_utility_mix_loss_weight
)
COSTARTS_SUBSET_UTILITY_LEARNING_RATE = (
    ROUTER_EXPERIMENT_CONFIG.costarts_subset_utility_learning_rate
)
COSTARTS_SUBSET_UTILITY_WEIGHT_DECAY = (
    ROUTER_EXPERIMENT_CONFIG.costarts_subset_utility_weight_decay
)
COSTARTS_SUBSET_UTILITY_GRAD_CLIP_NORM = (
    ROUTER_EXPERIMENT_CONFIG.costarts_subset_utility_grad_clip_norm
)
COSTARTS_SUBSET_UTILITY_BATCH_SIZE = (
    ROUTER_EXPERIMENT_CONFIG.costarts_subset_utility_batch_size
)
COSTARTS_SUBSET_UTILITY_MAX_EPOCHS = (
    ROUTER_EXPERIMENT_CONFIG.costarts_subset_utility_max_epochs
)
COSTARTS_SUBSET_UTILITY_PATIENCE = (
    ROUTER_EXPERIMENT_CONFIG.costarts_subset_utility_patience
)
SEED = ROUTER_EXPERIMENT_CONFIG.random_seed
RANDOM_SEED = ROUTER_EXPERIMENT_CONFIG.random_seed
DATASET_NAME = ROUTER_EXPERIMENT_CONFIG.dataset_name
DATA_DIR = ROUTER_EXPERIMENT_CONFIG.data_dir
INPUT_LEN = ROUTER_EXPERIMENT_CONFIG.input_length
INPUT_LENGTH = ROUTER_EXPERIMENT_CONFIG.input_length
OUTPUT_LEN = ROUTER_EXPERIMENT_CONFIG.forecast_horizon
FORECAST_HORIZON = ROUTER_EXPERIMENT_CONFIG.forecast_horizon
NUM_FEATURES = ROUTER_EXPERIMENT_CONFIG.num_features
SELECTED_VARIABLES = ROUTER_EXPERIMENT_CONFIG.selected_variables
CHECKPOINT_DIR = ROUTER_EXPERIMENT_CONFIG.checkpoint_dir
EXPERT_CHECKPOINT_PATHS = dict(ROUTER_EXPERIMENT_CONFIG.expert_checkpoint_paths)
CACHE_PATHS = dict(ROUTER_EXPERIMENT_CONFIG.cache_paths)
DEBUG_MODE = ROUTER_EXPERIMENT_CONFIG.debug_mode
