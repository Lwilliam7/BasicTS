"""Train a COSTARTS-style frozen-expert router without cosine similarity."""
# imports

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
#root path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.chronological_expert_training import (
    DEFAULT_INPUT_LEN,
    DEFAULT_NUM_FEATURES,
    DEFAULT_OUTPUT_LEN,
    _call_expert_model,
    _check_shapes,
    _load_torch_checkpoint,
    _prepare_forecasting_batch,
    _assert_full_data_contract,
    assert_experts_frozen,
    build_selected_candidate_experts,
    load_full_chronological_data,
    prepare_chronological_dataloaders,
)
from scripts.old.router_experiment_config import (
    RouterExperimentConfig,
    load_router_experiment_config,
    print_router_experiment_config,
    validate_router_experiment_config,
)


@dataclass
class COSTARTSTrainingConfig:
    data_dir: str = "datasets/ETTh1"
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "checkpoints/costarts_no_cosine"
    results_dir: str = "results/router_summary/costarts_no_cosine"
    batch_size: int = 512
    cache_batch_size: int = 512
    max_epochs: int = 50
    patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    device: str = "cpu"
    seed: int = 7
    error_temperature: float = 0.1
    map_loss_weight: float = 1.0
    ranking_loss_weight: float = 1.0
    query_loss_weight: float = 1.0
    stop_loss_weight: float = 1.0
    mix_forecast_loss_weight: float = 0.0
    teacher_forcing: bool = True
    sampled_rollout: bool = False
    force_rebuild_cache: bool = False
    debug: bool = False


class COSTARTSRouter(nn.Module):
    """Cost-aware sequential expert selector with no cosine expert matching."""
    #sequential costrat
    def __init__(
        self,
        num_experts: int,
        input_len: int = DEFAULT_INPUT_LEN,
        forecast_horizon: int = DEFAULT_OUTPUT_LEN,
        num_features: int = DEFAULT_NUM_FEATURES,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        queried_experts_cap_k: Optional[int] = None,
        routing_temperature: float = 1.0,
        uses_cosine_similarity: bool = False,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if input_len <= 0 or forecast_horizon <= 0 or num_features <= 0:
            raise ValueError("input_len, forecast_horizon, and num_features must be positive")
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embedding_dim and hidden_dim must be positive")
        if routing_temperature <= 0:
            raise ValueError("routing_temperature must be positive")
        if uses_cosine_similarity:
            raise ValueError("train_costarts_router_no_cosine.py does not support cosine similarity")

        self.num_experts = int(num_experts)
        self.input_len = int(input_len)
        self.forecast_horizon = int(forecast_horizon)
        self.num_features = int(num_features)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.queried_experts_cap_k = (
            self.num_experts
            if queried_experts_cap_k is None
            else min(int(queried_experts_cap_k), self.num_experts)
        )
        #set query limit
        if self.queried_experts_cap_k <= 0:
            raise ValueError("queried_experts_cap_k must be positive")
        self.routing_temperature = float(routing_temperature)

 
# Conv1d → detect time patterns
# GELU → nonlinear transformation
# GroupNorm → stabilize values
# Conv1d with dilation → detect wider time patterns
# AdaptiveAvgPool1d(1) → compress all 96 steps into one vector
        self.history_encoder = nn.Sequential(
            nn.Conv1d(num_features, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.GroupNorm(1, hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
            nn.GroupNorm(1, hidden_dim),
            nn.AdaptiveAvgPool1d(1),
        )
        self.query_projection = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        #create a trainable vector for each expert
        self.expert_embeddings = nn.Parameter(torch.empty(num_experts, embedding_dim))
        nn.init.normal_(self.expert_embeddings, mean=0.0, std=0.02)
# head for differnt thinks
# Each head is a small Linear layer that turns the shared 64-number embedding into scores.
        self.map_head = nn.Linear(embedding_dim, num_experts)
        self.ranking_head = nn.Linear(embedding_dim, num_experts)
        self.query_head = nn.Linear(embedding_dim, num_experts)
        self.mix_head = nn.Linear(embedding_dim, num_experts)
        self.stop_head = nn.Linear(embedding_dim, self.queried_experts_cap_k)
# turns history isnt a embedding
    def encode(self, history: torch.Tensor) -> torch.Tensor:
        assert history.ndim == 3
        assert history.shape[1:] == (self.input_len, self.num_features)
        encoded = self.history_encoder(history.transpose(1, 2)).squeeze(-1)
        query_embedding = self.query_projection(encoded)
        return F.normalize(query_embedding, p=2, dim=-1)
# This function takes the past time series and decides:

# which experts look best,
# the order to use them,
# how many experts to use,
# possible weights for combining them
    def forward(
        self,
        history: torch.Tensor,
        teacher_forced_order: Optional[torch.Tensor] = None,
        sampled_rollout: bool = False,
        debug: bool = False,
    ) -> dict:
        # compare history scores using costine simlitary
        # no-cosine version: keep this original comment, but do not add cosine scores to the heads
        batch_size = history.shape[0]
        query_embedding = self.encode(history)
        # original cosine code:
        # expert_vectors = F.normalize(self.expert_embeddings, p=2, dim=-1)
        # cosine_scores = torch.matmul(query_embedding, expert_vectors.T)
        # similarity_logits = cosine_scores / self.routing_temperature
        similarity_logits = torch.zeros(
            batch_size,
            self.num_experts,
            dtype=query_embedding.dtype,
            device=query_embedding.device,
        )
        #produce predictiosn for each expert  
        map_prediction = F.softplus(self.map_head(query_embedding))
        ranking_logits = self.ranking_head(query_embedding)
        query_logits = self.query_head(query_embedding)
        mix_weights = torch.softmax(self.mix_head(query_embedding), dim=-1)
        #scores weather to stop after 1,2,3
        stop_logits = self.stop_head(query_embedding)
        #scores expert order

        if teacher_forced_order is not None:
            query_order = teacher_forced_order[:, : self.queried_experts_cap_k]
        else:
            if sampled_rollout and self.training:
                probabilities = torch.softmax(query_logits, dim=-1)
                query_order = torch.multinomial(
                    probabilities,
                    num_samples=self.queried_experts_cap_k,
                    replacement=False,
                )
            else:
                query_order = torch.topk(
                    query_logits,
                    k=self.queried_experts_cap_k,
                    dim=-1,
                ).indices
        #decides when to stop  
        #check shapes
        stop_step = torch.argmax(stop_logits, dim=-1) + 1
        assert query_embedding.shape == (batch_size, self.embedding_dim)
        assert map_prediction.shape == (batch_size, self.num_experts)
        assert ranking_logits.shape == (batch_size, self.num_experts)
        assert query_logits.shape == (batch_size, self.num_experts)
        assert stop_logits.shape == (batch_size, self.queried_experts_cap_k)
        assert query_order.shape == (batch_size, self.queried_experts_cap_k)

        if debug:
            print("COSTARTS history:", tuple(history.shape))
            print("COSTARTS query_embedding:", tuple(query_embedding.shape))
            print("COSTARTS map_prediction:", tuple(map_prediction.shape))
            print("COSTARTS query_order:", tuple(query_order.shape))
            print("COSTARTS stop_logits:", tuple(stop_logits.shape))
            print("COSTARTS first selected experts:", query_order[:3].detach().cpu().tolist())
        #RETURN OUTPUTS
        return {
            "query_embedding": query_embedding,
            "similarity_logits": similarity_logits,
            "map_prediction": map_prediction,
            "ranking_logits": ranking_logits,
            "query_logits": query_logits,
            "mix_weights": mix_weights,
            "stop_logits": stop_logits,
            "query_order": query_order,
            "stop_step": stop_step,
        }

    def config_dict(self) -> dict:
        return {
            "num_experts": self.num_experts,
            "input_len": self.input_len,
            "forecast_horizon": self.forecast_horizon,
            "num_features": self.num_features,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "queried_experts_cap_k": self.queried_experts_cap_k,
            "routing_temperature": self.routing_temperature,
            "uses_cosine_similarity": False,
        }


class COSTARTSCacheDataset(Dataset):
    #saved cache
    def __init__(self, cache: Mapping[str, torch.Tensor]) -> None:
        self.cache = cache
        self.num_windows = int(cache["num_windows"])
    #reutn data set
    def __len__(self) -> int:
        return self.num_windows
    #return one window
    def __getitem__(self, index: int) -> dict:
        return {
            "history": self.cache["histories"][index],
            "target": self.cache["targets"][index],
            "target_mask": self.cache["target_masks"][index],
            "prediction_stack": self.cache["prediction_stack"][index],
            "error_matrix": self.cache["error_matrix"][index],
            "mse_matrix": self.cache["mse_matrix"][index],
            "target_probabilities": self.cache["target_probabilities"][index],
            "best_expert": self.cache["best_expert"][index],
            "sample_index": self.cache["sample_indices"][index],
        }
#reproducable

def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#make sure that the optimizer is not training the exper parameter
def assert_optimizer_excludes_experts(
    optimizer: torch.optim.Optimizer,
    experts: Sequence[nn.Module],
) -> None:
    expert_param_ids = {id(parameter) for expert in experts for parameter in expert.parameters()}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) in expert_param_ids:
                raise AssertionError("COSTARTS optimizer contains frozen expert parameters")

# checks that the forecasting experts are truly frozen
def assert_no_expert_gradients(experts: Sequence[nn.Module]) -> None:
    for expert in experts:
        if expert.training:
            raise AssertionError("Frozen expert left eval mode")
        for parameter in expert.parameters():
            if parameter.requires_grad:
                raise AssertionError("Frozen expert parameter has requires_grad=True")
            if parameter.grad is not None:
                raise AssertionError("Frozen expert received a gradient")

# This calculates the MAE and MSE of every expert for every sample.
def _sample_expert_errors(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = target_mask.to(predictions.dtype).unsqueeze(-1)
    denominator = mask.sum(dim=(1, 2)).clamp_min(1.0)
    abs_error = torch.abs(predictions - targets.unsqueeze(-1)) * mask
    squared_error = (predictions - targets.unsqueeze(-1)).pow(2) * mask
    mae = abs_error.sum(dim=(1, 2)) / denominator
    mse = squared_error.sum(dim=(1, 2)) / denominator
    return mae, mse

# Validates that a cached COSTAR-TS dataset has the correct split,
# experts, configuration, tensor shapes, and sample ordering
def validate_costarts_cache(
    cache: Mapping[str, torch.Tensor],
    split_role: str,
    expert_names: Sequence[str],
    input_len: int,
    forecast_horizon: int,
    num_features: int,
    error_temperature: Optional[float] = None,
) -> None:
    if cache.get("split_role") != split_role:
        raise ValueError(f"Cache split_role {cache.get('split_role')} does not match {split_role}")
    if tuple(cache.get("expert_names", ())) != tuple(expert_names):
        raise ValueError("Cache expert_names do not match the loaded experts")
    if error_temperature is not None:
        cached_temperature = float(cache.get("error_temperature", error_temperature))
        if abs(cached_temperature - float(error_temperature)) > 1e-12:
            raise ValueError(
                "Cache error_temperature does not match the requested value: "
                f"{cached_temperature} != {error_temperature}"
            )
    num_windows = int(cache["num_windows"])
    num_experts = len(expert_names)
    assert tuple(cache["histories"].shape) == (num_windows, input_len, num_features)
    assert tuple(cache["targets"].shape) == (num_windows, forecast_horizon, num_features)
    assert tuple(cache["target_masks"].shape) == (num_windows, forecast_horizon, num_features)
    assert tuple(cache["prediction_stack"].shape) == (
        num_windows,
        forecast_horizon,
        num_features,
        num_experts,
    )
    assert tuple(cache["error_matrix"].shape) == (num_windows, num_experts)
    assert tuple(cache["mse_matrix"].shape) == (num_windows, num_experts)
    assert tuple(cache["target_probabilities"].shape) == (num_windows, num_experts)
    assert tuple(cache["best_expert"].shape) == (num_windows,)
    assert tuple(cache["sample_indices"].shape) == (num_windows,)
    expected_indices = torch.arange(num_windows, dtype=cache["sample_indices"].dtype)
    if not torch.equal(cache["sample_indices"].cpu(), expected_indices):
        raise AssertionError("Cache sample_indices are not contiguous and aligned")


def build_costarts_expert_cache(
    loader: DataLoader,
    experts: Sequence[nn.Module],
    expert_names: Sequence[str],
    scaler,
    device: Union[str, torch.device],
    split_role: str,
    cache_path: Optional[Union[str, Path]] = None,
    error_temperature: float = 0.1,
    force_rebuild: bool = False,
    debug: bool = False,
) -> dict:
    #only allow router_train and router_val
    if split_role not in {"router_train", "router_val"}:
        raise ValueError("COSTARTS training cache must use router_train or router_val")
    if getattr(loader.dataset, "split_role", None) != split_role:
        raise ValueError(f"Loader split_role must be {split_role}")
    if error_temperature <= 0:
        raise ValueError("error_temperature must be positive")

    device = torch.device(device)
    cache_path = Path(cache_path) if cache_path is not None else None
    #reuse cache
    if cache_path is not None and cache_path.exists() and not force_rebuild:
        cached = _load_torch_checkpoint(cache_path, torch.device("cpu"))
        validate_costarts_cache(
            cached,
            split_role,
            expert_names,
            DEFAULT_INPUT_LEN,
            DEFAULT_OUTPUT_LEN,
            DEFAULT_NUM_FEATURES,
            error_temperature,
        )
        return cached
    #free and prepare experts
    assert_experts_frozen(*experts)
    for expert in experts:
        expert.to(device)
        expert.eval()
    #storage
    histories = []
    targets_list = []
    target_masks = []
    prediction_stacks = []
    error_matrices = []
    mse_matrices = []
    sample_indices = []
    cursor = 0
    #disable gradients

    with torch.no_grad():
        #Process each batch
        for batch in loader:
            inputs, targets, target_mask = _prepare_forecasting_batch(batch, device, scaler)
            assert inputs.shape[1:] == (DEFAULT_INPUT_LEN, DEFAULT_NUM_FEATURES)
            assert targets.shape[1:] == (DEFAULT_OUTPUT_LEN, DEFAULT_NUM_FEATURES)

            predictions = []
            #Run every expert
            for expert_name, expert in zip(expert_names, experts):
                prediction = _call_expert_model(expert, inputs).detach()
                _check_shapes(prediction, targets, f"COSTARTS cache {expert_name}")
                predictions.append(prediction)
            prediction_stack = torch.stack(predictions, dim=-1)
            assert prediction_stack.shape == (
                inputs.shape[0],
                DEFAULT_OUTPUT_LEN,
                DEFAULT_NUM_FEATURES,
                len(experts),
            )
            #Calculate every expert’s error
            mae, mse = _sample_expert_errors(prediction_stack, targets, target_mask)
            #Move results to CPU storage
            batch_size = inputs.shape[0]
            histories.append(inputs.cpu())
            targets_list.append(targets.cpu())
            target_masks.append(target_mask.cpu())
            prediction_stacks.append(prediction_stack.cpu())
            error_matrices.append(mae.cpu())
            mse_matrices.append(mse.cpu())
            sample_indices.append(torch.arange(cursor, cursor + batch_size, dtype=torch.long))
            cursor += batch_size

            if debug and len(histories) == 1:
                print("COSTARTS cache first batch")
                print("  inputs:", tuple(inputs.shape))
                print("  targets:", tuple(targets.shape))
                print("  prediction_stack:", tuple(prediction_stack.shape))
                print("  error_matrix:", tuple(mae.shape))
    #create soft targets to tell the router how good each expert is
    error_matrix = torch.cat(error_matrices, dim=0)
    target_probabilities = torch.softmax(-error_matrix / error_temperature, dim=-1)
    best_expert = torch.argmin(error_matrix, dim=-1)
#save cache
    cache = {
        "split_role": split_role,
        "expert_names": tuple(expert_names),
        "num_windows": int(error_matrix.shape[0]),
        "input_len": DEFAULT_INPUT_LEN,
        "forecast_horizon": DEFAULT_OUTPUT_LEN,
        "num_features": DEFAULT_NUM_FEATURES,
        "error_temperature": float(error_temperature),
        "histories": torch.cat(histories, dim=0).to(torch.float32),
        "targets": torch.cat(targets_list, dim=0).to(torch.float32),
        "target_masks": torch.cat(target_masks, dim=0).to(torch.bool),
        "prediction_stack": torch.cat(prediction_stacks, dim=0).to(torch.float32),
        "error_matrix": error_matrix.to(torch.float32),
        "mse_matrix": torch.cat(mse_matrices, dim=0).to(torch.float32),
        "target_probabilities": target_probabilities.to(torch.float32),
        "best_expert": best_expert.to(torch.long),
        "sample_indices": torch.cat(sample_indices, dim=0).to(torch.long),
    }
    #valide new cache
    validate_costarts_cache(
        cache,
        split_role,
        expert_names,
        DEFAULT_INPUT_LEN,
        DEFAULT_OUTPUT_LEN,
        DEFAULT_NUM_FEATURES,
        error_temperature,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        print(f"Saved COSTARTS {split_role} cache: {cache_path}")
    return cache

# This function returns the oracle ordering of experts for each sample based on their error.
def _oracle_order(error_matrix: torch.Tensor) -> torch.Tensor:
    return torch.argsort(error_matrix, dim=-1)

# This function returns a tensor of costs for each expert based on their names and the provided cost weights.
def _cost_tensor(
    expert_names: Sequence[str],
    cost_weights: Mapping[str, float],
    device: torch.device,
) -> torch.Tensor:
    values = [float(cost_weights.get(name, cost_weights.get(f"Candidate_{name}", 0.0))) for name in expert_names]
    return torch.tensor(values, dtype=torch.float32, device=device)
# This function computes the target stop index for each sample based on the error matrix, oracle order, expert costs, and a stop threshold.

def _target_stop_index(
    error_matrix: torch.Tensor,
    oracle_order: torch.Tensor,
    expert_costs: torch.Tensor,
    k: int,
    stop_threshold: float,
) -> torch.Tensor:
    ordered_errors = error_matrix.gather(1, oracle_order[:, :k])
    ordered_costs = expert_costs[oracle_order[:, :k]]
    cumulative_costs = torch.cumsum(ordered_costs, dim=-1)
    utility = ordered_errors + cumulative_costs
    best_utility = utility.min(dim=-1).values
    acceptable = utility <= (best_utility.unsqueeze(-1) + float(stop_threshold))
    first = acceptable.to(torch.int64).argmax(dim=-1)
    return first.clamp(0, k - 1)

# error_matrix contains each expert’s true MAE.
# best_expert is the lowest-error expert.
# oracle_order sorts experts from lowest to highest error.
def costarts_losses(
    router_outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    expert_names: Sequence[str],
    loss_weights: Mapping[str, float],
    cost_weights: Mapping[str, float],
    stop_threshold: float,
    teacher_forcing: bool,
    mix_forecast_loss_weight: float,
) -> Tuple[torch.Tensor, dict]:
    error_matrix = batch["error_matrix"]
    best_expert = batch["best_expert"]
    oracle_order = _oracle_order(error_matrix)
    k = router_outputs["stop_logits"].shape[-1]
    expert_costs = _cost_tensor(expert_names, cost_weights, error_matrix.device)
    stop_target = _target_stop_index(
        error_matrix,
        oracle_order,
        expert_costs,
        k,
        stop_threshold,
    )
    #normalize 
    scale = error_matrix.detach().mean(dim=-1, keepdim=True).clamp_min(1e-6)
    #create a loss for each of the outputs of the router
    map_loss = F.mse_loss(router_outputs["map_prediction"] / scale, error_matrix / scale)
    ranking_loss = F.cross_entropy(router_outputs["ranking_logits"], best_expert)
    query_loss = F.cross_entropy(router_outputs["query_logits"], best_expert)
    stop_loss = F.cross_entropy(router_outputs["stop_logits"], stop_target)
    mix_forecast_loss = torch.zeros((), device=error_matrix.device)
    #trains the router forecat mixing weights
    if mix_forecast_loss_weight > 0:
        prediction_stack = batch["prediction_stack"]
        targets = batch["target"]
        target_mask = batch["target_mask"].to(prediction_stack.dtype)
        mixed = torch.sum(
            prediction_stack * router_outputs["mix_weights"].view(-1, 1, 1, len(expert_names)),
            dim=-1,
        )
        assert mixed.shape == targets.shape
        denominator = target_mask.sum().clamp_min(1.0)
        mix_forecast_loss = (torch.abs(mixed - targets) * target_mask).sum() / denominator
    #total lost
    total = (
        float(loss_weights.get("map_regression", 1.0)) * map_loss
        + float(loss_weights.get("ranking", 1.0)) * ranking_loss
        + float(loss_weights.get("query", 1.0)) * query_loss
        + float(loss_weights.get("stop", 1.0)) * stop_loss
        + float(mix_forecast_loss_weight) * mix_forecast_loss
    )
    return total, {
        "map_regression_loss": float(map_loss.detach().cpu()),
        "ranking_loss": float(ranking_loss.detach().cpu()),
        "query_loss": float(query_loss.detach().cpu()),
        "stop_loss": float(stop_loss.detach().cpu()),
        "mix_forecast_loss": float(mix_forecast_loss.detach().cpu()),
        "teacher_forcing": bool(teacher_forcing),
    }


def _select_expert_from_outputs(
    #Read router outputs
    router_outputs: Mapping[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    query_order = router_outputs["query_order"]
    predicted_errors = router_outputs["map_prediction"]
    #Limit the stopping step
    stop_step = router_outputs["stop_step"].clamp(1, query_order.shape[1])
    selected = []
    #Process each sample and select the expert with the lowest predicted error among the queried experts
    for row_index in range(query_order.shape[0]):
        candidates = query_order[row_index, : int(stop_step[row_index].item())]
        candidate_errors = predicted_errors[row_index].gather(0, candidates)
        selected.append(candidates[torch.argmin(candidate_errors)])
    return torch.stack(selected, dim=0), stop_step

#disables gradient calculation for evaluation
@torch.no_grad()
#this function evaluates the COSTARTS router on a given cache dataset and computes various metrics such as MAE, MSE, oracle MAE, regret to oracle, average experts selected, average stop step, and routing entropy.
def evaluate_costarts_router(
    router: COSTARTSRouter,
    cache: Mapping[str, torch.Tensor],
    expert_names: Sequence[str],
    batch_size: int,
    device: Union[str, torch.device],
    debug: bool = False,
) -> dict:
    device = torch.device(device)
    router.eval()
    dataset = COSTARTSCacheDataset(cache)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_mae = 0.0
    total_mse = 0.0
    total_oracle = 0.0
    total_count = 0
    total_selected = 0.0
    total_stop = 0.0
    total_entropy = 0.0

    for batch_index, batch in enumerate(loader):
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        outputs = router(batch["history"], sampled_rollout=False, debug=debug and batch_index == 0)
        selected_expert, stop_step = _select_expert_from_outputs(outputs)
        selected_mae = batch["error_matrix"].gather(1, selected_expert.unsqueeze(-1)).squeeze(-1)
        selected_mse = batch["mse_matrix"].gather(1, selected_expert.unsqueeze(-1)).squeeze(-1)
        oracle_mae = batch["error_matrix"].min(dim=-1).values
        probabilities = torch.softmax(outputs["query_logits"], dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)

        count = batch["history"].shape[0]
        total_mae += float(selected_mae.sum().cpu())
        total_mse += float(selected_mse.sum().cpu())
        total_oracle += float(oracle_mae.sum().cpu())
        total_selected += float(stop_step.sum().cpu())
        total_stop += float(stop_step.sum().cpu())
        total_entropy += float(entropy.sum().cpu())
        total_count += count

    mae = total_mae / total_count
    oracle = total_oracle / total_count
    return {
        "validation_mae": mae,
        "validation_mse": total_mse / total_count,
        "validation_oracle_mae": oracle,
        "validation_regret_to_oracle": mae - oracle,
        "average_experts_selected": total_selected / total_count,
        "average_stop_step": total_stop / total_count,
        "routing_entropy": total_entropy / total_count,
    }
#It saves training or validation results into a CSV file.

def _write_curves(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

#saves a checkpoint 
def _save_checkpoint(
    path: Path,
    router: COSTARTSRouter,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Mapping[str, float],
    training_config: COSTARTSTrainingConfig,
    router_config: RouterExperimentConfig,
    expert_names: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "router_state_dict": router.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": dict(metrics),
            "training_config": asdict(training_config),
            "router_config": router.config_dict(),
            "experiment_config": asdict(router_config),
            "expert_names": tuple(expert_names),
            "selection_metric": "validation_mae",
        },
        path,
    )

# #Loads the data and frozen expert models.
# Builds or loads their cached forecasts.
# Creates the router and optimizer.
# Trains the router batch by batch.
# Evaluates validation MAE after every epoch.
# Saves the best router checkpoint.
# Stops early when validation stops improving.
# Saves training curves and a summary file.
def train_costarts_router(
    training_config: COSTARTSTrainingConfig,
) -> dict:
    set_reproducible_seed(training_config.seed)
    device = torch.device(training_config.device)
    from basicts.scaler import ZScoreScaler

    experiment_config = replace(
        load_router_experiment_config(),
        data_dir=training_config.data_dir,
        checkpoint_dir=training_config.checkpoint_dir,
        random_seed=training_config.seed,
        router_type="costarts",
    )
    experiment_config = validate_router_experiment_config(
        experiment_config,
        require_checkpoints=True,
        require_data=True,
        require_cache_parent=True,
    )
    print_router_experiment_config(experiment_config)

    full_data = load_full_chronological_data(training_config.data_dir)
    _assert_full_data_contract(full_data, DEFAULT_NUM_FEATURES)
    loaders, scaler = prepare_chronological_dataloaders(
        full_data=full_data,
        scaler=ZScoreScaler(norm_each_channel=True, rescale=False),
        batch_size=training_config.cache_batch_size,
        input_len=DEFAULT_INPUT_LEN,
        output_len=DEFAULT_OUTPUT_LEN,
    )
    #load the frozen forecasting experts
    experts, expert_names, _, expert_checkpoint_paths = build_selected_candidate_experts(
        checkpoint_dir=training_config.checkpoint_dir,
        device=device,
        scaler=scaler,
    )
    assert_experts_frozen(*experts)
    for expert in experts:
        expert.eval()

    cache_paths = dict(experiment_config.cache_paths)
    train_cache = build_costarts_expert_cache(
        loaders["router_train"],
        experts,
        expert_names,
        scaler,
        device,
        "router_train",
        cache_path=cache_paths.get("costarts_train", "cache/costarts_router_train_cache.pt"),
        error_temperature=training_config.error_temperature,
        force_rebuild=training_config.force_rebuild_cache,
        debug=training_config.debug,
    )
    val_cache = build_costarts_expert_cache(
        loaders["router_val"],
        experts,
        expert_names,
        scaler,
        device,
        "router_val",
        cache_path=cache_paths.get("costarts_val", "cache/costarts_router_val_cache.pt"),
        error_temperature=training_config.error_temperature,
        force_rebuild=training_config.force_rebuild_cache,
        debug=training_config.debug,
    )

    validate_costarts_cache(
        train_cache,
        "router_train",
        expert_names,
        DEFAULT_INPUT_LEN,
        DEFAULT_OUTPUT_LEN,
        DEFAULT_NUM_FEATURES,
        training_config.error_temperature,
    )
    validate_costarts_cache(
        val_cache,
        "router_val",
        expert_names,
        DEFAULT_INPUT_LEN,
        DEFAULT_OUTPUT_LEN,
        DEFAULT_NUM_FEATURES,
        training_config.error_temperature,
    )

    router = COSTARTSRouter(
        num_experts=len(expert_names),
        input_len=DEFAULT_INPUT_LEN,
        forecast_horizon=DEFAULT_OUTPUT_LEN,
        num_features=DEFAULT_NUM_FEATURES,
        embedding_dim=experiment_config.embedding_dim,
        hidden_dim=experiment_config.hidden_dim,
        queried_experts_cap_k=experiment_config.queried_experts_cap_k,
        routing_temperature=experiment_config.routing_temperature,
    ).to(device)

    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    #make sure that the optimizer is not training the frozen experts
    assert_optimizer_excludes_experts(optimizer, experts)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, training_config.patience // 3),
    )

    train_dataset = COSTARTSCacheDataset(train_cache)
    generator = torch.Generator()
    generator.manual_seed(training_config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=generator,
    )

    output_dir = Path(training_config.output_dir)
    results_dir = Path(training_config.results_dir)
    best_path = output_dir / "best_costarts_router.pt"
    last_path = output_dir / "last_costarts_router.pt"
    curves_path = results_dir / "costarts_training_curves.csv"
    summary_path = results_dir / "costarts_training_summary.json"

    best_val_mae = math.inf
    best_epoch = 0
    bad_epochs = 0
    curves = []

    loss_weights = {
        "map_regression": training_config.map_loss_weight,
        "ranking": training_config.ranking_loss_weight,
        "query": training_config.query_loss_weight,
        "stop": training_config.stop_loss_weight,
    }
    #train the router for a number of epochs, evaluating on the validation set after each epoch, and saving the best model based on validation MAE.             
    for epoch in range(1, training_config.max_epochs + 1):
        router.train()
        epoch_totals = {
            "total_loss": 0.0,
            "map_regression_loss": 0.0,
            "ranking_loss": 0.0,
            "query_loss": 0.0,
            "stop_loss": 0.0,
            "mix_forecast_loss": 0.0,
        }
        sample_count = 0

        for batch_index, batch in enumerate(train_loader):
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            oracle_order = _oracle_order(batch["error_matrix"])
            teacher_order = oracle_order if training_config.teacher_forcing else None
            optimizer.zero_grad(set_to_none=True)
            outputs = router(
                batch["history"],
                teacher_forced_order=teacher_order,
                sampled_rollout=training_config.sampled_rollout,
                debug=training_config.debug and epoch == 1 and batch_index == 0,
            )
            total_loss, parts = costarts_losses(
                outputs,
                batch,
                expert_names,
                loss_weights,
                experiment_config.cost_weights,
                experiment_config.stop_threshold,
                training_config.teacher_forcing,
                training_config.mix_forecast_loss_weight,
            )
            total_loss.backward()
            if training_config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(router.parameters(), training_config.grad_clip_norm)
            assert_no_expert_gradients(experts)
            optimizer.step()

            batch_size = batch["history"].shape[0]
            epoch_totals["total_loss"] += float(total_loss.detach().cpu()) * batch_size
            for key in ("map_regression_loss", "ranking_loss", "query_loss", "stop_loss", "mix_forecast_loss"):
                epoch_totals[key] += parts[key] * batch_size
            sample_count += batch_size

        train_metrics = {key: value / sample_count for key, value in epoch_totals.items()}
        val_metrics = evaluate_costarts_router(
            router,
            val_cache,
            expert_names,
            batch_size=training_config.batch_size,
            device=device,
            debug=training_config.debug and epoch == 1,
        )
        scheduler.step(val_metrics["validation_mae"])

        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **train_metrics,
            **val_metrics,
        }
        curves.append(row)
        improved = val_metrics["validation_mae"] < best_val_mae
        if improved:
            best_val_mae = val_metrics["validation_mae"]
            best_epoch = epoch
            bad_epochs = 0
            _save_checkpoint(
                best_path,
                router,
                optimizer,
                scheduler,
                epoch,
                val_metrics,
                training_config,
                experiment_config,
                expert_names,
            )
        else:
            bad_epochs += 1

        _save_checkpoint(
            last_path,
            router,
            optimizer,
            scheduler,
            epoch,
            val_metrics,
            training_config,
            experiment_config,
            expert_names,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_metrics['total_loss']:.6f} "
            f"map={train_metrics['map_regression_loss']:.6f} "
            f"rank={train_metrics['ranking_loss']:.6f} "
            f"query={train_metrics['query_loss']:.6f} "
            f"stop={train_metrics['stop_loss']:.6f} "
            f"mix={train_metrics['mix_forecast_loss']:.6f} | "
            f"val_mae={val_metrics['validation_mae']:.6f} "
            f"regret={val_metrics['validation_regret_to_oracle']:.6f} "
            f"avg_selected={val_metrics['average_experts_selected']:.3f} "
            f"avg_stop={val_metrics['average_stop_step']:.3f} "
            f"entropy={val_metrics['routing_entropy']:.3f} "
            f"saved={improved}"
        )

        if bad_epochs >= training_config.patience:
            print(f"COSTARTS early stopping after epoch {epoch}")
            break

    _write_curves(curves_path, curves)
    summary = {
        "best_epoch": best_epoch,
        "best_validation_mae": best_val_mae,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "training_curves_csv": str(curves_path),
        "training_config": asdict(training_config),
        "router_config": router.config_dict(),
        "expert_names": list(expert_names),
        "expert_checkpoint_paths": {name: str(path) for name, path in expert_checkpoint_paths.items()},
        "model_selection": "validation_mae on router_val only",
        "test_set_used": False,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"Saved best checkpoint: {best_path}")
    print(f"Saved last checkpoint: {last_path}")
    print(f"Saved curves CSV: {curves_path}")
    print(f"Saved summary JSON: {summary_path}")
    return summary

#commanline line setting to start the traing script
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train COSTARTSRouter without cosine similarity on router_train and select on router_val."
    )
    parser.add_argument("--data-dir", default="datasets/ETTh1")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="checkpoints/costarts_no_cosine")
    parser.add_argument("--results-dir", default="results/router_summary/costarts_no_cosine")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cache-batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--error-temperature", type=float, default=0.1)
    parser.add_argument("--map-loss-weight", type=float, default=1.0)
    parser.add_argument("--ranking-loss-weight", type=float, default=1.0)
    parser.add_argument("--query-loss-weight", type=float, default=1.0)
    parser.add_argument("--stop-loss-weight", type=float, default=1.0)
    parser.add_argument("--mix-forecast-loss-weight", type=float, default=0.0)
    parser.add_argument("--sampled-rollout", action="store_true")
    parser.add_argument("--no-teacher-forcing", action="store_true")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/imports without training.")
    return parser.parse_args()

# main funciton to run the training script with the provided command line arguments. It handles dry run validation and initiates the training process.
def main() -> None:
    args = parse_args()
    if args.dry_run:
        config = replace(
            load_router_experiment_config(),
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            random_seed=args.seed,
            router_type="costarts",
        )
        config = validate_router_experiment_config(
            config,
            require_checkpoints=True,
            require_data=True,
            require_cache_parent=True,
        )
        print_router_experiment_config(config)
        print("COSTARTS dry run passed.")
        return

    train_costarts_router(
        COSTARTSTrainingConfig(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            results_dir=args.results_dir,
            batch_size=args.batch_size,
            cache_batch_size=args.cache_batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            device=args.device,
            seed=args.seed,
            error_temperature=args.error_temperature,
            map_loss_weight=args.map_loss_weight,
            ranking_loss_weight=args.ranking_loss_weight,
            query_loss_weight=args.query_loss_weight,
            stop_loss_weight=args.stop_loss_weight,
            mix_forecast_loss_weight=args.mix_forecast_loss_weight,
            teacher_forcing=not args.no_teacher_forcing,
            sampled_rollout=args.sampled_rollout,
            force_rebuild_cache=args.force_rebuild_cache,
            debug=args.debug,
        )
    )


if __name__ == "__main__":
    main()
