"""Full end-to-end Sequential COSTAR-TS with jointly trained experts.

This module is intentionally separate from the frozen-expert
``SequentialCOSTARTSRouter`` baseline. During training it computes all expert
forecasts and uses differentiable sequential routing so the forecast loss can
update both COSTAR parameters and expert parameters. During hard inference it
queries experts one at a time and only executes selected experts.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from basicts.models.DLinear import DLinear, DLinearConfig
from basicts.models.PatchTST import PatchTSTConfig, PatchTSTForForecasting
from basicts.models.TimesNet import TimesNetConfig, TimesNetForForecasting
from basicts.models.iTransformer import iTransformerConfig, iTransformerForForecasting


EXPERT_ORDER = ("DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN")


@dataclass(frozen=True)
class EndToEndCOSTARTSConfig:
    input_len: int = 96
    forecast_horizon: int = 12
    num_features: int = 7
    expert_hidden_size: int = 64
    router_embedding_dim: int = 64
    router_hidden_dim: int = 64
    max_queries: int = 5
    route_temperature: float = 1.0
    gumbel_routing: bool = False
    straight_through: bool = False
    expert_dropout: float = 0.1
    finalizer_scale: float = 0.1


def _load_sourceless_modern_tcn() -> tuple[type[Any], type[nn.Module]]:
    """Load ModernTCN from repo bytecode when source files are absent.

    The local tree currently contains ModernTCN bytecode under ``__pycache__``
    but not the corresponding source files. This uses that implementation
    rather than substituting a proxy model.
    """

    try:
        from basicts.models.ModernTCN import ModernTCNConfig, ModernTCNForForecasting

        return ModernTCNConfig, ModernTCNForForecasting
    except Exception:
        pass

    tag = sys.implementation.cache_tag
    base = SRC / "basicts" / "models" / "ModernTCN"
    config_path = base / "config" / "__pycache__" / f"moderntcn_config.{tag}.pyc"
    arch_path = base / "arch" / "__pycache__" / f"moderntcn_arch.{tag}.pyc"
    if not config_path.exists() or not arch_path.exists():
        raise ImportError(
            "ModernTCN source/import is unavailable and matching bytecode was not found. "
            "Add src/basicts/models/ModernTCN/*.py before running full end-to-end COSTAR."
        )

    modules = (
        ("basicts.models.ModernTCN.config.moderntcn_config", config_path),
        ("basicts.models.ModernTCN.arch.moderntcn_arch", arch_path),
    )
    loaded: dict[str, Any] = {}
    for module_name, path in modules:
        loader = importlib.machinery.SourcelessFileLoader(module_name, str(path))
        spec = importlib.util.spec_from_loader(module_name, loader)
        if spec is None:
            raise ImportError(f"Could not create import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        sys.modules[module_name] = module
        loaded[module_name] = module
    return loaded[modules[0][0]].ModernTCNConfig, loaded[modules[1][0]].ModernTCNForForecasting


class ForecastingExpert(nn.Module):
    """Common expert interface: ``forecast = expert(history)``."""

    def __init__(self, name: str, model: nn.Module) -> None:
        super().__init__()
        self.name = name
        self.model = model

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if self.name == "TimesNet":
            return self.model(history, None)
        return self.model(history)


def build_expert(name: str, config: EndToEndCOSTARTSConfig) -> ForecastingExpert:
    input_len = config.input_len
    horizon = config.forecast_horizon
    features = config.num_features
    hidden = config.expert_hidden_size
    if name == "DLinear":
        model = DLinear(DLinearConfig(input_len=input_len, output_len=horizon, num_features=features, individual=False))
    elif name == "PatchTST":
        model = PatchTSTForForecasting(
            PatchTSTConfig(
                input_len=input_len,
                output_len=horizon,
                num_features=features,
                hidden_size=hidden,
                intermediate_size=hidden * 2,
                n_heads=1,
                num_layers=1,
                patch_len=16,
                patch_stride=8,
                use_revin=True,
                affine=True,
            )
        )
    elif name == "iTransformer":
        model = iTransformerForForecasting(
            iTransformerConfig(
                input_len=input_len,
                output_len=horizon,
                num_features=features,
                hidden_size=hidden,
                intermediate_size=hidden * 2,
                n_heads=1,
                num_layers=1,
                dropout=config.expert_dropout,
                use_revin=True,
            )
        )
    elif name == "TimesNet":
        model = TimesNetForForecasting(
            TimesNetConfig(
                input_len=input_len,
                output_len=horizon,
                num_features=features,
                hidden_size=hidden,
                intermediate_size=hidden * 2,
                num_layers=1,
                num_kernels=3,
                top_k=3,
                dropout=config.expert_dropout,
                use_timestamps=False,
            )
        )
    elif name == "ModernTCN":
        ModernTCNConfig, ModernTCNForForecasting = _load_sourceless_modern_tcn()
        model = ModernTCNForForecasting(
            ModernTCNConfig(
                input_len=input_len,
                output_len=horizon,
                num_features=features,
                hidden_size=hidden,
                num_layers=3,
                kernel_size=7,
                dropout=config.expert_dropout,
                use_revin=True,
                affine=False,
            )
        )
    else:
        raise ValueError(f"Unknown expert: {name}")
    return ForecastingExpert(name, model)


def masked_sample_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(prediction.dtype)
    denom = mask_f.flatten(1).sum(dim=1).clamp_min(1.0)
    return ((prediction - target).square() * mask_f).flatten(1).sum(dim=1) / denom


def masked_sample_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(prediction.dtype)
    denom = mask_f.flatten(1).sum(dim=1).clamp_min(1.0)
    return ((prediction - target).abs() * mask_f).flatten(1).sum(dim=1) / denom


class FullEndToEndCOSTARTS(nn.Module):
    """Jointly trained experts plus differentiable sequential COSTAR router."""

    def __init__(self, config: EndToEndCOSTARTSConfig | Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__()
        if config is None:
            config = EndToEndCOSTARTSConfig(**kwargs)
        elif isinstance(config, Mapping):
            values = dict(config)
            values.update(kwargs)
            config = EndToEndCOSTARTSConfig(**values)
        elif kwargs:
            values = asdict(config)
            values.update(kwargs)
            config = EndToEndCOSTARTSConfig(**values)
        self.config = config
        self.expert_names = EXPERT_ORDER
        self.num_experts = len(EXPERT_ORDER)
        self.max_queries = int(config.max_queries)
        self.experts = nn.ModuleDict({name: build_expert(name, config) for name in EXPERT_ORDER})

        hidden = config.router_hidden_dim
        emb = config.router_embedding_dim
        horizon_features = config.forecast_horizon * config.num_features
        self.history_encoder = nn.Sequential(
            nn.Conv1d(config.num_features, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.GroupNorm(1, hidden),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
            nn.GroupNorm(1, hidden),
            nn.AdaptiveAvgPool1d(1),
        )
        self.history_projection = nn.Sequential(nn.Linear(hidden, emb), nn.GELU(), nn.LayerNorm(emb))
        self.mask_encoder = nn.Sequential(nn.Linear(self.num_experts, emb), nn.GELU(), nn.LayerNorm(emb))
        self.current_forecast_encoder = nn.Sequential(nn.Linear(horizon_features, emb), nn.GELU(), nn.LayerNorm(emb))
        self.observed_forecast_encoder = nn.Sequential(nn.Linear(horizon_features, emb), nn.GELU(), nn.LayerNorm(emb))
        self.scalar_encoder = nn.Sequential(nn.Linear(4, emb), nn.GELU(), nn.LayerNorm(emb))
        self.fusion = nn.Sequential(
            nn.Linear(emb * 5, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, emb),
            nn.GELU(),
            nn.LayerNorm(emb),
        )
        self.route_head = nn.Linear(emb, self.num_experts)
        self.stop_head = nn.Linear(emb, 1)
        self.finalizer = nn.Sequential(
            nn.Linear(emb + horizon_features, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, horizon_features),
        )

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    def expert_parameters(self) -> list[nn.Parameter]:
        return list(self.experts.parameters())

    def router_parameters(self) -> list[nn.Parameter]:
        expert_ids = {id(param) for param in self.experts.parameters()}
        return [param for param in self.parameters() if id(param) not in expert_ids]

    def expert_forecasts_all(self, history: torch.Tensor) -> torch.Tensor:
        forecasts = [self.experts[name](history) for name in EXPERT_ORDER]
        return torch.stack(forecasts, dim=-1)

    def _encode_state(
        self,
        history: torch.Tensor,
        queried_soft_mask: torch.Tensor,
        current_forecast: torch.Tensor,
        observed_summary: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> torch.Tensor:
        batch = history.shape[0]
        h = self.history_projection(self.history_encoder(history.transpose(1, 2)).squeeze(-1))
        m = self.mask_encoder(queried_soft_mask)
        c = self.current_forecast_encoder(current_forecast.reshape(batch, -1))
        o = self.observed_forecast_encoder(observed_summary.reshape(batch, -1))
        s = self.scalar_encoder(scalar_features)
        return self.fusion(torch.cat((h, m, c, o, s), dim=-1))

    def _routing_weights(self, logits: torch.Tensor, queried_soft_mask: torch.Tensor, temperature: float) -> torch.Tensor:
        availability = (1.0 - queried_soft_mask).clamp_min(1e-6)
        masked_logits = logits + availability.log()
        if self.training and self.config.gumbel_routing:
            weights = F.gumbel_softmax(
                masked_logits,
                tau=max(float(temperature), 1e-3),
                hard=bool(self.config.straight_through),
                dim=-1,
            )
            weights = weights * availability
            return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return F.softmax(masked_logits / max(float(temperature), 1e-3), dim=-1)

    def forward_soft(
        self,
        history: torch.Tensor,
        temperature: float | None = None,
        expert_forecasts: torch.Tensor | None = None,
        minimum_queries: int = 1,
    ) -> dict[str, Any]:
        temperature = self.config.route_temperature if temperature is None else float(temperature)
        minimum_queries = max(1, min(int(minimum_queries), self.max_queries))
        if expert_forecasts is None:
            expert_forecasts = self.expert_forecasts_all(history)
        batch = history.shape[0]
        device = history.device
        dtype = history.dtype
        current = torch.zeros(
            batch,
            self.config.forecast_horizon,
            self.config.num_features,
            device=device,
            dtype=dtype,
        )
        observed = torch.zeros_like(current)
        queried_mask = torch.zeros(batch, self.num_experts, device=device, dtype=dtype)
        forecast_mass = torch.zeros(batch, device=device, dtype=dtype)
        continue_prob = torch.ones(batch, device=device, dtype=dtype)
        route_probs: list[torch.Tensor] = []
        route_logits: list[torch.Tensor] = []
        stop_logits: list[torch.Tensor] = []
        stop_probs: list[torch.Tensor] = []
        query_gates: list[torch.Tensor] = []
        current_before: list[torch.Tensor] = []
        current_after: list[torch.Tensor] = []
        reps: list[torch.Tensor] = []

        for step in range(self.max_queries):
            scalar = torch.stack(
                (
                    forecast_mass / max(float(self.max_queries), 1.0),
                    queried_mask.mean(dim=1),
                    continue_prob,
                    torch.full_like(continue_prob, float(step) / max(float(self.max_queries - 1), 1.0)),
                ),
                dim=1,
            )
            rep = self._encode_state(history, queried_mask, current, observed, scalar)
            logits = self.route_head(rep)
            stop_logit = self.stop_head(rep).squeeze(-1)
            stop_prob = torch.sigmoid(stop_logit)
            if step < minimum_queries:
                query_gate = torch.ones_like(continue_prob)
            else:
                query_gate = continue_prob * (1.0 - stop_prob)
            weights = self._routing_weights(logits, queried_mask, temperature)
            step_forecast = torch.einsum("bhfe,be->bhf", expert_forecasts, weights)
            current_before.append(current)
            next_mass = forecast_mass + query_gate
            current = (current * forecast_mass[:, None, None] + step_forecast * query_gate[:, None, None]) / next_mass.clamp_min(1e-6)[:, None, None]
            observed = current
            queried_mask = (queried_mask + query_gate[:, None] * weights).clamp(max=1.0)
            forecast_mass = next_mass
            if step > 0:
                continue_prob = query_gate
            route_probs.append(weights)
            route_logits.append(logits)
            stop_logits.append(stop_logit)
            stop_probs.append(stop_prob)
            query_gates.append(query_gate)
            current_after.append(current)
            reps.append(rep)

        final_rep = reps[-1]
        residual = self.finalizer(torch.cat((final_rep, current.reshape(batch, -1)), dim=-1)).reshape_as(current)
        final_forecast = current + float(self.config.finalizer_scale) * residual
        return {
            "forecast": final_forecast,
            "sequential_forecast": current,
            "expert_forecasts": expert_forecasts,
            "route_probs": torch.stack(route_probs, dim=1),
            "route_logits": torch.stack(route_logits, dim=1),
            "stop_logits": torch.stack(stop_logits, dim=1),
            "stop_probs": torch.stack(stop_probs, dim=1),
            "query_gates": torch.stack(query_gates, dim=1),
            "queried_soft_mask": queried_mask,
            "current_before": torch.stack(current_before, dim=1),
            "current_after": torch.stack(current_after, dim=1),
            "final_representation": final_rep,
        }

    def forward(self, history: torch.Tensor, temperature: float | None = None) -> dict[str, Any]:
        return self.forward_soft(history, temperature=temperature)

    @torch.no_grad()
    def forward_hard(
        self,
        history: torch.Tensor,
        stop_threshold: float = 0.5,
        max_queries: int | None = None,
        minimum_queries: int = 1,
    ) -> dict[str, Any]:
        was_training = self.training
        self.eval()
        max_queries = self.max_queries if max_queries is None else int(max_queries)
        minimum_queries = max(1, min(int(minimum_queries), max_queries))
        batch = history.shape[0]
        device = history.device
        dtype = history.dtype
        current = torch.zeros(batch, self.config.forecast_horizon, self.config.num_features, device=device, dtype=dtype)
        observed = torch.zeros_like(current)
        queried_mask = torch.zeros(batch, self.num_experts, device=device, dtype=dtype)
        forecast_sum = torch.zeros_like(current)
        counts = torch.zeros(batch, device=device, dtype=torch.long)
        queried_ids = torch.full((batch, max_queries), -1, device=device, dtype=torch.long)
        active = torch.ones(batch, device=device, dtype=torch.bool)
        forecast_cache: dict[int, torch.Tensor] = {}
        final_rep = None

        for step in range(max_queries):
            scalar = torch.stack(
                (
                    counts.to(dtype) / max(float(max_queries), 1.0),
                    queried_mask.mean(dim=1),
                    active.to(dtype),
                    torch.full((batch,), float(step) / max(float(max_queries - 1), 1.0), device=device, dtype=dtype),
                ),
                dim=1,
            )
            rep = self._encode_state(history, queried_mask, current, observed, scalar)
            final_rep = rep
            stop_prob = torch.sigmoid(self.stop_head(rep).squeeze(-1))
            should_query = active if step < minimum_queries else active & (stop_prob <= float(stop_threshold))
            if not bool(should_query.any()):
                break
            logits = self.route_head(rep).masked_fill(queried_mask.to(torch.bool), -1e9)
            next_ids = logits.argmax(dim=1)
            for expert_idx, expert_name in enumerate(EXPERT_ORDER):
                rows = should_query & (next_ids == expert_idx)
                if not bool(rows.any()):
                    continue
                if expert_idx not in forecast_cache:
                    forecast_cache[expert_idx] = self.experts[expert_name](history)
                forecast_sum[rows] = forecast_sum[rows] + forecast_cache[expert_idx][rows]
                queried_mask[rows, expert_idx] = 1.0
                queried_ids[rows, step] = expert_idx
                counts[rows] += 1
            current = forecast_sum / counts.clamp_min(1).to(dtype)[:, None, None]
            observed = current
            active = should_query & (counts < max_queries)

        if final_rep is None:
            scalar = torch.zeros(batch, 4, device=device, dtype=dtype)
            final_rep = self._encode_state(history, queried_mask, current, observed, scalar)
        residual = self.finalizer(torch.cat((final_rep, current.reshape(batch, -1)), dim=-1)).reshape_as(current)
        output = {
            "forecast": current + float(self.config.finalizer_scale) * residual,
            "sequential_forecast": current,
            "query_counts": counts,
            "queried_ids": queried_ids,
            "expert_usage": queried_mask,
        }
        if was_training:
            self.train()
        return output


def end_to_end_costarts_loss(
    outputs: Mapping[str, Any],
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha_expert: float,
    lambda_query: float,
    lambda_balance: float,
    lambda_stop: float,
    lambda_counterfactual_query: float = 0.0,
    lambda_counterfactual_stop: float = 0.0,
    counterfactual_tau: float = 0.1,
    minimum_queries: int = 1,
    query_cost: float = 0.0,
) -> dict[str, torch.Tensor]:
    final_per = masked_sample_mse(outputs["forecast"], target, mask)
    final_loss = final_per.mean()
    expert_forecasts = outputs["expert_forecasts"]
    expert_losses = []
    for index in range(expert_forecasts.shape[-1]):
        expert_losses.append(masked_sample_mse(expert_forecasts[..., index], target, mask).mean())
    expert_loss_tensor = torch.stack(expert_losses)
    expert_loss = expert_loss_tensor.mean()

    query_gates = outputs["query_gates"]
    expected_queries = query_gates.sum(dim=1).mean()
    query_loss = expected_queries

    route_probs = outputs["route_probs"]
    usage = (route_probs * query_gates.unsqueeze(-1)).sum(dim=(0, 1))
    usage = usage / usage.sum().clamp_min(1e-6)
    uniform = torch.full_like(usage, 1.0 / usage.numel())
    balance_loss = F.kl_div((usage + 1e-6).log(), uniform, reduction="sum")

    before = outputs["current_before"][:, 1:]
    after = outputs["current_after"][:, 1:]
    if before.numel() == 0:
        stop_loss = torch.zeros((), device=target.device, dtype=target.dtype)
    else:
        before_mse = masked_sample_mse(before.flatten(0, 1), target[:, None].expand_as(before).flatten(0, 1), mask[:, None].expand_as(before).flatten(0, 1))
        after_mse = masked_sample_mse(after.flatten(0, 1), target[:, None].expand_as(after).flatten(0, 1), mask[:, None].expand_as(after).flatten(0, 1))
        continue_target = ((before_mse - after_mse) > float(query_cost)).to(target.dtype).detach()
        stop_target = 1.0 - continue_target
        stop_logits = outputs["stop_logits"][:, 1:].flatten()
        weights = outputs["query_gates"][:, :-1].detach().flatten().clamp_min(0.0)
        bce = F.binary_cross_entropy_with_logits(stop_logits, stop_target, reduction="none")
        stop_loss = (bce * weights).sum() / weights.sum().clamp_min(1.0)

    counterfactual_query_loss, counterfactual_stop_loss, cf_stats = counterfactual_routing_losses(
        outputs,
        target,
        mask,
        query_cost=float(query_cost),
        tau=float(counterfactual_tau),
        minimum_queries=int(minimum_queries),
    )

    total = (
        final_loss
        + float(alpha_expert) * expert_loss
        + float(lambda_query) * query_loss
        + float(lambda_balance) * balance_loss
        + float(lambda_stop) * stop_loss
        + float(lambda_counterfactual_query) * counterfactual_query_loss
        + float(lambda_counterfactual_stop) * counterfactual_stop_loss
    )
    entropy = -(route_probs * (route_probs + 1e-8).log()).sum(dim=-1)
    result = {
        "total_loss": total,
        "final_loss": final_loss,
        "expert_loss": expert_loss,
        "query_loss": query_loss,
        "balance_loss": balance_loss,
        "stop_loss": stop_loss,
        "counterfactual_query_loss": counterfactual_query_loss,
        "counterfactual_stop_loss": counterfactual_stop_loss,
        "mean_entropy": entropy.mean(),
        "expected_queries": expected_queries.detach(),
    }
    result.update(cf_stats)
    for index, name in enumerate(EXPERT_ORDER):
        result[f"expert_loss_{name}"] = expert_loss_tensor[index]
        result[f"usage_{name}"] = usage[index].detach()
    return result


def counterfactual_routing_losses(
    outputs: Mapping[str, Any],
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    query_cost: float,
    tau: float,
    minimum_queries: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise query and stop heads with privileged counterfactual losses.

    The router inputs still only contain the current observed/queried forecast
    state. Counterfactual expert forecasts are used only here to build targets.
    """

    expert_forecasts = outputs["expert_forecasts"]
    route_logits = outputs["route_logits"]
    route_probs = outputs["route_probs"].detach()
    query_gates = outputs["query_gates"].detach()
    current_before = outputs["current_before"]
    stop_logits = outputs["stop_logits"]
    batch, steps, _, _, num_experts = route_probs.shape[0], route_probs.shape[1], *expert_forecasts.shape[1:]
    del batch  # only used to make the expected shape explicit

    mask_f = mask.to(expert_forecasts.dtype)
    denom = mask_f.flatten(1).sum(dim=1).clamp_min(1.0)
    state_weights = torch.ones_like(query_gates)
    if query_gates.shape[1] > 1:
        state_weights[:, 1:] = query_gates[:, :-1]

    queried_before = torch.zeros_like(route_probs)
    if route_probs.shape[1] > 1:
        prior_usage = route_probs[:, :-1] * query_gates[:, :-1, None]
        queried_before[:, 1:] = prior_usage.cumsum(dim=1)
    availability = (1.0 - queried_before).clamp_min(1e-6)
    masses = torch.zeros_like(query_gates)
    if query_gates.shape[1] > 1:
        masses[:, 1:] = query_gates[:, :-1].cumsum(dim=1)

    after_losses = []
    for expert_index in range(num_experts):
        expert_prediction = expert_forecasts[..., expert_index]
        candidate = (
            current_before * masses[:, :, None, None]
            + expert_prediction[:, None, :, :] * availability[:, :, expert_index, None, None]
        ) / (masses[:, :, None, None] + availability[:, :, expert_index, None, None]).clamp_min(1e-6)
        diff = (candidate - target[:, None, :, :]).square() * mask_f[:, None, :, :]
        loss = diff.flatten(2).sum(dim=2) / denom[:, None]
        after_losses.append(loss)
    counterfactual_loss = torch.stack(after_losses, dim=-1)

    masked_cf = counterfactual_loss.masked_fill(availability <= 1e-5, 1e6)
    target_probs = torch.softmax(-masked_cf / max(float(tau), 1e-4), dim=-1).detach()
    log_probs = torch.log_softmax(route_logits + availability.log(), dim=-1)
    query_kl = (target_probs * ((target_probs + 1e-8).log() - log_probs)).sum(dim=-1)
    query_loss = (query_kl * state_weights).sum() / state_weights.sum().clamp_min(1.0)

    current_loss = (
        ((current_before - target[:, None, :, :]).square() * mask_f[:, None, :, :]).flatten(2).sum(dim=2)
        / denom[:, None]
    )
    best_query_loss = masked_cf.min(dim=-1).values
    stop_target = (current_loss <= (best_query_loss + float(query_cost))).to(target.dtype)
    if minimum_queries > 1:
        stop_target[:, : min(int(minimum_queries), stop_target.shape[1])] = 0.0
    stop_bce = F.binary_cross_entropy_with_logits(stop_logits, stop_target.detach(), reduction="none")
    stop_loss = (stop_bce * state_weights).sum() / state_weights.sum().clamp_min(1.0)

    stats: dict[str, torch.Tensor] = {}
    for step in range(stop_logits.shape[1]):
        weight = state_weights[:, step]
        denom_step = weight.sum().clamp_min(1.0)
        stats[f"counterfactual_stop_target_step_{step + 1}"] = (stop_target[:, step] * weight).sum().detach() / denom_step
    return query_loss, stop_loss, stats


def gradient_report(model: FullEndToEndCOSTARTS) -> dict[str, float]:
    report: dict[str, float] = {}
    for name in EXPERT_ORDER:
        total = 0.0
        for param in model.experts[name].parameters():
            if param.grad is not None:
                total += float(param.grad.detach().abs().sum().cpu().item())
        report[name] = total
    router_total = 0.0
    for param in model.router_parameters():
        if param.grad is not None:
            router_total += float(param.grad.detach().abs().sum().cpu().item())
    report["COSTAR_router"] = router_total
    return report
