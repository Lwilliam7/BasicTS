"""Expert-native representation extraction for frozen forecasting experts.

Adapters here are intentionally observational: they register temporary PyTorch
hooks, run an eval-mode forward pass, pool the captured tensor to a compact
fixed vector, and remove the hooks immediately. They never mutate model
weights or replace forward methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch


EPS = 1e-8


@dataclass(frozen=True)
class AdapterSpec:
    expert: str
    hook_kind: str
    module_paths: tuple[str, ...]
    capture_description: str
    pooling_description: str


def module_by_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
    current: torch.nn.Module = model
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]  # type: ignore[index]
        else:
            current = getattr(current, part)
    return current


def adapter_spec(expert: str, model: torch.nn.Module) -> AdapterSpec:
    if expert == "DLinear":
        return AdapterSpec(
            expert=expert,
            hook_kind="forward",
            module_paths=("linear_seasonal", "linear_trend"),
            capture_description="DLinear decomposition branch outputs before seasonal+trend summation.",
            pooling_description="Stack seasonal/trend outputs as tokens and pool over branch/variable tokens.",
        )
    if expert == "PatchTST":
        return AdapterSpec(
            expert=expert,
            hook_kind="forward",
            module_paths=("backbone",),
            capture_description="PatchTST backbone encoder output before flatten and forecasting_head.",
            pooling_description="Treat variable-patch states as tokens; concatenate mean/std/max/min/first/last token summaries.",
        )
    if expert == "iTransformer":
        return AdapterSpec(
            expert=expert,
            hook_kind="forward",
            module_paths=("backbone",),
            capture_description="iTransformer final inverted-token encoder output before forecasting_head.",
            pooling_description="Treat variable tokens as tokens; concatenate mean/std/max/min/first/last token summaries.",
        )
    if expert == "TimesNet":
        return AdapterSpec(
            expert=expert,
            hook_kind="forward",
            module_paths=("backbone",),
            capture_description="TimesNet final temporal backbone state before output projection.",
            pooling_description="Use the last forecast-horizon temporal states; concatenate mean/std/max/min/first/last summaries.",
        )
    if expert == "ModernTCN":
        if hasattr(model, "backbone"):
            return AdapterSpec(
                expert=expert,
                hook_kind="forward",
                module_paths=("backbone",),
                capture_description="ModernTCN backbone convolutional state before temporal_head and feature_head.",
                pooling_description="Transpose channel-first backbone output to time tokens; concatenate mean/std/max/min/first/last summaries.",
            )
        return AdapterSpec(
            expert=expert,
            hook_kind="forward",
            module_paths=("norms.2",),
            capture_description="Compact walk-forward ModernTCN final normalization output before linear head.",
            pooling_description="Treat temporal hidden states as tokens; concatenate mean/std/max/min/first/last summaries.",
        )
    raise ValueError(f"No expert-native adapter for {expert}")


def _unwrap_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, Mapping):
        output = output["prediction"]
    if not torch.is_tensor(output):
        raise TypeError(f"Expected tensor-like hook output, got {type(output).__name__}")
    return output


def _tokens_last(expert: str, captured: Mapping[str, torch.Tensor], horizon: int) -> torch.Tensor:
    if expert == "DLinear":
        seasonal = captured["linear_seasonal"]
        trend = captured["linear_trend"]
        x = torch.stack((seasonal, trend), dim=1)  # [B,2,F,H]
        return x.reshape(x.shape[0], -1, x.shape[-1])

    x = next(iter(captured.values()))
    if expert == "PatchTST":
        return x.reshape(x.shape[0], -1, x.shape[-1])
    if expert == "iTransformer":
        return x
    if expert == "TimesNet":
        x = x[:, -horizon:, :]
        return x
    if expert == "ModernTCN":
        if x.ndim == 3 and x.shape[1] < x.shape[2]:
            return x.transpose(1, 2)
        return x.reshape(x.shape[0], -1, x.shape[-1])
    if x.ndim == 2:
        return x[:, None, :]
    return x.reshape(x.shape[0], -1, x.shape[-1])


def pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected [B,T,D] tokens, got {tuple(tokens.shape)}")
    mean = tokens.mean(dim=1)
    std = tokens.std(dim=1, unbiased=False)
    maxv = tokens.amax(dim=1)
    minv = tokens.amin(dim=1)
    first = tokens[:, 0, :]
    last = tokens[:, -1, :]
    return torch.cat((mean, std, maxv, minv, first, last), dim=1).to(torch.float32)


def extract_with_hooks(
    *,
    model: torch.nn.Module,
    expert: str,
    call_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    normalized_history: torch.Tensor,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    spec = adapter_spec(expert, model)
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(path: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            captured[path] = _unwrap_output(output).detach()

        return hook

    for path in spec.module_paths:
        handles.append(module_by_path(model, path).register_forward_hook(make_hook(path)))
    try:
        with torch.no_grad():
            prediction = call_fn(model, normalized_history)
    finally:
        for handle in handles:
            handle.remove()

    missing = [path for path in spec.module_paths if path not in captured]
    if missing:
        raise RuntimeError(f"{expert}: hooks did not capture {missing}")

    raw_shape = {path: list(t.shape) for path, t in captured.items()}
    tokens = _tokens_last(expert, captured, horizon)
    pooled = pool_tokens(tokens)
    manifest = {
        "expert": expert,
        "hook_kind": spec.hook_kind,
        "module_paths": list(spec.module_paths),
        "capture_description": spec.capture_description,
        "raw_shape_before_pooling": raw_shape,
        "token_shape_after_adapter": list(tokens.shape),
        "pooling": spec.pooling_description,
        "pooled_feature_dim": int(pooled.shape[1]),
    }
    return prediction, pooled.detach(), manifest
