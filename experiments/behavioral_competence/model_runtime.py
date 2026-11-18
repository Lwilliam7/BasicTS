"""Unified frozen-expert inference for perturbation forecasting.

Two checkpoint families exist in this repository and both are reused
unmodified:

- ETTh1 / ETTm1 / Weather / Electricity: the walk-forward family
  (`scripts/train_costarts_walkforward_experts.py` checkpoints, simplified
  configs, `pred_raw = pred_scaled * std + mean` output convention).
- ETTh2: the "clean_candidates" official-BasicTS-model family
  (`experiments/final_test_evaluation/run_final_frozen_test_evaluation.py`
  checkpoints, full model configs, raw-scale output directly -- this is the
  exact, already-verified inference path used for every ETTh2 result in this
  project).

Every model is loaded once, set to eval mode, and every parameter is
explicitly frozen (`requires_grad_(False)`). No training step is ever called.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.final_test_evaluation.run_final_frozen_test_evaluation import (  # noqa: E402
    build_etth2_model,
    call_etth2_model,
    etth2_checkpoint_path,
)
from scripts.train_costarts_walkforward_experts import build_model, call_model  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ExpertRuntime:
    dataset: str
    expert: str
    model: torch.nn.Module
    mean: torch.Tensor
    std: torch.Tensor
    call_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]
    rescale_output: bool
    checkpoint_path: Path
    checkpoint_sha256: str
    input_len: int
    horizon: int
    num_features: int
    device: torch.device

    def predict(self, history_raw: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
        """history_raw: [B, input_len, F] in raw (original) scale. Returns
        [B, horizon, F] forecasts in raw scale, matching the existing cache
        convention exactly (verified against the cached 'normal' forecast as
        an integrity check in run_behavioral_competence.py)."""
        outputs = []
        mean = self.mean.view(1, 1, -1)
        std = self.std.view(1, 1, -1)
        with torch.no_grad():
            for lo in range(0, history_raw.shape[0], batch_size):
                chunk = history_raw[lo : lo + batch_size].to(self.device)
                normalized = (chunk - mean) / std
                out = self.call_fn(self.model, normalized)
                if self.rescale_output:
                    out = out * std + mean
                outputs.append(out.detach().cpu().to(torch.float32))
        return torch.cat(outputs, dim=0)

    def predict_differentiable(self, history_raw: torch.Tensor) -> torch.Tensor:
        """Same forward pass as `predict`, but WITHOUT `torch.no_grad()` or
        `.detach()` -- used only to train the probe generator. The frozen
        expert's own parameters still have `requires_grad_(False)` (checked
        explicitly in run_learned_probe.py before and after every training
        step), so no expert weight is ever updated; gradients merely pass
        through the frozen forward computation to reach the perturbation
        `delta` that produced `history_raw`. No chunking: caller controls
        batch size directly since this keeps a single connected graph."""
        if history_raw.device != self.device:
            history_raw = history_raw.to(self.device)
        mean = self.mean.view(1, 1, -1)
        std = self.std.view(1, 1, -1)
        normalized = (history_raw - mean) / std
        out = self.call_fn(self.model, normalized)
        if self.rescale_output:
            out = out * std + mean
        return out


def _resolve_device(device: torch.device | str | None) -> torch.device:
    return torch.device("cpu") if device is None else torch.device(device)


def load_walkforward_expert(dataset: str, expert: str, checkpoint_root: Path, stage: str = "final_60", device: torch.device | str | None = None) -> ExpertRuntime:
    """`stage` selects which walk-forward checkpoint to load: "final_60" (used
    for router_val -- trained on 0-60%) or "block_a"/"block_ab" (used for the
    two out-of-sample sub-ranges that make up router_train -- trained on
    0-20%/0-40% respectively). Using the wrong stage for a given split would
    silently reintroduce in-sample leakage into the perturbation features."""
    path = checkpoint_root / stage / expert / "best_expert.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    model = build_model(ckpt["expert"], int(cfg["input_len"]), int(cfg["output_len"]), int(cfg["num_features"]), int(cfg["hidden_size"]))
    device_obj = _resolve_device(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device_obj)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return ExpertRuntime(
        dataset=dataset,
        expert=expert,
        model=model,
        mean=ckpt["scaler_mean"].to(torch.float32).to(device_obj),
        std=ckpt["scaler_std"].to(torch.float32).to(device_obj),
        call_fn=call_model,
        rescale_output=True,
        checkpoint_path=path,
        checkpoint_sha256=sha256_file(path),
        input_len=int(cfg["input_len"]),
        horizon=int(cfg["output_len"]),
        num_features=int(cfg["num_features"]),
        device=device_obj,
    )


def load_etth2_expert(expert: str, device: torch.device | str | None = None) -> ExpertRuntime:
    path = etth2_checkpoint_path(expert)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_etth2_model(ckpt)
    device_obj = _resolve_device(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"ETTh2/{expert}: checkpoint mismatch missing={missing}, unexpected={unexpected}")
    model.to(device_obj)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    stats = ckpt["scaler_stats"]
    return ExpertRuntime(
        dataset="ETTh2",
        expert=expert,
        model=model,
        mean=stats["mean"].to(torch.float32).view(-1).to(device_obj),
        std=stats["std"].to(torch.float32).view(-1).to(device_obj),
        call_fn=lambda m, h, _e=expert: call_etth2_model(m, _e, h),
        rescale_output=False,
        checkpoint_path=path,
        checkpoint_sha256=sha256_file(path),
        input_len=int(ckpt["input_len"]),
        horizon=int(ckpt["output_len"]),
        num_features=int(ckpt["num_features"]),
        device=device_obj,
    )


WALKFORWARD_CHECKPOINT_ROOTS = {
    "ETTh1": ROOT / "checkpoints/costarts_walkforward",
    "ETTm1": ROOT / "checkpoints/costarts_walkforward_ETTm1",
    "Weather": ROOT / "checkpoints/costarts_walkforward_Weather",
    "Electricity": ROOT / "checkpoints/costarts_walkforward_Electricity",
}


def load_expert_runtime(dataset: str, expert: str, stage: str = "final_60", device: torch.device | str | None = None) -> ExpertRuntime:
    """`stage` is only meaningful for the walk-forward family; ETTh2 uses a
    single fixed OOS checkpoint (trained on expert_train, 0-50%) for both
    router_train and router_val, so `stage` is ignored there."""
    if dataset == "ETTh2":
        return load_etth2_expert(expert, device=device)
    if dataset in WALKFORWARD_CHECKPOINT_ROOTS:
        return load_walkforward_expert(dataset, expert, WALKFORWARD_CHECKPOINT_ROOTS[dataset], stage=stage, device=device)
    raise ValueError(f"Unknown dataset: {dataset}")
