"""Train candidate forecasting experts on leakage-safe chronological splits."""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from basicts.models.DLinear import DLinear, DLinearConfig
from basicts.models.ModernTCN import ModernTCNConfig, ModernTCNForForecasting
from basicts.models.PatchTST import PatchTSTConfig, PatchTSTForForecasting
from basicts.models.TimesNet import TimesNetConfig, TimesNetForForecasting
from basicts.models.iTransformer import (
    iTransformerConfig,
    iTransformerForForecasting,
)
from basicts.scaler import ZScoreScaler
from scripts.chronological_expert_training import (
    DEFAULT_INPUT_LEN,
    DEFAULT_NUM_FEATURES,
    DEFAULT_OUTPUT_LEN,
    _accumulate_errors,
    _assert_full_data_contract,
    _configured_forecasting_loss,
    _dataset_config_summary,
    _prepare_forecasting_batch,
    _print_history_table,
    _prediction_tensor,
    build_expert_dataloaders,
    fit_scaler_on_expert_train,
    load_full_chronological_data,
)


class ExpertSpec:
    """Small container for a trainable candidate expert."""

    def __init__(
        self,
        key: str,
        display_name: str,
        checkpoint_name: str,
        model_class: type[nn.Module],
        config_factory: Callable[[], object],
        module_name: str,
        model_class_name: str,
        config_class_name: str,
        requires_timestamps: bool = False,
    ) -> None:
        self.key = key
        self.display_name = display_name
        self.checkpoint_name = checkpoint_name
        self.model_class = model_class
        self.config_factory = config_factory
        self.module_name = module_name
        self.model_class_name = model_class_name
        self.config_class_name = config_class_name
        self.requires_timestamps = requires_timestamps


def _dlinear_config() -> DLinearConfig:
    return DLinearConfig(
        input_len=DEFAULT_INPUT_LEN,
        output_len=DEFAULT_OUTPUT_LEN,
        num_features=DEFAULT_NUM_FEATURES,
        moving_avg=25,
        stride=1,
        individual=False,
    )


def _patchtst_config() -> PatchTSTConfig:
    return PatchTSTConfig(
        input_len=DEFAULT_INPUT_LEN,
        output_len=DEFAULT_OUTPUT_LEN,
        num_features=DEFAULT_NUM_FEATURES,
        patch_len=16,
        patch_stride=8,
        padding=True,
        hidden_size=64,
        n_heads=4,
        intermediate_size=128,
        num_layers=1,
        attn_dropout=0.1,
        fc_dropout=0.1,
        head_dropout=0.0,
        use_revin=False,
        output_attentions=False,
    )


def _itransformer_config() -> iTransformerConfig:
    return iTransformerConfig(
        input_len=DEFAULT_INPUT_LEN,
        output_len=DEFAULT_OUTPUT_LEN,
        num_features=DEFAULT_NUM_FEATURES,
        hidden_size=64,
        n_heads=4,
        intermediate_size=128,
        num_layers=1,
        dropout=0.1,
        use_revin=False,
        output_attentions=False,
    )


def _timesnet_config() -> TimesNetConfig:
    return TimesNetConfig(
        input_len=DEFAULT_INPUT_LEN,
        output_len=DEFAULT_OUTPUT_LEN,
        num_features=DEFAULT_NUM_FEATURES,
        hidden_size=64,
        intermediate_size=128,
        num_layers=1,
        num_kernels=3,
        top_k=3,
        dropout=0.1,
        use_timestamps=False,
        timestamp_sizes=None,
    )


def _moderntcn_config() -> ModernTCNConfig:
    return ModernTCNConfig(
        input_len=DEFAULT_INPUT_LEN,
        output_len=DEFAULT_OUTPUT_LEN,
        num_features=DEFAULT_NUM_FEATURES,
        hidden_size=64,
        num_layers=3,
        kernel_size=7,
        expansion=2,
        dropout=0.1,
        use_revin=True,
        affine=False,
        subtract_last=False,
    )


EXPERT_SPECS: Dict[str, ExpertSpec] = {
    "dlinear": ExpertSpec(
        key="dlinear",
        display_name="DLinear",
        checkpoint_name="best_dlinear.pt",
        model_class=DLinear,
        config_factory=_dlinear_config,
        module_name="DLinear",
        model_class_name="DLinear",
        config_class_name="DLinearConfig",
    ),
    "patchtst": ExpertSpec(
        key="patchtst",
        display_name="PatchTST",
        checkpoint_name="best_patchtst.pt",
        model_class=PatchTSTForForecasting,
        config_factory=_patchtst_config,
        module_name="PatchTST",
        model_class_name="PatchTSTForForecasting",
        config_class_name="PatchTSTConfig",
    ),
    "itransformer": ExpertSpec(
        key="itransformer",
        display_name="iTransformer",
        checkpoint_name="best_itransformer.pt",
        model_class=iTransformerForForecasting,
        config_factory=_itransformer_config,
        module_name="iTransformer",
        model_class_name="iTransformerForForecasting",
        config_class_name="iTransformerConfig",
    ),
    "timesnet": ExpertSpec(
        key="timesnet",
        display_name="TimesNet",
        checkpoint_name="best_timesnet.pt",
        model_class=TimesNetForForecasting,
        config_factory=_timesnet_config,
        module_name="TimesNet",
        model_class_name="TimesNetForForecasting",
        config_class_name="TimesNetConfig",
        requires_timestamps=True,
    ),
    "moderntcn": ExpertSpec(
        key="moderntcn",
        display_name="ModernTCN",
        checkpoint_name="best_moderntcn.pt",
        model_class=ModernTCNForForecasting,
        config_factory=_moderntcn_config,
        module_name="ModernTCN",
        model_class_name="ModernTCNForForecasting",
        config_class_name="ModernTCNConfig",
    ),
}

UNAVAILABLE_EXPERTS = {
    "nhits": (
        "N-HiTS is not implemented in src/basicts/models. FEDformer was "
        "checked as the requested fallback, but it is not implemented in this "
        "repository either, so no candidate checkpoint can be trained from "
        "existing project classes."
    ),
    "fedformer": (
        "FEDformer is not implemented in src/basicts/models, so it cannot be "
        "used as the N-HiTS substitute without adding a new model implementation."
    ),
}


def _call_model(
    model: nn.Module,
    inputs: torch.Tensor,
    requires_timestamps: bool,
) -> torch.Tensor:
    if requires_timestamps:
        output = model(inputs, None)
    else:
        output = model(inputs)
    return _prediction_tensor(output)


def evaluate_candidate(
    model: nn.Module,
    loader: Iterable[dict],
    model_name: str,
    device: torch.device,
    scaler,
    requires_timestamps: bool,
    print_shapes: bool = True,
) -> Tuple[float, float]:
    """Evaluate every expert-validation window without gradients."""

    model.eval()
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    element_count = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            inputs, targets, targets_mask = _prepare_forecasting_batch(
                batch,
                device,
                scaler,
            )
            prediction = _call_model(model, inputs, requires_timestamps)
            if tuple(prediction.shape) != tuple(targets.shape):
                raise ValueError(
                    f"{model_name} prediction shape {tuple(prediction.shape)} "
                    f"does not match target shape {tuple(targets.shape)}"
                )
            if print_shapes and batch_index == 0:
                print(f"\n{model_name} first expert-validation batch")
                print(f"input shape:      {list(inputs.shape)}")
                print(f"target shape:     {list(targets.shape)}")
                print(f"prediction shape: {list(prediction.shape)}")
            abs_sum, squared_sum, count = _accumulate_errors(
                prediction,
                targets,
                targets_mask,
            )
            absolute_error_sum += abs_sum
            squared_error_sum += squared_sum
            element_count += count

    if element_count == 0:
        raise ValueError("Validation loader produced no prediction elements")
    return absolute_error_sum / element_count, squared_error_sum / element_count


def train_candidate_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    spec: ExpertSpec,
    checkpoint_path: Path,
    train_loader: Iterable[dict],
    val_loader: Iterable[dict],
    max_epochs: int,
    patience: int,
    device: torch.device,
    scaler,
    model_config: dict,
    dataset_config: dict,
) -> Tuple[dict, ...]:
    """Train on expert_train, select by expert_val MAE, and save one checkpoint."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.to(device)
    history = []
    best_val_mae = float("inf")
    best_val_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_absolute_error_sum = 0.0
        train_squared_error_sum = 0.0
        train_element_count = 0

        for batch_index, batch in enumerate(train_loader):
            inputs, targets, targets_mask = _prepare_forecasting_batch(
                batch,
                device,
                scaler,
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = _call_model(model, inputs, spec.requires_timestamps)
            if tuple(prediction.shape) != tuple(targets.shape):
                raise ValueError(
                    f"{spec.display_name} prediction shape "
                    f"{tuple(prediction.shape)} does not match target shape "
                    f"{tuple(targets.shape)}"
                )
            if epoch == 1 and batch_index == 0:
                print(f"\n{spec.display_name} first expert-training batch")
                print(f"input shape:      {list(inputs.shape)}")
                print(f"target shape:     {list(targets.shape)}")
                print(f"prediction shape: {list(prediction.shape)}")
            loss = _configured_forecasting_loss(
                None,
                prediction,
                targets,
                targets_mask,
            )
            loss.backward()
            optimizer.step()

            abs_sum, squared_sum, count = _accumulate_errors(
                prediction,
                targets,
                targets_mask,
            )
            train_absolute_error_sum += abs_sum
            train_squared_error_sum += squared_sum
            train_element_count += count

        if train_element_count == 0:
            raise ValueError("Expert training loader produced no prediction elements")

        train_mae = train_absolute_error_sum / train_element_count
        train_mse = train_squared_error_sum / train_element_count
        val_mae, val_mse = evaluate_candidate(
            model=model,
            loader=val_loader,
            model_name=spec.display_name,
            device=device,
            scaler=scaler,
            requires_timestamps=spec.requires_timestamps,
            print_shapes=(epoch == 1),
        )

        checkpoint_saved = val_mae < best_val_mae
        if checkpoint_saved:
            best_val_mae = val_mae
            best_val_mse = val_mse
            best_epoch = epoch
            optimizer_state = optimizer.state_dict()
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer_state,
                    "optim_state_dict": optimizer_state,
                    "epoch": epoch,
                    "validation_mae": val_mae,
                    "validation_mse": val_mse,
                    "val_mae": val_mae,
                    "val_mse": val_mse,
                    "model_config": dict(model_config),
                    "dataset_config": dict(dataset_config),
                    "input_len": DEFAULT_INPUT_LEN,
                    "forecast_len": DEFAULT_OUTPUT_LEN,
                    "output_len": DEFAULT_OUTPUT_LEN,
                    "num_features": DEFAULT_NUM_FEATURES,
                    "best_validation_mae": val_mae,
                    "best_epoch": epoch,
                    "best_metrics": {
                        "val/MAE": val_mae,
                        "val/MSE": val_mse,
                    },
                    "expert_name": spec.display_name,
                    "model_key": spec.key,
                    "module_name": spec.module_name,
                    "model_class_name": spec.model_class_name,
                    "config_class_name": spec.config_class_name,
                    "requires_timestamps": spec.requires_timestamps,
                    **({"scaler_stats": scaler.stats} if scaler is not None else {}),
                },
                checkpoint_path,
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_mae": train_mae,
                "train_mse": train_mse,
                "val_mae": val_mae,
                "val_mse": val_mse,
                "checkpoint_saved": checkpoint_saved,
                "early_stopping_counter": epochs_without_improvement,
            }
        )
        print(
            f"{spec.display_name} epoch {epoch:>3d}/{max_epochs}: "
            f"training MAE={train_mae:.6f}, training MSE={train_mse:.6f}, "
            f"validation MAE={val_mae:.6f}, validation MSE={val_mse:.6f}, "
            f"early-stop counter={epochs_without_improvement}/{patience}"
        )

        if epochs_without_improvement >= patience:
            print(
                f"{spec.display_name}: early stopping after epoch {epoch} "
                f"({patience} epochs without lower validation MAE)."
            )
            break

    if best_epoch == 0:
        raise RuntimeError(
            f"{spec.display_name} never produced a finite validation MAE; "
            "no checkpoint was saved"
        )
    _print_history_table(spec.display_name, history)
    print(
        f"\nSelected {spec.display_name} epoch {best_epoch}: "
        f"validation MAE={best_val_mae:.6f}, validation MSE={best_val_mse:.6f}"
    )
    print(f"Saved: {checkpoint_path}")
    return tuple(history)


def parse_model_keys(raw_models: Sequence[str]) -> Sequence[str]:
    if not raw_models or raw_models == ["all"]:
        return tuple(EXPERT_SPECS)
    selected = []
    for model_key in raw_models:
        key = model_key.lower()
        if key in UNAVAILABLE_EXPERTS:
            print(f"{model_key}: {UNAVAILABLE_EXPERTS[key]}")
            continue
        if key not in EXPERT_SPECS:
            valid = ", ".join((*EXPERT_SPECS, *UNAVAILABLE_EXPERTS))
            raise ValueError(f"Unknown model {model_key!r}. Valid options: {valid}")
        selected.append(key)
    if not selected:
        raise ValueError("No trainable models were selected")
    return tuple(dict.fromkeys(selected))


def train_candidates(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    full_data = load_full_chronological_data(data_dir)
    _assert_full_data_contract(full_data, DEFAULT_NUM_FEATURES)
    train_loader, val_loader = build_expert_dataloaders(
        full_data=full_data,
        input_len=DEFAULT_INPUT_LEN,
        output_len=DEFAULT_OUTPUT_LEN,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if getattr(train_loader.dataset, "split_role", None) != "expert_train":
        raise RuntimeError("Training must use the chronological expert_train split")
    if getattr(val_loader.dataset, "split_role", None) != "expert_val":
        raise RuntimeError("Checkpoint selection must use expert_val")

    scaler = fit_scaler_on_expert_train(
        ZScoreScaler(norm_each_channel=True, rescale=False),
        train_loader,
    )
    dataset_config = _dataset_config_summary(len(full_data))
    model_keys = parse_model_keys(args.models)

    print("Candidate expert compatibility")
    print("  implemented: DLinear, PatchTST, iTransformer, TimesNet, ModernTCN")
    print(f"  unavailable: N-HiTS. {UNAVAILABLE_EXPERTS['nhits']}")
    print("\nTraining configuration")
    print(f"  data_dir: {data_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  device: {device}")
    print(f"  selected models: {', '.join(model_keys)}")
    print("  train split: expert_train")
    print("  selection split: expert_val")
    print("  router_train/router_val/test: not used")

    for model_key in model_keys:
        spec = EXPERT_SPECS[model_key]
        config = spec.config_factory()
        model = spec.model_class(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        checkpoint_path = output_dir / spec.checkpoint_name
        train_candidate_model(
            model=model,
            optimizer=optimizer,
            spec=spec,
            checkpoint_path=checkpoint_path,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=args.max_epochs,
            patience=args.patience,
            device=device,
            scaler=scaler,
            model_config=asdict(config),
            dataset_config=dataset_config,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train candidate forecasting experts on expert_train and select "
            "best checkpoints by expert_val MAE."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=(
            "Models to train: all, dlinear, patchtst, itransformer, timesnet, "
            "moderntcn. "
            "nhits/fedformer are reported as unavailable."
        ),
    )
    parser.add_argument("--data-dir", default="datasets/ETTh1")
    parser.add_argument("--output-dir", default="checkpoints/candidates")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print trainable and unavailable candidates without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_models:
        print("Trainable candidate experts:")
        for key, spec in EXPERT_SPECS.items():
            print(f"  {key}: {spec.display_name} -> {spec.checkpoint_name}")
        print("\nUnavailable requested/fallback experts:")
        for key, reason in UNAVAILABLE_EXPERTS.items():
            print(f"  {key}: {reason}")
        return
    train_candidates(args)


if __name__ == "__main__":
    main()
