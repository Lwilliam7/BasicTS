"""Evaluate COSTARTS forced-query and no-stop modes from cached predictions.

This script never runs or updates forecasting experts. It loads a saved
COSTARTS router checkpoint plus the offline COSTARTS train/validation caches
and computes counterfactual routing metrics from cached expert predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

try:
    from scripts.train_costarts_router import COSTARTSRouter, _select_expert_from_outputs
except ImportError:
    from train_costarts_router import COSTARTSRouter, _select_expert_from_outputs


DEFAULT_TRAIN_CACHE = "cache/costarts_router_train_cache.pt"
DEFAULT_VAL_CACHE = "cache/costarts_router_val_cache.pt"
DEFAULT_CHECKPOINT = "checkpoints/costarts/best_costarts_router.pt"
DEFAULT_OUTPUT_DIR = "results/router_summary/costarts/forced_query"

QUERY_SELECTORS = (
    "query_logits",
    "predicted_error",
    "oracle_improvement",
    "pairwise_reranker",
)
FINAL_SELECTORS = (
    "model",
    "oracle",
    "equal_average",
    "learned_sparse_mixer",
)


def _load_torch(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _assert_cache_contract(
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
) -> tuple[list[str], int, int]:
    if train_cache.get("split_role") != "router_train":
        raise ValueError("train cache must have split_role='router_train'")
    if val_cache.get("split_role") != "router_val":
        raise ValueError("validation cache must have split_role='router_val'")

    train_names = tuple(train_cache["expert_names"])
    val_names = tuple(val_cache["expert_names"])
    if train_names != val_names:
        raise ValueError(f"expert ordering mismatch: {train_names} != {val_names}")

    num_experts = len(val_names)
    num_windows = int(val_cache["num_windows"])
    assert tuple(val_cache["histories"].shape) == (num_windows, 96, 7)
    assert tuple(val_cache["prediction_stack"].shape) == (num_windows, 12, 7, num_experts)
    assert tuple(val_cache["targets"].shape) == (num_windows, 12, 7)
    assert tuple(val_cache["target_masks"].shape) == (num_windows, 12, 7)
    assert tuple(val_cache["error_matrix"].shape) == (num_windows, num_experts)
    assert tuple(val_cache["mse_matrix"].shape) == (num_windows, num_experts)
    assert tuple(val_cache["best_expert"].shape) == (num_windows,)
    assert torch.equal(
        train_cache["sample_indices"].cpu(),
        torch.arange(int(train_cache["num_windows"]), dtype=train_cache["sample_indices"].dtype),
    )
    assert torch.equal(
        val_cache["sample_indices"].cpu(),
        torch.arange(num_windows, dtype=val_cache["sample_indices"].dtype),
    )
    return list(val_names), num_windows, num_experts


def _load_router_outputs(
    checkpoint: Mapping[str, Any],
    histories: torch.Tensor,
    num_experts: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    router = COSTARTSRouter(**checkpoint["router_config"])
    router.load_state_dict(checkpoint["router_state_dict"])
    router.eval()

    output_chunks: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "map_prediction",
            "ranking_logits",
            "query_logits",
            "mix_weights",
            "stop_logits",
            "query_order",
            "stop_step",
        )
    }
    with torch.no_grad():
        for start in range(0, histories.shape[0], batch_size):
            history = histories[start : start + batch_size].to(torch.float32)
            assert history.shape[1:] == (96, 7)
            outputs = router(history, sampled_rollout=False)
            k = outputs["stop_logits"].shape[-1]
            assert outputs["query_logits"].shape == (history.shape[0], num_experts)
            assert outputs["stop_logits"].shape == (history.shape[0], k)
            assert outputs["query_order"].shape == (history.shape[0], k)
            for key in output_chunks:
                output_chunks[key].append(outputs[key].detach().cpu())
    return {key: torch.cat(chunks, dim=0) for key, chunks in output_chunks.items()}


def _ordered_experts(
    selector: str,
    router_outputs: Mapping[str, torch.Tensor],
    error_matrix: torch.Tensor,
    force_k: int,
) -> tuple[Optional[torch.Tensor], Optional[str]]:
    num_windows, num_experts = error_matrix.shape
    if selector == "query_logits":
        return torch.argsort(router_outputs["query_logits"], dim=-1, descending=True), None
    if selector == "predicted_error":
        top1 = torch.argmax(router_outputs["query_logits"], dim=-1, keepdim=True)
        predicted_order = torch.argsort(router_outputs["map_prediction"], dim=-1, descending=False)
        rows = []
        for row_index in range(num_windows):
            ordered = [int(top1[row_index, 0])]
            for expert_index in predicted_order[row_index].tolist():
                if expert_index not in ordered:
                    ordered.append(int(expert_index))
            rows.append(torch.tensor(ordered, dtype=torch.long))
        return torch.stack(rows, dim=0), None
    if selector == "oracle_improvement":
        first = torch.argmax(router_outputs["query_logits"], dim=-1, keepdim=True)
        oracle_order = torch.argsort(error_matrix, dim=-1, descending=False)
        rows = []
        for row_index in range(num_windows):
            ordered = [int(first[row_index, 0])]
            for expert_index in oracle_order[row_index].tolist():
                if expert_index not in ordered:
                    ordered.append(int(expert_index))
            rows.append(torch.tensor(ordered, dtype=torch.long))
        return torch.stack(rows, dim=0), None
    if selector == "pairwise_reranker":
        return None, "pairwise reranker is not available in the current COSTARTS checkpoint"
    raise ValueError(f"unknown query selector: {selector}")


def _selection_counts(
    expert_names: Sequence[str],
    indices: Optional[torch.Tensor],
    *,
    num_windows: int,
) -> dict[str, int]:
    if indices is None:
        return {name: 0 for name in expert_names}
    if indices.ndim == 1:
        counts = torch.bincount(indices.to(torch.long), minlength=len(expert_names))
    else:
        counts = torch.bincount(indices.reshape(-1).to(torch.long), minlength=len(expert_names))
    return {name: int(counts[index]) for index, name in enumerate(expert_names)}


def _forecast_metrics_for_average(
    prediction_stack: torch.Tensor,
    targets: torch.Tensor,
    target_masks: torch.Tensor,
    queried_indices: torch.Tensor,
) -> tuple[float, float]:
    num_windows, horizon, num_features, _ = prediction_stack.shape
    force_k = queried_indices.shape[1]
    gather_index = queried_indices[:, None, None, :].expand(num_windows, horizon, num_features, force_k)
    queried_predictions = prediction_stack.gather(dim=-1, index=gather_index)
    averaged = queried_predictions.mean(dim=-1)
    mask = target_masks.to(torch.float32)
    denominator = mask.sum().clamp_min(1.0)
    mae = (torch.abs(averaged - targets) * mask).sum() / denominator
    mse = ((averaged - targets).pow(2) * mask).sum() / denominator
    return float(mae), float(mse)


def _row_for_selected_expert(
    *,
    mode: str,
    query_selector: str,
    final_selector: str,
    force_k: Optional[int],
    stop_disabled: bool,
    selected: torch.Tensor,
    stop_steps: torch.Tensor,
    expert_names: Sequence[str],
    error_matrix: torch.Tensor,
    mse_matrix: torch.Tensor,
    oracle_best: torch.Tensor,
    full_oracle_mae: float,
) -> dict[str, Any]:
    selected_mae = error_matrix.gather(1, selected[:, None]).squeeze(1)
    selected_mse = mse_matrix.gather(1, selected[:, None]).squeeze(1)
    stop_counts = torch.bincount(stop_steps.to(torch.long), minlength=max(int(stop_steps.max()), 1) + 1)
    stop_distribution = {
        str(step): int(stop_counts[step])
        for step in range(1, len(stop_counts))
        if int(stop_counts[step]) > 0
    }
    return {
        "status": "ok",
        "mode": mode,
        "query_selector": query_selector,
        "final_selector": final_selector,
        "force_k": force_k,
        "stop_disabled": stop_disabled,
        "mae": float(selected_mae.mean()),
        "mse": float(selected_mse.mean()),
        "oracle_match_rate": float((selected == oracle_best).to(torch.float32).mean()),
        "regret_to_full_oracle": float(selected_mae.mean() - full_oracle_mae),
        "average_experts_used": float(stop_steps.to(torch.float32).mean()),
        "selection_counts": _selection_counts(expert_names, selected, num_windows=selected.shape[0]),
        "stop_step_distribution": stop_distribution,
        "note": "",
    }


def _row_for_equal_average(
    *,
    mode: str,
    query_selector: str,
    final_selector: str,
    force_k: int,
    stop_disabled: bool,
    queried_indices: torch.Tensor,
    expert_names: Sequence[str],
    prediction_stack: torch.Tensor,
    targets: torch.Tensor,
    target_masks: torch.Tensor,
    full_oracle_mae: float,
) -> dict[str, Any]:
    mae, mse = _forecast_metrics_for_average(
        prediction_stack,
        targets,
        target_masks,
        queried_indices,
    )
    return {
        "status": "ok",
        "mode": mode,
        "query_selector": query_selector,
        "final_selector": final_selector,
        "force_k": force_k,
        "stop_disabled": stop_disabled,
        "mae": mae,
        "mse": mse,
        "oracle_match_rate": None,
        "regret_to_full_oracle": mae - full_oracle_mae,
        "average_experts_used": float(force_k),
        "selection_counts": _selection_counts(
            expert_names,
            queried_indices,
            num_windows=queried_indices.shape[0],
        ),
        "stop_step_distribution": {str(force_k): int(queried_indices.shape[0])},
        "note": "selection_counts are utilization counts because forecasts are averaged",
    }


def _skipped_row(
    *,
    mode: str,
    query_selector: str,
    final_selector: str,
    force_k: Optional[int],
    stop_disabled: bool,
    note: str,
) -> dict[str, Any]:
    return {
        "status": "skipped",
        "mode": mode,
        "query_selector": query_selector,
        "final_selector": final_selector,
        "force_k": force_k,
        "stop_disabled": stop_disabled,
        "mae": None,
        "mse": None,
        "oracle_match_rate": None,
        "regret_to_full_oracle": None,
        "average_experts_used": None,
        "selection_counts": {},
        "stop_step_distribution": {},
        "note": note,
    }


def _evaluate_queried_indices(
    *,
    mode: str,
    query_selector: str,
    final_selector: str,
    force_k: int,
    stop_disabled: bool,
    queried_indices: torch.Tensor,
    router_outputs: Mapping[str, torch.Tensor],
    expert_names: Sequence[str],
    val_cache: Mapping[str, torch.Tensor],
    full_oracle_mae: float,
) -> dict[str, Any]:
    error_matrix = val_cache["error_matrix"].to(torch.float32)
    mse_matrix = val_cache["mse_matrix"].to(torch.float32)
    oracle_best = val_cache["best_expert"].to(torch.long)

    if final_selector == "model":
        predicted_errors = router_outputs["map_prediction"].to(torch.float32)
        candidate_errors = predicted_errors.gather(1, queried_indices)
        selected_positions = torch.argmin(candidate_errors, dim=1)
        selected = queried_indices[torch.arange(queried_indices.shape[0]), selected_positions]
        stop_steps = torch.full((queried_indices.shape[0],), force_k, dtype=torch.long)
        return _row_for_selected_expert(
            mode=mode,
            query_selector=query_selector,
            final_selector=final_selector,
            force_k=force_k,
            stop_disabled=stop_disabled,
            selected=selected,
            stop_steps=stop_steps,
            expert_names=expert_names,
            error_matrix=error_matrix,
            mse_matrix=mse_matrix,
            oracle_best=oracle_best,
            full_oracle_mae=full_oracle_mae,
        )
    if final_selector == "oracle":
        true_errors = error_matrix.gather(1, queried_indices)
        selected_positions = torch.argmin(true_errors, dim=1)
        selected = queried_indices[torch.arange(queried_indices.shape[0]), selected_positions]
        stop_steps = torch.full((queried_indices.shape[0],), force_k, dtype=torch.long)
        return _row_for_selected_expert(
            mode=mode,
            query_selector=query_selector,
            final_selector=final_selector,
            force_k=force_k,
            stop_disabled=stop_disabled,
            selected=selected,
            stop_steps=stop_steps,
            expert_names=expert_names,
            error_matrix=error_matrix,
            mse_matrix=mse_matrix,
            oracle_best=oracle_best,
            full_oracle_mae=full_oracle_mae,
        )
    if final_selector == "equal_average":
        return _row_for_equal_average(
            mode=mode,
            query_selector=query_selector,
            final_selector=final_selector,
            force_k=force_k,
            stop_disabled=stop_disabled,
            queried_indices=queried_indices,
            expert_names=expert_names,
            prediction_stack=val_cache["prediction_stack"].to(torch.float32),
            targets=val_cache["targets"].to(torch.float32),
            target_masks=val_cache["target_masks"],
            full_oracle_mae=full_oracle_mae,
        )
    if final_selector == "learned_sparse_mixer":
        return _skipped_row(
            mode=mode,
            query_selector=query_selector,
            final_selector=final_selector,
            force_k=force_k,
            stop_disabled=stop_disabled,
            note="learned sparse queried-subset mixer is not available in the current COSTARTS checkpoint",
        )
    raise ValueError(f"unknown final selector: {final_selector}")


def _current_costarts_row(
    router_outputs: Mapping[str, torch.Tensor],
    expert_names: Sequence[str],
    val_cache: Mapping[str, torch.Tensor],
    full_oracle_mae: float,
) -> dict[str, Any]:
    selected, stop_steps = _select_expert_from_outputs(router_outputs)
    return _row_for_selected_expert(
        mode="current_costarts",
        query_selector="learned_policy_with_stop",
        final_selector="model",
        force_k=None,
        stop_disabled=False,
        selected=selected.to(torch.long),
        stop_steps=stop_steps.to(torch.long),
        expert_names=expert_names,
        error_matrix=val_cache["error_matrix"].to(torch.float32),
        mse_matrix=val_cache["mse_matrix"].to(torch.float32),
        oracle_best=val_cache["best_expert"].to(torch.long),
        full_oracle_mae=full_oracle_mae,
    )


def _baseline_rows(
    expert_names: Sequence[str],
    val_cache: Mapping[str, torch.Tensor],
    full_oracle_mae: float,
) -> list[dict[str, Any]]:
    rows = []
    error_matrix = val_cache["error_matrix"].to(torch.float32)
    mse_matrix = val_cache["mse_matrix"].to(torch.float32)
    oracle_best = val_cache["best_expert"].to(torch.long)
    num_windows, num_experts = error_matrix.shape
    full_oracle_selected = torch.argmin(error_matrix, dim=1)
    rows.append(
        _row_for_selected_expert(
            mode="full_oracle_best_expert_per_window",
            query_selector="oracle_all_experts",
            final_selector="oracle",
            force_k=num_experts,
            stop_disabled=True,
            selected=full_oracle_selected,
            stop_steps=torch.full((num_windows,), num_experts, dtype=torch.long),
            expert_names=expert_names,
            error_matrix=error_matrix,
            mse_matrix=mse_matrix,
            oracle_best=oracle_best,
            full_oracle_mae=full_oracle_mae,
        )
    )
    mean_errors = error_matrix.mean(dim=0)
    best_fixed = int(torch.argmin(mean_errors))
    for expert_index, expert_name in enumerate(expert_names):
        selected = torch.full((num_windows,), expert_index, dtype=torch.long)
        mode = "best_fixed_expert" if expert_index == best_fixed else "fixed_single_expert"
        row = _row_for_selected_expert(
            mode=mode,
            query_selector="fixed",
            final_selector=expert_name,
            force_k=1,
            stop_disabled=True,
            selected=selected,
            stop_steps=torch.ones(num_windows, dtype=torch.long),
            expert_names=expert_names,
            error_matrix=error_matrix,
            mse_matrix=mse_matrix,
            oracle_best=oracle_best,
            full_oracle_mae=full_oracle_mae,
        )
        row["note"] = "actual best fixed expert on the validation cache" if expert_index == best_fixed else ""
        rows.append(row)
    return rows


def evaluate_forced_query_modes(
    *,
    train_cache_path: Path,
    val_cache_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    batch_size: int,
    mode: str,
    force_k: Optional[int],
    query_selector: str,
    final_selector: str,
    include_skipped: bool,
) -> dict[str, Any]:
    train_cache = _load_torch(train_cache_path)
    val_cache = _load_torch(val_cache_path)
    checkpoint = _load_torch(checkpoint_path)
    expert_names, num_windows, num_experts = _assert_cache_contract(train_cache, val_cache)

    if tuple(checkpoint.get("expert_names", expert_names)) != tuple(expert_names):
        raise ValueError("checkpoint expert_names do not match cache expert ordering")

    router_outputs = _load_router_outputs(
        checkpoint,
        val_cache["histories"],
        num_experts,
        batch_size,
    )

    full_oracle_mae = float(val_cache["error_matrix"].to(torch.float32).min(dim=1).values.mean())
    rows: list[dict[str, Any]] = []

    if mode in {"all", "current"}:
        rows.append(_current_costarts_row(router_outputs, expert_names, val_cache, full_oracle_mae))

    if mode in {"all", "baselines"}:
        rows.extend(_baseline_rows(expert_names, val_cache, full_oracle_mae))

    final_selectors = FINAL_SELECTORS if final_selector == "all" else (final_selector,)
    query_selectors = QUERY_SELECTORS if query_selector == "all" else (query_selector,)

    if mode in {"all", "forced"}:
        k_values = range(1, num_experts + 1) if force_k is None else (int(force_k),)
        for selector in query_selectors:
            order, skip_reason = _ordered_experts(
                selector,
                router_outputs,
                val_cache["error_matrix"].to(torch.float32),
                num_experts,
            )
            for k in k_values:
                if k < 1 or k > num_experts:
                    raise ValueError(f"force_k must be between 1 and {num_experts}")
                for selector_name in final_selectors:
                    if skip_reason is not None:
                        row = _skipped_row(
                            mode="forced_exact_k",
                            query_selector=selector,
                            final_selector=selector_name,
                            force_k=k,
                            stop_disabled=True,
                            note=skip_reason,
                        )
                    else:
                        queried = order[:, :k].to(torch.long)
                        row = _evaluate_queried_indices(
                            mode="forced_exact_k",
                            query_selector=selector,
                            final_selector=selector_name,
                            force_k=k,
                            stop_disabled=True,
                            queried_indices=queried,
                            router_outputs=router_outputs,
                            expert_names=expert_names,
                            val_cache=val_cache,
                            full_oracle_mae=full_oracle_mae,
                        )
                    if row["status"] == "ok" or include_skipped:
                        rows.append(row)

    if mode in {"all", "no-stop"}:
        for selector in query_selectors:
            order, skip_reason = _ordered_experts(
                selector,
                router_outputs,
                val_cache["error_matrix"].to(torch.float32),
                num_experts,
            )
            for selector_name in final_selectors:
                if skip_reason is not None:
                    row = _skipped_row(
                        mode="no_stop",
                        query_selector=selector,
                        final_selector=selector_name,
                        force_k=num_experts,
                        stop_disabled=True,
                        note=skip_reason,
                    )
                else:
                    row = _evaluate_queried_indices(
                        mode="no_stop",
                        query_selector=selector,
                        final_selector=selector_name,
                        force_k=num_experts,
                        stop_disabled=True,
                        queried_indices=order[:, :num_experts].to(torch.long),
                        router_outputs=router_outputs,
                        expert_names=expert_names,
                        val_cache=val_cache,
                        full_oracle_mae=full_oracle_mae,
                    )
                if row["status"] == "ok" or include_skipped:
                    rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "train_cache_path": str(train_cache_path),
        "val_cache_path": str(val_cache_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "expert_names": expert_names,
        "num_windows": num_windows,
        "num_experts": num_experts,
        "full_oracle_mae": full_oracle_mae,
        "test_data_used": False,
        "experts_updated": False,
        "shape_assertions": {
            "histories": "[B,96,7]",
            "prediction_stack": "[B,12,7,M]",
            "query_logits": "[B,M]",
            "stop_logits": "[B,K]",
            "query_order": "[B,K]",
        },
    }
    json_payload = {"metadata": metadata, "results": rows}
    json_path = output_dir / "forced_query_results.json"
    json_path.write_text(json.dumps(_jsonable(json_payload), indent=2), encoding="utf-8")

    csv_path = output_dir / "forced_query_results.csv"
    fieldnames = [
        "status",
        "mode",
        "query_selector",
        "final_selector",
        "force_k",
        "stop_disabled",
        "mae",
        "mse",
        "oracle_match_rate",
        "regret_to_full_oracle",
        "average_experts_used",
        "selection_counts",
        "stop_step_distribution",
        "note",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["selection_counts"] = json.dumps(row["selection_counts"], sort_keys=True)
            csv_row["stop_step_distribution"] = json.dumps(row["stop_step_distribution"], sort_keys=True)
            writer.writerow(csv_row)

    return json_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate forced-query and no-stop COSTARTS modes from cached predictions.",
    )
    parser.add_argument("--train-cache", default=DEFAULT_TRAIN_CACHE)
    parser.add_argument("--val-cache", default=DEFAULT_VAL_CACHE)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--mode",
        choices=("all", "current", "baselines", "forced", "no-stop"),
        default="all",
    )
    parser.add_argument("--force-k", type=int, default=None)
    parser.add_argument(
        "--query-selector",
        choices=QUERY_SELECTORS + ("all",),
        default="all",
    )
    parser.add_argument(
        "--final-selector",
        choices=FINAL_SELECTORS + ("all",),
        default="all",
    )
    parser.add_argument("--include-skipped", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_forced_query_modes(
        train_cache_path=Path(args.train_cache),
        val_cache_path=Path(args.val_cache),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        mode=args.mode,
        force_k=args.force_k,
        query_selector=args.query_selector,
        final_selector=args.final_selector,
        include_skipped=args.include_skipped,
    )
    output_dir = Path(args.output_dir)
    print(f"Saved: {output_dir / 'forced_query_results.csv'}")
    print(f"Saved: {output_dir / 'forced_query_results.json'}")
    ok_rows = [row for row in payload["results"] if row["status"] == "ok"]
    print(f"Evaluated rows: {len(ok_rows)}")
    for row in ok_rows[:12]:
        print(
            f"{row['mode']} | query={row['query_selector']} | final={row['final_selector']} "
            f"| K={row['force_k']} | MAE={row['mae']:.6f} | "
            f"avg_used={row['average_experts_used']:.3f}"
        )


if __name__ == "__main__":
    main()
