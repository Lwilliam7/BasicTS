"""Create preregistered final COSTAR-TS model freeze artifacts.

This script intentionally does not load cache tensors or evaluate anything.
It reads only completed non-test experiment reports and writes immutable model
metadata for the later, explicitly authorized test run.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experiments" / "final_test_freeze"


def rel(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/")


LOADED_PATHS: list[dict[str, str]] = []


def assert_not_test_loaded(path: str | Path, purpose: str) -> None:
    """Fail if any loaded/read source path contains 'test'."""
    normalized = rel(path).lower()
    if "test" in normalized:
        raise RuntimeError(f"Refusing to load path containing test for {purpose}: {path}")
    LOADED_PATHS.append({"purpose": purpose, "path": rel(path)})


def read_json_source(path: str | Path) -> dict[str, Any]:
    assert_not_test_loaded(path, "source_report")
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_dataset_freezes() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    timestamp = datetime.now(timezone.utc).isoformat()
    commit = git_head()

    etth1_report_path = "experiments/train_selected_core_etth1/final_report.json"
    etth2_report_path = "experiments/etth2_train_selected_core/final_report.json"
    etth2_canonical_path = "experiments/etth2_canonical_protocol/final_report.json"
    expanded_report_path = "experiments/expanded_expert_pool_costar/final_report.json"

    etth1_report = read_json_source(etth1_report_path)
    etth2_report = read_json_source(etth2_report_path)
    etth2_canonical = read_json_source(etth2_canonical_path)
    expanded_report = read_json_source(expanded_report_path)

    assert etth1_report["main_answers"]["three_experts_selected_from_router_train"] == ["PatchTST", "iTransformer", "TimesNet"]
    assert abs(etth1_report["train_selected_current_best_model"]["mae"] - 0.3631121516227722) < 1e-12
    assert abs(etth1_report["train_selected_current_best_model"]["mse"] - 0.30605703592300415) < 1e-12
    assert etth2_report["main_answers"]["three_experts_selected_from_router_train"] == ["DLinear", "PatchTST", "ModernTCN"]
    assert abs(etth2_report["main_answers"]["full_model_router_val_mae"] - 0.27683213353157043) < 1e-12

    etth1_cache_paths = {
        "router_train": "cache/costarts_walkforward/router_train_20_60_cache.pt",
        "router_validation": "cache/costarts_walkforward/router_val_60_80_cache.pt",
        "normalizer_checkpoint": "checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt",
    }
    etth2_cache_paths = {
        "router_train": "cache/costarts_fresh/ETTh2_96_12/router_train_cache.pt",
        "router_validation": "cache/costarts_fresh/ETTh2_96_12/router_val_cache.pt",
    }
    for purpose, path in {**etth1_cache_paths, **etth2_cache_paths}.items():
        assert_not_test_loaded(path, f"frozen_cache_path:{purpose}")

    expert_index_map = {
        "DLinear": 0,
        "PatchTST": 1,
        "iTransformer": 2,
        "TimesNet": 3,
        "ModernTCN": 4,
    }

    shared_hyperparameters = {
        "architecture": "hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75",
        "chronological_component": {
            "family": "EMA",
            "decay": 0.97,
            "temperature": 0.1,
            "online_blend_alpha": 0.5,
        },
        "horizon_variable_component": {
            "mode": "hv_lowrank",
            "rank": 1,
            "decay": 0.95,
            "temperature": 0.1,
            "alpha": 0.75,
        },
        "chrono_hv_blend": {
            "chrono_weight": 0.25,
            "hv_weight": 0.75,
        },
        "specialist_configuration": {
            "name": "both_variable_decay0.95_cap0.1_marginbp200_warm96",
            "specialists": ["DLinear", "ModernTCN"],
            "structure": "variable",
            "decay": 0.95,
            "cap": 0.1,
            "margin": 0.02,
            "margin_basis_points": 200,
            "warmup": 96,
        },
    }

    etth1 = {
        "dataset": "ETTh1",
        "model_frozen": True,
        "validation_tuning_complete": True,
        "test_loaded": False,
        "test_metrics_seen": False,
        "frozen_development_result": True,
        "selected_core_experts": ["PatchTST", "iTransformer", "TimesNet"],
        "how_core_was_selected": "router_train-only chronological OOF selection over all 10 fixed-three subsets",
        "core_expert_indices": [1, 2, 3],
        "expert_index_map": expert_index_map,
        "adaptive_weighting_parameters": shared_hyperparameters,
        "effective_specialist_handling": {
            "DLinear": "enabled_as_optional_specialist",
            "ModernTCN": "enabled_as_optional_specialist",
            "duplicate_specialists_disabled_if_in_core": True,
        },
        "random_seeds": [7, 11, 13, 17, 19],
        "validation": {
            "mae": 0.3631121516227722,
            "mse": 0.30605703592300415,
            "mae_std": etth1_report["train_selected_current_best_model"]["mae_std"],
            "mse_std": etth1_report["train_selected_current_best_model"]["mse_std"],
            "marked_final_frozen_development_mae": 0.363112,
            "marked_final_frozen_development_mse": 0.306057,
        },
        "cache_paths": etth1_cache_paths,
        "cache_hashes_from_prior_reports": {
            "router_train_sha256": etth1_report["phase_a_frozen_config"]["cache_hashes"]["router_train_sha256"],
            "router_validation_sha256": "7d8e9d98603a8392ca43eac86c99970f6fe1734f3ee29b50ffd2ce8b540cc2a6",
            "normalizer_sha256": etth1_report["phase_a_frozen_config"]["cache_hashes"]["normalizer_sha256"],
            "hash_note": "Hashes reused from completed non-test reports; no cache tensor was loaded during this freeze.",
        },
        "relevant_checkpoint_paths": {
            "DLinear": "checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt",
            "PatchTST": "checkpoints/costarts_walkforward/final_60/PatchTST/best_expert.pt",
            "iTransformer": "checkpoints/costarts_walkforward/final_60/iTransformer/best_expert.pt",
            "TimesNet": "checkpoints/costarts_walkforward/final_60/TimesNet/best_expert.pt",
            "ModernTCN": "checkpoints/costarts_walkforward/final_60/ModernTCN/best_expert.pt",
        },
        "code_scripts_used": {
            "core_selection_and_clean_validation": "experiments/train_selected_core_etth1/run_train_selected_core_eval.py",
            "specialist_source_experiment": "experiments/expanded_expert_pool_costar/run_expanded_expert_pool.py",
            "freeze_script": "experiments/final_test_freeze/create_final_model_freeze.py",
        },
        "source_reports": [etth1_report_path, expanded_report_path],
        "git_commit_hash": commit,
        "timestamp_utc": timestamp,
    }

    etth2 = {
        "dataset": "ETTh2",
        "model_frozen": True,
        "validation_tuning_complete": True,
        "test_loaded": False,
        "test_metrics_seen": False,
        "selected_core_experts": ["DLinear", "PatchTST", "ModernTCN"],
        "how_core_was_selected": "router_train-only chronological OOF selection over all 10 fixed-three subsets",
        "core_expert_indices": [0, 1, 4],
        "expert_index_map": expert_index_map,
        "adaptive_weighting_parameters": {
            **shared_hyperparameters,
            "chronological_component": {
                **shared_hyperparameters["chronological_component"],
                "static_prior": "equal_weights_no_etth2_static_neural_artifact",
            },
        },
        "effective_specialist_handling": {
            "DLinear": "disabled_as_duplicate_because_in_core",
            "ModernTCN": "disabled_as_duplicate_because_in_core",
            "duplicate_specialists_disabled_if_in_core": True,
        },
        "random_seeds": [],
        "random_seed_note": "ETTh2 frozen adaptive evaluation is deterministic in the clean script.",
        "validation": {
            "core_mae": 0.2808783948421478,
            "core_mse": 0.17193281650543213,
            "full_model_mae": 0.27683213353157043,
            "full_model_mse": 0.16727977991104126,
        },
        "validation_selected_reference_baselines_not_primary_model": {
            "DLinear+ModernTCN": {
                "mae": 0.2752290368080139,
                "mse": 0.1653451770544052,
                "selection_status": "validation-selected reference only, not frozen primary model",
            },
            "DLinear+TimesNet+ModernTCN": {
                "mae": 0.27664363384246826,
                "mse": 0.16693221032619476,
                "selection_status": "validation-selected fixed-3 reference only, not frozen primary model",
            },
        },
        "cache_paths": etth2_cache_paths,
        "cache_hashes_from_prior_reports": {
            "router_train_sha256": etth2_report["phase_a_frozen_config"]["cache_hashes"]["router_train_sha256"],
            "router_validation_sha256": etth2_canonical["protocol"]["val_cache_sha256"],
            "hash_note": "Hashes reused from completed non-test reports; no cache tensor was loaded during this freeze.",
        },
        "relevant_checkpoint_paths": {
            "DLinear": "checkpoints/costarts_fresh/ETTh2_96_12/clean_candidates/best_dlinear.pt",
            "PatchTST": "checkpoints/costarts_fresh/ETTh2_96_12/clean_candidates/best_patchtst.pt",
            "ModernTCN": "checkpoints/costarts_fresh/ETTh2_96_12/clean_candidates/best_moderntcn.pt",
        },
        "code_scripts_used": {
            "core_selection_and_clean_validation": "experiments/etth2_train_selected_core/run_etth2_train_selected_core_eval.py",
            "canonical_reference_baselines": "experiments/etth2_canonical_protocol/run_canonical_etth2_baselines.py",
            "freeze_script": "experiments/final_test_freeze/create_final_model_freeze.py",
        },
        "source_reports": [etth2_report_path, etth2_canonical_path],
        "git_commit_hash": commit,
        "timestamp_utc": timestamp,
    }

    freeze = {
        "freeze_name": "FINAL_COSTAR_TS_ETTh1_ETTh2_PRETEST_FREEZE",
        "timestamp_utc": timestamp,
        "git_commit_hash": commit,
        "validation_tuning_complete": True,
        "model_frozen": True,
        "test_loaded": False,
        "test_metrics_seen": False,
        "no_further_validation_tuning_allowed": True,
        "hard_rules": {
            "do_not_load_test_cache": True,
            "do_not_evaluate_test": True,
            "do_not_change_expert_identities": True,
            "do_not_change_subset_size": True,
            "do_not_change_decay": True,
            "do_not_change_temperature": True,
            "do_not_change_alpha": True,
            "do_not_change_blend_ratios": True,
            "do_not_change_specialist_cap": True,
            "do_not_change_margin": True,
            "do_not_change_warmup": True,
            "do_not_change_features": True,
            "do_not_run_another_validation_search": True,
        },
        "path_guard": {
            "assertion": "Any loaded/read source path containing 'test' raises RuntimeError.",
            "loaded_paths_checked": LOADED_PATHS,
            "test_cache_loaded": False,
        },
        "datasets": {
            "ETTh1": etth1,
            "ETTh2": etth2,
        },
        "unresolved_reproduction_issues": [],
    }
    return etth1, etth2, freeze


def build_report(etth1: dict[str, Any], etth2: dict[str, Any], freeze: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Final COSTAR-TS Pre-Test Model Freeze",
            "",
            "This is a preregistered snapshot created before any test cache was loaded.",
            "",
            "## ETTh1",
            "",
            f"- Core: `{'+'.join(etth1['selected_core_experts'])}`",
            "- Core selection: router_train only",
            f"- Model: `{etth1['adaptive_weighting_parameters']['architecture']}`",
            f"- Specialists: `{'+'.join(etth1['adaptive_weighting_parameters']['specialist_configuration']['specialists'])}`",
            f"- Specialist config: `{etth1['adaptive_weighting_parameters']['specialist_configuration']['name']}`",
            f"- Frozen validation MAE/MSE: `{etth1['validation']['marked_final_frozen_development_mae']:.6f}` / `{etth1['validation']['marked_final_frozen_development_mse']:.6f}`",
            "",
            "## ETTh2",
            "",
            f"- Core: `{'+'.join(etth2['selected_core_experts'])}`",
            "- Core selection: router_train only",
            f"- Model: `{etth2['adaptive_weighting_parameters']['architecture']}`",
            f"- Core validation MAE/MSE: `{etth2['validation']['core_mae']:.6f}` / `{etth2['validation']['core_mse']:.6f}`",
            f"- Full frozen adaptive validation MAE/MSE: `{etth2['validation']['full_model_mae']:.6f}` / `{etth2['validation']['full_model_mse']:.6f}`",
            "- Validation-selected `DLinear+ModernTCN` is retained only as a reference baseline.",
            "",
            "## Frozen Hyperparameters",
            "",
            "- Chronological EMA decay `0.97`, temperature `0.1`, online blend alpha `0.5`.",
            "- Horizon-variable low-rank rank `1`, decay `0.95`, temperature `0.1`, alpha `0.75`.",
            "- Chrono/HV blend: chrono `0.25`, HV `0.75`.",
            "- Specialist config: variable decay `0.95`, cap `0.1`, margin `0.02`, warmup `96`.",
            "",
            "## Freeze Status",
            "",
            f"- Validation tuning complete: `{freeze['validation_tuning_complete']}`",
            f"- Model frozen: `{freeze['model_frozen']}`",
            f"- Test loaded: `{freeze['test_loaded']}`",
            f"- Test metrics seen: `{freeze['test_metrics_seen']}`",
            f"- Git commit: `{freeze['git_commit_hash']}`",
            f"- Timestamp UTC: `{freeze['timestamp_utc']}`",
            "",
            "## Artifacts",
            "",
            "- `experiments/final_test_freeze/ETTh1_frozen_model.json`",
            "- `experiments/final_test_freeze/ETTh2_frozen_model.json`",
            "- `experiments/final_test_freeze/FINAL_MODEL_FREEZE.json`",
            "- `experiments/final_test_freeze/freeze_report.md`",
            "",
        ]
    )


def main() -> None:
    etth1, etth2, freeze = build_dataset_freezes()
    write_json(OUT_DIR / "ETTh1_frozen_model.json", etth1)
    write_json(OUT_DIR / "ETTh2_frozen_model.json", etth2)
    write_json(OUT_DIR / "FINAL_MODEL_FREEZE.json", freeze)
    (OUT_DIR / "freeze_report.md").write_text(build_report(etth1, etth2, freeze), encoding="utf-8")
    print(json.dumps({"wrote": [rel(p) for p in sorted(OUT_DIR.glob("*"))], "test_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
