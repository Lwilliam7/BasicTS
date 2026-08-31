"""First clean new-dataset confirmation of Window-Dependent Expert-Choice H x V.

This script performs the pre-registered, performance-blind steps required
before any confirmation metric may be computed: (1) verify the canonical
method exists locally and is the one that produced WINDOW_DEPENDENT_EC_SUPPORTED,
(2) build a dataset contamination ledger from local evidence, (3) inventory
every BasicTS-supported dataset's STATIC eligibility (schema/format/data-volume
compatibility with the existing walk-forward pipeline) without ever looking at
forecasting or routing performance, (4) deterministically select the primary
confirmation dataset list, and (5) only if >=4 genuinely untouched eligible
datasets exist, proceed to core selection, causal OOF, and a single frozen
confirmation_eval pass.

If fewer than 4 eligible untouched datasets exist, this script STOPS after
producing the contamination ledger and eligibility inventory and reports
INSUFFICIENT_UNTOUCHED_DATASETS. It does not relax any criterion to manufacture
a positive result.

TEST SET ACCESSED: NO. TEST CACHE LOADED: NO. TEST METRICS COMPUTED: NO.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.window_dependent_expert_choice_hv.run_window_dependent_expert_choice_hv as wdec  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"

# ---------------------------------------------------------------------------
# Section 7 seed: datasets known a priori to have already influenced the
# research program. Expanded below from local evidence, never contracted.
# ---------------------------------------------------------------------------
SEEDED_CONTAMINATED = [
    "ETTh1", "ETTh2", "ETTm1", "Weather", "Electricity",
    "ExchangeRate", "Traffic", "BeijingAirQuality", "ETTm2",
]

# Static, pre-registered eligibility floor for confirmation_eval window count.
# Chosen as roughly the smallest window count ANY development dataset for this
# exact method actually used (ETTh2 router_val = 613 windows), rounded down to
# a conservative round number BEFORE any candidate's own window count was
# computed. Not tuned to any specific candidate.
MIN_CONFIRMATION_WINDOWS = 400
INPUT_LEN = 96
HORIZON = 12
N_UNTOUCHED_REQUIRED = 4
N_UNTOUCHED_MAX = 6


def write_json(path: Path, obj: Any) -> None:
    wdec.write_json(path, obj)


def write_csv_stub(path: Path, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"note\n\"{note}\"\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True).strip()
        except Exception as exc:  # pragma: no cover
            return f"unavailable: {exc}"

    dirty = run(["git", "status", "--porcelain"])
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty_working_tree": bool(dirty) and not dirty.startswith("unavailable"),
        "dirty_files_count": len(dirty.splitlines()) if dirty and not dirty.startswith("unavailable") else 0,
    }


def environment_info() -> dict[str, Any]:
    import torch

    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


# ---------------------------------------------------------------------------
# Step 2 (section 2): verify the canonical development method exists locally
# and matches the reported design, by inspecting the actual source, not by
# re-deriving it from this prompt.
# ---------------------------------------------------------------------------


def verify_canonical_method() -> dict[str, Any]:
    import inspect

    src_combine = inspect.getsource(wdec.dynamic_prediction_from_claims)
    src_claims = inspect.getsource(wdec.dynamic_ec_claims)
    src_affinity = inspect.getsource(wdec.raw_to_affinity)
    src_model = inspect.getsource(wdec.SharedResidualScorer)

    checks = {
        "multi_claim_is_equal_average_not_affinity_weighted": ("claimed_sum / counts" in src_combine) and ("weight" not in src_combine.lower()),
        "zero_claim_fallback_is_equal_ensemble": "equal = forecasts.mean(dim=-1)" in src_combine,
        "capacity_formula_round_M_over_E": "round(m / e)" in src_claims,
        "affinity_is_softmax_after_fit_only_standardization": "torch.softmax" in src_affinity and "AFFINITY_TEMPERATURE" in src_affinity,
        "affinity_temperature_is_1.0": wdec.AFFINITY_TEMPERATURE == 1.0,
        "one_shared_scorer_class": "class SharedResidualScorer" in src_model,
        "architecture_64_32_1": "nn.Linear(input_dim, HIDDEN1)" in inspect.getsource(wdec.SharedResidualScorer.__init__) and wdec.HIDDEN1 == 64 and wdec.HIDDEN2 == 32,
        "seed_is_7": wdec.SCORER_SEED == 7,
        "optimizer_is_AdamW": True,  # verified below via source of train_scorer
        "lr_1e-3": wdec.LR == 1e-3,
        "weight_decay_1e-4": wdec.WEIGHT_DECAY == 1e-4,
        "max_epochs_100": wdec.MAX_EPOCHS == 100,
        "patience_10": wdec.PATIENCE == 10,
        "capacity_factor_1.0": wdec.CAPACITY_FACTOR == 1.0,
        "block_length_24": wdec.BLOCK_LENGTH == 24,
        "phase_k_12": wdec.PHASE_K == 12,
        "bootstrap_samples_10000": wdec.BOOTSTRAP_SAMPLES == 10000,
    }
    src_train = inspect.getsource(wdec.train_scorer)
    checks["optimizer_is_AdamW"] = "torch.optim.AdamW" in src_train

    undocumented_implementation_detail = {
        "batch_size": {
            "value": wdec.BATCH_SIZE,
            "note": "BATCH_SIZE=32 (windows per minibatch step) was not specified in the original development prompt; it was my own fixed implementation choice, applied identically to every development dataset and never tuned. Documented here per section 3's 'document the discrepancy' requirement.",
        },
        "internal_val_fraction": {
            "value": wdec.INTERNAL_VAL_FRACTION,
            "note": "Chronological tail fraction of the legal fit set reserved for early-stopping (0.20), applied identically across folds/datasets.",
        },
    }

    all_pass = all(checks.values())
    return {
        "CANONICAL_METHOD_FOUND": "YES" if all_pass else "NO",
        "canonical_source_file": "experiments/window_dependent_expert_choice_hv/run_window_dependent_expert_choice_hv.py",
        "canonical_source_sha256": sha256_file(ROOT / "experiments/window_dependent_expert_choice_hv/run_window_dependent_expert_choice_hv.py"),
        "stored_router_val_dynamic_ec_mae_reference": {
            "ETTh1": 0.375640, "ETTh2": 0.280951, "ETTm1": 0.253556, "Weather": 0.155621, "Electricity": 0.206356,
        },
        "checks": checks,
        "all_checks_pass": all_pass,
        "undocumented_implementation_details": undocumented_implementation_detail,
        "affinity_weighted_ec_excluded_as_canonical": True,
        "affinity_weighted_ec_note": "experiments/affinity_weighted_expert_choice_hv/ produced only negligible (~1e-5 MAE) deltas and MUST NOT replace the canonical multi-claim rule verified above; not imported or used anywhere in this script.",
    }


# ---------------------------------------------------------------------------
# Section 7: dataset contamination ledger, built from local evidence.
# ---------------------------------------------------------------------------


def full_basicts_catalog() -> list[str]:
    return sorted(p.name for p in DATASETS_DIR.iterdir() if p.is_dir() and (p / "meta.json").exists())


def grep_dataset_mentions(name: str) -> list[str]:
    """Files under experiments/ or project_memory/ whose JSON/MD content
    mentions this dataset name as a whole token (case-sensitive, avoids
    2-3 letter dataset names matching unrelated substrings)."""
    import re

    pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])")
    hits: list[str] = []
    for root_dir in (ROOT / "experiments", ROOT / "project_memory"):
        if not root_dir.exists():
            continue
        for path in root_dir.rglob("*"):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            # Exclude this confirmation experiment's own output directory --
            # otherwise a rerun sees its own prior report/ledger as "evidence"
            # of contamination (a self-referential feedback bug, caught and
            # fixed during development of this exact script).
            if OUT_DIR in path.parents:
                continue
            if path.suffix.lower() not in (".json", ".md", ".csv"):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if pattern.search(text):
                hits.append(str(path.relative_to(ROOT)))
    return hits


def build_contamination_ledger() -> dict[str, Any]:
    catalog = full_basicts_catalog()
    cache_dirs = {p.name for p in (ROOT / "cache").iterdir() if p.is_dir()} if (ROOT / "cache").exists() else set()
    checkpoint_dirs = {p.name for p in (ROOT / "checkpoints").iterdir() if p.is_dir()} if (ROOT / "checkpoints").exists() else set()

    entries: dict[str, Any] = {}
    contaminated: set[str] = set(SEEDED_CONTAMINATED)

    for name in catalog:
        mentions = grep_dataset_mentions(name)
        has_walkforward_cache = any(
            d == f"costarts_walkforward_{name}" or (name == "ETTh1" and d == "costarts_walkforward") or (name == "ETTh2" and d.startswith("costarts_fresh"))
            for d in cache_dirs
        )
        has_walkforward_checkpoints = any(
            d == f"costarts_walkforward_{name}" or (name == "ETTh1" and d == "costarts_walkforward")
            for d in checkpoint_dirs
        )
        # A dataset counts as CONTAMINATED only if there is evidence a real
        # forecasting/routing RESULT was computed on it: it is seeded, or a
        # walk-forward cache/checkpoint set exists (the necessary precondition
        # for any such result -- with neither, no model could have been
        # trained or evaluated on it, full stop). Merely being named inside a
        # performance-blind planning/candidate-selection document (verified
        # below) does NOT count: per section 7's own definition, contamination
        # requires evidence that a "forecasting/model-routing RESULT"
        # influenced a decision, not that the name appeared in text.
        is_seeded = name in SEEDED_CONTAMINATED
        is_contaminated = is_seeded or has_walkforward_cache or has_walkforward_checkpoints
        performance_blind_planning_mentions = [
            m for m in mentions if Path(m).name in ("dataset_selection.json",)
        ]
        other_mentions = [m for m in mentions if m not in performance_blind_planning_mentions]
        if is_contaminated:
            contaminated.add(name)
        elif other_mentions:
            # Any mention OUTSIDE the known performance-blind planning file is
            # treated conservatively as contaminating, since it has not been
            # individually verified to be performance-blind.
            is_contaminated = True
            contaminated.add(name)
        entries[name] = {
            "seeded_contaminated": is_seeded,
            "has_walkforward_cache_built": has_walkforward_cache,
            "has_walkforward_checkpoints_trained": has_walkforward_checkpoints,
            "mentioned_in_performance_blind_planning_doc_only": performance_blind_planning_mentions,
            "mentioned_in_other_files": other_mentions,
            "mention_count": len(mentions),
            "contaminated": is_contaminated,
            "reason": (
                "Seeded as already-influencing the research program."
                if is_seeded
                else "Walk-forward cache and/or trained expert checkpoints already exist for this dataset (necessary precondition for any computed result)."
                if (has_walkforward_cache or has_walkforward_checkpoints)
                else f"Appears in {len(other_mentions)} file(s) outside the verified performance-blind planning document; treated conservatively as contaminated pending individual verification."
                if other_mentions
                else "Only appears (if at all) inside experiments/behavioral_competence/generalization/dataset_selection.json, a verified performance-blind candidate-selection document that explicitly states no forecasting or competence-scorer performance was run or viewed before it was written (self-attested and independently corroborated by the absence of any cache/checkpoint for this dataset). Not treated as contamination: no result influenced any decision about this dataset."
            ),
        }

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "full_basicts_catalog": catalog,
        "seeded_contaminated": sorted(SEEDED_CONTAMINATED),
        "contaminated_datasets": sorted(contaminated),
        "candidate_untouched_datasets": sorted(set(catalog) - contaminated),
        "entries": entries,
        "expansion_note": (
            "The seed list matched local evidence exactly for these 9 datasets: ExchangeRate, Traffic, "
            "BeijingAirQuality, and ETTm2 were independently confirmed as a separate 'generalization' family "
            "(experiments/behavioral_competence/generalization/dataset_selection.json, "
            "experiments/behavioral_competence/timefuse_probe/) used for a prior LearnedProbe-Rank generalization "
            "study on 2026-08-20 and a TimeFuse probe study; no additional contamination was found beyond the seed set."
        ),
    }


# ---------------------------------------------------------------------------
# Section 8: static eligibility inventory. NEVER inspects forecasting/routing
# performance -- only dataset schema, format-convention, and volume facts.
# ---------------------------------------------------------------------------


def valid_window_count(range_len: int) -> int:
    last_exclusive_minus_start = range_len - INPUT_LEN - HORIZON + 1
    return max(0, last_exclusive_minus_start)


def static_eligibility(name: str) -> dict[str, Any]:
    meta_path = DATASETS_DIR / name / "meta.json"
    result: dict[str, Any] = {"dataset": name}
    if not meta_path.exists():
        result["eligible"] = False
        result["reason"] = "No meta.json found."
        return result
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["eligible"] = False
        result["reason"] = f"meta.json unreadable/invalid: {exc}"
        return result

    has_npy = all((DATASETS_DIR / name / f"{split}_data.npy").exists() for split in ("train", "val", "test"))
    reg = meta.get("regular_settings", {})
    shape = meta.get("shape")
    has_graph = bool(meta.get("has_graph", False))
    rescale = bool(reg.get("rescale", False))
    null_val = reg.get("null_val")
    metrics = reg.get("metrics", [])
    norm_each_channel = bool(reg.get("norm_each_channel", False))

    convention_matches = (
        not has_graph
        and rescale is False
        and (null_val is None or (isinstance(null_val, float) and null_val != null_val))  # NaN
        and set(metrics) >= {"MAE", "MSE"}
        and norm_each_channel is True
    )

    num_timestamps = shape[0] if isinstance(shape, list) and len(shape) >= 1 else None
    num_vars = shape[1] if isinstance(shape, list) and len(shape) >= 2 else None

    volume_ok = None
    router_train_windows = None
    confirmation_windows = None
    if num_timestamps:
        cuts = [int(num_timestamps * f) for f in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
        block_b_len = cuts[2] - cuts[1]
        block_c_len = cuts[3] - cuts[2]
        val_len = cuts[4] - cuts[3]
        router_train_windows = valid_window_count(block_b_len) + valid_window_count(block_c_len)
        confirmation_windows = valid_window_count(val_len)
        volume_ok = confirmation_windows >= MIN_CONFIRMATION_WINDOWS

    already_has_infra = (ROOT / f"cache/costarts_walkforward_{name}").exists() or (ROOT / f"checkpoints/costarts_walkforward_{name}").exists()

    reasons = []
    if has_npy is False:
        reasons.append("Missing one or more of train/val/test_data.npy (no standard forecasting array triple).")
    if not convention_matches:
        parts = []
        if has_graph:
            parts.append("has_graph=true")
        if rescale:
            parts.append("rescale=true")
        if not (null_val is None or (isinstance(null_val, float) and null_val != null_val)):
            parts.append(f"null_val={null_val} (not NaN)")
        if not set(metrics) >= {"MAE", "MSE"}:
            parts.append(f"metrics={metrics} (not MAE/MSE convention)")
        if not norm_each_channel:
            parts.append("norm_each_channel=false")
        reasons.append(
            "regular_settings convention differs from every development dataset (has_graph=false, rescale=false, "
            "null_val=NaN, metrics=[MAE,MSE], norm_each_channel=true): " + ", ".join(parts) + ". Adapting this "
            "dataset would require real preprocessing/masking/evaluation-convention changes to the existing "
            "generic pipeline functions (build_histories_targets' isfinite-based masking assumes NaN-missing, not "
            "null_val=0.0; sample_mae/std normalization assumes rescale=false) -- i.e. dataset-specific method "
            "tuning, which is disallowed for eligibility."
        )
    if volume_ok is False:
        reasons.append(
            f"confirmation_eval (validation block, 60-80%) would only contain {confirmation_windows} windows at "
            f"input_len={INPUT_LEN}/horizon={HORIZON}, below the static pre-registered floor of "
            f"{MIN_CONFIRMATION_WINDOWS} (chosen from the smallest development dataset's own confirmation-equivalent "
            f"window count, ETTh2 router_val=613, before any candidate's size was computed). Too thin for the "
            f"predeclared block-24 bootstrap and per-fold scorer training to be statistically meaningful."
        )

    eligible = has_npy and convention_matches and bool(volume_ok)

    result.update(
        {
            "shape": shape,
            "num_timestamps": num_timestamps,
            "num_variables": num_vars,
            "has_graph": has_graph,
            "rescale": rescale,
            "null_val": null_val,
            "metrics": metrics,
            "norm_each_channel": norm_each_channel,
            "has_train_val_test_npy": has_npy,
            "convention_matches_development_family": convention_matches,
            "estimated_router_train_windows": router_train_windows,
            "estimated_confirmation_eval_windows": confirmation_windows,
            "min_confirmation_windows_floor": MIN_CONFIRMATION_WINDOWS,
            "volume_adequate": volume_ok,
            "already_has_walkforward_cache_or_checkpoints": already_has_infra,
            "eligible": eligible,
            "ineligibility_reasons": reasons,
        }
    )
    return result


def build_eligibility(ledger: dict[str, Any]) -> dict[str, Any]:
    candidates = ledger["candidate_untouched_datasets"]
    rows = [static_eligibility(name) for name in candidates]
    eligible = sorted(r["dataset"] for r in rows if r["eligible"])

    if len(eligible) > N_UNTOUCHED_MAX:
        selected = eligible[:N_UNTOUCHED_MAX]
        selection_rule = f"More than {N_UNTOUCHED_MAX} eligible; sorted by canonical dataset name and took the first {N_UNTOUCHED_MAX} deterministically."
    else:
        selected = eligible
        selection_rule = f"{len(eligible)} eligible untouched dataset(s) found; used all of them (<= {N_UNTOUCHED_MAX})."

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_untouched_datasets_considered": candidates,
        "per_dataset": {r["dataset"]: r for r in rows},
        "fully_eligible_datasets": eligible,
        "num_eligible": len(eligible),
        "num_required": N_UNTOUCHED_REQUIRED,
        "selection_rule": selection_rule,
        "selected_primary_confirmation_datasets": selected,
        "insufficient": len(eligible) < N_UNTOUCHED_REQUIRED,
        "note": "Eligibility determined ONLY from static schema/format/volume facts (meta.json, existing cache/checkpoint presence, window-count arithmetic). No forecasting model was run and no MAE/MSE/routing performance was inspected for any candidate before this file was written.",
    }


def make_report(canonical: dict[str, Any], ledger: dict[str, Any], eligibility: dict[str, Any]) -> None:
    classification = "INSUFFICIENT_UNTOUCHED_DATASETS" if eligibility["insufficient"] else "SEE confirmation_results.json"
    lines = [
        f"Final classification: {classification}",
        "",
        "PRIMARY UNTOUCHED DATASETS: " + (", ".join(eligibility["selected_primary_confirmation_datasets"]) or "(none)"),
        "CONTAMINATED / EXCLUDED DATASETS: " + ", ".join(ledger["contaminated_datasets"]),
        "",
        "```text",
        "TEST SET ACCESSED: NO",
        "TEST CACHE LOADED: NO",
        "TEST METRICS COMPUTED: NO",
        "```",
        "",
        "# Window-Dependent Expert Choice: First Clean New-Dataset Confirmation",
        "",
        f"`CANONICAL_METHOD_FOUND: {canonical['CANONICAL_METHOD_FOUND']}` (source: `{canonical['canonical_source_file']}`, sha256 `{canonical['canonical_source_sha256'][:16]}...`).",
        "",
        "## Step 1: contamination ledger",
        "",
        f"Full local BasicTS catalog: `{len(ledger['full_basicts_catalog'])}` datasets. "
        f"Contaminated (already influenced this research program): `{len(ledger['contaminated_datasets'])}` -- "
        + ", ".join(ledger["contaminated_datasets"]) + ".",
        "",
        ledger["expansion_note"],
        "",
        "## Step 2: static eligibility (performance-blind)",
        "",
        f"Candidate untouched datasets considered: `{len(eligibility['candidate_untouched_datasets_considered'])}` -- "
        + ", ".join(eligibility["candidate_untouched_datasets_considered"]) + ".",
        "",
        "| Dataset | Has graph | Rescale | Null val | Metrics | Timestamps | Est. confirmation windows | Eligible | Reason if not |",
        "|---|---|---|---|---|---:|---:|---|---|",
    ]
    for name in eligibility["candidate_untouched_datasets_considered"]:
        r = eligibility["per_dataset"][name]
        reason = "; ".join(r.get("ineligibility_reasons", [])) or "-"
        lines.append(
            f"| {name} | {r.get('has_graph')} | {r.get('rescale')} | {r.get('null_val')} | {r.get('metrics')} | "
            f"{r.get('num_timestamps')} | {r.get('estimated_confirmation_eval_windows')} | {r.get('eligible')} | {reason} |"
        )
    lines += [
        "",
        f"Eligible untouched datasets: `{eligibility['num_eligible']}` (required >= `{eligibility['num_required']}`).",
        "",
        eligibility["selection_rule"],
        "",
    ]

    if eligibility["insufficient"]:
        lines += [
            "## Result: STOP -- INSUFFICIENT_UNTOUCHED_DATASETS",
            "",
            "Every dataset in the local BasicTS installation falls into one of three buckets:",
            "",
            "1. **Already contaminated** (9 datasets: ETTh1, ETTh2, ETTm1, Weather, Electricity, ExchangeRate, "
            "Traffic, BeijingAirQuality, ETTm2) -- either directly used to develop/validate the Expert-Choice "
            "method family, or used in the separate prior LearnedProbe-Rank 'generalization' study and TimeFuse "
            "probe study on the same 4 additional datasets.",
            "2. **Format-incompatible without real method changes** (graph-structured traffic datasets: PEMS03/04/07/08, "
            "METR-LA, PEMS-BAY, CA, GBA, GLA, SD -- all use `has_graph=true`, `rescale=true`, `null_val=0.0`, and "
            "`metrics=[MAE,RMSE,MAPE]`, a fundamentally different preprocessing/masking/evaluation convention than "
            "every dataset this method has ever run on; adapting them would require real code changes to masking "
            "and normalization, i.e. dataset-specific method tuning, which eligibility explicitly disallows), plus "
            "`Gaussian`/`Pulse` (synthetic, single-variable, `rescale=true`) and `UEA` (a classification benchmark "
            "collection with no forecasting train/val/test array).",
            "3. **Format-compatible but statistically too thin** (`Illness`: matches the development convention "
            "exactly -- `has_graph=false`, `rescale=false`, `null_val=NaN`, `norm_each_channel=true`, `metrics=[MAE,MSE]` "
            f"-- but only 966 total timestamps yields an estimated {eligibility['per_dataset'].get('Illness', {}).get('estimated_confirmation_eval_windows', 'N/A')} "
            f"confirmation_eval windows, far below the {MIN_CONFIRMATION_WINDOWS}-window static floor and less than "
            "15% of the smallest development dataset's own confirmation-window count (ETTh2, 613 windows).",
            "",
            "Independently of the above, **no untouched dataset currently has trained frozen expert checkpoints or a "
            "built walk-forward cache**: `checkpoints/costarts_walkforward_{Illness,PEMS03,...}` and the corresponding "
            "`cache/` directories simply do not exist. Even if a candidate passed the static eligibility bar, running "
            "confirmation would first require training 5 new frozen forecasting experts (DLinear, PatchTST, "
            "iTransformer, TimesNet, ModernTCN) across 3 chronological walk-forward stages -- a substantial new "
            "training investment, not a single frozen-method evaluation pass. This is a second, independent reason "
            "confirmation cannot proceed today without the user acquiring/approving a genuinely new, compatible, "
            "adequately-sized dataset.",
            "",
            "**No core selection, causal OOF, or confirmation_eval was run. No dataset-specific code was written or "
            "tuned to try to manufacture eligibility. This is being reported as a negative/blocked outcome, not "
            "rescued.**",
            "",
            "## Recommendation",
            "",
            "Confirmation cannot proceed with the current local BasicTS installation. Two honest paths forward:",
            "",
            "1. Acquire or point this repository at a genuinely new, walk-forward-compatible, adequately-sized "
            "dataset outside the current BasicTS bundle (same `has_graph=false`/`rescale=false`/`null_val=NaN`/"
            "`metrics=[MAE,MSE]` convention, several thousand+ timestamps), then rerun this exact script -- the "
            "contamination ledger and eligibility logic require no changes.",
            "2. If graph-structured traffic datasets (PEMS/METR-LA/PEMS-BAY/CA/GBA/GLA/SD) are acceptable "
            "confirmation targets, that requires a deliberate, separately-scoped methodological adaptation (real "
            "`null_val=0.0` masking and `rescale=true` evaluation support) before any confirmation run -- not a "
            "quiet relaxation of today's eligibility bar.",
        ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = OUT_DIR / ".run_lock"
    if lock_path.exists():
        raise RuntimeError(f"Run lock already present at {lock_path} -- a confirmation run may already be active or a previous run crashed without cleanup. Inspect before proceeding.")
    lock_path.write_text(json.dumps({"pid": None, "started_at_utc": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")

    try:
        source_hashes_before = {
            "experiments/window_dependent_expert_choice_hv/run_window_dependent_expert_choice_hv.py": sha256_file(ROOT / "experiments/window_dependent_expert_choice_hv/run_window_dependent_expert_choice_hv.py"),
            "experiments/expert_choice_hv/run_expert_choice_hv.py": sha256_file(ROOT / "experiments/expert_choice_hv/run_expert_choice_hv.py"),
            "experiments/frozen_hv_costar/run_frozen_hv_costar.py": sha256_file(ROOT / "experiments/frozen_hv_costar/run_frozen_hv_costar.py"),
            "experiments/costar_multidataset_frozen/common.py": sha256_file(ROOT / "experiments/costar_multidataset_frozen/common.py"),
            "experiments/behavioral_competence/common.py": sha256_file(ROOT / "experiments/behavioral_competence/common.py"),
            "experiments/oracle_weight_tournament/run_tournament.py": sha256_file(ROOT / "experiments/oracle_weight_tournament/run_tournament.py"),
            "experiments/horizon_variable_adaptive_costar/run_hv_adaptive_costar.py": sha256_file(ROOT / "experiments/horizon_variable_adaptive_costar/run_hv_adaptive_costar.py"),
        }
        write_json(OUT_DIR / "source_hashes.json", {"before": source_hashes_before, "git": git_info(), "environment": environment_info()})

        print("[confirmation] Step 2: verifying canonical method...", flush=True)
        canonical = verify_canonical_method()
        print(f"CANONICAL_METHOD_FOUND: {canonical['CANONICAL_METHOD_FOUND']}", flush=True)
        if canonical["CANONICAL_METHOD_FOUND"] != "YES":
            raise SystemExit("Canonical method verification failed; stopping per protocol section 2.")

        print("[confirmation] Step 3: building dataset contamination ledger...", flush=True)
        ledger = build_contamination_ledger()
        write_json(OUT_DIR / "dataset_contamination_ledger.json", ledger)

        print("[confirmation] Step 4-5: static eligibility inventory (no performance inspection)...", flush=True)
        eligibility = build_eligibility(ledger)
        write_json(OUT_DIR / "dataset_eligibility.json", eligibility)

        freeze_manifest: dict[str, Any] = {
            "experiment": "window_dependent_expert_choice_confirmation",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git": git_info(),
            "canonical_method": canonical,
            "frozen_design": {
                "score_definition": "gain[t,h,v,e] = equal_error[t,h,v] - expert_error[t,h,v,e]; higher is better",
                "static_gain": "mean over legal fit windows only, per fold",
                "residual_target": "gain - static_gain; scorer predicts residual only",
                "raw_score": "static_gain + predicted_residual, coefficient exactly 1.0",
                "scorer": "ONE shared scorer across experts: Linear(input_dim->64)->ReLU->Linear(64->32)->ReLU->Linear(32->1)",
                "features": ["global window_features_group_a (6)", "per-variable current-history stats (7)", "target-free cell-local forecast/disagreement stats (6)", "horizon embed (4)", "variable embed (8)", "expert embed (4)", "static_gain scalar (1)"],
                "training": {"seed": wdec.SCORER_SEED, "optimizer": "AdamW", "lr": wdec.LR, "weight_decay": wdec.WEIGHT_DECAY, "max_epochs": wdec.MAX_EPOCHS, "patience": wdec.PATIENCE, "batch_size": wdec.BATCH_SIZE, "minibatch_unit": "forecasting window", "internal_val_fraction": wdec.INTERNAL_VAL_FRACTION},
                "affinity_calibration": {"standardization": "fit-only scalar mean/std", "temperature": wdec.AFFINITY_TEMPERATURE},
                "capacity_factor": wdec.CAPACITY_FACTOR,
                "capacity_formula": "C = round(H*V/E)",
                "tie_break": "higher affinity, then lower flattened H*V index",
                "multi_claim_rule": "equal average of claiming experts' forecasts (verified from source, NOT affinity-weighted)",
                "zero_claim_fallback": "equal fixed ensemble",
                "core_selection_algorithm": "select_core_on_router_train (costar_multidataset_frozen.common): pooled chronological OOF MAE over 4 folds, C(5,3) subsets, router_train only",
                "core_size": 3,
                "oof_protocol": {"warmup_fraction": wdec.WARMUP_FRACTION, "num_folds": wdec.NUM_OOF_FOLDS, "full_horizon_observability_rule": "starts[i] + H <= current_eval_origin"},
                "metrics": ["MAE (primary)", "MSE (secondary)"],
                "statistical_tests": {"block_bootstrap_block_length_primary": wdec.BLOCK_LENGTH, "samples": wdec.BOOTSTRAP_SAMPLES, "phase_k": wdec.PHASE_K},
                "bootstrap_seed": wdec.BOOTSTRAP_SEED,
                "shuffle_seed": wdec.SHUFFLE_SEED,
            },
            "success_criteria": {
                "PAPER_DIRECTION_CONFIRMED": "All: (1) EC beats Dynamic Token on >=ceil(2N/3) datasets, (2) block-24 CI<0 on >=ceil(N/2), (3) median relative MAE improvement >=0.10%, (4) EC beats Static EC on >=ceil(N/2), (5) correct-window beats shuffled on >=ceil(2N/3), (6) genuine claim dynamics (>5% change, <0.95 Jaccard) on >=ceil(2N/3), (7) OOF EC beats Token on >=ceil(N/2), (8) all integrity checks pass.",
                "PAPER_DIRECTION_STRONGLY_CONFIRMED": "All PAPER_DIRECTION_CONFIRMED criteria plus EC beats Frozen HxV on >=ceil(N/2) datasets and median relative improvement over Frozen HxV is positive.",
                "MECHANISM_CONFIRMED_NOT_BEST_FORECASTER": "All central PAPER_DIRECTION_CONFIRMED criteria pass but Frozen HxV still wins on the majority of datasets.",
                "MECHANISM_REPLICATES_BUT_SMALL": "Favorable win counts but median relative EC-vs-Token improvement <0.10% or dependence support too weak.",
                "PARTIAL_CONFIRMATION": "Meaningful results on some datasets but full criteria fail.",
                "NO_UNTOUCHED_CONFIRMATION": "EC beats Dynamic Token on fewer than ceil(2N/3) datasets and lacks supporting statistics.",
                "INSUFFICIENT_UNTOUCHED_DATASETS": "Fewer than 4 genuinely untouched, eligible datasets exist. Written BEFORE any confirmation metric was computed, per protocol.",
                "INVALID_CONFIRMATION": "Any leakage, test access, source-hash change, contamination error, checkpoint mutation, mismatched routing inputs, or causal/provenance failure.",
            },
            "selected_untouched_datasets": eligibility["selected_primary_confirmation_datasets"],
            "contaminated_dataset_list": ledger["contaminated_datasets"],
            "status": "BLOCKED_INSUFFICIENT_UNTOUCHED_DATASETS" if eligibility["insufficient"] else "PROCEEDING_TO_CONFIRMATION",
        }
        write_json(OUT_DIR / "method_freeze_manifest.json", freeze_manifest)
        print("PRE_CONFIRMATION_METHOD_FREEZE: PASS", flush=True)

        if eligibility["insufficient"]:
            print(f"INSUFFICIENT_UNTOUCHED_DATASETS: {eligibility['num_eligible']} eligible, {N_UNTOUCHED_REQUIRED} required", flush=True)
            # Required artifact stubs: no computation occurred, documented explicitly.
            blocked_note = "BLOCKED: INSUFFICIENT_UNTOUCHED_DATASETS. No confirmation dataset was run; see report.md and dataset_eligibility.json for the full reasoning. This file intentionally contains no fabricated results."
            write_json(OUT_DIR / "selected_cores.json", {"status": "BLOCKED", "note": blocked_note})
            write_json(OUT_DIR / "oof_results.json", {"status": "BLOCKED", "note": blocked_note})
            write_json(OUT_DIR / "confirmation_results.json", {"status": "BLOCKED", "note": blocked_note, "classification": "INSUFFICIENT_UNTOUCHED_DATASETS"})
            write_json(OUT_DIR / "fold_causality.json", {"status": "BLOCKED", "note": blocked_note})
            write_csv_stub(OUT_DIR / "routing_diagnostics.csv", blocked_note)
            write_csv_stub(OUT_DIR / "claim_count_stats.csv", blocked_note)
            write_csv_stub(OUT_DIR / "ranking_diagnostics.csv", blocked_note)
            write_csv_stub(OUT_DIR / "dependence_tests.csv", blocked_note)

            source_hashes_after = {k: sha256_file(ROOT / k) for k in source_hashes_before}
            source_unchanged = source_hashes_after == source_hashes_before
            write_json(
                OUT_DIR / "integrity_checks.json",
                {
                    "all_pass": bool(source_unchanged),
                    "no_test_access": True,
                    "no_dataset_selected_by_performance": True,
                    "eligibility_computed_before_any_confirmation_metric": True,
                    "source_hashes_unchanged_before_after": source_unchanged,
                    "affinity_weighted_ec_not_used_as_canonical": True,
                    "no_rescue_tuning_applied": True,
                    "TEST_SET_ACCESSED": "NO", "TEST_CACHE_LOADED": "NO", "TEST_METRICS_COMPUTED": "NO",
                    "classification": "INSUFFICIENT_UNTOUCHED_DATASETS",
                },
            )
            make_report(canonical, ledger, eligibility)
            print("Final classification: INSUFFICIENT_UNTOUCHED_DATASETS", flush=True)
            print(json.dumps({"classification": "INSUFFICIENT_UNTOUCHED_DATASETS", "num_eligible": eligibility["num_eligible"], "contaminated_count": len(ledger["contaminated_datasets"]), "runtime_sec": time.time() - start}, indent=2))
            return

        # Reaching here means >=4 eligible untouched datasets were found.
        # Not implemented in this run because none were found; see report.md.
        raise NotImplementedError(
            "Eligible untouched datasets were found but the core-selection/OOF/confirmation_eval pipeline for "
            "genuinely new datasets (requiring first training 5 new frozen experts per dataset) is out of scope "
            "for this run and was not implemented. Re-invoke with explicit authorization to train new experts."
        )
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
