"""Actual Super-Linear port pipeline on synthetic pools; official-clone verification is server-only."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research_suite_20261020"))
sys.path.insert(0, r"C:\Data\Documents\Code\BasicTS")
import torch
import numpy as np
import run_router_checks as r
import experiments.super_linear_replication.run_super_linear_replication as sl
from conditional import metrics

torch.set_num_threads(2)
sl.POOLS = {"A": dict(sl.POOLS["A"])}
sl.C.SEEDS = (7,)
output = ROOT / "research_review/superlinear_component_checks_independent_forecasts"
summary = {}
for shuffled in (False, True):
    case = "shuffled" if shuffled else "predictable"
    bundle, winners = r.synthetic_pool(shuffled)
    sl.OUT_DIR = output / case
    sl.WORK_DIR = sl.OUT_DIR / "_work"
    sl.WORK_DIR.mkdir(parents=True, exist_ok=True)
    result, picks = r.execute("superlinear", sl, bundle, "ETTh1", health=True)
    avg = float(np.stack([v["per_window_mae"].numpy() for v in result.values()]).mean())
    accuracy = float(np.mean(picks == winners)) if picks is not None else None
    eq = float(metrics(bundle.val_cache, bundle.val_cache["prediction_stack"].mean(-1), bundle.std)[:, 0].mean())
    tr = bundle.train_cache
    best = min(range(5), key=lambda i: metrics(tr, tr["prediction_stack"][..., i], bundle.std)[:, 0].mean())
    single = float(metrics(bundle.val_cache, bundle.val_cache["prediction_stack"][..., best], bundle.std)[:, 0].mean())
    summary[case] = dict(mae=avg, equal_mae=eq, winner_accuracy=accuracy,
                         train_selected_single_mae=single,
                         official_source_verification="NOT_RUN_LOCAL_CLONE_UNAVAILABLE")
    assert picks is not None, "A fresh output directory is required for capture"
    if shuffled:
        assert avg >= .75 * single, summary[case]
    else:
        assert avg < .75 * min(eq, single), summary[case]
r.write(output / "summary.json", summary)
print(json.dumps(summary, indent=2))
