"""Isolated health checks and matched-pool extensions of the existing router ports."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace


METHODS = {"fame": "fame_official_replication", "timerouter": "timerouter_official_replication",
           "superlinear": "super_linear_replication"}
PORT_FILES = {"fame": "run_fame_official_replication", "timerouter": "run_timerouter_official_replication",
              "superlinear": "run_super_linear_replication"}
EXPERTS = ("DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN")
SEED = 20261020


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def synthetic_pool(shuffled=False):
    """Consistent histories/targets: the previous H-step target is the CV history tail."""
    import numpy as np
    import torch
    ntr, nval, length, horizon = 384, 128, 96, 12
    n = ntr + nval
    t = np.arange(length + n * horizon + horizon)
    regimes = (t // 384) % 2
    # Histories reveal two clearly different spectral regimes, repeatedly in both periods.
    period = np.where(regimes == 0, 8., 24.)
    base = 5. + np.sin(2 * np.pi * t / period) + .03 * np.sin(.37 * t)
    origins = length + np.arange(n) * horizon
    winners = regimes[origins - length // 2].copy()
    if shuffled:
        winners = np.random.default_rng(SEED + 1).integers(0, 2, n)
    # Fixed candidate forecasts carry no information about shuffled winners.
    # The target picks one of two offsets; its earlier realised values are also in later histories.
    series = base + 2 * regimes[np.maximum(t - length // 2, 0)] - 1
    for start, winner in zip(origins, winners):
        series[start:start+horizon] = base[start:start+horizon] + 2 * winner - 1
    y = torch.tensor(np.stack([series[s:s + horizon] for s in origins]), dtype=torch.float32)[:, :, None]
    baseline = torch.tensor(np.stack([base[s:s+horizon] for s in origins]), dtype=torch.float32)
    fc = baseline[:, :, None, None] + torch.tensor([-1., 1., 3., 4., 5.])[None, None, None, :] + .02
    histories = torch.tensor(np.stack([series[s-length:s] for s in origins]), dtype=torch.float32)[:, :, None]
    def cache(lo, hi, role):
        return dict(dataset="ETTh1", cache_role=role, num_windows=hi-lo, num_features=1,
                    forecast_horizon=horizon, input_len=length, expert_names=list(EXPERTS),
                    histories=histories[lo:hi], targets=y[lo:hi],
                    target_masks=torch.ones_like(y[lo:hi], dtype=torch.bool),
                    prediction_stack=fc[lo:hi], absolute_window_starts=torch.tensor(origins[lo:hi]))
    bundle = SimpleNamespace(train_cache=cache(0, ntr, "router_train_20_60"),
                             val_cache=cache(ntr, n, "router_val_60_80"), std=torch.ones(1), core_names=EXPERTS[:3])
    return bundle, winners[ntr:]


def initialize(method, root, dest, *, health=False):
    import torch
    sys.path.insert(0, str(root))
    module = importlib.import_module(f"experiments.{METHODS[method]}.{PORT_FILES[method]}")
    module.OUT_DIR, module.WORK_DIR = dest, dest / "_work"
    module.WORK_DIR.mkdir(parents=True, exist_ok=True)
    if method == "fame":
        module.import_fame(root / "third_party/FAME")
    elif method == "timerouter":
        module.check_official(root / "third_party/TimeRouter")
    else:
        module.check_official(root / "third_party/SuperLinear")
        module.POOLS = {"A": dict(module.POOLS["A"])}
        if health:
            module.C.SEEDS = (7,)
    torch.set_num_threads(8)
    return module


def execute(method, mod, bundle, dataset, *, health=False):
    """Call the actual port pipeline; capture its selected validation winner for health checks."""
    import numpy as np
    import torch
    picks = []
    if method == "fame":
        mod.load_bundle = lambda _d: bundle
        if not mod.fp_path(dataset).exists():
            mod.compute_fingerprints(dataset)
        original_predict = mod.predict_probs
        def capture(router, scaler, fp):
            p = original_predict(router, scaler, fp)
            if len(fp) == bundle.val_cache["num_windows"] * bundle.val_cache["num_features"]:
                picks.append(p.argmax(1).numpy())
            return p
        if health:
            mod.predict_probs = capture
        for seed in ((7,) if health else mod.SEEDS):
            if not (mod.WORK_DIR / f"seed{seed}_{dataset}.json").exists():
                mod.run_seed_dataset(dataset, seed)
        metrics = {}
        for seed in ((7,) if health else mod.SEEDS):
            z = torch.load(mod.WORK_DIR / f"seed{seed}_{dataset}.pt", map_location="cpu", weights_only=False)
            metrics[f"fame_top2_seed{seed}"] = z["top2"]
    elif method == "timerouter":
        mod.C.load_bundle = lambda _d: bundle
        original_decide = mod.OFFICIAL["R"].decide
        def capture(proba, div, **kw):
            d = original_decide(proba, div, **kw)
            if len(proba) == bundle.val_cache["num_windows"] * bundle.val_cache["num_features"]:
                picks.append(np.asarray(d["picks"]))
            return d
        if health:
            mod.OFFICIAL["R"].decide = capture
        if not (mod.WORK_DIR / f"{dataset}.json").exists():
            mod.run_dataset(dataset, str(mod.DEFAULT_REPO), workers=1, xgb_jobs=8)
        z = torch.load(mod.WORK_DIR / f"{dataset}.pt", map_location="cpu", weights_only=False)
        metrics = {"timerouter": z["timerouter"]}
    else:
        mod.C.load_bundle = lambda _d: bundle
        original_predict = mod.predict_val
        def capture(gate, b, pool, device):
            result = original_predict(gate, b, pool, device)
            with torch.no_grad():
                vals = []
                for lo in range(0, b.val_cache["num_windows"], mod.BATCH_WINDOWS):
                    win = torch.arange(lo, min(b.val_cache["num_windows"], lo + mod.BATCH_WINDOWS))
                    p, _, _, _ = mod.batch_rows(b.val_cache, b.std, win, False, device)
                    vals.append(gate.gating_network(p).argmax(1).cpu().numpy())
                picks.append(np.concatenate(vals))
            return result
        if health:
            mod.predict_val = capture
        if not (mod.WORK_DIR / f"{dataset}.json").exists():
            mod.run_dataset(dataset, torch.device("cpu"))
        z = torch.load(mod.WORK_DIR / f"{dataset}.pt", map_location="cpu", weights_only=False)
        metrics = {f"superlinear_{key}": v for key, v in z.items() if key.startswith("A_seed")}
    return metrics, picks[-1] if picks else None


def health(args):
    import numpy as np
    import torch
    bundle, winners = synthetic_pool(args.case == "shuffled")
    dest = args.output
    # An interrupted health case is rerun in a new attempt directory, preserving its artifacts.
    attempt = 1
    while (dest / f"attempt{attempt}").exists():
        attempt += 1
    mod = initialize(args.method, args.repo_root, dest / f"attempt{attempt}", health=True)
    result, picked = execute(args.method, mod, bundle, "ETTh1", health=True)
    from conditional import metrics
    tr, va = bundle.train_cache, bundle.val_cache
    train_scores = [metrics(tr, tr["prediction_stack"][..., i], bundle.std)[:, 0].mean() for i in range(5)]
    best = int(np.argmin(train_scores))
    equal = float(metrics(va, va["prediction_stack"].mean(-1), bundle.std)[:, 0].mean())
    single = float(metrics(va, va["prediction_stack"][..., best], bundle.std)[:, 0].mean())
    losses = np.stack([v["per_window_mae"].numpy() for v in result.values()])
    score = float(losses.mean())
    if picked is None or picked.shape != winners.shape:
        raise RuntimeError("Missing or misaligned validation routing diagnostics")
    accuracy = float(np.mean(picked == winners))
    finite = bool(np.isfinite(losses).all())
    # A convex combiner can reconstruct the best forecast with other candidates;
    # argmax-winner accuracy is therefore a diagnostic, not its correctness gate.
    passed = finite and (score >= .75 * single if args.case == "shuffled" else
                         score < .75 * min(equal, single))
    if args.method == "timerouter":
        passed = passed and (accuracy < .65 if args.case == "shuffled" else accuracy >= .75)
    write(dest / "health.json", dict(method=args.method, case=args.case, passed=passed,
          mae=score, equal_mae=equal, train_selected_single_mae=single, winner_accuracy=accuracy,
          attempt=attempt, synthetic_seed=SEED, finite=finite, test_set_accessed=False,
          criteria="Predictable MAE < 75% of equal and train-selected single; shuffled MAE >= 75% of single; TimeRouter winner-accuracy guard",
          evidence="Software positive/negative control; not a published benchmark reproduction"))
    if not passed:
        raise RuntimeError(f"{args.method} {args.case} health check failed; keep its negative result")


def unit(args):
    import numpy as np
    import torch
    from conditional import read_cache, server_unit, load_tensor_file, archived_reference, archived_files, metrics, paired
    spec = server_unit(args.repo_root, args.dataset, args.horizon)
    if args.method == "timerouter" and args.horizon > 96:
        write(args.output / "result.json", dict(status="UNSUPPORTED", reason="Existing TimeRouter CV requires H <= 96 history length",
                                                method=args.method, dataset=args.dataset, horizon=args.horizon, test_set_accessed=False))
        return
    paths = [Path(spec[k]) for k in ("train_cache", "val_cache", "normalizer")]
    paths += archived_files(Path(spec["archived_work"]), args.dataset)
    hashes = {str(p): sha(p) for p in paths}
    protocol = dict(method=args.method, dataset=args.dataset, horizon=args.horizon, hashes=hashes,
                    source_sha256=sha(__file__), test_set_accessed=False)
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("Inputs/source changed: use a new output directory")
    write(path, protocol)
    if (args.output / "result.json").exists():
        record = json.loads((args.output / "result.json").read_text())
        for name, digest in record.get("output_hashes", {}).items():
            if sha(args.output / name) != digest:
                raise RuntimeError("Completed router output changed")
        return
    train, val = read_cache(paths[0], "router_train_20_60"), read_cache(paths[1], "router_val_60_80")
    if train["dataset"] != args.dataset or val["dataset"] != args.dataset:
        raise RuntimeError("Dataset mismatch")
    if list(train["expert_names"]) != list(EXPERTS) or list(val["expert_names"]) != list(EXPERTS):
        raise RuntimeError("Unexpected expert order")
    if int(train["forecast_horizon"]) != args.horizon or int(val["forecast_horizon"]) != args.horizon:
        raise RuntimeError("Horizon mismatch")
    if not {"block_b_oos", "block_c_oos"}.issubset(train.get("provenance", {}).get("source_caches", {})):
        raise RuntimeError("Missing out-of-sample forecast provenance")
    if int(train["absolute_window_starts"][-1]) + args.horizon > int(val["absolute_window_starts"][0]):
        raise RuntimeError("Training labels overlap evaluation")
    std = torch.as_tensor(load_tensor_file(paths[2])["scaler_std"], dtype=torch.float32).flatten()
    if len(std) != train["num_features"] or not torch.isfinite(std).all() or not (std > 0).all():
        raise RuntimeError("Invalid checkpoint normalizer")
    primary, comparators, audit = archived_reference(Path(spec["archived_work"]), train, val, std)
    bundle = SimpleNamespace(train_cache=train, val_cache=val, std=std)
    mod = initialize(args.method, args.repo_root, args.output)
    if args.method == "fame":
        mod.SEASONAL_PERIOD[args.dataset] = spec["frequency"]
    else:
        mod.C.FREQ[args.dataset] = "10T" if spec["frequency"] == 144 else "H"
    result, _ = execute(args.method, mod, bundle, args.dataset)
    records = {}
    for name, arrays in result.items():
        loss = np.stack([np.asarray(arrays[f"per_window_{m}"]) for m in ("mae", "mse")], axis=1)
        if loss.shape != (len(val["targets"]), 2) or not np.isfinite(loss).all():
            raise RuntimeError("Nonfinite or misaligned router loss")
        records[name] = loss
    # Seed losses, rather than predictions, are averaged. FAME's top1 is secondary and not selected here.
    avg = np.stack(list(records.values())).mean(0)
    origins = val["absolute_window_starts"].numpy()
    contrasts = {base: {m: paired(avg[:, j], losses.mean(0)[:, j], origins, args.horizon)
                       for j, m in enumerate(("mae", "mse"))}
                 for base, losses in {"wide_top3": primary, **comparators}.items()}
    if hashes != {str(p): sha(p) for p in paths}:
        raise RuntimeError("Inputs changed during router run")
    output_hashes = {str(p.relative_to(args.output)): sha(p) for p in (args.output / "_work").glob("*")
                     if p.suffix in (".pt", ".json")}
    write(args.output / "result.json", dict(status="COMPLETE", **protocol, reference_audit=audit,
          mae=float(avg[:, 0].mean()), mse=float(avg[:, 1].mean()), contrasts=contrasts,
          per_seed_means={k: v.mean(0).tolist() for k, v in records.items()}, output_hashes=output_hashes,
          adaptation="Existing port on the matched frozen pool; development evaluation"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("health", "unit"))
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case", choices=("predictable", "shuffled"), default="predictable")
    p.add_argument("--dataset")
    p.add_argument("--horizon", type=int)
    a = p.parse_args()
    if a.mode == "unit" and (not a.dataset or a.horizon not in (96, 192)):
        p.error("A real-data unit needs --dataset and --horizon 96 or 192")
    a.repo_root, a.output = a.repo_root.resolve(), a.output.resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    a.output.mkdir(parents=True, exist_ok=True)
    (health if a.mode == "health" else unit)(a)


if __name__ == "__main__":
    main()
