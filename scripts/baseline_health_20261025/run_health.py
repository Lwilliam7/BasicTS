"""Four-port synthetic audit. Software health never clears paper-table eligibility."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


HERE = Path(__file__).resolve().parent
PORTS = {
    "fame": ("fame_official_replication", "run_fame_official_replication", "FAME"),
    "timerouter": ("timerouter_official_replication", "run_timerouter_official_replication", "TimeRouter"),
    "onenet": ("onenet_official_replication", "run_onenet_official_replication", "OneNet"),
    "superlinear": ("super_linear_replication", "run_super_linear_replication", "SuperLinear"),
}
CASES = {m: ("predictable", "shuffled", "permuted") for m in PORTS}
CASES["onenet"] = ("branch1", "branch2", "shuffled", "permuted", "poisoned")
LIMITS = {
    "fame": "Rolling-window frozen-pool adaptation; no published dataset/result reproduced.",
    "timerouter": "Port uses a retrained OVA head; the authors' head training code is not released.",
    "onenet": "Two-branch OCP block with frozen experts and delayed feedback; full branch co-adaptation is absent. Online protocol differs from offline routers.",
    "superlinear": "Spectral combination gate over frozen experts; full frequency-expert training is absent.",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def environment():
    packages = {}
    for n in ("numpy", "torch", "pandas", "scikit-learn", "xgboost", "tsfeatures"):
        try:
            packages[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            packages[n] = None
    return dict(python=sys.version, packages=packages)


def source_identity(root, official):
    files = list(HERE.glob("*.py")) + [HERE / "PROTOCOL.md"]
    for folder, script, _ in PORTS.values():
        files.append(root / "experiments" / folder / (script + ".py"))
    for rel in ("official_router_ports/common.py", "published_baseline_comparisons/run_published_baselines.py",
                "window_dependent_expert_choice_hv/run_window_dependent_expert_choice_hv.py"):
        files.append(root / "experiments" / rel)
    for folder in ("FAME/fame", "TimeRouter/gift_eval", "OneNet/exp", "SuperLinear/model_files"):
        files += list((official / folder).glob("*.py"))
    return {str(p.resolve()): sha(p) for p in sorted(files)}


def permute(bundle):
    from types import SimpleNamespace
    order = [4, 2, 0, 3, 1]
    caches = []
    for c in (bundle.train_cache, bundle.val_cache):
        d = dict(c)
        d["prediction_stack"] = c["prediction_stack"][..., order].clone()
        d["expert_names"] = [c["expert_names"][i] for i in order]
        caches.append(d)
    return SimpleNamespace(train_cache=caches[0], val_cache=caches[1], std=bundle.std,
                           core_names=bundle.core_names)


def onenet_pool(case):
    import numpy as np
    import torch
    from types import SimpleNamespace
    ntr, nv, h, length = 384, 128, 12, 96
    n = ntr + nv
    t = np.arange(length + (n + 1) * h)
    base = 5. + np.sin(2 * np.pi * t / 24) + .03 * np.sin(.37 * t)
    origins = length + np.arange(n) * h
    if case == "branch2":
        winners = np.ones(n, dtype=int)
    elif case == "shuffled":
        winners = np.random.default_rng(20261025).integers(0, 2, n)
    else:
        winners = np.zeros(n, dtype=int)
    series = base - 1.
    for start, winner in zip(origins, winners):
        series[start:start+h] = base[start:start+h] + 2 * winner - 1
    y = torch.tensor(np.stack([series[s:s+h] for s in origins]), dtype=torch.float32)[:, :, None]
    hist = torch.tensor(np.stack([series[s-length:s] for s in origins]), dtype=torch.float32)[:, :, None]
    ref = torch.tensor(np.stack([base[s:s+h] for s in origins]), dtype=torch.float32)[:, :, None, None]
    # The two actual OneNet branch names occupy columns 1 and 2.
    fc = ref + torch.tensor([3., -1., 1., 4., 5.])[None, None, None, :] + .02
    def cache(lo, hi, role):
        return dict(dataset="ETTh1", cache_role=role, num_windows=hi-lo, num_features=1,
                    forecast_horizon=h, input_len=length,
                    expert_names=["DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN"],
                    histories=hist[lo:hi], targets=y[lo:hi], target_masks=torch.ones_like(y[lo:hi], dtype=torch.bool),
                    prediction_stack=fc[lo:hi], absolute_window_starts=torch.tensor(origins[lo:hi]))
    return SimpleNamespace(train_cache=cache(0, ntr, "router_train_20_60"),
                           val_cache=cache(ntr, n, "router_val_60_80"), std=torch.ones(1),
                           core_names=("PatchTST", "iTransformer", "TimesNet")), winners[ntr:]


def metric(cache, pred, std):
    import torch
    m = cache["target_masks"].bool()
    e = torch.where(m, (pred - cache["targets"]) / std[None, None, :], 0.)
    den = m.flatten(1).sum(1).clamp_min(1)
    return torch.stack((e.abs().flatten(1).sum(1) / den, e.square().flatten(1).sum(1) / den), 1)


def baselines(bundle, method):
    import torch
    names = ("PatchTST", "iTransformer") if method == "onenet" else tuple(bundle.train_cache["expert_names"])
    def cols(c):
        return [c["expert_names"].index(n) for n in names]
    tr, va = bundle.train_cache, bundle.val_cache
    ti, vi = cols(tr), cols(va)
    train_means = [float(metric(tr, tr["prediction_stack"][..., i], bundle.std)[:, 0].mean()) for i in ti]
    selected = min(range(len(ti)), key=lambda i: train_means[i])
    equal = metric(va, va["prediction_stack"][..., vi].mean(-1), bundle.std)
    single = metric(va, va["prediction_stack"][..., vi[selected]], bundle.std)
    expert_error = (va["prediction_stack"][..., vi] - va["targets"][..., None]).abs()
    oracle = (expert_error / bundle.std[None, None, :, None]).mean((1, 2)).min(1).values
    require(torch.allclose(oracle, torch.full_like(oracle, .02), atol=2e-6, rtol=0), "Analytic oracle is not 0.02")
    return dict(equal_mae=float(equal[:, 0].mean()), single_mae=float(single[:, 0].mean()),
                oracle_mae=float(oracle.mean()), selected_single=names[selected])


def initialize(method, root, official, dest):
    import torch
    sys.path.insert(0, str(root))
    folder, script, clone = PORTS[method]
    mod = importlib.import_module(f"experiments.{folder}.{script}")
    mod.OUT_DIR, mod.WORK_DIR = dest, dest / "_work"
    mod.WORK_DIR.mkdir(parents=True, exist_ok=True)
    mod.DEFAULT_REPO = official / clone
    if method == "fame":
        info = mod.import_fame(official / clone)
        pin, hashes = mod.FAME_COMMIT, info["sha256"]
        # Windows on this host ignores an empty CUDA_VISIBLE_DEVICES mask.
        # Pin the execution device explicitly; never rely on the upstream auto selector.
        original_config = mod.fame_config
        def cpu_config(*a, **kw):
            kw["device"] = "cpu"
            return original_config(*a, **kw)
        mod.fame_config = cpu_config
        mod.health_training_devices = []
        original_train = mod.train_router
        def cpu_train(*a, **kw):
            router, history, cfg = original_train(*a, **kw)
            actual = str(router.device)
            require(actual == "cpu", f"Health audit unexpectedly trained FAME on {actual}")
            mod.health_training_devices.append(actual)
            return router, history, cfg
        mod.train_router = cpu_train
    else:
        mod.check_official(official / clone)
        pin = {"timerouter": "TR_COMMIT", "onenet": "ON_COMMIT", "superlinear": "SL_COMMIT"}[method]
        pin, hashes = getattr(mod, pin), mod.OFFICIAL["sha256"]
    if method == "superlinear":
        mod.POOLS = {"A": dict(mod.POOLS["A"])}
        mod.C.SEEDS = (7,)
    torch.set_num_threads(2)
    return mod, dict(status="PASS", commit_pinned=pin, file_hashes=hashes)


def run_onenet(mod, bundle, dest):
    import numpy as np
    import torch
    v1, v2, vy = mod.standardized(bundle, bundle.val_cache)
    origins = bundle.val_cache["absolute_window_starts"]
    h = int(bundle.val_cache["forecast_horizon"])
    predictions, weights, trace = [], [], []
    original = mod.OCP
    class AuditedOCP(original):
        def reset_for_test(self, d):
            super().reset_for_test(d)
            self.updated = 0
        def online_update(self, y1, y2, y):
            row, current = self.updated, len(predictions)
            require(current < len(origins) and row < current, "Update used a current/future row")
            require(int(origins[row]) + h <= int(origins[current]), "Feedback target has not resolved")
            for got, expected in ((y1, v1[row:row+1]), (y2, v2[row:row+1]), (y, vy[row:row+1])):
                require(torch.equal(got, expected), "Feedback tensor/window alignment failed")
            trace.append(dict(update_row=row, prediction_row=current, update_origin=int(origins[row]),
                              target_end=int(origins[row])+h, prediction_origin=int(origins[current])))
            super().online_update(y1, y2, y)
            self.updated += 1
        def predict(self, y1, y2):
            row = len(predictions)
            require(next(self.decision.parameters()).device.type == "cpu", "OneNet audit must use CPU")
            require(torch.equal(y1, v1[row]) and torch.equal(y2, v2[row]), "Prediction input/window alignment failed")
            result = super().predict(y1, y2)
            predictions.append(result.detach().clone())
            w = torch.sigmoid(self.weight.detach() + self.bias.detach().view(-1)[:self.weight.numel()])
            require(bool(torch.isfinite(w).all() and (w >= 0).all() and (w <= 1).all()), "Nonconvex effective OneNet weight")
            weights.append(w.cpu().numpy())
            return result
    mod.OCP = AuditedOCP
    mod.C.load_bundle = lambda _d: bundle
    mod.run_seed_dataset("ETTh1", 7)
    require(len(predictions) == len(origins) and len(trace) == len(origins)-1, "Unexpected causal update count")
    z = torch.load(mod.WORK_DIR / "seed7_ETTh1.pt", map_location="cpu", weights_only=False)
    values = np.stack([z[f"per_window_{m}"].numpy() for m in ("mae", "mse")], 1)
    pred = torch.stack(predictions).numpy()
    w = np.stack(weights)
    np.savez_compressed(dest / "one_net_trace.npz", standardized_predictions=pred, branch1_weights=w)
    write(dest / "causality_trace.json", dict(passed=True, update_count=len(trace), trace=trace))
    return values, dict(online_updates=len(trace), causality_passed=True,
                        branch1_weight_mean=float(w.mean()), branch1_weight_tail_mean=float(w[-32:].mean()),
                        target_feedback="Fully observed previous windows only")


def convex_checks(mod, method, checks):
    import numpy as np
    import torch
    def check_weight(w):
        require(w.device.type == "cpu", "Combiner audit must use CPU")
        require(bool(torch.isfinite(w).all() and (w >= -1e-7).all()), "Invalid combiner weight")
        require(bool(torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6, rtol=1e-6)), "Weights do not sum to one")
        checks["convex_calls"] += 1
    if method == "fame":
        original = mod.sparse_weights
        def sparse(*args, **kwargs):
            w = original(*args, **kwargs)
            check_weight(w)
            return w
        mod.sparse_weights = sparse
    elif method == "superlinear":
        original = mod.F.softmax
        def softmax(*args, **kwargs):
            w = original(*args, **kwargs)
            check_weight(w)
            return w
        mod.F.softmax = softmax
    elif method == "timerouter":
        original = mod.OFFICIAL["R"].cv_inverse_weighted
        def fallback(forecasts, *args, **kwargs):
            result = original(forecasts, *args, **kwargs)
            stack = np.stack(list(forecasts.values()))
            require(bool(np.isfinite(result).all() and (result >= stack.min(0)-1e-6).all()
                         and (result <= stack.max(0)+1e-6).all()), "Fallback outside convex forecast hull")
            checks["convex_calls"] += 1
            return result
        mod.OFFICIAL["R"].cv_inverse_weighted = fallback


def worker(args):
    import numpy as np
    import torch
    from unittest.mock import patch
    import reference_fixture as fixture
    dest = args.output
    dest.mkdir(parents=True, exist_ok=False)
    mod, upstream = initialize(args.method, args.repo_root, args.official_root, dest)
    if args.method == "onenet":
        bundle, winners = onenet_pool(args.case)
    else:
        bundle, winners = fixture.synthetic_pool(args.case == "shuffled")
    if args.case == "permuted":
        bundle = permute(bundle)
    if args.case == "poisoned":
        bundle.val_cache = dict(bundle.val_cache)
        bundle.val_cache["targets"] = bundle.val_cache["targets"].clone()
        bundle.val_cache["targets"][64:] += 37.
    controls = baselines(bundle, args.method) if args.case != "poisoned" else {}
    loads = []
    original_load = torch.load
    def guarded_load(path, *a, **kw):
        require(Path(path).resolve().is_relative_to(dest.resolve()), f"Research-data load forbidden in synthetic audit: {path}")
        loads.append(str(Path(path).resolve().relative_to(dest)))
        return original_load(path, *a, **kw)
    diag = dict(convex_calls=0)
    convex_checks(mod, args.method, diag)
    with patch.object(torch, "load", guarded_load):
        if args.method == "onenet":
            losses, extra = run_onenet(mod, bundle, dest)
            diag.update(extra)
        else:
            results, picked = fixture.execute(args.method, mod, bundle, "ETTh1", health=True)
            losses = np.stack([np.stack([entry[f"per_window_{m}"].numpy() for m in ("mae", "mse")], 1)
                               for entry in results.values()]).mean(0)
            require(picked is not None and picked.shape == winners.shape, "Missing/misaligned routing diagnostic")
            diag["winner_accuracy"] = float(np.mean(picked == winners))
            require(diag["convex_calls"] > 0, "Combiner convexity hook did not execute")
            if args.method == "timerouter":
                diag["fallback_frequency"] = read(mod.WORK_DIR / "ETTh1.json")["val_fallback_frequency"]
            if args.method == "fame":
                diag["routing"] = read(mod.WORK_DIR / "seed7_ETTh1.json")["top2"]
                require(bool(mod.health_training_devices), "FAME device trace is missing")
                diag["training_devices"] = mod.health_training_devices
            if args.method == "superlinear":
                diag["routing"] = read(mod.WORK_DIR / "ETTh1.json")["pools"]["A"]["per_seed"]
    require(losses.shape == (128, 2) and bool(np.isfinite(losses).all()), "Nonfinite/misaligned losses")
    np.savez_compressed(dest / "per_window_losses.npz", losses=losses)
    mae = float(losses[:, 0].mean())
    checks = dict(finite=True, source_gate_passed=True, synthetic_load_guard=True)
    if args.case not in ("permuted", "poisoned"):
        if args.method == "onenet":
            if args.case == "shuffled":
                checks["unpredictable_guard"] = mae >= .75 * controls["equal_mae"]
            else:
                checks["learns_known_branch"] = mae < controls["oracle_mae"] + .5 * (controls["equal_mae"] - controls["oracle_mae"])
            checks["causal_feedback"] = bool(diag["causality_passed"])
        elif args.case == "shuffled":
            checks["unpredictable_guard"] = mae >= .75 * controls["single_mae"]
            if args.method == "timerouter":
                checks["chance_winner_accuracy"] = diag["winner_accuracy"] < .65
        else:
            checks["learns_predictable_winner"] = mae < .75 * min(controls["equal_mae"], controls["single_mae"])
            if args.method == "timerouter":
                checks["winner_accuracy"] = diag["winner_accuracy"] >= .75
    artifacts = {str(p.relative_to(dest)): sha(p) for p in dest.rglob("*") if p.is_file()}
    record = dict(status="CHECKED", method=args.method, case=args.case, seed=7, mae=mae,
                  mse=float(losses[:, 1].mean()), tail32_mae=float(losses[-32:, 0].mean()),
                  checks=checks, software_case_passed=all(checks.values()), baselines=controls,
                  diagnostics=diag, upstream_source_check=upstream, synthetic_files_loaded=loads,
                  artifact_hashes=artifacts, paper_fidelity="NOT_ESTABLISHED", table_eligible=False,
                  test_set_accessed=False, real_research_data_accessed=False)
    write(dest / "result.json", record)
    print(json.dumps({k: record[k] for k in ("method", "case", "mae", "software_case_passed", "checks")}), flush=True)


def report(output):
    import numpy as np
    methods = {}
    for method in PORTS:
        cases = {}
        for case in CASES[method]:
            path = output / method / case / "result.json"
            cases[case] = read(path) if path.exists() else dict(status="MISSING", software_case_passed=False)
        checks = {case: bool(r.get("software_case_passed")) for case, r in cases.items()}
        base_case = "branch1" if method == "onenet" else "predictable"
        comparison = {}
        try:
            original = np.load(output / method / base_case / "per_window_losses.npz")["losses"]
            permuted = np.load(output / method / "permuted" / "per_window_losses.npz")["losses"]
            comparison["expert_name_column_permutation"] = bool(np.allclose(original, permuted, atol=1e-6, rtol=0))
            comparison["permutation_max_abs_difference"] = float(np.max(np.abs(original-permuted)))
        except (OSError, KeyError):
            comparison["expert_name_column_permutation"] = False
        if method == "onenet":
            try:
                a = np.load(output / method / "branch1" / "one_net_trace.npz")["standardized_predictions"]
                b = np.load(output / method / "poisoned" / "one_net_trace.npz")["standardized_predictions"]
                comparison["future_target_poison_prefix_unchanged"] = bool(np.array_equal(a[:65], b[:65]))
                comparison["future_target_poison_prefix_max_difference"] = float(np.max(np.abs(a[:65]-b[:65])))
            except (OSError, KeyError):
                comparison["future_target_poison_prefix_unchanged"] = False
        bool_checks = [v for v in comparison.values() if isinstance(v, bool)]
        passed = all(checks.values()) and all(bool_checks)
        methods[method] = dict(software_health="PASS" if passed else "FAIL_OR_INCOMPLETE", case_passes=checks,
                               cross_case_checks=comparison, cases=cases, paper_fidelity="NOT_ESTABLISHED",
                               table_eligible=False, limitation=LIMITS[method])
    result = dict(status="SOFTWARE_HEALTH_PASS" if all(v["software_health"] == "PASS" for v in methods.values()) else "HEALTH_FAILURE_OR_INCOMPLETE",
                  methods=methods, table_eligible=False, paper_fidelity="NOT_ESTABLISHED", test_set_accessed=False,
                  note="Known-answer synthetic checks validate this port's software behavior, not full-paper reproduction.")
    write(output / "results.json", result)
    lines = ["# Four-port synthetic health audit", "", f"Status: {result['status']}", "",
             "Paper-table eligibility: **NOT CLEARED for every port**. Full-paper fidelity has not been established.", "",
             "| Port | Software health | Control | MAE | Equal control | Oracle | Table eligible |",
             "| --- | --- | --- | ---: | ---: | ---: | --- |"]
    for method, m in methods.items():
        for case, r in m["cases"].items():
            b = r.get("baselines", {})
            def number(x):
                return f"{x:.6f}" if isinstance(x, (int, float)) else "—"
            lines.append(f"| {method} | {m['software_health']} | {case} | {number(r.get('mae'))} | {number(b.get('equal_mae'))} | {number(b.get('oracle_mae'))} | No |")
    for method, m in methods.items():
        lines += ["", f"{method}: {m['limitation']}", f"Cross-case checks: `{json.dumps(m['cross_case_checks'])}`", ""]
    lines += ["## Interpretation", "", "Failures remain failures; no thresholds or method settings were tuned after seeing outcomes.",
              "OneNet is its causal OCP block on two frozen branches. It cannot be presented as a full OneNet reproduction or ranked as an offline frozen router.",
              "Detailed diagnostics, source/environment manifests, per-window losses and OneNet causal traces accompany this report."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


@contextlib.contextmanager
def run_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                f.write(b"0"); f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "run", "worker", "report"))
    p.add_argument("--repo-root", type=Path, default=HERE.parents[1])
    p.add_argument("--official-root", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--method", choices=PORTS)
    p.add_argument("--case")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.official_root = (args.official_root or args.repo_root / "third_party").resolve()
    args.output = (args.output or args.repo_root / "experiments/baseline_health_20261025").resolve()
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    if args.mode == "worker":
        require(args.method in PORTS and args.case in CASES[args.method], "Invalid method/control")
        try:
            worker(args)
        except Exception as exc:
            args.output.mkdir(parents=True, exist_ok=True)
            write(args.output / "result.json", dict(status="ERROR", method=args.method, case=args.case,
                  error=str(exc), traceback=traceback.format_exc(), software_case_passed=False,
                  table_eligible=False, paper_fidelity="NOT_ESTABLISHED"))
            raise
        return
    if args.mode == "plan":
        required = [args.official_root / folder / rel for folder, rel in
                    (("FAME", "fame/router.py"), ("TimeRouter", "gift_eval/_router.py"),
                     ("OneNet", "exp/exp_onenet_fsnet.py"), ("SuperLinear", "model_files/modeling_super_linear.py"))]
        print(json.dumps(dict(methods=CASES, environment=environment(), missing_official_sources=[str(p) for p in required if not p.is_file()],
              output=str(args.output), cpu_only=True, table_eligible=False, paper_fidelity="NOT_ESTABLISHED"), indent=2))
        return
    if args.mode == "report":
        print(report(args.output)["status"])
        return
    output = args.output
    with run_lock(output.parent / (output.name + ".lock")):
        manifest = dict(source_hashes=source_identity(args.repo_root, args.official_root), environment=environment(),
                        controls=CASES, seed=7, test_set_accessed=False, table_eligible=False)
        manifest = json.loads(json.dumps(manifest))
        if output.exists():
            require(args.resume, "Existing output preserved; use --resume for an identical audit or a new directory")
            require(read(output / "manifest.json") == manifest, "Sources/environment changed; use a new audit output")
        else:
            output.mkdir(parents=True)
            write(output / "manifest.json", manifest)
        state = dict(status="RUNNING", pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(), cases={})
        for method, cases in CASES.items():
            for case in cases:
                dest = output / method / case
                path = dest / "result.json"
                key = f"{method}/{case}"
                if path.exists():
                    r = read(path)
                    for rel, digest in r.get("artifact_hashes", {}).items():
                        require(sha(dest / rel) == digest, "Completed audit artifact changed")
                    state["cases"][key] = r.get("software_case_passed", False)
                    continue
                state["active_case"] = key
                write(output / "status.json", state)
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker", "--repo-root", str(args.repo_root),
                           "--official-root", str(args.official_root), "--output", str(dest), "--method", method, "--case", case]
                print(f"START {key}", flush=True)
                with (output / f"{method}_{case}.log").open("a", encoding="utf-8") as log:
                    rc = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT).returncode
                state["cases"][key] = bool(rc == 0 and path.exists() and read(path).get("software_case_passed"))
                print(f"FINISHED {key}: {'PASS' if state['cases'][key] else 'FAIL'}", flush=True)
                write(output / "status.json", state)
        require(source_identity(args.repo_root, args.official_root) == manifest["source_hashes"], "Source changed during audit")
        result = report(output)
        state.update(status=result["status"], active_case=None, table_eligible=False)
        write(output / "status.json", state)
        print(f"{result['status']}: {output / 'report.md'}", flush=True)
        if result["status"] != "SOFTWARE_HEALTH_PASS":
            sys.exit(2)


if __name__ == "__main__":
    main()
