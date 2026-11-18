"""Gate ablation: see PREDECLARATION.md. --selftest uses synthetic CPU tensors only.

All scientific imports are lazy. One worker process handles exactly one unit;
the original experiment modules and their outputs are never written by this runner.
No error metric is printed before --analyze.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DATASETS = ("ETTh1", "Electricity", "Traffic", "AppliancesEnergy", "OikolabWeather", "SolarAL", "TetouanPower")
HORIZONS = (12, 192)
SEEDS = (7, 11, 13, 17, 19)
ARMS = ("core_only", "router_only", "constant", "static_hv", "full", "static_all")
LADDER = (("constant", "core_only"), ("static_hv", "constant"), ("full", "static_hv"),
          ("router_only", "full"), ("static_hv", "static_all"))
EVIDENCE = "Development evidence on previously scored 60-80% blocks; no test access; frozen method unchanged."


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    """Atomic completion records; readers never see half a job."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


@contextlib.contextmanager
def lock(path, shared=False, wait=False):
    """Kernel lock, released on crash. Windows is used only for local tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            if not f.read(1):
                f.write(b"0"); f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK if wait else msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | (0 if wait else fcntl.LOCK_NB))
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


def deps():
    import torch
    import experiments.topk_wide_router_ec.run_topk_wide_router_ec as topk
    import experiments.wide_top3_untouched_confirmation.run_untouched_confirmation as conf
    import experiments.long_horizon_protocol.run_long_horizon as lh
    return torch, topk, conf, lh


def source_hashes():
    """Bind resumability to the frozen path, wrappers, ablation and plan."""
    import experiments.wide_top3_method_freeze.freeze as fz
    paths = set(fz.SOURCES) | {
        "experiments/wide_top3_method_freeze/freeze.py", "experiments/wide_top3_method_freeze/freeze_manifest.json",
        "experiments/wide_top3_untouched_confirmation/run_untouched_confirmation.py",
        "experiments/wide_top3_untouched_confirmation/run_untouched_training.py",
        "experiments/wide_top3_confirmation_long_training/run_long_training.py",
        "experiments/expert_pool_convergence_ec/run_pool_evaluation.py",
        "experiments/expert_pool_convergence_ec/run_expert_pool_training.py",
        "experiments/long_horizon_protocol/run_long_horizon.py",
        "experiments/long_horizon_protocol/pool_plan.json",
        "experiments/long_horizon_protocol/dataset_hashes.json",
        "experiments/gate_ablation/PREDECLARATION.md", "experiments/gate_ablation/run_gate_ablation.py",
        "experiments/gate_ablation/launch.sh"}
    return {p: sha(ROOT / p) for p in sorted(paths)}


def manifest():
    _, topk, conf, _ = deps()
    conf.freeze_ok()
    current = source_hashes()
    path = OUT / "method_manifest.json"
    with lock(OUT / ".manifest.lock", wait=True):
        if path.exists():
            rec = read(path)
            require(rec["source_sha256"] == current, "Source/plan changed: do not mix runs")
        else:
            rec = {"source_sha256": current, "git": topk.wdec.git_info(), "seeds": SEEDS,
                   "datasets": DATASETS, "horizons": HORIZONS, "arms": ARMS,
                   "evidence": EVIDENCE, "test_set_accessed": False}
            write(path, rec)
    return sha(path)


def unit_name(ds, h):
    return f"{ds}_h{h}"


def result_path(ds, h, seed, ext="json"):
    return OUT / "_work" / unit_name(ds, h) / f"seed{seed}.{ext}"


def reference_paths(ds, h, seed):
    if h == 12 and ds in DATASETS[:3]:
        p = ROOT / "experiments/expert_pool_convergence_ec/_eval" / f"R4_{ds}"
        return p.with_suffix(".json"), None, p.with_suffix(".pt")
    base = (ROOT / "experiments/wide_top3_confirmation_long_training/_work" if h == 12 else
            ROOT / "experiments/long_horizon_protocol/units" / unit_name(ds, h) / "_work")
    return base / f"seed{seed}_{ds}.json", base / f"seed{seed}_{ds}.pt", base / f"comparators_{ds}.pt"


def ref_hashes(ds, h, seed):
    meta, full, comp = reference_paths(ds, h, seed)
    require(meta.exists(), f"Missing reference metadata: {meta}")
    if h == 192:
        require(full.exists() and comp.exists(), "H192 requires per-window reference and comparator files")
    if h == 12 and ds not in DATASETS[:3]:
        require(comp.exists(), "Long-training pool requires per-window comparators")
    paths = [meta, full, comp, comp.with_suffix(".json")]
    return {str(p.relative_to(ROOT).as_posix()): sha(p) for p in paths if p is not None and p.exists()}


def configure(ds, h):
    _, topk, conf, lh = deps()
    if h == 192:
        conf, settings = lh.configure(ds, h, scoring=True)
    else:
        if ds not in DATASETS[:3]:
            import experiments.wide_top3_confirmation_long_training.run_long_training as lt
            conf = lt.confirmation()
        require((topk.wdec.BATCH_SIZE, topk.wdec.STAT_CHUNK, topk.cf.BLOCK_LENGTH, topk.cf.PHASE_K) ==
                (lh.BATCH_12, lh.STAT_CHUNK_12, lh.BLOCK_12, lh.PHASE_12) == (32, 256, 24, 12),
                "Frozen H12 settings changed")
        settings = dict(horizon=12, stride=1, batch_size=32, stat_chunk=256, block_length=24, phase_k=12)
    # Redirection is in memory only. Never invoke conf.run_one/analyze or writers.
    return conf, dict(settings)


def load_unit(ds, h, conf):
    if h == 12 and ds in DATASETS[:3]:
        import experiments.expert_pool_convergence_ec.run_pool_evaluation as pe
        bundle = pe.pool_bundle("R4", ds)
        causality = pe.verify_router_train_out_of_sample(bundle.train_cache)
        _, chosen = pe.select_core_on_router_train(bundle.train_cache, bundle.std)
        core = list(chosen["experts"])
        cache_dir, checkpoints, _ = pe.pool_root("R4", ds)
    else:
        bundle, core, causality, chosen = conf.load_pool(ds)
        roots = conf.train.roots(ds)
        cache_dir, checkpoints = ROOT / roots["cache_dir"], ROOT / roots["checkpoint_root"]
    cache_paths = sorted(cache_dir.glob("router_train*cache.pt")) + sorted(cache_dir.glob("router_val*cache.pt"))
    require(len(cache_paths) == 2, "Ambiguous train/validation caches")
    for c in (bundle.train_cache, bundle.val_cache):
        require(int(c["forecast_horizon"]) == h and int(c["input_len"]) == 96, "Cache horizon/input mismatch")
    require(causality["result"] == "PASS", "Training forecasts are not out of sample")
    return bundle, core, causality, chosen, checkpoints, cache_paths


def checkpoint_hashes(ds, h, conf, root):
    if h == 12 and ds in DATASETS[:3]:
        got = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.glob("*/*/best_expert.pt"))}
    else:
        got = conf.checkpoint_hashes(ds)
    require(len(got) == conf.EXPERT_CHECKPOINTS == 15, "Expected all 15 expert checkpoints")
    return got


def static_fits(fc, affinity, target, mask, std, fallback, chunk, topk):
    """Only pooled OOF tensors enter this function. All reductions use float64."""
    import torch
    require(chunk > 0 and fc.ndim == 4 and fc.shape[-1] == 5, "Invalid OOF shape/chunk")
    require(fc.shape[:-1] == target.shape == mask.shape and fc.shape == affinity.shape, "OOF shapes differ")
    require(bool(torch.isfinite(fc).all() and torch.isfinite(affinity).all() and
                 torch.isfinite(target[mask.bool()]).all()), "Nonfinite observed OOF input")
    require(bool(torch.isfinite(std).all() and (std > 0).all()), "Invalid normalizer")
    shape = target.shape[1:]
    num = torch.zeros(shape, dtype=torch.float64)
    den = torch.zeros_like(num)
    errors = torch.zeros((*shape, 5), dtype=torch.float64)
    for lo in range(0, len(fc), chunk):
        sl = slice(lo, lo + chunk)
        f = fc[sl].double()
        m = mask[sl].bool()
        y = torch.where(m, target[sl].double(), 0.)
        c = (f * fallback.double()).sum(-1)
        r = (f * topk.topk_equal(affinity[sl], 3).double()).sum(-1)
        e = torch.where(m, c - y, 0.)
        d = torch.where(m, r - c, 0.)
        num -= (e * d).sum(0)
        den += d.square().sum(0)
        errors += torch.where(m[..., None], (f - y[..., None]).square(), 0.).sum(0)
    def solve(n, d):
        return torch.where(d > 0, n / d.clamp_min(torch.finfo(torch.float64).tiny), 0.).clamp(0, 1)
    hv = solve(num, den)
    invvar = std.double().reshape(1, -1).square().reciprocal()
    constant = solve((num * invvar).sum(), (den * invvar).sum())
    # Stable ascending order implements the predeclared candidate-order tie break.
    idx = torch.argsort(errors, dim=-1, stable=True)[..., :3]
    static_weights = torch.zeros_like(errors).scatter_(-1, idx, 1. / 3)
    sn, sd = torch.zeros_like(num), torch.zeros_like(den)
    for lo in range(0, len(fc), chunk):
        sl = slice(lo, lo + chunk)
        f, m = fc[sl].double(), mask[sl].bool()
        c = (f * fallback.double()).sum(-1)
        e = torch.where(m, c - target[sl].double(), 0.)
        d = torch.where(m, (f * static_weights).sum(-1) - c, 0.)
        sn -= (e * d).sum(0); sd += d.square().sum(0)
    return {"lambda_c": constant, "lambda_hv": hv, "lambda_static_all": solve(sn, sd),
            "static_weights": static_weights, "zero_denominator_cells": int((den == 0).sum()),
            "static_all_zero_denominator_cells": int((sd == 0).sum())}


def lambda_diagnostics(lam, hv):
    import torch
    x = lam.double()
    mean = x.mean(0)
    a, b = mean.flatten(), hv.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    scale = a.square().sum().sqrt() * b.square().sum().sqrt()
    return {"lambda_mean_full": mean, "variance_total": float(x.var(unbiased=False)),
            "variance_static": float(mean.var(unbiased=False)),
            "variance_input": float(x.var(dim=0, unbiased=False).mean()),
            "correlation_full_mean_static_hv": float((a * b).sum() / scale) if scale > 0 else None,
            "static_hv_at_zero": float((hv == 0).double().mean()),
            "static_hv_at_one": float((hv == 1).double().mean())}


def score_arms(bundle, hi, core, chunk):
    torch, topk, _, _ = deps()
    fb = topk.wide.fallback_vector(topk.CANDIDATES, core)
    weights = topk.weights_fn(fb, 3)
    fit = static_fits(hi["oof_fc"], hi["oof_aff"], hi["oof_target"], hi["oof_tmask"], bundle.std, fb, chunk, topk)
    captured = {}
    def capture(aff, lam):
        captured["lambda"] = lam.detach()  # last call in head_arm is the full validation tensor
        return weights(aff, lam)
    full = topk.head_arm(bundle, hi, capture, fb > 0)
    lam = captured["lambda"]
    require(lam.shape == hi["val_aff"].shape[:-1], "Did not capture final validation lambda")
    diagnostics = lambda_diagnostics(lam, fit["lambda_hv"])
    arms = {"full": full}
    for name, value in (("core_only", 0.), ("router_only", 1.), ("constant", fit["lambda_c"]),
                        ("static_hv", fit["lambda_hv"]), ("static_all", fit["lambda_static_all"])):
        gate = torch.as_tensor(value, dtype=hi["val_aff"].dtype).expand_as(lam)
        if name == "static_all":
            w = (1 - gate[..., None]) * fb + gate[..., None] * fit["static_weights"].to(gate.dtype)
        else:
            w = weights(hi["val_aff"], gate)
        met = topk.cf.metrics_of(bundle, (hi["val_fc"] * w).sum(-1))
        met.update(mean_lambda=float(gate.mean()), noncore_weight_share=float(w[..., fb == 0].sum(-1).mean()))
        arms[name] = met
    # Encode each unordered top-3 set as a bit mask; there are only 10 possibilities.
    counts = torch.zeros((32, *lam.shape[1:]), dtype=torch.int64)
    bits = (2 ** torch.arange(5)).view(1, 1, 1, -1)
    for lo in range(0, len(lam), chunk):
        code = ((topk.topk_equal(hi["val_aff"][lo:lo + chunk], 3) > 0) * bits).sum(-1)
        for k in range(32):
            counts[k] += (code == k).sum(0)
    diagnostics["routing_set_change_rate"] = float((1 - counts.max(0).values.double() / len(lam)).mean())
    return arms, fit, diagnostics


def compare_vector(actual, expected, tol):
    import torch
    require(actual.shape == expected.shape, "Reference per-window shape mismatch")
    require(bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()), "Nonfinite reference or result")
    delta = float((actual.double() - expected.double()).abs().max())
    return {"exact": torch.equal(actual, expected), "within_tol": delta <= tol, "max_abs_diff": delta}


def reproduction(ds, h, seed, arms, core):
    import torch
    meta, full, comp = reference_paths(ds, h, seed)
    rec = read(meta)
    require(rec["core"] == core, "Core differs from reference")
    require(rec.get("test_set_accessed") is False, "Reference test provenance missing/invalid")
    if full is None:
        rep = {"kind": "global_anchor_only_R4_first_wide_run", "within_tol": True, "exact": None}
    elif full.exists():
        ref = torch.load(full, map_location="cpu", weights_only=False)["wide_top3"]
        parts = {m: compare_vector(arms["full"][f"per_window_{m}"], ref[f"per_window_{m}"], 1e-3) for m in ("mae", "mse")}
        rep = {"kind": "per_window", "metrics": parts, "within_tol": all(v["within_tol"] for v in parts.values()),
               "exact": all(v["exact"] for v in parts.values())}
    else:
        delta = abs(arms["full"]["mae"] - rec["arms"]["wide_top3"]["mae"])
        rep = {"kind": "scalar_MAE_fallback", "abs_diff": delta, "within_tol": delta <= 1e-3, "exact": delta == 0}
    if full is None:
        delta = abs(arms["core_only"]["mae"] - rec["comparators"]["equal3"]["mae"])
        eq = {"within_tol": delta <= 1e-5, "kind": "scalar_MAE", "abs_diff": delta}
        if comp.exists():
            ref = torch.load(comp, map_location="cpu", weights_only=False)
            eq = compare_vector(arms["core_only"]["per_window_mae"], ref["equal3_per_window_mae"], 1e-5)
    else:
        ref = torch.load(comp, map_location="cpu", weights_only=False)["equal3"]
        eq = {"within_tol": all(compare_vector(arms["core_only"][f"per_window_{m}"], ref[f"per_window_{m}"], 1e-5)["within_tol"] for m in ("mae", "mse"))}
    return rep, eq


def scorer_checks(bundle, sc, wdec):
    import torch
    expected = torch.zeros_like(sc["oof_mask"])
    bounds = wdec.oof_bounds(len(expected))
    starts = bundle.train_cache["absolute_window_starts"]
    horizon = int(bundle.val_cache["forecast_horizon"])
    causal = len(sc["folds"]) == len(bounds) == 4
    for (lo, hi), f in zip(bounds, sc["folds"]):
        expected[lo:hi] = True
        legal = wdec.legal_fit_mask(starts, horizon, int(starts[lo]))
        causal &= (f["eval_lo"] == lo and f["eval_hi"] == hi and f["eval_origin"] == int(starts[lo])
                   and f["causal"] and len(legal) == f["num_legal_fit_windows"] and len(legal) > 0
                   and f["latest_fit_target_end"] == int((starts[legal] + horizon).max()) <= int(starts[lo]))
    return {"all_folds_causal": bool(causal),
            "oof_mask_covers_folds": bool(sc["oof_mask_covers_folds"] and torch.equal(expected, sc["oof_mask"])),
            "scorer_tensors_finite": bool(sc["tensors_finite"]),
            "oof_targets_observed_before_validation": bool((starts[expected] + horizon <= bundle.val_cache["absolute_window_starts"][0]).all())}


def finite_tree(value):
    import torch
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    return True


def anchor_valid(manifest_id):
    p = OUT / "anchor_check.json"
    if not p.exists():
        return False
    rec = read(p)
    return (rec.get("ok") is True and rec.get("exact") is True and rec.get("manifest_sha256") == manifest_id
            and bool(rec.get("reference_hashes"))
            and all((ROOT / p).exists() and sha(ROOT / p) == v for p, v in rec["reference_hashes"].items()))


def run_anchor():
    torch, topk, conf, lh = deps()
    mid = manifest()
    require((topk.wdec.BATCH_SIZE, topk.wdec.STAT_CHUNK, topk.cf.BLOCK_LENGTH, topk.cf.PHASE_K) ==
            (lh.BATCH_12, lh.STAT_CHUNK_12, lh.BLOCK_12, lh.PHASE_12) == (32, 256, 24, 12), "Anchor settings differ")
    refs = [topk.scores_path("ETTh1", 7), topk.result_path("ETTh1", 7, "pt")]
    require(all(p.exists() for p in refs), "Anchor requires saved ETTh1 seed7 scores and top3 results")
    before = {str(p.relative_to(ROOT)): sha(p) for p in refs}
    if anchor_valid(mid):
        require(read(OUT / "anchor_check.json")["reference_hashes"] == before, "Anchor references changed")
        print("[gate] anchor already EXACT")
        return
    require(not (OUT / "anchor_check.json").exists(), "An earlier failed/stale anchor is preserved; investigate first")
    topk.wdec.SCORER_SEED = 7
    base = topk.fhv.LOADERS["ETTh1"]()
    checks_before = topk.wide.expert_hashes("ETTh1", topk.CANDIDATES)
    require(len(checks_before) >= 5, "Missing anchor expert checkpoints")
    wb = topk.wide.widened(base, topk.CANDIDATES)
    sc = torch.load(refs[0], map_location="cpu", weights_only=False)
    require(sc.get("dataset") == "ETTh1" and sc.get("seed") == 7 and sc.get("candidates") == list(topk.CANDIDATES), "Anchor scores metadata mismatch")
    hi = topk.wide.head_inputs(wb, sc)
    arms, _, _ = score_arms(wb, hi, list(base.core_names), topk.wdec.STAT_CHUNK)
    ref = torch.load(refs[1], map_location="cpu", weights_only=False)["top3"]["per_window_mae"]
    exact = torch.equal(arms["full"]["per_window_mae"], ref)
    fb = topk.wide.fallback_vector(topk.CANDIDATES, base.core_names)
    c = topk.cf.metrics_of(wb, (hi["val_fc"] * fb).sum(-1))["per_window_mae"]
    r = topk.cf.metrics_of(wb, (hi["val_fc"] * topk.topk_equal(hi["val_aff"], 3)).sum(-1))["per_window_mae"]
    flags = {"exact": exact, "core_exact": torch.equal(c, arms["core_only"]["per_window_mae"]),
             "router_exact": torch.equal(r, arms["router_only"]["per_window_mae"]),
             "references_unchanged": before == {str(p.relative_to(ROOT)): sha(p) for p in refs},
             "checkpoints_unchanged": checks_before == topk.wide.expert_hashes("ETTh1", topk.CANDIDATES),
             **scorer_checks(wb, sc, topk.wdec)}
    conf.freeze_ok()
    require(source_hashes() == read(OUT / "method_manifest.json")["source_sha256"], "Source changed during anchor")
    write(OUT / "anchor_check.json", flags | {"ok": all(flags.values()), "manifest_sha256": mid,
          "reference_hashes": before, "dataset": "ETTh1", "seed": 7, "test_set_accessed": False})
    require(all(flags.values()), "Global anchor failed; no units may run")
    print("[gate] anchor EXACT; integrity PASS [values withheld until analysis]", flush=True)


def run_seed(ds, h, seed, conf, settings):
    torch, topk, _, lh = deps()
    t0 = time.monotonic()
    mid = manifest()
    require(anchor_valid(mid), "Run --anchor first; it must be EXACT for this source manifest")
    references = ref_hashes(ds, h, seed)
    path = result_path(ds, h, seed)
    if path.exists():
        rec = read(path)
        require(rec.get("manifest_sha256") == mid and rec.get("reference_hashes") == references,
                "Completed job source/reference identity changed")
        require(rec.get("status") == "done" and rec.get("integrity", {}).get("ok"), "Existing INVALID result is preserved")
        require(sha(path.with_suffix(".pt")) == rec["tensor_sha256"], "Completed tensor file changed")
        for p, expected in rec["input_hashes"].items():
            require(sha(ROOT / p) == expected, "Completed job input changed")
        print(f"[gate] {unit_name(ds, h)} seed {seed}: already done; identities checked", flush=True)
        return
    bundle, core, causality, chosen, ckroot, caches = load_unit(ds, h, conf)
    require(read(reference_paths(ds, h, seed)[0])["core"] == core, "Core differs from reference")
    before = checkpoint_hashes(ds, h, conf, ckroot)
    inputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in caches}
    inputs.update({(ckroot / p).relative_to(ROOT).as_posix(): v for p, v in before.items()})
    n = int(bundle.val_cache["num_windows"])
    settings = dict(settings)
    if h == 12:
        settings.update(evaluation_windows=n, independent_blocks=n / 24, interval_eligible=n / 24 >= lh.MIN_BLOCKS)
        require(settings["interval_eligible"], "H12 unit unexpectedly underpowered")
    checks = {"freeze_matches": True, "global_anchor_exact": True, "router_train_out_of_sample": causality["result"] == "PASS",
              "core_matches_reference": True, "checkpoint_count_15": len(before) == 15,
              "static_fits_oof_only": True, "test_set_not_accessed": True}
    try:
        topk.wdec.SCORER_SEED = seed
        wb = topk.wide.widened(bundle, topk.CANDIDATES)
        sc = topk.wide.train_scorer_bank_protocol(wb, seed)
        checks.update(scorer_checks(wb, sc, topk.wdec))
        require(all(checks.values()), "Scorer integrity failed")
        hi = topk.wide.head_inputs(wb, sc)
        require(finite_tree(hi), "Nonfinite head inputs")
        arms, fit, diagnostics = score_arms(wb, hi, core, topk.wdec.STAT_CHUNK)
        rep, eq = reproduction(ds, h, seed, arms, core)
        checks.update(full_reproduces_reference=rep["within_tol"], core_matches_equal3=eq["within_tol"],
                      tensors_finite=finite_tree((arms, fit, diagnostics)),
                      settings_match=(topk.wdec.BATCH_SIZE == settings["batch_size"] and topk.wdec.STAT_CHUNK == settings["stat_chunk"]
                                      and topk.cf.BLOCK_LENGTH == settings["block_length"] and topk.cf.PHASE_K == settings["phase_k"]))
    finally:
        # Also checked after a fitting exception; neither input group is ours to modify.
        checks["checkpoints_unchanged"] = before == checkpoint_hashes(ds, h, conf, ckroot)
        checks["references_unchanged"] = references == ref_hashes(ds, h, seed)
        checks["inputs_unchanged"] = all(sha(ROOT / p) == digest for p, digest in inputs.items())
        checks["sources_unchanged"] = source_hashes() == read(OUT / "method_manifest.json")["source_sha256"]
        write(path.with_suffix(".integrity.json"), checks)
    checks["ok"] = all(checks.values())
    payload = {"arms": {a: {f"per_window_{m}": arms[a][f"per_window_{m}"] for m in ("mae", "mse")} for a in ARMS},
               "fits": fit, "lambda_mean_full": diagnostics.pop("lambda_mean_full"),
               "absolute_window_starts": bundle.val_cache["absolute_window_starts"],
               "oof_absolute_window_starts": bundle.train_cache["absolute_window_starts"][sc["oof_mask"]]}
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp); tmp.replace(path.with_suffix(".pt"))
    rec = {"dataset": ds, "horizon": h, "seed": seed, "core": core, "candidates": list(topk.CANDIDATES),
           "settings": settings, "core_selection": chosen, "integrity": checks, "reproduction": rep, "equal3_check": eq,
           "scorer_folds": sc["folds"], "manifest_sha256": mid, "reference_hashes": references, "input_hashes": inputs,
           "tensor_sha256": sha(path.with_suffix(".pt")), "diagnostics": diagnostics,
           "lambda_c": float(fit["lambda_c"]), "zero_denominator_cells": fit["zero_denominator_cells"],
           "arms": {a: {k: v for k, v in arms[a].items() if not torch.is_tensor(v)} for a in ARMS},
           "status": "done" if checks["ok"] else "invalid", "elapsed_sec": time.monotonic() - t0,
           "test_set_accessed": False, "evidence": EVIDENCE}
    write(path, topk.wdec.jsonable(rec))
    require(checks["ok"], "Unit INVALID; preserved for investigation")
    print(f"[gate] {unit_name(ds, h)} seed {seed}: integrity PASS; reproduction {rep['kind']} "
          f"{'EXACT' if rep['exact'] else 'WITHIN_TOL/global anchor'} [values withheld until analysis]", flush=True)


def gate_verdict(mae_interval, mse_interval, mse_across):
    if mae_interval in ("UNDERPOWERED", "INCOMPLETE", "INVALID"):
        return mae_interval
    if mae_interval == "HOLDS":
        return "HELPS" if mse_interval != "REVERSED" and mse_across != "CONSISTENT_DISADVANTAGE" else "HELPS_MAE_MSE_COST"
    return {"PARTIAL": "MIXED", "NOT_SHOWN": "NO_GAIN", "REVERSED": "STATIC_BETTER"}[mae_interval]


def job_key(ds, h, seed):
    return f"{unit_name(ds, h)}_seed{seed}"


def job_status(ds, h, seed):
    p = result_path(ds, h, seed)
    if p.exists():
        return read(p).get("status", "invalid")
    check = p.with_suffix(".integrity.json")
    if check.exists() and not all(read(check).values()):
        return "invalid"
    state = OUT / "queue_state.json"
    return read(state).get("jobs", {}).get(job_key(ds, h, seed), {}).get("status", "pending") if state.exists() else "pending"


def status():
    total = 0
    for h in HORIZONS:
        for ds in DATASETS:
            states = [job_status(ds, h, s) for s in SEEDS]
            final = sum(s in ("done", "failed", "invalid") for s in states)
            total += final
            print(f"[gate] {unit_name(ds, h)}: {states.count('done')} done, {states.count('failed')} failed, "
                  f"{states.count('invalid')} invalid, {5-final} pending/running")
    print(f"[gate] {total}/70 jobs final [values withheld until analysis]")


def analyze():
    torch, topk, conf, lh = deps()
    mid = manifest()
    require(anchor_valid(mid), "Global anchor missing/stale")
    pending = [job_key(d, h, s) for h in HORIZONS for d in DATASETS for s in SEEDS
               if job_status(d, h, s) not in ("done", "failed", "invalid")]
    require(not pending, f"Analysis refused: {len(pending)} jobs have not finished")
    units, integrity, contrasts = {}, {}, []
    for h in HORIZONS:
        for ds in DATASETS:
            name = unit_name(ds, h)
            states = [job_status(ds, h, s) for s in SEEDS]
            if any(s != "done" for s in states):
                integrity[name] = {"ok": False, "status": "invalid" if "invalid" in states else "failed"}
                continue
            recs = {s: read(result_path(ds, h, s)) for s in SEEDS}
            runs = {s: torch.load(result_path(ds, h, s, "pt"), map_location="cpu", weights_only=False) for s in SEEDS}
            r0, x0 = recs[7], runs[7]
            flags = {"jobs_valid": all(r["integrity"]["ok"] and r["test_set_accessed"] is False for r in recs.values()),
                     "same_source": all(r["manifest_sha256"] == mid for r in recs.values()),
                     "tensors_match_hashes": all(sha(result_path(ds, h, s, "pt")) == recs[s]["tensor_sha256"] for s in SEEDS),
                     "references_unchanged": all(ref_hashes(ds, h, s) == recs[s]["reference_hashes"] for s in SEEDS),
                     "same_inputs": all(r["input_hashes"] == r0["input_hashes"] for r in recs.values()),
                     "same_core_settings": all(r["core"] == r0["core"] and r["settings"] == r0["settings"] for r in recs.values()),
                     "same_origins": all(torch.equal(x["absolute_window_starts"], x0["absolute_window_starts"]) and
                                         torch.equal(x["oof_absolute_window_starts"], x0["oof_absolute_window_starts"]) for x in runs.values()),
                     "static_all_identical": all(all(torch.equal(x["arms"]["static_all"][f"per_window_{m}"], x0["arms"]["static_all"][f"per_window_{m}"]) for m in ("mae", "mse"))
                                                 and torch.equal(x["fits"]["static_weights"], x0["fits"]["static_weights"])
                                                 and torch.equal(x["fits"]["lambda_static_all"], x0["fits"]["lambda_static_all"]) for x in runs.values())}
            integrity[name] = flags | {"ok": all(flags.values()), "status": "done" if all(flags.values()) else "invalid"}
            if not all(flags.values()):
                continue
            settings = r0["settings"]
            topk.cf.BLOCK_LENGTH, topk.cf.PHASE_K = settings["block_length"], settings["phase_k"]
            avg = {a: {m: torch.stack([runs[s]["arms"][a][f"per_window_{m}"] for s in SEEDS]).mean(0) for m in ("mae", "mse")} for a in ARMS}
            comps = list(LADDER)
            comp_path = reference_paths(ds, h, 7)[2]
            note = None
            if comp_path.exists():
                saved = torch.load(comp_path, map_location="cpu", weights_only=False)
                if h == 12 and ds in DATASETS[:3]:
                    avg["bates_all5"] = {"mae": saved["bates_all5_per_window_mae"]}
                    note = "R4 archived Bates has MAE only."
                else:
                    avg["bates_all5"] = {m: saved["bates_all5"][f"per_window_{m}"] for m in ("mae", "mse")}
                comps += [(a, "bates_all5") for a in ARMS]
            else:
                note = "R4 per-window Bates absent; descriptive comparison omitted."
            cells = {}
            for a, b in comps:
                key = f"{a}-{b}"
                cells[key] = {}
                for m in ("mae", "mse"):
                    if m not in avg[b]:
                        continue
                    require(avg[a][m].shape == avg[b][m].shape, "Comparator shape mismatch")
                    paired = topk.cf.paired(avg[a][m], avg[b][m])
                    deltas = [float(runs[s]["arms"][a][f"per_window_{m}"].mean() -
                                    (runs[s]["arms"][b][f"per_window_{m}"].mean() if b in ARMS else avg[b][m].mean())) for s in SEEDS]
                    c = lh.contrast(paired["block"], float(avg[b][m].mean()), settings["interval_eligible"], deltas)
                    cells[key][m] = c | {"paired": paired, "per_seed_deltas": deltas}
                    contrasts.append({"dataset": ds, "horizon": h, "contrast": key, "metric": m, **c})
            units[name] = {"core": r0["core"], "settings": settings, "contrasts": cells,
                           "arms": {a: {m: float(v.mean()) for m, v in mm.items()} for a, mm in avg.items()},
                           "diagnostics_by_seed": {s: recs[s]["diagnostics"] | {"lambda_c": recs[s]["lambda_c"],
                              "arms": {a: {k: recs[s]["arms"][a][k] for k in ("mean_lambda", "noncore_weight_share")} for a in ARMS}} for s in SEEDS},
                           "comparator_note": note}
    readings = {}
    for h in HORIZONS:
        missing = [d for d in DATASETS if integrity[unit_name(d, h)]["status"] == "failed"]
        invalid = [d for d in DATASETS if integrity[unit_name(d, h)]["status"] == "invalid"]
        result = {}
        for a, b in list(LADDER) + [(a, "bates_all5") for a in ARMS]:
            key = f"{a}-{b}"
            result[key] = {}
            for m in ("mae", "mse"):
                cells = {d: units[unit_name(d, h)]["contrasts"][key][m] for d in DATASETS
                         if unit_name(d, h) in units and m in units[unit_name(d, h)]["contrasts"].get(key, {})}
                absent = sorted(set(missing) | (set(DATASETS) - set(cells) - set(invalid)))
                result[key][m] = {"interval": lh.interval_reading(cells, absent, invalid),
                                  "across": lh.across_reading(cells, absent, invalid)}
        primary = result["full-static_hv"]
        readings[h] = {"verdict": gate_verdict(primary["mae"]["interval"]["label"], primary["mse"]["interval"]["label"], primary["mse"]["across"]["label"]),
                       "contrasts": result}
    write(OUT / "integrity_checks.json", integrity | {"test_set_accessed": False})
    write(OUT / "results.json", topk.wdec.jsonable({"evidence": EVIDENCE, "horizons": readings, "units": units}))
    topk.wdec.write_csv_rows(OUT / "contrasts.csv", contrasts)
    lines = ["# Gate ablation", "", EVIDENCE, "", "Primary contrast: full minus static_hv. Negative favours full.", ""]
    for h, r in readings.items():
        lines += [f"## Horizon {h}: {r['verdict']}", "", "| Contrast | Metric | Interval reading | Across reading | Exact Wilcoxon p |", "|---|---|---|---|---|"]
        for key, mm in r["contrasts"].items():
            for m, v in mm.items():
                lines.append(f"| {key} | {m} | {v['interval']['label']} | {v['across']['label']} | {v['across']['wilcoxon_p']} |")
        lines += ["", "Only HELPS licenses a claim that input-dependent gating improves on the best static gate at this horizon.", ""]
    for name, u in units.items():
        lines += [f"## {name}", "", "| Arm | MAE | MSE |", "|---|---:|---:|"]
        for a, v in u["arms"].items():
            lines.append(f"| {a} | {v.get('mae', 'unavailable')} | {v.get('mse', 'unavailable')} |")
        lines += ["", u["comparator_note"] or "Both metrics available for Bates.", "", "Diagnostics (by seed; population variance split):", "", "```json",
                  json.dumps(u["diagnostics_by_seed"], indent=2), "```", ""]
    lines += ["## Integrity", "", "```json", json.dumps(integrity, indent=2), "```", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:6]))
    for h, r in readings.items():
        print(f"[gate] horizon {h}: {r['verdict']}")


def runtime_estimate(ds, h, seed):
    p = reference_paths(ds, h, seed)[0]
    require(p.exists(), f"Missing previous runtime record: {p}")
    return float(read(p)["elapsed_sec"])


def queue():
    """Longest previous runtimes first; one subprocess per job, two slots/GPU."""
    mid = manifest()
    require(anchor_valid(mid), "Queue refused: run --anchor first")
    info = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"], text=True)
    gpus = [x.strip() for x in info.splitlines() if x.strip()]
    require(len(gpus) == 6 and len(set(gpus)) == 6, "Predeclared queue requires six GPUs")
    state_path = OUT / "queue_state.json"
    state = read(state_path) if state_path.exists() else {"manifest_sha256": mid, "jobs": {}}
    require(state["manifest_sha256"] == mid, "Queue manifest changed")
    # Token permits only this queue's children to bypass the exclusive campaign lock.
    import uuid
    state["token"] = uuid.uuid4().hex
    jobs = state["jobs"]
    order = sorted([(d, h, s) for h in HORIZONS for d in DATASETS for s in SEEDS],
                   key=lambda x: runtime_estimate(*x), reverse=True)
    for ds, h, s in order:
        k = job_key(ds, h, s)
        old = jobs.get(k, {})
        if result_path(ds, h, s).exists():
            r = read(result_path(ds, h, s))
            require(r["manifest_sha256"] == mid and sha(result_path(ds, h, s, "pt")) == r["tensor_sha256"], "Stale/corrupt completed job")
            require(r["reference_hashes"] == ref_hashes(ds, h, s), "Completed reference changed")
            jobs[k] = old | {"status": r["status"]}
        elif old.get("status") == "running":
            # Holding the campaign lock proves the prior queue is gone. Its workers
            # keep per-job locks; a still-live child causes refusal, never duplication.
            with lock(result_path(ds, h, s, "lock")):
                jobs[k] = old | {"status": "pending" if old.get("attempts", 0) < 2 else "failed"}
        elif job_status(ds, h, s) == "invalid":
            jobs[k] = old | {"status": "invalid"}
        else:
            jobs[k] = old or {"status": "pending", "attempts": 0}
    write(state_path, state)
    logs = ROOT / "logs/gate_ablation"
    logs.mkdir(parents=True, exist_ok=True)
    running = {}
    try:
        while True:
            for slot, (proc, stream, k, ds, h, s) in list(running.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                stream.close()
                final = job_status(ds, h, s)
                jobs[k].update(returncode=rc, status=("done" if rc == 0 and final == "done" else
                               "invalid" if final == "invalid" else "pending" if jobs[k]["attempts"] < 2 else "failed"))
                jobs[k].setdefault("attempt_history", []).append({"attempt": jobs[k]["attempts"], "returncode": rc,
                                                                 "log": jobs[k]["log"]})
                del running[slot]
                write(state_path, state)
            pending = [x for x in order if jobs[job_key(*x)]["status"] == "pending"]
            for gpu in gpus:
                for share in range(2):
                    slot = (gpu, share)
                    if slot in running or not pending:
                        continue
                    ds, h, s = pending.pop(0)
                    k = job_key(ds, h, s)
                    jobs[k]["attempts"] = jobs[k].get("attempts", 0) + 1
                    logfile = logs / f"{k}_attempt{jobs[k]['attempts']}.log"
                    stream = logfile.open("a", encoding="utf-8")
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
                               OPENBLAS_NUM_THREADS="8", GATE_QUEUE_TOKEN=state["token"], PYTHONUNBUFFERED="1")
                    cmd = [sys.executable, str(Path(__file__).resolve()), "--unit", ds, "--horizon", str(h), "--seeds", str(s)]
                    jobs[k].update(status="running", gpu=gpu, log=str(logfile.relative_to(ROOT)))
                    write(state_path, state)
                    try:
                        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                                stdout=stream, stderr=subprocess.STDOUT)
                    except BaseException:
                        stream.close()
                        jobs[k]["status"] = "pending" if jobs[k]["attempts"] < 2 else "failed"
                        write(state_path, state)
                        raise
                    jobs[k]["pid"] = proc.pid
                    running[slot] = (proc, stream, k, ds, h, s)
                    write(state_path, state)
                    print(f"[gate] started {k} on GPU {gpu}, attempt {jobs[k]['attempts']}", flush=True)
            if not running and not any(j["status"] == "pending" for j in jobs.values()):
                break
            time.sleep(5)
    finally:
        for proc, stream, *_ in running.values():
            if proc.poll() is None:
                proc.terminate()
            proc.wait()
            stream.close()
    analyze()
    # Keep experiment artifacts under its own directory (logs are the declared exception).
    bundle = OUT / "results_bundle_gate_ablation.tar"
    tmp = bundle.with_suffix(".tar.tmp")
    with tarfile.open(tmp, "w") as archive:
        for p in sorted(OUT.rglob("*")):
            if p.is_file() and p.suffix in (".json", ".pt", ".md", ".csv", ".py", ".sh"):
                archive.add(p, arcname=p.relative_to(ROOT).as_posix())
        for p in sorted(logs.glob("*.log")):
            archive.add(p, arcname=p.relative_to(ROOT).as_posix())
    tmp.replace(bundle)
    print("[gate] queue settled; report and results_bundle_gate_ablation.tar written", flush=True)


def selftest():
    """No cache loads, fitting of learned models, GPU operations or experiment writes."""
    import torch
    import experiments.topk_wide_router_ec.run_topk_wide_router_ec as topk
    torch.set_num_threads(2)
    gen = torch.Generator(device="cpu").manual_seed(20261017)
    tests = []
    def passed(name):
        tests.append(name)
        print(f"PASS {name}")
    fc = torch.randn(23, 3, 4, 5, generator=gen, dtype=torch.float64)
    aff = torch.rand(fc.shape, generator=gen, dtype=torch.float64)
    fb = torch.tensor([1/3, 1/3, 1/3, 0, 0], dtype=torch.float64)
    c, r = (fc * fb).sum(-1), (fc * topk.topk_equal(aff, 3)).sum(-1)
    target = c + .37 * (r - c) + .1 * torch.randn(c.shape, generator=gen, dtype=torch.float64)
    mask = torch.rand(c.shape, generator=gen) > .2
    std = torch.tensor([.2, 1., 3., 10.], dtype=torch.float64)
    fit = static_fits(fc, aff, target, mask, std, fb, 7, topk)
    grid = torch.linspace(0, 1, 10001, dtype=torch.float64)
    # Independent brute-force objective, in small grid chunks to bound memory.
    losses = []
    for g in grid.split(100):
        err = (c[None] + g[:, None, None, None] * (r - c)[None] - target[None]) / std
        losses.append((err.square() * mask).sum((1, 2, 3)))
    best = grid[torch.cat(losses).argmin()]
    require(abs(float(best - fit["lambda_c"])) <= 1e-4, "Constant grid mismatch")
    for i in range(3):
        for j in range(4):
            errors = c[:, i, j][None] + grid[:, None] * (r[:, i, j] - c[:, i, j])[None] - target[:, i, j][None]
            best = grid[(errors.square() * mask[:, i, j]).sum(1).argmin()]
            require(abs(float(best - fit["lambda_hv"][i, j])) <= 1e-4, "HxV grid mismatch")
    passed("closed forms match brute-force grid within 1e-4 (constant and HxV)")
    fn = topk.weights_fn(fb, 3)
    require(torch.equal((fc * fn(aff, torch.zeros_like(c))).sum(-1), c), "Zero endpoint differs")
    require(torch.equal((fc * fn(aff, torch.ones_like(c))).sum(-1), r), "One endpoint differs")
    passed("lambda 0/1 exactly reproduce fallback/router using frozen weight code")
    lam = torch.rand(c.shape, generator=gen, dtype=torch.float64)
    d = lambda_diagnostics(lam, fit["lambda_hv"])
    require(abs(d["variance_total"] - d["variance_static"] - d["variance_input"]) < 1e-12, "Variance split mismatch")
    passed("population variance split adds up")
    tied = fc[..., :1].expand_as(fc).clone()
    tie = static_fits(tied, aff, target, mask, std, fb, 7, topk)
    require(torch.equal(tie["static_weights"], fb.expand_as(tie["static_weights"])), "Static tie order differs")
    passed("static top-3 ties use candidate order")
    # Masked targets must not influence fit, even when NaN; all-masked cells use zero.
    masked = target.clone(); masked[~mask] = float("nan")
    fit2 = static_fits(fc, aff, masked, mask, std, fb, 7, topk)
    require(all(torch.equal(fit[k], fit2[k]) for k in ("lambda_c", "lambda_hv", "static_weights", "lambda_static_all")), "Masked target leakage")
    none = static_fits(fc, aff, masked, torch.zeros_like(mask), std, fb, 7, topk)
    require(none["lambda_c"] == 0 and not bool(none["lambda_hv"].any()) and none["zero_denominator_cells"] == 12, "Zero denominator rule failed")
    passed("masked targets ignored and zero denominators yield lambda 0")
    fit3 = static_fits(fc, aff, target, mask, std, fb, 23, topk)
    require(torch.allclose(fit["lambda_hv"], fit3["lambda_hv"], atol=1e-12, rtol=0), "Chunk reduction changed fit")
    other = static_fits(fc, 1 - aff, target, mask, std, fb, 7, topk)
    require(torch.equal(fit["static_weights"], other["static_weights"]) and
            torch.equal(fit["lambda_static_all"], other["lambda_static_all"]), "Static-all depends on scorer")
    sr = (fc * fit["static_weights"]).sum(-1)
    for i in range(3):
        for j in range(4):
            errors = c[:, i, j][None] + grid[:, None] * (sr[:, i, j] - c[:, i, j])[None] - target[:, i, j][None]
            best = grid[(errors.square() * mask[:, i, j]).sum(1).argmin()]
            require(abs(float(best - fit["lambda_static_all"][i, j])) <= 1e-4, "Static-all grid mismatch")
    passed("static-all is scorer-independent, grid-optimal; chunked fits agree")
    for value, expected in ((-1., 0.), (2., 1.)):
        clamp = static_fits(fc, aff, c + value * (r - c), mask, std, fb, 7, topk)
        require(abs(float(clamp["lambda_c"]) - expected) < 1e-12, "Constant clipping failed")
        require(torch.allclose(clamp["lambda_hv"], torch.full_like(clamp["lambda_hv"], expected), atol=1e-12), "HxV clipping failed")
    passed("closed forms clip to both boundaries")
    for labels, expected in ((("HOLDS", "NOT_SHOWN", "NOT_ESTABLISHED"), "HELPS"),
                             (("HOLDS", "REVERSED", "NOT_ESTABLISHED"), "HELPS_MAE_MSE_COST"),
                             (("HOLDS", "HOLDS", "CONSISTENT_DISADVANTAGE"), "HELPS_MAE_MSE_COST"),
                             (("NOT_SHOWN", "HOLDS", "CONSISTENT_ADVANTAGE"), "NO_GAIN"),
                             (("PARTIAL", "HOLDS", "NOT_ESTABLISHED"), "MIXED"),
                             (("REVERSED", "HOLDS", "NOT_ESTABLISHED"), "STATIC_BETTER")):
        require(gate_verdict(*labels) == expected, "Verdict mapping differs")
    for label in ("INVALID", "INCOMPLETE", "UNDERPOWERED"):
        require(gate_verdict(label, label, label) == label, "Status did not propagate")
    passed("all predeclared verdict branches")
    # Exercise six-arm plumbing without training a scorer or head. The real
    # head_arm calls the supplied weights during fitting and once on validation.
    from types import SimpleNamespace
    from unittest.mock import patch
    val_fc, val_aff = fc[:8].float(), aff[:8].float()
    val_y, val_mask = target[:8].float(), mask[:8]
    fake_bundle = SimpleNamespace(std=std.float(), val_cache={"targets": val_y, "target_masks": val_mask})
    hi = {"oof_fc": fc.float(), "oof_aff": aff.float(), "oof_target": target.float(), "oof_tmask": mask,
          "val_fc": val_fc, "val_aff": val_aff}
    planted_lambda = torch.linspace(0, 1, 8)[:, None, None].expand(8, 3, 4)
    def fake_head(bundle, inputs, weights, core_mask):
        weights(inputs["oof_aff"][:2], torch.zeros(2, 3, 4))
        w = weights(inputs["val_aff"], planted_lambda)
        met = topk.cf.metrics_of(bundle, (inputs["val_fc"] * w).sum(-1))
        return met | {"mean_lambda": float(planted_lambda.mean()),
                      "noncore_weight_share": float(w[..., ~core_mask].sum(-1).mean())}
    # Fail if any future self-test path tries to read a cache/checkpoint.
    with patch.object(torch, "load", side_effect=AssertionError("Selftest may not load data")), patch.object(topk, "head_arm", fake_head):
        arms, fits, diag = score_arms(fake_bundle, hi, list(topk.CANDIDATES[:3]), 7)
        require(set(arms) == set(ARMS) and finite_tree((arms, fits, diag)), "Six-arm scoring failed")
        require(torch.equal(diag["lambda_mean_full"], planted_lambda.double().mean(0)), "Captured wrong lambda")
        fake_bundle.val_cache["targets"] = val_y + 100
        _, changed, _ = score_arms(fake_bundle, hi, list(topk.CANDIDATES[:3]), 7)
        require(all(torch.equal(fits[k], changed[k]) for k in ("lambda_c", "lambda_hv", "lambda_static_all", "static_weights")), "Validation labels leaked into static fits")
    passed("six-arm plumbing, final lambda capture, validation-label isolation (mock head, no training)")
    print(f"SELFTEST PASS: {len(tests)} groups; synthetic CPU tensors only; no experiment run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--anchor", action="store_true")
    mode.add_argument("--unit", choices=DATASETS)
    mode.add_argument("--queue", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    parser.add_argument("--horizon", type=int, choices=HORIZONS)
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS))
    args = parser.parse_args()
    if args.selftest:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return selftest()
    if args.status:
        return status()
    if bool(args.unit) != bool(args.horizon):
        parser.error("--horizon is required with --unit and only valid with --unit")
    require(len(args.seeds) == len(set(args.seeds)), "Duplicate seeds")
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    token = os.environ.get("GATE_QUEUE_TOKEN")
    child = bool(token and (OUT / "queue_state.json").exists() and read(OUT / "queue_state.json").get("token") == token)
    require(not child or bool(args.unit), "Queue children must score one unit")
    guard = contextlib.nullcontext() if child else lock(OUT / ".campaign.lock", shared=bool(args.unit))
    with guard:
        if args.anchor:
            run_anchor()
        elif args.queue:
            queue()
        elif args.analyze:
            analyze()
        else:
            conf, settings = configure(args.unit, args.horizon)
            for seed in args.seeds:
                with lock(result_path(args.unit, args.horizon, seed, "lock")):
                    run_seed(args.unit, args.horizon, seed, conf, settings)


if __name__ == "__main__":
    main()
