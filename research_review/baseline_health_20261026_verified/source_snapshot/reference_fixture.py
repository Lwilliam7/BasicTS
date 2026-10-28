"""October 20 synthetic fixture with audit instrumentation; deployed files are unchanged."""
from types import SimpleNamespace
EXPERTS = ("DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN")

SEED = 20261020

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

def execute(method, mod, bundle, dataset, *, health=False):
    """Call the actual port pipeline; capture its selected validation winner for health checks."""
    import numpy as np
    import torch
    picks = []
    if method == "fame":
        mod.load_bundle = lambda _d: bundle
        if not mod.fp_path(dataset).exists():
            mod.compute_fingerprints(dataset)
        original_evaluate = mod.evaluate_val
        def capture(router, scaler, fp, cache, std, top_r, delta, experts):
            result = original_evaluate(router, scaler, fp, cache, std, top_r, delta, experts)
            # The secondary top-1 model may select a different tau. Capture the
            # primary top-2 model whose losses are returned below.
            if top_r == mod.TOP_R_PRIMARY:
                p = mod.predict_probs(router, scaler, fp)
                picks.append(p.argmax(1).numpy())
            return result
        if health:
            mod.evaluate_val = capture
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
