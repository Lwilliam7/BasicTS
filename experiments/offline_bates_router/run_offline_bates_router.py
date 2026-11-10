"""Offline replacement for the learned gate. See PREDECLARATION.md in this directory.

LambdaHead is removed. The routed top-3 forecast r is blended into a fitted Bates all-5 combination b as
b + lambda * (r - b). lambda is closed form on pooled OOF rows: one constant or one value per horizon-variable
cell, the form chosen by fitting on the first 70% of OOF windows and scoring on the last 30%. An optional
risk-controlled switch keeps routing only in variables whose late-OOF gain has a 95% block-bootstrap lower
bound above zero. Nothing realised in the evaluation period is used at forecast time.

The Bates fallback on OOF rows is refitted inside each OOF fold on windows whose targets end before that
fold starts, so lambda is fitted against an out-of-sample fallback. The router_val fallback uses all
router_train windows. Per-cell weights are inverse mean squared errors, as in the archived Bates comparator.

Development evidence on 60-80% blocks; no test cache is opened. Reuses the gate-ablation loaders, the frozen
scorer protocol and score_arms unchanged, and writes only under experiments/offline_bates_router/.

Usage (repo root):
  python experiments/offline_bates_router/run_offline_bates_router.py --selftest
  python -m experiments.offline_bates_router.run_offline_bates_router --run ETTh1:12 --seeds 7
  python -m experiments.offline_bates_router.run_offline_bates_router --run all --seeds 7 11 13 17 19
  python -m experiments.offline_bates_router.run_offline_bates_router --analyze
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
CHUNK = 256
LATE_SHARE = 0.3
ARMS = ("bates_only", "bates_router_constant", "bates_router_static", "bates_router_chosen", "bates_router_switch")
REFERENCE_ARMS = ("full", "core_only", "router_only", "constant", "static_hv")
CONTRASTS = (("bates_router_chosen", "full", "primary"), ("bates_router_chosen", "bates_all5", "primary"),
             ("bates_router_switch", "bates_all5", "primary"), ("bates_router_switch", "bates_router_chosen", "secondary"),
             ("bates_router_constant", "constant", "secondary"), ("bates_only", "bates_all5", "descriptive"))


def require(ok, message):
    if not ok:
        raise SystemExit(f"[offline] {message}")


def topk_equal(aff, k=3):
    return torch.zeros_like(aff).scatter_(-1, aff.topk(k, -1).indices, 1.0 / k)


def inverse_mse_weights(sq_sum, count):
    mse = sq_sum / count.clamp_min(1)
    w = 1.0 / mse.clamp_min(1e-12)
    return (w / w.sum(-1, keepdim=True)).float()


def solve(num, den):
    return torch.where(den > 0, num / den.clamp_min(1e-300), torch.zeros_like(num)).clamp(0, 1)


def closed_form(b, r, y, m, inv_var):
    """static_fits' estimator with fallback b: lambda = clip(-sum(e d) / sum(d^2), 0, 1), e = b - y, d = r - b.
    Returns (constant, per-cell [H, V]). inv_var [V] weights the constant like static_fits."""
    e = torch.where(m, b.double() - y.double(), 0.)
    d = torch.where(m, r.double() - b.double(), 0.)
    num, den = -(e * d).sum(0), d.square().sum(0)
    iv = inv_var.double().view(1, -1)
    return solve((num * iv).sum(), (den * iv).sum()).float(), solve(num, den).float()


def moving_block_lower(x, block, samples=4000, seed=20261110):
    rng = np.random.default_rng(seed)
    n = len(x)
    block = max(1, min(block, n))
    starts = rng.integers(0, n - block + 1, size=(samples, math.ceil(n / block)))
    idx = (starts[..., None] + np.arange(block)).reshape(samples, -1)[:, :n]
    return float(np.quantile(x[idx].mean(1), 0.025))


def fallback_and_router(fc_tr, y_tr, m_tr, origins_tr, oof_rows, fold_bounds, aff_oof, first_val_origin, horizon):
    """Causal Bates fallback and routed forecasts on the pooled OOF rows. Prefix sums of squared expert errors
    give every fit: fold k uses windows with origin + H <= fold origin (a prefix, origins ascending); router_val
    uses origin + H <= first val origin. Returns OOF fallback b_f and routed r_f [R,H,V], the fold and final
    weights, and the number of windows each fit used."""
    n_tr, h, v, e = fc_tr.shape
    starts = origins_tr.long()
    cuts = {int(lo): int((starts + horizon <= starts[lo]).sum()) for lo, _ in fold_bounds}
    cuts["final"] = int((starts + horizon <= first_val_origin).sum())
    require(all(cuts[int(lo)] <= lo for lo, _ in fold_bounds), "fold Bates would see the fold itself")
    sq, cnt = torch.zeros(h, v, e, dtype=torch.float64), torch.zeros(h, v, e, dtype=torch.float64)
    snapshots, done = {}, 0
    for target in sorted(set(cuts.values())):
        for lo in range(done, target, CHUNK):
            hi = min(lo + CHUNK, target)
            mm = m_tr[lo:hi, ..., None]
            sq += torch.where(mm, (fc_tr[lo:hi].double() - y_tr[lo:hi, ..., None].double()).square(), 0.).sum(0)
            cnt += mm.expand(-1, -1, -1, e).double().sum(0)
        done = target
        snapshots[target] = inverse_mse_weights(sq.clone(), cnt.clone())
    fold_w = {int(lo): snapshots[cuts[int(lo)]] for lo, _ in fold_bounds}
    final_w = snapshots[cuts["final"]]
    rows = oof_rows.long()
    b_f, r_f = torch.empty(len(rows), h, v), torch.empty(len(rows), h, v)
    fold_of = torch.zeros(n_tr, dtype=torch.long)
    for lo, hi in fold_bounds:
        fold_of[lo:hi] = lo
    for i in range(0, len(rows), CHUNK):
        sl = slice(i, i + CHUNK)
        f = fc_tr[rows[sl]]
        w = torch.stack([fold_w[int(fold_of[j])] for j in rows[sl]])
        b_f[sl] = (f * w).sum(-1)
        r_f[sl] = (f * topk_equal(aff_oof[sl])).sum(-1)
    return b_f, r_f, fold_w, final_w, cuts


def offline_arms(fc_tr, y_tr, m_tr, origins_tr, oof_rows, fold_bounds, aff_oof, fc_v, y_v, m_v, aff_v,
                 first_val_origin, horizon, std, block_length):
    """All tensors raw scale. fc_tr [N,H,V,E] router_train forecasts; oof_rows: train indices of the pooled OOF
    rows, in the order of aff_oof; fold_bounds: (lo, hi) train index ranges of the OOF folds.
    Returns router_val predictions per arm (raw scale) and the fitted choices."""
    h, v = fc_tr.shape[1:3]
    inv_var = 1.0 / std.double().square()
    rows = oof_rows.long()
    b_f, r_f, _, final_w, cuts = fallback_and_router(fc_tr, y_tr, m_tr, origins_tr, rows, fold_bounds, aff_oof,
                                                     first_val_origin, horizon)
    y_f, m_f = y_tr[rows], m_tr[rows]

    lam_c, lam_hv = closed_form(b_f, r_f, y_f, m_f, inv_var)
    cut = int(len(rows) * (1 - LATE_SHARE))
    early_c, early_hv = closed_form(b_f[:cut], r_f[:cut], y_f[:cut], m_f[:cut], inv_var)
    sd = std.view(1, 1, -1)

    def late_mae(lam):
        pred = b_f[cut:] + lam * (r_f[cut:] - b_f[cut:])
        return float(torch.where(m_f[cut:], (pred - y_f[cut:]).abs() / sd, 0.).sum() / m_f[cut:].sum().clamp_min(1))

    form = "constant" if late_mae(early_c) <= late_mae(early_hv) else "static"
    lam_chosen, lam_early = (lam_c, early_c) if form == "constant" else (lam_hv, early_hv)
    pred_late = b_f[cut:] + lam_early * (r_f[cut:] - b_f[cut:])
    gain = torch.where(m_f[cut:], ((b_f[cut:] - y_f[cut:]).abs() - (pred_late - y_f[cut:]).abs()) / sd, 0.)
    gain_wv = (gain.sum(1) / m_f[cut:].sum(1).clamp_min(1)).numpy()
    route = torch.tensor([moving_block_lower(gain_wv[:, j], block_length) > 0 for j in range(v)])

    b_v = torch.cat([(fc_v[i:i + CHUNK] * final_w).sum(-1) for i in range(0, len(fc_v), CHUNK)])
    r_v = torch.cat([(fc_v[i:i + CHUNK] * topk_equal(aff_v[i:i + CHUNK])).sum(-1) for i in range(0, len(fc_v), CHUNK)])
    blend = lambda lam: b_v + lam * (r_v - b_v)
    preds = {"bates_only": b_v, "bates_router_constant": blend(lam_c), "bates_router_static": blend(lam_hv),
             "bates_router_chosen": blend(lam_chosen),
             "bates_router_switch": blend(lam_chosen * route.float().view(1, -1).expand(h, v))}
    choices = {"chosen_form": form, "lambda_constant": float(lam_c), "lambda_static_mean": float(lam_hv.mean()),
               "lambda_static_share_0": float((lam_hv == 0).float().mean()), "lambda_static_share_1": float((lam_hv == 1).float().mean()),
               "late_oof_mae_constant": late_mae(early_c), "late_oof_mae_static": late_mae(early_hv),
               "variables_routed": int(route.sum()), "variables": v, "oof_rows": len(rows), "late_rows": len(rows) - cut,
               "bates_fit_windows": {str(k): c for k, c in cuts.items()}}
    return preds, choices


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def result_path(ds, h, seed, ext="json"):
    return OUT / "_work" / f"{ds}_h{h}" / f"seed{seed}.{ext}"


def run_job(ds, h, seed):
    import experiments.gate_ablation.run_gate_ablation as ga
    torch_, topk, _, lh = ga.deps()
    path = result_path(ds, h, seed)
    if path.exists():
        print(f"[offline] {ds} H{h} seed {seed}: already done", flush=True)
        return
    t0 = time.monotonic()
    conf, settings = ga.configure(ds, h)
    bundle, core, causality, chosen, ckroot, caches = ga.load_unit(ds, h, conf)
    require(not any("test" in Path(p).name.lower() for p in caches), "a test cache was selected")
    settings = dict(settings)
    if h == 12:
        n_val = int(bundle.val_cache["num_windows"])
        settings.update(evaluation_windows=n_val, independent_blocks=n_val / 24, interval_eligible=n_val / 24 >= lh.MIN_BLOCKS)
    require(causality["result"] == "PASS", "router_train forecasts are not out of sample")
    before = ga.checkpoint_hashes(ds, h, conf, ckroot)
    inputs = {Path(p).as_posix(): sha(p) for p in caches}
    topk.wdec.SCORER_SEED = seed
    wb = topk.wide.widened(bundle, topk.CANDIDATES)
    sc = topk.wide.train_scorer_bank_protocol(wb, seed)
    require(sc["tensors_finite"] and sc["oof_mask_covers_folds"], "scorer integrity failed")
    hi = topk.wide.head_inputs(wb, sc)
    ref_arms, fit, gate_diag = ga.score_arms(wb, hi, core, topk.wdec.STAT_CHUNK)
    fc_tr = wb.forecasts_fn(wb.train_cache, wb.expert_idx).float()
    preds, choices = offline_arms(
        fc_tr, wb.train_cache["targets"].float(), wb.train_cache["target_masks"].bool(),
        wb.train_cache["absolute_window_starts"].long(), torch.nonzero(sc["oof_mask"]).flatten(), sc["fold_bounds"],
        hi["oof_aff"].float(), hi["val_fc"].float(), wb.val_cache["targets"].float(), wb.val_cache["target_masks"].bool(),
        hi["val_aff"].float(), int(wb.val_cache["absolute_window_starts"][0]), h, wb.std.float(), int(settings["block_length"]))
    arms = {a: topk.cf.metrics_of(wb, p) for a, p in preds.items()}
    arms.update({a: ref_arms[a] for a in REFERENCE_ARMS})
    gate_path = ga.result_path(ds, h, seed, "pt")
    anchor = None
    if gate_path.exists():
        prior = torch.load(gate_path, map_location="cpu", weights_only=False)["arms"]["full"]["per_window_mae"]
        anchor = {"gate_ablation_full_exact": bool(torch.equal(prior, ref_arms["full"]["per_window_mae"])),
                  "max_abs_diff": float((prior - ref_arms["full"]["per_window_mae"]).abs().max())}
    checks = {"test_set_not_accessed": True, "checkpoints_unchanged": before == ga.checkpoint_hashes(ds, h, conf, ckroot),
              "inputs_unchanged": all(sha(p) == d for p, d in inputs.items()),
              "predictions_finite": all(bool(torch.isfinite(p).all()) for p in preds.values())}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arms": {a: {f"per_window_{m}": arms[a][f"per_window_{m}"] for m in ("mae", "mse")} for a in arms},
                "absolute_window_starts": wb.val_cache["absolute_window_starts"]}, path.with_suffix(".pt"))
    rec = {"dataset": ds, "horizon": h, "seed": seed, "core": core, "settings": settings, "choices": choices,
           "arms": {a: {k: float(arms[a][k]) for k in ("mae", "mse")} for a in arms}, "anchor": anchor,
           "lambda_diagnostics_learned_gate": {k: x for k, x in gate_diag.items() if not torch.is_tensor(x)},
           "integrity": checks | {"ok": all(checks.values())}, "input_hashes": inputs,
           "tensor_sha256": sha(path.with_suffix(".pt")), "elapsed_sec": time.monotonic() - t0, "test_set_accessed": False}
    path.write_text(json.dumps(topk.wdec.jsonable(rec), indent=1))
    require(all(checks.values()), "unit INVALID; record preserved")
    print(f"[offline] {ds} H{h} seed {seed}: done in {rec['elapsed_sec']:.0f}s [values withheld until analysis]", flush=True)


def analyze():
    import experiments.gate_ablation.run_gate_ablation as ga
    torch_, topk, _, lh = ga.deps()
    units = {}
    for h in ga.HORIZONS:
        for ds in ga.DATASETS:
            seeds = [s for s in ga.SEEDS if result_path(ds, h, s).exists()]
            if not seeds:
                continue
            recs = {s: json.loads(result_path(ds, h, s).read_text()) for s in seeds}
            require(all(r["integrity"]["ok"] for r in recs.values()), f"{ds} H{h}: an INVALID job is present")
            runs = {s: torch.load(result_path(ds, h, s, "pt"), map_location="cpu", weights_only=False) for s in seeds}
            settings = recs[seeds[0]]["settings"]
            topk.cf.BLOCK_LENGTH, topk.cf.PHASE_K = settings["block_length"], settings["phase_k"]
            names = list(runs[seeds[0]]["arms"])
            avg = {a: {m: torch.stack([runs[s]["arms"][a][f"per_window_{m}"] for s in seeds]).mean(0) for m in ("mae", "mse")} for a in names}
            comp_path = ga.reference_paths(ds, h, 7)[2]
            if comp_path.exists():
                saved = torch.load(comp_path, map_location="cpu", weights_only=False)
                if h == 12 and ds in ga.DATASETS[:3]:
                    avg["bates_all5"] = {"mae": saved["bates_all5_per_window_mae"]}
                else:
                    avg["bates_all5"] = {m: saved["bates_all5"][f"per_window_{m}"] for m in ("mae", "mse")}
            confirmatory = sorted(seeds) == sorted(ga.SEEDS)
            unit = {"seeds": seeds, "confirmatory": confirmatory, "arms": {a: {m: float(x.mean()) for m, x in mm.items()} for a, mm in avg.items()},
                    "choices_by_seed": {s: recs[s]["choices"] for s in seeds}, "anchor_by_seed": {s: recs[s]["anchor"] for s in seeds},
                    "contrasts": {}}
            for a, b, role in CONTRASTS:
                if b not in avg:
                    continue
                for m in ("mae", "mse"):
                    if m not in avg[b]:
                        continue
                    paired = topk.cf.paired(avg[a][m], avg[b][m])
                    cell = {"role": role, "delta": float(avg[a][m].mean() - avg[b][m].mean()),
                            "relative_pct": float(100 * (avg[a][m].mean() / avg[b][m].mean() - 1)), "paired": paired}
                    if confirmatory:
                        deltas = [float(runs[s]["arms"][a][f"per_window_{m}"].mean() - (runs[s]["arms"][b][f"per_window_{m}"].mean()
                                  if b in runs[s]["arms"] else avg[b][m].mean())) for s in seeds]
                        cell.update(lh.contrast(paired["block"], float(avg[b][m].mean()), settings["interval_eligible"], deltas))
                        cell["per_seed_deltas"] = deltas
                    unit["contrasts"][f"{a}-{b}:{m}"] = cell
            units[f"{ds}_h{h}"] = unit
    readings = {}
    for h in ga.HORIZONS:
        readings[h] = {}
        for a, b, role in CONTRASTS:
            for m in ("mae", "mse"):
                key = f"{a}-{b}:{m}"
                cells = {ds: units[f"{ds}_h{h}"]["contrasts"][key] for ds in ga.DATASETS
                         if f"{ds}_h{h}" in units and units[f"{ds}_h{h}"]["confirmatory"] and key in units[f"{ds}_h{h}"]["contrasts"]}
                if not cells:
                    continue
                absent = [ds for ds in ga.DATASETS if ds not in cells]
                readings[h][key] = {"role": role, "interval": lh.interval_reading(cells, absent, []),
                                    "across": lh.across_reading(cells, absent, [])}
    (OUT / "results.json").write_text(json.dumps(topk.wdec.jsonable({"units": units, "readings": readings,
                                                                     "test_set_accessed": False}), indent=1))
    lines = ["# Offline Bates-router experiment", "", "Development evidence on 60-80% blocks. No test access.",
             "Negative deltas favour the first arm. Confirmatory readings need all five scorer seeds.", "",
             "| Unit | Seeds | learned gate | constant (core) | Bates archived | Bates + router, chosen | Bates + router, switch | chosen form | variables routed |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for name, u in units.items():
        a = u["arms"]
        c0 = u["choices_by_seed"][u["seeds"][0]]
        lines.append(f"| {name} | {len(u['seeds'])} | {a['full']['mae']:.4f} | {a['constant']['mae']:.4f} | "
                     f"{a.get('bates_all5', {}).get('mae', float('nan')):.4f} | {a['bates_router_chosen']['mae']:.4f} | "
                     f"{a['bates_router_switch']['mae']:.4f} | {c0['chosen_form']} | {c0['variables_routed']}/{c0['variables']} |")
    for h, rr in readings.items():
        if rr:
            lines += ["", f"## Horizon {h}: confirmatory readings (all five seeds)", "",
                      "| Contrast | Role | Interval reading | Across reading | Exact Wilcoxon p |", "| --- | --- | --- | --- | --- |"]
            for key, r in rr.items():
                lines.append(f"| {key} | {r['role']} | {r['interval'].get('label')} | {r['across'].get('label')} | {r['across'].get('wilcoxon_p')} |")
    lines += ["", "## Contrasts (MAE)", "", "| Unit | Contrast | Role | Delta | Relative | 95% block interval |", "| --- | --- | --- | ---: | ---: | --- |"]
    for name, u in units.items():
        for key, cell in u["contrasts"].items():
            if key.endswith(":mae"):
                blk = cell["paired"].get("block", {})
                ci = blk.get("ci", blk.get("interval", "see results.json"))
                lines.append(f"| {name} | {key[:-4]} | {cell['role']} | {cell['delta']:+.5f} | {cell['relative_pct']:+.2f}% | {ci} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def synthetic(n=900, h=6, v=4, e=5, seed=0, harm_variable=3):
    """Experts 1 and 2 take turns being good in 30-window regimes and the affinities know which one; the
    other experts share a common error, so static Bates cannot cancel it and routing should help. In
    `harm_variable` the affinities point at three experts that share the common error, so routing hurts."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(n, h, v, generator=g)
    regime = (torch.arange(n) // 30) % 2
    common = torch.randn(n, h, v, generator=g)
    fc = y[..., None] + common[..., None] + 0.3 * torch.randn(n, h, v, e, generator=g)
    good = 1 + regime
    idx = torch.arange(n)
    fc[idx, :, :, good] = y[idx] + 0.3 * torch.randn(n, h, v, generator=g)
    aff = torch.full((n, h, v, e), 0.1)
    aff[idx, :, :, good] = 3.0
    aff[..., 0], aff[..., 3] = 1.0, 0.9
    aff[:, :, harm_variable] = torch.tensor([3.0, 0.0, 0.0, 2.5, 2.0])
    return fc, y, torch.ones(n, h, v, dtype=torch.bool), torch.arange(n) + 100, aff


def selftest():
    n, h, n_tr, warm = 900, 6, 600, 120
    fc, y, m, origins, aff = synthetic(n=n, h=h)
    tr, va = slice(0, n_tr), slice(n_tr, n)
    bounds = [(warm + (n_tr - warm) * i // 4, warm + (n_tr - warm) * (i + 1) // 4) for i in range(4)]
    rows = torch.arange(warm, n_tr)
    std = torch.ones(fc.shape[2])
    run = lambda yy: offline_arms(fc[tr], yy[tr], m[tr], origins[tr], rows, bounds, aff[tr][rows], fc[va], yy[va], m[va],
                                  aff[va], int(origins[n_tr]), h, std, 24)
    preds, ch = run(y)
    mae = {k: float((p - y[va]).abs().mean()) for k, p in preds.items()}
    assert mae["bates_router_static"] < mae["bates_only"], ("routing to the regime's good expert must help", mae)
    assert ch["variables_routed"] < ch["variables"], ("the switch must turn off the planted harmful variable", ch)

    y_val_poison = y.clone()
    y_val_poison[va] += 100.0
    poisoned, _ = run(y_val_poison)
    assert all(torch.equal(preds[k], poisoned[k]) for k in preds), "router_val targets must not change any prediction"

    lo2 = bounds[1][0]
    legal = lo2 - h + 1
    _, _, fold_w, _, cuts = fallback_and_router(fc[tr], y[tr], m[tr], origins[tr], rows, bounds, aff[tr][rows], int(origins[n_tr]), h)
    mse = (fc[tr][:legal] - y[tr][:legal, ..., None]).square().mean(0)
    expected = (1 / mse) / (1 / mse).sum(-1, keepdim=True)
    assert cuts[lo2] == legal and torch.allclose(fold_w[lo2], expected, atol=1e-5), "fold Bates = inverse MSE on its legal prefix"
    y_later = y.clone()
    y_later[legal:n_tr] += 7.0
    _, _, fold_w_later, _, _ = fallback_and_router(fc[tr], y_later[tr], m[tr], origins[tr], rows, bounds, aff[tr][rows], int(origins[n_tr]), h)
    assert torch.equal(fold_w_later[lo2], fold_w[lo2]), "a fold's Bates fit must not see windows whose targets end inside or after it"
    assert not torch.equal(fold_w_later[bounds[3][0]], fold_w[bounds[3][0]]), "later folds must see those windows"
    _, ch_again = run(y)
    assert ch_again == ch, "deterministic"
    print("[offline] selftest PASS", json.dumps({"mae": {k: round(x, 4) for k, x in mae.items()},
                                                 "choices": {k: ch[k] for k in ("chosen_form", "variables_routed", "variables")}}))


def lab(cache, run_dir, seeds):
    """Same arms on the throwaway ETTh1 lab (sigma = 1), checked against the lab pipeline's learned gate."""
    sys.path.insert(0, str(Path(run_dir).resolve().parent))
    import pipeline as P
    _, horizon, train_raw, val_raw = P.load(cache)
    train, val = P.Part(train_raw), P.Part(val_raw)
    pw_gate = dict(np.load(Path(run_dir) / "per_window.npz"))
    per_window, choices = {}, []
    for seed in seeds:
        sv = torch.load(Path(run_dir) / f"scorer_seed{seed}.pt")
        lo = sv["bounds"][0][0]
        rows = torch.arange(lo, len(train))
        preds, ch = offline_arms(train.fc, train.y, torch.ones_like(train.y, dtype=torch.bool), train.origins, rows, sv["bounds"],
                                 sv["oof_aff"][rows], val.fc, val.y, torch.ones_like(val.y, dtype=torch.bool), sv["val_aff"],
                                 int(val.origins[0]), horizon, torch.ones(train.y.shape[-1]), 24)
        choices.append({k: ch[k] for k in ("chosen_form", "lambda_constant", "variables_routed", "variables")})
        for a, p in preds.items():
            per_window.setdefault(a, []).append((p - val.y).abs().mean((1, 2)).numpy())
    avg = {a: np.mean(x, 0) for a, x in per_window.items()}
    avg["learned gate (lab pipeline)"] = pw_gate["full"]
    avg["core constant (lab pipeline)"] = pw_gate["constant"]
    out = {"horizon": horizon, "seeds": seeds, "choices": choices, "mae": {a: float(x.mean()) for a, x in avg.items()},
           "vs_learned_gate": {a: P.block_bootstrap(x - avg["learned gate (lab pipeline)"]) for a, x in avg.items()},
           "vs_bates_only": {a: P.block_bootstrap(x - avg["bates_only"]) for a, x in avg.items()}}
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", default=[], help="units like ETTh1:12, or 'all' for the 14 gate-ablation units")
    ap.add_argument("--seeds", nargs="*", type=int, default=[7])
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--lab", nargs=2, metavar=("CACHE", "RUN_DIR"))
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.lab:
        return lab(args.lab[0], args.lab[1], args.seeds)
    if args.analyze:
        return analyze()
    units = args.run
    if units == ["all"]:
        import experiments.gate_ablation.run_gate_ablation as ga
        units = [f"{ds}:{h}" for h in ga.HORIZONS for ds in ga.DATASETS]
    for unit in units:
        ds, h = unit.split(":")
        for seed in args.seeds:
            run_job(ds, int(h), seed)


if __name__ == "__main__":
    main()
