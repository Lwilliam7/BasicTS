#!/usr/bin/env python
"""Isolated, validation-only frozen-pool comparison. No BasicTS imports or writes.

Dependencies: numpy, torch, xgboost>=2.1, tsfeatures==0.4.5, threadpoolctl.
Run --help for usage; --self-test needs no research data or tsfeatures.
Server: --server plan, then --server run (resumable), --server status, --server pack.
Defaults: seven datasets at horizons 96 and 192; --horizons 96 192 336 720 extends it.

FFORMA source: robjhyndman/M4metalearning @
61ddc7101680e9df7219c359587d0b509d2b50d6, R/ensemble_classifier.R.
Adaptations: normalized MAE replaces M4 OWA; one history window is one
meta-example; THA features describe its cross-channel mean; no zero clamp.
The other methods are generic conditional controls, NOT FAME/TimeRouter replicas.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn
import xgboost as xgb


SEEDS = [7, 11, 13, 17, 19]
M4 = dict(max_depth=14, eta=0.575188, subsample=0.9161483,
          colsample_bytree=0.7670739, nrounds=94)
TREE_GRID = [M4,
    dict(max_depth=6, eta=0.1, subsample=0.8, colsample_bytree=0.8, nrounds=100),
    dict(max_depth=10, eta=0.05, subsample=0.7, colsample_bytree=0.7, nrounds=150),
    dict(max_depth=8, eta=0.3, subsample=0.9, colsample_bytree=0.6, nrounds=50),
    dict(max_depth=14, eta=0.01, subsample=0.6, colsample_bytree=0.9, nrounds=250)]
MLP_GRID = [dict(hidden=h, lr=lr, epochs=ep, weight_decay=wd)
            for h, lr, ep, wd in [(32,.001,20,.001),(64,.001,30,.001),
                                  (32,.003,30,.01),(64,.0003,40,.01),
                                  (128,.001,20,.01)]]
METHODS = ('fforma_mae', 'conditional_error_softmax', 'conditional_mlp')
SERVER_DATASETS = ('ETTh1', 'Electricity', 'Traffic', 'AppliancesEnergy',
                   'OikolabWeather', 'SolarAL', 'TetouanPower')
# Daily periods from the existing dataset-selection metadata: hourly or 10-minute.
SERVER_FREQUENCIES = {d: (144 if d in ('AppliancesEnergy','SolarAL','TetouanPower') else 24)
                      for d in SERVER_DATASETS}
GROUPS = {
    'acf_features': 'x_acf1 x_acf10 diff1_acf1 diff1_acf10 diff2_acf1 diff2_acf10 seas_acf1',
    'arch_stat': 'arch_lm', 'crossing_points': 'crossing_points',
    'entropy': 'entropy', 'flat_spots': 'flat_spots',
    'heterogeneity': 'arch_acf garch_acf arch_r2 garch_r2',
    'holt_parameters': 'alpha beta', 'hurst': 'hurst', 'lumpiness': 'lumpiness',
    'nonlinearity': 'nonlinearity',
    'pacf_features': 'x_pacf5 diff1x_pacf5 diff2x_pacf5 seas_pacf',
    'stl_features': 'nperiods seasonal_period trend spike linearity curvature e_acf1 e_acf10 seasonal_strength peak trough',
    'stability': 'stability', 'hw_parameters': 'hw_alpha hw_beta hw_gamma',
    'unitroot_kpss': 'unitroot_kpss', 'unitroot_pp': 'unitroot_pp',
}
FEATURE_NAMES = sorted({k for v in GROUPS.values() for k in v.split()}) + ['series_length']
SEASONAL = {'seas_acf1','seas_pacf','seasonal_strength','peak','trough'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f, **arrays)
    temporary.replace(path)


@contextlib.contextmanager
def run_lock(path):
    """OS lock is released on process exit; no manual stale-PID deletion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+b')
    if path.stat().st_size == 0:
        handle.write(b'0')
        handle.flush()
    try:
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError(f'Another process holds {path}; do not launch twice')
    try:
        yield
    finally:
        handle.close()


def feature_worker_init():
    # A feature worker must not inherit the parent's entire BLAS thread budget.
    from threadpoolctl import threadpool_limits
    global _feature_thread_limit
    _feature_thread_limit = threadpool_limits(limits=1)
    torch.set_num_threads(1)


def load_tensor_file(path):
    # Only explicitly supplied, trusted local research files should be loaded.
    return torch.load(path, map_location='cpu', weights_only=False)


def read_cache(path, expected):
    c = load_tensor_file(path)
    role = c.get('cache_role', c.get('split_role', ''))
    if role != expected:
        raise ValueError(f'{path}: role {role!r}, expected {expected!r}; test roles forbidden')
    for key in ('histories','targets','target_masks','prediction_stack','absolute_window_starts'):
        if key not in c:
            raise ValueError(f'{path}: missing {key}')
    n, h, v, k = c['prediction_stack'].shape
    if c['targets'].shape != (n,h,v) or c['target_masks'].shape != (n,h,v):
        raise ValueError('Targets/masks/forecasts have incompatible shapes')
    if c['histories'].ndim != 3 or c['histories'].shape[0] != n or c['histories'].shape[2] != v:
        raise ValueError('History dimensions do not match forecasts')
    names = list(c['expert_names'])
    if len(names) != k or len(set(names)) != k:
        raise ValueError('Invalid expert order')
    if int(c['forecast_horizon']) != h or int(c['num_windows']) != n:
        raise ValueError('Cache metadata disagrees with tensor dimensions')
    origins = c['absolute_window_starts'].long().numpy()
    if origins.shape != (n,) or n < 2 or not np.all(np.diff(origins) > 0):
        raise ValueError('Window starts must be unique and strictly chronological')
    for key in ('histories','prediction_stack'):
        if not torch.isfinite(c[key]).all():
            raise ValueError(f'Nonfinite {key}')
    mask = c['target_masks'].bool()
    if not torch.isfinite(c['targets'][mask]).all() or not mask.flatten(1).any(1).all():
        raise ValueError('Nonfinite observed target or entirely masked window')
    return c


def folds(origins, horizon, count=4, warmup=.2):
    bounds = np.linspace(math.ceil(len(origins)*warmup),len(origins),count+1,dtype=int)
    result = []
    for lo, hi in zip(bounds[:-1],bounds[1:]):
        fit = np.flatnonzero((np.arange(len(origins)) < lo) & (origins+horizon <= origins[lo]))
        if not len(fit) or hi <= lo:
            raise ValueError('Insufficient data for purged chronological folds')
        assert int((origins[fit]+horizon).max()) <= int(origins[lo])
        result.append((fit, np.arange(lo,hi)))
    return result


def metrics(cache, prediction, std):
    mask = cache['target_masks'].bool()
    # where prevents NaN * zero at missing targets.
    error = torch.where(mask, (prediction-cache['targets'])/std[None,None,:], 0.)
    denom = mask.flatten(1).sum(1).clamp_min(1)
    return torch.stack((error.abs().flatten(1).sum(1)/denom,
                        error.square().flatten(1).sum(1)/denom),1).double().numpy()


def expert_errors(cache, std):
    return np.stack([metrics(cache,cache['prediction_stack'][...,i],std)[:,0]
                     for i in range(len(cache['expert_names']))],1)


def subset(cache, indices):
    return {key: value[indices] if torch.is_tensor(value) and value.ndim and value.shape[0] == len(cache['targets'])
            else value for key,value in cache.items()}


def combine(forecasts, weights):
    w = np.asarray(weights)
    if not np.isfinite(w).all() or w.min() < -1e-6 or not np.allclose(w.sum(1),1,atol=1e-5):
        raise ValueError('Combiner produced invalid convex weights')
    return (forecasts * torch.as_tensor(w,dtype=forecasts.dtype)[:,None,None,:]).sum(-1)


def softmax(x):
    e = np.exp(x-x.max(1,keepdims=True))
    return e/e.sum(1,keepdims=True)


def fforma_objective(errors):
    def objective(margins, _):
        p = softmax(np.asarray(margins,dtype=np.float64).reshape(errors.shape))
        grad = p*(errors-(p*errors).sum(1,keepdims=True))
        # Official numerical surrogate, not the true diagonal Hessian.
        hess = errors*p*(1-p)-grad*p
        return grad,hess
    return objective


def extract_feature_row(task):
    import tsfeatures as tf
    series, frequency = task
    out, failed, padding, nonfinite = {}, 0, 0, 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for group, keys in GROUPS.items():
            try:
                values = getattr(tf,group)(series, freq=frequency)
            except Exception:
                values = {}
                failed += 1
            for key in keys.split():
                if key not in values and key in SEASONAL and frequency == 1:
                    out[key] = 0.
                    padding += 1
                    continue
                value = values.get(key, np.nan)
                value = float(value) if value is not None else np.nan
                if not np.isfinite(value):
                    nonfinite += 1
                    value = 0.
                out[key] = value
    out['series_length'] = float(len(series))
    return [out[k] for k in FEATURE_NAMES], failed, padding, nonfinite


def features(cache, frequency, workers, destination, identity):
    if destination.exists():
        z = np.load(destination,allow_pickle=False)
        if str(z['identity'].item()) != identity:
            raise ValueError('Feature-cache identity mismatch')
        return z['features'], json.loads(str(z['diagnostics'].item()))
    # This project's caches contain raw histories/forecasts. Use the history only.
    histories = np.concatenate([chunk.double().mean(2).numpy()
                                for chunk in cache['histories'].split(64)])
    print(f'Extracting THA features: {len(histories)} windows, workers={workers}',flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=feature_worker_init,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        rows = []
        for row in pool.map(extract_feature_row,((x,frequency) for x in histories),chunksize=16):
            rows.append(row)
            if len(rows) % 512 == 0:
                print(f'  THA features {len(rows)}/{len(histories)}', flush=True)
    array = np.array([r[0] for r in rows],dtype=np.float32)
    diag = dict(windows=len(rows), failed_groups=sum(r[1] for r in rows),
                seasonal_padding=sum(r[2] for r in rows), zeroed_nonfinite=sum(r[3] for r in rows))
    save_npz(destination,features=array,identity=identity,diagnostics=json.dumps(diag))
    return array,diag


def fit(method, config, x, cache, std, seed, threads):
    errors = expert_errors(cache,std)
    k = errors.shape[1]
    if method == 'fforma_mae':
        data = xgb.DMatrix(x,base_margin=np.zeros((len(x),k),dtype=np.float32))
        params = {key:value for key,value in config.items() if key != 'nrounds'}
        params.update(num_class=k,disable_default_eval_metric=1,seed=seed,nthread=threads,tree_method='hist')
        booster = xgb.train(params,data,num_boost_round=config['nrounds'],obj=fforma_objective(errors))
        def predict(new_x):
            test = xgb.DMatrix(new_x,base_margin=np.zeros((len(new_x),k),dtype=np.float32))
            return softmax(booster.predict(test,output_margin=True).reshape(-1,k))
        return predict
    if method == 'conditional_error_softmax':
        # Forecast each expert's conditional MAE, then convert to convex weights.
        # Temperature is fixed from fit-only losses, not validation performance.
        scale = max(float(np.mean(errors)),1e-8)
        models = []
        params = {key:value for key,value in config.items() if key != 'nrounds'}
        params.update(objective='reg:squarederror',seed=seed,nthread=threads,tree_method='hist')
        for i in range(k):
            models.append(xgb.train(params,xgb.DMatrix(x,label=errors[:,i]/scale),
                                    num_boost_round=config['nrounds']))
        return lambda new_x: softmax(-np.stack([m.predict(xgb.DMatrix(new_x)) for m in models],1)/.1)
    if method != 'conditional_mlp':
        raise ValueError(method)
    torch.manual_seed(seed)
    mean, scale = x.mean(0), np.maximum(x.std(0),1e-6)
    xx = torch.tensor((x-mean)/scale,dtype=torch.float32)
    model = nn.Sequential(nn.Linear(x.shape[1],config['hidden']),nn.ReLU(),nn.Linear(config['hidden'],k))
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    gen = torch.Generator().manual_seed(seed)
    forecasts = cache['prediction_stack'].float()
    for _ in range(config['epochs']):
        for indices in torch.randperm(len(x),generator=gen).split(128):
            weights = model(xx[indices]).softmax(1)
            prediction = (forecasts[indices]*weights[:,None,None,:]).sum(-1)
            target = cache['targets'][indices]
            mask = cache['target_masks'][indices].bool()
            error = torch.where(mask,(prediction-target)/std[None,None,:],0.)
            loss = (error.abs().flatten(1).sum(1)/mask.flatten(1).sum(1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    def predict(new_x):
        with torch.no_grad():
            return torch.cat([model(t).softmax(1) for t in
                torch.tensor((new_x-mean)/scale,dtype=torch.float32).split(1024)]).numpy()
    return predict


def select(method, grid, x, cache, std, splits, threads, progress_path=None):
    audit = json.loads(progress_path.read_text()) if progress_path and progress_path.exists() else []
    if len(audit) > len(grid) or any(r['config'] != grid[i] for i,r in enumerate(audit)):
        raise ValueError('Saved tuning grid differs from the declared grid')
    for config_index in range(len(audit),len(grid)):
        config = grid[config_index]
        total, count = 0., 0
        for fold_index, (fit_idx, val_idx) in enumerate(splits):
            print(f'  {method} config {config_index+1}/{len(grid)}, fold {fold_index+1}/{len(splits)}', flush=True)
            predictor = fit(method,config,x[fit_idx],subset(cache,fit_idx),std,7,threads)
            pred = combine(cache['prediction_stack'][val_idx],predictor(x[val_idx]))
            losses = metrics(subset(cache,val_idx),pred,std)[:,0]
            total += float(losses.sum())
            count += len(losses)
        audit.append(dict(config=config,oof_mae=total/count))
        if progress_path:
            write_json(progress_path,audit)
    # Deterministic tie-breaking: earliest declared configuration.
    winner = min(range(len(audit)),key=lambda i:audit[i]['oof_mae'])
    return grid[winner],audit


def paired(a,b,origins,horizon,samples=10000):
    delta = np.asarray(a)-np.asarray(b)
    # Match the existing non-wrapping moving-block bootstrap. Report point
    # estimates only with <8 blocks; never mistake scorer seeds for time samples.
    stride = int(np.median(np.diff(origins)))
    if np.any(np.diff(origins) != stride):
        raise ValueError('Evaluation starts must have a constant stride for block inference')
    length = math.ceil(2*horizon/stride)
    result = dict(mean_delta=float(delta.mean()),block_windows=length,bootstrap_samples=samples,
                  effective_blocks=len(delta)/length,interval_eligible=len(delta)/length >= 8)
    if not result['interval_eligible']:
        result.update(ci95=None,reason='Fewer than eight horizon-scaled blocks')
        return result
    rng = np.random.default_rng(20260821)
    means = []
    offsets = np.arange(length)
    for _ in range(samples):
        indices = (rng.integers(0,len(delta)-length+1,size=math.ceil(len(delta)/length))[:,None]+offsets).ravel()[:len(delta)]
        means.append(delta[indices].mean())
    result['ci95'] = np.quantile(means,[.025,.975]).tolist()
    result['bootstrap_fraction_negative'] = float(np.mean(np.asarray(means)<0))
    return result


def load_reference(path, train_hash, val_hash, normalizer_hash, names, origins):
    """NPZ: losses [seeds,N,2], origins [N], expert_names [K], input SHA256s.

    Predictions are NOT averaged across seeds: compare mean per-seed losses,
    as in the frozen primary's existing reports. Caller must export with these
    identity fields at the source; legacy unbound loss vectors are refused.
    """
    z = np.load(path,allow_pickle=False)
    if str(z['train_sha256'].item()) != train_hash or str(z['val_sha256'].item()) != val_hash:
        raise ValueError('Primary reference belongs to different forecast caches')
    if str(z['normalizer_sha256'].item()) != normalizer_hash:
        raise ValueError('Primary reference uses a different metric normalizer')
    if list(z['expert_names']) != names or not np.array_equal(z['origins'],origins):
        raise ValueError('Primary reference expert order/window alignment mismatch')
    losses = z['losses']
    if losses.shape != (len(SEEDS),len(origins),2) or not np.isfinite(losses).all() or (losses < 0).any():
        raise ValueError('Primary losses must be finite [seeds,windows,(MAE,MSE)]')
    if list(z['seeds']) != SEEDS:
        raise ValueError(f'Primary must contain the predeclared seeds {SEEDS}')
    return losses


def archived_files(work, dataset):
    return [work/f'comparators_{dataset}.{ext}' for ext in ('json','pt')] + [
        work/f'seed{seed}_{dataset}.{ext}' for seed in SEEDS for ext in ('json','pt')]


def archived_reference(work, train, val, std):
    """Bind existing server outputs using numerical sentinels and producer metadata.

    Legacy primary outputs have no original cache hashes. Never pretend that a new
    hash proves historical byte identity. Require matching per-window equal-five
    AND equal-core errors, training fold scores/origins, primary means and flags.
    Record this weaker historical association explicitly in the new protocol.
    """
    dataset, horizon = str(train['dataset']), int(train['forecast_horizon'])
    names = list(train['expert_names'])
    comp_meta = json.loads((work/f'comparators_{dataset}.json').read_text())
    comp = load_tensor_file(work/f'comparators_{dataset}.pt')
    if comp_meta.get('dataset') != dataset or comp_meta.get('test_set_accessed') is not False:
        raise ValueError('Archived comparator metadata has wrong dataset or test-access status')
    if comp_meta.get('router_train_causality') != 'PASS':
        raise ValueError('Archived comparator training causality did not pass')
    core = list(comp_meta['core'])
    indices = [names.index(name) for name in core]
    def losses_of(entry):
        result = np.stack([np.asarray(entry[f'per_window_{m}'],dtype=np.float64)
                           for m in ('mae','mse')],1)
        if result.shape != (len(val['targets']),2) or not np.isfinite(result).all() or (result < 0).any():
            raise ValueError('Invalid archived per-window losses')
        return result
    deltas = {}
    for key, prediction in [('equal5',val['prediction_stack'].mean(-1)),
                            ('equal3',val['prediction_stack'][...,indices].mean(-1))]:
        expected = metrics(val,prediction,std)
        saved = losses_of(comp[key])
        deltas[key] = float(np.max(np.abs(expected-saved)))
        if not np.allclose(expected,saved,atol=1e-6,rtol=1e-6):
            raise ValueError(f'Archived {key} window errors do not reproduce: max diff {deltas[key]}')
    comparators = {}
    for key, entry in comp.items():
        losses = losses_of(entry)
        if not np.allclose(losses.mean(0),[comp_meta['arms'][key][m] for m in ('mae','mse')],atol=1e-6,rtol=1e-6):
            raise ValueError(f'Archived comparator tensor/JSON mismatch: {key}')
        comparators['archived_'+key] = losses[None]
    origins = train['absolute_window_starts'].numpy()
    primary, fit_score_checks = [], []
    for seed in SEEDS:
        meta = json.loads((work/f'seed{seed}_{dataset}.json').read_text())
        for flag in ('freeze_matches','all_folds_causal','checkpoints_unchanged','tensors_finite','oof_mask_covers_folds'):
            if meta.get(flag) is not True:
                raise ValueError(f'Primary seed {seed}: {flag} did not pass')
        if (meta.get('dataset') != dataset or meta.get('seed') != seed or
            meta.get('horizon') != horizon or meta.get('test_set_accessed') is not False or
            meta.get('candidates') != names or meta.get('core') != core):
            raise ValueError(f'Primary seed {seed}: dataset/horizon/seed/expert metadata mismatch')
        fold_groups = meta['scorer_folds']
        if not fold_groups or not isinstance(fold_groups[0],list):
            raise ValueError('Expected current wide-top3 scorer-fold metadata')
        for group in fold_groups:
            for fold in group:
                lo, hi = int(fold['eval_lo']), int(fold['eval_hi'])
                if not 0 <= lo < hi <= len(origins) or int(origins[lo]) != int(fold['eval_origin']):
                    raise ValueError('Archived primary fold origins do not match training cache')
                legal = np.flatnonzero((np.arange(len(origins)) < lo) & (origins+horizon <= origins[lo]))
                if (not len(legal) or len(legal) != int(fold['num_legal_fit_windows']) or
                    int(origins[legal[-1]]+horizon) != int(fold['latest_fit_target_end'])):
                    raise ValueError('Archived primary purge boundary does not match training cache')
        if seed == SEEDS[0]:
            for fold in meta['core_selection']['fold_rows']:
                lo, hi = int(fold['eval_lo']), int(fold['eval_hi'])
                if not 0 <= lo < hi <= len(origins):
                    raise ValueError('Invalid archived core-selection fold bounds')
                small = subset(train,np.arange(lo,hi))
                current = metrics(small,small['prediction_stack'][...,indices].mean(-1),std).mean(0)
                saved = [fold[m] for m in ('mae','mse')]
                if not np.allclose(current,saved,atol=1e-6,rtol=1e-6):
                    raise ValueError('Archived core-selection scores do not match current training forecasts')
                fit_score_checks.append(float(np.max(np.abs(current-saved))))
        losses = losses_of(load_tensor_file(work/f'seed{seed}_{dataset}.pt')['wide_top3'])
        if not np.allclose(losses.mean(0),[meta['arms']['wide_top3'][m] for m in ('mae','mse')],atol=1e-6,rtol=1e-6):
            raise ValueError(f'Primary seed {seed}: tensor/JSON metric mismatch')
        primary.append(losses)
    return np.stack(primary),comparators,dict(
        binding='archived numerical sentinels and producer metadata; original input hashes unavailable',
        equal_window_max_differences=deltas,training_core_fold_max_differences=fit_score_checks,
        primary_seed_means_match=True,fold_origins_and_purge_match=True,
        historical_cache_byte_identity_proven=False)


def server_unit(root,dataset,horizon):
    if horizon == 96 and dataset != 'Traffic':
        pool = f'costarts_h96/{dataset}'
        experiment = root/'experiments/horizon_pilot_h96'
    else:
        pool = f'costarts_lh/h{horizon}/{dataset}'
        experiment = root/f'experiments/long_horizon_protocol/units/{dataset}_h{horizon}'
    cache = root/'cache'/pool
    return dict(name=f'{dataset}_h{horizon}',dataset=dataset,horizon=horizon,
        frequency=SERVER_FREQUENCIES[dataset],
        train_cache=str(cache/'router_train_20_60_cache.pt'),
        val_cache=str(cache/'router_val_60_80_cache.pt'),
        normalizer=str(root/'checkpoints'/pool/'final_60/DLinear/best_expert.pt'),
        archived_work=str(experiment/'_work'))


def server_inventory(args):
    root = args.repo_root.resolve()
    units = []
    for horizon in args.horizons:
        for dataset in args.datasets:
            unit = server_unit(root,dataset,horizon)
            if dataset == 'Traffic' and horizon == 720:
                unit.update(state='EXCLUDED',reason='Existing long-horizon protocol hardware exclusion')
            else:
                required = [Path(unit[k]) for k in ('train_cache','val_cache','normalizer')]
                required += archived_files(Path(unit['archived_work']),dataset)
                missing = [str(p.relative_to(root)) for p in required if not p.is_file()]
                unit.update(state='BLOCKED' if missing else 'READY',missing=missing)
            units.append(unit)
    return units


def server_report(output, inventory, outcomes):
    rows, states, contrasts = [], {}, {}
    for unit in inventory:
        name = unit['name']
        result = output/'units'/name/'results.json'
        current_state = outcomes[name]['state']
        if result.exists() and current_state == 'COMPLETE':
            record = json.loads(result.read_text())
            states[name] = record['status']
            contrasts[name] = record['contrasts']
            rows.extend(dict(dataset=unit['dataset'],horizon=unit['horizon'],method=m,**values)
                        for m,values in record['summary'].items())
        else:
            states[name] = current_state if current_state != 'COMPLETE' else 'FAILED'
    complete = sum(s == 'COMPLETE' for s in states.values())
    requested = sum(s != 'EXCLUDED' for s in states.values())
    status = 'COMPLETE' if complete == requested else 'INCOMPLETE'
    write_json(output/'summary.json',dict(status=status,completed=complete,requested=requested,
        units=states,rows=rows,contrasts=contrasts,test_set_accessed=False))
    with (output/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=['dataset','horizon','method','mae','mse','seed_mae_sd'])
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Server conditional comparison','',f'Status: {status}; {complete}/{requested} units complete.','',
             'Development evidence on the original frozen pools; no expert training or test access.','',
             '| Unit | Status |','| --- | --- |']
    lines += [f'| {name} | {state} |' for name,state in states.items()]
    lines += ['', 'Metrics: summary.csv. Paired differences: summary.json and units/*/results.json.',
              'Positive method-minus-wide_top3 differences favor wide_top3.',
              'Missing/failed units remain explicit; intervals with fewer than eight blocks are withheld.',
              'Archived primary association is checked numerically; historical input byte identity is not proven.',
              'FFORMA is adapted to rolling multivariate windows. Other learned methods are generic controls.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return status


def server_main(args):
    root = args.repo_root.resolve()
    output = (args.output or root/'experiments/conditional_comparison_server').resolve()
    if args.server in ('status','pack'):
        if not (output/'queue_manifest.json').exists():
            raise SystemExit(f'No queue started at {output}')
        if args.server == 'status':
            state = output/'queue_state.json'
            print(state.read_text() if state.exists() else 'Queue manifest exists; no state yet')
            return
        with run_lock(output/'.queue.lock'):
            archive = output.with_suffix('.tar.gz')
            temporary = archive.with_suffix('.tar.gz.tmp')
            with tarfile.open(temporary,'w:gz') as tar:
                for path in sorted(output.rglob('*')):
                    if path.is_file() and path.suffix in ('.json','.md','.csv','.log','.py','.npz') and not path.name.endswith('_features.npz'):
                        tar.add(path,arcname=str(Path(output.name)/path.relative_to(output)))
            temporary.replace(archive)
            print(f'Results bundle: {archive}')
            return
    inventory = server_inventory(args)
    if args.server == 'plan':
        print(json.dumps(dict(repo_root=str(root),output=str(output),jobs=args.jobs,
            threads_per_job=args.threads,feature_workers_per_job=args.workers,
            execution='CPU; bounded unit processes; all original experts frozen',
            dependencies={p:importlib.metadata.version(p) for p in ('numpy','torch','xgboost','tsfeatures','threadpoolctl')},
            note='Inventory only. Full numerical provenance checks run before each unit is fitted.',
            units=inventory),indent=2))
        return
    settings = dict(repo_root=str(root),datasets=args.datasets,horizons=args.horizons,
        jobs=args.jobs,threads=args.threads,workers=args.workers,bootstrap_samples=args.bootstrap_samples,
        source_sha256=sha256(Path(__file__)),frequencies=SERVER_FREQUENCIES)
    with run_lock(output/'.queue.lock'):
        manifest = output/'queue_manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != settings:
                raise ValueError('Queue settings/source changed: use a new output directory')
        else:
            write_json(manifest,settings)
            shutil.copy2(Path(__file__),output/'source_snapshot.py')
        state = dict(started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
                     status='RUNNING',units={u['name']:dict(state=u['state'],missing=u.get('missing',[])) for u in inventory})
        (output/'logs').mkdir(exist_ok=True)
        write_json(output/'queue_state.json',state)
        def worker(unit):
            destination = output/'units'/unit['name']
            command = [sys.executable,str(Path(__file__).resolve()),'--train-cache',unit['train_cache'],
                '--val-cache',unit['val_cache'],'--normalizer',unit['normalizer'],
                '--archived-work',unit['archived_work'],'--expected-dataset',unit['dataset'],
                '--expected-horizon',str(unit['horizon']), '--frequency',str(unit['frequency']),
                '--output',str(destination),'--resume','--threads',str(args.threads),
                '--workers',str(args.workers),'--bootstrap-samples',str(args.bootstrap_samples)]
            env = dict(os.environ,OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),
                       OPENBLAS_NUM_THREADS=str(args.threads),NUMEXPR_NUM_THREADS=str(args.threads),
                       PYTHONUNBUFFERED='1')
            log = output/'logs'/f"{unit['name']}.log"
            with log.open('a',encoding='utf-8') as stream:
                stream.write('\nAttempt '+datetime.now(timezone.utc).isoformat()+'\n')
                stream.flush()
                completed = subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if completed.returncode:
                destination.mkdir(parents=True,exist_ok=True)
                write_json(destination/'failure.json',dict(returncode=completed.returncode,log=str(log),
                    utc=datetime.now(timezone.utc).isoformat()))
            return completed.returncode
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            pending = {}
            for unit in inventory:
                if unit['state'] == 'READY':
                    state['units'][unit['name']]['state'] = 'SUBMITTED'
                    pending[executor.submit(worker,unit)] = unit
            write_json(output/'queue_state.json',state)
            for future in as_completed(pending):
                unit = pending[future]
                try:
                    rc = future.result()
                except Exception as exc:
                    rc = -1
                    state['units'][unit['name']]['error'] = str(exc)
                state['units'][unit['name']].update(state='COMPLETE' if rc == 0 else 'FAILED',returncode=rc)
                write_json(output/'queue_state.json',state)
                print(f"{unit['name']}: {state['units'][unit['name']]['state']}",flush=True)
        status = server_report(output,inventory,state['units'])
        state.update(status=status,finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output/'queue_state.json',state)
        print(f'{status}: {output}/report.md',flush=True)
        if status != 'COMPLETE':
            raise SystemExit(2)


def self_test(threads):
    """Planted, predictable regime: each of two experts wins half the windows."""
    torch.set_num_threads(threads)
    rng = np.random.default_rng(314)
    x = rng.normal(size=(480,4)).astype(np.float32)
    y = torch.zeros(480,2,1)
    forecasts = torch.zeros(480,2,1,2)
    forecasts[:, :, 0, 0] = torch.from_numpy(np.where(x[:,0] > 0,0.,2.))[:,None]
    forecasts[:, :, 0, 1] = 2.-forecasts[:, :, 0, 0]
    cache = dict(targets=y,target_masks=torch.ones_like(y,dtype=torch.bool),
                 prediction_stack=forecasts,expert_names=['a','b'])
    train, val = np.arange(320),np.arange(320,480)
    checks = {}
    for method in METHODS:
        config = (dict(hidden=16,lr=.02,epochs=30,weight_decay=0.) if method == 'conditional_mlp'
                  else dict(max_depth=2,eta=.2,subsample=1.,colsample_bytree=1.,nrounds=30))
        pred = fit(method,config,x[train],subset(cache,train),torch.ones(1),7,threads)
        weights = pred(x[val])
        error = metrics(subset(cache,val),combine(forecasts[val],weights),torch.ones(1))[:,0].mean()
        assert error < .5, (method,error)  # equal weighting has MAE exactly 1.
        checks[method] = float(error)
    for fit_idx, val_idx in folds(np.arange(100)*2,12):
        assert max(fit_idx*2+12) <= min(val_idx*2)
    g,h = fforma_objective(np.array([[1.,3.]]))(np.zeros((1,2)),None)
    assert np.allclose(g,[[-.5,.5]]) and np.all(h >= 0)
    masked = dict(targets=torch.tensor([[[2.,float('nan')]]]),target_masks=torch.tensor([[[True,False]]]))
    assert np.allclose(metrics(masked,torch.tensor([[[0.,9.]]]),torch.tensor([2.,1.])),[[1.,1.]])
    interval = paired(np.ones(100),np.zeros(100),np.arange(100),12,100)
    assert not interval['interval_eligible'] and interval['ci95'] is None
    print(json.dumps(dict(status='PASS',planted_mae=checks,equal_mae=1.,
                          checked=['predictable routing','convexity','purge','official objective','masked scaling','underpower guard']),indent=2))


def run(args):
    torch.set_num_threads(args.threads)
    train = read_cache(args.train_cache,'router_train_20_60')
    val = read_cache(args.val_cache,'router_val_60_80')
    names = list(train['expert_names'])
    if names != list(val['expert_names']) or train['dataset'] != val['dataset']:
        raise ValueError('Train/validation dataset or expert order mismatch')
    if train['prediction_stack'].shape[1:] != val['prediction_stack'].shape[1:]:
        raise ValueError('Train/validation forecast shape mismatch')
    if not {'block_b_oos','block_c_oos'}.issubset(train.get('provenance',{}).get('source_caches',{})):
        raise ValueError('Missing walk-forward out-of-sample training provenance')
    origins = train['absolute_window_starts'].long().numpy()
    val_origins = val['absolute_window_starts'].long().numpy()
    horizon = int(train['forecast_horizon'])
    if args.expected_dataset and train['dataset'] != args.expected_dataset:
        raise ValueError('Cache dataset does not match the requested server unit')
    if args.expected_horizon and horizon != args.expected_horizon:
        raise ValueError('Cache horizon does not match the requested server unit')
    if origins[-1]+horizon > val_origins[0]:
        raise ValueError('Training labels are not fully observable before validation')
    normalizer = load_tensor_file(args.normalizer)
    # Deliberately no silent unit-scaler fallback.
    std = torch.as_tensor(normalizer['scaler_std'],dtype=torch.float32).flatten()
    if len(std) != train['targets'].shape[2] or not torch.isfinite(std).all() or not (std>0).all():
        raise ValueError('Invalid expert checkpoint scaler_std')
    # Reported normalized forecast metrics match the project's stored raw tensors.
    splits = folds(origins,horizon)
    input_paths = [args.train_cache,args.val_cache,args.normalizer]+([args.primary] if args.primary else [])
    if args.archived_work:
        input_paths += archived_files(args.archived_work,str(train['dataset']))
    hashes = {str(p.resolve()):sha256(p) for p in input_paths}
    train_hash, val_hash = hashes[str(args.train_cache.resolve())],hashes[str(args.val_cache.resolve())]
    reference = load_reference(args.primary,train_hash,val_hash,hashes[str(args.normalizer.resolve())],names,val_origins) if args.primary else None
    archived, reference_audit = {}, None
    if args.archived_work:
        reference,archived,reference_audit = archived_reference(args.archived_work,train,val,std)
    plan = dict(dataset=train['dataset'],horizon=horizon,expert_names=names,
                train_windows=len(origins),evaluation_windows=len(val_origins),
                seeds=SEEDS,selection_seed=7,folds=[dict(fit_windows=len(t),eval_windows=len(v),
                max_fit_target_end=int(origins[t[-1]]+horizon),eval_origin=int(origins[v[0]])) for t,v in splits],
                hashes=hashes,frequency=args.frequency,methods=list(METHODS),configurations_per_method=5,
                bootstrap_samples=args.bootstrap_samples,bootstrap_seed=20260821,
                block_rule='ceil(2*horizon/evaluation_stride); nonwrapping; minimum eight blocks',
                primary_available=reference is not None,reference_audit=reference_audit,
                threads=args.threads,workers=args.workers,test_set_accessed=False)
    if args.plan:
        print(json.dumps(plan,indent=2))
        return
    protocol = dict(plan,tree_grid=TREE_GRID,mlp_grid=MLP_GRID,
        primary='frozen wide_top3; cache-bound NPZ or audited original server outputs',
        selection_metric='pooled purged-OOF normalized MAE; no validation selection',
        inference='static after router_train; no validation feedback',
        evidence='development; reused validation blocks; no new confirmation',
        adaptation='FFORMA-MAE with cross-channel mean THA features and window-level weights',
        tuning_budget='5 configurations x 4 folds, seed 7, per family; unequal runtime disclosed',
        limitations=['Generic conditional controls are not published-method replications',
                     'FFORMA uses original window-level granularity, not cell-level routing',
                     'Frozen primary has historical development selection; budgets are not retroactively equal'],
        source_sha256=sha256(Path(__file__)),python=platform.python_version(),
        packages={p:importlib.metadata.version(p) for p in ['numpy','torch','xgboost','tsfeatures','threadpoolctl']})
    if args.output.exists():
        if not args.resume:
            raise ValueError('Output already exists: use --resume for identical inputs/settings')
        saved = args.output/'protocol.json'
        if saved.exists():
            if json.loads(saved.read_text()) != protocol:
                raise ValueError('Cannot resume: input hashes, settings, source or package versions changed')
        elif any(p.name != 'failure.json' for p in args.output.iterdir()):
            raise ValueError('Output contains files but no resumable protocol')
    else:
        args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'protocol.json',protocol)
    if (args.output/'results.json').exists() and (args.output/'report.md').exists():
        print(f'Already complete; inputs and protocol revalidated: {args.output}',flush=True)
        return
    shutil.copy2(Path(__file__),args.output/'source_snapshot.py')
    (args.output/'checkpoints').mkdir(exist_ok=True)
    if reference is not None:
        save_npz(args.output/'primary_reference.npz',losses=reference,seeds=SEEDS,origins=val_origins,
                 expert_names=np.array(names),train_sha256=train_hash,val_sha256=val_hash,
                 normalizer_sha256=hashes[str(args.normalizer.resolve())],
                 binding=json.dumps(reference_audit or {'binding':'supplied cache-bound reference'}))
    fingerprint = sha256(Path(__file__))
    train_x,train_diag = features(train,args.frequency,args.workers,args.output/'train_features.npz',
                                f'{train_hash}:{args.frequency}:{fingerprint}')
    val_x,val_diag = features(val,args.frequency,args.workers,args.output/'val_features.npz',
                            f'{val_hash}:{args.frequency}:{fingerprint}')
    results, selection = dict(archived), {}
    equal = val['prediction_stack'].mean(-1)
    results['equal_all'] = metrics(val,equal,std)[None]
    best = int(expert_errors(train,std).mean(0).argmin())
    results['train_best_single'] = metrics(val,val['prediction_stack'][...,best],std)[None]
    for method in METHODS:
        print(f'Selecting {method}: five configurations, four purged folds',flush=True)
        config,audit = select(method,MLP_GRID if method == 'conditional_mlp' else TREE_GRID,
                              train_x,train,std,splits,args.threads,
                              args.output/'checkpoints'/f'{method}_selection.json')
        selection[method] = dict(selected=config,candidates=audit)
        write_json(args.output/'selection.json',selection)
        losses = []
        for seed in SEEDS:
            checkpoint = args.output/'checkpoints'/f'{method}_seed{seed}.npz'
            if checkpoint.exists():
                with np.load(checkpoint,allow_pickle=False) as stored:
                    loss = stored['losses']
                if loss.shape != (len(val_origins),2) or not np.isfinite(loss).all():
                    raise ValueError(f'Invalid saved final-fit losses: {checkpoint}')
                print(f'Reusing {method}, seed={seed}',flush=True)
            else:
                print(f'Fitting {method}, seed={seed}',flush=True)
                predict = fit(method,config,train_x,train,std,seed,args.threads)
                weights = predict(val_x)
                loss = metrics(val,combine(val['prediction_stack'],weights),std)
                save_npz(checkpoint,losses=loss,weights=weights,origins=val_origins)
            losses.append(loss)
        results[method] = np.stack(losses)
    if reference is not None:
        results['wide_top3'] = reference
    # Fingerprints are verified again before any scientific result is written.
    if any(sha256(p) != hashes[str(p.resolve())] for p in input_paths):
        raise ValueError('Input changed during execution; results invalid')
    if sha256(Path(__file__)) != protocol['source_sha256']:
        raise ValueError('Runner source changed during execution; results invalid')
    summary = {method:dict(mae=float(losses[:,:,0].mean()),mse=float(losses[:,:,1].mean()),
        seed_mae_sd=float(losses[:,:,0].mean(1).std(ddof=1)) if len(losses)>1 else None)
        for method,losses in results.items()}
    contrasts = {}
    for method in results:
        for baseline in ['equal_all']+(['wide_top3'] if reference is not None else [])+(['archived_bates_all5'] if 'archived_bates_all5' in results else []):
            if method == baseline:
                continue
            contrasts[f'{method}_minus_{baseline}'] = {metric:paired(results[method][:,:,i].mean(0),
                results[baseline][:,:,i].mean(0),val_origins,horizon,args.bootstrap_samples)
                for i,metric in enumerate(('mae','mse'))}
    save_npz(args.output/'per_window_losses.npz',**results,origins=val_origins)
    report = dict(status='COMPLETE' if reference is not None else 'BASELINES_ONLY_PRIMARY_MISSING',
        summary=summary,contrasts=contrasts,feature_diagnostics=dict(train=train_diag,val=val_diag),
        best_single_expert=names[best],reference_audit=reference_audit,
        inputs_unchanged=True,test_set_accessed=False)
    write_json(args.output/'results.json',report)
    lines = ['# Conditional combination comparison','',f"Status: {report['status']}",'',
             'Development evaluation on reused validation data. No test access.','',
             '| Method | MAE | MSE |','| --- | ---: | ---: |']
    lines += [f"| {name} | {value['mae']:.6f} | {value['mse']:.6f} |" for name,value in summary.items()]
    lines += ['', 'Paired differences, horizon-scaled moving-block intervals, selection audit,',
              'and feature failures are in results.json, selection.json and protocol.json.',
              'Intervals are descriptive pairwise intervals without multiplicity adjustment.',
              'Seeds average losses, not forecasts; they do not increase the temporal sample size.']
    if reference is None:
        lines += ['', 'The frozen primary was not supplied. This is not a completed comparison against CellCatch.']
    if reference_audit:
        lines += ['', 'Archived primary association passed numerical and metadata checks. Its original',
                  'cache hashes were not recorded, so historical cache byte identity is not proven.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report['summary'],indent=2))
    print(f"{report['status']}: {args.output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train-cache',type=Path)
    parser.add_argument('--val-cache',type=Path)
    parser.add_argument('--normalizer',type=Path,help='Matching frozen expert checkpoint with scaler_std')
    parser.add_argument('--frequency',type=int,help='Declare sampling/seasonal frequency before scoring')
    parser.add_argument('--primary',type=Path,help='Cache-bound NPZ reference: see load_reference docstring')
    parser.add_argument('--archived-work',type=Path,help='Original server _work: verify and import primary/comparators')
    parser.add_argument('--output',type=Path,help='Output directory, or server queue root')
    parser.add_argument('--resume',action='store_true',help='Resume only with identical hashes/settings/source/packages')
    parser.add_argument('--server',choices=['plan','run','status','pack'],help='Use documented server cache/result layouts')
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--datasets',nargs='+',choices=SERVER_DATASETS,default=list(SERVER_DATASETS))
    parser.add_argument('--horizons',nargs='+',type=int,choices=[96,192,336,720],default=[96,192])
    parser.add_argument('--jobs',type=int,default=2,help='Maximum concurrent CPU unit processes in server mode')
    parser.add_argument('--expected-dataset',help=argparse.SUPPRESS)
    parser.add_argument('--expected-horizon',type=int,help=argparse.SUPPRESS)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--bootstrap-samples',type=int,default=10000)
    parser.add_argument('--plan',action='store_true',help='Validate inputs and print protocol; no fitting or writes')
    parser.add_argument('--self-test',action='store_true',help='Run planted-signal and protocol checks')
    args = parser.parse_args()
    if args.self_test:
        self_test(args.threads)
        return
    if min(args.workers,args.threads,args.bootstrap_samples,args.jobs) < 1:
        parser.error('Worker/thread/sample/job counts must be positive')
    if args.primary and args.archived_work:
        parser.error('Use either --primary or --archived-work, not both')
    if args.server:
        if len(set(args.datasets)) != len(args.datasets) or len(set(args.horizons)) != len(args.horizons):
            parser.error('Duplicate dataset/horizon would create colliding jobs')
        server_main(args)
        return
    for field in ['train_cache','val_cache','normalizer','frequency']+([] if args.plan else ['output']):
        if getattr(args,field) is None:
            parser.error(f'--{field.replace("_","-")} is required')
    if min(args.workers,args.threads,args.bootstrap_samples,args.frequency) < 1:
        parser.error('Worker/thread/sample/frequency counts must be positive')
    if args.plan:
        run(args)
    else:
        with run_lock(args.output.parent/(args.output.name+'.lock')):
            run(args)


if __name__ == '__main__':
    main()
