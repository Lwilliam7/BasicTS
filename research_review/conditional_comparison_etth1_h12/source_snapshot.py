#!/usr/bin/env python
"""Isolated, validation-only frozen-pool comparison. No BasicTS imports or writes.

Dependencies: numpy, torch, xgboost>=2.1, tsfeatures==0.4.5.
Run --help for usage; --self-test needs no research data or tsfeatures.

FFORMA source: robjhyndman/M4metalearning @
61ddc7101680e9df7219c359587d0b509d2b50d6, R/ensemble_classifier.R.
Adaptations: normalized MAE replaces M4 OWA; one history window is one
meta-example; THA features describe its cross-channel mean; no zero clamp.
The other methods are generic conditional controls, NOT FAME/TimeRouter replicas.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
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
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


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
    histories = cache['histories'].double().mean(2).numpy()
    print(f'Extracting THA features: {len(histories)} windows, workers={workers}',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(extract_feature_row,((x,frequency) for x in histories),chunksize=16))
    array = np.array([r[0] for r in rows],dtype=np.float32)
    diag = dict(windows=len(rows), failed_groups=sum(r[1] for r in rows),
                seasonal_padding=sum(r[2] for r in rows), zeroed_nonfinite=sum(r[3] for r in rows))
    np.savez_compressed(destination,features=array,identity=identity,diagnostics=json.dumps(diag))
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


def select(method, grid, x, cache, std, splits, threads):
    audit = []
    for config in grid:
        total, count = 0., 0
        for fit_idx, val_idx in splits:
            predictor = fit(method,config,x[fit_idx],subset(cache,fit_idx),std,7,threads)
            pred = combine(cache['prediction_stack'][val_idx],predictor(x[val_idx]))
            losses = metrics(subset(cache,val_idx),pred,std)[:,0]
            total += float(losses.sum())
            count += len(losses)
        audit.append(dict(config=config,oof_mae=total/count))
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
    result = dict(mean_delta=float(delta.mean()),block_windows=length,
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


def load_reference(path, train_hash, val_hash, names, origins):
    """NPZ: losses [seeds,N,2], origins [N], expert_names [K], cache SHA256s.

    Predictions are NOT averaged across seeds: compare mean per-seed losses,
    as in the frozen primary's existing reports. Caller must export with these
    identity fields at the source; legacy unbound loss vectors are refused.
    """
    z = np.load(path,allow_pickle=False)
    if str(z['train_sha256'].item()) != train_hash or str(z['val_sha256'].item()) != val_hash:
        raise ValueError('Primary reference belongs to different forecast caches')
    if list(z['expert_names']) != names or not np.array_equal(z['origins'],origins):
        raise ValueError('Primary reference expert order/window alignment mismatch')
    losses = z['losses']
    if losses.ndim != 3 or losses.shape[1:] != (len(origins),2) or not np.isfinite(losses).all():
        raise ValueError('Primary losses must be finite [seeds,windows,(MAE,MSE)]')
    if list(z['seeds']) != SEEDS:
        raise ValueError(f'Primary must contain the predeclared seeds {SEEDS}')
    return losses


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
    hashes = {str(p.resolve()):sha256(p) for p in input_paths}
    train_hash, val_hash = hashes[str(args.train_cache.resolve())],hashes[str(args.val_cache.resolve())]
    reference = load_reference(args.primary,train_hash,val_hash,names,val_origins) if args.primary else None
    plan = dict(dataset=train['dataset'],horizon=horizon,expert_names=names,
                train_windows=len(origins),evaluation_windows=len(val_origins),
                seeds=SEEDS,selection_seed=7,folds=[dict(fit_windows=len(t),eval_windows=len(v),
                max_fit_target_end=int(origins[t[-1]]+horizon),eval_origin=int(origins[v[0]])) for t,v in splits],
                hashes=hashes,frequency=args.frequency,methods=list(METHODS),configurations_per_method=5,
                primary_available=reference is not None,test_set_accessed=False)
    if args.plan:
        print(json.dumps(plan,indent=2))
        return
    # Exclusive directory prevents overwriting an experiment, including failed runs.
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'protocol.json',dict(plan,tree_grid=TREE_GRID,mlp_grid=MLP_GRID,
        primary='frozen wide_top3; must be supplied as a cache-bound reference',
        selection_metric='pooled purged-OOF normalized MAE; no validation selection',
        inference='static after router_train; no validation feedback',
        evidence='development; reused validation blocks; no new confirmation',
        adaptation='FFORMA-MAE with cross-channel mean THA features and window-level weights',
        tuning_budget='5 configurations x 4 folds, seed 7, per family; unequal runtime disclosed',
        limitations=['Generic conditional controls are not published-method replications',
                     'FFORMA uses original window-level granularity, not cell-level routing',
                     'Frozen primary has historical development selection; budgets are not retroactively equal'],
        source_sha256=sha256(Path(__file__)),python=platform.python_version(),
        packages={p:importlib.metadata.version(p) for p in ['numpy','torch','xgboost','tsfeatures']}))
    torch.set_num_threads(args.threads)
    fingerprint = sha256(Path(__file__))
    train_x,train_diag = features(train,args.frequency,args.workers,args.output/'train_features.npz',
                                f'{train_hash}:{args.frequency}:{fingerprint}')
    val_x,val_diag = features(val,args.frequency,args.workers,args.output/'val_features.npz',
                            f'{val_hash}:{args.frequency}:{fingerprint}')
    results, selection = {}, {}
    equal = val['prediction_stack'].mean(-1)
    results['equal_all'] = metrics(val,equal,std)[None]
    best = int(expert_errors(train,std).mean(0).argmin())
    results['train_best_single'] = metrics(val,val['prediction_stack'][...,best],std)[None]
    for method in METHODS:
        print(f'Selecting {method}: five configurations, four purged folds',flush=True)
        config,audit = select(method,MLP_GRID if method == 'conditional_mlp' else TREE_GRID,
                              train_x,train,std,splits,args.threads)
        selection[method] = dict(selected=config,candidates=audit)
        write_json(args.output/'selection.json',selection)
        losses = []
        for seed in SEEDS:
            print(f'Fitting {method}, seed={seed}',flush=True)
            predict = fit(method,config,train_x,train,std,seed,args.threads)
            losses.append(metrics(val,combine(val['prediction_stack'],predict(val_x)),std))
        results[method] = np.stack(losses)
    if reference is not None:
        results['wide_top3'] = reference
    # Fingerprints are verified again before any scientific result is written.
    if any(sha256(p) != hashes[str(p.resolve())] for p in input_paths):
        raise ValueError('Input changed during execution; results invalid')
    summary = {method:dict(mae=float(losses[:,:,0].mean()),mse=float(losses[:,:,1].mean()),
        seed_mae_sd=float(losses[:,:,0].mean(1).std(ddof=1)) if len(losses)>1 else None)
        for method,losses in results.items()}
    contrasts = {}
    for method in results:
        for baseline in ['equal_all']+(['wide_top3'] if reference is not None else []):
            if method == baseline:
                continue
            contrasts[f'{method}_minus_{baseline}'] = {metric:paired(results[method][:,:,i].mean(0),
                results[baseline][:,:,i].mean(0),val_origins,horizon,args.bootstrap_samples)
                for i,metric in enumerate(('mae','mse'))}
    np.savez_compressed(args.output/'per_window_losses.npz',**results,origins=val_origins)
    report = dict(status='COMPLETE' if reference is not None else 'BASELINES_ONLY_PRIMARY_MISSING',
        summary=summary,contrasts=contrasts,feature_diagnostics=dict(train=train_diag,val=val_diag),
        best_single_expert=names[best],inputs_unchanged=True,test_set_accessed=False)
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
    parser.add_argument('--output',type=Path,help='New output directory; existing directories refused')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--bootstrap-samples',type=int,default=10000)
    parser.add_argument('--plan',action='store_true',help='Validate inputs and print protocol; no fitting or writes')
    parser.add_argument('--self-test',action='store_true',help='Run planted-signal and protocol checks')
    args = parser.parse_args()
    if args.self_test:
        self_test(args.threads)
        return
    for field in ['train_cache','val_cache','normalizer','frequency']+([] if args.plan else ['output']):
        if getattr(args,field) is None:
            parser.error(f'--{field.replace("_","-")} is required')
    if min(args.workers,args.threads,args.bootstrap_samples,args.frequency) < 1:
        parser.error('Worker/thread/sample/frequency counts must be positive')
    run(args)


if __name__ == '__main__':
    main()
