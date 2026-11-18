"""Synthetic integration checks. No research forecasts or remote server access."""
import argparse
import importlib.util
import json
import uuid
import shutil
import tarfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1]/'scripts/compare_conditional_combinations.py'
spec = importlib.util.spec_from_file_location('runner',SOURCE)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
c.torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parent/'server_runner_synthetic_checks'/uuid.uuid4().hex[:10]
ROOT.mkdir(parents=True,exist_ok=False)
work = ROOT/'work'
work.mkdir()
names = ['DLinear','PatchTST','iTransformer','TimesNet','ModernTCN']
core = names[:3]
gen = torch.Generator().manual_seed(18)
std = torch.tensor([1.,2.])

def cache(n, start, role):
    target = torch.randn(n,96,2,generator=gen)
    return dict(dataset='ETTh1',histories=torch.randn(n,96,2,generator=gen),
        targets=target,target_masks=torch.ones_like(target,dtype=torch.bool),
        prediction_stack=target[...,None]+torch.randn(n,96,2,5,generator=gen),
        absolute_window_starts=torch.arange(n)*96+start,expert_names=names,
        forecast_horizon=96,input_len=96,num_windows=n,num_features=2,cache_role=role,
        provenance=dict(source_caches={'block_b_oos':'synthetic','block_c_oos':'synthetic'}))

train=cache(80,0,'router_train_20_60')
val=cache(24,8000,'router_val_60_80')
def archived_losses(pred):
    array=c.metrics(val,pred,std)
    return dict(per_window_mae=torch.tensor(array[:,0]),per_window_mse=torch.tensor(array[:,1]))
comp={k:archived_losses(val['prediction_stack'][...,idx].mean(-1))
      for k,idx in [('equal5',list(range(5))),('equal3',list(range(3))),('bates_all5',list(range(5)))]}
torch.save(comp,work/'comparators_ETTh1.pt')
arms={k:{m:float(v['per_window_'+m].mean()) for m in ('mae','mse')} for k,v in comp.items()}
c.write_json(work/'comparators_ETTh1.json',dict(dataset='ETTh1',core=core,arms=arms,
    test_set_accessed=False,router_train_causality='PASS'))
origins=train['absolute_window_starts'].numpy()
folds=c.folds(origins,96)
fold_meta=[dict(eval_lo=int(v[0]),eval_hi=int(v[-1])+1,eval_origin=int(origins[v[0]]),
    num_legal_fit_windows=len(t),latest_fit_target_end=int(origins[t[-1]]+96)) for t,v in folds]
fold_scores=[]
for _,v in folds:
    small=c.subset(train,v)
    mae,mse=c.metrics(small,small['prediction_stack'][...,:3].mean(-1),std).mean(0)
    fold_scores.append(dict(eval_lo=int(v[0]),eval_hi=int(v[-1])+1,mae=float(mae),mse=float(mse)))
for seed in c.SEEDS:
    torch.save({'wide_top3':comp['equal3']},work/f'seed{seed}_ETTh1.pt')
    meta=dict(dataset='ETTh1',seed=seed,horizon=96,candidates=names,core=core,
        test_set_accessed=False,scorer_folds=[fold_meta,fold_meta],
        core_selection=dict(fold_rows=fold_scores),arms=dict(wide_top3=arms['equal3']))
    meta.update({k:True for k in ('freeze_matches','all_folds_causal','checkpoints_unchanged',
                                 'tensors_finite','oof_mask_covers_folds')})
    c.write_json(work/f'seed{seed}_ETTh1.json',meta)

reference,comparators,audit=c.archived_reference(work,train,val,std)
assert reference.shape==(5,24,2) and not audit['historical_cache_byte_identity_proven']
for bad in ('forecast','normalizer','origin','training'):
    bad_train=dict(train); bad_val=dict(val); bad_std=std.clone()
    if bad=='forecast':
        bad_val['prediction_stack']=val['prediction_stack'].clone()
        bad_val['prediction_stack'][0,0,0,0]+=100
    if bad=='normalizer': bad_std=std*2
    if bad=='origin': bad_train['absolute_window_starts']=train['absolute_window_starts']+1
    if bad=='training':
        bad_train['prediction_stack']=train['prediction_stack'].clone()
        bad_train['prediction_stack'][20,...,0]+=100
    try: c.archived_reference(work,bad_train,bad_val,bad_std)
    except ValueError: pass
    else: raise AssertionError('Accepted invalid archive binding: '+bad)

for name,value in [('train',train),('val',val),('normalizer',{'scaler_std':std})]:
    torch.save(value,ROOT/(name+'.pt'))
args=argparse.Namespace(train_cache=ROOT/'train.pt',val_cache=ROOT/'val.pt',normalizer=ROOT/'normalizer.pt',
    primary=None,archived_work=work,frequency=24,output=ROOT/'output',resume=True,
    threads=2,workers=1,bootstrap_samples=10,plan=False,expected_dataset='ETTh1',expected_horizon=96)
counter=[0]
def fake_fit(*args,**kwargs):
    counter[0]+=1
    if counter[0]==22: raise RuntimeError('Simulated interrupted final fit')
    return lambda x:np.full((len(x),5),.2,dtype=np.float32)
def fake_features(cache,*args):
    return np.zeros((len(cache['targets']),3),dtype=np.float32),{'synthetic':True}
with patch.object(c,'fit',side_effect=fake_fit),patch.object(c,'features',side_effect=fake_features):
    try: c.run(args)
    except RuntimeError as exc: assert 'Simulated' in str(exc)
    else: raise AssertionError('Interruption did not fire')
    saved=args.output/'checkpoints/fforma_mae_seed7.npz'
    assert saved.exists()
    before=saved.stat().st_mtime_ns
    counter[0]=100
    c.run(args)
    assert saved.stat().st_mtime_ns==before
    assert json.loads((args.output/'results.json').read_text())['status']=='COMPLETE'
    count=counter[0]
    c.run(args)
    assert counter[0]==count, 'Completed resume refit a model'
    args.frequency=12
    try: c.run(args)
    except ValueError as exc: assert 'Cannot resume' in str(exc)
    else: raise AssertionError('Changed protocol accepted')

lock=ROOT/'check.lock'
with c.run_lock(lock):
    try:
        with c.run_lock(lock): pass
    except RuntimeError: pass
    else: raise AssertionError('Duplicate queue lock accepted')
with c.run_lock(lock): pass

u=c.server_unit(Path('/repo'),'Traffic',96)
assert 'costarts_lh/h96/Traffic' in Path(u['train_cache']).as_posix()
u=c.server_unit(Path('/repo'),'ETTh1',96)
assert 'costarts_h96/ETTh1' in Path(u['train_cache']).as_posix()
inventory_args=argparse.Namespace(repo_root=ROOT,datasets=['Traffic'],horizons=[336,720])
units=c.server_inventory(inventory_args)
assert units[0]['state']=='BLOCKED' and units[1]['state']=='EXCLUDED'

# Exercise the queue/summary/pack lifecycle with a stand-in process. Numerical
# unit validation and interrupted fits were exercised independently above.
repo=ROOT/'repo'
u=c.server_unit(repo,'ETTh1',96)
for key,source in [('train_cache','train.pt'),('val_cache','val.pt'),('normalizer','normalizer.pt')]:
    dest=Path(u[key]); dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/source,dest)
shutil.copytree(work,Path(u['archived_work']))
queue_args=argparse.Namespace(server='run',repo_root=repo,output=ROOT/'queue',datasets=['ETTh1'],
    horizons=[96],jobs=1,threads=2,workers=1,bootstrap_samples=10)
def fake_process(command,**kwargs):
    dest=Path(command[command.index('--output')+1]); dest.mkdir(parents=True,exist_ok=True)
    for name in ['results.json','report.md']:
        shutil.copy2(args.output/name,dest/name)
    assert kwargs['env']['OMP_NUM_THREADS']=='2'
    assert '--archived-work' in command and '--resume' in command
    return SimpleNamespace(returncode=0)
with patch.object(c.subprocess,'run',side_effect=fake_process): c.server_main(queue_args)
assert json.loads((queue_args.output/'summary.json').read_text())['status']=='COMPLETE'
with patch.object(c.subprocess,'run',return_value=SimpleNamespace(returncode=1)):
    try: c.server_main(queue_args)
    except SystemExit as exc: assert exc.code==2
    else: raise AssertionError('Failed worker did not produce nonzero queue exit')
assert json.loads((queue_args.output/'summary.json').read_text())['status']=='INCOMPLETE', 'Stale success masked failure'
queue_args.server='pack'
c.server_main(queue_args)
with tarfile.open(queue_args.output.with_suffix('.tar.gz')) as archive:
    assert any(p.endswith('summary.csv') for p in archive.getnames())
print('PASS: archive alignment, training/normalizer drift rejection, interrupted/complete resume, changed-protocol refusal, duplicate lock, server layouts/exclusions, queue invocation, stale-success rejection and packing')
