"""Bounded FAME repeat/permutation diagnostic; observes inputs and RNG without changing fits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
sys.path.insert(0, str(ROOT / "scripts/baseline_health_20261025"))
import run_health as h

p = argparse.ArgumentParser()
p.add_argument("case", choices=("predictable", "permuted"))
a = p.parse_args()
dest = ROOT / "research_review/fame_health_diagnostic_20261025" / ("cpu_" + a.case)
records = []
original = h.initialize


def observe(*args, **kwargs):
    import numpy as np
    import torch
    mod, pin = original(*args, **kwargs)
    train = mod.train_router
    def digest(value):
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous().numpy()
        return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    def wrapper(dataset, tau, seed, fp_df, loss_df, pred_t, tgt_t, experts):
        row = dict(tau=tau, seed=seed, fp=digest(fp_df.to_numpy()), loss=digest(loss_df.to_numpy()),
                   pred=digest(pred_t), target=digest(tgt_t), pred_stride=pred_t.stride(), target_stride=tgt_t.stride())
        router, history, cfg = train(dataset, tau, seed, fp_df, loss_df, pred_t, tgt_t, experts)
        row.update(best_loss=history.best_loss, epochs=history.epochs, torch_seed=torch.initial_seed(), device=str(router.device),
                   model={k: digest(v) for k, v in router.model.state_dict().items()})
        records.append(row)
        return router, history, cfg
    mod.train_router = wrapper
    return mod, pin


h.initialize = observe
h.worker(SimpleNamespace(method="fame", case=a.case, output=dest,
         repo_root=Path(r"C:\Data\Documents\Code\BasicTS"),
         official_root=ROOT / "research_review/reference_checks/baseline_health_20261025/third_party"))
h.write(dest / "fit_input_diagnostic.json", records)
print(json.dumps(records, indent=2))
