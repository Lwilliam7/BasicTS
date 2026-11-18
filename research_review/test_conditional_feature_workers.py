from pathlib import Path
import sys
import uuid
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import compare_conditional_combinations as c

if __name__ == '__main__':
    destination=Path(__file__).resolve().parent/('feature_worker_check_'+uuid.uuid4().hex[:8]+'.npz')
    history=torch.randn(4,96,2,generator=torch.Generator().manual_seed(7))
    array,diag=c.features({'histories':history},24,2,destination,'synthetic-worker-check')
    assert array.shape==(4,42) and np.isfinite(array).all()
    again,_=c.features({'histories':history},24,2,destination,'synthetic-worker-check')
    assert np.array_equal(array,again)
    print('PASS: spawned bounded feature workers, finite features and cached reuse',diag)
