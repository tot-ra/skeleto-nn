"""Report an AMP critic's response to untouched reference clips (never a flight/game quality gate)."""
from pathlib import Path
import hashlib
import numpy as np
from motor.amp import Discriminator


def review(checkpoint, train_path, holdout_path):
    import torch
    ck = torch.load(checkpoint, weights_only=False)
    state = ck.get('disc')
    if not state or not state.get('reference'):
        return dict(available=False, reason='checkpoint has no provenance-verified AMP critic for this source')
    d = Discriminator(train_path)
    if state['reference'] != d.reference:
        return dict(available=False, reason='checkpoint critic belongs to another source/schema')
    d.load(state)
    with np.load(holdout_path) as data:
        holdout=data['X'].copy()
        schema=data['segments'].tolist() if 'segments' in data else None
        hz=float(data['hz']) if 'hz' in data else None
    if schema!=d.reference['segments'] or hz!=d.reference['hz'] or holdout.shape[1]!=d.real.shape[1]:
        raise ValueError('holdout reference/schema mismatch')
    def measure(x):
        out=[]
        with torch.no_grad():
            for k in range(0,len(x),1024):
                out.extend(d._d(torch.as_tensor(x[k:k+1024])).numpy().tolist())
        return dict(samples=len(out), mean=float(np.mean(out)), std=float(np.std(out)))
    return dict(available=True, train=measure(d.real), holdout=measure(holdout),
                holdout_sha256=hashlib.sha256(Path(holdout_path).read_bytes()).hexdigest(),
                meaning='LSGAN response to recorded motions only; does not establish policy realism or physical flight')
