"""Bounded, low-priority fine-tuning with paired evaluation and preserved champions.

python -m motor.finetune --family quadruped --init runs/motor/quad_muscle/ckpt.pt \
 --name quad_review --amp runs/dog/dog_train_amp.npz --reset-amp --rounds 6 --minutes 10

A rejected round never becomes the next round's starting policy. Complete reports,
commands, dataset hashes and checkpoints are retained; published/game weights are
never overwritten. Run one job per family, at most 9 simulation workers in total.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix+'.part')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def select_candidate(initial, champion, candidate):
    from motor.benchmark import compare
    decision=compare(champion,candidate)
    original_guard=compare(initial,candidate)
    decision['reasons'] += ['baseline: '+r for r in original_guard['reasons']]
    decision['promote'] = decision['promote'] and original_guard['promote']
    return decision


def validate_seeds(validation, holdout=(404,505,606)):
    if not validation or len(set(validation))!=len(validation) or set(validation)&set(holdout):
        raise ValueError('validation seeds must be unique and disjoint from final holdout 404,505,606')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--family',required=True,choices=['biped','quadruped','bird'])
    ap.add_argument('--init',required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--amp');ap.add_argument('--reset-amp',action='store_true')
    ap.add_argument('--amp-w',type=float,default=1.)
    ap.add_argument('--amp-species')
    ap.add_argument('--anchor',help='Frozen original policy used to retain behavior on all bodies')
    ap.add_argument('--anchor-weight',type=float,default=10.)
    ap.add_argument('--amp-holdout',help='Untouched whole-clip reference split, reviewed once after all rounds')
    ap.add_argument('--workers',type=int,default=2);ap.add_argument('--rounds',type=int,default=6)
    ap.add_argument('--minutes',type=float,default=10.);ap.add_argument('--lr',type=float,default=3e-5)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--bird-species',choices=['crow','gull','heron','duck'])
    ap.add_argument('--reset-curriculum',action='store_true')
    ap.add_argument('--speed0',type=float)
    ap.add_argument('--assist0',type=float,default=1.)
    ap.add_argument('--spread',type=float,default=0.)
    ap.add_argument('--seeds',default='101,202,303');ap.add_argument('--seconds',type=float,default=8.)
    a=ap.parse_args()
    if not 1<=a.workers<=8 or not 1<=a.rounds<=100 or not 0<a.minutes<=60 or not 0<a.seconds<=60:
        ap.error('invalid bounded run size')
    if a.bird_species and a.family!='bird':
        ap.error('--bird-species requires --family bird')
    if a.amp_holdout and not a.amp:
        ap.error('--amp-holdout requires --amp')
    if not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.name):
        ap.error('--name must be a simple run name')
    seeds=tuple(map(int,a.seeds.split(',')))
    validate_seeds(seeds)
    root=Path('runs/finetune')/a.name
    root.mkdir(parents=True,exist_ok=True)
    lock=root/'active.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:
        ap.error(f'{lock} already exists; inspect that job before restarting')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        if (root/'history.json').exists():
            ap.error('run already exists; choose a new name to retain the old audit')
        os.nice(max(0,19-os.nice(0)))
        for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
            os.environ[key]='1'
        os.environ['BIPED_MUSCLES']='1';os.environ['QUAD_MUSCLES']='1'
        from motor.benchmark import evaluate, compare
        from motor.show import load
        ck,_=load(a.init)
        if ck['family']!=a.family: raise ValueError('checkpoint family mismatch')
        baseline=root/'baseline.pt';shutil.copy2(a.init,baseline)
        best_path=baseline
        best=evaluate(best_path,seeds,a.seconds)
        initial=best
        atomic_json(root/'baseline.json',best)
        digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        history=dict(schema='skeleto-nn/finetune/1', args=vars(a),pid=os.getpid(),
                     started=time.time(),state='training',
                     source_sha256={str(p):digest(p) for p in sorted(Path('motor').glob('*.py'))},
                     amp_sha256=digest(a.amp) if a.amp else None,
                     anchor_sha256=digest(a.anchor) if a.anchor else None,
                     initial_sha256=digest(baseline),champion=str(best_path),rounds=[])
        atomic_json(root/'history.json',history)
        for rnd in range(a.rounds):
            if any(digest(p)!=sha for p,sha in history['source_sha256'].items()):
                raise RuntimeError('motor source changed during run; preserving audit and stopping')
            name=f'{a.name}_r{rnd:03d}'
            cmd=[sys.executable,'-u','-m','motor.ppo','--family',a.family,'--name',name,
                 '--init',str(best_path),'--workers',str(a.workers),'--hours',str(a.minutes/60),
                 '--lr',str(a.lr),'--seed',str(a.seed+rnd),
                 '--speed0',str(a.speed0 if a.speed0 is not None else (.5 if a.family=='quadruped' else .8)),
                 '--assist0',str(a.assist0),'--spread',str(a.spread),
                 '--fade','.0001','--fade-ok','.0002']
            if a.reset_curriculum and best_path==baseline:
                cmd += ['--reset-curriculum']
            if a.anchor:
                cmd += ['--kick','*='+a.anchor,'--kick-weight',str(a.anchor_weight)]
            if a.amp_species:
                cmd += ['--amp-species',a.amp_species]
            if a.bird_species:
                cmd += ['--bird-species',a.bird_species]
            if a.amp:
                cmd += ['--amp',a.amp,'--amp-w',str(a.amp_w)]
                # Only reset when the champion still comes from the old dataset.
                if a.reset_amp and best_path==baseline:cmd+=['--reset-amp','--vwarm','3']
            with (root/f'round{rnd:03d}.log').open('w') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            candidate=root/f'round{rnd:03d}.pt'
            shutil.copy2(Path('runs/motor')/name/'ckpt.pt',candidate)
            report=evaluate(candidate,seeds,a.seconds)
            decision=select_candidate(initial,best,report)
            atomic_json(root/f'round{rnd:03d}.json',dict(report,comparison=decision))
            if decision['promote']:
                best_path=candidate;best=report
            history['champion']=str(best_path)
            history['rounds'].append(dict(round=rnd,command=cmd,**decision))
            atomic_json(root/'history.json',history)
            print(f'{a.family} round {rnd}: {decision}',flush=True)
        # One untouched seed set is checked once at the end, never used to pick rounds.
        holdout=(404,505,606)
        hold_initial=evaluate(baseline,holdout,a.seconds)
        hold_best=evaluate(best_path,holdout,a.seconds) if best_path!=baseline else hold_initial
        atomic_json(root/'holdout_baseline.json',hold_initial)
        atomic_json(root/'holdout_champion.json',hold_best)
        if a.amp_holdout:
            from motor.prior_review import review
            atomic_json(root/'prior_holdout.json', review(best_path,a.amp,a.amp_holdout))
        history.update(state='finished',finished=time.time(),
                       holdout=compare(hold_initial,hold_best))
        atomic_json(root/'history.json',history)
    except BaseException as error:
        if 'history' in locals():
            history.update(state='failed',error=str(error));atomic_json(root/'history.json',history)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
