"""Deterministic, no-harness paired evaluation of motor checkpoints.

python -m motor.benchmark CHECKPOINT --out runs/review.json
The score alone is not a shipping gate: survival, direction, tilt, foot slip and
flight altitude are reported separately. Seeds and complete scenarios are kept
so candidate weights can be compared on exactly the same initial conditions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from motor.ppo import make_env, mlp_np
from motor.show import load, situations
from motor import evaluate as ev


def evaluate(path, seeds=(101, 202, 303), seconds=8.0, names=None):
    ck, pi = load(path)
    family = ck['family']
    mean, std = ck['mean'], np.sqrt(ck['var'] + 1e-6)
    environment = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [Path(n) for n in ("physics.py", "bodies.py", "muscles.py", "air.py",
                     "motor/family.py", f"motor/{family}.py", "motor/evaluate.py", "motor/show.py", "motor/benchmark.py")]}
    episodes = []
    for seed in seeds:
        probe = make_env(family, seed=seed)
        suite = list(situations(family, probe))
        # Every species seen by training must be visible to the forgetting gate.
        if family == 'quadruped':
            from bodies import QUADS
            from motor.quadruped import canonical
            for species in ('wolf', 'goat', 'bear'):
                suite += [(f'{species}_{gait}', dict(body=canonical(QUADS[species]), goal=(speed,0.)), '')
                          for gait,speed in [('walk',.5),('trot',1.)]]
        elif family == 'bird':
            from bodies import BIRDS
            suite += [('duck',dict(body=BIRDS['duck'],goal=(2.8,0.),climb=0.),'')]
        unknown = set(names or ()) - {name for name, _, _ in suite}
        if unknown:
            raise ValueError(f"unknown cases: {sorted(unknown)}")
        for name, kw, _title in suite:
            if names and name not in names:
                continue
            # Fresh environment for each case: adding a case cannot shift other RNG streams.
            env = make_env(family, seed=seed)
            body = kw.get('body', env.default_body())
            env._build(body)
            args = dict(body=body, friction=1., wind=np.zeros(3),
                        sores={k: 0. for k in env.SORES}, profile=dict(ev.DEFAULT_WEIGHTS))
            args.update(kw)
            args['goal'] = np.asarray(args.get('goal', (0., 0.))) * (env.speed_unit() if family == 'quadruped' else 1.)
            o = env.reset(**args)
            env.assist = 0.
            env.run_assist = 0.
            if o.shape != mean.shape or env.act_dim != ck['act_dim']:
                raise ValueError(f'{family}: checkpoint/environment shape mismatch')
            hz = env.CTRL_HZ
            goals, speeds, ups, facing, slips, costs = [], [], [], [], [], []
            p0, _ = env.pc.root_pose()
            foot_names = [leg.chain[-1] for leg in env.c.legs]
            # Leg.chain[-1] is a skeleton index; foot endpoints give a consistent slip proxy.
            contact_steps = 0
            prev = None
            done = False
            for _ in range(round(seconds * hz)):
                a = mlp_np(np.clip((o - mean) / std, -10, 10).astype(np.float32), pi)
                o, _r, done, info = env.step(a)
                p, R = env.pc.root_pose()
                if not np.all(np.isfinite(o)):
                    raise ValueError(f'non-finite rollout: {family}/{name}/{seed}')
                goals.append(info['goal_now']); speeds.append(info['speed'])
                ups.append(float(R[2, 2])); costs.append(info['costs'])
                g = env.goal_world
                if np.linalg.norm(g) > 1e-5:
                    facing.append(float(R[:2, 0] @ g / max(1e-9, np.linalg.norm(R[:2, 0]) * np.linalg.norm(g))))
                if family != 'bird':
                    _S, E, _R = env.pc.bone_ends()
                    feet = E[foot_names]
                    planted = env.pc.foot_contacts() > .5
                    contact_steps += int(planted.any())
                    if prev is not None:
                        shared = planted & prev[1]
                        if shared.any():
                            slips.extend((np.linalg.norm(feet[shared, :2] - prev[0][shared, :2], axis=1) * hz).tolist())
                    prev = (feet.copy(), planted)
                if done or info['trunc']:
                    break
            n = len(goals)
            survival = n / round(seconds * hz)
            episode = dict(case=name, seed=seed, seconds=n/hz, failed=bool(done),
                           survival=survival, goal=float(np.mean(goals)),
                           score=survival * (.3 + float(np.mean(goals))),
                           speed_m_s=float(np.mean(speeds)), min_up=min(ups),
                           tilt_p95_deg=float(np.percentile(np.degrees(np.arccos(np.clip(ups, -1, 1))), 95)),
                           facing_mean=float(np.mean(facing)) if facing else 1.,
                           slip_p95_m_s=float(np.percentile(slips, 95)) if slips else None,
                           altitude_delta_m=float(p[2] - p0[2]),
                           flight_height_error_spans=(abs(float(p[2] - p0[2]) - env.climb*n/hz) / env.wingspan() if family == "bird" else None),
                           contact_fraction=contact_steps/n if family != "bird" else None,
                           costs={k: float(np.mean([c[k] for c in costs])) for k in costs[0]})
            episodes.append(episode)
            print(f"{family}/{name}/{seed}: {n/hz:.2f}s goal={episode['goal']:.3f} score={episode['score']:.3f}", flush=True)
    cases = {}
    for name in sorted({e['case'] for e in episodes}):
        rows = [e for e in episodes if e['case'] == name]
        cases[name] = dict(score=float(np.mean([r['score'] for r in rows])),
                           survival=float(np.mean([r['survival'] for r in rows])),
                           failures=sum(r['failed'] for r in rows),
                           goal=float(np.mean([r['goal'] for r in rows])),
                           tilt_p95_deg=float(np.mean([r['tilt_p95_deg'] for r in rows])),
                           facing_mean=float(np.mean([r['facing_mean'] for r in rows])),
                           slip_p95_m_s=(float(np.mean([r['slip_p95_m_s'] for r in rows if r['slip_p95_m_s'] is not None]))
                                         if any(r['slip_p95_m_s'] is not None for r in rows) else None),
                           flight_height_error_spans=(float(np.mean([r['flight_height_error_spans'] for r in rows])) if family == 'bird' else None),
                           contact_fraction=float(np.mean([r['contact_fraction'] for r in rows])) if family != 'bird' else None)
    return dict(schema='skeleto-nn/motor-benchmark/1', checkpoint=str(path),
                sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                family=family, iteration=ck.get('it'), environment=environment, seconds=seconds, seeds=list(seeds),
                harness=0., score=float(np.mean([e['score'] for e in episodes])), cases=cases,
                episodes=episodes)


def compare(baseline, candidate, tolerance=.03, gain=.005):
    """Conservative paired promotion. No species/case may hide behind the mean."""
    keys = ('schema', 'family', 'seconds', 'seeds', 'harness', 'environment')
    if any(baseline[k] != candidate[k] for k in keys) or baseline['cases'].keys() != candidate['cases'].keys():
        raise ValueError('unpaired evaluation suites')
    reasons = []
    if not math.isfinite(candidate['score']) or candidate['score'] < baseline['score'] + gain:
        reasons.append('aggregate score did not improve')
    for name, old in baseline['cases'].items():
        new = candidate['cases'][name]
        if new['survival'] < old['survival'] - tolerance:
            reasons.append(f'{name}: survival regressed')
        if new['score'] < old['score'] - tolerance:
            reasons.append(f'{name}: score regressed')
        if new['failures'] > old['failures']:
            reasons.append(f'{name}: more failures')
        if new['tilt_p95_deg'] > old['tilt_p95_deg'] + 5.:
            reasons.append(f'{name}: tilt regressed')
        if new['facing_mean'] < old['facing_mean'] - .05:
            reasons.append(f'{name}: facing regressed')
        if old['slip_p95_m_s'] is not None:
            if new['slip_p95_m_s'] is None or new['slip_p95_m_s'] > old['slip_p95_m_s'] + .1:
                reasons.append(f'{name}: foot slip regressed or unmeasured')
        if old['contact_fraction'] is not None and new['contact_fraction'] < old['contact_fraction'] - .1:
            reasons.append(f'{name}: ground contact regressed')
        if old['flight_height_error_spans'] is not None and new['flight_height_error_spans'] > old['flight_height_error_spans'] + .25:
            reasons.append(f'{name}: flight height regressed')
    return dict(promote=not reasons, reasons=reasons,
                score_delta=candidate['score']-baseline['score'])


def main():
    os.environ.setdefault("BIPED_MUSCLES", "1")
    os.environ.setdefault("QUAD_MUSCLES", "1")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('checkpoint'); ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default='101,202,303'); ap.add_argument('--seconds', type=float, default=8.)
    ap.add_argument('--cases', default=None); ap.add_argument('--baseline')
    a = ap.parse_args()
    if not 0. < a.seconds <= 60.:
        ap.error('--seconds must be in (0,60]')
    result = evaluate(a.checkpoint, tuple(map(int, a.seeds.split(','))), a.seconds,
                      set(a.cases.split(',')) if a.cases else None)
    if not result['episodes']:
        ap.error('no selected cases')
    if a.baseline:
        result['comparison'] = compare(json.loads(Path(a.baseline).read_text()), result)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('family','score','cases')}, indent=2))


if __name__ == '__main__':
    main()
