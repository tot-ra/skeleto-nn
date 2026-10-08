"""Standing on a slippery slope without falling: a small feedback controller on the physical body, evolved with CMA-ES.

The body starts upright on a ramp whose friction is too low to hold it, so it slides. The controller sets a crouch posture and
corrects with ankle and hip feedback on the pelvis tilt and tilt rate (in the vertical frame) and the arms swing out.
"""
from __future__ import annotations
import os, sys, json, math
import numpy as np
from bodies import human
from physics import PhysChar, CTRL_HZ
from terrain import Terrain

NP = 18
def default_params():
    return np.array([0.35, 0.55, 0.15, 0.4,           # hip flex, knee flex, trunk lean, arm out
                     0.8, 0.1, 0.0, 0.0,              # ankle pitch gain on tilt, rate; hip pitch gain on tilt, rate
                     0.0, 0.0, 0.0, 0.0,              # roll gains: ankle X tilt, rate; hip X tilt, rate
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0])    # spare: bias per axis

def build(alpha_deg, mu, c=None):
    c = c or human()
    a = math.radians(alpha_deg); L = 60.0
    tr = Terrain(-5, 80, -6, 6)
    tr.add_ramp(-6.0, L, -4, 4, math.tan(a) * (L + 6.0), 0.0)         # descends along +x
    pc = PhysChar(c, tr, friction=mu)
    return c, tr, pc

def rollout(c, pc, alpha_deg, mu, params, T=5.0, record=False, passive=False):
    a = math.radians(alpha_deg); p = np.asarray(params, float)
    z_ramp = math.tan(a) * 6.0 + 0.0
    # surface height at x=0 is tan(a)*(L+6 - 6)... ramp starts at x=-6 with height tan(a)*66 and falls linearly; compute directly
    x0 = 0.0; zsurf = math.tan(a) * (60.0 - x0)
    pc.set_state(np.array([x0, 0, zsurf + c.z0 + 0.004]), np.eye(3), np.zeros(pc.nj))
    names = [(c.skel.bones[i].name, "XYZ"[ax]) for i, ax in pc.dof_list]
    n = int(T * CTRL_HZ); alive = 0; tilt_acc = 0.0; imp_acc = 0.0; traj = []
    dist = 0.0
    for k in range(n):
        d = pc.data; pos, R = pc.root_pose(); w = R.T @ np.zeros(3)
        up = np.array([0, 0, 1.0]); lean_vec = R[:, 2]            # pelvis axis in the world
        pitch = math.atan2(lean_vec[0], lean_vec[2]); roll = -math.atan2(lean_vec[1], lean_vec[2])
        wv = d.qvel[3:6]; wworld = R @ wv
        pitch_rate = wworld[1]; roll_rate = wworld[0]
        tgt = np.zeros(pc.nj)
        if not passive:
            for j, (nm, ax) in enumerate(names):
                v = 0.0
                if nm.startswith("thigh") and ax == "Y": v = -p[0] + p[6] * pitch + p[7] * pitch_rate
                elif nm.startswith("shank") and ax == "Y": v = p[1]
                elif nm.startswith("foot") and ax == "Y": v = p[4] * pitch + p[5] * pitch_rate - 0.5 * (p[1] - p[0])
                elif nm.startswith("foot") and ax == "X": v = (p[8] * roll + p[9] * roll_rate) * (1 if nm.endswith("L") else -1) * 0 + p[8] * roll + p[9] * roll_rate
                elif nm.startswith("thigh") and ax == "X": v = (p[10] * roll + p[11] * roll_rate) * (1 if nm.endswith("L") else -1) * 0 + p[10] * roll + p[11] * roll_rate
                elif nm in ("lumbar", "lumbar1", "lumbar2", "thorax", "chest") and ax == "Y": v = p[2] * 0.2
                elif nm.startswith("uarm") and ax == "X": v = p[3] * (1 if nm.endswith("L") else -1)
                tgt[j] = v
        pc.step(tgt)
        if pc.bad_contact() or R[2, 2] < 0.6 or pos[2] - (math.tan(a) * (60.0 - pos[0])) < 0.45 * c.z0: break
        alive += 1; tilt_acc += math.acos(max(-1, min(1, R[2, 2]))); imp_acc += min(3.0, pc.body_impact())
        dist = pos[0] - x0
        if record: traj.append(pc.bone_ends())
    frac = alive / n
    cost = 4.0 * (1 - frac) + tilt_acc / max(alive, 1) + 0.5 * imp_acc / max(alive, 1)          # a slip that ends in a fall also hurts
    return dict(cost=float(cost), alive=frac, dist=float(dist), traj=traj)

_G = {}
def _eval(args):
    p, scen = args
    cs = []
    for (al, mu) in scen:
        key = (round(al), round(mu, 2))
        c, tr, pc = _G.setdefault(key, build(al, mu))
        cs.append(rollout(c, pc, al, mu, p)["cost"])
    return float(np.mean(cs))

def evolve(gens=80, pop=24, nproc=6, out="runs/slide.json", init=None):
    import cma, multiprocessing as mp
    rng = np.random.default_rng(0)
    es = cma.CMAEvolutionStrategy(init if init is not None else default_params(), 0.25, dict(popsize=pop, seed=3, verbose=-9))
    ctx = mp.get_context("spawn")
    with ctx.Pool(nproc) as pool:
        for g in range(gens):
            scen = [(float(rng.choice([8, 12, 16, 20, 25])), float(rng.choice([0.12, 0.2, 0.3, 0.4]))) for _ in range(8)]
            X = es.ask(); f = pool.map(_eval, [(x, scen) for x in X]); es.tell(X, f)
            print(f"gen {g} best {min(f):.3f} mean {np.mean(f):.3f}", flush=True)
            json.dump(dict(params=[float(v) for v in es.result.xbest], gen=g), open(out, "w"))

if __name__ == "__main__":
    ap = __import__("argparse").ArgumentParser(); ap.add_argument("--gens", type=int, default=80); ap.add_argument("--procs", type=int, default=6); ap.add_argument("--out", default="runs/slide.json"); ap.add_argument("--init", default=None)
    a = ap.parse_args()
    evolve(a.gens, 28, a.procs, a.out, init=(np.array(json.load(open(a.init))["params"]) if a.init else None))
