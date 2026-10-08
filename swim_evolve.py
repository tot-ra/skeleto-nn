"""CMA-ES for the swimming stroke generator.
python swim_evolve.py under  --gens 150 --out runs/swim_under.json
python swim_evolve.py surface --gens 150 --out runs/swim_surface.json"""
import argparse, json, math, sys
import numpy as np
import swimmer as S

_G = {}
def _init():
    c, pc, sw = S.make_pool(); _G["pc"] = pc; _G["sw"] = sw

def sample_goals(rng, n):
    goals = []
    for _ in range(n):
        if rng.random() < 0.5: goals.append(np.array([rng.uniform(2.0, 5.0), rng.uniform(-1.0, 1.0), rng.uniform(0.3, 1.0)]))        # dive to the bottom
        else: goals.append(np.array([rng.uniform(5.0, 8.0), rng.uniform(-1.5, 1.5), S.W_SURF - rng.uniform(0.8, 2.0)]))               # swim far under water
    return goals

def _eval(args):
    x, mode, goals = args
    pc, sw = _G["pc"], _G["sw"]
    if mode == "surface": return float(S.rollout(pc, sw, x, "surface", None, T=12.0)["cost"])
    return float(np.mean([S.rollout(pc, sw, x, "under", g, T=9.0)["cost"] for g in goals]))

def evolve(mode, gens, pop, procs, out, init=None):
    import cma, multiprocessing as mp
    rng = np.random.default_rng(0)
    x0 = (np.array(json.load(open(init))["params"]) if init else S.DEFAULT.copy())
    sig = (S.HI - S.LO) / 4.0
    es = cma.CMAEvolutionStrategy((x0 - S.LO) / sig, 0.35, dict(popsize=pop, seed=2, verbose=-9, bounds=[(S.LO - S.LO) / sig, (S.HI - S.LO) / sig]))
    with mp.get_context("spawn").Pool(procs, initializer=_init) as pool:
        for g in range(gens):
            goals = sample_goals(rng, 3); X = es.ask()
            f = pool.map(_eval, [(S.LO + np.array(x) * sig, mode, goals) for x in X]); es.tell(X, f)
            print(f"gen {g} best {min(f):.3f} mean {np.mean(f):.3f}", flush=True)
            json.dump(dict(params=[float(v) for v in S.LO + np.array(es.result.xbest) * sig], gen=g, mode=mode), open(out, "w"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--gens", type=int, default=150); ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--procs", type=int, default=10); ap.add_argument("--out", default="runs/swim.json"); ap.add_argument("--init", default=None)
    a = ap.parse_args(); evolve(a.mode, a.gens, a.pop, a.procs, a.out, a.init)
