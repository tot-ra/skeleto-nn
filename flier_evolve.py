"""CMA-ES for the bird: python flier_evolve.py fly|takeoff|land --gens 200 --out runs/bird_fly.json"""
import argparse, json
import numpy as np
import flier as F

_G = {}
def _init():
    c, pc, ar = F.make_world(); _G["pc"] = pc; _G["ar"] = ar

def _eval(args):
    x, mode, tx = args
    return float(F.rollout(_G["pc"], _G["ar"], x, mode, T=(14.0 if mode == "fly" else 6.0 if mode == "takeoff" else 8.0), target_x=tx)["cost"])

def evolve(mode, gens, pop, procs, out, init=None):
    import cma, multiprocessing as mp
    rng = np.random.default_rng(0)
    x0 = np.array(json.load(open(init))["params"]) if init else F.DEFAULT.copy(); sig = (F.HI - F.LO) / 4.0
    es = cma.CMAEvolutionStrategy((x0 - F.LO) / sig, 0.35, dict(popsize=pop, seed=3, verbose=-9, bounds=[np.zeros_like(sig), (F.HI - F.LO) / sig]))
    with mp.get_context("spawn").Pool(procs, initializer=_init) as pool:
        for g in range(gens):
            tx = float(rng.uniform(9, 16)); X = es.ask()
            f = pool.map(_eval, [(F.LO + np.array(x) * sig, mode, tx) for x in X]); es.tell(X, f)
            print(f"gen {g} best {min(f):.3f} mean {np.mean(f):.3f}", flush=True)
            json.dump(dict(params=[float(v) for v in F.LO + np.array(es.result.xbest) * sig], gen=g, mode=mode), open(out, "w"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--gens", type=int, default=200); ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--procs", type=int, default=6); ap.add_argument("--out", default="runs/bird.json"); ap.add_argument("--init", default=None)
    a = ap.parse_args(); evolve(a.mode, a.gens, a.pop, a.procs, a.out, a.init)
