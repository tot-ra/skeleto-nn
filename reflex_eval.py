"""Held-out evaluation of the evolved fall/landing reflex against a stiff statue and the hand-made starting pose, plus side by side GIFs.
python reflex_eval.py runs/reflex.json [out_dir]"""
import os, sys, json, math
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
from reflex import *
from render import Scene, Actor, caption, save_gif
from terrain import Terrain

def stats(c, pc, kp, params, scens, passive=False):
    rows = [run_scenario(c, params, s, pc=pc, base_kp=kp, passive=passive) for s in scens]
    return rows

def summarize(rows, scens):
    out = {}
    for kind in ("shove", "drop", "all"):
        idx = [i for i, s in enumerate(scens) if kind == "all" or s["kind"] == kind]
        if not idx: continue
        r = [rows[i] for i in idx]
        out[kind] = dict(n=len(idx), cost=float(np.mean([x["cost"] for x in r])), head=float(np.mean([x["peak"]["head"] for x in r])), head_hit=float(np.mean([x["peak"]["head"] > 1.0 for x in r])),
                         torso=float(np.mean([x["peak"]["torso"] for x in r])), arms=float(np.mean([x["peak"]["arms"] for x in r])), legs=float(np.mean([x["peak"]["legs"] for x in r])),
                         head_risk=float(np.mean([x["risk"]["head"] for x in r])), torso_risk=float(np.mean([x["risk"]["torso"] for x in r])), arms_risk=float(np.mean([x["risk"]["arms"] for x in r])), legs_risk=float(np.mean([x["risk"]["legs"] for x in r])),
                         severe=float(np.mean([max(x["risk"].values()) > 0.5 for x in r])))
    return out

if __name__ == "__main__":
    pj = sys.argv[1]; outd = sys.argv[2] if len(sys.argv) > 2 else "runs/reflex_eval"
    os.makedirs(outd, exist_ok=True)
    params = np.array(json.load(open(pj))["params"])
    old = np.array(json.load(open(sys.argv[3]))["params"]) if len(sys.argv) > 3 else None
    c = human(); pc = PhysChar(c); kp = np.array([pc.model.actuator_gainprm[a, 0] for a in pc.aid])
    rng = np.random.default_rng(12345)
    scens = sample_scenarios(rng, 240)
    res = {}
    runs_ = [("statue (stiff, no reflex)", default_params(), True), ("hand-made starting pose", default_params(), False)]
    if old is not None: runs_.append(("evolved on peak force only (v2)", old if len(old) == len(default_params()) else np.concatenate([old, [1.0, 1.0]]), False))
    runs_.append(("evolved on bone-injury score", params, False))
    for name, p, pas in runs_:
        rows = stats(c, pc, kp, p, scens, pas); res[name] = summarize(rows, scens)
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(outd, "summary.json"), "w"), indent=1)
    # GIFs: same scenario, statue (left) vs evolved reflex (right)
    cases = {"shove_forward": dict(kind="shove", dir=(1.0, 0.0), mag=3.0), "shove_side": dict(kind="shove", dir=(0.0, 1.0), mag=3.0), "shove_back": dict(kind="shove", dir=(-1.0, 0.0), mag=3.0),
             "drop_1m": dict(kind="drop", h=1.0, v=np.array([0.8, 0.0, 0.0])), "drop_2m": dict(kind="drop", h=2.0, v=np.array([0.0, 0.0, 0.0])), "drop_2p5m_fwd": dict(kind="drop", h=2.5, v=np.array([1.6, 0.0, 0.0]))}
    for nm, sc in cases.items():
        a = run_scenario(c, default_params(), sc, record=True, passive=True, pc=pc, base_kp=kp)
        b = run_scenario(c, params, sc, record=True, pc=pc, base_kp=kp)
        n = min(len(a["traj"]), len(b["traj"]))
        scene = Scene(Terrain(), [Actor(c, "a", base=(0.55, 0.55, 0.58)), Actor(c, "b", base=(0.30, 0.45, 0.75))], 480, 300)
        frames = []
        for k in range(n):
            S, E, R, st = a["traj"][k]; S2, E2, R2, st2 = b["traj"][k]
            off = np.array([0, 1.1, 0]); off2 = np.array([0, -0.0, 0])
            scene.pose(scene.actors[0], S + off, E + off, R); scene.pose(scene.actors[1], S2 - np.array([0, 1.1, 0]), E2 - np.array([0, 1.1, 0]), R2)
            mid = 0.5 * (S[0] + S2[0]) + np.array([0, 0, 0.3])
            img = scene.render([mid[0] + 0.3, 0.0, 0.7], 5.6, 62, -8)
            frames.append(caption(img, f"{nm}: left = stiff statue (injury {a['cost']:.1f}, head risk {a['risk']['head']:.2f}), right = evolved reflex (injury {b['cost']:.1f}, head risk {b['risk']['head']:.2f})", f"t={k / 30:.1f}s"))
        scene.close(); save_gif(frames, os.path.join(outd, nm + ".gif"), fps=30)
        print(nm, "statue", a["peak"], "evolved", b["peak"])
