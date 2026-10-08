"""Pregnant body in a fall: stiff body vs the general reflex vs the reflex evolved for this body (the belly is scored most).
python preg_eval.py runs/reflex_preg.json runs/preg_eval"""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
from reflex import *
from reflex_eval import summarize
from bodies import persona
from render import Scene, Actor, caption, save_gif
from terrain import Terrain

pj = sys.argv[1]; outd = sys.argv[2] if len(sys.argv) > 2 else "runs/preg_eval"; os.makedirs(outd, exist_ok=True)
c = persona("pregnant"); pc = PhysChar(c); kp = np.array([pc.model.actuator_gainprm[a, 0] for a in pc.aid])
pre = np.array(json.load(open(pj))["params"]); gen = np.array(json.load(open("results/reflex_fall.json"))["params"])
sc = sample_scenarios(np.random.default_rng(321), 120); res = {}
for nm, p, pas in (("stiff body", default_params(), True), ("general reflex", gen, False), ("reflex evolved for this body", pre, False)):
    rows = [run_scenario(c, p, s, pc=pc, base_kp=kp, passive=pas) for s in sc]
    res[nm] = dict(cost=float(np.mean([r["cost"] for r in rows])), abdomen_risk=float(np.mean([r["risk"]["abdomen"] for r in rows])), head_risk=float(np.mean([r["risk"]["head"] for r in rows])),
                   torso_risk=float(np.mean([r["risk"]["torso"] for r in rows])), arms_risk=float(np.mean([r["risk"]["arms"] for r in rows])), legs_risk=float(np.mean([r["risk"]["legs"] for r in rows])),
                   abdomen_peak=float(np.mean([r["peak"]["abdomen"] for r in rows])))
print(json.dumps(res, indent=1)); json.dump(res, open(os.path.join(outd, "summary.json"), "w"), indent=1)
cases = {"shove_forward": dict(kind="shove", dir=(1.0, 0.0), mag=3.0), "shove_back": dict(kind="shove", dir=(-1.0, 0.0), mag=3.0), "drop_1m": dict(kind="drop", h=1.0, v=np.array([0.5, 0.0, 0.0]))}
for nm, s in cases.items():
    a = run_scenario(c, gen, s, record=True, pc=pc, base_kp=kp); b = run_scenario(c, pre, s, record=True, pc=pc, base_kp=kp)
    n = min(len(a["traj"]), len(b["traj"])); scene = Scene(Terrain(), [Actor(c, "a", base=(0.55, 0.55, 0.58)), Actor(c, "b", base=(0.55, 0.30, 0.50))], 480, 300); frames = []
    for k in range(n):
        S, E, R, st = a["traj"][k]; S2, E2, R2, st2 = b["traj"][k]; off = np.array([0, 1.1, 0])
        scene.pose(scene.actors[0], S + off, E + off, R); scene.pose(scene.actors[1], S2 - off, E2 - off, R2)
        mid = 0.5 * (S[0] + S2[0]) + np.array([0, 0, 0.3]); img = scene.render([mid[0] + 0.3, 0.0, 0.7], 5.6, 62, -8)
        frames.append(caption(img, f"{nm}: left general reflex (belly risk {a['risk']['abdomen']:.2f}), right reflex evolved for this body ({b['risk']['abdomen']:.2f})", f"t={k / 30:.1f}s"))
    scene.close(); save_gif(frames, os.path.join(outd, "preg_" + nm + ".gif"), fps=30)
