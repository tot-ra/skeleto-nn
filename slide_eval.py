"""Held-out evaluation of the slope controller (stand on a slippery slope without falling) and a GIF: stiff body vs evolved.
python slide_eval.py runs/slide2.json [out_dir]"""
import os, sys, json, math
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
from slide import build, rollout, default_params
from render import Scene, Actor, caption, save_gif
from terrain import Terrain

pj = sys.argv[1]; outd = sys.argv[2] if len(sys.argv) > 2 else "runs/slide_eval"; os.makedirs(outd, exist_ok=True)
params = np.array(json.load(open(pj))["params"])
rng = np.random.default_rng(777)
scen = [(float(rng.choice([8, 12, 16, 20, 25])), float(rng.choice([0.12, 0.2, 0.3, 0.4]))) for _ in range(40)]
res = {}
for name, p, passive in (("stiff body, no controller", default_params(), True), ("evolved controller", params, False)):
    alive = []; dist = []; cost = []
    for (al, mu) in scen:
        c, tr, pc = build(al, mu); r = rollout(c, pc, al, mu, p, passive=passive)
        alive.append(r["alive"]); dist.append(r["dist"]); cost.append(r["cost"])
    res[name] = dict(stays_up_5s=float(np.mean(np.array(alive) > 0.99)), mean_time_fraction_up=float(np.mean(alive)), mean_slide_m=float(np.mean(dist)), cost=float(np.mean(cost)))
print(json.dumps(res, indent=1)); json.dump(res, open(os.path.join(outd, "summary.json"), "w"), indent=1)
al, mu = 16.0, 0.20
c, tr, pc = build(al, mu); a = rollout(c, pc, al, mu, default_params(), passive=True, record=True)
c2, tr2, pc2 = build(al, mu); b = rollout(c2, pc2, al, mu, params, record=True)
n = min(len(a["traj"]), len(b["traj"])) if a["traj"] and b["traj"] else 0
scene = Scene(tr, [Actor(c, "a", base=(0.55, 0.55, 0.58)), Actor(c, "b", base=(0.30, 0.45, 0.75))], 480, 300)
frames = []
for k in range(0, max(len(a["traj"]), len(b["traj"]))):
    S, E, R = a["traj"][min(k, len(a["traj"]) - 1)]; S2, E2, R2 = b["traj"][min(k, len(b["traj"]) - 1)]
    off = np.array([0, 0.9, 0]); scene.pose(scene.actors[0], S + off, E + off, R); scene.pose(scene.actors[1], S2 - off, E2 - off, R2)
    mid = 0.5 * (S[0] + S2[0]) + np.array([0, 0, 0.3])
    img = scene.render([mid[0], 0.0, mid[2]], 5.0, 70, -10)
    frames.append(caption(img, f"slope {al:.0f} deg, friction {mu}: left stiff body (stays up {a['alive'] * 5:.1f} s), right evolved controller ({b['alive'] * 5:.1f} s)", f"t={k / 30:.1f}s"))
scene.close(); save_gif(frames, os.path.join(outd, "slide.gif"), fps=30)
