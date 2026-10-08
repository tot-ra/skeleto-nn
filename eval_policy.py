"""Evaluate a trained tracking policy: survival statistics and a GIF (physical body in colour, reference as a pale ghost).

python eval_policy.py runs/h1/ckpt.pt --episodes 40 --gif runs/h1_eval.gif [--push 1.0] [--clip-speeds 0.9,1.3]"""
from __future__ import annotations
import os, sys, argparse, json, math
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
import torch
from bodies import human, HumanBody, quadruped, bird
from ref import make_clip
from env import Env
from skeleton import rot

def load_policy(path):
    ck = torch.load(path, weights_only=False)
    def lay(sd, pre):
        ws = [(sd[f"{pre}.{i}.weight"].numpy().T, sd[f"{pre}.{i}.bias"].numpy()) for i in (0, 2, 4)]
        return ws
    pi = lay({k: v for k, v in ck["pi"].items()}, "")  if False else None
    sd = ck["pi"]; ws = [(sd[f"{i}.weight"].numpy().T, sd[f"{i}.bias"].numpy()) for i in (0, 2, 4)]
    mean, var = ck["mean"], ck["var"]
    def act(o):
        x = np.clip((o - mean) / np.sqrt(var + 1e-6), -10, 10)
        for i, (W, b) in enumerate(ws):
            x = x @ W + b
            if i < 2: x = np.tanh(x)
        return x
    return act, ck

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--episodes", type=int, default=30); ap.add_argument("--gif", default=None)
    ap.add_argument("--push", type=float, default=0.0); ap.add_argument("--seed", type=int, default=777); ap.add_argument("--speeds", default="0.0,0.9,1.3,1.8")
    ap.add_argument("--body", default="plain"); ap.add_argument("--seconds", type=float, default=10.0)
    a = ap.parse_args()
    act, ck = load_policy(a.ckpt)
    speeds = tuple(float(x) for x in a.speeds.split(","))
    c = human() if a.body == "plain" else human(HumanBody(**json.loads(a.body)))
    clips = [make_clip(c, 9000 + k, T=a.seconds + 2, speeds=speeds) for k in range(8)]
    env = Env(c, clips, seed=a.seed, push=(1.0 if a.push > 0 else 0.0, a.push), max_len=int(a.seconds * 30))
    lens = []; falls = 0; tot = 0.0
    best = None
    for ep in range(a.episodes):
        o = env.reset(); env.t = 0; env.t0 = 0
        # start from the clip's first frame for a clean comparison
        L = 0; traj = []
        while True:
            S, E, R = env.pc.bone_ends()
            traj.append((S.copy(), E.copy(), R.copy(), env.t))
            o, r, d, info = env.step(act(o)); L += 1
            if d or info["trunc"]: break
        lens.append(L); falls += int(d)
        if best is None or L > best[0]: best = (L, traj, env.clip, env.t0)
    lens = np.array(lens)
    print(f"episodes {a.episodes}: mean length {lens.mean() / 30:.2f} s, fell {falls}/{a.episodes}, full-length {np.mean(lens >= a.seconds * 30 - 2):.2f}")
    if a.gif:
        from render import Scene, Actor, caption, save_gif
        from terrain import Terrain
        L, traj, clip, t0 = best
        sc = Scene(Terrain(-60, 60, -60, 60), [Actor(c, "p", base=(0.30, 0.45, 0.75)), Actor(c, "g", base=(0.8, 0.8, 0.8))], 480, 300)
        sk = c.skel; frames = []
        for k, (S, E, R, tt) in enumerate(traj):
            sc.pose(sc.actors[0], S, E, R)
            kk = min(clip["q"].shape[0] - 1, k + t0)
            Sg, Eg, Rg = sk.fk(np.array([S[0][0], S[0][1] + 1.0, clip["root"][kk][2]]), clip["R"][kk], clip["q"][kk])
            sc.pose(sc.actors[1], Sg, Eg, Rg)
            img = sc.render([S[0][0], S[0][1] + 0.5, S[0][2] * 0.9], 4.5, 90, -6)
            frames.append(caption(img, "physical (blue/orange) vs planner reference (grey, shifted)", f"t={k / 30:.1f}s"))
        sc.close(); save_gif(frames, a.gif, fps=30)

if __name__ == "__main__":
    main()
