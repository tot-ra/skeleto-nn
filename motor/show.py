"""Render a trained family policy in fixed review situations (deterministic: the mean action, no harness).

    python -m motor.show runs/motor/biped_s1/ckpt.pt [out_dir] [--assist 0.4]   (harness level, default 0)

Writes one GIF per situation plus ``summary.json`` with the episode length, whether it fell, the mean goal score
and the mean of every evaluation term.
"""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "glfw")
import json, sys
import numpy as np

from bodies import HumanBody
from motor.ppo import make_env, mlp_np
from motor import evaluate as ev

NO_SORES = {"leg_L": 0.0, "leg_R": 0.0, "arm_L": 0.0, "arm_R": 0.0, "torso": 0.0, "head": 0.0}
BIPED = [
    ("stand", dict(goal=(0.0, 0.0)), "standing still"),
    ("walk", dict(goal=(1.3, 0.0)), "walk 1.3 m/s"),
    ("run", dict(goal=(3.2, 0.0)), "run 3.2 m/s"),
    ("turn", dict(goal=(0.0, 1.2)), "walk to the left"),
    ("ice", dict(goal=(1.3, 0.0), friction=0.1), "walk on ice"),
    ("sprain", dict(goal=(1.3, 0.0), sores={**NO_SORES, "leg_L": 0.9}), "sprained left ankle"),
    ("gale", dict(goal=(1.3, 0.0), wind=(0.0, -14.0, 0.0)), "gale from the left"),
    ("pack", dict(goal=(1.3, 0.0), body=HumanBody(mass=95.0, pack=20.0)), "20 kg pack"),
    ("child", dict(goal=(1.0, 0.0), body=HumanBody(height=1.2, mass=26.0, muscle=0.9, strength=0.9, flex=1.2)), "child"),
    ("elder", dict(goal=(0.8, 0.0), body=HumanBody(mass=68.0, muscle=0.55, strength=0.55, flex=0.75)), "weak elder"),
]


def situations(family, env):
    """(name, reset arguments, title); quadruped goals are in Froude speed units of the body shown."""
    if family == "biped":
        return BIPED
    if family == "bird":
        from bodies import BIRDS
        L = dict(climb=0.0)                                  # level flight unless a climb is named
        return [("crow_glide", dict(goal=(3.0, 0.0), climb=-0.45), "crow, glide"), ("crow_cruise", dict(goal=(2.8, 0.0), **L), "crow, level flight"),
                ("crow_climb", dict(goal=(2.8, 0.0), climb=0.3), "crow, climb"), ("crow_turn", dict(goal=(2.8, 0.6), **L), "crow, turn"),
                ("crow_wind", dict(goal=(2.8, 0.0), wind=(0.0, -6.0, 0.0), **L), "crow, side wind"),
                ("gull", dict(body=BIRDS["gull"], goal=(2.8, 0.0), **L), "gull"), ("heron", dict(body=BIRDS["heron"], goal=(2.8, 0.0), **L), "heron")]
    from bodies import QUADS
    from motor.quadruped import canonical
    out = []
    for sp in ("dog", "cat", "horse", "pig"):
        body = canonical(QUADS[sp])
        out += [(f"{sp}_walk", dict(body=body, goal=(0.5, 0.0)), f"{sp}, walk"),
                (f"{sp}_trot", dict(body=body, goal=(1.0, 0.0)), f"{sp}, trot")]
    out += [("dog_ice", dict(body=canonical(QUADS["dog"]), goal=(0.6, 0.0), friction=0.1), "dog on ice"),
            ("dog_lame", dict(body=canonical(QUADS["dog"]), goal=(0.6, 0.0),
                              sores={"leg_HL": 0.0, "leg_HR": 0.0, "leg_FL": 0.9, "leg_FR": 0.0, "torso": 0.0, "head": 0.0}),
             "dog, sore front left leg")]
    return out


def load(path):
    import torch
    ck = torch.load(path, weights_only=False)
    lay = lambda sd, n: [(sd[f"{k}.weight"].numpy().T, sd[f"{k}.bias"].numpy()) for k in n]
    names = sorted({k.rsplit(".", 1)[0] for k in ck["pi"]}, key=lambda s: int(s))
    return ck, lay(ck["pi"], names)


def run(env, pi, ck, kw, T=None, assist=0.0):
    T = T or int(10 * env.CTRL_HZ)                           # 10 s
    mean, std = ck["mean"], np.sqrt(ck["var"] + 1e-6)
    body = kw.get("body", env.default_body())
    env._build(body)
    args = dict(body=body, friction=1.0, wind=np.zeros(3), sores={k: 0.0 for k in env.SORES}, profile=dict(ev.DEFAULT_WEIGHTS))
    args.update({k: (np.array(v, float) if k in ("goal", "wind") else v) for k, v in kw.items()})
    args["goal"] = args.get("goal", np.zeros(2)) * (env.speed_unit() if env.family == "quadruped" else 1.0)
    o = env.reset(**args)
    env.assist = assist
    frames, costs, goals, speeds, fell = [], [], [], [], False
    for t in range(T):
        a = mlp_np(np.clip((o - mean) / std, -10, 10).astype(np.float32), pi)
        o, _r, d, info = env.step(a)
        frames.append(env.pc.bone_ends() + ((env.pc.muscle_paths(), env.pc.m_act.copy()) if env.pc.nm else ()))
        costs.append(info["costs"]); goals.append(info["goal_now"]); speeds.append(info["speed"])
        if d:
            fell = True
            break
    n = len(costs)
    mean_cost = {k: float(np.mean([c[k] for c in costs])) for k in costs[0]}
    return frames, dict(steps=n, fell=fell, goal=float(np.mean(goals)), speed=float(np.mean(speeds[30:] or speeds)), costs=mean_cost)


def render_clip(env, frames, title, path, size=(480, 300), seconds=None):
    """One GIF of a run's frames; birds are filmed closer, from above, every frame, with the height in the caption."""
    from render import Scene, Actor, caption, save_gif
    if seconds is not None:
        frames = frames[:int(seconds * env.CTRL_HZ)]
    nm = env.pc.nm                                            # line muscles: the skeleton thin, the muscles drawn
    sc = Scene(None, [Actor(env.c, "a", base=(0.85, 0.6, 0.45), muscles=nm, skeleton=nm > 0)], size[0], size[1])
    imgs = []
    bird = env.family == "bird"
    fps = 30 if bird else 15                                  # a 4.5 Hz wingbeat needs 30 frames a second
    step = max(1, env.CTRL_HZ // fps)
    for k, fr in enumerate(frames[::step]):
        S, E, R = fr[:3]
        sc.pose(sc.actors[0], S, E, R)
        if nm:
            sc.pose_muscles(sc.actors[0], *fr[3:5])
        look = np.array([S[0][0], S[0][1], S[0][2] if bird else 0.9 * env.c.z0])   # follow a flier up
        if bird:      # closer, from above and behind (both wings show), the height in the caption (a camera that
            sub = f"t={step * k / env.CTRL_HZ:.1f}s  height {S[0][2]:.1f} m"     # follows a flier hides whether it climbs or sinks)
            imgs.append(caption(sc.render(look, max(1.5, 2.5 * env.wingspan()), 160, -30), title, sub))
        else:
            imgs.append(caption(sc.render(look, 4.0, 120, -12), title, f"t={step * k / env.CTRL_HZ:.1f}s"))
    sc.close()
    save_gif(imgs, path, fps=fps)


def main():
    argv = [a for a in sys.argv[1:]]
    assist = 0.0
    if "--assist" in argv:
        k = argv.index("--assist"); assist = float(argv[k + 1]); del argv[k:k + 2]
    path = argv[0]; out = argv[1] if len(argv) > 1 else os.path.join(os.path.dirname(path), "show")
    os.makedirs(out, exist_ok=True)
    ck, pi = load(path)
    env = make_env(ck["family"], seed=0)
    summary = {}
    for name, kw, title in situations(ck["family"], env):
        frames, res = run(env, pi, ck, kw, assist=assist)
        render_clip(env, frames, title, os.path.join(out, name + ".gif"))
        summary[name] = res
        print(name, "steps", res["steps"], "fell", res["fell"], "goal %.2f speed %.2f" % (res["goal"], res["speed"]), {k: round(v, 3) for k, v in res["costs"].items() if v})
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
