"""What the leg muscles of a trained walker do: activation of every left-leg muscle over time (like EMG), with the
foot contacts, and the walk drawn with its muscles.

    BIPED_MUSCLES=1 python -m motor.emg runs/motor/biped_muscle/ckpt.pt [out_dir] [--assist 0.5] [--speed 1.3]

Writes ``muscle_walk.gif`` (muscles red while they contract) and ``muscle_emg.png``.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
from PIL import Image, ImageDraw

from motor.ppo import make_env, mlp_np
from motor.show import load, render_clip
from motor import evaluate as ev
from muscle_atlas import COLORS, font


def record(env, pi, ck, speed, assist, T):
    mean, std = ck["mean"], np.sqrt(ck["var"] + 1e-6)
    body = env.default_body(); env._build(body)
    o = env.reset(body=body, friction=1.0, wind=np.zeros(3), sores={k: 0.0 for k in env.SORES},
                  profile=dict(ev.DEFAULT_WEIGHTS), goal=np.array([speed, 0.0]))
    env.assist = assist
    frames, acts, contacts, speeds = [], [], [], []
    for _t in range(T):
        a = mlp_np(np.clip((o - mean) / std, -10, 10).astype(np.float32), pi)
        o, _r, d, info = env.step(a)
        frames.append(env.pc.bone_ends() + (env.pc.muscle_paths(), env.pc.m_act.copy()))
        acts.append(env.pc.m_act.copy()); contacts.append(env.pc.foot_contacts()); speeds.append(info["speed"])
        if d:
            break
    return frames, np.array(acts), np.array(contacts), float(np.mean(speeds[30:] or speeds))


def chart(env, acts, contacts, path, title, hz=30, W=900):
    names = [n for n in env.pc.mnames if n.endswith("_L")]
    rows = len(names) + 1; rh = 34; top = 40; left = 150
    H = top + rows * rh + 30
    im = Image.new("RGB", (W, H), (252, 251, 247)); dr = ImageDraw.Draw(im); f = font(13)
    n = len(acts); xs = lambda t: left + (W - left - 10) * t / max(n - 1, 1)
    dr.text((8, 8), title, fill=(30, 30, 30), font=font(15))
    for r, nm in enumerate(names):
        k = env.pc.mnames.index(nm); y0 = top + r * rh; col = COLORS[nm[:-2]]
        dr.text((8, y0 + 9), env.pc.ms.muscles[k].title, fill=col, font=f)
        dr.line([(left, y0 + rh - 4), (W - 10, y0 + rh - 4)], fill=(220, 220, 220))
        pts = [(xs(t), y0 + rh - 4 - (rh - 8) * float(acts[t, k])) for t in range(n)]
        dr.polygon(pts + [(xs(n - 1), y0 + rh - 4), (xs(0), y0 + rh - 4)], fill=tuple(int(0.35 * c + 0.65 * 252) for c in col))
        dr.line(pts, fill=col, width=2)
    y0 = top + len(names) * rh                                   # foot contacts: left on top, right below
    dr.text((8, y0 + 9), "foot on ground L / R", fill=(60, 60, 60), font=f)
    for side, dy in ((0, 6), (1, 18)):
        for t in range(n):
            if contacts[t, side] > 0:
                dr.line([(xs(t), y0 + dy), (xs(t), y0 + dy + 8)], fill=(90, 90, 90) if side == 0 else (170, 170, 170))
    for s in range(int(n / hz) + 1):
        x = xs(s * hz); dr.line([(x, top), (x, H - 26)], fill=(235, 235, 235)); dr.text((x - 4, H - 22), f"{s}s", fill=(90, 90, 90), font=f)
    im.save(path)


def main():
    argv = sys.argv[1:]
    opt = {"--assist": 0.0, "--speed": 1.3}
    for k in list(opt):
        if k in argv:
            j = argv.index(k); opt[k] = float(argv[j + 1]); del argv[j:j + 2]
    path = argv[0]; out = argv[1] if len(argv) > 1 else os.path.join(os.path.dirname(path), "show")
    os.makedirs(out, exist_ok=True)
    ck, pi = load(path)
    env = make_env(ck["family"], seed=0)
    frames, acts, contacts, sp = record(env, pi, ck, opt["--speed"], opt["--assist"], 6 * env.CTRL_HZ)
    harness = f", harness {opt['--assist']:.0%}" if opt["--assist"] else ""
    title = f"walk {opt['--speed']} m/s asked, {sp:.2f} m/s{harness}: {len(acts) / env.CTRL_HZ:.1f} s"
    render_clip(env, frames, title, os.path.join(out, "muscle_walk.gif"), size=(480, 400))
    chart(env, acts, contacts, os.path.join(out, "muscle_emg.png"), "Left leg muscle activation, " + title)
    print("wrote", out, title)


if __name__ == "__main__":
    main()
