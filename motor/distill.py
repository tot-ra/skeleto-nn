"""Policy distillation: one family network learns what specialists for some of its bodies know.

    QUAD_ONLY=  python -m motor.distill --family quadruped --student runs/motor/quad_pbt/m1/ckpt.pt \\
        --teacher horse=runs/snap/horse_only_20min.pt --out runs/motor/quad_distill/ckpt.pt

A network shared by a whole family can fail one body that a network trained on that body alone masters (the shared
quadruped network kept the horse standing; a horse-only network walked it within 20 minutes). Distillation keeps one
network: the student is rolled out over the family's bodies (DAgger: at first partly driven by the teachers, then by
itself), and every visited state is labelled with the action of that body's teacher: the specialist for the bodies it
names, a frozen copy of the student itself for all the others (so nothing it could do is forgotten). The student is
trained on all labels gathered so far (mean actions, squared error). Its value function, exploration noise,
observation normalisation and curriculum state are kept; continue with ``motor.ppo --init <out> --vwarm 50``.
"""
from __future__ import annotations
import argparse, copy, os
import numpy as np
import torch
import torch.nn as nn

from motor.ppo import make_env, mlp_np
from motor.show import load
from motor import evaluate as ev


def policy_net(sd):
    """The torch policy of a checkpoint (Linear-Tanh-Linear-Tanh-Linear, as motor.ppo builds it)."""
    i, h = sd["0.weight"].shape[1], sd["0.weight"].shape[0]; o = sd["4.weight"].shape[0]
    net = nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h // 2), nn.Tanh(), nn.Linear(h // 2, o))
    net.load_state_dict(sd); return net


def body_of(family, name, rng):
    """A body of the named kind, varied a little as the family's sampler varies it."""
    import dataclasses, math
    if family == "quadruped":
        from bodies import QUADS
        from motor.quadruped import canonical
        sp = QUADS[name]; j = lambda: math.exp(rng.normal(0, 0.08))
        return canonical(dataclasses.replace(sp, mass=sp.mass * j() ** 3, scale=sp.scale * j()))
    if family == "bird":
        from bodies import BIRDS
        return BIRDS[name]
    raise ValueError(family)


class Teacher:
    def __init__(self, path):
        self.ck, self.layers = load(path)
        self.mean, self.std = self.ck["mean"], np.sqrt(self.ck["var"] + 1e-6)

    def act(self, raw_obs):
        return mlp_np(np.clip((raw_obs - self.mean) / self.std, -10, 10).astype(np.float32), self.layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="quadruped"); ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", action="append", default=[], help="body=checkpoint (body: the skeleton name, e.g. horse)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=6); ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--share", type=float, default=0.4, help="share of episodes on the teachers' bodies")
    args = ap.parse_args()
    ck = torch.load(args.student, weights_only=False)
    student = policy_net(ck["pi"]); opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    s_mean, s_std = ck["mean"], np.sqrt(ck["var"] + 1e-6)
    self_teacher = Teacher(args.student)                       # the student as it was: the teacher of all other bodies
    teachers = {k: Teacher(v) for k, v in (t.split("=", 1) for t in args.teacher)}
    env = make_env(args.family, seed=args.seed)
    env.spread = float(ck.get("spread", 1.0)); env.max_speed = float(ck.get("max_speed", 1.0)); env.assist = 0.0
    rng = np.random.default_rng(args.seed)

    def reset():
        if teachers and rng.random() < args.share:            # a teacher's body (the family's sampler draws it rarely)
            return env.reset(body=body_of(args.family, str(rng.choice(sorted(teachers))), rng))
        return env.reset()
    O, Y = [], []
    for k in range(args.iters):
        beta = max(0.0, 0.5 * (1 - k / max(1, args.iters - 1)))  # how much the teacher drives the rollout
        o = reset(); n = 0; per = {}
        while n < args.steps:
            body = env.c.skel.name
            t = teachers.get(body, self_teacher)
            y = t.act(o)
            with torch.no_grad():
                a_s = student(torch.tensor(np.clip((o - s_mean) / s_std, -10, 10), dtype=torch.float32)).numpy()
            a = y if rng.random() < beta else a_s
            O.append(o.copy()); Y.append(y); per[body] = per.get(body, 0) + 1
            o, _r, d, info = env.step(a + 0.05 * rng.standard_normal(a.shape))
            n += 1
            if d or info["trunc"]:
                o = reset()
        X = torch.tensor(np.clip((np.array(O) - s_mean) / s_std, -10, 10), dtype=torch.float32)
        T = torch.tensor(np.array(Y), dtype=torch.float32)
        for _ in range(args.epochs):
            perm = torch.randperm(len(X))
            for s in range(0, len(X), 4096):
                idx = perm[s:s + 4096]
                loss = ((student(X[idx]) - T[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        print(f"iter {k}: beta {beta:.2f}, {len(X)} labels, loss {float(loss):.4f}, bodies {per}", flush=True)
    out = copy.copy(ck); out["pi"] = student.state_dict()
    os.makedirs(os.path.dirname(args.out), exist_ok=True); torch.save(out, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
