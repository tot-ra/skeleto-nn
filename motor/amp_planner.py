"""Reference motion for the motion prior from the procedural planner (``planner.Walker``): the planner's gaits, which
look natural by construction (Froude-scaled cadence and duty factor, planted feet, swinging arms and tail), become
the "recordings" the AMP discriminator compares a policy with. Training only: at run time there is no planner, the
network drives the muscles in the physics; it learns to arrive at motion like the planner's by its own means.

    python -m motor.amp_planner quadruped data/mocap/planner_quad_amp.npz [--seconds 120]

Every species of the family walks, trots and gallops on flat ground under random commands (speed in Froude units,
heading changes, stops) for ``--seconds`` each; features are those of the family's ``amp_features`` (``motor.amp``),
taken at the control rate (30 Hz), with mirrored copies.
"""
from __future__ import annotations
import argparse, math
import numpy as np

from motor import amp
from planner import Walker, Cmd
from terrain import Terrain

HZ = amp.HZ
SUB = 2                                                  # planner steps per control step (60 Hz)


def clip(c, points, segs, seconds, seed, max_froude):
    rng = np.random.default_rng(seed)
    W = Walker(c, Terrain(-400, 400, -400, 400, res=0.5), dt=1.0 / (HZ * SUB), seed=seed)
    leg = c.leg_len(); unit = math.sqrt(amp.G * leg)
    F = []; prev = None; heading = 0.0; v = 0.0; t_next = 0.0
    for k in range(int(seconds * HZ)):
        t = k / HZ
        if t >= t_next:                                  # a new command every 1.5 to 4 s
            v = 0.0 if rng.random() < 0.12 else rng.uniform(0.1, max_froude) * unit
            heading += rng.normal(0.0, 0.6); t_next = t + rng.uniform(1.5, 4.0)
        for _ in range(SUB):
            W.step(Cmd(v=np.array([math.cos(heading), math.sin(heading)]) * v))
        S, E = W.S, W.E
        root = S[0].copy(); yaw = W.heading
        if prev is not None and t > 1.0:                 # skip the first second (starting from rest)
            vel = (root - prev[0]) * HZ; yr = math.atan2(math.sin(yaw - prev[1]), math.cos(yaw - prev[1])) * HZ
            F.append(amp.features(points(c.skel, S, E), vel, yr, leg, segs))
        prev = (root, yaw)
    return np.array(F, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["quadruped"]); ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=120.0)
    a = ap.parse_args()
    from bodies import quadruped, QUADS
    from motor.quadruped import canonical, SPECIES, QuadrupedEnv
    pairs = []
    for n, sp in enumerate(SPECIES):
        c = quadruped(canonical(QUADS[sp]))
        F = clip(c, amp.quad_points, amp.QUAD_SEGS, a.seconds, n, QuadrupedEnv.MAX_SPEED)
        for G_ in (F, amp.mirror(F, amp.QUAD_SEGS)):
            pairs.append(np.concatenate([G_[:-1], G_[1:]], 1))
        print(sp, len(F), "frames", flush=True)
    X = np.concatenate(pairs).astype(np.float32)
    np.savez_compressed(a.out, X=X)
    print("pairs", X.shape)


if __name__ == "__main__":
    main()
