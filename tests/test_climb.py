"""The climber reaches the top of a random wall using only holds of the right kind."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from bodies import human
import climb as CL
rng = np.random.default_rng(2); holds = []
for _ in range(4000):
    p = np.array([3.0, rng.uniform(-0.8, 0.8), rng.uniform(0.25, 5.4)])
    if all(np.linalg.norm(p - q.p) > 0.34 for q in holds): holds.append(CL.Hold(p, str(rng.choice(["hand", "foot", "both", "both", "both"]))))
    if len(holds) > 110: break
c = human(); cl = CL.Climber(c, holds, (-1.0, 0.0, 0.0), (2.58, 0.0), 5.0)
sk = c.skel; i = sk.idx; closest = 9.0
for _ in range(60 * 45):
    S, E, R = cl.step(1 / 60)
    for a in ("thigh", "shank"):
        for b in ("thigh", "shank"):
            closest = min(closest, CL.seg_seg_dist(S[i[a + "_L"]], E[i[a + "_L"]], S[i[b + "_R"]], E[i[b + "_R"]]))
    if cl.done: break
assert cl.done and cl.pelvis[2] > 4.0, (cl.done, cl.pelvis)
assert closest > 0.06, "the legs pass through each other: %.3f m" % closest
print("legs stayed apart: %.2f m" % closest)
print("climb ok: %d moves, pelvis at %.1f m" % (len(cl.log), cl.pelvis[2]))

# the tree: branch stubs close to the trunk, alternating sides
rng = np.random.default_rng(5); holds = []; z = 0.55; side = 1.0
while z < 6.2:
    holds.append(CL.Hold(np.array([2.86, side * rng.uniform(0.12, 0.26), z]), "both")); z += rng.uniform(0.34, 0.50); side = -side
    if rng.random() < 0.7: holds.append(CL.Hold(np.array([2.86, -side * rng.uniform(0.12, 0.28), z - 0.17]), "foot"))
cl = CL.Climber(c, holds, (-1.0, 0.0, 0.0), (2.58, 0.0), 6.0); closest = 9.0
for _ in range(60 * 60):
    S, E, R = cl.step(1 / 60)
    for a in ("thigh", "shank"):
        for b in ("thigh", "shank"):
            closest = min(closest, CL.seg_seg_dist(S[i[a + "_L"]], E[i[a + "_L"]], S[i[b + "_R"]], E[i[b + "_R"]]))
    if cl.done: break
assert closest > 0.06 and cl.pelvis[2] > 4.5, (closest, cl.pelvis, cl.done)
print("tree: %d moves, legs apart %.2f m, pelvis at %.1f m" % (len(cl.log), closest, cl.pelvis[2]))
