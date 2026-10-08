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
for _ in range(60 * 45):
    cl.step(1 / 60)
    if cl.done: break
assert cl.done and cl.pelvis[2] > 4.0, (cl.done, cl.pelvis)
print("climb ok: %d moves, pelvis at %.1f m" % (len(cl.log), cl.pelvis[2]))
