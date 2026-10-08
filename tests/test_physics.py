import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from bodies import *
from physics import *

def check(c):
    pc = PhysChar(c)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(5):
        qv = np.array([rng.uniform(0.3 * lo, 0.3 * hi) for lo, hi in zip(pc.lo, pc.hi)])
        q = pc.vec_to_q(qv)
        Rr = rot("Z", 0.4) @ rot("Y", 0.2)
        S, E, R = c.skel.fk(np.array([0.0, 0, 1.0]), Rr, q)
        pc.set_state(np.array([0.0, 0, 1.0]), Rr, qv)
        S2, E2, R2 = pc.bone_ends()
        worst = max(worst, abs(S - S2).max(), abs(E - E2).max())
    return worst
from skeleton import rot
for name, c in [("human", human()), ("dog", quadruped("dog")), ("crow", bird("crow"))]:
    print(name, "max fk mismatch %.2e" % check(c), "dofs", PhysChar(c).nj, "mass", round(PhysChar(c).total_mass, 1))
