import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from bodies import *
from physics import *
from skeleton import rot
c = human(); pc = PhysChar(c)
for kp in (1.0, 2.0, 4.0):
    pc = PhysChar(c, kp_scale=kp)
    pc.set_state(np.array([0, 0, c.z0 + 0.003]), np.eye(3), np.zeros(pc.nj))
    for k in range(90):
        pc.step(np.zeros(pc.nj))
    p, R = pc.root_pose()
    print("kp x%.0f: after 3 s root z %.3f (start %.3f) tilt %.1f deg, bad contact %s" % (kp, p[2], c.z0, np.degrees(np.arccos(R[2, 2])), pc.bad_contact()))
