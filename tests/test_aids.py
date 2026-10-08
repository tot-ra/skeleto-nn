"""A peg leg, crutches with a held-up leg, and a three-legged dog all walk without falling over and move forward."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from bodies import human, HumanBody, quadruped
from planner import Walker, Cmd
from terrain import Terrain
import aids
def walk(w, v, n=600):
    for _ in range(n): w.step(Cmd(v=np.array([v, 0.0]), heading=0.0))
    return w
w = walk(Walker(human(HumanBody(peg="L")), Terrain()), 1.2); assert w.pos[0] > 6 and w.z > 0.6
w = Walker(human(), Terrain()); cr = aids.use_crutches(w, "leg_L"); walk(w, 0.9); assert w.pos[0] > 5 and w.z > 0.6 and cr.phase in ("planted", "moving")
w = walk(Walker(quadruped("dog", missing="HL"), Terrain()), 1.0); assert w.pos[0] > 5 and w.z > 0.3
w = Walker(human(), Terrain()); w.chronic = {"leg_L": 0.8}; walk(w, 1.3)
assert w.leg_duty["L"] < 0.8 and w.leg_duty["R"] > 0.99
print("aids ok")
