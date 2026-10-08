"""The legs limit a jump: a standing jump of 3.5 m is refused, a hop over a low wall is not, a run-up makes the long jump possible."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from bodies import human
from planner import Walker, Cmd
from terrain import Terrain
c = human(); tr = Terrain(); tr.add_pit(2.9, 4.1, -3, 3, 1.2)
w = Walker(c, tr, pos=(1.2, 0))
for _ in range(60): w.step(Cmd())
assert not w.start_jump((4.9, 0.0)), "standing 3.7 m jump must be refused"
w = Walker(c, tr, pos=(-2, 0))
for _ in range(400):
    w.step(Cmd(v=np.array([4.0, 0])))
    if w.pos[0] > 2.0: break
assert w.start_jump((5.0, 0.0)), "the same gap from a 4 m/s run-up must work"
tr2 = Terrain(); tr2.add_box(2.7, 3.1, -3, 3, 0.6); w = Walker(c, tr2, pos=(2.0, 0))
for _ in range(60): w.step(Cmd())
assert w.start_jump((3.8, 0.0))
print("jump limits ok")
