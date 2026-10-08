"""Pain makes an arm unusable, a knee blow becomes leg pain, the same groin blow hurts a woman less, a hard head blow knocks the body down."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from bodies import human, HumanBody
from planner import Walker, Cmd
from terrain import Terrain
def stand(**kw):
    w = Walker(human(HumanBody(**kw)), Terrain())
    for _ in range(60): w.step(Cmd())
    return w
w = stand(); assert w.arm_usable("R"); w.apply_hit("arm_R", np.array([1.0, 0.0]), 1.0); assert not w.arm_usable("R")
w = stand(); w.apply_hit("knee_L", np.array([1.0, 0.0]), 0.8); assert w.pain["leg_L"] > 0.5
m, f = stand(sex="m"), stand(sex="f"); m.apply_hit("groin", np.array([1.0, 0.0]), 0.6); f.apply_hit("groin", np.array([1.0, 0.0]), 0.6)
assert m.pain["groin"] > 2.5 * f.pain["groin"], (m.pain["groin"], f.pain["groin"])
w = stand(); w.apply_hit("head", np.array([-1.0, 0.0]), 2.4); assert w.down is not None
for _ in range(2400):
    w.step(Cmd())
    if w.down is None: break
assert w.down is None, "the body must get up again"
print("hit ok")
