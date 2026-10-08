"""Muscle, energy and flexibility change what a body can do; a self-model learned by practice finds the real jump range from any prior; fear stops a jump that cannot be made."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
from bodies import persona
from planner import Walker, Cmd
from terrain import Terrain
import jumper as J
def walker(name):
    w = Walker(persona(name), Terrain(-3, 300, -6, 6))
    for _ in range(60): w.step(Cmd())
    return w
rng = {n: walker(n).jump_range(np.array([1.0, 0.0]), 0.0) for n in ("elder", "adult", "athlete")}
assert rng["elder"] < rng["adult"] < rng["athlete"], rng
# energy: the same relative effort for 30 s drains a small reserve much further
st = {}
for n in ("starved", "adult"):
    w = walker(n); vs = w.c.params["v_sprint"]
    for _ in range(1800): w.step(Cmd(v=np.array([0.8 * vs, 0.0]), heading=0.0))
    st[n] = w.stamina
assert st["starved"] < st["adult"] - 0.2, st
# the self-model converges from an over-confident and from a fearful prior
w = walker("adult")
for prior in (1.5, 0.6):
    m = J.SelfModel(w, prior, 0.25, np.random.default_rng(1)); m.practise(12); mean, sd = m.mean_sd()
    assert abs(mean - m.C_true) < 0.15 * m.C_true and sd < 0.1 * m.C_true, (prior, mean, sd, m.C_true)
# a jump the body cannot make is not attempted when the belief is right
m = J.SelfModel(w, 1.0, 0.3, np.random.default_rng(1)); m.practise(12)
assert m.plan(m.C_true * 1.4, 1.5)[0] is None and m.plan(1.0, 1.5)[0] is not None
# pregnancy: a belly with a collision geom and slower steps
c = persona("pregnant"); assert c.params["pregnancy"] == 1.0 and c.params["v_max"] < 2.5
print("personas ok: standing jump range elder %.1f adult %.1f athlete %.1f m; reserve after 30 s of the same relative effort: starved %.2f adult %.2f" % (rng["elder"], rng["adult"], rng["athlete"], st["starved"], st["adult"]))
