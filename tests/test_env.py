import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from bodies import *
from ref import make_clip
from env import Env
c = human()
t = time.time()
clips = [make_clip(c, s, T=10.0) for s in range(4)]
print("4 clips %.1fs" % (time.time() - t), [cl["q"].shape for cl in clips][:1])
e = Env(c, clips, seed=0)
print("obs", e.obs_dim, "act", e.act_dim)
lens = []
t = time.time(); n = 0
for ep in range(12):
    o = e.reset(); L = 0
    while True:
        o, r, d, info = e.step(np.zeros(e.act_dim)); L += 1; n += 1
        if d or info["trunc"]: break
    lens.append(L)
print("zero-action survival (steps at 30 Hz):", lens, "steps/s %.0f" % (n / (time.time() - t)))
