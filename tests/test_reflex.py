import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from reflex import *
c = human(); pc = PhysChar(c); kp = np.array([pc.model.actuator_gainprm[a, 0] for a in pc.aid])
rng = np.random.default_rng(5)
sc = sample_scenarios(rng, 8)
t = time.time()
for s in sc:
    a = run_scenario(c, default_params(), s, pc=pc, base_kp=kp); b = run_scenario(c, default_params(), s, pc=pc, base_kp=kp, passive=True)
    print(s["kind"], "h=%.1f" % s.get("h", 0), "default cost %.2f  passive cost %.2f  head peak %.1f / %.1f mg" % (a["cost"], b["cost"], a["peak"]["head"], b["peak"]["head"]))
print("per scenario %.2fs" % ((time.time() - t) / 16))
