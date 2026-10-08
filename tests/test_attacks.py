"""Limb and animal attacks land on the victim's pain model and leave every body finite."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np, catalog, catalog_attacks, shots
want = {"kick_human": ("leg_L", 0.3), "wolf_bite": ("arm_R", 0.3), "bear_swipe": ("head", 0.5), "horse_kick": ("torso", 0.5)}
for n, (region, lo) in want.items():
    W, snaps, ctx = shots.simulate_shot(catalog.SHOTS[n]())
    assert all(np.isfinite(r["S"]).all() for t, row in snaps for r in row), n
    assert max(h[2] for h in W[0].hit_log if h[1].startswith(region[:3])) > 0 and W[0].hit_log, n
    print(n, "hit", [(h[1], round(h[2], 2)) for h in W[0].hit_log], "down" if W[0].down is not None else "up")
W, snaps, ctx = shots.simulate_shot(catalog.SHOTS["weapons_human"]())
assert all(np.isfinite(r["S"]).all() for t, row in snaps for r in row)
print("attacks ok")
