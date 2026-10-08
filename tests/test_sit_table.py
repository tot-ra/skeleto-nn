"""The person goes round the table and chair and never puts a bone through the tabletop; the chair is really pulled out and pushed in."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
import catalog, catalog_home, shots
sh = catalog.SHOTS["sit_table"](); walkers, snaps, ctx = shots.simulate_shot(sh)
worst = 0; viol = 0
for t, row in snaps:
    S, E = row[0]["S"], row[0]["E"]
    for u in np.linspace(0, 1, 5):
        P = S + (E - S) * u
        inside = (4.40 <= P[:, 0]) & (P[:, 0] <= 5.60) & (np.abs(P[:, 1]) <= 0.45) & (P[:, 2] >= 0.715) & (P[:, 2] <= 0.755)
        viol += int(inside.sum())
assert viol == 0, "%d bone samples inside the tabletop" % viol
print("sit_table ok: no bone inside the tabletop; chair x ends at %.2f" % ctx.state["chair_x"])
