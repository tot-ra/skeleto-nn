"""A defender that notices in time avoids the blow by predicting the geometry; one that notices too late is hit."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("MUJOCO_GL", "glfw")
import catalog, catalog_combat, shots
sh = catalog.SHOTS["dodge_reactions"](); sh.T = 3.0
walkers, snaps, ctx = shots.simulate_shot(sh)
log = ctx.state["arena"].log
outcomes = [(r["outcome"], r["react"]) for r in log]
assert any(o == "miss" for o, _ in outcomes), outcomes
assert any(o == "hit" and rc == "none" for o, rc in outcomes), outcomes      # the one that noticed too late
print("combat ok", outcomes)
