"""Simulate a catalog shot with a human whose proportions equal the shipped character rig and export the world
rotation of every mapped bone per frame (JSON), for retarget_blender.py.
python export_clip.py SHOT out.json"""
import sys, json
import numpy as np
import bodies, catalog, shots
sys.modules["catalog"] = catalog
import catalog_extra, catalog_home, catalog_birds, catalog_combat, catalog_moves
for mod in (catalog, catalog_extra, catalog_home, catalog_birds, catalog_combat, catalog_moves):
    mod.human = lambda b=None, name="human", rig=None: bodies.human_game_rig()
name, out = sys.argv[1], sys.argv[2]
shot = catalog.build(name)
walkers, snaps, ctx = shots.simulate_shot(shot)
c = shot.actors[0].creature; sk = c.skel
want = ["pelvis", "lumbar", "chest", "head", "thigh_L", "thigh_R", "shank_L", "shank_R", "foot_L", "foot_R", "uarm_L", "uarm_R", "farm_L", "farm_R"]
idx = [sk.idx[n] for n in want]
frames = []
for t, row in snaps:
    r = row[0]
    frames.append(dict(t=float(t), root=[float(v) for v in r["root"]], R={n: [float(v) for v in r["R"][i].ravel()] for n, i in zip(want, idx)}))
terrain = []
for kind, a, rgba in shot.terrain.prims:
    if kind in ("box", "ramp", "pit", "beam"):
        terrain.append(dict(kind=kind, a=[float(v) for v in a], rgba=list(rgba)))
door = None
if "phi_now" in ctx.state or "hist" in ctx.state:
    h = ctx.state.get("hist", {})
    if h and name == "door_open":
        door = dict(hinge=[4.0, -0.5], width=0.95, height=2.0, thick=0.05, phi=[float(h.get(round(f["t"] * 60), 0.0)) for f in frames])
json.dump(dict(door=door, fps=shot.fps, z0=float(c.z0), frames=frames, terrain=terrain, title=shot.title, name=shot.name), open(out, "w"))
print("frames", len(frames), "->", out)
