"""GIFs of the physically simulated swimmer, bird and fall: python phys_gifs.py out_dir   (uses runs/swim_*.json, runs/bird_*.json, results/*.json)"""
import os, sys, json, math
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
from render import Scene, Actor, caption, save_gif
from terrain import Terrain
import swimmer as SW, flier as FL

out = sys.argv[1] if len(sys.argv) > 1 else "runs/phys"; os.makedirs(out, exist_ok=True)
def load(f): return np.array(json.load(open(f))["params"])
WATER = lambda x, W: dict(geom=f'<geom type="box" size="14 5 0.02" pos="{x} 0 {W}" rgba="0.15 0.45 0.80 0.35"/>')
SPHERE = lambda r, rgba: dict(geom=f'<geom type="sphere" size="{r}" rgba="{rgba}"/>')

def render_swim(name, params_file, mode, goal, T, title):
    c, pc, sw = SW.make_pool(); p = load(params_file)
    r = SW.rollout(pc, sw, p, mode, goal, T=T, record=True)
    tr = Terrain(-3, 40, -6, 6)
    props = [WATER(10, SW.W_SURF), SPHERE(0.25, "0.95 0.8 0.1 1")]
    sc = Scene(tr, [Actor(c, "a", base=(0.85, 0.6, 0.45))], 480, 300, props=props); frames = []
    for k, (S, E, R, o2, mouth_out) in enumerate(r["traj"]):
        sc.pose(sc.actors[0], S, E, R); sc.set_prop(0, [S[0][0] + 4, 0, SW.W_SURF]); gp = goal if goal is not None else np.array([30.0, 0, SW.W_SURF])
        sc.set_prop(1, gp)
        look = np.array([S[0][0] + (1.5 if mode == "surface" else 1.0), 0, S[0][2] if mode == "under" else SW.W_SURF - 0.5])
        img = sc.render(look, 4.5 if mode == "surface" else 6.0, 20, -4)
        frames.append(caption(img, title, f"t={k / 30:.1f}s   oxygen {o2 * 100:.0f}%   {'breathing' if mouth_out else 'holding breath'}   speed {np.linalg.norm(S[0] - r['traj'][max(0, k - 5)][0][0]) * 6:.1f} m/s"))
    sc.close(); save_gif(frames, os.path.join(out, name + ".gif"), fps=30)
    print(name, "cost %.2f progress %.1f reached %s o2 %.2f" % (r["cost"], r["progress"], r["reached"], r["o2"]))

def render_bird(name, params_file, mode, T, title, target_x=12.0):
    c, pc, ar = FL.make_world(); p = load(params_file)
    r = FL.rollout(pc, ar, p, mode, T=T, target_x=target_x, record=True)
    tr = Terrain(-3, 120, -8, 8); sc = Scene(tr, [Actor(c, "a", base=(0.2, 0.2, 0.25))], 480, 300, props=[SPHERE(0.3, "0.95 0.8 0.1 1")]); frames = []
    for k, (S, E, R) in enumerate(r["traj"]):
        sc.pose(sc.actors[0], S, E, R); sc.set_prop(0, [target_x, 0, 0.3] if mode == "land" else [0, 0, -5])
        img = sc.render(np.array([S[0][0] + 0.5, 0, max(S[0][2], 0.8)]), 3.2, 80, -6); frames.append(caption(img, title, f"t={k / 30:.1f}s   height {S[0][2]:.1f} m"))
    sc.close(); save_gif(frames, os.path.join(out, name + ".gif"), fps=30)
    print(name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k not in ("traj", "touch")}, r.get("touch"))

if __name__ == "__main__":
    which = sys.argv[2:] or ["swim", "bird"]
    if "swim" in which:
        render_swim("swim_surface", "runs/swim_surface.json", "surface", None, 14.0, "swimming at the surface, learned: air or no air decides the stroke")
        render_swim("swim_under_bottom", "runs/swim_under.json", "under", np.array([4.0, 0, 0.3]), 9.0, "under water to a goal on the bottom (hands and feet push on the water)")
        render_swim("swim_under_far", "runs/swim_under.json", "under", np.array([7.0, 0, 3.0]), 9.0, "under water to a far goal at depth")
    if "bird" in which:
        render_bird("bird_fly", "runs/bird_fly.json", "fly", 14.0, "flight with a limited energy store: flap, then glide")
        render_bird("bird_land", "runs/bird_land.json", "land", 8.0, "landing on a chosen spot with the least injury", 12.0)
        render_bird("bird_takeoff", "runs/bird_takeoff.json", "takeoff", 6.0, "take-off from the ground")
