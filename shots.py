"""A Shot = terrain + one or more creatures with command functions + props + camera. `run_shot` simulates all
creatures in lock step (so they can react to each other) and renders a GIF or a contact sheet."""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "glfw")
from dataclasses import dataclass, field
import numpy as np
from PIL import Image
from planner import Walker, Cmd
from terrain import Terrain
from render import Scene, Actor, caption, save_gif, grid

@dataclass
class ActorSpec:
    creature: object
    pos: tuple = (0.0, 0.0)
    heading: float = 0.0
    cmd: object = None                  # fn(t, walker, ctx) -> Cmd
    events: list = field(default_factory=list)     # [(t, fn(walker, ctx))]
    label: str = ""
    base: tuple | None = None           # body tint
    walker_kw: dict = field(default_factory=dict)
    kinematic: object = None            # fn(t, ctx) -> (S, E, R) full override (snake, bird flight ...)

@dataclass
class Shot:
    name: str
    title: str
    terrain: Terrain
    actors: list
    T: float
    cam: dict = field(default_factory=dict)       # dist, azimuth, elevation, look_z, follow (index | "all")
    props: list = field(default_factory=list)     # [{"geom": xml}] static or moving
    prop_fn: object = None                        # fn(t, ctx, scene) sets prop poses
    notes: str = ""
    size: tuple = (400, 260)
    fps: int = 20
    caption_fn: object = None                     # fn(t, ctx) -> extra text shown in the header

class Ctx:
    def __init__(self, shot, walkers):
        self.shot = shot; self.walkers = walkers; self.t = 0.0; self.state = {}
        self.terrain = shot.terrain

def simulate_shot(shot: Shot, dt=1 / 60):
    walkers = []
    for a in shot.actors:
        if a.kinematic is None:
            walkers.append(Walker(a.creature, shot.terrain, pos=a.pos, heading=a.heading, dt=dt, **a.walker_kw))
        else:
            walkers.append(None)
    if hasattr(shot, 'post_init'): shot.post_init([w for w in walkers])
    ctx = Ctx(shot, walkers)
    n = int(shot.T / dt); every = max(1, int(round(1 / (shot.fps * dt))))
    events = [sorted(a.events, key=lambda e: e[0]) for a in shot.actors]; ei = [0] * len(shot.actors)
    snaps = []
    for k in range(n):
        t = k * dt; ctx.t = t
        row = []
        for i, a in enumerate(shot.actors):
            w = walkers[i]
            while ei[i] < len(events[i]) and events[i][ei[i]][0] <= t:
                events[i][ei[i]][1](w, ctx); ei[i] += 1
            if a.kinematic is not None:
                S, E, R = a.kinematic(t, ctx)
                row.append(dict(S=S, E=E, R=R, root=S[0].copy(), t=t, speed=0.0, q=None)); continue
            cmd = a.cmd(t, w, ctx) if a.cmd else Cmd()
            row.append(w.step(cmd))
        if k % every == 0:
            snaps.append((t, row))
    return walkers, snaps, ctx

def render_shot(shot: Shot, snaps, ctx, label=True, hud=True):
    props = shot.props
    sc = Scene(shot.terrain, [Actor(a.creature, f"a{i}", base=a.base) for i, a in enumerate(shot.actors)], shot.size[0], shot.size[1], props=props)
    cam = shot.cam; follow = cam.get("follow", 0)
    out = []
    for (t, row) in snaps:
        for i, a in enumerate(shot.actors):
            r = row[i]
            sc.pose(sc.actors[i], r["S"], r["E"], r["R"])
        if shot.prop_fn: shot.prop_fn(t, ctx, sc, row)
        if follow == "all":
            ps = np.array([r["root"] for r in row]); look = ps.mean(0)
        else:
            look = row[follow]["root"].copy()
        lz = cam.get("look_z"); look = np.array([look[0], look[1], lz if lz is not None else max(look[2] * 0.9, 0.3)])
        if "look_off" in cam: look = look + np.array(cam["look_off"])
        az = cam.get("azimuth", 90.0)
        if callable(az): az = az(t)
        img = sc.render(look, cam.get("dist", 5.0), az, cam.get("elevation", -8.0))
        if label:
            sub = shot.title if hud else None
            extra = shot.caption_fn(t, ctx) if shot.caption_fn else ""
            out.append(caption(img, shot.name.replace("_", " ") + (("   |   " + extra) if extra else ""), f"{shot.title}   t={t:.1f}s"))
        else:
            out.append(img)
    sc.close()
    return out

def run_shot(shot, gif=None, sheet=None, n_sheet=8, t_range=None):
    walkers, snaps, ctx = simulate_shot(shot)
    if t_range:
        snaps = [s for s in snaps if t_range[0] <= s[0] <= t_range[1]]
    frames = render_shot(shot, snaps, ctx)
    if gif: save_gif(frames, gif, fps=shot.fps)
    if sheet:
        idx = np.linspace(0, len(frames) - 1, n_sheet).astype(int)
        Image.fromarray(grid([frames[i] for i in idx], 4)).save(sheet)
    return walkers, snaps, frames
