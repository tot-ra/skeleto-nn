"""Fast moves in clutter: running into a wall (with and without the arms), running through a forest and through a crowd."""
from __future__ import annotations
import math
import numpy as np
from bodies import *
from planner import Cmd, smooth
from shots import *
from nav import Navigator
from catalog import shot, goto
import catalog_birds as CBd
import injury as INJ

G = 9.81

# ---------------------------------------------------------------------------- running into a wall
def _wall_runner(i, v0, perceive, wall_x, state):
    """Run at the wall; once the time to collision is below `perceive` put the hands on the wall and sink; the stopping stroke is what
    the arms and knees give (a braced runner stops over ~0.45 m, a blind one over the thickness of the chest)."""
    def cmd(t, w, ctx):
        st = state.setdefault(i, dict(phase="run", t_c=None))
        gap = wall_x - w.pos[0]
        arm_L = 0.62                                              # shoulder to hand
        shoulder_h = float(w.S[w.sk.idx["uarm_L"]][2]) if w.S is not None else 1.4
        if st["phase"] == "run":
            ttc = (gap - 0.25) / max(v0, 1e-3)
            if perceive > 0 and ttc < perceive: st["phase"] = "brace"; st["t_b"] = t
            if gap < 0.28 + (0.0 if perceive > 0 else 0.0): st["phase"] = "contact"; st["t_c"] = t; st["v"] = max(w.speed, 0.5)
        if st["phase"] == "brace":
            u = smooth((t - st["t_b"]) / 0.22)
            hands = {s: np.array([wall_x - 0.02, w.pos[1] + (0.22 if s == "L" else -0.22), shoulder_h - 0.05]) for s in "LR"}
            arms = {s: (w.S[w.sk.idx["uarm_" + s]] + np.array([0.3, 0, -0.5]) * (1 - u) + (hands[s] - w.S[w.sk.idx["uarm_" + s]]) * u) for s in "LR"}
            if gap < arm_L + 0.18: st["phase"] = "contact"; st["t_c"] = t; st["v"] = max(w.speed, 0.5)
            return Cmd(v=np.array([v0, 0.0]), heading=0.0, arms=arms, crouch=0.25 * u)
        if st["phase"] == "contact":
            stroke = {"braced": 0.45, "late": 0.20, "blind": 0.03}[st["kind"]] if "kind" in st else 0.03
            if "kind" not in st:
                st["kind"] = "braced" if perceive >= 0.45 else "late" if perceive > 0 else "blind"
                stroke = {"braced": 0.45, "late": 0.20, "blind": 0.03}[st["kind"]]
                v = st["v"]; a = v * v / (2 * stroke); st["a"] = a; st["stroke"] = stroke; st["F"] = a / G
                st["rate"] = st["F"] / (v / a * 0.4)                      # force rises over ~40% of the stopping time
                region = "arms" if st["kind"] != "blind" else "torso"
                share = 0.5 if st["kind"] != "blind" else 1.0
                risk = INJ.risk(st["F"] * share, st["rate"] * share, region)
                st["risk"] = risk; st["region"] = region
                dirv = np.array([-1.0, 0.0])
                if st["kind"] == "braced": w.apply_hit("arm_L", dirv, 0.25); w.apply_hit("arm_R", dirv, 0.25)
                elif st["kind"] == "late": w.apply_hit("arm_L", dirv, 0.9); w.apply_hit("arm_R", dirv, 0.9)
                else: w.apply_hit("head", dirv, 1.7); w.apply_hit("torso", dirv, 2.3)
                ctx.state.setdefault("impacts", []).append((t, i, st["kind"], st["F"], risk, region))
            tt = t - st["t_c"]; v = max(0.0, st["v"] - st["a"] * tt)
            w.v = np.array([v, 0.0])
            hands = {s: np.array([wall_x - 0.02, w.pos[1] + (0.22 if s == "L" else -0.22), shoulder_h - 0.05]) for s in "LR"}
            return Cmd(v=np.zeros(2), heading=0.0, arms=hands if st["kind"] != "blind" else None, crouch=0.35 if st["kind"] != "blind" else 0.0)
        return Cmd(v=np.array([v0, 0.0]), heading=0.0)
    return cmd

@shot
def run_into_wall():
    tr = Terrain(-6, 20, -12, 12); acts = []; state = {}
    lanes = [(0.55, "sees it early: hands up, knees give, stops over ~45 cm"), (0.22, "sees it late: hands up, short stroke"), (0.0, "does not see it: chest and head take it")]
    wall_x = 12.0
    for i, (per, lab) in enumerate(lanes):
        y = (i - 1) * 4.0
        c = human(HumanBody(height=1.75 + 0.02 * i))
        acts.append(ActorSpec(c, pos=(4.5 - 0.0, y), heading=0.0, cmd=_wall_runner(i, 4.5, per, wall_x, state), label=lab, base=(0.4 + 0.1 * i, 0.5, 0.65)))
    props = [dict(geom='<geom type="box" size="0.15 7 1.6" pos="0 0 1.6" rgba="0.62 0.55 0.5 1"/>')]
    def prop_fn(t, ctx, sc, row): sc.set_prop(0, [wall_x + 0.15, 0, 0])
    def cap(t, ctx):
        im = [r for r in ctx.state.get("impacts", []) if r[0] <= t]
        return " | ".join(f"{['braced', 'late', 'blind'][['braced','late','blind'].index(k)]}: {F:.1f} mg, {reg} risk {rk:.2f}" for (_, _, k, F, rk, reg) in im)
    return Shot("run_into_wall", "running at 4.5 m/s into a wall: three runners who see it early, late, or not at all; the arms and knees lengthen the stop and lower the force", tr, acts, 9.0,
                dict(dist=9.0, azimuth=60, elevation=-12, follow="all", look_z=0.9, look_off=(1.5, 0, 0)), props=props, prop_fn=prop_fn, size=(640, 340), caption_fn=cap)

# ---------------------------------------------------------------------------- forest
@shot
def run_forest():
    rng = np.random.default_rng(4)
    tr = Terrain(-5, 60, -12, 12)
    pts = []
    while len(pts) < 70:
        p = np.array([rng.uniform(3, 52), rng.uniform(-9, 9)])
        if all(np.linalg.norm(p - q) > 1.5 for q in pts) and abs(p[1]) > 0.0: pts.append(p)
    for (x, y) in pts:
        r = rng.uniform(0.14, 0.26); tr.add_box(x - r, x + r, y - r, y + r, 8.0, rgba=(0.35, 0.25, 0.17, 1))
    for k in range(6):          # low branches across the way
        x = 8 + k * 7 + rng.uniform(-1, 1); y = rng.uniform(-3, 3)
        tr.add_beam(x - 0.2, x + 0.2, y - 1.4, y + 1.4, 1.45, 1.6, rgba=(0.30, 0.22, 0.15, 1))
    c = human(HumanBody(mass=72)); nv = Navigator(tr, c, radius=0.34)
    a = ActorSpec(c, pos=(-3.0, 0.0), cmd=lambda t, w, ctx: None)
    nv.plan(np.array([-3.0, 0.0]), np.array([56.0, 0.0]))
    a.cmd = lambda t, w, ctx: nv.cmd(w.pos, 4.2)
    return Shot("run_forest", "running through a forest at 4 m/s: A* around the trunks, curved turns, ducking under low branches", tr, [a], 16.0,
                dict(dist=7.5, azimuth=75, elevation=-14, follow=0, look_z=1.0, look_off=(1.5, 0, 0)), size=(560, 320))

# ---------------------------------------------------------------------------- crowd
@shot
def run_crowd():
    tr = Terrain(-5, 40, -12, 12); rng = np.random.default_rng(8); acts = []
    for k in range(34):
        x = rng.uniform(4, 22); y = rng.uniform(-4.5, 4.5)
        acts.append(ActorSpec(human(HumanBody(height=1.75 * rng.uniform(0.92, 1.06), mass=75 * rng.uniform(0.8, 1.25))), pos=(x, y), heading=float(rng.uniform(-math.pi, math.pi)), cmd=lambda t, w, ctx: Cmd(), base=(0.55, 0.55, 0.5)))
    n0 = len(acts)
    R = ActorSpec(human(HumanBody(mass=72)), pos=(-3.0, 0.5), heading=0.0, base=(0.25, 0.45, 0.85))
    acts.append(R)
    R.cmd = CBd.squeeze_cmd(n0, (26.0, 0.0), 3.4)
    return Shot("run_crowd", "running through a crowd at 3.4 m/s: speed follows the gap width, shoulders turn, hands go up", tr, acts, 12.0,
                dict(dist=9.5, azimuth=80, elevation=-30, follow=n0, look_z=0.8, look_off=(2.5, 0, 0)), size=(560, 340))

# ---------------------------------------------------------------------------- clothing
@shot
def outfits_walk():
    """The same walk and the same stairs for eight outfits: ranges of motion, mass and foot shape change, the planner copes."""
    names = ["none", "skirt_tight", "skirt_wide", "trousers_tight", "boots_heavy", "heels_high", "chainmail", "plate"]
    tr = Terrain(-3, 30, -14, 14); tr.add_stairs(9.0, 0.0, 5, 0.17, 0.30, 14.0); tr.add_box(10.5, 30, -7, 7, 0.85)
    acts = []
    for i, n in enumerate(names):
        c = human(HumanBody(outfit=n, sex="f" if n in ("skirt_tight", "skirt_wide", "heels_high") else "m"))
        y = (i - (len(names) - 1) / 2) * 1.4
        acts.append(ActorSpec(c, pos=(0.0, y), heading=0.0, cmd=lambda t, w, ctx: Cmd(v=np.array([2.6 if w.pos[0] < 22.0 else 0.0, 0.0]), heading=0.0), label=n, base=(0.35 + 0.07 * (i % 4), 0.5, 0.7 - 0.05 * (i % 3))))
    return Shot("outfits_walk", "same walk and stairs in: plain, tight long skirt, wide skirt, tight trousers, heavy boots, high heels, chainmail, plate armour (limits, mass, foot shape)", tr, acts, 14.0,
                dict(dist=11.0, azimuth=60, elevation=-14, follow="all", look_z=0.9), size=(640, 340))

# ---------------------------------------------------------------------------- climbing where only some points can be held
import climb as CL

def _rock_holds(rng, x_wall, height, width=1.6, gap=0.34):
    pts = []
    for _ in range(4000):
        p = np.array([x_wall, rng.uniform(-width / 2, width / 2), rng.uniform(0.25, height)])
        if all(np.linalg.norm(p - q.p) > gap for q in pts):
            kind = rng.choice(["hand", "foot", "both", "both", "both"])
            pts.append(CL.Hold(p, str(kind)))
        if len(pts) > 110: break
    return pts

def _holds_geom(holds, wall_x, thick=0.2, w=1.8, h=7.0, rgba="0.5 0.47 0.43 1", tree=False):
    g = ""
    if not tree: g += f'<geom type="box" size="{thick / 2} {w / 2} {h / 2}" pos="{wall_x + thick / 2} 0 {h / 2}" rgba="{rgba}"/>'
    else: g += f'<geom type="cylinder" size="0.22 {h / 2}" pos="{wall_x + 0.22} 0 {h / 2}" rgba="0.38 0.27 0.18 1"/>'
    for hd in holds:
        col = {"hand": "0.9 0.75 0.3 1", "foot": "0.45 0.7 0.9 1", "both": "0.85 0.85 0.85 1"}[hd.kind]
        if tree: g += f'<geom type="capsule" fromto="{wall_x + 0.15} {hd.p[1] * 0.6} {hd.p[2]} {hd.p[0]} {hd.p[1]} {hd.p[2]}" size="0.025" rgba="0.33 0.23 0.15 1"/>'
        else: g += f'<geom type="ellipsoid" size="0.05 0.09 0.04" pos="{hd.p[0] + 0.02} {hd.p[1]} {hd.p[2]}" rgba="{col}"/>'
    return g

def _climb_shot(name, title, holds, wall_x, goal, tree, T=40.0):
    c = human(); tr = Terrain(-3, 8, -4, 4)
    cl = CL.Climber(c, holds, (-1.0, 0.0, 0.0), (wall_x - 0.42, 0.0), goal)
    def kin(t, ctx):
        S, E, R = cl.step(1 / 60); ctx.state["climber"] = cl; return S, E, R
    a = ActorSpec(c, kinematic=kin, base=(0.4, 0.5, 0.7))
    props = [dict(geom=_holds_geom(holds, wall_x, tree=tree))]
    def cap(t, ctx):
        k = ctx.state.get("climber")
        return "" if k is None else ("STUCK: no hold in reach" if k.stuck and k.moving is None else f"holds used: {len(k.log)}, height {k.pelvis[2]:.1f} m")
    return Shot(name, title, tr, [a], T, dict(dist=6.5, azimuth=20, elevation=-4, follow=0, look_z=None, look_off=(0, 0, 0.3)), props=props, prop_fn=lambda t, ctx, sc, row: sc.set_prop(0, [0, 0, 0]),
                size=(480, 420), caption_fn=cap, fps=20)

@shot
def climb_rock():
    rng = np.random.default_rng(2); wall_x = 3.0
    holds = _rock_holds(rng, wall_x, 5.4)
    return _climb_shot("climb_rock", "free climbing: only the coloured holds can be used (yellow hands, blue feet, grey both); three points stay, the fourth reaches for the best hold", holds, wall_x, 5.0, False, 42.0)

@shot
def climb_tree():
    rng = np.random.default_rng(5); wall_x = 3.0; holds = []; z = 0.55; side = 1.0
    while z < 6.2:
        holds.append(CL.Hold(np.array([wall_x - 0.14, side * rng.uniform(0.12, 0.26), z]), "both")); z += rng.uniform(0.42, 0.62); side = -side
        if rng.random() < 0.5: holds.append(CL.Hold(np.array([wall_x - 0.14, -side * rng.uniform(0.12, 0.28), z - 0.22]), "foot"))
    return _climb_shot("climb_tree", "climbing a tree: branch stubs are the only holds; the climber hugs the trunk and uses them alternately", holds, wall_x, 6.0, True, 42.0)

# ---------------------------------------------------------------------------- limping: a sore leg, a stick for a leg, crutches, a dog on three legs
import aids as AIDS
from render import z_to_quat

CRUTCH_GEOM = '<geom type="capsule" fromto="0 0 0 0 0 1.12" size="0.013" rgba="0.55 0.4 0.25 1"/><geom type="capsule" fromto="0 -0.05 0.5 0 0.05 0.5" size="0.012" rgba="0.3 0.3 0.3 1"/>'

def _walk_cmd(v):
    return lambda t, w, ctx: Cmd(v=np.array([v if t > 0.5 else 0.0, 0.0]), heading=0.0)

@shot
def limp_people():
    tr = Terrain(-3, 30, -8, 8); acts = []; labs = []
    specs = [("healthy", dict(), 1.3, None), ("sore leg (chronic pain)", dict(), 1.3, "pain"), ("rigid stick instead of the shank", dict(peg="L"), 1.3, None),
             ("on crutches, one leg held up", dict(), 0.9, "crutch")]
    for i, (lab, kw, v, mode) in enumerate(specs):
        c = human(HumanBody(**kw)); y = (i - 1.5) * 1.5
        base_cmd = _walk_cmd(v)
        def cmd(t, w, ctx, base_cmd=base_cmd):
            if w.aids is not None and w.aids.tops: ctx.state.setdefault("crutch", {})[round(t * 60)] = {sd: (a_.copy(), b_.copy()) for sd, (a_, b_) in w.aids.segments().items()}
            return base_cmd(t, w, ctx)
        a = ActorSpec(c, pos=(0.0, y), heading=0.0, cmd=cmd, label=lab, base=(0.35 + 0.08 * i, 0.5, 0.7 - 0.06 * i))
        if mode == "pain": a.events = [(0.0, lambda w, ctx: setattr(w, "chronic", {"leg_L": 0.8}))]
        if mode == "crutch": a.events = [(0.0, lambda w, ctx: AIDS.use_crutches(w, "leg_L"))]
        acts.append(a)
    props = [dict(geom=CRUTCH_GEOM), dict(geom=CRUTCH_GEOM)]
    def prop_fn(t, ctx, sc, row):
        seg = ctx.state.get("crutch", {}).get(round(t * 60)) or ctx.state.get("crutch", {}).get(round(t * 60) - 1) or {}
        for k, sd in enumerate("LR"):
            if sd in seg:
                tip, top = seg[sd]; d = top - tip; d = d / max(np.linalg.norm(d), 1e-6); sc.set_prop(k, tip, z_to_quat(d))
            else: sc.set_prop(k, np.array([0, 0, -5.0]), z_to_quat(np.array([0, 0, 1.0])))
    return Shot("limp_people", "walking with a sore leg (shorter time on it, trunk leans over it), a rigid stick for a lower leg (swung round, short stance), and on crutches with one leg held up",
                tr, acts, 11.0, dict(dist=6.5, azimuth=70, elevation=-12, follow="all", look_z=0.8), props=props, prop_fn=prop_fn, size=(640, 340))

@shot
def limp_dogs():
    tr = Terrain(-3, 30, -8, 8); acts = []
    specs = [("healthy", dict(), None), ("sore hind leg", dict(), "pain"), ("hind leg missing: three legs", dict(missing="HL"), None), ("stick for the lower hind leg", dict(peg="HL"), None)]
    for i, (lab, kw, mode) in enumerate(specs):
        c = quadruped("dog", **kw); y = (i - 1.5) * 1.1
        a = ActorSpec(c, pos=(0.0, y), heading=0.0, cmd=_walk_cmd(1.2), label=lab)
        if mode == "pain": a.events = [(0.0, lambda w, ctx: setattr(w, "chronic", {"leg_L": 0.8}))]
        acts.append(a)
    return Shot("limp_dogs", "dogs: sore leg, three legs, a stick in place of the lower hind leg", tr, acts, 11.0, dict(dist=5.0, azimuth=70, elevation=-12, follow="all", look_z=0.4), size=(640, 340))
