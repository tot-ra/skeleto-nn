"""Fast moves in clutter: running into a wall (with and without the arms), running through a forest and through a crowd."""
from __future__ import annotations
import math
import numpy as np
from bodies import *
from bodies import persona
from posing import Poser
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
        holds.append(CL.Hold(np.array([wall_x - 0.14, side * rng.uniform(0.12, 0.26), z]), "both")); z += rng.uniform(0.34, 0.50); side = -side
        if rng.random() < 0.7: holds.append(CL.Hold(np.array([wall_x - 0.14, -side * rng.uniform(0.12, 0.28), z - 0.17]), "foot"))
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

# ---------------------------------------------------------------------------- jumping different distances: the body knows what it can do
import jumper as JP
import dataclasses

def _practised(c, prior, sd, trials, seed=1):
    w = Walker(c, Terrain(-3, 40, -6, 6))
    for _ in range(60): w.step(Cmd())
    m = JP.SelfModel(w, prior, sd, np.random.default_rng(seed)); m.practise(trials); return m

def _gap_scene(name, title, lanes, T=20.0, depth=1.5, size=(640, 360)):
    """lanes: list of dict(gap, body=HumanBody kwargs, prior=(mean, sd), trials)."""
    tr = Terrain(-3, 46, -14, 14); acts = []; jumpers = []; models = []
    for k, ln in enumerate(lanes):
        y = (k - (len(lanes) - 1) / 2) * 2.4; x_edge = 12.0; gap = ln["gap"]
        tr.add_pit(x_edge, x_edge + gap, y - 1.0, y + 1.0, depth, rgba=(0.12, 0.10, 0.09, 1)) if "rgba" in tr.add_pit.__code__.co_varnames else tr.add_pit(x_edge, x_edge + gap, y - 1.0, y + 1.0, depth)
        c = human(HumanBody(**ln.get("body", {}))); m = _practised(c, ln.get("prior", (1.0, 0.3))[0], ln.get("prior", (1.0, 0.3))[1], ln.get("trials", 12), seed=k + 1)
        gj = JP.GapJumper(m, y, x_edge, gap, depth, label=ln.get("label", ""))
        a = ActorSpec(c, pos=(x_edge - 9.0, y), heading=0.0, cmd=gj, label=ln.get("label", ""), base=(0.35 + 0.08 * (k % 4), 0.5, 0.7 - 0.06 * (k % 3)))
        acts.append(a); jumpers.append(gj); models.append(m)
    def cap(t, ctx):
        parts = []
        for g in jumpers:
            if g.decision == "jump": parts.append(f"{g.gap:.1f} m: run {g.v_run:.1f} m/s (p {g.p:.2f}) {g.outcome}")
            else: parts.append(f"{g.gap:.1f} m: afraid (p {g.p:.2f})")
        return " | ".join(parts)
    sh = Shot(name, title, tr, acts, T, dict(dist=13.5, azimuth=70, elevation=-24, follow="all", look_z=0.5, look_off=(3.0, 0, 0)), size=size, caption_fn=cap)
    sh.jumpers = jumpers; sh.models = models
    return sh

@shot
def jump_gap():
    lanes = [dict(gap=gap) for gap in (0.8, 1.8, 2.8, 3.4)]
    return _gap_scene("jump_gap", "gaps of 0.8, 1.8, 2.8 and 3.4 m: the same person picks the run-up speed that the gap needs (a hop from a walk, a long jump from a sprint)", lanes)

@shot
def jump_fear():
    lanes = [dict(gap=3.4, label="knows its limit"), dict(gap=4.4, label="knows its limit"),
             dict(gap=3.8, prior=(1.5, 0.2), trials=2, label="over-confident"), dict(gap=2.8, prior=(0.6, 0.2), trials=2, label="over-cautious")]
    return _gap_scene("jump_fear", "knowing what you can jump: 3.4 m at full sprint; 4.4 m is refused (it stops and is afraid); an over-confident one runs and is refused at the edge; an over-cautious one will not try a jump it could make", lanes, T=22.0)

@shot
def obstacle_course():
    """Running over rough ground with things in the way: a log, a pit, a low wall, a longer pit, a waist-high wall (vaulted), a wall that is too high (gone round)."""
    rng = np.random.default_rng(7); tr = Terrain(-3, 90, -9, 9)
    for k in range(70):                                                        # uneven ground: low bumps
        x = 2 + k * 1.2 + rng.uniform(-0.3, 0.3); y = rng.uniform(-1.4, 1.4)
        tr.add_box(x, x + rng.uniform(0.4, 0.8), y - 0.4, y + 0.4, rng.uniform(0.03, 0.09), rgba=(0.62, 0.57, 0.50, 1))
    tr.add_box(14.0, 14.4, -2.0, 2.0, 0.30, rgba=(0.42, 0.30, 0.18, 1))        # a log
    tr.add_pit(23.0, 24.4, -3.0, 3.0, 1.5)                                     # a small pit
    tr.add_box(32.0, 32.25, -2.5, 2.5, 0.65, rgba=(0.55, 0.50, 0.46, 1))      # a low wall
    tr.add_pit(41.0, 43.4, -3.0, 3.0, 1.5)                                     # a long pit
    tr.add_box(52.0, 52.3, -2.5, 2.5, 1.0, rgba=(0.55, 0.50, 0.46, 1))        # a waist-high wall
    tr.add_box(64.0, 64.3, -2.5, 2.5, 1.7, rgba=(0.50, 0.45, 0.42, 1))        # too high
    tr.add_pit(76.0, 81.6, -9.0, 9.0, 1.5)                                     # wider than any jump: the way ends here
    c = human(); m = _practised(c, 1.0, 0.3, 12)
    rn = JP.ObstacleRunner(m, 0.0, v_run=4.2, detour_y=3.4, label="runner")
    a = ActorSpec(c, pos=(0.0, 0.0), heading=0.0, cmd=rn, base=(0.3, 0.5, 0.8))
    def cap(t, ctx):
        return " | ".join(f"{k} {v:.1f} m: {o}" for (tt, k, v, o) in rn.log if tt <= t)
    sh = Shot("obstacle_course", "running over uneven ground: a log is stepped over, pits are jumped with the run-up the body needs, a low wall is hopped, a waist-high wall is vaulted with the hands on top, a wall that is too high is gone round, a pit that is too wide stops it",
              tr, [a], 40.0, dict(dist=8.0, azimuth=75, elevation=-14, follow=0, look_z=0.9, look_off=(2.0, 0, 0)), size=(640, 340), caption_fn=cap)
    sh.runner = rn
    return sh

@shot
def start_from_squat():
    """A person squatting who suddenly has to run: unhurried (stands up first), hurried (drives out of the squat, trunk forward), and a sprinter's start with hands on the ground."""
    tr = Terrain(-3, 60, -8, 8); acts = []
    lanes = [("unhurried", dict(), 0.2, 4.5, False), ("hurried", dict(), 0.7, 5.0, False), ("sprinter, hands down", dict(muscle=1.4, energy=1.5), 1.0, 7.0, True)]
    for i, (lab, kw, u, v, hd) in enumerate(lanes):
        c = human(HumanBody(**kw)); y = (i - 1) * 1.6
        a = ActorSpec(c, pos=(0.0, y), heading=0.0, cmd=JP.Starter(u, v, t_go=1.2, hands_down=hd), label=lab, base=(0.35 + 0.1 * i, 0.5, 0.7 - 0.08 * i))
        acts.append(a)
    return Shot("start_from_squat", "from a squat to a run: unhurried it stands up first and then speeds up; hurried it drives out of the squat with the trunk forward and rises as the speed does; the sprinter starts with the hands on the ground",
                tr, acts, 7.0, dict(dist=8.0, azimuth=62, elevation=-10, follow="all", look_z=0.8, look_off=(2.0, 0, 0)), size=(640, 340))

@shot
def personas_course():
    """Five people with different muscle, energy and flexibility run the same course: log, pit, low wall, wider pit, waist-high wall, a pit nobody can cross.
    Each decides from what its own body can do (self-model learned by practice); the child also plays."""
    rng = np.random.default_rng(3); tr = Terrain(-3, 80, -9, 9)
    for k in range(60):
        x = 2 + k * 1.1 + rng.uniform(-0.3, 0.3); y = rng.uniform(-3.5, 3.5)
        tr.add_box(x, x + rng.uniform(0.4, 0.8), y - 0.4, y + 0.4, rng.uniform(0.03, 0.08), rgba=(0.62, 0.57, 0.50, 1))
    tr.add_box(12.0, 12.4, -8.5, 8.5, 0.30, rgba=(0.42, 0.30, 0.18, 1))
    tr.add_pit(20.0, 21.6, -8.5, 8.5, 1.5)
    tr.add_box(29.0, 29.25, -8.5, 8.5, 0.60, rgba=(0.55, 0.50, 0.46, 1))
    tr.add_pit(38.0, 40.4, -8.5, 8.5, 1.5)
    tr.add_box(48.0, 48.3, -8.5, 8.5, 0.90, rgba=(0.55, 0.50, 0.46, 1))
    tr.add_pit(57.0, 62.6, -8.5, 8.5, 1.5)
    people = [("child", (1.3, 0.25)), ("adult", (1.0, 0.3)), ("elder", (0.7, 0.25)), ("starved", (0.8, 0.3)), ("athlete", (1.0, 0.3))]
    acts = []; runners = []; plays = []
    for i, (nm, prior) in enumerate(people):
        c = persona(nm); y = (i - 2) * 1.5
        m = _practised(c, prior[0], prior[1], 12, seed=i + 1)
        w0 = Walker(c, Terrain(-3, 40, -6, 6)); vs = float(c.params.get("v_sprint", 6.2))
        rn = JP.ObstacleRunner(m, y, v_run=float(min(4.4, 0.72 * vs)), detour_y=3.4, label=nm); runners.append(rn)
        pl = JP.Playful(w0, np.random.default_rng(i)); plays.append(pl)
        def cmd(t, w, ctx, rn=rn, pl=pl, y=y):
            ev = rn._scan(w, look=5.0)
            if ev is None and w.speed > 0.5: pl.step(w, 1 / 60)
            return rn(t, w, ctx)
        acts.append(ActorSpec(c, pos=(0.0, y), heading=0.0, cmd=cmd, label=nm, base=(0.3 + 0.12 * i, 0.5, 0.8 - 0.12 * i)))
    sh = Shot("personas_course", "child, adult, elder, starved, athlete on the same course: muscle, energy and flexibility decide how fast they run, what they jump, what they refuse",
              tr, acts, 36.0, dict(dist=15.0, azimuth=72, elevation=-20, follow="all", look_z=0.6), size=(680, 360), caption_fn=lambda t, ctx: " | ".join(f"{r.label}: " + (lambda xs: (f"{xs[-1][1]} {xs[-1][3]}" if xs else "running"))([x for x in r.log if x[0] <= t]) for r in runners))
    sh.runners = runners
    return sh

# ---------------------------------------------------------------------------- input with inertia
import controls as CT

def _stick_script(t):
    """(angle, magnitude) of the stick over time: walk, run, reverse, sharp left, sharp right, stop, walk back."""
    if t < 1.0: return 0.0, 0.0
    if t < 3.5: return 0.0, 0.5                                   # walk forward
    if t < 8.0: return 0.0, 1.0                                   # run forward
    if t < 13.0: return math.pi, 1.0                              # the stick is thrown to the opposite side: run back
    if t < 16.0: return math.pi / 2, 1.0                          # then hard left (screen left = +y)
    if t < 19.0: return 3 * math.pi / 2, 1.0                      # and hard right
    if t < 21.5: return math.pi, 0.0                              # let go
    return 0.0, 0.5                                               # walk the other way

@shot
def joystick_control():
    """The same stick script for the planner alone (left) and with the stick rules (right): braking, turning on the spot, cornering limited by friction, lean."""
    tr = Terrain(-40, 40, -12, 12); acts = []; hud = {}
    for i, naive in enumerate((True, False)):
        c = human(); y = (i - 0.5) * 3.0
        ctl = CT.Joystick(_stick_script, naive=naive)
        acts.append(ActorSpec(c, pos=(0.0, y), heading=0.0, cmd=ctl, label="planner only" if naive else "stick rules: brake, pivot, corner", base=(0.75 - 0.3 * i, 0.45, 0.35 + 0.35 * i)))
        hud[i] = ctl
    def cap(t, ctx):
        ang, mag = _stick_script(t); names = ["right", "up", "left", "back"]
        d = "stop" if mag < 0.05 else {0: "forward", 1: "left", 2: "back", 3: "right"}.get(int(round(((ang % (2 * math.pi)) / (math.pi / 2))) % 4), "?")
        return f"stick: {d}{' (walk)' if 0.05 < mag <= 0.5 else ' (run)' if mag > 0.5 else ''}"
    return Shot("joystick_control", "the stick says where to go and how fast; a body cannot do it at once: it brakes (trunk leans back), turns on the spot, then accelerates, and its speed falls before a sharp turn (right: with the stick rules, left: the planner alone)",
                tr, acts, 26.0, dict(dist=11.0, azimuth=70, elevation=-22, follow="all", look_z=0.8), size=(640, 340), caption_fn=cap)

# ---------------------------------------------------------------------------- the horse: gaits, turns with a big body's inertia, fences
@shot
def horse_gaits():
    """One horse, walk to gallop (gait from the Froude number), with a long neck, a stiff trunk, hooves and a tail that follows."""
    tr = Terrain(-3, 120, -8, 8); c = quadruped("horse"); G = 9.81
    sched = [(0, 0.0), (1.0, 1.5), (5.0, 3.6), (9.0, 6.0), (13.0, 9.5), (17.5, 3.0), (20.0, 0.0)]
    def cmd(t, w, ctx):
        v = 0.0
        for ts, vs in sched:
            if t >= ts: v = vs
        return Cmd(v=np.array([v, 0.0]), heading=0.0)
    a = ActorSpec(c, pos=(0, 0), cmd=cmd, base=(0.45, 0.30, 0.2))
    return Shot("horse_gaits", "a horse: walk, trot, canter, gallop (the gait follows from the speed), neck and tail moving with it", tr, [a], 22.0, dict(dist=9.0, azimuth=90, elevation=-4, follow=0, look_z=1.1), size=(520, 300))

def _waypoint_cmd(pts, speed, tol=2.0):
    st = dict(i=0)
    def cmd(t, w, ctx):
        while st["i"] < len(pts) - 1 and np.linalg.norm(np.array(pts[st["i"]]) - w.pos) < tol: st["i"] += 1
        d = np.array(pts[st["i"]]) - w.pos; n = np.linalg.norm(d)
        fwd = np.array([math.cos(w.heading), math.sin(w.heading)])           # a big body moves along where it faces and turns towards the target
        return Cmd(v=fwd * speed, heading=math.atan2(d[1], d[0]))
    return cmd

def _slalom_shot(species, name, title, dist):
    tr = Terrain(-3, 90, -22, 22); G = 9.81
    c = quadruped(species); v = math.sqrt(1.4 * G * c.z0)
    pts = [(12, 4), (24, -4), (36, 4), (48, -4), (62, 0), (80, 0)]
    for (px, py) in pts[:-1]:                                  # flags at the corners of the slalom (drawn only: nothing to collide with)
        tr.prims.append(("box", (px - 0.15, px + 0.15, py + (1.6 if py < 0 else -1.6) - 0.15, py + (1.6 if py < 0 else -1.6) + 0.15, 0.0, 1.6), (0.85, 0.25, 0.2, 1)))
    a = ActorSpec(c, pos=(0, 0), cmd=_waypoint_cmd(pts, v), label=species, base=(0.55, 0.4, 0.3))
    return Shot(name, title, tr, [a], 22.0 if species == "horse" else 16.0, dict(dist=dist * 1.6, azimuth=90, elevation=-35, follow=0, look_z=0.3, look_off=(3, 0, 0)), size=(560, 340))

@shot
def slalom_cat(): return _slalom_shot("cat", "slalom_cat", "a cat at a run through a slalom: tight turns, the spine bends, the tail swings against the turn", 7.0)
@shot
def slalom_dog(): return _slalom_shot("dog", "slalom_dog", "a dog at a run through a slalom: turns in about a metre, the trunk leans into it", 9.0)
@shot
def slalom_horse(): return _slalom_shot("horse", "slalom_horse", "a horse at a canter through the same slalom: the big body swings wide (turning radius about 6 m at this speed) and cannot cut the turns", 24.0)

@shot
def horse_jumps():
    """Three horses take fences of 1.0, 1.6 and 2.7 m at a canter. Each asks its own body: the last cannot clear it and refuses, swerving aside."""
    tr = Terrain(-3, 60, -14, 14); acts = []; fences = []
    for k, h in enumerate((1.0, 1.6, 2.7)):
        y = (k - 1) * 4.5; xf = 26.0
        tr.add_box(xf, xf + 0.25, y - 2.5, y + 2.5, h, rgba=(0.6, 0.45, 0.3, 1))
        fj = JP.FenceJumper(y, xf, h, v_approach=6.5, label=f"{h:.1f} m"); fences.append(fj)
        c = quadruped("horse"); acts.append(ActorSpec(c, pos=(0, y), heading=0.0, cmd=fj, label=f"{h} m", base=(0.45, 0.30, 0.2)))
    return Shot("horse_jumps", "fences of 1.0, 1.6 and 2.7 m at a canter: the take-off point moves back with the height, the body rears and folds; the highest fence is beyond the horse and it refuses",
                tr, acts, 13.0, dict(dist=15.0, azimuth=85, elevation=-22, follow="all", look_z=0.8, look_off=(2, 0, 0)), size=(640, 340),
                caption_fn=lambda t, ctx: " | ".join(f"{f.label}: {f.outcome or 'approaching'}" for f in fences))

@shot
def rider_course():
    """A horse with a rider over a course: canter, a slalom with the big body's wide turns, a fence (the rider rises out of the saddle and leans forward), canter on."""
    import catalog_extra as CE
    tr = Terrain(-3, 100, -14, 14); xf = 52.0; tr.add_box(xf, xf + 0.25, -2.5, 2.5, 1.1, rgba=(0.6, 0.45, 0.3, 1))
    H = quadruped("horse"); P = human(HumanBody(mass=75)); poser = Poser(P)
    pts = [(10, 3), (20, -3), (30, 3), (40, 0), (50, 0)]
    steer = _waypoint_cmd(pts, 6.0); fj = JP.FenceJumper(0.0, xf, 1.1, v_approach=6.0)
    def horse_cmd(t, w, ctx):
        if w.pos[0] < 38.0: return steer(t, w, ctx)
        return fj(t, w, ctx)
    kin = CE.rider_kin(P, poser)
    h = ActorSpec(H, pos=(0, 0), cmd=horse_cmd, base=(0.45, 0.30, 0.2)); r = ActorSpec(P, kinematic=kin, base=(0.30, 0.38, 0.55))
    return Shot("rider_course", "a horse with a rider: canter, a slalom (wide turns, the rider leans with them), a 1.1 m fence (the rider rises out of the saddle and folds forward), canter on",
                tr, [h, r], 16.0, dict(dist=11.0, azimuth=80, elevation=-10, follow=0, look_z=1.4), size=(560, 320))

# ---------------------------------------------------------------------------- pregnancy
@shot
def pregnant_stairs():
    """A woman in a wide skirt on stairs, not pregnant (left) and heavily pregnant (right): the second slows before the steps, plants each foot fully on the tread,
    holds the skirt with one hand and supports the belly with the other, leans back, and sways from side to side."""
    tr = Terrain(-3, 30, -8, 8); xe = tr.add_stairs(6.0, 0.0, 9, 0.17, 0.30, 15.0); tr.add_box(xe, xe + 6.0, -7.5, 7.5, 9 * 0.17)
    acts = []
    for i, (lab, kw) in enumerate((("not pregnant", dict(sex="f", outfit="skirt_wide")), ("heavily pregnant", dict(pregnancy=1.0, outfit="skirt_wide")))):
        c = human(HumanBody(**kw)); y = (i - 0.5) * 2.0; care = c.params.get("pregnancy", 0.0)
        def cmd(t, w, ctx, care=care):
            near = (w.pos[0] > 6.0 - 2.5) and (w.pos[0] < xe + 0.5)
            v = 1.5 * (1.0 - 0.45 * care) if not near else 1.2 * (1.0 - 0.55 * care)
            arms = None
            if near and care > 0 and w.S is not None:
                f = np.array([math.cos(w.heading), math.sin(w.heading), 0.0]); l = np.array([-f[1], f[0], 0.0]); p = np.array([w.pos[0], w.pos[1], w.z])
                arms = {"R": p + f * 0.18 - l * 0.28 + np.array([0, 0, -0.22]), "L": p + f * 0.28 + np.array([0, 0, 0.12])}      # hem in one hand, the other under the belly
            if care > 0 and near: w.auto_duck = False
            return Cmd(v=np.array([v, 0.0]), heading=0.0, arms=arms)
        acts.append(ActorSpec(c, pos=(0.0, y), heading=0.0, cmd=cmd, label=lab, base=(0.6 - 0.2 * i, 0.4, 0.5 + 0.2 * i)))
    return Shot("pregnant_stairs", "stairs in a wide skirt: left not pregnant, right heavily pregnant (slower, feet fully on the treads, hand on the skirt and under the belly, trunk back, a waddle)",
                tr, acts, 16.0, dict(dist=7.0, azimuth=70, elevation=-8, follow="all", look_z=0.9, look_off=(1.5, 0, 0)), size=(560, 320))

@shot
def personas_race():
    """Everyone runs as fast as they can for 36 s: the energy reserve drains with effort, a tired body slows down; the child (a lot of energy, shorter legs) keeps its pace, the starved one is spent in seconds."""
    tr = Terrain(-3, 260, -9, 9); acts = []; walkers_ref = {}
    names = ["child", "adult", "elder", "starved", "athlete"]
    for i, nm in enumerate(names):
        c = persona(nm); y = (i - 2) * 1.6; vs = float(c.params.get("v_sprint", 6.2))
        acts.append(ActorSpec(c, pos=(0.0, y), heading=0.0, cmd=(lambda t, w, ctx, vs=vs: Cmd(v=np.array([0.88 * vs if t > 0.8 else 0.0, 0.0]), heading=0.0)), label=nm, base=(0.3 + 0.12 * i, 0.5, 0.8 - 0.12 * i)))
    def cap(t, ctx):
        ws = ctx.state.get("race", {}).get(round(t * 20)) or {}
        return " | ".join(f"{n}: {ws[n][0]:.0f} m, energy {ws[n][1] * 100:.0f}%" for n in names if n in ws)
    for i, a in enumerate(acts):
        inner = a.cmd
        def cmd(t, w, ctx, inner=inner, n=names[i]):
            ctx.state.setdefault("race", {}).setdefault(round(t * 20), {})[n] = (float(w.pos[0]), float(w.stamina)); return inner(t, w, ctx)
        a.cmd = cmd
    return Shot("personas_race", "a 36 s all-out run: the pace each body can hold falls as its energy reserve is spent (energy and muscle differ: child, adult, elder, starved, athlete)", tr, acts, 36.0,
                dict(dist=26.0, azimuth=78, elevation=-24, follow="all", look_z=0.5), size=(680, 340), caption_fn=cap, fps=12)

@shot
def animal_ages():
    """A dog, an old dog, a puppy and a kid goat on a walk: the old dog is slow and stiff, the young ones have more energy than they need and spend it in hops."""
    import dataclasses
    from bodies import QUADS
    tr = Terrain(-3, 60, -8, 8); acts = []; plays = []
    specs = [("adult dog", QUADS["dog"], 1.0), ("old dog", dataclasses.replace(QUADS["dog"], vigor=0.5, energy=0.5), 0.5),
             ("puppy", dataclasses.replace(QUADS["dog"], scale=0.55, mass=8.0, vigor=1.2, energy=2.0), 1.8), ("kid goat", dataclasses.replace(QUADS["goat"], scale=0.6, mass=12.0, vigor=1.3, energy=2.2), 2.0)]
    for i, (nm, sp, ev) in enumerate(specs):
        c = quadruped(sp); y = (i - 1.5) * 1.3; w0 = Walker(c, Terrain(-3, 40, -6, 6)); pl = JP.Playful(w0, np.random.default_rng(i), rate=0.5, hop=0.55); plays.append(pl)
        v = math.sqrt(0.3 * 9.81 * c.z0) * (0.6 if "old" in nm else 1.0)
        def cmd(t, w, ctx, v=v, pl=pl): pl.step(w, 1 / 60); return Cmd(v=np.array([v if t > 0.5 else 0.0, 0.0]), heading=0.0)
        acts.append(ActorSpec(c, pos=(0.0, y), heading=0.0, cmd=cmd, label=nm, base=(0.55, 0.4, 0.3 + 0.1 * i)))
    return Shot("animal_ages", "a dog, an old dog, a puppy and a kid goat: slow and stiff, or full of energy that is spent in hops", tr, acts, 14.0, dict(dist=7.0, azimuth=72, elevation=-14, follow="all", look_z=0.4, look_off=(2, 0, 0)), size=(640, 340))
