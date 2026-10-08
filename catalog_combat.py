"""Combat as reactions: being hit (head, body, arm, leg; several directions), avoiding a blow in time, and a free exchange of blows."""
from __future__ import annotations
import math
import numpy as np
from bodies import *
from planner import Cmd
from shots import *
from catalog import shot
from render import z_to_quat
import combat as CB

AXE_GEOM = '<geom type="capsule" fromto="0 0 -0.12 0 0 0.85" size="0.02" rgba="0.5 0.35 0.2 1"/><geom type="box" size="0.012 0.08 0.07" pos="0 0.05 0.80" rgba="0.55 0.57 0.6 1"/>'
BOW_GEOM = '<geom type="capsule" fromto="0.0 0 -0.45 0.05 0 0.0" size="0.012" rgba="0.45 0.3 0.15 1"/><geom type="capsule" fromto="0.05 0 0.0 0.0 0 0.45" size="0.012" rgba="0.45 0.3 0.15 1"/>'
ARROW_GEOM = '<geom type="capsule" fromto="0 0 -0.35 0 0 0.35" size="0.006" rgba="0.7 0.6 0.4 1"/><geom type="capsule" fromto="0 0 0.33 0 0 0.40" size="0.012" rgba="0.6 0.62 0.65 1"/>'
SHIELD_GEOM = '<geom type="cylinder" size="0.30 0.02" rgba="0.55 0.18 0.15 1"/><geom type="sphere" size="0.045" pos="0 0 0.03" rgba="0.8 0.8 0.8 1"/>'
STICK_GEOM = '<geom type="capsule" fromto="0 0 -0.12 0 0 0.85" size="0.02" rgba="0.55 0.38 0.20 1"/>'
FLOOR_ANGLE = {"front": 0.0, "side": math.pi / 2, "behind": math.pi, "side_r": -math.pi / 2}

def _arena(ctx):
    return ctx.state["arena"]

def _build(lanes, T, name, title, cam, dx=0.0, size=(600, 320), spacing=3.8, seed=0):
    """lanes: list of dicts(region, from, strength, delay, react, walk, kind) -> victim + attacker pairs side by side."""
    tr = Terrain(-8, 14, -14, 14); acts = []; fighters = []; props = []; weapon_props = []
    rng = np.random.default_rng(seed)
    n = len(lanes)
    for k, ln in enumerate(lanes):
        y = 0.0; dxk = (k - (n - 1) / 2) * spacing
        V = human(HumanBody(height=1.74 + 0.03 * (k % 3), mass=76 if ln.get("sex", "m") == "m" else 62, sex=ln.get("sex", "m"))); A = human(HumanBody(height=1.8, mass=84))
        ang = FLOOR_ANGLE[ln.get("from", "front")]
        vi = len(acts); ai = vi + 1
        acts.append(ActorSpec(V, pos=(dx + dxk, y), heading=0.0, base=(0.35, 0.5, 0.7)))
        rng_ = ln.get("range", 1.3); apos = (dx + dxk + rng_ * math.cos(ang), y + rng_ * math.sin(ang))
        acts.append(ActorSpec(A, pos=apos, heading=math.atan2(y - apos[1], dx + dxk - apos[0]), base=(0.65, 0.45, 0.35)))
        fv = CB.Fighter(vi, ai, rng, attacks=False, react=ln.get("react", False), delay=(ln.get("delay", 0.2),) * 2, forced=ln.get("forced"), shield=ln.get("shield", False), leg_block=ln.get("leg_block", False), dist=ln.get("range", 1.3))
        if "walk" in ln: fv.walk = (ln["walk"][0], np.array(ln["walk"][1], float))
        fa = CB.Fighter(ai, vi, rng, attacks=True, react=False, regions={ln["region"]: 1}, strength=(ln.get("strength", 1.2),) * 2, cooldown=(99, 99), first=ln.get("t", 1.0), dist=ln.get("range", 1.25), weapon=ln.get("weapon", "stick"))
        if ln.get("kind"): fa.kind = ln["kind"]
        fighters += [fv, fa]; wg = {"stick": STICK_GEOM, "axe": AXE_GEOM, "bow": BOW_GEOM}[ln.get("weapon", "stick")]; weapon_props += [dict(geom=STICK_GEOM), dict(geom=wg)]
        for a_, f_ in ((acts[vi], fv), (acts[ai], fa)): pass
    arena = CB.Arena(fighters)
    def mk(i):
        def cmd(t, w, ctx):
            ctx.state.setdefault("arena", arena); arena.tick(t, ctx); return arena.cmds[i]
        return cmd
    for i, a in enumerate(acts): a.cmd = mk(i)
    n_a = len(acts); props = weapon_props + [dict(geom=ARROW_GEOM)] * n_a + [dict(geom=SHIELD_GEOM)] * n_a
    def near(tab, i, k):
        for dk in (0, -1, 1, -2, 2, -3, 3):
            hd = tab.get((i, k + dk))
            if hd: return hd
        return None
    def prop_fn(t, ctx, sc, row):
        k = round(t * 60)
        for i in range(n_a):
            hd = near(arena.hist, i, k)
            if hd is None: hd = (np.array([0, 0, -5.0]), np.array([0, 0, 1.0]))
            sc.set_prop(i, hd[0], z_to_quat(hd[1]))
            ar = near(arena.arrow_hist, i, k)
            if ar is not None:
                # the arrow stays where it hit (stuck) once it has landed
                sc.set_prop(n_a + i, ar[0], z_to_quat(ar[1]))
            else: sc.set_prop(n_a + i, np.array([0, 0, -5.0]), z_to_quat(np.array([0, 0, 1.0])))
            sh = near(arena.shield_hist, i, k)
            if sh is not None: sc.set_prop(2 * n_a + i, sh[0], z_to_quat(sh[1]))
            else: sc.set_prop(2 * n_a + i, np.array([0, 0, -5.0]), z_to_quat(np.array([0, 0, 1.0])))
    def cap(t, ctx):
        a = ctx.state.get("arena"); return CB.caption_text(a, t) if a else ""
    return Shot(name, title, tr, acts, T, cam, props=props, prop_fn=prop_fn, size=size, caption_fn=cap)

@shot
def hit_reactions_a():
    lanes = [dict(region="head", **{"from": "front"}, strength=1.2, kind="chop"), dict(region="torso", **{"from": "side"}, strength=1.5, walk=(3.2, (1.0, 0))),
             dict(region="arm_R", **{"from": "front"}, strength=1.4, walk=(3.2, (1.0, 0)))]
    return _build(lanes, 9.0, "hit_reactions_a", "stick blows without a reaction: head (dazed, hand to the head), torso from the side (doubled over), right arm (drops the stick, clutches it)",
                  dict(dist=9.5, azimuth=90, elevation=-8, follow="all", look_z=0.9))

@shot
def hit_reactions_b():
    lanes = [dict(region="leg_L", **{"from": "side"}, strength=1.2, walk=(2.6, (1.2, 0))), dict(region="head", **{"from": "front"}, strength=2.5),
             dict(region="torso", **{"from": "behind"}, strength=2.8)]
    return _build(lanes, 16.0, "hit_reactions_b", "left leg (limp, leans off it), a hard head blow (knocked flat, lies, gets up), a hard blow in the back (falls onto hands and knees)",
                  dict(dist=9.5, azimuth=90, elevation=-8, follow="all", look_z=0.9))

@shot
def hit_reactions_c():
    lanes = [dict(region="groin", sex="m", strength=1.2, **{"from": "front"}, walk=(3.4, (1.0, 0))), dict(region="groin", sex="f", strength=1.2, **{"from": "front"}, walk=(3.4, (1.0, 0))),
             dict(region="knee_L", sex="m", strength=1.0, **{"from": "side"}, walk=(3.0, (1.2, 0))), dict(region="knee_R", sex="f", strength=1.0, **{"from": "side_r"}, walk=(3.0, (1.2, 0)))]
    return _build(lanes, 9.0, "hit_reactions_c", "the same blow to the groin hurts a man about three times more than a woman (doubled over, hands down, slow); a blow to the knee makes the leg give way and the walk a limp",
                  dict(dist=11.0, azimuth=90, elevation=-8, follow="all", look_z=0.9), spacing=3.0)

@shot
def weapon_reactions():
    lanes = [dict(region="torso", weapon="axe", kind="chop", shield=True, react=True, delay=0.2, strength=1.5, forced="shield"),
             dict(region="head", weapon="axe", kind="chop", react=False, strength=1.4),
             dict(region="torso", weapon="bow", range=6.0, react=True, delay=0.15, strength=1.6),
             dict(region="torso", weapon="bow", range=6.0, react=True, shield=True, delay=0.15, strength=1.6, forced="shield"),
             dict(region="torso", weapon="bow", range=6.0, react=False, strength=1.6),
             dict(region="leg_R", weapon="stick", react=True, leg_block=True, delay=0.2, strength=1.5, forced="block_leg")]
    return _build(lanes, 5.5, "weapon_reactions", "axe vs a shield, axe on an unguarded head, arrows (dodged, caught by a shield, taken), and a shin block of a low blow",
                  dict(dist=13.5, azimuth=90, elevation=-8, follow="all", look_z=0.9), spacing=3.0, size=(680, 340))

@shot
def dodge_reactions():
    lanes = [dict(region="head", kind="chop", delay=0.18, react=True, strength=1.6), dict(region="head", delay=0.18, react=True, strength=1.6),
             dict(region="leg_L", delay=0.18, react=True, strength=1.6), dict(region="torso", delay=0.26, react=True, strength=1.6), dict(region="head", kind="chop", delay=0.55, react=True, strength=1.6)]
    return _build(lanes, 4.6, "dodge_reactions", "the same kind of blow, different notice time and aim: the defender picks side step / duck / hop / block by predicting the geometry; too late = hit",
                  dict(dist=11.5, azimuth=90, elevation=-8, follow="all", look_z=0.9), spacing=2.9)

@shot
def duel():
    tr = Terrain(-8, 14, -8, 8); rng = np.random.default_rng(11)
    A = human(HumanBody(armour=8)); D = human(HumanBody(height=1.8, armour=10))
    acts = [ActorSpec(A, pos=(0, 0), heading=0.0, base=(0.6, 0.55, 0.5)), ActorSpec(D, pos=(3.2, 0), heading=math.pi, base=(0.45, 0.5, 0.6))]
    fa = CB.Fighter(0, 1, rng, attacks=True, react=True, delay=(0.16, 0.30), strength=(0.9, 2.2), cooldown=(1.1, 2.0), first=1.4)
    fd = CB.Fighter(1, 0, rng, attacks=True, react=True, delay=(0.18, 0.40), strength=(0.9, 2.0), cooldown=(1.3, 2.4), first=2.3)
    arena = CB.Arena([fa, fd])
    def mk(i):
        def cmd(t, w, ctx):
            ctx.state.setdefault("arena", arena); arena.tick(t, ctx); return arena.cmds[i]
        return cmd
    for i, a in enumerate(acts): a.cmd = mk(i)
    def prop_fn(t, ctx, sc, row):
        k = round(t * 60)
        for i in range(2):
            hd = next((arena.hist.get((i, k + dk)) for dk in (0, -1, 1, -2, 2, -3, 3) if arena.hist.get((i, k + dk))), None)
            if hd is None: hd = (np.array([0, 0, -5.0]), np.array([0, 0, 1.0]))
            sc.set_prop(i, hd[0], z_to_quat(hd[1]))
    cap = lambda t, ctx: CB.caption_text(ctx.state["arena"], t) if "arena" in ctx.state else ""
    return Shot("duel", "two fighters, no script: each picks targets, notices blows after its own delay and chooses to duck, step aside, hop, block or take the hit; pain and knock-downs follow", tr, acts, 28.0,
                dict(dist=6.5, azimuth=100, elevation=-10, follow="all"), props=[dict(geom=STICK_GEOM)] * 2, prop_fn=prop_fn, size=(560, 320), caption_fn=cap)
