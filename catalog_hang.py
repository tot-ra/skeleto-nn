"""Hanging and pulling: pull-ups on a bar, a pipe and rings (legs swing to help), mounting a ledge (feet walk up the wall, elbows on the edge, a knee up),
and climbing a rope with hands and feet. Goals for hands, feet and pelvis with the shared IK (a goal script, labelled as such)."""
from __future__ import annotations
import math
import numpy as np
from bodies import *
from planner import Cmd, wrap
from shots import *
from catalog import shot
from posing import Poser, foot_rot, ankle_for
from goals import Key, pose_from_keys, R_ypr, minjerk, spine_split

def _hang_geometry(c):
    sk = c.skel; Larm = float(sum(sk.length[c.arms[0].chain]))
    return dict(Larm=Larm, shoulder=0.52 * (c.z0 / 0.935), hang=1.14 * (c.z0 / 0.935))

def _pull_keys(c, bar_z, bar_x, reps, kip=False, grip_w=0.24, t0=0.5, period=3.2, ring=False):
    """Hang, pull the chin over the bar, lower; legs bent behind and kicking a little on the pull."""
    g = _hang_geometry(c); keys = []; t = t0
    hands = lambda dx=0.0: {"L": np.array([bar_x + dx, grip_w, bar_z]), "R": np.array([bar_x + dx, -grip_w, bar_z])}
    def key(tt, z_off, x_off, pitch, knee, hip, kick=0.0):
        pel = np.array([bar_x + x_off, 0.0, bar_z - z_off])
        feet = {s: (np.array([bar_x + x_off - 0.10 - 0.30 * (knee - 0.2) + kick * (1 if s == "L" else -1) * 0.1, 0.10 * (1 if s == "L" else -1), pel[2] - 0.60 + 0.30 * knee - 0.35 * hip]), foot_rot(0.0, 0.7)) for s in "LR"}
        return Key(tt, pel, R_ypr(0.0, pitch, 0.0), (0, 0.0, 0), (0, 0.0, 0), feet=feet, hands=hands())
    keys.append(key(0.0, g["hang"], -0.05, 0.0, 0.2, 0.0))
    keys.append(key(t, g["hang"], -0.05, 0.0, 0.2, 0.0))
    for r in range(reps):
        k = kip and r >= 1
        keys.append(key(t + 0.25 * period, g["hang"] - 0.30, -0.12 - (0.10 if k else 0), -0.18 - (0.18 if k else 0), 0.6, 0.15 if k else 0.0, kick=1.0 if k else 0.0))
        keys.append(key(t + 0.5 * period, g["shoulder"] + 0.04, -0.20, -0.30 - (0.10 if k else 0), 0.7, 0.2, kick=0.5 if k else 0.0))                  # chin over the bar
        keys.append(key(t + 0.58 * period, g["shoulder"] + 0.04, -0.20, -0.30, 0.7, 0.2))
        keys.append(key(t + 0.85 * period, g["hang"] - 0.35, -0.10, -0.08, 0.35, 0.05))
        t += period
        keys.append(key(t, g["hang"], -0.05, 0.0, 0.2, 0.0))
    keys.append(key(t + 1.0, g["hang"], -0.05, 0.0, 0.2, 0.0))
    return keys, t + 1.0

def _kin_from_keys(c, keys):
    poser = Poser(c)
    def kin(t, ctx):
        S, E, R, q = pose_from_keys(c, poser, keys, t)
        return S, E, R
    return kin

def _bar_shot(name, title, bar_geom, bar_z, reps, kip, ring=False):
    c = human(); tr = Terrain(-3, 6, -4, 4)
    keys, T = _pull_keys(c, bar_z, 0.0, reps, kip=kip, ring=ring)
    a = ActorSpec(c, kinematic=_kin_from_keys(c, keys), base=(0.35, 0.5, 0.75))
    props = [dict(geom=bar_geom)]
    return Shot(name, title, tr, [a], T + 0.5, dict(dist=4.2, azimuth=60, elevation=-6, follow=0, look_z=1.5), props=props, prop_fn=lambda t, ctx, sc, row: sc.set_prop(0, [0, 0, 0]), size=(480, 360))

POST = lambda z: f'<geom type="capsule" fromto="0 -0.7 0 0 -0.7 {z}" size="0.03" rgba="0.4 0.4 0.45 1"/><geom type="capsule" fromto="0 0.7 0 0 0.7 {z}" size="0.03" rgba="0.4 0.4 0.45 1"/>'

@shot
def pullup_bar():
    z = 2.3
    return _bar_shot("pullup_bar", "pull-ups on a flat bar: hang, chin over the bar, lower; from the second repetition the legs swing to help (the goals are the hands on the bar and the chin; the elbows and the trunk follow)",
                     POST(z) + f'<geom type="capsule" fromto="0 -0.7 {z} 0 0.7 {z}" size="0.02" rgba="0.75 0.75 0.8 1"/>', z, 4, True)

@shot
def pullup_pipe():
    z = 2.3
    return _bar_shot("pullup_pipe", "pull-ups on a thick pipe: the same hang and pull with a wider, thicker grip",
                     POST(z) + f'<geom type="capsule" fromto="0 -0.7 {z} 0 0.7 {z}" size="0.05" rgba="0.55 0.5 0.45 1"/>', z, 3, False)

@shot
def pullup_rings():
    z = 2.3
    rings = "".join(f'<geom type="cylinder" size="0.12 0.01" pos="0 {s * 0.24} {z - 0.12}" euler="90 0 0" rgba="0.55 0.4 0.25 1"/><geom type="capsule" fromto="0 {s * 0.24} {z} 0 {s * 0.24} {z + 0.9}" size="0.006" rgba="0.8 0.8 0.8 1"/>' for s in (-1, 1))
    return _bar_shot("pullup_rings", "pull-ups on rings: the hands hold the rings, a little apart, and the body rises between them", rings, z + 0.0, 3, False, ring=True)

@shot
def mantle_ledge():
    """From a hang on a ledge to standing on it: the feet walk up the wall, the elbows come over the edge, a knee goes up, the body stands."""
    c = human(); tr = Terrain(-3, 6, -4, 4); H = 2.0; g = _hang_geometry(c); z0 = c.z0
    hands = lambda x=0.12, z=H: {"L": np.array([x, 0.22, z]), "R": np.array([x, -0.22, z])}
    wallfoot = lambda z, dz=0.0: {"L": (np.array([-0.06, 0.12, z + dz]), R_ypr(0.0, -1.35, 0)), "R": (np.array([-0.06, -0.12, z - dz]), R_ypr(0.0, -1.35, 0))}
    keys = [
        Key(0.0, np.array([-0.32, 0.0, H - g["hang"]]), R_ypr(0.0, -0.10, 0), feet=wallfoot(H - 1.55), hands=hands()),
        Key(1.0, np.array([-0.32, 0.0, H - g["hang"]]), R_ypr(0.0, -0.10, 0), feet=wallfoot(H - 1.55), hands=hands()),
        Key(2.0, np.array([-0.28, 0.0, H - 0.95]), R_ypr(0.0, -0.10, 0), feet=wallfoot(H - 1.25, 0.12), hands=hands()),
        Key(3.0, np.array([-0.26, 0.0, H - 0.72]), R_ypr(0.0, -0.05, 0), feet=wallfoot(H - 1.00, -0.12), hands=hands()),
        Key(4.0, np.array([-0.22, 0.0, H - 0.50]), R_ypr(0.0, 0.05, 0), feet=wallfoot(H - 0.80, 0.10), hands=hands()),
        Key(5.0, np.array([-0.10, 0.0, H - 0.22]), R_ypr(0.0, 0.45, 0), (0, 0.2, 0), (0, 0.3, 0), feet=wallfoot(H - 0.70), hands=hands(0.32, H + 0.02)),                 # chest over the edge, hands pushing down
        Key(6.0, np.array([0.10, 0.0, H + 0.10]), R_ypr(0.0, 0.85, 0), (0, 0.3, 0), (0, 0.4, 0), feet={"L": (np.array([-0.06, 0.12, H - 0.62]), R_ypr(0.0, -1.35, 0)), "R": (np.array([0.22, -0.16, H + 0.03]), foot_rot(0.0, 0.0))}, hands=hands(0.34, H + 0.02)),   # a knee onto the ledge
        Key(7.0, np.array([0.34, 0.0, H + 0.45]), R_ypr(0.0, 0.7, 0), (0, 0.3, 0), (0, 0.3, 0), feet={"L": (np.array([0.18, 0.12, H + 0.03]), foot_rot(0.0, 0.0)), "R": (np.array([0.30, -0.14, H + 0.03]), foot_rot(0.0, 0.0))}, hands={"L": np.array([0.40, 0.25, H + 0.02]), "R": np.array([0.40, -0.25, H + 0.02])}),
        Key(8.2, np.array([0.45, 0.0, H + z0 * 0.99]), R_ypr(0.0, 0.0, 0), feet={"L": (np.array([0.40, 0.10, H + 0.03]), foot_rot(0.0, 0.0)), "R": (np.array([0.42, -0.10, H + 0.03]), foot_rot(0.0, 0.0))}, hands={"L": np.array([0.55, 0.28, H + 0.55]), "R": np.array([0.55, -0.28, H + 0.55])}),
        Key(9.5, np.array([0.45, 0.0, H + z0 * 0.99]), R_ypr(0.0, 0.0, 0), feet={"L": (np.array([0.40, 0.10, H + 0.03]), foot_rot(0.0, 0.0)), "R": (np.array([0.42, -0.10, H + 0.03]), foot_rot(0.0, 0.0))}, hands={"L": np.array([0.45, 0.30, H + 0.4]), "R": np.array([0.45, -0.30, H + 0.4])}),
    ]
    props = [dict(geom=f'<geom type="box" size="0.1 1.5 {H / 2}" pos="-0.1 0 {H / 2}" rgba="0.6 0.55 0.5 1"/><geom type="box" size="1.2 1.5 0.05" pos="1.0 0 {H - 0.05}" rgba="0.62 0.57 0.52 1"/>')]
    a = ActorSpec(c, kinematic=_kin_from_keys(c, keys), base=(0.35, 0.5, 0.75))
    return Shot("mantle_ledge", "from a hang on a ledge to standing on it: feet walk up the wall, the elbows come over the edge, a knee goes up, then the body stands (hand, foot and pelvis goals)", tr, [a], 10.2,
                dict(dist=5.0, azimuth=50, elevation=-6, follow=0, look_z=H * 0.8), props=props, prop_fn=lambda t, ctx, sc, row: sc.set_prop(0, [0, 0, 0]), size=(480, 360))

@shot
def rope_climb():
    """Climbing a rope: pull the knees up, clamp the rope with the feet, stand up on it pushing with the legs while the hands reach higher, hand over hand."""
    c = human(); tr = Terrain(-3, 6, -4, 4); g = _hang_geometry(c); keys = []; zt = 5.0
    cyc = 3.0; z_h = 2.35; z_p = 1.20           # hand height and pelvis height at the start
    foot = lambda z: {"L": (np.array([0.05, 0.07, z]), R_ypr(0.0, -1.2, 0.5)), "R": (np.array([0.05, -0.07, z]), R_ypr(0.0, -1.2, -0.5))}
    hands = lambda zl, zr: {"L": np.array([0.0, 0.10, zl]), "R": np.array([0.0, -0.10, zr])}
    t = 0.0; zp = z_p; zh = z_h
    keys.append(Key(0.0, np.array([-0.22, 0.0, zp]), R_ypr(0.0, 0.05, 0), feet=foot(zp - 0.80), hands=hands(zh, zh - 0.1)))
    for k in range(4):
        # 1: pull the knees up (feet and pelvis ride up the rope, the arms bend)
        keys.append(Key(t + 0.9, np.array([-0.26, 0.0, zp + 0.18]), R_ypr(0.0, -0.20, 0), (0, 0.15, 0), (0, 0.15, 0), feet=foot(zp - 0.55 + 0.18), hands=hands(zh, zh - 0.1)))
        # 2: the legs push (feet clamp the rope), the body stands up on the rope, the right hand reaches
        keys.append(Key(t + 1.7, np.array([-0.22, 0.0, zp + 0.42]), R_ypr(0.0, 0.0, 0), feet=foot(zp - 0.45 + 0.30), hands=hands(zh, zh + 0.35)))
        # 3: the left hand reaches past it
        keys.append(Key(t + 2.4, np.array([-0.22, 0.0, zp + 0.45]), R_ypr(0.0, 0.0, 0), feet=foot(zp - 0.35 + 0.30), hands=hands(zh + 0.40, zh + 0.35)))
        zp += 0.45; zh += 0.40; t += cyc
        keys.append(Key(t, np.array([-0.22, 0.0, zp]), R_ypr(0.0, 0.05, 0), feet=foot(zp - 0.80 + 0.1), hands=hands(zh, zh - 0.05)))
    props = [dict(geom=f'<geom type="capsule" fromto="0 0 0 0 0 {zt + 1.5}" size="0.018" rgba="0.75 0.65 0.4 1"/><geom type="sphere" size="0.05" pos="0 0 {zt + 1.5}" rgba="0.4 0.4 0.4 1"/>')]
    a = ActorSpec(c, kinematic=_kin_from_keys(c, keys), base=(0.35, 0.5, 0.75))
    return Shot("rope_climb", "climbing a rope: knees up, feet clamp the rope, stand up on it with the hands reaching higher, hand over hand", tr, [a], t + 0.5,
                dict(dist=4.5, azimuth=60, elevation=-4, follow=0, look_z=None), props=props, prop_fn=lambda t, ctx, sc, row: sc.set_prop(0, [0, 0, 0]), size=(420, 420))
