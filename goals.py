"""Goal-space sequences: a motion is a list of end-effector and pelvis goals (positions and orientations in the
world), not joint-angle keyframes. Goals are blended with minimum jerk, and the Poser's IK finds the angles."""
from __future__ import annotations
import math
import numpy as np
from skeleton import rot, rot_axis
from posing import Poser, ankle_for

def minjerk(x):
    x = min(1.0, max(0.0, x)); return 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5

def rotvec(R):
    a = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if a < 1e-7: return np.zeros(3), 0.0
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(a)) if abs(math.sin(a)) > 1e-6 else np.array([0, 0, 1.0])
    return ax, a

def slerp(R0, R1, u):
    ax, a = rotvec(R0.T @ R1)
    return R0 @ rot_axis(ax, a * u) if a > 1e-7 else R0

def R_ypr(yaw=0.0, pitch=0.0, roll=0.0):
    return rot("Z", yaw) @ rot("Y", pitch) @ rot("X", roll)

class Key:
    def __init__(self, t, pos, R, lumbar=(0, 0, 0), chest=(0, 0, 0), neck=(0, 0, 0), head=(0, 0, 0), feet=None, hands=None, extra=None):
        self.t = t; self.pos = np.asarray(pos, float); self.R = R
        self.lumbar = np.asarray(lumbar, float); self.chest = np.asarray(chest, float); self.neck = np.asarray(neck, float); self.head = np.asarray(head, float)
        self.feet = feet or {}      # side -> (contact centre world, Rf)
        self.hands = hands or {}    # side -> world point
        self.extra = extra or {}    # bone name -> (x,y,z) angles

def blend(keys, t):
    if t <= keys[0].t: return keys[0], keys[0], 0.0
    for a, b in zip(keys[:-1], keys[1:]):
        if a.t <= t <= b.t:
            return a, b, minjerk((t - a.t) / max(b.t - a.t, 1e-6))
    return keys[-1], keys[-1], 0.0

def spine_split(creature, lumbar, chest):
    """One lumbar and one chest value for a goal pose become shares of the whole vertebral chain."""
    i = creature.skel.idx; out = {}
    lum = [i[n] for n in ("lumbar", "lumbar1", "lumbar2") if n in i]; chs = [i[n] for n in ("thorax", "chest") if n in i]
    for j in lum: out[j] = np.asarray(lumbar, float) / len(lum)
    for j in chs: out[j] = np.asarray(chest, float) / len(chs)
    return out

def pose_from_keys(creature, poser: Poser, keys, t, hand_pole=None):
    a, b, u = blend(keys, t)
    pos = a.pos * (1 - u) + b.pos * u
    R = slerp(a.R, b.R, u)
    sk = creature.skel; i = sk.idx
    sq = spine_split(creature, a.lumbar * (1 - u) + b.lumbar * u, a.chest * (1 - u) + b.chest * u)
    if "neck" in i: sq[i["neck"]] = a.neck * (1 - u) + b.neck * u
    sq[i["head"]] = a.head * (1 - u) + b.head * u
    feet = {}
    for s in set(a.feet) | set(b.feet):
        fa = a.feet.get(s, b.feet.get(s)); fb = b.feet.get(s, a.feet[s] if s in a.feet else None)
        ctr = fa[0] * (1 - u) + fb[0] * u; Rf = slerp(fa[1], fb[1], u)
        feet[s] = (ankle_for(creature, s, ctr, Rf), Rf)
    hands = {}
    for s in set(a.hands) | set(b.hands):
        ha = a.hands.get(s); hb = b.hands.get(s)
        if ha is None and hb is None: continue
        hands[s] = hb if ha is None else ha if hb is None else ha * (1 - u) + hb * u
    ex = {}
    for nm in set(a.extra) | set(b.extra):
        ea = np.asarray(a.extra.get(nm, b.extra.get(nm)), float); eb = np.asarray(b.extra.get(nm, a.extra.get(nm)), float)
        ex[i[nm]] = ea * (1 - u) + eb * u
    return poser.pose(pos, R, sq, feet or None, hands or None, hand_pole, extra_q=ex)
