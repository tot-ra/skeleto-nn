"""High-level behaviours expressed as commands to the planner (arm targets, pelvis path, torso twist).
They contain no animation data: only end-effector goals in space and time, and the planner's IK, balance
and footstep logic do the rest."""
from __future__ import annotations
import math
import numpy as np
from planner import Cmd, wrap, smooth
from render import z_to_quat

def minjerk(x):
    x = min(1.0, max(0.0, x)); return 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5

def body_frame(w):
    fwd = np.array([math.cos(w.heading), math.sin(w.heading), 0.0]); left = np.array([-fwd[1], fwd[0], 0.0])
    return fwd, left, np.array([0, 0, 1.0])

def shoulder(w, side):
    sk = w.sk
    i = sk.idx["uarm_" + side]
    return w.S[i].copy()

def arm_reach(w):
    return float(sum(w.sk.length[w.c.arms[0].chain]))

# ---- sword strike ---------------------------------------------------------------------------
STRIKE = dict(windup=0.40, strike=0.16, hold=0.05, recover=0.45)

def strike_phase(t, t0):
    """Returns (phase name, progress 0..1) of a strike that starts at t0."""
    a = t - t0
    if a < 0: return "guard", 0.0
    W, S, H, R = STRIKE["windup"], STRIKE["strike"], STRIKE["hold"], STRIKE["recover"]
    if a < W: return "windup", a / W
    if a < W + S: return "strike", (a - W) / S
    if a < W + S + H: return "hold", 1.0
    if a < W + S + H + R: return "recover", (a - W - S - H) / R
    return "guard", 0.0

def strike_hand(w, t, t0, target, side="R", kind="chop"):
    """World position of the sword hand and the progress of the swing."""
    fwd, left, up = body_frame(w)
    sgn = -1.0 if side == "R" else 1.0
    L = arm_reach(w)
    sh = shoulder(w, side)
    guard = sh + fwd * 0.52 * L + left * sgn * 0.12 * L + up * -0.10 * L
    if kind == "thrust":                      # spear and fist: pull back, then straight at the target
        wind = sh + fwd * 0.05 * L + left * sgn * 0.12 * L + up * -0.05 * L
        d = np.asarray(target, float) - sh; dn = np.linalg.norm(d); hit = sh + d / max(dn, 1e-6) * min(dn, 0.97 * L)
        ph, p = strike_phase(t, t0)
        if ph == "guard": return guard, ph, 0.0
        if ph == "windup": return guard + (wind - guard) * minjerk(p), ph, 0.0
        if ph == "strike": return wind + (hit - wind) * p ** 0.7, ph, p
        if ph == "hold": return hit, ph, 1.0
        return hit + (guard - hit) * minjerk(p), ph, 1.0
    if kind == "chop":
        wind = sh + fwd * 0.05 * L + left * sgn * 0.40 * L + up * 0.85 * L
    else:   # horizontal slash from the right side
        wind = sh + fwd * 0.10 * L + left * sgn * 0.85 * L + up * 0.05 * L
    d = np.asarray(target, float) - sh; d[2] = d[2]; dn = np.linalg.norm(d)
    hit = sh + d / max(dn, 1e-6) * min(dn, 0.92 * L)
    ph, p = strike_phase(t, t0)
    if ph == "guard": return guard, ph, 0.0
    if ph == "windup": return guard + (wind - guard) * minjerk(p), ph, 0.0
    if ph == "strike":
        # accelerate through the target, arc via a slightly lifted midpoint
        q = p ** 1.6
        mid = 0.5 * (wind + hit) + up * 0.12 * L
        pos = (1 - q) ** 2 * wind + 2 * (1 - q) * q * mid + q * q * hit
        return pos, ph, p
    if ph == "hold": return hit, ph, 1.0
    return hit + (guard - hit) * minjerk(p), ph, 1.0

def blade_dir(w, hand, ph, p, target, side="R"):
    fwd, left, up = body_frame(w)
    to_t = np.asarray(target, float) - hand; to_t = to_t / max(np.linalg.norm(to_t), 1e-6)
    carry = (fwd * 0.5 + up * 0.85); carry /= np.linalg.norm(carry)
    wind = (-fwd * 0.4 + up * 0.9); wind /= np.linalg.norm(wind)
    if ph == "guard": d = carry
    elif ph == "windup": d = carry * (1 - p) + wind * p
    elif ph == "strike": d = wind * (1 - p) + to_t * p
    elif ph == "hold": d = to_t
    else: d = to_t * (1 - p) + carry * p
    return d / np.linalg.norm(d)

# ---- shield block ---------------------------------------------------------------------------
def shield_pose(w, threat_pos, side="L", dist=0.52):
    """Hand position that puts the shield between the chest and the incoming weapon tip."""
    sk = w.sk
    chest = w.S[sk.idx["chest"]] + 0.5 * (w.E[sk.idx["chest"]] - w.S[sk.idx["chest"]])
    d = np.asarray(threat_pos, float) - chest; n = np.linalg.norm(d); d = d / max(n, 1e-6)
    fwd, left, up = body_frame(w)
    sgn = 1.0 if side == "L" else -1.0
    pos = chest + d * dist + left * sgn * 0.05 + up * -0.05
    return pos, d

# ---- sit and stand --------------------------------------------------------------------------
def sit_cmd(w, t, t0, chair_c, seat_h, facing, dur=1.5, stand=False, rest=0.0):
    """Pelvis path from standing to the seat (or back). The feet stay planted; the trunk leans forward so that
    the centre of mass stays over the feet, which is what makes the motion look like a person sitting."""
    tau = minjerk((t - t0) / dur)
    if stand: tau = 1.0 - tau
    f = np.array([math.cos(facing), math.sin(facing)])
    z0 = w.c.z0
    hip_rad = 0.10
    z = z0 * (1 - tau) + (seat_h + hip_rad) * tau
    # pelvis slides back over the seat as it goes down (feet stay)
    stand_xy = np.asarray(chair_c, float) + f * 0.40 * w.reach / 0.86
    seat_xy = np.asarray(chair_c, float) + f * 0.02
    xy = stand_xy * (1 - smooth(tau * 1.1)) + seat_xy * smooth(tau * 1.1)
    lean = (0.55 * math.sin(math.pi * min(1.0, tau * 1.25)) + 0.05) * (1.0 if not stand else 1.25)
    return dict(z=z, xy=xy, pitch=lean), tau

def knee_hands(w, side):
    """Hand targets on the thighs for a seated person."""
    sk = w.sk
    i = sk.idx["thigh_" + side]
    k = sk.idx["shank_" + side]
    return w.S[k] + np.array([0, 0, 0.07])

# ---- head hit -------------------------------------------------------------------------------
def hit(w, direction, strength=1.4):
    w.apply_push(direction, strength, chest=True)
