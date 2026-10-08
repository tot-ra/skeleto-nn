"""Being hit: pain per body region, clutching, limping, staggering and knock-downs for the planner's Walker.

apply_hit(walker, region, direction, strength) is the only entry point. Everything else follows from numbers:
 * the blow pushes the body (spring-damper trunk + velocity) and adds pain to the region (decays over seconds);
 * pain on an arm makes it unusable: the hand leaves its goal and clutches the injured place, a held weapon is dropped;
 * pain on a leg slows the body and makes it lean off that leg (limp);
 * pain on the torso doubles the body over and the hands go to the belly; on the head the body weaves and a hand goes to the head;
 * a hard enough blow, or too much pain, knocks the body down: goal sequence to the floor, lie, get up (injured limbs do not push).
"""
from __future__ import annotations
import math
from dataclasses import replace
import numpy as np
from skeleton import rot
from goals import Key, R_ypr, pose_from_keys, blend, slerp
from posing import Poser, foot_rot, ankle_for
from planner import smooth, wrap

REGIONS = ("head", "torso", "groin", "arm_L", "arm_R", "leg_L", "leg_R")
HIT_REGIONS = REGIONS + ("knee_L", "knee_R")          # everything a blow can be aimed at; knees add to the leg pain
# push: share of the blow that moves the body; pain: pain per m/s; knock: strength that floors the body; tau: seconds for the pain to fade
SPEC = {"head": dict(push=0.8, pain=0.55, knock=1.9, tau=6.0), "torso": dict(push=1.0, pain=0.45, knock=2.6, tau=5.0),
        "groin": dict(push=0.35, pain=1.30, knock=99.0, tau=9.0), "knee": dict(push=0.3, pain=1.0, knock=1.2, tau=9.0),
        "arm": dict(push=0.3, pain=0.80, knock=99.0, tau=7.0), "leg": dict(push=0.35, pain=0.80, knock=1.7, tau=8.0)}
def _spec(region): return SPEC[region.split("_")[0]]

def _fl(w):
    f = np.array([math.cos(w.heading), math.sin(w.heading)]); return f, np.array([-f[1], f[0]])

def init(w):
    w.pain = {r: 0.0 for r in REGIONS}; w.down = None; w.last_hit = None
    w.inj_roll = 0.0; w.inj_dip = 0.0; w.poser = None; w.hit_log = []

def arm_usable(w, side): return w.pain["arm_" + side] < 0.3

def region_segments(w, region):
    """Bone segments (start, end, radius) that a blow to the region can touch."""
    sk = w.sk; i = sk.idx; S, E = w.S, w.E
    if region == "head": return [(S[i["head"]], E[i["head"]] + np.array([0, 0, 0.02]), 0.13)]
    if region == "torso": return [(S[i["lumbar"]], E[i["chest"]], 0.20), (S[i["pelvis"]], E[i["pelvis"]], 0.18)]
    if region == "groin":
        Rp = w.R[i["pelvis"]]; c = S[i["pelvis"]] + Rp @ np.array([0.07, 0.0, -0.05]); return [(c, c + np.array([0, 0, 0.01]), 0.085)]
    kind, side = region.split("_")
    if kind == "knee": return [(S[i["shank_" + side]] - np.array([0, 0, 0.02]), S[i["shank_" + side]] + np.array([0, 0, 0.06]), 0.075)]
    if kind == "arm": return [(S[i["uarm_" + side]], E[i["uarm_" + side]], 0.07), (S[i["farm_" + side]], E[i["hand_" + side]], 0.065)]
    return [(S[i["thigh_" + side]], E[i["thigh_" + side]], 0.10), (S[i["shank_" + side]], E[i["shank_" + side]], 0.08)]

def region_point(w, region):
    segs = region_segments(w, region); a, b, r = segs[0] if region in ("head",) else segs[len(segs) // 2 if region.startswith("leg") else 0]
    if region.startswith("arm") or region.startswith("leg"):
        a, b, r = segs[1]
    return 0.5 * (a + b)

def apply_hit(w, region, direction, strength):
    """direction: world direction the blow pushes the body (from the attacker); strength ~ velocity change in m/s (0.5 light, 1.5 hard, 2.5 brutal)."""
    if getattr(w, "pain", None) is None: init(w)
    sp = _spec(region); real = region
    if region.startswith("knee"): real = "leg_" + region[-1]; strength = strength * 1.25      # the knee takes the leg's pain and gives way sooner
    d = np.array(direction[:2], float); d = d / (np.linalg.norm(d) + 1e-9)
    chest = region in ("head", "torso")
    if w.down is None:
        w.apply_push(np.array([d[0], d[1], 0.0]), strength * sp["push"], chest=chest)
        if region == "head":
            f, l = _fl(w); w.wob_v += 3.0 * strength * np.array([-float(d @ l), float(d @ f), 0.0])      # the head whips the trunk
    gain = sp["pain"]
    if region == "groin": gain *= 1.0 if w.c.params.get("sex", "m") == "m" else 0.35        # the same blow is felt about three times less by a woman
    w.pain[real] = min(1.0, w.pain[real] + strength * gain)
    w.last_hit = dict(region=real, t=w.t, dir=d.copy(), strength=strength)
    w.hit_log.append((w.t, region, float(strength)))
    total = sum(w.pain.values())
    if w.down is None and w.jumpplan is None and (strength >= sp["knock"] or total > 1.7):
        begin_down(w, d, region)

# ---- every-step effects -------------------------------------------------------------------------
def modify_cmd(w, cmd, dt):
    """Pain changes what the body is able to do: called by Walker.step before the pose is made."""
    if getattr(w, "pain", None) is None: init(w)
    for r in REGIONS: w.pain[r] *= math.exp(-dt / _spec(r)["tau"])
    for r, floor in getattr(w, "chronic", {}).items(): w.pain[r] = max(w.pain[r], floor)          # pain that does not go away
    if w.S is None or w.down is not None: return cmd
    p = w.pain; c = w.c; sk = w.sk
    f, l = _fl(w); f3 = np.append(f, 0.0); l3 = np.append(l, 0.0)
    cap = (1 - 0.6 * max(p["torso"], p["head"])) * (1 - 0.65 * max(p["leg_L"], p["leg_R"])) * (1 - 0.75 * p["groin"])
    v = np.asarray(cmd.v, float) * cap
    heading = cmd.heading
    if p["head"] > 0.15 and np.linalg.norm(v) > 0.1:
        base = heading if heading is not None else math.atan2(v[1], v[0])
        heading = base + 0.45 * p["head"] * math.sin(2 * math.pi * 0.6 * w.t)
    crouch = max(cmd.crouch, 0.40 * smooth(p["torso"] / 0.5), 0.10 * max(p["leg_L"], p["leg_R"]), 0.60 * smooth(p["groin"] / 0.4))
    arms = dict(cmd.arms or {})
    pel = w.S[sk.idx["pelvis"]]
    for arm in c.arms:
        side = arm.side; sg = 1.0 if side == "L" else -1.0
        u = arm.chain[0]; sh = w.S[u]; L = float(sum(sk.length[k] for k in arm.chain[:2]))
        goal = arms.get(side)
        hang = sh + np.array([0, 0, -0.95 * L])
        base = goal if goal is not None else hang
        wt = 0.0; clutch = base
        pa = p["arm_" + side]
        if pa > 0.1:        # the injured arm is held against the body, the hand cupping the hurt place
            wt = smooth((pa - 0.1) / 0.35)
            clutch = pel + np.array([0, 0, 0.30 * L]) + f3 * 0.20 * L + l3 * sg * 0.10 * L
        if p["groin"] > 0.1 and wt < 0.95:
            wt2 = smooth((p["groin"] - 0.1) / 0.3)
            if wt2 > wt: wt, clutch = wt2, pel + f3 * 0.13 * L + np.array([0, 0, -0.10 * L]) + l3 * sg * 0.04 * L
        if p["torso"] > 0.15 and wt < 0.9:
            wt2 = smooth((p["torso"] - 0.15) / 0.35)
            if wt2 > wt: wt, clutch = wt2, pel + np.array([0, 0, 0.22 * L]) + f3 * 0.28 * L + l3 * sg * 0.06 * L
        hl = w.last_hit
        if p["head"] > 0.25 and wt < 0.9 and hl is not None:
            hside = "L" if (hl["dir"] @ l) < 0 else "R"          # the hand on the side that was struck
            if side == hside:
                wt2 = smooth((p["head"] - 0.25) / 0.35)
                if wt2 > wt: wt, clutch = wt2, w.S[c.head] + f3 * 0.05 + l3 * sg * 0.11 + np.array([0, 0, 0.10])
        if wt > 0.0: arms[side] = base * (1 - wt) + clutch * wt
    # limp
    lt_roll = 0.0; lt_dip = 0.0
    for ls in w.legs:
        pl = p["leg_" + ls.leg.side]
        if pl > 0.1 and ls.stance:
            sg = 1.0 if ls.leg.side == "L" else -1.0
            lt_roll += 0.11 * pl * sg; lt_dip += 0.05 * w.reach * pl
    # an antalgic gait: less time on the sore leg, the other leg steps short
    for sd in "LR": w.leg_duty[sd] = 1.0 - 0.38 * min(1.0, 1.4 * p["leg_" + sd])
    for ls in w.legs:
        lean = w.leg_gait.get(ls.leg.name, w.leg_gait.get(ls.leg.side, {})).get("lean", 0.0)
        if lean and ls.stance: lt_roll += lean * (1.0 if ls.leg.side == "L" else -1.0)
    k = min(1.0, 10 * dt)
    w.inj_roll += (lt_roll - w.inj_roll) * k; w.inj_dip += (lt_dip - w.inj_dip) * k
    return replace(cmd, v=v, heading=heading, crouch=crouch, arms=arms or None)

# ---- knock-down ---------------------------------------------------------------------------------
def getup_keys(w, P, yaw, t0, injured_arm=None, injured_leg=None, hold=1.0, lie_hands=None):
    """Lie -> sit up -> roll over the feet -> squat -> stand, in a frame whose origin is the pelvis and +x points from head to feet."""
    z0 = w.c.z0; flat = foot_rot(0.0); Rz = rot("Z", yaw)
    R_lie = R_ypr(0.0, -math.pi / 2, 0.0)
    ys = 0.0 if injured_leg is None else (-0.08 if injured_leg == "L" else 0.08)        # pelvis shifts over the healthy leg
    def hands(h):
        h = {s: np.array(v, float) for s, v in h.items()}
        if injured_arm in h: h[injured_arm] = np.array([-0.10, 0.12 if injured_arm == "L" else -0.12, 0.30]) if False else np.array([0.05, 0.18 if injured_arm == "L" else -0.18, 0.28])
        return h
    foot = lambda x, y, z, R=flat: (np.array([x, y, z]), R)
    def K(t, pos, R, lum, ch, feet, hnd):
        pos = np.asarray(pos, float)
        return Key(t0 + t, P + Rz @ pos, Rz @ R, lum, ch, feet={s: (P + Rz @ c[0], Rz @ c[1]) for s, c in feet.items()},
                   hands={s: P + Rz @ v for s, v in hands(hnd).items()})
    P0 = np.array([0, 0, 0.14])
    lie_feet = {"L": foot(0.85, 0.10, 0.04, R_lie), "R": foot(0.85, -0.10, 0.04, R_lie)}
    lh = lie_hands or {"L": [-0.2, 0.28, 0.05], "R": [-0.2, -0.28, 0.05]}
    ks = [K(0.0, P0, R_lie, (0, 0, 0), (0, 0, 0), lie_feet, lh), K(hold, P0, R_lie, (0, 0, 0), (0, 0, 0), lie_feet, lh)]
    h = hold
    seq = [(1.2, P0, R_lie, (0, 0, 0), (0, 0, 0), {"L": foot(0.45, 0.12, 0.05), "R": foot(0.45, -0.12, 0.05)}, {"L": [-0.1, 0.30, 0.05], "R": [-0.1, -0.30, 0.05]}),
           (2.5, P0, R_ypr(0, -0.55, 0), (0, 0.2, 0), (0, 0.2, 0), {"L": foot(0.45, 0.12, 0.05), "R": foot(0.45, -0.12, 0.05)}, {"L": [0.30, 0.26, 0.45], "R": [0.30, -0.26, 0.45]}),
           (3.6, [0.28, ys, 0.36], R_ypr(0, 0.75, 0), (0, 0.25, 0), (0, 0.25, 0), {"L": foot(0.45, 0.12, 0.05), "R": foot(0.45, -0.12, 0.05)}, {"L": [0.62, 0.22, 0.06], "R": [0.62, -0.22, 0.06]}),
           (4.7, [0.40, ys, z0 * 0.78], R_ypr(0, 0.55, 0), (0, 0.2, 0), (0, 0.2, 0), {"L": foot(0.45, 0.12, 0.05), "R": foot(0.45, -0.12, 0.05)}, {"L": [0.65, 0.22, 0.30], "R": [0.65, -0.22, 0.30]}),
           (5.8, [0.42, ys, z0 * 0.99], R_ypr(0, 0.05, 0), (0, 0, 0), (0, 0, 0), {"L": foot(0.45, 0.10, 0.05), "R": foot(0.45, -0.10, 0.05)}, {"L": [0.42, 0.30, 0.55], "R": [0.42, -0.30, 0.55]}),
           (7.0, [0.42, 0.0, z0 * 0.99], R_ypr(0, 0.0, 0), (0, 0, 0), (0, 0, 0), {"L": foot(0.45, 0.10, 0.05), "R": foot(0.45, -0.10, 0.05)}, {"L": [0.42, 0.28, 0.35], "R": [0.42, -0.28, 0.35]})]
    for (t, pos, R, lum, ch, feet, hnd) in seq: ks.append(K(h + t, pos, R, lum, ch, feet, hnd))
    return ks

_PHYS = {}
def _fall_params(c=None):
    """The evolved fall reflex for this body: a body with a heavy belly has its own (the belly is what must not hit the ground)."""
    import json, os
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    name = "reflex_fall_pregnant.json" if (c is not None and c.params.get("pregnancy", 0) > 0.3) else "reflex_fall.json"
    p = os.path.join(here, name)
    return np.array(json.load(open(p))["params"]) if os.path.exists(p) else None

def begin_down(w, d, region):
    """Floor the body. With the evolved reflex available the fall itself is simulated on a physical body (muscle-like PD joints that
    yield on impact, head tucked); otherwise a goal sequence. Either way the get-up follows from the pose the body ends in."""
    params = _fall_params(w.c)
    if params is not None and w.kind == "biped":
        try:
            if _begin_down_physical(w, d, region, params): return
        except ImportError:
            pass
    _begin_down_kinematic(w, d, region)

def _key_from_state(w, S, E, R, q, t):
    sk = w.sk; i = sk.idx
    feet = {}
    for ls in w.legs:
        f = ls.leg.chain[-1]; feet[ls.leg.side] = (S[f] + R[f] @ w.mid_local[ls.idx], R[f].copy())
    hands = {arm.side: E[arm.chain[2]].copy() for arm in w.c.arms}
    return Key(t, S[0].copy(), R[0].copy(), q[i["lumbar"]].copy(), q[i["chest"]].copy(), q[i["neck"]].copy() if "neck" in i else (0, 0, 0), q[i["head"]].copy(), feet=feet, hands=hands)

def _begin_down_physical(w, d, region, params):
    import reflex as RF
    from physics import PhysChar
    key = id(w.c)
    if key not in _PHYS: _PHYS[key] = PhysChar(w.c, w.terrain)
    pc = _PHYS[key]
    if w.poser is None: w.poser = Poser(w.c)
    vel = np.array([w.v[0], w.v[1], w.vz]); ang = np.array([w.wob_v[0], w.wob_v[1], 0.0]) * 1.0
    res = RF.fall_from_state(w.c, pc, params, w.root.copy(), w.R_p.copy(), pc.q_to_vec(w.q), vel, ang)
    w.fall_report = res["parts"]; w.hit_log.append((w.t, "fall_injury", float(res["cost"])))
    traj = res["traj"]
    if res["upright"] and len(traj) < 30 * 2.5 or not traj: pass
    fps = 30.0; T0 = w.t; sk = w.sk; i = sk.idx
    S, E, R, q = traj[-1]
    lying = float(R[0][2, 0])                       # pelvis forward axis: up = on the back, down = on the face
    pain_max = max(w.pain.values()); hold = 0.8 + 2.4 * pain_max
    t_phys = T0 + len(traj) / fps
    inj_arm = "L" if w.pain["arm_L"] > 0.45 else "R" if w.pain["arm_R"] > 0.45 else None
    inj_leg = "L" if w.pain["leg_L"] > 0.45 else "R" if w.pain["leg_R"] > 0.45 else None
    k0 = _key_from_state(w, S, E, R, q, t_phys)
    if R[0][2, 2] > 0.85:           # still on its feet: the reflex absorbed the blow
        w.down = dict(keys=None, phys=traj, t0=T0, fps=fps, t_end=t_phys, resume_key=k0); return True
    f_mean = np.mean([E[ls.leg.chain[-1]] for ls in w.legs], axis=0)
    head = E[w.c.head]
    if lying > -0.3:                 # on the back or side: legs point away from the head
        v = f_mean[:2] - S[0][:2]; yaw = math.atan2(v[1], v[0])
        gk = getup_keys(w, np.append(S[0][:2], 0.0), yaw, t_phys + 0.6, injured_arm=inj_arm, injured_leg=inj_leg, hold=hold)
        ks = [k0] + gk
    else:                            # face down: onto hands and knees, then up
        v = head[:2] - S[0][:2]; yaw = math.atan2(v[1], v[0]); Rz = rot("Z", yaw)
        fc = np.mean([np.append(k0.feet[s][0][:2], 0.0) for s in "LR"], axis=0)
        O = fc - Rz @ np.array([0.45, 0, 0]); O[2] = 0.0
        gk = getup_keys(w, O, yaw, 0.0, injured_arm=inj_arm, injured_leg=inj_leg, hold=0.0)
        sel = [k for k in gk if k.t >= 3.6 - 1e-6]
        tb = t_phys + 0.9
        ks = [k0, Key(tb, sel[0].pos, sel[0].R, sel[0].lumbar, sel[0].chest, feet=sel[0].feet, hands=sel[0].hands),
              Key(tb + hold, sel[0].pos, sel[0].R, sel[0].lumbar, sel[0].chest, feet=sel[0].feet, hands=sel[0].hands)]
        for k in sel[1:]: ks.append(Key(tb + hold + (k.t - 3.6), k.pos, k.R, k.lumbar, k.chest, feet=k.feet, hands=k.hands))
    ks.sort(key=lambda k: k.t)
    w.down = dict(keys=ks, phys=traj, t0=T0, fps=fps, t_end=ks[-1].t)
    w.v[:] = 0.0
    return True

def _begin_down_kinematic(w, d, region):
    """Goal-sequence fall (fallback without the physics package): from the current pose to the ground and back up."""
    c = w.c; sk = w.sk
    if w.poser is None: w.poser = Poser(c)
    f, l = _fl(w); d = np.asarray(d[:2], float); d = d / (np.linalg.norm(d) + 1e-9)
    fd = float(d @ f); z0 = c.z0; P0 = w.pos.copy(); T0 = w.t
    inj_arm = "L" if w.pain["arm_L"] > 0.45 else "R" if w.pain["arm_R"] > 0.45 else None
    inj_leg = "L" if w.pain["leg_L"] > 0.45 else "R" if w.pain["leg_R"] > 0.45 else None
    feet = {ls.leg.side: (ls.planted.copy() if ls.stance else w._swing_pose(ls)[0], foot_rot(w.heading)) for ls in w.legs}
    cur_h = {arm.side: w.E[arm.chain[2]].copy() for arm in c.arms}
    pain_max = max(w.pain.values())
    hold = 0.8 + 2.6 * pain_max
    kneel = fd > 0.25 or region.startswith("leg")           # pushed forward or the leg gave way: onto the knees and hands
    d3 = np.array([d[0], d[1], 0.0])
    if kneel:
        yaw = math.atan2(d[1], d[0]) if fd > 0.25 else w.heading
        Rz = rot("Z", yaw); O = np.append(feet["L"][0][:2] * 0.5 + feet["R"][0][:2] * 0.5, 0.0) - Rz @ np.array([0.45, 0, 0])
        ks = [Key(T0, np.array([w.pos[0], w.pos[1], w.z]), R_ypr(w.heading, w.pitch, 0), feet=feet, hands=cur_h)]
        gk = getup_keys(w, O, yaw, T0 + 0.5 + hold, injured_arm=inj_arm, injured_leg=inj_leg, hold=0.0)
        # skip the lying part: start the sequence from the hands-and-knees key (3.6 s into the get-up) and hold there while it hurts
        sel = [k for k in gk if k.t - (T0 + 0.5 + hold) >= 3.6 - 1e-6]
        t_hold = T0 + 0.55
        sel0 = sel[0]
        ks.append(Key(t_hold, sel0.pos, sel0.R, sel0.lumbar, sel0.chest, feet=sel0.feet, hands=sel0.hands))
        ks.append(Key(t_hold + hold, sel0.pos, sel0.R, sel0.lumbar, sel0.chest, feet=sel0.feet, hands=sel0.hands))
        base_t = t_hold + hold
        for k in sel[1:]:
            ks.append(Key(base_t + (k.t - (T0 + 0.5 + hold) - 3.6), k.pos, k.R, k.lumbar, k.chest, feet=k.feet, hands=k.hands))
    else:
        yaw = math.atan2(-d[1], -d[0])                                     # feet point back towards the attacker
        Pl = np.append(P0 + d * 0.95, 0.0)
        if region == "head": lh = {"L": [-0.55, 0.14, 0.2], "R": [-0.55, -0.14, 0.2]}
        elif region == "torso": lh = {"L": [-0.15, 0.12, 0.26], "R": [-0.15, -0.12, 0.26]}
        else: lh = None
        if lh is not None:
            lh = {s: v for s, v in lh.items()}
        ks = [Key(T0, np.array([w.pos[0], w.pos[1], w.z]), R_ypr(w.heading, w.pitch, 0), feet=feet, hands=cur_h)]
        shoulder_up = lambda s: np.append(P0 + d * 0.35 + (l if s == "L" else -l) * 0.6, z0 * 1.05)
        ks.append(Key(T0 + 0.32, np.append(P0 + d * 0.18, z0 * 0.72), R_ypr(w.heading, -0.35, 0), (0, -0.1, 0), (0, -0.1, 0), feet=feet, hands={s: shoulder_up(s) for s in "LR"}))
        ks.append(Key(T0 + 0.68, np.append(P0 + d * 0.55, z0 * 0.38), R_ypr(yaw, -1.0, 0), (0, -0.1, 0), (0, -0.1, 0),
                      feet={s: (np.append(feet[s][0][:2], 0.12), feet[s][1]) for s in "LR"}, hands={s: np.append(P0 + d * 0.8 + (l if s == "L" else -l) * 0.5, 0.12) for s in "LR"}))
        gk = getup_keys(w, Pl, yaw, T0 + 1.05, injured_arm=inj_arm, injured_leg=inj_leg, hold=hold, lie_hands=lh)
        ks.extend(gk)
    ks.sort(key=lambda k: k.t)
    w.down = dict(keys=ks, t_end=ks[-1].t, kneel=kneel)
    w.v[:] = 0.0

def down_step(w, dt):
    sk = w.sk; dn = w.down; keys = dn["keys"]
    ph = dn.get("phys")
    if ph is not None and w.t - dn["t0"] < len(ph) / dn["fps"]:
        k = min(len(ph) - 1, int((w.t - dn["t0"]) * dn["fps"]))
        S, E, R, q = ph[k]; Rp = R[0]; pos = S[0]
    elif keys is None:
        S, E, R, q = ph[-1]; Rp = R[0]; pos = S[0]
    else:
        S, E, R, q = pose_from_keys(w.c, w.poser, keys, w.t)
        a, b, u = blend(keys, w.t)
        pos = a.pos * (1 - u) + b.pos * u; Rp = slerp(a.R, b.R, u)
    w.S, w.E, w.R, w.q = S, E, R, q
    w.root = np.asarray(pos, float).copy(); w.R_p = Rp; w.com = sk.com(S, E, R)
    w.pos = w.root[:2].copy(); w.speed = 0.0; w.v[:] = 0.0
    w.hit_flash = 0.0
    if w.t >= w.down["t_end"]:
        k = keys[-1] if keys else dn["resume_key"]
        heading = math.atan2(Rp[1, 0], Rp[0, 0])
        w.heading = heading; w.pitch = 0.0; w.roll = 0.0; w.crouch = 0.0; w.trunk_wobble[:] = 0; w.wob_v[:] = 0
        w.mode = "stand"; w.yaw_rate = 0.0
        for ls in w.legs:
            ctr = k.feet[ls.leg.side][0]; ls.planted = np.array([ctr[0], ctr[1], float(w.terrain.h(ctr[0], ctr[1]))]); ls.yaw = heading; ls.stance = True; ls.u = 0.5
        w.z = float(w.root[2]); w.z_f = w.z; w.flight = False
        w.down = None
    return w.snapshot()
