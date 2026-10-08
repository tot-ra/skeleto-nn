"""More shots: bird flight, snake, rider, crowd, ladder. Imported by catalog.py."""
from __future__ import annotations
import math
import numpy as np
from bodies import *
from planner import Cmd, wrap, smooth
from shots import *
from nav import Navigator
from skeleton import rot, rot_axis, align
from posing import Poser, foot_rot, ankle_for
from catalog import shot, straight_cmd, nav_actor, goto
import skills as SK
from goals import spine_split

def sstep(x): return smooth(x)

# ---------------------------------------------------------------------------- bird flight (kinematic flapping model)
def bird_flight_track(c, T, dt=1 / 60):
    """Precomputed pose track of a take-off, climb, turn, descent and landing. Wing beat frequency scales with mass^-0.28."""
    sk = c.skel; i = sk.idx
    mass = c.params["mass"]; s = c.z0 / 0.255
    f_beat = 3.2 * (mass / 0.45) ** -0.28
    n = int(T / dt); out = []
    x = 0.0; y = 0.0; yaw = 0.0; phase = 0.0
    for k in range(n):
        t = k * dt
        up = sstep((t - 1.6) / 2.4)                 # climb
        down = sstep((t - 6.6) / 2.4)               # descent
        alt = 3.0 * s * (up - down)
        spd = 5.0 * np.sqrt(s) * sstep((t - 1.5) / 1.8) * (1 - 0.8 * sstep((t - 7.0) / 2.0))
        yaw_t = math.radians(70) * sstep((t - 4.2) / 2.8)
        yaw = yaw_t
        x += spd * math.cos(yaw) * dt; y += spd * math.sin(yaw) * dt
        vz = (3.0 * s * (sstep((t + dt - 1.6) / 2.4) - sstep((t + dt - 6.6) / 2.4)) - alt) / dt
        climb = float(np.clip(math.atan2(vz, max(spd, 0.5)), -0.6, 0.7))
        flapping = 1.0 if (1.4 < t < 5.2 or 6.0 < t < 9.5) else 0.0
        glide = sstep((t - 5.2) / 0.5) * (1 - sstep((t - 6.0) / 0.5))
        phase += 2 * math.pi * f_beat * dt * (1.4 if (1.4 < t < 2.4) else 1.0) * flapping
        out.append((t, x, y, alt, yaw, climb, spd, phase, glide))
    return out

def bird_pose(c, row, c_stand_z):
    sk = c.skel; i = sk.idx
    t, x, y, alt, yaw, climb, spd, phase, glide = row
    q = sk.zeros()
    airborne = sstep((t - 1.45) / 0.3) * (1 - sstep((t - 9.3) / 0.3))
    crouch = sstep((t - 0.9) / 0.5) * (1 - sstep((t - 1.5) / 0.15))
    landing = sstep((t - 8.2) / 1.0)
    z = c_stand_z + alt - 0.35 * c_stand_z * crouch
    # body attitude: nose up for the climb, flare at landing
    pitch = -(climb * 0.9 + math.radians(8)) * airborne - math.radians(45) * landing * (1 - sstep((t - 9.4) / 0.3)) + 0.25 * crouch
    bank = -0.5 * sstep((t - 4.2) / 1.0) * (1 - sstep((t - 6.8) / 1.0)) * airborne
    R = rot("Z", yaw) @ rot("Y", pitch) @ rot("X", bank)
    flap_A = (1.0 if t < 3.5 else 0.6) * (1 - glide) + 0.12 * glide
    flap_A = flap_A * (1.0 - landing * 0.2)
    ph = phase
    for wi, (a, b, cc) in enumerate(c.wings):
        sg = 1.0 if wi == 0 else -1.0
        fold = 1.0 - airborne
        spread_land = landing * (1 - sstep((t - 9.2) / 0.4))
        el = 0.5 + 0.5 * math.cos(ph + 0.5)
        wr = 0.5 + 0.5 * math.cos(ph + 1.2)
        q[a] = [sg * (flap_A * 0.95 * math.cos(ph) + 0.12) * airborne + sg * 0.25 * fold, 0.0, sg * (0.18 * airborne + 1.4 * fold)]
        q[b] = [0.0, 0.0, sg * (-(0.15 + 0.55 * el * flap_A) * airborne - 2.55 * fold)]
        q[cc] = [0.0, 0.0, sg * ((0.1 + 0.5 * wr * flap_A) * airborne + 2.2 * fold)]
    # legs: tuck in flight, reach forward for landing
    tuck = airborne * (1 - landing)
    for li, l in enumerate(c.legs):
        th, sh, me, to = l.chain
        q[th] = [0, 0.7 * tuck - 0.6 * landing * airborne, 0]; q[sh] = [0, 1.2 * tuck, 0]; q[me] = [0, -0.9 * tuck, 0]; q[to] = [0, 0.9 * tuck, 0]
    nk = c.neck
    pull = airborne * (1 - 0.6 * landing)
    q[nk[0], 1] = -0.5 * pull; q[nk[1], 1] = 0.2 * pull; q[nk[2], 1] = 0.8 * pull; q[nk[3], 1] = 0.5 * pull
    q[c.head, 1] = -pitch * 0.7 * airborne
    for tj in c.tail:
        q[tj, 1] = (0.25 * landing - 0.1) * airborne
    for k, b in enumerate(sk.bones):
        for ax, nm in enumerate("XYZ"):
            if nm in b.lim:
                lo, hi = b.lim[nm]; q[k, ax] = float(np.clip(q[k, ax], math.radians(lo), math.radians(hi)))
    S, E, Rw = sk.fk(np.array([x, y, z]), R, q)
    return S, E, Rw

@shot
def bird_flight():
    tr = Terrain(); c = bird("crow"); T = 11.0
    track = bird_flight_track(c, T)
    def kin(t, ctx):
        k = min(int(t * 60), len(track) - 1)
        return bird_pose(c, track[k], c.z0)
    a = ActorSpec(c, kinematic=kin, base=(0.2, 0.2, 0.25))
    def prop_fn(t, ctx, sc, row): pass
    return Shot("bird_flight", "bird: take-off, climb, banked turn, glide, descent, flare and landing (kinematic flapping model, beat rate from mass)", tr, [a], T,
                dict(dist=3.0, azimuth=35, elevation=6, follow=0, look_z=None), size=(480, 300))

# ---------------------------------------------------------------------------- snake
@shot
def snake_around():
    tr = Terrain(); c = snake(26, 1.8, 0.9)
    tr.add_box(3.0, 3.5, -3.0, 0.8, 0.6); tr.add_box(6.0, 6.5, -0.8, 3.0, 0.6); tr.add_box(9.0, 9.5, -3.0, 0.8, 0.6)
    nv = Navigator(tr, human(), radius=0.25, res=0.12); path = nv.plan(np.array([0.0, 0.0]), np.array([12.0, 0.0]))
    n = c.skel.n; seg = c.skel.length[0]
    # head position along the (resampled) path, speed 0.9 m/s
    pts = [np.array([-2.0, 0.0])] + [p for p in path]
    P = np.array(pts); d = np.linalg.norm(np.diff(P, axis=0), axis=1); s_cum = np.concatenate([[0], np.cumsum(d)])
    def at(s):
        s = float(np.clip(s, 0, s_cum[-1]))
        j = int(np.searchsorted(s_cum, s, side="right") - 1); j = min(j, len(P) - 2)
        u = (s - s_cum[j]) / max(d[j], 1e-6)
        return P[j] * (1 - u) + P[j + 1] * u, (P[j + 1] - P[j]) / max(d[j], 1e-6)
    def kin(t, ctx):
        s_head = 2.0 + 0.9 * t + 0.0
        lam = 0.55; amp = 0.07
        nodes = []
        for k in range(n + 1):
            s = s_head - (n - k) * seg
            p, tang = at(s)
            nrm = np.array([-tang[1], tang[0]])
            env = 0.35 + 0.65 * (k / n)
            off = amp * env * math.sin(2 * math.pi * (s / lam) - 2 * math.pi * 0.9 * t)
            nodes.append(np.array([p[0] + nrm[0] * off, p[1] + nrm[1] * off, c.z0]))
        nodes = np.array(nodes)
        S = nodes[:-1].copy(); E = nodes[1:].copy(); R = np.zeros((n, 3, 3))
        for k in range(n):
            dv = E[k] - S[k]; R[k] = rot("Z", math.atan2(dv[1], dv[0]))
        return S, E, R
    a = ActorSpec(c, kinematic=kin, base=(0.35, 0.45, 0.25))
    return Shot("snake_around", "snake: undulation travels along the body, body follows the head's A* path around walls", tr, [a], 16.0,
                dict(dist=4.2, azimuth=140, elevation=-55, follow=0, look_z=0.1), size=(480, 300))

# ---------------------------------------------------------------------------- rider
def rider_kin(P, poser):
    """Kinematic rider on the horse that is actor 0 of the shot: a damped spring on the saddle, legs and hands to stirrups and reins; over a fence it rises and folds."""
    def kin(t, ctx):
        w = ctx.walkers[0]; sk = w.sk
        if w.S is None: return None
        i = sk.idx
        Sp, Ep = w.S[i["spine"]], w.E[i["spine"]]
        jumping = w.jumpplan is not None
        saddle = 0.5 * (Sp + Ep) + np.array([0, 0, 0.30 * (w.c.z0 / 1.4) + 0.12 + (0.13 if jumping else 0.0)])
        rp = ctx.state.setdefault("rider", dict(off=np.zeros(3), v=np.zeros(3), pitch=0.0, pv=0.0, t=0.0))
        dt = 1 / 24; target = saddle
        if "pos" not in rp: rp["pos"] = target.copy(); rp["vel"] = np.zeros(3)
        acc = -90.0 * (rp["pos"] - target) - 16.0 * rp["vel"]
        rp["vel"] += acc * dt; rp["pos"] += rp["vel"] * dt
        pos = rp["pos"]; hy = w.heading; hp = w.pitch
        tgt_pitch = 0.25 * hp + 0.12 * min(1.0, w.speed / 5.0) + (0.55 if jumping else 0.0) + 0.5 * float(np.clip(w.yaw_rate * w.speed / 9.81, -0.3, 0.3)) * 0.0
        rp["pv"] += (-60.0 * (rp["pitch"] - tgt_pitch) - 12.0 * rp["pv"]) * dt; rp["pitch"] += rp["pv"] * dt
        roll = -0.9 * float(np.clip(w.yaw_rate * w.speed / 9.81, -0.35, 0.35))                     # leans with the horse into a turn
        Rp = rot("Z", hy) @ rot("Y", rp["pitch"]) @ rot("X", roll * 0.6)
        fwd = np.array([math.cos(hy), math.sin(hy), 0]); left = np.array([-fwd[1], fwd[0], 0])
        feet = {}
        for side, sg in (("L", 1.0), ("R", -1.0)):
            ctr = pos + left * sg * 0.30 * (w.c.z0 / 1.4) ** 0.3 - np.array([0, 0, 0.78 * P.leg_len() / 0.86 - (0.1 if jumping else 0.0)]) + fwd * 0.05
            Rf = foot_rot(hy, 0.35); feet[side] = (ankle_for(P, side, ctr, Rf), Rf)
        neck = w.S[i["neck"]] + 0.6 * (w.E[i["neck"]] - w.S[i["neck"]])
        hands = {"L": neck + left * 0.16 + np.array([0, 0, 0.12]), "R": neck - left * 0.16 + np.array([0, 0, 0.12])}
        spq = spine_split(P, (0, 0.12 - rp["pitch"] * 0.5, 0), (0, 0.10 - rp["pitch"] * 0.5, 0))
        S, E, R, q = poser.pose(pos, Rp, spq, feet, hands)
        return S, E, R
    return kin

@shot
def rider():
    tr = Terrain(); H = quadruped("horse"); P = human(HumanBody(mass=75))
    poser = Poser(P)
    sched = [(0, 0.0), (1.0, 1.3), (5.0, 3.0), (9.0, 5.5), (13.0, 1.3), (16.0, 0.0)]
    def horse_cmd(t, w, ctx):
        v = 0.0
        for ts, vs in sched:
            if t >= ts: v = vs
        return Cmd(v=np.array([v, 0.0]))
    st = dict(z=None, vz=0.0, x=0.0, vx=0.0, pitch=0.0, vp=0.0)
    def kin(t, ctx):
        w = ctx.walkers[0]; sk = w.sk
        if w.S is None: return None
        i = sk.idx
        # saddle point: just behind the middle of the trunk, on the back
        Sp, Ep = w.S[i["spine"]], w.E[i["spine"]]
        saddle = 0.5 * (Sp + Ep) + np.array([0, 0, 0.30 * (w.c.z0 / 1.4) + 0.12])
        rp = ctx.state.setdefault("rider", dict(off=np.zeros(3), v=np.zeros(3), pitch=0.0, pv=0.0, t=0.0))
        dt = 1 / 24
        # rider pelvis is a damped spring on the saddle: the horse's bounce reaches the rider late and softened
        target = saddle
        if "pos" not in rp: rp["pos"] = target.copy(); rp["vel"] = np.zeros(3)
        acc = -90.0 * (rp["pos"] - target) - 16.0 * rp["vel"]
        rp["vel"] += acc * dt; rp["pos"] += rp["vel"] * dt
        pos = rp["pos"]
        hy = w.heading; hp = w.pitch
        tgt_pitch = 0.25 * hp + 0.12 * min(1.0, w.speed / 5.0) + 0.0
        rp["pv"] += (-60.0 * (rp["pitch"] - tgt_pitch) - 12.0 * rp["pv"]) * dt; rp["pitch"] += rp["pv"] * dt
        Rp = rot("Z", hy) @ rot("Y", rp["pitch"])
        fwd = np.array([math.cos(hy), math.sin(hy), 0]); left = np.array([-fwd[1], fwd[0], 0])
        # feet in stirrups hang from the horse's barrel
        feet = {}
        for side, sg in (("L", 1.0), ("R", -1.0)):
            ctr = pos + left * sg * 0.30 * (w.c.z0 / 1.4) ** 0.3 - np.array([0, 0, 0.78 * P.leg_len() / 0.86]) + fwd * 0.05
            Rf = foot_rot(hy, 0.35)
            feet[side] = (ankle_for(P, side, ctr, Rf), Rf)
        # hands on the reins (neck of the horse)
        neck = w.S[i["neck"]] + 0.6 * (w.E[i["neck"]] - w.S[i["neck"]])
        hands = {"L": neck + left * 0.16 + np.array([0, 0, 0.12]), "R": neck - left * 0.16 + np.array([0, 0, 0.12])}
        hk = P.skel.idx
        spq = spine_split(P, (0, 0.12 - rp["pitch"] * 0.5, 0), (0, 0.10 - rp["pitch"] * 0.5, 0))
        S, E, R, q = poser.pose(pos, Rp, spq, feet, hands)
        return S, E, R
    h = ActorSpec(H, pos=(0, 0), cmd=horse_cmd, base=(0.45, 0.30, 0.2))
    r = ActorSpec(P, kinematic=kin, base=(0.30, 0.38, 0.55))
    def hook(t, ctx, sc, row): pass
    return Shot("rider", "horse with a rider: walk, trot, canter, gallop; the rider is a damped spring on the saddle, legs and hands IK to stirrups and reins", tr, [h, r], 17.0,
                dict(dist=8.0, azimuth=95, elevation=-6, follow=0, look_z=1.4), size=(520, 300))

# ---------------------------------------------------------------------------- crowd
@shot
def crowd():
    tr = Terrain(); rng = np.random.default_rng(3); acts = []; navs = []
    people = []
    for i in range(14):
        left_to_right = i % 2 == 0
        b = HumanBody(height=1.75 * rng.uniform(0.9, 1.08), mass=75 * rng.uniform(0.8, 1.3), belly=float(rng.choice([0, 0, 0, 12])))
        c = human(b)
        x0 = -2.0 if left_to_right else 12.0; y0 = rng.uniform(-2.2, 2.2)
        x1 = 12.0 if left_to_right else -2.0; y1 = rng.uniform(-1.5, 1.5)
        speed = rng.uniform(1.0, 1.5)
        acts.append(ActorSpec(c, pos=(x0 + rng.uniform(-0.5, 0.5), y0), heading=0.0 if left_to_right else math.pi,
                              base=tuple(rng.uniform(0.3, 0.8, 3)) if False else None))
        people.append((np.array([x1, y1]), speed))
    def mk(i):
        goal, speed = people[i]
        def cmd(t, w, ctx):
            others = [np.append(o.pos, 0.0) for j, o in enumerate(ctx.walkers) if j != i]
            d = goal - w.pos; dist = np.linalg.norm(d)
            if dist < 0.4: return Cmd()
            v = d / dist * speed
            for o in others:
                rel = w.pos - o[:2]; dd = np.linalg.norm(rel)
                if 1e-6 < dd < 1.5:
                    dirv = d / dist
                    side = np.array([-dirv[1], dirv[0]])
                    sgn = 1.0 if (rel @ side) >= 0 else -1.0
                    push = (1.5 - dd) / 1.5
                    v = v + (rel / dd) * speed * 0.9 * push + side * sgn * speed * 0.8 * push
            n = np.linalg.norm(v)
            if n > speed * 1.1: v = v / n * speed * 1.1
            return Cmd(v=v)
        return cmd
    for i, a in enumerate(acts): a.cmd = mk(i)
    return Shot("crowd", "14 people cross a plaza in two streams: steering around each other, same gait planner, no collisions", tr, acts, 14.0,
                dict(dist=12.0, azimuth=100, elevation=-35, follow="all", look_z=0.3), size=(520, 320))

# ---------------------------------------------------------------------------- ladder
@shot
def ladder_climb():
    c = human(); tr = Terrain(); poser = Poser(c)
    LX = 1.0; rung_dz = 0.30; n_r = 14
    props = [dict(geom='<geom type="capsule" fromto="0 -0.22 0 0 -0.22 4.4" size="0.025" rgba="0.5 0.35 0.2 1"/><geom type="capsule" fromto="0 0.22 0 0 0.22 4.4" size="0.025" rgba="0.5 0.35 0.2 1"/>'
                      + "".join(f'<geom type="capsule" fromto="0 -0.22 {0.3 + k * rung_dz:.2f} 0 0.22 {0.3 + k * rung_dz:.2f}" size="0.018" rgba="0.6 0.45 0.3 1"/>' for k in range(n_r)))]
    sk = c.skel; i = sk.idx
    v_climb = 0.28; cyc = 2.0
    z_body0 = 0.95
    def limb_target(t, phase_off, base_fn, jump):
        """Hold a rung, then move to the rung `jump` above: returns position along z during the swing (smooth)."""
        u = ((t / cyc) + phase_off) % 1.0
        idx = math.floor(t / cyc + phase_off)
        return u, idx
    def kin(t, ctx):
        t = max(t - 0.5, 0.0) if t > 0.5 else 0.0
        zb = z_body0 + v_climb * t
        hy = 0.0
        pos = np.array([LX - 0.36, 0.0, zb])
        Rp = rot("Z", hy) @ rot("Y", 0.06)
        feet = {}; hands = {}
        # limbs alternate: (hand L, foot R) then (hand R, foot L); each moves 2 rungs per cycle
        for side, ph in (("L", 0.0), ("R", 0.5)):
            for kind in ("hand", "foot"):
                opp = (side == "L") == (kind == "hand")           # hand L with foot R
                p0 = 0.0 if opp else 0.5
                u = ((t / cyc) - p0) % 1.0
                m = math.floor((t / cyc) - p0)
                swing = u < 0.30
                tau = u / 0.30
                if kind == "hand":
                    z_prev = z_body0 + 0.55 + rung_dz * 2 * m; z_next = z_prev + 2 * rung_dz
                else:
                    z_prev = z_body0 - 0.80 + rung_dz * 2 * m; z_next = z_prev + 2 * rung_dz
                # snap to rung heights
                z_prev = 0.3 + rung_dz * round((z_prev - 0.3) / rung_dz); z_next = 0.3 + rung_dz * round((z_next - 0.3) / rung_dz)
                z = z_prev if not swing else z_prev + (z_next - z_prev) * sstep(tau)
                x = LX - 0.03 - (0.10 * math.sin(math.pi * tau) if swing else 0.0) * (1 if kind == "hand" else 0.5)
                y = (0.17 if side == "L" else -0.17)
                if kind == "hand": hands[side] = np.array([x, y, z])
                else:
                    Rf = foot_rot(0.0, 0.0)
                    ball = np.array([x - 0.0 + 0.02, y * 0.7, z + 0.02])
                    leg = next(l for l in c.legs if l.side == side)
                    feet[side] = (ball - Rf @ leg.ball, Rf)
        spq = spine_split(c, (0, 0.05, 0), (0, 0.10, 0))
        S, E, R, q = poser.pose(pos, Rp, spq, feet, hands, poles={"L": np.array([-0.6, 0.8, -0.2]), "R": np.array([-0.6, -0.8, -0.2])})
        return S, E, R
    a = ActorSpec(c, kinematic=kin, base=(0.4, 0.45, 0.6))
    def prop_fn(t, ctx, sc, row): sc.set_prop(0, [LX, 0.0, 0.0])
    return Shot("ladder_climb", "ladder: four limbs alternate (hand L + foot R, then hand R + foot L), each reaching the next free rung", tr, [a], 14.0,
                dict(dist=4.5, azimuth=90, elevation=-6, follow=0, look_z=None), props=props, prop_fn=prop_fn, size=(400, 300))
