"""Bird landings: ground (run out), branch (grip with the toes) and water (skid and float). Kinematic model: the approach is solved
from the landing point, so speed, glide slope and flare follow from the distance and the height, not from a stored clip."""
from __future__ import annotations
import math
import numpy as np
from bodies import *
from planner import smooth
from shots import *
from skeleton import rot
from catalog import shot, SHOTS
import catalog_extra as CE

KIND = {   # touch-down speed (fraction of body lengths/s), flare pitch (rad), post-touch speed decay (s)
    "ground": dict(vtd=1.6, flare=0.95, decay=0.45),
    "branch": dict(vtd=0.35, flare=1.30, decay=0.12),
    "water": dict(vtd=2.2, flare=0.70, decay=0.70),
}

def landing_track(c, kind, T=6.5, dt=1 / 60, h_branch=1.5):
    k = KIND[kind]; s = c.z0 / 0.255
    Ta = 3.0; t_td = 1.2 + Ta
    D = 8.5 * s; H0 = 2.0 * s
    vtd = k["vtd"] * 0.6 * s * 1.6
    B = vtd * Ta; A = D - B
    zt = {"ground": c.z0, "branch": c.z0 + h_branch * s, "water": 0.01 + 0.55 * c.z0}[kind]
    zf = {"ground": c.z0, "branch": c.z0 + h_branch * s, "water": 0.01 + 0.40 * c.z0}[kind]
    xt = 0.0
    f_beat = 3.2 * (c.params["mass"] / 0.45) ** -0.28
    out = []; phase = 0.0; x_extra = 0.0
    for n in range(int(T / dt)):
        t = n * dt
        u = float(np.clip((t - 1.2) / Ta, 0.0, 1.0))
        tau = max(0.0, t - t_td)
        x = xt - A * (1 - u) ** 2 - B * (1 - u)
        if tau > 0: x += vtd * k["decay"] * (1 - math.exp(-tau / k["decay"]))
        z = zt + H0 * (1 - u) ** 2
        if tau > 0: z += (zf - zt) * smooth(tau / 0.8)
        vx = (2 * A * (1 - u) + B) / Ta if u < 1 else vtd * math.exp(-tau / k["decay"])
        flare = smooth((u - 0.45) / 0.5)
        phase += 2 * math.pi * f_beat * dt * (1 + 0.7 * flare) * (1.0 if t < t_td + 0.25 else 0.0)
        out.append(dict(t=t, x=x, z=z, u=u, tau=tau, flare=flare, phase=phase, vx=vx, kind=kind, zf=zf))
    return out

def landing_pose(c, r):
    sk = c.skel; q = sk.zeros(); kind = r["kind"]; k = KIND[kind]
    u, tau, flare = r["u"], r["tau"], r["flare"]
    air = 1.0 - smooth(tau / 0.35)                        # 1 while flying, 0 once settled
    ground = smooth(tau / 0.25)
    fold = smooth((tau - 0.35) / 0.7)
    pitch = -(math.radians(8) + (k["flare"] - math.radians(8)) * flare) * (1 - smooth((tau - 0.05) / 0.6))
    if kind == "water": pitch = -(math.radians(8) + (k["flare"] - math.radians(8)) * flare) * (1 - 0.7 * smooth((tau - 0.1) / 0.9))
    R = rot("Z", 0.0) @ rot("Y", pitch)
    flap_A = (0.7 + 0.55 * flare) * (1 - fold) if r["u"] < 1 or tau < 0.4 else 0.0
    ph = r["phase"]
    spread = 1.0 - fold
    for wi, (a, b, cc) in enumerate(c.wings):
        sg = 1.0 if wi == 0 else -1.0
        el = 0.5 + 0.5 * math.cos(ph + 0.5); wr = 0.5 + 0.5 * math.cos(ph + 1.2)
        q[a] = [sg * (flap_A * 0.95 * math.cos(ph) + 0.12) * spread + sg * 0.25 * fold, 0.0, sg * (0.18 * spread + 1.4 * fold)]
        q[b] = [0.0, 0.0, sg * (-(0.15 + 0.55 * el * flap_A) * spread - 2.55 * fold)]
        q[cc] = [0.0, 0.0, sg * ((0.1 + 0.5 * wr * flap_A) * spread + 2.2 * fold)]
    # legs: tucked in cruise, thrust out ahead in the flare, absorb at contact, then stand (or float)
    reach_leg = smooth((u - 0.55) / 0.35)
    tuck = (1 - reach_leg) * (1 - ground) * smooth(u / 0.1)
    absorb = math.sin(math.pi * min(1.0, tau / 0.5)) if tau < 0.5 else 0.0
    for l in c.legs:
        th, sh, me, to = l.chain
        a = np.array([0.7, 1.2, -0.9, 0.9]) * tuck + np.array([-0.8, 0.5, -0.2, -0.1 if kind != "water" else -0.4]) * reach_leg * (1 - ground)
        a = a + np.array([0.35, 0.8, -0.5, 0.0]) * absorb
        if kind == "branch": a[3] = a[3] * (1 - ground) + 1.25 * ground                    # toes close round the twig
        if kind == "water":                                                                   # legs fold under the floating body
            a = a * (1 - ground) + np.array([0.5, 1.3, -1.0, 0.3]) * ground
        q[th] = [0, a[0], 0]; q[sh] = [0, a[1], 0]; q[me] = [0, a[2], 0]; q[to] = [0, a[3], 0]
    nk = c.neck; pull = air * (1 - 0.6 * flare)
    q[nk[0], 1] = -0.5 * pull; q[nk[1], 1] = 0.2 * pull; q[nk[2], 1] = 0.8 * pull; q[nk[3], 1] = 0.5 * pull
    q[c.head, 1] = -pitch * 0.7 * air
    for tj in c.tail: q[tj, 1] = 0.35 * flare * air
    for kk, b in enumerate(sk.bones):
        for ax, nm in enumerate("XYZ"):
            if nm in b.lim:
                lo, hi = b.lim[nm]; q[kk, ax] = float(np.clip(q[kk, ax], math.radians(lo), math.radians(hi)))
    return sk.fk(np.array([r["x"], 0.0, r["z"]]), R, q)

def _land_shot(kind, species, title):
    tr = Terrain(-16, 8, -6, 6); c = bird(species); s = c.z0 / 0.255
    track = landing_track(c, kind); T = len(track) / 60
    def kin(t, ctx): return landing_pose(c, track[min(int(t * 60), len(track) - 1)])
    a = ActorSpec(c, kinematic=kin, base=(0.2, 0.2, 0.25))
    props = []
    if kind == "branch":
        hb = 1.5 * s
        props = [dict(geom=f'<geom type="capsule" fromto="0 -0.7 {hb} 0 0.7 {hb}" size="{0.018 * s}" rgba="0.42 0.28 0.17 1"/>'),
                 dict(geom=f'<geom type="capsule" fromto="0.5 0 {hb - 0.04} 0.5 0 0" size="{0.05 * s}" rgba="0.36 0.25 0.16 1"/>')]
    if kind == "water":
        props = [dict(geom='<geom type="box" size="5 3 0.012" pos="1.2 0 0" rgba="0.15 0.45 0.80 0.55"/>')]
    return Shot(f"bird_land_{kind}", title, tr, [a], T, dict(dist=3.4 * max(1.0, s * 0.8), azimuth=80, elevation=4, follow=0, look_z=None),
                props=props, prop_fn=lambda t, ctx, sc, row: None, size=(480, 300))

@shot
def bird_land_ground(): return _land_shot("ground", "crow", "bird landing on the ground: glide slope from the distance, flare, feet first, a few steps to stop")
@shot
def bird_land_branch(): return _land_shot("branch", "crow", "bird landing on a branch: steeper approach, near-stall flare, toes close round the twig, wings fold")
@shot
def bird_land_water(): return _land_shot("water", "duck", "duck landing on water: shallow approach, tail down, feet forward as skis, skid, then floats low")

# ---------------------------------------------------------------------------- squeezing past people (arms and shoulders take part)
import skills as SK
from planner import Cmd

def squeeze_cmd(i, goal, speed, rad=0.24, ws=0.46):
    """Passing through a gap between people: slow down in proportion to how tight it is, turn the shoulders sideways, put the hands up
    in front of the chest and steer towards the middle of the gap. No stored clip: the gap width decides everything."""
    goal = np.asarray(goal, float)
    def cmd(t, w, ctx):
        d = goal - w.pos; dist = np.linalg.norm(d)
        if dist < 0.3: return Cmd()
        fw = d / dist; lf = np.array([-fw[1], fw[0]])
        left_c, right_c = None, None
        for j, o in enumerate(ctx.walkers):
            if j == i: continue
            rel = o.pos - w.pos; f = rel @ fw; l = rel @ lf
            if -0.5 < f < 1.5 and abs(l) < 1.6:
                gap = abs(l) - rad
                if l >= 0 and (left_c is None or gap < left_c[0]): left_c = (gap, f)
                if l < 0 and (right_c is None or gap < right_c[0]): right_c = (gap, f)
        if left_c is not None and right_c is not None: free = left_c[0] + right_c[0]; centre = 0.5 * (left_c[0] - right_c[0])
        elif left_c is not None: free = 2 * left_c[0]; centre = left_c[0] - ws * 0.8
        elif right_c is not None: free = 2 * right_c[0]; centre = -(right_c[0] - ws * 0.8)
        else: free = 9.0; centre = 0.0
        prox = float(np.clip((ws * 1.5 - free) / (ws * 1.5 - ws * 0.8), 0.0, 1.0))
        turn = prox * 1.15 * (1.0 if (left_c is not None and (right_c is None or left_c[0] < right_c[0])) else -1.0)
        v = fw * speed * (1 - 0.6 * prox) + lf * float(np.clip(centre, -0.4, 0.4)) * 1.2 * prox
        arms = None
        if prox > 0.02 and w.S is not None:
            arms = {}
            for arm in w.c.arms:
                u = arm.chain[0]; sh = w.S[u]; sgn = 1.0 if arm.side == "L" else -1.0
                L = float(sum(w.sk.length[k] for k in arm.chain[:2]))
                natural = sh + np.array([0, 0, -0.95 * L]) + np.append(fw * 0.0, 0.0)
                near_neigh = (arm.side == "L") == (turn > 0)          # the arm on the side the chest turns to goes out a little more
                protect = sh + np.append(fw * (0.55 * L + 0.12 * L * near_neigh), -0.28 * L) + np.append(lf * sgn * (0.10 - 0.08 * near_neigh), 0.0)
                arms[arm.side] = natural * (1 - prox) + protect * prox
        return Cmd(v=v, heading=float(math.atan2(fw[1], fw[0])), torso_yaw=turn, arms=arms, crouch=0.0)
    return cmd

@shot
def crowd_squeeze():
    """A dense group with narrow gaps and two people who have to pass each other inside it."""
    tr = Terrain(); rng = np.random.default_rng(5); acts = []
    stand = [(4.0, -1.55), (4.0, -0.25), (4.1, 1.10), (5.3, -0.90), (5.3, 0.50), (5.4, 1.75), (6.6, -1.55), (6.6, -0.20), (6.7, 1.20), (3.2, 0.5), (7.7, -0.9), (7.7, 0.55)]
    for (x, y) in stand:
        c = human(HumanBody(height=1.75 * rng.uniform(0.92, 1.05), mass=75 * rng.uniform(0.85, 1.2)))
        a = ActorSpec(c, pos=(x, y), heading=float(rng.uniform(-math.pi, math.pi)), cmd=lambda t, w, ctx: Cmd())
        acts.append(a)
    n0 = len(acts)
    A = ActorSpec(human(HumanBody(height=1.78, mass=82)), pos=(0.0, -0.5), heading=0.0)
    B = ActorSpec(human(HumanBody(height=1.66, mass=64)), pos=(11.0, 0.0), heading=math.pi)
    acts += [A, B]
    A.cmd = squeeze_cmd(n0, (11.0, 0.0), 1.1); B.cmd = squeeze_cmd(n0 + 1, (0.0, -0.4), 1.0)
    return Shot("crowd_squeeze", "squeezing through a dense group: gap width sets speed, shoulder turn and hand position; two people pass each other inside it",
                tr, acts, 15.0, dict(dist=8.5, azimuth=95, elevation=-38, follow="all", look_z=0.3), size=(560, 340))

# ---------------------------------------------------------------------------- species: spine, neck and tail behave differently
def _species_shot(names, name, title, cam, spacing, size):
    tr = Terrain(-3, 40, -10, 10); acts = []; G = 9.81
    for i, n in enumerate(names):
        c = quadruped(n); y = (i - (len(names) - 1) / 2) * spacing
        vw = math.sqrt(0.2 * G * c.z0); vt = math.sqrt(1.0 * G * c.z0)
        def cmd(t, w, ctx, vw=vw, vt=vt):
            w.mood = 1.0 if t > 9.5 else 0.3
            if t < 3.0: return Cmd(v=np.array([vw, 0.0]), heading=0.0)
            if t < 6.0: return Cmd(v=np.array([vt, 0.0]), heading=0.0)
            if t < 8.5: return Cmd(v=np.array([vt * 0.5 * math.cos(1.2), vt * 0.5 * math.sin(1.2)]))
            return Cmd(heading=1.2)
        acts.append(ActorSpec(c, pos=(0.0, y), cmd=cmd, label=n))
    return Shot(name, title, tr, acts, 13.0, cam, size=size)

@shot
def quad_species_tails():
    """Same script for four small animals: walk, trot, turn, stop (mood up: wag, curl)."""
    return _species_shot(["cat", "dog", "wolf", "pig"], "quad_species_tails",
                         "cat (8 spine joints, balance tail), dog (wag), wolf (low stiff tail), pig (curl): the same script, different spine, neck and tail",
                         dict(dist=5.2, azimuth=75, elevation=-14, follow="all", look_z=0.35), 1.15, (640, 340))

@shot
def horse_gait_tail():
    """Horse alone: five neck joints nod with the gait, the hair tail flags at speed and flicks when it stands."""
    return _species_shot(["horse"], "horse_gait_tail", "horse: 5-joint neck nods and stretches with speed, stiff 3-joint trunk, hoofed feet, hair tail lifts at the trot and flicks flies at rest",
                         dict(dist=6.0, azimuth=90, elevation=-6, follow=0, look_z=1.0), 2.0, (560, 320))
