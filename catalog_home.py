"""Everyday situations: doors, tables, beds, getting up from the floor, swimming, diving. Imported by catalog.py."""
from __future__ import annotations
import math
import numpy as np
from bodies import *
from planner import Cmd, Walker, wrap, smooth
from shots import *
from nav import Navigator
from skeleton import rot, rot_axis
from posing import Poser, foot_rot, ankle_for
from goals import Key, pose_from_keys, R_ypr, minjerk, slerp
from render import z_to_quat, mat2quat
from catalog import shot, straight_cmd, nav_actor, goto
import skills as SK
from goals import spine_split

def chair_geom(seat_h, c=(0.5, 0.35, 0.2)):
    legs = "".join(f'<geom type="box" size="0.02 0.02 {seat_h/2}" pos="{sx*0.21} {sy*0.21} {seat_h/2-0.04}" rgba="0.4 0.28 0.16 1"/>' for sx in (-1, 1) for sy in (-1, 1))
    return (f'<geom type="box" size="0.24 0.24 0.02" pos="0 0 {seat_h-0.02}" rgba="0.5 0.35 0.2 1"/>' + legs +
            f'<geom type="box" size="0.02 0.24 0.25" pos="-0.22 0 {seat_h+0.24}" rgba="0.5 0.35 0.2 1"/>')

def table_geom(h=0.76):
    legs = "".join(f'<geom type="box" size="0.03 0.03 {h/2}" pos="{sx*0.52} {sy*0.38} {h/2}" rgba="0.42 0.3 0.18 1"/>' for sx in (-1, 1) for sy in (-1, 1))
    return f'<geom type="box" size="0.6 0.45 0.025" pos="0 0 {h-0.025}" rgba="0.55 0.4 0.25 1"/>' + legs + \
           f'<geom type="cylinder" size="0.045 0.05" pos="0.2 -0.15 {h+0.05}" rgba="0.7 0.7 0.75 1"/>'

# ---------------------------------------------------------------------------- door
@shot
def door_open():
    c = human(); tr = Terrain()
    DX = 4.0; hinge = np.array([DX, -0.5]); WD = 0.95
    tr.add_box(DX - 0.1, DX + 0.1, -4.0, -0.5, 2.3, rgba=(0.55, 0.5, 0.45, 1)); tr.add_box(DX - 0.1, DX + 0.1, 0.5, 4.0, 2.3, rgba=(0.55, 0.5, 0.45, 1))
    tr.add_beam(DX - 0.1, DX + 0.1, -0.5, 0.5, 2.05, 2.3, rgba=(0.55, 0.5, 0.45, 1))
    def handle(phi):
        u = np.array([math.sin(phi), math.cos(phi)]); n = np.array([-math.cos(phi), math.sin(phi)])
        p = hinge + u * 0.82 + n * 0.07
        return np.array([p[0], p[1], 1.0])
    stand = np.array([DX - 0.62, 0.22])
    nv = Navigator(tr, c); nv.plan(np.array([0.0, 0.0]), stand)
    PHI = math.radians(100)
    def cmd(t, w, ctx):
        s = ctx.state; ph = s.setdefault("ph", "walk"); s.setdefault("phi", {})
        phi = s.get("phi_now", 0.0)
        fwd = np.array([1.0, 0.0])
        out = Cmd(heading=0.0)
        if ph == "walk":
            out = nv.cmd(w.pos, 1.2)
            if np.linalg.norm(w.pos - stand) < 0.18 and w.speed < 0.2: s["ph"] = "reach"; s["t0"] = t
        elif ph == "reach":
            u = minjerk((t - s["t0"]) / 0.7)
            guard = w.S[w.sk.idx["uarm_L"]] + np.array([0.25, 0.0, -0.45])
            out = Cmd(heading=0.0, arms={"L": guard * (1 - u) + handle(0.0) * u})
            if t - s["t0"] > 0.9: s["ph"] = "push"; s["t0"] = t
        elif ph == "push":
            a = (t - s["t0"]) / 2.0
            phi = PHI * minjerk(a / 0.9) if a < 0.9 else PHI
            s["phi_now"] = phi
            shoulder = w.S[w.sk.idx["uarm_L"]]
            hp = handle(phi)
            d_ = np.linalg.norm(hp - shoulder)
            hand = hp
            if a > 0.5:                      # release: the hand falls back to the side while the door keeps swinging
                k = minjerk((a - 0.5) / 0.35)
                hand = hp * (1 - k) + (shoulder + np.array([0.15, 0.1, -0.5])) * k
            v = np.array([0.8, 0.0]) * min(1.0, max(0.0, (a - 0.15) / 0.3))
            out = Cmd(v=v, heading=0.0, arms={"L": hand}, torso_yaw=0.15 * (1 - minjerk(a)))
            if t - s["t0"] > 2.0: s["ph"] = "through"; s["nv"] = Navigator(tr, c); s["nv"].plan(w.pos, np.array([8.0, 0.0]))
        else:
            out = s["nv"].cmd(w.pos, 1.3)
        s.setdefault("hist", {})[round(t * 60)] = s.get("phi_now", 0.0)
        return out
    props = [dict(geom='<geom type="box" size="0.025 0.475 1.0" pos="0 0 0" rgba="0.5 0.33 0.18 1"/><geom type="sphere" size="0.035" pos="-0.06 0.33 0.0" rgba="0.8 0.7 0.3 1"/>')]
    def prop_fn(t, ctx, sc, row):
        phi = ctx.state.get("hist", {}).get(round(t * 60), 0.0)
        u = np.array([math.sin(phi), math.cos(phi)])
        ctr = hinge + u * 0.475
        R = rot("Z", -phi)
        sc.set_prop(0, [ctr[0], ctr[1], 1.0], mat2quat(R))
    return Shot("door_open", "open a door: reach to the handle, push while stepping through, release; the door angle follows the hand", tr,
                [ActorSpec(c, pos=(0.0, 0.0), cmd=cmd)], 13.0, dict(dist=5.5, azimuth=50, elevation=-12, look_z=1.0), props=props, prop_fn=prop_fn, size=(480, 300))

# ---------------------------------------------------------------------------- sit at a table
@shot
def sit_table():
    """Sit at a table with real obstacles: the chair is tucked under the table, the person pulls it out by the back, goes round it, sits,
    scoots in, eats, scoots out, stands and leaves. The table and the chair block the body (A* goes round them), the hands grip the chair back."""
    c = human(); tr = Terrain(-2, 12, -4, 4)
    T = np.array([5.0, 0.0]); chair_in, chair_out = 4.55, 3.55
    tr.add_hidden_ceiling(T[0] - 0.6, T[0] + 0.6, -0.45, 0.45, 0.72)                  # nobody walks under the tabletop
    for sx in (-1, 1):
        for sy in (-1, 1): tr.add_block(T[0] + sx * 0.52 - 0.04, T[0] + sx * 0.52 + 0.04, sy * 0.38 - 0.04, sy * 0.38 + 0.04, 0.76)
    block = [None]
    def put_chair(x):
        if block[0] is not None: tr.clear_block(*block[0])
        block[0] = (x - 0.26, x + 0.26, -0.26, 0.26); tr.add_block(*block[0], 0.5)
    put_chair(chair_in)
    props = [dict(geom=chair_geom(0.46)), dict(geom=table_geom())]
    seat_h = 0.46; top = 0.78
    nv0 = Navigator(tr, c, radius=0.22); nv0.plan(np.array([0.0, 1.6]), np.array([chair_in - 0.22 - 0.37, 0.0]))
    hist = {}
    def cmd(t, w, ctx):
        s = ctx.state; ph = s.setdefault("ph", "walk"); cx = s.setdefault("chair_x", chair_in); hist[round(t * 60)] = cx
        w.auto_duck = False
        if ph == "walk":
            cm = nv0.cmd(w.pos, 1.2)
            if np.linalg.norm(w.pos - np.array([chair_in - 0.22 - 0.37, 0.0])) < 0.2 and w.speed < 0.15: s["ph"] = "turn1"; s["t"] = t
            return cm
        grip = lambda x: {"L": np.array([x - 0.22, 0.13, 0.93]), "R": np.array([x - 0.22, -0.13, 0.93])}
        if ph == "turn1":
            if abs(wrap(0.0 - w.heading)) < 0.08 and t - s["t"] > 0.6 and all(l.stance for l in w.legs): s["ph"] = "grab"; s["t"] = t
            return Cmd(heading=0.0)
        if ph == "grab":
            u = minjerk((t - s["t"]) / 1.0); sh = {sd: w.S[w.sk.idx["uarm_" + sd]] for sd in "LR"}
            g = grip(cx); arms = {sd: (sh[sd] + np.array([0.15, 0, -0.45])) * (1 - u) + g[sd] * u for sd in "LR"}
            if t - s["t"] > 1.0: s["ph"] = "pull"; s["t"] = t
            return Cmd(heading=0.0, arms=arms, crouch=0.1)
        if ph == "pull":
            s["chair_x"] = float(w.pos[0] + 0.59)                                             # the chair follows the hands
            if s["chair_x"] <= chair_out: s["chair_x"] = chair_out; s["ph"] = "release"; s["t"] = t
            put_chair(s["chair_x"]); return Cmd(v=np.array([-0.75, 0.0]), heading=0.0, arms=grip(s["chair_x"]), crouch=0.1)
        if ph == "release":
            if "nv" not in s:
                s["nv"] = Navigator(tr, c, radius=0.2); s["nv"].plan(w.pos.copy(), np.array([chair_out + 0.40, 0.0]))
            cm = s["nv"].cmd(w.pos, 1.0)
            if np.linalg.norm(w.pos - np.array([chair_out + 0.40, 0.0])) < 0.2 and w.speed < 0.15: s["ph"] = "turn"; s["t"] = t
            return cm
        if ph == "turn":
            if abs(wrap(0.0 - w.heading)) < 0.08 and t - s["t"] > 0.8 and all(l.stance for l in w.legs): s["ph"] = "sit"; s["t"] = t
            return Cmd(heading=0.0)
        if ph in ("sit", "scoot", "seated", "unscoot", "rise"):
            if ph == "sit":
                pel, tau = SK.sit_cmd(w, t, s["t"], (cx, 0.0), seat_h, 0.0, dur=1.7)
                if tau >= 1.0: s["ph"] = "scoot"; s["t"] = t
            elif ph == "scoot":
                u = minjerk((t - s["t"]) / 1.0); s["chair_x"] = chair_out + 0.40 * u
                pel, tau = SK.sit_cmd(w, 9e9, 0.0, (s["chair_x"], 0.0), seat_h, 0.0)
                if t - s["t"] > 1.0: s["ph"] = "seated"; s["t"] = t
            elif ph == "seated":
                pel, tau = SK.sit_cmd(w, 9e9, 0.0, (cx, 0.0), seat_h, 0.0)
                if t - s["t"] > 7.0: s["ph"] = "unscoot"; s["t"] = t
            elif ph == "unscoot":
                u = minjerk((t - s["t"]) / 1.0); s["chair_x"] = (chair_out + 0.40) - 0.40 * u
                pel, tau = SK.sit_cmd(w, 9e9, 0.0, (s["chair_x"], 0.0), seat_h, 0.0)
                if t - s["t"] > 1.0: s["ph"] = "rise"; s["t"] = t; put_chair(chair_out)
            else:
                pel, tau = SK.sit_cmd(w, t, s["t"], (cx, 0.0), seat_h, 0.0, stand=True, dur=1.5)
                if t - s["t"] > 1.5:
                    s["ph"] = "leave"; s["nv"] = Navigator(tr, c, radius=0.2); s["nv"].plan(w.pos.copy(), np.array([0.5, 1.8]))
            if ph in ("sit", "scoot", "seated"): put_chair(s["chair_x"]) if ph != "sit" else None
            hl = np.array([T[0] - 0.55, 0.20, top + 0.05]); hr_t = np.array([T[0] - 0.50, -0.15, top + 0.09])
            head = w.S[w.sk.idx["head"]] + (w.E[w.sk.idx["head"]] - w.S[w.sk.idx["head"]]) * 0.4; mouth = head + np.array([0.12, 0.0, -0.06])
            cyc = (t - s.get("t", 0.0)) % 3.6 if ph == "seated" else 0.0
            k = minjerk(min(cyc, 1.2) / 1.2) - minjerk(max(0.0, cyc - 2.2) / 1.2) if ph == "seated" and (t - s["t"]) > 0.8 else 0.0
            hr = hr_t * (1 - k) + mouth * k
            arms = {"L": hl, "R": hr} if ph in ("scoot", "seated", "unscoot") else ({"L": SK.knee_hands(w, "L"), "R": SK.knee_hands(w, "R")} if ph == "sit" else None)
            return Cmd(heading=0.0, pelvis=pel, arms=arms)
        return s["nv"].cmd(w.pos, 1.2)
    def prop_fn(t, ctx, sc, row):
        k = round(t * 60); cx = hist.get(k) or hist.get(k - 1) or chair_in
        sc.set_prop(0, [cx, 0.0, 0.0]); sc.set_prop(1, [T[0], 0.0, 0.0])
    return Shot("sit_table", "sit at a table with real obstacles: pull the chair out by its back, go round it, sit, scoot in, eat, scoot out, stand and leave (the table and chair block the body)", tr,
                [ActorSpec(c, pos=(0.0, 1.6), heading=0.0, cmd=cmd)], 34.0, dict(dist=5.0, azimuth=70, elevation=-14, follow=0, look_z=0.8, look_off=(1.2, 0, 0)), props=props, prop_fn=prop_fn, size=(520, 320))

# ---------------------------------------------------------------------------- bed: lie down, lie, get up
@shot
def bed_lie_rise():
    c = human(); tr = Terrain(); poser = Poser(c)
    bed_top = 0.50; seat_c = np.array([0.35, 0.33]); facing = math.pi / 2
    props = [dict(geom='<geom type="box" size="1.05 0.5 0.18" pos="0 0 0.32" rgba="0.45 0.3 0.2 1"/><geom type="box" size="1.0 0.46 0.07" pos="0 0 0.43" rgba="0.82 0.8 0.74 1"/>'
                       '<geom type="box" size="0.22 0.3 0.06" pos="-0.78 0 0.55" rgba="0.95 0.95 0.92 1"/><geom type="box" size="0.8 0.45 0.04" pos="0.28 0 0.55" rgba="0.38 0.45 0.62 1"/>')]
    nv = Navigator(tr, c); stand_pt = seat_c + np.array([0.0, 0.40])
    nv.plan(np.array([0.35, 4.0]), stand_pt)
    st = {}
    R_lie = R_ypr(0.0, -math.pi / 2, 0.0)
    def kin(t, ctx):
        s = ctx.state
        w = s.get("w")
        if w is None:
            w = Walker(c, tr, pos=(0.35, 4.0), heading=-math.pi / 2, dt=1 / 60); s["w"] = w; s["ph"] = "walk"
        ph = s["ph"]
        def snap(): return w.S, w.E, w.R
        if ph == "walk":
            cm = nv.cmd(w.pos, 1.2); w.step(cm)
            if np.linalg.norm(w.pos - stand_pt) < 0.18 and w.speed < 0.15: s["ph"] = "turn"; s["t"] = t
            return snap()
        if ph == "turn":
            w.step(Cmd(heading=facing))
            if abs(wrap(facing - w.heading)) < 0.08 and t - s["t"] > 0.8 and all(l.stance for l in w.legs): s["ph"] = "sit"; s["t"] = t
            return snap()
        if ph == "sit":
            pel, tau = SK.sit_cmd(w, t, s["t"], tuple(seat_c), bed_top, facing, dur=1.5)
            w.step(Cmd(heading=facing, pelvis=pel, arms={"L": SK.knee_hands(w, "L"), "R": SK.knee_hands(w, "R")}))
            if tau >= 1.0:
                s["ph"] = "lie"; s["t"] = t
                # goal keys start from the pose reached
                f = np.array([math.cos(facing), math.sin(facing), 0.0])
                feet = {ls.leg.side: (ls.planted.copy(), foot_rot(facing)) for ls in w.legs}
                T0 = t
                pel_end = np.array([0.18, 0.0, bed_top + 0.14])
                feet_end = {"L": (np.array([1.0, 0.10, bed_top + 0.10]), R_lie), "R": (np.array([1.0, -0.10, bed_top + 0.10]), R_lie)}
                hands_end = {"L": np.array([0.0, 0.17, bed_top + 0.24]), "R": np.array([0.0, -0.17, bed_top + 0.24])}
                knee = lambda side: SK.knee_hands(w, side)
                keys = [
                    Key(T0, w.root.copy(), w.R_p.copy(), (0, 0.1, 0), (0, 0.05, 0), feet=feet, hands={"L": knee("L"), "R": knee("R")}),
                    Key(T0 + 1.0, [0.35, 0.28, bed_top + 0.17], R_ypr(facing, -0.55, 0), (0, -0.2, 0), (0, -0.2, 0), feet=feet, hands={"L": np.array([0.55, 0.0, bed_top + 0.09]), "R": np.array([0.15, 0.0, bed_top + 0.09])}),
                    Key(T0 + 2.4, [0.28, 0.12, bed_top + 0.19], R_ypr(facing * 0.5, -1.05, 0), (0, -0.1, 0), (0, -0.1, 0),
                        feet={"L": (np.array([0.55, 0.45, bed_top + 0.35]), R_ypr(0.9, 0, 0)), "R": (np.array([0.55, 0.25, bed_top + 0.40]), R_ypr(0.9, 0, 0))},
                        hands={"L": np.array([0.2, 0.28, bed_top + 0.09]), "R": np.array([0.1, -0.2, bed_top + 0.09])}),
                    Key(T0 + 4.0, pel_end, R_lie, (0, 0, 0), (0, 0, 0), feet=feet_end, hands=hands_end),
                    Key(T0 + 7.5, pel_end + np.array([0, 0, 0.005]), R_lie, (0, 0, 0), (0, 0, 0), feet=feet_end, hands=hands_end),
                    Key(T0 + 9.0, [0.28, 0.12, bed_top + 0.19], R_ypr(facing * 0.5, -1.05, 0), (0, -0.1, 0), (0, -0.1, 0),
                        feet={"L": (np.array([0.55, 0.45, bed_top + 0.35]), R_ypr(0.9, 0, 0)), "R": (np.array([0.55, 0.25, bed_top + 0.40]), R_ypr(0.9, 0, 0))},
                        hands={"L": np.array([0.2, 0.28, bed_top + 0.09]), "R": np.array([0.1, -0.2, bed_top + 0.09])}),
                    Key(T0 + 10.2, [0.35, 0.28, bed_top + 0.17], R_ypr(facing, -0.55, 0), (0, -0.2, 0), (0, -0.2, 0), feet=feet, hands={"L": np.array([0.55, 0.0, bed_top + 0.09]), "R": np.array([0.15, 0.0, bed_top + 0.09])}),
                    Key(T0 + 11.2, w.root.copy(), w.R_p.copy(), (0, 0.1, 0), (0, 0.05, 0), feet=feet, hands={"L": knee("L"), "R": knee("R")}),
                ]
                s["keys"] = keys; s["t_end"] = T0 + 11.2
            return snap()
        if ph == "lie":
            S, E, R, q = pose_from_keys(c, poser, s["keys"], t)
            if t >= s["t_end"]: s["ph"] = "rise"; s["t"] = t
            return S, E, R
        if ph == "rise":
            pel, tau = SK.sit_cmd(w, t, s["t"], tuple(seat_c), bed_top, facing, stand=True, dur=1.4)
            w.step(Cmd(heading=facing, pelvis=pel))
            if t - s["t"] > 1.4: s["ph"] = "leave"; s["nv"] = Navigator(tr, c); s["nv"].plan(w.pos, np.array([0.35, 4.0]))
            return snap()
        w.step(s["nv"].cmd(w.pos, 1.2)); return snap()
    a = ActorSpec(c, kinematic=kin, base=(0.4, 0.45, 0.6))
    def prop_fn(t, ctx, sc, row): sc.set_prop(0, [0, 0, 0])
    return Shot("bed_lie_rise", "walk to a bed, sit on its edge, lean back, swing the legs up, lie down; then get up and leave (goals for pelvis, hands and feet, IK does the rest)", tr, [a], 32.0,
                dict(dist=5.0, azimuth=20, elevation=-18, follow=0, look_z=0.5), props=props, prop_fn=prop_fn, size=(480, 300))

# ---------------------------------------------------------------------------- get up from the floor
@shot
def get_up_floor():
    c = human(); tr = Terrain(); poser = Poser(c)
    R_lie = R_ypr(0.0, -math.pi / 2, 0.0); z0 = c.z0
    flat = foot_rot(0.0)
    P0 = np.array([0.0, 0.0, 0.14])
    L = c.leg_len()
    keys = [
        Key(0.0, P0, R_lie, feet={"L": (np.array([0.85, 0.10, 0.04]), R_lie), "R": (np.array([0.85, -0.10, 0.04]), R_lie)}, hands={"L": np.array([-0.2, 0.28, 0.05]), "R": np.array([-0.2, -0.28, 0.05])}),
        Key(1.2, P0, R_lie, feet={"L": (np.array([0.45, 0.12, 0.05]), flat), "R": (np.array([0.45, -0.12, 0.05]), flat)}, hands={"L": np.array([-0.1, 0.30, 0.05]), "R": np.array([-0.1, -0.30, 0.05])}),
        Key(2.5, P0 + np.array([0, 0, 0.0]), R_ypr(0, -0.55, 0), (0, 0.2, 0), (0, 0.2, 0), feet={"L": (np.array([0.45, 0.12, 0.05]), flat), "R": (np.array([0.45, -0.12, 0.05]), flat)},
            hands={"L": np.array([0.30, 0.26, 0.45]), "R": np.array([0.30, -0.26, 0.45])}),
        Key(3.6, np.array([0.28, 0.0, 0.36]), R_ypr(0, 0.75, 0), (0, 0.25, 0), (0, 0.25, 0), feet={"L": (np.array([0.45, 0.12, 0.05]), flat), "R": (np.array([0.45, -0.12, 0.05]), flat)},
            hands={"L": np.array([0.62, 0.22, 0.06]), "R": np.array([0.62, -0.22, 0.06])}),
        Key(4.7, np.array([0.40, 0.0, z0 * 0.78]), R_ypr(0, 0.55, 0), (0, 0.2, 0), (0, 0.2, 0), feet={"L": (np.array([0.45, 0.12, 0.05]), flat), "R": (np.array([0.45, -0.12, 0.05]), flat)},
            hands={"L": np.array([0.65, 0.22, 0.30]), "R": np.array([0.65, -0.22, 0.30])}),
        Key(5.8, np.array([0.42, 0.0, z0 * 0.99]), R_ypr(0, 0.05, 0), (0, 0, 0), (0, 0, 0), feet={"L": (np.array([0.45, 0.10, 0.05]), flat), "R": (np.array([0.45, -0.10, 0.05]), flat)},
            hands={"L": np.array([0.42, 0.30, 0.55]), "R": np.array([0.42, -0.30, 0.55])}),
        Key(7.0, np.array([0.42, 0.0, z0 * 0.99]), R_ypr(0, 0.0, 0), (0, 0, 0), (0, 0, 0), feet={"L": (np.array([0.45, 0.10, 0.05]), flat), "R": (np.array([0.45, -0.10, 0.05]), flat)},
            hands={"L": np.array([0.42, 0.28, 0.35]), "R": np.array([0.42, -0.28, 0.35])}),
    ]
    def kin(t, ctx):
        # stay supine for a moment, then rise
        S, E, R, q = pose_from_keys(c, poser, keys, max(0.0, t - 0.8))
        return S, E, R
    a = ActorSpec(c, kinematic=kin, base=(0.4, 0.45, 0.6))
    return Shot("get_up_floor", "get up from the floor: knees up, sit up, roll over the feet, squat, stand (goal sequence of pelvis, feet and hands)", tr, [a], 9.0,
                dict(dist=4.6, azimuth=30, elevation=-12, follow=0, look_z=0.6), size=(480, 300))

# ---------------------------------------------------------------------------- swimming
WATER = dict(geom='<geom type="box" size="12 6 0.02" pos="0 0 -0.02" rgba="0.15 0.45 0.80 0.45"/>')

@shot
def swim_freestyle():
    c = human(); tr = Terrain(); tr.ground_z = -2.5; poser = Poser(c); sk = c.skel; ix = sk.idx
    f_stroke = 0.75; speed = 0.9
    R0 = rot("Y", math.pi / 2)
    def kin(t, ctx):
        ph = (t * f_stroke) % 1.0
        pos = np.array([speed * t, 0.0, -0.10])
        roll = math.radians(38) * math.sin(2 * math.pi * (t * f_stroke) )
        yaw_t = 0.05 * math.sin(2 * math.pi * t * f_stroke)
        R = rot("Z", yaw_t) @ R0 @ rot("Z", roll)
        hands = {}
        # shoulders for target generation come from a first pass
        S0, E0, Rr0, q0 = poser.pose(pos, R, None, None, None)
        for side, off in (("L", 0.0), ("R", 0.5)):
            p = (ph + off) % 1.0
            sh = S0[ix["uarm_" + side]]
            lat = 1.0 if side == "L" else -1.0
            if p < 0.45:        # pull
                u = p / 0.45
                f = 0.78 - 0.85 * u; h = -0.15 - 0.22 * math.sin(math.pi * u * 0.9); l = 0.20 - 0.08 * u
            elif p < 0.62:      # push and exit
                u = (p - 0.45) / 0.17
                f = -0.07 - 0.20 * u; h = -0.20 + 0.22 * u; l = 0.12 + 0.05 * u
            else:               # recovery above water, elbow high
                u = (p - 0.62) / 0.38
                f = -0.27 + 1.05 * u; h = 0.02 + 0.30 * math.sin(math.pi * u); l = 0.17 + 0.18 * math.sin(math.pi * u)
            hands[side] = np.array([sh[0] + f, sh[1] + lat * l, max(h + 0.10, -0.45)])
        # flutter kick: hips and knees alternate
        ex = {}
        for side, sgn in (("L", 1.0), ("R", -1.0)):
            kick = math.sin(2 * math.pi * (6 * t * f_stroke) * 0.5 * 1.0 + (0 if side == "L" else math.pi))
            ex["thigh_" + side] = (0, 0.28 * kick - 0.05, 0)
            ex["shank_" + side] = (0, 0.35 * (0.5 + 0.5 * math.sin(2 * math.pi * (6 * t * f_stroke) * 0.5 + (0.9 if side == "L" else math.pi + 0.9))) + 0.05, 0)
            ex["foot_" + side] = (0, 0.7, 0)
        # breathing: turn the head to the side of the recovering arm once per cycle
        pb = (ph + 0.0) % 1.0
        breath = math.sin(math.pi * min(1.0, max(0.0, (pb - 0.55) / 0.35))) if pb > 0.55 else 0.0
        spq = {ix["head"]: (0, -0.25 * 0 , 0.0), }
        extra = {ix["neck"]: (0, -0.15, 0.9 * breath), ix["head"]: (0, -0.25, 0.4 * breath)}
        spq.update(extra)
        S, E, Rr, q = poser.pose(pos, R, spq, None, hands, poles={"L": np.array([-0.3, 0.8, 0.6]), "R": np.array([-0.3, -0.8, 0.6])}, extra_q={ix[k]: v for k, v in ex.items()})
        return S, E, Rr
    a = ActorSpec(c, kinematic=kin, base=(0.85, 0.6, 0.45))
    return Shot("swim_freestyle", "swimming: the body floats with the head at the surface, arms stroke through pull, push and high recovery, legs flutter, the body rolls and the head turns to breathe", tr,
                [a], 10.0, dict(dist=3.4, azimuth=40, elevation=-10, follow=0, look_z=0.0), props=[WATER], prop_fn=lambda t, ctx, sc, row: sc.set_prop(0, [row[0]["root"][0], 0, 0]), size=(480, 300))

@shot
def dive_swim():
    c = human(); tr = Terrain(); tr.ground_z = -3.0; poser = Poser(c); sk = c.skel; ix = sk.idx
    PH = 1.0     # platform height
    tr.add_box(-3.0, 0.8, -1.0, 1.0, PH, rgba=(0.6, 0.6, 0.62, 1), bottom=-3.0)
    dt = 1 / 60; T = 12.0
    track = []
    # phase times
    t_crouch, t_jump = 1.2, 1.9
    pos = np.array([0.55, 0.0, PH + c.z0]); vel = np.zeros(3)
    theta = 0.0; spin = 0.0
    for k in range(int(T / dt)):
        t = k * dt
        if t < t_jump:
            pos = np.array([0.55, 0.0, PH + c.z0 * (1 - 0.30 * minjerk((t - t_crouch) / 0.6) * (1 if t < t_crouch + 0.6 else 1.0))]) if t >= t_crouch else np.array([0.55, 0.0, PH + c.z0])
            if t >= t_crouch + 0.6: pos[2] = PH + c.z0 * 0.70 + (c.z0 * 0.30) * minjerk((t - t_crouch - 0.6) / (t_jump - t_crouch - 0.6)) * 0.9
            vel = np.array([0, 0, 0.0]); theta = 0.0
            if t >= t_crouch + 0.6: vel = np.array([2.0, 0, 3.2]) * minjerk((t - t_crouch - 0.6) / (t_jump - t_crouch - 0.6))
        else:
            under = pos[2] < -0.05
            if not under:
                vel = vel + np.array([0, 0, -9.81]) * dt
                theta = min(theta + (math.radians(165) / 0.95) * dt, math.radians(165))
            else:
                # underwater: drag, buoyancy, the body aligns with the velocity and curves up
                sp = np.linalg.norm(vel)
                vel = vel - 2.4 * vel * dt + np.array([0, 0, 3.6]) * dt
                ang = math.atan2(vel[2], max(vel[0], 1e-3))       # direction of travel, 0 horizontal, + up
                theta_t = math.pi / 2 - ang                       # head leads along the velocity
                theta += (theta_t - theta) * min(1.0, 4 * dt)
            pos = pos + vel * dt
            if pos[2] > -0.12 and vel[2] > 0 and (t - t_jump) > 1.6:      # surfaced: float with the head out
                pos[2] = -0.12; vel = np.array([0.6, 0, 0]); theta = min(theta + 1.5 * dt, math.radians(95))
        track.append((t, pos.copy(), theta, vel.copy()))
    def kin(t, ctx):
        k = min(int(t / dt), len(track) - 1); _, pos, th, vel = track[k]
        tt = t
        crouch = minjerk((tt - t_crouch) / 0.6) if tt < t_crouch + 0.6 else 1.0
        airborne = 1.0 if (tt > t_jump and pos[2] > -0.05) else 0.0
        under = pos[2] <= -0.05
        R = rot("Y", th)
        S0, E0, R0, _ = poser.pose(pos, R, None, None, None)
        zb = R @ np.array([0, 0, 1.0])
        hands = {}
        feet = None
        spq = {}
        extra = {}
        if tt < t_jump:
            swing = math.sin(math.pi * min(1.0, max(0.0, (tt - 0.4) / 1.4)))
            for side, lat in (("L", 1.0), ("R", -1.0)):
                sh = S0[ix["uarm_" + side]]
                if tt < t_crouch + 0.6: hands[side] = sh + np.array([0.15 * crouch, 0.05 * lat, -0.55 + 0.1 * crouch]) + np.array([-0.35, 0, 0.45]) * swing * 0.0
                else:
                    u = minjerk((tt - t_crouch - 0.6) / (t_jump - t_crouch - 0.6)); hands[side] = sh + np.array([0.15, 0.05 * lat, -0.55]) * (1 - u) + zb * 0.65 * u + np.array([0, 0.0, 0])
            # feet planted on the platform while crouching and pushing off
            ank_ctr = np.array([0.55, 0.0, PH + 0.02])
            feet = {s: (ankle_for(c, s, ank_ctr + np.array([0.03, 0.10 * (1 if s == "L" else -1), 0.0]), foot_rot(0.0)), foot_rot(0.0)) for s in ("L", "R")}
            if tt >= t_crouch + 0.6 + 0.5: feet = None
            spq = spine_split(c, (0, 0.5 * crouch, 0), (0, 0.3 * crouch, 0))
            if feet is None:
                extra = {"thigh_L": (0, 0, 0), "thigh_R": (0, 0, 0)}
        else:
            # flight and entry: arms stretched past the head, body straight, slight pike in the middle
            pike = 0.55 * math.sin(math.pi * min(1.0, max(0.0, (tt - t_jump) / 0.9))) if not under else 0.0
            for side, lat in (("L", 1.0), ("R", -1.0)):
                sh = S0[ix["uarm_" + side]]
                hands[side] = sh + zb * 0.72 + np.array([0, 0.0, 0]) * 0
            spq = spine_split(c, (0, -pike * 0.5, 0), (0, -pike * 0.3, 0))
            extra = {"thigh_L": (0, -pike * 1.2, 0), "thigh_R": (0, -pike * 1.2, 0), "foot_L": (0, 0.8, 0), "foot_R": (0, 0.8, 0)}
            if under and (tt - t_jump) > 3.2:      # kicking while surfacing
                kick = math.sin(2 * math.pi * 2.2 * tt)
                extra["thigh_L"] = (0, 0.3 * kick, 0); extra["thigh_R"] = (0, -0.3 * kick, 0)
        S, E, Rr, q = poser.pose(pos, R, spq, feet, hands, extra_q={ix[k]: v for k, v in extra.items()})
        return S, E, Rr
    a = ActorSpec(c, kinematic=kin, base=(0.85, 0.6, 0.45))
    return Shot("dive_swim", "dive: crouch and jump, rotate head first, enter the water, glide and curve up with drag and buoyancy, surface with the head out", tr, [a], T,
                dict(dist=6.0, azimuth=60, elevation=-6, follow=0, look_z=0.3, look_off=(1.2, 0, 0)), props=[dict(geom='<geom type="box" size="9 6 0.02" pos="0 0 -0.02" rgba="0.15 0.45 0.80 0.40"/>')],
                prop_fn=lambda t, ctx, sc, row: sc.set_prop(0, [7.8, 0, 0.0]), size=(480, 300))


# ---------------------------------------------------------------------------- bump the head
@shot
def bump_head():
    """Walking into a low beam without looking: the head stops, the body is thrown back, the hand goes to the forehead."""
    c = human(); tr = Terrain(); tr.add_beam(4.2, 5.2, -1.5, 1.5, 1.55)
    def cmd(t, w, ctx):
        s = ctx.state
        if "hit" not in s:
            head_top = w.S[w.sk.idx["head"]][2] + w.sk.length[w.sk.idx["head"]] if w.S is not None else 0.0
            ahead = w.pos[0] + 0.18
            if w.S is not None and head_top > 1.55 and w.S[w.sk.idx["head"]][0] > 4.1:
                s["hit"] = t; w.apply_push((-1, 0), 1.7, chest=True); w.wob_v[1] += -2.5
            return Cmd(v=np.array([1.3, 0.0]))
        a = t - s["hit"]
        shoulder = w.S[w.sk.idx["uarm_L"]]; head = w.S[w.sk.idx["head"]] + 0.6 * (w.E[w.sk.idx["head"]] - w.S[w.sk.idx["head"]])
        u = minjerk(a / 0.5)
        fore = head + np.array([0.09, 0.03, 0.04])
        arms = {"L": (shoulder + np.array([0.1, 0.1, -0.5])) * (1 - u) + fore * u} if 0.15 < a < 3.0 else None
        return Cmd(v=np.zeros(2), heading=0.0, arms=arms)
    a = ActorSpec(c, cmd=cmd, pos=(0, 0), walker_kw=dict())
    sh = Shot("bump_head", "not looking: the head hits a low beam, the body is thrown back and the hand goes to the forehead", tr, [a], 7.0, dict(dist=4.4, azimuth=70, elevation=-6))
    sh.post_init = lambda ws: setattr(ws[0], "auto_duck", False)
    return sh
