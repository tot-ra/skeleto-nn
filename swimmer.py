"""A swimmer in water, learned: a rhythmic stroke generator whose numbers are found by CMA-ES against the physics of the water.

The body is the same physical skeleton as everywhere else, in a pool (water.py): buoyancy, drag on every segment, flat palms and soles
that push on the water. Nothing about a stroke is drawn: the generator only says how each joint moves in time (mean, amplitude, phase
of the shoulders, elbows, wrists, hips, knees, ankles, trunk twist and pitch, neck turn) and the search keeps what moves the swimmer.
Two objectives are used:
  * "under": reach a goal under water as fast as possible - on the bottom (dive down) or far away at depth (swim with the hands);
  * "surface": cover as much distance as possible in a fixed time at the surface, with a body that needs air: oxygen is spent while the
    mouth is under water (faster the harder the muscles work), muscle strength falls as the reserve falls, and at zero the swimmer blacks
    out. The head must turn to the air from time to time; when is part of what is learned (a threshold on the oxygen reserve).
Every run also pays a small price in effort (mechanical work) so that flailing is not free."""
from __future__ import annotations
import math, json
import numpy as np
from bodies import human
from physics import PhysChar, CTRL_HZ, SUBSTEPS
from skeleton import rot
import water as WT

W_SURF = 4.0                       # surface height; the pool floor is at z = 0

# (name, default, lo, hi)
SPEC = [
    ("f", 0.9, 0.35, 1.7),                        # stroke frequency (Hz)
    ("arm_y_m", -80.0, -150.0, 0.0), ("arm_y_a", 65.0, 0.0, 90.0),        # shoulder flexion: mean and amplitude (deg), + = arm back
    ("arm_x_m", 20.0, -10.0, 90.0), ("arm_x_a", 15.0, 0.0, 60.0), ("arm_x_ph", 0.0, -3.14, 3.14),     # abduction
    ("arm_z_m", 0.0, -60.0, 60.0), ("arm_z_a", 20.0, 0.0, 60.0), ("arm_z_ph", 0.0, -3.14, 3.14),      # upper-arm rotation
    ("elb_m", -50.0, -140.0, 0.0), ("elb_a", 35.0, 0.0, 70.0), ("elb_ph", 0.8, -3.14, 3.14),
    ("hand_m", 0.0, -50.0, 50.0), ("hand_a", 25.0, 0.0, 50.0), ("hand_ph", 0.0, -3.14, 3.14),
    ("arm_dphi", 3.14, 0.0, 6.28),                # phase between the arms (pi = alternating, 0 = together)
    ("hip_m", 5.0, -30.0, 30.0), ("hip_a", 20.0, 0.0, 50.0), ("hip_x", 0.0, -20.0, 20.0),
    ("knee_m", 25.0, 0.0, 90.0), ("knee_a", 25.0, 0.0, 60.0), ("knee_ph", 0.9, -3.14, 3.14),
    ("foot_m", 25.0, -30.0, 50.0), ("foot_a", 15.0, 0.0, 40.0), ("foot_ph", 0.4, -3.14, 3.14),
    ("kick_ratio", 2.0, 0.5, 4.0),                # kick frequency over stroke frequency
    ("twist_a", 5.0, 0.0, 12.0), ("twist_ph", 0.0, -3.14, 3.14),            # body roll: trunk twist per joint (deg)
    ("pitch_b", 0.0, -12.0, 12.0), ("pitch_g", 10.0, -20.0, 40.0),          # trunk pitch bias per joint and gain with the goal elevation
    ("neck_b", 0.0, -40.0, 40.0), ("neck_g", 15.0, -40.0, 40.0),
    ("steer", 6.0, 0.0, 20.0),                     # trunk twist per joint per radian of heading error
    ("breath_thr", 0.55, 0.05, 0.98), ("breath_turn", 55.0, 0.0, 70.0), ("breath_roll", 8.0, 0.0, 14.0), ("breath_lift", 15.0, 0.0, 45.0), ("breath_ph", 0.0, -3.14, 3.14),
]
NAMES = [s[0] for s in SPEC]
DEFAULT = np.array([s[1] for s in SPEC], float); LO = np.array([s[2] for s in SPEC], float); HI = np.array([s[3] for s in SPEC], float)
D2R = math.pi / 180.0

def unpack(x):
    x = np.clip(np.asarray(x, float), LO, HI); return {n: float(v) for n, v in zip(NAMES, x)}

class Controller:
    def __init__(self, pc, params, mode):
        self.pc = pc; self.p = unpack(params); self.mode = mode; self.names = [(pc.sk.bones[i].name, "XYZ"[a]) for i, a in pc.dof_list]
        self.breathing = False; self.breath_t0 = -9.0; self.phase = 0.0; self.stroke_n = 0; self.last_arm_phase = 0.0

    def targets(self, t, dt, goal, o2):
        pc = self.pc; p = self.p; d = pc.data
        self.phase += 2 * math.pi * p["f"] * dt
        ph = self.phase
        com = pc.com(); to = goal - com; dist = float(np.linalg.norm(to)) + 1e-6
        elev = math.atan2(to[2], math.hypot(to[0], to[1]))
        Rr = d.xmat[pc.bid[0]].reshape(3, 3); fwd = Rr[:, 2]                                   # the head points along local z
        yaw_err = math.atan2(to[1], to[0]) - math.atan2(fwd[1], fwd[0]); yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
        # breathing: start a breath at the first stroke after the reserve falls below the threshold
        if self.mode == "surface":
            if (not self.breathing) and o2 < p["breath_thr"] and math.sin(ph + p["breath_ph"]) > 0.9: self.breathing = True; self.breath_t0 = t
            if self.breathing and t - self.breath_t0 > 0.65 / p["f"]: self.breathing = False
        br = 0.0
        if self.breathing:
            u = (t - self.breath_t0) * p["f"] / 0.65; br = math.sin(math.pi * min(1.0, u))
        q = np.zeros(pc.nj)
        for j, (nm, ax) in enumerate(self.names):
            side = 0 if nm.endswith("_L") else 1 if nm.endswith("_R") else -1
            s = 1.0 if side == 0 else -1.0
            ph_arm = ph + (p["arm_dphi"] if side == 1 else 0.0)
            ph_leg = p["kick_ratio"] * ph + (math.pi if side == 1 else 0.0)
            v = 0.0
            if nm.startswith("uarm"):
                if ax == "Y": v = p["arm_y_m"] + p["arm_y_a"] * math.sin(ph_arm)
                elif ax == "X": v = s * (p["arm_x_m"] + p["arm_x_a"] * math.sin(ph_arm + p["arm_x_ph"]))
                elif ax == "Z": v = s * (p["arm_z_m"] + p["arm_z_a"] * math.sin(ph_arm + p["arm_z_ph"]))
            elif nm.startswith("farm") and ax == "Y": v = p["elb_m"] + p["elb_a"] * math.sin(ph_arm + p["elb_ph"])
            elif nm.startswith("hand"):
                if ax == "Y": v = p["hand_m"] + p["hand_a"] * math.sin(ph_arm + p["hand_ph"])
            elif nm.startswith("thigh"):
                if ax == "Y": v = p["hip_m"] + p["hip_a"] * math.sin(ph_leg)
                elif ax == "X": v = s * p["hip_x"]
            elif nm.startswith("shank") and ax == "Y": v = p["knee_m"] + p["knee_a"] * math.sin(ph_leg + p["knee_ph"])
            elif nm.startswith("foot") and ax == "Y": v = p["foot_m"] + p["foot_a"] * math.sin(ph_leg + p["foot_ph"])
            elif nm in ("lumbar", "lumbar1", "lumbar2", "thorax", "chest"):
                if ax == "Y": v = p["pitch_b"] + p["pitch_g"] * elev * 0.5
                elif ax == "Z": v = p["twist_a"] * math.sin(ph + p["twist_ph"]) + p["steer"] * math.degrees(yaw_err) * 0.2 / 5 + (p["breath_roll"] * br)
            elif nm == "neck":
                if ax == "Y": v = p["neck_b"] + p["neck_g"] * elev - p["breath_lift"] * br
                elif ax == "Z": v = p["breath_turn"] * br * 0.6
            elif nm == "head":
                if ax == "Z": v = p["breath_turn"] * br * 0.4
                elif ax == "Y": v = -p["breath_lift"] * br * 0.3
            q[j] = v * D2R
        return np.clip(q, pc.lo * 0.98, pc.hi * 0.98)

def make_pool():
    c = human(); pc = PhysChar(c); sw = WT.Water(pc, W_SURF); return c, pc, sw

def rollout(pc, sw, params, mode, goal=None, T=10.0, record=False, depth0=0.8, seed=0):
    """Run one swim. Returns dict(cost, progress, time_to_goal, o2_min, blackout, traj)."""
    c = pc.c; m = pc.model; d = pc.data; br = WT.Breath(pc, W_SURF)
    ctl = Controller(pc, params, mode)
    R0 = rot("Y", math.pi / 2)                                        # prone: the head points along +x
    z0 = W_SURF - (0.14 if mode == "surface" else depth0)
    pc.set_state(np.array([0.0, 0.0, z0]), R0, np.zeros(pc.nj), root_vel=np.array([0.6, 0.0, 0.0]))
    goal = np.array([8.0, 0.0, W_SURF - 0.14]) if goal is None else np.asarray(goal, float)
    tau0 = np.array([m.actuator_forcerange[a, 1] for a in pc.aid]); n = int(T * CTRL_HZ); dt = 1.0 / CTRL_HZ
    work = 0.0; reached = None; traj = []; last_o2_state = []
    for k in range(n):
        t = k * dt
        sc = br.strength()
        for j, a in enumerate(pc.aid): m.actuator_forcerange[a, 0] = -tau0[j] * sc; m.actuator_forcerange[a, 1] = tau0[j] * sc
        tgt = ctl.targets(t, dt, goal, br.s)
        pc.step(tgt, pre=sw.apply)
        P = float(np.sum(np.abs(d.actuator_force[pc.aid] * d.qvel[pc.vadr])))                     # mechanical power at the joints
        br.step(dt, P); work += P * dt
        com = pc.com()
        if record: S, E, Rw = pc.bone_ends(); traj.append((S.copy(), E.copy(), Rw.copy(), br.s, br.mouth()[2] > W_SURF - 0.01))
        if not np.all(np.isfinite(d.qpos)) or br.blackout: break
        if mode == "under" and np.linalg.norm(com - goal) < 0.7 and reached is None: reached = t
        if mode == "under" and reached is not None: break
    com = pc.com(); p0 = np.array([0.0, 0.0, z0])
    if mode == "surface":
        progress = float(com[0]); cost = -progress / T + 0.0004 * work / T + (3.0 if br.blackout else 0.0) + 0.5 * max(0.0, 0.3 - br.s)
    else:
        remain = float(np.linalg.norm(com - goal)); t_goal = reached if reached is not None else T + remain * 1.5
        cost = t_goal / T + 0.00015 * work / T + (3.0 if br.blackout else 0.0) + (0.5 if (br.mouth()[2] > W_SURF + 0.05) else 0.0)
        progress = float(np.linalg.norm(p0 - goal) - remain)
    return dict(cost=float(cost), progress=progress, reached=reached, o2=br.s, blackout=br.blackout, work=work, traj=traj, x=float(com[0]))
