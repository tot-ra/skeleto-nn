"""A bird in the air, learned: a flapping generator with feedback, evolved against the physics of the air (air.py).

The bird is the same physical skeleton as everywhere else; the wings and tail are flat plates in the air. What the search finds are the
numbers of a rhythmic stroke (flap, feather, sweep, fold of elbow and wrist), of a few reflexes that keep the body steady (tail and wing
angle of attack against pitch error and pitch rate, differential flapping against roll) and of the behaviour at the two ends of a flight
(the push of the legs in take-off, the flare before landing). Three objectives, three searches:
  fly      stay in the air as long as possible with a limited energy store: after the store is used the muscles can only hold the wings out, so
           the search has to learn to flap, to glide and how to mix them;
  takeoff  from standing on the ground, reach a safe height with speed;
  land     arrive at a chosen spot with the least injury: the force through head, body, legs and wings at touch-down is scored by the same bone-injury
           model as everything else (injury.py)."""
from __future__ import annotations
import math, json
import numpy as np
from bodies import bird
from physics import PhysChar, CTRL_HZ
import air as AIR
import injury as INJ

G = 9.81
SPEC = [
    ("f", 4.5, 1.5, 9.0),
    ("ax_m", 5.0, -20.0, 40.0), ("ax_a", 30.0, 0.0, 80.0),                         # flap about the body axis (deg)
    ("ay_m", 0.0, -25.0, 35.0), ("ay_a", 25.0, 0.0, 60.0), ("ay_ph", 1.2, -3.14, 3.14),     # wing twist = angle of attack
    ("az_m", 0.0, -40.0, 40.0), ("az_a", 12.0, 0.0, 40.0), ("az_ph", 0.0, -3.14, 3.14),     # sweep
    ("eb_m", -25.0, -140.0, 10.0), ("eb_a", 35.0, 0.0, 80.0), ("eb_ph", 2.4, -3.14, 3.14),  # elbow fold
    ("wc_m", 15.0, -10.0, 100.0), ("wc_a", 20.0, 0.0, 60.0), ("wc_ph", 2.4, -3.14, 3.14),   # wrist fold
    ("tail_m", 0.0, -35.0, 35.0), ("k_tp", 25.0, 0.0, 80.0), ("k_td", 10.0, 0.0, 30.0),    # tail: mean, pitch gain, pitch-rate gain (deg per rad, deg per rad/s)
    ("k_wp", 0.0, -30.0, 60.0), ("k_roll", 0.5, 0.0, 2.0), ("k_tr", 20.0, 0.0, 60.0),
    ("flap_duty", 1.0, 0.0, 1.0), ("flap_T", 1.5, 0.4, 4.0),                       # flap in bursts: a fraction of a period, then wings held out
    ("leg_tuck", 60.0, 0.0, 90.0), ("pitch0", 5.0, -15.0, 25.0), ("neck_f", 17.0, 0.0, 30.0),
    ("push_amp", 40.0, 0.0, 70.0), ("push_t", 0.30, 0.1, 0.7), ("lift_ph", 0.0, -3.14, 3.14),                  # take-off: leg thrust
    ("d_flare", 4.0, 1.0, 10.0), ("flare_pitch", 40.0, 0.0, 70.0), ("flare_aoa", 20.0, 0.0, 50.0), ("flare_leg", 50.0, 0.0, 90.0), ("flare_flap", 0.7, 0.0, 1.5),   # landing
]
NAMES = [s[0] for s in SPEC]
DEFAULT = np.array([s[1] for s in SPEC], float); LO = np.array([s[2] for s in SPEC], float); HI = np.array([s[3] for s in SPEC], float)
D2R = math.pi / 180.0
def unpack(x): return {n: float(v) for n, v in zip(NAMES, np.clip(np.asarray(x, float), LO, HI))}

def make_world(species="crow"):
    c = bird(species); pc = PhysChar(c); ar = AIR.Air(pc); return c, pc, ar

def region_of(name):
    if name.startswith(("head", "neck")): return "head"
    if name.startswith("wing"): return "arms"
    if name.startswith(("thigh", "shank", "meta", "toes")): return "legs"
    return "torso"

class Flier:
    def __init__(self, pc, params, mode):
        self.pc = pc; self.p = unpack(params); self.mode = mode
        self.names = [(pc.sk.bones[i].name, "XYZ"[a]) for i, a in pc.dof_list]; self.phase = 0.0

    def pitch_state(self):
        d = self.pc.data; R = d.xmat[self.pc.bid[0]].reshape(3, 3)
        pitch = math.asin(float(np.clip(R[2, 0], -1, 1))); roll = math.asin(float(np.clip(R[2, 1], -1, 1)))      # body x = forward, y = left
        w = R.T @ d.qvel[3:6]; return pitch, roll, -float(w[1]), float(w[0])

    def targets(self, t, dt, ctx):
        p = self.p; pc = self.pc
        pitch, roll, pitch_rate, roll_rate = self.pitch_state()
        gate = 1.0
        if p["flap_duty"] < 0.999: gate = 1.0 if ((t / p["flap_T"]) % 1.0) < p["flap_duty"] else 0.0
        flare = 0.0; push = 0.0
        if self.mode == "land":
            flare = float(np.clip((p["d_flare"] - ctx["dist"]) / max(p["d_flare"] * 0.5, 1e-3) + 0.5, 0.0, 1.0)) if ctx["dist"] < p["d_flare"] else 0.0
        if self.mode == "takeoff": push = 1.0 if t < p["push_t"] else 0.0
        self.phase += 2 * math.pi * p["f"] * dt * gate
        ph = self.phase; amp_f = (1.0 - flare * (1.0 - p["flare_flap"])) * (1.0 if gate else 0.0)
        pit_err = (p["pitch0"] + p["flare_pitch"] * flare) * D2R - pitch
        q = np.zeros(pc.nj)
        for j, (nm, ax) in enumerate(self.names):
            sd = 1.0 if nm.endswith("_L") else -1.0 if nm.endswith("_R") else 0.0
            v = 0.0
            if nm.startswith("wing_a"):
                diff = 1.0 + sd * p["k_roll"] * roll * 2.0
                if ax == "X": v = sd * (p["ax_m"] + amp_f * p["ax_a"] * diff * math.sin(ph))
                elif ax == "Y": v = p["ay_m"] + p["ay_a"] * amp_f * math.sin(ph + p["ay_ph"]) - p["k_wp"] * pit_err / D2R * 0.4 - p["flare_aoa"] * flare
                elif ax == "Z": v = sd * (p["az_m"] + p["az_a"] * amp_f * math.sin(ph + p["az_ph"]))
            elif nm.startswith("wing_b") and ax == "Z": v = sd * (p["eb_m"] + p["eb_a"] * amp_f * math.sin(ph + p["eb_ph"]))
            elif nm.startswith("wing_c") and ax == "Z": v = sd * (p["wc_m"] + p["wc_a"] * amp_f * math.sin(ph + p["wc_ph"]))
            elif nm == "tail0" and ax == "Y": v = p["tail_m"] + p["k_tp"] * pit_err / D2R * 0.4 - p["k_td"] * pitch_rate * 0.5 + 25.0 * flare
            elif nm == "tail0" and ax == "Z": v = p["k_tr"] * roll
            elif nm.startswith("thigh") and ax == "Y":
                v = p["leg_tuck"] * (1 - flare) - p["flare_leg"] * flare * 0.5 - p["push_amp"] * push
            elif nm.startswith("shank") and ax == "Y": v = -p["leg_tuck"] * 1.2 * (1 - flare) + p["flare_leg"] * flare * 0.3 + 30 * push
            elif nm.startswith("meta") and ax == "Y": v = p["leg_tuck"] * 0.6 * (1 - flare) + p["push_amp"] * 0.7 * push
            elif nm.startswith("neck") and ax == "Y": v = p["neck_f"] * (1 - 0.6 * flare) - pitch / D2R * 0.2
            q[j] = v * D2R
        return np.clip(q, pc.lo * 0.98, pc.hi * 0.98)

def rollout(pc, ar, params, mode, T=15.0, target_x=12.0, energy_j=150.0, record=False, seed=0):
    c = pc.c; m = pc.model; d = pc.data
    fl = Flier(pc, params, mode); dt = 1.0 / CTRL_HZ
    if mode == "takeoff":
        pc.set_state(np.array([0.0, 0.0, c.z0 + 0.01]), np.eye(3), np.zeros(pc.nj))
    else:
        z0 = 6.0 if mode == "fly" else 4.0; v0 = 8.0 if mode == "fly" else 7.0
        pitch0 = math.radians(2.0); R0 = np.array([[math.cos(pitch0), 0, -math.sin(pitch0)], [0, 1, 0], [math.sin(pitch0), 0, math.cos(pitch0)]])
        pc.set_state(np.array([0.0, 0.0, z0]), R0, np.zeros(pc.nj), root_vel=np.array([v0, 0.0, 0.0]))
    tau0 = np.array([m.actuator_forcerange[a, 1] for a in pc.aid]); work = 0.0; t_air = 0.0; traj = []
    floor_ids = {pc.floor}; regions = {}
    for gi in range(m.ngeom):
        nm = m.geom(gi).name
        if nm.startswith("g_"): regions[gi] = region_of(nm[2:])
    series = {k: [] for k in INJ.REGIONS}; mg0 = pc.total_mass * G
    def contacts():
        out = {k: 0.0 for k in INJ.REGIONS}; f6 = np.zeros(6)
        for k in range(d.ncon):
            cn = d.contact[k]; g1, g2 = cn.geom1, cn.geom2
            g = g2 if g1 == pc.floor else g1 if g2 == pc.floor else None
            if g is None or g not in regions: continue
            mujoco_cf(m, d, k, f6); out[regions[g]] += abs(f6[0])
        return out
    import mujoco
    mujoco_cf = mujoco.mj_contactForce
    touch = None; dead = False; crashed = False; alt_ok_t = 0.0; reach_t = None
    n = int(T * CTRL_HZ)
    for k in range(n):
        t = k * dt; com = pc.com()
        ctx = dict(dist=max(0.0, target_x - com[0]), alt=float(com[2]))
        sc = 1.0 if work < energy_j else 0.04                                         # the energy store is used up: the muscles only hold the wings
        for j, a in enumerate(pc.aid): m.actuator_forcerange[a, 0] = -tau0[j] * sc; m.actuator_forcerange[a, 1] = tau0[j] * sc
        tgt = fl.targets(t, dt, ctx)
        def probe():
            cf = contacts()
            for kk in cf: series[kk].append(cf[kk] / mg0)
        pc.step(tgt, pre=ar.apply, probe=probe)
        P = float(np.sum(np.abs(d.actuator_force[pc.aid] * d.qvel[pc.vadr]))); work += P * dt
        com = pc.com()
        if record: S, E, Rw = pc.bone_ends(); traj.append((S.copy(), E.copy(), Rw.copy()))
        if not np.all(np.isfinite(d.qpos)): crashed = True; break
        touching = any(series[kk][-1] > 0.05 for kk in series if series[kk]) if mode != "takeoff" else False
        if mode in ("fly", "land"):
            if touching and touch is None: touch = (t, com.copy(), float(np.linalg.norm(d.qvel[0:3])))
            if touch is not None and t - touch[0] > 0.8: break
            t_air = t
        if mode == "takeoff":
            if com[2] > 2.2 and np.linalg.norm(d.qvel[0:3]) > 3.5:
                alt_ok_t += dt
                if alt_ok_t > 1.0 and reach_t is None: reach_t = t - 1.0; break
            else: alt_ok_t = 0.0
            if com[2] > 12: break
    com = pc.com(); res = dict(work=work, traj=traj, x=float(com[0]), z=float(com[2]), t_air=t_air)
    if mode == "fly":
        t_end = touch[0] if touch is not None else T
        res["cost"] = -(t_end / T) + 0.0003 * max(0.0, work - energy_j * 0.0) / max(T, 1) * 0.0 + (1.0 if crashed else 0.0) - 0.002 * float(com[0]) / T
        res["t_air"] = t_end
    elif mode == "takeoff":
        res["cost"] = (reach_t / T if reach_t is not None else 1.0 + max(0.0, 2.2 - float(com[2])) * 0.5) + (1.0 if crashed else 0.0)
        res["reached"] = reach_t
    else:
        if touch is None: res["cost"] = 3.0 + 0.1 * abs(float(com[0]) - target_x) + (1.0 if crashed else 0.0); res["touch"] = None
        else:
            icost, parts = INJ.score(series, dt / 6.0)
            res["cost"] = icost + 0.12 * abs(float(touch[1][0]) - target_x) + 0.04 * touch[2] + (1.0 if crashed else 0.0)
            res["touch"] = dict(t=touch[0], x=float(touch[1][0]), speed=touch[2], risk={k: v["risk"] for k, v in parts.items()}, peak={k: v["peak"] for k, v in parts.items()})
    return res
