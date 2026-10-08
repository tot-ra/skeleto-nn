"""Evolved protective reflexes on the physical body: falling after a shove, landing after a drop, standing on a slippery slope.

The controller is tiny (a few dozen numbers): a pose that depends linearly on the direction of the fall, a trigger angle and a
stiffness for the landing. CMA-ES searches it against random scenarios; the cost is the peak contact force on the head and body.
"""
from __future__ import annotations
import math, json, os, sys, time
import numpy as np
import mujoco
from bodies import human, HumanBody
from physics import PhysChar, SUBSTEPS, CTRL_HZ
from skeleton import rot
from terrain import Terrain
import injury as INJ

G = 9.81
# groups: (bone, axis, sym, mirror_sign_for_right)
GROUPS = [("uarm", "Y"), ("uarm", "X"), ("farm", "Y"), ("thigh", "Y"), ("thigh", "X"), ("shank", "Y"), ("lumbar", "Y"), ("chest", "Y"), ("lumbar", "X"), ("neck", "Y"), ("head", "Y"), ("foot", "Y")]
NG = len(GROUPS)
REGIONS = {"head": ("head", "neck"), "torso": ("pelvis", "lumbar", "lumbar1", "lumbar2", "thorax", "chest"), "arms": ("clav_L", "clav_R", "uarm_L", "uarm_R", "farm_L", "farm_R", "hand_L", "hand_R"), "legs": ("thigh_L", "thigh_R", "shank_L", "shank_R", "foot_L", "foot_R", "toes_L", "toes_R")}

def default_params():
    base = np.zeros(NG); fx = np.zeros(NG); fy = np.zeros(NG)
    base[3] = -0.35; base[5] = 0.6; base[9] = 0.5; base[10] = 0.3; base[6] = 0.3
    air = np.array([-0.2, 0.9, 0.2, -0.3])      # hip, knee, ankle, shoulder before touching down
    return np.concatenate([base, fx, fy, air, [math.radians(10), 0.5, 1.0, 1.0, 0.3, 1.0, 1.0, 0.35]])

def smooth01(x):
    x = min(1.0, max(0.0, x)); return x * x * x * (x * (6 * x - 15) + 10)

def _spine_share(nm):
    """Names of the vertebral chain map onto the two groups the reflex was written for, sharing the value between the joints."""
    if nm in ("lumbar", "lumbar1", "lumbar2"): return "lumbar", 1 / 3
    if nm in ("thorax", "chest"): return "chest", 1 / 2
    return nm, 1.0

class Reflex:
    def stiff(self, ts):
        """Stiffness scales t seconds after the brace began: muscles first stop the body, then yield so that the stroke is long (low peak force)."""
        u = smooth01(ts / self.soft_t)
        return self.kp_b, self.m_arm * (1 + (self.soft_arm - 1) * u), self.m_leg * (1 + (self.soft_leg - 1) * u)

    def __init__(self, c, params, drop=False):
        self.c = c; self.sk = c.skel
        p = np.asarray(params, float)
        self.base, self.fx, self.fy = p[:NG], p[NG:2 * NG], p[2 * NG:3 * NG]
        self.air = p[3 * NG:3 * NG + 4]; self.trig = abs(p[3 * NG + 4]); self.kp_b = float(np.clip(p[3 * NG + 5], 0.12, 1.5))
        self.state = "stand"; self.t_brace = None
        tail = p[3 * NG + 6:3 * NG + 8] if len(p) >= 3 * NG + 8 else np.ones(2)
        self.m_arm, self.m_leg = [float(np.clip(x, 0.15, 2.5)) for x in tail]      # limb muscles can yield (soft) or stay rigid
        soft = p[3 * NG + 8:3 * NG + 11] if len(p) >= 3 * NG + 11 else np.array([0.3, 1.0, 1.0])
        self.trig_land = float(np.clip(abs(p[3 * NG + 11]), 0.1, 1.0)) if len(p) > 3 * NG + 11 else 0.35      # tilt at which a landing turns into a fall
        self.soft_t = float(np.clip(abs(soft[0]), 0.05, 1.2)); self.soft_arm = float(np.clip(soft[1], 0.15, 2.5)); self.soft_leg = float(np.clip(soft[2], 0.15, 2.5))

    def target(self, pc: PhysChar, fall_dir_body):
        fx, fy = fall_dir_body
        q = np.zeros(pc.nj)
        names = [(self.sk.bones[i].name, "XYZ"[a]) for i, a in pc.dof_list]
        for j, (nm, ax) in enumerate(names):
            for g, (bone, gax) in enumerate(GROUPS):
                if nm.startswith(bone + "_") and ax == gax:
                    side = nm[-1]
                    if side == "L": v = self.base[g] + self.fx[g] * fx + self.fy[g] * fy
                    else:
                        v = self.base[g] + self.fx[g] * fx - self.fy[g] * fy
                        if ax in "XZ": v = -v
                    q[j] = v
            if nm in ("lumbar_X", "lumbar") and False: pass
        # single (unpaired) bones: lumbar, chest, neck, head have no side suffix
        for j, (nm, ax) in enumerate(names):
            base_nm, share = _spine_share(nm)
            for g, (bone, gax) in enumerate(GROUPS):
                if base_nm == bone and ax == gax: q[j] = (self.base[g] + self.fx[g] * fx + self.fy[g] * fy) * share
        return q

    def air_target(self, pc):
        q = np.zeros(pc.nj)
        for j, (i, a) in enumerate(pc.dof_list):
            nm = self.sk.bones[i].name; ax = "XYZ"[a]
            if nm.startswith("thigh") and ax == "Y": q[j] = self.air[0]
            elif nm.startswith("shank") and ax == "Y": q[j] = self.air[1]
            elif nm.startswith("foot") and ax == "Y": q[j] = self.air[2]
            elif nm.startswith("uarm") and ax == "Y": q[j] = self.air[3]
        return q

def set_stiffness(pc: PhysChar, scale, base_kp, m_arm=1.0, m_leg=1.0):
    m = pc.model
    for j, a in enumerate(pc.aid):
        nm = pc.skel_names[j] if hasattr(pc, "skel_names") else ""
        mult = m_arm if nm.startswith(("uarm", "farm", "hand")) else m_leg if nm.startswith(("thigh", "shank", "foot")) else 1.0
        kp = base_kp[j] * scale * mult
        m.actuator_gainprm[a, 0] = kp; m.actuator_biasprm[a, 1] = -kp; m.actuator_biasprm[a, 2] = -0.08 * kp * (0.7 + 0.6 * scale)

def contact_forces(pc: PhysChar, geom_region):
    d = pc.data; m = pc.model
    out = {k: 0.0 for k in ("head", "torso", "arms", "legs")}
    f6 = np.zeros(6)
    for k in range(d.ncon):
        c = d.contact[k]
        g = c.geom2 if m.geom_contype[c.geom1] == 1 else c.geom1
        if g in geom_region:
            mujoco.mj_contactForce(m, d, k, f6)
            out[geom_region[g]] += abs(f6[0])
    return out

def run_scenario(c, params, scen, T=3.5, record=False, passive=False, pc=None, base_kp=None):
    """scen: dict(kind='shove'|'drop', dir, mag, h, v)."""
    if pc is None: pc = PhysChar(c)
    if base_kp is None: base_kp = np.array([pc.model.actuator_gainprm[a, 0] for a in pc.aid])
    m = pc.model; sk = c.skel
    pc.skel_names = [sk.bones[i].name for i, a in pc.dof_list]
    geom_region = {}
    for reg, names in REGIONS.items():
        for nm in names: geom_region[m.geom("g_" + nm).id] = reg
    rf = Reflex(c, params)
    z0 = c.z0 + 0.003
    h = scen.get("h", 0.0)
    pc.set_state(np.array([0, 0, z0 + h]), np.eye(3), np.zeros(pc.nj))
    set_stiffness(pc, 1.0, base_kp)
    mass = pc.total_mass
    if scen["kind"] == "drop" and scen.get("v") is not None:
        pc.data.qvel[0:3] = scen["v"]
    n = int(T * CTRL_HZ); peak = {k: 0.0 for k in ("head", "torso", "arms", "legs")}; traj = []
    series = {k: [] for k in peak}; mg0 = pc.total_mass * G
    state = "stand"; t_brace = None; t_land = 0.0; d_dir = np.array([1.0, 0.0])
    air_t = 0
    for k in range(n):
        t = k / CTRL_HZ
        p, R = pc.root_pose(); v = pc.data.qvel[0:3]
        yaw = math.atan2(R[1, 0], R[0, 0]); fwd = np.array([math.cos(yaw), math.sin(yaw)]); left = np.array([-fwd[1], fwd[0]])
        com = pc.com(); vel_h = v[:2]
        tilt_vec = R[:, 2][:2]
        tilt = math.acos(max(-1, min(1, R[2, 2])))
        # direction of the fall: where the body is heading (velocity plus lean)
        dirv = vel_h * 0.6 + tilt_vec * 3.0
        if np.linalg.norm(dirv) < 1e-3: dirv = fwd
        dirv = dirv / np.linalg.norm(dirv)
        fb = (float(dirv @ fwd), float(dirv @ left))
        con = pc.foot_contacts().sum()
        if passive:
            tgt = np.zeros(pc.nj)
        else:
            if state == "stand":
                tgt = np.zeros(pc.nj)
                if scen["kind"] == "drop" and con == 0 and v[2] < -0.8: state = "air"
                elif tilt > rf.trig: state = "brace"; t_brace = t; set_stiffness(pc, rf.kp_b, base_kp, rf.m_arm, rf.m_leg)
            if state == "air":
                tgt = rf.air_target(pc)
                if con > 0 or pc.bad_contact() or p[2] < z0 + 0.05:
                    state = "land"; t_land = t
            if state == "land":
                # landing on the feet: knees bend and the limbs yield; it turns into a fall only if the body tips over or something else hits the ground
                tgt = rf.air_target(pc)
                kb_, ma_, ml_ = rf.stiff(t - t_land); set_stiffness(pc, kb_, base_kp, ma_, ml_)
                if tilt > rf.trig_land or pc.bad_contact():
                    state = "brace"; t_brace = t
                elif t - t_land > 0.9:
                    state = "stand"; set_stiffness(pc, 1.0, base_kp)
            if state == "brace":
                set_stiffness(pc, *rf.stiff(t - t_brace), ) if False else None
                tgt = rf.target(pc, fb)
                kb_, ma_, ml_ = rf.stiff(t - t_brace); set_stiffness(pc, kb_, base_kp, ma_, ml_)
                if scen["kind"] == "drop" and tilt < 0.25 and (t - t_brace) > 1.2:
                    state = "stand"; set_stiffness(pc, 1.0, base_kp)
        force = None
        if scen["kind"] == "shove" and 0.3 <= t < 0.45:
            force = np.array([scen["dir"][0], scen["dir"][1], 0.0]) * scen["mag"] * mass / 0.15
        def probe():
            cf_ = contact_forces(pc, geom_region)
            for kk in peak: series[kk].append(cf_[kk] / mg0)
        pc.step(tgt, ext_force=force, probe=probe)
        cf = {kk: series[kk][-1] * mg0 for kk in peak}
        for kk in peak: peak[kk] = max(peak[kk], max(series[kk][-6:]) * mg0)
        if record:
            S, E, Rw = pc.bone_ends(); traj.append((S.copy(), E.copy(), Rw.copy(), state))
        if not np.all(np.isfinite(pc.data.qpos)): break
    mg = mass * G
    p, R = pc.root_pose()
    upright = float(R[2, 2] > 0.8 and p[2] > 0.6 * c.z0)
    icost, parts = INJ.score(series, 1.0 / (CTRL_HZ * SUBSTEPS))
    cost = icost
    if scen["kind"] == "drop" and scen.get("h", 0) < 1.1: cost += 1.0 * (1.0 - upright)
    out = dict(cost=float(cost), peak={k: float(v / mg) for k, v in peak.items()}, upright=upright, risk={k: parts[k]["risk"] for k in parts}, rate={k: parts[k]["rate"] for k in parts})
    if record: out["traj"] = traj
    return out

def sample_scenarios(rng, n, kinds=("shove", "drop")):
    sc = []
    for _ in range(n):
        kind = rng.choice(kinds)
        if kind == "shove":
            a = rng.uniform(0, 2 * math.pi); sc.append(dict(kind="shove", dir=(math.cos(a), math.sin(a)), mag=rng.uniform(2.0, 3.6)))
        else:
            a = rng.uniform(0, 2 * math.pi); sp = rng.uniform(0, 2.0)
            sc.append(dict(kind="drop", h=float(rng.uniform(0.4, 2.6)), v=np.array([sp * math.cos(a), sp * math.sin(a), 0.0])))
    return sc

_G = {}
def _init_worker():
    c = human(); pc = PhysChar(c)
    _G["c"] = c; _G["pc"] = pc; _G["kp"] = np.array([pc.model.actuator_gainprm[a, 0] for a in pc.aid])

def _eval(args):
    params, scens = args
    cs = [run_scenario(_G["c"], params, s, pc=_G["pc"], base_kp=_G["kp"])["cost"] for s in scens]
    return float(np.mean(cs))

def evolve(gens=60, pop=24, nproc=16, nscen=14, out="runs/reflex.json", seed=0, init=None, sigma=0.4):
    import cma, multiprocessing as mp
    rng = np.random.default_rng(seed)
    x0 = np.array(init) if init is not None else default_params()
    es = cma.CMAEvolutionStrategy(x0, sigma, dict(popsize=pop, seed=seed + 1, verbose=-9))
    ctx = mp.get_context("spawn")
    best = (1e9, x0)
    with ctx.Pool(nproc, initializer=_init_worker) as pool:
        for g in range(gens):
            scens = sample_scenarios(rng, nscen)        # fresh scenarios every generation: no overfitting to a fixed set
            X = es.ask()
            f = pool.map(_eval, [(x, scens) for x in X])
            es.tell(X, f)
            i = int(np.argmin(f))
            if f[i] < best[0] or True:
                cur = es.result.xbest
            print(f"gen {g} best {min(f):.3f} mean {np.mean(f):.3f}", flush=True)
            json.dump(dict(params=[float(v) for v in es.result.xbest], gen=g), open(out, "w"))
    return es.result.xbest

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--gens", type=int, default=60); ap.add_argument("--pop", type=int, default=24); ap.add_argument("--out", default="runs/reflex.json")
    ap.add_argument("--procs", type=int, default=8); ap.add_argument("--init", default=None)
    a = ap.parse_args()
    init = json.load(open(a.init))["params"] if a.init else None
    evolve(a.gens, a.pop, a.procs, out=a.out, init=init)

def fall_from_state(c, pc, params, root_pos, root_R, qvec, vel, angvel, T=3.2, base_kp=None, min_still=0.5):
    """Let the evolved reflex handle a fall that starts from an arbitrary pose and velocity. Returns the recorded poses and the injury report."""
    if base_kp is None: base_kp = np.array([pc.model.actuator_gainprm[a, 0] for a in pc.aid])
    m = pc.model; sk = c.skel
    pc.skel_names = [sk.bones[i].name for i, a in pc.dof_list]
    geom_region = {}
    for reg, names in REGIONS.items():
        for nm in names: geom_region[m.geom("g_" + nm).id] = reg
    rf = Reflex(c, params)
    pc.set_state(root_pos, root_R, qvec, root_vel=vel, root_angvel=angvel)
    set_stiffness(pc, rf.kp_b, base_kp, rf.m_arm, rf.m_leg)
    mg0 = pc.total_mass * G
    series = {k: [] for k in REGIONS}
    def probe():
        cf = contact_forces(pc, geom_region)
        for kk in series: series[kk].append(cf[kk] / mg0)
    traj = []; still = 0.0; n = int(T * CTRL_HZ); t_brace = 0.0
    for k in range(n):
        t = k / CTRL_HZ
        p, R = pc.root_pose(); v = pc.data.qvel[0:3]
        yaw = math.atan2(R[1, 0], R[0, 0]); fwd = np.array([math.cos(yaw), math.sin(yaw)]); left = np.array([-fwd[1], fwd[0]])
        dirv = v[:2] * 0.6 + R[:, 2][:2] * 3.0
        if np.linalg.norm(dirv) < 1e-3: dirv = fwd
        dirv = dirv / np.linalg.norm(dirv)
        tgt = rf.target(pc, (float(dirv @ fwd), float(dirv @ left)))
        kb_, ma_, ml_ = rf.stiff(t); set_stiffness(pc, kb_, base_kp, ma_, ml_)
        pc.step(tgt, probe=probe)
        S, E, Rw = pc.bone_ends(); traj.append((S.copy(), E.copy(), Rw.copy(), pc.vec_to_q(pc.qvec())))
        sp = float(np.linalg.norm(pc.data.qvel[0:3]))
        still = still + 1 / CTRL_HZ if (sp < 0.08 and t > 0.6) else 0.0
        if still > min_still: break
        if not np.all(np.isfinite(pc.data.qpos)): break
    cost, parts = INJ.score(series, 1.0 / (CTRL_HZ * SUBSTEPS))
    p, R = pc.root_pose()
    return dict(traj=traj, cost=float(cost), parts=parts, upright=bool(R[2, 2] > 0.85))
