"""Imitation environment: a physical creature tracks the planner's reference (DeepMimic-style) with a residual policy."""
from __future__ import annotations
import math
import numpy as np
from physics import PhysChar, CTRL_HZ, DT, SUBSTEPS
from skeleton import rot

G = 9.81

def yaw_of(R): return math.atan2(R[1, 0], R[0, 0])

class Env:
    def __init__(self, creature, clips, seed=0, push=(0.0, 0.0), terrain=None, kp_scale=1.0, act_scale=0.30, max_len=300):
        self.c = creature; self.sk = creature.skel
        self.pc = PhysChar(creature, terrain, kp_scale)
        self.clips = clips; self.rng = np.random.default_rng(seed)
        self.J = self.pc.nj
        self.push = push                    # (probability per episode, max impulse in m/s of velocity change)
        self.assist = 0.0                   # harness strength 0..1 (annealed to zero during training)
        self.max_len = max_len
        # per-DOF weights and action scales
        w = np.ones(self.J); sc = np.full(self.J, act_scale)
        for j, (i, a) in enumerate(self.pc.dof_list):
            g = self.sk.bones[i].group
            if g in ("neck", "head"): w[j] = 0.2; sc[j] = 0.0
            elif g == "arm": w[j] = 0.5
            elif g == "foot": w[j] = 0.8
            elif g in ("tail", "wing"): w[j] = 0.1; sc[j] = 0.0
        self.wj = w / w.sum() * self.J * 0.35
        self.act_scale = sc
        self.ee_names = [l.chain[-1] for l in creature.legs] + [a.chain[-1] for a in creature.arms]
        self.legs = len(creature.legs)
        self.desc = np.array([creature.params.get("height", 1.0), math.log(max(self.sk.total_mass(), 1e-3) / 75.0) if creature.kind == "biped" else 0.0,
                              creature.params.get("body", {}).get("belly", 0.0) / 75.0, creature.leg_len()], float)
        self.z_stand = creature.z0
        self.obs_dim = len(self.reset())
        self.act_dim = self.J

    # ---- reference access ---------------------------------------------------------------
    def rq(self, k):
        k = min(k, len(self.clip["q"]) - 1); return self.pc.q_to_vec(self.clip["q"][k])

    def ree_rel(self, k):
        k = min(k, len(self.clip["q"]) - 1)
        R = self.clip["R"][k]; yaw = yaw_of(R); Rh = rot("Z", yaw)
        return (self.clip["ee"][k] - self.clip["root"][k]) @ Rh      # row vectors times Rh == Rh^T v

    # ---- episode ------------------------------------------------------------------------
    def reset(self):
        ci = self.rng.integers(len(self.clips)); self.clip = self.clips[ci]
        T = len(self.clip["q"])
        self.t = int(self.rng.integers(0, max(T - 60, 1)))
        self.t0 = self.t
        k = self.t
        root = self.clip["root"][k]; R = self.clip["R"][k]
        qv = self.rq(k)
        k2 = min(k + 1, T - 1)
        vroot = (self.clip["root"][k2] - self.clip["root"][k]) * CTRL_HZ
        dR = self.clip["R"][k].T @ self.clip["R"][k2]
        wv = self.clip["R"][k] @ self._rotvec(dR) * CTRL_HZ
        qd = (self.rq(k2) - qv) * CTRL_HZ
        self.pc.set_state(root + np.array([0, 0, 0.005]), R, qv, vroot, wv, qd)
        self.steps = 0
        self.push_t = -1.0
        if self.rng.random() < self.push[0]:
            self.push_t = int(self.rng.integers(30, 200)); self.push_dir = self._rand_dir(); self.push_len = int(self.rng.integers(3, 8))
            self.push_mag = self.rng.uniform(0.3, 1.0) * self.push[1] * self.pc.total_mass * CTRL_HZ / self.push_len   # impulse -> force
        self.ep_ret = 0.0
        return self.obs()

    def _rand_dir(self):
        a = self.rng.uniform(0, 2 * math.pi); return np.array([math.cos(a), math.sin(a), 0.0])

    @staticmethod
    def _rotvec(R):
        a = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
        if a < 1e-6: return np.zeros(3)
        ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(a))
        return ax * a

    # ---- observation ---------------------------------------------------------------------
    def ee_rel(self):
        S, E, R = self.pc.bone_ends()
        root = self.pc.data.qpos[0:3]; Rr = self.pc.data.xmat[self.pc.bid[0]].reshape(3, 3)
        Rh = rot("Z", yaw_of(Rr))
        pts = []
        for l in self.c.legs:
            f = l.chain[-1]; pts.append(S[f] + R[f] @ (0.5 * (l.heel + l.ball)))
        for a in self.c.arms:
            pts.append(E[a.chain[-1]])
        return (np.array(pts) - root) @ Rh

    def obs(self):
        pc = self.pc; d = pc.data
        p, R = pc.root_pose()
        g_loc = R.T @ np.array([0, 0, -1.0])
        v_loc = R.T @ d.qvel[0:3]; w_loc = d.qvel[3:6]
        qv = pc.qvec(); qd = pc.qdvec()
        k = self.t + 1
        Rr = self.clip["R"][min(k, len(self.clip["R"]) - 1)]
        g_ref = Rr.T @ np.array([0, 0, -1.0])
        vref = (self.clip["root"][min(k + 1, len(self.clip["root"]) - 1)] - self.clip["root"][min(k, len(self.clip["root"]) - 1)]) * CTRL_HZ
        vref_loc = Rr.T @ vref
        dyaw = yaw_of(R) - yaw_of(Rr)
        ee = self.ee_rel(); er = self.ree_rel(k)
        st = self.clip["stance"][min(k, len(self.clip["stance"]) - 1)]
        o = np.concatenate([g_loc, v_loc, w_loc * 0.2, [p[2] - 0.0], qv, qd * 0.1,
                            self.rq(k) - qv, self.rq(k + 3) - qv, self.rq(k + 7) - qv,
                            g_ref, vref_loc, [self.clip["root"][min(k, len(self.clip["root"]) - 1)][2] - p[2]], [math.sin(dyaw), math.cos(dyaw)],
                            (er - ee).ravel(), er.ravel(), st, self.desc])
        return o.astype(np.float32)

    # ---- step ---------------------------------------------------------------------------
    def step(self, a):
        a = np.clip(a, -1, 1)
        k = self.t + 1
        target = self.rq(k) + self.act_scale * a
        force = None
        if self.push_t >= 0 and self.push_t <= self.steps < self.push_t + self.push_len:
            force = self.push_dir * self.push_mag
        wrench = None
        if self.assist > 0:
            p, R = self.pc.root_pose(); vz = self.pc.data.qvel[2]
            kk = min(k, len(self.clip["root"]) - 1); zr = self.clip["root"][kk][2]; Rr = self.clip["R"][kk]
            Fz = self.assist * (0.9 * self.pc.total_mass * G + 4000.0 * (zr - p[2]) - 400.0 * vz)
            ee = np.cross(R[:, 2], Rr[:, 2]); wv = R @ self.pc.data.qvel[3:6]
            tq = self.assist * (900.0 * ee - 90.0 * wv * np.array([1.0, 1.0, 0.0]))
            wrench = np.array([0.0, 0.0, Fz, tq[0], tq[1], tq[2]])
        self.pc.step(target, ext_force=force, wrench=wrench)
        self.t += 1; self.steps += 1
        T = len(self.clip["q"])
        r, done = self.reward()
        trunc = (self.t >= T - 2) or (self.steps >= self.max_len)
        self.ep_ret += r
        info = dict(trunc=trunc and not done)
        if done or trunc:
            info["ep_ret"] = self.ep_ret; info["ep_len"] = self.steps; info["fell"] = done
        return self.obs(), r, done, info

    def reward(self):
        pc = self.pc; k = self.t
        p, R = pc.root_pose()
        qv = pc.qvec(); qd = pc.qdvec()
        qr = self.rq(k); qdr = (self.rq(k + 1) - qr) * CTRL_HZ
        e_pose = float(np.sum(self.wj * (qv - qr) ** 2))
        e_vel = float(np.sum(self.wj * (qd - qdr) ** 2)) * 0.01
        ee = self.ee_rel(); er = self.ree_rel(k)
        e_ee = float(np.sum((ee - er) ** 2))
        Rr = self.clip["R"][min(k, len(self.clip["R"]) - 1)]
        zr = self.clip["root"][min(k, len(self.clip["root"]) - 1)][2]
        e_z = (p[2] - zr) ** 2
        e_o = float(np.sum((R[:, 2] - Rr[:, 2]) ** 2))
        vr = (self.clip["root"][min(k + 1, len(self.clip["root"]) - 1)] - self.clip["root"][min(k, len(self.clip["root"]) - 1)]) * CTRL_HZ
        v_loc = R.T @ pc.data.qvel[0:3]; vr_loc = Rr.T @ vr
        e_v = float(np.sum((v_loc - vr_loc) ** 2))
        pain = min(3.0, pc.body_impact())            # body parts other than the feet carrying load: every learned skill is also scored on pain
        r = -0.10 * pain + 0.45 * math.exp(-2.0 * e_pose) + 0.10 * math.exp(-0.5 * e_vel) + 0.25 * math.exp(-8.0 * e_ee) + 0.20 * math.exp(-10 * e_z - 4 * e_o - 1.0 * e_v)
        tilt = math.degrees(math.acos(max(-1, min(1, R[2, 2]))))
        fell = (p[2] < 0.55 * self.z_stand) or tilt > 65 or pc.bad_contact() or not np.all(np.isfinite(pc.data.qpos))
        return r, fell
