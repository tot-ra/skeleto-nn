"""Pose a biped from end-effector goals (pelvis pose, feet, hands) without the gait planner:
used for riders, ladders and other situations where the body is attached to something."""
from __future__ import annotations
import math
import numpy as np
from skeleton import rot, euler_from_R
from ik import PlanarLeg, leg_ik, arm_ik, clamp_lim

class Poser:
    def __init__(self, creature):
        self.c = creature; self.sk = creature.skel
        self.pl = [PlanarLeg(self.sk, l) for l in creature.legs]

    def pose(self, root_pos, R_p, spine_q=None, feet=None, hands=None, poles=None, head_q=None, extra_q=None):
        """feet: {side: (ankle_world, foot_R)}, hands: {side: world point}, spine_q: {bone idx: (x,y,z)}"""
        sk, c = self.sk, self.c
        q = sk.zeros()
        for k, v in (spine_q or {}).items(): q[k] = v
        for k, v in (extra_q or {}).items(): q[k] = v
        S, E, R = sk.fk(np.asarray(root_pos, float), R_p, q)
        for li, leg in enumerate(c.legs):
            if feet is None or leg.side not in feet: continue
            a, Rf = feet[leg.side]
            ch = leg.chain; par = sk.parent[ch[0]]
            rows = leg_ik(sk, leg, self.pl[li], R[par], S[ch[0]], a, 0.0)
            for k, rw in zip(ch[:-1], rows): q[k] = rw
            Rc = R[par]
            for k in ch[:-1]:
                o = sk.axes_order[k]; d = {"X": q[k, 0], "Y": q[k, 1], "Z": q[k, 2]}
                Rc = Rc @ rot(o[0], d[o[0]]) @ rot(o[1], d[o[1]]) @ rot(o[2], d[o[2]])
            e = euler_from_R(Rc.T @ Rf, sk.axes_order[ch[-1]])
            for ax, nm in enumerate("XYZ"):
                e[ax] = clamp_lim(sk, ch[-1], nm, e[ax]) if nm in sk.bones[ch[-1]].lim else 0.0
            q[ch[-1]] = e
        S, E, R = sk.fk(np.asarray(root_pos, float), R_p, q)
        if hands is not None:                      # shoulder girdle follows the reach (clavicle lift and protraction)
            fwd_h = R_p[:, 0].copy(); fwd_h[2] = 0.0; fwd_h /= max(np.linalg.norm(fwd_h), 1e-6)
            for arm in c.arms:
                if arm.clav is None or arm.side not in hands: continue
                d = np.asarray(hands[arm.side], float) - S[arm.chain[0]]; L_ = float(sum(sk.length[k] for k in arm.chain[:2])) + 1e-6
                sg_ = 1.0 if arm.side == "L" else -1.0
                q[arm.clav] = [sg_ * float(np.clip((d[2] / L_ - 0.3) * 0.6, -0.1, 0.65)), 0.0, -sg_ * float(np.clip((d @ fwd_h / L_ - 0.3) * 0.5, -0.1, 0.5))]
            S, E, R = sk.fk(np.asarray(root_pos, float), R_p, q)
        for arm in c.arms:
            if hands is None or arm.side not in hands: continue
            u, f, h = arm.chain; par = sk.parent[u]
            pole = (poles or {}).get(arm.side, np.array([-0.5, 0.6 if arm.side == "L" else -0.6, -0.4]))
            qu, qf, Eb, Rua = arm_ik(sk, arm, R[par], S[u], np.asarray(hands[arm.side], float), pole)
            q[u] = qu; q[f] = qf
        for i, b in enumerate(sk.bones):
            for ax, nm in enumerate("XYZ"):
                if nm in b.lim: q[i, ax] = clamp_lim(sk, i, nm, q[i, ax])
                elif i != 0 and b.group not in ("leg", "foot"): q[i, ax] = 0.0
        S, E, R = sk.fk(np.asarray(root_pos, float), R_p, q)
        return S, E, R, q

def foot_rot(yaw=0.0, pitch=0.0, roll=0.0):
    return rot("Z", yaw) @ rot("Y", pitch) @ rot("X", roll)

def ankle_for(creature, leg_side, contact_ctr, Rf):
    """Ankle position that puts the middle of the sole at contact_ctr for foot rotation Rf."""
    leg = next(l for l in creature.legs if l.side == leg_side)
    mid = 0.5 * (leg.heel + leg.ball)
    return np.asarray(contact_ctr, float) - Rf @ mid
